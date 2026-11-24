"""FastAPI workbench for the Ka-band video SAR shadow project.

Drop-in replacement for the Streamlit UI. Reuses videosar_ui/ read-only helpers
and shells out to ``python -m videosar`` without modifying videosar/ or
deliverables/.

Run from repository root:
    python server.py
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from videosar_ui import commands as cmd
from videosar_ui import imaging as viz
from videosar_ui import scanning as scan

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="Ka Video SAR Shadow Workbench", version="2.0.0")
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ---------------------------------------------------------------------------
# Lightweight mtime-aware LRU caches (prevents re-reading .npy on every frame)
# ---------------------------------------------------------------------------
class _MtimeLRU:
    def __init__(self, maxsize: int = 12):
        self._maxsize = maxsize
        self._store: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Any, loader):
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
        value = loader()
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)
        return value


_SEQ_CACHE = _MtimeLRU(maxsize=8)
_PRED_CACHE = _MtimeLRU(maxsize=16)
_TRACK_CACHE = _MtimeLRU(maxsize=16)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


def _opt_path(text: str | None) -> Path | None:
    if not text or not str(text).strip():
        return None
    return Path(str(text).strip()).expanduser()


def _cached_sequence(seq_dir: Path):
    key = (str(seq_dir.resolve()), _mtime(seq_dir / "frames.npy"), _mtime(seq_dir / "masks.npy"))
    return _SEQ_CACHE.get(key, lambda: scan.load_sequence(seq_dir))


def _cached_predictions(pred_dir: Path | None, shape: tuple[int, ...]):
    if pred_dir is None:
        return None
    key = (str(pred_dir.resolve()), _mtime(pred_dir / "predictions.npy"), tuple(shape))
    return _PRED_CACHE.get(key, lambda: scan.load_predictions(pred_dir, shape))


def _cached_tracks(track_dir: Path | None):
    if track_dir is None:
        return scan.load_tracks(None)
    key = (str(track_dir.resolve()), _mtime(track_dir / "tracks.csv"), _mtime(track_dir / "tracks.json"))
    return _TRACK_CACHE.get(key, lambda: scan.load_tracks(track_dir))


def _clean(obj: Any) -> Any:
    """Make numpy / Path / non-finite floats JSON-safe."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _encode_png(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _media_url(path: Any, download: bool = False, name: str | None = None) -> str:
    """Build a /api/media URL; Windows paths carry backslashes and spaces."""
    url = "/api/media?path=" + quote(str(path), safe="")
    if download:
        url += "&download=1&name=" + quote(str(name or Path(str(path)).name), safe="")
    return url


def _recorded_sequence(folder: Path) -> str:
    """The sequence path a result folder records in its source.json receipt."""
    receipt = scan.read_json_safe(Path(folder) / "source.json")
    if isinstance(receipt, dict) and receipt.get("sequence"):
        return str(receipt["sequence"])
    return ""


def _tail_key(text: str, parts: int = 2) -> str:
    bits = [bit for bit in str(text).replace("\\", "/").lower().split("/") if bit]
    return "/".join(bits[-parts:])


def _pred_shape_ok(folder: Path, shape: tuple[int, ...]) -> bool:
    try:
        stack = np.load(Path(folder) / "predictions.npy", mmap_mode="r")
    except (OSError, ValueError):
        return False
    return tuple(stack.shape) == tuple(shape)


_INDEX_LOCK = threading.Lock()
_INDEX_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_INDEX_TTL = 60.0
_INDEX_FILE_TTL = 600.0


def _build_index(runs_root: Path) -> dict[str, list[dict[str, Any]]]:
    """One directory walk serving both sequence discovery and result lookup.

    ``scan.find_results`` walks the whole tree and re-reads source.json per
    candidate, which costs ~4 s on this repository (7 800 folders / 56 k files).
    The workbench needs the listing on every sequence switch, so it is built
    once and reused until the TTL expires.
    """
    index: dict[str, list[dict[str, Any]]] = {
        "sequences": [], "predictions": [], "tracks": [], "metrics": [],
    }
    if not runs_root.is_dir():
        return index
    markers = (("predictions", "predictions.npy"),
               ("tracks", "tracks.csv"),
               ("metrics", "metrics.json"))
    for dirpath, dirnames, filenames in os.walk(runs_root):
        dirnames[:] = [name for name in dirnames if name not in {"__pycache__", ".git"}]
        names = set(filenames)
        folder = Path(dirpath)
        if "frames.npy" in names:
            index["sequences"].append({"path": folder, "mtime": _mtime(folder / "frames.npy")})
        kinds = [kind for kind, marker in markers if marker in names]
        if not kinds:
            continue
        recorded = _recorded_sequence(folder)
        entry = {"path": folder, "recorded": recorded,
                 "tail": _tail_key(recorded), "mtime": _mtime(folder)}
        for kind in kinds:
            index[kind].append(entry)
    return index


def _index_cache_file(runs_root: Path) -> Path:
    digest = hashlib.md5(str(runs_root).lower().encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / "videosar-workbench" / f"runs-index-{digest}.json"


def _index_to_payload(index: dict[str, list[dict[str, Any]]], runs_root: Path) -> dict[str, Any]:
    return {
        "runs_root": str(runs_root),
        "built_at": time.time(),
        "sequences": [[str(e["path"]), e["mtime"]] for e in index["sequences"]],
        **{kind: [[str(e["path"]), e["recorded"], e["mtime"]] for e in index[kind]]
           for kind in ("predictions", "tracks", "metrics")},
    }


def _index_from_payload(payload: Any) -> dict[str, list[dict[str, Any]]] | None:
    if not isinstance(payload, dict):
        return None
    try:
        index: dict[str, list[dict[str, Any]]] = {
            "sequences": [{"path": Path(p), "mtime": float(m or -1.0)} for p, m in payload["sequences"]],
        }
        for kind in ("predictions", "tracks", "metrics"):
            entries = []
            for path, recorded, mtime in payload[kind]:
                entries.append({"path": Path(path), "recorded": recorded,
                                "tail": _tail_key(recorded), "mtime": float(mtime or -1.0)})
            index[kind] = entries
    except (KeyError, TypeError, ValueError):
        return None
    return index


def _runs_index(runs_root: Path, refresh: bool = False) -> dict[str, list[dict[str, Any]]]:
    key = str(runs_root.resolve()).lower()
    now = time.monotonic()
    with _INDEX_LOCK:
        cached = _INDEX_CACHE.get(key)
        if cached and not refresh and now - cached[0] < _INDEX_TTL:
            return cached[1]

    cache_file = _index_cache_file(runs_root)
    if not refresh:
        payload = scan.read_json_safe(cache_file)
        if isinstance(payload, dict) and time.time() - float(payload.get("built_at") or 0) < _INDEX_FILE_TTL:
            restored = _index_from_payload(payload)
            if restored is not None:
                with _INDEX_LOCK:
                    _INDEX_CACHE[key] = (now, restored)
                return restored

    index = _build_index(runs_root)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(_index_to_payload(index, runs_root)), encoding="utf-8")
    except OSError:
        pass  # a cache miss only costs time, never correctness
    with _INDEX_LOCK:
        _INDEX_CACHE[key] = (now, index)
    return index


def _same_path(recorded: str, target: Path) -> bool:
    """Lexical comparison: ``resolve()`` would cost one stat per candidate."""
    if not recorded:
        return False
    try:
        left = os.path.normcase(os.path.abspath(str(recorded)))
        right = os.path.normcase(os.path.abspath(str(target)))
    except (OSError, ValueError):
        return False
    return left == right


def _pair_valid(kind: str, folder: Path, sequence) -> bool:
    """Reject receipts whose payload cannot describe these frames."""
    frames = sequence["frames"]
    if kind == "predictions":
        return _pred_shape_ok(folder, tuple(frames.shape))
    if kind == "tracks":
        rows = _cached_tracks(folder)["rows"]
        return bool(rows) and rows[-1]["frame"] <= len(frames) - 1
    metrics = scan.load_metrics(folder)
    logged = (metrics or {}).get("detection", {}).get("frames")
    return bool(metrics) and (not isinstance(logged, int) or logged <= len(frames))


def _ranked_results(index, seq_path: Path, sequence) -> tuple[dict, dict, dict]:
    """Mirror ``scan.find_results`` (exact link first) from the cached index.

    ``find_results`` compares absolute paths, so a result produced from a
    renamed dataset root (``echo-stress`` vs ``echo-stress-repro``) counts as
    unlinked although the frames are identical. Matching the tail of the
    recorded path plus a shape / frame-count check restores that pairing.
    """
    tail = _tail_key(seq_path)
    buckets: dict[str, list[Path]] = {}
    auto: dict[str, Path] = {}
    linked: dict[str, bool] = {}
    for kind in ("predictions", "tracks", "metrics"):
        entries = index[kind]
        exact = [e for e in entries if _same_path(e["recorded"], seq_path)]
        near = sorted((e for e in entries if e["tail"] and e["tail"] == tail),
                      key=lambda e: -e["mtime"])
        pool = exact or near or sorted(entries, key=lambda e: -e["mtime"])[:60]
        linked[kind] = bool(exact)
        buckets[kind] = [e["path"] for e in pool]
        chosen = next((e["path"] for e in near if _pair_valid(kind, e["path"], sequence)), None)
        if chosen is not None:
            auto[kind] = chosen
    return buckets, auto, linked


def _sequence_label(path: Path, runs_root: Path) -> str:
    text = scan._relative_text(path, runs_root)
    if text.endswith(scan.PREFERRED_SEQUENCE):
        return f"[结题演示·优先] {text}"
    if "gui/" in text or text.startswith("gui"):
        return f"[GUI 现场结果] {text}"
    return text


def _build_views(sequence, predictions, track_rows, index: int, settings: dict[str, Any]):
    frames = sequence["frames"]
    truth = sequence["masks"]
    prediction = None if predictions is None else predictions[index]
    base = viz.enhance(
        frames[index],
        settings.get("mode", "percentile"),
        settings.get("cmap", "gray"),
        bool(settings.get("denoise", False)),
    )

    view1 = base.copy()
    if settings.get("show_truth", True) and truth is not None:
        view1 = viz.overlay_mask(view1, truth[index], viz.TRUTH_COLOR)
    if settings.get("show_pred", True) and prediction is not None:
        view1 = viz.overlay_mask(view1, prediction, viz.PREDICTION_COLOR)

    view2 = viz.temporal_difference(frames, index, settings.get("diff_ref", "median"))

    view3 = base.copy()
    if prediction is not None:
        view3 = viz.overlay_mask(view3, prediction, viz.PREDICTION_COLOR)
    view3 = viz.draw_tracks(view3, track_rows, index)

    center = viz.centroid(prediction)
    if center is None and truth is not None:
        center = viz.centroid(truth[index])
    view4, _roi = viz.roi_zoom(view1, center, factor=4)
    return [view1, view2, view3, view4], center


# ---------------------------------------------------------------------------
# Background command runner with SSE streaming
# ---------------------------------------------------------------------------
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_MAX_JOBS = 40


def _start_job(commands: list[list[str]], result_builder) -> str:
    job_id = uuid.uuid4().hex[:12]
    state: dict[str, Any] = {"logs": [], "done": False, "ok": False, "result": None}
    with _JOBS_LOCK:
        # Keep only the most recent finished jobs so a long session cannot grow
        # the log buffer without bound.
        if len(_JOBS) > _MAX_JOBS:
            for stale in [k for k, v in _JOBS.items() if v["done"]][: len(_JOBS) - _MAX_JOBS]:
                _JOBS.pop(stale, None)
        _JOBS[job_id] = state

    def worker():
        def emit(line: Any):
            state["logs"].append(str(line))

        ok = cmd.run_commands(commands, scan.REPO_ROOT, emit)
        state["ok"] = ok
        if ok:
            try:
                state["result"] = _clean(result_builder())
            except Exception as exc:  # pragma: no cover - defensive
                state["logs"].append(f"[后处理异常] {exc}")
                state["ok"] = False
        state["done"] = True

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def _start_grouped_job(groups: list[tuple[str, list[list[str]]]], result_builder) -> str:
    """Run several independent command chains; one failing does not kill the rest."""
    job_id = uuid.uuid4().hex[:12]
    state: dict[str, Any] = {"logs": [], "done": False, "ok": False, "result": None}
    with _JOBS_LOCK:
        if len(_JOBS) > _MAX_JOBS:
            for stale in [k for k, v in _JOBS.items() if v["done"]][: len(_JOBS) - _MAX_JOBS]:
                _JOBS.pop(stale, None)
        _JOBS[job_id] = state

    def worker():
        def emit(line: Any):
            state["logs"].append(str(line))

        outcomes: dict[str, bool] = {}
        for name, commands in groups:
            emit(f"=== {name} ===")
            outcomes[name] = cmd.run_commands(commands, scan.REPO_ROOT, emit)
            if not outcomes[name]:
                emit(f"[{name}] 失败，继续其余方法")
        state["ok"] = any(outcomes.values())
        if state["ok"]:
            try:
                state["result"] = _clean(result_builder(outcomes))
            except Exception as exc:  # pragma: no cover - defensive
                state["logs"].append(f"[后处理异常] {exc}")
                state["ok"] = False
        state["done"] = True

    threading.Thread(target=worker, daemon=True).start()
    return job_id


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------
class DisplaySettings(BaseModel):
    mode: str = "percentile"
    cmap: str = "gray"
    denoise: bool = False
    diff_ref: str = "median"
    show_truth: bool = True
    show_pred: bool = True


class BrowserMetaRequest(BaseModel):
    sequence_dir: str
    detect_dir: str | None = None
    track_dir: str | None = None
    evaluate_dir: str | None = None


class FrameRequest(BrowserMetaRequest):
    index: int = 0
    settings: DisplaySettings = DisplaySettings()


class BatchFrameRequest(FrameRequest):
    indices: list[int] = []


class ExportRequest(BrowserMetaRequest):
    format: str = "gif"  # gif | mp4
    settings: DisplaySettings = DisplaySettings()


class CompareSide(BaseModel):
    detect_dir: str
    evaluate_dir: str | None = None
    label: str | None = None


class CompareFrameRequest(BaseModel):
    sequence_dir: str
    left_dir: str | None = None
    right_dir: str | None = None
    sides: list[CompareSide] | None = None
    index: int = 0
    settings: DisplaySettings = DisplaySettings()


class DetectJobRequest(BaseModel):
    sequence_dir: str
    methods: list[str] = ["threshold"]
    model: str | None = None
    threshold: float | None = None
    min_area: int = 4
    gate_px: float = 12.0
    max_gap: int = 2
    match_iou: float = 0.5


class DemoJobRequest(BaseModel):
    config: str = str(cmd.DEMO_CONFIG)
    frames: int = 16


class SarJobRequest(BaseModel):
    bandwidth_hz: int = 450_000_000
    method: str = "threshold"
    model: str = str(cmd.UNET_MODEL)


class CompareJobRequest(BaseModel):
    sequence_dir: str


class VideoInspectRequest(BaseModel):
    video_path: str


class DetectVideoJobRequest(BaseModel):
    video_path: str
    crop: list[int] | None = None
    start_frame: int = 0
    max_frames: int = 24
    model: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root_page():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/bootstrap")
def api_bootstrap(runs_root: str | None = Query(default=None)):
    runs_path = _opt_path(runs_root) or scan.DEFAULT_RUNS
    env = cmd.environment_report()
    tables = [{"label": label, "path": str(path)} for label, path in scan.archive_tables()]
    return JSONResponse(_clean({
        "repo_root": str(scan.REPO_ROOT),
        "default_runs": str(scan.DEFAULT_RUNS),
        "runs_root": str(runs_path),
        "runs_exists": runs_path.is_dir(),
        "modes": list(viz.MODES),
        "mode_labels": viz.MODE_LABELS,
        "cmaps": list(viz.CMAPS),
        "defaults": {
            "demo_config": str(cmd.DEMO_CONFIG),
            "unet_model": str(cmd.UNET_MODEL),
            "snl_model": str(cmd.SNL_MODEL),
            "ffmpeg_install": cmd.FFMPEG_INSTALL,
            "torch_install": cmd.TORCH_INSTALL,
        },
        "environment": env,
        "snl_model_info": scan.snl_model_info(),
        "archive_tables": tables,
        "preview_videos": scan.preview_videos(runs_path),
    }))


@app.get("/api/sequences")
def api_sequences(runs_root: str | None = Query(default=None), refresh: int = Query(default=0)):
    runs_path = _opt_path(runs_root) or scan.DEFAULT_RUNS
    index = _runs_index(runs_path, refresh=bool(refresh))
    ranked = sorted(
        index["sequences"],
        key=lambda e: (scan._sequence_rank(e["path"], runs_path), -e["mtime"], str(e["path"])),
    )
    seqs = [entry["path"] for entry in ranked[:250]]
    return JSONResponse(_clean({
        "runs_exists": runs_path.is_dir(),
        "sequences": [{"path": str(p), "label": _sequence_label(p, runs_path)} for p in seqs],
    }))


@app.get("/api/results")
def api_results(runs_root: str = Query(...), sequence_dir: str = Query(...),
                refresh: int = Query(default=0)):
    runs_path = _opt_path(runs_root) or scan.DEFAULT_RUNS
    seq_path = _opt_path(sequence_dir)
    if seq_path is None or not seq_path.is_dir():
        raise HTTPException(status_code=400, detail="序列目录无效")

    seq = _cached_sequence(seq_path)
    if seq is None:
        raise HTTPException(status_code=400, detail="序列缺少 frames.npy，无法列出结果")
    index = _runs_index(runs_path, refresh=bool(refresh))
    results, auto, linked = _ranked_results(index, seq_path, seq)

    live = scan.live_dirs(runs_path)
    preferred = {
        "predictions": live.get("live-detect"),
        "tracks": live.get("live-track"),
        "metrics": live.get("live-eval"),
    }
    for kind, folder in preferred.items():
        if folder and folder.is_dir() and folder not in results[kind]:
            results[kind] = [folder] + results[kind]
    for kind, folder in auto.items():
        results[kind] = [folder] + [p for p in results[kind] if p != folder]

    def fmt_items(paths: list[Path], check_shape: bool = False):
        items = []
        for path in paths:
            item = {"path": str(path), "label": scan._relative_text(path, runs_path)}
            if check_shape and seq is not None:
                item["shape_ok"] = _pred_shape_ok(path, tuple(seq["frames"].shape))
            items.append(item)
        return items

    return JSONResponse(_clean({
        "predictions": fmt_items(results["predictions"], check_shape=True),
        "tracks": fmt_items(results["tracks"]),
        "metrics": fmt_items(results["metrics"]),
        "linked": linked,
        "auto": {kind: str(path) for kind, path in auto.items()},
        "auto_used": bool(auto),
    }))


@app.post("/api/browser/meta")
def api_browser_meta(req: BrowserMetaRequest):
    seq_path = _opt_path(req.sequence_dir)
    if seq_path is None or not seq_path.is_dir():
        raise HTTPException(status_code=400, detail=f"目录不存在：{req.sequence_dir}")
    seq = _cached_sequence(seq_path)
    if seq is None:
        raise HTTPException(status_code=400, detail=f"无法加载序列（需 frames.npy）：{seq_path}")

    frames = seq["frames"]
    count = int(len(frames))
    detect_path = _opt_path(req.detect_dir)
    track_path = _opt_path(req.track_dir)
    eval_path = _opt_path(req.evaluate_dir)

    tracks = _cached_tracks(track_path)
    metrics = scan.load_metrics(eval_path)

    predictions = _cached_predictions(detect_path, frames.shape)
    if predictions is not None:
        pred_state = "ok"
    elif detect_path is not None and (detect_path / "predictions.npy").is_file():
        pred_state = "mismatch"  # file exists but belongs to another frame shape
    elif detect_path is not None:
        pred_state = "absent"
    else:
        pred_state = "none"

    missing = []
    if predictions is None:
        missing.append("predictions.npy")
    if track_path is None or not tracks["rows"]:
        missing.append("tracks.csv")
    if metrics is None:
        missing.append("metrics.json")

    cards = [
        {"label": label, "value": scan.format_metric(val), "raw": val}
        for label, val in scan.metric_cards(metrics, count)
    ]
    tracking = [
        {"label": label, "value": scan.format_metric(val), "raw": val}
        for label, val in scan.tracking_summary(metrics, tracks).items()
    ]
    frame_indices, speeds = scan.per_frame_speed(tracks["rows"]) if tracks["rows"] else ([], [])

    return JSONResponse(_clean({
        "count": count,
        "shape": list(frames.shape),
        "has_truth": seq["masks"] is not None,
        "metadata": seq["metadata"],
        "missing": missing,
        "pred_state": pred_state,
        "cards": cards,
        "tracking": tracking,
        "match_iou": scan.format_metric(metrics["match_iou"]) if metrics and metrics.get("match_iou") is not None else None,
        "metrics_path": str(metrics["path"]) if metrics else None,
        "tracks_path": str(tracks.get("path")) if tracks.get("path") else None,
        "track_rows": tracks["rows"],
        "speed_series": {"frames": frame_indices, "speeds": speeds},
    }))


def _frame_payload(req: FrameRequest) -> dict[str, Any]:
    seq_path = _opt_path(req.sequence_dir)
    if seq_path is None:
        raise HTTPException(status_code=400, detail="缺少序列目录")
    seq = _cached_sequence(seq_path)
    if seq is None:
        raise HTTPException(status_code=400, detail=f"无法加载序列：{seq_path}")

    frames = seq["frames"]
    count = int(len(frames))
    idx = max(0, min(count - 1, int(req.index)))
    predictions = _cached_predictions(_opt_path(req.detect_dir), frames.shape)
    tracks = _cached_tracks(_opt_path(req.track_dir))

    views, center = _build_views(seq, predictions, tracks["rows"], idx, req.settings.model_dump())
    return {
        "index": idx,
        "count": count,
        "center": list(center) if center else None,
        "panels": [_encode_png(v) for v in views],
    }


@app.post("/api/browser/frame")
def api_browser_frame(req: FrameRequest):
    return JSONResponse(_clean(_frame_payload(req)))


@app.post("/api/browser/frames")
def api_browser_frames(req: BatchFrameRequest):
    """Batch render used by the frontend frame cache pool."""
    if not req.indices:
        return JSONResponse({"frames": []})
    payloads = []
    for index in req.indices[:24]:
        single = req.model_copy(update={"index": index})
        payloads.append(_frame_payload(single))
    return JSONResponse(_clean({"frames": payloads}))


@app.post("/api/browser/export")
def api_browser_export(req: ExportRequest):
    seq_path = _opt_path(req.sequence_dir)
    if seq_path is None:
        raise HTTPException(status_code=400, detail="缺少序列目录")
    seq = _cached_sequence(seq_path)
    if seq is None:
        raise HTTPException(status_code=400, detail="无法加载序列")

    frames = seq["frames"]
    predictions = _cached_predictions(_opt_path(req.detect_dir), frames.shape)
    tracks = _cached_tracks(_opt_path(req.track_dir))
    settings = req.settings.model_dump()

    output = cmd.gui_run_dir("export")
    panels = [
        viz.compose_quad(_build_views(seq, predictions, tracks["rows"], i, settings)[0])
        for i in range(len(frames))
    ]
    if req.format == "mp4":
        try:
            saved = viz.save_mp4(panels, output / "quad.mp4", fps=8)
        except (ImportError, OSError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail=f"MP4 导出失败：{exc}（请安装 imageio-ffmpeg）")
        filename = "sar_quad.mp4"
    else:
        saved = viz.save_gif(panels, output / "quad.gif", duration_ms=125)
        filename = "sar_quad.gif"

    return JSONResponse(_clean({
        "path": str(saved),
        "filename": filename,
        "download_url": _media_url(saved, download=True, name=filename),
    }))


def _side_summary(evaluate_dir: Path | None) -> dict[str, Any] | None:
    """Four headline numbers plus diagnostics, straight from metrics.json."""
    metrics = scan.load_metrics(evaluate_dir)
    if not metrics:
        return None
    detection = metrics.get("detection") or {}
    tracking = metrics.get("tracking") or {}
    return {
        "precision": detection.get("precision"),
        "recall": detection.get("recall"),
        "target_f1": detection.get("target_f1"),
        "iou": detection.get("iou"),
        "fp_per_frame": detection.get("false_positive_per_frame"),
        "matched": detection.get("matched"),
        "missed": detection.get("missed"),
        "false_positive": detection.get("false_positive"),
        "continuity": tracking.get("continuity"),
        "location_rmse_px": tracking.get("location_rmse_px"),
        "velocity_error_mps": tracking.get("velocity_error_mps"),
        "match_iou": metrics.get("match_iou"),
    }


@app.post("/api/compare/frame")
def api_compare_frame(req: CompareFrameRequest):
    seq_path = _opt_path(req.sequence_dir)
    if seq_path is None:
        raise HTTPException(status_code=400, detail="缺少序列目录")
    seq = _cached_sequence(seq_path)
    if seq is None:
        raise HTTPException(status_code=400, detail="无法加载序列")

    if req.sides:
        targets = [(side.detect_dir, side.evaluate_dir, side.label) for side in req.sides]
    else:
        targets = [(req.left_dir, None, None), (req.right_dir, None, None)]
    targets = [item for item in targets if item[0]]
    if not targets:
        raise HTTPException(status_code=400, detail="未提供检测结果目录")

    frames = seq["frames"]
    count = int(len(frames))
    idx = max(0, min(count - 1, int(req.index)))
    truth = seq["masks"]
    s = req.settings.model_dump()
    base = viz.enhance(frames[idx], s["mode"], s["cmap"], s["denoise"])

    out_sides = []
    for detect_dir, evaluate_dir, label in targets:
        pred = _cached_predictions(_opt_path(detect_dir), frames.shape)
        view = base.copy()
        if pred is not None:
            view = viz.overlay_mask(view, pred[idx], viz.PREDICTION_COLOR)
        if truth is not None:
            view = viz.overlay_mask(view, truth[idx], viz.TRUTH_COLOR)
        components = viz.count_components(pred[idx]) if pred is not None else None
        overlap = int(np.count_nonzero(pred[idx] & truth[idx])) if pred is not None and truth is not None else None
        out_sides.append({
            "label": label or Path(detect_dir).name,
            "path": detect_dir,
            "image": _encode_png(view),
            "components": scan.format_metric(components),
            "overlap": scan.format_metric(overlap),
            "summary": _side_summary(_opt_path(evaluate_dir)),
        })

    return JSONResponse(_clean({"index": idx, "count": count, "sides": out_sides}))


@app.get("/api/archive_table")
def api_archive_table(path: str = Query(...)):
    file_path = _opt_path(path)
    if file_path is None or not file_path.is_file():
        raise HTTPException(status_code=404, detail="归档 CSV 不存在")
    fields, rows = scan.read_csv_rows(file_path)
    if not fields or not rows:
        return JSONResponse({"fields": fields or [], "rows": [], "chart": None})

    chart = None
    if "condition" in fields and "recall" in fields and "method" in fields:
        # Grouped recall by condition x method
        conditions = list(dict.fromkeys(r["condition"] for r in rows if r.get("condition")))
        methods = list(dict.fromkeys(r["method"] for r in rows if r.get("method")))
        series = []
        for m in methods:
            vals = []
            for c in conditions:
                nums = [scan.numeric(r["recall"]) for r in rows if r.get("condition") == c and r.get("method") == m]
                nums = [n for n in nums if n is not None]
                vals.append(round(sum(nums) / len(nums), 4) if nums else 0.0)
            series.append({"name": m, "values": vals})
        chart = {"title": "分条件召回率对比 (Recall)", "categories": conditions, "series": series}
    elif "method" in fields:
        metric_cols = [c for c in ("precision", "recall", "target_f1", "region_iou") if c in fields]
        if metric_cols:
            methods = [r["method"] for r in rows]
            series = [
                {"name": col, "values": [scan.numeric(r.get(col)) or 0.0 for r in rows]}
                for col in metric_cols
            ]
            chart = {"title": "各方法核心指标对比", "categories": methods, "series": series}

    return JSONResponse(_clean({"fields": fields, "rows": rows, "chart": chart}))


@app.post("/api/jobs/demo")
def api_job_demo(req: DemoJobRequest):
    output = cmd.gui_run_dir("demo")
    demo_dir = output / "demo"
    commands = [cmd.demo_command(req.config, req.frames, demo_dir)]
    job_id = _start_job(
        commands,
        lambda: {
            "sequence": demo_dir / "sequence",
            "predictions": demo_dir / "detect",
            "tracks": demo_dir / "track",
            "metrics": demo_dir / "evaluate",
        },
    )
    return JSONResponse({"job_id": job_id})


@app.post("/api/jobs/sar")
def api_job_sar(req: SarJobRequest):
    output = cmd.gui_run_dir("sar")
    plan = cmd.sar_chain_commands(req.bandwidth_hz, req.method, Path(req.model), output)
    job_id = _start_job(
        plan["commands"],
        lambda: {
            "sequence": plan["sequence"],
            "predictions": plan["predictions"],
            "tracks": plan["tracks"],
            "metrics": plan["metrics"],
        },
    )
    return JSONResponse({"job_id": job_id})


@app.post("/api/jobs/compare")
def api_job_compare(req: CompareJobRequest):
    seq_path = _opt_path(req.sequence_dir)
    if seq_path is None or not seq_path.is_dir():
        raise HTTPException(status_code=400, detail="序列目录无效")
    output = cmd.gui_run_dir("compare")
    plan = cmd.compare_commands(seq_path, output)
    job_id = _start_job(
        plan["commands"],
        lambda: {"predictions": [str(p) for p in plan["predictions"]]},
    )
    return JSONResponse({"job_id": job_id})


METHOD_LABELS = {
    "threshold": "局部暗度阈值",
    "difference": "时序背景差分",
    "learned": "U-Net 时序",
}


def _method_chain(sequence: Path, method: str, root: Path, req: DetectJobRequest) -> list[list[str]]:
    """detect -> track -> evaluate for one method on an existing sequence."""
    detect_dir, track_dir, evaluate_dir = root / method / "detect", root / method / "track", root / method / "evaluate"
    detect = [sys.executable, "-m", "videosar", "detect",
              "--input", str(sequence), "--method", method,
              "--min-area", str(int(req.min_area)), "--output-dir", str(detect_dir)]
    if req.threshold is not None:
        detect += ["--threshold", f"{float(req.threshold):g}"]
    if method == "learned":
        detect += ["--model", str(_opt_path(req.model) or cmd.UNET_MODEL)]
    track = [sys.executable, "-m", "videosar", "track",
             "--input", str(sequence), "--predictions", str(detect_dir),
             "--gate-px", f"{float(req.gate_px):g}", "--max-gap", str(int(req.max_gap)),
             "--output-dir", str(track_dir)]
    evaluate = [sys.executable, "-m", "videosar", "evaluate",
                "--input", str(sequence), "--predictions", str(detect_dir), "--tracks", str(track_dir),
                "--match-iou", f"{float(req.match_iou):g}", "--output-dir", str(evaluate_dir)]
    return [detect, track, evaluate]


@app.post("/api/jobs/detect")
def api_job_detect(req: DetectJobRequest):
    """Run one or more detection methods on the selected sequence."""
    seq_path = _opt_path(req.sequence_dir)
    if seq_path is None or not seq_path.is_dir():
        raise HTTPException(status_code=400, detail="序列目录无效")
    methods = [m for m in req.methods if m in METHOD_LABELS]
    if not methods:
        raise HTTPException(status_code=400, detail="请至少选择一种检测方法")
    if "learned" in methods:
        env = cmd.environment_report()
        if not env["torch"]:
            raise HTTPException(status_code=400, detail="学习模型需要 PyTorch，请查看环境自检")

    output = cmd.gui_run_dir("detect")
    groups = [(m, _method_chain(seq_path, m, output, req)) for m in methods]

    def collect(outcomes: dict[str, bool]):
        results = []
        for method in methods:
            folder = output / method
            results.append({
                "method": method,
                "label": METHOD_LABELS[method],
                "ok": bool(outcomes.get(method)),
                "detect": folder / "detect",
                "track": folder / "track",
                "evaluate": folder / "evaluate",
            })
        return {"sequence": seq_path, "results": results}

    job_id = _start_grouped_job(groups, collect)
    return JSONResponse({"job_id": job_id, "methods": methods})



def _extract_preview(video_path: Path, workspace: Path) -> dict[str, Any]:
    ffmpeg = cmd.ffmpeg_path()
    if not ffmpeg:
        raise HTTPException(status_code=400, detail=f"未检测到 FFmpeg，请先安装：{cmd.FFMPEG_INSTALL}")
    first = cmd.extract_first_frame(video_path, ffmpeg, workspace / "first_frame.png")
    if first is None:
        raise HTTPException(status_code=400, detail="FFmpeg 无法解码该视频首帧，请检查文件格式")
    with Image.open(first) as img:
        w, h = img.size
    return {
        "video_path": str(video_path),
        "first_frame_path": str(first),
        "first_frame_url": _media_url(first),
        "width": w,
        "height": h,
    }


@app.post("/api/video/upload")
async def api_video_upload(file: UploadFile = File(...)):
    workspace = cmd.gui_run_dir("video-input")
    workspace.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "input.mp4").name
    dest = workspace / safe_name
    content = await file.read()
    dest.write_bytes(content)
    return JSONResponse(_clean(_extract_preview(dest, workspace)))


@app.post("/api/video/inspect")
def api_video_inspect(req: VideoInspectRequest):
    vpath = _opt_path(req.video_path)
    if vpath is None or not vpath.is_file():
        raise HTTPException(status_code=404, detail=f"视频文件不存在：{req.video_path}")
    workspace = cmd.gui_run_dir("video-preview")
    return JSONResponse(_clean(_extract_preview(vpath, workspace)))


@app.post("/api/jobs/detect_video")
def api_job_detect_video(req: DetectVideoJobRequest):
    vpath = _opt_path(req.video_path)
    if vpath is None or not vpath.is_file():
        raise HTTPException(status_code=400, detail="请先上传视频或提供有效的本地视频路径")
    env = cmd.environment_report()
    if not env["ffmpeg"] or not env["torch"]:
        raise HTTPException(status_code=400, detail="缺少 FFmpeg 或 PyTorch 依赖，请查看环境自检提示")

    output = cmd.gui_run_dir("video")
    crop_tuple = tuple(req.crop) if req.crop and len(req.crop) == 4 else None
    command = cmd.detect_video_command(
        vpath,
        output,
        crop=crop_tuple,
        start_frame=req.start_frame,
        max_frames=req.max_frames,
        model=_opt_path(req.model),
        ffmpeg=env["ffmpeg"],
    )

    def collect():
        overlay = output / "overlay.mp4"
        preds = output / "predictions.csv"
        vinfo = scan.read_json_safe(output / "video.json")
        fields, rows = scan.read_csv_rows(preds) if preds.is_file() else (None, [])
        return {
            "output_dir": str(output),
            "overlay_url": _media_url(overlay) if overlay.is_file() else None,
            "overlay_download": _media_url(overlay, download=True, name="overlay.mp4") if overlay.is_file() else None,
            "csv_download": _media_url(preds, download=True, name="predictions.csv") if preds.is_file() else None,
            "csv_fields": fields or [],
            "csv_rows": rows[:300],
            "csv_total": len(rows),
            "video_json": vinfo,
        }

    job_id = _start_job([command], collect)
    return JSONResponse({"job_id": job_id})


@app.get("/api/jobs/{job_id}/stream")
async def api_job_stream(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    async def event_gen():
        cursor = 0
        while True:
            logs = job["logs"]
            while cursor < len(logs):
                line = logs[cursor]
                cursor += 1
                yield f"event: log\ndata: {json.dumps({'line': line}, ensure_ascii=False)}\n\n"
            if job["done"] and cursor >= len(logs):
                payload = {"ok": job["ok"], "result": job["result"]}
                yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                break
            await asyncio.sleep(0.08)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/api/media")
def api_media(path: str = Query(...), download: int = Query(default=0), name: str | None = Query(default=None)):
    target = _opt_path(path)
    if target is None or not target.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在：{path}")
    headers = {}
    if download:
        fname = name or target.name
        # RFC 5987: keep non-ASCII / Windows names from breaking the header.
        headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{quote(fname, safe='')}"
    return FileResponse(target, headers=headers)


if __name__ == "__main__":
    print("Ka Video SAR Workbench running at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
