"""Read-only discovery and parsing of experiment outputs for the GUI.

All metric values shown in the interface come from the files here; nothing is
hard-coded. Missing files always yield ``None`` / empty collections instead of
raising, so the GUI can show a clear hint and offer a manual path.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = REPO_ROOT / "runs"
PREFERRED_SEQUENCE = "echo-stress-repro/dataset/stress-00-reference"
PREFERRED_LIVE = ("live-detect", "live-track", "live-eval")

ARCHIVE_TABLES = (
    ("申报书正式测试各方法对比", "deliverables/proposal/metrics.csv"),
    ("纯回波压测各方法对比", "deliverables/closeout/stress/metrics.csv"),
    ("纯回波压测分条件召回率变化", "deliverables/closeout/stress/metrics_by_condition.csv"),
    ("申报书分组指标", "deliverables/proposal/metrics_by_group.csv"),
)
SNL_MODEL = REPO_ROOT / "deliverables/snl-causal-study/models/stacked-center.json"
SNL_PREVIEWS = (
    "snl-causal-study/prospective/development-preview.mp4",
    "snl-video-prefix-v2/overlay.mp4",
)


def read_json_safe(path):
    """Return a JSON object, or None when missing / malformed."""
    try:
        with Path(path).open(encoding="utf-8-sig") as stream:
            return json.load(stream)
    except (OSError, ValueError, TypeError):
        return None


def read_csv_rows(path):
    """Return (fieldnames, rows). Absent or broken files return (None, [])."""
    source = Path(path)
    if not source.is_file():
        return None, []
    try:
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)
    except (OSError, UnicodeError, csv.Error):
        return None, []


def numeric(value):
    """Convert a CSV cell to float, tolerating blanks and text."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_metric(value):
    """Format a metric for a card; '—' when unavailable."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _relative_text(path, root):
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _sequence_rank(path, root):
    text = _relative_text(path, root).lower()
    if text.endswith(PREFERRED_SEQUENCE):
        return 0
    if text.startswith("gui/") or "/gui/" in text:
        return 1
    if any(name in text for name in PREFERRED_LIVE):
        return 2
    return 3


def find_sequences(runs_root, max_results=250):
    """Find directories containing frames.npy, preferred demo data first."""
    root = Path(runs_root)
    found, seen = [], set()
    if not root.is_dir():
        return found
    for dirpath, _dirnames, filenames in os.walk(root):
        if "frames.npy" not in filenames:
            continue
        candidate = Path(dirpath)
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(candidate)
    # Sort the complete set before truncating: an early break would drop the
    # preferred closeout demo whenever runs/ contains many newer sequences.
    found.sort(key=lambda path: (_sequence_rank(path, root), -path.stat().st_mtime, str(path)))
    return found[:max_results]


def load_sequence(directory):
    """Load frames, optional truth masks and metadata from a sequence folder."""
    root = Path(directory)
    frames_path = root / "frames.npy"
    if not frames_path.is_file():
        return None
    try:
        frames = np.asarray(np.load(frames_path, allow_pickle=False), dtype=np.float32)
    except (OSError, ValueError):
        return None
    if frames.ndim != 3 or frames.size == 0:
        return None
    masks = None
    masks_path = root / "masks.npy"
    if masks_path.is_file():
        try:
            candidate = np.load(masks_path, allow_pickle=False)
            if candidate.shape == frames.shape:
                masks = candidate.astype(bool)
        except (OSError, ValueError):
            masks = None
    return {"root": root, "frames": frames, "masks": masks,
            "metadata": read_json_safe(root / "metadata.json") or {}}


def _linked_to(directory, target):
    receipt = read_json_safe(Path(directory) / "source.json")
    if not isinstance(receipt, dict):
        return False
    sequence = receipt.get("sequence")
    if not sequence:
        return False
    try:
        return Path(sequence).resolve() == Path(target).resolve()
    except (OSError, ValueError):
        return False


def find_results(runs_root, sequence_root, max_each=60):
    """Locate detect / track / evaluate folders for one sequence.

    A source.json receipt links a result folder to its input sequence. When no
    receipt matches, all candidates are still returned so the user can pick
    manually instead of hitting an error.
    """
    root = Path(runs_root)
    buckets = {"predictions": [], "tracks": [], "metrics": [], "_linked": {}}
    if not root.is_dir():
        return buckets
    for dirpath, _dirnames, filenames in os.walk(root):
        folder = Path(dirpath)
        if "predictions.npy" in filenames:
            buckets["predictions"].append(folder)
        if "tracks.csv" in filenames:
            buckets["tracks"].append(folder)
        if "metrics.json" in filenames:
            buckets["metrics"].append(folder)
    target = Path(sequence_root)
    for kind in ("predictions", "tracks", "metrics"):
        candidates = buckets[kind]
        linked = [path for path in candidates if _linked_to(path, target)]
        buckets["_linked"][kind] = bool(linked)
        chosen = linked if linked else candidates
        buckets[kind] = sorted(chosen, key=lambda path: -path.stat().st_mtime)[:max_each]
    return buckets


def load_predictions(detect_dir, expected_shape):
    """Load predictions.npy as a boolean stack with a matching shape."""
    if detect_dir is None:
        return None
    path = Path(detect_dir) / "predictions.npy"
    try:
        masks = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    return masks.astype(bool) if masks.shape == expected_shape else None


def load_tracks(track_dir):
    """Read tracks.csv plus the velocity summary from tracks.json."""
    if track_dir is None:
        return {"rows": [], "velocity_px_s": None, "shadow_velocity_mps": None,
                "true_shadow_velocity_px_s": None}
    folder = Path(track_dir)
    fields, raw_rows = read_csv_rows(folder / "tracks.csv")
    rows = []
    if fields:
        for row in raw_rows:
            frame = numeric(row.get("frame"))
            if frame is None:
                continue
            rows.append({"frame": int(frame), "time_s": numeric(row.get("time_s")),
                         "x_px": numeric(row.get("x_px")), "y_px": numeric(row.get("y_px")),
                         "source": row.get("source"), "segment": row.get("segment")})
    info = read_json_safe(folder / "tracks.json") or {}
    return {"rows": rows, "velocity_px_s": info.get("velocity_px_s"),
            "shadow_velocity_mps": info.get("shadow_velocity_mps"),
            "true_shadow_velocity_px_s": None, "path": folder / "tracks.csv"}


def load_metrics(evaluate_dir):
    """Read evaluate/metrics.json without assuming any values."""
    if evaluate_dir is None:
        return None
    path = Path(evaluate_dir) / "metrics.json"
    data = read_json_safe(path)
    if not isinstance(data, dict):
        return None
    detection = data.get("detection") if isinstance(data.get("detection"), dict) else {}
    tracking = data.get("tracking") if isinstance(data.get("tracking"), dict) else {}
    return {"path": path, "detection": detection, "tracking": tracking,
            "match_iou": data.get("match_iou")}


def metric_cards(metrics, frame_count=None):
    """Return (label, value) pairs for the KPI cards; all values are dynamic."""
    detection = (metrics or {}).get("detection") or {}
    tracking = (metrics or {}).get("tracking") or {}
    if not metrics:
        return [(label, None) for label in
                ("查准率 Precision", "召回率 Recall", "目标级 F1", "区域 IoU",
                 "每帧虚警", "匹配帧", "漏检帧", "虚警总数", "帧数")]
    return [
        ("查准率 Precision", detection.get("precision")),
        ("召回率 Recall", detection.get("recall")),
        ("目标级 F1", detection.get("target_f1")),
        ("区域 IoU", detection.get("iou")),
        ("每帧虚警", detection.get("false_positive_per_frame")),
        ("匹配帧", detection.get("matched")),
        ("漏检帧", detection.get("missed")),
        ("虚警总数", detection.get("false_positive")),
        ("帧数", frame_count if frame_count is not None else len(detection.get("frames") or []) or None),
    ]


def tracking_summary(metrics, tracks):
    """Dynamic velocity / continuity values from tracks.json and metrics.json."""
    tracking = (metrics or {}).get("tracking") or {}
    shadow = (tracks or {}).get("shadow_velocity_mps")
    speed = None
    if isinstance(shadow, (list, tuple)) and len(shadow) == 2:
        speed = float(np.linalg.norm(np.asarray(shadow, dtype=float)))
    return {
        "表观阴影速度 (m/s)": speed,
        "阴影速度 (px/s)": (float(np.linalg.norm(np.asarray(tracks["velocity_px_s"], dtype=float)))
                          if isinstance((tracks or {}).get("velocity_px_s"), (list, tuple)) else None),
        "真实阴影速度 (px/s)": (float(np.linalg.norm(np.asarray(tracking["true_shadow_velocity_px_s"], dtype=float)))
                            if isinstance(tracking.get("true_shadow_velocity_px_s"), (list, tuple)) else None),
        "速度误差 (m/s)": tracking.get("velocity_error_mps"),
        "轨迹连续性": tracking.get("continuity"),
        "定位 RMSE (px)": tracking.get("location_rmse_px"),
    }


def per_frame_speed(rows):
    """Apparent per-frame speed (px/s) computed from consecutive track points."""
    speeds, indices = [], []
    previous = None
    for row in rows:
        x, y, time = row.get("x_px"), row.get("y_px"), row.get("time_s")
        if x is None or y is None or time is None:
            previous = None
            speeds.append(None)
            indices.append(row["frame"])
            continue
        if previous is not None and time > previous[2]:
            speeds.append(float(np.hypot(x - previous[0], y - previous[1]) / (time - previous[2])))
        else:
            speeds.append(None)
        indices.append(row["frame"])
        previous = (x, y, time)
    return indices, speeds


def archive_tables(repo_root=REPO_ROOT):
    """Existing archived comparison tables only."""
    root = Path(repo_root)
    return [(label, root / relative) for label, relative in ARCHIVE_TABLES
            if (root / relative).is_file()]


def snl_model_info():
    return read_json_safe(SNL_MODEL) or {}


def preview_videos(runs_root):
    root = Path(runs_root)
    found = []
    for relative in SNL_PREVIEWS:
        for candidate in (root / relative, REPO_ROOT / "runs" / relative):
            if candidate.is_file() and str(candidate) not in found:
                found.append(str(candidate))
                break
    return found


def live_dirs(runs_root):
    root = Path(runs_root)
    return {name: (root / name if (root / name).is_dir() else None) for name in PREFERRED_LIVE}
