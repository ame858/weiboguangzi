"""Read-only parsing of Video-SAR v2.0 benchmark artifacts for the web workbench.

A v2 run directory is any directory holding both ``config.json`` and
``metrics.json``.  The remaining two artifacts (``summary.csv`` and
``froc_curve.csv``) are parsed when present; nothing is hard-coded and missing
files yield empty collections rather than raising.

The two metric partitions -- ``real_blind_eval`` and ``synthetic_ablation`` --
are deliberately kept separate end to end.  This module never averages or
merges them.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from videosar_ui.scanning import numeric, read_csv_rows, read_json_safe

V2_MODES = ("smoke", "simulate", "eval-vsmb", "eval-snl")
V2_ARTIFACTS = ("config.json", "metrics.json", "summary.csv", "froc_curve.csv")

SUMMARY_COLUMNS = (
    "split_name",
    "domain",
    "map_50",
    "map_30",
    "fa_per_frame_50",
    "recall_at_1_fa",
    "mota",
    "idf1",
    "hota",
    "id_switches",
)
SUMMARY_NUMERIC = (
    "map_50", "map_30", "fa_per_frame_50", "recall_at_1_fa",
    "mota", "idf1", "hota",
)
FROC_COLUMNS = ("split_name", "domain", "fa_per_frame", "recall", "score")
FROC_NUMERIC = ("fa_per_frame", "recall", "score")

SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _mode_from_config(config: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    name = str(config.get("experiment_name") or metrics.get("experiment_name") or "")
    for mode in V2_MODES:
        if name == mode or name == f"videosar_{mode}" or name.endswith(f"_{mode}"):
            return mode
    return name or "unknown"


def _partitions(metrics: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    return (_as_dict(metrics.get("real_blind_eval")),
            _as_dict(metrics.get("synthetic_ablation")))


def is_v2_run(directory: Path) -> bool:
    """True when a directory holds both required v2.0 JSON artifacts."""
    path = Path(directory)
    return (path / "config.json").is_file() and (path / "metrics.json").is_file()


def list_v2_runs(outputs_root: Path) -> List[Dict[str, Any]]:
    """List v2 runs under ``outputs_root``, most recently modified first."""
    root = Path(outputs_root)
    if not root.is_dir():
        return []
    runs: List[Dict[str, Any]] = []
    root_resolved = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        if "config.json" not in filenames or "metrics.json" not in filenames:
            continue
        path = Path(dirpath)
        if path.resolve() == root_resolved:
            continue
        config = _as_dict(read_json_safe(path / "config.json"))
        metrics = _as_dict(read_json_safe(path / "metrics.json"))
        real, synthetic = _partitions(metrics)
        try:
            run_id = path.relative_to(root).as_posix()
        except ValueError:  # pragma: no cover - os.walk stays under root
            run_id = path.name
        mtime = max((path / "config.json").stat().st_mtime,
                    (path / "metrics.json").stat().st_mtime)
        timestamp = metrics.get("timestamp_utc")
        runs.append({
            "run_id": run_id,
            "path": str(path),
            "mode": _mode_from_config(config, metrics),
            "timestamp": str(timestamp) if timestamp else datetime.fromtimestamp(
                mtime, timezone.utc).isoformat(),
            "mtime": mtime,
            "has_real": bool(real),
            "has_synthetic": bool(synthetic),
        })
    runs.sort(key=lambda item: (-item["mtime"], item["run_id"]))
    return runs


def resolve_run_dir(outputs_root: Path, run_id: str) -> Optional[Path]:
    """Resolve a URL-supplied run id, rejecting traversal and non-runs.

    Returns ``None`` for absolute paths, drive letters, ``..`` segments, paths
    escaping ``outputs_root``, and directories without both v2 artifacts.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        return None
    relative = run_id.strip().replace("\\", "/")
    if not relative or relative.startswith("/") or ":" in relative or Path(relative).is_absolute():
        return None
    if any(part == ".." for part in relative.split("/")):
        return None
    root = Path(outputs_root).resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if is_v2_run(candidate) else None


def _summary_rows(path: Path) -> List[Dict[str, Any]]:
    _, raw_rows = read_csv_rows(path)
    rows: List[Dict[str, Any]] = []
    for raw in raw_rows:
        row: Dict[str, Any] = {key: raw.get(key) for key in SUMMARY_COLUMNS}
        for key in SUMMARY_NUMERIC:
            row[key] = numeric(row[key])
        switches = numeric(row.get("id_switches"))
        row["id_switches"] = int(switches) if switches is not None else None
        # Keep any additional columns written by the CLI (num_frames, etc.).
        for key, value in raw.items():
            if key is not None and key not in row:
                row[key] = value
        rows.append(row)
    return rows


def _froc_points(path: Path) -> List[Dict[str, Any]]:
    _, raw_rows = read_csv_rows(path)
    points: List[Dict[str, Any]] = []
    for raw in raw_rows:
        point: Dict[str, Any] = {key: raw.get(key) for key in FROC_COLUMNS}
        for key in FROC_NUMERIC:
            point[key] = numeric(point[key])
        points.append(point)
    return points


def load_v2_run(run_dir: Path) -> Dict[str, Any]:
    """Parse the four standard v2.0 artifacts of one run directory."""
    path = Path(run_dir)
    config = _as_dict(read_json_safe(path / "config.json"))
    metrics = _as_dict(read_json_safe(path / "metrics.json"))
    real, synthetic = _partitions(metrics)
    return {
        "run_id": path.name,
        "path": str(path),
        "mode": _mode_from_config(config, metrics),
        "experiment_name": metrics.get("experiment_name") or config.get("experiment_name"),
        "git_commit": metrics.get("git_commit"),
        "seed": metrics.get("seed"),
        "timestamp_utc": metrics.get("timestamp_utc"),
        # Kept verbatim and isolated: never merged with each other.
        "real_blind_eval": real,
        "synthetic_ablation": synthetic,
        "has_real": bool(real),
        "has_synthetic": bool(synthetic),
        "summary": _summary_rows(path / "summary.csv"),
        "froc_curve": _froc_points(path / "froc_curve.csv"),
        "config": config,
    }
