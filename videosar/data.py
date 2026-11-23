import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


@contextmanager
def run_directory(output, command, config):
    now = datetime.now(timezone.utc)
    root = Path(output) if output else Path("runs") / (
        f"{command}-{now:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    )
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    info = {"command": command, "started_utc": now.isoformat(), "status": "running"}
    write_json(root / "run.json", info)
    try:
        write_json(root / "config.json", config)
        yield root
    except BaseException as exc:
        info.update(status="failed", error=str(exc))
        raise
    else:
        info["status"] = "completed"
    finally:
        info["elapsed_s"] = perf_counter() - start
        write_json(root / "run.json", info)


def save_sequence(root, frames, masks, metadata, geometry=None):
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "frames.npy", np.asarray(frames, dtype=np.float32))
    np.save(root / "masks.npy", np.asarray(masks, dtype=np.uint8))
    if geometry is not None:
        np.save(root / "geometric_masks.npy", np.asarray(geometry, dtype=np.uint8))
    write_json(root / "metadata.json", metadata)


def load_observations(root):
    root = Path(root)
    frames = np.load(root / "frames.npy", allow_pickle=False)
    meta = read_json(root / "metadata.json")
    if (frames.ndim != 3 or min(frames.shape, default=0) <= 0 or
            frames.dtype.kind not in "fiu" or not np.isfinite(frames).all() or
            np.any(frames < 0)):
        raise ValueError("frames.npy must contain finite nonnegative numeric data in (t, y, x) order")
    interval = meta.get("frame_interval_s")
    if not isinstance(interval, (int, float)) or not np.isfinite(interval) or interval <= 0:
        raise ValueError("frame_interval_s must be positive")
    times = np.asarray(meta.get("timestamps_s", np.arange(len(frames)) * interval), dtype=float)
    if times.shape != (len(frames),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps_s must contain one increasing finite value per frame")
    scale = meta.get("pixel_size_m")
    if scale is not None and (not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0):
        raise ValueError("pixel_size_m must be positive when present")
    # The detector gets no target positions, visibility labels or geometry derived from truth.
    return frames.astype(np.float32, copy=False), times, scale


def load_masks(path, expected_shape):
    masks = np.load(path, allow_pickle=False)
    if masks.shape != expected_shape or masks.dtype.kind not in "bui" or np.any((masks != 0) & (masks != 1)):
        raise ValueError(f"{path} must contain binary masks with shape {expected_shape}")
    return masks.astype(bool, copy=False)


def save_receipt(root, source, **extra):
    write_json(root / "source.json", {"sequence": str(Path(source).resolve()), **extra})


def verify_receipt(root, source):
    receipt = read_json(Path(root) / "source.json")
    if Path(receipt["sequence"]).resolve() != Path(source).resolve():
        raise ValueError(f"{root} was produced for a different sequence")
    return receipt
