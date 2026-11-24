"""Export strictly causal 3-channel SNL Video SAR dataset in YOLO format."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snl_inputs import DEFAULT_OUTPUT, TOTAL, WIDTH, HEIGHT, load_gray, load_labels  # noqa: E402

# prospective 官方划分（严禁修改，封存验收段禁止进入此表）
SPLITS: Dict[str, Tuple[int, int]] = {
    "train": (2, 296),   # 295 帧
    "cal": (310, 323),   # 14 帧
    "dev": (332, 379),   # 48 帧
}


def load_snl_frames_and_boxes(
    labels_dir: Path,
) -> Tuple[np.ndarray, Dict[int, List[Tuple[float, float, float, float]]]]:
    """Load the real SNL frames and annotations through the repository's own readers.

    load_gray returns a (899, 720, 660) uint8 memmap of the native radar crop.
    load_labels validates annotation provenance/frame alignment and returns
    annotation-index keys with [x1, y1, x2, y2] boxes already scaled to 660x720.
    It searches labels_dir, DEFAULT_OUTPUT/labels, the public inputs and the
    independent relabel directory, which is required because the dev frames
    are not all present under prospective/labels.
    """
    gray = load_gray(labels_dir.parent, "ffmpeg")
    ids = [i for start, end in SPLITS.values() for i in range(start, end + 1)]
    labels, _, missing = load_labels(labels_dir, ids, gray, allow_partial=True)
    if missing:
        raise FileNotFoundError(f"Missing annotated frames: {missing}")
    if gray.shape != (TOTAL, HEIGHT, WIDTH) or gray.dtype != np.uint8:
        raise ValueError(f"Unexpected frame cache: {gray.shape} {gray.dtype}")
    return np.asarray(gray), {index: [tuple(map(float, box)) for box in boxes]
                              for index, boxes in labels.items()}


def _compute_causal_raw_channels(
    frames: np.ndarray,
    mode: str = "causal-motion",
    ema_alpha: float = 0.15,
    local_ksize: int = 15,
) -> np.ndarray:
    """严格因果逐帧计算浮点 3 通道特征，形状 (N, H, W, 3)。"""
    n_frames, height, width = frames.shape
    out = np.zeros((n_frames, height, width, 3), dtype=np.float32)

    if mode == "3frame":
        for t in range(n_frames):
            out[t, :, :, 0] = frames[max(0, t - 2)]
            out[t, :, :, 1] = frames[max(0, t - 1)]
            out[t, :, :, 2] = frames[t]
        return out

    if mode != "causal-motion":
        raise ValueError(f"Unsupported mode: {mode}")

    bg = frames[0].astype(np.float32).copy()
    for t in range(n_frames):
        cur = frames[t].astype(np.float32)
        ch0 = cur
        ch1 = np.maximum(0.0, bg - cur)
        bg = (1.0 - ema_alpha) * bg + ema_alpha * cur
        local_mean = cv2.GaussianBlur(cur, (local_ksize, local_ksize), 0)
        local_dark = np.maximum(0.0, local_mean - cur)
        short_diff = np.maximum(0.0, frames[max(0, t - 2)].astype(np.float32) - cur)
        ch2 = 0.5 * local_dark + 0.5 * short_diff

        out[t, :, :, 0] = ch0
        out[t, :, :, 1] = ch1
        out[t, :, :, 2] = ch2
    return out


def normalize_with_train_stats(
    raw_3ch: np.ndarray,
    train_range: Tuple[int, int],
    mode: str,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """仅使用 train 帧统计分位数，将 3 通道特征映射到 uint8 [0, 255]。"""
    t0, t1 = train_range
    train_slice = raw_3ch[t0:t1 + 1]
    lo_list: List[float] = []
    hi_list: List[float] = []
    u8 = np.zeros_like(raw_3ch, dtype=np.uint8)

    for c in range(3):
        vals = train_slice[..., c]
        if mode == "causal-motion" and c in (1, 2):
            lo, hi = 0.0, float(np.percentile(vals, 99.5))
        else:
            lo, hi = float(np.percentile(vals, 1.0)), float(np.percentile(vals, 99.0))
        if hi <= lo + 1e-6:
            hi = lo + 1.0
        lo_list.append(lo)
        hi_list.append(hi)
        scaled = np.clip((raw_3ch[..., c] - lo) / (hi - lo), 0.0, 1.0)
        u8[..., c] = np.round(scaled * 255.0).astype(np.uint8)

    return u8, {"lo": lo_list, "hi": hi_list}


def verify_strict_causality(frames: np.ndarray, mode: str, check_frame: int = 335) -> None:
    """因果自检：破坏 check_frame 之后的所有未来帧，第 check_frame 帧输出必须逐像素一致。"""
    sub_a = frames[:check_frame + 5].astype(np.float32)
    sub_b = sub_a.copy()
    sub_b[check_frame + 1:] = 9999.0
    feat_a = _compute_causal_raw_channels(sub_a, mode=mode)
    feat_b = _compute_causal_raw_channels(sub_b, mode=mode)
    if not np.array_equal(feat_a[check_frame], feat_b[check_frame]):
        raise RuntimeError(f"Causality check FAILED at frame {check_frame}!")


def xyxy_to_yolo_line(
    box: Tuple[float, float, float, float], width: int, height: int
) -> str:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width), float(x1)))
    y1 = max(0.0, min(float(height), float(y1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y2 = max(0.0, min(float(height), float(y2)))
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0.5 or bh <= 0.5:
        return ""
    cx = (x1 + x2) * 0.5 / width
    cy = (y1 + y2) * 0.5 / height
    return f"0 {cx:.6f} {cy:.6f} {bw / width:.6f} {bh / height:.6f}"


def export_dataset(
    labels_dir: Path,
    output_dir: Path,
    mode: str = "causal-motion",
    ema_alpha: float = 0.15,
) -> None:
    frames, boxes_by_frame = load_snl_frames_and_boxes(labels_dir)
    max_needed = max(end for _, end in SPLITS.values())
    if len(frames) <= max_needed:
        raise ValueError(f"Expected > {max_needed} frames, got {len(frames)}")

    verify_strict_causality(frames, mode=mode)
    raw_3ch = _compute_causal_raw_channels(frames[:max_needed + 1], mode=mode, ema_alpha=ema_alpha)
    u8_3ch, norm_stats = normalize_with_train_stats(raw_3ch, SPLITS["train"], mode=mode)

    _, height, width = frames.shape
    counts: Dict[str, int] = {}
    box_counts: Dict[str, int] = {}

    for split_name, (start_f, end_f) in SPLITS.items():
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        n_imgs = n_boxes = 0
        for fidx in range(start_f, end_f + 1):
            stem = f"frame_{fidx:04d}"
            bgr = cv2.cvtColor(u8_3ch[fidx], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_dir / f"{stem}.png"), bgr)
            lines = [line for box in boxes_by_frame.get(fidx, [])
                     if (line := xyxy_to_yolo_line(box, width=width, height=height))]
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            n_imgs += 1
            n_boxes += len(lines)

        counts[split_name] = n_imgs
        box_counts[split_name] = n_boxes

    assert counts == {"train": 295, "cal": 14, "dev": 48}, f"Unexpected split counts: {counts}"

    yaml_text = (
        f"path: {output_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/cal\n"
        f"test: images/dev\n"
        f"names:\n"
        f"  0: moving_shadow\n"
    )
    (output_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    meta = {
        "mode": mode,
        "ema_alpha": ema_alpha,
        "height": height,
        "width": width,
        "splits": SPLITS,
        "frame_counts": counts,
        "box_counts": box_counts,
        "norm_stats_train_only": norm_stats,
        "gt_boxes_by_frame": {
            str(f): [list(b) for b in boxes_by_frame.get(f, [])]
            for split_range in SPLITS.values()
            for f in range(split_range[0], split_range[1] + 1)
        },
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Export complete:", json.dumps({"counts": counts, "box_counts": box_counts}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["causal-motion", "3frame"], default="causal-motion")
    parser.add_argument("--ema-alpha", type=float, default=0.15)
    args = parser.parse_args()
    export_dataset(args.labels_dir, args.output_dir, mode=args.mode, ema_alpha=args.ema_alpha)


if __name__ == "__main__":
    main()
