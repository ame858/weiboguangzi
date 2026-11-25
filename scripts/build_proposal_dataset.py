#!/usr/bin/env python3
"""Build the proposal-controlled Video-SAR dataset (120 sequences / 3000 frames).

Six strictly disjoint splits (6:2:2 in-distribution + three OOD arms) with
per-sequence mask / kinematics / MOT truth exported by ``VideoSARSimulator``.
Two anti-leakage rules are enforced by construction:

* SNL real empty-road backgrounds are read ONLY from the configured train
  window ``[--snl-min-train-frame, --snl-max-train-frame]`` (default 1..600).
  Frames beyond the max are never opened; pass the train/test boundary (e.g.
  399 for the annotated 150..527 subset) to prevent background leakage.
* ``train_norm_stats`` is computed ONLY from id_train sequences; val/test/OOD
  pixels never contribute to mean / std / p01 / p99.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from videosar.config import MWPConfig, RadarConfig, SimulationConfig
from videosar.simulation import VideoSARSimulator

SPLIT_SPECS: List[Tuple[str, int, int, str]] = [
    # (split name, full-mode groups, smoke-mode groups, condition tag)
    ("id_train", 54, 5, "normal"),
    ("id_val", 18, 2, "normal"),
    ("id_test", 18, 2, "normal"),
    ("ood_speed", 10, 1, "unseen_speed"),
    ("ood_background", 10, 1, "dense_clutter"),
    ("ood_mwp_noise", 10, 1, "strong_speckle_mwp"),
]


def _config_for_split(split: str, resolution: int, num_frames: int, seed: int) -> Tuple[SimulationConfig, RadarConfig, MWPConfig]:
    """Return sim / radar / MWP configs for one split's condition arm."""
    sim_cfg = SimulationConfig(
        height=resolution,
        width=resolution,
        num_frames=num_frames,
        seed=seed,
    )
    radar_cfg = RadarConfig()
    mwp = MWPConfig(enabled=False)
    if split == "ood_speed":
        sim_cfg.min_speed_mps = 1.0
        sim_cfg.max_speed_mps = 25.0
        sim_cfg.enable_stop_and_go = True
    elif split == "ood_background":
        sim_cfg.num_static_distractors = 14
        sim_cfg.shadow_contrast_range = (0.10, 0.22)
    elif split == "ood_mwp_noise":
        radar_cfg.n_looks = 1.5
        mwp = MWPConfig(enabled=True)
    return sim_cfg, radar_cfg, mwp


def _collect_real_backgrounds(
    snl_dir: str | None,
    rng: np.random.Generator,
    min_train_frame: int = 1,
    max_train_frame: int = 600,
) -> List[np.ndarray]:
    """Extract up to 10 time-median empty-road backgrounds from SNL train frames.

    Only frames in ``[min_train_frame, max_train_frame]`` are ever read. The
    caller sets the upper bound to the train/test boundary (default 600, or
    399 for the annotated-subset contrast) so test frames are excluded before
    any I/O. Each group of 100 train frames contributes one background.
    """
    if not snl_dir:
        return []
    root = Path(snl_dir)
    frames_dir = root / "frames" if (root / "frames").is_dir() else root
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    candidates: List[Tuple[int, Path]] = []
    for p in frames_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        if not digits:
            continue
        idx = int(digits)
        if idx < min_train_frame or idx > max_train_frame:  # hard guard
            continue
        candidates.append((idx, p))
    candidates.sort(key=lambda x: x[0])
    if not candidates:
        return []

    backgrounds: List[np.ndarray] = []
    for start in range(0, len(candidates), 100):
        group = [p for idx, p in candidates[start : start + 100]]
        if len(group) < 8:
            continue
        stack = []
        for p in group[::4]:  # subsample for speed; still within train frames
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                stack.append(img.astype(np.float32) / 255.0)
        if not stack:
            continue
        median = np.median(np.stack(stack), axis=0)
        backgrounds.append(np.clip(median, 0.05, 0.95).astype(np.float32))
    return backgrounds


def _write_temp_background(bg: np.ndarray, path: Path) -> str:
    cv2.imwrite(str(path), np.clip(bg * 255.0, 0, 255).astype(np.uint8))
    return str(path)


def build_dataset(args: argparse.Namespace) -> Dict[str, object]:
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    resolution = args.resolution
    if resolution < 256:
        raise ValueError("--resolution must be >= 256 (legacy 96x64 regime forbidden).")

    rng = np.random.default_rng(args.seed)
    # getattr keeps direct Namespace construction (tests) backward-compatible.
    min_train_frame = int(getattr(args, "snl_min_train_frame", 1))
    max_train_frame = int(getattr(args, "snl_max_train_frame", 600))
    if min_train_frame < 1 or max_train_frame < min_train_frame:
        raise ValueError(
            f"Invalid SNL train frame window {min_train_frame}..{max_train_frame}."
        )
    real_bgs = _collect_real_backgrounds(
        args.snl_train_frames_dir, rng,
        min_train_frame=min_train_frame, max_train_frame=max_train_frame,
    )
    # Record the exact train-only window actually used for background extraction.
    used_frame_ceiling = max_train_frame if real_bgs else 0
    splits_manifest: Dict[str, List[Dict[str, object]]] = {name: [] for name, _, _, _ in SPLIT_SPECS}
    seq_counter = 0

    for split_name, n_full, n_smoke, condition_tag in SPLIT_SPECS:
        n_groups = n_smoke if args.smoke_scale else n_full
        for gi in range(n_groups):
            seq_counter += 1
            seed = args.seed * 100000 + seq_counter * 7919
            num_frames = 8 if args.smoke_scale else args.num_frames
            sim_cfg, radar_cfg, mwp_cfg = _config_for_split(split_name, resolution, num_frames, seed)

            # Real background for 70% of id_train groups when available.
            use_real = bool(real_bgs) and split_name == "id_train" and (gi % 10 < 7)
            tmp_bg_path = None
            if use_real:
                bg = real_bgs[gi % len(real_bgs)]
                tmp_bg_path = out_root / f"_bg_{split_name}_{gi:03d}.png"
                sim_cfg.background_image_path = _write_temp_background(bg, tmp_bg_path)

            sequence_id = f"{split_name}_{gi:03d}"
            seq = VideoSARSimulator(sim_cfg, radar_cfg=radar_cfg, mwp_cfg=mwp_cfg).generate_sequence(
                sequence_id=sequence_id, seed=seed
            )
            rel = seq.export_mot_directory(out_root / "sequences")
            if tmp_bg_path is not None and tmp_bg_path.exists():
                tmp_bg_path.unlink()

            splits_manifest[split_name].append({
                "sequence_id": sequence_id,
                "rel_path": str(rel.relative_to(out_root)).replace("\\", "/"),
                "seed": seed,
                "condition_tag": condition_tag,
                "num_frames": num_frames,
                "num_gt": len(seq.ground_truths),
                "real_bg_used": use_real,
            })

    # Norm stats from id_train ONLY.
    train_pixels: List[np.ndarray] = []
    for rec in splits_manifest["id_train"]:
        seq_dir = out_root / str(rec["rel_path"])
        for img_path in sorted((seq_dir / "img1").glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                train_pixels.append(img.astype(np.float32).ravel())
    all_train = np.concatenate(train_pixels) if train_pixels else np.zeros(1, dtype=np.float32)
    stats = {
        "mean": float(np.mean(all_train)),
        "std": float(np.std(all_train)),
        "p01": float(np.percentile(all_train, 1)),
        "p99": float(np.percentile(all_train, 99)),
    }

    all_ids = [r["sequence_id"] for recs in splits_manifest.values() for r in recs]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("Sequence ID leakage detected across splits.")

    manifest = {
        "dataset_version": "proposal_v2",
        "total_sequences": sum(len(recs) for recs in splits_manifest.values()),
        "total_frames": sum(
            int(r["num_frames"]) for recs in splits_manifest.values() for r in recs
        ),
        "real_bg_used": bool(real_bgs),
        "snl_max_train_frame_used": used_frame_ceiling,
        "snl_min_train_frame_used": min_train_frame if real_bgs else 0,
        "snl_train_frame_window": [min_train_frame, max_train_frame],
        "train_norm_stats": stats,
        "splits": splits_manifest,
    }
    (out_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build proposal_v2 controlled Video-SAR dataset")
    parser.add_argument("--output-dir", default="data/proposal_v2")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snl-train-frames-dir", default=None)
    parser.add_argument(
        "--snl-min-train-frame", type=int, default=1,
        help="Lowest SNL frame index allowed for real-background extraction (1-based).",
    )
    parser.add_argument(
        "--snl-max-train-frame", type=int, default=600,
        help="Highest SNL frame index allowed for real-background extraction (1-based). "
             "Set to the train/test boundary (e.g. 399) to prevent background leakage.",
    )
    parser.add_argument("--smoke-scale", action="store_true")
    args = parser.parse_args()

    manifest = build_dataset(args)
    print(
        f"[proposal_v2] {manifest['total_sequences']} sequences / "
        f"{manifest['total_frames']} frames -> {args.output_dir}"
    )
    print(f"[proposal_v2] real_bg_used={manifest['real_bg_used']} "
          f"snl_train_frame_window={manifest['snl_train_frame_window']} "
          f"snl_max_train_frame_used={manifest['snl_max_train_frame_used']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
