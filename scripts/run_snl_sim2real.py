#!/usr/bin/env python3
"""Sim-to-Real contrast on the annotated SNL subset (train 150..399, test 416..527).

Three regimes are compared on the SAME frozen real test block (SNL 416..527):

  A) ``synth_only``   -- z_thresh tuned on simulated training sequences only
                         (no real SNL frames). Measures pure sim-to-real transfer.
  B) ``snl_train``    -- z_thresh tuned on real SNL train frames 150..399 (GT
                         available). This is the "SNL training only" baseline.
  C) ``snl_plus_aug`` -- z_thresh tuned on real SNL train frames PLUS the
                         real-background-injected simulated sequences (built with
                         --snl-min-train-frame 150 --snl-max-train-frame 399,
                         so backgrounds never leak past the train boundary).

The causal temporal-median baseline is used throughout; only its detection
threshold is fitted, on each regime's own training data. All three are then
evaluated on the frozen real test block. Output: contrast table + full
ExperimentRunner artifacts, each carrying git_commit / seed / frame ranges.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from videosar.config import VideoSARConfig
from videosar.datasets.snl import SNLDataset
from videosar.experiment import ExperimentRunner
from videosar.metrics import BoxDetection, evaluate_detection_and_froc
from videosar.tracker import detect_moving_shadows_baseline

Z_GRID = [1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 4.0, 5.0]


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "uncommitted"


def _load_seq_frames(seq_dir: Path, n_frames: int) -> np.ndarray:
    frames = []
    for t in range(1, n_frames + 1):
        img = cv2.imread(str(seq_dir / "img1" / f"{t:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        frames.append(img.astype(np.float32) / 255.0)
    return np.stack(frames, axis=0)


def _load_sim_gt(seq_dir: Path) -> List[BoxDetection]:
    gt: List[BoxDetection] = []
    path = seq_dir / "gt" / "gt.txt"
    if not path.exists():
        return gt
    with path.open(encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 6:
                continue
            try:
                fid, tid = int(row[0]), int(row[1])
                x, y, w, h = map(float, row[2:6])
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            gt.append(BoxDetection(fid, x, y, x + w, y + h, 1.0, 1, tid))
    return gt


def _mean_ap_over(sequences: List[Tuple[np.ndarray, List[BoxDetection]]], z: float) -> float:
    aps = []
    for frames, gt in sequences:
        if frames.shape[0] < 2:
            continue
        dets = detect_moving_shadows_baseline(frames, z_thresh=z)
        res = evaluate_detection_and_froc(dets, gt, frames.shape[0], iou_threshold=0.5)
        aps.append(float(res["ap"]))
    return float(np.mean(aps)) if aps else 0.0


# Tuning only needs a stable threshold, so cap each sequence at a causal prefix.
# This keeps z-selection reproducible while avoiding the O(T^2) full-length cost.
TUNE_PREFIX_FRAMES = 64


def tune_z(sequences: List[Tuple[np.ndarray, List[BoxDetection]]]) -> Tuple[float, Dict[str, float]]:
    """Pick z_thresh maximising mean mAP@0.5 on a causal prefix of each sequence."""
    if not sequences:
        return Z_GRID[2], {str(z): 0.0 for z in Z_GRID}
    clipped: List[Tuple[np.ndarray, List[BoxDetection]]] = []
    for frames, gt in sequences:
        n = min(frames.shape[0], TUNE_PREFIX_FRAMES)
        kept_gt = [b for b in gt if 1 <= b.frame_id <= n]
        clipped.append((frames[:n], kept_gt))
    grid: Dict[str, float] = {}
    for z in Z_GRID:
        grid[str(z)] = _mean_ap_over(clipped, z)
        print(f"    [tune] z={z}: mean mAP@0.5={grid[str(z)]:.4f}", flush=True)
    best = max(grid, key=lambda k: (grid[k], -float(k)))
    return float(best), grid


def run_contrast(args: argparse.Namespace) -> Dict[str, object]:
    root = REPO_ROOT
    manifest_path = Path(args.dataset_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_root = manifest_path.parent

    # Simulated training sequences (real-bg injected) -- backgrounds from 150..399 only.
    train_recs = manifest["splits"]["id_train"]
    sim_dirs = [dataset_root / r["rel_path"] for r in train_recs]
    num_frames = int(train_recs[0]["num_frames"])
    sim_sequences = [
        (_load_seq_frames(d, num_frames), _load_sim_gt(d))
        for d in sim_dirs
    ]
    sim_sequences = [(f, g) for f, g in sim_sequences if f.shape[0] >= 2 and g]

    # Real SNL training frames 150..399 (GT remapped to 1..250 internally).
    snl_kw = dict(
        train_start_frame=args.train_start,
        train_end_frame=args.train_end,
        guard_gap_frames=args.guard_gap,
        test_end_frame=args.test_end,
    )
    snl_train_ds = SNLDataset(args.snl_root, split="train", **snl_kw)
    snl_train_frames = np.stack(
        [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
         for p in snl_train_ds.sequences[0].frame_paths], axis=0
    )
    snl_train_gt = [
        b for boxes in snl_train_ds.sequences[0].annotations_by_frame.values() for b in boxes
    ]
    snl_sequences = [(snl_train_frames, snl_train_gt)]

    # A = sim only; B = SNL real only; C = SNL real + real-bg-augmented sim.
    z_synth, grid_synth = tune_z(sim_sequences)
    z_snl, grid_snl = tune_z(snl_sequences)
    z_aug, grid_aug = tune_z(snl_sequences + sim_sequences)

    # Frozen real test block.
    test_ds = SNLDataset(args.snl_root, split="test", **snl_kw)
    test_frames = np.stack(
        [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
         for p in test_ds.sequences[0].frame_paths], axis=0
    )
    test_gt = [b for boxes in test_ds.sequences[0].annotations_by_frame.values() for b in boxes]
    test_frame_ids = test_ds.original_frame_indices
    n_test = test_frames.shape[0]

    cfg = VideoSARConfig(experiment_name="snl_sim2real", seed=args.seed, output_dir=str(args.output_dir))
    runner = ExperimentRunner(cfg, run_dir=args.output_dir)

    grid_map = {
        "synth_only": grid_synth,
        "snl_train": grid_snl,
        "snl_plus_aug": grid_aug,
    }
    z_map = {"synth_only": z_synth, "snl_train": z_snl, "snl_plus_aug": z_aug}
    results: Dict[str, Dict[str, object]] = {}
    for regime, z in z_map.items():
        dets = detect_moving_shadows_baseline(test_frames, z_thresh=z)
        res = runner.evaluate_split(
            split_name=f"snl_test_{test_frame_ids[0]}_{test_frame_ids[-1]}_{regime}",
            domain="real",
            predictions=dets,
            ground_truths=test_gt,
            num_frames=n_test,
            condition_tag=regime,
        )
        results[regime] = {
            "z_thresh": z,
            "train_tune_map_grid": grid_map[regime],
            "map_50": res["map_50"],
            "map_30": res["map_30"],
            "fa_per_frame": res["fa_per_frame_50"],
            "recall_50": res["recall_50"],
            "num_gt": res["num_gt"],
            "num_pred": res["num_pred"],
        }

    artifacts = runner.save_artifacts()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gain_aug_vs_snl = float(results["snl_plus_aug"]["map_50"]) - float(results["snl_train"]["map_50"])
    gain_snl_vs_synth = float(results["snl_train"]["map_50"]) - float(results["synth_only"]["map_50"])
    report = {
        "experiment": "SNL Sim-to-Real gain contrast (annotated subset)",
        "git_commit": _git_commit(),
        "seed": args.seed,
        "dataset_version": manifest.get("dataset_version"),
        "train_frame_range": [args.train_start, args.train_end],
        "guard_gap_frames": args.guard_gap,
        "guard_frame_range": [args.train_end + 1, args.train_end + args.guard_gap],
        "test_frame_range": [test_frame_ids[0], test_frame_ids[-1]],
        "test_num_frames": n_test,
        "test_num_gt": len(test_gt),
        "snl_background_window": manifest.get("snl_train_frame_window"),
        "regimes": results,
        "map50_gain_snl_vs_synth": gain_snl_vs_synth,
        "map50_gain_aug_vs_snl": gain_aug_vs_snl,
        "artifacts": {
            k: str(Path(v).resolve().relative_to(root)) if Path(v).resolve().is_relative_to(root)
            else str(v)
            for k, v in artifacts.items()
        },
        "note": "z_thresh tuned per regime on its own training data (synth sim / real SNL "
                "150..399 / real SNL + real-bg-augmented sim), evaluated on frozen real "
                "SNL 416..527. Backgrounds restricted to frames 150..399.",
    }
    (out_dir / "sim2real_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (out_dir / "sim2real_contrast.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["regime", "z_thresh", "map_50", "map_30", "fa_per_frame", "recall_50", "num_gt", "num_pred"])
        for regime, r in results.items():
            w.writerow([regime, r["z_thresh"], f"{float(r['map_50']):.4f}",
                        f"{float(r['map_30']):.4f}", f"{float(r['fa_per_frame']):.4f}",
                        f"{float(r['recall_50']):.4f}", r["num_gt"], r["num_pred"]])
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="SNL Sim-to-Real gain contrast")
    p.add_argument("--dataset-manifest", default="data/proposal_v2_snl399/dataset_manifest.json")
    p.add_argument("--snl-root", default="E:/workspace/weiboguangzi-refactor/data/snl_v2")
    p.add_argument("--train-start", type=int, default=150)
    p.add_argument("--train-end", type=int, default=399)
    p.add_argument("--guard-gap", type=int, default=16)
    p.add_argument("--test-end", type=int, default=527)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="deliverables/snl-sim2real-contrast")
    args = p.parse_args()

    report = run_contrast(args)
    print(f"[sim2real] test frames {report['test_frame_range']} | num_gt={report['test_num_gt']}")
    for regime, r in report["regimes"].items():
        print(f"  {regime}: z={r['z_thresh']} mAP50={float(r['map_50']):.4f} "
              f"FA/frame={float(r['fa_per_frame']):.4f} recall={float(r['recall_50']):.4f}")
    print(f"[sim2real] gain snl-synth={report['map50_gain_snl_vs_synth']:+.4f} | "
          f"aug-snl={report['map50_gain_aug_vs_snl']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
