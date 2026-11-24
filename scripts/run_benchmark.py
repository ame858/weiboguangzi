#!/usr/bin/env python3
"""Unified CLI entrypoint for Video-SAR v2.0 simulation & benchmark evaluation.

Examples:
    python3 scripts/run_benchmark.py --mode smoke --output-dir outputs/smoke
    python3 scripts/run_benchmark.py --mode simulate --mwp-enabled --output-dir outputs/mwp_ablation
    python3 scripts/run_benchmark.py --mode eval-vsmb --data-root /path/to/VSMB --output-dir outputs/vsmb
    python3 scripts/run_benchmark.py --mode eval-snl --data-root /path/to/SNL --output-dir outputs/snl
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is importable when executed directly as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from videosar.config import DatasetConfig, MWPConfig, SimulationConfig, VideoSARConfig
from videosar.datasets.snl import SNLDataset
from videosar.datasets.vsmb import VSMBDataset
from videosar.experiment import ExperimentRunner
from videosar.simulation import VideoSARSimulator
from videosar.tracker import ByteTrackLite, detect_moving_shadows_baseline


def _run_tracker_on_detections(detections, num_frames: int):
    tracker = ByteTrackLite()
    by_frame = {}
    for d in detections:
        by_frame.setdefault(d.frame_id, []).append(d)
    tracked = []
    for fid in range(1, num_frames + 1):
        tracked.extend(tracker.update(fid, by_frame.get(fid, [])))
    return tracked


def main() -> int:
    parser = argparse.ArgumentParser(description="Video-SAR v2.0 Benchmark & Simulation CLI")
    parser.add_argument(
        "--mode",
        choices=["smoke", "simulate", "eval-vsmb", "eval-snl"],
        default="smoke",
    )
    parser.add_argument("--data-root", type=str, default="data/vsmb")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark_run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--mwp-enabled", action="store_true")
    parser.add_argument("--bg-image", type=str, default=None)
    args = parser.parse_args()

    cfg = VideoSARConfig(
        experiment_name=f"videosar_{args.mode}",
        seed=args.seed,
        output_dir=args.output_dir,
        mwp=MWPConfig(enabled=args.mwp_enabled),
        simulation=SimulationConfig(
            height=args.resolution,
            width=args.resolution,
            num_frames=args.num_frames,
            seed=args.seed,
            background_image_path=args.bg_image,
        ),
        dataset=DatasetConfig(root_dir=args.data_root),
    )
    cfg.validate()

    runner = ExperimentRunner(cfg, run_dir=args.output_dir)

    if args.mode in {"smoke", "simulate"}:
        sim = VideoSARSimulator(cfg.simulation, cfg.radar, cfg.mwp)
        seq = sim.generate_sequence(sequence_id="sim_ka_curved_01", seed=args.seed)
        if args.mode == "simulate":
            seq.export_mot_directory(Path(args.output_dir) / "simulated_mot")

        raw_dets = detect_moving_shadows_baseline(seq.frames)
        tracked = _run_tracker_on_detections(raw_dets, num_frames=seq.frames.shape[0])
        res = runner.evaluate_split(
            split_name="synthetic_ka_baseline" + ("_mwp" if args.mwp_enabled else ""),
            domain="synthetic",
            predictions=raw_dets,
            ground_truths=seq.ground_truths,
            num_frames=seq.frames.shape[0],
            tracked_predictions=tracked,
        )
        artifacts = runner.save_artifacts()
        print(
            f"[{args.mode}] mAP@0.5={res['map_50']:.4f} | mAP@0.3={res['map_30']:.4f} | "
            f"FA/frame={res['fa_per_frame_50']:.3f} | MOTA={res['mot']['mota']:.4f} | "
            f"IDF1={res['mot']['idf1']:.4f} | HOTA={res['mot']['hota']:.4f}"
        )
        print(f"Saved artifacts to: {artifacts['metrics'].parent}")
        return 0

    if args.mode == "eval-vsmb":
        ds = VSMBDataset(root_dir=args.data_root, split="test", tile_size=1024, tile_overlap=0)
        print(f"Loaded VSMB test split: {len(ds.sequence_ids)} sequences, {len(ds)} windows.")
        return 0

    if args.mode == "eval-snl":
        ds = SNLDataset(root_dir=args.data_root, split="test")
        print(f"Loaded SNL test split: {len(ds.original_frame_indices)} frames, {len(ds)} windows.")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
