"""Comprehensive unit & end-to-end integration tests for Video-SAR v2.0 refactor."""

from __future__ import annotations

from pathlib import Path
import json
import math
import tempfile
import unittest
import cv2
import numpy as np

from videosar.config import (
    DatasetConfig,
    MWPConfig,
    RadarConfig,
    SimulationConfig,
    VideoSARConfig,
)
from videosar.data import verify_no_sequence_leakage
from videosar.datasets.base import compute_tile_grid, split_sequence_ids
from videosar.datasets.snl import SNLDataset
from videosar.datasets.vsmb import VSMBDataset
from videosar.experiment import ExperimentRunner
from videosar.metrics import (
    BoxDetection,
    evaluate_detection_and_froc,
    evaluate_mot,
)
from videosar.mwp import MicrowavePhotonicsChannel
from videosar.proposal_data import (
    LegacyProposalDataRemovedError,
    load_proposal_dataset,
)
from videosar.simulation import VideoSARSimulator
from videosar.tracker import ByteTrackLite, detect_moving_shadows_baseline


class TestVideoSARRefactor(unittest.TestCase):
    def test_legacy_proposal_data_is_disabled(self) -> None:
        with self.assertWarns(DeprecationWarning):
            with self.assertRaises(LegacyProposalDataRemovedError):
                load_proposal_dataset()

    def test_config_rejects_legacy_low_resolution_and_mixed_eval(self) -> None:
        with self.assertRaises(ValueError):
            SimulationConfig(height=96, width=64).validate()

        cfg = VideoSARConfig()
        cfg.evaluation.allow_mixed_synthetic_real = True
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_empty_scene_metrics_produce_no_nans(self) -> None:
        false_alarms = [
            BoxDetection(frame_id=1, x1=10, y1=10, x2=24, y2=24, score=0.9),
            BoxDetection(frame_id=3, x1=50, y1=50, x2=66, y2=66, score=0.7),
        ]
        det = evaluate_detection_and_froc(
            predictions=false_alarms,
            ground_truths=[],
            num_frames=10,
            iou_threshold=0.5,
        )
        self.assertTrue(det["empty_scene"])
        self.assertAlmostEqual(float(det["fa_per_frame"]), 0.2, places=6)
        self.assertFalse(math.isnan(float(det["ap"])))
        self.assertFalse(math.isnan(float(det["precision"])))

        mot = evaluate_mot(predictions=false_alarms, ground_truths=[])
        for v in mot.values():
            self.assertFalse(math.isnan(float(v)))

    def test_experiment_runner_forbids_mixed_domain_pooling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = VideoSARConfig(experiment_name="guard_test", output_dir=tmp)
            runner = ExperimentRunner(cfg, run_dir=Path(tmp) / "run1")
            box = BoxDetection(frame_id=1, x1=10, y1=10, x2=30, y2=30, score=0.9, track_id=1)

            runner.evaluate_split("vsmb_test", "real", [box], [box], num_frames=1)
            runner.evaluate_split("sim_ka", "synthetic", [box], [box], num_frames=1)

            with self.assertRaises(ValueError):
                runner.assert_no_mixed_pooling(["vsmb_test", "sim_ka"])

            with self.assertRaises(ValueError):
                runner.evaluate_split("bad", "mixed", [box], [box], num_frames=1)

            paths = runner.save_artifacts()
            payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
            self.assertIn("vsmb_test", payload["real_blind_eval"])
            self.assertIn("sim_ka", payload["synthetic_ablation"])
            self.assertTrue(paths["summary"].exists())
            self.assertTrue(paths["froc"].exists())

    def test_sequence_id_split_and_vsmb_snl_loaders(self) -> None:
        seq_ids = [f"vsmb_seq_{i:02d}" for i in range(10)]
        splits = split_sequence_ids(seq_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=7)
        counts = verify_no_sequence_leakage(splits["train"], splits["val"], splits["test"])
        self.assertEqual(sum(counts.values()), 10)

        tiles = compute_tile_grid(1024, 1024, tile_size=512, overlap=64)
        self.assertGreaterEqual(len(tiles), 4)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Create 4 mini VSMB sequences (640x640) with MOT gt.txt
            for s_idx in range(4):
                seq_dir = root / "vsmb" / f"seq_{s_idx:02d}"
                img_dir = seq_dir / "img1"
                gt_dir = seq_dir / "gt"
                img_dir.mkdir(parents=True)
                gt_dir.mkdir(parents=True)
                gt_rows = []
                for f in range(1, 8):
                    frame = np.full((640, 640), 120, dtype=np.uint8)
                    cv2.imwrite(str(img_dir / f"{f:06d}.png"), frame)
                    gt_rows.append(f"{f},1,{40 + f * 4},100,24,18,1,car,1.0")
                (gt_dir / "gt.txt").write_text("\n".join(gt_rows) + "\n", encoding="utf-8")

            ds_train = VSMBDataset(root / "vsmb", split="train", window_size=5, tile_size=512, tile_overlap=64)
            ds_test = VSMBDataset(root / "vsmb", split="test", window_size=5, tile_size=512, tile_overlap=64)
            verify_no_sequence_leakage(ds_train.sequence_ids, [], ds_test.sequence_ids)
            sample = ds_train[0]
            self.assertEqual(sample.frames.shape, (5, 512, 512))
            self.assertEqual(len(sample.frame_indices), 5)
            # Causal window check: strictly non-decreasing and ends at current frame
            self.assertEqual(sample.frame_indices, sorted(sample.frame_indices))

            # Create mini SNL directory (30 frames) with temporal guard gap
            snl_frames = root / "snl" / "frames"
            snl_frames.mkdir(parents=True)
            snl_gt = []
            for f in range(1, 31):
                cv2.imwrite(str(snl_frames / f"frame_{f:04d}.png"), np.full((320, 360), 100, dtype=np.uint8))
                snl_gt.append(f"{f},1,{30 + f * 2},120,16,12,1,1,1.0")
            (root / "snl" / "gt.txt").write_text("\n".join(snl_gt) + "\n", encoding="utf-8")

            snl_train = SNLDataset(root / "snl", split="train", train_end_frame=18, guard_gap_frames=4, window_size=5)
            snl_test = SNLDataset(root / "snl", split="test", train_end_frame=18, guard_gap_frames=4, window_size=5)
            self.assertEqual(max(snl_train.original_frame_indices), 18)
            self.assertEqual(min(snl_test.original_frame_indices), 23)
            self.assertTrue(len(snl_test) > 0)

    def test_eval_vsmb_writes_real_only_blind_metrics(self) -> None:
        """eval-vsmb logic must populate real_blind_eval and leave synthetic_ablation empty."""
        import importlib.util

        scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
        spec = importlib.util.spec_from_file_location("run_benchmark", scripts_dir / "run_benchmark.py")
        self.assertIsNotNone(spec and spec.loader)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vsmb_root = root / "vsmb"
            # 4 mini sequences so the 70/15/15 split yields a non-empty test partition.
            for s_idx in range(4):
                seq_dir = vsmb_root / f"seq_{s_idx:02d}"
                img_dir = seq_dir / "img1"
                gt_dir = seq_dir / "gt"
                img_dir.mkdir(parents=True)
                gt_dir.mkdir(parents=True)
                gt_rows = []
                for f in range(1, 9):
                    frame = np.full((128, 128), 100 + s_idx * 10, dtype=np.uint8)
                    cv2.imwrite(str(img_dir / f"{f:06d}.png"), frame)
                    gt_rows.append(f"{f},1,20,20,16,12,1,car,1.0")
                (gt_dir / "gt.txt").write_text("\n".join(gt_rows) + "\n", encoding="utf-8")

            cfg = VideoSARConfig(
                experiment_name="vsmb_blind_test",
                seed=7,
                output_dir=str(root / "out"),
                dataset=DatasetConfig(root_dir=str(vsmb_root), split_seed=7),
            )
            runner = ExperimentRunner(cfg, run_dir=root / "out")
            ds = VSMBDataset(root_dir=vsmb_root, split="test", tile_size=128, tile_overlap=0, split_seed=7)
            results, artifacts = mod.evaluate_real_dataset(ds, runner, "vsmb", "vsmb")

            self.assertGreater(len(results), 0)
            self.assertTrue(artifacts["metrics"].exists())
            payload = json.loads((root / "out" / "metrics.json").read_text(encoding="utf-8"))
            self.assertGreater(len(payload["real_blind_eval"]), 0)
            self.assertEqual(payload["synthetic_ablation"], {})
            for name in ("config", "summary", "froc"):
                self.assertTrue(artifacts[name].exists())

    def test_simulation_mwp_and_end_to_end_tracking(self) -> None:
        sim_cfg = SimulationConfig(height=256, width=256, num_frames=16, min_targets=3, max_targets=5, seed=123)
        sim_clean = VideoSARSimulator(sim_cfg, RadarConfig(), MWPConfig(enabled=False))
        seq_clean = sim_clean.generate_sequence("sim_clean", seed=123)
        self.assertEqual(seq_clean.frames.shape, (16, 256, 256))
        self.assertGreater(len(seq_clean.ground_truths), 0)
        self.assertEqual(len(seq_clean.static_distractors), sim_cfg.num_static_distractors)

        # Verify MWP channel measurably perturbs complex echoes & is reproducible
        mwp = MicrowavePhotonicsChannel(MWPConfig(enabled=True))
        iq = np.exp(1j * np.linspace(0, 8 * np.pi, 512))
        deg1, diag1 = mwp.apply(iq, rng=np.random.default_rng(99))
        deg2, _ = mwp.apply(iq, rng=np.random.default_rng(99))
        np.testing.assert_allclose(deg1, deg2)
        self.assertGreater(diag1["evm_rms"], 0.0)

        # End-to-end detection + ByteTrackLite + FROC/MOT
        dets = detect_moving_shadows_baseline(seq_clean.frames)
        tracker = ByteTrackLite()
        by_f = {}
        for d in dets:
            by_f.setdefault(d.frame_id, []).append(d)
        tracked = []
        for fid in range(1, seq_clean.frames.shape[0] + 1):
            tracked.extend(tracker.update(fid, by_f.get(fid, [])))

        det_res = evaluate_detection_and_froc(dets, seq_clean.ground_truths, num_frames=16, iou_threshold=0.3)
        mot_res = evaluate_mot(tracked, seq_clean.ground_truths, iou_threshold=0.3)
        self.assertGreater(float(det_res["ap"]), 0.0)
        self.assertGreaterEqual(float(mot_res["hota"]), 0.0)


class TestCausalEvaluationContract(unittest.TestCase):
    """Protocol rules carried over from the SNL research branch.

    These are invariants, not model-quality checks: no detector may read future
    frames, blocks must be split by time, and normalization statistics must come
    from the training partition only.
    """

    def test_detector_is_strictly_causal(self) -> None:
        """Changing any future frame must not alter detections on earlier frames."""
        rng = np.random.default_rng(58)
        frames = rng.integers(0, 170, (24, 32, 32), dtype=np.uint8)
        check_frame = 12  # 0-based; frames after this index are "future"

        baseline = detect_moving_shadows_baseline(frames)
        tampered = frames.copy()
        tampered[check_frame + 1:] = rng.integers(180, 255, tampered[check_frame + 1:].shape, dtype=np.uint8)
        after = detect_moving_shadows_baseline(tampered)

        def positions(dets):
            return sorted((d.frame_id, d.x1, d.y1, d.x2, d.y2, d.score) for d in dets
                          if d.frame_id <= check_frame)

        self.assertEqual(positions(baseline), positions(after),
                         "Future frames changed detections at or before the current frame")

    def test_temporal_block_split_has_guard_gap(self) -> None:
        """Frame-block splits must never overlap, and the guard gap is respected."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames_dir = root / "frames"
            frames_dir.mkdir()
            for i in range(1, 41):
                cv2.imwrite(str(frames_dir / f"frame_{i:04d}.png"), np.full((16, 16), i, np.uint8))
            train = SNLDataset(root, split="train", train_end_frame=24, guard_gap_frames=4,
                               window_size=3, tile_size=16, tile_overlap=0)
            test = SNLDataset(root, split="test", train_end_frame=24, guard_gap_frames=4,
                              window_size=3, tile_size=16, tile_overlap=0)
            self.assertEqual(max(train.original_frame_indices), 24)
            self.assertEqual(min(test.original_frame_indices), 29)
            self.assertTrue(set(train.original_frame_indices).isdisjoint(test.original_frame_indices))

    def test_normalization_uses_training_statistics_only(self) -> None:
        """Fitted normalization must ignore held-out frames entirely.

        v2.0 currently only has stateless per-frame normalization (normalize_db),
        so there is no fitted statistic to audit yet. This test guards the
        contract once a train-fitted normalizer is added: it must expose fit over
        training frames and apply without refitting on hold-out frames.
        """
        try:
            from videosar.datasets.base import fit_normalization_stats, apply_normalization
        except ImportError:
            self.skipTest("v2.0 has no train-fitted normalizer yet; contract pending implementation")

        train = np.array([[[10, 20], [30, 40]], [[20, 30], [40, 50]]], dtype=np.float32)
        held_out = np.full((2, 2), 250.0, dtype=np.float32)
        stats = fit_normalization_stats([train])
        first = apply_normalization(held_out, stats).copy()
        contaminated = fit_normalization_stats([train, held_out])
        self.assertNotAlmostEqual(float(contaminated["mean"]), float(stats["mean"]))
        np.testing.assert_array_equal(apply_normalization(held_out, stats), first)


if __name__ == "__main__":
    unittest.main()
