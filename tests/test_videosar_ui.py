"""Basic tests for the presentation-only GUI helpers.

These tests need only NumPy / Pillow, so they run in the existing .venv without
installing requirements-gui.txt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from videosar_ui import commands as cmd
from videosar_ui import imaging as viz
from videosar_ui import scanning as scan


class ImagingTest(TestCase):
    def test_normalize_modes_stay_in_range_and_handle_constant_frames(self):
        frame = np.random.default_rng(3).random((12, 16)).astype(np.float32) * 10
        for mode in viz.MODES:
            result = viz.normalize(frame, mode)
            self.assertEqual(result.shape, frame.shape)
            self.assertTrue(np.isfinite(result).all())
            self.assertGreaterEqual(float(result.min()), 0.0)
            self.assertLessEqual(float(result.max()), 1.0)
        constant = viz.normalize(np.full((4, 4), 7.0), "percentile")
        self.assertTrue(np.isfinite(constant).all())

    def test_colorize_luts_and_enhance_shape(self):
        frame = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
        gray = viz.colorize(frame, "gray")
        magma = viz.colorize(frame, "magma")
        self.assertEqual(gray.shape, (8, 8, 3))
        self.assertEqual(gray.dtype, np.uint8)
        self.assertFalse(np.array_equal(gray, magma))
        enhanced = viz.enhance(np.linspace(0, 10, 64).reshape(8, 8), "log_db", "viridis", True)
        self.assertEqual(enhanced.shape, (8, 8, 3))

    def test_smooth_keeps_constant_frame(self):
        self.assertTrue(np.allclose(viz.smooth(np.full((6, 6), 3.0)), 3.0))

    def test_temporal_difference_cancels_static_darkness_and_highlights_motion(self):
        frames = np.full((4, 16, 16), 10.0, dtype=np.float32)
        frames[:, 4:8, 4:8] = 1.0  # static dark area, present in every frame
        static = viz.temporal_difference(frames, 2)
        moving = frames.copy()
        moving[2, 10:14, 10:14] = 1.0  # moving shadow appears at frame 2
        changed = viz.temporal_difference(moving, 2)
        self.assertEqual(static.shape, (16, 16, 3))
        self.assertEqual(int(static[6, 6, 0]), int(changed[6, 6, 0]))  # static area cancelled
        self.assertGreater(int(changed[12, 12, 0]), int(static[12, 12, 0]))  # warm highlight
        previous = viz.temporal_difference(moving, 2, reference="previous")
        self.assertEqual(previous.shape, (16, 16, 3))

    def test_boundary_overlay_and_centroid(self):
        mask = np.ones((3, 3), dtype=bool)
        self.assertEqual(int(viz.boundary(mask).sum()), 8)
        self.assertEqual(viz.centroid(np.zeros((3, 3), dtype=bool)), None)
        offset = np.zeros((10, 10), dtype=bool)
        offset[2:4, 6:8] = True
        self.assertEqual(viz.centroid(offset), (6.5, 2.5))
        base = np.zeros((10, 10, 3), dtype=np.uint8)
        overlaid = viz.overlay_mask(base, offset, viz.TRUTH_COLOR)
        self.assertGreater(np.count_nonzero(overlaid), 0)
        self.assertEqual(int(overlaid[5, 5].sum()), 0)  # untouched outside the contour

    def test_count_components_roi_and_quad(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[1:3, 1:3] = True
        mask[8:10, 8:10] = True
        self.assertEqual(viz.count_components(mask), 2)
        rgb = np.zeros((12, 12, 3), dtype=np.uint8)
        zoom, roi = viz.roi_zoom(rgb, (6.0, 6.0), factor=4)
        self.assertEqual(zoom.shape, (192, 256, 3))
        self.assertEqual(roi[2], 64)
        quad = viz.compose_quad([rgb, rgb, rgb, rgb])
        self.assertEqual(quad.shape, (2 * (12 + 22) + 3 * 8, 2 * 12 + 3 * 8, 3))

    def test_draw_helpers_and_gif(self):
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        rows = [{"frame": 0, "x_px": 4, "y_px": 4, "source": "measured", "segment": 1},
                {"frame": 1, "x_px": 8, "y_px": 8, "source": "imputed", "segment": 1}]
        drawn = viz.draw_tracks(rgb, rows, 1)
        self.assertEqual(drawn.shape, rgb.shape)
        self.assertGreater(np.count_nonzero(drawn), 0)
        boxed = viz.draw_crop_box(rgb, (2, 2, 8, 8))
        self.assertEqual(int(boxed[2, 2, 0]), viz.CROP_COLOR[0])
        with TemporaryDirectory() as folder:
            path = viz.save_gif([rgb, rgb], Path(folder) / "preview.gif")
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)
            self.assertIsNone(viz.save_gif([], Path(folder) / "empty.gif"))


class ScanningTest(TestCase):
    def test_read_json_and_csv_tolerate_missing_and_bom(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertIsNone(scan.read_json_safe(root / "absent.json"))
            self.assertEqual(scan.read_csv_rows(root / "absent.csv"), (None, []))
            (root / "meta.json").write_text(json.dumps({"frame_interval_s": 0.2}), encoding="utf-8-sig")
            self.assertEqual(scan.read_json_safe(root / "meta.json")["frame_interval_s"], 0.2)
            (root / "table.csv").write_text("﻿frame,x_px,source\n0,1.5,measured\n1,,missing\n", encoding="utf-8")
            fields, rows = scan.read_csv_rows(root / "table.csv")
            self.assertEqual(fields, ["frame", "x_px", "source"])
            self.assertEqual(rows[1]["x_px"], "")
            self.assertEqual(scan.numeric(rows[0]["x_px"]), 1.5)
            self.assertIsNone(scan.numeric(rows[1]["x_px"]))
            self.assertEqual(scan.format_metric(None), "—")
            self.assertEqual(scan.format_metric(0.123456), "0.123")

    def test_sequence_discovery_prioritises_closeout_demo(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            preferred = root / scan.PREFERRED_SEQUENCE
            other = root / "some-other-run" / "scene-001"
            for path in (preferred, other):
                path.mkdir(parents=True)
                np.save(path / "frames.npy", np.ones((3, 8, 8), dtype=np.float32))
            found = scan.find_sequences(root)
            self.assertEqual(len(found), 2)
            self.assertEqual(found[0], preferred)

    def test_scan_limit_keeps_preferred_demo_first(self):
        """runs/ can hold more sequences than the UI shows; the demo still wins."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for index in range(30):
                path = root / "center-study" / f"scene-{index:03d}"
                path.mkdir(parents=True)
                np.save(path / "frames.npy", np.ones((2, 4, 4), dtype=np.float32))
            preferred = root / scan.PREFERRED_SEQUENCE
            preferred.mkdir(parents=True)
            np.save(preferred / "frames.npy", np.ones((2, 4, 4), dtype=np.float32))
            found = scan.find_sequences(root, max_results=5)
            self.assertEqual(len(found), 5)
            self.assertEqual(found[0], preferred)

    def test_load_sequence_validates_shapes_and_missing_files(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertIsNone(scan.load_sequence(root))
            np.save(root / "frames.npy", np.ones((4, 8, 10), dtype=np.float32))
            np.save(root / "masks.npy", np.ones((4, 8, 10), dtype=np.uint8))
            np.save(root / "geometric_masks.npy", np.ones((4, 8, 10), dtype=np.uint8))
            (root / "metadata.json").write_text(json.dumps({"frame_interval_s": 0.2, "pixel_size_m": 0.25}))
            sequence = scan.load_sequence(root)
            self.assertEqual(sequence["frames"].shape, (4, 8, 10))
            self.assertEqual(sequence["masks"].shape, (4, 8, 10))
            self.assertEqual(sequence["metadata"]["pixel_size_m"], 0.25)
            np.save(root / "masks.npy", np.ones((2, 8, 10), dtype=np.uint8))
            self.assertIsNone(scan.load_sequence(root)["masks"])

    def test_results_are_matched_by_source_receipt(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            sequence = root / "dataset" / "scene-000"
            sequence.mkdir(parents=True)
            linked = root / "detect-linked"
            unlinked = root / "detect-other"
            for path in (linked, unlinked):
                path.mkdir()
                np.save(path / "predictions.npy", np.zeros((4, 8, 10), dtype=np.uint8))
            (linked / "source.json").write_text(json.dumps({"sequence": str(sequence.resolve())}))
            (unlinked / "source.json").write_text(json.dumps({"sequence": str(root / "elsewhere")}))
            results = scan.find_results(root, sequence)
            self.assertTrue(results["_linked"]["predictions"])
            self.assertEqual(results["predictions"], [linked])
            self.assertTrue(scan.load_predictions(linked, (4, 8, 10)) is not None)
            self.assertIsNone(scan.load_predictions(linked, (5, 8, 10)))
            self.assertIsNone(scan.load_predictions(None, (4, 8, 10)))

    def test_tracks_metrics_and_speed_are_read_not_hardcoded(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tracks.csv").write_text(
                "frame,time_s,x_px,y_px,source,segment\n0,0,10,5,measured,1\n1,0.2,12,9,imputed,1\n2,0.4,,,missing,\n",
                encoding="utf-8-sig")
            (root / "tracks.json").write_text(json.dumps({"velocity_px_s": [10.0, 20.0],
                                                          "shadow_velocity_mps": [0.5, 1.0]}))
            tracks = scan.load_tracks(root)
            self.assertEqual(len(tracks["rows"]), 3)
            self.assertIsNone(tracks["rows"][2]["x_px"])
            indices, speeds = scan.per_frame_speed(tracks["rows"])
            self.assertEqual(indices, [0, 1, 2])
            self.assertAlmostEqual(speeds[1], np.hypot(2, 4) / 0.2)
            self.assertIsNone(speeds[2])

            metrics = {"detection": {"precision": 0.9123, "recall": 0.8, "target_f1": 0.85,
                                     "iou": 0.7, "false_positive_per_frame": 0.25,
                                     "matched": 19, "missed": 1, "false_positive": 5, "frames": [{}] * 20},
                       "tracking": {"continuity": 0.95, "true_shadow_velocity_px_s": [3.0, 4.0],
                                    "velocity_error_mps": 0.12},
                       "match_iou": 0.5}
            (root / "metrics.json").write_text(json.dumps(metrics))
            loaded = scan.load_metrics(root)
            self.assertEqual(loaded["match_iou"], 0.5)
            cards = dict(scan.metric_cards(loaded, 20))
            self.assertAlmostEqual(cards["查准率 Precision"], 0.9123)
            self.assertEqual(cards["帧数"], 20)
            self.assertIsNone(dict(scan.metric_cards(None, 20))["查准率 Precision"])
            summary = scan.tracking_summary(loaded, tracks)
            self.assertAlmostEqual(summary["表观阴影速度 (m/s)"], np.hypot(0.5, 1.0))
            self.assertAlmostEqual(summary["真实阴影速度 (px/s)"], 5.0)
            self.assertEqual(summary["速度误差 (m/s)"], 0.12)

    def test_archive_tables_and_previews_only_return_existing_files(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            proposal = root / "deliverables" / "proposal"
            proposal.mkdir(parents=True)
            (proposal / "metrics.csv").write_text("method,precision\nthreshold,0.5\n", encoding="utf-8")
            tables = scan.archive_tables(root)
            self.assertEqual(len(tables), 1)
            self.assertEqual(tables[0][0], "申报书正式测试各方法对比")
            preview_root = root / "runs"
            target = preview_root / scan.SNL_PREVIEWS[0]
            target.parent.mkdir(parents=True)
            target.write_bytes(b"fake")
            self.assertEqual(scan.preview_videos(preview_root), [str(target)])
            self.assertEqual(scan.live_dirs(preview_root)["live-detect"], None)

    def test_snl_model_info_exposes_development_scores(self):
        info = scan.snl_model_info()
        self.assertIn("development", info)
        self.assertIn("input_channels", info)
        self.assertEqual(info["development"]["frames"], 48)


class CommandsTest(TestCase):
    def test_demo_and_sar_commands_use_real_flag_names(self):
        demo = cmd.demo_command(cmd.DEMO_CONFIG, 12, Path("runs/gui/demo"))
        self.assertEqual(demo[2:4], ["videosar", "demo"])
        self.assertIn("--config", demo)
        self.assertIn("--frames", demo)
        self.assertIn("--output-dir", demo)
        plan = cmd.sar_chain_commands(450000000, "learned", cmd.UNET_MODEL, Path("runs/gui/sar"))
        self.assertEqual(len(plan["commands"]), 4)
        joined = [part for command in plan["commands"] for part in command]
        for flag in ("--bandwidth-hz", "--input", "--method", "--model", "--predictions",
                     "--tracks", "--match-iou"):
            self.assertIn(flag, joined)
        self.assertIn("0.50", joined)
        traditional = cmd.sar_chain_commands(300000000, "threshold", None, Path("runs/gui/sar2"))
        self.assertNotIn("--model", [part for command in traditional["commands"] for part in command])

    def test_detect_video_and_compare_commands(self):
        command = cmd.detect_video_command(Path("in.mp4"), Path("runs/gui/v"), crop=(10, 20, 300, 400),
                                           start_frame=150, max_frames=24)
        self.assertEqual(command[2:4], ["videosar", "detect-video"])
        self.assertEqual(command[command.index("--crop") + 1:command.index("--crop") + 5],
                         ["10", "20", "300", "400"])
        self.assertIn("--start-frame", command)
        self.assertIn("--max-frames", command)
        self.assertIn("--output-dir", command)
        self.assertNotIn("--model", command)
        comparison = cmd.compare_commands(Path("dataset/s1"), Path("runs/gui/cmp"))
        self.assertEqual(len(comparison["commands"]), 2)
        self.assertIn("--method", comparison["commands"][0])
        self.assertIn("learned", comparison["commands"][1])

    def test_gui_run_dir_is_under_runs_gui(self):
        with TemporaryDirectory() as folder:
            path = cmd.gui_run_dir("demo", Path(folder))
            self.assertEqual(path.parent.name, "gui")
            self.assertEqual(path.parent.parent.name, "runs")
            self.assertTrue(path.name.endswith("_demo"))

    def test_run_commands_streams_output_and_reports_failure(self):
        with TemporaryDirectory() as folder:
            lines = []
            ok = cmd.run_commands([[sys.executable, "-c", "print('hello-gui')"]], folder, lines.append)
            self.assertTrue(ok)
            self.assertIn("hello-gui", "\n".join(lines))
            self.assertFalse(cmd.run_commands([[sys.executable, "-c", "raise SystemExit(3)"]],
                                              folder, lines.append))
            self.assertTrue(any("退出码 3" in line for line in lines))

    def test_environment_report_keys(self):
        report = cmd.environment_report()
        for key in ("ffmpeg", "torch", "torchvision", "imageio", "imageio_ffmpeg", "python"):
            self.assertIn(key, report)
        self.assertTrue(report["python"])
