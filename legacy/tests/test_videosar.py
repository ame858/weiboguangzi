import json
from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np

from videosar.__main__ import main
from videosar.analysis import baseline_scores, score_masks, score_track, segment, target_summary, track
from videosar.aperture_motion_study import CONDITIONS as MOTION_CONDITIONS, CONTRASTS as MOTION_CONTRASTS
from videosar.center_box import center_loss, center_masks
from videosar.center_study import rectangle_oracle
from videosar.contrast_study import with_contrast_stem
from videosar.echo_attribution import factorial_metrics
from videosar.open_compare import new_test_specs
from videosar.config import SIMULATION, merged
from videosar.data import read_json
from videosar.experiment import read_split
from videosar.learning import fit, load_model, predict, save_model
from videosar.proposal_data import translate
from videosar.simulation import (_focus_fields, calibrated_receiver_video, form_video, occluded, paired_aperture_video,
                                 paired_shadow_video, point_check, positions, simulate)
from videosar.small_shadow import independent_specs, operating_rank
from videosar.snl_causal import CausalFeatures, TrackFilter, component_boxes, detection_counts
from videosar.stress import diagnostics, scene_specs
from videosar.temporal_study import expand_history_stem
from videosar.trajectory import accumulate_centers, low_rank_scores, predictions_from_history, recover_masks
from videosar.study import calibrate, cleanup
from videosar.unet import (FPN, LRASPP, UNET, align, create_model, input_features, load_weights, logits,
                           loss_value, predict as predict_unet, runtime, training_stats)


class VideoSarTest(TestCase):
    def test_snl_causal_prefix_is_invariant_to_future_frames(self):
        rng = np.random.default_rng(58)
        frames = rng.integers(0, 170, (45, 24, 24), dtype=np.uint8)
        changed = frames.copy()
        changed[33:] = 255 - changed[33:]

        def run(images, profile):
            history = CausalFeatures(profile)
            track = TrackFilter(0.08, 0.03)
            result = []
            for i, image in enumerate(images):
                channels, _, fused = history.update(image)
                boxes, scores = component_boxes(fused, 0.03, 3)
                result.append((channels, *track.update(i, boxes, scores)))
            return result

        for profile in ("raw", "standardized", "registered", "histogram", "histogram-registered", "stacked",
                        "stacked-standardized", "stacked-long", "background-history"):
            expected = run(frames[:33], profile)
            for full in (run(frames, profile), run(changed, profile)):
                for first, second in zip(expected, full[:33], strict=True):
                    for a, b in zip(first, second, strict=True):
                        np.testing.assert_array_equal(a, b)

    def test_snl_background_warmup_reuses_oldest_observation(self):
        history = CausalFeatures("background-history")
        first, _, _ = history.update(np.full((12, 12), 20, dtype=np.uint8))
        history.update(np.full((12, 12), 40, dtype=np.uint8))
        third, _, _ = history.update(np.full((12, 12), 60, dtype=np.uint8))
        self.assertEqual(first.shape[0], 7)
        np.testing.assert_array_equal(np.rint(255 * first[:, 0, 0]), np.full(7, 20))
        np.testing.assert_array_equal(np.rint(255 * third[:, 0, 0]), [60, 40, 20, 20, 20, 20, 20])

        longer = CausalFeatures("stacked-long")
        for value in range(92):
            channels, _, _ = longer.update(np.full((4, 4), value, dtype=np.uint8))
        np.testing.assert_array_equal(np.rint(255 * channels[:, 0, 0]), [91, 90, 85, 76, 61, 31, 1])

    def test_snl_video_network_channels_match_the_research_scanner(self):
        frames = np.random.default_rng(11).integers(0, 256, (95, 12, 16), dtype=np.uint8)
        for profile in ("stacked", "stacked-long", "background-history", "stacked-standardized", "raw"):
            production, research = CausalFeatures(profile), CausalFeatures(profile)
            for frame in frames:
                np.testing.assert_array_equal(production.network_channels(frame), research.update(frame)[0])

    def test_snl_causal_registration_reduces_history_residual_for_camera_motion(self):
        from scipy import ndimage

        rng = np.random.default_rng(874)
        background = ndimage.gaussian_filter(rng.uniform(50, 170, (96, 112)), 1.2)
        raw, registered = CausalFeatures(), CausalFeatures("registered")
        for i in range(38):
            frame = ndimage.shift(background, (i*0.1, i*-0.13), order=1, mode="nearest").astype(np.uint8)
            plain, _, _ = raw.update(frame)
            corrected, _, _ = registered.update(frame)
        self.assertLess(corrected[2, 12:-12, 12:-12].mean(), plain[2, 12:-12, 12:-12].mean() * 0.65)

    def test_snl_track_keeps_current_box_only_with_recent_strong_detection(self):
        tracker = TrackFilter(0.8, 0.3)
        first = np.array([[2, 3, 10, 16]], dtype=np.float32)
        current = first + [1, 0, 1, 0]
        self.assertEqual(len(tracker.update(0, first, np.array([0.4]))[0]), 0)
        out, _, ids = tracker.update(1, first, np.array([0.85]))
        np.testing.assert_array_equal(out, first)
        out, _, current_ids = tracker.update(7, current, np.array([0.4]))
        np.testing.assert_array_equal(out, current)
        np.testing.assert_array_equal(ids, current_ids)
        self.assertEqual(len(tracker.update(8, current, np.array([0.4]))[0]), 0)
        self.assertEqual(detection_counts(np.concatenate((first, first)), first), (1, 1, 0))

        corroborated = TrackFilter(0.8, 0.3, confirmations=2)
        corroborated.update(0, first, np.array([0.85]))
        self.assertEqual(len(corroborated.update(1, current, np.array([0.4]))[0]), 0)
        corroborated.update(2, current, np.array([0.85]))
        np.testing.assert_array_equal(corroborated.update(3, current, np.array([0.4]))[0], current)

    @classmethod
    def setUpClass(cls):
        Path("runs").mkdir(exist_ok=True)

    def test_trajectory_recovery_uses_only_present_box_and_past_detections(self):
        rows = [{"source": "measured", "segment": 1, "xy_px": [1, 2], "time_s": 0},
                {"source": "measured", "segment": 1, "xy_px": [2, 2], "time_s": 1},
                {"source": "measured", "segment": 1, "xy_px": [99, 99], "time_s": 2}]
        expected = predictions_from_history(rows, np.arange(3))
        np.testing.assert_array_equal(expected[2], [3, 2])
        masks = np.zeros((3, 8, 8), dtype=bool)
        masks[1, 2, 2] = True
        masks[2, 7, 7] = True
        boxes = np.zeros_like(masks)
        boxes[2, 2, 3] = True
        observed = dict(masks=masks, boxes=boxes, expected=expected, scores=np.array([0.9, 0.2, 0.3]),
                        xy=np.array([[4, 4], [2, 2], [3, 2]]),
                        mask_centers=[np.empty((0, 2)), np.array([[2, 2]]), np.array([[7, 7]])])
        recovered, selected = recover_masks(observed, dict(method="association", gate=2, threshold=0.15, replace=True))
        np.testing.assert_array_equal(selected, [False, False, True])
        np.testing.assert_array_equal(recovered[:2], masks[:2])
        np.testing.assert_array_equal(recovered[2], boxes[2])

    def test_temporal_scores_are_causal_and_stationary_peak_is_controlled(self):
        heatmap = np.zeros((5, 8, 8))
        heatmap[:3, 2, 2:5] = np.eye(3) * 0.4
        heatmap[3:, 5, 1] = 1
        full = accumulate_centers(heatmap, 2, 0.5)
        np.testing.assert_array_equal(full[:3], accumulate_centers(heatmap[:3], 2, 0.5))
        self.assertAlmostEqual(full[2], 0.4)
        constant = np.repeat(heatmap[:1], 3, axis=0)
        self.assertLess(accumulate_centers(constant, 2, 0.5)[2], 0.4)
        power = np.concatenate([constant, heatmap[3:]])
        np.testing.assert_allclose(low_rank_scores(power, 5, 1)[:3], low_rank_scores(power[:3], 5, 1))

    def test_center_rectangles_use_peak_threshold_and_image_bounds(self):
        heatmaps = np.zeros((2, 8, 8))
        heatmaps[0, 0, 7] = 0.8
        heatmaps[0, 5, 2] = 0.7
        heatmaps[1, 0, 0] = 0.3
        distances = np.full((2, 4, 8, 8), np.log1p(2))
        masks = center_masks(heatmaps, distances, 0.5, 1)
        self.assertEqual(int(masks[0].sum()), 9)
        self.assertTrue(masks[0, :3, 5:].all())
        self.assertFalse(masks[1].any())
        self.assertFalse(center_masks(heatmaps, distances, 0.5, 10).any())
        disjoint = np.zeros((1, 6), dtype=bool)
        disjoint[0, [0, 5]] = True
        self.assertEqual(rectangle_oracle(disjoint), (2 / 6, 0.5))

    @skipUnless(find_spec("segmentation_models_pytorch") is not None, "segmentation runtime is optional")
    def test_center_loss_handles_negative_and_positive_batches(self):
        torch, _, _ = runtime()
        for positive in (False, True):
            labels = torch.zeros((2, 8, 8))
            if positive:
                labels[0, 1:4, 3:7] = 1
            output = torch.zeros((2, 6, 8, 8), requires_grad=True)
            loss = center_loss(torch, output, labels)
            loss.backward()
            self.assertTrue(torch.isfinite(loss).item())
            self.assertTrue(torch.isfinite(output.grad).all().item())
            self.assertEqual(bool(output.grad[:, 2:].any()), positive)

    def test_local_contrast_is_spatial_and_zero_stem_preserves_logits(self):
        frames = np.ones((2, 64, 80), dtype=np.float32)
        frames[0, 30:35, 40:45] = 0.15
        features = input_features(frames, True, local_contrast=True)
        self.assertEqual(features.shape, (2, 4, 64, 80))
        self.assertTrue((features[0, 2:, 32, 42] > 0).all())
        np.testing.assert_allclose(features[1, 2:], 0, atol=1e-7)
        np.testing.assert_array_equal(features[:, :2], input_features(frames, True))
        if find_spec("segmentation_models_pytorch") is not None:
            torch, smp, _ = runtime()
            torch.set_num_threads(2)
            torch.manual_seed(101645)
            parent = create_model(torch, smp, True).to(memory_format=torch.channels_last).eval()
            model = with_contrast_stem(torch, smp, parent).eval()
            self.assertEqual(torch.count_nonzero(model.encoder.conv1.weight[:, 2:]).item(), 0)
            with torch.inference_mode():
                a = logits(torch, parent, torch.from_numpy(features[:, :2]))
                b = logits(torch, model, torch.from_numpy(features))
            torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-5)

    def test_stress_variants_preserve_paired_controls(self):
        from collections import Counter

        scenes = list(scene_specs())
        self.assertEqual(len(scenes), 72)
        self.assertEqual(len({row["config"]["seed"] for row in scenes}), 24)
        self.assertTrue(all(n == 12 for n in Counter(row["condition"] for row in scenes).values()))
        reference, slow, small, dark, combined = scenes[:5]
        self.assertEqual(dark["config"], reference["config"])
        self.assertEqual(dark["offsets_xy_px"], reference["offsets_xy_px"])
        self.assertEqual(combined["static_dark"], dark["static_dark"])
        for axis in ("x", "y"):
            speed, start = f"velocity_{axis}_mps", f"start_{axis}_m"
            cfg, slower = reference["config"], slow["config"]
            self.assertAlmostEqual(cfg[start] + 1.9 * cfg[speed], slower[start] + 1.9 * slower[speed])
        for key in ("object_height_m", "object_width_m", "object_depth_m"):
            self.assertAlmostEqual(small["config"][key], reference["config"][key] * 0.6)
        config = reference["config"] | {"frames": 2, "pulses_per_frame": 5}
        a = form_video(config, normalize_power=False)[0]
        b = form_video(config, normalize_power=False, static_dark=dark["static_dark"][:-1] + [1.0])[0]
        np.testing.assert_array_equal(a, b)

    def test_stress_diagnostics_tracks_removed_region(self):
        raw = {"predicted_components": 2, "best_component_iou": 0.6}
        closed = {"predicted_components": 1, "best_component_iou": 0.7}
        final = {"predicted_components": 0, "best_component_iou": 0, "target_visible": True}
        self.assertEqual(diagnostics(raw, closed, final)["miss_stage"], "area_filter")
        no_target = {"predicted_components": 2, "best_component_iou": 0, "target_visible": False}
        actual = diagnostics(raw, closed, no_target)
        self.assertFalse(actual["missed"])
        self.assertEqual(actual["false_alarms"], 2)

    def test_calibrated_receiver_noise_keeps_clean_labels_and_reference(self):
        config = next(scene_specs())["config"] | {"frames": 2, "pulses_per_frame": 5,
                                                   "noise": 0, "thermal_noise": 0}
        clean, labels, _, geometry, reference = form_video(config, normalize_power=False)
        levels = calibrated_receiver_video(config, (25, 10))
        for snr, (observed, mask, info, geometric, unblocked) in levels.items():
            self.assertAlmostEqual(info["snr_measured_db"], snr)
            np.testing.assert_array_equal(mask, labels)
            np.testing.assert_array_equal(geometric, geometry)
            np.testing.assert_array_equal(unblocked, reference)
            self.assertFalse(np.array_equal(observed, clean))
        self.assertAlmostEqual(levels[10][2]["receiver_noise_scale"] /
                               levels[25][2]["receiver_noise_scale"], 10**(15 / 20))

    def test_aperture_subsets_share_receiver_samples(self):
        config = next(scene_specs())["config"] | {"frames": 2, "frame_interval_s": 0.01,
                                                   "noise": 0, "thermal_noise": 0}
        with patch("videosar.simulation._focus_fields", wraps=_focus_fields) as focus:
            short, long = list(paired_aperture_video(config, (3, 5), 25))
        short_call, long_call = focus.call_args_list
        short_noise = short_call.kwargs["noise_echoes"]
        long_noise = long_call.kwargs["noise_echoes"]
        self.assertIs(short_call.kwargs["processor"], long_call.kwargs["processor"])
        np.testing.assert_array_equal(short_noise, long_noise[:, 1:-1])
        np.testing.assert_array_equal(long_noise[0, 2:], long_noise[1, :-2])
        np.testing.assert_array_equal(short[1][3], long[1][3])
        for _, (_, _, meta, _, _) in (short, long):
            self.assertAlmostEqual(meta["snr_measured_db"], 25)

    def test_held_aperture_uses_the_target_center_of_each_frame(self):
        config = next(scene_specs())["config"] | {"frames": 2}
        with patch("videosar.simulation.occluded", wraps=occluded) as blocker:
            _, held = next(paired_aperture_video(config, (5,), 25, hold_target_within_frame=True))
        for frame in range(2):
            calls = blocker.call_args_list[frame * 6:(frame + 1) * 6]
            center = positions(config, frame * config["frame_interval_s"])
            self.assertTrue(all(call.args[3] == center for call in calls))
            self.assertEqual(len({call.args[2] for call in calls[:5]}), 5)
        _, moving = next(paired_aperture_video(config, (5,), 25))
        np.testing.assert_array_equal(held[3], moving[3])
        np.testing.assert_array_equal(held[4], moving[4])

    def test_paired_shadow_attenuation_agrees_with_separate_echo_formation(self):
        config = SIMULATION | {"frames": 3, "width": 40, "height": 32, "pulses_per_frame": 17,
                               "start_x_m": -1.5}
        for retention, result in paired_shadow_video(config, (1.0, 0.30, 0.60), 25):
            adjusted = config if retention == 1 else config | {
                "shadow_factor": 1 - retention * (1 - config["shadow_factor"]),
            }
            _, direct = next(paired_aperture_video(adjusted, (17,), 25))
            if retention == 1:
                np.testing.assert_array_equal(result[0], direct[0])
                self.assertTrue(result[1].any())
            else:
                np.testing.assert_allclose(result[0], direct[0], rtol=1e-12, atol=1e-12)
            for i in (1, 3, 4):
                np.testing.assert_array_equal(result[i], direct[i])
            self.assertEqual(result[2]["receiver_noise_power"], direct[2]["receiver_noise_power"])

    def test_fixed_reference_interaction_uses_matches_on_shared_frames(self):
        rows = []
        for frame in (0, 1):
            for pulses, motion in MOTION_CONDITIONS:
                successful = 1 if (pulses, motion) == (91, "held") else 0
                rows.append(dict(group="small", scene_family="test", frame=frame, pulses=pulses, motion=motion,
                                 geometry_pixels=1, visible_pixels=1, local_contrast_db=2.0 * successful,
                                 reference_common_visible=frame == 0,
                                 unet_reference_matched=successful, fpn_reference_matched=0,
                                 difference_reference_matched=0))
        _, estimates = factorial_metrics(rows, {"interaction": MOTION_CONTRASTS["interaction"]},
                                         ("pulses", "motion"), fixed_reference=True)
        checked = {row["metric"]: (row["frames"], row["delta"]) for row in estimates}
        self.assertEqual(checked["unet_common_visible_recall"], (1, 1.0))
        self.assertEqual(checked["local_contrast_db"], (1, 2.0))
        self.assertEqual(checked["formation_rate"], (2, 0.0))

    def test_small_shadow_splits_and_validation_priority(self):
        from collections import Counter

        specs = list(independent_specs())
        counts = Counter(spec["split"] for spec in specs)
        self.assertEqual(counts, {"train": 48, "validation": 24, "test": 36})
        seeds = {spec["config"]["seed"] for spec in specs}
        self.assertEqual(len(seeds), 108)
        self.assertFalse(seeds & {spec["config"]["seed"] for spec in scene_specs()})
        holdout = list(new_test_specs())
        self.assertEqual(len(holdout), 36)
        self.assertFalse(seeds & {spec["config"]["seed"] for spec in holdout})
        self.assertEqual(Counter(s["condition"] for s in specs if s["split"] == "test"),
                         {"normal": 9, "small": 9, "small_slow_dark": 9, "empty": 9})
        eligible = {"precision": 0.88, "recall": 0.83, "target_f1": 0.85, "normal_recall": 0.90,
                    "small_recall": 0.75, "false_positive": 6}
        low_precision = eligible | {"precision": 0.77, "small_recall": 0.95}
        normal_regression = eligible | {"normal_recall": 0.83, "small_recall": 0.92}
        for alternative in (low_precision, normal_regression):
            self.assertGreater(operating_rank(eligible, 0.92), operating_rank(alternative, 0.92))

    def test_legacy_regression(self):
        config = merged(SIMULATION, {"scene": "legacy"})
        frames, truth, meta, _ = simulate(config)
        self.assertEqual(frames.shape, (24, 128, 192))
        np.testing.assert_array_equal(truth.sum(axis=(1, 2)), 512)
        np.testing.assert_array_equal(np.unique(frames), [45, 180])
        predicted = segment(baseline_scores(frames, "threshold"), 0.45, 4)
        result = track(predicted, meta["timestamps_s"], None, 12, 2)
        np.testing.assert_allclose(result["velocity_px_s"], [30, 5])
        self.assertIsNone(result["shadow_velocity_mps"])
        self.assertEqual(score_masks(predicted, truth, 0.3)["matched"], 24)

    def test_geometry_and_target_velocity(self):
        config = merged(SIMULATION, {"noise": 0, "background": "uniform"})
        frames, truth, meta, _ = simulate(config)
        np.testing.assert_array_equal(frames, simulate(config)[0])
        predicted = segment(baseline_scores(frames, "threshold"), 0.45, 4)
        result = track(predicted, meta["timestamps_s"], config["pixel_size_m"], 12, 2)
        motion = score_track(result, truth, config["pixel_size_m"], meta)
        self.assertLess(motion["target_velocity_error_mps"], 0.15)
        outside = config | {"start_x_m": 10.0, "velocity_x_mps": 3.0}
        _, labels, _, _ = simulate(outside)
        self.assertTrue(labels[0].any())
        self.assertFalse(labels[-1].any())

    def test_empty_and_missing(self):
        frames, truth, meta, _ = simulate(merged(SIMULATION, {"scene": "empty", "background": "uniform", "noise": 0}))
        predicted = segment(baseline_scores(frames, "threshold"), 0.45, 4)
        metrics = score_masks(predicted, truth, 0.3)
        self.assertIsNone(metrics["iou"])
        self.assertEqual(metrics["false_positive"], 0)
        self.assertIsNone(track(predicted, meta["timestamps_s"], 0.25, 12, 2)["velocity_px_s"])

    def test_short_gap(self):
        masks = np.zeros((6, 10, 20), dtype=bool)
        for t in (0, 1, 2, 5):
            masks[t, 4:6, 2 + t:4 + t] = True
        times = np.arange(6) * 0.25
        result = track(masks, times, 0.5, 3, 2)
        self.assertEqual([row["source"] for row in result["frames"]][3:5], ["imputed", "imputed"])
        np.testing.assert_allclose(result["shadow_velocity_mps"], [2, 0])
        self.assertEqual(track(masks, times, 0.5, 3, 0)["frames"][3]["source"], "missing")

    def test_sar_peak_and_obstructed_counterfactual(self):
        config = merged(SIMULATION, {"frames": 3, "noise": 0})
        point, _ = point_check(config)
        narrow, _ = point_check(config | {"bandwidth_hz": config["bandwidth_hz"] / 2})
        self.assertLessEqual(point["point_error_px"], 1)
        self.assertLessEqual(point["range_peak_error_bins"], 1)
        self.assertAlmostEqual(narrow["range_mainlobe_half_power_width_m"] / point["range_mainlobe_half_power_width_m"], 2)
        frames, masks, _, geometry, ref = form_video(config)
        self.assertTrue(masks.any() and geometry.any())
        self.assertFalse(np.allclose(frames, ref))
        frames, masks, _, geometry, ref = form_video(config | {"scene": "empty"})
        self.assertFalse(masks.any() or geometry.any())
        np.testing.assert_array_equal(frames, ref)

    def test_learning_is_causal(self):
        frames, truth, _, _ = simulate(merged(SIMULATION, {"frames": 8}))
        frames = frames.astype(np.float32)
        with TemporaryDirectory(dir=Path.cwd() / "runs") as temporary:
            for network in ("compact", "context"):
                with self.subTest(network=network):
                    model, loss = fit([(frames, truth)], True, 3, 42, network)
                    self.assertLess(loss[-1], loss[0])
                    path = Path(temporary) / f"{network}.npz"
                    save_model(path, model)
                    reloaded = load_model(path)
                    np.testing.assert_allclose(predict(frames[:4], reloaded), predict(frames, model)[:4])

    def test_cli_config_and_unlabelled_input(self):
        with TemporaryDirectory(dir=Path.cwd() / "runs") as temporary:
            root = Path(temporary)
            config = root / "sample.json"
            config.write_text(json.dumps({"scene": "legacy", "frame_interval_s": 0.4}), encoding="utf-8")
            sequence = root / "sequence"
            self.assertEqual(main(["simulate", "--config", str(config), "--frames", "10", "--shadow-width-px", "20",
                                   "--output-dir", str(sequence)]), 0)
            self.assertEqual(read_json(sequence / "config.json")["frames"], 10)
            np.testing.assert_array_equal(np.load(sequence / "masks.npy").sum(axis=(1, 2)), 320)
            (sequence / "masks.npy").unlink()
            detections = root / "detections"
            tracks = root / "tracks"
            self.assertEqual(main(["detect", "--input", str(sequence), "--output-dir", str(detections)]), 0)
            self.assertEqual(main(["track", "--input", str(sequence), "--predictions", str(detections),
                                   "--output-dir", str(tracks)]), 0)
            np.testing.assert_allclose(read_json(tracks / "tracks.json")["velocity_px_s"], [15, 2.5])
            self.assertEqual(main(["simulate", "--frame-interval-s", "0", "--output-dir", str(root / "invalid")]), 2)
            self.assertFalse((root / "invalid").exists())

    def test_scene_split_leakage_is_rejected(self):
        with TemporaryDirectory(dir=Path.cwd() / "runs") as temporary:
            root = Path(temporary)
            entries = [{"path": split, "scene_id": split, "trajectory_family": "rotated-copy", "split": split}
                       for split in ("train", "validation", "test")]
            (root / "dataset.json").write_text(json.dumps({"version": 1, "entries": entries}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "crosses splits|across splits"):
                read_split(root)

    def test_unet_input_features_are_causal(self):
        frames = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
        features = input_features(frames, True)
        self.assertEqual(features.shape, (3, 2, 8, 8))
        np.testing.assert_array_equal(features[:2], input_features(frames[:2], True))
        np.testing.assert_allclose(features[2, 1], np.log1p(frames[2]) - np.log1p(frames[1]))
        mean, std = training_stats([features[:1], features[1:]])
        np.testing.assert_allclose(mean, features.mean(axis=(0, 2, 3), dtype=np.float64))
        np.testing.assert_allclose(std, features.std(axis=(0, 2, 3), dtype=np.float64))
        history = input_features(frames, True, 5)
        np.testing.assert_array_equal(history[:, :2], features)
        np.testing.assert_array_equal(history[:2], input_features(frames[:2], True, 5))
        np.testing.assert_allclose(history[2, 2], np.log1p(frames[2]) - np.median(np.log1p(frames[:2]), axis=0))
        self.assertFalse(history[0, 2].any())

    @skipUnless(all(find_spec(name) for name in ("torch", "segmentation_models_pytorch", "skimage")),
                "optional segmentation dependencies unavailable")
    def test_history_expansion_preserves_initial_logits_and_weighted_loss_is_finite(self):
        torch, smp, _ = runtime()
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)
        model = create_model(torch, smp, True).eval()
        extended = expand_history_stem(torch, smp, model, 5).eval()
        self.assertTrue(torch.equal(extended.encoder.conv1.weight[:, :2], model.encoder.conv1.weight))
        self.assertEqual(torch.count_nonzero(extended.encoder.conv1.weight[:, 2]).item(), 0)
        x = torch.rand(3, 3, 24, 32)
        with torch.inference_mode():
            np.testing.assert_allclose(logits(torch, extended, x).numpy(), logits(torch, model, x[:, :2]).numpy(),
                                       atol=2e-5, rtol=1e-5)
        labels = torch.zeros(3, 24, 32)
        labels[0, 2:5, 2:5] = 1
        labels[1, 2:10, 2:10] = 1
        output = torch.zeros_like(labels, requires_grad=True)
        value = loss_value(torch, output, labels, 64.0)
        value.backward()
        self.assertTrue(torch.isfinite(output.grad).all())
        self.assertGreater(abs(output.grad[0, 2, 2].item()), abs(output.grad[1, 2, 2].item()))

    @skipUnless(find_spec("skimage"), "optional study dependencies unavailable")
    def test_alignment_and_validation_matcher(self):
        from skimage.registration import phase_cross_correlation

        frame = np.random.default_rng(13).uniform(0.2, 1.2, (32, 48)).astype(np.float32)
        shifted = translate(frame, 1.4, -0.8)
        _, correction = align(np.stack((frame, shifted)), phase_cross_correlation)
        np.testing.assert_allclose(correction[1], [-1.4, 0.8], atol=0.25)
        truth = np.zeros((2, 14, 14), dtype=bool)
        truth[0, 4:8, 4:8] = True
        scores = truth.astype(float) * 0.8
        scores[:, 1, 1] = 0.95
        settings, scan = calibrate([(scores, truth)], (0.7, 0.9), (1, 3), False, 0.5)
        self.assertEqual(settings, {"threshold": 0.7, "min_area": 3, "closing": False})
        masks = cleanup(scores, **settings)
        metrics = score_masks(masks, truth, 0.5)
        self.assertEqual(target_summary(metrics["matched"], metrics["false_positive"], metrics["missed"])["target_f1"], 1)
        self.assertEqual(len(scan), 4)

    @skipUnless(all(find_spec(name) for name in ("torch", "segmentation_models_pytorch", "skimage")),
                "optional U-Net dependencies unavailable")
    def test_unet_checkpoint_supports_unpadded_frame_sizes(self):
        torch, smp, _ = runtime()
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)
        frames = np.full((2, 31, 43), 0.7, dtype=np.float32)
        settings = {"format_version": 1, "architecture": "smp.Unet/resnet18/depth3",
                    "temporal": True, "mean": [0, 0], "std": [1, 1]}
        with TemporaryDirectory(dir=Path.cwd() / "runs") as temporary:
            path = Path(temporary) / "weights.pt"
            model = create_model(torch, smp, True).to(memory_format=torch.channels_last)
            expected = predict_unet(torch, model, frames, settings, "cpu", 2)
            torch.save(model.state_dict(), path)
            settings["weight_sha256"] = sha256(path.read_bytes()).hexdigest()
            reloaded = load_weights(torch, smp, path, settings, "cpu")
            actual = predict_unet(torch, reloaded, frames, settings, "cpu", 2)
            self.assertEqual(actual.shape, frames.shape)
            np.testing.assert_array_equal(actual, expected)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_weights(torch, smp, path, settings | {"weight_sha256": "wrong"}, "cpu")

    @skipUnless(all(find_spec(name) for name in ("torch", "torchvision", "segmentation_models_pytorch", "skimage")),
                "optional segmentation dependencies unavailable")
    def test_open_models_return_original_grid_and_reload(self):
        torch, smp, _ = runtime()
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)
        frames = np.full((2, 31, 43), 0.7, dtype=np.float32)
        with TemporaryDirectory(dir=Path.cwd() / "runs") as temporary:
            for i, (architecture, scale) in enumerate(((FPN, 1), (LRASPP, 1), (UNET, 2))):
                with self.subTest(architecture=architecture, scale=scale):
                    model = create_model(torch, smp, True, architecture).to(memory_format=torch.channels_last)
                    settings = {"format_version": 1, "architecture": architecture, "input_scale": scale,
                                "temporal": True, "mean": [0, 0], "std": [1, 1]}
                    expected = predict_unet(torch, model, frames, settings, "cpu", 2)
                    path = Path(temporary) / f"weights-{i}.pt"
                    torch.save(model.state_dict(), path)
                    settings["weight_sha256"] = sha256(path.read_bytes()).hexdigest()
                    loaded = load_weights(torch, smp, path, settings, "cpu")
                    actual = predict_unet(torch, loaded, frames, settings, "cpu", 2)
                    self.assertEqual(actual.shape, frames.shape)
                    np.testing.assert_array_equal(actual, expected)
