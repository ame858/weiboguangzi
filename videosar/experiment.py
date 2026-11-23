from pathlib import Path
from time import perf_counter

import numpy as np

from .analysis import baseline_scores, score_masks, score_track, segment, track
from .config import SIMULATION, merged
from .data import load_masks, load_observations, read_json, save_receipt, save_sequence, write_json
from .learning import fit, predict, save_model
from .proposal_data import make_proposal_dataset
from .simulation import form_video, simulate
from .visuals import save_preview, write_csv


SAR_CONDITIONS = ("texture", "clutter", "empty", "high_noise", "small_fast", "exit")


def sar_scene(rng, condition, split):
    direction = rng.choice([-1, 1])
    speed = rng.uniform(2.5, 3.5) if condition in ("small_fast", "exit") else rng.uniform(0.75, 2.4)
    if condition == "small_fast" and split == "test":
        speed *= 1.28
    noise = rng.uniform(0.38, 0.56) if condition == "high_noise" else rng.uniform(0.06, 0.28)
    if condition == "high_noise" and split == "test":
        noise += 0.20
    start_x = direction * rng.uniform(5.5, 6.8) if condition == "exit" else rng.uniform(-4.0, 4.0)
    return merged(SIMULATION, {"scene": "empty" if condition == "empty" else "geometry",
                               "background": "texture" if condition == "texture" else "clutter",
                               "noise": float(noise), "seed": int(rng.integers(0, 2**32 - 1)),
                               "width": 72, "height": 56, "frames": 18, "pulses_per_frame": 61,
                               "start_x_m": float(start_x), "start_y_m": float(rng.uniform(-3.0, 0.5)),
                               "velocity_x_mps": float(direction * speed), "velocity_y_mps": float(rng.uniform(-0.65, 0.65)),
                               "object_height_m": float(rng.uniform(0.65, 1.1) if condition == "small_fast" else rng.uniform(1.2, 2.4)),
                               "object_width_m": float(rng.uniform(0.5, 0.8) if condition == "small_fast" else rng.uniform(0.85, 1.6)),
                               "object_depth_m": float(rng.uniform(0.85, 1.6)), "shadow_factor": float(rng.uniform(0.06, 0.20)),
                               "bandwidth_hz": float(rng.uniform(550e6, 720e6)), "carrier_hz": float(rng.uniform(34e9, 36e9))})


def make_dataset(root, count, seed, profile="mixed"):
    if profile == "proposal":
        make_proposal_dataset(root, count, seed)
        return
    if count < 12:
        raise ValueError("dataset_size must be at least 12")
    if profile == "sar" and count < 30:
        raise ValueError("dataset_size must be at least 30 for the sar profile to cover all six validation conditions")
    rng = np.random.default_rng(seed)
    n_train, n_val = count // 2, max(3, count // 5)
    entries, labels = [], []
    for index in range(count):
        split = "train" if index < n_train else "validation" if index < n_train + n_val else "test"
        local_index = index - {"train": 0, "validation": n_train, "test": n_train + n_val}[split]
        source = "sar" if local_index % 3 == 1 else "geometry"
        condition = ("texture", "clutter", "empty")[local_index % 3]
        if profile == "sar":
            source, condition = "sar", SAR_CONDITIONS[local_index % len(SAR_CONDITIONS)]
        elif split == "test" and local_index >= 3:
            condition = ("high_noise", "small_fast", "exit")[((local_index - 3) % 3)]
            source = "sar" if condition in ("high_noise", "exit") else "geometry"
        group = f"scene-{index:03d}"
        config = sar_scene(rng, condition, split) if profile == "sar" else merged(SIMULATION, {
            "scene": "empty" if condition == "empty" else "geometry",
            "background": "texture" if condition == "texture" else "clutter",
            "noise": 0.65 if condition == "high_noise" else float(rng.uniform(0.08, 0.28)),
            "seed": int(rng.integers(0, 2**32 - 1)), "width": 72, "height": 56,
            "frames": 14, "pulses_per_frame": 41,
            "start_x_m": 6.3 if condition == "exit" else float(rng.uniform(-5.2, -2.2)),
            "start_y_m": float(rng.uniform(-3.2, 1.0)),
            "velocity_x_mps": (3.1 if condition in ("small_fast", "exit") else
                               float(rng.uniform(0.8, 2.3)) * (-1 if local_index % 4 == 3 else 1)),
            "velocity_y_mps": float(rng.uniform(-0.7, 0.75)),
            "object_height_m": 0.75 if condition == "small_fast" else float(rng.uniform(1.15, 2.25)),
            "object_width_m": 0.6 if condition == "small_fast" else float(rng.uniform(0.8, 1.5)),
            "object_depth_m": float(rng.uniform(0.8, 1.6)),
            "shadow_factor": float(rng.uniform(0.08, 0.22)),
        })
        frames, masks, meta, geom = (form_video(config)[:4] if source == "sar" else simulate(config))
        if profile == "sar":
            # Visibility can vary with the random scattering field. Recreate the same condition when needed.
            for _ in range(8):
                visible = masks.any(axis=(1, 2))
                if condition == "empty" or (visible.any() and
                        (condition != "exit" or (visible[0] and not visible[-1]))):
                    break
                config = sar_scene(rng, condition, split)
                frames, masks, meta, geom = form_video(config)[:4]
            visible = masks.any(axis=(1, 2))
            if (bool(visible.any()) != (condition != "empty") or
                    (condition == "exit" and (not visible[0] or visible[-1]))):
                raise ValueError(f"{group}: generated labels do not satisfy condition {condition}")
        meta.update(scene_id=group, trajectory_family=group, split=split, condition=condition)
        save_sequence(root / group, frames, masks, meta, geom)
        entries.append({"path": group, "scene_id": group, "trajectory_family": group, "split": split,
                        "condition": condition, "source": source})
        if profile == "sar":
            labels.append(dict(entries[-1], frames=len(frames), visible_frames=int(visible.sum()),
                               foreground_pixels=int(masks.sum()), pixels=int(masks.size)))
    manifest = {"version": 1, "seed": seed, "entries": entries}
    if profile == "sar":
        manifest["profile"] = profile
        write_csv(root / "labels_by_scene.csv", labels)
    write_json(root / "dataset.json", manifest)


def read_split(root):
    root = Path(root).resolve()
    manifest = read_json(root / "dataset.json")
    if manifest.get("version") != 1 or not isinstance(manifest.get("entries"), list):
        raise ValueError("invalid dataset manifest")
    partitions = {"train": [], "validation": [], "test": []}
    seen_paths = set()
    groups = {"scene_id": {}, "trajectory_family": {}}
    for row in manifest["entries"]:
        split = row["split"]
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or path in seen_paths or split not in partitions:
            raise ValueError("dataset paths must be unique, internal and assigned to a valid split")
        seen_paths.add(path)
        for field, group_splits in groups.items():
            if not row[field] or group_splits.setdefault(row[field], split) != split:
                raise ValueError(f"{field} appears across splits: {row[field]}")
        partitions[split].append((path, row))
    if any(not rows for rows in partitions.values()):
        raise ValueError("dataset requires nonempty train, validation and test splits")
    return partitions


def get_sample(path):
    frames, times, scale = load_observations(path)
    return frames, load_masks(path / "masks.npy", frames.shape), times, scale, read_json(path / "metadata.json")


def tune(samples, scorer, thresholds, area):
    scored = [(scorer(frame), mask) for frame, mask, *_ in samples]
    choice, best, results = thresholds[0], -1, []
    for threshold in thresholds:
        intersection = union = 0
        for scores, truth in scored:
            prediction = segment(scores, threshold, area)
            intersection += np.count_nonzero(prediction & truth)
            union += np.count_nonzero(prediction | truth)
        value = float(intersection / union) if union else 0.0
        results.append({"threshold": threshold, "validation_iou": value})
        if value > best:
            choice, best = threshold, value
    return choice, results


def train_models(root, train, validation, epochs, seed, area, network="compact"):
    results = {}
    for name, temporal in (("single_frame", False), ("causal_temporal", True)):
        model, history = fit([(f, m) for f, m, *_ in train], temporal, epochs, seed, network)
        threshold, scores = tune(validation, lambda f: predict(f, model),
                                 ((0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98)
                                  if network == "context" else (0.35, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.98)), area)
        save_model(root / f"{name}.npz", model)
        settings = {"threshold": threshold, "min_area": area, "epochs": epochs, "seed": seed,
                    "input": "current_and_past_frames" if temporal else "current_frame",
                    "weighted_training_loss": history, "validation": scores, "backend": "NumPy CPU"}
        if network != "compact":
            settings.update(network=network, parameters=sum(model[key].size for key in ("kernel", "bias", "head", "intercept")))
        write_json(root / f"{name}.json", settings)
        results[name] = (model, threshold)
    return results


def aggregate(rows):
    totals = {key: sum(row[key] for row in rows) for key in (
        "intersection_pixels", "union_pixels", "predicted_pixels", "truth_pixels", "matched", "false_positive", "missed",
        "visible_frames", "located_visible_frames", "measured_visible_frames", "location_squared_error_px2", "predicted_on_invisible_frames")}
    support = totals["predicted_pixels"] + totals["truth_pixels"]
    located = totals["located_visible_frames"]
    visible = totals["visible_frames"]
    stats = dict(totals, scenes=len(rows), iou=totals["intersection_pixels"] / totals["union_pixels"] if totals["union_pixels"] else None,
                 f1=2 * totals["intersection_pixels"] / support if support else None,
                 continuity=located / visible if visible else None,
                 location_rmse_px=float(np.sqrt(totals["location_squared_error_px2"] / located)) if located else None,
                 elapsed_s=sum(row["elapsed_s"] for row in rows))
    for key in ("velocity_error_px_s", "velocity_error_mps", "target_velocity_error_mps", "target_location_rmse_m", "target_direction_error_deg"):
        observed = [row[key] for row in rows if row.get(key) is not None]
        stats[key] = float(np.mean(observed)) if observed else None
        stats[f"{key}_sample_count"] = len(observed)
    return stats


def paired_iou_intervals(rows, methods, seed):
    """Bootstrap whole test sequences, using the same sampled scenes for each method."""
    cases = sorted({row["scene_id"] for row in rows})
    by_method = {(row["method"], row["scene_id"]): row for row in rows}
    rng = np.random.default_rng(seed)
    selection = rng.integers(len(cases), size=(2000, len(cases)))
    sampled = {}
    for name in methods:
        stats = [by_method[name, case] for case in cases]
        intersection = np.array([row["intersection_pixels"] for row in stats])[selection].sum(axis=1)
        union = np.array([row["union_pixels"] for row in stats])[selection].sum(axis=1)
        sampled[name] = np.divide(intersection, union, out=np.full(union.shape, np.nan), where=union > 0)
    intervals = {}
    for name, values in sampled.items():
        if name == "causal_temporal":
            continue
        gain = sampled["causal_temporal"] - values
        valid = gain[np.isfinite(gain)]
        intervals[name] = {"valid_replicates": int(valid.size),
                           "iou_gain_lower_95": float(np.quantile(valid, 0.025)) if valid.size else None,
                           "iou_gain_upper_95": float(np.quantile(valid, 0.975)) if valid.size else None}
    return {"unit": "test_sequence", "replicates": len(selection), "seed": seed,
            "reference": "causal_temporal", "paired_iou_gain": intervals}


def benchmark(root, dataset_root, config):
    splits = read_split(dataset_root)
    train = [get_sample(path) for path, _ in splits["train"]]
    validation = [get_sample(path) for path, _ in splits["validation"]]
    models_dir = root / "models"
    models_dir.mkdir()
    models = train_models(models_dir, train, validation, config["epochs"], config["seed"], config["min_area"], config["network"])
    methods = {}
    selections = {}
    for method in ("threshold", "difference"):
        scorer = lambda frames, name=method: baseline_scores(frames, name)
        threshold, stats = tune(validation, scorer, (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85), config["min_area"])
        methods[method] = (scorer, threshold)
        selections[method] = stats
    for name, (model, threshold) in models.items():
        methods[name] = (lambda frames, weights=model: predict(frames, weights), threshold)
        selections[name] = read_json(models_dir / f"{name}.json")["validation"]
    write_json(root / "validation_selection.json", selections)
    rows, ablation = [], []
    examples = {}
    for path, entry in splits["test"]:
        frames, truth, times, scale, meta = get_sample(path)
        for method, (scorer, threshold) in methods.items():
            start = perf_counter()
            prediction = segment(scorer(frames), threshold, config["min_area"])
            tracked = track(prediction, times, scale, config["gate_px"], config["max_gap"])
            elapsed = perf_counter() - start
            metrics = score_masks(prediction, truth, config["match_iou"])
            metrics.pop("frames")
            motion = score_track(tracked, truth, scale, meta)
            motion.pop("true_shadow_velocity_px_s")
            row = dict(entry, method=method, threshold=threshold, **metrics, **motion, elapsed_s=elapsed)
            rows.append(row)
            artifact = root / "predictions" / entry["scene_id"] / method
            artifact.mkdir(parents=True)
            np.save(artifact / "predictions.npy", prediction.astype(np.uint8))
            write_json(artifact / "tracks.json", tracked)
            save_receipt(artifact, path, detector=method, threshold=threshold)
            if method == "causal_temporal":
                if truth.any():
                    examples[entry["scene_id"]] = (row["iou"], frames, truth, prediction, tracked)
    for name in ("best", "failure"):
        selector = max if name == "best" else min
        if examples:
            case_id = selector(examples, key=lambda case: examples[case][0])
            value, frames, truth, prediction, tracked = examples[case_id]
            cases = np.unique(np.linspace(0, len(frames) - 1, min(4, len(frames))).round().astype(int)).tolist()
            save_preview(root / f"{name}.png", frames, truth, prediction, tracked, cases)
            write_json(root / f"{name}.json", {"scene_id": case_id, "iou": value, "frames": cases})
    for path, entry in splits["test"]:
        frames, truth, times, scale, meta = get_sample(path)
        if np.count_nonzero(truth.any(axis=(1, 2))) < 8:
            continue
        scorer, threshold = methods["causal_temporal"]
        prediction = segment(scorer(frames), threshold, config["min_area"])
        absent = (len(frames) // 2, len(frames) // 2 + 1)
        prediction[list(absent)] = False
        for gap in (0, 2):
            tracked = track(prediction, times, scale, config["gate_px"], gap)
            metrics = score_track(tracked, truth, scale, meta)
            ablation.append({"scene_id": entry["scene_id"], "max_gap": gap, "dropped_frames": str(absent),
                             "imputed_dropped_frames": sum(tracked["frames"][i]["source"] == "imputed" for i in absent),
                             "continuity": metrics["continuity"], "location_rmse_px": metrics["location_rmse_px"],
                             "location_squared_error_px2": metrics["location_squared_error_px2"],
                             "located_visible_frames": metrics["located_visible_frames"], "visible_frames": metrics["visible_frames"]})
    overall = [dict(method=name, **aggregate([row for row in rows if row["method"] == name])) for name in methods]
    groups = []
    for name in methods:
        for condition in sorted({row["condition"] for row in rows}):
            group = [row for row in rows if row["method"] == name and row["condition"] == condition]
            groups.append(dict(method=name, condition=condition, **aggregate(group)))
    write_csv(root / "metrics_by_scene.csv", rows)
    write_csv(root / "metrics_by_condition.csv", groups)
    write_csv(root / "metrics.csv", overall)
    write_csv(root / "tracking_ablation.csv", ablation)
    write_json(root / "iou_intervals.json", paired_iou_intervals(rows, methods, config["seed"]))
    write_json(root / "results.json", {"summary": overall, "conditions": groups, "tracking_ablation": ablation,
                                      "test_cases": [row for _, row in splits["test"]], "matched_iou": config["match_iou"]})
