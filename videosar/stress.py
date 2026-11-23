from collections import Counter
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from .analysis import baseline_scores, score_masks, target_summary, track
from .config import SIMULATION, merged
from .data import load_masks, load_observations, read_json, save_sequence, write_json
from .proposal_data import scene_digest, translate
from .simulation import form_video
from .study import aggregate_detection, cleanup, read_registered
from .unet import load_weights, predict, runtime
from .visuals import save_preview, write_csv


SEED = 73191
MATCH = 0.5
GATE = 0.80
VARIANTS = ("reference", "slow", "small", "static_dark", "combined")
CONDITIONS = (*VARIANTS, "empty_static_dark")
MISS_STAGES = ("raw_segmentation", "closing", "area_filter")
PROPOSAL = Path(__file__).resolve().parents[1] / "deliverables" / "proposal"


def scene_specs(seed=SEED, paired_families=12, empty_families=12):
    for i in range(paired_families + empty_families):
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        empty = i >= paired_families
        heading = rng.choice([-1, 1])
        speed = heading * rng.uniform(1.45, 2.15)
        slower = heading * rng.uniform(0.35, 0.65)
        lateral = rng.uniform(-0.50, 0.50)
        mid_x, mid_y = rng.uniform(-0.8, 0.8), rng.uniform(-3.5, -2.0)
        strong_noise = i % 3 == 2
        base = merged(SIMULATION, {
            "scene": "empty" if empty else "geometry", "background": "texture" if i % 2 else "uniform",
            "frames": 20, "height": 64, "width": 80, "pulses_per_frame": 61,
            "velocity_x_mps": float(speed), "velocity_y_mps": float(lateral),
            "start_x_m": float(mid_x - speed * 1.9), "start_y_m": float(mid_y - lateral * 1.9),
            "object_width_m": float(rng.uniform(0.65, 1.6)), "object_depth_m": float(rng.uniform(0.75, 1.6)),
            "object_height_m": float(rng.uniform(0.8, 2.45)), "shadow_factor": float(rng.uniform(0.045, 0.22)),
            "noise": float(rng.uniform(0.22, 0.36) if strong_noise else rng.uniform(0.045, 0.14)),
            "thermal_noise": float(rng.uniform(0.20, 0.36) if strong_noise else rng.uniform(0.055, 0.12)),
            "seed": int(rng.integers(0, 2**32 - 1)), "carrier_hz": float(rng.uniform(34e9, 36e9)),
            "bandwidth_hz": float(rng.uniform(550e6, 720e6)),
        })
        dark = [float(rng.uniform(-5, 5)), float(rng.uniform(3.8, 5.0)), 2.8, 2.2, 0.18]
        offsets = np.zeros((20, 2))
        if i % 2:
            offsets[1:] = rng.uniform(-1.65, 1.65, size=(19, 2))
        for variant in ("empty_static_dark",) if empty else VARIANTS:
            cfg = base.copy()
            if variant in ("slow", "combined"):
                slow_lateral = lateral * (slower / speed)
                cfg.update(velocity_x_mps=float(slower), start_x_m=float(mid_x - slower * 1.9),
                           velocity_y_mps=float(slow_lateral), start_y_m=float(mid_y - slow_lateral * 1.9))
            if variant in ("small", "combined"):
                for key in ("object_width_m", "object_depth_m", "object_height_m"):
                    cfg[key] *= 0.60
            patch = dark if variant in ("static_dark", "combined", "empty_static_dark") else None
            yield {"scene_id": f"stress-{i:02d}-{variant}", "scene_family": f"stress-{i:02d}",
                   "condition": variant, "config": cfg, "static_dark": patch, "offsets_xy_px": offsets.tolist()}


def save_scene(dataset, spec):
    frames, masks, meta, geometric, _ = form_video(spec["config"], normalize_power=False, static_dark=spec["static_dark"])
    offsets = np.asarray(spec["offsets_xy_px"])
    if offsets.any():
        frames = np.stack([translate(frame, dx, dy) for frame, (dx, dy) in zip(frames, offsets)])
    meta.update(scene_id=spec["scene_id"], scene_family=spec["scene_family"], condition=spec["condition"],
                static_dark=spec["static_dark"], observed_offsets_xy_px=spec["offsets_xy_px"], label_coordinates="fixed_ground_grid")
    if "split" in spec:
        meta["split"] = spec["split"]
    path = dataset / spec["scene_id"]
    save_sequence(path, frames, masks, meta, geometric)
    return path


def build_dataset(root, specs):
    dataset = root / "dataset"
    rows, labels = [], []
    for spec in specs:
        path = save_scene(dataset, spec)
        frames, _, _ = load_observations(path)
        if frames.shape != (20, 64, 80):
            raise ValueError(f"invalid stress sequence shape: {path}")
        labels_true = load_masks(path / "masks.npy", frames.shape)
        geometry = load_masks(path / "geometric_masks.npy", frames.shape)
        if spec["condition"] == "empty_static_dark" and (labels_true.any() or geometry.any()):
            raise ValueError(f"negative sequence contains target labels: {path}")
        meta = read_json(path / "metadata.json")
        if meta["effective_config"] != spec["config"] or meta["static_dark"] != spec["static_dark"] or meta["observed_offsets_xy_px"] != spec["offsets_xy_px"]:
            raise ValueError(f"stress sequence parameters differ from the protocol: {path}")
        row = {key: spec[key] for key in ("scene_id", "scene_family", "condition")}
        row.update(path=path.name, source="sar", sha256=scene_digest(path))
        rows.append((path, row))
        geometric_visible = geometry.any(axis=(1, 2))
        image_visible = labels_true.any(axis=(1, 2))
        labels.append(dict(row, frames=len(frames), visible_frames=int(image_visible.sum()),
                           geometric_without_visible_label=int((geometric_visible & ~image_visible).sum())))
        if len(rows) % 12 == 0:
            print(f"Generated {len(rows)}/72 SAR sequences", flush=True)
    if Counter(row["condition"] for _, row in rows) != {name: 12 for name in CONDITIONS}:
        raise ValueError("stress dataset condition counts differ from the protocol")
    with TemporaryDirectory(dir=root) as repeat:
        for spec, (_, row) in zip(specs[:5], rows[:5]):
            if scene_digest(save_scene(Path(repeat), spec)) != row["sha256"]:
                raise ValueError("stress scene family failed byte-for-byte replay")
    write_json(dataset / "dataset.json", {"profile": "stress", "seed": SEED, "entries": [row for _, row in rows]})
    write_csv(dataset / "labels_by_scene.csv", labels)
    return rows


def differences(rows):
    reference = {(row["method"], row["scene_family"]): row for row in rows if row["condition"] == "reference"}
    paired = []
    for row in rows:
        if row["condition"] not in VARIANTS[1:]:
            continue
        control = reference[row["method"], row["scene_family"]]
        values = {f"delta_{key}": row[key] - control[key] if row[key] is not None and control[key] is not None else None
                  for key in ("precision", "recall", "target_f1", "false_positive_per_frame")}
        paired.append({key: row[key] for key in ("method", "scene_family", "condition")} | values)
    return paired


def diagnostics(raw, closed, final, match_iou=MATCH):
    missed = final["target_visible"] and final["best_component_iou"] < match_iou
    false_alarms = final["predicted_components"] - int(final["target_visible"] and not missed)
    stage = None
    if missed:
        if closed["best_component_iou"] >= match_iou:
            stage = "area_filter"
        elif raw["best_component_iou"] >= match_iou:
            stage = "closing"
        else:
            stage = "raw_segmentation"
    return {"missed": missed, "false_alarms": false_alarms, "miss_stage": stage,
            "raw_components": raw["predicted_components"], "closed_components": closed["predicted_components"],
            "raw_best_iou": raw["best_component_iou"], "closed_best_iou": closed["best_component_iou"], **final}


def run_stress(root, model_path):
    model_path = model_path or PROPOSAL / "models" / "unet_temporal_post.json"
    settings = read_json(model_path)
    weights = model_path.with_name(settings.get("checkpoint", model_path.with_suffix(".pt").name))
    baseline_record = PROPOSAL / "frozen_validation.json"
    baseline = read_json(baseline_record)["methods"]["difference"]["settings"]
    operating = {key: settings[key] for key in ("threshold", "min_area", "closing")}
    torch, smp, correlate = runtime()
    torch.set_num_threads(min(6, torch.get_num_threads()))
    model = load_weights(torch, smp, weights, settings, "cpu")
    specs = list(scene_specs())
    protocol = {"seed": SEED, "frames_per_sequence": 20, "match_iou": MATCH, "precision_recall_gate": GATE,
                "model_sha256": sha256(weights.read_bytes()).hexdigest(), "settings_sha256": sha256(model_path.read_bytes()).hexdigest(),
                "baseline_source_sha256": sha256(baseline_record.read_bytes()).hexdigest(),
                "torch_version": torch.__version__, "smp_version": smp.__version__,
                "operating_points": {"unet": operating, "difference": baseline}, "scenes": specs}
    write_json(root / "protocol.json", protocol)
    entries = build_dataset(root, specs)
    print("Data verified. Evaluating both frozen methods.", flush=True)
    rows, examples, failure_counts = [], {}, Counter()
    for sample in read_registered(entries, correlate):
        entry, aligned, truth = (sample[key] for key in ("entry", "aligned", "truth"))
        for name, score in (("unet", lambda: predict(torch, model, aligned, settings, "cpu", 32)),
                            ("difference", lambda: baseline_scores(aligned, "difference"))):
            point = protocol["operating_points"][name]
            start = perf_counter()
            scores = score()
            predicted = cleanup(scores, **point)
            elapsed = perf_counter() - start + sample["registration_s"]
            metrics = score_masks(predicted, truth, MATCH)
            frame_metrics = metrics.pop("frames")
            target = root / "predictions" / entry["scene_id"] / name
            target.mkdir(parents=True)
            np.save(target / "predictions.npy", predicted.astype(np.uint8))
            if name == "unet":
                raw = scores >= point["threshold"]
                closed = cleanup(scores, point["threshold"], 1, point["closing"])
                raw_metrics = score_masks(raw, truth, MATCH)["frames"]
                closed_metrics = score_masks(closed, truth, MATCH)["frames"]
                frame_metrics = [diagnostics(*frames) for frames in zip(raw_metrics, closed_metrics, frame_metrics)]
                np.save(target / "scores.npy", scores)
                np.save(target / "raw.npy", raw.astype(np.uint8))
                np.save(target / "before_area_filter.npy", closed.astype(np.uint8))
                for i, frame in enumerate(frame_metrics):
                    if frame["false_alarms"] > 0:
                        rank = (not frame["target_visible"], frame["false_alarms"])
                        if "false_alarm" not in examples or rank > examples["false_alarm"][0]:
                            examples["false_alarm"] = (rank, sample, predicted, i, frame)
                    if frame["missed"]:
                        failure_counts[entry["condition"], frame["miss_stage"]] += 1
                        rank = (frame["miss_stage"] == "area_filter", -frame["best_component_iou"])
                        if "miss" not in examples or rank > examples["miss"][0]:
                            examples["miss"] = (rank, sample, predicted, i, frame)
            write_csv(target / "per_frame.csv", [dict(frame=i, **item) for i, item in enumerate(frame_metrics)])
            saved = load_masks(target / "predictions.npy", truth.shape)
            replay = score_masks(saved, truth, MATCH)
            if any(replay[key] != value for key, value in metrics.items()):
                raise ValueError(f"{entry['scene_id']} {name}: saved-mask evaluation differs")
            rows.append(dict(entry, method=name, frames=len(truth), elapsed_s=elapsed, **metrics,
                             **target_summary(metrics["matched"], metrics["false_positive"], metrics["missed"]),
                             false_positive_per_frame=metrics["false_positive"] / len(truth)))
    for name, (_, sample, predicted, i, frame) in examples.items():
        tracked = track(predicted, sample["times"], sample["scale"], 12, 2)
        save_preview(root / f"{name}.png", sample["aligned"], sample["truth"], predicted, tracked, [i])
        write_json(root / f"{name}.json", {
            "scene_id": sample["entry"]["scene_id"], "condition": sample["entry"]["condition"],
            "frame": i, **frame})
    summary = [dict(method=name, **aggregate_detection([row for row in rows if row["method"] == name]))
               for name in ("unet", "difference")]
    groups = [dict(method=name, condition=condition, **aggregate_detection(
        [row for row in rows if row["method"] == name and row["condition"] == condition]))
        for name in ("unet", "difference") for condition in CONDITIONS]
    paired = differences(rows)
    mean_delta = []
    for name in ("unet", "difference"):
        for condition in VARIANTS[1:]:
            subset = [row for row in paired if row["method"] == name and row["condition"] == condition]
            result = {"method": name, "condition": condition, "families": len(subset)}
            for key in ("delta_precision", "delta_recall", "delta_target_f1", "delta_false_positive_per_frame"):
                values = [row[key] for row in subset if row[key] is not None]
                result[key] = float(np.mean(values)) if values else None
                result[f"{key}_n"] = len(values)
            mean_delta.append(result)
    write_csv(root / "metrics.csv", summary)
    write_csv(root / "metrics_by_condition.csv", groups)
    write_csv(root / "metrics_by_scene.csv", rows)
    write_csv(root / "paired_by_family.csv", paired)
    write_csv(root / "paired_summary.csv", mean_delta)
    write_csv(root / "miss_stages.csv", [
        {"condition": condition, **{stage: failure_counts[condition, stage] for stage in MISS_STAGES}}
        for condition in CONDITIONS])
    passed = all((summary[0][key] or 0) >= GATE for key in ("precision", "recall"))
    write_json(root / "results.json", {"passed": passed, "summary": summary, "conditions": groups,
                                      "paired": mean_delta, "cases_present": sorted(examples)})
