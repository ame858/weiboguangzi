from collections import Counter
from hashlib import sha256
from time import perf_counter

import numpy as np

from .analysis import score_masks, target_summary, track
from .data import load_masks, load_observations, read_json, write_json
from .proposal_data import scene_digest
from .stress import MATCH, PROPOSAL, save_scene, scene_specs
from .study import aggregate_detection, calibrate, cleanup, read_registered
from .unet import fit, load_weights, predict, runtime
from .visuals import save_preview, write_csv


SEED = 88731
METHODS = ("frozen", "retuned", "finetuned")
CONDITIONS = ("normal", "small", "small_slow_dark", "empty")
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.78, 0.83, 0.88, 0.93)
AREAS = (4, 8, 12, 16, 24)
OPERATING_FIELDS = ("threshold", "min_area", "closing")
EPOCHS = 8
LEARNING_RATE = 0.00035


def independent_specs(seed=SEED):
    for offset, (split, count) in enumerate((("train", 12), ("validation", 6), ("test", 9))):
        for row in scene_specs(seed + offset, count * 3, count):
            i = int(row["scene_family"].split("-")[-1])
            group = i // count
            condition = CONDITIONS[group]
            variant = ("static_dark" if i % 2 else "reference", "small", "combined", "empty_static_dark")[group]
            if row["condition"] == variant:
                identifier = f"small-{split}-{i:02d}"
                yield row | {"scene_id": identifier, "scene_family": identifier, "condition": condition, "split": split}


def make_dataset(root, specs, *, seed=SEED, profile="small_shadow", counts=(("train", 12), ("validation", 6), ("test", 9))):
    dataset = root / "dataset"
    splits = {key: [] for key, _ in counts}
    labels = []
    known_seeds = {spec["config"]["seed"] for spec in scene_specs()}
    for spec in specs:
        seed = spec["config"]["seed"]
        if seed in known_seeds:
            raise ValueError("scene seed repeats between independent evaluations")
        known_seeds.add(seed)
        path = save_scene(dataset, spec)
        frames, _, _ = load_observations(path)
        if frames.shape != (20, 64, 80):
            raise ValueError(f"unexpected sequence shape: {path}")
        truth = load_masks(path / "masks.npy", frames.shape)
        geometry = load_masks(path / "geometric_masks.npy", frames.shape)
        if bool(truth.any()) != (spec["condition"] != "empty") or bool(geometry.any()) != bool(truth.any()):
            raise ValueError(f"generated masks differ from the planned condition: {path}")
        entry = {key: spec[key] for key in ("scene_id", "scene_family", "condition", "split")}
        entry.update(path=path.name, source="sar", sha256=scene_digest(path))
        splits[entry["split"]].append((path, entry))
        labels.append(dict(entry, frames=len(truth), visible_frames=int(truth.any(axis=(1, 2)).sum())))
    for split, count in counts:
        entries = splits[split]
        if Counter(row["condition"] for _, row in entries) != dict.fromkeys(CONDITIONS, count):
            raise ValueError("dataset condition counts differ from the protocol")
    write_json(dataset / "dataset.json", {"profile": profile, "seed": seed,
                                         "entries": [row for entries in splits.values() for _, row in entries]})
    write_csv(dataset / "labels_by_scene.csv", labels)
    return splits


def operating_rank(row, normal_reference):
    precision, recall = row["precision"] or 0, row["recall"] or 0
    stable = row["normal_recall"] >= normal_reference - 0.05
    valid = precision >= 0.8
    return (valid and stable and recall >= 0.8, valid and stable, valid,
            row["small_recall"] if valid else row["target_f1"] or 0,
            row["target_f1"] or 0, -row["false_positive"])


def tune(scored, validation, normal_reference, thresholds=THRESHOLDS, areas=AREAS):
    by_group = {group: [(scores, truth) for (scores, truth), sample in zip(scored, validation)
                        if sample["entry"]["condition"] in conditions]
                for group, conditions in (("normal", ("normal",)), ("small", ("small", "small_slow_dark")))}
    _, rows = calibrate(scored, thresholds, areas, True, MATCH)
    for group, pairs in by_group.items():
        _, checks = calibrate(pairs, thresholds, areas, True, MATCH)
        by_point = {(row["threshold"], row["min_area"]): row["recall"] for row in checks}
        for row in rows:
            row[f"{group}_recall"] = by_point[row["threshold"], row["min_area"]] or 0
    return max(rows, key=lambda row: operating_rank(row, normal_reference)), rows


def measure(predicted, truth):
    metrics = score_masks(predicted, truth, MATCH)
    frames = metrics.pop("frames")
    metrics.update(target_summary(metrics["matched"], metrics["false_positive"], metrics["missed"]))
    return metrics, frames


def recall_for(scored, validation, point, conditions):
    tp = missed = 0
    for (scores, truth), sample in zip(scored, validation):
        if sample["entry"]["condition"] in conditions:
            metrics, _ = measure(cleanup(scores, **point), truth)
            tp += metrics["matched"]
            missed += metrics["missed"]
    return tp / (tp + missed)


def run_small_shadow(root):
    settings_path = PROPOSAL / "models" / "unet_temporal_post.json"
    source = read_json(settings_path)
    weights = settings_path.with_name(source["checkpoint"])
    original = {key: source[key] for key in OPERATING_FIELDS}
    torch, smp, correlate = runtime()
    torch.set_num_threads(min(6, torch.get_num_threads()))
    frozen = load_weights(torch, smp, weights, source, "cpu")
    specs = list(independent_specs())
    write_json(root / "protocol.json", {
        "seed": SEED, "match_iou": MATCH, "source_checkpoint_sha256": source["weight_sha256"],
        "source_settings_sha256": sha256(settings_path.read_bytes()).hexdigest(), "initial_operating_point": original,
        "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "batch_size": 32,
        "limits": {"precision": 0.8, "recall": 0.8, "small_recall_gain": 0.10, "normal_recall_max_drop": 0.05},
        "thresholds": THRESHOLDS, "areas": AREAS, "scenes": specs})
    splits = make_dataset(root, specs)
    print("108 SAR sequences generated. Loading train and validation splits.", flush=True)
    train = read_registered(splits["train"], correlate)
    validation = read_registered(splits["validation"], correlate)
    val_scores = [(predict(torch, frozen, s["aligned"], source, "cpu", 32), s["truth"]) for s in validation]
    normal_reference = recall_for(val_scores, validation, original, ("normal",))
    retuned, scan = tune(val_scores, validation, normal_reference)
    write_csv(root / "validation_retuned.csv", scan)
    fresh_model = load_weights(torch, smp, weights, source, "cpu")
    pairs = lambda samples: [(s["aligned"], s["truth"]) for s in samples]
    adapted, settings = fit(torch, smp, pairs(train), pairs(validation), temporal=True, augment_noise=False, epochs=EPOCHS,
                            seed=SEED, device="cpu", batch_size=32, initialization=(fresh_model, source),
                            learning_rate=LEARNING_RATE)
    val_scores = [(predict(torch, adapted, s["aligned"], settings, "cpu", 32), s["truth"]) for s in validation]
    finetuned, scan = tune(val_scores, validation, normal_reference)
    write_csv(root / "validation_finetuned.csv", scan)

    models = root / "models"
    models.mkdir()
    checkpoint = models / "small_shadow.pt"
    torch.save({key: value.detach().cpu() for key, value in adapted.state_dict().items()}, checkpoint)
    settings.update(checkpoint=checkpoint.name, weight_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
                    parent_weight_sha256=source["weight_sha256"], registered=True,
                    **{key: finetuned[key] for key in OPERATING_FIELDS})
    write_json(checkpoint.with_suffix(".json"), settings)
    chosen = max(("retuned", "finetuned"), key=lambda method: operating_rank(
        {"retuned": retuned, "finetuned": finetuned}[method], normal_reference))
    points = {"frozen": original, "retuned": {k: retuned[k] for k in OPERATING_FIELDS},
              "finetuned": {k: finetuned[k] for k in OPERATING_FIELDS}}
    write_json(root / "frozen_validation.json", {
        "dataset_sha256": sha256((root / "dataset" / "dataset.json").read_bytes()).hexdigest(),
        "selected_candidate": chosen, "normal_recall_reference": normal_reference, "retuned": retuned,
        "finetuned": finetuned, "operating_points": points, "finetuned_sha256": settings["weight_sha256"]})
    print("Operating points frozen. Evaluating independent test split.", flush=True)
    rows, rescue = [], None
    for sample in read_registered(splits["test"], correlate):
        entry, aligned, truth = (sample[key] for key in ("entry", "aligned", "truth"))
        scores = {}
        for name, model, meta in (("frozen", frozen, source), ("finetuned", adapted, settings)):
            start = perf_counter()
            scores[name] = predict(torch, model, aligned, meta, "cpu", 32), perf_counter() - start
        saved = {}
        for name in METHODS:
            probability, inference = scores["frozen" if name == "retuned" else name]
            start = perf_counter()
            prediction = cleanup(probability, **points[name])
            elapsed = inference + sample["registration_s"] + (perf_counter() - start)
            metrics, details = measure(prediction, truth)
            folder = root / "predictions" / entry["scene_id"] / name
            folder.mkdir(parents=True)
            np.save(folder / "predictions.npy", prediction.astype(np.uint8))
            write_csv(folder / "per_frame.csv", [dict(frame=i, **row) for i, row in enumerate(details)])
            rows.append(dict(entry, method=name, frames=len(truth), elapsed_s=elapsed, **metrics))
            saved[name] = prediction, details
        if rescue is None and entry["condition"] in ("small", "small_slow_dark"):
            selected, selected_frames = saved[chosen]
            for i, (before, after) in enumerate(zip(saved["frozen"][1], selected_frames)):
                if before["target_visible"] and before["best_component_iou"] < MATCH <= after["best_component_iou"]:
                    rescue = {"scene_id": entry["scene_id"], "frame": i, "method": chosen,
                              "before_iou": before["best_component_iou"], "after_iou": after["best_component_iou"]}
                    for label, prediction in (("before", saved["frozen"][0]), ("after", selected)):
                        tracked = track(prediction, sample["times"], sample["scale"], 12, 2)
                        save_preview(root / f"recovery_{label}.png", aligned, truth, prediction, tracked, [i])
                    break

    groups = []
    for name in METHODS:
        for group in ("all", *CONDITIONS, "all_small"):
            selected = [row for row in rows if row["method"] == name and
                        (group == "all" or row["condition"] == group or
                         (group == "all_small" and row["condition"] in ("small", "small_slow_dark")))]
            groups.append(dict(method=name, group=group, **aggregate_detection(selected)))
    metrics = {(row["method"], row["group"]): row for row in groups}
    gates = {}
    for method in METHODS[1:]:
        all_cases = metrics[method, "all"]
        gain = metrics[method, "all_small"]["recall"] - metrics["frozen", "all_small"]["recall"]
        guard = metrics[method, "normal"]["recall"] - metrics["frozen", "normal"]["recall"]
        gates[method] = {"passed": (all_cases["precision"] or 0) >= 0.8 and all_cases["recall"] >= 0.8
                         and gain >= 0.10 and guard >= -0.05, "small_recall_gain": gain, "normal_recall_change": guard}
    write_csv(root / "metrics_by_scene.csv", rows)
    write_csv(root / "metrics_by_group.csv", groups)
    write_json(root / "results.json", {"selected_candidate": chosen, "gates": gates, "groups": groups, "recovery_case": rescue})
