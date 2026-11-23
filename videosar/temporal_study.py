from collections import Counter
from hashlib import sha256

import numpy as np

from .analysis import score_masks
from .data import load_masks, read_json, write_json
from .open_compare import (COMMON_THRESHOLDS, HISTORICAL, comparison_gate, evaluate_models,
                           new_test_specs, read_reference_dataset)
from .small_shadow import AREAS, OPERATING_FIELDS, make_dataset, operating_rank, recall_for, tune
from .stress import MATCH, PROPOSAL
from .study import read_registered
from .unet import create_model, fit, load_weights, predict, runtime
from .visuals import write_csv


SEED = 97341
EPOCHS = 6
LEARNING_RATE = 0.0002
VARIANTS = (("continued", 0, False), ("median_history", 5, False), ("weighted_history", 5, True))
SIZE_BANDS = ("1–11", "12–23", "24–47", "48+")


def expand_history_stem(torch, smp, original, history_frames):
    if not history_frames:
        return original
    model = create_model(torch, smp, True, history_frames=history_frames).to(memory_format=torch.channels_last)
    weights = original.state_dict()
    key = "encoder.conv1.weight"
    stem = weights[key]
    weights[key] = torch.cat((stem, torch.zeros_like(stem[:, :1])), dim=1)
    model.load_state_dict(weights)
    return model


def write_size_breakdown(root, test, methods):
    counts, matched = Counter(), Counter()
    for path, entry in test:
        truth = load_masks(path / "masks.npy", (20, 64, 80))
        areas = truth.sum(axis=(1, 2))
        bands = np.digitize(areas, (12, 24, 48))
        counts.update(SIZE_BANDS[i] for i in bands[areas > 0])
        for name in methods:
            prediction = load_masks(root / "predictions" / entry["scene_id"] / name / "predictions.npy", truth.shape)
            records = score_masks(prediction, truth, MATCH)["frames"]
            matched.update((name, SIZE_BANDS[bands[i]]) for i, row in enumerate(records)
                           if row["target_visible"] and row["best_component_iou"] >= MATCH)
    write_csv(root / "recall_by_shadow_area.csv", [
        {"method": name, "label_area_px": band, "visible_frames": counts[band], "matched": matched[name, band],
         "recall": matched[name, band] / counts[band] if counts[band] else None}
        for name in methods for band in SIZE_BANDS])


def run_temporal_study(root, dataset):
    torch, smp, correlate = runtime()
    torch.set_num_threads(min(6, torch.get_num_threads()))
    prior = PROPOSAL.parent / "small-shadow"
    comparison = PROPOSAL.parent / "open-compare"
    native_path = prior / "models" / "small_shadow.json"
    fpn_path = comparison / "models" / "fpn.json"
    native, fpn = read_json(native_path), read_json(fpn_path)
    native_weights = native_path.with_name(native["checkpoint"])
    original = load_weights(torch, smp, native_weights, native, "cpu")
    fpn_model = load_weights(torch, smp, fpn_path.with_name(fpn["checkpoint"]), fpn, "cpu")
    original_point = {key: native[key] for key in OPERATING_FIELDS}
    reference_digest = read_json(prior / "frozen_validation.json")["dataset_sha256"]
    source = read_reference_dataset(dataset, reference_digest)
    specs = list(new_test_specs(SEED, "history"))
    seen_seeds = {row["config"]["seed"] for row in read_json(prior / "protocol.json")["scenes"]}
    seen_seeds.update(row["config"]["seed"] for row in read_json(comparison / "protocol.json")["test_scenes"])
    if seen_seeds & {row["config"]["seed"] for row in specs}:
        raise ValueError("new test seed overlaps the prior datasets")
    write_json(root / "protocol.json", {
        "seed": SEED, "match_iou": MATCH, "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
        "batch_size": 32, "thresholds": COMMON_THRESHOLDS, "areas": AREAS, "variants": VARIANTS,
        "training_dataset_sha256": reference_digest, "test_scenes": specs,
        "native_sha256": native["weight_sha256"], "fpn_sha256": fpn["weight_sha256"],
        "torch_version": torch.__version__, "smp_version": smp.__version__})
    test = make_dataset(root, specs, seed=SEED, profile="temporal_test", counts=(("test", 9),))["test"]
    train = read_registered(source["train"], correlate)
    validation = read_registered(source["validation"], correlate)
    train_pairs = [(s["aligned"], s["truth"]) for s in train]
    val_pairs = [(s["aligned"], s["truth"]) for s in validation]
    scores = [(predict(torch, original, s["aligned"], native, "cpu", 32), s["truth"]) for s in validation]
    normal_ref = recall_for(scores, validation, original_point, ("normal",))
    registry = {HISTORICAL: (original, native), "fpn": (fpn_model, fpn)}
    models = root / "models"
    models.mkdir()
    selected = {}
    for name, length, weighted in VARIANTS:
        print(f"Training {name} with {length} earlier frames, weighted loss: {weighted}", flush=True)
        parent = load_weights(torch, smp, native_weights, native, "cpu")
        expanded = expand_history_stem(torch, smp, parent, length)
        model, settings = fit(torch, smp, train_pairs, val_pairs, temporal=True, augment_noise=False, epochs=EPOCHS,
                             seed=SEED, device="cpu", batch_size=32, initialization=(expanded, native),
                             learning_rate=LEARNING_RATE, history_frames=length, weight_small_shadows=weighted)
        scores = [(predict(torch, model, s["aligned"], settings, "cpu", 32), s["truth"]) for s in validation]
        point, scan = tune(scores, validation, normal_ref, COMMON_THRESHOLDS)
        write_csv(root / f"validation_{name}.csv", scan)
        checkpoint = models / f"{name}.pt"
        torch.save(model.state_dict(), checkpoint)
        settings.update(checkpoint=checkpoint.name, weight_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
                        registered=True, parent_weight_sha256=native["weight_sha256"],
                        **{key: point[key] for key in OPERATING_FIELDS})
        write_json(checkpoint.with_suffix(".json"), settings)
        selected[name] = {"weight_sha256": settings["weight_sha256"], "setting": point}
        registry[name] = (model, settings)
    candidate = max(selected, key=lambda name: operating_rank(selected[name]["setting"], normal_ref))
    write_json(root / "frozen_validation.json", {
        "test_dataset_sha256": sha256((root / "dataset" / "dataset.json").read_bytes()).hexdigest(),
        "selected_candidate": candidate, "native_normal_recall": normal_ref, "candidates": selected,
        "baseline_operating_points": {name: {k: settings[k] for k in OPERATING_FIELDS}
                                     for name, settings in ((HISTORICAL, native), ("fpn", fpn))}})
    print("Validation choices frozen. Evaluating the fresh test set.", flush=True)
    groups = evaluate_models(root, test, registry, torch, correlate)
    write_size_breakdown(root, test, registry)
    table = {(row["method"], row["group"]): row for row in groups}
    write_json(root / "results.json", {"groups": groups, "selected_candidate": candidate,
                                     "gates": {name: comparison_gate(table, name) for name in selected}})
