from hashlib import sha256
from shutil import copyfile
from time import perf_counter

import numpy as np

from .analysis import baseline_scores
from .center_box import center_masks
from .data import read_json, write_json
from .small_shadow import (AREAS, CONDITIONS, EPOCHS, LEARNING_RATE, OPERATING_FIELDS, THRESHOLDS,
                           SEED as TRAIN_SEED, independent_specs, make_dataset, measure, recall_for, tune)
from .stress import MATCH, PROPOSAL
from .study import aggregate_detection, cleanup, read_registered
from .unet import FPN, LRASPP, UNET, fit, load_weights, predict, predict_centers, runtime
from .visuals import write_csv


SEED = 96272
INITIAL_TEST_SEED = 96241
COMMON_THRESHOLDS = (0.005, 0.0095, 0.01, 0.0105, 0.0108, 0.011, 0.0112, 0.01135, 0.0115, 0.0117,
                     0.0119, 0.0121, 0.0123, 0.0125, 0.013, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, *THRESHOLDS)
TRAINED = (("unet_scratch", UNET, 1), ("fpn", FPN, 1), ("lraspp", LRASPP, 1), ("unet_2x", UNET, 2))
HISTORICAL = "unet_native_finetuned"
SOURCES = {"segmentation_models_pytorch": "https://github.com/qubvel-org/segmentation_models.pytorch",
           "torchvision": "https://github.com/pytorch/vision/tree/v0.22.1"}


def new_test_specs(seed=SEED, prefix="open"):
    for sample in independent_specs(seed):
        if sample["split"] == "test":
            identifier = sample["scene_id"].replace("small-", f"{prefix}-", 1)
            yield sample | {"scene_id": identifier, "scene_family": identifier}


def read_reference_dataset(dataset, expected_sha):
    path = dataset / "dataset.json"
    if sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("open comparison requires the frozen small-shadow dataset")
    rows = read_json(path)["entries"]
    return {split: [(dataset / entry["path"], entry) for entry in rows if entry["split"] == split]
            for split in ("train", "validation")}


def comparison_gate(metrics, method):
    total = metrics[method, "all"]
    small_gain = metrics[method, "all_small"]["recall"] - metrics[HISTORICAL, "all_small"]["recall"]
    normal_gain = metrics[method, "normal"]["recall"] - metrics[HISTORICAL, "normal"]["recall"]
    return {"passed": ((total["precision"] or 0) >= 0.8 and total["recall"] >= 0.8 and
                       small_gain >= 0.10 and normal_gain >= -0.05),
            "small_recall_change": small_gain, "normal_recall_change": normal_gain}


def run_open_comparison(root, dataset, reuse_models=None):
    torch, smp, correlate = runtime()
    import torchvision

    torch.set_num_threads(min(6, torch.get_num_threads()))
    old = PROPOSAL.parent / "small-shadow"
    source_path = PROPOSAL / "models" / "unet_temporal_post.json"
    old_path = old / "models" / "small_shadow.json"
    source, historical = read_json(source_path), read_json(old_path)
    reference_digest = read_json(old / "frozen_validation.json")["dataset_sha256"]
    reference_splits = read_reference_dataset(dataset, reference_digest)
    initial_weights = source_path.with_name(source["checkpoint"])
    native = load_weights(torch, smp, old_path.with_name(historical["checkpoint"]), historical, "cpu")
    reference_point = {key: historical[key] for key in OPERATING_FIELDS}
    specs = list(new_test_specs())
    past_seeds = {row["config"]["seed"] for row in read_json(old / "protocol.json")["scenes"]}
    past_seeds.update(row["config"]["seed"] for row in independent_specs(INITIAL_TEST_SEED) if row["split"] == "test")
    if past_seeds & {row["config"]["seed"] for row in specs}:
        raise ValueError("new test seed overlaps the previous small-shadow experiment")
    write_json(root / "protocol.json", {
        "test_seed": SEED, "train_seed": TRAIN_SEED, "match_iou": MATCH, "epochs": EPOCHS, "batch_size": 32,
        "scratch_learning_rate": 0.001, "warm_start_learning_rate": LEARNING_RATE,
        "thresholds": COMMON_THRESHOLDS, "areas": AREAS, "methods": TRAINED, "source_projects": SOURCES,
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__, "smp": smp.__version__},
        "historical_sha256": historical["weight_sha256"], "initial_sha256": source["weight_sha256"],
        "train_validation_dataset_sha256": reference_digest, "test_scenes": specs})
    test = make_dataset(root, specs, seed=SEED, profile="open_compare_test", counts=(("test", 9),))["test"]

    validation = read_registered(reference_splits["validation"], correlate)
    if reuse_models:
        prior = read_json(reuse_models / "frozen_validation.json")
        if prior["train_validation_dataset_sha256"] != reference_digest:
            raise ValueError("reused weights must use the same frozen training and validation data")
    else:
        training = read_registered(reference_splits["train"], correlate)
        train_targets = [(s["aligned"], s["truth"]) for s in training]
    validation_targets = [(s["aligned"], s["truth"]) for s in validation]
    historical_scores = [(predict(torch, native, s["aligned"], historical, "cpu", 32), s["truth"]) for s in validation]
    normal_reference = recall_for(historical_scores, validation, reference_point, ("normal",))
    registry = {HISTORICAL: (native, historical)}
    selected = {}
    models = root / "models"
    models.mkdir()
    for name, architecture, scale in TRAINED:
        checkpoint = models / f"{name}.pt"
        if reuse_models:
            previous = reuse_models / "models" / checkpoint.name
            settings = read_json(previous.with_suffix(".json"))
            if (settings["weight_sha256"] != prior["candidates"][name]["weight_sha256"] or
                    settings["architecture"] != architecture or settings.get("input_scale", 1) != scale):
                raise ValueError(f"reused checkpoint disagrees with the recorded configuration: {name}")
            model = load_weights(torch, smp, previous, settings, "cpu")
            copyfile(previous, checkpoint)
        else:
            print(f"Training {name}: {architecture} at {scale}x", flush=True)
            initialization = (load_weights(torch, smp, initial_weights, source, "cpu"), source) if scale == 2 else None
            model, settings = fit(torch, smp, train_targets, validation_targets, temporal=True, augment_noise=False,
                                 epochs=EPOCHS, seed=TRAIN_SEED, device="cpu", batch_size=32,
                                 initialization=initialization, learning_rate=LEARNING_RATE if scale == 2 else 0.001,
                                 architecture=architecture, input_scale=scale)
            torch.save(model.state_dict(), checkpoint)
        scores = [(predict(torch, model, s["aligned"], settings, "cpu", 32), s["truth"]) for s in validation]
        point, scan = tune(scores, validation, normal_reference, COMMON_THRESHOLDS)
        write_csv(root / f"validation_{name}.csv", scan)
        settings.update(checkpoint=checkpoint.name, weight_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
                        registered=True, initialization="proposal_checkpoint" if scale == 2 else "random",
                        **{key: point[key] for key in OPERATING_FIELDS})
        write_json(checkpoint.with_suffix(".json"), settings)
        registry[name] = (model, settings)
        selected[name] = {"weight_sha256": settings["weight_sha256"], "setting": point}
    write_json(root / "frozen_validation.json", {
        "train_validation_dataset_sha256": reference_digest,
        "test_dataset_sha256": sha256((root / "dataset" / "dataset.json").read_bytes()).hexdigest(),
        "normal_recall_reference": normal_reference,
        "historical": {"weight_sha256": historical["weight_sha256"], "setting": reference_point}, "candidates": selected})
    print("All operating points frozen. Evaluating all methods on the shared test set.", flush=True)
    groups = evaluate_models(root, test, registry, torch, correlate)
    table = {(row["method"], row["group"]): row for row in groups}
    write_json(root / "results.json", {"groups": groups, "gates": {
        name: comparison_gate(table, name) for name, _, _ in TRAINED}})


def evaluate_models(root, test, registry, torch, correlate, *, prediction_overrides=None):
    prediction_overrides = prediction_overrides or {}
    rows = []
    for sample in read_registered(test, correlate):
        entry, frames, truth = (sample[key] for key in ("entry", "aligned", "truth"))
        for name, (model, settings) in registry.items():
            start = perf_counter()
            if name in prediction_overrides:
                prediction = prediction_overrides[name](sample)
            elif settings.get("detection_mode") == "center_box":
                scores, distances = predict_centers(torch, model, frames, settings, "cpu", 32)
                prediction = center_masks(scores, distances, settings["threshold"], settings["min_area"])
            else:
                scores = baseline_scores(frames, name) if model is None else predict(torch, model, frames, settings, "cpu", 32)
                prediction = cleanup(scores, **{key: settings[key] for key in OPERATING_FIELDS})
            elapsed = perf_counter() - start + sample["registration_s"]
            metrics, per_frame = measure(prediction, truth)
            rows.append(dict(entry, method=name, frames=len(truth), elapsed_s=elapsed, **metrics))
            folder = root / "predictions" / entry["scene_id"] / name
            folder.mkdir(parents=True)
            np.save(folder / "predictions.npy", prediction.astype(np.uint8))
            write_csv(folder / "per_frame.csv", [dict(frame=i, **r) for i, r in enumerate(per_frame)])

    groups = []
    for method, (_, settings) in registry.items():
        for group in ("all", *CONDITIONS, "all_small"):
            subset = [r for r in rows if r["method"] == method and
                      (group == "all" or r["condition"] == group or
                       (group == "all_small" and r["condition"] in ("small", "small_slow_dark")))]
            groups.append(dict(method=method, group=group, trainable_parameters=settings.get("trainable_parameters", 0),
                               **aggregate_detection(subset)))
    write_csv(root / "metrics_by_scene.csv", rows)
    write_csv(root / "metrics_by_group.csv", groups)
    return groups
