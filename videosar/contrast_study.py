from collections import Counter
import csv

import numpy as np

from .analysis import score_masks, target_summary
from .aperture_study import BANDWIDTH_MHZ, SNR_DB
from .data import load_masks, write_json
from .echo_attribution import FIXED, families, file_hash, frozen_settings, save_variant
from .experiment import read_split
from .open_compare import HISTORICAL, SOURCES, comparison_gate, evaluate_models
from .simulation import paired_aperture_video
from .small_shadow import CONDITIONS, OPERATING_FIELDS, independent_specs, operating_rank, recall_for, tune
from .stress import MATCH, MISS_STAGES, diagnostics, scene_specs
from .study import cleanup, read_registered
from .temporal_study import SIZE_BANDS, write_size_breakdown
from .unet import CONTRAST_WINDOWS, create_model, fit, load_weights, predict, runtime
from .visuals import write_csv


SEED = 101641
TRAIN_SEED = 101645
BOOTSTRAP_SEED = 101646
BOOTSTRAP_DRAWS = 2000
EPOCHS = 6
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
COUNTS = {"train": 48, "validation": 24, "test": 36}
PULSES = (61, 91)
THRESHOLDS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.93)
AREAS = (1, 4, 8, 12, 16, 24)
SMALL = ("small", "small_slow_dark")
TRAINED = (("continued", False), ("local_contrast", True))


def locked_specs(seed=SEED):
    rng = np.random.default_rng(seed)
    source = list(independent_specs(seed))
    existing_seeds = {row["config"]["seed"] for seed in (88731, 96272, 97341)
                      for row in independent_specs(seed)}
    existing_seeds.update(row["config"]["seed"] for seed in (101621, 101631) for row in families(seed))
    existing_seeds.update(row["config"]["seed"] for row in scene_specs())
    if seed != SEED:
        existing_seeds.update(row["config"]["seed"] for row in independent_specs(SEED))
    seeds = {row["config"]["seed"] for row in source}
    if len(seeds) != sum(COUNTS.values()) or seeds & existing_seeds:
        raise ValueError("contrast study requires new independent scene seeds")
    result = []
    for split, total in COUNTS.items():
        for i, group in enumerate(CONDITIONS):
            subset = [row for row in source if (row["split"], row["condition"]) == (split, group)]
            size = total // len(CONDITIONS)
            count_61 = (size + (i < 2)) // 2
            allocation = rng.permutation([61] * count_61 + [91] * (size - count_61))
            for spec, pulses in zip(subset, allocation, strict=True):
                identifier = spec["scene_id"].replace("small-", "contrast-", 1)
                config = FIXED | {"bandwidth_hz": BANDWIDTH_MHZ * 1e6, "pulses_per_frame": int(pulses)}
                # Preserve the original per-scene scatterer, dimensions and trajectory.
                result.append(spec | {"scene_id": identifier, "scene_family": identifier,
                                      "group": "small" if group in SMALL else group, "pulses": int(pulses),
                                      "config": spec["config"] | config})
    return result


def save_contrast_scene(dataset, spec):
    config, output = next(paired_aperture_video(spec["config"], (spec["pulses"],), SNR_DB, spec["static_dark"]))
    row = {key: spec[key] for key in ("scene_id", "scene_family", "split", "condition", "group", "pulses")}
    row.update(path=spec["scene_id"], trajectory_family=spec["scene_family"], source="sar",
               bandwidth_mhz=BANDWIDTH_MHZ, snr_db=SNR_DB, has_static_dark=spec["static_dark"] is not None)
    entry, _, visible, geometry, _ = save_variant(dataset, spec, config, row, output)
    if spec["group"] != "empty" and (not visible.any() or not geometry.any()):
        raise ValueError(f"{spec['scene_id']}: positive scene has no target labels")
    return entry


def build_contrast_dataset(root, specs):
    dataset = root / "dataset"
    rows = []
    for spec in specs:
        _, row = save_contrast_scene(dataset, spec)
        rows.append(row)
        if len(rows) % 12 == 0:
            print(f"Generated and checked {len(rows)}/{len(specs)} independent SAR scenes", flush=True)
    for split, total in COUNTS.items():
        subset = [row for row in rows if row["split"] == split]
        if Counter(row["condition"] for row in subset) != dict.fromkeys(CONDITIONS, total // len(CONDITIONS)):
            raise ValueError(f"{split}: condition counts differ from the protocol")
        if Counter(row["pulses"] for row in subset) != dict.fromkeys(PULSES, total // len(PULSES)):
            raise ValueError(f"{split}: aperture counts differ from the protocol")
    write_json(dataset / "dataset.json", {"version": 1, "profile": "contrast-study", "seed": SEED, "entries": rows})
    write_csv(dataset / "manifest.csv", rows)
    return read_split(dataset)


def with_contrast_stem(torch, smp, parent):
    model = create_model(torch, smp, True, local_contrast=True).to(memory_format=torch.channels_last)
    weights = parent.state_dict()
    stem_key = "encoder.conv1.weight"
    stem = weights[stem_key]
    added = stem.new_zeros((stem.shape[0], len(CONTRAST_WINDOWS), *stem.shape[2:]))
    weights[stem_key] = torch.cat((stem, added), dim=1)
    model.load_state_dict(weights)
    return model


def score_validation(torch, model, settings, validation):
    return [(predict(torch, model, row["aligned"], settings, "cpu", BATCH_SIZE), row["truth"]) for row in validation]


def write_miss_stages(root, entries, torch, correlate, model, settings):
    support, misses = Counter(), Counter()
    threshold, min_area, closing = (settings[key] for key in OPERATING_FIELDS)
    for sample in read_registered(entries, correlate):
        truth = sample["truth"]
        scores = predict(torch, model, sample["aligned"], settings, "cpu", BATCH_SIZE)
        final = cleanup(scores, threshold, min_area, closing)
        path = root / "predictions" / sample["entry"]["scene_id"] / "local_contrast/predictions.npy"
        if not np.array_equal(final, load_masks(path, truth.shape)):
            raise ValueError(f"{sample['entry']['scene_id']}: diagnostic prediction differs from the locked test")
        masks = (scores >= threshold, cleanup(scores, threshold, 1, closing), final)
        metrics = [score_masks(mask, truth, MATCH)["frames"] for mask in masks]
        areas = truth.sum(axis=(1, 2))
        bands = np.digitize(areas, (12, 24, 48))
        for i, records in enumerate(zip(*metrics)):
            if areas[i]:
                band = SIZE_BANDS[bands[i]]
                support[band] += 1
                stage = diagnostics(*records, MATCH)["miss_stage"]
                if stage is not None:
                    misses[band, stage] += 1
    write_csv(root / "misses_by_stage_and_area.csv", [
        dict(label_area_px=band, visible_frames=support[band],
             **{stage: misses[band, stage] for stage in MISS_STAGES}) for band in SIZE_BANDS])


def paired_and_aperture(root, registry, comparisons=None):
    with (root / "metrics_by_scene.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_method = {}
    aperture_rows = []
    for method in registry:
        subset = [row for row in rows if row["method"] == method]
        by_method[method] = {row["scene_id"]: row for row in subset}
        for pulses in PULSES:
            for group in ("all", "small", "normal", "empty"):
                conditions = [row for row in subset if int(row["pulses"]) == pulses and (group == "all" or row["group"] == group)]
                matched, fp, missed, frames = (sum(int(row[key]) for row in conditions)
                                              for key in ("matched", "false_positive", "missed", "frames"))
                aperture_rows.append(dict(method=method, pulses=pulses, group=group, sequences=len(conditions),
                                         frames=frames, matched=matched, false_positive=fp, missed=missed,
                                         false_positive_per_frame=fp / frames, **target_summary(matched, fp, missed)))
    if comparisons is None:
        comparisons = (("retuned", HISTORICAL), ("continued", "retuned"), ("local_contrast", "continued"),
                       ("local_contrast", HISTORICAL), ("continued", HISTORICAL))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    paired = []
    for method, reference in comparisons:
        for group, metric in (("small", "recall"), ("normal", "recall"), ("empty", "false_positive_per_frame")):
            first = [row for row in by_method[method].values() if row["group"] == group]
            second = [by_method[reference][row["scene_id"]] for row in first]
            numerator = "false_positive" if group == "empty" else "matched"
            difference = np.array([int(a[numerator]) - int(b[numerator]) for a, b in zip(first, second)])
            support = np.array([int(row["frames"]) if group == "empty" else int(row["matched"]) + int(row["missed"])
                                for row in first])
            reference_support = [int(row["frames"]) if group == "empty" else int(row["matched"]) + int(row["missed"])
                                 for row in second]
            if not np.array_equal(support, reference_support) or not (support > 0).all():
                raise ValueError("paired scenes must have the same nonempty evaluation support")
            draws = rng.integers(0, len(first), size=(BOOTSTRAP_DRAWS, len(first)))
            low, high = np.quantile(difference[draws].sum(axis=1) / support[draws].sum(axis=1), [0.025, 0.975])
            paired.append(dict(method=method, reference=reference, group=group, metric=metric, scenes=len(first),
                               support=int(support.sum()), delta=float(difference.sum() / support.sum()),
                               ci95_low=float(low), ci95_high=float(high)))
    write_csv(root / "metrics_by_aperture.csv", aperture_rows)
    write_csv(root / "paired_summary.csv", paired)
    return paired


def run_contrast_study(root):
    torch, smp, correlate = runtime()
    torch.set_num_threads(min(6, torch.get_num_threads()))
    settings, checkpoints, operating, hashes = frozen_settings()
    native = settings["unet"]
    specs = locked_specs()
    write_json(root / "protocol.json", dict(seed=SEED, train_seed=TRAIN_SEED, bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_draws=BOOTSTRAP_DRAWS, scenes=specs, partition_counts=COUNTS, match_iou=MATCH, epochs=EPOCHS,
        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, contrast_windows=CONTRAST_WINDOWS,
        features="mean log1p power in surrounding square ring minus mean in center square",
        thresholds=THRESHOLDS, min_areas=AREAS, methods=[HISTORICAL, "fpn", "difference", "retuned", *dict(TRAINED)],
        operating_points=operating, **hashes, torch_version=torch.__version__, smp_version=smp.__version__,
        source_projects=SOURCES, gate={"precision": 0.8, "recall": 0.8, "small_recall_gain": 0.10,
                                     "normal_recall_max_drop": 0.05}))
    splits = build_contrast_dataset(root, specs)
    train = read_registered(splits["train"], correlate)
    validation = read_registered(splits["validation"], correlate)
    train_pairs = [(row["aligned"], row["truth"]) for row in train]
    validation_pairs = [(row["aligned"], row["truth"]) for row in validation]
    frozen = load_weights(torch, smp, checkpoints["unet"], native, "cpu")
    fpn = load_weights(torch, smp, checkpoints["fpn"], settings["fpn"], "cpu")
    registry = {HISTORICAL: (frozen, native), "fpn": (fpn, settings["fpn"]),
                "difference": (None, operating["difference"])}
    frozen_scores = score_validation(torch, frozen, native, validation)
    normal_reference = recall_for(frozen_scores, validation, operating["unet"], ("normal",))
    point, grid = tune(frozen_scores, validation, normal_reference, THRESHOLDS, AREAS)
    write_csv(root / "validation_retuned.csv", grid)
    registry["retuned"] = (frozen, native | {key: point[key] for key in OPERATING_FIELDS})
    chosen = {"retuned": dict(operating_point=point, weight_sha256=native["weight_sha256"])}
    models = root / "models"
    models.mkdir()
    for name, local in TRAINED:
        print(f"Training {name} on the same scenario split and budget", flush=True)
        parent = load_weights(torch, smp, checkpoints["unet"], native, "cpu")
        initial = with_contrast_stem(torch, smp, parent) if local else parent
        model, trained = fit(torch, smp, train_pairs, validation_pairs, temporal=True, augment_noise=False,
                             epochs=EPOCHS, seed=TRAIN_SEED, device="cpu", batch_size=BATCH_SIZE,
                             initialization=(initial, native), learning_rate=LEARNING_RATE, local_contrast=local)
        scores = score_validation(torch, model, trained, validation)
        point, grid = tune(scores, validation, normal_reference, THRESHOLDS, AREAS)
        write_csv(root / f"validation_{name}.csv", grid)
        path = models / f"{name}.pt"
        torch.save(model.state_dict(), path)
        trained.update(checkpoint=path.name, weight_sha256=file_hash(path), registered=True,
                       parent_weight_sha256=native["weight_sha256"], **{key: point[key] for key in OPERATING_FIELDS})
        write_json(path.with_suffix(".json"), trained)
        chosen[name] = dict(operating_point=point, weight_sha256=trained["weight_sha256"])
        # Reload the saved deployable candidate before touching the test predictions.
        registry[name] = (load_weights(torch, smp, path, trained, "cpu"), trained)
    preferred = max(chosen, key=lambda name: operating_rank(chosen[name]["operating_point"], normal_reference))
    write_json(root / "frozen_validation.json", dict(dataset_sha256=file_hash(root / "dataset/dataset.json"),
        normal_reference=normal_reference, selected_candidate=preferred, candidates=chosen))
    print("Validation choices frozen. Evaluating on 36 independent test scenes.", flush=True)
    groups = evaluate_models(root, splits["test"], registry, torch, correlate)
    write_size_breakdown(root, splits["test"], registry)
    write_miss_stages(root, splits["test"], torch, correlate, *registry["local_contrast"])
    paired = paired_and_aperture(root, registry)
    table = {(row["method"], row["group"]): row for row in groups}
    write_json(root / "results.json", dict(groups=groups, paired=paired, selected_candidate=preferred,
        gates={name: comparison_gate(table, name) for name in chosen}))
