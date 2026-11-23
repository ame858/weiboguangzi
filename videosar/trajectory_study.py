from collections import Counter
from itertools import product
from shutil import copyfile

import numpy as np

from .analysis import score_masks, target_summary
from .contrast_study import locked_specs, paired_and_aperture, save_contrast_scene
from .data import load_observations, read_json, save_receipt, write_json
from .echo_attribution import file_hash, frozen_settings
from .experiment import read_split
from .open_compare import HISTORICAL, comparison_gate, evaluate_models
from .small_shadow import CONDITIONS
from .stress import MATCH
from .study import calibrate, cleanup, read_registered
from .temporal_study import write_size_breakdown
from .trajectory import accumulate_centers, candidate_observations, choose_boxes, low_rank_scores, recover_masks
from .unet import align, load_weights, runtime
from .visuals import write_csv


SEED = 101661
WEAK_THRESHOLDS = (0.10, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.45, 0.50, 0.55, 1.0)
RANK_THRESHOLDS = (0.06, 0.09, 0.12, 0.15, 0.18, 0.23, 0.30, 0.40)
RANK_AREAS = (4, 8, 12, 16)
RANK_WINDOWS = (5, 9)
GATES = (2, 3, 4)
METHODS = ("low_score", "association", "accumulation", "low_rank")
PAPERS = {
    "association": ["https://arxiv.org/abs/2110.06864", "https://github.com/ifzhang/ByteTrack",
                    "https://www.sciencedirect.com/science/article/pii/S0957417425015039"],
    "accumulation": ["https://doi.org/10.1109/JSTARS.2021.3104603"],
    "low_rank": ["https://doi.org/10.3390/rs15061542"],
}


def frame_counts(masks, truth):
    rows = score_masks(masks, truth, MATCH)["frames"]
    matched = np.array([r["best_component_iou"] >= MATCH for r in rows], dtype=int)
    fp = np.array([r["predicted_components"] for r in rows]) - matched
    return np.stack((matched, fp), axis=1)


def validation_rows(samples, observed):
    base = np.stack([frame_counts(item["masks"], s["truth"]) for s, item in zip(samples, observed)])
    boxes = np.stack([frame_counts(item["boxes"], s["truth"]) for s, item in zip(samples, observed)])
    group = np.array([s["entry"]["group"] for s in samples])
    visible = np.stack([s["truth"].any(axis=(1, 2)) for s in samples])
    support = {name: int(visible[group == name].sum()) for name in ("small", "normal")}

    def summarize(config, values):
        tp, fp = (int(n) for n in values.sum(axis=(0, 1)))
        return dict(config, normal_recall=float(values[group == "normal", :, 0].sum() / support["normal"]),
                    small_recall=float(values[group == "small", :, 0].sum() / support["small"]),
                    empty_fp=int(values[group == "empty", :, 1].sum()), false_positive=fp,
                    **target_summary(tp, fp, int(visible.sum()) - tp))

    baseline = summarize({"method": "joint_mask"}, base)
    rows = [baseline]
    for method in METHODS[:3]:
        motions = (0.0, 0.5) if method == "accumulation" else (0.0,)
        for gate, displacement in product(GATES if method != "low_score" else (3,), motions):
            accumulated = [accumulate_centers(o["heatmap"], gate, displacement)
                           for o in observed] if method == "accumulation" else [None] * len(observed)
            for threshold, replace in product(WEAK_THRESHOLDS, (False, True)):
                config = dict(method=method, gate=gate, minimum_displacement=displacement,
                              threshold=threshold, replace=replace)
                chosen = np.stack([choose_boxes(o, config, a) for o, a in zip(observed, accumulated)])
                rows.append(summarize(config, np.where(chosen[:, :, None], boxes, base)))
    return baseline, rows


def rank_validation(samples):
    rows = []
    for window, rank in product(RANK_WINDOWS, (1, 2)):
        scored = [(low_rank_scores(s["aligned"], window, rank), s["truth"]) for s in samples]
        subsets = {"all": scored}
        for name in ("small", "normal", "empty"):
            subsets[name] = [pair for sample, pair in zip(samples, scored) if sample["entry"]["group"] == name]
        grids = {name: calibrate(pairs, RANK_THRESHOLDS, RANK_AREAS, True, MATCH)[1]
                 for name, pairs in subsets.items()}
        for total, small, normal, empty in zip(*(grids[name] for name in ("all", "small", "normal", "empty"))):
            rows.append(dict(total, method="low_rank", window=window, rank=rank, empty_fp=empty["false_positive"],
                             small_recall=small["recall"], normal_recall=normal["recall"]))
    return rows


CONFIG_FIELDS = ("method", "threshold", "gate", "replace", "minimum_displacement", "window", "rank", "min_area", "closing")


def choose_validation(rows, baseline):
    def eligible(row):
        return row["empty_fp"] <= baseline["empty_fp"] and row["normal_recall"] >= baseline["normal_recall"] - 0.05

    def rank(row):
        precision, recall = row["precision"] or 0, row["recall"] or 0
        return (eligible(row), precision >= 0.8 and recall >= 0.8, row["small_recall"],
                row["target_f1"] or 0, -row["false_positive"], row["method"] == "joint_mask")

    for row in rows:
        row["within_validation_constraints"] = eligible(row)
    winners = {method: max((r for r in rows if r["method"] == method), key=rank) for method in METHODS}
    return winners, max([baseline, *winners.values()], key=rank)["method"]


def apply_profile(config, sample, torch, model, settings, device="cpu"):
    method = config["method"]
    if method == "low_rank":
        scores = low_rank_scores(sample["aligned"], config["window"], config["rank"])
        return cleanup(scores, config["threshold"], config["min_area"], config["closing"])
    observed = candidate_observations(torch, model, sample["aligned"], sample["times"], settings, device)
    return observed["masks"] if method == "joint_mask" else recover_masks(observed, config)[0]


def trajectory_detect(source, output, profile):
    config = read_json(profile)
    settings_path = profile.parent / config["model_settings"]
    settings = read_json(settings_path)
    torch, smp, correlate = runtime()
    torch.set_num_threads(min(6, torch.get_num_threads()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_weights(torch, smp, settings_path.with_name(settings["checkpoint"]), settings, device)
    frames, times, _ = load_observations(source)
    aligned, offsets = align(frames, correlate)
    masks = apply_profile(config, {"aligned": aligned, "times": times}, torch, model, settings, device)
    np.save(output / "predictions.npy", masks.astype(np.uint8))
    np.save(output / "aligned_frames.npy", aligned)
    np.save(output / "registration_offsets_xy_px.npy", offsets)
    save_receipt(output, source, detector=config["method"], profile=str(profile.resolve()), registered=True)


def run_trajectory_study(root, training_run, center_run):
    torch, smp, correlate = runtime()
    torch.set_num_threads(min(6, torch.get_num_threads()))
    source = training_run / "dataset"
    center_protocol = read_json(center_run / "protocol.json")
    center_freeze = read_json(center_run / "frozen_validation.json")
    if file_hash(source / "dataset.json") != center_protocol["source_dataset_sha256"]:
        raise ValueError("trajectory study requires the validation split used in the center experiment")
    joint_path = center_run / "models/joint_mask.json"
    joint = read_json(joint_path)
    if (joint["weight_sha256"] != center_freeze["weight_sha256"]
            or any(joint[key] != center_freeze["candidates"]["joint_mask"][key]
                   for key in ("threshold", "min_area", "closing"))):
        raise ValueError("joint segmentation model differs from the center-study freeze")
    checkpoints_root = root / "models"
    checkpoints_root.mkdir()
    weights = joint_path.with_name(joint["checkpoint"])
    model = load_weights(torch, smp, weights, joint, "cpu")
    copyfile(weights, checkpoints_root / weights.name)
    copyfile(joint_path, checkpoints_root / joint_path.name)
    settings, checkpoints, operating, hashes = frozen_settings()
    prior_seeds = {s["config"]["seed"] for s in center_protocol["test_scenes"]}
    specs = []
    for spec in locked_specs(SEED):
        if spec["split"] == "test":
            name = spec["scene_id"].replace("contrast-", "trajectory-", 1)
            specs.append(spec | {"scene_id": name, "scene_family": name})
    if prior_seeds & {row["config"]["seed"] for row in specs}:
        raise ValueError("test scene seed overlaps the earlier center-study test")
    write_json(root / "protocol.json", dict(seed=SEED, scenes=specs, match_iou=MATCH, **hashes,
        joint_weight_sha256=joint["weight_sha256"], joint_settings_sha256=file_hash(joint_path),
        validation_dataset_sha256=file_hash(source / "dataset.json"), weak_thresholds=WEAK_THRESHOLDS,
        gates=GATES, rank_thresholds=RANK_THRESHOLDS, rank_areas=RANK_AREAS, rank_windows=RANK_WINDOWS,
        association_history_frames=3, accumulation_frames=3, paper_sources=PAPERS,
        selection="validation no-target FP and normal recall constraints, then dual-80%, small recall and F1",
        test_policy="replace only if original adoption gate passes and no-target FP does not rise; "
                    "keep an improving constrained validation winner as experimental otherwise"))
    validation = read_registered(read_split(source)["validation"], correlate)
    observed = [candidate_observations(torch, model, s["aligned"], s["times"], joint) for s in validation]
    baseline, grid = validation_rows(validation, observed)
    grid.extend(rank_validation(validation))
    winners, preferred = choose_validation(grid, baseline)
    write_csv(root / "validation_grid.csv", grid)
    write_json(root / "frozen_validation.json", dict(baseline=baseline, candidates=winners, selected=preferred))
    print(f"Validation frozen. Selected: {preferred}. Generating independent test scenes.", flush=True)
    test = [save_contrast_scene(root / "dataset", spec) for spec in specs]
    if (Counter(r["condition"] for _, r in test) != dict.fromkeys(CONDITIONS, 9)
            or Counter(r["pulses"] for _, r in test) != {61: 18, 91: 18}):
        raise ValueError("test condition counts differ from the protocol")
    write_json(root / "dataset/dataset.json", {"profile": "trajectory-study", "seed": SEED, "entries": [r for _, r in test]})
    write_csv(root / "dataset/manifest.csv", [r for _, r in test])
    native = load_weights(torch, smp, checkpoints["unet"], settings["unet"], "cpu")
    fpn = load_weights(torch, smp, checkpoints["fpn"], settings["fpn"], "cpu")
    registry = {HISTORICAL: (native, settings["unet"]), "fpn": (fpn, settings["fpn"]),
                "difference": (None, operating["difference"]), "joint_mask": (model, joint)}
    registry.update({name: (None, {} if name == "low_rank" else joint) for name in winners})
    overrides = {name: (lambda sample, config=config: apply_profile(config, sample, torch, model, joint))
                 for name, config in winners.items()}
    groups = evaluate_models(root, test, registry, torch, correlate, prediction_overrides=overrides)
    write_size_breakdown(root, test, registry)
    paired = paired_and_aperture(root, registry, (
        *((name, "joint_mask") for name in winners), ("association", "low_score"), ("accumulation", "low_score")))
    table = {(r["method"], r["group"]): r for r in groups}
    gates = {name: comparison_gate(table, name) for name in ("joint_mask", *METHODS)}
    reference = table["joint_mask", "empty"]["false_positive"]
    winner_valid = (table[preferred, "empty"]["false_positive"] <= reference
                    and table[preferred, "normal"]["recall"] >= table["joint_mask", "normal"]["recall"] - 0.05)
    improves = table[preferred, "all_small"]["recall"] > table["joint_mask", "all_small"]["recall"]
    adopted = preferred != "joint_mask" and winner_valid and gates[preferred]["passed"]
    experimental = preferred != "joint_mask" and winner_valid and improves
    retained = preferred if adopted or experimental else "joint_mask"
    config = winners[retained] if retained in winners else baseline
    write_json(root / "retained.json", {key: config[key] for key in CONFIG_FIELDS if key in config}
               | {"model_settings": "models/joint_mask.json", "adopted": adopted})
    write_json(root / "results.json", dict(groups=groups, paired=paired, validation_winner=preferred,
               retained=retained, adopted=adopted, experimental=experimental, gates=gates))
