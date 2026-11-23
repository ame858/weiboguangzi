"""Select a small-shadow detector by forward validation before evaluating the fixed SNL holdouts."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from audit_detection import paired_block_f1
from experiment import CONFIDENCES, NMS, filtered, quality, raw_candidates, summarize, with_nms
from snl_center import train_center
from snl_detector import backend
from snl_inputs import DEFAULT_OUTPUT, PROFILES, ROOT, VIDEO, feature_cache, load_gray, load_labels, read_json, sha_file, write_csv, write_json
from snl_protocol import passed
from videosar.snl_causal import CausalFeatures, detection_metrics


FOLDS = ((104, 120, 150), (154, 170, 200), (214, 230, 260))
EPOCHS = (12, 24)
DENSE_LIMITS = (0, 1024)
ARMS = {"raw_giou": ("stacked", None), "raw_nwd": ("stacked", 12.0),
        "quantile_giou": ("stacked-quantile", None), "quantile_nwd": ("stacked-quantile", 12.0)}


def fit(folder, features, truth, arm, settings):
    folder.mkdir(parents=True, exist_ok=True)
    profile, nwd = ARMS[arm]
    recipe = dict(settings, arm=arm, fit_ids=list(truth))
    path = folder / "checkpoints.json"
    if path.is_file():
        saved = read_json(path)
        if saved["recipe"] != recipe:
            raise ValueError(f"Training recipe changed for {folder}")
        if any(sha_file(folder / row["path"]) != row["sha256"] for row in saved["checkpoints"]):
            raise ValueError(f"Checkpoint checksum mismatch in {folder}")
        return saved["checkpoints"]
    checkpoints = list(train_center(folder, features, truth, input_profile=profile, dense_offsets=True,
                                    nwd_scale=nwd, epoch_budget=EPOCHS[-1]))
    write_json(path, dict(recipe=recipe, checkpoints=checkpoints))
    return checkpoints


def choose(out_of_fold, labels):
    trials = []
    for (epoch, limit), raw in out_of_fold.items():
        for nms in NMS:
            boxes = with_nms(raw, nms)
            candidates = [(float(t), summarize(filtered(boxes, float(t), None), labels)[0]) for t in CONFIDENCES]
            threshold, metrics = max(candidates, key=lambda row: quality(row[1]))
            trials.append(dict(epoch=epoch, dense_candidate_limit=limit, nms=nms, threshold=threshold, cv=metrics))
    selected = max(trials, key=lambda row: quality(row["cv"]))
    return selected, trials


def at_point(raw, point):
    return filtered(with_nms(raw, point["nms"]), point["threshold"], None)


def center_metrics(predictions, labels):
    counts = np.zeros(3, dtype=np.int64)
    for index, boxes in labels.items():
        predicted = predictions[index][0]
        center = (predicted[:, :2] + predicted[:, 2:]) / 2
        truth = (boxes[:, :2] + boxes[:, 2:]) / 2
        near = np.linalg.norm(center[:, None] - truth, axis=2) <= 5
        a, b = linear_sum_assignment(near, maximize=True)
        hits = int(near[a, b].sum())
        counts += (hits, len(predicted) - hits, len(boxes) - hits)
    return detection_metrics(counts, len(labels))


def verify_prefix(gray, features, profile, index):
    scanner = CausalFeatures(profile)
    for image in gray[:index - 1]:
        channels = scanner.network_channels(image)
    np.testing.assert_array_equal(np.rint(255 * channels).astype(np.uint8), features[index - 2])


def run(output, ffmpeg):
    train_ids, cal_ids, dev_ids = PROFILES["prospective"]
    if any(not 2 <= end < first <= last < min(cal_ids) for end, first, last in FOLDS) or max(cal_ids) >= min(dev_ids):
        raise ValueError("Forward selection and holdout dates are not correctly isolated")
    study = output / "rolling-nwd"
    study.mkdir(exist_ok=True)
    gray = load_gray(output, ffmpeg)
    training = load_labels(output, train_ids, gray)[0]
    source_paths = (Path(__file__), Path(__file__).with_name("snl_center.py"), ROOT / "videosar/snl_causal.py")
    settings = dict(video_sha256=sha_file(VIDEO), source_hashes={path.name: sha_file(path) for path in source_paths},
                    folds=[list(row) for row in FOLDS], epochs=list(EPOCHS), seed=1729, torch=str(torch.__version__),
                    device=backend().type)
    fold_labels = [{i: training[i] for i in range(first, last + 1)} for _, first, last in FOLDS]
    cv_labels = {i: targets for block in fold_labels for i, targets in block.items()}
    if len(cv_labels) != sum(map(len, fold_labels)):
        raise ValueError("Forward selection reuses a holdout frame")

    profiles = {profile for profile, _ in ARMS.values()}
    through = max(cv_labels)
    input_cache = {profile: feature_cache(study, gray, through, CausalFeatures(profile)) for profile in profiles}
    arms = {}
    cv_rows = []
    for arm, (profile, _) in ARMS.items():
        stack = input_cache[profile]
        predictions = {(epoch, limit): {} for epoch, limit in product(EPOCHS, DENSE_LIMITS)}
        for fold, ((end, _, _), labels) in enumerate(zip(FOLDS, fold_labels, strict=True), start=1):
            folder = study / arm / f"fold-{fold}"
            folder.parent.mkdir(exist_ok=True)
            checkpoints = fit(folder, stack, {i: training[i] for i in train_ids if i <= end}, arm, settings)
            for checkpoint, limit in product(checkpoints, DENSE_LIMITS):
                config = checkpoint | {"dense_candidate_limit": limit}
                predictions[(config["epoch"], limit)].update(raw_candidates(folder, config, stack, list(labels), "forward"))
        point, trials = choose(predictions, cv_labels)
        raw = predictions[(point["epoch"], point["dense_candidate_limit"])]
        selected = at_point(raw, point)
        per_fold = []
        for index, labels in enumerate(fold_labels, start=1):
            metrics, rows = summarize(selected, labels)
            per_fold.append(metrics)
            cv_rows.extend(dict(arm=arm, fold=index, **row) for row in rows)
        arms[arm] = dict(point=point, trials=trials, forward_blocks=per_fold)
        print(arm, "rolling selection", point, flush=True)
    chosen = max(arms, key=lambda arm: quality(arms[arm]["point"]["cv"]))
    freeze = dict(settings=settings, selected_by_rolling_cv=chosen, cv_frames=len(cv_labels), arms=arms,
                  matching_iou=0.50, center_distance_px=5.0)
    write_json(study / "rolling-selection.json", freeze)
    write_csv(study / "rolling-development-rows.csv", cv_rows)

    input_cache = {profile: feature_cache(study, gray, max(dev_ids), CausalFeatures(profile)) for profile in profiles}
    for profile, stack in input_cache.items():
        verify_prefix(gray, stack, profile, min(cal_ids))

    configs = {}
    for arm, (profile, _) in ARMS.items():
        folder = study / arm / "refit"
        checkpoints = fit(folder, input_cache[profile], training, arm, settings)
        point = arms[arm]["point"]
        configs[arm] = next(row for row in checkpoints if row["epoch"] == point["epoch"]) | {
            "dense_candidate_limit": point["dense_candidate_limit"], "nms": point["nms"],
            "threshold": point["threshold"], "low": None}
    write_json(study / "frozen-refit.json", dict(selected=chosen, configs=configs, validated_prefix=min(cal_ids)))

    rows, development_rows = [], {}
    for scope, ids in (("calibration_audit", cal_ids), ("development", dev_ids)):
        labels = load_labels(output, ids, gray)[0]
        for arm, config in configs.items():
            stack = input_cache[ARMS[arm][0]]
            raw = raw_candidates(study / arm / "refit", config, stack, list(ids), scope)
            predictions = at_point(raw, config)
            metrics, counts = summarize(predictions, labels)
            arms[arm][scope] = dict(iou50=metrics, center5=center_metrics(predictions, labels))
            rows.extend(dict(arm=arm, split=scope, **row) for row in counts)
            if scope == "development":
                development_rows[arm] = counts
            print(scope, arm, arms[arm][scope], flush=True)
    for arm, result in arms.items():
        result["development_gate_passed"] = all(passed(result[scope]["iou50"]) for scope in ("calibration_audit", "development"))
        result["paired_f1_vs_raw_giou"] = paired_block_f1(development_rows[arm], development_rows["raw_giou"])
    write_json(study / "comparison.json", freeze | {"refit_checkpoints": configs})
    write_csv(study / "holdout-rows.csv", rows)
    print("Rolling selection:", chosen, "development gate:", arms[chosen]["development_gate_passed"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.output.resolve(), args.ffmpeg)
