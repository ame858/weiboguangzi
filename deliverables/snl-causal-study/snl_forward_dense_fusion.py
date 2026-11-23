"""Fit dense two-source detection scores on a chronological earlier-model holdout."""
import argparse
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
from scipy.special import expit

from audit_detection import paired_block_f1
from experiment import CONFIDENCES, filtered, load_candidates, quality, raw_candidates, select_operation, summarize
from experiment import warm_ids, with_nms
from snl_bytetrack import REFERENCE
from snl_fusion import measurement, score
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, WIDTH, feature_cache, load_gray, load_labels, read_json, sha_file
from snl_inputs import write_csv, write_json
from snl_local_shadow_proposals import coverage, evaluated, fit_anchors, proposals, supervision
from videosar.snl_causal import CausalFeatures, box_iou


DENSE_LIMIT = 4096
PAST_LAGS = (1, 3, 6)


def pixels(boxes):
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    x, y = np.floor(centers).astype(np.int32).clip(0, (WIDTH - 1, HEIGHT - 1)).T
    return y, x


def score_map(boxes, values):
    density = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    np.maximum.at(density, pixels(boxes), values)
    return density


def expanded(density, radius_x, radius_y):
    window = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_x + 1, 2 * radius_y + 1))
    return cv2.dilate(density, window)


def records(neural, features, transforms):
    result = {}
    density = {i: score_map(boxes, confidence) for i, (boxes, confidence) in neural.items()}
    for index, (network_boxes, confidence) in neural.items():
        frame = features[index - 2]
        local_boxes, cues = proposals(frame)
        boxes = np.concatenate((network_boxes, local_boxes))
        positions = pixels(boxes)
        local_score = (cues[:, 0] - 0.5 * cues[:, 1] + 0.4 * cues[:, 2]).clip(min=0)
        local_map = score_map(local_boxes, local_score)
        source = np.arange(len(boxes)) < len(network_boxes)
        direct = np.concatenate((confidence, np.zeros(len(local_boxes), dtype=np.float32)))
        context = [source, direct, expanded(density[index], 2, 4)[positions],
                   expanded(density[index], 5, 8)[positions], expanded(local_map, 3, 5)[positions]]
        current_inverse = cv2.invertAffineTransform(transforms[index])
        for lag in PAST_LAGS:
            previous = index - lag
            if previous not in density:
                context.append(np.zeros(len(boxes), dtype=np.float32))
                continue
            transform = current_inverse @ np.vstack((transforms[previous], (0, 0, 1)))
            aligned = cv2.warpAffine(density[previous], transform, (WIDTH, HEIGHT), flags=cv2.INTER_NEAREST)
            context.append(expanded(aligned, 3 + 2 * lag, 4 + 2 * lag)[positions])
        measurements = np.column_stack((measurement(frame, boxes), *context)).astype(np.float32)
        result[index] = boxes, measurements
    return result


def train_frame_ranker(labels):
    data = np.concatenate([row[0] for row in labels.values()])
    overlaps = np.concatenate([row[2] for row in labels.values()])
    grades = np.searchsorted((0.20, 0.35, 0.50, 0.65, 0.80), overlaps, side="right")
    sizes = [len(row[2]) for row in labels.values()]
    options = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[10, 30], lambdarank_truncation_level=45,
                   label_gain=[0, 0, 1, 4, 8, 12], num_leaves=23, max_depth=5, learning_rate=0.05,
                   min_data_in_leaf=22, seed=331, num_threads=5, verbosity=-1, force_col_wise=True)
    return lgb.train(options, lgb.Dataset(data, label=grades, group=sizes), num_boost_round=120)


def fit_frame_ranker(labels, regression_dir):
    classifier = train_frame_ranker(labels)
    models = read_json(regression_dir / "comparison.json")["weights"]
    paths = [regression_dir / f"{key}.txt" for key in ("x0", "y0", "x1", "y1")]
    if any(sha_file(path) != models[path.name] for path in paths):
        raise ValueError("The archived geometry models have changed")
    return classifier, [lgb.Booster(model_file=str(path)) for path in paths]


def candidate_scores(ranker, row, refine, *, frame_ranked=False):
    boxes, confidence = score(ranker, row, refine)
    if frame_ranked:
        confidence = expit(confidence)
    return boxes, confidence


def soft_nms(boxes, scores, sigma, floor):
    active = np.flatnonzero(scores >= floor)
    remaining = scores.copy()
    chosen, confidence = [], []
    while len(active):
        best = int(remaining[active].argmax())
        index = int(active[best])
        chosen.append(index)
        confidence.append(remaining[index])
        active = np.delete(active, best)
        overlap = box_iou(boxes[index:index + 1], boxes[active])[0]
        remaining[active] *= np.exp(-overlap**2 / sigma)
        active = active[remaining[active] >= floor]
    return boxes[chosen], np.asarray(confidence, dtype=np.float32)


def suppress(raw, method, strength):
    if method == "hard":
        return with_nms(raw, strength)
    floor = float(CONFIDENCES[0]) / 2
    if method == "soft":
        return {i: soft_nms(boxes, scores, strength, floor) for i, (boxes, scores) in raw.items()}
    pruned = {i: (boxes[scores >= floor], scores[scores >= floor]) for i, (boxes, scores) in raw.items()}
    pooled = with_nms(pruned, strength)
    result = {}
    for index, (boxes, scores) in pooled.items():
        source, confidences = pruned[index]
        overlaps = box_iou(boxes, source)
        weights = (overlaps >= 0.50) * confidences[None, :]**2
        voted = (weights @ source) / np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
        result[index] = voted.astype(np.float32), scores
    return result


def compare_suppression(study, cal, dev, calibration, development, confirmations, baseline_rows):
    trials = []
    choices = {"hard": (0.35, 0.50, 0.65, 0.80), "soft": (0.30, 0.50, 0.80), "vote": (0.35, 0.50)}
    for method, values in choices.items():
        for strength in values:
            candidates = suppress(cal, method, strength)
            for threshold in CONFIDENCES:
                for low in (None, float(threshold / 2)):
                    predicted = filtered(candidates, float(threshold), low, confirmations=confirmations)
                    metrics = summarize(predicted, calibration)[0]
                    trials.append(dict(method=method, strength=strength, threshold=float(threshold), low=low, metrics=metrics))
    selected = max(trials, key=lambda row: quality(row["metrics"]))
    chosen = suppress(dev, selected["method"], selected["strength"])
    predictions = filtered(chosen, selected["threshold"], selected["low"], confirmations=confirmations)
    metrics, rows = summarize(predictions, development)
    report = {"selected": selected, "development": metrics, "candidate_matches": coverage(chosen, development),
              "paired_f1_vs_forward_fusion": paired_block_f1(rows, baseline_rows), "calibration_trials": trials}
    write_json(study / "suppression-comparison.json", report)
    write_csv(study / "suppression-development.csv", rows)
    print("Candidate suppression", selected, metrics, flush=True)


def audit_ranking(study, output, features, transforms, calibration, development):
    report = read_json(study / "comparison.json")
    weights = report["weights"]
    paths = [study / f"{key}.txt" for key in ("ranker", "x0", "y0", "x1", "y1")]
    if any(sha_file(path) != weights[path.name] for path in paths):
        raise ValueError("The forward scorer weights differ from the archived evaluation")
    models = [lgb.Booster(model_file=str(path)) for path in paths]
    ranker = models[0], models[1:]
    ranked_scopes = []
    for scope in ("validation", "development"):
        dense = load_candidates(output, report["evaluation_detector"], scope)
        data = records(dense, features, transforms)
        ranked_scopes.append({i: candidate_scores(ranker, row, report["selected"]["refine"],
                                                  frame_ranked=report.get("frame_ranking", False))
                              for i, row in data.items()})
    cal, replay = ranked_scopes
    op = report["selected"]["operation"] | {"track_support": report["evaluation_detector"].get("track_support", 1)}
    metrics, baseline_rows = evaluated(replay, op, development)
    if metrics != report["development"]:
        raise ValueError("Archived settings failed to reproduce forward fusion")
    ranked = {}
    for index, (boxes, confidence) in replay.items():
        order = np.argsort(-confidence, kind="stable")
        ranked[index] = boxes[order], confidence[order]
    budgets = (10, 20, 40, 80, 160, 320, 640, 1280)
    rows = {}
    for name, predictions in (("before_nms", ranked), ("after_selected_nms", with_nms(replay, op["nms"]))):
        rows[name] = [{"limit_per_frame": limit, "matches": coverage({i: (p[0][:limit], p[1][:limit])
                                                                       for i, p in predictions.items()}, development)}
                      for limit in budgets]
        rows[name].append({"limit_per_frame": None, "matches": coverage(predictions, development)})
    write_json(study / "ranking-audit.json", {"targets": sum(map(len, development.values())), "match_iou": 0.50,
               "reproduced_development": report["development"], "selected_nms": op["nms"], "coverage": rows})
    print("Forward fusion candidate coverage", rows, flush=True)
    compare_suppression(study, cal, replay, calibration, development, op["track_support"], baseline_rows)


def evaluate(output, ffmpeg, *, audit_only=False, frame_ranked=False):
    original = output / "forward-dense-fusion"
    study = output / "forward-dense-frame-ranking" if frame_ranked else original
    study.mkdir(exist_ok=True)
    earlier = DEFAULT_OUTPUT / "early-stacked"
    train_ids, cal_ids, dev_ids = PROFILES["prospective"]
    train_earlier = read_json(earlier / "split-protocol.json")["train"]
    fit_ids = tuple(index for index in train_ids if index >= 242)
    if max(train_earlier) >= min(fit_ids) or max(fit_ids) >= min(cal_ids) or max(cal_ids) >= min(dev_ids):
        raise ValueError("Earlier detector, scorer, calibration, and development must follow time order")
    gray = load_gray(output, ffmpeg)
    fitting, calibration, development = (load_labels(output, ids, gray)[0] for ids in (fit_ids, cal_ids, dev_ids))
    release = read_json(output / REFERENCE)["config"]
    first = next(item for item in read_json(earlier / "checkpoints-center-stacked.json") if item["epoch"] == 36)
    first = first | {"contrast_gain": release["contrast_gain"], "dense_candidate_limit": DENSE_LIMIT}
    latest = release | {"dense_candidate_limit": DENSE_LIMIT}
    stacked = feature_cache(output, gray, max(dev_ids), CausalFeatures("stacked"))
    features = np.load(output / "motion-integration/motion-integration.npy", mmap_mode="r", allow_pickle=False)
    registration = read_json(output / "canonical-center-chained/canonical-registration.json")
    transforms = {int(index): np.asarray(matrix) for index, matrix in registration["transforms"].items()}
    if audit_only:
        audit_ranking(study, output, features, transforms, calibration, development)
        return
    first_boxes = raw_candidates(earlier, first, stacked, warm_ids(fitting), "forward-dense-fit")
    training_rows = records(first_boxes, features, transforms)
    labels = {i: supervision(training_rows[i], truth) for i, truth in fitting.items()}
    examples = sum(len(row[2]) for row in labels.values())
    positives = sum(int((row[2] >= 0.50).sum()) for row in labels.values())
    print(f"Fitting the forward scorer on {examples} candidates, {positives} positives", flush=True)
    ranker = fit_frame_ranker(labels, original) if frame_ranked else fit_anchors(labels)
    del labels, training_rows
    weights = {}
    for name, estimator in zip(("ranker", "x0", "y0", "x1", "y1"), (ranker[0], *ranker[1]), strict=True):
        model_path = study / f"{name}.txt"
        estimator.save_model(str(model_path))
        weights[model_path.name] = sha_file(model_path)

    val_neural = raw_candidates(output, latest, stacked, warm_ids(calibration), "validation")
    dev_neural = raw_candidates(output, latest, stacked, warm_ids(development), "development")
    support = release.get("track_support", 1)
    baseline_config, baseline_cal, _ = select_operation(val_neural, calibration, confirmations=support)
    baseline, baseline_rows = evaluated(dev_neural, baseline_config | {"track_support": support}, development)
    val = records(val_neural, features, transforms)
    stop = next(iter(calibration)) + 2
    prefix = records({i: value for i, value in val_neural.items() if i <= stop}, features, transforms)
    for index, (boxes, data) in prefix.items():
        np.testing.assert_array_equal(boxes, val[index][0])
        np.testing.assert_array_equal(data, val[index][1])
    dev = records(dev_neural, features, transforms)
    trials = []
    for refine in (False, True):
        data = {i: candidate_scores(ranker, row, refine, frame_ranked=frame_ranked) for i, row in val.items()}
        operation, metrics, _ = select_operation(data, calibration, confirmations=support)
        trials.append({"refine": refine, "operation": operation, "calibration": metrics})
    best = max(trials, key=lambda row: quality(row["calibration"]))
    predictions = {i: candidate_scores(ranker, row, best["refine"], frame_ranked=frame_ranked) for i, row in dev.items()}
    scorecard, rows = evaluated(predictions, best["operation"] | {"track_support": support}, development)
    coverage_summary = {"neural_only": coverage(dev_neural, development),
                        "union_before_scoring": coverage(dev, development),
                        "selected_geometry": coverage(predictions, development)}
    report = {"fit_detector": first, "evaluation_detector": latest, "fit_ids": fit_ids, "fit_examples": examples,
              "fit_positive": positives, "past_lags": PAST_LAGS, "weights": weights, "prefix_verified_through": stop,
              "frame_ranking": frame_ranked,
              "calibration_trials": trials, "selected": best, "development": scorecard, "coverage": coverage_summary,
              "baseline": {"operation": baseline_config, "calibration": baseline_cal, "development": baseline},
              "paired_f1_vs_dense_baseline": paired_block_f1(rows, baseline_rows)}
    write_json(study / "comparison.json", report)
    write_csv(study / "development.csv", rows)
    print("Forward dense fusion", scorecard, coverage_summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--audit-ranking", action="store_true", help="Check scored candidates from a completed run")
    parser.add_argument("--rank-by-frame", action="store_true", help="Fit a LambdaRank scorer with frames as groups")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    evaluate(args.output.resolve(), args.ffmpeg, audit_only=args.audit_ranking, frame_ranked=args.rank_by_frame)
