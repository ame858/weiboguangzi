"""Train a size-independent center scorer on the fixed forward SNL candidate pool."""
import argparse
from itertools import product
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from audit_detection import count_pairs, paired_block_f1
from experiment import CONFIDENCES, filtered, load_candidates, quality, summarize
from snl_box_quality import GEOMETRY_KEYS, TREE, fit_geometry
from snl_forward_dense_fusion import candidate_scores, records, suppress
from snl_inputs import DEFAULT_OUTPUT, PROFILES, load_gray, load_labels, read_json, sha_file, write_csv, write_json
from snl_local_shadow_proposals import coverage, supervision


RADIUS = 5.0
GRID_X, GRID_Y = np.meshgrid(np.arange(-4, 5, 2, dtype=np.float32), np.arange(-8, 9, 4, dtype=np.float32))


def centers(boxes):
    return (boxes[:, :2] + boxes[:, 2:]) * 0.5


def descriptors(frame, boxes, original_data):
    xy = centers(boxes)
    xs = xy[:, :1] + GRID_X.ravel()
    ys = xy[:, 1:] + GRID_Y.ravel()
    patterns = [cv2.remap(frame[ch], xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) / np.float32(255)
                for ch in (0, 1, 4, 5)]
    return np.column_stack((*patterns, original_data[:, -6:])).astype(np.float32, copy=False)


def refined(parent, geometry, raw, features):
    output = {}
    for i, (original, data) in raw.items():
        boxes, scores = candidate_scores((parent, geometry), (original, data), True, frame_ranked=True)
        if len(boxes) != len(original):
            raise ValueError("The archived offset model removed proposals")
        output[i] = boxes, scores, descriptors(features[i - 2], boxes, data)
    return output


def center_targets(boxes, truth):
    distance = ((centers(boxes)[:, None] - centers(truth)[None, :]) ** 2).sum(axis=2)
    target = distance.min(axis=1, initial=np.inf) <= RADIUS**2
    weights = np.ones(len(boxes), dtype=np.float32)
    if target.any():
        groups = distance[target].argmin(axis=1)
        count = np.bincount(groups)[groups]
        weights[target] = np.clip(6 * np.median(count) / count, 2, 18)
    return target, weights


def fit(parent, raw, features, truth):
    labels = {i: supervision(raw[i], boxes) for i, boxes in truth.items()}
    training = []
    for holdout in np.array_split(list(labels), 3):
        geometry = fit_geometry([row for i, row in labels.items() if i not in holdout])
        for i, (boxes, _, data) in refined(parent, geometry, {i: raw[i] for i in holdout}, features).items():
            target, weights = center_targets(boxes, truth[i])
            training.append((data, target, weights))
    data, target, weights = (np.concatenate(part) for part in zip(*training, strict=True))
    model = lgb.train(dict(TREE, objective="binary", num_leaves=23),
                      lgb.Dataset(data, label=target, weight=weights), num_boost_round=120)
    return model, len(target), int(target.sum())


def center_suppress(raw, radius):
    output = {}
    for i, (boxes, scores) in raw.items():
        visible = scores >= float(CONFIDENCES[0]) / 2
        order = np.flatnonzero(visible)
        order = order[np.argsort(-scores[order], kind="stable")]
        tree = cKDTree(centers(boxes[order]))
        blocked = np.zeros(len(order), dtype=bool)
        kept = []
        for j in range(len(order)):
            if not blocked[j]:
                kept.append(order[j])
                blocked[tree.query_ball_point(tree.data[j], radius)] = True
        output[i] = boxes[kept], scores[kept]
    return output


def pool(raw, weight, method, strength):
    mixed = {i: (boxes, (1 - weight) * base + weight * center) for i, (boxes, base, center) in raw.items()}
    return center_suppress(mixed, strength) if method == "center" else suppress(mixed, method, strength)


def select(raw, truth, support, methods, *, weights=(0.25, 0.50, 0.75, 1.0)):
    trials = []
    for weight, (method, strength) in product(weights, methods):
        candidates = pool(raw, weight, method, strength)
        best = None
        for threshold, track in product(CONFIDENCES, (False, True)):
            low = float(threshold) / 2 if track else None
            predictions = filtered(candidates, float(threshold), low, confirmations=support)
            metrics = summarize(predictions, truth)[0]
            trial = dict(weight=weight, method=method, strength=strength, threshold=float(threshold), low=low, calibration=metrics)
            if best is None or quality(metrics) > quality(best["calibration"]):
                best = trial
        trials.append(best)
    return max(trials, key=lambda row: quality(row["calibration"])), trials


def matched_centers(candidates, truth):
    return sum(count_pairs(((centers(candidates[i][0])[:, None] - centers(boxes)[None, :])**2).sum(axis=2) <= RADIUS**2)
               for i, boxes in truth.items())


def compare_scores(raw, truth):
    scores, labels = [], []
    for i, boxes in truth.items():
        proposals, base, center_scores = raw[i]
        target, _ = center_targets(proposals, boxes)
        scores.append(np.column_stack((base, center_scores)))
        labels.append(target)
    target = np.concatenate(labels)
    positive = int(target.sum())
    ranks = rankdata(np.concatenate(scores), axis=0)
    auc = (ranks[target].sum(axis=0) - positive * (positive + 1) / 2) / (positive * (len(target) - positive))
    return {"pool_size": len(target), "positive_centers": positive, "original_auc": float(auc[0]), "center_auc": float(auc[1])}


def ranking_budget(study, branch, cal_raw, dev_raw, cal, dev, base_setup, support):
    results, rows = {}, []
    for split, raw, labels in (("calibration", cal_raw, cal), ("development", dev_raw, dev)):
        baseline_pool = pool(raw, 0, base_setup["method"], base_setup["strength"])
        baseline = filtered(baseline_pool, base_setup["threshold"], base_setup["low"], confirmations=support)
        control, control_rows = summarize(baseline, labels)
        predictions = pool(raw, branch["weight"], base_setup["method"], base_setup["strength"])
        ranked = {}
        for i in labels:
            boxes, scores = predictions[i]
            cap = len(baseline[i][0])
            if len(boxes) < cap:
                raise ValueError("Same-count comparison exhausted the rescored candidate pool")
            keep = np.argsort(-scores, kind="stable")[:cap]
            ranked[i] = boxes[keep], scores[keep], None
        metrics, points = summarize(ranked, labels)
        results[split] = {"reference": control, "center_rescore": metrics,
                          "reference_centers": matched_centers(baseline, labels), "rescored_centers": matched_centers(ranked, labels),
                          "paired_vs_reference": paired_block_f1(points, control_rows)}
        rows.extend(dict(split=split, **row) for row in points)

    decision, _ = select(cal_raw, cal, support, (("center", 3.0), ("center", 5.0)), weights=(0,))
    tied = pool(dev_raw, 0, decision["method"], decision["strength"])
    output = filtered(tied, decision["threshold"], decision["low"], confirmations=support)
    control, _ = summarize(output, dev)
    results["center_suppression_control"] = {"selected": decision, "development": control}
    write_json(study / "ranking-budget.json", results)
    write_csv(study / "ranking-budget-per-frame.csv", rows)
    print("Fixed output count", results["development"], flush=True)


def evaluate(output, ffmpeg, *, audit_only=False):
    archive = output / "forward-dense-frame-ranking"
    setup = read_json(archive / "comparison.json")
    archived = read_json(archive / "suppression-comparison.json")
    paths = [archive / f"{key}.txt" for key in ("ranker", *GEOMETRY_KEYS)]
    if any(sha_file(path) != setup["weights"][path.name] for path in paths):
        raise ValueError("The reference candidate scorer changed")
    parent, *geometry = (lgb.Booster(model_file=str(path)) for path in paths)
    detector = setup["evaluation_detector"]
    support = detector.get("track_support", 1)
    _, cal_ids, dev_ids = PROFILES["prospective"]
    fit_ids = setup["fit_ids"]
    if max(fit_ids) >= min(cal_ids) or max(cal_ids) >= min(dev_ids):
        raise ValueError("The training, calibration, and development splits must be chronological")
    gray = load_gray(output, ffmpeg)
    train, cal, dev = (load_labels(output, ids, gray)[0] for ids in (fit_ids, cal_ids, dev_ids))
    transforms = {int(i): np.asarray(matrix) for i, matrix in
                  read_json(output / "canonical-center-chained/canonical-registration.json")["transforms"].items()}
    features = np.load(output / "motion-integration/motion-integration.npy", mmap_mode="r", allow_pickle=False)
    study = output / "center-rescore"
    study.mkdir(exist_ok=True)
    path = study / "center.txt"
    if audit_only:
        original = read_json(study / "comparison.json")
        if sha_file(path) != original["scorer_sha256"]:
            raise ValueError("Archived center scorer weights have changed")
        model = lgb.Booster(model_file=str(path))
    else:
        source = load_candidates(DEFAULT_OUTPUT / "early-stacked", setup["fit_detector"], "forward-dense-fit")
        model, examples, positive = fit(parent, records(source, features, transforms), features, train)
        print(f"Fit center scorer on {examples} candidates, {positive} within {RADIUS:g} px", flush=True)
        model.save_model(str(path))
    outputs = []
    for scope in ("validation", "development"):
        neural = load_candidates(output, detector, scope)
        data = refined(parent, geometry, records(neural, features, transforms), features)
        outputs.append({i: (boxes, scores, model.predict(x, num_threads=5).astype(np.float32))
                        for i, (boxes, scores, x) in data.items()})

    cal_raw, dev_raw = outputs
    base_setup = archived["selected"]
    ref_pool = pool(dev_raw, 0, base_setup["method"], base_setup["strength"])
    detections = filtered(ref_pool, base_setup["threshold"], base_setup["low"], confirmations=support)
    reference, reference_rows = summarize(detections, dev)
    if reference != archived["development"]:
        raise ValueError("Archived detector settings failed to replay on the fixed boxes")

    if audit_only:
        saved = original["results"]["box_overlap"]
        branch = saved["selected"]
        candidates = pool(dev_raw, branch["weight"], branch["method"], branch["strength"])
        replayed = filtered(candidates, branch["threshold"], branch["low"], confirmations=support)
        if summarize(replayed, dev)[0] != saved["development"]:
            raise ValueError("Archived center scores failed to reproduce the selected operation")
        ranking_budget(study, branch, cal_raw, dev_raw, cal, dev, base_setup, support)
        return

    results, rows = {}, []
    for name, methods in (("box_overlap", (("vote", 0.35), ("hard", 0.35))), ("center_distance", (("center", 3.0), ("center", 5.0)))):
        selected, trials = select(cal_raw, cal, support, methods)
        candidates = pool(dev_raw, selected["weight"], selected["method"], selected["strength"])
        predicted = filtered(candidates, selected["threshold"], selected["low"], confirmations=support)
        metrics, frame_rows = summarize(predicted, dev)
        results[name] = dict(selected=selected, trials=trials, development=metrics,
                             output_center_matches=matched_centers(predicted, dev), candidate_center_matches=matched_centers(candidates, dev),
                             candidate_iou50_matches=coverage(candidates, dev), paired_vs_reference=paired_block_f1(frame_rows, reference_rows))
        rows.extend(dict(method=name, **row) for row in frame_rows)
        print(name, metrics, results[name]["output_center_matches"], flush=True)
    report = {"fit_ids": fit_ids, "center_radius": RADIUS, "scorer_sha256": sha_file(path), "fit_examples": examples,
              "positive_training_centers": positive, "matching_iou": 0.50, "center_discrimination": compare_scores(dev_raw, dev),
              "reference": {"development": reference, "output_center_matches": matched_centers(detections, dev),
                            "candidate_center_matches": matched_centers(ref_pool, dev)}, "results": results}
    write_json(study / "comparison.json", report)
    write_csv(study / "development.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--audit-ranking", action="store_true", help="Check the saved scorer at identical per-frame output counts")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    evaluate(args.output.resolve(), args.ffmpeg, audit_only=args.audit_ranking)
