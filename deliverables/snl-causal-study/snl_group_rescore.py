"""Train scores for complete candidate groups with forward-held-out detector predictions."""
import argparse
from itertools import product
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
from scipy.spatial import cKDTree

from audit_detection import paired_block_f1
from experiment import CONFIDENCES, filtered, load_candidates, nonmax, quality, summarize
from snl_box_quality import GEOMETRY_KEYS, TREE, fit_geometry
from snl_forward_dense_fusion import candidate_scores, records, suppress, train_frame_ranker
from snl_fusion import measurement
from snl_inputs import DEFAULT_OUTPUT, PROFILES, load_gray, load_labels, read_json, sha_file, write_csv, write_json
from snl_local_shadow_proposals import coverage, supervision
from videosar.snl_causal import matched_boxes


GROUPINGS = {"vote": (0.35, True), "loose": (0.80, False)}


def centers(boxes):
    return (boxes[:, :2] + boxes[:, 2:]) * 0.5


def overlapping_members(seeds, boxes):
    """Return all pairs with IoU at least 0.50 without a dense IoU matrix."""
    size = boxes[:, 2:] - boxes[:, :2]
    seed_size = seeds[:, 2:] - seeds[:, :2]
    tree = cKDTree(centers(boxes))
    radius = (0.5 * (seed_size + size.max(axis=0))).max(axis=1) + 1e-4
    neighborhoods = tree.query_ball_point(centers(seeds), radius, p=np.inf)
    rows = np.repeat(np.arange(len(seeds)), [len(group) for group in neighborhoods])
    cols = np.concatenate([np.asarray(group, dtype=np.int64) for group in neighborhoods])
    lo = np.maximum(seeds[rows, :2], boxes[cols, :2])
    hi = np.minimum(seeds[rows, 2:], boxes[cols, 2:])
    intersection = np.maximum(hi - lo, 0).prod(axis=1)
    union = seed_size[rows].prod(axis=1) + size[cols].prod(axis=1) - intersection
    overlap = intersection / union
    keep = overlap >= 0.50
    return rows[keep], cols[keep], overlap[keep]


def grouped_row(boxes, scores, frame, threshold, vote):
    visible = scores >= float(CONFIDENCES[0]) / 2
    boxes, scores = boxes[visible], scores[visible]
    seeds, confidence = nonmax(boxes, scores, threshold)
    if not len(seeds):
        columns = measurement(frame, seeds).shape[1] + 11
        return seeds, confidence, np.empty((0, columns), dtype=np.float32)
    rows, cols, overlaps = overlapping_members(seeds, boxes)
    weight = scores[cols]**2
    count = np.bincount(rows, minlength=len(seeds))
    mass = np.bincount(rows, weights=weight, minlength=len(seeds))
    offsets = centers(boxes[cols]) - centers(seeds[rows])

    def aggregate(values, weights=None):
        return np.bincount(rows, weights=values if weights is None else values * weights, minlength=len(seeds))

    mean_score = aggregate(scores[cols]) / count
    score_sd = np.sqrt(np.maximum(aggregate(scores[cols]**2) / count - mean_score**2, 0))
    centroid = np.column_stack([aggregate(boxes[cols, j], weight) / mass for j in range(4)]).astype(np.float32)
    shifted = np.linalg.norm(centers(centroid) - centers(seeds), axis=1)
    dispersion = [np.sqrt(aggregate(offsets[:, j]**2, weight) / mass) for j in range(2)]
    broad = aggregate(overlaps) / count
    tight = aggregate(overlaps >= 0.70) / count
    strong = aggregate(scores[cols] >= 0.80 * confidence[rows]) / count
    if vote:
        seeds = centroid
    data = np.column_stack((measurement(frame, seeds), confidence, np.log1p(count), np.log1p(mass), mean_score,
                            score_sd, shifted, *dispersion, broad, tight, strong)).astype(np.float32)
    return seeds, confidence, data


def group_scope(raw, features, settings):
    return {i: grouped_row(boxes, scores, features[i - 2], *settings) for i, (boxes, scores) in raw.items()}


def crossfit(source, truth):
    supervised = {i: supervision(source[i], values) for i, values in truth.items()}
    out = {}
    for index, holdout in enumerate(np.array_split(list(truth), 3), start=1):
        fit = {i: value for i, value in supervised.items() if i not in holdout}
        ranker = train_frame_ranker(fit), fit_geometry(list(fit.values()))
        out.update({i: candidate_scores(ranker, source[i], True, frame_ranked=True) for i in holdout})
        print(f"Replayed held-out boxes for block {index}/3", flush=True)
    return out


def fit_groups(rows, truth):
    labels, features = [], []
    for i, (boxes, _, data) in rows.items():
        target = np.zeros(len(boxes), dtype=bool)
        target[matched_boxes(boxes, truth[i])[:, 0]] = True
        labels.append(target)
        features.append(data)
    labels, features = np.concatenate(labels), np.concatenate(features)
    if not 0 < labels.sum() < len(labels):
        raise ValueError("Group training requires both matched and unmatched examples")
    model = lgb.train(dict(TREE, objective="binary"),
                      lgb.Dataset(features, label=labels, weight=np.where(labels, 6, 1)), num_boost_round=120)
    return model, len(labels), int(labels.sum())


def scored(groups, model):
    return {i: (boxes, base, model.predict(x, num_threads=5).astype(np.float32)) for i, (boxes, base, x) in groups.items()}


def mixed(rows, weight):
    return {i: (boxes, (1 - weight) * base + weight * score) for i, (boxes, base, score) in rows.items()}


def tune(points, labels, support, weights):
    trials = []
    for weight in weights:
        candidate = mixed(points, weight)
        for threshold, track in product(CONFIDENCES, (False, True)):
            low = float(threshold) / 2 if track else None
            output = filtered(candidate, float(threshold), low, confirmations=support)
            metrics = summarize(output, labels)[0]
            trials.append(dict(weight=weight, threshold=float(threshold), low=low, calibration=metrics))
    return max(trials, key=lambda row: quality(row["calibration"]))


def replay(raw, labels, point, support):
    output = filtered(mixed(raw, point["weight"]), point["threshold"], point["low"], confirmations=support)
    return summarize(output, labels)


def same_count(raw, labels, base, weight):
    ranked = {}
    for i in labels:
        boxes, score = raw[i]
        k = len(base[i][0])
        if k > len(boxes):
            raise ValueError("A fixed output count exceeds the candidate pool")
        keep = np.argsort(-score, kind="stable")[:k]
        ranked[i] = boxes[keep], score[keep], None
    return summarize(ranked, labels)[0] | {"group_weight": weight}


def run(output, ffmpeg):
    archive = output / "forward-dense-frame-ranking"
    recipe = read_json(archive / "comparison.json")
    baseline = read_json(archive / "suppression-comparison.json")
    paths = [archive / f"{key}.txt" for key in ("ranker", *GEOMETRY_KEYS)]
    if any(sha_file(path) != recipe["weights"][path.name] for path in paths):
        raise ValueError("Archived candidate weights changed")
    ranker, *geometry = (lgb.Booster(model_file=str(path)) for path in paths)
    cfg = recipe["evaluation_detector"]
    support = cfg.get("track_support", 1)
    _, cal_ids, dev_ids = PROFILES["prospective"]
    fit_ids = recipe["fit_ids"]
    if max(fit_ids) >= min(cal_ids) or max(cal_ids) >= min(dev_ids):
        raise ValueError("The three experiment splits must remain chronological")
    gray = load_gray(output, ffmpeg)
    train, cal, dev = (load_labels(output, ids, gray)[0] for ids in (fit_ids, cal_ids, dev_ids))
    transforms = {int(i): np.asarray(matrix) for i, matrix in
                  read_json(output / "canonical-center-chained/canonical-registration.json")["transforms"].items()}
    features = np.load(output / "motion-integration/motion-integration.npy", mmap_mode="r", allow_pickle=False)
    earlier = load_candidates(DEFAULT_OUTPUT / "early-stacked", recipe["fit_detector"], "forward-dense-fit")
    fit_raw = crossfit(records(earlier, features, transforms), train)
    scored_scopes = []
    for scope in ("validation", "development"):
        candidates = records(load_candidates(output, cfg, scope), features, transforms)
        scored_scopes.append({i: candidate_scores((ranker, geometry), data, recipe["selected"]["refine"], frame_ranked=True)
                              for i, data in candidates.items()})
    cal_raw, dev_raw = scored_scopes
    point = baseline["selected"]
    reference_pool = suppress(dev_raw, point["method"], point["strength"])
    reference_predictions = filtered(reference_pool, point["threshold"], point["low"], confirmations=support)
    reference, frame_rows = summarize(reference_predictions, dev)
    if reference != baseline["development"]:
        raise ValueError("The original scoring and suppression settings failed to replay")
    study = output / "group-rescore"
    study.mkdir(exist_ok=True)
    results, all_rows = {}, []
    for name, grouping in GROUPINGS.items():
        model, examples, positives = fit_groups(group_scope(fit_raw, features, grouping), train)
        path = study / f"{name}.txt"
        model.save_model(str(path))
        train_cal = scored(group_scope(cal_raw, features, grouping), model)
        train_dev = scored(group_scope(dev_raw, features, grouping), model)
        selected = tune(train_cal, cal, support, (0.25, 0.50, 0.75, 1.0))
        control = tune(train_cal, cal, support, (0,))
        metrics, rows = replay(train_dev, dev, selected, support)
        control_metrics, _ = replay(train_dev, dev, control, support)
        fixed_count = same_count(mixed(train_dev, selected["weight"]), dev, reference_predictions, selected["weight"])
        results[name] = {"grouping": grouping, "fit_examples": examples, "fit_positives": positives, "weights_sha256": sha_file(path),
                         "selected": selected, "development": metrics, "same_geometry_control": control_metrics,
                         "fixed_output_count": fixed_count, "iou50_ceiling": coverage(train_dev, dev),
                         "paired_vs_reference": paired_block_f1(rows, frame_rows)}
        all_rows.extend(dict(method=name, **row) for row in rows)
        print(f"{name}: {metrics}; original-score control {control_metrics}; IoU 0.50 coverage {results[name]['iou50_ceiling']}", flush=True)
    chosen = max(results, key=lambda key: quality(results[key]["selected"]["calibration"]))
    write_json(study / "comparison.json", {"fit_ids": fit_ids, "crossfit_blocks": 3, "selected_by_calibration": chosen,
                                          "reference": reference, "results": results})
    write_csv(study / "development.csv", all_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    run(args.output.resolve(), args.ffmpeg)
