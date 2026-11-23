"""Separate shadow presence and localization quality on a fixed candidate pool."""
import argparse
from itertools import product
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
from scipy.special import expit

from audit_detection import paired_block_f1
from experiment import CONFIDENCES, filtered, load_candidates, quality, summarize
from snl_forward_dense_fusion import candidate_scores, records, suppress
from snl_fusion import box_means
from snl_inputs import DEFAULT_OUTPUT, PROFILES, load_gray, load_labels, read_json, sha_file, write_csv, write_json
from snl_local_shadow_proposals import coverage, supervision
from videosar.snl_causal import box_iou


GEOMETRY_KEYS = ("x0", "y0", "x1", "y1")
TREE = dict(num_leaves=15, max_depth=5, learning_rate=0.05, min_data_in_leaf=22,
            seed=331, num_threads=5, verbosity=-1, force_col_wise=True)


def fit_geometry(samples):
    data, boxes, overlap, target = (np.concatenate(part) for part in zip(*samples, strict=True))
    near = overlap >= 0.22
    boxes = boxes[near]
    size = np.tile(np.maximum(3, boxes[:, 2:] - boxes[:, :2]), (1, 2))
    offsets = ((target[near] - boxes) / size).clip(-0.50, 0.50)
    data = np.ascontiguousarray(data[near])
    return [lgb.train(dict(TREE, objective="regression"),
                      lgb.Dataset(data, label=offsets[:, j].copy(), weight=np.sqrt(overlap[near])),
                      num_boost_round=120) for j in range(4)]


def alignment_features(frame, boxes):
    x0, y0, x1, y1 = boxes.T
    widths, heights = x1 - x0, y1 - y0
    cuts_x = (x0 - 0.35 * widths, x0, x0 + widths / 3, x1 - widths / 3, x1, x1 + 0.35 * widths)
    cuts_y = (y0 - 0.25 * heights, y0, y0 + heights / 3, y1 - heights / 3, y1, y1 + 0.25 * heights)
    output = [widths, heights]
    for channel in (0, 1, 4):
        integral = cv2.integral(frame[channel], sdepth=cv2.CV_64F) / 255
        horizontal = [box_means(integral, np.column_stack((left, y0, right, y1)))
                      for left, right in zip(cuts_x[:-1], cuts_x[1:], strict=True)]
        vertical = [box_means(integral, np.column_stack((x0, top, x1, bottom)))
                    for top, bottom in zip(cuts_y[:-1], cuts_y[1:], strict=True)]
        for profile in (horizontal, vertical):
            output.extend((profile[2], profile[1] - profile[0], profile[3] - profile[4],
                           profile[2] - 0.5 * (profile[0] + profile[4]), profile[1] - profile[3]))
    return np.column_stack(output).astype(np.float32)


def refined_records(parent, geometry, source_records, features):
    result = {}
    for index, (raw_boxes, data) in source_records.items():
        boxes, scores = candidate_scores((parent, geometry), (raw_boxes, data), True, frame_ranked=True)
        # Archived box refinement clips offsets to less than half a box per edge.
        if len(boxes) != len(raw_boxes):
            raise ValueError("Refinement removed candidates from the fixed pool")
        alignment = alignment_features(features[index - 2], boxes)
        result[index] = boxes, scores, np.column_stack((data, alignment))
    return result


def crossfit(parent, source_records, features, fitting, *, group_by_target=False):
    groups = np.array_split(list(fitting), 3)
    targets = {i: supervision(source_records[i], truth) for i, truth in fitting.items()}
    out = {}
    for fold in groups:
        geometry = fit_geometry([row for i, row in targets.items() if i not in fold])
        for i, (boxes, _, data) in refined_records(parent, geometry, {i: source_records[i] for i in fold}, features).items():
            overlap = box_iou(boxes, fitting[i])
            best = overlap.max(axis=1, initial=0)
            if group_by_target and overlap.shape[1]:
                owner = overlap.argmax(axis=1)
                for target in range(overlap.shape[1]):
                    near = (owner == target) & (best >= 0.20)
                    grades = best[near] >= 0.50
                    if grades.any() and not grades.all():
                        out[i, target] = data[near], best[near]
            elif not group_by_target:
                out[i] = data, best
    return out


def fit_quality(samples):
    data, overlap = (np.concatenate(part) for part in zip(*samples.values(), strict=True))
    foreground = overlap >= 0.20
    good_box = overlap >= 0.50
    presence = lgb.train(dict(TREE, objective="binary"),
                         lgb.Dataset(data, label=foreground, weight=np.where(foreground, 5, 1)), num_boost_round=120)
    localization = lgb.train(dict(TREE, objective="binary"),
                             lgb.Dataset(data[foreground], label=good_box[foreground]), num_boost_round=120)
    return presence, localization, int(foreground.sum()), int(good_box.sum())


def fit_local_ranker(groups):
    data, overlap = (np.concatenate(part) for part in zip(*groups.values(), strict=True))
    grades = np.searchsorted((0.35, 0.50, 0.65, 0.80), overlap, side="right")
    options = dict(TREE, objective="lambdarank", lambdarank_truncation_level=5, label_gain=[0, 0, 3, 6, 10])
    grouped = lgb.Dataset(data, label=grades, group=[len(row[1]) for row in groups.values()])
    return lgb.train(options, grouped, num_boost_round=120), len(overlap)


def predictions(data, models):
    return {i: (boxes, baseline, *(model.predict(x, num_threads=5).astype(np.float32) for model in models))
            for i, (boxes, baseline, x) in data.items()}


def blend(raw, weight, use_localization):
    out = {}
    for i, (boxes, baseline, presence, localization) in raw.items():
        estimated = presence * localization if use_localization else presence
        conf = (1 - weight) * baseline + weight * estimated
        out[i] = boxes, conf
    return out


def tune(raw, labels, support, weight, use_localization):
    candidates = blend(raw, weight, use_localization)
    choice = []
    for method, strength in (("hard", 0.35), ("hard", 0.65), ("vote", 0.35)):
        pooled = suppress(candidates, method, strength)
        for threshold, track in product(CONFIDENCES, (False, True)):
            low = float(threshold) / 2 if track else None
            evaluation = summarize(filtered(pooled, float(threshold), low, confirmations=support), labels)[0]
            choice.append(dict(method=method, strength=strength, threshold=float(threshold), low=low,
                               weight=weight, use_localization=use_localization, calibration=evaluation))
    return max(choice, key=lambda row: quality(row["calibration"]))


def replay(raw, labels, operation, support):
    candidates = blend(raw, operation["weight"], operation["use_localization"])
    pooled = suppress(candidates, operation["method"], operation["strength"])
    selected = filtered(pooled, operation["threshold"], operation["low"], confirmations=support)
    result, rows = summarize(selected, labels)
    return result, rows, coverage(pooled, labels)


def choose_representatives(raw, group_iou, weight):
    """Keep cluster confidence and select one of its existing boxes by localization score."""
    output = {}
    for i, (boxes, baseline, presence, localization) in raw.items():
        visible = baseline >= float(CONFIDENCES[0]) / 2
        seeds, scores = suppress({i: (boxes[visible], baseline[visible])}, "hard", group_iou)[i]
        evidence = (1 - weight) * baseline + weight * presence * localization
        selected = []
        for start in range(0, len(seeds), 64):
            members = box_iou(seeds[start:start + 64], boxes) >= group_iou
            selected.extend(np.where(members, evidence, -1).argmax(axis=1))
        unique = np.sort(np.unique(selected, return_index=True)[1])
        output[i] = boxes[np.asarray(selected, dtype=np.int32)[unique]], scores[unique]
    return output


def audit_local_ranking(raw, labels):
    eligible = 0
    correct = dict(archived=0, presence=0, localization=0, product=0)
    for i, truth in labels.items():
        boxes, baseline, presence, localization = raw[i]
        for overlaps in box_iou(boxes, truth).T:
            good = overlaps >= 0.50
            nearby = overlaps >= 0.20
            if not good.any() or not (nearby & ~good).any():
                continue
            eligible += 1
            for key, scores in zip(correct, (baseline, presence, localization, presence * localization), strict=True):
                choice = np.where(nearby, scores, -1).argmax()
                correct[key] += int(good[choice])
    return {"truth_grouped_examples": eligible, "highest_ranked_matches": correct}


def cluster_comparison(study, cal, dev, calibration, development, support, baseline_rows):
    trials = []
    for group_iou, weight in product((0.35, 0.50), (0.50, 1.0)):
        candidates = choose_representatives(cal, group_iou, weight)
        settings = []
        for threshold, track in product(CONFIDENCES, (False, True)):
            low = float(threshold) / 2 if track else None
            predicted = filtered(candidates, float(threshold), low, confirmations=support)
            settings.append(dict(group_iou=group_iou, quality_weight=weight, threshold=float(threshold), low=low,
                                 calibration=summarize(predicted, calibration)[0]))
        trials.append(max(settings, key=lambda row: quality(row["calibration"])))
    best = max(trials, key=lambda row: quality(row["calibration"]))
    candidates = choose_representatives(dev, best["group_iou"], best["quality_weight"])
    predictions = filtered(candidates, best["threshold"], best["low"], confirmations=support)
    metrics, rows = summarize(predictions, development)
    write_json(study / "cluster-comparison.json",
               dict(trials=trials, selected=best, development=metrics, post_suppression_matches=coverage(candidates, development),
                    paired_vs_parent=paired_block_f1(rows, baseline_rows),
                    truth_grouped_local_ranking=audit_local_ranking(dev, development)))
    write_csv(study / "cluster-development.csv", rows)
    print("cluster_representatives", metrics, flush=True)


def run(output, ffmpeg, *, clusters_only=False, local_rank=False):
    archive = output / "forward-dense-frame-ranking"
    parent_report = read_json(archive / "comparison.json")
    parent_vote = read_json(archive / "suppression-comparison.json")
    parent_paths = [archive / f"{key}.txt" for key in ("ranker", *GEOMETRY_KEYS)]
    if any(sha_file(path) != parent_report["weights"][path.name] for path in parent_paths):
        raise ValueError("Archived ranking or refinement weights have changed")
    parent, *geometry = (lgb.Booster(model_file=str(path)) for path in parent_paths)
    evaluation_detector = parent_report["evaluation_detector"]
    support = evaluation_detector.get("track_support", 1)
    _, cal_ids, dev_ids = PROFILES["prospective"]
    fit_ids = parent_report["fit_ids"]
    if max(fit_ids) >= min(cal_ids) or max(cal_ids) >= min(dev_ids):
        raise ValueError("Box-quality data must remain chronological")
    gray = load_gray(output, ffmpeg)
    fitting, calibration, development = (load_labels(output, ids, gray)[0] for ids in (fit_ids, cal_ids, dev_ids))
    features = np.load(output / "motion-integration/motion-integration.npy", mmap_mode="r", allow_pickle=False)
    registration = read_json(output / "canonical-center-chained/canonical-registration.json")
    transforms = {int(i): np.asarray(matrix) for i, matrix in registration["transforms"].items()}
    study = output / ("local-box-ranking" if local_rank else "box-quality")
    study.mkdir(exist_ok=True)
    if local_rank:
        early = load_candidates(DEFAULT_OUTPUT / "early-stacked", parent_report["fit_detector"], "forward-dense-fit")
        groups = crossfit(parent, records(early, features, transforms), features, fitting, group_by_target=True)
        model, examples = fit_local_ranker(groups)
        path = study / "local-ranker.txt"
        model.save_model(str(path))
        write_json(study / "fitting.json", {"fit_ids": fit_ids, "groups": len(groups), "examples": examples,
                                            "weights_sha256": sha_file(path)})
        del groups
        print(f"Fit local ranking on {examples} competing boxes", flush=True)
    elif clusters_only:
        weights = read_json(study / "comparison.json")["weights"]
        paths = [study / f"{name}.txt" for name in ("presence", "localization")]
        if any(sha_file(path) != weights[path.name] for path in paths):
            raise ValueError("Archived quality models have changed")
        presence, localization = (lgb.Booster(model_file=str(path)) for path in paths)
    else:
        early = load_candidates(DEFAULT_OUTPUT / "early-stacked", parent_report["fit_detector"], "forward-dense-fit")
        folds = crossfit(parent, records(early, features, transforms), features, fitting)
        presence, localization, nearby, good = fit_quality(folds)
        del folds
        print(f"Trained quality scorers on {nearby} nearby candidates, including {good} IoU 0.50 boxes", flush=True)
        weights = {}
        for name, model in (("presence", presence), ("localization", localization)):
            path = study / f"{name}.txt"
            model.save_model(str(path))
            weights[path.name] = sha_file(path)

    processed = []
    for scope in ("validation", "development"):
        detector_rows = records(load_candidates(output, evaluation_detector, scope), features, transforms)
        data = refined_records(parent, geometry, detector_rows, features)
        if local_rank:
            processed.append({i: (boxes, confidence, np.ones_like(confidence),
                                  expit(model.predict(x, num_threads=5)).astype(np.float32))
                              for i, (boxes, confidence, x) in data.items()})
        else:
            processed.append(predictions(data, (presence, localization)))
    cal, dev = processed
    archived = dict(parent_vote["selected"], weight=0.0, use_localization=False)
    baseline, baseline_rows, _ = replay(dev, development, archived, support)
    if baseline != parent_vote["development"]:
        raise ValueError("Fixed geometry and original scores failed to reproduce the archived detector")
    if clusters_only or local_rank:
        cluster_comparison(study, cal, dev, calibration, development, support, baseline_rows)
        return

    results, rows, trials = {}, [], {}
    for name, conditional in (("presence_only", False), ("presence_and_localization", True)):
        options = [tune(cal, calibration, support, w, conditional) for w in (0.25, 0.50, 0.75, 1.0)]
        selected = max(options, key=lambda row: quality(row["calibration"]))
        metrics, points, retained = replay(dev, development, selected, support)
        rows.extend(dict(method=name, **record) for record in points)
        trials[name] = options
        results[name] = dict(operation=selected, development=metrics, post_suppression_matches=retained,
                             paired_vs_parent=paired_block_f1(points, baseline_rows))
        print(name, metrics, f"coverage after suppression {retained}", flush=True)
    selected = max(results, key=lambda key: quality(results[key]["operation"]["calibration"]))
    report = dict(fit_ids=fit_ids, weights=weights, neighborhood_label_iou=0.20, localization_label_iou=0.50,
                  geometry_crossfit_blocks=3, examples_nearby=nearby, examples_iou50=good,
                  candidate_matches=coverage(dev, development), archived_parent=baseline,
                  selected_by_calibration=selected, trials=trials, results=results)
    write_json(study / "comparison.json", report)
    write_csv(study / "development.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--clusters", action="store_true", help="Reuse trained scorers to select cluster representatives")
    modes.add_argument("--local-rank", action="store_true", help="Rank nearby training candidates within each target")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    run(args.output.resolve(), args.ffmpeg, clusters_only=args.clusters, local_rank=args.local_rank)
