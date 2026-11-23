"""Evaluate learned local position, size and velocity priors on fixed SNL candidates."""
import argparse
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.special import expit, logit
from scipy.stats import rankdata

from audit_detection import paired_block_f1
from experiment import CONFIDENCES, filtered, load_candidates, quality, summarize
from route_prior import centers
from snl_box_quality import GEOMETRY_KEYS
from snl_forward_dense_fusion import candidate_scores, expanded, records, score_map, suppress
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, WIDTH, load_gray, load_labels, read_json, sha_file, write_csv, write_json
from snl_local_shadow_proposals import coverage
from snl_protocol import passed
from snl_registered_route import project
from videosar.snl_causal import box_iou


LAGS = (2, 4, 6)
CUES = ("position", "shape", "directional_history")
ARMS = {"position": (0,), "position_and_shape": (0, 1), "position_and_direction": (0, 2),
        "all_cues": (0, 1, 2)}
SIZE_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)


def matched_tracks(registered):
    tracks, active = [], {}
    for i, position in registered.items():
        prior = registered.get(i - 1)
        follow = {}
        if prior is not None:
            separation = np.linalg.norm(prior[:, None] - position, axis=2)
            a, b = linear_sum_assignment(np.where(separation <= 9, separation, 1e4))
            for left, right in zip(a, b, strict=True):
                if separation[left, right] <= 9:
                    follow[right] = active[left]
                    tracks[active[left]].append(position[right])
        for j, point in enumerate(position):
            if j not in follow:
                follow[j] = len(tracks)
                tracks.append([point])
        active = follow
    return tracks


def velocities(tracks):
    locations, deltas = [], []
    time = np.arange(-2, 3, dtype=np.float32)
    for track in tracks:
        for j in range(2, len(track) - 2, 2):
            window = np.asarray(track[j - 2:j + 3])
            velocity = (time[:, None] * window).sum(axis=0) / (time @ time)
            fitted = window.mean(axis=0) + time[:, None] * velocity
            error = np.sqrt(np.mean(np.square(window - fitted)))
            if error <= 2 and 0.65 <= np.linalg.norm(velocity) <= 5.5:
                locations.append(window[2])
                deltas.append(velocity)
    if not locations:
        raise ValueError("Training does not contain enough continuous moving tracks")
    return np.asarray(locations), np.asarray(deltas, dtype=np.float32)


class LocalContext:
    def __init__(self, labels, transforms):
        self.transforms = transforms
        positions = {i: project(centers(boxes), transforms[i]) for i, boxes in labels.items()}
        self.locations = cKDTree(np.concatenate(list(positions.values())))
        self.sizes = np.concatenate([np.log(np.maximum(1, b[:, 2:] - b[:, :2]) * np.linalg.norm(transforms[i][:, 0]))
                                     for i, b in labels.items()])
        tracks = matched_tracks(positions)
        locations, self.deltas = velocities(tracks)
        self.movement = cKDTree(locations)
        self.counts = dict(labeled_points=len(self.sizes), connected_tracks=len(tracks), velocity_observations=len(self.deltas))

    def static_features(self, i, boxes):
        matrix = self.transforms[i]
        scale = np.linalg.norm(matrix[:, 0])
        transformed = project(centers(boxes), matrix).astype(np.float32)
        distances, index = self.locations.query(transformed, k=20)
        position = -np.log1p(distances[:, :8].mean(axis=1))
        weights = np.exp(-(distances - distances[:, :1]) / 8)
        weights /= weights.sum(axis=1, keepdims=True)
        nearby = self.sizes[index]
        mean = (weights[:, :, None] * nearby).sum(axis=1)
        std = np.sqrt((weights[:, :, None] * (nearby - mean[:, None])**2).sum(axis=1) + .25**2)
        size = np.log(np.maximum(1, boxes[:, 2:] - boxes[:, :2]) * scale)
        shape = -np.abs((size - mean) / std).mean(axis=1)
        return transformed, position, shape, mean - np.log(scale)

    def dynamic_feature(self, i, points, maps):
        history = [(lag, maps[i - lag]) for lag in LAGS if i - lag in maps]
        if not history:
            return np.zeros(len(points), dtype=np.float32)
        distances, index = self.movement.query(points, k=min(12, len(self.deltas)))
        offsets = self.deltas[index]
        offsets = np.concatenate((np.zeros_like(offsets[:, :1]), offsets), axis=1)
        votes = []
        for lag, density in history:
            x, y = (points[:, None] - lag * offsets).transpose(2, 0, 1)
            votes.append(cv2.remap(density, x.copy(), y.copy(), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
        average = np.exp(np.log(np.stack(votes) + 1e-4).mean(axis=0))
        moving = np.where(distances <= 40, average[:, 1:], 0).max(axis=1)
        return np.maximum(moving - average[:, 0], 0)

    def describe(self, raw, neural):
        maps = {i: expanded(cv2.warpAffine(score_map(boxes, confidence), self.transforms[i], (WIDTH, HEIGHT),
                                          flags=cv2.INTER_NEAREST), 3, 5)
                for i, (boxes, confidence) in neural.items()}
        output = {}
        for i, (boxes, _) in raw.items():
            points, position, shape, _ = self.static_features(i, boxes)
            output[i] = np.column_stack((position, shape, self.dynamic_feature(i, points, maps)))
        return output


def mix(raw, cues, point):
    columns = point["cue_indices"]
    if point["weight"] == 0:
        return raw
    center, spread = np.asarray(point["center"]), np.asarray(point["spread"])
    mixed = {}
    for i, (boxes, score) in raw.items():
        shift = ((cues[i][:, columns] - center) / spread).clip(-3, 3).mean(axis=1)
        mixed[i] = boxes, expit(logit(score.clip(1e-6, 1 - 1e-6)) + point["weight"] * shift).astype(np.float32)
    return mixed


def choose(raw, evidence, truth, columns, suppression):
    data = np.concatenate([evidence[i][:, columns] for i in truth])
    q1, center, q3 = np.quantile(data, (0.25, 0.50, 0.75), axis=0)
    spread = np.maximum(q3 - q1, 0.05)
    trials = []
    for weight in (0.0, 0.15, 0.35, 0.65, 1.0):
        rule = dict(cue_indices=columns, center=center.tolist(), spread=spread.tolist(), weight=weight)
        pooled = suppress(mix(raw, evidence, rule), suppression["method"], suppression["strength"])
        rows = [(float(t), summarize(filtered(pooled, float(t), None), truth)[0]) for t in CONFIDENCES]
        threshold, result = max(rows, key=lambda trial: quality(trial[1]))
        trials.append(dict(rule, threshold=threshold, calibration=result))
    return max(trials, key=lambda trial: quality(trial["calibration"])), trials


def feature_auc(raw, cues, labels):
    features, outcomes = [], []
    for i, targets in labels.items():
        boxes, scores = raw[i]
        features.append(np.column_stack((scores, cues[i])))
        outcomes.append(box_iou(boxes, targets).max(axis=1, initial=0) >= 0.50)
    positive = np.concatenate(outcomes)
    rank = rankdata(np.concatenate(features), axis=0)
    count = int(positive.sum())
    auc = (rank[positive].sum(axis=0) - count * (count + 1) / 2) / (count * (len(positive) - count))
    return dict(zip(("lambda_rank", *CUES), auc.tolist(), strict=True))


def change_size(raw, model, weight, *, retain_original=False):
    if weight == 0:
        return raw
    adjusted = {}
    for i, (boxes, scores) in raw.items():
        _, _, _, expected = model.static_features(i, boxes)
        size = np.maximum(1, boxes[:, 2:] - boxes[:, :2])
        half = 0.5 * np.exp((1 - weight) * np.log(size) + weight * expected)
        center = centers(boxes)
        bounds = (WIDTH, HEIGHT)
        corrected = np.column_stack(((center - half).clip(0, bounds), (center + half).clip(0, bounds))).astype(np.float32)
        adjusted[i] = ((np.concatenate((boxes, corrected)), np.concatenate((scores, scores * 0.85)))
                       if retain_original else (corrected, scores))
    return adjusted


def size_experiment(study, source, context, calibration, dev_ids, output, gray, reference):
    cal, _ = source("validation")
    rule = reference["selected"]
    results = {}
    for mode, retain in (("replacement", False), ("addition", True)):
        choices = []
        for weight in SIZE_WEIGHTS:
            pooled = suppress(change_size(cal, context, weight, retain_original=retain), rule["method"], rule["strength"])
            stats = [(float(t), summarize(filtered(pooled, float(t), None), calibration)[0]) for t in CONFIDENCES]
            threshold, metrics = max(stats, key=lambda trial: quality(trial[1]))
            choices.append(dict(size_weight=weight, threshold=threshold, calibration=metrics))
        results[mode] = dict(selected=max(choices, key=lambda trial: quality(trial["calibration"])), trials=choices)
    write_json(study / "size-calibration.json", results)
    dev, _ = source("development")
    truth = load_labels(output, dev_ids, gray)[0]
    base_pool = suppress(dev, rule["method"], rule["strength"])
    base, previous = summarize(filtered(base_pool, rule["threshold"], rule["low"]), truth)
    if base != reference["development"]:
        raise ValueError("Unmodified box sizing did not reproduce the archived comparison")
    rows = [dict(method="original", **row) for row in previous]
    for mode, arm in results.items():
        point = arm["selected"]
        retain = mode == "addition"
        audit = []
        for weight in SIZE_WEIGHTS:
            altered = change_size(dev, context, weight, retain_original=retain)
            pooled = suppress(altered, rule["method"], rule["strength"])
            before, after = coverage(altered, truth), coverage(pooled, truth)
            audit.append(dict(size_weight=weight, before_suppression=before, after_suppression=after))
            if weight == point["size_weight"]:
                outcome, per_frame = summarize(filtered(pooled, point["threshold"], None), truth)
                arm.update(development=outcome, development_gate_passed=passed(point["calibration"]) and passed(outcome),
                           paired_f1=paired_block_f1(per_frame, previous))
                rows.extend(dict(method=mode, **row) for row in per_frame)
        arm["candidate_coverage"] = audit
        print(mode, point, arm["development"], audit, flush=True)
    write_json(study / "size-comparison.json", dict(reference=base, results=results))
    write_csv(study / "size-development.csv", rows)


def evaluate(output, ffmpeg, *, audit_size=False):
    archive = output / "forward-dense-frame-ranking"
    parent = read_json(archive / "comparison.json")
    reference = read_json(archive / "suppression-comparison.json")
    paths = [archive / f"{part}.txt" for part in ("ranker", *GEOMETRY_KEYS)]
    if any(sha_file(path) != parent["weights"][path.name] for path in paths):
        raise ValueError("Archived ranking or geometry weights changed")
    ranker, *geometry = (lgb.Booster(model_file=str(path)) for path in paths)
    alignment_path = output / "canonical-center-chained/canonical-registration.json"
    alignment = read_json(alignment_path)
    transforms = {int(i): np.asarray(matrix) for i, matrix in alignment["transforms"].items()}
    train_ids, cal_ids, dev_ids = PROFILES["prospective"]
    if alignment["settings"]["anchor"] > max(train_ids):
        raise ValueError("The scene registration anchor cannot come from holdout footage")
    gray = load_gray(output, ffmpeg)
    train = load_labels(output, train_ids, gray)[0]
    calibration = load_labels(output, cal_ids, gray)[0]
    context = LocalContext(train, transforms)
    features = np.load(output / "motion-integration/motion-integration.npy", mmap_mode="r", allow_pickle=False)

    def source(scope):
        raw_neural = load_candidates(output, parent["evaluation_detector"], scope)
        measured = records(raw_neural, features, transforms)
        dense = {i: candidate_scores((ranker, geometry), value, parent["selected"]["refine"], frame_ranked=True)
                 for i, value in measured.items()}
        return dense, raw_neural

    study = output / "route-context"
    study.mkdir(exist_ok=True)
    if audit_size:
        size_experiment(study, source, context, calibration, dev_ids, output, gray, reference)
        return

    cal, neural = source("validation")
    cal_context = context.describe(cal, neural)
    first = min(cal_ids)
    prefix = context.describe({i: row for i, row in cal.items() if i <= first}, {i: row for i, row in neural.items() if i <= first})
    np.testing.assert_array_equal(cal_context[first], prefix[first])
    baseline_point = reference["selected"]
    arms = {}
    for name, columns in ARMS.items():
        selected, trials = choose(cal, cal_context, calibration, columns, baseline_point)
        arms[name] = dict(selected=selected, trials=trials)
    chosen = max(arms, key=lambda name: quality(arms[name]["selected"]["calibration"]))
    report = dict(fit_ids=train_ids, model=context.counts, alignment_sha256=sha_file(alignment_path), matching_iou=0.50,
                  prefix_check=first, selected_by_calibration=chosen, arms=arms)
    write_json(study / "calibration.json", report)
    print("Training route:", context.counts, "calibration selection:", chosen, arms[chosen]["selected"], flush=True)

    dev, neural = source("development")
    dev_context = context.describe(dev, neural)
    development = load_labels(output, dev_ids, gray)[0]
    ctrl_pool = suppress(dev, baseline_point["method"], baseline_point["strength"])
    base_prediction = filtered(ctrl_pool, baseline_point["threshold"], baseline_point["low"])
    baseline, base_rows = summarize(base_prediction, development)
    if baseline != reference["development"]:
        raise ValueError("Original detection reference did not replay")
    rows = [dict(method="reference", **row) for row in base_rows]
    for name, arm in arms.items():
        point = arm["selected"]
        pooled = suppress(mix(dev, dev_context, point), baseline_point["method"], baseline_point["strength"])
        eval_rows = filtered(pooled, point["threshold"], None)
        result, per_frame = summarize(eval_rows, development)
        fixed = {}
        for i in development:
            boxes, scores = pooled[i]
            limit = len(base_prediction[i][0])
            order = np.argsort(-scores, kind="stable")[:limit]
            if len(order) != limit:
                raise ValueError("Rescoring did not leave enough candidates for the fixed output budget")
            fixed[i] = boxes[order], scores[order], None
        fixed_result, _ = summarize(fixed, development)
        arm.update(development=result, fixed_output_count=fixed_result, ceiling=coverage(pooled, development),
                   gate=passed(point["calibration"]) and passed(result),
                   paired_vs_reference=paired_block_f1(per_frame, base_rows))
        rows.extend(dict(method=name, **row) for row in per_frame)
        print(name, result, "equal-count hits:", fixed_result["tp"], flush=True)
    report.update(reference_development=baseline, development_auc=feature_auc(dev, dev_context, development))
    write_json(study / "comparison.json", report)
    write_csv(study / "development.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--audit-size", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(3)
    evaluate(args.output.resolve(), args.ffmpeg, audit_size=args.audit_size)
