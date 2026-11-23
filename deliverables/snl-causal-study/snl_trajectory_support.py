"""Compare independent history with coherent causal motion support on fixed SNL boxes."""
import argparse
from itertools import product
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata

from audit_detection import audit, paired_block_f1
from experiment import CONFIDENCES, filtered, load_candidates, quality, summarize
from snl_forward_dense_fusion import candidate_scores, expanded, records, score_map, suppress
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, WIDTH, load_gray, load_labels, read_json, sha_file
from snl_inputs import write_csv, write_json
from snl_local_shadow_proposals import coverage
from videosar.snl_causal import box_iou


LAGS = (2, 4, 6)
SPEEDS = (1.0, 2.5, 4.5)
DIRECTIONS = 12
METHODS = ("independent", "consistent", "motion_advantage")


def velocity_bank():
    angle = np.arange(DIRECTIONS, dtype=np.float32) * (2 * np.pi / DIRECTIONS)
    directions = np.column_stack((np.cos(angle), np.sin(angle)))
    moving = np.asarray(SPEEDS, dtype=np.float32)[:, None, None] * directions[None, ...]
    return np.concatenate((np.zeros((1, 2), dtype=np.float32), moving.reshape(-1, 2)))


def score_centers(centers, past_maps, velocities):
    if not len(centers) or not past_maps:
        return np.zeros((len(centers), len(METHODS)), dtype=np.float32)
    votes = []
    for lag, density in past_maps:
        x, y = (centers[:, None, :] - lag * velocities[None, :, :]).transpose(2, 0, 1)
        votes.append(cv2.remap(density, x.copy(), y.copy(), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
    log_score = np.log(np.stack(votes) + 1e-4)
    trajectory = log_score.mean(axis=0)
    independent = np.exp(log_score.max(axis=2).mean(axis=0))
    consistent = np.exp(trajectory.max(axis=1))
    motion_gain = np.maximum(consistent - np.exp(trajectory[:, 0]), 0)
    return np.column_stack((independent, consistent, motion_gain))


def history_evidence(candidates, neural, transforms):
    velocities = velocity_bank()
    native_maps = {i: score_map(boxes, confidence) for i, (boxes, confidence) in neural.items()}
    response = {}
    for i, (boxes, _) in candidates.items():
        past_maps = []
        inverse = cv2.invertAffineTransform(transforms[i])
        for lag in LAGS:
            j = i - lag
            if j in native_maps:
                to_current = inverse @ np.vstack((transforms[j], (0, 0, 1)))
                density = cv2.warpAffine(native_maps[j], to_current, (WIDTH, HEIGHT), flags=cv2.INTER_NEAREST)
                past_maps.append((lag, expanded(density, 3, 5)))
        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        # Convert canonical velocity vectors to the current native grid.
        native_velocities = (velocities @ inverse[:, :2].T).astype(np.float32)
        response[i] = score_centers(centers, past_maps, native_velocities)
    return response


def fused(raw, evidence, cue, center, spread, weight):
    return {i: (boxes, expit(logit(confidence.clip(1e-6, 1 - 1e-6))
                             + weight * ((evidence[i][:, cue] - center) / spread).clip(-3, 3)).astype(np.float32))
            for i, (boxes, confidence) in raw.items()}


def score_candidate_predictions(raw, labels, operation, support):
    pool = suppress(raw, operation["method"], operation["strength"])
    output = filtered(pool, operation["threshold"], operation["low"], confirmations=support)
    result, rows = summarize(output, labels)
    return result, rows, coverage(pool, labels)


def fit_calibration(raw, evidence, calibration, support, cue):
    values = np.concatenate([evidence[i][:, cue] for i in calibration])
    low, center, high = map(float, np.quantile(values, (0.25, 0.50, 0.75)))
    spread = max(high - low, 1e-4)
    best = None
    for weight in (0.25, 0.50, 1.0):
        predictions = fused(raw, evidence, cue, center, spread, weight)
        for method, strength in (("hard", 0.35), ("vote", 0.35)):
            pool = suppress(predictions, method, strength)
            for threshold, track in product(CONFIDENCES, (False, True)):
                point = dict(cue=cue, center=center, spread=spread, weight=weight, method=method, strength=strength,
                             threshold=float(threshold), low=float(threshold) / 2 if track else None)
                output = filtered(pool, point["threshold"], point["low"], confirmations=support)
                metrics = summarize(output, calibration)[0]
                if best is None or quality(metrics) > quality(best["calibration"]):
                    best = dict(point, calibration=metrics)
    return best


def conditional_audit(candidates, evidence, labels):
    samples, matches = [], []
    for i, truth in labels.items():
        boxes, scores = candidates[i]
        overlap = box_iou(boxes, truth).max(axis=1, initial=0)
        near = overlap >= 0.20
        samples.append(np.column_stack((scores[near], evidence[i][near])))
        matches.append(overlap[near] >= 0.50)
    values = np.concatenate(samples)
    positive = np.concatenate(matches)
    count = int(positive.sum())
    ranked = rankdata(values, axis=0)
    if 0 < count < len(positive):
        auc = (ranked[positive].sum(axis=0) - count * (count + 1) / 2) / (count * (len(positive) - count))
        scores = auc.tolist()
    else:
        scores = [None] * values.shape[1]
    return {"nearby_candidates": len(positive), "iou50_candidates": count,
            "auc": dict(zip(("original", *METHODS), scores, strict=True))}


def budget_comparison(study, scopes, truth, baseline_settings, support):
    settings = read_json(study / "comparison.json")["results"]
    results = {name: {} for name in METHODS}
    rows = []
    for (raw, evidence), labels, split in zip(scopes, truth, ("calibration", "development"), strict=True):
        original = suppress(raw, baseline_settings["method"], baseline_settings["strength"])
        baseline = filtered(original, baseline_settings["threshold"], baseline_settings["low"], confirmations=support)
        control, control_rows = summarize(baseline, labels)
        for name in METHODS:
            point = settings[name]["settings"]
            prediction = fused(raw, evidence, point["cue"], point["center"], point["spread"], point["weight"])
            pool = suppress(prediction, baseline_settings["method"], baseline_settings["strength"])
            limited = {}
            for i in labels:
                boxes, scores = pool[i]
                cap = len(baseline[i][0])
                if len(boxes) < cap:
                    raise ValueError(f"Historical candidate pool could not reproduce frame {i}'s output count")
                keep = np.argsort(-scores, kind="stable")[:cap]
                limited[i] = boxes[keep], scores[keep], None
            metrics, frames = summarize(limited, labels)
            results[name][split] = {"reference": control, "historical": metrics,
                                    "paired_f1_vs_reference": paired_block_f1(frames, control_rows)}
            rows.extend(dict(method=name, split=split, **row) for row in frames)
    best = max(results, key=lambda name: quality(results[name]["calibration"]["historical"]))
    write_json(study / "fixed-budget-comparison.json", {"selected_by_calibration": best, "results": results})
    write_csv(study / "fixed-budget-per-frame.csv", rows)
    print("Fixed output count", best, results[best], flush=True)


def run(output, ffmpeg, *, geometry_audit=False, matched_budget=False):
    source = output / "forward-dense-frame-ranking"
    parent = read_json(source / "comparison.json")
    suppression = read_json(source / "suppression-comparison.json")
    keys = ("ranker", "x0", "y0", "x1", "y1")
    paths = [source / f"{key}.txt" for key in keys]
    if any(sha_file(path) != parent["weights"][path.name] for path in paths):
        raise ValueError("Cached ranking weights differ from the recorded comparison")
    models = [lgb.Booster(model_file=str(path)) for path in paths]
    detector = parent["evaluation_detector"]
    _, cal_ids, dev_ids = PROFILES["prospective"]
    gray = load_gray(output, ffmpeg)
    calibration, development = (load_labels(output, ids, gray)[0] for ids in (cal_ids, dev_ids))
    registration = read_json(output / "canonical-center-chained/canonical-registration.json")
    transforms = {int(i): np.asarray(matrix) for i, matrix in registration["transforms"].items()}
    features = np.load(output / "motion-integration/motion-integration.npy", mmap_mode="r", allow_pickle=False)
    study = output / "trajectory-support"
    study.mkdir(exist_ok=True)
    scopes = []
    for name in (("development",) if geometry_audit else ("validation", "development")):
        neural = load_candidates(output, detector, name)
        candidate_rows = records(neural, features, transforms)
        raw = {i: candidate_scores((models[0], models[1:]), data, parent["selected"]["refine"], frame_ranked=True)
               for i, data in candidate_rows.items()}
        if geometry_audit:
            chosen = suppression["selected"]
            pool = suppress(raw, chosen["method"], chosen["strength"])
            rows = filtered(pool, chosen["threshold"], chosen["low"], confirmations=detector.get("track_support", 1))
            if summarize(rows, development)[0] != suppression["development"]:
                raise ValueError("Geometry audit failed to reproduce the archived operating point")
            report = audit(pool, rows, development)
            write_json(study / "geometry-audit.json", report)
            print(report, flush=True)
            return
        evidence = history_evidence(raw, neural, transforms)
        if name == "validation":
            smaller = sorted(raw)[:9]
            prefix = history_evidence({i: raw[i] for i in smaller}, {i: neural[i] for i in smaller}, transforms)
            for i in smaller:
                np.testing.assert_array_equal(prefix[i], evidence[i])
        scopes.append((raw, evidence))

    (cal, cal_evidence), (dev, dev_evidence) = scopes
    support = detector.get("track_support", 1)
    baseline, baseline_rows, baseline_coverage = score_candidate_predictions(dev, development, suppression["selected"], support)
    if baseline != suppression["development"]:
        raise ValueError("Unmodified candidate scores failed to reproduce the baseline")
    print("Baseline replay and historical feature prefix verified", flush=True)
    if matched_budget:
        budget_comparison(study, scopes, (calibration, development), suppression["selected"], support)
        return
    options = [fit_calibration(cal, cal_evidence, calibration, support, cue) for cue in range(len(METHODS))]
    results, rows = {}, []
    for name, point in zip(METHODS, options, strict=True):
        raw = fused(dev, dev_evidence, point["cue"], point["center"], point["spread"], point["weight"])
        result, frames, retained = score_candidate_predictions(raw, development, point, support)
        results[name] = dict(settings=point, development=result, post_suppression_matches=retained,
                             paired_vs_reference=paired_block_f1(frames, baseline_rows))
        rows.extend(dict(method=name, **row) for row in frames)
        print(name, result, retained, flush=True)

    best = max(results, key=lambda name: quality(results[name]["settings"]["calibration"]))
    report = dict(lags=LAGS, speeds=SPEEDS, directions=DIRECTIONS, prefix_verified=smaller,
                  matching_iou=0.5, before_suppression_matches=coverage(dev, development),
                  reference={'development': baseline, 'post_suppression_matches': baseline_coverage},
                  development_local_ranking=conditional_audit(dev, dev_evidence, development),
                  selected_by_calibration=best, results=results)
    write_json(study / "comparison.json", report)
    write_csv(study / "development.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--geometry-audit", action="store_true", help="Replay the fixed baseline for geometry diagnostics")
    modes.add_argument("--matched-budget", action="store_true", help="Compare ranking with the original per-frame output count")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    run(args.output.resolve(), args.ffmpeg, geometry_audit=args.geometry_audit, matched_budget=args.matched_budget)
