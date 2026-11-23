"""Fit a dense local-shadow candidate scorer on the open SNL training frames."""
import argparse
from itertools import product
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np

from audit_detection import paired_block_f1
from experiment import filter_by_config, load_candidates, quality, select_operation, summarize, warm_ids, with_nms
from snl_bytetrack import REFERENCE
from snl_fusion import measurement, score
from snl_inputs import DEFAULT_OUTPUT, PROFILES, load_gray, load_labels, read_json, sha_file, write_csv, write_json
from snl_templates import SIZES
from videosar.snl_causal import box_iou, detection_counts


PER_SIZE = 160
MODES = ("scored_anchors", "refined_anchors")
PEAK_WINDOW = np.ones((5, 5), np.uint8)


def top_peaks(field, width, height):
    maxima = cv2.dilate(field, PEAK_WINDOW)
    margin_x, margin_y = width // 2 + 1, height // 2 + 1
    interior = (slice(margin_y, -margin_y), slice(margin_x, -margin_x))
    y, x = np.nonzero((field[interior] == maxima[interior]) & (field[interior] > 0))
    y, x = y + margin_y, x + margin_x
    score = field[y, x]
    if len(score) > PER_SIZE:
        top = np.argpartition(score, -PER_SIZE)[-PER_SIZE:]
        y, x, score = y[top], x[top], score[top]
    return x, y, score


def proposals(frame):
    """Retain multiple shapes per peak and score local darkness before any ranking."""
    local, residual = (frame[i].astype(np.float32) / 127.5 - 1 for i in (1, 4))
    candidates, cues = [], []
    for width, height in SIZES:
        inner = cv2.blur(local, (width, height))
        outer = cv2.blur(local, (width + 8, height + 8))
        motion = cv2.blur(residual, (width, height))
        contrast = inner - 0.5 * outer
        for field in (contrast, contrast + 0.40 * motion):
            x, y, _ = top_peaks(field, width, height)
            centers = np.column_stack((x, y)).astype(np.float32) + 0.5
            half = np.array([width, height], dtype=np.float32) * 0.5
            candidates.append(np.column_stack((centers - half, centers + half)))
            cues.append(np.column_stack((inner[y, x], outer[y, x], motion[y, x])))
    boxes, unique = np.unique(np.concatenate(candidates), axis=0, return_index=True)
    return boxes, np.concatenate(cues)[unique]


def samples(frame):
    boxes, cues = proposals(frame)
    features = np.column_stack((measurement(frame, boxes), cues))
    return boxes, features


def supervision(record, truth):
    boxes, data = record
    overlaps = box_iou(boxes, truth)
    high = overlaps.max(axis=1, initial=0)
    nearest = truth[overlaps.argmax(axis=1)] if len(truth) else np.zeros_like(boxes)
    return data, boxes, high, nearest


def fit_anchors(samples):
    data, boxes, overlap, nearest = (np.concatenate(parts) for parts in zip(*samples.values(), strict=True))
    keep = overlap >= 0.22
    positive = overlap >= 0.50
    options = dict(num_leaves=15, max_depth=5, learning_rate=0.05, min_data_in_leaf=22,
                   seed=331, num_threads=5, verbosity=-1, force_col_wise=True)
    classifier = lgb.train(dict(options, objective="binary"),
                           lgb.Dataset(data, label=positive, weight=np.where(positive, 7, 1)), num_boost_round=120)
    ratios = np.tile(np.maximum(3, boxes[keep, 2:] - boxes[keep, :2]), (1, 2))
    offsets = ((nearest[keep] - boxes[keep]) / ratios).clip(-0.5, 0.5)
    data, weights = data[keep], np.sqrt(overlap[keep])
    regressors = [lgb.train(dict(options, objective="regression"), lgb.Dataset(data, label=target, weight=weights),
                            num_boost_round=120) for target in offsets.T]
    return classifier, regressors


def coverage(raw, targets, neural=None):
    total = 0
    for index, truth in targets.items():
        boxes = raw[index][0]
        if neural is not None:
            boxes = np.concatenate((boxes, neural[index][0]))
        total += detection_counts(boxes, truth)[0]
    return total


def evaluated(raw, operation, targets):
    nms = with_nms(raw, operation["nms"])
    predicted = filter_by_config(nms, operation)
    return summarize(predicted, targets)


def agreement_features(neural, local):
    result = {}
    for index, (boxes, scores) in neural.items():
        local_boxes, local_scores = local[index]
        order = np.argsort(-local_scores, kind="stable")[:1024]
        local_boxes, local_scores = local_boxes[order], local_scores[order]
        shared = box_iou(boxes, local_boxes) >= 0.30
        local_support = (shared * local_scores[None]).max(axis=1, initial=0)
        neural_support = (shared * scores[:, None]).max(axis=0, initial=0)
        result[index] = boxes, scores, local_support, local_boxes, local_scores, neural_support
    return result


def combine_agreement(features, neural_floor, local_floor, boost, weight):
    def normalize(scores, floor):
        return scores / (scores + floor)

    result = {}
    for index, (a, scores, support, b, local_scores, local_support) in features.items():
        primary = normalize(scores, neural_floor)
        first = primary + boost * normalize(support, local_floor) * (1 - primary)
        if weight:
            secondary = normalize(local_scores, local_floor / weight)
            second = secondary + boost * normalize(local_support, neural_floor) * (1 - secondary)
            result[index] = np.concatenate((a, b)), np.concatenate((first, second))
        else:
            result[index] = a, first
    return result


def fusion_study(output, ffmpeg):
    study = output / "local-shadow-proposals"
    result = read_json(study / "comparison.json")
    model = [lgb.Booster(model_file=str(study / file)) for file in ("ranker.txt", "x0.txt", "y0.txt", "x1.txt", "y1.txt")]
    if any(sha_file(study / file) != value for file, value in result["weights"].items()):
        raise ValueError("The proposal scorer weights differ from the archived run")
    ranker = model[0], model[1:]
    mode = result["selected_by_calibration"]
    refine = mode == "refined_anchors"
    local_floor = result["results"][mode]["settings"]["threshold"]
    features = np.load(output / "motion-integration/motion-integration.npy", mmap_mode="r", allow_pickle=False)
    gray = load_gray(output, ffmpeg)
    _, cal_ids, dev_ids = PROFILES["prospective"]
    cal, dev = (load_labels(output, ids, gray)[0] for ids in (cal_ids, dev_ids))
    release = read_json(output / REFERENCE)["config"]
    neural = release | {"dense_candidate_limit": 1024}
    neural_cal, neural_dev = (load_candidates(output, neural, scope) for scope in ("validation", "development"))
    local_cal, local_dev = ({index: score(ranker, samples(features[index - 2]), refine) for index in anchors}
                            for anchors in (neural_cal, neural_dev))
    op, val_score, _ = select_operation(neural_cal, cal, confirmations=release.get("track_support", 1))
    original, original_rows = evaluated(neural_dev, op | {"track_support": release.get("track_support", 1)}, dev)
    agreement = agreement_features(neural_cal, local_cal)
    trials = []
    for boost, weight in product((0.0, 0.20, 0.40), (0.0, 0.5, 1.0)):
        if not boost and not weight:
            continue
        fused = combine_agreement(agreement, op["threshold"], local_floor, boost, weight)
        operation, metric, _ = select_operation(fused, cal, confirmations=release.get("track_support", 1))
        trials.append({"boost": boost, "local_weight": weight, "operation": operation, "calibration": metric})
    best = max(trials, key=lambda row: quality(row["calibration"]))
    fused = combine_agreement(agreement_features(neural_dev, local_dev), op["threshold"], local_floor,
                              best["boost"], best["local_weight"])
    metrics, rows = evaluated(fused, best["operation"] | {"track_support": release.get("track_support", 1)}, dev)
    caps = {"neural_only": coverage(neural_dev, dev), "local_only": coverage(local_dev, dev),
            "all_candidates_union": coverage(neural_dev, dev, local_dev), "fusion_pool": coverage(fused, dev)}
    report = {"local_mode": mode, "dense_neural_config": neural, "local_threshold": local_floor,
              "baseline_calibration": val_score, "baseline_development": original,
              "calibration_trials": trials, "selected": best, "development": metrics, "coverage": caps,
              "paired_f1_vs_dense_baseline": paired_block_f1(rows, original_rows)}
    write_json(study / "dense-neural-fusion.json", report)
    write_csv(study / "dense-neural-fusion-development.csv", rows)
    print("Dense candidate fusion", best, metrics, caps, flush=True)


def evaluate(output, ffmpeg):
    features_dir = output / "motion-integration"
    features = np.load(features_dir / "motion-integration.npy", mmap_mode="r", allow_pickle=False)
    train_ids, cal_ids, dev_ids = PROFILES["prospective"]
    fit_ids = train_ids[::2]
    gray = load_gray(output, ffmpeg)
    training, calibration, development = (load_labels(output, ids, gray)[0] for ids in (fit_ids, cal_ids, dev_ids))
    fit_samples = {}
    for k, (index, truth) in enumerate(training.items(), 1):
        fit_samples[index] = supervision(samples(features[index - 2]), truth)
        if k % 30 == 0:
            print(f"Prepared anchors on {k}/{len(training)} training frames", flush=True)
    supervised_count = sum(len(entry[2]) for entry in fit_samples.values())
    positive_count = sum(int((entry[2] >= 0.5).sum()) for entry in fit_samples.values())
    if not 0 < positive_count < supervised_count:
        raise ValueError("Anchor fitting requires both positive and negative examples")
    print(f"Training anchor scores from {supervised_count} boxes, {positive_count} positive", flush=True)
    ranker = fit_anchors(fit_samples)
    del fit_samples
    study = output / "local-shadow-proposals"
    study.mkdir(exist_ok=True)
    weights = {}
    for name, estimator in zip(("ranker", "x0", "y0", "x1", "y1"), (ranker[0], *ranker[1]), strict=True):
        path = study / f"{name}.txt"
        estimator.save_model(str(path))
        weights[path.name] = sha_file(path)

    reference = read_json(output / REFERENCE)
    release = reference["config"]
    cached = load_candidates(output, release, "development")
    baseline, baseline_rows = evaluated(cached, release, development)
    if baseline != reference["development"]:
        raise ValueError("The cached reference predictions do not reproduce the baseline")
    cal = {index: samples(features[index - 2]) for index in warm_ids(calibration)}
    val_raw = {mode: {index: score(ranker, record, j == 1) for index, record in cal.items()} for j, mode in enumerate(MODES)}
    operations = {mode: select_operation(raw, calibration) for mode, raw in val_raw.items()}
    del cal
    dev = {index: samples(features[index - 2]) for index in warm_ids(development)}
    selected = max(MODES, key=lambda mode: quality(operations[mode][1]))
    results, frame_rows = {}, []
    for j, mode in enumerate(MODES):
        operation, validation, _ = operations[mode]
        raw = {index: score(ranker, record, j == 1) for index, record in dev.items()}
        metrics, rows = evaluated(raw, operation, development)
        results[mode] = {"settings": operation, "calibration": validation, "development": metrics,
                         "candidate_count": sum(len(raw[i][0]) for i in development),
                         "unfiltered_matches": coverage(raw, development),
                         "unfiltered_with_reference_matches": coverage(raw, development, cached),
                         "paired_f1_vs_reference": paired_block_f1(rows, baseline_rows)}
        frame_rows.extend(dict(method=mode, **row) for row in rows)
        print(mode, results[mode], flush=True)
    write_json(study / "comparison.json", {"features": read_json(features_dir / "features.json"), "weights": weights,
               "fit_ids": fit_ids, "supervised_anchors": supervised_count, "positive_anchors": positive_count,
               "per_size_per_cue": PER_SIZE, "anchor_shapes_wh": SIZES, "reference": baseline,
               "selected_by_calibration": selected, "results": results})
    write_csv(study / "development.csv", frame_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--fusion", action="store_true", help="Evaluate a saved scorer with dense neural candidates")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    experiment = fusion_study if args.fusion else evaluate
    experiment(args.output.resolve(), args.ffmpeg)
