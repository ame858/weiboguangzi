"""Compare cross-detector support on SNL dense candidates using forward folds."""
import argparse
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from audit_detection import paired_block_f1
from experiment import CONFIDENCES, NMS, filtered, quality, raw_candidates, summarize, with_nms
from snl_independent_review import REPLAY, load_predictions
from snl_inputs import DEFAULT_OUTPUT, PROFILES, VIDEO, feature_cache, load_gray, load_labels, read_json, sha_file
from snl_inputs import write_csv, write_json
from snl_protocol import passed
from snl_rolling_nwd import ARMS, FOLDS, center_metrics, verify_prefix
from videosar.snl_causal import CausalFeatures, box_iou


EPOCH = 24
DENSE_LIMIT = 1024
SUPPORT_RADIUS = 5.0
SUPPORT_IOU = 0.35
SUPPORT_WEIGHTS = (0.5, 1.0, 2.0)
SUPPLEMENT_LIMITS = (1, 2, 4)
SUPPLEMENT_METHODS = ("single_dense", "dense_union", "overlap_consensus")
METHODS = {"single_peaks": (0.0,), "single_dense": (0.0,), "dense_union": (0.0,),
           "center_consensus": SUPPORT_WEIGHTS, "overlap_consensus": SUPPORT_WEIGHTS}


def cross_support(observations):
    """Collect at most one score per other detector for each candidate and support rule."""
    trees = [cKDTree((boxes[:, :2] + boxes[:, 2:]) / 2) for boxes, _ in observations]
    areas = [(boxes[:, 2:] - boxes[:, :2]).prod(axis=1) for boxes, _ in observations]
    totals = [np.zeros((2, len(boxes)), dtype=np.float32) for boxes, _ in observations]
    for a, b in combinations(range(len(observations)), 2):
        distances = trees[a].sparse_distance_matrix(trees[b], SUPPORT_RADIUS, output_type="ndarray")
        i, j, distance = distances["i"], distances["j"], distances["v"]
        boxes_a, scores_a = observations[a]
        boxes_b, scores_b = observations[b]
        low = np.maximum(boxes_a[i, :2], boxes_b[j, :2])
        high = np.minimum(boxes_a[i, 2:], boxes_b[j, 2:])
        intersection = np.maximum(high - low, 0).prod(axis=1)
        iou = intersection / (areas[a][i] + areas[b][j] - intersection)
        similarity = (np.exp(-0.5 * (distance / (SUPPORT_RADIUS / 2)) ** 2),
                      np.where(iou >= SUPPORT_IOU, iou, 0))
        for row, agreement in enumerate(similarity):
            vote_a, vote_b = np.zeros_like(scores_a), np.zeros_like(scores_b)
            np.maximum.at(vote_a, i, agreement * scores_b[j])
            np.maximum.at(vote_b, j, agreement * scores_a[i])
            totals[a][row] += vote_a
            totals[b][row] += vote_b
    return np.concatenate(totals, axis=1) / (len(observations) - 1)


def describe(dense):
    rows = {}
    for index in dense["raw_giou"]:
        observations = [dense[arm][index] for arm in ARMS]
        boxes, scores = (np.concatenate(part) for part in zip(*observations, strict=True))
        center, overlap = cross_support(observations)
        rows[index] = boxes, scores, center, overlap
    return rows


def candidates(method, weight, peaks, dense, rows):
    if method == "single_peaks":
        return peaks
    if method == "single_dense":
        return dense["raw_giou"]
    if method == "dense_union":
        return {i: (boxes, own) for i, (boxes, own, _, _) in rows.items()}
    support_index = {"center_consensus": 2, "overlap_consensus": 3}[method]
    return {i: (row[0], (row[1] + weight * row[support_index]) / (1 + weight)) for i, row in rows.items()}


def select(method, peaks, dense, rows, labels):
    trials = []
    for weight, nms in product(METHODS[method], NMS):
        pool = with_nms(candidates(method, weight, peaks, dense, rows), nms)
        scored = ((float(t), summarize(filtered(pool, float(t), None), labels)[0]) for t in CONFIDENCES)
        threshold, metrics = max(scored, key=lambda row: quality(row[1]))
        trials.append(dict(weight=weight, nms=nms, threshold=threshold, metrics=metrics))
    point = max(trials, key=lambda row: quality(row["metrics"]))
    pool = with_nms(candidates(method, point["weight"], peaks, dense, rows), point["nms"])
    predictions = filtered(pool, point["threshold"], None)
    return point, trials, predictions


def budgeted(pool, release):
    kept = {}
    for index, (boxes, scores) in pool.items():
        count = len(release[index][0])
        if len(boxes) < count:
            raise ValueError(f"Frame {index} needs {count} candidates for the same-count comparison")
        take = np.argsort(-scores, kind="stable")[:count]
        kept[index] = boxes[take], scores[take], None
    return kept


def measure(predictions, labels):
    iou, rows = summarize(predictions, labels)
    return dict(iou50=iou, center5=center_metrics(predictions, labels)), rows


def supplement_pool(raw, baseline):
    remaining = {}
    for i, (boxes, scores) in raw.items():
        reference = baseline[i][0]
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        reference_centers = (reference[:, :2] + reference[:, 2:]) / 2
        delta = centers[:, None] - reference_centers
        separate = (np.square(delta).sum(axis=2) > SUPPORT_RADIUS**2).all(axis=1)
        separate &= (box_iou(boxes, reference) < SUPPORT_IOU).all(axis=1)
        remaining[i] = boxes[separate], scores[separate]
    return with_nms(remaining, SUPPORT_IOU)


def append_boxes(baseline, extra):
    return {i: (np.concatenate((baseline[i][0], boxes)), None, None) for i, boxes in extra.items()}


def supplement(baseline, pool, point):
    extra = {i: boxes[scores >= point["threshold"]][:point["per_frame_limit"]] for i, (boxes, scores) in pool.items()}
    return append_boxes(baseline, extra)


def select_supplement(baseline, pool, labels):
    initial, _ = summarize(baseline, labels)
    trials = [dict(per_frame_limit=0, threshold=1.0, metrics=initial)]
    for limit, threshold in product(SUPPLEMENT_LIMITS, CONFIDENCES):
        point = dict(per_frame_limit=limit, threshold=float(threshold))
        metrics, _ = summarize(supplement(baseline, pool, point), labels)
        trials.append(point | {"metrics": metrics})
    return dict(point=max(trials, key=lambda row: quality(row["metrics"])), trials=trials)


def get_candidates(rolling, scope, ids, stacks, archived, train_ids, fold=None):
    dense, models = {}, {}
    for arm, (profile, _) in ARMS.items():
        folder = rolling / arm / ("refit" if fold is None else f"fold-{fold}")
        receipt = read_json(folder / "checkpoints.json")
        if receipt["recipe"]["fit_ids"] != list(train_ids):
            raise ValueError(f"Unexpected training frame IDs for {folder}")
        config = next(row for row in receipt["checkpoints"] if row["epoch"] == EPOCH)
        if fold is None and config["sha256"] != archived[arm]["sha256"]:
            raise ValueError(f"Refit checkpoint differs from the rolling archive for {arm}")
        if sha_file(folder / config["path"]) != config["sha256"]:
            raise ValueError(f"Checkpoint hash changed for {folder}")
        stack = stacks[profile]
        dense[arm] = raw_candidates(folder, config | {"dense_candidate_limit": DENSE_LIMIT}, stack, list(ids), scope)
        models[arm] = config["sha256"]
        if arm == "raw_giou":
            peaks = raw_candidates(folder, config, stack, list(ids), scope)
    return peaks, dense, models


def expanded_coverage(output, study, dense, labels):
    from experiment import load_candidates
    from snl_bytetrack import REFERENCE
    from snl_local_shadow_proposals import proposals
    from videosar.snl_causal import detection_counts

    config = read_json(output / REFERENCE)["config"] | {"dense_candidate_limit": 4096}
    archived = load_candidates(output, config, "development")
    motion = np.load(output / "motion-integration/motion-integration.npy", mmap_mode="r", allow_pickle=False)
    rows = []
    for i, truth in labels.items():
        local = proposals(motion[i - 2])[0]
        original = np.concatenate((archived[i][0], local))
        expanded = np.concatenate((original, *(source[i][0] for source in dense.values())))
        rows.append(dict(annotation_id=i, targets=len(truth), archived_pool=len(original), expanded_pool=len(expanded),
                         archived_matches=detection_counts(original, truth)[0], expanded_matches=detection_counts(expanded, truth)[0]))
    totals = {key: sum(row[key] for row in rows) for key in rows[0] if key != "annotation_id"}
    audit = dict(match_iou=0.50, ground_truth_assisted=True, totals=totals, rows=rows)
    write_json(study / "expanded-coverage-audit.json", audit)
    print("expanded coverage", totals, flush=True)


def run(output, ffmpeg, replay, *, audit_expanded=False):
    rolling = output / "rolling-nwd"
    selection = read_json(rolling / "rolling-selection.json")
    archived = read_json(rolling / "frozen-refit.json")["configs"]
    train_ids, cal_ids, dev_ids = PROFILES["prospective"]
    source_hash = sha_file(VIDEO)
    if (selection["settings"]["video_sha256"] != source_hash or
            selection["settings"]["folds"] != [list(row) for row in FOLDS] or
            any(not train_ids[0] <= cutoff < first <= last <= train_ids[-1] for cutoff, first, last in FOLDS) or
            not train_ids[-1] < cal_ids[0] < cal_ids[-1] < dev_ids[0]):
        raise ValueError("The rolling archive or time boundaries differ from the protocol")
    study = output / "candidate-consensus"
    study.mkdir(exist_ok=True)
    gray = load_gray(output, ffmpeg)
    cv_ids = [i for _, first, last in FOLDS for i in range(first, last + 1)]
    labels = load_labels(output, cv_ids, gray)[0]
    profiles = sorted({p for p, _ in ARMS.values()})
    stacks = {p: feature_cache(rolling, gray, max(cv_ids), CausalFeatures(p)) for p in profiles}
    peaks, dense, fold_weights = {}, {arm: {} for arm in ARMS}, []
    for fold, (cutoff, first, last) in enumerate(FOLDS, start=1):
        ids = range(first, last + 1)
        fit_ids = [i for i in train_ids if i <= cutoff]
        block_peaks, block_dense, weights = get_candidates(rolling, "forward", ids, stacks, archived, fit_ids, fold=fold)
        peaks.update(block_peaks)
        for arm in ARMS:
            dense[arm].update(block_dense[arm])
        fold_weights.append(weights)
    rows = describe(dense)
    selections = {}
    for method in METHODS:
        point, trials, predictions = select(method, peaks, dense, rows, labels)
        folds = [summarize(predictions, {i: labels[i] for i in range(first, last + 1)})[0] for _, first, last in FOLDS]
        selections[method] = dict(point=point, trials=trials, forward_blocks=folds)
        print("forward", method, point, flush=True)
    chosen = max(selections, key=lambda method: quality(selections[method]["point"]["metrics"]))
    peak_point = selections["single_peaks"]["point"]
    baseline = filtered(with_nms(peaks, peak_point["nms"]), peak_point["threshold"], None)
    supplement_selections = {}
    for method in SUPPLEMENT_METHODS:
        raw = candidates(method, selections[method]["point"]["weight"], peaks, dense, rows)
        supplement_selections[method] = select_supplement(baseline, supplement_pool(raw, baseline), labels)
        print("forward supplement", method, supplement_selections[method]["point"], flush=True)
    chosen_supplement = max(supplement_selections, key=lambda method: quality(supplement_selections[method]["point"]["metrics"]))
    freeze = dict(selected=chosen, source_sha256=source_hash, script_sha256=sha_file(Path(__file__)),
                  folds=FOLDS, fold_weights=fold_weights, cv_frames=len(labels), epoch=EPOCH, dense_limit=DENSE_LIMIT,
                  match_iou=0.50, support_radius_px=SUPPORT_RADIUS, support_iou=SUPPORT_IOU, methods=selections,
                  supplement_methods=supplement_selections, selected_supplement=chosen_supplement,
                  refit_weights={arm: archived[arm]["sha256"] for arm in ARMS})
    write_json(study / "frozen-selection.json", freeze)

    stacks = {p: feature_cache(rolling, gray, max(dev_ids), CausalFeatures(p)) for p in profiles}
    for profile, stack in stacks.items():
        verify_prefix(gray, stack, profile, min(cal_ids))
    holdouts, supplement_holdouts, all_rows = {}, {}, []
    for scope, ids in (("calibration_audit", cal_ids), ("development", dev_ids)):
        peaks, dense, _ = get_candidates(rolling, scope, ids, stacks, archived, train_ids)
        rows = describe(dense)
        labels = load_labels(output, ids, gray)[0]
        release_boxes, release_hash = load_predictions(replay, source_hash, [{"source_frame_zero_based": i + 148} for i in ids])
        release = {i: (release_boxes[i + 148], None, None) for i in ids}
        release_metrics, release_rows = measure(release, labels)
        scores = {"delivery": release_metrics}
        supplemental, supplement_pools, supplement_counts = {}, {}, {}
        all_rows.extend(dict(split=scope, method="delivery", budget="frozen", **row) for row in release_rows)
        for method, decision in selections.items():
            point = decision["point"]
            raw = candidates(method, point["weight"], peaks, dense, rows)
            pool = with_nms(raw, point["nms"])
            metrics, predictions = measure(filtered(pool, point["threshold"], None), labels)
            fixed, fixed_rows = measure(budgeted(pool, release), labels)
            scores[method] = dict(frozen=metrics, equal_delivery_count=fixed, candidates_before_nms=summarize(
                {i: (box, None, None) for i, (box, _) in raw.items()}, labels)[0]["tp"],
                paired_frozen_vs_delivery=paired_block_f1(predictions, release_rows),
                paired_equal_count_vs_delivery=paired_block_f1(fixed_rows, release_rows))
            for budget, counts in (("frozen", predictions), ("equal_delivery_count", fixed_rows)):
                all_rows.extend(dict(split=scope, method=method, budget=budget, **row) for row in counts)
            print(scope, method, metrics["iou50"], "matched at delivery count", fixed["iou50"]["tp"], flush=True)
        for method, decision in supplement_selections.items():
            raw = candidates(method, selections[method]["point"]["weight"], peaks, dense, rows)
            supplement_pools[method] = supplement_pool(raw, release)
            predictions = supplement(release, supplement_pools[method], decision["point"])
            supplement_counts[method] = {i: len(predictions[i][0]) - len(release[i][0]) for i in ids}
            metrics, counts = measure(predictions, labels)
            extra_tp = metrics["iou50"]["tp"] - release_metrics["iou50"]["tp"]
            extra_fp = metrics["iou50"]["fp"] - release_metrics["iou50"]["fp"]
            supplemental[method] = dict(metrics=metrics, extra_tp=extra_tp, extra_fp=extra_fp,
                                        paired_vs_delivery=paired_block_f1(counts, release_rows))
            all_rows.extend(dict(split=scope, method=method, budget="supplement", **row) for row in counts)
            print(scope, "supplement", method, metrics["iou50"], "added TP/FP", extra_tp, extra_fp, flush=True)
        for method, other in (("dense_union", "overlap_consensus"), ("overlap_consensus", "dense_union")):
            caps = supplement_counts[other]
            extra = {i: supplement_pools[method][i][0][:caps[i]] for i in ids}
            if any(len(extra[i]) != caps[i] for i in ids):
                raise ValueError(f"{method} exhausted its complementary candidates at the {other} output count")
            metrics, counts = measure(append_boxes(release, extra), labels)
            supplemental[method]["at_other_addition_count"] = dict(reference=other, metrics=metrics)
            all_rows.extend(dict(split=scope, method=method, budget="equal_supplement_count", **row) for row in counts)
        if audit_expanded and scope == "development":
            expanded_coverage(output, study, dense, labels)
        holdouts[scope] = scores
        supplement_holdouts[scope] = supplemental
    gates = {name: all(passed(scores[name]["frozen"]["iou50"]) for scores in holdouts.values()) for name in METHODS}
    supplement_gates = {name: all(passed(scores[name]["metrics"]["iou50"]) for scores in supplement_holdouts.values())
                        for name in SUPPLEMENT_METHODS}
    write_json(study / "comparison.json", freeze | dict(delivery_model_sha256=release_hash, holdouts=holdouts,
                                                        gates=gates, supplement_holdouts=supplement_holdouts,
                                                        supplement_gates=supplement_gates))
    write_csv(study / "holdout-rows.csv", all_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--replay", type=Path, default=REPLAY)
    parser.add_argument("--audit-expanded-coverage", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.output.resolve(), args.ffmpeg, args.replay.resolve(), audit_expanded=args.audit_expanded_coverage)
