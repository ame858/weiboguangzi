"""Train a candidate ranker on later early footage scored by a truly earlier model."""
import argparse
from pathlib import Path

import numpy as np

from experiment import (candidate_file, diagnostic_coverage, filtered, load_candidates, raw_candidates,
                        select_operation, summarize, with_nms)
from snl_fusion import fit, make_records, score
from snl_inputs import DEFAULT_OUTPUT, PROFILES, feature_cache, load_gray, load_labels, read_json, sha_file, write_json
from videosar.snl_causal import CausalFeatures, box_iou


def blend_scores(rank, base, weight):
    boxes, confidence = rank
    if weight == 1:
        return boxes, confidence
    original_boxes, original_scores = base
    overlap = box_iou(boxes, original_boxes)
    support = (overlap >= 0.50) * original_scores[None]
    center_score = support.max(axis=1, initial=0)
    confidence = np.power(confidence.clip(0.001, 1), weight) * np.power(center_score.clip(0.001, 1), 1 - weight)
    return boxes, confidence.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=DEFAULT_OUTPUT / "early-stacked")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    output, previous = args.output.resolve(), args.previous.resolve()
    early, _, _ = PROFILES["original"]
    train_ids, cal_ids, dev_ids = PROFILES["prospective"]
    fit_ids = tuple(index for index in train_ids if index >= 242)
    if read_json(previous / "split-protocol.json")["train"] != list(early) or max(early) >= min(fit_ids):
        raise ValueError("Ranking examples must follow the source detector's training segment")
    registry = read_json(previous / "checkpoints-center-stacked.json")
    starter = next(config for config in registry if config["epoch"] == 36)
    current = read_json(args.decision)["config"]
    if (current.get("input_profile") != "stacked" or current["architecture"] != "radar-center"
            or current.get("dense_offsets") or current.get("tiled") or current.get("flip_inference")):
        raise ValueError("The paired ranker requires comparable untiled center detectors")
    if gain := current.get("contrast_gain"):
        starter = starter | {"contrast_gain": gain}
    gray = load_gray(output, args.ffmpeg)
    fit_labels, _, _ = load_labels(output, fit_ids, gray)
    validation, _, _ = load_labels(output, cal_ids, gray)
    development, _, _ = load_labels(output, dev_ids, gray)
    stacked = feature_cache(output, gray, max(dev_ids), CausalFeatures("stacked"))
    statistics = feature_cache(output, gray, max(dev_ids), CausalFeatures())
    cached = candidate_file(previous, starter, "forward-rank-fit", starter.get("device"))
    if cached.is_file() and read_json(cached)["ids"] == list(fit_ids):
        train_candidates = load_candidates(previous, starter, "forward-rank-fit")
    else:
        train_candidates = raw_candidates(previous, starter, stacked, fit_ids, "forward-rank-fit")
    _, samples = make_records([train_candidates], fit_labels, statistics)
    estimators = fit(samples, fit_ids)
    raw_cal = load_candidates(output, current, "validation")
    raw_dev = load_candidates(output, current, "development")
    cal, _ = make_records([raw_cal], {}, statistics)
    dev, _ = make_records([raw_dev], {}, statistics)
    comparisons = []
    for refine in (False, True):
        cal_rank = {index: score(estimators, record, refine) for index, record in cal.items()}
        dev_rank = {index: score(estimators, record, refine) for index, record in dev.items()}
        for weight in (1.0, 0.25, 0.50, 0.75):
            refitted = {i: blend_scores(rank, raw_cal[i], weight) for i, rank in cal_rank.items()}
            operation, calibration, _ = select_operation(refitted, validation)
            ranked = {i: blend_scores(rank, raw_dev[i], weight) for i, rank in dev_rank.items()}
            pooled = with_nms(ranked, operation["nms"])
            predictions = filtered(pooled, operation["threshold"], operation["low"])
            evaluation, _ = summarize(predictions, development)
            coverage, centers = diagnostic_coverage(pooled, development)
            row = {"refine_boxes": refine, "ranker_weight": weight, "operation": operation, "validation": calibration,
                   "development": evaluation, "development_recall_at_60": coverage,
                   "development_center_coverage_at_60": centers}
            comparisons.append(row)
            print(row, flush=True)
    paths = []
    for index, estimator in enumerate((estimators[0], *estimators[1])):
        path = output / f"forward-rank-{index}.txt"
        estimator.save_model(str(path))
        paths.append(path)
    write_json(output / "forward-ranking-comparison.json",
               {"first_detector": starter, "ranking_fit_ids": list(fit_ids), "current_detector": current,
                "estimators": {path.name: sha_file(path) for path in paths}, "comparisons": comparisons})


if __name__ == "__main__":
    main()
