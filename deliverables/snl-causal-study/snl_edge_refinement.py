"""Measure and test bounded, image-driven SNL box geometry changes."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np

from audit_detection import count_pairs, paired_block_f1
from experiment import filter_by_config, load_candidates, quality, select_operation, summarize, with_nms
from snl_bytetrack import REFERENCE
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, WIDTH, load_gray, load_labels, read_json, write_csv, write_json
from videosar.snl_causal import box_iou


SHIFTS = (-2, 0, 2)
SCALES = (0.8, 1.0, 1.2)
MOVES = np.array(list(product(SHIFTS, SHIFTS, SCALES, SCALES)), dtype=np.float32)
BASE = int(np.flatnonzero(np.all(MOVES == (0, 0, 1, 1), axis=1))[0])
MODES = {"original": (MOVES[:, :2] == 0).all(axis=1) & (MOVES[:, 2:] == 1).all(axis=1),
         "shifts": (MOVES[:, 2:] == 1).all(axis=1), "scales": (MOVES[:, :2] == 0).all(axis=1),
         "joint": np.ones(len(MOVES), dtype=bool)}


def geometry_options(boxes):
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    half_lengths = (boxes[:, 2:] - boxes[:, :2]) * 0.5
    centers = centers[:, None] + MOVES[None, :, :2]
    half_lengths = half_lengths[:, None] * MOVES[None, :, 2:]
    options = np.concatenate((centers - half_lengths, centers + half_lengths), axis=2)
    options = options.clip(0, (WIDTH, HEIGHT, WIDTH, HEIGHT)).astype(np.float32)
    options[:, BASE] = boxes
    return options


def rectangle_mean(sums, x0, y0, x1, y1):
    area = (x1 - x0) * (y1 - y0)
    value = sums[y1, x1] - sums[y0, x1] - sums[y1, x0] + sums[y0, x0]
    return np.divide(value, area, out=np.zeros_like(value), where=area != 0)


def edge_strength(gray, boxes):
    sums = np.pad(gray.astype(np.float64).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    x0, y0 = np.floor(boxes[..., :2]).astype(int).transpose(2, 0, 1)
    x1, y1 = np.ceil(boxes[..., 2:]).astype(int).transpose(2, 0, 1)
    left, right = np.maximum(x0 - 2, 0), np.minimum(x1 + 2, WIDTH)
    top, bottom = np.maximum(y0 - 2, 0), np.minimum(y1 + 2, HEIGHT)
    inner_left, inner_right = np.minimum(x0 + 2, x1), np.maximum(x1 - 2, x0)
    inner_top, inner_bottom = np.minimum(y0 + 2, y1), np.maximum(y1 - 2, y0)
    contrasts = np.stack([
        rectangle_mean(sums, left, y0, x0, y1) - rectangle_mean(sums, x0, y0, inner_left, y1),
        rectangle_mean(sums, x1, y0, right, y1) - rectangle_mean(sums, inner_right, y0, x1, y1),
        rectangle_mean(sums, x0, top, x1, y0) - rectangle_mean(sums, x0, y0, x1, inner_top),
        rectangle_mean(sums, x0, y1, x1, bottom) - rectangle_mean(sums, x0, inner_bottom, x1, y1),
    ])
    strength = (0.75 * contrasts.mean(axis=0) + 0.25 * contrasts.min(axis=0)) / 255
    return np.where((x1 > x0) & (y1 > y0), strength, -np.inf)


def measures(raw, options, truth, limits=(15, 60, 256)):
    table = {limit: {mode: 0 for mode in MODES} for limit in limits}
    for index, boxes in truth.items():
        order = np.argsort(-raw[index][1], kind="stable")
        geometries = options[index][order]
        observed = box_iou(geometries.reshape(-1, 4), boxes).reshape(len(geometries), len(MOVES), len(boxes))
        for mode, moves in MODES.items():
            can_match = (observed[:, moves] >= 0.50).any(axis=1)
            for limit, tally in table.items():
                tally[mode] += count_pairs(can_match[:limit])
    return table


def refine(raw, options, evidence, penalty):
    if penalty is None:
        return raw
    shift_cost = np.square(MOVES[:, :2] / 2).sum(axis=1)
    scale_cost = np.square((MOVES[:, 2:] - 1) / 0.2).sum(axis=1)
    cost = penalty * (shift_cost + scale_cost)
    predictions = {}
    for index, (boxes, score) in raw.items():
        utility = evidence[index] - cost
        choice = utility.argmax(axis=1)
        improves = utility[np.arange(len(boxes)), choice] > utility[:, BASE]
        refined = options[index][np.arange(len(boxes)), np.where(improves, choice, BASE)]
        refined[~improves] = boxes[~improves]
        predictions[index] = refined, score
    return predictions


def study(output, ffmpeg):
    gray = load_gray(output, ffmpeg)
    _, cal_ids, dev_ids = PROFILES["prospective"]
    calibration, development = (load_labels(output, ids, gray)[0] for ids in (cal_ids, dev_ids))
    reference = read_json(output / REFERENCE)
    config = reference["config"]
    cal_raw, dev_raw = (load_candidates(output, config, scope) for scope in ("validation", "development"))
    raw = cal_raw | dev_raw
    options = {index: geometry_options(boxes) for index, (boxes, _) in raw.items()}
    evidence = {index: edge_strength(gray[index - 2], boxes) for index, boxes in options.items()}
    baseline, baseline_rows = summarize(filter_by_config(with_nms(dev_raw, config["nms"]), config), development)
    if baseline != reference["development"]:
        raise ValueError("The candidate cache no longer reproduces the reference detector")

    trials = []
    for penalty in (None, 0.003, 0.008, 0.016):
        recentered = refine(cal_raw, options, evidence, penalty)
        point, fitted, _ = select_operation(recentered, calibration, confirmations=config.get("track_support", 1))
        trials.append({"edge_penalty": penalty, "operation": point, "calibration": fitted})
    selected = max(trials, key=lambda row: quality(row["calibration"]))
    decision = config | selected["operation"]
    adjusted = refine(dev_raw, options, evidence, selected["edge_penalty"])
    found, records = summarize(filter_by_config(with_nms(adjusted, decision["nms"]), decision), development)
    report = {"source": config, "pixel_shifts_each_axis": SHIFTS, "length_scales_each_axis": SCALES,
              "truth_boxes": sum(map(len, development.values())), "match_iou": 0.50,
              "development_oracle_matches_by_candidate_budget": measures(dev_raw, options, development),
              "calibration_trials": trials, "selected": selected, "development": found,
              "reference_development": baseline, "f1_paired_difference": paired_block_f1(records, baseline_rows)}
    write_json(output / "edge-refinement-comparison.json", report)
    write_csv(output / "edge-refinement-development.csv", records)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    result = study(args.output.resolve(), args.ffmpeg)
    for key in ("development_oracle_matches_by_candidate_budget", "selected", "development", "f1_paired_difference"):
        print(key, result[key], flush=True)


if __name__ == "__main__":
    main()
