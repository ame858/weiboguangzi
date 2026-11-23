"""Test a calibration-only correction of predicted box dimensions."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from experiment import filter_by_config, load_candidates, quality, select_operation, summarize, with_nms
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, WIDTH, feature_cache, load_gray, load_labels, read_json, write_json
from videosar.snl_causal import CausalFeatures


def rescale_candidates(candidates, scale):
    factors = np.asarray(scale, dtype=np.float32)
    adjusted = {}
    for index, (boxes, scores) in candidates.items():
        half = 0.5 * (boxes[:, 2:] - boxes[:, :2])
        center = boxes[:, :2] + half
        resized = np.column_stack((center - half * factors, center + half * factors))
        adjusted[index] = resized.clip(0, [WIDTH, HEIGHT, WIDTH, HEIGHT]).astype(np.float32), scores
    return adjusted


def appearance_maps(gray, history, indices):
    maps = {}
    for index in indices:
        current = gray[index-2].astype(np.float32) / 255
        local = uniform_filter(current, size=(17, 13), mode="nearest")
        interior = uniform_filter(current, size=(7, 5), mode="nearest")
        residual = uniform_filter(history[index-2, 2].astype(np.float32) / 1020, size=3, mode="nearest")
        maps[index] = (local - interior, residual)
    return maps


def recenter(candidates, maps, penalty, motion_weight):
    if penalty is None:
        return candidates
    offsets = np.asarray(list(product(range(-3, 4), repeat=2)), dtype=int)
    price = penalty * np.square(offsets).sum(axis=1)
    result = {}
    for index, (boxes, scores) in candidates.items():
        spatial, residual = maps[index]
        evidence = spatial + motion_weight * residual
        anchor = np.floor((boxes[:, :2] + boxes[:, 2:]) * 0.5).astype(int)
        anchor = anchor.clip(0, [WIDTH - 1, HEIGHT - 1])
        coordinates = anchor[:, None] + offsets[None]
        x, y = coordinates.clip(0, [WIDTH - 1, HEIGHT - 1]).transpose(2, 0, 1)
        utility = evidence[y, x] - price
        best = utility.argmax(axis=1)
        improve = utility[np.arange(len(boxes)), best] - evidence[anchor[:, 1], anchor[:, 0]] >= 0.004
        shifts = offsets[best] * improve[:, None]
        moved = boxes + np.tile(shifts, (1, 2))
        moved = moved.clip(0, [WIDTH, HEIGHT, WIDTH, HEIGHT]).astype(np.float32)
        valid = np.all(moved[:, 2:] > moved[:, :2], axis=1)
        result[index] = moved[valid], scores[valid]
    return result


def compare_center_shifts(output, config, gray, calibration, development, cal_raw, dev_raw):
    history = feature_cache(output, gray, max(development), CausalFeatures())
    images = appearance_maps(gray, history, set(cal_raw) | set(dev_raw))
    trials = []
    settings = [(None, 0.0), *product((0.0015, 0.003, 0.006), (0.0, 0.25))]
    for penalty, weight in settings:
        recentered = recenter(cal_raw, images, penalty, weight)
        op, metrics, _ = select_operation(recentered, calibration, confirmations=config.get("track_support", 1))
        trials.append({"distance_penalty": penalty, "motion_weight": weight, "operation": op, "validation": metrics})
    chosen = max(trials, key=lambda value: quality(value["validation"]))
    scored = config | chosen["operation"]
    shifted = recenter(dev_raw, images, chosen["distance_penalty"], chosen["motion_weight"])
    estimates = filter_by_config(with_nms(shifted, scored["nms"]), scored)
    chosen["development"], _ = summarize(estimates, development)
    write_json(output / "image-center-calibration.json", {"source": config, "selected": chosen, "trials": trials})
    print(chosen, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--image-center", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    decision = read_json(args.decision) if args.decision else read_json(output / "development-status.json")["best"]
    config = decision["config"]
    gray = load_gray(output, args.ffmpeg)
    profile = read_json(output / "split-protocol.json")["profile"]
    _, calibration_ids, development_ids = PROFILES[profile]
    calibration, _, _ = load_labels(output, calibration_ids, gray)
    development, _, _ = load_labels(output, development_ids, gray)
    cal_raw = load_candidates(output, config, "validation")
    dev_raw = load_candidates(output, config, "development")
    if args.image_center:
        compare_center_shifts(output, config, gray, calibration, development, cal_raw, dev_raw)
        return
    trials = []
    for scale in product((0.85, 1.00, 1.15, 1.30), repeat=2):
        candidates = rescale_candidates(cal_raw, scale)
        operation, metrics, _ = select_operation(candidates, calibration, confirmations=config.get("track_support", 1))
        trials.append({"scale_xy": scale, "operation": operation, "validation": metrics})
    chosen = max(trials, key=lambda v: (quality(v["validation"]), -sum(abs(x - 1) for x in v["scale_xy"])))
    comparison = config | chosen["operation"]
    scaled = rescale_candidates(dev_raw, chosen["scale_xy"])
    predicted = filter_by_config(with_nms(scaled, comparison["nms"]), comparison)
    chosen["development"], _ = summarize(predicted, development)
    result = {"source": config, "selected": chosen, "baseline_development": decision["development"],
              "calibration_trials": trials}
    write_json(output / "box-dimension-calibration.json", result)
    print(chosen, flush=True)


if __name__ == "__main__":
    main()
