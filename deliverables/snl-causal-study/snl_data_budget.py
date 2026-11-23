"""Compare two training prefixes with identical calibration and development frames."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from experiment import filtered, quality, raw_candidates, select_operation, summarize, torch, warm_ids, with_nms
from snl_detector import backend
from snl_inputs import DEFAULT_OUTPUT, PROFILES, feature_cache, load_gray, load_labels, read_json, write_json
from snl_motion_aug import local_darkness
from videosar.snl_causal import CausalFeatures


def label_observations(gray, labels):
    centers, contrast = [], []
    for index, boxes in labels.items():
        frame = gray[index - 2].astype(np.float32) / 255
        centers.extend((boxes[:, :2] + boxes[:, 2:]) / 2)
        contrast.extend(local_darkness(frame, box) for box in boxes)
    return np.asarray(centers), np.asarray(contrast)


def compare_coverage(output, gray, development):
    dev_positions, dev_contrast = label_observations(gray, development)
    report = {"development_boxes": len(dev_positions),
              "development_contrast_q25_q50_q75": np.quantile(dev_contrast, (0.25, 0.5, 0.75)).tolist(),
              "prefixes": {}}
    for profile in ("original", "prospective"):
        ids = PROFILES[profile][0]
        labels, _, _ = load_labels(output, ids, gray)
        positions, contrast = label_observations(gray, labels)
        distances = cKDTree(positions).query(dev_positions)[0]
        quantiles = np.quantile(contrast, (0.25, 0.5, 0.75))
        report["prefixes"][profile] = {
            "train_frames": len(ids), "train_boxes": len(positions),
            "occupied_32_px_cells": len(np.unique((positions / 32).astype(int), axis=0)),
            "train_contrast_q25_q50_q75": quantiles.tolist(),
            "development_below_training_q25": int((dev_contrast < quantiles[0]).sum()),
            "development_nearest_training_centers_px_q25_q50_q75": np.quantile(distances, (0.25, 0.5, 0.75)).tolist(),
            "development_within_radius_px": {r: int((distances <= r).sum()) for r in (5, 10, 20, 40)},
        }
    write_json(output / "training-coverage-comparison.json", report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early", type=Path, default=DEFAULT_OUTPUT / "early-stacked")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--coverage-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    early, output = args.early.resolve(), args.output.resolve()
    gray = load_gray(output, args.ffmpeg)
    _, cal_ids, dev_ids = PROFILES["prospective"]
    development, _, _ = load_labels(output, dev_ids, gray)
    if args.coverage_only:
        compare_coverage(output, gray, development)
        return
    calibration, _, _ = load_labels(output, cal_ids, gray)
    features = feature_cache(output, gray, max(dev_ids), CausalFeatures("stacked"))
    cal_warm, dev_warm = warm_ids(calibration), warm_ids(development)
    result = {"device": backend().type, "calibration_ids": list(cal_ids), "development_ids": list(dev_ids), "prefixes": {}}
    for path, profile in ((early, "original"), (output, "prospective")):
        train_ids = PROFILES[profile][0]
        protocol = read_json(path / "split-protocol.json")
        if protocol["train"] != list(train_ids) or max(train_ids) >= min(cal_ids):
            raise ValueError("The fitting prefixes must match the declared chronology")
        registry = read_json(path / "checkpoints-center-stacked.json")
        rows = []
        for config in registry:
            if config.get("input_profile") != "stacked" or config["training_label_frames"] != len(train_ids):
                raise ValueError("Training budget comparison requires the same directly trained detector")
            val = raw_candidates(path, config, features, cal_warm, "budget-validation")
            operation, recalibrated, _ = select_operation(val, calibration)
            dev = raw_candidates(path, config, features, dev_warm, "budget-development")
            predicted = filtered(with_nms(dev, operation["nms"]), operation["threshold"], operation["low"])
            metrics, _ = summarize(predicted, development)
            rows.append({"config": config | operation, "calibration": recalibrated, "development": metrics})
        best = max(rows, key=lambda row: quality(row["calibration"]))
        result["prefixes"][profile] = {"training_frames": len(train_ids), "selected_by_calibration": best, "checkpoints": rows}
        print(profile, len(train_ids), best, flush=True)
    write_json(output / "training-prefix-comparison.json", result)


if __name__ == "__main__":
    main()
