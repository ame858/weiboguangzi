"""Measure how many development misses come from scoring, position and box size."""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from experiment import (filter_by_config, load_candidates, model_key, raw_candidates, require_evaluation_device,
                        summarize, torch, warm_ids, with_nms)
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, WIDTH, feature_cache, load_gray, load_labels, read_json, write_json
from snl_motion_aug import local_darkness
from videosar.snl_causal import CausalFeatures, box_iou, matched_boxes


def count_pairs(valid):
    row, col = linear_sum_assignment(valid, maximize=True)
    return int(valid[row, col].sum())


def size_oracle(boxes, truth):
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    overlaps = []
    for target in truth:
        half = 0.5 * (target[2:] - target[:2])
        rebuilt = np.column_stack((centers - half, centers + half))
        overlaps.append(box_iou(rebuilt, target[None])[:, 0])
    return np.stack(overlaps, axis=1) if overlaps else np.empty((len(boxes), 0))


def pair_diagnostics(boxes, truth):
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    targets = (truth[:, :2] + truth[:, 2:]) / 2
    near = np.linalg.norm(centers[:, None] - targets[None, :], axis=-1) <= 5
    iou = box_iou(boxes, truth)
    fixed_size = size_oracle(boxes, truth)
    return np.asarray([count_pairs(test) for test in (iou >= 0.50, iou >= 0.25, near,
                                                      (fixed_size >= 0.50) | (iou >= 0.50))])


def audit(candidates, predicted, labels):
    names = ("matches_iou50", "matches_iou25", "matches_within_5_px", "matches_with_oracle_size")
    result = {}
    for key, cap in (("selected_threshold", 0), ("top_15", 15), ("top_60", 60), ("all_candidates", None)):
        rows = []
        for index, target in labels.items():
            boxes, scores = candidates[index]
            if cap == 0:
                boxes = predicted[index][0]
            elif cap is not None:
                boxes = boxes[np.argsort(-scores, kind="stable")[:cap]]
            rows.append([len(boxes), *pair_diagnostics(boxes, target)])
        total = np.sum(rows, axis=0)
        result[key] = {k: int(v) for k, v in zip(("candidate_count", *names), total, strict=True)}
    return {"frames": len(labels), "targets": sum(map(len, labels.values())), "candidate_groups": result}


def remap_for_sensitivity(labels, coefficients):
    reference_size = np.array([644, 704], dtype=np.float32)
    scale = np.array([WIDTH, HEIGHT], dtype=np.float32) / reference_size
    revised = {}
    for index, boxes in labels.items():
        coordinates = boxes.reshape(-1, 2) / scale
        design = np.column_stack((np.ones(len(coordinates)), coordinates / reference_size - 0.5))
        mapped = (coordinates + design @ coefficients) * scale
        revised[index] = mapped.clip(0, [WIDTH, HEIGHT]).astype(np.float32).reshape(-1, 4)
    return revised


def appearance(gray, labels):
    values = {}
    for index, boxes in labels.items():
        image = gray[index-2].astype(np.float32) / 255
        values[index] = np.asarray([local_darkness(image, box) for box in boxes], dtype=float)
    return values


def stratify(predicted, labels, contrast, train_contrast, train_centers):
    darkness, distances, hit = [], [], []
    for index, boxes in labels.items():
        darkness.extend(contrast[index])
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        distances.extend(train_centers.query(center)[0])
        found = np.zeros(len(boxes), dtype=bool)
        found[matched_boxes(predicted[index][0], boxes)[:, 1]] = True
        hit.extend(found)
    darkness, distances, hit = map(np.asarray, (darkness, distances, hit))
    weak, high = np.percentile(train_contrast, (25, 75))
    groups = {"at_or_below_zero": darkness <= 0, "positive_below_training_q25": (darkness > 0) & (darkness < weak),
              "between_training_q25_q75": (darkness >= weak) & (darkness < high), "at_or_above_training_q75": darkness >= high,
              "within_10_px_of_train": distances <= 10, "farther_than_10_px_from_train": distances > 10}
    summary = {key: {"targets": int(keep.sum()), "matched": int(hit[keep].sum()),
                     "recall": float(hit[keep].mean()) if keep.any() else None} for key, keep in groups.items()}
    return {"train_darkness_quartiles": [float(weak), float(high)],
            "split_darkness_q25_q50_q75": np.percentile(darkness, (25, 50, 75)).tolist(), "subgroups": summary}


def paired_block_f1(rows, reference, *, block=6, draws=3000):
    rng = np.random.default_rng(629)
    frames = len(rows)
    starts = rng.integers(frames, size=(draws, (frames+block-1)//block))
    indices = ((starts[:, :, None] + np.arange(block)) % frames).reshape(draws, -1)[:, :frames]

    def f1(records):
        counts = np.asarray([[row[key] for key in ("tp", "fp", "fn")] for row in records])
        tp, fp, fn = counts[indices].sum(axis=1).T
        return 2 * tp / np.maximum(2 * tp + fp + fn, 1)

    return {"resamples": draws, "block_frames": block, "sampling": "circular",
            "f1_difference_ci95": np.percentile(f1(rows) - f1(reference), (2.5, 97.5)).tolist()}


def sample_training(output, config, gray, training, contrast, centers, high):
    require_evaluation_device(config)
    torch.set_num_threads(4)
    sample_ids = tuple(training)[30::5]
    sample = {index: training[index] for index in sample_ids}
    features = gray[:, None]
    if config["method"] == "temporal":
        features = feature_cache(output, gray, high, CausalFeatures(config.get("input_profile", "raw")))
    ids = warm_ids(sample) if config["low"] is not None else sample_ids
    raw = raw_candidates(output, config, features, ids, "training-sample")
    candidates = with_nms(raw, config["nms"])
    predicted = filter_by_config(candidates, config)
    values = appearance(gray, sample)
    return {"ids": list(sample_ids), "in_sample": True, "metrics": summarize(predicted, sample)[0],
            "appearance_stratification": stratify(predicted, sample, values, contrast, centers)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--decision", type=Path, help="Audit a specific decision JSON instead of the current leader")
    parser.add_argument("--compare-decision", type=Path, help="Cross two evaluated operating points for a diagnostic")
    parser.add_argument("--training-sample", action="store_true", help="Also run the selected checkpoint on early fitting examples")
    args = parser.parse_args()
    output = args.output.resolve()
    decision = read_json(args.decision) if args.decision else read_json(output / "development-status.json")["best"]
    config = decision["config"]
    comparison = read_json(args.compare_decision)["config"] if args.compare_decision else None
    profile = read_json(output / "split-protocol.json")["profile"]
    gray = load_gray(output, args.ffmpeg)
    train_ids, *held = PROFILES[profile]
    splits = zip(("validation", "development"), held, strict=True)
    train, _, _ = load_labels(output, train_ids, gray)
    train_contrast = np.concatenate(list(appearance(gray, train).values()))
    boxes = np.concatenate(list(train.values()))
    train_centers = cKDTree((boxes[:, :2] + boxes[:, 2:]) / 2)
    result = {"config": config}
    if comparison is not None:
        result["compared_config"] = comparison
    if args.training_sample:
        result["training_sample"] = sample_training(output, config, gray, train, train_contrast, train_centers, max(held[-1]))
    spatial_audit = output / "reference-spatial-alignment.json"
    coefficients = None
    if spatial_audit.is_file():
        coefficients = read_json(spatial_audit)["train"]["affine_coefficients_bias_normalized_x_normalized_y"]
    for name, indices in splits:
        labels, _, _ = load_labels(output, indices, gray)
        candidates = with_nms(load_candidates(output, config, name), config["nms"])
        predicted = filter_by_config(candidates, config)
        result[name] = audit(candidates, predicted, labels)
        result[name]["appearance_stratification"] = stratify(predicted, labels, appearance(gray, labels), train_contrast,
                                                            train_centers)
        if comparison is not None:
            comparisons = []
            same_operation_rows = []
            for source in (config, comparison):
                raw = load_candidates(output, source, name)
                for operating in (config, comparison):
                    boxes = with_nms(raw, operating["nms"])
                    rows = filter_by_config(boxes, operating)
                    metrics, per_frame = summarize(rows, labels)
                    comparisons.append({"predictions": model_key(source), "operating_point": model_key(operating),
                                        "metrics": metrics})
                    if operating is comparison:
                        same_operation_rows.append(per_frame)
            result[name]["operating_point_crosscheck"] = comparisons
            result[name]["f1_difference_at_reference_operating_point"] = paired_block_f1(*same_operation_rows)
        if coefficients is not None:
            shifted = remap_for_sensitivity(labels, coefficients)
            result[name]["spatial_mapping_sensitivity"] = {
                "fit": "training reference images", "nominal": summarize(predicted, labels)[0],
                "affine_reference_mapping": summarize(predicted, shifted)[0],
            }
        print(name, result[name], flush=True)
    suffix = f"-{model_key(config)}" if args.decision else ""
    write_json(output / f"detection-error-audit{suffix}.json", result)


if __name__ == "__main__":
    main()
