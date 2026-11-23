"""Audit the candidate coverage ceiling and an equal-size spatially random control."""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiment import load_candidates, model_key
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, WIDTH, load_gray, load_labels, read_json, write_json
from videosar.snl_causal import matched_boxes


def coverage(boxes, truth):
    left = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    right = (truth[:, :2] + truth[:, 2:]) * 0.5
    inside = np.linalg.norm(left[:, None] - right[None, :], axis=-1) <= 5
    row, column = linear_sum_assignment(inside, maximize=True)
    return np.array([len(matched_boxes(boxes, truth)), int(inside[row, column].sum())])


def random_centers(boxes, rng):
    half = (boxes[:, 2:] - boxes[:, :2]) * 0.5
    center = rng.uniform(half, np.asarray([WIDTH, HEIGHT]) - half)
    return np.column_stack((center - half, center + half)).astype(np.float32)


def main():
    from snl_fusion import choose_sources

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    output = args.output.resolve()
    profile = read_json(output / "split-protocol.json")["profile"]
    gray = load_gray(output, args.ffmpeg)
    target, _, _ = load_labels(output, PROFILES[profile][2], gray)
    configs = choose_sources(output, include_full_frame=True)
    sources = [load_candidates(output, config, "development") for config in configs]
    rng = np.random.default_rng(66813)
    total = sum(map(len, target.values()))
    reports = []
    for cap in (15, 30, 60, 150, None):
        actual, shuffled = np.zeros(2, dtype=int), np.zeros(2, dtype=float)
        proposed = 0
        controls = 8
        for index, truth in target.items():
            selected = []
            for source in sources:
                boxes, scores = source[index]
                top = np.argsort(-scores, kind="stable")[:cap]
                selected.append(boxes[top])
            joined = np.concatenate(selected)
            proposed += len(joined)
            actual += coverage(joined, truth)
            for _ in range(controls):
                shuffled += coverage(random_centers(joined, rng), truth) / controls
        row = {"per_source_cap": cap, "proposals": proposed, "targets": total,
               "matches_iou50": int(actual[0]), "center_matches_5_px": int(actual[1]),
               "random_center_mean_iou50": float(shuffled[0]), "random_center_mean_matches_5_px": float(shuffled[1])}
        reports.append(row)
        print(row, flush=True)
    write_json(output / "proposal-coverage-with-control.json",
               {"models": [model_key(cfg) for cfg in configs], "control_seed": 66813, "measurements": reports})


if __name__ == "__main__":
    main()
