"""Compare causal low-rank and sparse background models on the open development splits."""
import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from experiment import baseline_predictions, quality, summarize
from snl_inputs import DEFAULT_OUTPUT, PROFILES, load_gray, load_labels, write_json
from videosar.trajectory import low_rank_basis

LAGS = (0, 3, 6, 9, 12, 15, 18, 24, 30, 42, 54, 66)


def residuals(gray, ids):
    @lru_cache(maxsize=72)
    def smoothed(index):
        return uniform_filter(gray[index].astype(np.float32) / 255, 3, mode="nearest")

    result = {name: {} for name in ("low_rank", "robust_low_rank", "robust_with_local_contrast")}
    for index in ids:
        images = np.stack([smoothed(index - 2 - lag) for lag in reversed(LAGS) if lag <= index - 2])
        stack = images.reshape(len(images), -1)
        basis = low_rank_basis(stack, rank=2)
        first = (basis @ basis.T) @ stack
        mismatch = np.abs(stack - first)
        outliers = mismatch >= np.quantile(mismatch, 0.985, axis=1, keepdims=True)
        cleaned = np.where(outliers, first, stack)
        robust_basis = low_rank_basis(cleaned, rank=2)
        next_background = (robust_basis[-1] @ robust_basis.T) @ cleaned
        robust = np.maximum(next_background.reshape(images[-1].shape) - images[-1], 0)
        local = uniform_filter(images[-1], 15, mode="nearest") - images[-1]
        result["low_rank"][index] = np.maximum(first[-1].reshape(images[-1].shape) - images[-1], 0)
        result["robust_low_rank"][index] = robust
        result["robust_with_local_contrast"][index] = robust * np.clip(local / 0.045, 0, 1)
    return result


def fit_operation(maps, labels):
    best = None
    calibration = {index: maps[index] for index in labels}
    for threshold in (0.006, 0.009, 0.013, 0.018, 0.025, 0.034, 0.046, 0.060):
        for area in (8, 15, 25, 45):
            result, _ = summarize(baseline_predictions(calibration, threshold, area), labels)
            if best is None or quality(result) > quality(best[2]):
                best = (threshold, area, result)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    output = args.output.resolve()
    gray = load_gray(output, args.ffmpeg)
    _, cal_ids, dev_ids = PROFILES["prospective"]
    validation, _, _ = load_labels(output, cal_ids, gray)
    development, _, _ = load_labels(output, dev_ids, gray)
    results = {}
    for name, maps in residuals(gray, [*cal_ids, *dev_ids]).items():
        threshold, area, calibration = fit_operation(maps, validation)
        predicted = baseline_predictions({index: maps[index] for index in development}, threshold, area)
        evaluation, _ = summarize(predicted, development)
        results[name] = {"threshold": threshold, "min_area": area, "validation": calibration, "development": evaluation}
        print(name, results[name], flush=True)
    write_json(output / "low-rank-comparison.json", {"lags": list(LAGS), "rank": 2, "comparisons": results})


if __name__ == "__main__":
    main()
