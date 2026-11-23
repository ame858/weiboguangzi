"""Test whether causal dense optical flow adds small-shadow proposals on SNL."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import maximum_filter

from experiment import CONFIDENCES, filter_by_config, load_candidates, quality, summarize, with_nms
from snl_bytetrack import REFERENCE
from snl_inputs import DEFAULT_OUTPUT, PROFILES, load_gray, load_labels, read_json, write_csv, write_json
from snl_templates import SIZES
from videosar.snl_causal import detection_counts


LAGS = (3, 6)
CUES = ("contrast", "relative_motion", "contrast_with_motion")
CAP = 256


def relative_speed(flow, gray, index):
    current = gray[index - 2]
    speeds = []
    for lag in LAGS:
        displacement = flow.calc(current, gray[index - 2 - lag], None)
        common_shift = np.median(displacement[::8, ::8], axis=(0, 1))
        remaining = (displacement - common_shift) / lag
        speeds.append(np.hypot(remaining[..., 0], remaining[..., 1]))
    return np.sqrt(speeds[0] * speeds[1])


def box_mean(image, width, height):
    return cv2.boxFilter(image, -1, (width, height), borderType=cv2.BORDER_REPLICATE)


def proposals(gray, speed):
    image = gray.astype(np.float32) / 255
    fields = np.full((len(CUES), *image.shape), -np.inf, dtype=np.float32)
    size_id = np.zeros(fields.shape, dtype=np.uint8)
    for index, (width, height) in enumerate(SIZES):
        inner = box_mean(image, width, height)
        outer = box_mean(image, width + 12, height + 12)
        dark = np.maximum(outer - inner, 0)
        motion = box_mean(speed, width, height)
        joint = dark * np.sqrt(motion)
        y, x = height // 2, width // 2
        for field, selected, candidate in zip(fields, size_id, (dark, motion, joint), strict=True):
            candidate[:y] = candidate[-y:] = -np.inf
            candidate[:, :x] = candidate[:, -x:] = -np.inf
            better = candidate > field
            field[better], selected[better] = candidate[better], index
    output = {}
    dims = np.asarray(SIZES, dtype=np.float32)
    for cue, field, selected in zip(CUES, fields, size_id, strict=True):
        y, x = np.nonzero((field > 0) & (field == maximum_filter(field, size=5, mode="nearest")))
        order = np.argsort(-field[y, x], kind="stable")[:CAP]
        y, x = y[order], x[order]
        half = dims[selected[y, x]] / 2
        centers = np.column_stack((x + 0.5, y + 0.5)).astype(np.float32)
        output[cue] = np.column_stack((centers - half, centers + half)), field[y, x]
    return output


def predictions(candidates, threshold):
    return {index: (boxes[scores >= threshold], scores[scores >= threshold], None)
            for index, (boxes, scores) in candidates.items()}


def counts(candidates, truth, limit, other=None):
    total = 0
    for index, target in truth.items():
        boxes = candidates[index][0][:limit]
        if other is not None:
            boxes = np.concatenate((boxes, other[index][0][:limit]))
        total += detection_counts(boxes, target)[0]
    return total


def mean_motion(speed, boxes):
    bounds = np.array(speed.shape[::-1])
    start = np.floor(boxes[:, :2]).astype(int).clip(0, bounds - 1)
    end = np.ceil(boxes[:, 2:]).astype(int).clip(start + 1, bounds)
    left, top = start.T
    right, bottom = end.T
    integral = cv2.integral(speed)
    values = integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left]
    return values / ((right - left) * (bottom - top))


def rescore(raw, motion, weight, center, scale):
    if weight == 0:
        return raw
    result = {}
    for index, (boxes, scores) in raw.items():
        p = np.clip(scores, 1e-6, 1 - 1e-6)
        delta = np.clip((motion[index] - center) / scale, -3, 3)
        confidence = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + weight * delta)))
        result[index] = boxes, confidence.astype(np.float32)
    return result


def neural_rescore(reference, raw, motion, validation, development):
    config = reference["config"]
    pool = np.concatenate([motion[index] for index in validation])
    lo, center, hi = np.quantile(pool, (0.25, 0.50, 0.75))
    scale = max(hi - lo, 1e-6)
    replay = with_nms({i: raw[i] for i in development}, config["nms"])
    baseline = summarize(filter_by_config(replay, config), development)[0]
    if baseline != reference["development"]:
        raise ValueError("The saved candidates no longer reproduce the neural reference")
    results = []
    for weight in (0, 0.25, 0.50, 1.0):
        candidate = rescore(raw, motion, weight, center, scale)
        ca = with_nms({i: candidate[i] for i in validation}, config["nms"])
        for threshold in CONFIDENCES:
            fitted = summarize(predictions(ca, threshold), validation)[0]
            results.append((weight, float(threshold), fitted))
    weight, threshold, fitted = max(results, key=lambda row: quality(row[2]))
    replay = rescore({i: raw[i] for i in development}, motion, weight, center, scale)
    result = summarize(predictions(with_nms(replay, config["nms"]), threshold), development)[0]
    return {"weight": weight, "threshold": threshold, "calibration": fitted, "development": result,
            "reference_development": baseline, "calibration_motion_median": float(center),
            "calibration_motion_iqr": float(scale)}


def study(output, ffmpeg):
    gray = load_gray(output, ffmpeg)
    _, cal_ids, dev_ids = PROFILES["prospective"]
    calibration, development = (load_labels(output, ids, gray)[0] for ids in (cal_ids, dev_ids))
    reference = read_json(output / REFERENCE)
    existing = load_candidates(output, reference["config"], "development")
    cached = load_candidates(output, reference["config"], "validation") | existing
    raw = {i: cached[i] for i in (*cal_ids, *dev_ids)}
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    all_candidates = {cue: {} for cue in CUES}
    motion = {}
    for index in (*cal_ids, *dev_ids):
        speed = relative_speed(flow, gray, index)
        motion[index] = mean_motion(speed, raw[index][0])
        for cue, values in proposals(gray[index - 2], speed).items():
            all_candidates[cue][index] = values
    results, frame_rows = {}, []
    for cue, candidates in all_candidates.items():
        ca = {i: candidates[i] for i in cal_ids}
        scores = np.concatenate([v[1] for v in ca.values()])
        grid = np.quantile(scores, np.linspace(0.1, 1, 37))
        threshold, fitted = max(((value, summarize(predictions(ca, value), calibration)[0]) for value in grid),
                                key=lambda pair: quality(pair[1]))
        replay, rows = summarize(predictions({i: candidates[i] for i in dev_ids}, threshold), development)
        results[cue] = {"threshold": float(threshold), "calibration": fitted, "development": replay,
                        "max_matches": {limit: counts(candidates, development, limit) for limit in (60, CAP)},
                        "with_center_max_matches": {limit: counts(candidates, development, limit, existing)
                                                    for limit in (60, CAP)}}
        frame_rows.extend({"cue": cue, **row} for row in rows)
    receipt = {"method": "OpenCV DIS medium", "opencv": cv2.__version__, "past_lags": LAGS, "box_wh": SIZES,
               "calibration_frames": list(cal_ids), "development_frames": list(dev_ids),
               "development_truth_boxes": sum(map(len, development.values())), "match_iou": 0.50,
               "center_max_matches": {limit: counts(existing, development, limit) for limit in (60, CAP)},
               "results": results, "neural_rescore": neural_rescore(reference, raw, motion, calibration, development)}
    write_json(output / "flow-candidate-study.json", receipt)
    write_csv(output / "flow-candidate-development.csv", frame_rows)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    print(study(args.output.resolve(), args.ffmpeg), flush=True)


if __name__ == "__main__":
    main()
