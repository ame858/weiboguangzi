"""Evaluate a causal adaptation of Yan et al. on the public SNL splits."""
import argparse
from collections import deque
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import expit, logit

from experiment import CONFIDENCES, filter_by_config, load_candidates, nonmax, quality, summarize, with_nms
from snl_bytetrack import REFERENCE
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, WIDTH, load_gray, load_labels, read_json, write_csv, write_json
from videosar.snl_causal import HistoryRegistration, detection_counts


LAGS = (6, 12, 18, 24, 30)
MAPS = ("raw", "registered_median", "subspace", "subspace_admm")
MODES = ("mce_variance", "mce_std", "mce_relaxed", "dark_tail")
JS_STEP = 0.05


def js_lookup():
    """Integrate in the pixel Gaussian's units, where the two densities overlap."""
    distance = np.arange(201)[None, :] * JS_STEP
    log_scale = np.arange(241)[:, None] * JS_STEP
    nodes, weights = np.polynomial.legendre.leggauss(64)
    overlap = np.zeros((len(log_scale), distance.size))
    for node, weight in zip(12 * nodes, 12 * weights, strict=True):
        pixel = -0.5 * node**2
        neighborhood = -log_scale - 0.5 * (node * np.exp(-log_scale) + distance)**2
        overlap += weight * (np.exp(pixel) * np.logaddexp(0, neighborhood - pixel)
                             + np.exp(neighborhood) * np.logaddexp(0, pixel - neighborhood))
    return np.clip(np.log(2) - overlap / (2 * np.sqrt(2 * np.pi)), 0, np.log(2))


def registered_history(gray, start, stop):
    """Move only prior frames into the current radar grid."""
    reg = HistoryRegistration()
    history = deque(maxlen=max(LAGS))
    for index in range(max(2, start - max(LAGS)), stop + 1):
        image = gray[index - 2].astype(np.float32) / 255
        position = reg.estimate(image)
        if index >= start:
            aligned = [reg.align(history[-lag], lag, position) for lag in LAGS]
            yield index, image, position, np.stack(aligned, axis=-1)
        reg.positions.append(position)
        history.append(image)


def robust_background(image, stack):
    """Fit a rolling historical subspace with two-Gaussian residual weights."""
    mean = stack.mean(axis=-1)
    history = (stack - mean[..., None]).reshape(-1, len(LAGS))
    _, vectors = np.linalg.eigh(history.T @ history)
    basis = np.ascontiguousarray(history @ vectors[:, -2:])
    sample = basis[::5]
    target = (image - mean).ravel()[::5]
    weight = np.ones_like(target)
    sigma = np.array([0.035, 0.15], dtype=np.float32)
    mixture = np.array([0.85, 0.15], dtype=np.float32)
    for _ in range(4):
        weighted = sample.T * weight
        coef = np.linalg.solve(weighted @ sample + np.eye(2, dtype=np.float32), weighted @ target)
        error_squared = (target - sample @ coef) ** 2
        likelihood = (mixture / sigma) * np.exp(-0.5 * error_squared[:, None] / sigma**2)
        responsibility = likelihood / np.maximum(likelihood.sum(axis=1, keepdims=True), 1e-20)
        weight = (responsibility / sigma**2).sum(axis=1)
        variance = (responsibility.T @ error_squared) / np.maximum(responsibility.sum(axis=0), 1e-20)
        sigma = np.sqrt(np.clip(variance, 0.015**2, 0.25**2))
        mixture = (responsibility.mean(axis=0) + 0.01) / 1.02
    return mean + (basis @ coef).reshape(image.shape)


def low_rank_foreground(residuals):
    """Nuclear norm plus sparse decomposition on a trailing residual block."""
    shape = residuals[-1].shape
    matrix = np.column_stack([frame.ravel() for frame in residuals]).astype(np.float32)
    sparse = np.zeros_like(matrix)
    dual = np.zeros_like(matrix)
    scale = max(float(np.sqrt(matrix.size * np.mean(matrix**2))), 1e-4)
    rho = 1.25 / scale
    penalty = 1 / np.sqrt(matrix.shape[0])
    for _ in range(12):
        adjusted = matrix - sparse + dual
        values, vectors = np.linalg.eigh(adjusted.T @ adjusted)
        singular = np.sqrt(np.maximum(values, 0))
        shrink = np.maximum(1 - 1 / (rho * np.maximum(singular, 1e-6)), 0)
        low_rank = adjusted @ ((vectors * shrink) @ vectors.T)
        error = matrix - low_rank + dual
        sparse = np.sign(error) * np.maximum(np.abs(error) - penalty / rho, 0)
        dual += matrix - low_rank - sparse
        rho *= 1.6
    return low_rank[:, -1].reshape(shape)


def gray_bytes(values):
    return np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)


def js_features(values, lookup):
    """Equation (4) under the two readings of the paper's pixel spread."""
    pixels = values.astype(np.float32) / 255
    window = (20, 20)
    mu = cv2.blur(pixels, window)
    variance = cv2.blur(pixels**2, window) - mu**2
    for narrow in (np.sqrt(1e-5), 1e-5):
        sigma = np.sqrt(np.maximum(variance, narrow**2))
        distance = np.abs(pixels - mu) / (JS_STEP * sigma)
        scale = np.log(sigma / narrow) / JS_STEP
        js = cv2.remap(lookup, distance.astype(np.float32), scale.astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
        yield gray_bytes((js - js.min()) / max(float(np.ptp(js)), 1e-12))


def mce_threshold(gray, js):
    """Equations (8)–(13), with a retained-mass floor to avoid a trivial edge cutoff."""
    levels = np.arange(256, dtype=np.float64)
    hist = np.bincount(256 * gray.ravel().astype(np.int32) + js.ravel(), minlength=65536).reshape(256, 256)
    hist = hist / gray.size
    log_moments = levels * np.log(np.maximum(levels, 1))
    mass, a, b, terms = [array.cumsum(axis=0).cumsum(axis=1) for array in
                         (hist, hist * levels[:, None], hist * levels, hist * (log_moments[:, None] + log_moments))]
    upper = [array[-1:] - array for array in (mass, a, b)]
    cost = terms[-1:] - (a * np.log(np.maximum(a / np.maximum(mass, 1e-15), 1e-15))
                         + b * np.log(np.maximum(b / np.maximum(mass, 1e-15), 1e-15)))
    other_mass, other_a, other_b = upper
    cost -= (other_a * np.log(np.maximum(other_a / np.maximum(other_mass, 1e-15), 1e-15))
             + other_b * np.log(np.maximum(other_b / np.maximum(other_mass, 1e-15), 1e-15)))
    valid = (mass[-1:] >= 0.70) & (mass >= 0.001) & (other_mass >= 0.001)
    if not valid.any():
        return int(np.quantile(gray, 0.02)), 255
    row, column = np.unravel_index(np.argmin(np.where(valid, cost / np.maximum(mass[-1:], 1e-15), np.inf)), cost.shape)
    return int(row), int(column)


def regions(values, lookup, edges):
    js_var, js_std = js_features(values, lookup)
    threshold, edge_cutoff = mce_threshold(values, js_var)
    std_threshold, std_cutoff = mce_threshold(values, js_std)
    masks = ((values <= threshold) & (js_var <= edge_cutoff), (values <= std_threshold) & (js_std <= std_cutoff),
             values <= threshold, values <= np.quantile(values, 0.02))
    result = {}
    for mode, mask in zip(MODES, masks, strict=True):
        count, components, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        weight = np.bincount(components.ravel(), weights=(255 - values).ravel(), minlength=count)
        overlap = np.bincount(components.ravel(), weights=edges.ravel(), minlength=count)
        x, y, w, h, area = stats.T
        keep = (w >= 3) & (h >= 4) & (w <= 36) & (h <= 56) & (area >= 8) & (overlap <= 0.5 * area)
        keep[0] = False
        boxes = np.column_stack((x, y, x + w, y + h))[keep].astype(np.float32)
        scores = (weight[keep] / (255 * area[keep])).astype(np.float32)
        result[mode] = boxes, scores
    return result


def mean_darkness(image, boxes):
    bounds = np.array(image.shape[::-1])
    left, top = np.floor(boxes[:, :2]).astype(int).clip(0, bounds - 1).T
    right, bottom = np.ceil(boxes[:, 2:]).astype(int).clip(np.column_stack((left, top)) + 1, bounds).T
    integral = cv2.integral(image)
    total = integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left]
    return (1 - total / (255 * (right - left) * (bottom - top))).astype(np.float32)


def collect(gray, start, stop, neural_raw):
    result = {(method, mode): {} for method, mode in product(MAPS, MODES)}
    positions = {}
    cues = {method: {} for method in MAPS[1:]}
    residual_history = deque(maxlen=5)
    lookup = js_lookup()
    for index, image, position, aligned in registered_history(gray, start, stop):
        residual = image - robust_background(image, aligned)
        residual_history.append((position, residual))
        block = [cv2.warpAffine(frame, np.float32([[1, 0, position[1] - origin[1]], [0, 1, position[0] - origin[0]]]),
                                image.shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                 for origin, frame in residual_history]
        foreground = low_rank_foreground(block)
        original = gray[index - 2].astype(np.float32)
        edges = np.hypot(cv2.Sobel(original, -1, 1, 0), cv2.Sobel(original, -1, 0, 1)) >= 50
        images = (image, 0.5 + 2 * (image - np.median(aligned, axis=-1)), 0.5 + 2 * residual, 0.5 + 2 * foreground)
        for method, values in zip(MAPS, images, strict=True):
            values = gray_bytes(values)
            for mode, candidates in regions(values, lookup, edges).items():
                result[method, mode][index] = candidates
            if method in cues and index in neural_raw:
                cues[method][index] = mean_darkness(values, neural_raw[index][0])
        positions[index] = position
        if (index - start + 1) % 10 == 0:
            print(f"Processed through frame {index}/{stop}", flush=True)
    return result, positions, cues


def track_support(candidates, positions, distance):
    """Make one-to-one links over five prior frames and confirm in a 20-frame history."""
    active = []
    result = {}
    for index, (boxes, _) in candidates.items():
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2 - positions[index][::-1]
        active = [track for track in active if index - track[-1][0] <= 5]
        gaps = cdist(np.asarray([track[-1][1] for track in active]).reshape(-1, 2), centers)
        rows, cols = linear_sum_assignment(np.where(gaps <= distance, gaps, (len(centers) + 1) * (distance + 1)))
        assigned = {col: active[row] for row, col in zip(rows, cols, strict=True) if gaps[row, col] <= distance}
        votes = np.empty(len(boxes), dtype=np.int32)
        for col, center in enumerate(centers):
            track = assigned.get(col)
            if track is None:
                track = deque()
                active.append(track)
            track.append((index, center))
            while index - track[0][0] >= 20:
                track.popleft()
            motion = np.max(np.linalg.norm(np.asarray([point for _, point in track]) - center, axis=1))
            votes[col] = len(track) if motion >= 2 else 0
        result[index] = votes
    return result


def select(candidates, support, config):
    result = {}
    for index, (boxes, scores) in candidates.items():
        dimensions = boxes[:, 2:] - boxes[:, :2]
        keep = (scores >= config["threshold"]) & (dimensions.prod(axis=1) >= config["min_box_area"])
        if config["confirmations"]:
            keep &= support[index] >= config["confirmations"]
        boxes = boxes[keep].copy()
        pad = dimensions[keep] * (config["scale"] - 1) / 2
        boxes[:, :2] = (boxes[:, :2] - pad).clip(0)
        boxes[:, 2:] = (boxes[:, 2:] + pad).clip(max=(WIDTH, HEIGHT))
        result[index] = (boxes, scores[keep], None)
    return result


def union(neural, geometric):
    result = {}
    for index, (boxes, scores, _) in neural.items():
        added, added_scores, _ = geometric[index]
        boxes, scores = nonmax(np.concatenate((boxes, added)), np.concatenate((scores + 1, added_scores)), 0.35)
        result[index] = boxes, scores, None
    return result


def coverage(candidates, truth, neural=None):
    total = 0
    for index, target in truth.items():
        boxes = candidates[index][0]
        if neural is not None:
            boxes = np.concatenate((boxes, neural[index][0]))
        total += detection_counts(boxes, target)[0]
    return total


def fit_rescore(neural, val_raw, dev_raw, cues, calibration, development):
    def adjust(candidates, cue, center, spread, weight):
        return {i: (boxes, expit(logit(scores.clip(1e-6, 1 - 1e-6))
                                 + weight * ((cue[i] - center) / spread).clip(-3, 3)).astype(np.float32))
                for i, (boxes, scores) in candidates.items()}

    best = None
    for method, cue in cues.items():
        q1, center, q3 = np.quantile(np.concatenate([cue[index] for index in calibration]), (0.25, 0.50, 0.75))
        spread = max(float(q3 - q1), 1e-6)
        for weight in (0.125, 0.25, 0.5, 1.0):
            rescored = with_nms(adjust(val_raw, cue, center, spread, weight), neural["nms"])
            for threshold in CONFIDENCES:
                settings = dict(neural, threshold=float(threshold))
                metric = summarize(filter_by_config(rescored, settings), calibration)[0]
                if best is None or quality(metric) > quality(best[0]):
                    best = metric, method, float(center), spread, weight, settings
    value, method, center, spread, weight, settings = best
    rescored = with_nms(adjust(dev_raw, cues[method], center, spread, weight), neural["nms"])
    result, rows = summarize(filter_by_config(rescored, settings), development)
    config = dict(method=method, center=center, spread=spread, weight=weight, threshold=settings["threshold"])
    return {"config": config, "calibration": value, "development": result}, rows


def evaluate(output, ffmpeg):
    _, cal_ids, dev_ids = PROFILES["prospective"]
    gray = load_gray(output, ffmpeg)
    calibration, development = (load_labels(output, ids, gray)[0] for ids in (cal_ids, dev_ids))
    reference = read_json(output / REFERENCE)
    neural = reference["config"]
    val_raw = load_candidates(output, neural, "validation")
    neural_raw_dev = load_candidates(output, neural, "development")
    val_raw = {index: val_raw[index] for index in cal_ids}
    neural_raw_dev = {index: neural_raw_dev[index] for index in dev_ids}
    neural_val, neural_dev = (filter_by_config(with_nms(raw, neural["nms"]), neural) for raw in (val_raw, neural_raw_dev))
    baseline_val = summarize(neural_val, calibration)[0]
    baseline_dev, baseline_rows = summarize(neural_dev, development)
    if baseline_dev != reference["development"]:
        raise ValueError("Archived neural predictions failed to reproduce the release metrics")

    start = cal_ids[0] - 19
    neural_raw = val_raw | neural_raw_dev
    candidates, positions, cues = collect(gray, start, dev_ids[-1], neural_raw)
    prefix, prefix_positions, prefix_cues = collect(gray, start, cal_ids[0] + 2, neural_raw)
    for key, frames in prefix.items():
        for index, (boxes, scores) in frames.items():
            np.testing.assert_array_equal(boxes, candidates[key][index][0])
            np.testing.assert_array_equal(scores, candidates[key][index][1])
            np.testing.assert_array_equal(prefix_positions[index], positions[index])
    for method, frames in prefix_cues.items():
        for index, cue in frames.items():
            np.testing.assert_array_equal(cue, cues[method][index])

    best = {method: (None, None) for method in MAPS}
    fused = None, None
    supports = {}
    diagnostics = {}
    cal_frames = tuple(range(start, cal_ids[-1] + 1))
    for (method, mode), raw in candidates.items():
        raw_cal = {index: raw[index] for index in cal_frames}
        pool = np.concatenate([raw[index][1] for index in cal_ids])
        thresholds = [0.0, *np.quantile(pool, (0.5, 0.75, 0.9))] if len(pool) else [0.0]
        for distance in (12, 30):
            key = method, mode, distance
            short = track_support({index: raw[index] for index in prefix[method, mode]}, prefix_positions, distance)
            support = track_support(raw, positions, distance)
            for index, votes in short.items():
                np.testing.assert_array_equal(votes, support[index])
            supports[key] = support
            for threshold, area, confirmations, scale in product(thresholds, (0, 50), (0, 3, 5), (1.0, 1.25)):
                if not confirmations and distance != 12:
                    continue
                config = dict(method=method, mode=mode, distance=distance, threshold=float(threshold),
                              min_box_area=area, confirmations=confirmations, scale=scale)
                picked = select({index: raw_cal[index] for index in cal_ids}, support, config)
                score = summarize(picked, calibration)[0]
                if best[method][1] is None or quality(score) > quality(best[method][1]):
                    best[method] = config, score
                fused_score = summarize(union(neural_val, picked), calibration)[0]
                if any(len(boxes) for boxes, _, _ in picked.values()) and (fused[1] is None or quality(fused_score) > quality(fused[1])):
                    fused = config, fused_score
        diagnostic_config = dict(threshold=0, min_box_area=0, confirmations=0, scale=1.0)
        raw_dev = {index: raw[index] for index in dev_ids}
        expanded = select(raw_dev, {}, dict(diagnostic_config, scale=1.25))
        diagnostics[f"{method}/{mode}"] = {"proposals": sum(len(row[0]) for row in raw_dev.values()),
                                            "matches": coverage(raw_dev, development),
                                            "with_neural": coverage(raw_dev, development, neural_raw_dev),
                                            "expanded_with_neural": coverage(expanded, development, neural_raw_dev)}
    print("Selected operating points from calibration only", flush=True)

    rows = [dict(method="released_seven_frame", **row) for row in baseline_rows]
    results = {"released_seven_frame": {"calibration": baseline_val, "development": baseline_dev}}
    for label, (config, score) in (*best.items(), ("roi_union", fused)):
        if config is not None:
            raw = candidates[config["method"], config["mode"]]
            support = supports[config["method"], config["mode"], config["distance"]]
            selected = select({index: raw[index] for index in dev_ids}, support, config)
            replay = union(neural_dev, selected) if label == "roi_union" else selected
        else:
            replay = neural_dev
        dev, per_frame = summarize(replay, development)
        rows.extend(dict(method=label, **row) for row in per_frame)
        results[label] = {"config": config, "calibration": score or baseline_val, "development": dev}
    results["neural_rescore"], rescored_rows = fit_rescore(neural, val_raw, neural_raw_dev, cues, calibration, development)
    rows.extend(dict(method="neural_rescore", **row) for row in rescored_rows)
    report = {"paper": "https://doi.org/10.1109/JSTARS.2025.3568810", "preprocessing": "https://arxiv.org/abs/2501.09341",
              "adaptation": "rolling two-Gaussian weighted subspace, trailing ADMM, causal track confirmation",
              "opencv": cv2.__version__, "past_lags": LAGS, "calibration_ids": cal_ids, "development_ids": dev_ids,
              "match_iou": 0.50, "prefix_verified_through": max(prefix_positions), "results": results,
              "unfiltered_neural_matches": coverage(neural_raw_dev, development), "unfiltered_roi_coverage": diagnostics}
    write_json(output / "information-geometry-study.json", report)
    write_csv(output / "information-geometry-development.csv", rows)
    for label, row in results.items():
        print(label, row["development"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    evaluate(args.output.resolve(), args.ffmpeg)


if __name__ == "__main__":
    main()
