from collections import deque

import numpy as np

from .simulation import box_mean


def intensity_transform(frames):
    return np.log1p(frames)


def baseline_scores(frames, method):
    local = np.asarray([box_mean(frame, 1) for frame in frames])
    reference = np.maximum(np.median(local, axis=(1, 2), keepdims=True), 1e-8)
    if method == "threshold":
        return 1 - local / reference
    score = np.zeros_like(local)
    for t in range(1, len(local)):
        history = np.max(local[max(0, t - 5):t], axis=0)
        score[t] = np.minimum((history - local[t]) / reference[t], 1 - local[t] / reference[t])
    return score


def components(mask):
    """Return four-connected foreground index arrays in row-major seed order."""
    h, w = mask.shape
    remaining = set(np.flatnonzero(mask))
    groups = []
    for seed in np.flatnonzero(mask):
        if seed not in remaining:
            continue
        remaining.remove(seed)
        pending = deque([seed])
        group = []
        while pending:
            index = pending.popleft()
            group.append(index)
            column = index % w
            neighbors = []
            if column:
                neighbors.append(index - 1)
            if column + 1 < w:
                neighbors.append(index + 1)
            if index >= w:
                neighbors.append(index - w)
            if index < (h - 1) * w:
                neighbors.append(index + w)
            for neighbor in neighbors:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
        groups.append(np.asarray(group, dtype=np.intp))
    return groups


def segment(scores, threshold, min_area):
    masks = scores >= threshold
    if min_area > 1:
        for mask in masks:
            for group in components(mask):
                if group.size < min_area:
                    mask.ravel()[group] = False
    return masks


def centroid(indices, width):
    y, x = np.divmod(indices, width)
    return np.array([np.mean(x), np.mean(y)])


def fit_velocity(times, points):
    if len(times) < 2:
        return None
    centered = np.asarray(times) - np.mean(times)
    return (centered @ np.asarray(points)) / (centered @ centered)


def track(masks, times, scale, gate, max_gap):
    tracks = []
    measured = []
    active, gap = False, 0
    segment_id = 0
    for t, (mask, time) in enumerate(zip(masks, times)):
        candidates = sorted(components(mask), key=len, reverse=True)
        centers = [centroid(group, mask.shape[1]) for group in candidates]
        point, origin = None, "missing"
        if active:
            velocity = fit_velocity(*zip(*measured[-5:]))
            last_time, last_point = measured[-1]
            expected = last_point if velocity is None else last_point + velocity * (time - last_time)
            if centers:
                best = min(centers, key=lambda center: np.linalg.norm(center - expected))
                if np.linalg.norm(best - expected) <= gate * (gap + 1):
                    point, origin = best, "measured"
            if point is None:
                gap += 1
                if gap <= max_gap and velocity is not None:
                    point, origin = expected, "imputed"
                if gap > max_gap:
                    active = False
        if not active and centers:
            segment_id += 1
            measured = []
            active = True
            point, origin = centers[0], "measured"
        if origin == "measured":
            measured.append((time, point))
            gap = 0
        tracks.append({"frame": t, "time_s": float(time), "xy_px": point.tolist() if point is not None else None,
                       "source": origin, "segment": segment_id if point is not None else None})
    segments = {}
    for row in tracks:
        if row["source"] == "measured":
            segments.setdefault(row["segment"], []).append(row)
    longest = max(segments.values(), key=len, default=[])
    velocity = fit_velocity([r["time_s"] for r in longest], [r["xy_px"] for r in longest])
    return {"frames": tracks, "velocity_segment": longest[0]["segment"] if velocity is not None else None,
            "velocity_px_s": velocity.tolist() if velocity is not None else None,
            "shadow_velocity_mps": (velocity * scale).tolist() if velocity is not None and scale is not None else None}


def score_masks(prediction, truth, match_iou):
    intersection = int(np.count_nonzero(prediction & truth))
    predicted_pixels = int(np.count_nonzero(prediction))
    truth_pixels = int(np.count_nonzero(truth))
    matched = false_positive = missed = 0
    frame_rows = []
    for mask, label in zip(prediction, truth):
        label_flat = label.ravel()
        area = int(np.count_nonzero(label))
        objects = components(mask)
        best = 0.0
        for group in objects:
            overlap = int(np.count_nonzero(label_flat[group]))
            best = max(best, overlap / (len(group) + area - overlap))
        match = int(area > 0 and best >= match_iou)
        matched += match
        false_positive += len(objects) - match
        missed += int(area > 0) - match
        overlap = int(np.count_nonzero(mask & label))
        union = int(np.count_nonzero(mask | label))
        frame_rows.append({"iou": overlap / union if union else None, "best_component_iou": best,
                           "predicted_components": len(objects), "target_visible": bool(area)})
    union = predicted_pixels + truth_pixels - intersection
    support = predicted_pixels + truth_pixels
    return {"intersection_pixels": intersection, "union_pixels": union, "predicted_pixels": predicted_pixels,
            "truth_pixels": truth_pixels, "iou": intersection / union if union else None,
            "f1": 2 * intersection / support if support else None, "matched": matched,
            "false_positive": false_positive, "missed": missed, "frames": frame_rows}


def target_summary(matched, false_positive, missed):
    positive = matched + false_positive
    visible = matched + missed
    denominator = positive + visible
    return {"precision": matched / positive if positive else None,
            "recall": matched / visible if visible else None,
            "target_f1": 2 * matched / denominator if denominator else None}


def target_motion(estimated, shape, config):
    # Midpoint between ground and the top-face projection gives an approximate shadow-to-target map.
    fraction = config["object_height_m"] / config["radar_height_m"]
    coefficient = fraction / (2 * (1 - fraction))
    times = np.asarray([row["time_s"] for row in estimated])
    shadow = np.asarray([row["xy_px"] for row in estimated])
    shadow = (shadow - (np.array(shape[::-1]) - 1) / 2) * config["pixel_size_m"]
    radar = np.column_stack((times * config["platform_speed_mps"], np.full(times.shape, -config["radar_offset_m"])))
    return (shadow + coefficient * radar) / (1 + coefficient)


def score_track(tracked, truth, scale, metadata):
    visible = truth.any(axis=(1, 2))
    truth_centers = {i: centroid(np.flatnonzero(frame), frame.shape[1]) for i, frame in enumerate(truth) if visible[i]}
    located = [row for row in tracked["frames"] if row["xy_px"] is not None and visible[row["frame"]]]
    measured = [row for row in located if row["source"] == "measured"]
    distances = [float(np.linalg.norm(np.asarray(row["xy_px"]) - truth_centers[row["frame"]])) for row in located]
    segment = tracked["velocity_segment"]
    velocity_rows = [row for row in tracked["frames"] if row["source"] == "measured" and row["segment"] == segment]
    shared = [row for row in velocity_rows if visible[row["frame"]]]
    # Refit both velocities over identical visible timestamps for the error metric.
    reference = fit_velocity([r["time_s"] for r in shared], [truth_centers[r["frame"]] for r in shared])
    predicted = fit_velocity([r["time_s"] for r in shared], [r["xy_px"] for r in shared])
    error = float(np.linalg.norm(predicted - reference)) if reference is not None else None
    result = {"visible_frames": len(truth_centers), "located_visible_frames": len(located),
              "measured_visible_frames": len(measured), "continuity": len(located) / len(truth_centers) if truth_centers else None,
              "location_rmse_px": float(np.sqrt(np.mean(np.square(distances)))) if distances else None,
              "location_squared_error_px2": float(np.sum(np.square(distances))),
              "predicted_on_invisible_frames": sum(r["xy_px"] is not None and not visible[r["frame"]] for r in tracked["frames"]),
              "velocity_error_px_s": error, "velocity_error_mps": error * scale if error is not None and scale is not None else None,
              "true_shadow_velocity_px_s": reference.tolist() if reference is not None else None}
    config = metadata.get("effective_config", {})
    true_velocity = metadata.get("target_velocity_mps")
    if len(velocity_rows) >= 2 and true_velocity is not None and scale is not None and config.get("scene") == "geometry":
        mapped = target_motion(velocity_rows, truth.shape[1:], config)
        estimated_velocity = fit_velocity([r["time_s"] for r in velocity_rows], mapped)
        difference = float(np.linalg.norm(estimated_velocity - true_velocity))
        direction_error = None
        if np.linalg.norm(estimated_velocity) > 0.1 and np.linalg.norm(true_velocity) > 0.1:
            angles = np.arctan2([estimated_velocity[1], true_velocity[1]], [estimated_velocity[0], true_velocity[0]])
            direction_error = float(abs((np.rad2deg(angles[0] - angles[1]) + 180) % 360 - 180))
        actual = np.asarray(metadata["target_centers_m"])[[r["frame"] for r in velocity_rows]]
        result.update(target_velocity_mps=estimated_velocity.tolist(), target_velocity_error_mps=difference,
                      target_direction_error_deg=direction_error,
                      target_location_rmse_m=float(np.sqrt(np.mean(np.sum((mapped - actual)**2, axis=1)))))
    return result
