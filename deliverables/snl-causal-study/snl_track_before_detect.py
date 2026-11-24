"""Compare causal continuous-frame shadow accumulation and online background updates."""
import argparse
from collections import deque
from itertools import product
from pathlib import Path
from warnings import catch_warnings, simplefilter

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from audit_detection import paired_block_f1
from experiment import filter_by_config, load_candidates, with_nms
from route_prior import centers
from snl_bytetrack import REFERENCE
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, VIDEO, WIDTH, load_gray, load_labels
from snl_inputs import read_json, sha_file, write_csv, write_json
from snl_registered_route import chained_registration, project
from videosar.analysis import fit_velocity
from videosar.data import run_directory
from videosar.snl_causal import detection_counts, detection_metrics


ANCHOR = 242
INITIAL_FRAMES = 48
WINDOW = 20
STRIDE = 2
BOX_XY = (8, 15)
SPEEDS = ((0., 0.), *(tuple(v) for v in product(np.arange(-4, 4.01, .5), repeat=2)
                        if 0 < np.linalg.norm(v) <= 4.25))
ALPHA = 0.035
CENTER_GATE = 5
SPEED_WINDOW = 12
SHORT_SPEED_WINDOW = 5
BUDGET = 256
ARMS = tuple(f"{background}_{mode}" for background in ("fixed", "online") for mode in
             ("single", "static", "tbd", "tbd_motion"))
HEAT_ARMS = ("heat_single", "heat_static", "heat_tbd", "heat_tbd_motion", "heat_tbd_current")


def shift_add(destination, source, offset, sign):
    dx, dy = map(int, offset)
    height, width = destination.shape[-2:]
    if abs(dx) >= width or abs(dy) >= height:
        return
    dy_dst, dx_dst = max(dy, 0), max(dx, 0)
    dy_src, dx_src = max(-dy, 0), max(-dx, 0)
    size_y, size_x = height - abs(dy), width - abs(dx)
    window = destination[..., dy_dst:dy_dst + size_y, dx_dst:dx_dst + size_x]
    aligned = source[..., dy_src:dy_src + size_y, dx_src:dx_src + size_x]
    if sign > 0:
        window += aligned
    else:
        window -= aligned


class PathAccumulator:
    def __init__(self, speeds, window):
        self.speeds = np.asarray(speeds, dtype=np.float32)
        self.window = window
        self.history = deque(maxlen=window)
        self.index = 0
        self.total = self.support = None

    def advance(self, evidence, valid):
        current = np.rint(self.index * self.speeds / STRIDE).astype(np.int32)
        previous = np.rint((self.index - 1) * self.speeds / STRIDE).astype(np.int32)
        new = np.broadcast_to(np.where(valid, evidence, 0), (len(self.speeds), *evidence.shape)).copy()
        counts = np.broadcast_to(valid.astype(np.int16), (len(self.speeds), *valid.shape)).copy()
        if self.total is not None:
            for row, offset in enumerate(current - previous):
                shift_add(new[row], self.total[row], offset, 1)
                shift_add(counts[row], self.support[row], offset, 1)
        if len(self.history) == self.window:
            old, usable, position = self.history[0]
            for row, offset in enumerate(current - position):
                shift_add(new[row], old, offset, -1)
                shift_add(counts[row], usable, offset, -1)
        self.total, self.support = new, counts
        self.history.append((np.where(valid, evidence, 0), valid.astype(np.int16), current))
        self.index += 1

    def scores(self):
        ready = self.support == self.window
        result = []
        for values in self.total.transpose(1, 0, 2, 3):
            paths = np.where(ready, values / self.window, -np.inf)
            best = paths[1:].argmax(axis=0) + 1
            moving = np.take_along_axis(paths, best[None], axis=0)[0]
            stationary = paths[0]
            adjusted = np.full_like(moving, -np.inf)
            valid = np.isfinite(moving) & np.isfinite(stationary)
            adjusted[valid] = moving[valid] - 0.5 * np.maximum(stationary[valid], 0)
            result.append((stationary, moving, adjusted, best))
        return result


def warp(image, matrix):
    shape = image.shape[::-1]
    pixels = cv2.warpAffine(image.astype(np.float32) / 255, matrix, shape, flags=cv2.INTER_LINEAR)
    footprint = cv2.warpAffine(np.ones_like(image, dtype=np.float32), matrix, shape, flags=cv2.INTER_LINEAR)
    return pixels, footprint >= 1 - 1e-6


def initialize(registered):
    stack = np.stack([np.where(valid, image, np.nan) for image, valid in registered])
    exists = np.isfinite(stack).mean(axis=0) >= 0.80
    with catch_warnings():
        simplefilter("ignore", RuntimeWarning)
        background = np.nanmedian(stack, axis=0)
        noise = 1.4826 * np.nanmedian(abs(stack - background), axis=0)
    return np.where(exists, background, 0), np.where(exists, np.maximum(noise, 0.016), 0.016), exists


class Background:
    def __init__(self, registered):
        background, noise, self.exists = initialize(registered)
        self.estimate = np.stack((background, background))
        self.noise = np.stack((noise, noise))

    def advance(self, image, valid):
        usable = self.exists & valid
        # Require full spatial kernel support before using a residual as path evidence.
        smooth_safe = cv2.erode(usable.astype(np.uint8), np.ones((29, 25), np.uint8),
                                borderType=cv2.BORDER_CONSTANT, borderValue=0)
        kernel_shape = (image.shape[1] // STRIDE, image.shape[0] // STRIDE)
        safe = cv2.resize(smooth_safe.astype(np.float32), kernel_shape, interpolation=cv2.INTER_AREA) == 1
        evidence = []
        difference = image - self.estimate
        for slot in range(len(self.estimate)):
            sampled = usable[::4, ::4]
            delta = difference[slot]
            delta -= np.median(delta[::4, ::4][sampled]) if sampled.any() else 0
            delta[~usable] = 0
            local = cv2.blur(delta, (5, 9))
            surround = cv2.blur(delta, (25, 29))
            spread = cv2.blur(self.noise[slot], (5, 9))
            signed = np.clip((surround - local) / spread, -4, 4)
            evidence.append(cv2.resize(signed, kernel_shape, interpolation=cv2.INTER_AREA))
        value, scale = difference[1], self.noise[1]
        self.estimate[1] += ALPHA * np.clip(value, -2 * scale, 3 * scale)
        change = np.clip(1.2533 * abs(value), 0, 4 * scale) - scale
        self.noise[1] = np.where(usable, np.maximum(0.016, scale + ALPHA * change), scale)
        return np.stack(evidence), safe


def local_candidates(score, velocity, inverse, shape):
    valid = np.isfinite(score)
    field = np.where(valid, score, -1e6).astype(np.float32)
    peaks = valid & (field == cv2.dilate(field, np.ones((5, 5), dtype=np.uint8)))
    y, x = np.where(peaks)
    order = np.argsort(-field[y, x], kind="stable")
    x, y = x[order], y[order]
    positions = STRIDE * (np.column_stack((x, y)) + 0.5) - 0.5
    positions = project(positions, inverse).astype(np.float32)
    limits = np.array(shape[::-1])
    keep = np.flatnonzero(((positions >= 0) & (positions < limits)).all(axis=1))[:BUDGET]
    value = field[y[keep], x[keep]]
    vector = np.asarray(SPEEDS)[velocity[y[keep], x[keep]]] if velocity is not None else None
    return positions[keep], value, vector


class Stream:
    def __init__(self, gray, matrices, start, window=WINDOW, registered=True, initial_frames=INITIAL_FRAMES, heat=None):
        self.gray, self.matrices, self.start = gray, matrices, start
        self.registered = registered
        self.heat = heat
        self.paths = PathAccumulator(SPEEDS, window)
        initial = [self.frame(index) for index in range(start, start + initial_frames)]
        self.background = Background(initial)
        self.next = start + initial_frames

    def frame(self, index):
        image = self.gray[index - self.start]
        if self.registered:
            return warp(image, self.matrices[index])
        return image.astype(np.float32) / 255, np.ones_like(image, dtype=bool)

    def run(self, ids):
        names = ARMS + (HEAT_ARMS if self.heat is not None else ())
        predictions = {name: {} for name in names}
        support = []
        wanted = set(ids)
        for index in range(self.next, max(ids) + 1):
            image, valid = self.frame(index)
            evidence, safe = self.background.advance(image, valid)
            if self.heat is not None:
                evidence = np.concatenate((evidence, self.heat(index, self.matrices[index], safe)[None]))
            self.paths.advance(evidence, safe)
            if index not in wanted:
                continue
            inverse = cv2.invertAffineTransform(self.matrices[index]) if self.registered else np.eye(2, 3)
            for slot, (static, tbd, motion, velocity) in enumerate(self.paths.scores()):
                prefix = ("fixed", "online", "heat")[slot]
                common = safe & np.isfinite(static) & np.isfinite(tbd)
                comparison = [("single", evidence[slot], None), ("static", static, None),
                              ("tbd", tbd, velocity), ("tbd_motion", motion, velocity)]
                if prefix == "heat":
                    comparison.append(("tbd_current", 0.5 * (tbd + evidence[slot]), velocity))
                for suffix, score, vector in comparison:
                    name = f"{prefix}_{suffix}"
                    score = np.where(common, score, -np.inf)
                    predictions[name][index] = local_candidates(score, vector, inverse, image.shape)
                    support.append(dict(annotation_id=index, arm=name, supported_cells=int(np.isfinite(score).sum()),
                                        candidates=len(predictions[name][index][0])))
        self.next = max(ids) + 1
        return predictions, support


def center_pairs(predicted, truth):
    distance = np.linalg.norm(predicted[:, None] - truth[None], axis=2)
    valid = distance <= CENTER_GATE
    reward = valid * (1 + (CENTER_GATE - distance) / (CENTER_GATE * (1 + min(distance.shape))))
    left, right = linear_sum_assignment(reward, maximize=True)
    keep = valid[left, right]
    return left[keep], right[keep]


def paired_center_change(predicted, blank, truth_centers):
    detected = set(center_pairs(predicted, truth_centers)[1])
    before = set(center_pairs(blank, truth_centers)[1])
    return dict(blank_path_hits=len(before), shared_hits=len(detected & before),
                new_hits=len(detected - before), lost_hits=len(before - detected))


def boxes_at(points, shape):
    height, width = shape
    half = np.asarray(BOX_XY) / 2
    return np.column_stack((points - half, points + half)).clip(0, (width, height, width, height))


def thresholded(predictions, threshold=None, counts=None):
    output = {}
    for index, (positions, likelihood, vectors) in predictions.items():
        selected = (int((likelihood >= threshold).sum()) if counts is None else counts[index])
        output[index] = (positions[:selected], likelihood[:selected], None if vectors is None else vectors[:selected])
    return output


def velocity_references(labels, matrices, window=SPEED_WINDOW):
    previous, chains, references = None, [], {}
    for index, boxes in labels.items():
        points = project(centers(boxes), matrices[index])
        tails = {j: [point] for j, point in enumerate(points)}
        if previous is not None and index == previous[0] + 1:
            distance = np.linalg.norm(previous[1][:, None] - points[None], axis=2)
            possible = distance <= 9
            unique = possible & (possible.sum(0) == 1)[None] & (possible.sum(1) == 1)[:, None]
            for before, after in zip(*np.where(unique), strict=True):
                tails[after] = [*chains[before][-(window - 1):], points[after]]
        velocity = np.full((len(points), 2), np.nan)
        for number, path in tails.items():
            if len(path) == window:
                times = np.arange(window)
                vector = fit_velocity(times, path)
                fits = np.mean(path, axis=0) + (times - times.mean())[:, None] * vector
                rms = np.sqrt(np.mean(np.sum((path - fits)**2, axis=1)))
                if rms <= 2:
                    velocity[number] = vector
        previous, chains = (index, points), tails
        references[index] = velocity
    return references


def evaluate(predictions, labels, shape, *, original_boxes=None, truth_velocity=None):
    records, matched_velocity = [], []
    for index, boxes in labels.items():
        points, scores, velocity = predictions[index]
        left, right = center_pairs(points, centers(boxes))
        iou = detection_counts(original_boxes[index] if original_boxes is not None else boxes_at(points, shape), boxes)
        records.append(dict(annotation_id=index, tp=len(left), fp=len(points) - len(left), fn=len(boxes) - len(left),
                            iou_tp=iou[0], iou_fp=iou[1], iou_fn=iou[2]))
        if velocity is None or truth_velocity is None:
            continue
        for a, b in zip(left, right, strict=True):
            truth = truth_velocity[index][b]
            if np.isfinite(truth).all():
                matched_velocity.append(dict(annotation_id=index, truth_box_index=int(b), x=float(points[a, 0]),
                                             y=float(points[a, 1]), score=float(scores[a]), vx=float(velocity[a, 0]),
                                             vy=float(velocity[a, 1]), reference_vx=float(truth[0]),
                                             reference_vy=float(truth[1]), vector_error=float(np.linalg.norm(velocity[a] - truth)),
                                             speed_error=float(abs(np.linalg.norm(velocity[a]) - np.linalg.norm(truth)))))
    center = detection_metrics([sum(r[k] for r in records) for k in ("tp", "fp", "fn")], len(records))
    iou = detection_metrics([sum(r[k] for r in records) for k in ("iou_tp", "iou_fp", "iou_fn")], len(records))
    result = dict(center5=center, iou50=iou)
    if truth_velocity is not None:
        result["velocity"] = dict(proxy_targets=sum(int(np.isfinite(v).all(axis=1).sum()) for v in truth_velocity.values()),
                                  matched=len(matched_velocity), units="registered pixels per observation",
                                  speed_mae=(float(np.mean([r["speed_error"] for r in matched_velocity]))
                                             if matched_velocity else None),
                                  mean_vector_error=(float(np.mean([r["vector_error"] for r in matched_velocity]))
                                                     if matched_velocity else None))
    return result, records, matched_velocity


def select(predictions, labels, shape):
    points = {}
    for name, frames in predictions.items():
        peak_scores = np.concatenate([row[1] for row in frames.values()])
        if not len(peak_scores):
            raise ValueError(f"{name} had no complete supported calibration paths")
        thresholds = np.unique(np.quantile(peak_scores, np.r_[np.linspace(0.75, 0.99, 25), 0.995, 1.0]))
        trials = [dict(threshold=float(t), **evaluate(thresholded(frames, t), labels, shape)[0]) for t in thresholds]
        points[name] = max(trials, key=lambda row: (row["center5"]["f1"] or 0, row["center5"]["precision"] or 0))
    return points


def reference_predictions(source, scope):
    config = read_json(source / REFERENCE)["config"]
    raw = load_candidates(source, config, scope)
    selected = filter_by_config(with_nms(raw, config["nms"]), config)
    return {i: (centers(b), s, None) for i, (b, s, _) in selected.items()}, {i: b for i, (b, _, _) in selected.items()}


class DenseHeat:
    def __init__(self, source, gray, last):
        from snl_detector import load_trained, torch
        from snl_inputs import feature_cache
        from videosar.snl_causal import CausalFeatures

        self.config = read_json(source / REFERENCE)["config"]
        self.model = load_trained(source, self.config)
        self.frames = feature_cache(source, gray, last, CausalFeatures("stacked"))
        self.torch = torch
        self.device = next(self.model.parameters()).device

    def __call__(self, index, transform, mask):
        from snl_center import predict_views

        image = self.torch.from_numpy(self.frames[index - 2].copy()).to(self.device).float()[None] / 255
        with self.torch.inference_mode():
            logits = predict_views(self.model, image)[0, 0].cpu().numpy()
        half = transform.copy()
        half[:, 2] = (project(np.full((1, 2), 0.5), transform)[0] - 0.5) / STRIDE
        aligned = cv2.warpAffine(logits, half, mask.shape[::-1], flags=cv2.INTER_LINEAR)
        median = np.median(aligned[mask]) if mask.any() else 0
        return np.clip(aligned - median, -6, 6)


def detection_rows(predictions, scope, arm):
    for index, (positions, scores, velocity) in predictions.items():
        for j, (xy, score) in enumerate(zip(positions, scores, strict=True)):
            vector = (None, None) if velocity is None else velocity[j].tolist()
            yield dict(split=scope, arm=arm, annotation_id=index, x=float(xy[0]), y=float(xy[1]), score=float(score),
                       vx_registered=vector[0], vy_registered=vector[1])


def transform_prefix(gray, end):
    cv2.setRNGSeed(1729)
    matrices, counts = chained_registration(gray, ANCHOR, range(ANCHOR, end + 1))
    missing = sorted(set(range(ANCHOR, end + 1)) - matrices.keys())
    if missing:
        raise ValueError(f"Registration missing for observation(s) {missing}; complete paths require real samples")
    return matrices, counts


def run(source, output, ffmpeg, *, learned_heat=False, window=WINDOW):
    _, calibration, development = PROFILES["prospective"]
    available = min(calibration) - (ANCHOR + INITIAL_FRAMES) + 1
    if not 2 <= window <= available:
        raise ValueError(f"The first calibration frame supports a window of 2–{available} observations")
    gray = load_gray(source, ffmpeg)
    span = gray[ANCHOR - 2:max(development) - 1]
    shape = span[0].shape
    heat = DenseHeat(source, gray, max(development)) if learned_heat else None
    settings = dict(anchor=ANCHOR, initial_frames=INITIAL_FRAMES, window=window, stride=STRIDE, alpha=ALPHA,
                    speeds=SPEEDS, fixed_box_xy=BOX_XY, speed_windows=(SPEED_WINDOW, SHORT_SPEED_WINDOW),
                    source_sha256=sha_file(VIDEO),
                    calibration_ids=calibration, development_ids=development, budget=BUDGET,
                    heat_checkpoint=heat.config["sha256"] if heat is not None else None)
    with run_directory(output, "snl-track-before-detect", settings) as root:
        initial_matrices, _ = transform_prefix(gray, max(calibration))
        stream = Stream(span, initial_matrices, ANCHOR, window=window, heat=heat)
        cal_predictions, cal_support = stream.run(calibration)
        cal_labels = load_labels(source, calibration, gray)[0]
        points = select(cal_predictions, cal_labels, shape)
        write_json(root / "frozen-selection.json", dict(settings=settings, points=points))
        print("calibration frozen", {k: round(v["center5"]["f1"] * 100, 2) for k, v in points.items()}, flush=True)

        full_matrices, inliers = transform_prefix(gray, max(development))
        for index, matrix in initial_matrices.items():
            np.testing.assert_array_equal(matrix, full_matrices[index])
        stream.matrices = full_matrices
        dev_predictions, dev_support = stream.run(development)
        dev_labels = load_labels(source, development, gray)[0]
        results, frame_rows, speed_rows, emitted = {}, [], [], []
        for scope, (labels, predicted) in (("validation", (cal_labels, cal_predictions)),
                                           ("development", (dev_labels, dev_predictions))):
            reference, ref_boxes = reference_predictions(source, scope)
            base, base_rows, _ = evaluate(reference, labels, shape, original_boxes=ref_boxes)
            fixed = {i: len(reference[i][0]) for i in labels}
            if any(len(frames[i][0]) < count for frames in predicted.values() for i, count in fixed.items()):
                raise ValueError("Incomplete candidate budget for a matched-output comparison")
            truth_velocity = {window: velocity_references(labels, full_matrices, window)
                              for window in (SPEED_WINDOW, SHORT_SPEED_WINDOW)}
            results[scope] = {"release": base}
            frame_rows.extend(dict(split=scope, arm="release", protocol="frozen", **r) for r in base_rows)
            for name, arm in predicted.items():
                results[scope][name] = {}
                for protocol, selected in (("frozen", thresholded(arm, points[name]["threshold"])),
                                            ("equal_output", thresholded(arm, counts=fixed))):
                    score, rows, _ = evaluate(selected, labels, shape)
                    if any(record[2] is not None for record in selected.values()):
                        score["velocity"] = {}
                        for window, proxy in truth_velocity.items():
                            scored, _, speed = evaluate(selected, labels, shape, truth_velocity=proxy)
                            score["velocity"][window] = scored["velocity"]
                            speed_rows.extend(dict(split=scope, arm=name, protocol=protocol, speed_proxy_frames=window, **r)
                                              for r in speed)
                    if protocol == "frozen":
                        emitted.extend(detection_rows(selected, scope, name))
                    frame_rows.extend(dict(split=scope, arm=name, protocol=protocol, **r) for r in rows)
                    score["center_f1_vs_release"] = paired_block_f1(rows, base_rows, block=WINDOW)
                    results[scope][name][protocol] = score
                print(scope, name, {k: round(v["center5"]["f1"] * 100, 2) for k, v in results[scope][name].items()}, flush=True)
        write_json(root / "comparison.json", dict(settings=settings, points=points, results=results,
                                                     prefix_registration_exact=True, registration_inliers=inliers))
        write_csv(root / "frame-rows.csv", frame_rows)
        write_csv(root / "support-rows.csv", cal_support + dev_support)
        if emitted:
            write_csv(root / "center-predictions.csv", emitted)
        if speed_rows:
            write_csv(root / "matched-velocities.csv", speed_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--learned-heat", action="store_true")
    parser.add_argument("--window", type=int, default=WINDOW, help="Consecutive observations in each accumulation path")
    args = parser.parse_args()
    cv2.setNumThreads(4)
    default = "track-before-detect-heat" if args.learned_heat else "track-before-detect"
    if args.window != WINDOW:
        default += f"-window-{args.window}"
    run(args.source.resolve(), (args.output or args.source / default).resolve(), args.ffmpeg,
        learned_heat=args.learned_heat, window=args.window)
