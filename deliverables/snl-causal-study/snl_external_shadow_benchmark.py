"""Measure injected-shadow detection on three disjoint real VideoSAR backgrounds."""
import argparse
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from experiment import quality
from infer_video import DEFAULT_MODEL, find_ffmpeg, torch
from snl_detector import TILE, load_trained, nonmax, predict_frame
from snl_external_background_transfer import tone_map
from snl_inputs import DEFAULT_OUTPUT, PROFILES, load_gray, load_labels, read_json, sha_file, write_csv, write_json
from snl_motion_aug import measurements, shape_on_tile
from snl_qatar_pilot import EXTERNAL, PINNED_MODEL, decode, observations
from videosar.data import run_directory
from videosar.snl_causal import STACKED_HISTORY_LAGS, box_iou, detection_metrics


CALIBRATION = "Rotterdam"
HELD_OUT = ("Qatar", "Australia")
POSITIONS = (0, 264, 528)
OBSERVATIONS = (8, 10, 12, 14, 16, 18, 21, 24)
DEPTHS = (0, 5, 15)
SCALES = (1.0, 1.25, 1.50)
THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.55)
SEED = 21072


def bank(frames, sizes, displacements, *, placements=None, seed=SEED, observation_order=OBSERVATIONS, strong_overrides=None):
    rng = np.random.default_rng(seed)
    if placements is None:
        placements = product(POSITIONS, POSITIONS, OBSERVATIONS)
    overrides = strong_overrides or {}
    for group, (x, y, index) in enumerate(placements):
        lags = np.where(np.asarray((0, *STACKED_HISTORY_LAGS)) <= index, (0, *STACKED_HISTORY_LAGS), 0)
        background = frames[index - lags, y:y + TILE, x:x + TILE]
        scale = 0.72 if (x // 264 + y // 264 + observation_order.index(index)) % 2 else 1.0
        half = np.clip(sizes[rng.integers(len(sizes))] * scale / 2, 2, 20)
        angle = rng.uniform(-0.25, 0.25)
        delta = displacements[rng.integers(len(displacements))]
        shifts = lags[:, None] * delta
        border = int(np.ceil(np.linalg.norm(half))) + 6
        low = border + np.maximum(0, shifts.max(axis=0))
        high = TILE - border + np.minimum(0, shifts.min(axis=0))
        center = rng.uniform(low, high)
        projected = np.abs([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]) @ half
        box = np.r_[center - projected, center + projected].astype(np.float32).reshape(1, 4)
        spec = dict(tile_x=x, tile_y=y, observation=index, size_scale=scale, box=box[0].tolist())
        for depth in (*DEPTHS[:2], overrides.get(group, DEPTHS[2])):
            label = box if depth else np.empty((0, 4), dtype=np.float32)
            image = background.astype(np.float32) / 255
            if depth:
                for frame, shift in zip(image, shifts, strict=True):
                    shape_on_tile(frame, center - shift, half, angle, depth / 255)
            yield spec | {"injected_depth": depth}, np.rint(255 * image).astype(np.uint8), label


def take_predictions(model, image, mapping, size, nms):
    if mapping is not None:
        image = mapping[image]
    if size != TILE:
        image = np.stack([cv2.resize(channel, (size, size), interpolation=cv2.INTER_LINEAR) for channel in image])
    boxes, scores = nonmax(*predict_frame(model, image, "temporal"), nms)
    return boxes * np.float32(TILE / size), scores


def score_cases(cases, predictions, threshold, mode):
    rows = []
    for spec, truth, (boxes, scores) in zip((row[0] for row in cases), (row[2] for row in cases), predictions, strict=True):
        predicted = boxes[scores >= threshold]
        if mode == "iou50":
            found = bool((box_iou(predicted, truth) >= 0.50).any())
        else:
            x = (predicted[:, :2] + predicted[:, 2:]) / 2
            y = (truth[:, :2] + truth[:, 2:]) / 2
            found = bool((np.square(x[:, None] - y).sum(axis=2) <= 25).any())
        rows.append(dict(spec, tp=int(found), fp=len(predicted) - int(found), fn=len(truth) - int(found)))
    return rows


def metrics(rows):
    return detection_metrics((sum(row[key] for row in rows) for key in ("tp", "fp", "fn")), len(rows))


def summarize(cases, predictions, threshold):
    iou = score_cases(cases, predictions, threshold, "iou50")
    centers = score_cases(cases, predictions, threshold, "center5")
    by_depth = {str(depth): metrics([row for row in iou if row["injected_depth"] == depth]) for depth in DEPTHS}
    return dict(iou50=metrics(iou), center5=metrics(centers), by_depth=by_depth), iou


def open_site(site, ffmpeg, target, sizes, displacements):
    source = EXTERNAL / "vsdformer" / f"Noisy_{site}.mp4"
    frames, ids, timing = observations(*decode(source, ffmpeg), minimum_transition=2.0)
    mapping, bounds = tone_map(frames[:min(OBSERVATIONS)], target)
    lut = np.rint(255 * mapping).astype(np.uint8)
    cases = list(bank(frames, sizes, displacements))
    receipt = dict(video_sha256=sha_file(source), representative_playback_frames=ids, transition_check=timing,
                   prefix_observations=min(OBSERVATIONS), prefix_quantiles=bounds.tolist())
    return cases, lut, receipt


def predict(model, cases, tone, scale, lut, nms):
    mapping = lut if tone == "prefix-tone" else None
    return [take_predictions(model, image, mapping, int(TILE * scale), nms) for _, image, _ in cases]


def compare(output, ffmpeg):
    source = DEFAULT_OUTPUT / "prospective"
    train_ids = PROFILES["prospective"][0]
    gray = load_gray(source, ffmpeg)
    training = load_labels(source, train_ids, gray)[0]
    sizes, _, displacements = measurements(gray, training)
    target = np.percentile(gray[np.array(train_ids[::7]) - 2, ::4, ::4], (5, 50, 95))
    config = read_json(DEFAULT_MODEL)
    if (config["sha256"], config["threshold"], config["nms"], config["low"]) != (PINNED_MODEL, 0.25, 0.35, None):
        raise ValueError("The frozen SNL delivery recipe changed")
    model = load_trained(DEFAULT_MODEL.parent, config)
    settings = dict(model_sha256=PINNED_MODEL, seed=SEED, matching_iou=0.50, nms=config["nms"], nominal_depths=DEPTHS,
                    calibration_scene=CALIBRATION, held_out_scenes=HELD_OUT, observations=OBSERVATIONS,
                    spatial_tiles=POSITIONS, tone_target_from_snl_training=target.tolist())
    with run_directory(output, "snl-external-shadow-benchmark", settings) as root:
        cases, mapping, calibration = open_site(CALIBRATION, ffmpeg, target, sizes, displacements)
        trials = []
        for tone, scale in product(("raw", "prefix-tone"), SCALES):
            predictions = predict(model, cases, tone, scale, mapping, config["nms"])
            for threshold in THRESHOLDS:
                rows = score_cases(cases, predictions, threshold, "iou50")
                trials.append(dict(tone=tone, scale=scale, threshold=threshold, calibration=metrics(rows)))
        raw_trials = [trial for trial in trials if (trial["tone"], trial["scale"]) == ("raw", 1)]
        selected = max(trials, key=lambda trial: quality(trial["calibration"]))
        raw = next(trial for trial in raw_trials if trial["threshold"] == config["threshold"])
        best_raw = max(raw_trials, key=lambda trial: quality(trial["calibration"]))
        arms = dict(frozen=raw, threshold_only=best_raw, cross_scene_selected=selected)
        write_json(root / "frozen-selection.json", dict(settings=settings, calibration=calibration, arms=arms, trials=trials))
        print("Frozen on Rotterdam", arms, flush=True)

        results, all_rows = {}, []
        for site in HELD_OUT:
            cases, mapping, receipt = open_site(site, ffmpeg, target, sizes, displacements)
            evaluated = {}
            raw_by_variant = {}
            for name, arm in arms.items():
                variant = arm["tone"], arm["scale"]
                if variant not in raw_by_variant:
                    raw_by_variant[variant] = predict(model, cases, *variant, mapping, config["nms"])
                evaluated[name], rows = summarize(cases, raw_by_variant[variant], arm["threshold"])
                all_rows.extend(dict(site=site, arm=name, **row) for row in rows)
                print(site, name, evaluated[name]["iou50"], flush=True)
            results[site] = dict(source=receipt, arms=evaluated)
        pooled = {name: metrics([row for row in all_rows if row["arm"] == name]) for name in arms}
        write_json(root / "comparison.json", dict(settings=settings, frozen_arms=arms, results=results, pooled_iou50=pooled))
        write_csv(root / "case-rows.csv", all_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXTERNAL / "injected-shadow-benchmark")
    parser.add_argument("--ffmpeg")
    args = parser.parse_args()
    cv2.setNumThreads(3)
    torch.set_num_threads(4)
    compare(args.output.resolve(), find_ffmpeg(args.ffmpeg))
