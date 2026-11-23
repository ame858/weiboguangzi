"""Compare equal-budget SNL initialization and scratch fitting on injected Rotterdam shadows."""
import argparse
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from experiment import CONFIDENCES, quality
from infer_video import DEFAULT_MODEL, find_ffmpeg, torch
from snl_center import train_center
from snl_detector import TILE, load_trained
from snl_external_shadow_benchmark import CALIBRATION, HELD_OUT, OBSERVATIONS, POSITIONS, bank, metrics, open_site
from snl_external_shadow_benchmark import predict, score_cases, summarize
from snl_inputs import DEFAULT_OUTPUT, PROFILES, load_gray, load_labels, read_json, sha_file, write_csv, write_json
from snl_motion_aug import measurements
from snl_qatar_pilot import EXTERNAL, PINNED_MODEL, decode, observations
from videosar.data import run_directory


TRAINING_TRIPLETS = 160
TRAINING_SEED = 71521
EPOCHS = 12
CALIBRATION_X = POSITIONS[-1]
OUTPUT = EXTERNAL / "injected-shadow-adaptation"


def training_placements():
    rng = np.random.default_rng(TRAINING_SEED)
    return [(int(rng.integers(0, CALIBRATION_X - TILE + 1)), int(rng.integers(0, 720 - TILE + 1)), int(index))
            for index in rng.choice(OBSERVATIONS, TRAINING_TRIPLETS)]


def fit_samples(sizes, velocities, ffmpeg, *, strong_overrides=None):
    path = EXTERNAL / "vsdformer" / f"Noisy_{CALIBRATION}.mp4"
    frames, _, _ = observations(*decode(path, ffmpeg), minimum_transition=2)
    placements = training_placements()
    assert max(x + TILE for x, _, _ in placements) <= CALIBRATION_X
    fitting = list(bank(frames, sizes, velocities, placements=placements, seed=TRAINING_SEED, strong_overrides=strong_overrides))
    calibration = list(bank(frames, sizes, velocities, placements=product((CALIBRATION_X,), POSITIONS, OBSERVATIONS)))
    return fitting, calibration, sha_file(path)


def fit_model(folder, fitting, calibration, baseline, prior=None):
    folder.mkdir()
    data = np.stack([row[1] for row in fitting])
    labels = {index + 2: row[2] for index, row in enumerate(fitting)}
    config, = list(train_center(folder, data, labels, input_profile="stacked", pretrained=prior, epoch_budget=EPOCHS))
    config |= {"contrast_gain": baseline["contrast_gain"]}
    model = load_trained(folder, config)
    predictions = predict(model, calibration, "raw", 1, None, baseline["nms"])
    scored = [(float(t), metrics(score_cases(calibration, predictions, float(t), "iou50"))) for t in CONFIDENCES]
    threshold, result = max(scored, key=lambda row: quality(row[1]))
    trials = [dict(threshold=value, metrics=measurement) for value, measurement in scored]
    return dict(config=config, model_dir=str(folder), threshold=threshold, calibration=result), trials


def compare(output, ffmpeg):
    source = DEFAULT_OUTPUT / "prospective"
    train_ids = PROFILES["prospective"][0]
    gray = load_gray(source, ffmpeg)
    training = load_labels(source, train_ids, gray)[0]
    sizes, _, velocities = measurements(gray, training)
    target_tone = np.percentile(gray[np.array(train_ids[::7]) - 2, ::4, ::4], (5, 50, 95))
    baseline = read_json(DEFAULT_MODEL)
    if (baseline["sha256"], baseline["threshold"], baseline["nms"], baseline["low"]) != (PINNED_MODEL, 0.25, 0.35, None):
        raise ValueError("The frozen delivery experiment used a different operation point")
    initialized = baseline | {"path": str(DEFAULT_MODEL.with_name(baseline["path"]))}
    fitting, calibration, video_hash = fit_samples(sizes, velocities, ffmpeg)
    settings = dict(source_site=CALIBRATION, source_sha256=video_hash, calibration_tile_x=CALIBRATION_X, training_frames=len(fitting),
                    calibration_frames=len(calibration), held_out_sites=HELD_OUT, initialization_sha256=PINNED_MODEL,
                    training_seed=TRAINING_SEED, epoch_budget=EPOCHS, match_iou=0.50)
    with run_directory(output, "snl-external-shadow-adaptation", settings) as root:
        trials = {}
        arms = {"frozen": dict(config=baseline, model_dir=str(DEFAULT_MODEL.parent), threshold=baseline["threshold"])}
        for name, prior in (("scratch", None), ("snl_initialized", initialized)):
            arms[name], trials[name] = fit_model(root / name, fitting, calibration, baseline, prior)
            print(f"Frozen {name} on Rotterdam: {arms[name]['threshold']} {arms[name]['calibration']}", flush=True)
        write_json(root / "frozen-selection.json", dict(settings=settings, arms=arms, trials=trials))

        results, rows = {}, []
        models = {name: load_trained(Path(arm["model_dir"]), arm["config"]) for name, arm in arms.items()}
        for scene in HELD_OUT:
            cases, _, receipt = open_site(scene, ffmpeg, target_tone, sizes, velocities)
            evaluated = {}
            for name, member in arms.items():
                predictions = predict(models[name], cases, "raw", 1, None, baseline["nms"])
                evaluated[name], counts = summarize(cases, predictions, member["threshold"])
                rows.extend(dict(site=scene, arm=name, **row) for row in counts)
                print(scene, name, evaluated[name]["iou50"], flush=True)
            results[scene] = dict(source=receipt, arms=evaluated)
        pooled = {name: metrics([row for row in rows if row["arm"] == name]) for name in arms}
        write_json(root / "comparison.json", dict(settings=settings, frozen_arms=arms, results=results, pooled_iou50=pooled))
        write_csv(root / "case-rows.csv", rows)


def compare_mixed(output, ffmpeg, reference):
    prior_path = reference / "comparison.json"
    prior = read_json(prior_path)
    source = DEFAULT_OUTPUT / "prospective"
    train_ids = PROFILES["prospective"][0]
    gray = load_gray(source, ffmpeg)
    sizes, _, velocities = measurements(gray, load_labels(source, train_ids, gray)[0])
    target_tone = np.percentile(gray[np.array(train_ids[::7]) - 2, ::4, ::4], (5, 50, 95))
    base = read_json(DEFAULT_MODEL)
    fitting, calibration, rotterdam_hash = fit_samples(sizes, velocities, ffmpeg)
    if rotterdam_hash != prior["settings"]["source_sha256"] or len(fitting) != prior["settings"]["training_frames"]:
        raise ValueError("The single-background fitting recipe changed")
    split = len(fitting) // 2
    settings = dict(total_training_frames=len(fitting), frames_per_training_scene=split, epoch_budget=EPOCHS,
                    box_recipe_unchanged=True, calibration_scene=CALIBRATION, calibration_tile_x=CALIBRATION_X,
                    single_background_result_sha256=sha_file(prior_path))
    with run_directory(output, "snl-external-shadow-mixed", settings) as root:
        arms = {}
        for held_out, partner in (HELD_OUT, HELD_OUT[::-1]):
            path = EXTERNAL / "vsdformer" / f"Noisy_{partner}.mp4"
            frames, _, _ = observations(*decode(path, ffmpeg), minimum_transition=2)
            cases = list(bank(frames, sizes, velocities, placements=training_placements(), seed=TRAINING_SEED))
            if [record[0] for record in cases] != [record[0] for record in fitting]:
                raise ValueError("The two fitting banks use different geometry")
            arm, trials = fit_model(root / held_out, fitting[:split] + cases[split:], calibration, base)
            arms[held_out] = arm | {"training_scenes": [CALIBRATION, partner], "training_video_sha256": [rotterdam_hash, sha_file(path)],
                                    "threshold_trials": trials}
            print("Frozen mixed training for", held_out, arm["threshold"], arm["calibration"], flush=True)
        write_json(root / "frozen-selection.json", dict(settings=settings, arms=arms))
        results, rows = {}, []
        for held_out, arm in arms.items():
            cases, _, receipt = open_site(held_out, ffmpeg, target_tone, sizes, velocities)
            if receipt != prior["results"][held_out]["source"]:
                raise ValueError(f"Held-out data changed for {held_out}")
            model = load_trained(Path(arm["model_dir"]), arm["config"])
            predictions = predict(model, cases, "raw", 1, None, base["nms"])
            scores, counts = summarize(cases, predictions, arm["threshold"])
            rows.extend(dict(site=held_out, arm="mixed_scratch", **row) for row in counts)
            results[held_out] = dict(mixed_scratch=scores, single_scratch=prior["results"][held_out]["arms"]["scratch"])
            print(held_out, "mixed backgrounds", scores["iou50"], flush=True)
        write_json(root / "comparison.json", dict(settings=settings, frozen_arms=arms, results=results,
                   pooled_iou50=dict(mixed_scratch=metrics(rows), single_scratch=prior["pooled_iou50"]["scratch"])))
        write_csv(root / "case-rows.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--mixed", action="store_true", help="Replace half the fitting samples with a second training scene")
    parser.add_argument("--reference-run", type=Path, default=OUTPUT, help="Single-background run used by --mixed")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(3)
    output = args.output or (EXTERNAL / "injected-shadow-mixed" if args.mixed else OUTPUT)
    ffmpeg = find_ffmpeg(args.ffmpeg)
    if args.mixed:
        compare_mixed(output.resolve(), ffmpeg, args.reference_run.resolve())
    else:
        compare(output.resolve(), ffmpeg)
