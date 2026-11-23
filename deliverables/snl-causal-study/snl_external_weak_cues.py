"""Compare causal local and temporal contrast cues on faint injected shadows."""
import argparse
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

from experiment import CONFIDENCES, quality
from infer_video import find_ffmpeg, torch
from snl_center import train_center
from snl_detector import load_trained, nonmax, predict_frame
from snl_external_shadow_adaptation import EPOCHS, OUTPUT as REFERENCE, fit_samples
from snl_external_shadow_benchmark import bank, metrics, predict, score_cases, summarize
from snl_inputs import DEFAULT_OUTPUT, PROFILES, load_gray, load_labels, read_json, sha_file, write_csv, write_json
from snl_motion_aug import measurements
from snl_qatar_pilot import EXTERNAL, decode, observations
from videosar.data import run_directory
from videosar.snl_causal import STACKED_HISTORY_LAGS, box_iou


DENVER = EXTERNAL / "iceye-open/ICEYE_9XJ64F_20251104T180718Z_6897825_X31_SLEDF_VID.mp4"
MODES = ("raw", "local", "local_residual")
NOVEL_TIMES = (7, 9, 11, 13, 15, 17, 19, 23)
NOVEL_POSITIONS = (24, 240, 480)
TEST_SEED = 152307
WINDOW = 25
FOLLOWUP_TIMES = (8, 10, 12, 14, 16, 18, 20, 22)
FOLLOWUP_POSITIONS = (12, 228, 444)
FOLLOWUP_SEED = 803061
SOURCES = {"Qatar": EXTERNAL / "vsdformer/Noisy_Qatar.mp4", "Australia": EXTERNAL / "vsdformer/Noisy_Australia.mp4",
           "Denver": DENVER}


def local_standardize(images, floor):
    footprint = (1, WINDOW, WINDOW)
    mean = uniform_filter(images, size=footprint, mode="nearest")
    square = uniform_filter(images**2, size=footprint, mode="nearest")
    deviation = np.sqrt(np.maximum(square - mean**2, floor**2))
    return np.clip(0.5 + 0.18 * (images - mean) / deviation, 0, 1)


def encode(image, mode, observation):
    if mode == "raw":
        return image
    signal = image.astype(np.float32) / 255
    corrected = local_standardize(signal, 0.012) if mode.startswith("local") else signal.copy()
    if mode.endswith("residual"):
        if observation >= STACKED_HISTORY_LAGS[-1]:
            raise ValueError("Residual replaces the unavailable last history channel in short sequences")
        before = np.asarray(STACKED_HISTORY_LAGS) <= observation
        residual = np.median(signal[1:][before], axis=0) - signal[0]
        corrected[-1] = local_standardize(residual[None], 0.008)[0]
    return np.rint(255 * corrected).astype(np.uint8)


def coded(cases, mode):
    return [(spec, encode(image, mode, spec["observation"]), target) for spec, image, target in cases]


def table(cases, predictions, thresholds=CONFIDENCES):
    trials = []
    for threshold in thresholds:
        rows = score_cases(cases, predictions, float(threshold), "iou50")
        weak = [row for row in rows if row["injected_depth"] == 5]
        blank = [row for row in rows if row["injected_depth"] == 0]
        trials.append(dict(threshold=float(threshold), iou50=metrics(rows), weak=metrics(weak),
                           blank_outputs=sum(row["fp"] for row in blank)))
    return trials


def choose(trials, cap):
    feasible = [row for row in trials if row["blank_outputs"] <= cap]
    if not feasible:
        raise ValueError(f"No predeclared threshold satisfies the calibration budget of {cap} control outputs")
    return max(feasible, key=lambda row: (row["weak"]["tp"], quality(row["iou50"]), -row["blank_outputs"]))


def test_cases(site, path, ffmpeg, sizes, motion, *, times, positions, seed):
    raw, fps = decode(path, ffmpeg)
    crop = None
    if site in ("Denver", "MexicoCity"):
        if len(raw) != 25:
            raise ValueError(f"Expected all 25 official {site} observations")
        height, width = raw.shape[1:]
        y, x = (height - 720) // 2, (width - 720) // 2
        if min(x, y) < 0:
            raise ValueError(f"The {site} frame does not support the fixed central crop")
        frames = raw[:, y:y + 720, x:x + 720]
        crop = [x, y, 720, 720]
    else:
        frames, _, _ = observations(raw, fps, minimum_transition=2)
    placements = product(positions, positions, times)
    cases = list(bank(frames, sizes, motion, placements=placements, seed=seed, observation_order=times))
    return cases, dict(video_sha256=sha_file(path), crop_xywh=crop)


def realized_depth(cases):
    drops = []
    for start in range(0, len(cases), 3):
        spec, image, _ = cases[start]
        weak, changed, _ = cases[start + 1]
        if (spec["injected_depth"], weak["injected_depth"]) != (0, 5):
            raise ValueError("Incorrect paired control order")
        left, top, right, bottom = np.rint(spec["box"]).astype(int)
        pixels = np.s_[0, top:bottom, left:right]
        drops.append(np.median(image[pixels].astype(np.int16) - changed[pixels]))
    return dict(median=float(np.median(drops)), p10_p90=np.percentile(drops, (10, 90)).tolist())


def training_priors(ffmpeg):
    snl = DEFAULT_OUTPUT / "prospective"
    gray = load_gray(snl, ffmpeg)
    labels = load_labels(snl, PROFILES["prospective"][0], gray)[0]
    sizes, _, motion = measurements(gray, labels)
    return sizes, motion


def evaluate(output, ffmpeg, *, retain_raw=False):
    modes = ("raw", "raw_residual") if retain_raw else MODES
    times, positions, seed = (FOLLOWUP_TIMES, FOLLOWUP_POSITIONS, FOLLOWUP_SEED) if retain_raw else (
        NOVEL_TIMES, NOVEL_POSITIONS, TEST_SEED)
    archive = read_json(REFERENCE / "comparison.json")
    old = archive["frozen_arms"]["scratch"]
    sizes, motion = training_priors(ffmpeg)
    fitting, calibration, training_hash = fit_samples(sizes, motion, ffmpeg)
    if training_hash != archive["settings"]["source_sha256"] or len(fitting) != archive["settings"]["training_frames"]:
        raise ValueError("The fitting or calibration source differs from the matched raw control")
    settings = dict(training_source="Rotterdam", training_cases=len(fitting), calibration_cases=len(calibration),
                    training_sha256=training_hash, budget=EPOCHS, modes=modes, test_times=times, test_positions=positions,
                    test_seed=seed, test_sites=list(SOURCES), weak_gray_levels=5, local_window_px=WINDOW,
                    gate="pooled weak hits increase, control outputs do not increase, and pooled overall F1 does not decrease")
    with run_directory(output, "snl-external-weak-cues", settings) as root:
        configs = {"raw": (Path(old["model_dir"]), old["config"])}
        targets = {i + 2: target for i, (_, _, target) in enumerate(fitting)}
        for mode in modes[1:]:
            destination = root / mode
            destination.mkdir()
            data = np.stack([image for _, image, _ in coded(fitting, mode)])
            (cfg,) = list(train_center(destination, data, targets, input_profile="stacked", epoch_budget=EPOCHS))
            configs[mode] = destination, cfg | {"contrast_gain": old["config"]["contrast_gain"]}

        models = {mode: load_trained(folder, cfg) for mode, (folder, cfg) in configs.items()}
        trials = {mode: table(calibration, predict(model, coded(calibration, mode), "raw", 1, None, 0.35))
                  for mode, model in models.items()}
        reference = next(row for row in trials["raw"] if row["threshold"] == old["threshold"])
        if reference["iou50"] != old["calibration"]:
            raise ValueError("The archived raw calibration counts did not reproduce")
        cap = reference["blank_outputs"]
        points = {mode: choose(rows, cap) for mode, rows in trials.items()}
        frozen = dict(settings=settings, raw_original_point=reference, calibration_blank_cap=cap,
                      points=points, trials=trials,
                      checkpoints={mode: {"folder": str(folder), "config": cfg} for mode, (folder, cfg) in configs.items()})
        write_json(root / "frozen-selection.json", frozen)
        print("Frozen points", points, flush=True)

        results, rows = {}, []
        for site, path in SOURCES.items():
            cases, source = test_cases(site, path, ffmpeg, sizes, motion, times=times, positions=positions, seed=seed)
            comparisons = {}
            for mode, model in models.items():
                predictions = predict(model, coded(cases, mode), "raw", 1, None, 0.35)
                comparisons[mode], counts = summarize(cases, predictions, points[mode]["threshold"])
                rows.extend(dict(site=site, arm=mode, **row) for row in counts)
                print(site, mode, comparisons[mode]["by_depth"]["5"], flush=True)
            results[site] = dict(source=source, realized_weak_drop=realized_depth(cases), arms=comparisons)
        pooled = {}
        for mode in modes:
            counts = [row for row in rows if row["arm"] == mode]
            pooled[mode] = dict(iou50=metrics(counts), weak=metrics([row for row in counts if row["injected_depth"] == 5]),
                                blank_outputs=sum(row["fp"] for row in counts if row["injected_depth"] == 0))
        control = pooled["raw"]
        gates = {mode: (row["weak"]["tp"] > control["weak"]["tp"] and row["blank_outputs"] <= control["blank_outputs"]
                        and row["iou50"]["f1"] >= control["iou50"]["f1"]) for mode, row in pooled.items() if mode != "raw"}
        write_json(root / "comparison.json", frozen | dict(results=results, pooled=pooled, passed_controlled_gates=gates))
        write_csv(root / "case-rows.csv", rows)
        print("Controlled gates", gates, flush=True)


def audit_dense(output, ffmpeg):
    report = read_json(output / "comparison.json")
    settings = report["settings"]
    sizes, motion = training_priors(ffmpeg)
    results = {}
    for site in settings["test_sites"]:
        cases, source = test_cases(site, SOURCES[site], ffmpeg, sizes, motion, times=tuple(settings["test_times"]),
                                   positions=tuple(settings["test_positions"]), seed=settings["test_seed"])
        if source != report["results"][site]["source"]:
            raise ValueError(f"The frozen {site} media receipt changed")
        weak = [row for row in cases if row[0]["injected_depth"] == 5]
        results[site] = {}
        for mode, saved in report["checkpoints"].items():
            folder, config = Path(saved["folder"]), saved["config"]
            normal = load_trained(folder, config)
            dense = load_trained(folder, config | {"dense_candidate_limit": 1024})
            rows = []
            for spec, image, truth in coded(weak, mode):
                boxes, score = nonmax(*predict_frame(normal, image, "temporal"), 0.35)
                pool, raw_score = predict_frame(dense, image, "temporal")
                remaining, _ = nonmax(pool, raw_score, 0.35)
                stages = dict(selected=boxes[score >= report["points"][mode]["threshold"]], peak_after_nms=boxes,
                              dense_before_nms=pool, dense_after_nms=remaining)
                rows.append({key: spec[key] for key in ("observation", "tile_x", "tile_y")} |
                            {key: int((box_iou(candidate, truth) >= 0.50).any()) for key, candidate in stages.items()})
            totals = {key: sum(row[key] for row in rows) for key in stages}
            if totals["selected"] != report["results"][site]["arms"][mode]["by_depth"]["5"]["tp"]:
                raise ValueError(f"The frozen weak counts did not reproduce for {site}/{mode}")
            results[site][mode] = dict(totals=totals, rows=rows)
            print(site, mode, totals, flush=True)
    write_json(output / "weak-candidate-audit.json", dict(iou_threshold=0.50, weak_per_site=len(weak), dense_limit=1024, results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--retain-raw", action="store_true", help="Compare raw grayscale plus a causal residual on fresh injected cases")
    parser.add_argument("--audit-dense", action="store_true", help="Audit pre-threshold weak coverage in an existing run")
    args = parser.parse_args()
    cv2.setNumThreads(3)
    torch.set_num_threads(4)
    output = args.output or (EXTERNAL / ("injected-weak-raw-residual" if args.retain_raw else "injected-weak-cues"))
    ffmpeg = find_ffmpeg(args.ffmpeg)
    if args.audit_dense:
        audit_dense(output.resolve(), ffmpeg)
    else:
        evaluate(output.resolve(), ffmpeg, retain_raw=args.retain_raw)
