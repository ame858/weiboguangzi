"""Test frozen and online backgrounds on controlled injections into independent SAR acquisitions."""
import argparse
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from infer_video import find_ffmpeg
from snl_inputs import read_json, sha_file, write_csv, write_json
from snl_motion_aug import shape_on_tile
from snl_qatar_pilot import EXTERNAL, decode
from snl_track_before_detect import ALPHA, ARMS, BUDGET, SPEEDS, Stream, evaluate, paired_center_change, thresholded
from videosar.data import run_directory


SITES = {
    "Lima": "ICEYE_6MC5KR_20250526T033241Z_4770346_X31_SLEDF",
    "Pucallpa": "ICEYE_6QE022_20260331T033546Z_9417828_X44_SLEDF",
    "SouthernScotland": "ICEYE_GCVEE1_20260318T213827Z_9335125_X61_SLEDF",
    "Denver": "ICEYE_9XJ64F_20251104T180718Z_6897825_X31_SLEDF",
    "MexicoCity": "ICEYE_9G3MZC_20260317T141841Z_9307864_X49_SLEDF",
}
TILE = 192
PLACEMENTS = tuple(product((120, 408), repeat=2))
WARMUP = 5
WINDOW = 15
TEST = tuple(range(WARMUP + WINDOW - 1, 25))
BLANK_LIMIT = 24
FIXED_OUTPUT = 8
DOSES = (0, 5, 15)
SEED = 572391


def video(site, ffmpeg):
    identifier = SITES[site]
    base = EXTERNAL / "iceye-open"
    path = base / f"{identifier}_VID.mp4"
    item = read_json(base / f"{identifier}.json")
    raw, _ = decode(path, ffmpeg)
    if len(raw) != 25 or min(raw.shape[1:]) < 720 or item["id"] != identifier:
        raise ValueError(f"Unexpected native acquisition for {site}")
    y, x = ((dimension - 720) // 2 for dimension in raw.shape[1:])
    return raw[:, y:y + 720, x:x + 720], dict(stac_id=identifier, sha256=sha_file(path), crop_xywh=[x, y, 720, 720])


def paired_clips(frames):
    rng = np.random.default_rng(SEED)
    nominal = np.array(((53, 54), (139, 52), (98, 142)), dtype=np.float32)
    velocities = np.asarray([v for v in SPEEDS if 0.7 <= np.linalg.norm(v) <= 1.8], dtype=np.float32)
    for tile, (x, y) in enumerate(PLACEMENTS):
        original = frames[:, y:y + TILE, x:x + TILE]
        end = nominal + rng.uniform(-5, 5, size=nominal.shape)
        angle = rng.uniform(-.20, .20, len(end))
        half = rng.uniform(0.85, 1.15, size=end.shape) * (4, 7.5)
        rotated = np.abs(np.stack((np.stack((np.cos(angle), -np.sin(angle)), axis=1),
                                   np.stack((np.sin(angle), np.cos(angle)), axis=1)), axis=1))
        projected = np.einsum("nij,nj->ni", rotated, half)
        velocity = velocities[rng.integers(len(velocities), size=len(end))]
        path = end + (np.arange(len(original))[:, None, None] - TEST[0]) * velocity
        truth = np.concatenate((path - projected, path + projected), axis=2)
        for depth in DOSES:
            images = original.copy()
            if depth:
                for i in range(WARMUP, len(images)):
                    frame = images[i].astype(np.float32) / 255
                    for point, size, rotation in zip(path[i], half, angle, strict=True):
                        shape_on_tile(frame, point, size, rotation, depth / 255)
                    images[i] = np.rint(255 * frame).astype(np.uint8)
            labels = {i: truth[i] if depth else np.empty((0, 4), dtype=np.float32) for i in TEST}
            vectors = velocity if depth else np.empty((0, 2), dtype=np.float32)
            yield tile, depth, images, labels, vectors, path


def acquire(site, ffmpeg):
    frames, receipt = video(site, ffmpeg)
    outputs = {arm: {} for arm in ARMS}
    truth, cases, velocities, centers = {}, {}, {}, {}
    for tile, depth, images, labels, vectors, path in paired_clips(frames):
        runner = Stream(images, None, 0, window=WINDOW, registered=False, initial_frames=WARMUP)
        predicted, _ = runner.run(TEST)
        for index in TEST:
            key = len(truth)
            truth[key] = labels[index]
            velocities[key] = vectors
            centers[key] = path[index]
            cases[key] = dict(tile=tile, observation=index, injected_depth=depth)
            for arm in outputs:
                outputs[arm][key] = predicted[arm][index]
    return outputs, truth, cases, velocities, centers, receipt


def measure(predicted, labels, cases, threshold=None, *, fixed_count=None, velocities=None, centers=None):
    counts = None if fixed_count is None else {index: fixed_count for index in predicted}
    if counts is not None and any(len(row[0]) < fixed_count for row in predicted.values()):
        raise ValueError("An external clip did not supply the common output count")
    selected = thresholded(predicted, threshold, counts)
    summary, rows, speed = evaluate(selected, labels, (TILE, TILE), truth_velocity=velocities)
    for row in rows:
        row.update(cases[row["annotation_id"]])
    if centers is not None:
        controls = {(case["tile"], case["observation"]): index for index, case in cases.items() if case["injected_depth"] == 0}
        for row in rows:
            index = row["annotation_id"]
            before = selected[controls[row["tile"], row["observation"]]][0]
            row.update(paired_center_change(selected[index][0], before, centers[index]))
        fields = ("blank_path_hits", "shared_hits", "new_hits", "lost_hits")
        summary["paired_path_effect"] = {str(dose): {field: sum(row[field] for row in rows if row["injected_depth"] == dose)
                                                  for field in fields} for dose in DOSES}
    summary["blank_outputs"] = sum(row["fp"] for row in rows if row["injected_depth"] == 0)
    summary["weak_hits"] = sum(row["tp"] for row in rows if row["injected_depth"] == 5)
    summary["strong_hits"] = sum(row["tp"] for row in rows if row["injected_depth"] == 15)
    return summary, rows, [dict(**row, **cases[row["annotation_id"]]) for row in speed]


def calibrate(outputs, truth, cases):
    points = {}
    for arm, predictions in outputs.items():
        combined = np.concatenate([row[1] for row in predictions.values()])
        thresholds = np.unique(np.r_[np.quantile(combined, np.r_[np.linspace(.90, .99, 25), .995, .999, 1]), combined.max() + 1])
        trials = []
        for value in thresholds:
            summary, _, _ = measure(predictions, truth, cases, float(value))
            trials.append(dict(threshold=float(value), **summary))
        feasible = [row for row in trials if row["blank_outputs"] <= BLANK_LIMIT]
        points[arm] = max(feasible, key=lambda row: (row["weak_hits"], row["center5"]["f1"], -row["blank_outputs"]))
    return points


def normalize(outputs):
    normalized = {}
    for arm, samples in outputs.items():
        normalized[arm] = {}
        for index, (positions, scores, vectors) in samples.items():
            midpoint = np.median(scores)
            spread = max(.02, 1.4826 * np.median(np.abs(scores - midpoint)))
            normalized[arm][index] = (positions, (scores - midpoint) / spread, vectors)
    return normalized


def branches(outputs, compare_normalization):
    if not compare_normalization:
        return outputs
    return outputs | {f"{arm}_normalized": scores for arm, scores in normalize(outputs).items()}


def run(output, ffmpeg, *, compare_normalization=False):
    sites = ("Denver", "MexicoCity") if compare_normalization else ("Pucallpa", "SouthernScotland")
    settings = dict(sites={site: SITES[site] for site in ("Lima", *sites)}, warmup=WARMUP, window=WINDOW,
                    test_observations=TEST, placements=PLACEMENTS, compare_normalization=compare_normalization,
                    seed=SEED, doses=DOSES, max_calibration_blank_outputs=BLANK_LIMIT, fixed_output=FIXED_OUTPUT,
                    background_alpha=ALPHA, candidate_budget=BUDGET, velocity_bank=SPEEDS)
    with run_directory(output, "snl-tbd-external", settings) as root:
        outputs, truth, cases, _, _, fitting = acquire("Lima", ffmpeg)
        points = calibrate(branches(outputs, compare_normalization), truth, cases)
        frozen = dict(settings=settings, calibration=fitting, points=points)
        write_json(root / "frozen-selection.json", frozen)
        print("external calibration frozen", {name: row["weak_hits"] for name, row in points.items()}, flush=True)
        results, observations, speeds, seen = {}, [], [], {fitting["sha256"]}
        for site in sites:
            outputs, truth, cases, velocities, centers, source = acquire(site, ffmpeg)
            if source["sha256"] in seen:
                raise ValueError(f"Duplicate source for {site}")
            seen.add(source["sha256"])
            scores, fixed = {}, {}
            for arm, predicted in branches(outputs, compare_normalization).items():
                scores[arm], rows, measured = measure(predicted, truth, cases, points[arm]["threshold"],
                                                      velocities=velocities, centers=centers)
                observations.extend(dict(site=site, arm=arm, protocol="frozen", **row) for row in rows)
                speeds.extend(dict(site=site, arm=arm, protocol="frozen", **row) for row in measured)
                fixed[arm], rows, measured = measure(predicted, truth, cases, fixed_count=FIXED_OUTPUT,
                                                     velocities=velocities, centers=centers)
                observations.extend(dict(site=site, arm=arm, protocol="equal_output", **row) for row in rows)
                speeds.extend(dict(site=site, arm=arm, protocol="equal_output", **row) for row in measured)
            results[site] = dict(source=source, scores=scores, equal_output=fixed)
            print(site, {name: (row["weak_hits"], row["blank_outputs"]) for name, row in scores.items()}, flush=True)
        write_json(root / "comparison.json", frozen | {"results": results})
        write_csv(root / "case-rows.csv", observations)
        if speeds:
            write_csv(root / "matched-velocities.csv", speeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--compare-normalization", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(3)
    default = "tbd-online-score-normalization" if args.compare_normalization else "tbd-online-injection"
    run((args.output or EXTERNAL / default).resolve(), find_ffmpeg(args.ffmpeg),
        compare_normalization=args.compare_normalization)
