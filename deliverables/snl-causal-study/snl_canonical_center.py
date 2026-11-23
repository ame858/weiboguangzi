"""Train a center detector in registered SNL image coordinates and score in native coordinates."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from audit_detection import paired_block_f1
from experiment import filter_by_config, load_candidates, quality, select_operation, summarize, warm_ids, with_nms
from route_prior import centers
from snl_bytetrack import REFERENCE
from snl_center import train_center
from snl_detector import load_trained, predict_frame, torch
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, VIDEO, WIDTH, load_gray, load_labels, read_json, sha_file
from snl_inputs import write_csv, write_json
from snl_registered_route import chained_registration, project, registration
from videosar.snl_causal import CausalFeatures, matched_boxes


GRID = np.array([WIDTH, HEIGHT])


def moved_boxes(boxes, matrix):
    points = project(centers(boxes), matrix)
    scale = np.linalg.norm(matrix[:, 0])
    half = (boxes[:, 2:] - boxes[:, :2]) * (0.5 * scale)
    result = np.column_stack((points - half, points + half)).clip(0, np.tile(GRID, 2)).astype(np.float32)
    keep = ((points >= 0) & (points < GRID)).all(axis=1) & (result[:, 2:] > result[:, :2]).all(axis=1)
    return result[keep], keep


def prepare(study, gray, anchor, last, *, chained=False):
    alignment = study / "canonical-registration.json"
    settings = {"anchor": anchor, "video_sha256": sha_file(VIDEO), "through": last}
    if alignment.exists():
        cache = read_json(alignment)
        if cache["settings"] != settings:
            raise ValueError("Cached registration belongs to different input frames")
        matrices = {int(i): np.asarray(matrix) for i, matrix in cache["transforms"].items()}
    else:
        estimator = chained_registration if chained else registration
        matrices, counts = estimator(gray, anchor, range(2, last + 1))
        cache = {"settings": settings, "transforms": {i: matrix.tolist() for i, matrix in matrices.items()},
                 "inliers": counts}
        write_json(alignment, cache)
    shape = (last - 1, 7, HEIGHT, WIDTH)
    path = study / "canonical-stacked.npy"
    identity = np.eye(2, 3)
    frame_matrices = {i: matrices.get(i, identity) for i in range(2, last + 1)}
    if not path.exists():
        temporary = path.with_suffix(".partial.npy")
        features = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=shape)
        scanner = CausalFeatures("stacked")
        for index, matrix in frame_matrices.items():
            image = cv2.warpAffine(gray[index - 2], matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
            features[index - 2] = np.rint(255 * scanner.network_channels(image)).astype(np.uint8)
        features.flush()
        del features
        temporary.replace(path)
    features = np.load(path, mmap_mode="r", allow_pickle=False)
    if features.shape != shape or features.dtype != np.uint8:
        raise ValueError("Invalid canonical feature cache")
    return features, frame_matrices, matrices


def native_candidates(model, features, transforms, labels):
    raw = {}
    for index in warm_ids(labels):
        boxes, scores = predict_frame(model, features[index - 2], "temporal")
        native, keep = moved_boxes(boxes, cv2.invertAffineTransform(transforms[index]))
        raw[index] = native, scores[keep]
    return raw


def evaluate(output, ffmpeg, *, chained=False):
    study = output / ("canonical-center-chained" if chained else "canonical-center")
    study.mkdir(exist_ok=True)
    train_ids, cal_ids, dev_ids = PROFILES["prospective"]
    gray = load_gray(output, ffmpeg)
    train, validation, development = (load_labels(output, ids, gray)[0] for ids in (train_ids, cal_ids, dev_ids))
    features, transforms, successful = prepare(study, gray, max(train_ids), max(dev_ids), chained=chained)
    fit = {i: moved_boxes(truth, transforms[i])[0] for i, truth in train.items()}
    path = study / "checkpoints-canonical-center.json"
    if path.exists():
        checkpoints = read_json(path)
    else:
        checkpoints = list(train_center(study, features, fit, input_profile="stacked"))
        write_json(path, checkpoints)
    trials = []
    for config in checkpoints:
        model = load_trained(study, config)
        calibration = native_candidates(model, features, transforms, validation)
        point, scores, _ = select_operation(calibration, validation)
        trials.append({"model": config, "operation": point, "calibration": scores})
    selected = max(trials, key=lambda row: quality(row["calibration"]))
    model = load_trained(study, selected["model"])
    raw = native_candidates(model, features, transforms, development)
    point = selected["operation"]
    result, rows = summarize(filter_by_config(with_nms(raw, point["nms"]), point), development)
    release = read_json(output / REFERENCE)
    native = load_candidates(output, release["config"], "development")
    base = release["config"]
    _, base_rows = summarize(filter_by_config(with_nms(native, base["nms"]), base), development)
    coverage = sum(len(matched_boxes(raw[index][0], truth)) for index, truth in development.items())
    registered = {name: sum(i in successful for i in ids) for name, ids
                  in (("train", train_ids), ("calibration", cal_ids), ("development", dev_ids))}
    report = {"registered_frames": registered, "fitting_boxes_native": sum(map(len, train.values())),
              "fitting_boxes_canonical": sum(map(len, fit.values())), "trials": trials, "selected": selected,
              "development": result, "all_candidate_matches_iou50": coverage, "development_reference": release["development"],
              "paired_f1_vs_reference": paired_block_f1(rows, base_rows)}
    write_json(output / f"{study.name}-comparison.json", report)
    write_csv(output / f"{study.name}-development.csv", rows)
    print(study.name, {key: report[key] for key in ("registered_frames", "fitting_boxes_canonical",
                                                           "development", "all_candidate_matches_iou50")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--chain-registration", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(4)
    cv2.setRNGSeed(1729)
    evaluate(args.output.resolve(), args.ffmpeg, chained=args.chain_registration)
