"""Audit whether external VideoSAR previews share acquisitions and compare frozen detections."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from infer_video import DEFAULT_MODEL, find_ffmpeg, infer, torch
from snl_inputs import read_json, sha_file, write_json
from snl_qatar_pilot import EXTERNAL, OUTPUT, PINNED_MODEL, ROI, decode, in_review_area, load_boxes, lossless_video, observations
from videosar.data import run_directory
from videosar.snl_causal import matched_boxes


SCENES = {"Qatar": "ICEYE_THKXTW_20240429T222841Z_4049621_X20_SLEDF",
          "Rotterdam": "ICEYE_U14ZW3_20240904T003536Z_4228243_X38_SLEDP"}
PREVIEWS = EXTERNAL / "vsdformer"
OFFICIAL = EXTERNAL / "iceye-open"


def align(preview, official):
    """Fit one spatial map on temporal median images, then compare temporal residuals."""
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(10, 10))
    sift = cv2.SIFT_create(nfeatures=9500, contrastThreshold=0.012)
    first, second = (sift.detectAndCompute(gray.apply(np.median(data, axis=0).astype(np.uint8)), None)
                     for data in (preview, official))
    matches = cv2.BFMatcher().knnMatch(first[1], second[1], k=2)
    close = [a for a, b in matches if a.distance < 0.80 * b.distance]
    xy = np.asarray([first[0][match.queryIdx].pt for match in close], dtype=np.float32)
    targets = np.asarray([second[0][match.trainIdx].pt for match in close], dtype=np.float32)
    if len(xy) < 15:
        raise ValueError("Insufficient spatial correspondences")
    grid, inliers = cv2.estimateAffine2D(xy, targets, method=cv2.RANSAC, ransacReprojThreshold=2.5,
                                         maxIters=5000, confidence=0.999)
    if grid is None:
        raise ValueError("Could not fit a spatial map")
    kept = inliers[:, 0].astype(bool)
    if kept.sum() < 20 or np.min(np.ptp(xy[kept], axis=0)) < 180:
        raise ValueError("The spatial correspondences do not cover the preview")
    errors = np.linalg.norm(xy[kept] @ grid[:, :2].T + grid[:, 2] - targets[kept], axis=1)
    height, width = preview.shape[1:]
    registered = np.stack([cv2.warpAffine(frame, grid, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                            for frame in official])
    residuals = []
    for data in (preview, registered):
        samples = data[:, 8:-8:2, 8:-8:2].astype(np.float32)
        samples -= np.median(samples, axis=0)
        flatten = samples.reshape(len(data), -1)
        residuals.append(flatten / np.linalg.norm(flatten, axis=1, keepdims=True).clip(min=1e-6))
    correlation = residuals[0] @ residuals[1].T
    indexes = np.arange(len(preview))
    best = correlation.argmax(axis=1)
    distant = np.abs(indexes[:, None] - indexes[None, :]) >= 3
    record = dict(preview_to_official_pixels=grid.tolist(), scale_per_axis=np.linalg.norm(grid[:, :2], axis=0).tolist(),
                  spatial_inliers=int(kept.sum()), spatial_residual_median_px=float(np.median(errors)),
                  nearest_frame=best.tolist(), exact_index_matches=int((best == indexes).sum()),
                  within_one_observation=int((abs(best - indexes) <= 1).sum()),
                  median_same_index_correlation=float(np.median(np.diag(correlation))),
                  median_best_distant_correlation=float(np.median(np.where(distant, correlation, -np.inf).max(axis=1))),
                  temporal_correlation=correlation.tolist())
    return grid, record, registered


def mapped_boxes(boxes, grid):
    x0, y0, x1, y1 = boxes.T
    corners = np.stack([np.column_stack((x, y)) for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))], axis=1)
    mapped = corners @ grid[:, :2].T + grid[:, 2]
    return np.column_stack((mapped.min(axis=1), mapped.max(axis=1))).astype(np.float32)


def prediction_agreement(first, second, indices):
    rows = []
    for i in indices:
        a, b = (in_review_area(boxes[i]) for boxes in (first, second))
        centers_a, centers_b = ((boxes[:, :2] + boxes[:, 2:]) / 2 for boxes in (a, b))
        nearby = np.square(centers_a[:, None] - centers_b).sum(axis=2) <= 25
        left, right = linear_sum_assignment(nearby, maximize=True)
        rows.append(dict(observation_zero_based=i, first_boxes=len(a), second_boxes=len(b),
                         iou50_agreements=len(matched_boxes(a, b)), center5_agreements=int(nearby[left, right].sum())))
    totals = {key: sum(row[key] for row in rows) for key in rows[0] if key != "observation_zero_based"}
    return dict(totals=totals, rows=rows)


def qatar_replay(root, video, ffmpeg, grid, evidence, registered):
    prior = read_json(OUTPUT / "frozen-pilot.json")
    old = OUTPUT / "frozen-predictions"
    if (any(sha_file(old / name) != digest for name, digest in prior["prediction_hashes"].items()) or
            sha_file(DEFAULT_MODEL) != read_json(old / "video.json")["model_config_sha256"] or
            prior["model_sha256"] != PINNED_MODEL):
        raise ValueError("Qatar frozen preview predictions or model configuration changed")
    height, width = evidence["official_shape"][1:]
    full = mapped_boxes(np.array([[0, 0, 720, 720]], dtype=np.float32), grid)[0]
    low = np.floor(full[:2]).astype(int).clip(0, (width - 1, height - 1))
    high = np.ceil(full[2:]).astype(int).clip(0, (width, height))
    crop = tuple(map(int, (*low, *(high - low))))
    official_output = root / "qatar-official-predictions"
    official_output.mkdir()
    infer(video, official_output, DEFAULT_MODEL, ffmpeg, crop=crop)
    backward = cv2.invertAffineTransform(grid)
    offset = np.tile(low, 2)
    original = load_boxes(old)
    alternate = {i: mapped_boxes(boxes + offset, backward) for i, boxes in load_boxes(official_output).items()}

    aligned_video, aligned_output = root / "qatar-official-aligned.mp4", root / "qatar-aligned-predictions"
    aligned_output.mkdir()
    lossless_video(aligned_video, registered, ffmpeg)
    infer(aligned_video, aligned_output, DEFAULT_MODEL, ffmpeg)
    aligned = load_boxes(aligned_output)
    indices = [i for i in prior["reviewed_observations"] if evidence["nearest_frame"][i] == i]
    if not indices:
        raise ValueError("No exactly aligned Qatar review observations")
    comparisons = dict(preview_vs_official=prediction_agreement(original, alternate, indices),
                       preview_vs_aligned=prediction_agreement(original, aligned, indices),
                       official_vs_aligned=prediction_agreement(alternate, aligned, indices))
    write_json(root / "qatar-presentation-agreement.json", dict(
               model_sha256=PINNED_MODEL, official_crop_xywh=crop, preview_review_crop_xywh=ROI,
               paired_observations=indices, comparisons=comparisons))
    for name, result in comparisons.items():
        print(f"Qatar {name}, {len(indices)} paired observations: {result['totals']}", flush=True)


def run(output, ffmpeg):
    records = {}
    with run_directory(output, "snl-external-provenance", {"scenes": SCENES}) as root:
        for name, item in SCENES.items():
            source, official = PREVIEWS / f"Noisy_{name}.mp4", OFFICIAL / f"{item}_VID.mp4"
            preview, _, _ = observations(*decode(source, ffmpeg), minimum_transition=2.0)
            frames, _ = decode(official, ffmpeg)
            if len(frames) != len(preview):
                raise ValueError(f"Unexpected sequence length for {name}")
            grid, record, registered = align(preview, frames)
            record.update(official_item=item, preview_sha256=sha_file(source), official_sha256=sha_file(official),
                          official_shape=list(frames.shape))
            records[name] = record
            print(name, record["spatial_inliers"], "inliers", record["within_one_observation"], "near-time pairs", flush=True)
            if name == "Qatar":
                qatar_replay(root, official, ffmpeg, grid, record, registered)
        write_json(root / "acquisition-alignment.json", records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXTERNAL / "source-provenance-controlled")
    parser.add_argument("--ffmpeg")
    args = parser.parse_args()
    cv2.setNumThreads(4)
    cv2.setRNGSeed(314)
    torch.set_num_threads(4)
    run(args.output.resolve(), find_ffmpeg(args.ffmpeg))
