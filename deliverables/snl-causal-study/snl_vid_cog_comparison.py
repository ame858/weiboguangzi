"""Compare paired Lima video products on the same radar grid and injection paths."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

from infer_video import find_ffmpeg
from snl_inputs import read_json, sha_file, write_csv, write_json
from snl_qatar_pilot import decode
from snl_track_before_detect import Stream, evaluate, paired_center_change, thresholded
from snl_vid_cog_window import EXTERNAL
from videosar.data import run_directory


WORK = 768
ROI_WH = (192, 112)
WARMUP = 5
WINDOW = 15
TEST = tuple(range(WARMUP + WINDOW - 1, 25))
SEEDS = (627101, 627103, 627107, 627109)
LEVELS = (0.0, 0.5, 2.0, 5.0)
OUTPUT_COUNT = 8
METHOD = "online_tbd"
PRODUCTS = ("cog_float", "cog_rounded_u8", "official_mp4")


def matrix(a):
    return np.vstack((a, (0.0, 0.0, 1.0)))


def resize_map(width, height, new_width, new_height):
    sx, sy = new_width / width, new_height / height
    return np.array(((sx, 0, (sx - 1) / 2), (0, sy, (sy - 1) / 2), (0, 0, 1)))


def point_map(points, transform):
    return points @ transform[:2, :2].T + transform[:2, 2]


def display_values(values):
    lo, hi = np.quantile(values, (0.02, 0.995))
    if hi <= lo:
        raise ValueError("Video band has no radiometric variation for alignment")
    return np.rint(np.clip((values - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def match_geometry(preview, mp4, full_yx, preview_bands):
    high = np.median(np.log1p(preview.astype(np.float32)), axis=0)
    video = np.median(mp4[list(preview_bands)], axis=0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(10, 10))
    left, right = [clahe.apply(cv2.resize(display_values(a), (WORK, WORK), interpolation=cv2.INTER_AREA))
                   for a in (high, video)]
    sift = cv2.SIFT_create(nfeatures=5500, contrastThreshold=.01)
    target_points, target_desc = sift.detectAndCompute(right, None)
    fits = []
    for flip in (None, 0, 1):
        candidate = left if flip is None else cv2.flip(left, flip)
        points, desc = sift.detectAndCompute(candidate, None)
        matched = cv2.BFMatcher().knnMatch(desc, target_desc, k=2)
        pairs = [a for a, b in matched if a.distance < .78 * b.distance]
        if len(pairs) < 15:
            continue
        source_xy = np.array([points[m.queryIdx].pt for m in pairs])
        target_xy = np.array([target_points[m.trainIdx].pt for m in pairs])
        if flip is not None:
            source_xy[:, 1 - flip] = WORK - 1 - source_xy[:, 1 - flip]
        aligned, inliers = cv2.estimateAffine2D(source_xy, target_xy, ransacReprojThreshold=2.5, maxIters=6000,
                                                confidence=.999)
        if aligned is None:
            continue
        keep = inliers[:, 0].astype(bool)
        if keep.sum() < 20 or np.min(np.ptp(source_xy[keep], axis=0)) < WORK * .18:
            continue
        errors = np.linalg.norm(point_map(source_xy[keep], aligned) - target_xy[keep], axis=1)
        fits.append((int(keep.sum()), float(np.median(errors)), float(np.quantile(errors, .9)), aligned))
    if not fits:
        raise ValueError("Could not verify an affine correspondence between the same-acquisition products")
    inliers, median, p90, aligned = max(fits, key=lambda row: (row[0], -row[1]))
    height, width = mp4.shape[-2:]
    result = np.linalg.inv(resize_map(width, height, WORK, WORK)) @ matrix(aligned) @ resize_map(*full_yx[::-1], WORK, WORK)
    return result, dict(spatial_inliers=inliers, coarse_median_error_preview_px=median, coarse_p90_error_preview_px=p90)


def roi_for_tile(transform, tile_yx, mp4_yx):
    width, height = ROI_WH
    middle = point_map((np.asarray(tile_yx[::-1]) - 1) / 2, transform)
    x, y = np.rint(middle - (width / 2, height / 2)).astype(int)
    if min(x, y) < 0 or x + width > mp4_yx[1] or y + height > mp4_yx[0]:
        raise ValueError("The requested matched window lies outside the MP4")
    return int(x), int(y), width, height


def remap(frames, transform, shape):
    height, width = shape
    return np.stack([cv2.warpAffine(image.astype(np.float32), transform[:2], (width, height), flags=cv2.INTER_LINEAR)
                     for image in frames])


def alignment(native, overview, mp4, receipt):
    transform, detail = match_geometry(overview, mp4, receipt["full_shape_yx"], receipt["preview_bands"])
    origin = np.eye(3)
    origin[:2, 2] = receipt["origin_xy"]
    tile_transform = transform @ origin
    x, y, width, height = roi_for_tile(tile_transform, native.shape[-2:], mp4.shape[-2:])
    translate = np.eye(3)
    translate[:2, 2] = -x, -y
    initial = translate @ tile_transform
    video = mp4[:, y:y + height, x:x + width]
    source = np.log1p(native.astype(np.float32))
    projected = remap(source[:WARMUP], initial, (height, width))
    raw_median = display_values(np.median(projected, axis=0))
    video_median = display_values(np.median(video[:WARMUP], axis=0))
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
    first, second = (clahe.apply(a) for a in (video_median, raw_median))
    correction = np.eye(3)
    try:
        _, fitted = cv2.findTransformECC(first, second, np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 140, 1e-6), None, 5)
        corners = np.array(((0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)))
        if np.max(np.linalg.norm(point_map(corners, fitted) - corners, axis=1)) <= 14:
            correction = matrix(fitted)
    except cv2.error:
        pass
    final = np.linalg.inv(correction) @ initial
    footprint = remap(np.ones_like(native[:1], dtype=np.float32), final, (height, width))[0]
    if np.min(footprint) < 1 - 1e-6:
        raise ValueError("Aligned window is not fully contained within the retrieved native tile")
    aligned = remap(source, final, (height, width))
    rank = [float(spearmanr(a[::2, ::2].ravel(), b[::2, ::2].ravel()).statistic) for a, b in zip(aligned, video, strict=True)]
    residual = []
    for values in (aligned, video.astype(np.float32)):
        change = values - np.median(values[:WARMUP], axis=0)
        rows = change[:, ::2, ::2].reshape(len(values), -1)
        rows -= rows.mean(axis=1, keepdims=True)
        residual.append(rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-6))
    similarity = residual[0] @ residual[1].T
    closest = similarity.argmax(axis=1)
    near = np.abs(closest - np.arange(len(video))) <= 1
    detail.update(mp4_roi_xywh=[x, y, width, height], cog_tile_to_roi=final.tolist(), local_ecc_correction=correction.tolist(),
                  spatial_spearman_by_band=rank, temporal_similarity=similarity.tolist(), closest_mp4_band=closest.tolist(),
                  temporal_within_one=int(near.sum()), scored_observations_within_one=int(near[list(TEST)].sum()),
                  median_spatial_rank_held_out=float(np.median(rank[WARMUP:])))
    return aligned, video, detail


def normalize(aligned, mp4):
    quantiles = np.r_[.005, np.linspace(.01, .99, 99), .995]
    raw_levels = np.quantile(aligned[:WARMUP], quantiles)
    video_levels = np.quantile(mp4[:WARMUP], quantiles) / 255
    unique, positions = np.unique(raw_levels, return_index=True)
    rendered = np.interp(aligned, unique, video_levels[positions]).astype(np.float32)
    return rendered, mp4.astype(np.float32) / 255, dict(cog_log_quantiles=raw_levels.tolist(), mp4_gray_quantiles=video_levels.tolist())


def paths(seed):
    rng = np.random.default_rng(seed)
    centers = np.array(((51, 36), (140, 77))) + rng.uniform(-4, 4, (2, 2))
    speed = np.array(((1.0, .6), (-.8, -.75))) * rng.choice((-1, 1), size=(2, 1))
    return centers + (np.arange(25)[:, None, None] - TEST[0]) * speed


def inject(frames, trajectories, level):
    copy = frames.copy()
    yy, xx = np.indices(frames.shape[-2:])
    for i in range(WARMUP, len(copy)):
        for x, y in trajectories[i]:
            distance = np.hypot(np.maximum(np.abs(xx - x) - 4, 0), np.maximum(np.abs(yy - y) - 7.5, 0))
            copy[i] -= (level / 255) * np.exp(-.5 * (distance / .75)**2).astype(np.float32)
    return np.clip(copy, 0, 1)


def score_windows(cog, mp4):
    results = []
    blank = (cog, np.rint(255 * cog) / 255, mp4)
    controls = [predict_fixed(frames) for frames in blank]
    for seed in SEEDS:
        trajectory = paths(seed)
        half = np.array([4, 7.5])
        for level in LEVELS:
            targets = {i: np.column_stack((trajectory[i] - half, trajectory[i] + half)) if level else np.empty((0, 4))
                       for i in TEST}
            raw = inject(cog, trajectory, level)
            assets = (raw, np.rint(255 * raw) / 255, np.rint(255 * inject(mp4, trajectory, level)) / 255)
            for name, asset, control, placebo in zip(PRODUCTS, assets, blank, controls, strict=True):
                selected = predict_fixed(asset) if level else placebo
                _, rows, _ = evaluate(selected, targets, asset.shape[-2:])
                for row in rows:
                    index = row["annotation_id"]
                    xx, yy = np.rint(trajectory[index]).astype(int).T
                    drops = (control[index, yy, xx] - asset[index, yy, xx]) * 255
                    change = paired_center_change(selected[index][0], placebo[index][0], trajectory[index])
                    results.append(dict(product=name, seed=seed, nominal_drop=level, mean_center_drop=float(drops.mean()),
                                        **row, **change))
    return results


def predict_fixed(frames):
    stream = Stream(frames * 255, None, 0, window=WINDOW, registered=False, initial_frames=WARMUP)
    predicted, _ = stream.run(TEST)
    if any(len(row[0]) < OUTPUT_COUNT for row in predicted[METHOD].values()):
        raise ValueError("A product supplied fewer candidate peaks than the equal-output count")
    return thresholded(predicted[METHOD], counts=dict.fromkeys(TEST, OUTPUT_COUNT))


def run(source, output, ffmpeg):
    receipt = read_json(source / "receipt.json")
    for name, digest in receipt["array_sha256"].items():
        if sha_file(source / name) != digest:
            raise ValueError(f"Source array failed its checksum: {name}")
    mp4_path = EXTERNAL / "iceye-open" / f"{receipt['stac_id']}_VID.mp4"
    mp4, _ = decode(mp4_path, ffmpeg)
    native, overview = (np.load(source / name, allow_pickle=False) for name in ("native-tile.npy", "overview.npy"))
    if len(mp4) != 25 or native.shape[0] != 25:
        raise ValueError("Both products must have 25 observations")
    settings = dict(source=receipt, mp4_sha256=sha_file(mp4_path), motion_window=WINDOW, warmup=WARMUP, test=TEST,
                    seed=SEEDS, levels=LEVELS, outputs_per_window=OUTPUT_COUNT, method=METHOD, products=PRODUCTS)
    with run_directory(output, "snl-vid-cog-comparison", settings) as root:
        aligned, video, detail = alignment(native, overview, mp4, receipt)
        write_json(root / "alignment.json", detail)
        print("Alignment", {key: value for key, value in detail.items() if key not in
                            ("temporal_similarity", "spatial_spearman_by_band")}, flush=True)
        if (detail["median_spatial_rank_held_out"] < .70 or detail["temporal_within_one"] < 19
                or detail["scored_observations_within_one"] < 5):
            raise ValueError("The native ROI did not meet the spatial and temporal correspondence criteria")
        float_cog, actual, levels = normalize(aligned, video)
        rows = score_windows(float_cog, actual)
        counts = ("tp", "fp", "fn", "iou_tp", "blank_path_hits", "shared_hits", "new_hits", "lost_hits")
        totals = {name: {str(level): {key: sum(row[key] for row in rows if row["product"] == name and row["nominal_drop"] == level)
                                     for key in counts} for level in LEVELS} for name in PRODUCTS}
        write_csv(root / "case-rows.csv", rows)
        write_json(root / "comparison.json", dict(settings=settings, alignment=detail, radiometric_mapping=levels, results=totals))
        print(totals, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=EXTERNAL / "lima-vid-cog-window")
    parser.add_argument("--output", type=Path, default=EXTERNAL / "lima-vid-cog-comparison")
    parser.add_argument("--ffmpeg")
    args = parser.parse_args()
    cv2.setNumThreads(4)
    cv2.setRNGSeed(1729)
    run(args.source.resolve(), args.output.resolve(), find_ffmpeg(args.ffmpeg))
