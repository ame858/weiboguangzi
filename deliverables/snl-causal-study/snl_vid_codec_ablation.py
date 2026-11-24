"""Measure controlled MPEG-4 effects on two held-out windows of one SAR video acquisition."""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

from infer_video import find_ffmpeg
from snl_inputs import read_json, sha_file, write_csv, write_json
from snl_qatar_pilot import decode
from snl_track_before_detect import Stream, evaluate, paired_center_change, thresholded
from snl_vid_cog_comparison import METHOD, TEST, WARMUP, WINDOW, inject, normalize, remap
from snl_vid_cog_window import EXTERNAL
from videosar.data import run_directory


WINDOW_WIDTH = 80
GAP = 8
OUTPUTS = 4
QUANTIZERS = (2, 4, 8)
SEEDS = (82301, 82309, 82331, 82333)
LEVELS = (0., .5, 2., 5.)
ARMS = ("cog_u8", *(f"mpeg4_q{q}" for q in QUANTIZERS), "cog_limited", "mpeg4_q4_limited", "official_mp4")
HALF_BOX = np.array((4., 7.5))


def recode(ffmpeg, frames, fps, quantizer=None):
    height, width = frames.shape[-2:]
    luminance = height * width
    chroma = np.full((len(frames), luminance // 2), 128, dtype=np.uint8)
    payload = np.concatenate((frames.reshape(len(frames), luminance), chroma), axis=1)
    codec = (["-c:v", "libx264", "-crf", "0", "-tune", "zerolatency"] if quantizer is None else
             ["-c:v", "mpeg4", "-q:v", str(quantizer), "-sc_threshold", "0"])
    encode = [ffmpeg, "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size",
              f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0", "-an", *codec,
              "-threads", "1", "-g", "12", "-bf", "0", "-pix_fmt", "yuv420p", "-f", "matroska", "pipe:1"]
    encoded = subprocess.run(encode, input=payload.tobytes(), check=True, stdout=subprocess.PIPE).stdout
    decoded = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                              "pipe:1"], input=encoded, check=True, stdout=subprocess.PIPE).stdout
    restored = np.frombuffer(decoded, dtype=np.uint8)
    if restored.size != payload.size:
        raise ValueError("Controlled encode/decode changed the number of pixels or observations")
    return restored.reshape(payload.shape)[:, :luminance].reshape(frames.shape), len(encoded)


def quantize(frames):
    return np.rint(np.clip(frames, 0, 1) * 255).astype(np.uint8)


def limited_luma(frames):
    return np.rint(16 + frames.astype(np.float32) * (219 / 255)).astype(np.uint8)


def expanded_luma(frames):
    return np.rint(np.clip((frames.astype(np.float32) - 16) * (255 / 219), 0, 255)).astype(np.uint8)


def load_windows(source, comparison, ffmpeg):
    receipt = read_json(source / "receipt.json")
    reference = read_json(comparison / "comparison.json")
    native_file = source / "native-tile.npy"
    if sha_file(native_file) != receipt["array_sha256"][native_file.name] or receipt != reference["settings"]["source"]:
        raise ValueError("COG evidence differs from the frozen alignment run")
    native = np.load(native_file, allow_pickle=False)
    video_path = EXTERNAL / "iceye-open" / f"{receipt['stac_id']}_VID.mp4"
    if sha_file(video_path) != reference["settings"]["mp4_sha256"]:
        raise ValueError("MP4 evidence differs from the frozen alignment run")
    video, fps = decode(video_path, ffmpeg)
    point = reference["alignment"]
    if point["closest_mp4_band"] != list(range(25)) or len(native) != 25 or len(video) != 25:
        raise ValueError("Frozen frame correspondence did not reproduce all matched pairs")
    x, y, width, height = point["mp4_roi_xywh"]
    shapes = {"west": (x - GAP - WINDOW_WIDTH, y), "east": (x + width + GAP, y)}
    for side, (left, top) in shapes.items():
        shifted = np.asarray(point["cog_tile_to_roi"], dtype=np.float64).copy()
        shifted[:2, 2] += (x - left, y - top)
        shape = (height, WINDOW_WIDTH)
        support = remap(np.ones_like(native[:1], dtype=np.float32), shifted, shape)
        if np.min(support) < 1 - 1e-6:
            raise ValueError(f"{side} window is outside the common raw footprint")
        raw = remap(np.log1p(native.astype(np.float32)), shifted, shape)
        mp4 = video[:, top:top + height, left:left + WINDOW_WIDTH]
        if mp4.shape != raw.shape:
            raise ValueError(f"{side} window is outside the native MP4 grid")
        matches = [float(spearmanr(a.ravel(), b.ravel()).statistic) for a, b in zip(raw, mp4, strict=True)]
        cog_float, mp4_float, _ = normalize(raw, mp4)
        yield side, cog_float, mp4_float, fps, dict(roi_xywh=[left, top, WINDOW_WIDTH, height],
                                                   median_holdout_spearman=float(np.median(matches[WARMUP:])))


def tracks(seed, shape):
    rng = np.random.default_rng(seed)
    height, width = shape
    endpoints = np.asarray(((.33 * width, .34 * height), (.65 * width, .67 * height)))
    endpoints += rng.uniform(-3, 3, endpoints.shape)
    velocity = np.asarray(((.25, .75), (-.25, -.75))) * rng.choice((-1, 1), (2, 1))
    return endpoints + (np.arange(25)[:, None, None] - TEST[0]) * velocity


def fixed_predictions(frames):
    stream = Stream(frames, None, 0, window=WINDOW, registered=False, initial_frames=WARMUP)
    predictions, _ = stream.run(TEST)
    arm = predictions[METHOD]
    if any(len(row[0]) < OUTPUTS for row in arm.values()):
        raise ValueError("A window supplied fewer candidate peaks than the fixed output budget")
    return thresholded(arm, counts=dict.fromkeys(TEST, OUTPUTS))


def clip_scores(selected, placebo, frames, blank, trajectory, level):
    truth = {i: np.column_stack((trajectory[i] - HALF_BOX, trajectory[i] + HALF_BOX)) if level else np.empty((0, 4))
             for i in TEST}
    _, rows, _ = evaluate(selected, truth, frames.shape[-2:])
    for row in rows:
        index = row["annotation_id"]
        difference = blank[index].astype(np.float32) - frames[index]
        center_mean = cv2.blur(difference, (3, 3))
        xx, yy = np.rint(trajectory[index]).astype(int).T
        row["mean_center_darkening"] = float(center_mean[yy, xx].mean())
        change = paired_center_change(selected[index][0], placebo[index][0], trajectory[index])
        if level and change["shared_hits"] + change["new_hits"] != row["tp"]:
            raise ValueError("Matched target IDs disagree with the center metric")
        row.update(change)
    return rows


def product_clips(cog, mp4, ffmpeg, fps):
    variants, quality = {"cog_u8": cog, "official_mp4": mp4}, []
    for quantizer in QUANTIZERS:
        name = f"mpeg4_q{quantizer}"
        decoded, size = recode(ffmpeg, cog, fps, quantizer)
        variants[name] = decoded
        quality.append(dict(arm=name, encoded_bytes=size))
    limited = limited_luma(cog)
    limited_decoded, size = recode(ffmpeg, limited, fps, 4)
    variants["cog_limited"] = expanded_luma(limited)
    variants["mpeg4_q4_limited"] = expanded_luma(limited_decoded)
    quality.append(dict(arm="mpeg4_q4_limited", encoded_bytes=size))
    return variants, quality


def evaluate_windows(windows, ffmpeg):
    cases, quality, registration = [], [], {}
    for side, cog, mp4, fps, alignment in windows:
        registration[side] = alignment
        base = quantize(cog)
        original = quantize(mp4)
        lossless, _ = recode(ffmpeg, base, fps)
        np.testing.assert_array_equal(lossless, base, err_msg=f"Lossless luminance verification of {side}")
        controls, quality_rows = product_clips(base, original, ffmpeg, fps)
        placebo = {arm: fixed_predictions(controls[arm]) for arm in ARMS}
        for row in quality_rows:
            error = np.mean((controls[row["arm"]].astype(np.float32) - base)**2)
            quality.append(dict(window=side, blank_rmse_gray=float(np.sqrt(error)), fps=float(fps), **row))
        for seed in SEEDS:
            trajectory = tracks(seed, base.shape[-2:])
            for level in LEVELS:
                uncompressed = quantize(inject(cog, trajectory, level))
                actual = quantize(inject(mp4, trajectory, level))
                variants = product_clips(uncompressed, actual, ffmpeg, fps)[0] if level else controls
                for arm in ARMS:
                    frames, blank = variants[arm], controls[arm]
                    np.testing.assert_array_equal(frames[:WARMUP], blank[:WARMUP],
                                                  err_msg="Future injection changed the warm-up prefix")
                    selected = fixed_predictions(frames) if level else placebo[arm]
                    rows = clip_scores(selected, placebo[arm], frames, blank, trajectory, level)
                    cases.extend(dict(window=side, seed=seed, nominal_drop=level, arm=arm, **row) for row in rows)
    return cases, quality, registration


def run(source, comparison, output, ffmpeg):
    settings = dict(source=str(source), prior_alignment=str(comparison), windows=("west", "east"), window_width=WINDOW_WIDTH,
                    margin_to_previous_roi=GAP, quantizers=QUANTIZERS, seeds=SEEDS, levels=LEVELS, outputs_per_frame=OUTPUTS,
                    arms=ARMS, scoring_indices=TEST, accumulation_window=WINDOW, warmup=WARMUP, codec="mpeg4", gop=12,
                    b_frames=0, limited_luma_bounds=(16, 235))
    with run_directory(output, "snl-vid-codec-ablation", settings) as root:
        windows = list(load_windows(source, comparison, ffmpeg))
        write_json(root / "alignment.json", {w[0]: w[-1] for w in windows})
        print("Independent windows", {w[0]: w[-1] for w in windows}, flush=True)
        rows, quality, registration = evaluate_windows(windows, ffmpeg)
        result = {}
        for arm in ARMS:
            result[arm] = {}
            for dose in LEVELS:
                cases = [r for r in rows if r["arm"] == arm and r["nominal_drop"] == dose]
                counts = ("tp", "fp", "fn", "iou_tp", "blank_path_hits", "shared_hits", "new_hits", "lost_hits")
                result[arm][str(dose)] = dict(**{k: sum(r[k] for r in cases) for k in counts},
                                               mean_center_darkening=float(np.mean([r["mean_center_darkening"] for r in cases])))
        write_csv(root / "case-rows.csv", rows)
        write_json(root / "comparison.json", dict(settings=settings, alignment=registration, compression=quality, results=result))
        print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=EXTERNAL / "lima-vid-cog-window-paired")
    parser.add_argument("--comparison", type=Path, default=EXTERNAL / "lima-vid-cog-comparison")
    parser.add_argument("--output", type=Path, default=EXTERNAL / "lima-vid-codec-ablation")
    parser.add_argument("--ffmpeg")
    args = parser.parse_args()
    cv2.setNumThreads(4)
    run(args.source.resolve(), args.comparison.resolve(), args.output.resolve(), find_ffmpeg(args.ffmpeg))
