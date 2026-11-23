"""Train with registered background residuals and causal motion integration."""
import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from audit_detection import paired_block_f1
from experiment import filter_by_config, load_candidates, quality, raw_candidates, select_operation, summarize, warm_ids, with_nms
from snl_bytetrack import REFERENCE
from snl_canonical_center import prepare
from snl_center import train_center
from snl_detector import backend
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, WIDTH, load_gray, load_labels, read_json, sha_file, write_csv, write_json
from videosar.snl_causal import detection_counts


RECENT_LAGS = (6, 12, 18, 24)
OLDER_LAGS = (30, 42, 54, 66)
MOTION_LAGS = (2, 4, 6, 8)
SPEEDS = (0.75, 2.0, 4.0)
DIRECTIONS = 16


def box_mean(image, size):
    return cv2.blur(image, (size, size), borderType=cv2.BORDER_REFLECT_101)


def local_response(image):
    smooth = cv2.GaussianBlur(image, (5, 5), 0.9)
    background = box_mean(smooth, 21)
    spread = np.sqrt(np.maximum(box_mean(smooth**2, 21) - background**2, 0.015**2))
    return smooth, ((background - smooth) / spread).clip(-3, 3)


class MotionIntegration:
    def __init__(self, device):
        self.history = deque(maxlen=max(MOTION_LAGS))
        self.device = device
        self.height, self.width = HEIGHT // 2, WIDTH // 2
        x = torch.linspace(-1, 1, self.width, device=device)
        y = torch.linspace(-1, 1, self.height, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        plane = torch.stack((grid_x, grid_y), dim=-1)
        angles = np.arange(DIRECTIONS) * (2 * np.pi / DIRECTIONS)
        velocity = np.stack((np.cos(angles), np.sin(angles)), axis=1)
        velocity = (velocity[None] * np.asarray(SPEEDS)[:, None, None]).reshape(-1, 2)
        factors = np.array([2 / (WIDTH - 2), 2 / (HEIGHT - 2)])
        offsets = np.asarray(MOTION_LAGS)[None, :, None] * velocity[:, None, :] * factors
        offsets = torch.from_numpy(offsets.astype(np.float32)).to(device)
        # Reuse the same fixed translation grid for all frames.
        self.grid = (plane[None, None] - offsets[:, :, None, None]).flatten(0, 1)
        self.velocities = len(velocity)

    @torch.inference_mode()
    def update(self, response):
        half = cv2.resize(response, (self.width, self.height), interpolation=cv2.INTER_AREA)
        current = torch.from_numpy(half).to(self.device)
        while len(self.history) < max(MOTION_LAGS):
            self.history.appendleft(current)
        past = torch.stack([self.history[-lag] for lag in MOTION_LAGS])
        moving = F.grid_sample(past[:, None].repeat(self.velocities, 1, 1, 1), self.grid,
                               padding_mode="border", align_corners=True)
        averages = moving.reshape(self.velocities, len(MOTION_LAGS), self.height, self.width).mean(dim=1)
        stationary = past.mean(dim=0)
        best = averages.amax(dim=0)
        joint = (current + best) / 2
        maps = torch.stack((joint, best - stationary)).cpu().numpy()
        self.history.append(current)
        return [cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR) for image in maps]


def encode_signed(image, width):
    return np.rint(((image / width + 1) / 2).clip(0, 1) * 255).astype(np.uint8)


def compute_features(destination, native, canonical, transforms, device):
    temp = destination.with_suffix(".partial.npy")
    count = len(canonical)
    output = np.lib.format.open_memmap(temp, mode="w+", dtype=np.uint8, shape=(count, 7, HEIGHT, WIDTH))
    integration = MotionIntegration(device)
    history = deque(maxlen=max(OLDER_LAGS))
    for offset in range(count):
        index = offset + 2
        image = canonical[offset, 0].astype(np.float32) / 255
        smooth, local = local_response(image)
        prior = np.stack([history[-lag] if lag <= len(history) else history[0] if history else smooth
                          for lag in (*RECENT_LAGS, *OLDER_LAGS)])
        recent = np.median(prior[:len(RECENT_LAGS)], axis=0)
        older = np.median(prior[len(RECENT_LAGS):], axis=0)
        spread = np.median(np.abs(prior - np.median(prior, axis=0)), axis=0)
        short = recent - smooth
        long = older - smooth
        standardized = (0.5 * (short + long)) / (spread + 0.018)
        integrated, motion_gain = integration.update(local)
        inverse = cv2.invertAffineTransform(transforms[index])
        # Warp only derived channels back, preserving the unresampled image and label grid.
        output[offset, 0] = native[offset]
        for channel, (values, gain) in enumerate(zip((local, short, long, standardized, integrated, motion_gain),
                                                      (2.5, 0.12, 0.12, 4, 2.5, 2.5), strict=True), 1):
            raw = cv2.warpAffine(values, inverse, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REPLICATE)
            output[offset, channel] = encode_signed(raw, gain)
        history.append(smooth)
        if index % 50 == 0:
            print(f"Motion features through frame {index}", flush=True)
    output.flush()
    del output
    temp.replace(destination)


def evaluate(output, ffmpeg):
    study = output / "motion-integration"
    study.mkdir(exist_ok=True)
    train_ids, cal_ids, dev_ids = PROFILES["prospective"]
    gray = load_gray(output, ffmpeg)
    last = max(dev_ids)
    canonical_dir = output / "canonical-center-chained"
    canonical_dir.mkdir(exist_ok=True)
    canonical, transforms, _ = prepare(canonical_dir, gray, max(train_ids), last, chained=True)
    features_file = study / "motion-integration.npy"
    code_hash = sha_file(Path(__file__))
    cache_key = dict(source=sha_file(canonical_dir / "canonical-registration.json"), code=code_hash,
                     through_frame=last)
    cache_receipt = study / "features.json"
    if not features_file.exists() or not cache_receipt.exists() or read_json(cache_receipt) != cache_key:
        compute_features(features_file, gray, canonical, transforms, backend())
        write_json(cache_receipt, cache_key)
    features = np.load(features_file, mmap_mode="r", allow_pickle=False)
    train, cal, dev = (load_labels(output, ids, gray)[0] for ids in (train_ids, cal_ids, dev_ids))
    checkpoints = study / "checkpoints.json"
    if checkpoints.exists():
        registry = read_json(checkpoints)
        if registry["features"] != cache_key:
            raise ValueError("Existing checkpoints were trained with different input features")
        configs = registry["checkpoints"]
    else:
        configs = list(train_center(study, features, train, input_profile="motion-integration"))
        write_json(checkpoints, {"features": cache_key, "checkpoints": configs})

    trials = []
    for config in configs:
        raw = raw_candidates(study, config, features, warm_ids(cal), "calibration")
        operation, score, _ = select_operation(raw, cal)
        trials.append({"config": config | operation, "calibration": score})
    best = max(trials, key=lambda row: quality(row["calibration"]))
    config = best["config"]
    dev_raw = raw_candidates(study, config, features, warm_ids(dev), "development")
    predicted = filter_by_config(with_nms(dev_raw, config["nms"]), config)
    score, rows = summarize(predicted, dev)
    reference = read_json(output / REFERENCE)
    release = reference["config"]
    cached = load_candidates(output, release, "development")
    baseline, baseline_rows = summarize(filter_by_config(with_nms(cached, release["nms"]), release), dev)
    if baseline != reference["development"]:
        raise ValueError("The archived predictions differ from the released baseline")
    coverage = sum(detection_counts(dev_raw[index][0], boxes)[0] for index, boxes in dev.items())
    merged = sum(detection_counts(np.concatenate((dev_raw[index][0], cached[index][0])), boxes)[0]
                 for index, boxes in dev.items())
    report = {"features": cache_key, "motion_lags": MOTION_LAGS, "motion_speeds": SPEEDS, "directions": DIRECTIONS,
              "background_lags": RECENT_LAGS + OLDER_LAGS, "train_frames": len(train),
              "calibration_trials": trials, "selected": best, "development": score, "reference": baseline,
              "unfiltered_matches": coverage, "unfiltered_with_reference_matches": merged,
              "paired_f1_vs_reference": paired_block_f1(rows, baseline_rows)}
    write_json(study / "comparison.json", report)
    write_csv(study / "development.csv", rows)
    print(f"Selected epoch {config['epoch']}: {score}; all-candidate coverage {coverage}/{sum(map(len, dev.values()))}",
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    evaluate(args.output.resolve(), args.ffmpeg)
