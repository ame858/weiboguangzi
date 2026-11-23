"""Train a native-grid center detector with causal, locally registered motion cues."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from audit_detection import paired_block_f1
from experiment import filter_by_config, load_candidates, quality, raw_candidates, select_operation, summarize, warm_ids, with_nms
from snl_bytetrack import REFERENCE
from snl_canonical_center import prepare
from snl_center import train_center
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, WIDTH, load_gray, load_labels, read_json, sha_file, write_csv, write_json
from snl_motion_integration import encode_signed, local_response
from snl_protocol import passed
from videosar.snl_causal import detection_counts


LAGS = (2, 5, 11)
CHANNELS = ("native", "local_darkness", "short_darkening", "long_darkening", "local_motion",
            "registration_gain", "tracked_darkness")


def remap(image, xy):
    return cv2.remap(image, xy, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class MotionFeatures:
    def __init__(self):
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        x, y = np.meshgrid(np.arange(WIDTH // 2, dtype=np.float32), np.arange(HEIGHT // 2, dtype=np.float32))
        self.grid = np.stack((x, y), axis=-1)

    def frame(self, native, canonical, transform, offset):
        half_size = (WIDTH // 2, HEIGHT // 2)
        current = cv2.resize(canonical[offset, 0], half_size, interpolation=cv2.INTER_AREA)
        curr_smooth, curr_dark = local_response(current.astype(np.float32) / 255)
        displacements, gains, shadows, residuals = [], [], [], []
        for lag in LAGS:
            previous = cv2.resize(canonical[max(offset - lag, 0), 0], half_size, interpolation=cv2.INTER_AREA)
            prev_smooth, prev_dark = local_response(previous.astype(np.float32) / 255)
            backward = self.flow.calc(current, previous, None)
            forward = self.flow.calc(previous, current, None)
            coordinates = self.grid + backward
            inside = ((coordinates >= 0) & (coordinates <= np.asarray(half_size) - 1)).all(axis=-1)
            cycle_error = np.linalg.norm(backward + remap(forward, coordinates), axis=-1)
            confidence = inside * np.exp(-cycle_error / 1.5).astype(np.float32)
            broad_motion = cv2.blur(backward, (25, 25))
            displacement = np.linalg.norm(backward - broad_motion, axis=-1)
            displacements.append(confidence * displacement / lag)
            gains.append(confidence * (np.abs(curr_smooth - prev_smooth) -
                                       np.abs(curr_smooth - remap(prev_smooth, coordinates))))
            shadows.append(confidence * (remap(prev_dark, coordinates) - prev_dark))
            residuals.append(prev_smooth - curr_smooth)

        inverse = cv2.invertAffineTransform(transform)
        output = np.empty((len(CHANNELS), HEIGHT, WIDTH), dtype=np.uint8)
        output[0] = native[offset]
        _, native_dark = local_response(output[0].astype(np.float32) / 255)
        output[1] = encode_signed(native_dark, 3)
        planes = (np.mean(residuals[:2], axis=0), residuals[-1], np.mean(displacements, axis=0),
                  np.mean(gains, axis=0), np.mean(shadows, axis=0))
        for channel, (plane, scale) in enumerate(zip(planes, (0.14, 0.14, 1.25, 0.08, 1.5), strict=True), 2):
            full = cv2.resize(plane, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            values = cv2.warpAffine(full, inverse, (WIDTH, HEIGHT), borderMode=cv2.BORDER_CONSTANT)
            if channel == 4:
                output[channel] = np.rint(np.clip(values / scale, 0, 1) * 255).astype(np.uint8)
            else:
                output[channel] = encode_signed(values, scale)
        return output


def prepare_features(study, gray, canonical, transforms, registration_hash):
    destination = study / "local-flow.npy"
    receipt = study / "features.json"
    settings = dict(registration_sha256=registration_hash, code_sha256=sha_file(Path(__file__)),
                    through=len(canonical) + 1, lags=list(LAGS), channels=list(CHANNELS), opencv=cv2.__version__)
    shape = (len(canonical), len(CHANNELS), HEIGHT, WIDTH)
    if destination.is_file() and receipt.is_file() and read_json(receipt) == settings:
        cached = np.load(destination, mmap_mode="r", allow_pickle=False)
        if cached.shape != shape or cached.dtype != np.uint8:
            raise ValueError("The flow feature cache has the wrong shape or type")
        return cached, settings
    temporary = destination.with_suffix(".partial.npy")
    volumes = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=shape)
    extractor = MotionFeatures()
    for offset in range(len(canonical)):
        volumes[offset] = extractor.frame(gray, canonical, transforms[offset + 2], offset)
        if (offset + 2) % 50 == 0:
            print(f"Local flow features through frame {offset + 2}", flush=True)
    volumes.flush()
    del volumes
    temporary.replace(destination)
    write_json(receipt, settings)
    return np.load(destination, mmap_mode="r", allow_pickle=False), settings


def evaluate(output, ffmpeg):
    train_ids, cal_ids, dev_ids = PROFILES["prospective"]
    gray = load_gray(output, ffmpeg)
    canonical_dir = output / "canonical-center-chained"
    canonical_dir.mkdir(exist_ok=True)
    canonical, transforms, _ = prepare(canonical_dir, gray, max(train_ids), max(dev_ids), chained=True)
    study = output / "local-flow"
    study.mkdir(exist_ok=True)
    registration = sha_file(canonical_dir / "canonical-registration.json")
    features, settings = prepare_features(study, gray, canonical, transforms, registration)
    first_cal = min(cal_ids)
    offset = first_cal - 2
    prefix = MotionFeatures().frame(gray[:offset + 1], canonical[:offset + 1], transforms[first_cal], offset)
    if not np.array_equal(features[offset], prefix):
        raise ValueError("The local motion input changed when the video suffix was removed")

    train = load_labels(output, train_ids, gray)[0]
    cal = load_labels(output, cal_ids, gray)[0]
    registry = study / "checkpoints.json"
    if registry.is_file():
        saved = read_json(registry)
        if saved["features"] != settings:
            raise ValueError("The saved weights used different flow features")
        configs = saved["checkpoints"]
    else:
        configs = list(train_center(study, features, train, input_profile="local-flow"))
        write_json(registry, dict(features=settings, checkpoints=configs))

    trials = []
    for config in configs:
        candidates = raw_candidates(study, config, features, warm_ids(cal), "calibration")
        operation, metrics, _ = select_operation(candidates, cal)
        trials.append(dict(config=config | operation, calibration=metrics))
    selected = max(trials, key=lambda result: quality(result["calibration"]))
    report = dict(features=settings, training_frames=len(train), calibration_trials=trials, selected=selected,
                  iou_threshold=0.50, feature_prefix_check_frame=first_cal)
    write_json(study / "calibration.json", report)
    print("Selected on calibration:", selected["calibration"], flush=True)

    config = selected["config"]
    candidates = raw_candidates(study, config, features, warm_ids(dev_ids), "development")
    development = load_labels(output, dev_ids, gray)[0]
    detections = filter_by_config(with_nms(candidates, config["nms"]), config)
    result, rows = summarize(detections, development)
    released = read_json(output / REFERENCE)
    release = released["config"]
    original = load_candidates(output, release, "development")
    baseline, base_rows = summarize(filter_by_config(with_nms(original, release["nms"]), release), development)
    if baseline != released["development"]:
        raise ValueError("The released seven-frame baseline did not replay")
    ceiling = sum(detection_counts(candidates[i][0], truth)[0] for i, truth in development.items())
    merged = sum(detection_counts(np.concatenate((candidates[i][0], original[i][0])), truth)[0]
                 for i, truth in development.items())
    report.update(development=result, reference=baseline, unfiltered_matches=ceiling, combined_matches=merged,
                  paired_f1_vs_reference=paired_block_f1(rows, base_rows),
                  development_gate_passed=passed(selected["calibration"]) and passed(result))
    write_json(study / "comparison.json", report)
    write_csv(study / "development.csv", rows)
    print("Development:", result, "all-candidate matches:", ceiling, "combined with release:", merged, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    cv2.setNumThreads(3)
    torch.set_num_threads(4)
    evaluate(args.output.resolve(), args.ffmpeg)
