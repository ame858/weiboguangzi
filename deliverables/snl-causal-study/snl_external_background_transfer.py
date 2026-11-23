"""Test shadow pretraining on independent real SAR backgrounds before SNL adaptation."""
import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np

from audit_detection import paired_block_f1
from experiment import filter_by_config, quality, raw_candidates, select_operation, summarize, warm_ids, with_nms
from snl_center import train_center
from snl_detector import TILE, backend, torch
from snl_inputs import DEFAULT_OUTPUT, PROFILES, feature_cache, load_gray, load_labels, read_json, sha_file, write_csv, write_json
from snl_motion_aug import background_tile, inject_shadows, measurements
from snl_protocol import passed
from videosar.snl_causal import CausalFeatures, STACKED_HISTORY_LAGS, detection_counts


SCENES = ("Australia", "Qatar")
REVISION = "c61fb8e744c06f22815f07ad1a0377bebd7c9feb"
CLIPS_PER_SCENE = 720
SOURCE_EPOCHS = 8
LAGS = (0, *STACKED_HISTORY_LAGS)
QUANTILES = (5, 50, 95)
NEGATIVE_FRACTION = 0.20
SNL_FPS = 30


def read_video(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode {path.name}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = []
        while (decoded := capture.read())[0]:
            frames.append(cv2.cvtColor(decoded[1], cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
    if not frames or fps <= 0:
        raise ValueError(f"No usable frames in {path.name}")
    gray = np.stack(frames)
    if min(gray.shape[1:]) < TILE:
        raise ValueError(f"{path.name} does not contain a native {TILE}-pixel crop")
    return gray, fps


def tone_map(gray, target_scale):
    step = max(1, len(gray) // 32)
    bounds = np.percentile(gray[::step, ::4, ::4], QUANTILES)
    if np.min(np.diff(bounds)) < 1:
        raise ValueError("The SAR scene has insufficient tone variation")
    mapping = np.interp(np.arange(256), np.r_[0, bounds, 255], np.r_[0, target_scale, 255])
    return (mapping / 255).astype(np.float32), bounds


def external_crop(gray, lag_offsets, mapping, rng):
    frame_count, height, width = gray.shape
    for _ in range(100):
        index = rng.integers(lag_offsets[-1], frame_count)
        x, y = rng.integers((0, 0), (width - TILE + 1, height - TILE + 1))
        crop = gray[index - lag_offsets, y:y + TILE, x:x + TILE]
        if crop[0, ::4, ::4].std() >= 5:
            return mapping[crop]
    raise ValueError("No textured crop found on the native SAR grid")


def build_source(study, payloads, gray, train, *, external):
    path = study / "source.json"
    settings = dict(generator_sha256=sha_file(Path(__file__).with_name("snl_motion_aug.py")),
                    experiment_sha256=sha_file(Path(__file__)), train_ids=list(train),
                    clips=len(SCENES) * CLIPS_PER_SCENE, epochs=SOURCE_EPOCHS,
                    source_kind="external" if external else "early-snl", source_revision=REVISION if external else None,
                    videos={p.name: sha_file(p) for p in payloads} if external else {})
    if path.is_file():
        receipt = read_json(path)
        if receipt["settings"] != settings:
            raise ValueError("Source checkpoint belongs to a different synthesis recipe")
        return receipt

    background_rng = np.random.default_rng(781)
    shadow_rng = np.random.default_rng(928)
    geometry = measurements(gray, train)
    if not len(geometry[2]):
        raise ValueError("Early SNL training does not contain measurable motion")
    source_metadata = []
    inputs = np.empty((settings["clips"], len(LAGS), TILE, TILE), dtype=np.uint8)
    labels, pos_depths = {}, []
    usable = [i for i in train if all(i - lag in train for lag in LAGS)]
    target_scale = np.percentile(gray[np.asarray(list(train)[::7]) - 2, ::4, ::4], QUANTILES)

    for group, video in enumerate(payloads):
        if external:
            source, fps = read_video(video)
            mapping, bounds = tone_map(source, target_scale)
            offsets = np.rint(np.asarray(LAGS) * fps / SNL_FPS).astype(np.int64)
            if offsets[-1] >= len(source):
                raise ValueError(f"{video.name} is too short for the causal history window")
            source_metadata.append(dict(video=video.name, frames=len(source), fps=fps, shape=source.shape[1:],
                                        quantiles=bounds.tolist(), lag_offsets=offsets.tolist()))
        for clip in range(CLIPS_PER_SCENE):
            index = group * CLIPS_PER_SCENE + clip
            tile = (external_crop(source, offsets, mapping, background_rng) if external
                    else background_tile(gray, train, background_rng, usable, LAGS))
            original = tile[0].copy()
            boxes = (inject_shadows(tile, geometry, shadow_rng, LAGS) if shadow_rng.random() >= NEGATIVE_FRACTION
                     else np.empty((0, 4), dtype=np.float32))
            inputs[index] = np.rint(255 * tile).astype(np.uint8)
            labels[index + 2] = boxes
            for box in boxes:
                left, top, right, bottom = np.rint(box).astype(int)
                pos_depths.append(float(np.median((original - tile[0])[top:bottom, left:right])))
        print(f"{settings['source_kind']}: built {(group + 1) * CLIPS_PER_SCENE} source crops", flush=True)

    checkpoint, = train_center(study, inputs, labels, input_profile="stacked", pretraining_epochs=SOURCE_EPOCHS)
    labels_digest = hashlib.sha256()
    for boxes in labels.values():
        labels_digest.update(len(boxes).to_bytes(2, "little"))
        labels_digest.update(boxes.tobytes())
    receipt = dict(settings=settings, source_metadata=source_metadata, checkpoint=checkpoint,
                   synthetic_boxes_sha256=labels_digest.hexdigest(), positive_patches=sum(bool(len(x)) for x in labels.values()),
                   injected_boxes=len(pos_depths), median_core_darkening=float(np.median(pos_depths)),
                   target_intensity_quantiles=target_scale.tolist())
    write_json(path, receipt)
    return receipt


def checkpoints(study, features, train, source):
    path = study / "checkpoints.json"
    expected = source["checkpoint"]["sha256"]
    if path.is_file():
        result = read_json(path)
        if any(c["pretrained_sha256"] != expected for c in result):
            raise ValueError("Adaptation checkpoints used a different source checkpoint")
        return result
    result = list(train_center(study, features, train, input_profile="stacked", pretrained=source["checkpoint"]))
    write_json(path, result)
    return result


def select(study, features, configs, calibration):
    trials = []
    for config in configs:
        if ((config["device"], config["torch"]) != (backend().type, str(torch.__version__))
                or sha_file(study / config["path"]) != config["sha256"]):
            raise ValueError("A comparison checkpoint has a different runtime or checksum")
        raw = raw_candidates(study, config, features, warm_ids(calibration), "validation")
        operation, result, _ = select_operation(raw, calibration)
        trials.append(dict(config=config | operation, calibration=result))
    return dict(selected=max(trials, key=lambda trial: quality(trial["calibration"])), trials=trials)


def compare(output, sources, ffmpeg):
    payloads = [sources / f"Noisy_{name}.mp4" for name in SCENES]
    if any(not path.is_file() for path in payloads):
        raise FileNotFoundError(f"Expected the pinned independent SAR videos in {sources}")
    gray = load_gray(output, ffmpeg)
    fit_ids, cal_ids, dev_ids = PROFILES["prospective"]
    train = load_labels(output, fit_ids, gray)[0]
    study = output / "external-background-transfer"
    study.mkdir(exist_ok=True)
    features = feature_cache(output, gray, max(dev_ids), CausalFeatures("stacked"))
    scope = {name: study / name for name in ("early-snl", "external")}
    archives = {}
    for name, destination in scope.items():
        destination.mkdir(exist_ok=True)
        archives[name] = build_source(destination, payloads, gray, train, external=name == "external")
    if len({source["synthetic_boxes_sha256"] for source in archives.values()}) != 1:
        raise ValueError("The two source branches did not receive exactly the same synthetic boxes")
    fitted = {name: checkpoints(destination, features, train, archives[name]) for name, destination in scope.items()}
    scope["random"] = output / "simulation-pretraining"
    fitted["random"] = read_json(scope["random"] / "checkpoints-control.json")

    calibration = load_labels(output, cal_ids, gray)[0]
    arms = {name: select(destination, features, fitted[name], calibration) for name, destination in scope.items()}
    report = dict(source=archives, matching_iou=0.50, training_frames=len(fit_ids),
                  calibration_frames=len(cal_ids), arms=arms)
    write_json(study / "calibration.json", report)
    for name, arm in arms.items():
        print(name, "calibration:", arm["selected"]["calibration"], flush=True)

    candidates = {name: raw_candidates(destination, arms[name]["selected"]["config"], features,
                                       warm_ids(dev_ids), "development") for name, destination in scope.items()}
    development = load_labels(output, dev_ids, gray)[0]
    per_arm = {}
    for name, raw in candidates.items():
        selected = arms[name]["selected"]
        config = selected["config"]
        result, rows = summarize(filter_by_config(with_nms(raw, config["nms"]), config), development)
        selected.update(development=result, development_gate_passed=passed(selected["calibration"]) and passed(result),
                        unfiltered_matches=sum(detection_counts(raw[i][0], truth)[0] for i, truth in development.items()))
        per_arm[name] = rows
        print(name, "development:", result, flush=True)
    report["paired_external_vs_early_snl"] = paired_block_f1(per_arm["external"], per_arm["early-snl"])
    report["paired_external_vs_random"] = paired_block_f1(per_arm["external"], per_arm["random"])
    write_json(study / "comparison.json", report)
    write_csv(study / "development.csv", [dict(arm=name, **row) for name, rows in per_arm.items() for row in rows])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--sources", type=Path, default=DEFAULT_OUTPUT / "external-sar/vsdformer")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    cv2.setNumThreads(3)
    torch.set_num_threads(4)
    compare(args.output.resolve(), args.sources.resolve(), args.ffmpeg)
