"""Train a matched SNL control and a detector exposed to earlier high-score background errors."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from audit_detection import paired_block_f1
from experiment import filter_by_config, load_candidates, nonmax, raw_candidates, summarize, warm_ids, with_nms
from snl_center import train_center
from snl_detector import TILE
from snl_external_background_transfer import select
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, WIDTH, feature_cache, load_gray, load_labels, read_json, sha_file, write_csv, write_json
from snl_protocol import passed
from snl_registered_route import project
from videosar.snl_causal import CausalFeatures, box_iou


LAGS = (-3, -1, 1, 3)
PATCH = (13, 21)
SEED = 5183


def centers(boxes):
    return (boxes[:, :2] + boxes[:, 2:]) / 2


def stable(canonical, index, point):
    center = tuple(map(float, point))
    patch = cv2.getRectSubPix(canonical[index - 2, 0], PATCH, center).astype(np.float32)
    patch -= patch.mean()
    length = np.linalg.norm(patch)
    if length < 2 * np.sqrt(patch.size):
        return None
    ncc = []
    for lag in LAGS:
        other = cv2.getRectSubPix(canonical[index + lag - 2, 0], PATCH, center).astype(np.float32)
        other -= other.mean()
        ncc.append(float(np.sum(patch * other) / max(float(length * np.linalg.norm(other)), 1e-6)))
    score = float(np.median(ncc))
    return score if score >= 0.80 else None


def locate(point):
    return np.rint(point - TILE / 2).astype(int).clip(0, [WIDTH - TILE, HEIGHT - TILE])


def paired_tiles(output, train):
    rank = read_json(output / "forward-dense-frame-ranking/comparison.json")
    scope = output / "set-ranking-expanded"
    early = DEFAULT_OUTPUT / "early-stacked"
    early_train = read_json(early / "split-protocol.json")["train"]
    initial = [(early, rank["fit_detector"], "forward-dense-fit", rank["fit_ids"], max(early_train))]
    for cutoff in (100, 160):
        directory = scope / f"detector-through-{cutoff}"
        frozen = read_json(directory / "checkpoint.json")
        initial.append((directory, frozen["config"], "set-fitting", frozen["held_out_ids"], frozen["training_ids"][-1]))
    source = {}
    hashes = []
    for directory, config, key, scope_ids, train_last in initial:
        if train_last >= min(scope_ids) or max(scope_ids) > max(train):
            raise ValueError("Background error mining needs fitting frames excluded from the generating detector")
        if set(scope_ids) & source.keys():
            raise ValueError("The earlier detector candidate scopes overlap")
        raw = load_candidates(directory, config, key)
        source.update({i: raw[i] for i in scope_ids})
        hashes.append(dict(sha256=config["sha256"], fit_last=train_last, mining_ids=scope_ids))

    folder = output / "canonical-center-chained"
    canonical = np.load(folder / "canonical-stacked.npy", mmap_mode="r", allow_pickle=False)
    register = read_json(folder / "canonical-registration.json")
    transforms = {int(i): np.asarray(m) for i, m in register["transforms"].items()}
    ground = {i: project(centers(boxes), transforms[i]) for i, boxes in train.items() if i in transforms}
    rng = np.random.default_rng(SEED)
    records = []
    for i in sorted(source):
        ids = (i, *(i + lag for lag in LAGS))
        if not all(t in ground for t in ids):
            continue
        boxes, scores = source[i]
        top = np.argsort(-scores, kind="stable")[:512]
        boxes, scores = nonmax(boxes[top], scores[top], 0.35)
        boxes, scores = boxes[:128], scores[:128]
        native = centers(boxes)
        fixed = project(native, transforms[i])
        neighbors = np.concatenate([ground[t] for t in ids])
        separation = np.linalg.norm(fixed[:, None] - neighbors, axis=2).min(axis=1, initial=np.inf)
        safe = (separation > 18) & (box_iou(boxes, train[i]).max(axis=1, initial=0) < 0.05)
        safe &= ((fixed >= 24) & (fixed < [WIDTH - 24, HEIGHT - 24])).all(axis=1)
        eligible = [(j, score) for j in np.flatnonzero(safe) if (score := stable(canonical, i, fixed[j])) is not None]
        if len(eligible) < 2:
            continue
        best, best_ncc = eligible[0]
        hard_start = locate(native[best])
        controls = [(j, score) for j, score in eligible[1:] if np.linalg.norm(locate(native[j]) - hard_start) >= 32]
        if not controls:
            continue
        control, control_ncc = controls[rng.integers(len(controls))]
        random_start = locate(native[control])
        records.append(dict(annotation_id=i, hard_left=int(hard_start[0]), hard_top=int(hard_start[1]),
                            random_left=int(random_start[0]), random_top=int(random_start[1]),
                            hard_score=float(scores[best]), random_score=float(scores[control]),
                            hard_static_ncc=best_ncc, random_static_ncc=control_ncc))
    if len(records) < 32:
        raise ValueError("Too few paired background crops to compare")
    return records, hashes


def evaluate(output, ffmpeg):
    fit_ids, cal_ids, dev_ids = PROFILES["prospective"]
    gray = load_gray(output, ffmpeg)
    training = load_labels(output, fit_ids, gray)[0]
    features = feature_cache(output, gray, max(dev_ids), CausalFeatures("stacked"))
    pairs, source = paired_tiles(output, training)
    study = output / "hard-background"
    study.mkdir(exist_ok=True)
    windows = study / "paired-training-tiles.csv"
    write_csv(windows, pairs)
    report = dict(source_detectors=source, pairs=len(pairs), mining_seed=SEED, training_frames=len(training),
                  median_hard_score=float(np.median([row["hard_score"] for row in pairs])),
                  median_random_score=float(np.median([row["random_score"] for row in pairs])),
                  windows_sha256=sha_file(windows), arms={})
    print(f"Paired background windows: {len(pairs)}, median scores {report['median_hard_score']:.3f} vs {report['median_random_score']:.3f}", flush=True)

    calibration = load_labels(output, cal_ids, gray)[0]
    for name in ("random", "hard"):
        folder = study / name
        folder.mkdir(exist_ok=True)
        path = folder / "checkpoints.json"
        extra = [(row["annotation_id"], row[f"{name}_left"], row[f"{name}_top"]) for row in pairs]
        if path.is_file():
            receipt = read_json(path)
            if receipt["windows_sha256"] != report["windows_sha256"]:
                raise ValueError("A source crop list changed")
            configs = receipt["checkpoints"]
        else:
            configs = list(train_center(folder, features, training, input_profile="stacked", hard_tiles=extra))
            write_json(path, dict(windows_sha256=report["windows_sha256"], checkpoints=configs))
        report["arms"][name] = select(folder, features, configs, calibration)
        print(name, "calibration", report["arms"][name]["selected"]["calibration"], flush=True)
    write_json(study / "calibration.json", report)

    candidates = {name: raw_candidates(study / name, arm["selected"]["config"], features, warm_ids(dev_ids), "development")
                  for name, arm in report["arms"].items()}
    development = load_labels(output, dev_ids, gray)[0]
    frame_rows = {}
    for name, raw in candidates.items():
        point = report["arms"][name]["selected"]
        config = point["config"]
        metrics, frame_rows[name] = summarize(filter_by_config(with_nms(raw, config["nms"]), config), development)
        point.update(development=metrics, development_gate_passed=passed(point["calibration"]) and passed(metrics))
        print(name, "development", metrics, flush=True)
    report["paired_f1_hard_vs_random"] = paired_block_f1(frame_rows["hard"], frame_rows["random"])
    write_json(study / "comparison.json", report)
    write_csv(study / "development.csv", [dict(method=name, **row) for name, rows in frame_rows.items() for row in rows])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    cv2.setNumThreads(3)
    torch.set_num_threads(4)
    evaluate(args.output.resolve(), args.ffmpeg)
