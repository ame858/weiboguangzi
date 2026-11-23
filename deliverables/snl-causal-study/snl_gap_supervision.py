"""Estimate recoverable single-frame annotation gaps and test their training value."""
import argparse
import hashlib
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from audit_detection import paired_block_f1
from experiment import filter_by_config, quality, raw_candidates, select_operation, summarize, warm_ids, with_nms
from snl_center import train_center
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, WIDTH, feature_cache, load_gray, load_labels, read_json, sha_file, write_csv, write_json
from snl_motion_aug import local_darkness
from snl_protocol import passed
from videosar.snl_causal import CausalFeatures, box_iou, detection_counts, detection_metrics


GRID = np.array((WIDTH, HEIGHT))
WINDOW = 6
HOLDOUT_SEED = 5179
RECOVERY_PRECISION = 0.80
MIN_RECOVERED = 16


def centers(boxes):
    return (boxes[:, :2] + boxes[:, 2:]) / 2


def match_position(template_frame, target_frame, box, expected):
    left, top, right, bottom = np.rint(box).astype(int)
    template = template_frame[top:bottom, left:right]
    height, width = template.shape
    if min(width, height) < 3 or template.std() < 1:
        return None
    nominal = np.rint(expected - np.array((width, height)) / 2).astype(int)
    low = np.maximum(nominal - WINDOW, 0)
    high = np.minimum(nominal + WINDOW + [width, height], GRID)
    x0, y0 = low
    x1, y1 = high
    if x1 - x0 < width or y1 - y0 < height:
        return None
    score = cv2.matchTemplate(target_frame[y0:y1, x0:x1], template, cv2.TM_CCOEFF_NORMED)
    _, correlation, _, location = cv2.minMaxLoc(score)
    return low + location + np.array((width, height)) / 2, correlation


def frame_candidates(before_frame, current, after_frame, before, after):
    left, right = centers(before), centers(after)
    sizes_left, sizes_right = before[:, 2:] - before[:, :2], after[:, 2:] - after[:, :2]
    distance = np.linalg.norm(left[:, None] - right, axis=2)
    size_change = np.abs(np.log2(sizes_left[:, None] / sizes_right)).max(axis=2)
    valid = (distance <= 19) & (size_change < 0.8)
    a, b = linear_sum_assignment(np.where(valid, distance + 4 * size_change, 1e4))
    rows = []
    for j, k in zip(a[valid[a, b]], b[valid[a, b]], strict=True):
        midpoint = (left[j] + right[k]) / 2
        previous = match_position(before_frame, current, before[j], midpoint)
        following = match_position(after_frame, current, after[k], midpoint)
        if previous is None or following is None:
            continue
        earlier_center, earlier_score = previous
        later_center, later_score = following
        final = (midpoint + earlier_center + later_center) / 3
        half = (sizes_left[j] + sizes_right[k]) / 4
        box = np.concatenate(((final - half).clip(0, GRID), (final + half).clip(0, GRID))).astype(np.float32)
        confidence = min(earlier_score, later_score)
        consistency = np.linalg.norm(earlier_center - later_center)
        darkness = local_darkness(current.astype(np.float32), box)
        rows.append((box, confidence, consistency, darkness))
    return rows


def candidates(gray, training):
    result = {}
    for i in training:
        if i - 1 in training and i + 1 in training:
            result[i] = frame_candidates(gray[i - 3], gray[i - 2], gray[i - 1], training[i - 1], training[i + 1])
    return result


def select_boxes(rows, known, params):
    accepted = [box for box, corr, gap, dark in rows
                if corr >= params["min_correlation"] and gap <= params["max_center_disagreement"] and dark >= params["min_darkness"]]
    if not accepted:
        return np.empty((0, 4), dtype=np.float32)
    boxes = np.stack(accepted)
    overlap = box_iou(boxes, known).max(axis=1, initial=0)
    distance = np.linalg.norm(centers(boxes)[:, None] - centers(known), axis=2).min(axis=1, initial=np.inf)
    return boxes[(overlap < 0.20) & (distance > 4)]


def mask_training_targets(raw, training, allowed_ids):
    rng = np.random.default_rng(HOLDOUT_SEED)
    masked = {}
    for i in allowed_ids[::3]:
        if i not in raw or len(training[i]) < 2:
            continue
        targets = training[i]
        missing = np.zeros(len(targets), dtype=bool)
        missing[rng.choice(len(targets), size=max(1, len(targets) // 3), replace=False)] = True
        masked[i] = targets[~missing], targets[missing]
    return masked


def recovery_metrics(raw, masked, params):
    counts = np.sum([detection_counts(select_boxes(raw[i], known, params), hidden)
                     for i, (known, hidden) in masked.items()], axis=0)
    return detection_metrics(counts, len(masked))


def recovery_calibration(raw, training, allowed_ids):
    masked = mask_training_targets(raw, training, allowed_ids)
    trials = []
    for corr, distance, darkness in product((0.70, 0.77, 0.84, 0.90, 0.94), (2, 3, 4), (0, 2, 4)):
        params = dict(min_correlation=corr, max_center_disagreement=distance, min_darkness=darkness)
        trials.append(dict(params=params, metrics=recovery_metrics(raw, masked, params)))
    eligible = [row for row in trials if (row["metrics"]["precision"] or 0) >= RECOVERY_PRECISION
                and row["metrics"]["tp"] >= MIN_RECOVERED]
    selected = max(eligible, key=lambda row: (row["metrics"]["recall"], row["metrics"]["precision"])) if eligible else None
    return dict(selected=selected, trials=trials, masking_seed=HOLDOUT_SEED, masked_ids=list(masked),
                minimum_precision=RECOVERY_PRECISION, minimum_recovered=MIN_RECOVERED,
                hidden_boxes=sum(len(hidden) for _, hidden in masked.values()))


def training_digest(labels):
    digest = hashlib.sha256()
    for i, boxes in labels.items():
        digest.update(int(i).to_bytes(4, "little"))
        digest.update(len(boxes).to_bytes(4, "little"))
        digest.update(boxes.tobytes())
    return digest.hexdigest()


def train_or_replay(folder, features, labels, seed):
    folder.mkdir(exist_ok=True)
    registry = folder / "checkpoints.json"
    identity = training_digest(labels)
    if registry.is_file():
        saved = read_json(registry)
        if saved["labels_sha256"] != identity or any(config["seed"] != seed for config in saved["checkpoints"]):
            raise ValueError("The checkpoints used different training observations or a different seed")
        return saved["checkpoints"]
    configs = list(train_center(folder, features, labels, input_profile="stacked", seed=seed))
    write_json(registry, dict(labels_sha256=identity, checkpoints=configs))
    return configs


def compare_at_control_count(predictions, results, development):
    control_cfg = results["archived_control"]["selected"]["config"]
    control = filter_by_config(with_nms(predictions["archived_control"], control_cfg["nms"]), control_cfg)
    treatment_cfg = results["gap_supervision"]["selected"]["config"]
    treatment = with_nms(predictions["gap_supervision"], treatment_cfg["nms"])
    limited = {}
    for i in development:
        boxes, scores = treatment[i]
        count = len(control[i][0])
        if len(boxes) < count:
            raise ValueError("The treatment has fewer candidates than the requested detection count")
        order = np.argsort(-scores, kind="stable")[:count]
        limited[i] = boxes[order], scores[order], None
    base, base_rows = summarize(control, development)
    score, rows = summarize(limited, development)
    return dict(control=base, gap_supervision=score, paired_f1=paired_block_f1(rows, base_rows))


def evaluate(output, ffmpeg, seed):
    train_ids, cal_ids, dev_ids = PROFILES["prospective"]
    gray = load_gray(output, ffmpeg)
    train = load_labels(output, train_ids, gray)[0]
    raw = candidates(gray, train)
    early_ids, later_ids, _ = PROFILES["original"]
    recovery = recovery_calibration(raw, train, early_ids)
    chosen = recovery["selected"]
    if chosen:
        later = mask_training_targets(raw, train, [i for i in later_ids if i in train])
        recovery["later_training_recheck"] = dict(masked_ids=list(later), metrics=recovery_metrics(raw, later, chosen["params"]))
    study = output / ("gap-supervision" if seed == 1729 else f"gap-supervision-seed{seed}")
    study.mkdir(exist_ok=True)
    write_json(study / "recovery-calibration.json", recovery)
    if not chosen:
        print("No recovery operating point met the fitting precision and support requirements", flush=True)
        return
    augmented, added = {}, []
    for i, boxes in train.items():
        recovered = select_boxes(raw.get(i, ()), boxes, chosen["params"])
        augmented[i] = np.concatenate((boxes, recovered))
        added.extend(dict(annotation_id=i, x0=float(b[0]), y0=float(b[1]), x1=float(b[2]), y1=float(b[3])) for b in recovered)
    report = dict(recovery=chosen, seed=seed, original_boxes=sum(map(len, train.values())), added_boxes=len(added),
                  labels_sha256=training_digest(augmented), modified_frames=len({r["annotation_id"] for r in added}))
    write_csv(study / "proposed-training-boxes.csv", added, fields=("annotation_id", "x0", "y0", "x1", "y1"))
    write_json(study / "training-labels.json", report)
    print("Training augmentation:", report, flush=True)
    if not added:
        return

    features = feature_cache(output, gray, max(dev_ids), CausalFeatures("stacked"))
    configs = train_or_replay(study, features, augmented, seed)
    if seed == 1729:
        baseline = output / "simulation-pretraining"
        baseline_configs = read_json(baseline / "checkpoints-control.json")
    else:
        baseline = study / "matched-control"
        baseline_configs = train_or_replay(baseline, features, train, seed)
    scopes = {"archived_control": (baseline, baseline_configs), "gap_supervision": (study, configs)}
    calibration = load_labels(output, cal_ids, gray)[0]
    results = {}
    for name, (directory, checkpoints) in scopes.items():
        trials = []
        for config in checkpoints:
            if sha_file(directory / config["path"]) != config["sha256"]:
                raise ValueError("Checkpoint checksum differs")
            predictions = raw_candidates(directory, config, features, warm_ids(cal_ids), "validation")
            operation, metrics, _ = select_operation(predictions, calibration)
            trials.append(dict(config=config | operation, calibration=metrics))
        results[name] = dict(selected=max(trials, key=lambda x: quality(x["calibration"])), trials=trials)
    write_json(study / "detector-calibration.json", results)
    predictions = {name: raw_candidates(folder, results[name]["selected"]["config"], features, warm_ids(dev_ids), "development")
                   for name, (folder, _) in scopes.items()}
    development = load_labels(output, dev_ids, gray)[0]
    rows = {}
    for name, raw in predictions.items():
        choice = results[name]["selected"]
        config = choice["config"]
        score, rows[name] = summarize(filter_by_config(with_nms(raw, config["nms"]), config), development)
        choice.update(development=score, development_gate_passed=passed(choice["calibration"]) and passed(score))
        print(name, score, flush=True)
    report.update(detectors=results, paired_f1_vs_control=paired_block_f1(rows["gap_supervision"], rows["archived_control"]))
    report["fixed_output_count"] = compare_at_control_count(predictions, results, development)
    write_json(study / "comparison.json", report)
    write_csv(study / "development.csv", [dict(method=name, **row) for name, values in rows.items() for row in values])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    cv2.setNumThreads(3)
    torch.set_num_threads(4)
    evaluate(args.output.resolve(), args.ffmpeg, args.seed)
