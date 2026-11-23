"""Test causal forward box propagation with fixed SNL detector weights."""
import argparse
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from audit_detection import paired_block_f1
from experiment import filter_by_config, filtered, load_candidates, quality, select_operation, summarize, with_nms
from snl_bytetrack import REFERENCE
from snl_canonical_center import prepare
from snl_detector_agreement import combine
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, WIDTH, load_gray, load_labels, read_json, write_csv, write_json
from snl_protocol import passed
from snl_registered_route import project
from videosar.snl_causal import box_iou, detection_counts


LAGS = (1, 3)
CAP_PER_HISTORY_FRAME = 64
LIMIT = np.array((WIDTH, HEIGHT))
OFFSETS = np.array(((0, 0), (-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)))


def sample(flow, points):
    x, y = points.astype(np.float32).transpose(2, 0, 1)
    return cv2.remap(flow, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def predicted_boxes(boxes, source, target, forward=None, backward=None):
    size = boxes[:, 2:] - boxes[:, :2]
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    canonical = project(center, source)
    certainty = np.ones(len(boxes), dtype=np.float32)
    if forward is not None:
        origin = project(center[:, None] + OFFSETS * size[:, None], source) / 2
        displacement = sample(forward, origin)
        destination = origin + displacement
        rolled_back = sample(backward, destination)
        cycle = np.linalg.norm(displacement + rolled_back, axis=2)
        inside = ((origin >= 0) & (origin < LIMIT / 2) & (destination >= 0) & (destination < LIMIT / 2)).all(axis=2)
        certainty = np.mean(inside * np.exp(-cycle / 1.5), axis=1).astype(np.float32)
        canonical += 2 * np.median(displacement, axis=1)
    native = project(canonical, cv2.invertAffineTransform(target))
    scale = np.linalg.norm(source[:, 0]) / np.linalg.norm(target[:, 0])
    half_size = size * (0.5 * scale)
    result = np.column_stack(((native - half_size).clip(0, LIMIT), (native + half_size).clip(0, LIMIT)))
    valid = ((native >= 0) & (native < LIMIT)).all(axis=1) & (result[:, 2:] > result[:, :2]).all(axis=1)
    return result[valid].astype(np.float32), certainty[valid], valid


def propagated_candidates(raw, canonical, transforms):
    stream = {i: cv2.resize(canonical[i - 2, 0], (WIDTH // 2, HEIGHT // 2), interpolation=cv2.INTER_AREA)
              for i in raw}
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    result = {"global": {}, "local": {}}
    for i in raw:
        history = {method: [] for method in result}
        for lag in LAGS:
            prior = i - lag
            if prior not in raw:
                continue
            boxes, scores = raw[prior]
            top = np.argsort(-scores, kind="stable")[:CAP_PER_HISTORY_FRAME]
            if not len(top):
                continue
            forward = flow.calc(stream[prior], stream[i], None)
            backward = flow.calc(stream[i], stream[prior], None)
            for method, directions in (("global", (None, None)), ("local", (forward, backward))):
                projected, certainty, valid = predicted_boxes(boxes[top], transforms[prior], transforms[i], *directions)
                confidence = scores[top][valid] * certainty * (0.9**lag)
                history[method].append((projected, confidence))
        for method, rows in history.items():
            result[method][i] = (tuple(np.concatenate(columns) for columns in zip(*rows, strict=True)) if rows
                                 else (np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)))
    return result


def add_proposals(raw, history, weight, require_support):
    result = {}
    for i, (boxes, scores) in raw.items():
        shifted, confidence = history[i]
        overlap = box_iou(shifted, boxes) >= 0.20
        support = (overlap * scores[None]).max(axis=1, initial=0)
        if require_support:
            confidence = confidence * np.minimum(support / 0.25, 1)
        result[i] = np.concatenate((boxes, shifted)), np.concatenate((scores, weight * confidence))
    return result


def coverage(raw, labels):
    return sum(detection_counts(raw[i][0], truth)[0] for i, truth in labels.items())


def evaluate(output, ffmpeg):
    train_ids, cal_ids, dev_ids = PROFILES["prospective"]
    gray = load_gray(output, ffmpeg)
    canonical, transforms, _ = prepare(output / "canonical-center-chained", gray, max(train_ids), max(dev_ids), chained=True)
    study = output / "local-flow"
    flow_config = read_json(study / "calibration.json")["selected"]["config"]
    reference = read_json(output / REFERENCE)
    release = reference["config"]

    def pool(scope):
        first = load_candidates(output, release, scope if scope == "development" else "validation")
        second = load_candidates(study, flow_config, scope)
        common = sorted(first.keys() & second.keys())
        if not common:
            raise ValueError("The source predictions cover no common frames")
        current = combine({i: first[i] for i in common}, {i: second[i] for i in common},
                          release["threshold"], flow_config["threshold"], 0, 1)
        return first, current

    _, cal = pool("calibration")
    labels = load_labels(output, cal_ids, gray)[0]
    prepared = propagated_candidates(cal, canonical, transforms)
    prefix = min(cal_ids)
    cropped = propagated_candidates({i: row for i, row in cal.items() if i <= prefix}, canonical[:prefix - 1], transforms)
    if not all(np.array_equal(a, b) for a, b in zip(prepared["local"][prefix], cropped["local"][prefix], strict=True)):
        raise ValueError("The box propagation changed for the same input prefix")

    def choose(candidates, method, weight, support):
        operation, metrics, _ = select_operation(candidates, labels)
        return dict(method=method, weight=weight, require_current_support=support, operation=operation, calibration=metrics)

    control = choose(cal, "unpropagated", 0, False)
    trials = [choose(add_proposals(cal, prepared[method], weight, support), method, weight, support)
              for method, weight, support in product(prepared, (0.5, 0.75, 1.0), (False, True))]
    selection = {method: max((t for t in trials if t["method"] == method), key=lambda t: quality(t["calibration"]))
                 for method in prepared}
    receipt = dict(lags=LAGS, max_history_boxes_per_frame=CAP_PER_HISTORY_FRAME, reference_checkpoint=release["sha256"],
                   flow_checkpoint=flow_config["sha256"], prefix_check_frame=prefix, control=control, trials=trials,
                   selected=selection, selected_by_calibration=max((control, *selection.values()), key=lambda t: quality(t["calibration"]))["method"])
    write_json(study / "propagation-calibration.json", receipt)

    original, dev = pool("development")
    prepared = propagated_candidates(dev, canonical, transforms)
    development = load_labels(output, dev_ids, gray)[0]
    baseline, baseline_rows = summarize(filter_by_config(with_nms(original, release["nms"]), release), development)
    if baseline != reference["development"]:
        raise ValueError("The released seven-frame baseline did not replay")
    rows = [dict(method="release", **row) for row in baseline_rows]
    for point in (control, *selection.values()):
        name = point["method"]
        candidates = dev if name == "unpropagated" else add_proposals(dev, prepared[name], point["weight"], point["require_current_support"])
        operation = point["operation"]
        prediction = filtered(with_nms(candidates, operation["nms"]), operation["threshold"], operation["low"])
        metrics, frame_rows = summarize(prediction, development)
        point.update(development=metrics, unfiltered_matches=coverage(candidates, development),
                     paired_f1_vs_release=paired_block_f1(frame_rows, baseline_rows),
                     development_gate_passed=passed(point["calibration"]) and passed(metrics))
        rows.extend(dict(method=name, **row) for row in frame_rows)
        print(name, metrics, "candidate coverage:", point["unfiltered_matches"], flush=True)
    receipt["release_development"] = baseline
    write_json(study / "propagation-comparison.json", receipt)
    write_csv(study / "propagation-development.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    cv2.setNumThreads(3)
    evaluate(args.output.resolve(), args.ffmpeg)
