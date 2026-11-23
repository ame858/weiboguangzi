"""Test integer translation phases of a frozen real SNL center detector."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import nms

from audit_detection import count_pairs, paired_block_f1
from experiment import CONFIDENCES, filter_by_config, load_candidates, quality, summarize, with_nms
from snl_bytetrack import REFERENCE
from snl_center import center_boxes_at, decode_center_map, predict_views
from snl_detector import load_trained
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, VIDEO, WIDTH, feature_cache, load_gray, load_labels
from snl_inputs import read_json, sha_file, write_csv, write_json
from snl_rolling_nwd import center_metrics
from videosar.data import run_directory
from videosar.snl_causal import CausalFeatures, box_iou


IDENTITY = (0, 0)
PHASES = {"single": (IDENTITY,), "same_phase": (IDENTITY, (8, 0), (0, 8), (8, 8)),
          "mixed_phase": (IDENTITY, (9, 0), (0, 9), (9, 9))}
MARGIN = 24
PER_VIEW = 2048
BUDGET = 1024
REFINE_IOU = 0.25
BLENDS = (0.0, 0.5, 1.0)


def interior(boxes):
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    return ((centers >= MARGIN) & (centers < (WIDTH - MARGIN, HEIGHT - MARGIN))).all(axis=1)


def translate(tensor, dx, dy):
    height, width = tensor.shape[-2:]
    return F.pad(tensor, (dx, 0, dy, 0), mode="replicate")[..., :height, :width].contiguous()


def dense_view(model, output, dx, dy):
    probability = output[0].sigmoid().flatten()
    strength, index = probability.topk(4 * PER_VIEW)
    boxes = center_boxes_at(model, output, index, HEIGHT, WIDTH).cpu().numpy()
    boxes -= np.array([dx, dy, dx, dy], dtype=np.float32)
    np.clip(boxes, 0, (WIDTH, HEIGHT, WIDTH, HEIGHT), out=boxes)
    valid = (boxes[:, 2:] > boxes[:, :2]).all(axis=1) & interior(boxes)
    selected = np.flatnonzero(valid)[:PER_VIEW]
    if len(selected) != PER_VIEW:
        raise ValueError("A translated image has too few interior proposals for the common per-view budget")
    return boxes[selected], strength.cpu().numpy()[selected]


def merged_view(views, shifts, iou):
    boxes = np.concatenate([views[shift][0] for shift in shifts])
    scores = np.concatenate([views[shift][1] for shift in shifts])
    order = np.argsort(-scores, kind="stable")
    boxes, scores = boxes[order], scores[order]
    priority = torch.arange(len(boxes), 0, -1, dtype=torch.float32)
    kept = nms(torch.from_numpy(boxes), priority, iou).numpy()
    return boxes[:BUDGET], boxes[kept[:BUDGET]], scores[kept[:BUDGET]]


def refine_geometry(reference, views, shifts):
    total = reference.copy()
    votes = np.ones((len(reference), 1), dtype=np.float32)
    for shift in shifts:
        if shift == IDENTITY or not len(reference):
            continue
        boxes, scores = views[shift]
        overlap = box_iou(reference, boxes)
        best = overlap.argmax(axis=1)
        allowed = interior(reference) & (overlap[np.arange(len(reference)), best] >= REFINE_IOU) & (scores[best] >= 0.005)
        total[allowed] += boxes[best[allowed]]
        votes[allowed] += 1
    return total / votes, int((votes > 1).sum())


@torch.inference_mode()
def predict(model, sequence, ids, raw, release, nms_iou):
    device = next(model.parameters()).device
    all_shifts = tuple(dict.fromkeys(shift for shifts in PHASES.values() for shift in shifts))
    geometry, dense, common_counts = {}, {}, {}
    for index in ids:
        input_frame = torch.from_numpy(sequence[index - 2].copy()).to(device).float()[None] / 255
        views = {}
        for dx, dy in all_shifts:
            frame = input_frame if (dx, dy) == IDENTITY else translate(input_frame, dx, dy)
            output = predict_views(model, frame)[0]
            if (dx, dy) == IDENTITY:
                boxes, scores = decode_center_map(model, output, HEIGHT, WIDTH)
                np.testing.assert_allclose(boxes, raw[index][0], rtol=0, atol=1e-4)
                np.testing.assert_allclose(scores, raw[index][1], rtol=0, atol=1e-6)
            views[dx, dy] = dense_view(model, output, dx, dy)
        geometry[index] = {name: refine_geometry(release[index][0], views, shifts) for name, shifts in PHASES.items()}
        merged = {name: merged_view(views, shifts, nms_iou) for name, shifts in PHASES.items()}
        common_count = min(len(post) for _, post, _ in merged.values())
        dense[index] = {name: (pre[:common_count], post[:common_count], scores[:common_count])
                        for name, (pre, post, scores) in merged.items()}
        common_counts[index] = common_count
    return geometry, dense, common_counts


def refined(geometry, release, name, blend):
    return {index: (boxes + blend * (geometry[index][name][0] - boxes), scores, tracks)
            for index, (boxes, scores, tracks) in release.items() if index in geometry}


def dense_predictions(dense, name, *, threshold=None, counts=None):
    out = {}
    for index, views in dense.items():
        _, boxes, scores = views[name]
        count = int((scores >= threshold).sum()) if counts is None else counts[index]
        if count > len(boxes):
            raise ValueError("A dense phase arm did not supply the requested number of outputs")
        out[index] = boxes[:count], scores[:count], np.full(count, -1)
    return out


def choose(geometry, dense, release, truth):
    restricted = {index: boxes[interior(boxes)] for index, boxes in truth.items()}
    detail = {}
    for name in PHASES:
        geometry_trials = [{"blend": blend, "iou50": summarize(refined(geometry, release, name, blend), truth)[0]}
                           for blend in BLENDS]
        dense_trials = [{"threshold": float(t), "iou50": summarize(dense_predictions(dense, name, threshold=t), restricted)[0]}
                        for t in CONFIDENCES]
        detail[name] = dict(geometry=max(geometry_trials, key=lambda row: quality(row["iou50"])),
                            dense=max(dense_trials, key=lambda row: quality(row["iou50"])),
                            geometry_trials=geometry_trials, dense_trials=dense_trials)
    return detail


def report(geometry, dense, common_counts, release, truth, points, scope):
    restricted = {index: boxes[interior(boxes)] for index, boxes in truth.items()}
    fixed_counts = {index: int(interior(release[index][0]).sum()) for index in truth}
    results, rows, coverage_rows = {}, [], []
    metrics_rows = {}
    for name in PHASES:
        a = refined(geometry, release, name, points[name]["geometry"]["blend"])
        b = dense_predictions(dense, name, threshold=points[name]["dense"]["threshold"])
        c = dense_predictions(dense, name, counts=fixed_counts)
        variants = {"geometry_only": (a, truth), "dense_frozen": (b, restricted), "dense_equal_output": (c, restricted)}
        result = {}
        for variant, (predictions, labels) in variants.items():
            iou, counts = summarize(predictions, labels)
            rows.extend(dict(split=scope, arm=name, variant=variant, **row) for row in counts)
            result[variant] = dict(iou50=iou, center5=center_metrics(predictions, labels))
            metrics_rows[name, variant] = counts
        coverage = []
        for index, targets in restricted.items():
            pre, post, _ = dense[index][name]
            coverage.append(dict(split=scope, arm=name, annotation_id=index, truth=len(targets), kept=common_counts[index],
                                 pre_nms=count_pairs(box_iou(pre, targets) >= 0.50),
                                 post_nms=count_pairs(box_iou(post, targets) >= 0.50),
                                 geometry_votes=geometry[index][name][1]))
        coverage_rows.extend(coverage)
        results[name] = result | {key: sum(row[key] for row in coverage) for key in ("pre_nms", "post_nms", "geometry_votes")}
    variant_names = ("geometry_only", "dense_frozen", "dense_equal_output")
    results["paired_phase_comparison"] = {variant: paired_block_f1(metrics_rows["mixed_phase", variant],
                                                                   metrics_rows["same_phase", variant])
                                          for variant in variant_names}
    return results, rows, coverage_rows


def run(source, output, ffmpeg):
    reference = read_json(source / REFERENCE)
    config = reference["config"]
    if config.get("stitch_inference") or config.get("occupancy_power") or config.get("dense_candidate_limit"):
        raise ValueError("This comparison requires the original local-peak center checkpoint")
    _, cal_ids, dev_ids = PROFILES["prospective"]
    settings = dict(config=config, video_sha256=sha_file(VIDEO), phases=PHASES, margin=MARGIN, per_view=PER_VIEW,
                    budget=BUDGET, geometry_match_iou=REFINE_IOU, geometry_blends=BLENDS, match_iou=0.50,
                    calibration_ids=list(cal_ids), development_ids=list(dev_ids))
    gray = load_gray(source, ffmpeg)
    features = feature_cache(source, gray, max(dev_ids), CausalFeatures("stacked"))
    model = load_trained(source, config)
    with run_directory(output, "snl-integer-phase", settings) as root:
        result, counts, coverage = {}, [], []
        for scope, ids in (("validation", cal_ids), ("development", dev_ids)):
            raw = load_candidates(source, config, scope)
            release = filter_by_config(with_nms(raw, config["nms"]), config)
            geometry, dense, common = predict(model, features, ids, raw, release, config["nms"])
            truth = load_labels(source, ids, gray)[0]
            if scope == "validation":
                points = choose(geometry, dense, release, truth)
                write_json(root / "frozen-selection.json", dict(settings=settings, points=points))
                print("Frozen calibration", {key: (p["geometry"]["blend"], p["dense"]["threshold"])
                                              for key, p in points.items()}, flush=True)
            results, rows, coverage_rows = report(geometry, dense, common, release, truth, points, scope)
            archived = reference[scope]
            if any(archived[key] != value for key, value in results["single"]["geometry_only"]["iou50"].items()):
                raise ValueError(f"{scope} original output differs from the archived detector")
            result[scope] = results
            counts.extend(rows)
            coverage.extend(coverage_rows)
            print(scope, {name: {variant: results[name][variant]["iou50"] for variant in
                                 ("geometry_only", "dense_frozen", "dense_equal_output")} for name in PHASES}, flush=True)
        write_json(root / "comparison.json", dict(settings=settings, points=points, results=result))
        write_csv(root / "frame-rows.csv", counts)
        write_csv(root / "coverage-rows.csv", coverage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.source.resolve(), (args.output or args.source / "integer-phase").resolve(), args.ffmpeg)
