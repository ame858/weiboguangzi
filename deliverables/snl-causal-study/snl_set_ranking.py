"""Compare pointwise and neighborhood-aware one-to-one scores on fixed dense SNL boxes."""
import argparse
from dataclasses import dataclass
from itertools import islice, product
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

from audit_detection import paired_block_f1
from experiment import CONFIDENCES, filtered, load_candidates, quality, raw_candidates, summarize, warm_ids
from snl_box_quality import GEOMETRY_KEYS
from snl_center import train_center
from snl_detector import backend
from snl_forward_dense_fusion import candidate_scores, records, suppress
from snl_fusion import measurement
from snl_group_rescore import centers, crossfit
from snl_inputs import DEFAULT_OUTPUT, PROFILES, feature_cache, load_gray, load_labels, read_json, sha_file, write_csv, write_json
from snl_local_shadow_proposals import coverage
from snl_protocol import passed
from videosar.snl_causal import CausalFeatures, box_iou


EPOCHS = (8, 16, 24)
NEIGHBORS = 12
SEED = 4103
OPERATIONS = (("none", None), ("hard", 0.50), ("vote", 0.35))
EXPANDED_FOLDS = ((100, 123, 160), (160, 183, 218))


@dataclass
class Frame:
    boxes: np.ndarray
    base: np.ndarray
    data: np.ndarray
    neighbors: np.ndarray
    edges: np.ndarray


def describe(raw, features):
    frames = {}
    for i, (boxes, score) in raw.items():
        sizes = boxes[:, 2:] - boxes[:, :2]
        center = centers(boxes)
        points = np.column_stack((center / np.array([5, 8]), np.log2(sizes)))
        neighbors = cKDTree(points).query(points, k=min(NEIGHBORS + 1, len(boxes)))[1][:, 1:]
        neighbor_boxes = boxes[neighbors]
        top_left = np.maximum(boxes[:, None, :2], neighbor_boxes[:, :, :2])
        bottom_right = np.minimum(boxes[:, None, 2:], neighbor_boxes[:, :, 2:])
        area = np.maximum(bottom_right - top_left, 0).prod(axis=2)
        intersection = area / np.maximum(sizes.prod(axis=1)[:, None] + sizes.prod(axis=1)[neighbors] - area, 1e-6)
        delta = (center[neighbors] - center[:, None]) / sizes[:, None]
        ratio = np.log2(sizes[neighbors] / sizes[:, None])
        edges = np.concatenate((delta.clip(-3, 3), ratio.clip(-3, 3), intersection[:, :, None]), axis=2)
        data = np.column_stack((measurement(features[i - 2], boxes), score))
        frames[i] = Frame(boxes, score, data.astype(np.float32), neighbors, edges.astype(np.float32))
    return frames


class CandidateSet(nn.Module):
    def __init__(self, width, use_neighbors):
        super().__init__()
        self.use_neighbors = use_neighbors
        self.encode = nn.Sequential(nn.Linear(width, 32), nn.LayerNorm(32), nn.SiLU())
        self.message = nn.Sequential(nn.Linear(37, 32), nn.SiLU(), nn.Linear(32, 24))
        self.head = nn.Sequential(nn.Linear(56, 32), nn.LayerNorm(32), nn.SiLU(), nn.Linear(32, 2))
        nn.init.constant_(self.head[-1].bias, -3.0)

    def forward(self, data, neighbors, edges):
        hidden = self.encode(data)
        if self.use_neighbors and neighbors.shape[1]:
            context = self.message(torch.cat((hidden[neighbors] - hidden[:, None], edges), dim=-1)).amax(dim=1)
        else:
            dummy_edges = hidden.new_zeros((len(hidden), 5))
            context = self.message(torch.cat((hidden, dummy_edges), dim=-1))
        return self.head(torch.cat((hidden, context), dim=1))


def labels_for_set(overlap, confidence):
    """Allocate one positive to each coverable truth with geometric eligibility fixed."""
    valid = overlap >= 0.50
    maximum = min(overlap.shape)
    priority = valid * (1 + (overlap + 0.35 * confidence[:, None]) / (2 + 2 * maximum))
    a, b = linear_sum_assignment(priority, maximize=True)
    target = np.zeros(len(overlap), dtype=np.float32)
    target[a[valid[a, b]]] = 1
    return target


def tensor_frames(frames, center, spread, device):
    return {i: (torch.from_numpy(((f.data - center) / spread).clip(-8, 8)).to(device),
                torch.from_numpy(f.neighbors).to(device), torch.from_numpy(f.edges).to(device))
            for i, f in frames.items()}


def fit(study, name, fitting, truth, center, spread):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = backend()
    model = CandidateSet(len(center), name == "neighbors").to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, EPOCHS[-1], eta_min=1e-4)
    inputs = tensor_frames(fitting, center, spread, device)
    overlaps = {i: box_iou(fitting[i].boxes, boxes) for i, boxes in truth.items()}
    quality_targets = {i: torch.from_numpy(values.max(axis=1, initial=0)).to(device) for i, values in overlaps.items()}
    checkpoints = []
    for epoch in range(1, EPOCHS[-1] + 1):
        model.train()
        losses = []
        for i in rng.permutation(list(truth)):
            prediction = model(*inputs[i])
            box_probability = prediction[:, 0].sigmoid()
            selected = labels_for_set(overlaps[i], box_probability.detach().cpu().numpy())
            target = torch.from_numpy(selected).to(device)
            positive = F.softplus(-prediction[:, 0]) * (1 - box_probability).square() * target
            negative = F.softplus(prediction[:, 0]) * box_probability.square() * (1 - target)
            assignment = (positive.sum() + negative.sum()) / target.sum().clamp_min(1)
            scores = quality_targets[i]
            weights = torch.where(scores >= 0.20, 1.0, 0.05)
            quality_loss = (F.binary_cross_entropy_with_logits(prediction[:, 1], scores, reduction="none") * weights).sum() / weights.sum()
            loss = assignment + 0.25 * quality_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optim.step()
            losses.append(float(loss.detach()))
        schedule.step()
        if epoch in EPOCHS:
            path = study / f"{name}-{epoch}.pt"
            torch.save(model.state_dict(), path)
            checkpoints.append(dict(epoch=epoch, path=path.name, sha256=sha_file(path)))
            print(f"{name} epoch {epoch}/{EPOCHS[-1]} loss {np.mean(losses):.4f}", flush=True)
    return checkpoints


@torch.inference_mode()
def probabilities(study, name, checkpoint, frames, center, spread):
    device = backend()
    path = study / checkpoint["path"]
    if sha_file(path) != checkpoint["sha256"]:
        raise ValueError("Candidate scorer checkpoint checksum differs")
    model = CandidateSet(len(center), name == "neighbors").to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    result = {}
    for i, row in tensor_frames(frames, center, spread, device).items():
        probabilities = model(*row).sigmoid()
        result[i] = (probabilities[:, 0] * probabilities[:, 1]).sqrt().cpu().numpy()
    return result


def candidate_pool(frames, learned, weight, method, strength):
    raw = {i: (frame.boxes, (1 - weight) * frame.base + weight * learned[i]) for i, frame in frames.items()}
    if method == "none":
        return raw
    return suppress(raw, method, strength)


def choose(frames, learned, labels, weights, operations=OPERATIONS):
    trials = []
    for weight, (method, strength) in product(weights, operations):
        candidates = candidate_pool(frames, learned, weight, method, strength)
        rows = [(float(t), summarize(filtered(candidates, float(t), None), labels)[0]) for t in CONFIDENCES]
        threshold, calibration = max(rows, key=lambda row: quality(row[1]))
        trials.append(dict(weight=weight, method=method, strength=strength, threshold=threshold, calibration=calibration))
    return max(trials, key=lambda row: quality(row["calibration"]))


def audit_without_suppression(study, cal_frames, dev_frames, calibration, development, saved):
    center = np.asarray(saved["feature_center"], dtype=np.float32)
    spread = np.asarray(saved["feature_scale"], dtype=np.float32)
    outcomes = {}
    for name, arm in saved["arms"].items():
        checkpoint = arm["selected"].get("checkpoint")
        if checkpoint:
            cal_score = probabilities(study, name, checkpoint, cal_frames, center, spread)
            dev_score = probabilities(study, name, checkpoint, dev_frames, center, spread)
            weights = (0.25, 0.5, 0.75, 1.0)
        else:
            cal_score = {i: f.base for i, f in cal_frames.items()}
            dev_score = {i: f.base for i, f in dev_frames.items()}
            weights = (0.0,)
        point = choose(cal_frames, cal_score, calibration, weights, (("none", None),))
        raw = candidate_pool(dev_frames, dev_score, point["weight"], "none", None)
        kept = filtered(raw, point["threshold"], None)
        result, _ = summarize(kept, development)
        eligible = sum(np.count_nonzero(box_iou(kept[i][0], target).max(axis=1, initial=0) >= 0.50)
                       for i, target in development.items())
        outcomes[name] = dict(selected_without_suppression=point, development=result,
                              geometrically_eligible_unmatched=eligible - result["tp"],
                              development_with_selected_suppression=arm["development"])
        print(name, "without suppression:", result, "unmatched despite valid IoU:", eligible - result["tp"], flush=True)
    write_json(study / "without-suppression.json", dict(fixed_checkpoint=True, outcomes=outcomes))


def expanded_fit(study, output, gray, features, transforms, parent, recipe):
    stack = feature_cache(output, gray, EXPANDED_FOLDS[-1][-1], CausalFeatures("stacked"))
    out, truth, audits = {}, {}, []
    for cutoff, first, last in EXPANDED_FOLDS:
        directory = study / f"detector-through-{cutoff}"
        directory.mkdir(exist_ok=True)
        fit_ids = list(range(2, cutoff + 1))
        holdout = list(range(first, last + 1))
        if max(fit_ids) >= min(warm_ids(holdout)):
            raise ValueError("Prefix training overlaps held-out candidate inference")
        receipt = directory / "checkpoint.json"
        if receipt.is_file():
            audit = read_json(receipt)
            if audit["training_ids"] != fit_ids or audit["held_out_ids"] != holdout:
                raise ValueError("The earlier source checkpoint has a different time split")
            config = audit["config"]
            if sha_file(directory / config["path"]) != config["sha256"]:
                raise ValueError("Earlier source checkpoint checksum differs")
        else:
            fitting = load_labels(output, fit_ids, gray)[0]
            config = list(islice(train_center(directory, stack, fitting, input_profile="stacked"), 3))[-1]
            options = {k: recipe["fit_detector"][k] for k in ("contrast_gain", "dense_candidate_limit")
                       if k in recipe["fit_detector"]}
            config = config | options
            audit = dict(training_ids=fit_ids, held_out_ids=holdout, config=config)
            write_json(receipt, audit)
        raw = raw_candidates(directory, config, stack, warm_ids(holdout), "set-fitting")
        measured = records(raw, features, transforms)
        out.update({i: candidate_scores(parent, measured[i], recipe["selected"]["refine"], frame_ranked=True) for i in holdout})
        truth.update(load_labels(output, holdout, gray)[0])
        audits.append(audit)
        print(f"Generated independent set-scoring candidates for {first}–{last}", flush=True)
    return out, truth, audits


def run(output, ffmpeg, *, audit_only=False, expanded=False):
    archive = output / "forward-dense-frame-ranking"
    recipe = read_json(archive / "comparison.json")
    baseline = read_json(archive / "suppression-comparison.json")
    paths = [archive / f"{part}.txt" for part in ("ranker", *GEOMETRY_KEYS)]
    if any(sha_file(path) != recipe["weights"][path.name] for path in paths):
        raise ValueError("Parent candidate scorer weights changed")
    ranker, *geometry = (lgb.Booster(model_file=str(path)) for path in paths)
    _, cal_ids, dev_ids = PROFILES["prospective"]
    fit_ids = recipe["fit_ids"]
    if max(fit_ids) >= min(cal_ids) or max(cal_ids) >= min(dev_ids):
        raise ValueError("Set score fitting and evaluation must be chronological")
    gray = load_gray(output, ffmpeg)
    cal = load_labels(output, cal_ids, gray)[0]
    transforms = {int(i): np.asarray(matrix) for i, matrix in
                  read_json(output / "canonical-center-chained/canonical-registration.json")["transforms"].items()}
    features = np.load(output / "motion-integration/motion-integration.npy", mmap_mode="r", allow_pickle=False)
    def scoring_frames(scope):
        inputs = records(load_candidates(output, recipe["evaluation_detector"], scope), features, transforms)
        return describe({i: candidate_scores((ranker, geometry), entry, recipe["selected"]["refine"], frame_ranked=True)
                         for i, entry in inputs.items()}, features)

    study = output / ("set-ranking-expanded" if expanded else "set-ranking")
    study.mkdir(exist_ok=True)
    cal_frames = scoring_frames("validation")
    if audit_only:
        saved = read_json(study / "comparison.json")
        if saved["candidate_scorers"] != recipe["weights"]:
            raise ValueError("The audit needs the same archived candidate scorers")
        dev_frames = scoring_frames("development")
        development = load_labels(output, dev_ids, gray)[0]
        audit_without_suppression(study, cal_frames, dev_frames, cal, development, saved)
        return

    truth = load_labels(output, fit_ids, gray)[0]
    earlier = load_candidates(DEFAULT_OUTPUT / "early-stacked", recipe["fit_detector"], "forward-dense-fit")
    fit_candidates = crossfit(records(earlier, features, transforms), truth)
    extra_folds = []
    if expanded:
        extra, extra_truth, extra_folds = expanded_fit(study, output, gray, features, transforms, (ranker, geometry), recipe)
        if set(extra_truth) & set(truth):
            raise ValueError("Expanded set scoring reuses an existing fitting label")
        fit_candidates.update(extra)
        truth = dict(sorted((truth | extra_truth).items()))
    fitting = describe(fit_candidates, features)
    data = np.concatenate([entry.data for entry in fitting.values()])
    center, spread = data.mean(axis=0), np.maximum(data.std(axis=0), 0.001)
    zero = {i: f.base for i, f in cal_frames.items()}
    arms = {"parent": dict(selected=choose(cal_frames, zero, cal, (0.0,)))}
    for name in ("pointwise", "neighbors"):
        checkpoints = fit(study, name, fitting, truth, center, spread)
        trials = []
        for entry in checkpoints:
            scores = probabilities(study, name, entry, cal_frames, center, spread)
            trials.append(choose(cal_frames, scores, cal, (0.25, 0.5, 0.75, 1.0)) | {"checkpoint": entry})
        arms[name] = dict(selected=max(trials, key=lambda step: quality(step["calibration"])), trials=trials)
        print(name, arms[name]["selected"], flush=True)
    chosen = max(arms, key=lambda name: quality(arms[name]["selected"]["calibration"]))
    report = dict(fit_ids=list(truth), chronological_extra_folds=extra_folds, calibration_ids=cal_ids, development_ids=dev_ids,
                  fitting_candidates=len(data), epochs=EPOCHS, neighbors=NEIGHBORS, feature_center=center.tolist(),
                  feature_scale=spread.tolist(), candidate_scorers=recipe["weights"], selected_by_calibration=chosen, arms=arms)
    write_json(study / "calibration.json", report)

    dev_frames = scoring_frames("development")
    development = load_labels(output, dev_ids, gray)[0]
    parent = {i: (f.boxes, f.base) for i, f in dev_frames.items()}
    operation = baseline["selected"]
    reference, original = summarize(filtered(suppress(parent, operation["method"], operation["strength"]),
                                             operation["threshold"], operation["low"],
                                             confirmations=recipe["evaluation_detector"].get("track_support", 1)), development)
    if reference != baseline["development"] or coverage(parent, development) != recipe["coverage"]["selected_geometry"]:
        raise ValueError("The archived baseline and dense geometry did not replay")
    rows = [dict(method="reference", **r) for r in original]
    for name, arm in arms.items():
        point = arm["selected"]
        scores = (probabilities(study, name, point["checkpoint"], dev_frames, center, spread) if "checkpoint" in point
                  else {i: f.base for i, f in dev_frames.items()})
        candidates = candidate_pool(dev_frames, scores, point["weight"], point["method"], point["strength"])
        prediction = filtered(candidates, point["threshold"], None)
        metrics, per_frame = summarize(prediction, development)
        fixed_budget = {}
        for limit in (10, 20, 40, 80):
            limited = {i: (boxes[np.argsort(-score, kind="stable")[:limit]], None)
                       for i, (boxes, score) in candidates.items()}
            fixed_budget[limit] = coverage(limited, development)
        arm.update(development=metrics, candidate_ceiling=coverage(candidates, development),
                   fixed_budget=fixed_budget, paired_vs_reference=paired_block_f1(per_frame, original),
                   development_gate_passed=passed(point["calibration"]) and passed(metrics))
        rows.extend(dict(method=name, **r) for r in per_frame)
        print(name, metrics, "fixed-budget coverage:", fixed_budget, flush=True)
    report.update(reference=reference, raw_candidate_ceiling=coverage(parent, development))
    write_json(study / "comparison.json", report)
    write_csv(study / "development.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--audit-without-suppression", action="store_true")
    parser.add_argument("--expanded-fitting", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(3)
    torch.set_num_threads(4)
    run(args.output.resolve(), args.ffmpeg, audit_only=args.audit_without_suppression, expanded=args.expanded_fitting)
