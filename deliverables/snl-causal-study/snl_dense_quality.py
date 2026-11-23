"""Learn to rerank dense candidates while preserving their original box geometry."""
import argparse
from pathlib import Path

import numpy as np
from torch.nn import functional as F

from audit_detection import paired_block_f1
from experiment import filter_by_config, quality, raw_candidates, select_operation, summarize, warm_ids, with_nms
from snl_bytetrack import REFERENCE
from snl_detector import backend, torch
from snl_inputs import DEFAULT_OUTPUT, PROFILES, feature_cache, load_gray, load_labels, read_json, sha_file, write_csv, write_json
from snl_patch import PatchScore, patch_inputs
from videosar.snl_causal import CausalFeatures, box_iou, matched_boxes


DENSE_LIMIT = 1024
SEED = 1831


def training_patches(predictions, features, labels):
    rng = np.random.default_rng(SEED)
    patches, scores = [], []
    for index, truth in labels.items():
        boxes, confidence = predictions[index]
        overlap = box_iou(boxes, truth).max(axis=1, initial=0)
        positive = np.flatnonzero(overlap >= 0.5)
        borderline = np.flatnonzero((overlap >= 0.2) & (overlap < 0.5))
        negative = np.flatnonzero(overlap < 0.2)
        hardest = negative[np.argsort(-confidence[negative], kind="stable")[:64]]
        selected = np.unique(np.concatenate((
            rng.permutation(positive)[:64], rng.permutation(borderline)[:48], hardest,
            rng.permutation(negative)[:48],
        )))
        patches.append(patch_inputs(features[index - 2], boxes[selected]))
        scores.append(overlap[selected])
    return np.concatenate(patches), np.concatenate(scores)


def fit_quality(patches, scores):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = backend()
    model = PatchScore(outputs=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0002)
    for epoch in range(1, 13):
        model.train()
        losses = []
        order = rng.permutation(len(scores))
        for start in range(0, len(order), 96):
            batch = order[start:start + 96]
            x = torch.from_numpy(patches[batch]).to(device).float() / 255
            target = torch.from_numpy(scores[batch]).to(device)
            loss = F.binary_cross_entropy_with_logits(model(x)[:, 0], target)
            if not torch.isfinite(loss):
                raise RuntimeError("Dense quality fit produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"Dense candidate quality epoch {epoch}/12: {np.mean(losses):.4f}", flush=True)
    return model.eval()


def learned_scores(model, features, predictions):
    scores = {}
    device = next(model.parameters()).device
    with torch.inference_mode():
        for index, (boxes, base) in predictions.items():
            values = torch.from_numpy(patch_inputs(features[index - 2], boxes))
            scores[index] = (torch.cat([model(part.to(device).float() / 255)[:, 0].sigmoid().cpu()
                                        for part in values.split(160)]).numpy() if len(boxes) else base.copy())
    return scores


def blend(predictions, learned, share):
    return {index: (boxes, ((1 - share) * scores + share * learned[index]).astype(np.float32))
            for index, (boxes, scores) in predictions.items()}


def evaluate(output, ffmpeg):
    old = DEFAULT_OUTPUT / "early-stacked"
    study = output / "dense-quality"
    study.mkdir(exist_ok=True)
    train_ids, cal_ids, dev_ids = PROFILES["prospective"]
    early_ids = read_json(old / "split-protocol.json")["train"]
    fit_ids = tuple(i for i in train_ids if 242 <= i <= 296)
    if max(early_ids) >= min(fit_ids) or max(fit_ids) >= min(cal_ids):
        raise ValueError("Earlier detector fitting, quality fitting, and calibration must be chronological")
    reference = read_json(output / REFERENCE)
    dense = reference["config"] | {"dense_candidate_limit": DENSE_LIMIT}
    earlier = next(row for row in read_json(old / "checkpoints-center-stacked.json") if row["epoch"] == 36)
    earlier |= {"contrast_gain": dense["contrast_gain"], "dense_candidate_limit": DENSE_LIMIT}
    gray = load_gray(output, ffmpeg)
    fit, cal, dev = [load_labels(output, ids, gray)[0] for ids in (fit_ids, cal_ids, dev_ids)]
    stacked, registered = [feature_cache(output, gray, max(dev_ids), CausalFeatures(kind))
                           for kind in ("stacked", "registered")]
    candidates = raw_candidates(old, earlier, stacked, fit_ids, "dense-quality-fit")
    patches, scores = training_patches(candidates, registered, fit)
    model = fit_quality(patches, scores)
    path = study / "quality.pt"
    torch.save(model.state_dict(), path)

    raw_cal = raw_candidates(output, dense, stacked, warm_ids(cal), "validation")
    raw_dev = raw_candidates(output, dense, stacked, warm_ids(dev), "development")
    learned_cal = learned_scores(model, registered, raw_cal)
    learned_dev = learned_scores(model, registered, raw_dev)
    trials = []
    for share in (0.0, 0.25, 0.50, 0.75, 1.0):
        point, cal_metrics, _ = select_operation(blend(raw_cal, learned_cal, share), cal,
                                                  confirmations=dense.get("track_support", 1))
        trials.append({"learned_weight": share, "point": point, "calibration": cal_metrics})
    selected = max(trials, key=lambda trial: quality(trial["calibration"]))
    metrics = {}
    for label, trial in (("fixed_geometry_control", trials[0]), ("selected", selected)):
        chosen = blend(raw_dev, learned_dev, trial["learned_weight"])
        config = dense | trial["point"]
        result, rows = summarize(filter_by_config(with_nms(chosen, config["nms"]), config), dev)
        metrics[label] = result
        write_csv(study / f"{label}-development.csv", rows)
        if label == "fixed_geometry_control":
            comparison_rows = rows
    ceiling = sum(len(matched_boxes(raw_dev[i][0], truth)) for i, truth in dev.items())
    report = {"detector": dense, "earlier_detector": earlier, "earlier_fit_through": max(early_ids),
              "quality_fit_ids": list(fit_ids), "quality_weights": path.relative_to(output).as_posix(),
              "checkpoint_sha256": sha_file(path), "seed": SEED, "fit_patches": len(scores),
              "fit_patches_at_iou50": int((scores >= 0.5).sum()), "candidate_ceiling": ceiling,
              "trials": trials, "selected": selected, "development": metrics,
              "paired_f1": paired_block_f1(rows, comparison_rows), "reference_development": reference["development"]}
    write_json(output / "dense-quality-comparison.json", report)
    print({key: report[key] for key in ("selected", "development", "candidate_ceiling", "paired_f1")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    evaluate(args.output.resolve(), args.ffmpeg)
