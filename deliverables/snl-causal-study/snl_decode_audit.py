"""Locate recall losses in the seven-frame detector's dense decoding stages."""
import argparse
from pathlib import Path

import numpy as np
from torch.nn import functional as F

from audit_detection import count_pairs
from experiment import load_candidates
from snl_bytetrack import REFERENCE
from snl_center import STRIDE, predict_views
from snl_detector import load_trained, torch
from snl_inputs import DEFAULT_OUTPUT, PROFILES, feature_cache, load_gray, load_labels, read_json, write_json
from videosar.snl_causal import CausalFeatures, box_iou


def dense_decoding(model, frame):
    height, width = frame.shape[-2:]
    device = next(model.parameters()).device
    with torch.inference_mode():
        output = predict_views(model, torch.from_numpy(frame.copy()).to(device)[None].float() / 255)[0]
        scores = output[0].sigmoid()
        local = scores == F.max_pool2d(scores[None, None], 3, 1, 1)[0, 0]
        indices = torch.arange(scores.numel(), device=device)
        y = torch.div(indices, scores.shape[1], rounding_mode="floor")
        x = indices % scores.shape[1]
        offsets = model.center_offsets(output[1:3, y, x].T)
        center = STRIDE * (torch.stack((x, y), dim=1) + offsets)
        half = output[3:5, y, x].T.clamp(0, 4.6).exp() / 2
        boxes = torch.cat((center - half, center + half), dim=1)
        boxes[:, 0::2].clamp_(0, width)
        boxes[:, 1::2].clamp_(0, height)
        kept = scores.masked_fill(~local, 0).flatten().topk(min(256, len(indices))).indices
    return boxes.cpu().numpy(), scores.flatten().cpu().numpy(), local.flatten().cpu().numpy(), kept.cpu().numpy()


def coverage(valid):
    return count_pairs(valid[valid.any(axis=1)])


def stage_counts(model, features, truth, published):
    counts = {stage: {"candidates": 0, "one_to_one_matches": 0} for stage in
              ("all_grid_cells", "grid_score_floor", "all_local_peaks", "peak_score_floor", "original_top256")}
    best_scores = []
    for index, target in truth.items():
        boxes, scores, peaks, top = dense_decoding(model, features[index - 2])
        floor = scores >= 0.005
        nonempty = (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
        top = top[floor[top] & peaks[top] & nonempty[top]]
        archived_boxes, archived_scores = published[index]
        np.testing.assert_allclose(boxes[top], archived_boxes, rtol=0, atol=1e-4)
        np.testing.assert_allclose(scores[top], archived_scores, rtol=0, atol=1e-6)
        valid = box_iou(boxes, target) >= 0.50
        best_scores.extend(np.where(valid, scores[:, None], 0).max(axis=0))
        masks = (nonempty, nonempty & floor, nonempty & peaks, nonempty & floor & peaks, top)
        for stage, selected in zip(counts, masks, strict=True):
            subtotal = valid[selected]
            counts[stage]["candidates"] += len(subtotal)
            counts[stage]["one_to_one_matches"] += coverage(subtotal)
    scores = np.asarray(best_scores)
    return {"frames": len(truth), "truth_boxes": len(scores), "stages": counts,
            "best_matching_grid_score_q10_q50_q90": np.quantile(scores, (0.10, 0.50, 0.90)).tolist(),
            "published_candidates_reproduced": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    output = args.output.resolve()
    config = read_json(output / REFERENCE)["config"]
    train, calibration, development = PROFILES["prospective"]
    gray = load_gray(output, args.ffmpeg)
    features = feature_cache(output, gray, max(development), CausalFeatures("stacked"))
    model = load_trained(output, config)
    report = {}
    for name, ids in (("training-sample", train[30::5]), ("validation", calibration), ("development", development)):
        truth, _, _ = load_labels(output, ids, gray)
        published = load_candidates(output, config, name)
        report[name] = stage_counts(model, features, truth, published)
        print(name, report[name], flush=True)
    write_json(output / "center-decode-audit.json", {"config": config, "splits": report})


if __name__ == "__main__":
    main()
