"""Object-level detection & multi-object tracking metrics for Video-SAR.

Replaces legacy pixel-level IoU (which produced NaNs on empty scenes with
``union_pixels=0`` and masked hundreds of pixel-level false alarms) with:
- FROC curve & Recall @ FA/frame (safely counting FPs on empty frames)
- mAP@IoU (VOC/COCO interpolated AP, default IoU=0.5 + small-target IoU=0.3)
- MOTA, IDF1, ID switches, and HOTA (DetA / AssA / HOTA)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class BoxDetection:
    """Single predicted or ground-truth bounding box in ``[x1, y1, x2, y2]`` format."""

    frame_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    class_id: int = 0
    track_id: int = -1

    def as_xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float64)


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized pairwise IoU between ``(M, 4)`` and ``(N, 4)`` xyxy boxes."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(
        0.0, boxes_a[:, 3] - boxes_a[:, 1]
    )
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(
        0.0, boxes_b[:, 3] - boxes_b[:, 1]
    )
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / np.maximum(union, 1e-12), 0.0)


def _match_frame_detections(
    preds: List[BoxDetection],
    gts: List[BoxDetection],
    iou_threshold: float,
    class_agnostic: bool = False,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Hungarian 1-to-1 matching for a single frame."""
    if not preds or not gts:
        return [], list(range(len(preds))), list(range(len(gts)))

    pred_arr = np.stack([p.as_xyxy() for p in preds], axis=0)
    gt_arr = np.stack([g.as_xyxy() for g in gts], axis=0)
    iou = compute_iou_matrix(pred_arr, gt_arr)

    if not class_agnostic:
        for i, p in enumerate(preds):
            for j, g in enumerate(gts):
                if p.class_id != g.class_id:
                    iou[i, j] = 0.0

    cost = 1.0 - iou
    row_idx, col_idx = linear_sum_assignment(cost)

    matches: List[Tuple[int, int, float]] = []
    matched_p = set()
    matched_g = set()
    for r, c in zip(row_idx, col_idx):
        if iou[r, c] >= iou_threshold:
            matches.append((int(r), int(c), float(iou[r, c])))
            matched_p.add(int(r))
            matched_g.add(int(c))

    unmatched_p = [i for i in range(len(preds)) if i not in matched_p]
    unmatched_g = [j for j in range(len(gts)) if j not in matched_g]
    return matches, unmatched_p, unmatched_g


def evaluate_detection_and_froc(
    predictions: Sequence[BoxDetection],
    ground_truths: Sequence[BoxDetection],
    num_frames: int,
    iou_threshold: float = 0.5,
    fa_points: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    class_agnostic: bool = True,
) -> Dict[str, object]:
    """Compute mAP@IoU, FROC curve, and Recall @ FA/frame safely (even on empty scenes)."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    gt_by_frame: Dict[int, List[BoxDetection]] = {}
    for g in ground_truths:
        gt_by_frame.setdefault(g.frame_id, []).append(g)

    total_gt = len(ground_truths)
    sorted_preds = sorted(predictions, key=lambda d: d.score, reverse=True)

    # Track which GT boxes have been claimed (greedy score-sorted matching for AP/FROC)
    matched_gt_flags: Dict[int, List[bool]] = {
        fid: [False] * len(boxes) for fid, boxes in gt_by_frame.items()
    }

    tp = np.zeros(len(sorted_preds), dtype=np.float64)
    fp = np.zeros(len(sorted_preds), dtype=np.float64)
    scores = np.array([p.score for p in sorted_preds], dtype=np.float64)

    for idx, pred in enumerate(sorted_preds):
        frame_gts = gt_by_frame.get(pred.frame_id, [])
        if not frame_gts:
            fp[idx] = 1.0
            continue

        pred_box = pred.as_xyxy()[None, :]
        gt_boxes = np.stack([g.as_xyxy() for g in frame_gts], axis=0)
        ious = compute_iou_matrix(pred_box, gt_boxes)[0]

        best_j = -1
        best_iou = 0.0
        for j, g in enumerate(frame_gts):
            if matched_gt_flags[pred.frame_id][j]:
                continue
            if not class_agnostic and g.class_id != pred.class_id:
                continue
            if ious[j] >= iou_threshold and ious[j] > best_iou:
                best_iou = float(ious[j])
                best_j = j

        if best_j >= 0:
            tp[idx] = 1.0
            matched_gt_flags[pred.frame_id][best_j] = True
        else:
            fp[idx] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    fa_per_frame_curve = cum_fp / float(num_frames)
    total_fp = float(cum_fp[-1]) if cum_fp.size else 0.0
    total_tp = float(cum_tp[-1]) if cum_tp.size else 0.0

    if total_gt == 0:
        # Empty-scene safe path: no NaNs, report exact FA/frame and zero-GT flag
        recall_at_fa = {f"recall_at_{fa:g}_fa": 0.0 for fa in fa_points}
        return {
            "num_frames": num_frames,
            "num_gt": 0,
            "num_pred": len(sorted_preds),
            "tp": 0,
            "fp": int(total_fp),
            "fn": 0,
            "fa_per_frame": total_fp / float(num_frames),
            "precision": 1.0 if total_fp == 0 else 0.0,
            "recall": 0.0,
            "ap": 0.0,
            "empty_scene": True,
            "recall_at_fa": recall_at_fa,
            "froc_curve": [
                {"fa_per_frame": float(fa), "recall": 0.0, "score": float(sc)}
                for fa, sc in zip(fa_per_frame_curve, scores)
            ],
        }

    recalls = cum_tp / float(total_gt)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    # All-point interpolated AP (COCO / VOC2010 standard)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    change_idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[change_idx + 1] - mrec[change_idx]) * mpre[change_idx + 1]))

    recall_at_fa: Dict[str, float] = {}
    for target_fa in fa_points:
        valid = np.where(fa_per_frame_curve <= target_fa)[0]
        rec_val = float(recalls[valid[-1]]) if valid.size > 0 else 0.0
        recall_at_fa[f"recall_at_{target_fa:g}_fa"] = rec_val

    return {
        "num_frames": num_frames,
        "num_gt": total_gt,
        "num_pred": len(sorted_preds),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_gt - total_tp),
        "fa_per_frame": total_fp / float(num_frames),
        "precision": float(total_tp / max(total_tp + total_fp, 1.0)),
        "recall": float(total_tp / float(total_gt)),
        "ap": ap,
        "empty_scene": False,
        "recall_at_fa": recall_at_fa,
        "froc_curve": [
            {"fa_per_frame": float(fa), "recall": float(rec), "score": float(sc)}
            for fa, rec, sc in zip(fa_per_frame_curve, recalls, scores)
        ],
    }


def evaluate_mot(
    predictions: Sequence[BoxDetection],
    ground_truths: Sequence[BoxDetection],
    iou_threshold: float = 0.5,
    hota_iou_thresholds: Sequence[float] = (
        0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95
    ),
) -> Dict[str, float]:
    """Compute MOTA, IDF1, ID switches, and HOTA (DetA, AssA, HOTA)."""
    frame_ids = sorted(
        {d.frame_id for d in predictions} | {g.frame_id for g in ground_truths}
    )
    preds_by_f: Dict[int, List[BoxDetection]] = {}
    gts_by_f: Dict[int, List[BoxDetection]] = {}
    for p in predictions:
        preds_by_f.setdefault(p.frame_id, []).append(p)
    for g in ground_truths:
        gts_by_f.setdefault(g.frame_id, []).append(g)

    total_gt = len(ground_truths)
    total_pred = len(predictions)
    if total_gt == 0:
        return {
            "mota": 1.0 if total_pred == 0 else 0.0,
            "idf1": 1.0 if total_pred == 0 else 0.0,
            "hota": 1.0 if total_pred == 0 else 0.0,
            "deta": 1.0 if total_pred == 0 else 0.0,
            "assa": 1.0 if total_pred == 0 else 0.0,
            "id_switches": 0.0,
            "fp": float(total_pred),
            "fn": 0.0,
            "tp": 0.0,
        }

    # --- 1. MOTA & ID Switches at primary iou_threshold ---
    total_tp = 0
    total_fp = 0
    total_fn = 0
    id_switches = 0
    last_gt_to_pred: Dict[int, int] = {}

    # For IDF1: track co-occurrence counts between GT track_id and Pred track_id
    pair_counts: Dict[Tuple[int, int], int] = {}
    gt_track_len: Dict[int, int] = {}
    pred_track_len: Dict[int, int] = {}
    for g in ground_truths:
        gt_track_len[g.track_id] = gt_track_len.get(g.track_id, 0) + 1
    for p in predictions:
        pred_track_len[p.track_id] = pred_track_len.get(p.track_id, 0) + 1

    for fid in frame_ids:
        p_list = preds_by_f.get(fid, [])
        g_list = gts_by_f.get(fid, [])
        matches, unm_p, unm_g = _match_frame_detections(
            p_list, g_list, iou_threshold=iou_threshold, class_agnostic=True
        )
        total_tp += len(matches)
        total_fp += len(unm_p)
        total_fn += len(unm_g)

        for pi, gi, _ in matches:
            gt_tid = g_list[gi].track_id
            pr_tid = p_list[pi].track_id
            if gt_tid in last_gt_to_pred and last_gt_to_pred[gt_tid] != pr_tid:
                id_switches += 1
            last_gt_to_pred[gt_tid] = pr_tid
            pair_counts[(gt_tid, pr_tid)] = pair_counts.get((gt_tid, pr_tid), 0) + 1

    mota = 1.0 - float(total_fn + total_fp + id_switches) / float(total_gt)

    # --- 2. IDF1 via global bipartite assignment of GT tracks to Pred tracks ---
    gt_tids = sorted(gt_track_len.keys())
    pr_tids = sorted(pred_track_len.keys())
    if gt_tids and pr_tids:
        cost = np.zeros((len(gt_tids), len(pr_tids)), dtype=np.float64)
        for i, gt_tid in enumerate(gt_tids):
            for j, pr_tid in enumerate(pr_tids):
                cost[i, j] = -float(pair_counts.get((gt_tid, pr_tid), 0))
        r_idx, c_idx = linear_sum_assignment(cost)
        idtp = sum(pair_counts.get((gt_tids[r], pr_tids[c]), 0) for r, c in zip(r_idx, c_idx))
    else:
        idtp = 0
    idf1 = float(2 * idtp) / float(total_gt + total_pred) if (total_gt + total_pred) > 0 else 0.0

    # --- 3. HOTA (DetA, AssA, HOTA averaged over alpha thresholds) ---
    deta_list: List[float] = []
    assa_list: List[float] = []
    hota_list: List[float] = []

    for alpha in hota_iou_thresholds:
        tp_a = 0
        fp_a = 0
        fn_a = 0
        tpa_pairs: Dict[Tuple[int, int], int] = {}
        matched_pairs_list: List[Tuple[int, int]] = []

        for fid in frame_ids:
            p_list = preds_by_f.get(fid, [])
            g_list = gts_by_f.get(fid, [])
            matches, unm_p, unm_g = _match_frame_detections(
                p_list, g_list, iou_threshold=alpha, class_agnostic=True
            )
            tp_a += len(matches)
            fp_a += len(unm_p)
            fn_a += len(unm_g)
            for pi, gi, _ in matches:
                pair = (g_list[gi].track_id, p_list[pi].track_id)
                tpa_pairs[pair] = tpa_pairs.get(pair, 0) + 1
                matched_pairs_list.append(pair)

        denom_det = tp_a + fp_a + fn_a
        deta = float(tp_a) / float(denom_det) if denom_det > 0 else 0.0

        if tp_a > 0:
            ass_sum = 0.0
            for gt_tid, pr_tid in matched_pairs_list:
                tpa = tpa_pairs[(gt_tid, pr_tid)]
                fna = gt_track_len[gt_tid] - tpa
                fpa = pred_track_len[pr_tid] - tpa
                ass_sum += float(tpa) / float(max(tpa + fna + fpa, 1))
            assa = ass_sum / float(tp_a)
        else:
            assa = 0.0

        deta_list.append(deta)
        assa_list.append(assa)
        hota_list.append(float(np.sqrt(max(deta * assa, 0.0))))

    return {
        "mota": float(mota),
        "idf1": float(idf1),
        "hota": float(np.mean(hota_list)),
        "deta": float(np.mean(deta_list)),
        "assa": float(np.mean(assa_list)),
        "id_switches": float(id_switches),
        "fp": float(total_fp),
        "fn": float(total_fn),
        "tp": float(total_tp),
    }
