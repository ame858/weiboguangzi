"""Calibrate YOLO11-P2 on cal(310-323) and evaluate on dev(332-379) using IoU>=0.50 1-to-1 matching."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    ax1, ay1, ax2, ay2 = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def match_frame_hungarian(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_thresh: float = 0.50
) -> Tuple[int, int, int]:
    """逐帧一对一匈牙利匹配，返回 (tp, fp, fn)。"""
    n_p, n_g = len(pred_boxes), len(gt_boxes)
    if n_p == 0:
        return 0, 0, n_g
    if n_g == 0:
        return 0, n_p, 0
    iou = box_iou_matrix(pred_boxes, gt_boxes)
    row_ind, col_ind = linear_sum_assignment(-iou)
    tp = int(sum(1 for r, c in zip(row_ind, col_ind) if iou[r, c] >= iou_thresh))
    return tp, n_p - tp, n_g - tp


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    order = np.argsort(-scores)
    keep: List[int] = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = box_iou_matrix(boxes[i:i + 1], boxes[rest])[0]
        order = rest[ious <= iou_thresh]
    return boxes[keep]


def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if (p + r) > 0 else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def predict_split_raw(
    model: YOLO,
    img_dir: Path,
    frame_range: Tuple[int, int],
    imgsz: int = 736,
    device: str = "0",
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """以极低阈值导出每个帧的候选框，便于后续在校准段快速网格扫描。"""
    preds: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    start_f, end_f = frame_range
    for fidx in range(start_f, end_f + 1):
        res = model.predict(
            source=str(img_dir / f"frame_{fidx:04d}.png"),
            imgsz=imgsz, conf=0.01, iou=0.70, max_det=100,
            device=device, verbose=False,
        )[0]
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            scores = res.boxes.conf.cpu().numpy().astype(np.float32)
        else:
            boxes, scores = np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        preds[fidx] = (boxes, scores)
    return preds


def evaluate_with_op(
    raw_preds: Dict[int, Tuple[np.ndarray, np.ndarray]],
    gt_boxes_by_frame: Dict[int, np.ndarray],
    conf_thresh: float,
    nms_iou: float,
    iou_thresh: float = 0.50,
) -> Tuple[Dict[str, float], List[Dict[str, int]]]:
    total_tp = total_fp = total_fn = 0
    per_frame: List[Dict[str, int]] = []
    for fidx in sorted(raw_preds.keys()):
        boxes, scores = raw_preds[fidx]
        mask = scores >= conf_thresh
        kept = nms_xyxy(boxes[mask], scores[mask], iou_thresh=nms_iou)
        gt = gt_boxes_by_frame.get(fidx, np.zeros((0, 4), dtype=np.float32))
        tp, fp, fn = match_frame_hungarian(kept, gt, iou_thresh=iou_thresh)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_frame.append({"frame": fidx, "tp": tp, "fp": fp, "fn": fn})
    return compute_prf(total_tp, total_fp, total_fn), per_frame


def paired_block_bootstrap(
    cand_pf: List[Dict[str, int]],
    base_pf: List[Dict[str, int]],
    block_size: int = 6,
    n_boot: int = 3000,
    seed: int = 629,
) -> Dict[str, float]:
    assert len(cand_pf) == len(base_pf)
    n = len(cand_pf)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    diffs: List[float] = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idxs = np.concatenate([(np.arange(s, s + block_size) % n) for s in starts])[:n]
        c_tp = sum(cand_pf[i]["tp"] for i in idxs)
        c_fp = sum(cand_pf[i]["fp"] for i in idxs)
        c_fn = sum(cand_pf[i]["fn"] for i in idxs)
        b_tp = sum(base_pf[i]["tp"] for i in idxs)
        b_fp = sum(base_pf[i]["fp"] for i in idxs)
        b_fn = sum(base_pf[i]["fn"] for i in idxs)
        diffs.append((compute_prf(c_tp, c_fp, c_fn)["f1"] - compute_prf(b_tp, b_fp, b_fn)["f1"]) * 100.0)
    arr = np.asarray(diffs, dtype=np.float64)
    return {"delta_f1_pp_mean": float(arr.mean()),
            "ci95_low_pp": float(np.percentile(arr, 2.5)),
            "ci95_high_pp": float(np.percentile(arr, 97.5))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=736)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--baseline-per-frame", type=Path, default=None)
    args = parser.parse_args()

    meta = json.loads((args.dataset_dir / "meta.json").read_text(encoding="utf-8"))
    gt_boxes_by_frame = {int(k): np.asarray(v, dtype=np.float32).reshape(-1, 4)
                         for k, v in meta["gt_boxes_by_frame"].items()}
    cal_range, dev_range = tuple(meta["splits"]["cal"]), tuple(meta["splits"]["dev"])

    ckpts = sorted(args.weights_dir.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"未找到检查点: {args.weights_dir}")

    conf_grid = [round(0.05 + 0.025 * i, 3) for i in range(23)]
    iou_grid = [0.30, 0.40, 0.50]
    best_cal_record: Optional[Dict] = None
    cal_summary: List[Dict] = []

    for ckpt in ckpts:
        model = YOLO(str(ckpt))
        raw_cal = predict_split_raw(model, args.dataset_dir / "images" / "cal", cal_range, args.imgsz, args.device)
        ckpt_best = None
        for nms_iou in iou_grid:
            for conf in conf_grid:
                metrics, _ = evaluate_with_op(raw_cal, gt_boxes_by_frame, conf, nms_iou)
                rec = {"checkpoint": ckpt.name, "checkpoint_path": str(ckpt),
                       "conf": conf, "nms_iou": nms_iou, "cal_metrics": metrics}
                if ckpt_best is None or metrics["f1"] > ckpt_best["cal_metrics"]["f1"]:
                    ckpt_best = rec
        cal_summary.append(ckpt_best)
        if best_cal_record is None or ckpt_best["cal_metrics"]["f1"] > best_cal_record["cal_metrics"]["f1"]:
            best_cal_record = ckpt_best

    best_model = YOLO(best_cal_record["checkpoint_path"])
    raw_dev = predict_split_raw(best_model, args.dataset_dir / "images" / "dev", dev_range, args.imgsz, args.device)
    dev_metrics, dev_per_frame = evaluate_with_op(
        raw_dev, gt_boxes_by_frame, best_cal_record["conf"], best_cal_record["nms_iou"]
    )
    result = {"selected_on_cal": best_cal_record, "dev_metrics": dev_metrics,
              "dev_per_frame": dev_per_frame, "all_checkpoints_cal_best": cal_summary}

    if args.baseline_per_frame is not None and args.baseline_per_frame.exists():
        base_pf = json.loads(args.baseline_per_frame.read_text(encoding="utf-8"))
        result["bootstrap_vs_baseline"] = paired_block_bootstrap(dev_per_frame, base_pf)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Selected on cal:", best_cal_record["checkpoint"],
          "conf=", best_cal_record["conf"], "nms_iou=", best_cal_record["nms_iou"])
    print(f"DEV (IoU>=0.50 1-to-1): P={dev_metrics['precision']*100:.2f}% "
          f"R={dev_metrics['recall']*100:.2f}% F1={dev_metrics['f1']*100:.2f}% "
          f"(TP={dev_metrics['tp']}, FP={dev_metrics['fp']}, FN={dev_metrics['fn']})")


if __name__ == "__main__":
    main()
