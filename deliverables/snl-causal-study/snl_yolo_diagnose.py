"""Diagnostic sweep of saved checkpoints on dev; not used for model selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from ultralytics import YOLO

from snl_yolo_eval import evaluate_with_op, predict_split_raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=736)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    meta = json.loads((args.dataset_dir / "meta.json").read_text(encoding="utf-8"))
    gt = {int(k): np.asarray(v, dtype=np.float32).reshape(-1, 4)
          for k, v in meta["gt_boxes_by_frame"].items()}
    dev_range = tuple(meta["splits"]["dev"])
    conf_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    rows: List[Dict] = []
    for ckpt in sorted(args.weights_dir.glob("*.pt"),
                       key=lambda p: (p.name != "best.pt", p.name)):
        model = YOLO(str(ckpt))
        raw = predict_split_raw(model, args.dataset_dir / "images" / "dev", dev_range,
                                args.imgsz, args.device)
        by_conf = {}
        for conf in conf_grid:
            metrics, _ = evaluate_with_op(raw, gt, conf_thresh=conf, nms_iou=0.30)
            by_conf[f"{conf:.2f}"] = metrics
        row = {"checkpoint": ckpt.name,
               "f1_at_conf": {c: round(m["f1"] * 100, 2) for c, m in by_conf.items()},
               "recall_at_conf": {c: round(m["recall"] * 100, 2) for c, m in by_conf.items()},
               "precision_at_conf": {c: round(m["precision"] * 100, 2) for c, m in by_conf.items()},
               "tp_fp_fn_at_conf_005": {k: by_conf["0.05"][k] for k in ("tp", "fp", "fn")}}
        rows.append(row)
        best = max(by_conf.items(), key=lambda kv: kv[1]["f1"])
        print(f"{ckpt.name:12s} best dev F1 {best[1]['f1']*100:5.2f}% @conf {best[0]}  "
              f"conf0.05 R {by_conf['0.05']['recall']*100:5.2f}%  "
              f"conf0.20 F1 {by_conf['0.20']['f1']*100:5.2f}%", flush=True)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"note": "diagnostic only, not model selection",
                                            "conf_grid": conf_grid, "rows": rows}, indent=2),
                                encoding="utf-8")


if __name__ == "__main__":
    main()
