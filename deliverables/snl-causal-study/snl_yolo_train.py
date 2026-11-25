"""Train YOLO11-P2 on strictly causal SNL Video SAR dataset."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from shutil import copyfile

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="dataset.yaml 路径")
    parser.add_argument("--model-yaml", type=Path, default=Path("yolo11n_p2.yaml"))
    parser.add_argument("--weights", type=Path, required=True, help="本地预训练权重路径，如 yolo11n.pt")
    parser.add_argument("--project", type=Path, required=True, help="输出父目录")
    parser.add_argument("--name", type=str, default="yolo11n-p2-s1729")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=736)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"离线预训练权重不存在: {args.weights}")

    # Ultralytics 8.3.200 runs an AMP check that fetches yolo11n.pt from GitHub into
    # the current directory even when the model was loaded from a local checkpoint.
    # Place the pinned offline weights there first so the check hits a local file.
    if not Path("yolo11n.pt").is_file():
        copyfile(args.weights, "yolo11n.pt")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"

    model = YOLO(str(args.model_yaml)).load(str(args.weights))
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        cos_lr=True,
        save=True,
        save_period=args.save_period,
        patience=0,  # 禁止依赖 Ultralytics mAP 早停，统一由校准段正式口径选权重
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.25,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.3,
        mixup=0.0,
        close_mosaic=20,
        verbose=True,
    )


if __name__ == "__main__":
    main()
