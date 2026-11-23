"""Run the upstream Ultralytics YOLO11s baseline on the chronological SNL split."""
import os
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from PIL import Image

from snl_detector import backend
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, VIDEO, WIDTH, read_json, sha_file, write_json

YOLO_VERSION = "8.3.200"
YOLO_WEIGHTS = "yolo11s.pt"
YOLO_SHA256 = "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"
IMAGE_SIZE = 768


def official_weights():
    weights = DEFAULT_OUTPUT / YOLO_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"Obtain the official Ultralytics checkpoint at {weights}")
    if sha_file(weights) != YOLO_SHA256:
        raise ValueError("YOLO11s checkpoint differs from the pinned official release")
    return weights


def yolo_runtime(output):
    config_dir = output / "yolo-settings"
    config_dir.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_VERBOSE"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir / "matplotlib"))
    from matplotlib import font_manager
    from ultralytics import YOLO, __version__, settings
    from ultralytics.utils import USER_CONFIG_DIR

    if __version__ != YOLO_VERSION:
        raise ValueError(f"This comparison uses Ultralytics {YOLO_VERSION}, found {__version__}")
    services = ("sync", "hub", "wandb", "clearml", "comet", "mlflow", "neptune", "raytune", "dvc")
    settings.update({name: False for name in services if name in settings})
    for name in ("Arial.ttf", "Arial.Unicode.ttf"):
        destination = USER_CONFIG_DIR / name
        if not destination.is_file():
            copyfile(font_manager.findfont("DejaVu Sans"), destination)
    return YOLO


def prepare_dataset(output, features, training, calibration, *, temporal=False):
    root = output / ("yolo-temporal-input" if temporal else "yolo-input")
    manifest = {"train": list(training), "val": list(calibration), "video_sha256": sha_file(VIDEO)}
    definition = root / "dataset-source.json"
    if definition.is_file() and read_json(definition) != manifest:
        raise ValueError("Existing YOLO inputs have a different video or chronological split")
    native_limits = np.array([WIDTH, HEIGHT], dtype=np.float32)
    for split, labels in (("train", training), ("val", calibration)):
        images, annotations = root / "images" / split, root / "labels" / split
        for directory in (images, annotations):
            directory.mkdir(parents=True, exist_ok=True)
        for index, boxes in labels.items():
            raster = images / f"{index:06}.png"
            if not raster.is_file():
                frame = features[index - 2]
                encoded = np.moveaxis(frame, 0, -1) if temporal else frame[0]
                Image.fromarray(encoded).save(raster)
            centers = (boxes[:, :2] + boxes[:, 2:]) * (0.5 / native_limits)
            sizes = (boxes[:, 2:] - boxes[:, :2]) / native_limits
            coordinates = np.column_stack((centers, sizes))
            lines = ["0 " + " ".join(f"{v:.8f}" for v in row) for row in coordinates]
            (annotations / f"{index:06}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    yaml = root / "snl.yaml"
    yaml.write_text(f"path: '{root.as_posix()}'\ntrain: images/train\nval: images/val\nnames:\n  0: vehicle-shadow\n",
                    encoding="utf-8")
    write_json(definition, manifest)
    return yaml


def p2_definition(output):
    import ultralytics
    import yaml

    baseline = Path(ultralytics.__file__).resolve().parent / "cfg/models/11/yolo11.yaml"
    config = yaml.safe_load(baseline.read_text(encoding="utf-8"))
    detect = config["head"].pop()
    if detect[2] != "Detect":
        raise ValueError("The pinned YOLO11 detection head changed")
    stride8, *coarser = detect[0]
    start = len(config["backbone"]) + len(config["head"])
    config["scales"] = {"s": config["scales"]["s"]}
    config["head"].extend([
        [stride8, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 2], 1, "Concat", [1]],
        [-1, 2, "C3k2", [128, False]],
        [[start + 2, stride8, *coarser], *detect[1:]],
    ])
    path = output / "yolo11s-snl-p2.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def train_yolo(output, features, labels, calibration, seed, *, pretrained=True, temporal=False, p2=False,
               trunk_only=False, auxiliary=None):
    source = "yolo11s.yaml"
    source_digest = None
    external_dataset = None
    init = "scratch"
    if auxiliary is not None:
        fit = read_json(auxiliary)
        weights = auxiliary.parent / fit["checkpoint"]["path"]
        source_digest = sha_file(weights)
        if source_digest != fit["checkpoint"]["sha256"]:
            raise ValueError("SIVED transfer checkpoint differs from its training record")
        init = f"{auxiliary.stem.removesuffix('-fit')}-trunk"
        source = str(weights)
        external_dataset = fit["dataset"]
    elif pretrained:
        source = str(official_weights())
        source_digest = YOLO_SHA256
        init = "coco-trunk" if trunk_only else "coco"
    dataset = prepare_dataset(output, features, labels, calibration, temporal=temporal)
    yolo = yolo_runtime(output)
    device = backend().type
    recipe = {"epochs": 60, "imgsz": IMAGE_SIZE, "batch": 4, "workers": 0, "optimizer": "AdamW",
              "lr0": 0.001, "patience": 20, "close_mosaic": 8, "seed": seed, "amp": False}
    method = "temporal" if temporal else "gray"
    if temporal:
        recipe |= {"hsv_h": 0.0, "hsv_s": 0.0}
    if trunk_only or auxiliary is not None:
        torch.manual_seed(seed)
        model = yolo("yolo11s.yaml")
        head = model.model.model[-1]
        fresh = {key: value.clone() for key, value in head.state_dict().items()}
        model.load(source)
        head.load_state_dict(fresh)
    elif p2:
        if not pretrained:
            raise ValueError("The P2 comparison uses the same pretrained YOLO11s feature extractor")
        model = yolo(str(p2_definition(output))).load(source)
        init = "coco-p2"
    else:
        model = yolo(source)
    model.train(data=str(dataset), device=0 if device == "cuda" else "cpu", project=str(output),
                name=f"yolo11s-{init}{'-temporal' if temporal else ''}-seed{seed}", exist_ok=True,
                pretrained=pretrained, deterministic=True,
                plots=False, verbose=False, **recipe)
    run = Path(model.trainer.save_dir)
    for role in ("best", "last"):
        checkpoint = run / "weights" / f"{role}.pt"
        yield {"architecture": "ultralytics-yolo11", "method": method, "scale": 1, "device": device,
               "input_profile": "raw",
               "torch": str(torch.__version__), "ultralytics": YOLO_VERSION, "checkpoint_role": role,
               "path": checkpoint.relative_to(output).as_posix(), "sha256": sha_file(checkpoint),
               "base_checkpoint_sha256": source_digest, "initialization": init,
               "auxiliary_dataset": external_dataset,
               "prediction_strides": [4, 8, 16, 32] if p2 else [8, 16, 32],
               "training_label_frames": len(labels), "training": recipe}


class YoloPredictor:
    def __init__(self, output, path, config):
        self.model = yolo_runtime(output)(str(path))
        self.size = config["training"]["imgsz"]
        self.device = 0 if backend().type == "cuda" else "cpu"
        self.temporal = config["method"] == "temporal"

    def predict_boxes(self, frame):
        if self.temporal:
            image = np.ascontiguousarray(np.moveaxis(frame[::-1], 0, -1))
        else:
            image = np.repeat(frame[0, :, :, None], 3, axis=-1)
        detected = self.model.predict(image, imgsz=self.size, conf=0.005, iou=0.7, max_det=256,
                                      device=self.device, verbose=False)[0].boxes
        return detected.xyxy.cpu().numpy(), detected.conf.cpu().numpy()
