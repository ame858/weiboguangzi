"""Pretrain YOLO11 on the authors' verified, independent SIVED SAR release."""
import argparse
import io
from collections import Counter
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps

from snl_detector import backend, torch
from snl_inputs import DEFAULT_OUTPUT, read_json, sha_file, write_json
from snl_yolo import YOLO_SHA256, YOLO_VERSION, official_weights, yolo_runtime

SIVED_SHA256 = "2addb9d09b19b80db78d41fe6e972e0f7ed803970d11ca8ca8f6c157075545c4"
SIVED_COMMIT = "97616f4a6497721258652926762d8234f073c14d"
SIVED_REPOSITORY = "https://github.com/CAESAR-Radi/SIVED"
STATIC_SPLITS = (("train", "train", 837), ("valid", "val", 104))


def oriented_to_axis_aligned(source, width, height, name):
    rows, clipped = [], 0
    extent = np.array([width, height])
    for line in source.decode("utf-8-sig").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 10 or fields[8].lower() != "vehicle" or fields[9] not in ("0", "1", "2"):
            raise ValueError(f"Invalid SIVED polygon in {name}: {line!r}")
        polygon = np.asarray(fields[:8], dtype=np.float64).reshape(4, 2)
        if not np.isfinite(polygon).all():
            raise ValueError(f"Non-finite polygon in {name}")
        lower, upper = polygon.min(axis=0), polygon.max(axis=0)
        clipped += int(np.any(lower < 0) or np.any(upper > extent))
        lower, upper = lower.clip(0, extent), upper.clip(0, extent)
        if np.any(lower >= upper):
            raise ValueError(f"Degenerate visible SIVED polygon in {name}")
        center, size = (lower + upper) / (2 * extent), (upper - lower) / extent
        rows.append("0 " + " ".join(f"{value:.8f}" for value in (*center, *size)))
    return rows, clipped


def prepare_sived(output, *, inverted=False):
    archive = output / "SIVED.zip"
    if sha_file(archive) != SIVED_SHA256:
        raise ValueError("SIVED archive differs from the author's Git LFS SHA-256")
    root = output / ("sived-yolo-inverted" if inverted else "sived-yolo")
    suffix = ".png" if inverted else ".jpg"
    root.mkdir(exist_ok=True)
    manifest_path = root / "dataset-source.json"
    yaml_path = root / "sived.yaml"
    provenance = {"repository": SIVED_REPOSITORY, "commit": SIVED_COMMIT, "archive_sha256": SIVED_SHA256,
                  "labels": "static SAR vehicles with axis-aligned envelopes of published polygons"}
    if inverted:
        provenance["image_transform"] = "255 minus decoded RGB, stored losslessly"
    if manifest_path.is_file():
        previous = read_json(manifest_path)
        if any(previous.get(key) != value for key, value in provenance.items()):
            raise ValueError("The existing prepared dataset has different provenance")
        if yaml_path.is_file() and all(
                len(list((root / "images" / split).glob(f"*{suffix}"))) == count
                and len(list((root / "labels" / split).glob("*.txt"))) == count
                for _, split, count in STATIC_SPLITS):
            return yaml_path, previous
    summary = {}
    with ZipFile(archive) as source:
        names = set(source.namelist())
        for original, split, expected in STATIC_SPLITS:
            prefix = f"ImageSets/images/{original}/"
            image_members = sorted(name for name in names if name.startswith(prefix) and name.endswith(".jpg"))
            if len(image_members) != expected or len({PurePosixPath(name).name for name in image_members}) != expected:
                raise ValueError(f"Unexpected number or names of SIVED {split} images")
            images, labels = root / "images" / split, root / "labels" / split
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            bands, boxes, clipped = Counter(), 0, 0
            for member in image_members:
                name = PurePosixPath(member)
                data = source.read(member)
                with Image.open(io.BytesIO(data)) as image:
                    width, height = image.size
                    image.verify()
                annotation = f"ImageSets/labelTxt/{original}/{name.stem}.txt"
                rows, outside = oriented_to_axis_aligned(source.read(annotation), width, height, annotation)
                xml = ElementTree.fromstring(source.read(f"Annotations/{name.stem}.xml"))
                band = xml.findtext("band", default="unknown").strip().replace("-band", "")
                bands[band] += 1
                boxes += len(rows)
                clipped += outside
                if inverted:
                    with Image.open(io.BytesIO(data)) as image:
                        ImageOps.invert(image.convert("RGB")).save(images / name.with_suffix(suffix).name)
                else:
                    (images / name.name).write_bytes(data)
                (labels / f"{name.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            summary[split] = {"frames": expected, "boxes": boxes, "clipped_boxes": clipped, "images_by_band": dict(bands)}
    yaml_path.write_text(f"path: '{root.as_posix()}'\ntrain: images/train\nval: images/val\nnames:\n  0: vehicle\n",
                         encoding="utf-8")
    manifest = {**provenance, "splits": summary}
    write_json(manifest_path, manifest)
    return yaml_path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "fit"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "external-sar")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--invert", action="store_true", help="Invert source image contrast before pretraining")
    parser.add_argument("--image-size", type=int, default=512, choices=(160, 512), help="source training input size")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    output = args.output.resolve()
    data, manifest = prepare_sived(output, inverted=args.invert)
    print("Verified SIVED splits:", manifest["splits"], flush=True)
    if args.phase == "prepare":
        return
    model = yolo_runtime(output)(str(official_weights()))
    device = backend().type
    recipe = {"epochs": 25, "imgsz": args.image_size, "batch": 8, "workers": 0, "optimizer": "AdamW", "lr0": 0.001,
              "patience": 10, "close_mosaic": 5, "seed": args.seed, "amp": False}
    variant = "sived-inverted" if args.invert else "sived"
    if args.image_size != 512:
        variant += f"-{args.image_size}"
    model.train(data=str(data), device=0 if device == "cuda" else "cpu", project=str(output),
                name=f"{variant}-yolo11s-coco-seed{args.seed}", exist_ok=True, pretrained=True, deterministic=True,
                plots=False, verbose=False, **recipe)
    checkpoint = Path(model.trainer.save_dir) / "weights/best.pt"
    record = {"dataset": manifest, "device": device, "torch": str(torch.__version__), "ultralytics": YOLO_VERSION,
              "initial_sha256": YOLO_SHA256, "recipe": recipe,
              "checkpoint": {"path": checkpoint.relative_to(output).as_posix(), "sha256": sha_file(checkpoint)}}
    write_json(output / f"{variant}-fit.json", record)
    print("SIVED checkpoint:", record["checkpoint"], flush=True)


if __name__ == "__main__":
    main()
