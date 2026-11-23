"""Measure the paired SNL effect of SIVED pretraining with a fresh detection head."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from audit_detection import paired_block_f1
from experiment import load_candidates, model_key, quality
from snl_inputs import DEFAULT_OUTPUT, PROFILES, VIDEO, load_gray, load_labels, read_json, sha_file, write_json
from snl_motion_aug import local_darkness
from snl_protocol import passed
from snl_sived import prepare_sived
from snl_yolo import YOLO_SHA256
from videosar.snl_causal import detection_counts, detection_metrics


def appearance_summary(samples, input_sizes):
    areas, relative_areas, darkness = [], [], []
    for frame, boxes in samples:
        frame_areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        areas.extend(frame_areas)
        relative_areas.extend(frame_areas / max(frame.shape) ** 2)
        normalized = frame.astype(np.float32) / 255.0
        darkness.extend(local_darkness(normalized, box) for box in boxes)
    positive = int(np.count_nonzero(np.asarray(darkness) > 0))
    quantiles = np.percentile(relative_areas, (25, 50, 75))
    return {"boxes": len(areas), "area_px_q25_q50_q75": np.percentile(areas, (25, 50, 75)).tolist(),
            "resized_area_px_q25_q50_q75": {size: (quantiles * size ** 2).tolist() for size in input_sizes},
            "darkness_q25_q50_q75": np.percentile(darkness, (25, 50, 75)).tolist(),
            "darker_than_local_background": positive, "darker_fraction": positive / len(darkness)}


def sived_training_samples(directory, suffix):
    for path in sorted((directory / "images/train").glob(f"*{suffix}")):
        label = directory / "labels/train" / path.with_suffix(".txt").name
        rows = [row.split()[1:] for row in label.read_text(encoding="utf-8").splitlines() if row.strip()]
        with Image.open(path) as image:
            frame = np.asarray(image.convert("L"))
        native = np.asarray(rows, dtype=float).reshape(-1, 4) * np.tile(frame.shape[::-1], 2)
        centers, sizes = np.split(native, 2, axis=1)
        yield frame, np.column_stack((centers - sizes / 2, centers + sizes / 2))


def audit_appearance(output, auxiliary, ffmpeg, *, inverted=False):
    definition, source = prepare_sived(auxiliary, inverted=inverted)
    source_stats = appearance_summary(sived_training_samples(definition.parent, ".png" if inverted else ".jpg"), (160, 512))
    if source_stats["boxes"] != source["splits"]["train"]["boxes"]:
        raise ValueError("Prepared SIVED box counts differ from the verified training archive")
    gray = load_gray(output, ffmpeg)
    labels = load_labels(output, PROFILES["prospective"][0], gray)[0]
    target_stats = appearance_summary(((gray[index - 2], boxes) for index, boxes in labels.items()), (768,))
    report = {"definition": "median 4-pixel rim gray minus median inside box, on native images scaled by 1/255",
              "sived_static_vehicle_train": source_stats, "snl_moving_shadow_train": target_stats}
    variant = "sived-inverted" if inverted else "sived"
    write_json(output / f"{variant}-snl-training-appearance.json", report)
    return report


def selected_checkpoint(output, phase, initialization, seed):
    registry = read_json(output / f"checkpoints-{phase}-seed{seed}.json")
    candidates = []
    for checkpoint in registry:
        decision = read_json(output / f"decision-{model_key(checkpoint)}.json")
        config = decision["config"]
        if config["initialization"] != initialization or any(config[k] != v for k, v in checkpoint.items()):
            raise ValueError(f"Decision and checkpoint record disagree for {checkpoint['path']}")
        if sha_file(output / config["path"]) != config["sha256"]:
            raise ValueError(f"Check the checkpoint hash for {config['path']}")
        candidates.append(decision)
    return max(candidates, key=lambda row: quality(row["validation"]))


def development_rows(output, decision, ids):
    path = output / f"development-{model_key(decision['config'])}.csv"
    with path.open(newline="", encoding="utf-8") as file:
        rows = [{name: int(value) for name, value in row.items()} for row in csv.DictReader(file)]
    if tuple(row["annotation_id"] for row in rows) != ids:
        raise ValueError(f"Unexpected development frames in {path}")
    counts = [sum(row[key] for row in rows) for key in ("tp", "fp", "fn")]
    if detection_metrics(counts, len(ids)) != decision["development"]:
        raise ValueError(f"Development counts do not reproduce {path}")
    return rows


def proposal_ceiling(output, control, transfer, development, ffmpeg, names):
    gray = load_gray(output, ffmpeg)
    labels = load_labels(output, development, gray)[0]
    reference = read_json(output / "decision-radar-center-stacked-epoch36-contrast1.35-support3.json")["config"]
    first, second = names
    sources = {"center": reference, first: control, second: transfer}
    pools = {name: load_candidates(output, config, "development") for name, config in sources.items()}
    groups = (("center",), (first,), (second,), ("center", first), ("center", second))
    report = {}
    for group in groups:
        matched, proposals = 0, 0
        for index, truth in labels.items():
            boxes = np.concatenate([pools[name][index][0] for name in group])
            matched += detection_counts(boxes, truth)[0]
            proposals += len(boxes)
        report["+".join(group)] = {"boxes": proposals, "max_matches_iou50": int(matched)}
    return {"ground_truth": sum(map(len, labels.values())), "all_exported_candidates": report}


def matched_source(config, fit):
    return (fit["initial_sha256"] == YOLO_SHA256 and config["base_checkpoint_sha256"] == fit["checkpoint"]["sha256"]
            and config["auxiliary_dataset"] == fit["dataset"])


def compare(output, auxiliary, seed, ffmpeg, *, variant="sived", temporal=False):
    has_source_control = variant != "sived" and not temporal
    names = ("sived" if has_source_control else "coco", variant)
    first_phase = "yolo11-sived" if has_source_control else "yolo11-coco-transfer"
    suffix = "-temporal" if temporal else ""
    control = selected_checkpoint(output, first_phase + suffix, f"{names[0]}-trunk", seed)
    transfer = selected_checkpoint(output, f"yolo11-{names[1]}{suffix}", f"{names[1]}-trunk", seed)
    fit = read_json(auxiliary / f"{names[1]}-fit.json")
    base, candidate = control["config"], transfer["config"]
    shared = ("architecture", "method", "device", "torch", "ultralytics", "input_profile", "prediction_strides",
              "training_label_frames", "training", "evaluation_device", "evaluation_torch")
    expected_method = "temporal" if temporal else "gray"
    if any(base[key] != candidate[key] for key in shared) or base["method"] != expected_method:
        raise ValueError("The comparison requires matched SNL recipes and evaluation runtimes")
    if not matched_source(candidate, fit):
        raise ValueError("The transfer lineage does not match the experiment")
    if has_source_control:
        original = read_json(auxiliary / "sived-fit.json")
        if not matched_source(base, original):
            raise ValueError("The source reference must use the pinned original SIVED checkpoint")
        if variant == "sived-inverted":
            unchanged = {key: value for key, value in fit["dataset"].items() if key != "image_transform"}
            if (not fit["dataset"].get("image_transform") or unchanged != original["dataset"]
                    or fit["recipe"] != original["recipe"]):
                raise ValueError("The source arms must differ only in image polarity")
        elif fit["dataset"] != original["dataset"] or fit["recipe"] != (original["recipe"] | {"imgsz": 160}):
            raise ValueError("The source arms must differ only in pretraining input size")
    elif base["base_checkpoint_sha256"] != YOLO_SHA256:
        raise ValueError("The reference must use the pinned COCO initialization")
    train, calibration, development = PROFILES["prospective"]
    manifest = read_json(output / f"yolo{suffix}-input/dataset-source.json")
    protocol = read_json(output / "split-protocol.json")
    if (manifest != {"train": list(train), "val": list(calibration), "video_sha256": sha_file(VIDEO)}
            or protocol["development"] != list(development) or base["training_label_frames"] != len(train)):
        raise ValueError("The comparison requires the prospective split on the pinned video")
    control_rows = development_rows(output, control, development)
    transfer_rows = development_rows(output, transfer, development)
    if any(a["tp"] + a["fn"] != b["tp"] + b["fn"] for a, b in zip(control_rows, transfer_rows, strict=True)):
        raise ValueError("Development ground-truth counts differ between arms")
    difference = {key: transfer["development"][key] - control["development"][key]
                  if None not in (transfer["development"][key], control["development"][key]) else None
                  for key in ("precision", "recall", "f1")}
    report = {
        "selection": "calibration only, for the checkpoint and operating point",
        "match_iou": 0.50, "seed": seed, "sived_pretraining": fit, "control": control, "transfer": transfer,
        "transfer_minus_control": difference | paired_block_f1(transfer_rows, control_rows),
        "transfer_passed_both_development_splits": all(passed(transfer[split]) for split in ("validation", "development")),
        "proposal_ceiling": proposal_ceiling(output, base, candidate, development, ffmpeg, names),
    }
    if has_source_control:
        report["original_sived_pretraining"] = original
    write_json(output / f"{names[1]}{suffix}-transfer-comparison-seed{seed}.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--auxiliary", type=Path, default=DEFAULT_OUTPUT / "external-sar")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--audit-data", action="store_true")
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--invert", dest="variant", action="store_const", const="sived-inverted",
                         help="Compare original and inverted SIVED pretraining")
    variant.add_argument("--small-source", dest="variant", action="store_const", const="sived-160",
                         help="Compare SIVED pretraining at 160 and 512 pixels")
    parser.set_defaults(variant="sived")
    parser.add_argument("--temporal", action="store_true", help="Compare both temporal arms against the COCO trunk")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if args.audit_data:
        print(audit_appearance(args.output.resolve(), args.auxiliary.resolve(), args.ffmpeg,
                               inverted=args.variant == "sived-inverted"), flush=True)
        return
    report = compare(args.output.resolve(), args.auxiliary.resolve(), args.seed, args.ffmpeg,
                     variant=args.variant, temporal=args.temporal)
    for name in ("control", "transfer"):
        print(name, report[name]["development"], flush=True)
    print("Paired difference", report["transfer_minus_control"], flush=True)


if __name__ == "__main__":
    main()
