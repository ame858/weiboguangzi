"""Compare seven-frame SNL training with and without proposal simulation pretraining."""
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, shift

from audit_detection import appearance, count_pairs, paired_block_f1
from experiment import develop, filter_by_config, load_candidates, quality, summarize, with_nms
from snl_center import train_center
from snl_detector import TILE, backend, torch
from snl_inputs import DEFAULT_OUTPUT, PROFILES, ROOT, feature_cache, load_gray, load_labels, read_json, sha_file, write_json
from snl_motion_aug import local_darkness
from videosar.proposal_data import scene_digest
from videosar.snl_causal import CausalFeatures, box_iou


def soften_simulated_shadow(gray, mask, box, desired):
    current = gray.astype(np.float32) / 255
    excess = local_darkness(current, box) - desired
    if excess <= 0:
        return gray
    opacity = np.maximum(0, 1 - distance_transform_edt(~mask) / 3)
    return np.rint(np.clip(current + excess * opacity, 0, 1) * 255).astype(np.uint8)


def source_training(dataset, contrast_reference=None):
    manifest = read_json(dataset / "dataset.json")
    if manifest["profile"] != "proposal":
        raise ValueError("Expected the proposal simulation dataset")
    entries = [entry for entry in manifest["entries"] if entry["split"] == "train"]
    features = np.empty((20 * len(entries), 7, TILE, TILE), dtype=np.uint8)
    labels = {}
    rng = np.random.default_rng(53129)
    for entry in entries:
        path = dataset / entry["path"]
        if scene_digest(path) != entry["sha256"]:
            raise ValueError(f"Simulation checksum mismatch: {path}")
        frames = np.load(path / "frames.npy", allow_pickle=False)
        masks = np.load(path / "masks.npy", allow_pickle=False)
        if frames.shape != (20, 64, 80) or masks.shape != frames.shape:
            raise ValueError(f"Unexpected simulation grid: {path}")
        metadata = read_json(path / "metadata.json")
        low, high = np.percentile(frames, [2, 98])
        normalized = np.rint(np.clip((frames - low) / max(high - low, 1e-6), 0, 1) * 255).astype(np.uint8)
        scanner = CausalFeatures("stacked")
        top, left = (TILE - frames.shape[1]) // 2, (TILE - frames.shape[2]) // 2
        desired = float(rng.choice(contrast_reference)) if contrast_reference is not None else None
        for image, mask, (dx, dy) in zip(normalized, masks, metadata["observed_offsets_xy_px"], strict=True):
            moved = shift(mask, (dy, dx), order=0, mode="reflect", prefilter=False).astype(bool)
            y, x = np.nonzero(moved)
            if x.size:
                box = np.array((x.min(), y.min(), x.max() + 1, y.max() + 1), dtype=np.float32)
                if desired is not None:
                    image = soften_simulated_shadow(image, moved, box, desired)
                boxes = (box + [left, top, left, top])[None].astype(np.float32)
            else:
                boxes = np.empty((0, 4), dtype=np.float32)
            padded = np.pad(image, ((top, top), (left, left)), mode="edge")
            index = len(labels)
            features[index] = np.rint(scanner.network_channels(padded) * 255).astype(np.uint8)
            labels[index + 2] = boxes
    return features, labels, entries


def score_selected(output, decision, labels):
    config = decision["config"]
    raw = load_candidates(output, config, "development")
    selected = filter_by_config(with_nms(raw, config["nms"]), config)
    metrics, rows = summarize(selected, labels)
    if metrics != decision["development"]:
        raise ValueError("Cached predictions do not reproduce the decision")
    ceiling = sum(count_pairs(box_iou(raw[i][0], truth) >= 0.50) for i, truth in labels.items())
    return rows, ceiling


def compare(output, dataset, ffmpeg, *, match_contrast=False):
    train_ids, validation_ids, development_ids = PROFILES["prospective"]
    gray = load_gray(output, ffmpeg)
    train, validation, development = (load_labels(output, ids, gray)[0]
                                      for ids in (train_ids, validation_ids, development_ids))
    features = feature_cache(output, gray, max(development_ids), CausalFeatures("stacked"))
    control_study = output / "simulation-pretraining"
    study = output / "simulation-pretraining-contrast" if match_contrast else control_study
    study.mkdir(exist_ok=True)
    control_study.mkdir(exist_ok=True)
    source_config = study / "checkpoint-simulation-source.json"
    source_hash = sha_file(dataset / "dataset.json")
    if source_config.exists():
        source = read_json(source_config)
        if source["simulation_manifest_sha256"] != source_hash:
            raise ValueError("Source checkpoint was made from a different simulation manifest")
    else:
        contrast_reference = None
        if match_contrast:
            real_contrast = np.concatenate(tuple(appearance(gray, train).values()))
            contrast_reference = real_contrast[real_contrast > 0]
        simulated, boxes, entries = source_training(dataset, contrast_reference)
        source, = train_center(study, simulated, boxes, input_profile="stacked", pretraining_epochs=8)
        source |= {"simulation_manifest_sha256": source_hash, "scenes": len(entries),
                   "geometry_scenes": sum(entry["source"] == "geometry" for entry in entries)}
        if match_contrast:
            observed = np.concatenate(tuple(appearance(simulated[:, 0], boxes).values()))
            source["contrast_matching"] = {
                "seed": 53129, "snl_positive_q25_q50_q75": np.percentile(contrast_reference, (25, 50, 75)).tolist(),
                "simulated_q25_q50_q75": np.percentile(observed, (25, 50, 75)).tolist(),
            }
        write_json(source_config, source)
        del simulated, boxes

    decisions = {}
    runtime = (backend().type, str(torch.__version__))
    for arm, initial, work_dir in (("control", None, control_study), ("transfer", source, study)):
        registry_file = work_dir / f"checkpoints-{arm}.json"
        if registry_file.exists():
            checkpoints = read_json(registry_file)
            if any(config.get("pretrained_sha256") != (initial and initial["sha256"]) for config in checkpoints):
                raise ValueError("Checkpoints were made from a different initialization")
            if any((config["device"], config["torch"]) != runtime for config in checkpoints):
                raise ValueError(f"Resume with the device and PyTorch version recorded in {registry_file}")
        else:
            checkpoints = list(train_center(work_dir, features, train, input_profile="stacked", pretrained=initial))
            write_json(registry_file, checkpoints)
        decisions[arm] = max(develop(work_dir, features, validation, development, checkpoints),
                             key=lambda row: quality(row["validation"]))
    control, transfer = decisions["control"], decisions["transfer"]
    control_rows, control_ceiling = score_selected(control_study, control, development)
    transfer_rows, transfer_ceiling = score_selected(study, transfer, development)
    difference = paired_block_f1(transfer_rows, control_rows)
    report = {"seed": 1729, "match_iou": 0.50, "source": source, "control": control, "transfer": transfer,
              "development_truth_boxes": sum(map(len, development.values())),
              "development_max_matches_all_candidates": {"control": control_ceiling, "transfer": transfer_ceiling},
              "development_f1_difference": transfer["development"]["f1"] - control["development"]["f1"],
              "paired_uncertainty": difference,
              "passes_both_development_splits": all(
                  transfer[split][metric] is not None and transfer[split][metric] >= 0.80
                  for split in ("validation", "development") for metric in ("recall", "precision"))}
    report_name = f"{study.name}-comparison.json"
    write_json(output / report_name, report)
    for key in ("development_max_matches_all_candidates", "development_f1_difference", "paired_uncertainty"):
        print(key, report[key], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--dataset", type=Path, default=ROOT / "runs/proposal-final-data")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--match-contrast", action="store_true", help="Match the source to early SNL shadow contrast")
    args = parser.parse_args()
    compare(args.output.resolve(), args.dataset.resolve(), args.ffmpeg, match_contrast=args.match_contrast)
