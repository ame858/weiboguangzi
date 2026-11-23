import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np

from .analysis import score_masks
from .data import read_json, run_directory, write_json
from .open_compare import COMMON_THRESHOLDS, read_reference_dataset
from .small_shadow import OPERATING_FIELDS
from .stress import MATCH, MISS_STAGES, PROPOSAL, diagnostics
from .study import cleanup, read_registered
from .temporal_study import SIZE_BANDS
from .unet import load_weights, predict, runtime
from .visuals import write_csv


def threshold_probe(probability, label, threshold):
    from scipy import ndimage

    area = int(label.sum())
    best_iou, best_threshold = -1, None
    for candidate in sorted(set(COMMON_THRESHOLDS) | {threshold}):
        components, count = ndimage.label(probability >= candidate)
        sizes = np.bincount(components.ravel(), minlength=count + 1)[1:]
        overlap = np.bincount(components[label], minlength=count + 1)[1:]
        ious = overlap / (sizes + area - overlap)
        best = float(ious.max(initial=0))
        if best > best_iou:
            best_iou, best_threshold = best, candidate
    return best_iou, best_threshold


def diagnose(root, dataset):
    torch, smp, correlate = runtime()
    torch.set_num_threads(min(6, torch.get_num_threads()))
    prior = PROPOSAL.parent / "small-shadow"
    fpn = PROPOSAL.parent / "open-compare" / "models" / "fpn.json"
    models = {"unet": prior / "models" / "small_shadow.json", "fpn": fpn}
    dataset_hash = read_json(prior / "frozen_validation.json")["dataset_sha256"]
    entries = read_reference_dataset(dataset, dataset_hash)["validation"]
    samples = read_registered(entries, correlate)
    rows, totals, protocol = [], {}, {}
    for name, path in models.items():
        settings = read_json(path)
        point = {key: settings[key] for key in OPERATING_FIELDS}
        weights = path.with_name(settings["checkpoint"])
        model = load_weights(torch, smp, weights, settings, "cpu")
        protocol[name] = {"weight_sha256": settings["weight_sha256"],
                          "config_sha256": sha256(path.read_bytes()).hexdigest(), **point}
        aggregate = dict.fromkeys(("matched", "false_positive", "missed"), 0)
        for sample in samples:
            scene = sample["entry"]["scene_id"]
            truth = sample["truth"]
            scores = predict(torch, model, sample["aligned"], settings, "cpu", 32)
            thresholded = scores >= point["threshold"]
            closed = cleanup(scores, point["threshold"], 1, point["closing"])
            final = cleanup(scores, **point)
            raw_rows, closed_rows, final_rows = (score_masks(mask, truth, MATCH) for mask in (thresholded, closed, final))
            for key in aggregate:
                aggregate[key] += final_rows[key]
            output = root / "scenes" / scene / name
            output.mkdir(parents=True)
            np.savez_compressed(output / "stages.npz", scores=scores, thresholded=thresholded,
                                closed=closed, final=final)
            for i, frame_stages in enumerate(zip(raw_rows["frames"], closed_rows["frames"], final_rows["frames"])):
                row = diagnostics(*frame_stages)
                area = int(truth[i].sum())
                band = SIZE_BANDS[np.digitize(area, (12, 24, 48))] if area else "no_label"
                row.update(method=name, scene_id=scene, condition=sample["entry"]["condition"], frame=i,
                           label_area_px=area, band=band, fg_probability_max=float(scores[i, truth[i]].max()) if area else None,
                           fg_probability_mean=float(scores[i, truth[i]].mean()) if area else None,
                           geometry_blocks_match=bool(area and area * 2 < point["min_area"]))
                if row["miss_stage"] == "raw_segmentation":
                    best, threshold = threshold_probe(scores[i], truth[i], point["threshold"])
                    row.update(probe_best_iou=best, probe_threshold=threshold, probe_match=best >= MATCH)
                rows.append(row)
        totals[name] = aggregate

    summary = []
    for name in models:
        model_rows = [row for row in rows if row["method"] == name]
        for band in (*SIZE_BANDS, "no_label"):
            selected = [row for row in model_rows if row["band"] == band]
            stages = Counter(row["miss_stage"] for row in selected if row["missed"])
            summary.append({"method": name, "band": band, "frames": len(selected),
                            "matched": sum(row["target_visible"] and not row["missed"] for row in selected),
                            "missed": sum(stages.values()), **{stage: stages[stage] for stage in MISS_STAGES},
                            "threshold_probe_matches": sum(row.get("probe_match", False) for row in selected),
                            "area_rule_impossible": sum(row["geometry_blocks_match"] for row in selected),
                            "false_alarms": sum(row["false_alarms"] for row in selected)})
        assert sum(row["false_alarms"] for row in model_rows) == totals[name]["false_positive"]
        assert sum(row["missed"] for row in model_rows) == totals[name]["missed"]
    write_csv(root / "by_area_and_stage.csv", summary)
    write_csv(root / "per_frame.csv", rows)
    write_json(root / "protocol.json", {"dataset_sha256": dataset_hash, "split": "validation", "sequences": len(entries),
                                       "match_iou": MATCH, "methods": protocol, "probe_thresholds": COMMON_THRESHOLDS,
                                       "totals": totals})


def main():
    parser = argparse.ArgumentParser(description="Diagnose misses on the frozen small-shadow validation split.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    with run_directory(args.output_dir, "miss-diagnostics", {"dataset": str(args.dataset.resolve())}) as root:
        diagnose(root, args.dataset)
    print(root)


if __name__ == "__main__":
    main()
