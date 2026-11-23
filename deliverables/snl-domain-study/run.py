"""Train controlled U-Nets and evaluate them on locked SNL frames."""
import argparse
import base64
import io
import subprocess
from hashlib import sha1, sha256
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
from scipy import ndimage

from prepare import (
    DEFAULT_OUTPUT, GROUPS, SOURCE, TEST_BLOCKS, read_json, save_csv, save_json,
)
from candidate_probe import box_mean, nearby_controls
from evaluate import box_iou, counts, label_boxes, match, metrics, pearson, thumbnail
from factorial import (
    EPOCHS, SEEDS, THRESHOLDS, UNET, VALIDATION_IDS, create_model, features,
    probability_maps, runtime, train,
)

MAX_BOXES = 20
MIN_AREA = 5


def load_samples(ffmpeg, labels_dir, ids):
    blobs = {x["path"]: x["sha"] for x in read_json(SOURCE / "jiang-tree.json")["tree"]}
    labels, frame_indices = [], set()
    for annotation_id in ids:
        name = f"{annotation_id:06}.json"
        raw = (labels_dir / name).read_bytes()
        blob = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if blob != blobs[f"data/json/{name}"]:
            raise ValueError(f"{name} failed Git blob verification")
        data = read_json(labels_dir / name)
        if (data["imageWidth"], data["imageHeight"]) != (644, 704):
            raise ValueError(f"{name} has unexpected dimensions")
        if any(s["shape_type"] != "rectangle" or s["label"] != "car" for s in data["shapes"]):
            raise ValueError(f"{name} has unexpected annotation types")
        video_index = annotation_id + 148
        rectangles = [shape["points"] for shape in data["shapes"]]
        ref = Image.open(io.BytesIO(base64.b64decode(data["imageData"]))).convert("L")
        labels.append((annotation_id, video_index, blob, rectangles,
                       np.asarray(ref.resize((80, 88), Image.Resampling.BILINEAR))))
        frame_indices.update((*range(video_index - 2, video_index + 3), video_index - 6))

    order = sorted(frame_indices)
    select = "+".join(rf"eq(n\,{i})" for i in order)
    raw_frames = subprocess.check_output([
        ffmpeg, "-v", "error", "-i", str(SOURCE / "eubank-snl.mp4"),
        "-vf", f"select={select},crop=660:720:310:0,format=gray",
        "-fps_mode", "passthrough", "-threads", "1", "-f", "rawvideo", "-",
    ])
    frames = dict(zip(order, np.frombuffer(raw_frames, np.uint8).reshape(-1, 720, 660), strict=True))
    samples, audit = [], []
    for annotation_id, n, blob, boxes, reference in labels:
        comparison = [pearson(reference, thumbnail(frames[i])) for i in range(n - 2, n + 3)]
        gap = max(comparison) - comparison[2]
        if comparison[2] < 0.96 or gap > 0.006:
            raise ValueError(f"{annotation_id}: reference frame alignment failed ({comparison})")
        samples.append((features(frames[n], frames[n - 6]), label_boxes({"rectangles": boxes})))
        audit.append(dict(annotation_id=annotation_id, video_index=n, git_blob_sha1=blob,
                          correlation=comparison[2], deficit_to_best_neighbor=gap))
    return samples, audit


def candidate_boxes(scores, threshold):
    regions, total = ndimage.label(scores >= threshold)
    areas = np.bincount(regions.ravel(), minlength=total + 1)
    use = np.flatnonzero(areas[1:] >= MIN_AREA) + 1
    sums = np.bincount(regions.ravel(), weights=scores.ravel(), minlength=total + 1)
    ranked = use[np.argsort(-sums[use] / areas[use], kind="stable")[:MAX_BOXES]]
    slices = ndimage.find_objects(regions)
    boxes = []
    for label in ranked:
        ys, xs = slices[label - 1]
        boxes.append((xs.start, ys.start, xs.stop, ys.stop))
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def evaluate(scores, samples, ids, threshold, *, ranking=False):
    rng = np.random.default_rng(43)
    rows = []
    for probability, (_, truth), annotation_id in zip(scores, samples, ids, strict=True):
        iou = box_iou(candidate_boxes(probability, threshold), truth)
        matched, _ = match(iou)
        margins = [box_mean(probability, target) - box_mean(probability, control)
                   for target, control in nearby_controls(truth, probability.shape, rng)] if ranking else []
        rows.append({"annotation_id": annotation_id, **counts(iou, len(matched)),
                     "ranking_pairs": len(margins), "ranking_wins": sum(m > 0 for m in margins),
                     "ranking_ties": sum(m == 0 for m in margins)})
    return rows


def aggregate(rows):
    tally = {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn")}
    pairs, wins, ties = (sum(row[key] for row in rows)
                         for key in ("ranking_pairs", "ranking_wins", "ranking_ties"))
    return metrics(**tally) | {
        "frames": len(rows), "false_positives_per_frame": tally["fp"] / len(rows),
        "ranking_pairs": pairs, "ranking_wins": wins, "ranking_ties": ties,
        "ranking_win_rate": (wins + 0.5 * ties) / pairs if pairs else None,
    }


def fingerprint(path):
    return sha256(path.read_bytes()).hexdigest()


def train_and_freeze(out, ffmpeg, torch, smp, center, spread):
    training = out / "training.npz"
    validation, alignment = load_samples(ffmpeg, SOURCE / "factorial-labels", VALIDATION_IDS)
    saved, validation_grid = [], []
    with np.load(training, allow_pickle=False) as arrays:
        for seed in SEEDS:
            for group in GROUPS:
                x, y = arrays[f"x_{group}"], arrays[f"y_{group}"]
                model = train(torch, smp, UNET, (x - center) / spread, y, seed)
                probabilities = probability_maps(torch, model, validation, center, spread)
                grid = [dict(group=group, seed=seed, threshold=t, **aggregate(
                    evaluate(probabilities, validation, VALIDATION_IDS, t))) for t in THRESHOLDS]
                validation_grid.extend(grid)
                chose = max(grid, key=lambda r: (r["recall"] or 0.0, r["precision"] or 0.0, r["threshold"]))
                path = out / f"unet-{group}-{seed}.pt"
                torch.save(model.state_dict(), path)
                saved.append({"group": group, "seed": seed, "threshold": chose["threshold"],
                              "checkpoint": path.name, "checkpoint_sha256": fingerprint(path)})
                print(f"Frozen {group}/{seed}: threshold {chose['threshold']:.1f}, "
                      f"validation {chose['tp']} TP / {chose['fp']} FP.", flush=True)
    save_csv(out / "validation.csv", validation_grid)
    frozen = {
        "training_sha256": fingerprint(training), "training_seeds": SEEDS, "epochs": EPOCHS,
        "validation_annotation_ids": list(VALIDATION_IDS), "test_annotation_blocks": TEST_BLOCKS,
        "max_predictions_per_frame": MAX_BOXES, "min_component_area": MIN_AREA, "match_iou": 0.5,
        "normalization_mean": center.flatten().tolist(), "normalization_std": spread.flatten().tolist(),
        "validation_alignment": alignment, "models": saved,
        "threshold_grid": THRESHOLDS,
    }
    save_json(out / "frozen.json", frozen)
    return frozen


def score_blind(out, ffmpeg, torch, smp, frozen, center, spread):
    ids = tuple(i for block in TEST_BLOCKS.values() for i in block)
    labels = out / "labels"
    block_for_id = {i: group for group, block in TEST_BLOCKS.items() for i in block}
    try:
        samples, audit = load_samples(ffmpeg, labels, ids)
    except (FileNotFoundError, ValueError, OSError, subprocess.CalledProcessError) as error:
        save_json(out / "blind-status.json", {"status": "pending", "reason": str(error)})
        print(f"Blind scoring pending: {error}", flush=True)
        return
    save_json(out / "blind-alignment.json", audit)

    per_frame, results = [], []
    for settings in frozen["models"]:
        seed, group, threshold = (settings[field] for field in ("seed", "group", "threshold"))
        path = out / settings["checkpoint"]
        if fingerprint(path) != settings["checkpoint_sha256"]:
            raise ValueError(f"{path.name} differs from its frozen checksum")
        model = create_model(torch, smp, True, UNET).to(memory_format=torch.channels_last)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        predictions = probability_maps(torch, model.eval(), samples, center, spread)
        frames = evaluate(predictions, samples, ids, threshold, ranking=True)
        for row in frames:
            per_frame.append(dict(group=group, seed=seed, block=block_for_id[row["annotation_id"]], **row))
        for block in ("all", *TEST_BLOCKS):
            rows = [r for r in frames if block == "all" or block_for_id[r["annotation_id"]] == block]
            results.append(dict(group=group, seed=seed, block=block, threshold=threshold, **aggregate(rows)))
        print(f"Blind {group}/{seed}: {results[-4]['tp']} TP / {results[-4]['fp']} FP.", flush=True)

    save_csv(out / "blind-per-frame.csv", per_frame)
    save_csv(out / "blind-results.csv", results)
    by_key = {(r["group"], r["seed"], r["block"]): r for r in results}
    comparisons = []
    effects = {"B-A": {"A": -1, "B": 1}, "C-A": {"A": -1, "C": 1},
               "D-A": {"A": -1, "D": 1}, "D-B-C+A": {"A": 1, "B": -1, "C": -1, "D": 1}}
    for block in ("all", *TEST_BLOCKS):
        for seed in SEEDS:
            for name, weights in effects.items():
                for metric in ("recall", "precision", "f1", "false_positives_per_frame", "ranking_win_rate"):
                    values = [by_key[group, seed, block][metric] for group in weights]
                    delta = sum(weight * value for weight, value in zip(weights.values(), values, strict=True)
                                ) if all(value is not None for value in values) else None
                    comparisons.append(dict(block=block, seed=seed, effect=name, metric=metric, difference=delta))
    save_csv(out / "paired-effects.csv", comparisons)

    def gain(block, seed, metric):
        return next(r["difference"] for r in comparisons
                    if (r["block"], r["seed"], r["effect"], r["metric"]) == (block, seed, "D-A", metric))

    seed_recalls = [gain("all", seed, "recall") for seed in SEEDS]
    seed_precisions = [gain("all", seed, "precision") for seed in SEEDS]
    block_recalls = {block: mean(gain(block, seed, "recall") for seed in SEEDS) for block in TEST_BLOCKS}
    adopted = (all(delta > 0 for delta in seed_recalls) and mean(seed_recalls) >= 0.05
               and sum(delta > 0 for delta in block_recalls.values()) >= 2
               and all(delta is not None for delta in seed_precisions) and mean(seed_precisions) >= 0)
    save_json(out / "blind-status.json", {
        "status": "complete", "test_frames": len(samples),
        "test_boxes": sum(len(boxes) for _, boxes in samples),
        "joint_minus_baseline_recall_by_seed": dict(zip(SEEDS, seed_recalls, strict=True)),
        "joint_minus_baseline_precision_by_seed": dict(zip(SEEDS, seed_precisions, strict=True)),
        "joint_minus_baseline_mean_recall_by_block": block_recalls,
        "joint_configuration_retained": adopted,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    protocol = read_json(out / "simulation-protocol.json")
    if fingerprint(out / "training.npz") != protocol["training_npz_sha256"]:
        raise ValueError("Training windows differ from their checksum")
    shared = read_json(SOURCE / "factorial-protocol.json")
    center, spread = (np.asarray(shared[k], dtype=np.float32)[None, :, None, None]
                      for k in ("input_mean", "input_std"))
    torch, smp, _ = runtime()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.score_only:
        frozen = read_json(out / "frozen.json")
        if (frozen["training_sha256"] != protocol["training_npz_sha256"]
                or frozen["normalization_mean"] != center.flatten().tolist()
                or frozen["normalization_std"] != spread.flatten().tolist()):
            raise ValueError("Frozen configuration differs from the current inputs")
    else:
        frozen = train_and_freeze(out, args.ffmpeg, torch, smp, center, spread)
    score_blind(out, args.ffmpeg, torch, smp, frozen, center, spread)


if __name__ == "__main__":
    main()
