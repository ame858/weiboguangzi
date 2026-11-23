from hashlib import sha256
from time import perf_counter

import numpy as np

from .analysis import baseline_scores, score_masks, score_track, target_summary, track
from .data import read_json, save_receipt, write_json
from .experiment import get_sample, read_split
from .proposal_data import scene_digest
from .unet import align, fit, predict, runtime
from .visuals import save_preview, write_csv


def cleanup(scores, threshold, min_area, closing):
    from scipy import ndimage

    masks = scores >= threshold
    if closing:
        structure = ndimage.generate_binary_structure(2, 1)
        masks = ndimage.binary_closing(masks, structure=structure[None])
    if min_area > 1:
        for frame in masks:
            regions, _ = ndimage.label(frame)
            counts = np.bincount(regions.ravel())
            counts[0] = 0
            frame[:] = counts[regions] >= min_area
    return masks


def calibrate(scored, thresholds, areas, closing, match_iou):
    from scipy import ndimage

    rows = []
    for threshold in thresholds:
        counts = {area: np.zeros(3, dtype=int) for area in areas}
        for scores, truth in scored:
            masks = cleanup(scores, threshold, 1, closing)
            for frame, label in zip(masks, truth):
                region_map, total = ndimage.label(frame)
                sizes = np.bincount(region_map.ravel(), minlength=total + 1)[1:]
                target_area = np.count_nonzero(label)
                hits = np.bincount(region_map[label], minlength=total + 1)[1:]
                ious = hits / (sizes + target_area - hits)
                for area in areas:
                    eligible = sizes >= area
                    matched = int(np.any(eligible & (ious >= match_iou)))
                    counts[area] += [matched, int(eligible.sum()) - matched, int(target_area > 0) - matched]
        for area, (matched, fp, missed) in counts.items():
            rows.append({"threshold": float(threshold), "min_area": area, "closing": closing,
                         "false_positive": int(fp), **target_summary(int(matched), int(fp), int(missed))})
    best = max(rows, key=lambda row: (row["target_f1"] or 0, -row["false_positive"]))
    return {key: best[key] for key in ("threshold", "min_area", "closing")}, rows


def read_registered(entries, correlate):
    result = []
    for path, entry in entries:
        if scene_digest(path) != entry["sha256"]:
            raise ValueError(f"dataset contents differ from the frozen manifest for {entry['scene_id']}")
        frames, truth, times, scale, meta = get_sample(path)
        start = perf_counter()
        aligned, correction = align(frames, correlate)
        elapsed = perf_counter() - start
        offset = np.asarray(meta["observed_offsets_xy_px"])
        result.append({"path": path, "entry": entry, "aligned": aligned, "truth": truth, "times": times, "scale": scale,
                       "metadata": meta, "registration_s": elapsed, "registration_rss": float(np.square(offset + correction).sum()),
                       "uncorrected_rss": float(np.square(offset).sum())})
    return result


def aggregate_detection(rows):
    numbers = {key: sum(row[key] for row in rows) for key in (
        "frames", "matched", "false_positive", "missed", "truth_pixels", "predicted_pixels",
        "intersection_pixels", "union_pixels", "elapsed_s")}
    matched, fp, missed = (numbers[key] for key in ("matched", "false_positive", "missed"))
    support = numbers["truth_pixels"] + numbers["predicted_pixels"]
    return {"sequences": len(rows), **numbers, **target_summary(matched, fp, missed),
            "pixel_f1": 2 * numbers["intersection_pixels"] / support if support else None,
            "region_iou": numbers["intersection_pixels"] / numbers["union_pixels"] if numbers["union_pixels"] else None,
            "false_positive_per_frame": fp / numbers["frames"], "milliseconds_per_frame": 1000 * numbers["elapsed_s"] / numbers["frames"]}


def study(root, dataset_root, config):
    import os

    torch, smp, correlate = runtime()
    manifest_path = dataset_root / "dataset.json"
    dataset_hash = sha256(manifest_path.read_bytes()).hexdigest()
    manifest = read_json(manifest_path)
    if manifest.get("profile") != "proposal":
        raise ValueError("study expects dataset --profile proposal")
    splits = read_split(dataset_root)
    train = read_registered(splits["train"], correlate)
    validation = read_registered(splits["validation"], correlate)
    val_pairs = [(sample["aligned"], sample["truth"]) for sample in validation]
    train_pairs = [(sample["aligned"], sample["truth"]) for sample in train]
    cuda = torch.cuda.is_available()
    device = ("cuda" if cuda else "cpu") if config["device"] == "auto" else config["device"]
    if device == "cuda" and not cuda:
        raise ValueError("CUDA was requested but is unavailable")
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    if device == "cuda":
        torch.backends.cudnn.deterministic = True
    models_dir = root / "models"
    models_dir.mkdir()
    selections, registry, settings_registry = {}, {}, {}
    match_iou = config["match_iou"]
    for name in ("threshold", "difference"):
        registry[name] = lambda frames, kind=name: baseline_scores(frames, kind)
        values = [(registry[name](frames), truth) for frames, truth in val_pairs]
        settings, scan = calibrate(values, (.17, .25, .35, .45, .55, .65, .72, .79, .86),
                                   (4, 8, 12, 20, 32, 55), name == "threshold", match_iou)
        selections[name] = dict(settings=settings, validation=scan)

    variants = (("unet_single", False, False), ("unet_temporal", True, False), ("unet_temporal_noise", True, True))
    for name, temporal, noise in variants:
        print(f"Training {name} on {device}", flush=True)
        model, settings = fit(torch, smp, train_pairs, val_pairs, temporal=temporal, augment_noise=noise,
                              epochs=config["epochs"], seed=config["seed"], device=device, batch_size=config["batch_size"])
        weights = models_dir / f"{name}.pt"
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, weights)
        weight_digest = sha256(weights.read_bytes()).hexdigest()
        registry[name] = lambda frames, network=model, spec=settings: predict(
            torch, network, frames, spec, device, config["batch_size"])
        values = [(registry[name](frames), truth) for frames, truth in val_pairs]
        operating_point, scan = calibrate(values, (.15, .25, .35, .45, .55, .65, .75, .83, .90, .95, .98),
                                          (1,), False, match_iou)
        selections[name] = dict(settings=operating_point, validation=scan, weight_sha256=weight_digest)
        settings_registry[name] = dict(settings, **operating_point, seed=config["seed"], registered=True, weight_sha256=weight_digest)
        write_json(weights.with_suffix(".json"), settings_registry[name])
    deploy = "unet_temporal_post"
    parent = "unet_temporal_noise"
    threshold = selections[parent]["settings"]["threshold"]
    operating_point, scan = calibrate(values, (threshold,), (3, 5, 8, 12, 16, 24), True, match_iou)
    selections[deploy] = dict(settings=operating_point, validation=scan, weights=f"models/{parent}.pt")
    write_json(models_dir / f"{deploy}.json", settings_registry[parent] | operating_point |
               {"checkpoint": f"{parent}.pt"})
    if sha256(manifest_path.read_bytes()).hexdigest() != dataset_hash:
        raise ValueError("the frozen dataset manifest changed during training")
    frozen = {"dataset_sha256": dataset_hash, "match_iou": match_iou,
              "platform": torch.cuda.get_device_name() if device == "cuda" else "CPU",
              "torch_version": torch.__version__, "smp_version": smp.__version__, "methods": selections}
    write_json(root / "frozen_validation.json", frozen)
    print("Validation operating points frozen. Evaluating the independent test split.", flush=True)
    test = read_registered(splits["test"], correlate)
    rows, examples = [], {}
    for sample in test:
        path, entry, aligned, truth = (sample[key] for key in ("path", "entry", "aligned", "truth"))
        scene_id = entry["scene_id"]
        scored = {}
        for name, scorer in registry.items():
            start = perf_counter()
            scores = scorer(aligned)
            scored[name] = (scores, perf_counter() - start)
        scored[deploy] = scored[parent]
        for name, (scores, seconds) in scored.items():
            operating_point = selections[name]["settings"]
            start = perf_counter()
            predicted = cleanup(scores, **operating_point)
            elapsed = seconds + sample["registration_s"] + (perf_counter() - start)
            metrics = score_masks(predicted, truth, match_iou)
            per_frame = metrics.pop("frames")
            tracked = track(predicted, sample["times"], sample["scale"], config["gate_px"], config["max_gap"])
            motion = score_track(tracked, truth, sample["scale"], sample["metadata"])
            motion.pop("true_shadow_velocity_px_s")
            rows.append(dict(entry, method=name, frames=len(truth), **metrics, **motion, elapsed_s=elapsed,
                             **target_summary(metrics["matched"], metrics["false_positive"], metrics["missed"])))
            target = root / "predictions" / scene_id / name
            target.mkdir(parents=True)
            np.save(target / "predictions.npy", predicted.astype(np.uint8))
            write_csv(target / "per_frame.csv", [dict(frame=i, **frame) for i, frame in enumerate(per_frame)])
            write_json(target / "tracks.json", tracked)
            save_receipt(target, path, detector=name, **operating_point)
            if name == deploy and truth.any():
                examples[scene_id] = (metrics["iou"], aligned, truth, predicted, tracked)
    overall = [dict(method=name, **aggregate_detection([row for row in rows if row["method"] == name])) for name in selections]
    groups = []
    for field in ("source", "background", "speed_band", "noise_band", "registration_perturbation", "condition"):
        for name in selections:
            method_rows = [r for r in rows if r["method"] == name]
            for value in dict.fromkeys(row[field] for row in method_rows):
                group = [r for r in method_rows if r[field] == value]
                groups.append(dict(method=name, field=field, value=value, **aggregate_detection(group)))
    for label, selector in (("best", max), ("failure", min)):
        scene_id = selector(examples, key=lambda key: examples[key][0])
        iou, frames, truth, predicted, tracked = examples[scene_id]
        save_preview(root / f"{label}.png", frames, truth, predicted, tracked, [0, 6, 13, 19])
        write_json(root / f"{label}.json", {"scene_id": scene_id, "region_iou": iou, "method": deploy})
    perturbed = [s for s in test if s["entry"]["registration_perturbation"]]
    frames = sum(len(s["aligned"]) for s in perturbed)
    registration = {"scenes": len(perturbed), "frames": frames,
                    "before_rmse_px": float(np.sqrt(sum(s["uncorrected_rss"] for s in perturbed) / frames)) if frames else None,
                    "after_rmse_px": float(np.sqrt(sum(s["registration_rss"] for s in perturbed) / frames)) if frames else None}
    write_csv(root / "metrics.csv", overall)
    write_csv(root / "metrics_by_scene.csv", rows)
    write_csv(root / "metrics_by_group.csv", groups)
    write_json(root / "results.json", {"summary": overall, "match_iou": match_iou, "registration": registration})
