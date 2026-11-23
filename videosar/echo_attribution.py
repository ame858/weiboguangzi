from collections import Counter
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import distance_transform_edt

from .analysis import baseline_scores, score_masks, target_summary
from .data import load_masks, load_observations, read_json, save_sequence, write_json
from .proposal_data import scene_digest, translate
from .simulation import calibrated_receiver_video, point_check
from .stress import MATCH, PROPOSAL, scene_specs
from .study import cleanup, read_registered
from .unet import load_weights, predict, runtime
from .visuals import write_csv


SEED = 101621
SHAPE = (20, 64, 80)
BANDWIDTHS = (300, 600)
SNRS = (25, 10)
CONDITIONS = tuple((bandwidth, snr) for bandwidth in BANDWIDTHS for snr in SNRS)
CONDITION_KEYS = ("bandwidth_mhz", "snr_db")
GROUPS = {"small": 12, "normal": 6, "empty": 6}
METHODS = ("unet", "fpn", "difference")
BOOTSTRAP_SEED = 101622
BOOTSTRAP_DRAWS = 2000
SNR_TOLERANCE = 0.1
POWER_FLOOR = 1e-12
FIXED = {"carrier_hz": 35e9, "noise": 0.0, "thermal_noise": 0.0, "frames": SHAPE[0],
         "height": SHAPE[1], "width": SHAPE[2], "pulses_per_frame": 61, "pulse_interval_s": 0.005,
         "frame_interval_s": 0.2, "pixel_size_m": 0.25}
CONTRASTS = {
    "bandwidth_at_25dB": {(600, 25): 1, (300, 25): -1},
    "bandwidth_at_10dB": {(600, 10): 1, (300, 10): -1},
    "noise_at_300MHz": {(300, 10): 1, (300, 25): -1},
    "noise_at_600MHz": {(600, 10): 1, (600, 25): -1},
    "interaction": {(600, 10): 1, (300, 10): -1, (600, 25): -1, (300, 25): 1},
}


def families(seed=SEED, prefix="echo"):
    specs = {}
    for spec in scene_specs(seed, paired_families=18, empty_families=6):
        specs.setdefault(spec["scene_family"], {})[spec["condition"]] = spec
    result = []
    for i, variants in enumerate(specs.values()):
        group = "small" if i < 12 else "normal" if i < 18 else "empty"
        variant = {"small": "small", "normal": "reference", "empty": "empty_static_dark"}[group]
        selected = variants[variant]
        patch = variants["static_dark"]["static_dark"] if i < 12 and i % 2 else selected["static_dark"]
        # Each bandwidth is carried in CONDITIONS and set at generation time.
        common = {key: value for key, value in selected["config"].items() if key != "bandwidth_hz"}
        result.append({"scene_family": f"{prefix}-{i:02d}", "group": group, "static_dark": patch,
                       "offsets_xy_px": selected["offsets_xy_px"], "config": common | FIXED})
    return result


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def save_variant(dataset, family, config, row, output):
    frames, visible, info, geometry, reference = output
    offsets = np.asarray(family["offsets_xy_px"])
    if offsets.any():
        frames = np.stack([translate(frame, dx, dy) for frame, (dx, dy) in zip(frames, offsets)])
    scene_id = row["scene_id"]
    info.update(row, static_dark=family["static_dark"], observed_offsets_xy_px=family["offsets_xy_px"],
                label_coordinates="fixed_ground_grid", visibility_definition=(
                    "blocked for >=50% of pulses and local noise-free power drops by configured fraction"),
                noise_model="calibrated complex Gaussian receiver noise; no image-domain perturbation")
    path = dataset / scene_id
    save_sequence(path, frames, visible, info, geometry)
    np.save(path / "unoccluded_power.npy", reference.astype(np.float32))
    observed, _, _ = load_observations(path)
    visible = load_masks(path / "masks.npy", SHAPE)
    geometry = load_masks(path / "geometric_masks.npy", SHAPE)
    reference = np.load(path / "unoccluded_power.npy", allow_pickle=False)
    meta = read_json(path / "metadata.json")
    if (observed.shape != SHAPE or reference.shape != SHAPE or not np.isfinite(reference).all()
            or np.any(reference < 0) or meta["effective_config"] != config
            or meta["observed_offsets_xy_px"] != family["offsets_xy_px"]
            or meta["static_dark"] != family["static_dark"]):
        raise ValueError(f"{scene_id}: invalid saved arrays or scene parameters")
    measured = 10 * np.log10(reference.mean(dtype=np.float64) / meta["receiver_noise_power"])
    if (not np.isfinite(measured) or abs(measured - row["snr_db"]) > SNR_TOLERANCE
            or abs(measured - meta["snr_measured_db"]) > SNR_TOLERANCE):
        raise ValueError(f"{scene_id}: measured SNR outside tolerance")
    if family["group"] == "empty" and (visible.any() or geometry.any()):
        raise ValueError(f"{scene_id}: empty control has target labels")
    row.update(sha256=scene_digest(path), reference_sha256=file_hash(path / "unoccluded_power.npy"),
               snr_measured_db=float(measured), frames=len(observed),
               geometry_frames=int(geometry.any(axis=(1, 2)).sum()),
               visible_frames=int(visible.any(axis=(1, 2)).sum()))
    return (path, row), observed, visible, geometry, reference


def save_family(dataset, family):
    rows = []
    geometry_anchor = None
    for bandwidth in BANDWIDTHS:
        config = family["config"] | {"bandwidth_hz": bandwidth * 1e6}
        outputs = calibrated_receiver_video(config, SNRS, family["static_dark"])
        anchor = None
        for snr in SNRS:
            scene_id = f"{family['scene_family']}-{bandwidth}MHz-{snr}dB"
            row = {"scene_id": scene_id, "scene_family": family["scene_family"], "group": family["group"],
                   "bandwidth_mhz": bandwidth, "snr_db": snr, "path": scene_id,
                   "has_static_dark": family["static_dark"] is not None}
            entry, observed, visible, geometry, reference = save_variant(dataset, family, config, row, outputs[snr])
            if geometry_anchor is None:
                geometry_anchor = geometry
            if not np.array_equal(geometry_anchor, geometry):
                raise ValueError(f"{scene_id}: paired geometry differs")
            if anchor is None:
                anchor = (observed, visible, reference)
            elif (np.array_equal(anchor[0], observed) or not np.array_equal(anchor[1], visible)
                  or not np.array_equal(anchor[2], reference)):
                raise ValueError(f"{scene_id}: invalid paired observations or clean labels")
            rows.append(entry)
    return rows


def build_dataset(root, specs, *, save=save_family, conditions=CONDITIONS, condition_keys=CONDITION_KEYS,
                  profile="echo-attribution", seed=SEED):
    expected = {(group, *condition): count for group, count in GROUPS.items() for condition in conditions}
    dataset = root / "dataset"
    entries = []
    for i, family in enumerate(specs, 1):
        entries.extend(save(dataset, family))
        if i % 4 == 0:
            print(f"Generated and checked {i}/{len(specs)} paired families", flush=True)
    rows = [row for _, row in entries]
    if Counter((row["group"], *(row[key] for key in condition_keys)) for row in rows) != expected:
        raise ValueError("dataset counts differ from the frozen protocol")
    with TemporaryDirectory(dir=root) as repeat:
        rebuilt = save(Path(repeat), specs[1])
        if [row for _, row in rebuilt] != rows[len(conditions):2 * len(conditions)]:
            raise ValueError("paired family failed byte-for-byte replay")
    write_json(dataset / "dataset.json", {"profile": profile, "seed": seed, "entries": rows})
    write_csv(dataset / "manifest.csv", rows)
    write_json(dataset / "validation.json", {"families": len(specs), "sequences": len(rows),
               "frames": sum(row["frames"] for row in rows), "replayed_family": specs[1]["scene_family"],
               "max_snr_error_db": max(abs(row["snr_db"] - row["snr_measured_db"]) for row in rows)})
    return entries


def local_contrast(frame, geometry):
    if not geometry.any():
        return None
    distance = distance_transform_edt(~geometry)
    ring = (distance >= 2) & (distance <= 5)
    if not ring.any():
        return None
    inside = max(float(np.median(frame[geometry])), POWER_FLOOR)
    outside = max(float(np.median(frame[ring])), POWER_FLOOR)
    return float(10 * np.log10(outside / inside))


def evaluate(root, entries, torch, correlate, networks, operating, condition_keys=CONDITION_KEYS):
    rows = []
    for i, entry_pair in enumerate(entries, 1):
        sample, = read_registered([entry_pair], correlate)
        path, entry = entry_pair
        aligned, truth = sample["aligned"], sample["truth"]
        geometry = load_masks(path / "geometric_masks.npy", SHAPE)
        base = {key: entry[key] for key in ("scene_id", "scene_family", "group", "has_static_dark", *condition_keys)}
        frames = [dict(base, frame=j, geometry_pixels=int(mask.sum()), visible_pixels=int(label.sum()),
                       local_contrast_db=local_contrast(image, mask))
                  for j, (image, mask, label) in enumerate(zip(aligned, geometry, truth))]
        target_dir = root / "predictions" / entry["scene_id"]
        target_dir.mkdir(parents=True)
        np.save(target_dir / "aligned_frames.npy", aligned)
        for name in METHODS:
            if name == "difference":
                scores = baseline_scores(aligned, name)
            else:
                model, config = networks[name]
                scores = predict(torch, model, aligned, config, "cpu", 32)
            if scores.shape != SHAPE or not np.isfinite(scores).all():
                raise ValueError(f"{entry['scene_id']} {name}: invalid detection scores")
            prediction = cleanup(scores, **operating[name])
            prediction_file = target_dir / f"{name}.npy"
            np.save(prediction_file, prediction.astype(np.uint8))
            final = score_masks(load_masks(prediction_file, SHAPE), truth, MATCH)
            raw = score_masks(scores >= operating[name]["threshold"], truth, MATCH)
            for row, checked, thresholded in zip(frames, final["frames"], raw["frames"]):
                matched = checked["target_visible"] and checked["best_component_iou"] >= MATCH
                row.update({f"{name}_matched": int(matched),
                            f"{name}_false_positives": checked["predicted_components"] - int(matched),
                            f"{name}_best_iou": checked["best_component_iou"],
                            f"{name}_threshold_best_iou": thresholded["best_component_iou"]})
        rows.extend(frames)
        if i % 16 == 0:
            print(f"Evaluated {i}/{len(entries)} sequences with three frozen methods", flush=True)
    return rows


def mean(values):
    return float(np.mean(values)) if len(values) else None


def detection_metrics(rows, method):
    matched = sum(row[f"{method}_matched"] for row in rows)
    fp = sum(row[f"{method}_false_positives"] for row in rows)
    visible = sum(row["visible_pixels"] > 0 for row in rows)
    empty = [row for row in rows if row["group"] == "empty"]
    empty_fp = sum(row[f"{method}_false_positives"] for row in empty)
    return dict(method=method, matched=matched, false_positives=fp, missed=visible - matched,
                empty_frames=len(empty), empty_false_positives=empty_fp,
                empty_fp_per_frame=empty_fp / len(empty) if empty else None,
                **target_summary(matched, fp, visible - matched))


def condition_metrics(rows, conditions=CONDITIONS, condition_keys=CONDITION_KEYS):
    imaging, detection = [], []
    for values in conditions:
        condition = [row for row in rows if tuple(row[key] for key in condition_keys) == values]
        for group in (*GROUPS, "all"):
            subset = condition if group == "all" else [row for row in condition if row["group"] == group]
            geometry = [row for row in subset if row["geometry_pixels"]]
            visible = [row for row in subset if row["visible_pixels"]]
            both = sum(row["visible_pixels"] > 0 for row in geometry)
            contrast = [row["local_contrast_db"] for row in geometry if row["local_contrast_db"] is not None]
            key = dict(group=group, **dict(zip(condition_keys, values)))
            imaging.append(dict(key, frames=len(subset), geometry_frames=len(geometry), visible_frames=len(visible),
                                both_frames=both, visible_without_geometry_frames=len(visible) - both,
                                formation_rate=both / len(geometry) if geometry else None,
                                mean_local_contrast_db=mean(contrast), contrast_frames=len(contrast),
                                mean_visible_area_px=mean([row["visible_pixels"] for row in visible])))
            detection.extend(dict(key, **detection_metrics(subset, method)) for method in METHODS)
    return imaging, detection


def factorial_metrics(rows, contrasts=CONTRASTS, condition_keys=CONDITION_KEYS, seed=BOOTSTRAP_SEED, *,
                      fixed_reference=False):
    by_family = {}
    for row in rows:
        key = (row["group"], row["scene_family"])
        condition = tuple(row[key] for key in condition_keys)
        by_family.setdefault(key, {}).setdefault(row["frame"], {})[condition] = row
    paired = []
    for (group, family), frames in by_family.items():
        for name, weights in contrasts.items():
            changes = {"formation_rate": [], "local_contrast_db": []}
            if group == "empty":
                changes = {f"{method}_empty_fp_per_frame": [] for method in METHODS}
            else:
                changes.update({f"{method}_common_visible_recall": [] for method in METHODS})
            for versions in frames.values():
                selected = [(weight, versions[condition]) for condition, weight in weights.items()]
                if group == "empty":
                    for method in METHODS:
                        changes[f"{method}_empty_fp_per_frame"].append(
                            sum(weight * row[f"{method}_false_positives"] for weight, row in selected))
                    continue
                if all(row["geometry_pixels"] for _, row in selected):
                    changes["formation_rate"].append(
                        sum(weight * (row["visible_pixels"] > 0) for weight, row in selected))
                common = not fixed_reference or all(row["reference_common_visible"] for row in versions.values())
                if common and all(row["local_contrast_db"] is not None for _, row in selected):
                    changes["local_contrast_db"].append(
                        sum(weight * row["local_contrast_db"] for weight, row in selected))
                if common and all(row["visible_pixels"] for _, row in selected):
                    match_suffix = "reference_matched" if fixed_reference else "matched"
                    for method in METHODS:
                        changes[f"{method}_common_visible_recall"].append(
                            sum(weight * row[f"{method}_{match_suffix}"] for weight, row in selected))
            paired.extend(dict(group=group, scene_family=family, contrast=name, metric=metric,
                               frames=len(values), delta=mean(values)) for metric, values in changes.items())
    rng = np.random.default_rng(seed)
    grouped = {}
    for row in paired:
        if row["frames"]:
            key = (row["group"], row["contrast"], row["metric"])
            grouped.setdefault(key, []).append(row)
    estimates = []
    for (group, contrast, metric), subset in grouped.items():
        values = np.array([row["delta"] for row in subset])
        draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
        estimates.append(dict(group=group, contrast=contrast, metric=metric,
                              families=len(subset), frames=sum(row["frames"] for row in subset),
                              delta=mean(values), ci95_low=float(low), ci95_high=float(high)))
    return paired, estimates


def mismatch_cases(rows, num_conditions=len(CONDITIONS)):
    common = Counter((row["scene_family"], row["frame"]) for row in rows
                     if row["group"] == "small" and row["visible_pixels"])
    cases = []
    for row in rows:
        if (common[row["scene_family"], row["frame"]] == num_conditions and row["difference_matched"]
                and row["local_contrast_db"] is not None and row["local_contrast_db"] > 0
                and not (row["unet_matched"] and row["fpn_matched"])):
            cases.append(row)
    return sorted(cases, key=lambda row: (row["unet_matched"] + row["fpn_matched"],
                                         -row["local_contrast_db"]))


def frozen_settings():
    delivery = PROPOSAL.parent
    settings_paths = {"unet": delivery / "small-shadow/models/small_shadow.json",
                      "fpn": delivery / "open-compare/models/fpn.json"}
    baseline_path = PROPOSAL / "frozen_validation.json"
    settings = {name: read_json(path) for name, path in settings_paths.items()}
    checkpoints = {name: path.with_name(settings[name].get("checkpoint", path.with_suffix(".pt").name))
                   for name, path in settings_paths.items()}
    operating = {name: {key: config[key] for key in ("threshold", "min_area", "closing")}
                 for name, config in settings.items()}
    operating["difference"] = read_json(baseline_path)["methods"]["difference"]["settings"]
    hashes = {
        "settings_sha256": {name: file_hash(path) for name, path in
                            (settings_paths | {"difference": baseline_path}).items()},
        "weights_sha256": {name: file_hash(path) for name, path in checkpoints.items()},
    }
    return settings, checkpoints, operating, hashes


def run_attribution(root):
    torch, smp, correlate = runtime()
    torch.set_num_threads(min(6, torch.get_num_threads()))
    settings, checkpoints, operating, hashes = frozen_settings()
    specs = families()
    protocol = dict(seed=SEED, groups=GROUPS, conditions=[dict(bandwidth_mhz=bw, snr_db=snr)
                    for bw, snr in CONDITIONS], scenes=specs, match_iou=MATCH, operating_points=operating,
                    precision_recall_target=0.8, snr_tolerance_db=SNR_TOLERANCE,
                    torch_version=torch.__version__, smp_version=smp.__version__, numpy_version=np.__version__,
                    **hashes,
                    snr_definition="10 log10(mean clean unblocked power / mean focused receiver-noise power); scaled before coherent addition",
                    local_contrast="10 log10(median ring power / median geometry power) after registration; Euclidean distance 2 to 5 pixels inclusive",
                    local_power_floor=POWER_FLOOR, bootstrap_seed=BOOTSTRAP_SEED,
                    bootstrap_draws=BOOTSTRAP_DRAWS, pairing="same frame index; recall uses intersection of visible labels",
                    paired_estimator="difference per frame, average within family, equal-family mean and percentile bootstrap",
                    mismatch_rule="small targets visible in all four conditions; local geometry contrast >0 dB; background difference matches; at least one network misses")
    write_json(root / "protocol.json", protocol)
    points = {str(bw): point_check(specs[0]["config"] | {"bandwidth_hz": bw * 1e6})[0] for bw in BANDWIDTHS}
    widths = [point["range_mainlobe_half_power_width_m"] for point in points.values()]
    if (any(point["point_error_px"] > 1 or point["range_peak_error_bins"] > 1 for point in points.values())
            or not 1.9 <= widths[0] / widths[1] <= 2.1):
        raise ValueError("point checks failed for paired bandwidths")
    write_json(root / "point_checks.json", points)
    entries = build_dataset(root, specs)
    print("All paired scenes verified. Evaluating frozen methods.", flush=True)
    networks = {name: (load_weights(torch, smp, checkpoints[name], config, "cpu"), config)
                for name, config in settings.items()}
    rows = evaluate(root, entries, torch, correlate, networks, operating)
    imaging, detection = condition_metrics(rows)
    paired, estimates = factorial_metrics(rows)
    cases = mismatch_cases(rows)
    totals = [detection_metrics(rows, method) for method in METHODS]
    for filename, records in (("per_frame", rows), ("imaging_by_condition", imaging),
                              ("detection_by_condition", detection), ("paired_by_family", paired),
                              ("paired_summary", estimates), ("mismatch_cases", cases), ("metrics", totals)):
        write_csv(root / f"{filename}.csv", records)
    write_json(root / "results.json", dict(imaging=imaging, detection=detection, paired=estimates,
                                          totals=totals, mismatch_cases=len(cases),
                                          mismatch_examples=cases[:5]))
