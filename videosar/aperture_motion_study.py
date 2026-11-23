from functools import partial

import numpy as np

from .analysis import score_masks
from .aperture_study import BANDWIDTH_MHZ, SEED, SNR_DB, aperture_config
from .data import load_masks, read_json, write_json
from .echo_attribution import (BOOTSTRAP_DRAWS, GROUPS, METHODS, POWER_FLOOR, SHAPE, SNR_TOLERANCE,
                               build_dataset, condition_metrics, evaluate, factorial_metrics, families,
                               file_hash, frozen_settings, save_variant)
from .proposal_data import scene_digest
from .simulation import paired_aperture_video
from .stress import MATCH
from .unet import load_weights, runtime
from .visuals import write_csv


BOOTSTRAP_SEED = 101633
PULSES = (61, 91)
MODES = ("moving", "held")
CONDITIONS = tuple((pulses, mode) for pulses in PULSES for mode in MODES)
KEYS = ("pulses", "motion")
CONTRASTS = {
    "moving_91_minus_61": {(91, "moving"): 1, (61, "moving"): -1},
    "held_91_minus_61": {(91, "held"): 1, (61, "held"): -1},
    "held_minus_moving_61": {(61, "held"): 1, (61, "moving"): -1},
    "held_minus_moving_91": {(91, "held"): 1, (91, "moving"): -1},
    "interaction": {(91, "held"): 1, (61, "held"): -1, (91, "moving"): -1, (61, "moving"): 1},
}
ARRAYS = ("frames.npy", "masks.npy", "geometric_masks.npy", "unoccluded_power.npy")


def save_family(dataset, family, reference_entries):
    entries = []
    geometry_anchor = None
    anchors = {}
    for mode in MODES:
        variants = paired_aperture_video(aperture_config(family), PULSES, SNR_DB, family["static_dark"],
                                         hold_target_within_frame=mode == "held")
        for config, output in variants:
            count = config["pulses_per_frame"]
            prior_id = f"{family['scene_family']}-{count}p"
            scene_id = f"{prior_id}-{mode}"
            row = dict(scene_id=scene_id, scene_family=family["scene_family"], group=family["group"],
                       bandwidth_mhz=BANDWIDTH_MHZ, snr_db=SNR_DB, pulses=count, motion=mode,
                       aperture_span_s=(count - 1) * config["pulse_interval_s"],
                       has_static_dark=family["static_dark"] is not None, path=scene_id)
            entry, observed, visible, geometry, reference = save_variant(dataset, family, config, row, output)
            path, _ = entry
            if geometry_anchor is None:
                geometry_anchor = geometry
            if not np.array_equal(geometry, geometry_anchor):
                raise ValueError(f"{scene_id}: center-time geometry differs across conditions")
            info = output[2]
            noise = (info["receiver_noise_power"], info["receiver_noise_scale"])
            if mode == "moving":
                anchors[count] = (observed, visible, reference, noise)
                if prior_id in reference_entries:
                    prior_path = reference_entries[prior_id]
                    if any((path / name).read_bytes() != (prior_path / name).read_bytes() for name in ARRAYS):
                        raise ValueError(f"{scene_id}: saved arrays differ from the aperture-study reference")
            else:
                prior_frames, prior_masks, prior_ref, prior_noise = anchors[count]
                if noise != prior_noise or not np.array_equal(reference, prior_ref):
                    raise ValueError(f"{scene_id}: noise or unblocked reference changed")
                if family["group"] == "empty" and not (
                        np.array_equal(observed, prior_frames) and np.array_equal(visible, prior_masks)):
                    raise ValueError(f"{scene_id}: no-target control changed")
            entries.append(entry)
    return entries


def verify_zero_speed(spec):
    config = aperture_config(spec) | {"frames": 3, "velocity_x_mps": 0.0, "velocity_y_mps": 0.0}
    moving = paired_aperture_video(config, PULSES, SNR_DB)
    held = paired_aperture_video(config, PULSES, SNR_DB, hold_target_within_frame=True)
    for (_, a), (_, b) in zip(moving, held):
        if not a[1].any():
            raise ValueError("zero-speed control contains no visible target")
        if any(not np.array_equal(a[i], b[i]) for i in (0, 1, 3, 4)) or a[2] != b[2]:
            raise ValueError("zero-speed control differs between motion modes")


def verify_reference(reference_run, specs, hashes):
    if reference_run is None:
        return {}
    protocol = read_json(reference_run / "protocol.json")
    if protocol["scenes"] != specs or any(protocol[key] != value for key, value in hashes.items()):
        raise ValueError("prior aperture study used different scenes or frozen models")
    dataset = reference_run / "dataset"
    manifest = read_json(dataset / "dataset.json")
    entries = {}
    for row in manifest["entries"]:
        if row["pulses"] not in PULSES:
            continue
        path = dataset / row["path"]
        if scene_digest(path) != row["sha256"] or file_hash(path / "unoccluded_power.npy") != row["reference_sha256"]:
            raise ValueError(f"{row['scene_id']}: prior data failed hash verification")
        entries[row["scene_id"]] = path
    expected = {f"{spec['scene_family']}-{count}p" for spec in specs for count in PULSES}
    if entries.keys() != expected:
        raise ValueError("prior aperture study is missing required moving sequences")
    return entries


def score_fixed_reference(root, entries, rows):
    frames = {(row["scene_id"], row["frame"]): row for row in rows}
    families_data = {}
    for path, row in entries:
        row_key = (row["pulses"], row["motion"])
        families_data.setdefault(row["scene_family"], {})[row_key] = (path, row)
    for versions in families_data.values():
        labels = {key: load_masks(path / "masks.npy", SHAPE) for key, (path, _) in versions.items()}
        common = np.logical_and.reduce([label.any(axis=(1, 2)) for label in labels.values()])
        reference = labels[61, "moving"]
        for _, entry in versions.values():
            scene_id = entry["scene_id"]
            directory = root / "predictions" / scene_id
            for i, flag in enumerate(common):
                frames[scene_id, i]["reference_common_visible"] = bool(flag)
            for method in METHODS:
                prediction = load_masks(directory / f"{method}.npy", SHAPE)
                evaluated = score_masks(prediction[common], reference[common], MATCH)["frames"]
                for i, item in zip(np.flatnonzero(common), evaluated):
                    best = item["best_component_iou"]
                    frames[scene_id, i].update({f"{method}_reference_best_iou": best,
                                               f"{method}_reference_matched": int(best >= MATCH)})

    summaries = []
    for group in ("small", "normal"):
        for pulses, mode in CONDITIONS:
            subset = [row for row in rows if (row["group"], row["pulses"], row["motion"]) == (group, pulses, mode)
                      and row["reference_common_visible"]]
            for method in METHODS:
                matched = sum(row[f"{method}_reference_matched"] for row in subset)
                summaries.append(dict(group=group, pulses=pulses, motion=mode, method=method, matched=matched,
                                      common_frames=len(subset), recall=matched / len(subset) if subset else None))
    return summaries


def run_aperture_motion_study(root, reference_run=None):
    torch, smp, correlate = runtime()
    torch.set_num_threads(min(6, torch.get_num_threads()))
    settings, checkpoints, operating, hashes = frozen_settings()
    specs = families(SEED, "aperture")
    references = verify_reference(reference_run, specs, hashes)
    protocol = dict(seed=SEED, groups=GROUPS, scenes=specs, bandwidth_mhz=BANDWIDTH_MHZ, snr_db=SNR_DB,
                    conditions=[dict(pulses=n, motion=mode) for n, mode in CONDITIONS],
                    matches_prior_aperture_study=bool(references), match_iou=MATCH, operating_points=operating, **hashes,
                    torch_version=torch.__version__, smp_version=smp.__version__, numpy_version=np.__version__,
                    controls="shared frame-center trajectory, platform, ground, range grid, unit noise and observation offsets",
                    held_motion="target center is held at the frame time for every pulse; platform still moves within frame",
                    precision_recall_target=0.8, snr_tolerance_db=SNR_TOLERANCE,
                    local_contrast="10 log10(median ring power / median geometry power) after registration; ring 2 to 5 pixels",
                    local_power_floor=POWER_FLOOR, bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=BOOTSTRAP_DRAWS,
                    recall_reference="61-pulse moving label on frames visible in all four conditions",
                    primary_interaction="(91 held - 61 held) - (91 moving - 61 moving)",
                    primary_metrics=["small target local contrast", "small target U-Net fixed-reference recall"],
                    evidence_rule="both small target primary interaction intervals must be above zero")
    write_json(root / "protocol.json", protocol)
    verify_zero_speed(specs[0])
    entries = build_dataset(root, specs, save=partial(save_family, reference_entries=references),
                            conditions=CONDITIONS, condition_keys=KEYS, profile="aperture-motion-study", seed=SEED)
    validation = read_json(root / "dataset/validation.json")
    validation.update(prior_moving_sequences_reproduced=len(references), zero_speed_apertures_verified=len(PULSES),
                      no_target_mode_pairs_verified=GROUPS["empty"] * len(PULSES),
                      noise_and_reference_mode_pairs_verified=len(specs) * len(PULSES))
    write_json(root / "dataset/validation.json", validation)
    print("Motion pairs and controls verified. Evaluating frozen methods.", flush=True)
    networks = {name: (load_weights(torch, smp, checkpoints[name], config, "cpu"), config)
                for name, config in settings.items()}
    rows = evaluate(root, entries, torch, correlate, networks, operating, KEYS)
    fixed = score_fixed_reference(root, entries, rows)
    imaging, detection = condition_metrics(rows, CONDITIONS, KEYS)
    paired, estimates = factorial_metrics(rows, CONTRASTS, KEYS, BOOTSTRAP_SEED, fixed_reference=True)
    primary_metrics = {"local_contrast_db", "unet_common_visible_recall"}
    primary = {row["metric"]: row for row in estimates if row["group"] == "small"
               and row["contrast"] == "interaction" and row["metric"] in primary_metrics}
    supports_hypothesis = primary.keys() == primary_metrics and all(row["ci95_low"] > 0 for row in primary.values())
    for filename, records in (("per_frame", rows), ("imaging_by_condition", imaging),
                              ("detection_by_condition", detection), ("fixed_label_recall", fixed),
                              ("paired_by_family", paired), ("paired_summary", estimates)):
        write_csv(root / f"{filename}.csv", records)
    write_json(root / "results.json", dict(imaging=imaging, detection=detection, fixed_label_recall=fixed,
                                          paired=estimates, primary=primary, supports_hypothesis=supports_hypothesis))
