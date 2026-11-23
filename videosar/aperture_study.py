import numpy as np

from .analysis import score_masks
from .data import load_masks, write_json
from .echo_attribution import (BOOTSTRAP_DRAWS, GROUPS, METHODS, POWER_FLOOR, SHAPE, SNR_TOLERANCE,
                               build_dataset, condition_metrics, evaluate, factorial_metrics, families,
                               frozen_settings, mismatch_cases, save_variant)
from .simulation import paired_aperture_video, point_check
from .stress import MATCH
from .unet import load_weights, runtime
from .visuals import write_csv


SEED = 101631
BOOTSTRAP_SEED = 101632
PULSES = (31, 61, 91)
BANDWIDTH_MHZ = 600
SNR_DB = 25
CONDITIONS = tuple((count,) for count in PULSES)
KEYS = ("pulses",)
CONTRASTS = {f"{end}_minus_{start}": {(end,): 1, (start,): -1}
             for start, end in ((31, 61), (61, 91), (31, 91))}


def aperture_config(family):
    return family["config"] | {"bandwidth_hz": BANDWIDTH_MHZ * 1e6}


def save_family(dataset, family):
    entries = []
    reference_geometry = None
    previous = None
    for config, output in paired_aperture_video(aperture_config(family), PULSES, SNR_DB, family["static_dark"]):
        count = config["pulses_per_frame"]
        scene_id = f"{family['scene_family']}-{count}p"
        row = dict(scene_id=scene_id, scene_family=family["scene_family"], group=family["group"],
                   bandwidth_mhz=BANDWIDTH_MHZ, snr_db=SNR_DB, pulses=count,
                   aperture_span_s=(count - 1) * config["pulse_interval_s"],
                   has_static_dark=family["static_dark"] is not None, path=scene_id)
        entry, observed, _, geometry, _ = save_variant(dataset, family, config, row, output)
        if reference_geometry is None:
            reference_geometry = geometry
        if not np.array_equal(geometry, reference_geometry):
            raise ValueError(f"{scene_id}: center-time geometry differs across apertures")
        if previous is not None and np.array_equal(observed, previous):
            raise ValueError(f"{scene_id}: aperture observations unexpectedly identical")
        previous = observed
        entries.append(entry)
    return entries


def azimuth_checks(config):
    offsets = np.linspace(-1.5, 1.5, 1501)
    checks = []
    curves = []
    for count in PULSES:
        check, _ = point_check(config | {"pulses_per_frame": count}, azimuth_offsets_m=offsets)
        power = np.asarray(check.pop("azimuth_power"))
        peak = np.argmax(power)
        ratio = power / power[peak]
        left = np.flatnonzero(ratio[:peak] < 0.5)[-1]
        right = np.flatnonzero(ratio[peak:] < 0.5)[0] + peak
        start = np.interp(0.5, ratio[left:left + 2], offsets[left:left + 2])
        end = np.interp(0.5, ratio[right - 1:right + 1][::-1], offsets[right - 1:right + 1][::-1])
        span = (count - 1) * config["pulse_interval_s"]
        checks.append(dict(pulses=count, aperture_span_s=span, platform_travel_m=span * config["platform_speed_mps"],
                           azimuth_peak_offset_m=float(offsets[peak]),
                           azimuth_mainlobe_half_power_width_m=float(end - start), **check))
        curves.extend(dict(pulses=count, offset_m=float(offset), normalized_power=float(level))
                      for offset, level in zip(offsets, ratio))
    widths = [row["azimuth_mainlobe_half_power_width_m"] for row in checks]
    ranges = {row["range_mainlobe_half_power_width_m"] for row in checks}
    if (not widths[0] > widths[1] > widths[2] or len(ranges) != 1
            or any(row["point_error_px"] > 1 or row["range_peak_error_bins"] > 1
                   or abs(row["azimuth_peak_offset_m"]) > 0.01 for row in checks)):
        raise ValueError(f"aperture point checks failed: {checks}")
    return checks, curves


def fixed_label_recall(root, entries):
    truth = {(row["scene_family"], row["pulses"]): load_masks(path / "masks.npy", SHAPE)
             for path, row in entries if row["group"] != "empty"}
    rows = []
    for _, row in entries:
        family, count = row["scene_family"], row["pulses"]
        if row["group"] == "empty":
            continue
        reference = truth[family, 61]
        common = np.logical_and.reduce([truth[family, pulses].any(axis=(1, 2)) for pulses in PULSES])
        for method in METHODS:
            prediction = load_masks(root / "predictions" / row["scene_id"] / f"{method}.npy", SHAPE)
            matches = score_masks(prediction[common], reference[common], MATCH)["matched"]
            rows.append(dict(scene_family=family, group=row["group"], pulses=count, method=method,
                             matched=matches, common_frames=int(common.sum())))
    totals = []
    for group in ("small", "normal"):
        for method in METHODS:
            for count in PULSES:
                subset = [row for row in rows if (row["group"], row["method"], row["pulses"]) == (group, method, count)]
                matched = sum(row["matched"] for row in subset)
                frames = sum(row["common_frames"] for row in subset)
                totals.append(dict(group=group, method=method, pulses=count, matched=matched,
                                   common_frames=frames, recall=matched / frames if frames else None))
    return rows, totals


def run_aperture_study(root):
    torch, smp, correlate = runtime()
    torch.set_num_threads(min(6, torch.get_num_threads()))
    settings, checkpoints, operating, hashes = frozen_settings()
    specs = families(SEED, "aperture")
    protocol = dict(seed=SEED, groups=GROUPS, bandwidth_mhz=BANDWIDTH_MHZ, snr_db=SNR_DB,
                    pulse_counts=PULSES, aperture_spans_s=[(n - 1) * 0.005 for n in PULSES],
                    scenes=specs, match_iou=MATCH, operating_points=operating, **hashes,
                    torch_version=torch.__version__, smp_version=smp.__version__, numpy_version=np.__version__,
                    precision_recall_target=0.8, snr_tolerance_db=SNR_TOLERANCE,
                    noise="one scene-level complex Gaussian train on the longest aperture range grid; same input sample for overlapping pulses; independently scaled per aperture to the same formed SNR",
                    snr_definition="10 log10(mean clean unblocked power / mean focused receiver-noise power)",
                    local_contrast="10 log10(median ring power / median geometry power) after registration; ring 2 to 5 pixels from geometry",
                    local_power_floor=POWER_FLOOR, bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=BOOTSTRAP_DRAWS,
                    pairing="same frame index; recall on frames visible under both apertures; equal family means",
                    fixed_labels="61-pulse visible masks on frames visible under all three apertures")
    write_json(root / "protocol.json", protocol)
    checks, curves = azimuth_checks(aperture_config(specs[0]))
    write_json(root / "point_checks.json", checks)
    write_csv(root / "azimuth_response.csv", curves)
    entries = build_dataset(root, specs, save=save_family, conditions=CONDITIONS, condition_keys=KEYS,
                            profile="aperture-study", seed=SEED)
    print("All aperture pairs verified. Evaluating frozen methods.", flush=True)
    networks = {name: (load_weights(torch, smp, checkpoints[name], config, "cpu"), config)
                for name, config in settings.items()}
    rows = evaluate(root, entries, torch, correlate, networks, operating, KEYS)
    imaging, detection = condition_metrics(rows, CONDITIONS, KEYS)
    paired, estimates = factorial_metrics(rows, CONTRASTS, KEYS, BOOTSTRAP_SEED)
    fixed_rows, fixed_totals = fixed_label_recall(root, entries)
    cases = mismatch_cases(rows, len(PULSES))
    for filename, records in (("per_frame", rows), ("imaging_by_aperture", imaging),
                              ("detection_by_aperture", detection), ("paired_by_family", paired),
                              ("paired_summary", estimates), ("fixed_label_by_family", fixed_rows),
                              ("fixed_label_recall", fixed_totals), ("mismatch_cases", cases)):
        write_csv(root / f"{filename}.csv", records)
    write_json(root / "results.json", dict(point=checks, imaging=imaging, detection=detection, paired=estimates,
                                          fixed_label_recall=fixed_totals, mismatch_cases=len(cases),
                                          mismatch_examples=cases[:5]))
