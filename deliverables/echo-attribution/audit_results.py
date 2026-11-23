"""Recompute saved-mask metrics and check bandwidth effects against fixed labels."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import label


METHODS = ("unet", "fpn", "difference")
MATCH = 0.5
BANDWIDTHS = (300, 600)
SNRS = (25, 10)
GROUPS = ("small", "normal", "empty")


def counts(prediction, truth):
    regions, n = label(prediction)
    areas = np.bincount(regions.ravel(), minlength=n + 1)[1:]
    intersections = np.bincount(regions[truth], minlength=n + 1)[1:]
    iou = intersections / (areas + truth.sum() - intersections)
    best = float(iou.max(initial=0))
    match = int(best >= MATCH)
    return best, match, n - match


def audit(root):
    manifest = json.loads((root / "dataset/dataset.json").read_text(encoding="utf-8"))["entries"]
    with (root / "per_frame.csv").open(encoding="utf-8-sig", newline="") as stream:
        frames = {(r["scene_id"], int(r["frame"])): r for r in csv.DictReader(stream)}
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    totals = defaultdict(lambda: np.zeros(3, dtype=int))
    by_family = defaultdict(dict)
    fixed = defaultdict(dict)
    count = 0
    for row in manifest:
        scene, family, group = (row[key] for key in ("scene_id", "scene_family", "group"))
        bw, snr = row["bandwidth_mhz"], row["snr_db"]
        folder = root / "dataset" / scene
        digest = hashlib.sha256()
        for name in ("frames.npy", "masks.npy", "geometric_masks.npy", "metadata.json"):
            digest.update((folder / name).read_bytes())
        assert digest.hexdigest() == row["sha256"], scene
        reference = (folder / "unoccluded_power.npy").read_bytes()
        assert hashlib.sha256(reference).hexdigest() == row["reference_sha256"], scene
        truth = np.load(folder / "masks.npy", allow_pickle=False).astype(bool)
        assert truth.shape == (20, 64, 80) and len(truth) == row["frames"], scene
        labels = {basis: np.load(root / "dataset" / f"{family}-{basis}MHz-{snr}dB" / "masks.npy",
                                 allow_pickle=False).astype(bool) for basis in BANDWIDTHS}
        for method in METHODS:
            prediction = np.load(root / "predictions" / scene / f"{method}.npy", allow_pickle=False)
            assert prediction.shape == truth.shape and np.isin(prediction, (0, 1)).all(), (scene, method)
            match_array = []
            fp_array = []
            for i, (mask, target) in enumerate(zip(prediction, truth)):
                record = frames[scene, i]
                best, match, fp = counts(mask, target)
                expected = (float(record[f"{method}_best_iou"]), int(record[f"{method}_matched"]),
                            int(record[f"{method}_false_positives"]))
                assert (best, match, fp) == expected, (scene, method, i)
                totals[group, bw, snr, method] += [match, fp, int(target.any()) - match]
                match_array.append(match)
                fp_array.append(fp)
                count += 1
            support = truth.any(axis=(1, 2))
            by_family[group, method, family][bw, snr] = (np.array(match_array), np.array(fp_array), support)
            if group == "empty":
                assert not truth.any(), scene
                continue
            for basis, targets in labels.items():
                common = labels[300].any(axis=(1, 2)) & labels[600].any(axis=(1, 2))
                matches = [counts(mask, target)[1] for mask, target in zip(prediction[common], targets[common])]
                fixed[group, snr, method, basis, family][bw] = (sum(matches), int(common.sum()))
    assert len(frames) == 1920 and count == 5760
    for r in results["detection"]:
        group, bw, snr, method = (r[k] for k in ("group", "bandwidth_mhz", "snr_db", "method"))
        observed = (sum((totals[g, bw, snr, method] for g in GROUPS), start=np.zeros(3, dtype=int))
                    if group == "all" else totals[group, bw, snr, method])
        assert observed.tolist() == [r[k] for k in ("matched", "false_positives", "missed")]
    pairs_checked = paired_recall_audit(results["paired"], by_family)

    grouped = defaultdict(list)
    for (group, snr, method, basis, _), by_bw in fixed.items():
        low, n = by_bw[300]
        high, other_n = by_bw[600]
        assert n == other_n
        if n:
            grouped[group, snr, method, basis].append((low, high, n))
    sensitivity = []
    for (group, snr, method, basis), families in grouped.items():
        data = np.array(families)
        low, high, n = data.sum(axis=0)
        sensitivity.append(dict(group=group, snr_db=snr, method=method, fixed_label_bandwidth_mhz=basis,
                                families=len(families), common_visible_frames=int(n),
                                recall_300mhz=low / n, recall_600mhz=high / n,
                                equal_family_delta=float(np.mean((data[:, 1] - data[:, 0]) / data[:, 2]))))
    assert sensitivity
    with (root / "fixed_label_sensitivity.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sensitivity[0]))
        writer.writeheader()
        writer.writerows(sensitivity)
    (root / "prediction_audit.json").write_text(json.dumps(dict(independent_mask_frame_checks=count,
        dataset_hashes_verified=len(manifest), condition_totals_verified=len(results["detection"]),
        paired_deltas_verified=pairs_checked, sensitivity_rows=len(sensitivity)), indent=2) + "\n", encoding="utf-8")


def paired_recall_audit(estimates, by_family):
    # Slots are 300/25, 300/10, 600/25, and 600/10.
    conditions = [(bw, snr) for bw in BANDWIDTHS for snr in SNRS]
    contrasts = {"bandwidth_at_25dB": [-1, 0, 1, 0], "bandwidth_at_10dB": [0, -1, 0, 1],
                 "noise_at_300MHz": [-1, 1, 0, 0], "noise_at_600MHz": [0, 0, -1, 1],
                 "interaction": [1, -1, -1, 1]}
    checked = 0
    for row in estimates:
        group, metric = row["group"], row["metric"]
        method = metric.split("_", 1)[0]
        if method not in METHODS:
            continue
        weights = np.array(contrasts[row["contrast"]])
        paired = []
        support = 0
        for (family_group, family_method, _), versions in by_family.items():
            if (family_group, family_method) != (group, method):
                continue
            matched, fp, visible = (np.stack(arrays) for arrays in zip(*(versions[c] for c in conditions)))
            common = np.all(visible[weights != 0], axis=0) if group != "empty" else np.ones(20, dtype=bool)
            if common.any():
                paired.append(np.mean((weights @ (fp if group == "empty" else matched))[common]))
                support += int(common.sum())
        assert len(paired) == row["families"] and support == row["frames"], row
        assert np.isclose(np.mean(paired), row["delta"], atol=1e-12, rtol=0), row
        checked += 1
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    audit(parser.parse_args().run)
