"""Test emphasis on faint shadows with an equal-budget, paired training ablation."""
import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np

from infer_video import find_ffmpeg, torch
from snl_center import train_center
from snl_detector import load_trained, nonmax, predict_frame
from snl_external_shadow_adaptation import EPOCHS, TRAINING_TRIPLETS, fit_samples
from snl_external_shadow_benchmark import metrics, predict, summarize
from snl_external_weak_cues import SOURCES, choose, coded, realized_depth, table, test_cases, training_priors
from snl_inputs import read_json, sha_file, write_csv, write_json
from snl_qatar_pilot import EXTERNAL
from videosar.data import run_directory
from videosar.snl_causal import box_iou


REFERENCE = EXTERNAL / "injected-weak-raw-residual"
MEXICO_CITY = EXTERNAL / "iceye-open/ICEYE_9G3MZC_20260317T141841Z_9307864_X49_SLEDF_VID.mp4"
TIMES = (7, 8, 10, 13, 15, 17, 20, 22)
POSITIONS = (36, 252, 504)
TEST_SEED = 890217
SUBSET_SEED = 152181
STRONG_MARGIN = 0.05
MODE = "raw_residual"


def digest(cases):
    result = hashlib.sha256()
    for _, image, target in cases:
        result.update(image.tobytes())
        result.update(target.tobytes())
    return result.hexdigest()


def verify_training(control, tailored, replacements):
    rows = []
    for i, (before, after) in enumerate(zip(control, tailored, strict=True)):
        original, first, old_target = before
        adjusted, second, new_target = after
        np.testing.assert_array_equal(old_target, new_target)
        edited = i // 3 in replacements and i % 3 == 2
        if not edited:
            if original != adjusted or not np.array_equal(first, second):
                raise ValueError(f"An unselected fitting sample changed at {i}")
            continue
        if original["injected_depth"] != 15 or original | {"injected_depth": 7} != adjusted:
            raise ValueError(f"Geometry or adjusted intensity did not match at {i}")
        if not np.all(second >= first) or np.array_equal(first, second):
            raise ValueError(f"The lower-depth sample was not a strict intensity relaxation at {i}")
        left, top, right, bottom = np.rint(adjusted["box"]).astype(int)
        region = np.s_[0, top:bottom, left:right]
        background = control[i - 2][1][region].astype(np.int16)
        rows.append(dict(training_group=i // 3, **adjusted,
                         original_drop=float(np.median(background - first[region])),
                         adjusted_drop=float(np.median(background - second[region]))))
    if len(rows) != len(replacements):
        raise ValueError("The number of changed training samples differs from the frozen design")
    return rows


def candidate_audit(cases, selected, threshold, folder, config):
    model = load_trained(folder, config | {"dense_candidate_limit": 1024})
    rows = []
    for (spec, image, truth), (peaks, scores) in zip(cases, selected, strict=True):
        if spec["injected_depth"] != 5:
            continue
        dense, score = predict_frame(model, image, "temporal")
        nms_dense, _ = nonmax(dense, score, 0.35)
        stages = {"peak_pool": peaks, "frozen_output": peaks[scores >= threshold],
                  "dense_pool": dense, "dense_after_nms": nms_dense}
        rows.append(dict(spec, dense_count=len(dense), dense_nms_count=len(nms_dense),
                         **{name: int((box_iou(boxes, truth) >= 0.50).any()) for name, boxes in stages.items()}))
    return rows


def evaluate(output, ffmpeg, reference):
    previous = read_json(reference / "comparison.json")
    frozen_control = previous["checkpoints"][MODE]
    control_dir, control_config = Path(frozen_control["folder"]), frozen_control["config"]
    sizes, motion = training_priors(ffmpeg)
    control_bank, calibration, source_hash = fit_samples(sizes, motion, ffmpeg)
    selected = np.random.default_rng(SUBSET_SEED).choice(TRAINING_TRIPLETS, size=TRAINING_TRIPLETS // 2, replace=False)
    replacements = dict.fromkeys(map(int, selected), 7)
    targeted, check, trained_hash = fit_samples(sizes, motion, ffmpeg, strong_overrides=replacements)
    if (source_hash != previous["settings"]["training_sha256"] or source_hash != trained_hash or
            digest(calibration) != digest(check) or len(control_bank) != previous["settings"]["training_cases"]):
        raise ValueError("The Rotterdam training source or control cases changed")
    changed = verify_training(control_bank, targeted, replacements)
    sources = SOURCES | {"MexicoCity": MEXICO_CITY}
    settings = dict(training_frames=len(control_bank), training_source_sha256=source_hash, calibration_cases=len(calibration),
                    training_digests=dict(balance=digest(control_bank), weak_emphasis=digest(targeted)),
                    calibration_digest=digest(calibration), replaced_groups=sorted(replacements), subset_seed=SUBSET_SEED,
                    original_depth=15, replacement_depth=7, epoch_budget=EPOCHS, model_seed=control_config["seed"],
                    test_times=TIMES, test_positions=POSITIONS, test_seed=TEST_SEED, test_sites=list(sources),
                    dense_budget=1024, dense_decoder_score_floor=0.005, strong_recall_maximum_loss=STRONG_MARGIN,
                    acceptance="more weak hits, no extra blank outputs, at most five percentage points of strong recall loss, and no overall F1 loss")
    with run_directory(output, "snl-external-weak-curriculum", settings) as root:
        model_dir = root / "weak-emphasis"
        model_dir.mkdir()
        features = np.stack([row[1] for row in coded(targeted, MODE)])
        labels = {i + 2: row[2] for i, row in enumerate(targeted)}
        (config,) = list(train_center(model_dir, features, labels, input_profile="stacked", epoch_budget=EPOCHS,
                                     seed=control_config["seed"]))
        config |= {"contrast_gain": control_config["contrast_gain"]}
        checkpoints = {"balanced": (control_dir, control_config), "weak_emphasis": (model_dir, config)}
        models = {name: load_trained(folder, cfg) for name, (folder, cfg) in checkpoints.items()}
        observed = coded(calibration, MODE)
        trials = {name: table(calibration, predict(model, observed, "raw", 1, None, 0.35)) for name, model in models.items()}
        baseline = previous["points"][MODE]
        cap = baseline["blank_outputs"]
        points = {name: choose(rows, cap) for name, rows in trials.items()}
        if points["balanced"] != baseline:
            raise ValueError("Balanced model did not exactly reproduce its archived calibration point")
        frozen = dict(settings=settings, calibration_blank_cap=cap, points=points, trials=trials,
                      checkpoints={name: dict(folder=str(folder), config=cfg) for name, (folder, cfg) in checkpoints.items()})
        write_json(root / "frozen-selection.json", frozen)
        write_csv(root / "training-changes.csv", changed)
        print("Frozen curriculum", points, flush=True)

        results, case_rows, geometry_rows = {}, [], []
        for site, path in sources.items():
            cases, source = test_cases(site, path, ffmpeg, sizes, motion, times=TIMES, positions=POSITIONS, seed=TEST_SEED)
            features = coded(cases, MODE)
            site_results = {}
            for name, (folder, cfg) in checkpoints.items():
                raw = predict(models[name], features, "raw", 1, None, 0.35)
                stats, counts = summarize(cases, raw, points[name]["threshold"])
                _, common = summarize(cases, raw, baseline["threshold"])
                candidates = candidate_audit(features, raw, points[name]["threshold"], folder, cfg)
                case_rows.extend(dict(site=site, model=name, operating_point="calibrated", **row) for row in counts)
                case_rows.extend(dict(site=site, model=name, operating_point="shared", **row) for row in common)
                geometry_rows.extend(dict(site=site, model=name, **row) for row in candidates)
                coverage = {key: sum(row[key] for row in candidates) for key in
                            ("peak_pool", "dense_pool", "dense_after_nms", "frozen_output", "dense_count", "dense_nms_count")}
                if coverage["frozen_output"] != stats["by_depth"]["5"]["tp"]:
                    raise ValueError("Candidate replay disagrees with the frozen endpoint")
                site_results[name] = dict(calibrated=stats, weak_candidate_coverage=coverage)
                print(site, name, stats["iou50"], coverage, flush=True)
            results[site] = dict(source=source, realized_weak_drop=realized_depth(cases), arms=site_results)

        pooled = {}
        for point, name in ((p, name) for p in ("calibrated", "shared") for name in checkpoints):
            rows = [row for row in case_rows if (row["operating_point"], row["model"]) == (point, name)]
            pooled.setdefault(point, {})[name] = dict(iou50=metrics(rows), weak=metrics([r for r in rows if r["injected_depth"] == 5]),
                strong=metrics([r for r in rows if r["injected_depth"] == 15]), blank_outputs=sum(r["fp"] for r in rows if r["injected_depth"] == 0))
        old, new = (pooled["calibrated"][name] for name in checkpoints)
        gates = dict(weak_hits_increase=new["weak"]["tp"] > old["weak"]["tp"],
                     blank_outputs_do_not_increase=new["blank_outputs"] <= old["blank_outputs"],
                     strong_recall_within_margin=new["strong"]["recall"] - old["strong"]["recall"] >= -STRONG_MARGIN,
                     no_overall_f1_loss=new["iou50"]["f1"] >= old["iou50"]["f1"])
        write_json(root / "comparison.json", frozen | dict(results=results, pooled=pooled, gates=gates))
        write_csv(root / "case-rows.csv", case_rows)
        write_csv(root / "candidate-rows.csv", geometry_rows)
        print("Controlled gates", gates, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXTERNAL / "injected-weak-curriculum")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--reference-run", type=Path, default=REFERENCE)
    args = parser.parse_args()
    cv2.setNumThreads(3)
    torch.set_num_threads(4)
    evaluate(args.output.resolve(), find_ffmpeg(args.ffmpeg), args.reference_run.resolve())
