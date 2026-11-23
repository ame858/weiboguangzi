"""Evaluate a registered background on forward-held-out SNL shadow candidates."""
import argparse
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np

from audit_detection import paired_block_f1
from experiment import filtered, load_candidates, quality, summarize
from snl_box_quality import GEOMETRY_KEYS
from snl_forward_dense_fusion import candidate_scores, records, suppress
from snl_fusion import measurement
from snl_group_rescore import crossfit, fit_groups, grouped_row, mixed, replay, same_count, scored, tune
from snl_inputs import DEFAULT_OUTPUT, PROFILES, load_gray, load_labels, read_json, sha_file, write_csv, write_json
from snl_local_shadow_proposals import coverage
from snl_protocol import passed


LAGS = (3, 8, 15, 24, 36, 51, 68)
ATLAS_IDS = tuple(range(110, 219, 3))
GROUPING = (0.35, True)


def homogeneous(matrix):
    return np.vstack((matrix, (0, 0, 1)))


def standardized(image):
    values = image.astype(np.float32)
    mean = cv2.GaussianBlur(values, (0, 0), 12)
    square = cv2.GaussianBlur(values**2, (0, 0), 12)
    return (values - mean) / np.sqrt(np.maximum(square - mean**2, 0) + 16)


def warp(image, transform, shape):
    return cv2.warpAffine(image, transform[:2], shape[::-1], flags=cv2.INTER_LINEAR, borderValue=np.nan)


def quantiles(images, levels):
    stack = np.stack(images)
    count = np.isfinite(stack).sum(axis=0)
    stack.sort(axis=0)
    positions = np.asarray(levels, dtype=np.float32)[:, None, None] * np.maximum(count - 1, 0)
    lower = np.floor(positions).astype(np.intp)
    upper = np.ceil(positions).astype(np.intp)
    left = np.take_along_axis(stack, lower, axis=0)
    right = np.take_along_axis(stack, upper, axis=0)
    return (left + (right - left) * (positions - lower)).astype(np.float32)


class Background:
    def __init__(self, gray, transforms):
        self.gray = gray
        self.transforms = {i: homogeneous(matrix) for i, matrix in transforms.items()}
        self.shape = gray.shape[1:]
        reference = [warp(standardized(gray[i - 2]), self.transforms[i], self.shape)
                     for i in ATLAS_IDS if i in self.transforms]
        if len(reference) < 10:
            raise ValueError("The early background needs at least 10 registered reference frames")
        self.atlas_size = len(reference)
        self.atlas = quantiles(reference, (0.50, 0.75))
        supported = np.count_nonzero(np.isfinite(reference), axis=0) >= 5
        self.atlas[:, ~supported] = np.nan

    def frame(self, index):
        current = standardized(self.gray[index - 2])
        inverse = np.linalg.inv(self.transforms[index])
        history = [warp(standardized(self.gray[i - 2]), inverse @ self.transforms[i], self.shape)
                   for lag in LAGS if (i := index - lag) in self.transforms]
        if len(history) < 3:
            raise ValueError(f"Insufficient registered history for frame {index}")
        past = quantiles(history, (0.25, 0.50, 0.75))
        atlas = [warp(layer, inverse, self.shape) for layer in self.atlas]
        residuals = np.stack((past[1] - current, past[2] - current,
                              (past[1] - current) / (0.5 + past[2] - past[0]),
                              atlas[0] - current, atlas[1] - current))
        valid = np.isfinite(residuals[[0, 3]])
        return np.concatenate((np.nan_to_num(residuals), valid)).astype(np.float32)

    def prefix_check(self, index):
        prefix = Background(self.gray[:index - 1], {i: m[:2] for i, m in self.transforms.items() if i <= index})
        if not np.array_equal(self.frame(index), prefix.frame(index)):
            raise ValueError("Background features changed when removing the video suffix")


def make_groups(raw, features, background):
    original, extended = {}, {}
    for i, (boxes, scores) in raw.items():
        boxes, base, appearance = grouped_row(boxes, scores, features[i - 2], *GROUPING)
        background_data = measurement(255 * background.frame(i), boxes)[:, 3:]
        original[i] = boxes, base, appearance
        extended[i] = boxes, base, np.column_stack((appearance, background_data))
    return original, extended


def run(output, ffmpeg):
    archive = output / "forward-dense-frame-ranking"
    recipe = read_json(archive / "comparison.json")
    reference = read_json(archive / "suppression-comparison.json")
    paths = [archive / f"{key}.txt" for key in ("ranker", *GEOMETRY_KEYS)]
    if any(sha_file(path) != recipe["weights"][path.name] for path in paths):
        raise ValueError("Archived candidate weights changed")
    ranker, *geometry = (lgb.Booster(model_file=str(path)) for path in paths)
    fit_ids = recipe["fit_ids"]
    _, cal_ids, dev_ids = PROFILES["prospective"]
    if not max(ATLAS_IDS) < min(fit_ids) <= max(fit_ids) < min(cal_ids) <= max(cal_ids) < min(dev_ids):
        raise ValueError("The atlas, fit, calibration and development splits must be chronological")
    gray = load_gray(output, ffmpeg)
    transforms_path = output / "canonical-center-chained/canonical-registration.json"
    registration = read_json(transforms_path)
    anchor = registration["settings"]["anchor"]
    if anchor >= min(cal_ids):
        raise ValueError("The registration anchor must precede evaluation")
    transforms = {int(i): np.asarray(m) for i, m in registration["transforms"].items()}
    features = np.load(output / "motion-integration/motion-integration.npy", mmap_mode="r", allow_pickle=False)
    background = Background(gray, transforms)
    background.prefix_check(cal_ids[0])
    training, calibration = (load_labels(output, ids, gray)[0] for ids in (fit_ids, cal_ids))
    source = load_candidates(DEFAULT_OUTPUT / "early-stacked", recipe["fit_detector"], "forward-dense-fit")
    training_boxes = crossfit(records(source, features, transforms), training)
    cfg = recipe["evaluation_detector"]
    support = cfg.get("track_support", 1)

    def candidate_pool(scope):
        raw = records(load_candidates(output, cfg, scope), features, transforms)
        return {i: candidate_scores((ranker, geometry), row, recipe["selected"]["refine"], frame_ranked=True)
                for i, row in raw.items()}

    study = output / "background-atlas"
    study.mkdir(exist_ok=True)
    models, operations = {}, {}
    train_groups = make_groups(training_boxes, features, background)
    cal_groups = make_groups(candidate_pool("validation"), features, background)
    for method, fit, cal in zip(("appearance", "background"), train_groups, cal_groups, strict=True):
        model, examples, positives = fit_groups(fit, training)
        path = study / f"{method}.txt"
        model.save_model(str(path))
        decision = tune(scored(cal, model), calibration, support, (0, 0.25, 0.5, 0.75, 1.0))
        operations[method] = dict(selected=decision, examples=examples, positives=positives, sha256=sha_file(path))
        models[method] = model
        print(method, decision, flush=True)
    selected = max(operations, key=lambda key: quality(operations[key]["selected"]["calibration"]))
    report = dict(atlas_ids=ATLAS_IDS, atlas_frames_registered=background.atlas_size, historical_lags=LAGS,
                  fit_ids=fit_ids, calibration_ids=cal_ids, development_ids=dev_ids, matching_iou=0.50,
                  registration_anchor=anchor, transforms_sha256=sha_file(transforms_path), prefix_check_frame=cal_ids[0],
                  selected_by_calibration=selected, operations=operations)
    write_json(study / "calibration.json", report)

    dev_pool = candidate_pool("development")
    dev_groups = make_groups(dev_pool, features, background)
    development = load_labels(output, dev_ids, gray)[0]
    point = reference["selected"]
    base = filtered(suppress(dev_pool, point["method"], point["strength"]), point["threshold"], point["low"],
                    confirmations=support)
    baseline, base_rows = summarize(base, development)
    if baseline != reference["development"]:
        raise ValueError("The archived reference did not replay")
    rows = [dict(method="reference", **row) for row in base_rows]
    comparisons = {}
    for (method, model), groups in zip(models.items(), dev_groups, strict=True):
        decision = operations[method]["selected"]
        scores = scored(groups, model)
        metrics, per_frame = replay(scores, development, decision, support)
        operations[method].update(development=metrics, iou50_ceiling=coverage(scores, development),
                                  fixed_output_count=same_count(mixed(scores, decision["weight"]), development, base,
                                                                decision["weight"]),
                                  paired_vs_reference=paired_block_f1(per_frame, base_rows))
        comparisons[method] = per_frame
        rows.extend(dict(method=method, **row) for row in per_frame)
        print(method, metrics, flush=True)
    report["reference_development"] = baseline
    report["background_vs_appearance"] = paired_block_f1(comparisons["background"], comparisons["appearance"])
    decision = operations[selected]
    report["development_gate_passed"] = passed(decision["selected"]["calibration"]) and passed(decision["development"])
    write_json(study / "comparison.json", report)
    write_csv(study / "development.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    run(args.output.resolve(), args.ffmpeg)
