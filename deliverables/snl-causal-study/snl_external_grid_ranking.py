"""Rank full-grid VideoSAR proposals with geometry and causal box-scale features."""
import argparse
from itertools import product
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
from torchvision.ops import nms

from experiment import CONFIDENCES
from infer_video import find_ffmpeg, torch
from snl_center import center_boxes_at, predict_views
from snl_detector import TILE, load_trained
from snl_external_candidate_quota import BUDGET
from snl_external_shadow_adaptation import fit_samples
from snl_external_shadow_benchmark import bank, metrics, score_cases, summarize
from snl_external_weak_cues import SOURCES, choose, encode, table, test_cases, training_priors
from snl_external_weak_curriculum import MEXICO_CITY, MODE, REFERENCE
from snl_fusion import measurement
from snl_inputs import read_json, sha_file, write_csv, write_json
from snl_qatar_pilot import EXTERNAL, decode
from videosar.data import run_directory
from videosar.snl_causal import STACKED_HISTORY_LAGS, box_iou


LIMA = EXTERNAL / "iceye-open/ICEYE_6MC5KR_20250526T033241Z_4770346_X31_SLEDF_VID.mp4"
FIT_TIMES = (8, 10, 12, 14, 16, 18, 21, 24)
TEST_TIMES = (7, 9, 12, 14, 17, 19, 21, 24)
TEST_POSITIONS = (24, 264, 516)
FIT_WINDOWS = 192
FIT_SEED = 730921
TEST_SEED = 426503
NMS = 0.35
ITERATIONS = 120
BASIC_WIDTH = 5
PAIRED_TEST_SEED = 613027
PAIRED_TEST_TIMES = (8, 10, 11, 13, 16, 18, 22, 23)
PAIRED_TEST_POSITIONS = (16, 252, 500)
PAIRED_THRESHOLDS = (*CONFIDENCES, 0.975, 0.99, 1.0)


@torch.inference_mode()
def proposals(model, image, observation, *, with_grid=False):
    encoded = encode(image, MODE, observation)
    device = next(model.parameters()).device
    network = torch.from_numpy(encoded).to(device).float()[None] / 255
    output = predict_views(model, network)[0]
    indices = torch.arange(output[0].numel(), device=device)
    boxes = center_boxes_at(model, output, indices, *image.shape[-2:])
    valid = (boxes[:, 2:] > boxes[:, :2]).all(dim=1)
    geometry = boxes[valid].cpu().numpy()
    heat, occupancy = output[[0, 5]].reshape(2, -1)[:, valid].T.cpu().numpy().T
    width, height = (geometry[:, 2:] - geometry[:, :2]).T
    basic = np.column_stack((heat, occupancy, width, height, height / width)).astype(np.float32)
    if with_grid:
        return geometry, basic, indices[valid].cpu().numpy()
    return geometry, basic


def causal_radar_views(image, observation):
    valid = np.asarray(STACKED_HISTORY_LAGS) <= observation
    current = image[0].astype(np.float32)
    history = image[1:][valid].astype(np.float32)
    reference = np.median(history, axis=0)
    variation = np.median(np.abs(history - reference), axis=0)
    return np.stack((current, reference, reference - current, variation))


def features(image, observation, boxes, basic):
    if basic.shape[1] != BASIC_WIDTH:
        raise ValueError("Unexpected geometry and heat-map feature count")
    physical = measurement(causal_radar_views(image, observation), boxes)[:, 3:]
    return np.column_stack((basic, physical)).astype(np.float32)


def take_random(values, size, rng):
    return rng.choice(values, min(size, len(values)), replace=False)


def training_candidates(model, cases):
    rng = np.random.default_rng(FIT_SEED)
    features_by_case, labels_by_case, weights_by_case = [], [], []
    counts = dict(sampled=0, positives=0, positive_below_old_floor=0, windows_with_positive=0)
    for spec, image, truth in cases:
        boxes, basic = proposals(model, image, spec["observation"])
        iou = box_iou(boxes, truth).max(axis=1, initial=0)
        positives = np.flatnonzero(iou >= 0.50)
        negative = np.flatnonzero(iou < 0.50)
        high = negative[np.argsort(-basic[negative, 0], kind="stable")[:160]]
        near = negative[iou[negative] >= 0.10]
        background = np.unique(np.r_[high, take_random(near, 64, rng), take_random(negative, 320, rng)])
        selected = np.r_[positives, background]
        features_by_case.append(features(image, spec["observation"], boxes[selected], basic[selected]))
        labels_by_case.append(iou[selected] >= 0.50)
        weights_by_case.append(np.r_[np.full(len(positives), 64 / max(len(positives), 1)),
                                     np.full(len(background), 64 / max(len(background), 1))])
        counts["sampled"] += len(selected)
        counts["positives"] += len(positives)
        counts["positive_below_old_floor"] += int((basic[positives, 0] < np.log(0.005 / 0.995)).sum())
        counts["windows_with_positive"] += bool(len(positives))
    return tuple(np.concatenate(part) for part in (features_by_case, labels_by_case, weights_by_case)), counts


def paired_training_candidates(model, cases):
    rng = np.random.default_rng(FIT_SEED)
    common_inputs, common_labels, common_weights = [], [], []
    extra_inputs = {"background_control": [], "paired_background": []}
    extra_weights = []
    counts = dict(cases=len(cases), shared_rows=0, positives=0, extra_rows=0, unmatched_grid=0, repeated_anchor=0,
                  weak_pairs=0, strong_pairs=0)
    heat_gaps = []
    for blank, weak, strong in zip(cases[::3], cases[1::3], cases[2::3], strict=True):
        blank_spec, blank_image, blank_truth = blank
        observation = blank_spec["observation"]
        if tuple(case[0]["injected_depth"] for case in (blank, weak, strong)) != (0, 5, 15) or len(blank_truth):
            raise ValueError("Expected matched blank, weak and strong fitting windows")
        if any({k: v for k, v in case[0].items() if k != "injected_depth"} !=
               {k: v for k, v in blank_spec.items() if k != "injected_depth"} for case in (weak, strong)):
            raise ValueError("Pair metadata differs across injection doses")
        blank_boxes, blank_basic, blank_grid = proposals(model, blank_image, observation, with_grid=True)
        used_pairs = np.zeros(len(blank_grid), dtype=bool)
        used_controls = np.zeros(len(blank_grid), dtype=bool)
        for spec, image, truth in (weak, strong, blank):
            boxes, basic, grid = (blank_boxes, blank_basic, blank_grid) if not len(truth) else proposals(
                model, image, observation, with_grid=True)
            iou = box_iou(boxes, truth).max(axis=1, initial=0)
            positive = np.flatnonzero(iou >= 0.50)
            can_sample = iou < 0.50
            if not len(truth):
                can_sample &= ~(used_pairs | used_controls)
            negative = np.flatnonzero(can_sample)
            high = negative[np.argsort(-basic[negative, 0], kind="stable")[:160]]
            background = np.unique(np.r_[high, take_random(negative[iou[negative] >= 0.10], 64, rng),
                                         take_random(negative, 320, rng)])

            paired_grid = np.isin(blank_grid, grid[positive], assume_unique=True)
            counts["unmatched_grid"] += len(positive) - int(paired_grid.sum())
            counts["repeated_anchor"] += int((paired_grid & (used_pairs | used_controls)).sum())
            paired = np.flatnonzero(paired_grid & ~(used_pairs | used_controls))
            if len(paired):
                # Control negatives come from the same blank scene and have the closest original heat values outside the injection region.
                center_difference = (blank_boxes[:, :2] + blank_boxes[:, 2:] - truth[0, :2] - truth[0, 2:]) / 2
                distant = np.flatnonzero((box_iou(blank_boxes, truth)[:, 0] < 0.10) &
                                         (np.square(center_difference).sum(axis=1) >= 32**2) &
                                         ~(used_pairs | used_controls | paired_grid))
                heat = blank_basic[distant, 0]
                if len(distant) < len(paired):
                    raise ValueError("Not enough distinct blank control candidates")
                controls = []
                for index in paired:
                    distance = np.abs(heat - blank_basic[index, 0])
                    distance[controls] = np.inf
                    choice = int(distance.argmin())
                    controls.append(choice)
                    heat_gaps.append(float(distance[choice]))
                for name, indices in (("background_control", distant[controls]), ("paired_background", paired)):
                    extra_inputs[name].append(features(blank_image, observation, blank_boxes[indices], blank_basic[indices]))
                used_pairs[paired] = True
                used_controls[distant[controls]] = True
                extra_weights.append(np.full(len(paired), 32 / len(paired)))
                counts["extra_rows"] += len(paired)
                counts["weak_pairs" if spec["injected_depth"] == 5 else "strong_pairs"] += len(paired)

            shared = np.r_[positive, background]
            common_inputs.append(features(image, observation, boxes[shared], basic[shared]))
            common_labels.append(iou[shared] >= 0.50)
            common_weights.append(np.r_[np.full(len(positive), 64 / max(1, len(positive))),
                                        np.full(len(background), (32 if len(paired) else 64) / len(background))])
            counts["shared_rows"] += len(shared)
            counts["positives"] += len(positive)
    if not counts["extra_rows"]:
        raise ValueError("No valid same-grid blank matches in the training bank")
    shared = np.concatenate(common_inputs)
    labels = np.r_[np.concatenate(common_labels), np.zeros(counts["extra_rows"], dtype=bool)]
    weights = np.concatenate([*common_weights, *extra_weights])
    counts["blank_control_abs_heat_difference"] = np.percentile(heat_gaps, (50, 95, 100)).tolist()
    return {name: np.concatenate([shared, *extra]) for name, extra in extra_inputs.items()}, labels, weights, counts


def fit_rankers(root, model, cases, *, paired_negatives=False):
    if paired_negatives:
        matrices, labels, weights, counts = paired_training_candidates(model, cases)
    else:
        (inputs, labels, weights), counts = training_candidates(model, cases)
        matrices = {"heat_geometry": inputs[:, :BASIC_WIDTH], "causal_physics": inputs}
    if not labels.any() or labels.all():
        raise ValueError("The independent fitting bank has no detectable positive or negative grid examples")
    options = dict(objective="binary", learning_rate=0.05, num_leaves=15, max_depth=5, min_data_in_leaf=36,
                   lambda_l2=0.1, seed=FIT_SEED, num_threads=5, verbosity=-1, deterministic=True, force_col_wise=True)
    rankers, hashes = {}, {}
    for name, matrix in matrices.items():
        scorer = lgb.train(options, lgb.Dataset(matrix, label=labels, weight=weights), num_boost_round=ITERATIONS)
        path = root / f"{name}.txt"
        scorer.save_model(str(path))
        hashes[name] = sha_file(path)
        rankers[name] = lgb.Booster(model_file=str(path))
    return rankers, hashes, counts


def fit_bank(ffmpeg, sizes, motion):
    stac = read_json(LIMA.with_name(LIMA.name.removesuffix("_VID.mp4") + ".json"))
    data, _ = decode(LIMA, ffmpeg)
    if len(data) != 25 or min(data.shape[1:]) < 720:
        raise ValueError("Expected all 25 official Lima observations and a central 720-pixel crop")
    y, x = ((dimension - 720) // 2 for dimension in data.shape[1:])
    cropped = data[:, y:y + 720, x:x + 720]
    rng = np.random.default_rng(FIT_SEED)
    positions = [(int(rng.integers(0, 720 - TILE + 1)), int(rng.integers(0, 720 - TILE + 1)), int(observation))
                 for observation in rng.choice(FIT_TIMES, FIT_WINDOWS)]
    cases = list(bank(cropped, sizes, motion, placements=positions, seed=FIT_SEED, observation_order=FIT_TIMES))
    return cases, dict(stac_id=stac["id"], acquired_utc=stac["properties"]["datetime"],
                       video_sha256=sha_file(LIMA), crop_xywh=[x, y, 720, 720], sample_count=len(cases))


def official_test_bank(path, ffmpeg, sizes, motion, times, positions, seed):
    item = read_json(path.with_name(path.name.removesuffix("_VID.mp4") + ".json"))
    if path.name != item["id"] + "_VID.mp4":
        raise ValueError(f"MP4 filename does not match STAC item {item['id']}")
    raw, _ = decode(path, ffmpeg)
    if len(raw) != 25 or min(raw.shape[1:]) < 720:
        raise ValueError(f"Expected 25 official observations and a 720-pixel crop for {item['id']}")
    y, x = ((dimension - 720) // 2 for dimension in raw.shape[1:])
    cases = list(bank(raw[:, y:y + 720, x:x + 720], sizes, motion, placements=product(positions, positions, times),
                      seed=seed, observation_order=times))
    return cases, dict(video_sha256=sha_file(path), stac_id=item["id"], acquired_utc=item["properties"]["datetime"],
                       bbox=item["bbox"], crop_xywh=[x, y, 720, 720])


def rank(model, rankers, cases, *, audit_score_response=False):
    scored = {name: [] for name in ("heatmap", *rankers)}
    evidence = {name: [] for name in scored}
    responses = []
    blank = None
    for spec, image, truth in cases:
        decoded = proposals(model, image, spec["observation"], with_grid=audit_score_response)
        boxes, basic = decoded[:2]
        if len(boxes) < BUDGET:
            raise ValueError("The full-grid decode did not supply a common 1,024-candidate budget")
        x = features(image, spec["observation"], boxes, basic)
        scores = {"heatmap": 1 / (1 + np.exp(-np.clip(basic[:, 0], -40, 40)))}
        scores.update({name: scorer.predict(x[:, :scorer.num_feature()], num_threads=4) for name, scorer in rankers.items()})
        if audit_score_response:
            grid = decoded[2]
            key = (spec["tile_x"], spec["tile_y"], spec["observation"], spec["box"])
            if spec["injected_depth"] == 0:
                blank = key, grid, scores
            else:
                if blank is None or key != blank[0]:
                    raise ValueError("Score response requires the corresponding blank trial first")
                matching = np.flatnonzero(box_iou(boxes, truth)[:, 0] >= 0.50)
                common, injected_rows, blank_rows = np.intersect1d(grid[matching], blank[1], assume_unique=True,
                                                                   return_indices=True)
                for name, values in scores.items():
                    change = values[matching[injected_rows]] - blank[2][name][blank_rows]
                    responses.append(dict(spec, model=name, matching_grids=len(matching), compared_grids=len(common),
                                          scores_increased=int((change > 0).sum()),
                                          scores_decreased=int((change < 0).sum()),
                                          median_score_change=float(np.median(change)) if len(change) else None))
        for name, values in scores.items():
            # Stable grid order breaks ties after score and original heat.
            order = np.lexsort((-basic[:, 0], -values))[:BUDGET]
            candidates, probability = boxes[order], values[order]
            priority = torch.arange(len(candidates), 0, -1, dtype=torch.float32)
            kept = nms(torch.from_numpy(candidates), priority, NMS).numpy()
            nms_boxes, nms_scores = candidates[kept], probability[kept]
            scored[name].append((nms_boxes, nms_scores))
            if spec["injected_depth"] == 5:
                evidence[name].append(dict(spec, before_nms=int((box_iou(candidates, truth) >= 0.50).any()),
                                           after_nms=int((box_iou(nms_boxes, truth) >= 0.50).any()), candidates=len(candidates)))
    return scored, evidence, responses


def pooled_results(points, count_rows, candidate_rows):
    pooled = {}
    for name in points:
        counts = [r for r in count_rows if r["model"] == name]
        candidates = [r for r in candidate_rows if r["model"] == name]
        coverage = {phase: sum(r[phase] for r in candidates) for phase in ("before_nms", "after_nms")}
        pooled[name] = dict(iou50=metrics(counts), weak=metrics([r for r in counts if r["injected_depth"] == 5]),
                            blank_outputs=sum(r["fp"] for r in counts if r["injected_depth"] == 0),
                            weak_candidate_coverage=coverage)
    return pooled


def advances(row, base):
    return (row["weak"]["tp"] > base["weak"]["tp"] and row["blank_outputs"] <= base["blank_outputs"]
            and row["iou50"]["f1"] >= base["iou50"]["f1"])


def equal_output_rows(cases, predictions, heatmap_threshold):
    heatmap_counts = [int((scores >= heatmap_threshold).sum()) for _, scores in predictions["heatmap"]]
    capacities = [min(len(boxes) for boxes, _ in window) for window in zip(*predictions.values(), strict=True)]
    output_counts = [min(requested, capacity) for requested, capacity in zip(heatmap_counts, capacities, strict=True)]
    for name, raw in predictions.items():
        top = [(boxes[:count], scores[:count]) for (boxes, scores), count in zip(raw, output_counts, strict=True)]
        for row, count, requested in zip(score_cases(cases, top, -np.inf, "iou50"), output_counts, heatmap_counts,
                                         strict=True):
            yield dict(model=name, outputs=count, heatmap_outputs=requested, **row)


def evaluate(output, ffmpeg, *, paired_negatives=False, fresh_paths=()):
    prior = read_json(REFERENCE / "comparison.json")
    geometry = prior["checkpoints"][MODE]
    model = load_trained(Path(geometry["folder"]), geometry["config"])
    sizes, motion = training_priors(ffmpeg)
    fitting, fitting_source = fit_bank(ffmpeg, sizes, motion)
    _, calibration, cal_hash = fit_samples(sizes, motion, ffmpeg)
    if cal_hash != prior["settings"]["training_sha256"] or fitting_source["video_sha256"] == cal_hash:
        raise ValueError("Fitting and calibration source provenance changed")
    fresh = {path.stem.removesuffix("_VID"): path for path in fresh_paths}
    if len(fresh) != len(fresh_paths):
        raise ValueError("Duplicate fresh-site names")
    sites = fresh | SOURCES | {"MexicoCity": MEXICO_CITY}
    times, positions, seed = (PAIRED_TEST_TIMES, PAIRED_TEST_POSITIONS, PAIRED_TEST_SEED) if paired_negatives else (
        TEST_TIMES, TEST_POSITIONS, TEST_SEED)
    settings = dict(fitting=fitting_source, calibration_video_sha256=cal_hash, calibration_cases=len(calibration),
                    geometry=geometry, sites=list(sites), fresh_sites=list(fresh), paired_negatives=paired_negatives,
                    test_times=times, test_positions=positions,
                    fit_seed=FIT_SEED, test_seed=seed, budget=BUDGET, decoder_score_floor=None,
                    tree_iterations=ITERATIONS, nms=NMS, iou=0.50)
    with run_directory(output, "snl-external-grid-ranking", settings) as root:
        rankers, hashes, counts = fit_rankers(root, model, fitting, paired_negatives=paired_negatives)
        print("Ranker training", counts, flush=True)
        predictions, _, _ = rank(model, rankers, calibration)
        thresholds = PAIRED_THRESHOLDS if paired_negatives else CONFIDENCES
        trials = {name: table(calibration, output, thresholds) for name, output in predictions.items()}
        cap = prior["points"][MODE]["blank_outputs"]
        points = {name: choose(options, cap) if any(row["blank_outputs"] <= cap for row in options) else None
                  for name, options in trials.items()}
        frozen = dict(settings=settings, training_counts=counts, ranker_sha256=hashes, trials=trials,
                      calibration_blank_cap=cap, points=points)
        write_json(root / "frozen-selection.json", frozen)
        print("Frozen grid rankers", points, flush=True)
        if any(point is None for point in points.values()):
            write_json(root / "comparison.json", frozen | dict(status="no_common_feasible_operating_points"))
            return

        results, all_counts, all_candidates, all_responses, equal_counts = {}, [], [], [], []
        seen_videos = {cal_hash, fitting_source["video_sha256"]}
        for site, path in sites.items():
            if site in fresh:
                cases, receipt = official_test_bank(path, ffmpeg, sizes, motion, times, positions, seed)
            else:
                cases, receipt = test_cases(site, path, ffmpeg, sizes, motion, times=times, positions=positions, seed=seed)
            if receipt["video_sha256"] in seen_videos:
                raise ValueError(f"Duplicate source media appeared in held-out {site}")
            seen_videos.add(receipt["video_sha256"])
            predictions, candidates, responses = rank(model, rankers, cases, audit_score_response=paired_negatives)
            all_responses.extend(dict(site=site, **row) for row in responses)
            if paired_negatives:
                equal_counts.extend(dict(site=site, **row)
                                    for row in equal_output_rows(cases, predictions, points["heatmap"]["threshold"]))
            outcome = {}
            for name, raw in predictions.items():
                scores, rows = summarize(cases, raw, points[name]["threshold"])
                pool = {phase: sum(row[phase] for row in candidates[name]) for phase in ("before_nms", "after_nms")}
                outcome[name] = dict(metrics=scores, weak_candidate_coverage=pool)
                all_counts.extend(dict(site=site, model=name, **row) for row in rows)
                all_candidates.extend(dict(site=site, model=name, **row) for row in candidates[name])
                print(site, name, scores["iou50"], scores["by_depth"]["5"]["tp"], pool, flush=True)
            results[site] = dict(source=receipt, arms=outcome)
        pooled = pooled_results(points, all_counts, all_candidates)
        gates = {name: advances(row, pooled["heatmap"]) for name, row in pooled.items() if name != "heatmap"}
        comparison = frozen | dict(results=results, pooled=pooled, endpoint_gates=gates)
        if equal_counts:
            comparison["equal_output_pooled"] = pooled_results(points, equal_counts, all_candidates)
        if fresh:
            new_counts = [row for row in all_counts if row["site"] in fresh]
            new_candidates = [row for row in all_candidates if row["site"] in fresh]
            primary = pooled_results(points, new_counts, new_candidates)
            comparison["fresh_pooled"] = primary
            comparison["fresh_endpoint_gates"] = {name: advances(row, primary["heatmap"])
                                                   for name, row in primary.items() if name != "heatmap"}
            if paired_negatives:
                comparison["paired_vs_control_fresh_gate"] = advances(primary["paired_background"], primary["background_control"])
                new_equal = [row for row in equal_counts if row["site"] in fresh]
                comparison["equal_output_fresh_pooled"] = pooled_results(points, new_equal, new_candidates)
        write_json(root / "comparison.json", comparison)
        write_csv(root / "case-rows.csv", all_counts)
        write_csv(root / "candidate-rows.csv", all_candidates)
        if all_responses:
            write_csv(root / "score-response-rows.csv", all_responses)
        if equal_counts:
            write_csv(root / "equal-output-rows.csv", equal_counts)
        print(pooled, gates, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--paired-negatives", action="store_true")
    parser.add_argument("--fresh-video", type=Path, action="append", default=[])
    args = parser.parse_args()
    cv2.setNumThreads(3)
    torch.set_num_threads(4)
    output = args.output or EXTERNAL / ("injected-weak-grid-paired" if args.paired_negatives else "injected-weak-grid-ranking")
    evaluate(output.resolve(), find_ffmpeg(args.ffmpeg), paired_negatives=args.paired_negatives, fresh_paths=args.fresh_video)
