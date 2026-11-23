"""Compare global dense decoding with fixed spatial quotas on the same weak-shadow cases."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from experiment import quality
from infer_video import find_ffmpeg, torch
from snl_center import decode_center_map, predict_views
from snl_detector import load_trained, nonmax
from snl_external_shadow_adaptation import fit_samples
from snl_external_shadow_benchmark import metrics, summarize
from snl_external_weak_cues import SOURCES, choose, coded, table, test_cases, training_priors
from snl_external_weak_curriculum import MEXICO_CITY, MODE
from snl_inputs import read_json, sha_file, write_csv, write_json
from snl_qatar_pilot import EXTERNAL
from videosar.data import run_directory
from videosar.snl_causal import box_iou


BUDGET = 1024
FLOOR = 0.005
GRIDS = {"global": 0, "quota4": 4, "quota8": 8}
CURRICULUM = EXTERNAL / "injected-weak-curriculum"


def quota_logits(logits, grid):
    heat = logits[0].sigmoid()
    height, width = heat.shape
    if height % grid or width % grid or BUDGET % grid**2:
        raise ValueError("Forecast heat map and budget must divide into the spatial grid")
    block_h, block_w = height // grid, width // grid
    blocks = heat.reshape(grid, block_h, grid, block_w).transpose(1, 2).reshape(grid**2, -1)
    score, index = blocks.topk(BUDGET // grid**2, dim=1)
    keep = torch.zeros_like(blocks, dtype=torch.bool)
    keep.scatter_(1, index, score >= FLOOR)
    keep = keep.reshape(grid, grid, block_h, block_w).transpose(1, 2).reshape(-1)
    if remainder := BUDGET - int(keep.sum()):
        score, index = heat.flatten().masked_fill(keep, -1).topk(remainder)
        keep[index[score >= FLOOR]] = True
    result = logits.clone()
    result[0].masked_fill_(~keep.reshape(height, width), -torch.inf)
    return result


@torch.inference_mode()
def decode(model, cases, modes):
    result = {name: [] for name in modes}
    device = next(model.parameters()).device
    for _, image, _ in coded(cases, MODE):
        height, width = image.shape[-2:]
        model_input = torch.from_numpy(image).to(device).float()[None] / 255
        logits = predict_views(model, model_input)[0]
        for name in modes:
            grid = GRIDS[name]
            chosen = quota_logits(logits, grid) if grid else logits
            boxes, scores = decode_center_map(model, chosen, height, width)
            result[name].append((boxes, scores))
    if len(result) > 1 and len({tuple(len(row[0]) for row in raw) for raw in result.values()}) != 1:
        raise ValueError("Compared decoders returned different candidate counts")
    return result


def coverage(cases, candidates, after_nms):
    rows = []
    for (spec, _, truth), (before, _), (after, _) in zip(cases, candidates, after_nms, strict=True):
        if spec["injected_depth"] == 5:
            rows.append(dict(spec, candidates=len(before), nms_candidates=len(after),
                             geometric_before_nms=int((box_iou(before, truth) >= 0.50).any()),
                             geometric_after_nms=int((box_iou(after, truth) >= 0.50).any())))
    totals = {key: sum(row[key] for row in rows) for key in
              ("candidates", "nms_candidates", "geometric_before_nms", "geometric_after_nms")}
    return totals, rows


def evaluate(output, ffmpeg, cases_run):
    source_path = cases_run / "comparison.json"
    source = read_json(source_path)
    checkpoint = source["checkpoints"]["balanced"]
    config = checkpoint["config"] | {"dense_candidate_limit": BUDGET}
    model = load_trained(Path(checkpoint["folder"]), config)
    sizes, displacement = training_priors(ffmpeg)
    _, calibration, video_hash = fit_samples(sizes, displacement, ffmpeg)
    if video_hash != source["settings"]["training_source_sha256"]:
        raise ValueError("The calibration source changed")
    case_settings = source["settings"]
    sources = SOURCES | {"MexicoCity": MEXICO_CITY}
    settings = dict(dense_budget=BUDGET, decoder_floor=FLOOR, nms=0.35, matching_iou=0.50, grids=GRIDS,
                    checkpoint=config, calibration_video_sha256=video_hash,
                    case_manifest_sha256=sha_file(source_path), heldout_samples_reused=True)
    with run_directory(output, "snl-external-candidate-quota", settings) as root:
        cap = source["calibration_blank_cap"]
        preds = decode(model, calibration, GRIDS)
        trials, points, proposed = {}, {}, {}
        for name, candidate in preds.items():
            nms = [nonmax(boxes, scores, 0.35) for boxes, scores in candidate]
            trials[name] = table(calibration, nms)
            points[name] = choose(trials[name], cap)
            proposed[name] = coverage(calibration, candidate, nms)[0]
        quota = max((name for name in GRIDS if name != "global"), key=lambda name:
                    (points[name]["weak"]["tp"], quality(points[name]["iou50"]), proposed[name]["geometric_after_nms"]))
        freeze = dict(settings=settings, selected_quota=quota, calibration_blank_cap=cap, points=points,
                      calibration_geometry=proposed, trials=trials)
        write_json(root / "frozen-selection.json", freeze)
        print("Frozen spatial quota", quota, points, flush=True)

        results, all_counts, all_geometry = {}, [], []
        for site, path in sources.items():
            cases, receipt = test_cases(site, path, ffmpeg, sizes, displacement, times=tuple(case_settings["test_times"]),
                                        positions=tuple(case_settings["test_positions"]), seed=case_settings["test_seed"])
            if source["results"][site]["source"] != receipt:
                raise ValueError("Held-out samples changed")
            results[site] = {}
            for name, raw in decode(model, cases, ("global", quota)).items():
                nms = [nonmax(boxes, scores, 0.35) for boxes, scores in raw]
                score, counts = summarize(cases, nms, points[name]["threshold"])
                support, geometry = coverage(cases, raw, nms)
                all_counts.extend(dict(site=site, method=name, **row) for row in counts)
                all_geometry.extend(dict(site=site, method=name, **row) for row in geometry)
                results[site][name] = dict(metrics=score, weak_coverage=support)
                print(site, name, score["iou50"], support, flush=True)
        pooled = {}
        for name in ("global", quota):
            counts = [row for row in all_counts if row["method"] == name]
            pooled[name] = dict(iou50=metrics(counts), weak=metrics([row for row in counts if row["injected_depth"] == 5]),
                                blank_outputs=sum(row["fp"] for row in counts if row["injected_depth"] == 0))
        write_json(root / "comparison.json", freeze | dict(results=results, pooled=pooled))
        write_csv(root / "case-rows.csv", all_counts)
        write_csv(root / "candidate-rows.csv", all_geometry)


@torch.inference_mode()
def audit_heatmap(output, ffmpeg, cases_run):
    comparison = read_json(output / "comparison.json")
    source_path = cases_run / "comparison.json"
    curriculum = read_json(source_path)
    if sha_file(source_path) != comparison["settings"]["case_manifest_sha256"]:
        raise ValueError("The held-out sample manifest changed")
    frozen = curriculum["checkpoints"]["balanced"]
    config = comparison["settings"]["checkpoint"]
    model = load_trained(Path(frozen["folder"]), config)
    device = next(model.parameters()).device
    sizes, displacement = training_priors(ffmpeg)
    settings = curriculum["settings"]
    results = {}
    for site, path in (SOURCES | {"MexicoCity": MEXICO_CITY}).items():
        cases, receipt = test_cases(site, path, ffmpeg, sizes, displacement, times=tuple(settings["test_times"]),
                                    positions=tuple(settings["test_positions"]), seed=settings["test_seed"])
        if receipt != curriculum["results"][site]["source"]:
            raise ValueError(f"The {site} media receipt changed")
        rows = []
        for spec, image, truth in coded([case for case in cases if case[0]["injected_depth"] == 5], MODE):
            height, width = image.shape[-2:]
            tensor = torch.from_numpy(image).to(device).float()[None] / 255
            heatmap = predict_views(model, tensor)[0]
            model.dense_candidate_limit = BUDGET
            truncated, _ = decode_center_map(model, heatmap, height, width)
            model.dense_candidate_limit = int(heatmap[0].numel())
            all_above_floor, _ = decode_center_map(model, heatmap, height, width)
            full = heatmap.clone()
            full[0].zero_()
            all_boxes, _ = decode_center_map(model, full, height, width)
            candidates = dict(top1024=truncated, entire_grid_above_floor=all_above_floor, entire_grid_ignoring_score=all_boxes)
            rows.append(dict(spec, **{key: int((box_iou(value, truth) >= 0.50).any()) for key, value in candidates.items()}))
        totals = {key: sum(row[key] for row in rows) for key in candidates}
        if totals["top1024"] != comparison["results"][site]["global"]["weak_coverage"]["geometric_before_nms"]:
            raise ValueError(f"The {site} top-k coverage failed to reproduce")
        results[site] = dict(totals=totals, rows=rows)
        print(site, totals, flush=True)
    write_json(output / "weak-heatmap-audit.json", dict(score_floor=FLOOR, matching_iou=0.50, source=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXTERNAL / "injected-weak-spatial-quota")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--cases-run", type=Path, default=CURRICULUM)
    parser.add_argument("--audit-heatmap", action="store_true", help="Audit the full decoded heat grid in an existing run")
    args = parser.parse_args()
    cv2.setNumThreads(3)
    torch.set_num_threads(4)
    run = audit_heatmap if args.audit_heatmap else evaluate
    run(args.output.resolve(), find_ffmpeg(args.ffmpeg), args.cases_run.resolve())
