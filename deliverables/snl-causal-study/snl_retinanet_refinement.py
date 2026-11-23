"""Train and test a neural box refiner on out-of-time RetinaNet candidates."""
import argparse
from itertools import islice, product
from pathlib import Path

import numpy as np

from audit_detection import paired_block_f1
from experiment import filter_by_config, load_candidates, quality, raw_candidates, select_operation, summarize, with_nms
from snl_bytetrack import REFERENCE
from snl_detector import backend, torch, train_model
from snl_inputs import DEFAULT_OUTPUT, PROFILES, feature_cache, load_gray, load_labels, read_json, sha_file
from snl_inputs import write_csv, write_json
from snl_patch import PatchScore, choose, evaluate_neural, train, training_patches
from videosar.snl_causal import CausalFeatures, matched_boxes


def early_retinanet(folder, features, labels):
    registry = folder / "early-retinanet.json"
    ids = list(labels)
    if registry.is_file():
        receipt = read_json(registry)
        if receipt["training_ids"] != ids:
            raise ValueError("Cached early RetinaNet uses a different split")
        if receipt["checkpoints"][-1]["epoch"] == 8:
            return receipt["checkpoints"][-1]
    training = train_model(folder, features, labels, "temporal", 3, checkpoint_epochs=(4, 8, 12),
                           architecture="retinanet", input_profile="registered")
    checkpoints = []
    for checkpoint in islice(training, 2):
        checkpoints.append(checkpoint)
        write_json(registry, {"training_ids": ids, "checkpoints": checkpoints})
    return checkpoints[-1]


def collect_model(folder, sources, features, truth, metadata):
    registry = folder / "retina-refinement-fit.json"
    identifiers = list(truth)
    if registry.exists():
        artifact = read_json(registry)
        if artifact["fit_ids"] != identifiers or artifact["early_detectors"] != metadata:
            raise ValueError("Cached refiner uses different earlier detectors or fitting frames")
        path = folder / artifact["checkpoint"]["path"]
        if sha_file(path) != artifact["checkpoint"]["sha256"]:
            raise ValueError("Refiner checkpoint checksum differs")
        model = PatchScore().to(backend())
        model.load_state_dict(torch.load(path, map_location=backend(), weights_only=True))
        return model.eval(), artifact
    records = training_patches(sources, features, truth, identifiers, source_cap=None, pool_cap=None)
    model, checkpoint = train(folder, records)
    artifact = {"fit_ids": identifiers, "early_detectors": metadata, "checkpoint": checkpoint,
                "patches": len(records[1]), "positive_patches": int(records[1].sum())}
    write_json(registry, artifact)
    return model, artifact


def evaluate(output, ffmpeg):
    if backend().type == "cuda":
        torch.set_float32_matmul_precision("high")
    previous = DEFAULT_OUTPUT / "early-stacked"
    fit_folder = output / "retinanet-refinement"
    fit_folder.mkdir(exist_ok=True)
    early_ids = PROFILES["original"][0]
    train_ids, cal_ids, dev_ids = PROFILES["prospective"]
    fit_ids = tuple(i for i in train_ids if i >= 242)
    if max(early_ids) >= min(fit_ids) or max(fit_ids) >= min(cal_ids):
        raise ValueError("Source detector training, refiner fitting and calibration must be chronological")
    if read_json(previous / "split-protocol.json")["train"] != list(early_ids):
        raise ValueError("Early center model uses a different split")
    gray = load_gray(output, ffmpeg)
    early, fitting, calibration, development = [load_labels(output, ids, gray)[0]
                                                for ids in (early_ids, fit_ids, cal_ids, dev_ids)]
    maps = {profile: feature_cache(output, gray, max(dev_ids), CausalFeatures(profile))
            for profile in ("registered", "stacked")}
    reference = read_json(output / REFERENCE)
    retina = max((read_json(path) for path in output.glob("decision-retinanet-registered-temporal-3x-epoch*.json")),
                 key=lambda row: quality(row["validation"]))
    old_center = next(row for row in read_json(previous / "checkpoints-center-stacked.json") if row["epoch"] == 36)
    old_center |= {"contrast_gain": reference["config"]["contrast_gain"]}
    old_retina = early_retinanet(fit_folder, maps["registered"], early)
    earlier = (old_center, old_retina)
    fit_sources = [raw_candidates(location, model, maps[model["input_profile"]], fit_ids, "retina-refiner-fit")
                   for location, model in zip((previous, fit_folder), earlier, strict=True)]
    model, artifact = collect_model(fit_folder, fit_sources, maps["registered"], fitting, list(earlier))

    configs = (reference["config"], retina["config"])
    raw = {}
    for split in ("validation", "development"):
        sources = [load_candidates(output, config, split) for config in configs]
        indices = sorted(set(sources[0]) & set(sources[1]))
        raw[split] = evaluate_neural(model, sources, maps["registered"], indices, source_cap=None, pool_cap=None)

    trials = []
    for share, refine in product((0.0, 0.25, 0.50, 0.75, 1.0), (False, True)):
        point, summary, _ = select_operation(choose(raw["validation"], share, refine), calibration)
        trials.append({"source_score_weight": share, "refine_boxes": refine, "operation": point, "calibration": summary})
    selected = max(trials, key=lambda row: quality(row["calibration"]))
    point = selected["operation"]
    reranked = choose(raw["development"], selected["source_score_weight"], selected["refine_boxes"])
    result, rows = summarize(filter_by_config(with_nms(reranked, point["nms"]), point), development)
    base_pool = load_candidates(output, configs[0], "development")
    _, base_rows = summarize(filter_by_config(with_nms(base_pool, configs[0]["nms"]), configs[0]), development)
    ceilings = {}
    for refine in (False, True):
        candidates = choose(raw["development"], 0.0, refine)
        ceilings["refined" if refine else "unrefined"] = sum(
            len(matched_boxes(candidates[index][0], truth)) for index, truth in development.items())
    report = {"refiner": artifact, "calibration_trials": trials, "selected": selected, "development": result,
              "reference_development": reference["development"], "candidate_ceiling_iou50": ceilings,
              "paired_f1_vs_reference": paired_block_f1(rows, base_rows), "detectors": list(configs)}
    write_json(output / "retinanet-refinement-comparison.json", report)
    write_csv(output / "retinanet-refinement-development.csv", rows)
    print(selected, result, ceilings, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    evaluate(args.output.resolve(), args.ffmpeg)
