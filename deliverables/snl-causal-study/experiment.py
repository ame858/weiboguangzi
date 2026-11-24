"""Develop, freeze, and evaluate the causal SNL detector."""
import argparse
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from snl_inputs import (
    BLIND, DEFAULT_OUTPUT, HEIGHT, PROFILES, ROOT, TOTAL,
    VIDEO, WIDTH, feature_cache, load_gray, load_labels, read_json, sha_file,
    write_csv, write_json,
)
from snl_protocol import GATE, causal_prefix_length, causal_receipt, exclusive_run, freeze_selection, passed, verify_render

sys.path.insert(0, str(ROOT))

from snl_detector import (
    DENSE_ARCHITECTURES, FULL_FRAME_ARCHITECTURES, TILE, backend, load_trained, map_tile_boxes, nonmax, predict_frame,
    target_in_tile, torch, train_model,
)
from videosar.snl_causal import (
    CausalFeatures, TrackFilter, component_boxes, detection_counts, detection_metrics,
)

CONFIDENCES = np.round(np.arange(0.05, 0.96, 0.05), 2)
NMS = (0.35, 0.50)
YOLO_PHASES = ("yolo11", "yolo11-scratch", "yolo11-temporal", "yolo11-p2", "yolo11-coco-transfer",
               "yolo11-sived", "yolo11-sived-inverted", "yolo11-sived-160", "yolo11-coco-transfer-temporal",
               "yolo11-sived-temporal", "yolo11-sived-inverted-temporal", "yolo11-sived-160-temporal")
CENTER_INPUTS = {
    "center": "raw", "center-domain": "raw", "center-motion": "raw", "center-iou": "raw",
    "center-recent": "raw", "center-stable-labels": "raw", "center-soft-labels": "raw",
    "center-normalized": "standardized", "center-registered": "registered",
    "center-histogram": "histogram", "center-histogram-registered": "histogram-registered",
    "center-stacked": "stacked", "center-stacked-soft": "stacked", "center-stacked-motion": "stacked",
    "center-stacked-recent": "stacked", "center-space-depth": "stacked", "center-shared-temporal": "stacked",
    "center-recurrent-temporal": "stacked", "center-mixed-temporal": "stacked",
    "center-peak-ranking": "stacked",
    "center-task-aligned": "stacked",
    "center-line-context": "stacked", "center-weak-shadows": "stacked", "center-background": "background-history",
    "center-quality-focal": "stacked", "center-stacked-normalized": "stacked-standardized",
    "center-stacked-long": "stacked-long", "center-weak-sampling": "stacked", "center-box-projection": "stacked",
    "center-local-gain": "stacked", "center-stacked-copy": "stacked",
    "center-stacked-fused": "stacked",
}
# Historical 28.06% operating point: seven-frame causal stack with local-contrast
# view fusion and three-frame history confirmation.
FUSED_VIEW = {"contrast_gain": 1.35, "track_support": 3}
def quality(result):
    return (passed(result), result["f1"] or 0, result["recall"] or 0, result["precision"] or 0)


def summarize(rows, targets):
    records = []
    for i, truth in targets.items():
        boxes, _, _ = rows[i]
        tp, fp, fn = detection_counts(boxes, truth)
        records.append({"annotation_id": i, "video_frame": i + 148, "tp": tp, "fp": fp, "fn": fn})
    counts = [sum(r[k] for r in records) for k in ("tp", "fp", "fn")]
    return detection_metrics(counts, len(records)), records


def legacy_probability(gray, ids):
    from videosar.unet import load_weights, logits, normalize, runtime

    _, smp, _ = runtime()
    path = ROOT / "deliverables/proposal/models/unet_temporal_post.json"
    cfg = read_json(path)
    device = torch.device("cpu")
    model = load_weights(torch, smp, path.with_name(cfg["checkpoint"]), cfg, device).eval()
    power = np.log1p(gray[:max(ids)-1].astype(np.float32) / 49.80579350265945)
    result = {}
    order = list(ids)
    with torch.inference_mode():
        for k in range(0, len(order), 3):
            chunk = order[k:k+3]
            x = np.stack([np.stack((power[i-2], power[i-2] - power[i-8])) for i in chunk])
            probability = logits(torch, model, torch.from_numpy(normalize(x, cfg["mean"], cfg["std"]))).sigmoid().numpy()
            result.update(zip(chunk, probability, strict=True))
    return result, cfg


def baseline_maps(gray, ids):
    wanted = set(ids)
    history = CausalFeatures()
    maps = {"background_difference": {}, "multi_interval": {}}
    for i in range(2, max(ids) + 1):
        _, old, fused = history.update(gray[i - 2])
        if i in wanted:
            maps["background_difference"][i] = old
            maps["multi_interval"][i] = fused
    return maps


def baseline_predictions(maps, threshold, area, *, closing=False):
    from videosar.study import cleanup

    output = {}
    for i, score in maps.items():
        if closing:
            mask = cleanup(score[None], threshold, area, True)[0]
            boxes, scores = component_boxes(mask, 1, area)
        else:
            boxes, scores = component_boxes(score, threshold, area)
        output[i] = (boxes, scores, np.full(len(boxes), -1))
    return output


def run_baselines(output, gray, validation, development):
    maps = baseline_maps(gray, (*validation, *development))
    ranges = {"background_difference": (0.15, 0.25, 0.35, 0.45, 0.55),
              "multi_interval": (0.008, 0.015, 0.025, 0.04, 0.06, 0.09, 0.12)}
    grid, selected = [], []
    for method, values in maps.items():
        rows = []
        for threshold in ranges[method]:
            for area in (5, 12, 24, 32):
                pred = baseline_predictions({i: values[i] for i in validation}, threshold, area)
                score, _ = summarize(pred, validation)
                rows.append(dict(method=method, threshold=threshold, min_area=area, closing=False, **score))
        grid.extend(rows)
        best = max(rows, key=quality)
        config = {key: best[key] for key in ("method", "threshold", "min_area", "closing")}
        pred = baseline_predictions({i: values[i] for i in development},
                                    config["threshold"], config["min_area"])
        dev, _ = summarize(pred, development)
        selected.append({"config": config, "validation": best, "development": dev})
        print(method, "development", dev, flush=True)
    legacy, settings = legacy_probability(gray, (*validation, *development))
    config = {"method": "legacy_unet", **{k: settings[k] for k in ("threshold", "min_area", "closing")}}
    predictions = baseline_predictions(legacy, config["threshold"], config["min_area"], closing=config["closing"])
    val, _ = summarize(predictions, validation)
    dev, _ = summarize(predictions, development)
    selected.append({"config": config, "validation": val, "development": dev})
    print("legacy_unet development", dev, flush=True)
    write_csv(output / "baseline-validation.csv", grid)
    write_json(output / "baselines.json", selected)


def warm_ids(targets):
    ids = set()
    for i in targets:
        ids.update(range(max(2, i - 6), i + 1))
    return sorted(ids)


def candidate_arrays(payload):
    return {int(i): (np.asarray(value[0], dtype=np.float32).reshape(-1, 4),
                     np.asarray(value[1], dtype=np.float32)) for i, value in payload["predictions"].items()}


def candidate_file(output, config, scope, device):
    variant = f"-{device}" if device is not None and device != config.get("device", device) else ""
    return output / f"raw-{model_key(config, predictions_only=True)}-{scope}{variant}.json"


def load_candidates(output, config, scope):
    device = config.get("evaluation_device", config.get("device"))
    payload = read_json(candidate_file(output, config, scope, device))
    if payload["checkpoint_sha256"] != config["sha256"]:
        raise ValueError(f"{config['path']}: cached candidates differ from the selected model")
    return candidate_arrays(payload)


def raw_candidates(output, config, features, ids, scope):
    device = backend().type
    snapshot = candidate_file(output, config, scope, device)
    fingerprint = f"{device}:{torch.__version__}:" + sha_file(Path(__file__).with_name("snl_detector.py"))
    fingerprint += ":" + sha_file(ROOT / "videosar/snl_causal.py")
    if config.get("architecture", "").startswith("radar-center"):
        fingerprint += ":" + sha_file(Path(__file__).with_name("snl_center.py"))
    if config.get("learned_background"):
        fingerprint += ":" + sha_file(Path(__file__).with_name("snl_background.py"))
    if config.get("architecture") == "ultralytics-yolo11":
        fingerprint += ":" + sha_file(Path(__file__).with_name("snl_yolo.py"))
    if snapshot.exists():
        cached = read_json(snapshot)
        if (cached["checkpoint_sha256"], cached["inference_sha256"], cached["ids"]) == (
                config["sha256"], fingerprint, list(ids)):
            return candidate_arrays(cached)
    model = load_trained(output, config)
    result = {}
    for k, i in enumerate(ids, 1):
        boxes, scores = predict_frame(model, features[i - 2], config["method"])
        result[i] = boxes, scores
        if k % 30 == 0:
            print(f"{Path(config['path']).stem} {scope}: {k}/{len(ids)} frames", flush=True)
    write_json(snapshot, {"checkpoint_sha256": config["sha256"], "inference_sha256": fingerprint,
                          "ids": list(ids), "predictions": {i: [b.tolist(), s.tolist()] for i, (b, s) in result.items()}})
    return result


def model_key(config, *, predictions_only=False):
    parts = Path(config["path"]).with_suffix("").parts
    views = (("tiled", "tiled"), ("flip_inference", "flip"), ("stitch_inference", "stitched"))
    suffix = [label for key, label in views if config.get(key)]
    if power := config.get("occupancy_power"):
        suffix.append(f"occupancy{power:g}")
    if gain := config.get("contrast_gain"):
        suffix.append(f"contrast{gain:g}")
    if limit := config.get("dense_candidate_limit"):
        suffix.append(f"dense{limit}")
    if (support := config.get("track_support")) and not predictions_only:
        suffix.append(f"support{support}")
    return "-".join((*parts, *suffix))


def with_nms(candidates, nms):
    return {i: nonmax(boxes, scores, nms) for i, (boxes, scores) in candidates.items()}


def filtered(candidates, threshold, low, *, confirmations=1):
    tracker = TrackFilter(threshold, low, confirmations=confirmations) if low is not None else None
    rows = {}
    for i, (boxes, scores) in candidates.items():
        if tracker is not None:
            rows[i] = tracker.update(i, boxes, scores)
        else:
            keep = scores >= threshold
            rows[i] = boxes[keep], scores[keep], np.full(keep.sum(), -1)
    return rows


def filter_by_config(candidates, config):
    return filtered(candidates, config["threshold"], config["low"], confirmations=config.get("track_support", 1))


def diagnostic_coverage(candidates, targets, limit=60):
    from scipy.optimize import linear_sum_assignment

    tally = np.zeros(3, dtype=int)
    near_centers = 0
    for i, truth in targets.items():
        boxes, scores = candidates[i]
        if len(boxes) > limit:
            boxes = boxes[np.argsort(-scores, kind="stable")[:limit]]
        tally += detection_counts(boxes, truth)
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        truth_centers = (truth[:, :2] + truth[:, 2:]) / 2
        near = np.linalg.norm(centers[:, None] - truth_centers[None, :], axis=-1) <= 5
        row, col = linear_sum_assignment(near, maximize=True)
        near_centers += int(near[row, col].sum())
    metrics = detection_metrics(tally, len(targets))
    coverage = metrics["recall"]
    center_coverage = near_centers / (metrics["tp"] + metrics["fn"]) if coverage is not None else None
    return coverage, center_coverage


def select_operation(candidates, targets, *, confirmations=1):
    grid = []
    for nms in NMS:
        after_nms = with_nms(candidates, nms)
        for threshold in CONFIDENCES:
            for low in (None, float(threshold * 0.5)):
                result, _ = summarize(filtered(after_nms, threshold, low, confirmations=confirmations), targets)
                grid.append(dict(threshold=float(threshold), low=low, nms=nms, **result))
    selected = max(grid, key=quality)
    return {k: selected[k] for k in ("threshold", "low", "nms")}, selected, grid


def train(output, features, train_labels, method, scale, hard=()):
    registry_path = output / f"checkpoints-{method}-{scale}x.json"
    registry = []
    for checkpoint in train_model(output, features, train_labels, method, scale, hard=hard):
        registry.append(checkpoint)
        write_json(registry_path, registry)


def records_for_fit(output, methods, scale):
    return [row for method in methods for row in read_json(output / f"checkpoints-{method}-{scale}x.json")]


def develop(output, features, validation, development, models, *, saved_predictions=False):
    val_ids, dev_ids = warm_ids(validation), warm_ids(development)
    grid, decisions = [], []
    for checkpoint in models:
        if saved_predictions:
            raw_val = load_candidates(output, checkpoint, "validation")
            raw_dev = load_candidates(output, checkpoint, "development")
        else:
            checkpoint = checkpoint | {"evaluation_device": backend().type, "evaluation_torch": str(torch.__version__)}
            raw_val = raw_candidates(output, checkpoint, features, val_ids, "validation")
            raw_dev = raw_candidates(output, checkpoint, features, dev_ids, "development")
        op, val, sweep = select_operation(raw_val, validation, confirmations=checkpoint.get("track_support", 1))
        grid.extend(dict(checkpoint=checkpoint["path"], **r) for r in sweep)
        merged = with_nms(raw_dev, op["nms"])
        predicted = filter_by_config(merged, checkpoint | op)
        dev, per_frame = summarize(predicted, development)
        coverage, centers = diagnostic_coverage(merged, development)
        row = {"config": checkpoint | op, "validation": val, "development": dev,
               "development_recall_at_60": coverage, "development_center_coverage_at_60": centers}
        decisions.append(row)
        write_json(output / f"decision-{model_key(checkpoint)}.json", row)
        write_csv(output / f"development-{model_key(checkpoint)}.csv", per_frame)
        write_csv(output / f"validation-{model_key(checkpoint)}.csv", sweep)
        print(checkpoint["path"], "validation", val["precision"], val["recall"],
              "development", dev["precision"], dev["recall"],
              "coverage", row["development_recall_at_60"], flush=True)
    write_csv(output / f"detector-validation-{model_key(models[0])}.csv", grid)
    return decisions


def mine_negatives(output, features, model, labels):
    from videosar.snl_causal import box_iou

    subset = tuple(labels)[::3]
    collection = raw_candidates(output, model, features, subset, "hard-negative-mining")
    negative = []
    for index, (boxes, scores) in collection.items():
        truth = labels[index]
        overlap = box_iou(boxes, truth).max(axis=1, initial=0)
        candidates = np.flatnonzero(overlap < 0.10)
        for j in candidates[np.argsort(-scores[candidates], kind="stable")]:
            ctr = 0.5 * (boxes[j, :2] + boxes[j, 2:])
            left, top = np.clip(ctr.astype(int) - TILE // 2, 0, [WIDTH-TILE, HEIGHT-TILE])
            if not len(target_in_tile(truth, left, top)):
                negative.append((index, int(left), int(top)))
                break
    write_json(output / "hard-negative-windows.json", negative)
    return negative


def propose(output):
    all_decisions = [read_json(p) for p in sorted(output.glob("decision-*.json"))]
    baselines = output / "baselines.json"
    partition_file = output / "split-protocol.json"
    profile = read_json(partition_file)["profile"] if partition_file.is_file() else "original"
    if not baselines.is_file() and profile == "original":
        baselines = DEFAULT_OUTPUT / baselines.name
    if baselines.is_file():
        all_decisions += read_json(baselines)
    tried = [read_json(path) for name in BLIND for path in DEFAULT_OUTPUT.glob(f"**/blind-{name}.json")]
    rejected = [result["config"] for result in tried if not result["passed"]]
    reached = [r for r in all_decisions if passed(r["validation"]) and passed(r["development"])
               and r["config"] not in rejected and r["config"]["method"] != "legacy_unet"]
    best = max(reached or all_decisions, key=lambda row: (quality(row["development"]), quality(row["validation"])))
    report = {"status": "ready" if reached else "continue_development", "best": best,
              "candidates": all_decisions, "test_pass_threshold": GATE}
    write_json(output / "development-status.json", report)
    print("Development status:", report["status"], best, flush=True)
    return report


def process(output, config, gray, labels, name):
    method = config["method"]
    horizon = max(labels)
    if method in ("gray", "temporal"):
        profile = config.get("input_profile", "raw")
        features = feature_cache(output, gray, horizon, CausalFeatures(profile)) if method == "temporal" else gray[:, None]
        raw = raw_candidates(output, config, features, warm_ids(labels), name)
        return filter_by_config(with_nms(raw, config["nms"]), config)
    if method == "dense-radar-statistics":
        from snl_dense import predictions

        raw = predictions(output, config, gray, labels)
        return filter_by_config(with_nms(raw, config["nms"]), config)
    if method == "legacy_unet":
        raise ValueError("The unchanged legacy reference is diagnostic; use the full-prefix causal replacement")
    maps = baseline_maps(gray, labels)[method]
    return baseline_predictions(maps, config["threshold"], config["min_area"])


def blind(output, gray, key):
    config = freeze_selection(output, key)["selected"]["config"]
    require_evaluation_device(config)
    ids = tuple(i for block in BLIND[key].values() for i in block)
    start = perf_counter()
    all_predictions = process(output, config, gray, ids, f"blind-{key}")
    labels, audit, _ = load_labels(output, ids, gray)
    result, per_frame = summarize(all_predictions, labels)
    block_results = {segment: summarize(all_predictions, {i: labels[i] for i in frames})[0]
                     for segment, frames in BLIND[key].items()}
    write_csv(output / f"blind-{key}-frames.csv", per_frame)
    write_json(output / f"blind-{key}-alignment.json", audit)
    write_json(output / f"blind-{key}.json", {"passed": passed(result), "result": result, "blocks": block_results,
                                             "seconds": perf_counter() - start, "config": config})
    print(key, "blind", result, flush=True)


def check_causality(output, gray):
    decision = read_json(output / "development-status.json")["best"]
    config = decision["config"]
    require_evaluation_device(config)
    method = config["method"]
    model = load_trained(output, config) if method in ("gray", "temporal") else None
    prefix = causal_prefix_length(config)
    sequence = gray[205:205+prefix+6, 95:95+TILE, 140:140+TILE]
    perturbed = np.asarray(sequence).copy()
    perturbed[prefix:] = 255 - perturbed[prefix:]

    def scan(frames):
        if method == "dense-radar-statistics":
            from snl_dense import DenseDetector

            causal = DenseDetector(output, config)
        else:
            causal = CausalFeatures(config.get("input_profile", "raw"))
        tracker = TrackFilter(config["threshold"], config.get("low") or config["threshold"],
                              confirmations=config.get("track_support", 1))
        prefix_outputs = []
        for i, frame in enumerate(frames):
            if method == "dense-radar-statistics":
                native = causal.update(frame, used=i < prefix)
            else:
                channels, old, fused = causal.update(frame)
                if i < prefix:
                    inputs = np.rint(channels * 255).astype(np.uint8)
                    if model is None:
                        score = old if method == "background_difference" else fused
                        native = component_boxes(score, config["threshold"], config["min_area"])
                    else:
                        native = predict_frame(model, inputs, method)
            if i < prefix:
                results = nonmax(*native, config["nms"]) if "nms" in config else native
                prefix_outputs.append(tracker.update(i, *results))
        return prefix_outputs

    expected = scan(sequence[:prefix])
    for future in (sequence, perturbed):
        for a, b in zip(expected, scan(future), strict=True):
            for left, right in zip(a, b, strict=True):
                np.testing.assert_array_equal(left, right)
    same_box = np.array([[158, 151, 174, 174]], dtype=np.float32)
    strength = np.array([0.9], dtype=np.float32)
    a, score_a = map_tile_boxes(same_box, strength, 0, 0, WIDTH, HEIGHT)
    b, score_b = map_tile_boxes(same_box - [144, 144, 144, 144], strength, 144, 144, WIDTH, HEIGHT)
    merged, _ = nonmax(np.concatenate((a, b)).astype(np.float32), np.concatenate((score_a, score_b)), 0.35)
    np.testing.assert_array_equal(merged, same_box)
    write_json(output / "causality-check.json", causal_receipt(config))
    print("Causal prefix and overlap remapping passed.", flush=True)


def render(output, ffmpeg, gray, key):
    receipt = read_json(output / f"blind-{key}.json")
    if not receipt["passed"]:
        raise ValueError("The final video requires a passed blind evaluation")
    config = receipt["config"]
    verify_render(output, key, config)
    require_evaluation_device(config)
    method = config["method"]
    ids = range(2, TOTAL + 2)
    if method in ("gray", "temporal"):
        profile = config.get("input_profile", "raw")
        features = feature_cache(output, gray, TOTAL+1, CausalFeatures(profile)) if method == "temporal" else gray[:, None]
        model = load_trained(output, config)
    elif method == "dense-radar-statistics":
        from snl_dense import DenseDetector

        features = None
        scan = DenseDetector(output, config)
    else:
        features = None
        scan = CausalFeatures()
    tracker = None
    if config.get("low") is not None:
        tracker = TrackFilter(config["threshold"], config["low"], confirmations=config.get("track_support", 1))
    destination = output / "eubank-causal-overlay.mp4"
    temporary = destination.with_suffix(".partial.mp4")
    command = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", "30000/1001", "-i", "-",
               "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(temporary)]
    rows = []
    t0 = perf_counter()
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for i in ids:
            if features is not None:
                boxes, scores = nonmax(*predict_frame(model, features[i-2], method), config["nms"])
            elif method == "dense-radar-statistics":
                boxes, scores = nonmax(*scan.update(gray[i-2]), config["nms"])
            else:
                _, old, fused = scan.update(gray[i-2])
                score = old if method == "background_difference" else fused
                boxes, scores = component_boxes(score, config["threshold"], config["min_area"])
            if tracker is not None:
                boxes, scores, tracks = tracker.update(i, boxes, scores)
            else:
                take = scores >= config["threshold"]
                boxes, scores = boxes[take], scores[take]
                tracks = np.full(len(boxes), -1)
            image = Image.fromarray(gray[i-2]).convert("RGB")
            draw = ImageDraw.Draw(image)
            for box, score, track in zip(boxes, scores, tracks, strict=True):
                draw.rectangle(tuple(box), outline="#57f1aa", width=2)
                rows.append(dict(annotation_id=i, video_frame=i+148, x0=float(box[0]), y0=float(box[1]),
                                 x1=float(box[2]), y1=float(box[3]), score=float(score),
                                 track_id=int(track) if track > 0 else ""))
            draw.text((12, 12), f"SNL frame {i+149} | {len(boxes)} detections", fill="white",
                      stroke_width=2, stroke_fill="black")
            encoder.stdin.write(image.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("FFmpeg could not encode the detection video")
    temporary.replace(destination)
    write_csv(output / "video-predictions.csv", rows,
              ("annotation_id", "video_frame", "x0", "y0", "x1", "y1", "score", "track_id"))
    write_json(output / "video-run.json", {"frames": TOTAL, "seconds": perf_counter()-t0, "config": config})


def require_evaluation_device(config):
    expected = config.get("evaluation_device", config.get("device"))
    if expected is not None and backend().type != expected:
        raise ValueError(f"The selected run was evaluated on {expected}; reproduce it on the same backend")
    if config.get("evaluation_torch", str(torch.__version__)) != str(torch.__version__):
        raise ValueError("Reproduce the selected evaluation with the recorded PyTorch build")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("verify", "baselines", "train", "develop", "iterate",
                                          *CENTER_INPUTS, "center-tiled", "center-full",
                                          "center-map-ensemble", "center-spatial-views", "center-occupancy",
                                          "center-contrast-views", "center-stitch", "track-support", *DENSE_ARCHITECTURES,
                                          *FULL_FRAME_ARCHITECTURES, *YOLO_PHASES,
                                          "causality", "freeze", "blind", "render"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="original")
    parser.add_argument("--method", choices=("gray", "temporal"), default="temporal")
    parser.add_argument("--scale", choices=(2, 3), default=2, type=int)
    parser.add_argument("--blind-set", choices=tuple(BLIND), default="first")
    from snl_backbones import BACKBONES

    parser.add_argument("--backbone", choices=BACKBONES, default="tiny",
                        help="CenterDetector feature extractor (SNL architecture upgrade)")
    parser.add_argument("--epoch-budget", type=int, default=None,
                        help="Override the center detector training epoch count")
    parser.add_argument("--random-backbone", action="store_true",
                        help="Initialize the selected backbone randomly instead of with ImageNet weights")
    parser.add_argument("--checkpoint-interval", type=int, default=12,
                        help="Checkpoint cadence for center training (6 reduces calibration selection bias)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_ids, validate_ids, dev_ids = PROFILES[args.profile]
    phase_protocol = output / "split-protocol.json"
    partitions = {"profile": args.profile, "train": list(train_ids), "validation": list(validate_ids),
                  "development": list(dev_ids), "blind": {name: {key: list(ids) for key, ids in value.items()}
                                                         for name, value in BLIND.items()}}
    if phase_protocol.is_file():
        if read_json(phase_protocol) != partitions:
            raise ValueError("This output directory already belongs to a different split")
    else:
        write_json(phase_protocol, partitions)
    if args.phase == "freeze":
        with exclusive_run(args.blind_set):
            freeze_selection(output, args.blind_set, create=True)
        print(f"Frozen the qualified configuration for {args.blind_set} blind scoring", flush=True)
        return
    gray = load_gray(output, args.ffmpeg)
    if args.phase == "causality":
        check_causality(output, gray)
        return
    if args.phase in ("blind", "render"):
        if args.phase == "blind":
            with exclusive_run(args.blind_set):
                blind(output, gray, args.blind_set)
        else:
            render(output, args.ffmpeg, gray, args.blind_set)
        return
    if args.phase == "train":
        labels, alignment, _ = load_labels(output, train_ids, gray)
        write_json(output / f"train-alignment-{args.method}.json", alignment)
        features = feature_cache(output, gray, max(dev_ids), CausalFeatures()) if args.method == "temporal" else gray[:, None]
        train(output, features, labels, args.method, args.scale)
        return
    validation, v_audit, _ = load_labels(output, validate_ids, gray)
    development, d_audit, _ = load_labels(output, dev_ids, gray)
    if args.phase == "verify":
        labels, t_audit, _ = load_labels(output, train_ids, gray)
        audit = {"train": t_audit, "validation": v_audit, "development": d_audit}
        write_json(output / "development-alignment.json", audit)
        print("Verified development splits:", {k: (len(v), sum(r["boxes"] for r in v)) for k, v in audit.items()},
              flush=True)
    elif args.phase == "baselines":
        run_baselines(output, gray, validation, development)
    elif args.phase == "develop":
        checkpoints = records_for_fit(output, ("gray", "temporal"), args.scale)
        features = feature_cache(output, gray, max(dev_ids), CausalFeatures())
        develop(output, features, validation, development, checkpoints)
        propose(output)
    elif args.phase in CENTER_INPUTS:
        from snl_center import train_center

        labels, _, _ = load_labels(output, train_ids, gray)
        profile = CENTER_INPUTS[args.phase]
        features = feature_cache(output, gray, max(dev_ids), CausalFeatures(profile))
        stabilized = args.phase == "center-stable-labels"
        if stabilized:
            from audit_labels import stabilize_training_boxes

            labels, audit = stabilize_training_boxes(gray, labels)
            write_json(output / "stabilized-training-labels.json", audit)
        bank = None
        initial = None
        if args.phase in ("center-motion", "center-stacked-motion"):
            from snl_motion_aug import generate_bank

            bank = generate_bank(gray, labels, stacked=profile == "stacked")
        if args.phase in ("center-recent", "center-stacked-recent"):
            source = "stacked" if profile == "stacked" else "augmented"
            choices = [read_json(path) for path in output.glob(f"decision-radar-center-{source}-epoch*.json")
                       if "tiled" not in path.name]
            initial = max(choices, key=lambda row: quality(row["validation"]))["config"]
        soft_labels = args.phase in ("center-soft-labels", "center-stacked-soft")
        quality_focal = args.phase == "center-quality-focal"
        registry = []
        for checkpoint in train_center(output, features, labels, augment=args.phase in ("center-domain", "center-stacked-copy"),
                                       motion_bank=bank, input_profile=profile,
                                       dense_offsets=args.phase == "center-iou" or soft_labels or quality_focal,
                                       initial=initial, stabilized_labels=stabilized,
                                       soft_heat=soft_labels, learned_background=args.phase == "center-background",
                                       space_to_depth=args.phase == "center-space-depth",
                                       shared_temporal=args.phase == "center-shared-temporal",
                                       recurrent_temporal=args.phase == "center-recurrent-temporal",
                                       mixed_temporal=args.phase == "center-mixed-temporal",
                                       rank_peaks=args.phase == "center-peak-ranking",
                                       task_alignment=args.phase == "center-task-aligned",
                                       line_context=args.phase == "center-line-context",
                                       weak_shadow_augment=args.phase == "center-weak-shadows",
                                       quality_focal=quality_focal, rebalance_targets=args.phase == "center-weak-sampling",
                                       box_projection=args.phase == "center-box-projection",
                                       local_gain_augment=args.phase == "center-local-gain", seed=args.seed,
                                       backbone=args.backbone, epoch_budget=args.epoch_budget,
                                       pretrained_backbone=not args.random_backbone,
                                       checkpoint_interval=args.checkpoint_interval):
            registry.append(checkpoint)
            suffix = f"-seed{args.seed}" if args.seed != 1729 else ""
            write_json(output / f"checkpoints-{args.phase}{suffix}.json", registry)
        candidates = registry
        if args.phase == "center-task-aligned":
            candidates = registry + [row | {"dense_candidate_limit": 1024} for row in registry]
        if args.phase == "center-stacked-fused":
            candidates = registry + [row | FUSED_VIEW for row in registry]
        develop(output, features, validation, development, candidates)
        propose(output)
    elif args.phase == "center-tiled":
        checkpoints = []
        for file in ("checkpoints-radar-center.json", "checkpoints-center.json", "checkpoints-center-domain.json"):
            registry = output / file
            if registry.is_file():
                checkpoints += [read_json(registry)[-1] | {"tiled": True}]
        if not checkpoints:
            raise FileNotFoundError("The tiled comparison needs a completed center detector")
        features = feature_cache(output, gray, max(dev_ids), CausalFeatures())
        develop(output, features, validation, development, checkpoints)
        propose(output)
    elif args.phase in DENSE_ARCHITECTURES:
        labels, _, _ = load_labels(output, train_ids, gray)
        profile = "registered"
        features = feature_cache(output, gray, max(dev_ids), CausalFeatures(profile))
        registry = []
        checkpoints = (8, 16, 24) if args.phase == "fcos" else (4, 8, 12)
        for checkpoint in train_model(output, features, labels, "temporal", args.scale,
                                      checkpoint_epochs=checkpoints, architecture=args.phase, input_profile=profile):
            registry.append(checkpoint)
            write_json(output / f"checkpoints-{args.phase}.json", registry)
        develop(output, features, validation, development, registry)
        propose(output)
    elif args.phase in FULL_FRAME_ARCHITECTURES:
        labels, _, _ = load_labels(output, train_ids, gray)
        features = gray[:, None]
        registry = []
        for checkpoint in train_model(output, features, labels, "gray", 1, architecture=args.phase):
            registry.append(checkpoint)
            write_json(output / f"checkpoints-{args.phase}.json", registry)
        develop(output, features, validation, development, registry)
        propose(output)
    elif args.phase in YOLO_PHASES:
        from snl_yolo import train_yolo

        labels, _, _ = load_labels(output, train_ids, gray)
        temporal = args.phase.endswith("-temporal")
        source_phase = args.phase.removesuffix("-temporal")
        features = feature_cache(output, gray, max(dev_ids), CausalFeatures()) if temporal else gray[:, None]
        auxiliary = None
        if source_phase.startswith("yolo11-sived"):
            variant = source_phase.removeprefix("yolo11-")
            auxiliary = DEFAULT_OUTPUT / "external-sar" / f"{variant}-fit.json"
        registry = list(train_yolo(output, features, labels, validation, args.seed, temporal=temporal,
                                   pretrained=args.phase != "yolo11-scratch", p2=args.phase == "yolo11-p2",
                                   trunk_only=source_phase == "yolo11-coco-transfer", auxiliary=auxiliary))
        write_json(output / f"checkpoints-{args.phase}-seed{args.seed}.json", registry)
        develop(output, features, validation, development, registry)
        propose(output)
    elif args.phase == "center-full":
        from snl_full_context import train_full

        labels, _, _ = load_labels(output, train_ids, gray)
        features = feature_cache(output, gray, max(dev_ids), CausalFeatures("registered"))
        registry = []
        for checkpoint in train_full(output, features, labels):
            registry.append(checkpoint)
            write_json(output / "checkpoints-center-full.json", registry)
        develop(output, features, validation, development, registry)
        propose(output)
    elif args.phase == "center-map-ensemble":
        from snl_center import CenterMapEnsemble

        seeds = {}
        for result_path in sorted(output.glob("decision-radar-center-stacked-*epoch*.json")):
            row = read_json(result_path)
            cfg = row["config"]
            if any(cfg.get(option) for option in ("dense_offsets", "recent_fine_tune", "tiled", "flip_inference",
                                                  "stitch_inference",
                                                  "space_to_depth", "shared_temporal", "recurrent_temporal", "mixed_temporal",
                                                  "line_context",
                                                  "training_motion_synthesis",
                                                  "training_copy_paste",
                                                  "training_peak_ranking",
                                                  "task_alignment",
                                                  "weak_shadow_augmentation", "weak_target_sampling",
                                                  "box_projection_supervision", "local_gain_augmentation",
                                                  "occupancy_power", "contrast_gain",
                                                  "track_support")):
                continue
            seed = cfg.get("seed", 1729)
            if seed not in seeds or quality(row["validation"]) > quality(seeds[seed]["validation"]):
                seeds[seed] = row
        members = [row["config"] for row in seeds.values()]
        if len(members) < 2:
            raise ValueError("Map ensembling needs independently trained stacked models")
        model = CenterMapEnsemble(len(members),7)
        for member, cfg in zip(model.models,members,strict=True):
            path = output / cfg["path"]
            if sha_file(path) != cfg["sha256"]:
                raise ValueError(f"Ensemble member {path.name} differs from its frozen checksum")
            member.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
        name = output / "radar-center-map-ensemble.pt"
        torch.save(model.state_dict(), name)
        config = {"method":"temporal","architecture":"radar-center-map-ensemble","scale":1,
                  "input_profile":"stacked","path":name.name,"sha256":sha_file(name),
                  "members":len(members),"input_channels":7,"source_models":members,
                  "device":backend().type,"torch":str(torch.__version__)}
        features = feature_cache(output,gray,max(dev_ids),CausalFeatures("stacked"))
        develop(output,features,validation,development,[config])
        propose(output)
    elif args.phase == "center-spatial-views":
        groups = {}
        for path in sorted(output.glob("decision-radar-center*.json")):
            row = read_json(path)
            cfg = row["config"]
            profile = cfg.get("input_profile")
            if (profile not in ("stacked", "stacked-standardized", "background-history") or cfg.get("dense_offsets")
                    or cfg.get("tiled") or cfg.get("stitch_inference") or cfg.get("flip_inference") or cfg.get("recent_fine_tune")
                    or cfg.get("occupancy_power") or cfg.get("contrast_gain") or cfg.get("track_support")):
                continue
            key = (profile, cfg["architecture"], bool(cfg.get("space_to_depth")), bool(cfg.get("shared_temporal")),
                   bool(cfg.get("line_context")), bool(cfg.get("recurrent_temporal")), bool(cfg.get("mixed_temporal")))
            if key not in groups or quality(row["validation"]) > quality(groups[key]["validation"]):
                groups[key] = row
        if not groups:
            raise FileNotFoundError("Complete a stacked center comparison before evaluating spatial views")
        for (profile, architecture, *_), row in groups.items():
            base = row["config"]
            variants = [base | {"flip_inference": True}]
            if architecture == "radar-center":
                variants.extend((base | {"tiled": True}, base | {"tiled": True, "flip_inference": True}))
            features = feature_cache(output, gray, max(dev_ids), CausalFeatures(profile))
            develop(output, features, validation, development, variants)
        propose(output)
    elif args.phase == "center-occupancy":
        base = read_json(output / "development-status.json")["best"]["config"]
        if not base.get("architecture", "").startswith("radar-center"):
            raise ValueError("The occupancy comparison needs a center detector with a supervised foreground map")
        features = feature_cache(output, gray, max(dev_ids), CausalFeatures(base.get("input_profile", "raw")))
        variants = [base | {"occupancy_power": power} for power in (0.5, 1.0)]
        develop(output, features, validation, development, variants)
        propose(output)
    elif args.phase == "center-contrast-views":
        previous = read_json(output / "development-status.json")["candidates"]
        stacked = [row for row in previous if row["config"].get("input_profile") == "stacked"
                   and not row["config"].get("contrast_gain")]
        if not stacked:
            raise FileNotFoundError("Complete a stacked center comparison before evaluating contrast views")
        base = max(stacked, key=lambda row: quality(row["development"]))["config"]
        features = feature_cache(output, gray, max(dev_ids), CausalFeatures("stacked"))
        variants = [base | {"contrast_gain": gain} for gain in (1.35, 1.75)]
        develop(output, features, validation, development, variants)
        propose(output)
    elif args.phase == "track-support":
        base = read_json(output / "development-status.json")["best"]["config"]
        if base["method"] not in ("gray", "temporal"):
            raise ValueError("The confirmation comparison needs cached neural detections")
        variants = [base | {"track_support": support} for support in (2, 3)]
        develop(output, None, validation, development, variants, saved_predictions=True)
        propose(output)
    elif args.phase == "center-stitch":
        previous = read_json(output / "development-status.json")["candidates"]
        center = [r for r in previous if r["config"].get("architecture") == "radar-center"
                  and not r["config"].get("stitch_inference") and not r["config"].get("tiled")]
        if not center:
            raise FileNotFoundError("Complete a center comparison before stitching tiled maps")
        base = max(center, key=lambda row: quality(row["development"]))["config"]
        features = feature_cache(output, gray, max(dev_ids), CausalFeatures(base.get("input_profile", "raw")))
        develop(output, features, validation, development, [base | {"stitch_inference": True}])
        propose(output)
    elif args.phase == "iterate":
        decision = propose(output)
        if decision["status"] == "ready":
            return
        eligible = [r for r in decision["candidates"] if "path" in r["config"]]
        best = max(eligible, key=lambda row: (row["development_recall_at_60"], quality(row["development"])))
        features = feature_cache(output, gray, max(dev_ids), CausalFeatures())
        labels, _, _ = load_labels(output, train_ids, gray)
        if best["development_recall_at_60"] < 0.90:
            train(output, features, labels, "temporal", 3)
            advanced = records_for_fit(output, ("temporal",), 3)
        else:
            negatives = mine_negatives(output, features, best["config"], labels)
            with_revision = output / "hard-negative"
            with_revision.mkdir(exist_ok=True)
            train(with_revision, features, labels, "temporal", best["config"]["scale"], negatives)
            advanced = [r | {"path": f"hard-negative/{r['path']}"}
                        for r in records_for_fit(with_revision, ("temporal",), best["config"]["scale"])]
        develop(output, features, validation, development, advanced)
        propose(output)


if __name__ == "__main__":
    main()
