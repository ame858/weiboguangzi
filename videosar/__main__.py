import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

from .analysis import baseline_scores, score_masks, score_track, segment, target_summary, track
from .config import CHOICES, DETECTION, EVALUATION, SIMULATION, TRACKING, merged
from .data import (load_masks, load_observations, read_json, run_directory, save_receipt,
                   save_sequence, verify_receipt, write_json)
from .experiment import benchmark, get_sample, make_dataset, read_split, train_models
from .learning import load_model, predict
from .simulation import form_video, point_check, simulate
from .visuals import save_point, save_preview, write_csv


COMMANDS = {
    "demo": (SIMULATION | DETECTION | TRACKING | EVALUATION, "run the image-domain chain and render a preview"),
    "simulate": (SIMULATION, "generate a labelled image-domain sequence"),
    "image": (SIMULATION, "form a single-channel SAR video and run point checks"),
    "detect": (DETECTION, "detect shadows from a sequence"),
    "detect-video": ({}, "process a video file and export causal boxes and an MP4 overlay"),
    "track": (TRACKING, "estimate one shadow track from detections"),
    "evaluate": (EVALUATION, "compare prediction and tracks with simulation truth"),
    "dataset": ({"dataset_size": 30, "seed": 42, "profile": "mixed"}, "generate grouped train, validation and independent test scenes"),
    "train": ({"epochs": 16, "seed": 42, "min_area": 4, "network": "compact"}, "train single-frame and causal-temporal models"),
    "benchmark": ({"epochs": 16, "seed": 42, "min_area": 4, "network": "compact"} | TRACKING | EVALUATION,
                  "train, tune and compare four methods on the independent test split"),
    "study": ({"epochs": 16, "seed": 20261119, "batch_size": 32, "device": "auto"} |
              TRACKING | {"match_iou": 0.5}, "train open-source U-Net variants and evaluate the proposal dataset"),
    "stress": ({}, "run the frozen comparison on 72 independent and paired SAR sequences"),
    "echo-attribution": ({}, "compare four paired bandwidth and receiver SNR conditions"),
    "aperture-study": ({}, "compare three paired SAR aperture lengths at fixed bandwidth and SNR"),
    "aperture-motion-study": ({}, "isolate target motion within paired SAR apertures"),
    "contrast-study": ({}, "test multiscale local contrast features on independent SAR scenes"),
    "center-study": ({}, "compare point-based shadow proposals with segmentation on independent SAR scenes"),
    "trajectory-study": ({}, "compare sequence-based weak shadow candidates on independent SAR scenes"),
    "trajectory-detect": ({}, "run a retained trajectory study profile on one sequence"),
    "small-shadow": ({}, "compare recalibration and fine-tuning on a new small-shadow dataset"),
    "open-compare": ({}, "compare open segmentation implementations on a new held-out SAR test set"),
    "temporal-study": ({}, "test causal background history and small-shadow-weighted training"),
}


def parser():
    root = argparse.ArgumentParser(prog="python -m videosar", description="Reproducible video SAR shadow experiments")
    sub = root.add_subparsers(dest="command", required=True)
    for name, (defaults, help_text) in COMMANDS.items():
        command = sub.add_parser(name, help=help_text, description=help_text)
        command.add_argument("--config", type=Path, help="JSON parameters; explicit flags take priority")
        command.add_argument("--output-dir", type=Path, help="new directory; defaults to a unique directory under runs")
        for key, default in defaults.items():
            flag = "--" + key.replace("_", "-")
            options = {"choices": CHOICES[key]} if key in CHOICES else {"type": float if default is None else type(default)}
            help_text = f"default: {default}; 120 for proposal" if key == "dataset_size" else f"default: {default}"
            command.add_argument(flag, dest=key, default=argparse.SUPPRESS, help=help_text, **options)
        if name in ("detect", "track", "evaluate", "trajectory-detect"):
            command.add_argument("--input", type=Path, required=True, help="sequence directory")
        if name in ("demo", "detect"):
            command.add_argument("--model", type=Path, help="NPZ weights, U-Net PT weights or a U-Net JSON deployment file")
        if name == "stress":
            command.add_argument("--model", type=Path, help="U-Net JSON deployment file; defaults to the proposal checkpoint")
        if name == "aperture-motion-study":
            command.add_argument("--reference-run", type=Path, help="previous aperture-study run for exact replay checks")
        if name == "center-study":
            command.add_argument("--training-run", type=Path, default=Path("runs/contrast-study"),
                                 help="verified contrast-study data and checkpoint")
        if name == "trajectory-study":
            command.add_argument("--training-run", type=Path, default=Path("runs/contrast-study"))
            command.add_argument("--center-run", type=Path, default=Path("runs/center-study"))
        if name == "trajectory-detect":
            command.add_argument("--profile", type=Path, required=True, help="retained profile JSON from trajectory-study")
        if name == "open-compare":
            command.add_argument("--reuse-models", type=Path, help="previous open-compare run directory containing hashed checkpoints")
        if name in ("track", "evaluate"):
            command.add_argument("--predictions", type=Path, required=True, help="detect output directory")
        if name == "evaluate":
            command.add_argument("--tracks", type=Path, required=True, help="track output directory")
        if name in ("train", "benchmark", "study", "open-compare", "temporal-study"):
            command.add_argument("--dataset", type=Path, required=True, help="dataset output directory")
    return root


def detect(source, output, config, model_path, default_area=False):
    frames, _, _ = load_observations(source)
    method = config["method"]
    threshold = config["threshold"]
    min_area = config["min_area"]
    aligned, correction = None, None
    center_distances = None
    use_unet = method == "learned" and model_path.suffix.lower() in (".pt", ".json")
    if use_unet:
        from .study import cleanup
        from .unet import align, load_weights, predict as predict_unet, predict_centers, runtime

        torch, smp, correlate = runtime()
        torch.set_num_threads(min(6, torch.get_num_threads()))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        settings = read_json(model_path.with_suffix(".json"))
        weights = model_path if model_path.suffix.lower() == ".pt" else model_path.with_name(
            settings.get("checkpoint", model_path.with_suffix(".pt").name))
        model = load_weights(torch, smp, weights, settings, device)
        aligned, correction = align(frames, correlate)
        if settings.get("detection_mode") == "center_box":
            scores, center_distances = predict_centers(torch, model, aligned, settings, device, 32)
        else:
            scores = predict_unet(torch, model, aligned, settings, device, 32)
        if threshold is None:
            threshold = settings["threshold"]
        if default_area:
            min_area = settings["min_area"]
    elif method == "learned":
        scores = predict(frames, load_model(model_path))
        if threshold is None:
            settings = model_path.with_suffix(".json")
            threshold = read_json(settings)["threshold"] if settings.is_file() else 0.5
    else:
        scores = baseline_scores(frames, method)
        if threshold is None:
            threshold = 0.45 if method == "threshold" else 0.35
    merged(DETECTION, {"threshold": threshold, "min_area": min_area})
    if center_distances is not None:
        from .center_box import center_masks

        masks = center_masks(scores, center_distances, threshold, min_area)
    else:
        masks = cleanup(scores, threshold, min_area, settings["closing"]) if use_unet else segment(scores, threshold, min_area)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "scores.npy", scores)
    np.save(output / "predictions.npy", masks.astype(np.uint8))
    if aligned is not None:
        np.save(output / "aligned_frames.npy", aligned)
        np.save(output / "registration_offsets_xy_px.npy", correction)
    save_receipt(output, source, detector=method, threshold=threshold, min_area=min_area,
                 model=str(model_path.resolve()) if model_path else None, registered=use_unet)


def run_tracking(source, predictions, output, config):
    frames, times, scale = load_observations(source)
    verify_receipt(predictions, source)
    masks = load_masks(predictions / "predictions.npy", frames.shape)
    result = track(masks, times, scale, config["gate_px"], config["max_gap"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "tracks.json", result)
    rows = []
    for row in result["frames"]:
        x, y = row["xy_px"] or (None, None)
        rows.append({"frame": row["frame"], "time_s": row["time_s"], "x_px": x, "y_px": y,
                     "source": row["source"], "segment": row["segment"]})
    write_csv(output / "tracks.csv", rows)
    save_receipt(output, source, predictions=str(predictions.resolve()), gate_px=config["gate_px"], max_gap=config["max_gap"])


def validate_tracks(tracked, times):
    rows = tracked["frames"]
    if len(rows) != len(times):
        raise ValueError("tracks.json must have one row per frame")
    for index, (row, time) in enumerate(zip(rows, times)):
        point = row["xy_px"]
        origin = row["source"]
        if row["frame"] != index or row["time_s"] != time or origin not in ("measured", "imputed", "missing"):
            raise ValueError("tracks.json frame, time or source is invalid")
        if origin == "missing" and point is None:
            continue
        if point is None or origin == "missing" or np.asarray(point).shape != (2,) or not np.isfinite(point).all():
            raise ValueError("tracks.json contains an invalid x/y position")


def evaluate(source, predictions, tracks, output, config):
    frames, times, scale = load_observations(source)
    detection_receipt = verify_receipt(predictions, source)
    receipt = verify_receipt(tracks, source)
    if Path(receipt["predictions"]).resolve() != predictions.resolve():
        raise ValueError("tracks were produced from a different prediction directory")
    prediction = load_masks(predictions / "predictions.npy", frames.shape)
    truth = load_masks(source / "masks.npy", frames.shape)
    tracked = read_json(tracks / "tracks.json")
    validate_tracks(tracked, times)
    metadata = read_json(source / "metadata.json")
    metrics = score_masks(prediction, truth, config["match_iou"])
    metrics.update(target_summary(metrics["matched"], metrics["false_positive"], metrics["missed"]))
    metrics["false_positive_per_frame"] = metrics["false_positive"] / len(frames)
    motion = score_track(tracked, truth, scale, metadata)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", {"detection": metrics, "tracking": motion, "match_iou": config["match_iou"]})
    write_csv(output / "per_frame.csv", [{"frame": i, "time_s": times[i], **row} for i, row in enumerate(metrics["frames"])])
    indices = np.unique(np.linspace(0, len(frames) - 1, min(4, len(frames))).round().astype(int)).tolist()
    displayed = np.load(predictions / "aligned_frames.npy", allow_pickle=False) if detection_receipt.get("registered") else frames
    save_preview(output / "preview.png", displayed, truth, prediction, tracked, indices)
    save_receipt(output, source, predictions=str(predictions.resolve()), tracks=str(tracks.resolve()))
    iou = f"{metrics['iou']:.3f}" if metrics["iou"] is not None else "undefined (empty union)"
    print(f"IoU: {iou}; matches: {metrics['matched']}; false positives: {metrics['false_positive']}; misses: {metrics['missed']}")


def image(output, config):
    check, point_power = point_check(config)
    alternate, _ = point_check(config | {"bandwidth_hz": config["bandwidth_hz"] / 2})
    width_ratio = alternate["range_mainlobe_half_power_width_m"] / check["range_mainlobe_half_power_width_m"]
    check.update(half_bandwidth_mainlobe_m=alternate["range_mainlobe_half_power_width_m"],
                 half_bandwidth_width_ratio=width_ratio,
                 range_peak_within_bin=check["range_peak_error_bins"] <= 1,
                 point_peak_within_pixel=check["point_error_px"] <= 1,
                 width_scales_with_bandwidth=1.9 <= width_ratio <= 2.1)
    write_json(output / "point_check.json", check)
    save_point(output / "point_check.png", point_power, check)
    frames, masks, info, geometry, reference = form_video(config)
    save_sequence(output, frames, masks, info, geometry)
    np.save(output / "unoccluded_power.npy", reference.astype(np.float32))
    print(f"Point range error: {check['range_peak_error_bins']:.3f} bins; image peak error: {check['point_error_px']:.3f} px")
    if not check["point_peak_within_pixel"]:
        print("Point focus exceeds one pixel for the selected aperture; inspect point_check.json.", file=sys.stderr)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["detect-video"]:
        entry = Path(__file__).resolve().parents[1] / "deliverables/snl-causal-study/infer_video.py"
        return subprocess.run([sys.executable, str(entry), *argv[1:]], check=False).returncode
    cli = parser()
    args = cli.parse_args(argv)
    try:
        defaults = COMMANDS[args.command][0]
        file_config = read_json(args.config) if args.config is not None else {}
        if not isinstance(file_config, dict):
            raise ValueError("config must be a JSON object")
        config = merged(defaults, file_config | {key: vars(args)[key] for key in defaults if key in vars(args)})
        default_area = "min_area" not in file_config and "min_area" not in vars(args)
        if args.command == "image" and config["scene"] == "legacy":
            raise ValueError("scene legacy is only available to image-domain simulation")
        if args.command in ("demo", "detect") and ((config["method"] == "learned") != (args.model is not None)):
            raise ValueError("--model is required with --method learned and valid only for that method")
        with run_directory(args.output_dir, args.command, config) as output:
            if args.command in ("demo", "simulate"):
                source = output / "sequence" if args.command == "demo" else output
                simulation_config = {key: config[key] for key in SIMULATION}
                save_sequence(source, *simulate(simulation_config))
                if args.command == "demo":
                    detect(source, output / "detect", config, args.model, default_area)
                    run_tracking(source, output / "detect", output / "track", config)
                    evaluate(source, output / "detect", output / "track", output / "evaluate", config)
            elif args.command == "image":
                image(output, config)
            elif args.command == "detect":
                detect(args.input, output, config, args.model, default_area)
            elif args.command == "track":
                run_tracking(args.input, args.predictions, output, config)
            elif args.command == "evaluate":
                evaluate(args.input, args.predictions, args.tracks, output, config)
            elif args.command == "dataset":
                make_dataset(output, config["dataset_size"], config["seed"], config["profile"])
            elif args.command == "train":
                splits = read_split(args.dataset)
                train = [get_sample(path) for path, _ in splits["train"]]
                validation = [get_sample(path) for path, _ in splits["validation"]]
                train_models(output, train, validation, config["epochs"], config["seed"], config["min_area"], config["network"])
            elif args.command == "benchmark":
                benchmark(output, args.dataset, config)
            elif args.command == "study":
                from .study import study

                study(output, args.dataset, config)
            elif args.command == "stress":
                from .stress import run_stress

                run_stress(output, args.model)
            elif args.command == "echo-attribution":
                from .echo_attribution import run_attribution

                run_attribution(output)
            elif args.command == "aperture-study":
                from .aperture_study import run_aperture_study

                run_aperture_study(output)
            elif args.command == "aperture-motion-study":
                from .aperture_motion_study import run_aperture_motion_study

                run_aperture_motion_study(output, args.reference_run)
            elif args.command == "contrast-study":
                from .contrast_study import run_contrast_study

                run_contrast_study(output)
            elif args.command == "center-study":
                from .center_study import run_center_study

                run_center_study(output, args.training_run)
            elif args.command == "trajectory-study":
                from .trajectory_study import run_trajectory_study

                run_trajectory_study(output, args.training_run, args.center_run)
            elif args.command == "trajectory-detect":
                from .trajectory_study import trajectory_detect

                trajectory_detect(args.input, output, args.profile)
            elif args.command == "small-shadow":
                from .small_shadow import run_small_shadow

                run_small_shadow(output)
            elif args.command == "open-compare":
                from .open_compare import run_open_comparison

                run_open_comparison(output, args.dataset, args.reuse_models)
            elif args.command == "temporal-study":
                from .temporal_study import run_temporal_study

                run_temporal_study(output, args.dataset)
        print(output)
        return 0
    except (OSError, ValueError, KeyError, TypeError, OverflowError, RuntimeError) as exc:
        print(f"videosar: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
