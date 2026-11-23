"""Run causal shadow detection on a video and export native-grid boxes and an MP4 overlay."""
import argparse
import csv
from fractions import Fraction
from pathlib import Path
from shutil import which
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from snl_detector import backend, load_trained, nonmax, predict_frame, torch
from snl_inputs import ROOT, read_json, sha_file, write_json

sys.path.insert(0, str(ROOT))
from videosar.data import run_directory
from videosar.snl_causal import CausalFeatures, TrackFilter


DEFAULT_MODEL = Path(__file__).resolve().parent / "models/stacked-center.json"
PREDICTION_FIELDS = ("frame", "source_frame", "x0_px", "y0_px", "x1_px", "y1_px", "score", "track_id")


def find_ffmpeg(explicit):
    if explicit:
        return explicit
    if binary := which("ffmpeg"):
        return binary
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise ValueError("Pass --ffmpeg or install FFmpeg so it is on PATH") from error
    return imageio_ffmpeg.get_ffmpeg_exe()


def stream_metadata(stream):
    line = stream.readline(4096).decode("ascii")
    tokens = line.split()
    if not tokens or tokens[0] != "YUV4MPEG2":
        raise ValueError("FFmpeg did not produce a video stream")
    header = {token[0]: token[1:] for token in tokens[1:]}
    rate = Fraction(header["F"].replace(":", "/"))
    width, height = int(header["W"]), int(header["H"])
    if header.get("C") != "mono" or min(width, height, rate) <= 0:
        raise ValueError("Expected grayscale video with a positive nominal frame rate and dimensions")
    return width, height, rate


def raw_frames(stream, width, height):
    while marker := stream.readline(4096):
        if marker.split(maxsplit=1)[0] != b"FRAME":
            raise ValueError("Unexpected FFmpeg frame delimiter")
        raw = stream.read(width * height)
        if len(raw) != width * height:
            raise ValueError("Video ends inside a decoded frame")
        yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width)


def decoder_command(ffmpeg, source, start, limit):
    command = [ffmpeg, "-nostdin", "-v", "error", "-xerror", "-i", str(source), "-map", "0:v:0"]
    if start:
        command += ["-vf", rf"select=gte(n\,{start})"]
    if limit:
        command += ["-frames:v", str(limit)]
    return command + ["-fps_mode", "passthrough", "-pix_fmt", "gray", "-f", "yuv4mpegpipe", "pipe:1"]


def encoder_command(ffmpeg, destination, width, height, rate):
    return [ffmpeg, "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size",
            f"{width}x{height}", "-framerate", str(rate), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]


def write_predictions(stream, encoder, prediction_stream, frame_stream, roi, model, config, start):
    writer = csv.writer(prediction_stream)
    writer.writerow(PREDICTION_FIELDS)
    overview = csv.writer(frame_stream)
    overview.writerow(("frame", "source_frame", "detections"))
    x, y, width, height = roi
    method = config["method"]
    scanner = CausalFeatures(config.get("input_profile", "raw")) if method == "temporal" else None
    high, low = config["threshold"], config.get("low")
    tracker = TrackFilter(high, low, confirmations=config.get("track_support", 1)) if low is not None else None
    frames, detections = 0, 0
    for index, source_gray in enumerate(stream):
        gray = source_gray[y:y + height, x:x + width]
        if scanner:
            channels = scanner.network_channels(gray)
            features = np.rint(channels * 255).astype(np.uint8)
        else:
            features = gray[None]
        boxes, scores = nonmax(*predict_frame(model, features, method), config["nms"])
        if tracker:
            boxes, scores, ids = tracker.update(index, boxes, scores)
        else:
            keep = scores >= high
            boxes, scores = boxes[keep], scores[keep]
            ids = [None] * len(boxes)
        source_index = start + index
        image = Image.fromarray(gray).convert("RGB")
        draw = ImageDraw.Draw(image)
        for box, score, track_id in zip(boxes, scores, ids, strict=True):
            writer.writerow((index, source_index, *map(float, box), float(score), track_id))
            draw.rectangle(tuple(box), outline="#57f1aa", width=2)
        overview.writerow((index, source_index, len(boxes)))
        draw.text((12, 12), f"Frame {source_index} | {len(boxes)} detections", fill="white",
                  stroke_width=2, stroke_fill="black")
        encoder.write(image.tobytes())
        frames += 1
        detections += len(boxes)
        if frames % 100 == 0:
            print(f"Processed {frames} video frames", flush=True)
    return frames, detections


def infer(source, output, model_path, ffmpeg, crop=None, start=0, limit=None):
    if start < 0 or (limit is not None and limit <= 0):
        raise ValueError("Use a nonnegative --start-frame and positive --max-frames")
    if not source.is_file():
        raise FileNotFoundError(source)
    source_before = source.stat()
    settings_digest = sha_file(model_path)
    config = read_json(model_path)
    if config["method"] not in ("gray", "temporal"):
        raise ValueError("The deployment model must use gray or temporal frame input")
    model = load_trained(model_path.parent, config)
    with subprocess.Popen(decoder_command(ffmpeg, source, start, limit), stdout=subprocess.PIPE) as decoder:
        source_width, source_height, rate = stream_metadata(decoder.stdout)
        roi = crop or (0, 0, source_width, source_height)
        x, y, width, height = roi
        if min(x, y) < 0 or min(width, height) < 2 or x + width > source_width or y + height > source_height:
            if decoder.poll() is None:
                decoder.terminate()
            raise ValueError(f"Crop {tuple(roi)} lies outside the {source_width}x{source_height} source")
        video = output / "overlay.mp4"
        with (subprocess.Popen(encoder_command(ffmpeg, video, width, height, rate), stdin=subprocess.PIPE) as encoder,
              (output / "predictions.csv").open("w", newline="", encoding="utf-8") as predictions,
              (output / "frames.csv").open("w", newline="", encoding="utf-8") as summary):
            stream = raw_frames(decoder.stdout, source_width, source_height)
            frames, detections = write_predictions(stream, encoder.stdin, predictions, summary,
                                                   roi, model, config, start)
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("FFmpeg could not write the overlay")
        if decoder.wait():
            raise RuntimeError("FFmpeg could not finish decoding the video")
    if not frames:
        raise ValueError("No video frames were decoded in the specified range")
    if sha_file(model_path) != settings_digest:
        raise ValueError("The model configuration changed during inference")
    digest = sha_file(source)
    source_after = source.stat()
    if (source_before.st_size, source_before.st_mtime_ns) != (source_after.st_size, source_after.st_mtime_ns):
        raise ValueError("The source video changed during inference")
    write_json(output / "video.json", {
        "source": str(source), "source_sha256": digest, "nominal_fps": str(rate),
        "source_size": [source_width, source_height], "crop_xywh": list(roi), "first_source_frame": start,
        "last_source_frame": start + frames - 1, "frames": frames, "detections": detections,
        "coordinates": "zero-based frame and pixel coordinates relative to the crop",
        "model": str(model_path), "model_config_sha256": settings_digest, "model_sha256": config["sha256"],
        "device": backend().type, "torch": str(torch.__version__), "input_profile": config.get("input_profile", "raw"),
        "threshold": config["threshold"], "low": config.get("low"), "nms": config["nms"],
    })


def add_arguments(parser):
    parser.add_argument("--input", type=Path, required=True, help="video file")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="detector deployment JSON")
    parser.add_argument("--ffmpeg", help="FFmpeg executable; autodetected when omitted")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--start-frame", type=int, default=0, help="zero-based source frame where causal inference starts")
    parser.add_argument("--max-frames", type=int, help="optional maximum frame count")
    parser.add_argument("--threads", type=int, default=4, help="number of PyTorch CPU threads")


def run(args, output):
    torch.set_num_threads(args.threads)
    infer(args.input.resolve(), output, args.model.resolve(), find_ffmpeg(args.ffmpeg), args.crop,
          args.start_frame, args.max_frames)


def main():
    parser = argparse.ArgumentParser(prog="python -m videosar detect-video", description=__doc__)
    add_arguments(parser)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        config = {"input": str(args.input.resolve()), "model": str(args.model.resolve()), "crop": args.crop,
                  "start_frame": args.start_frame, "max_frames": args.max_frames, "threads": args.threads}
        with run_directory(args.output_dir, "detect-video", config) as output:
            run(args, output)
        print(output)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"videosar: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
