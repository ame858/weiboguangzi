"""Prepare a blind Qatar road-shadow pilot and score returned LabelMe files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image

from experiment import summarize
from infer_video import DEFAULT_MODEL, decoder_command, find_ffmpeg, infer, raw_frames, stream_metadata, torch
from snl_independent_review import read_review, review_boxes
from snl_inputs import DEFAULT_OUTPUT, read_json, sha_file, write_csv, write_json
from snl_rolling_nwd import center_metrics
from videosar.data import run_directory


EXTERNAL = DEFAULT_OUTPUT / "external-sar"
SOURCE = EXTERNAL / "vsdformer/Noisy_Qatar.mp4"
SOURCE_HASH = "510f2dca9414ed95f3f1d0985a4d0e34f788d4f24619ea6f79c035f4715be563"
SOURCE_URL = "https://github.com/BFY-official/VSDFormer/tree/c61fb8e744c06f22815f07ad1a0377bebd7c9feb/Videos"
OUTPUT = EXTERNAL / "qatar-transfer-pilot"
ROI = (0, 280, 720, 266)
HISTORY = 7
PINNED_MODEL = "7713d072ff43c7b84de2d9a5fa9abb9137b7d9bd1fe0e6cc43413d7bc6591375"
REPLAY_FILES = ("video.json", "predictions.csv", "frames.csv")
INSTRUCTIONS = """卡塔尔 VideoSAR 车辆阴影盲标

请按编号处理 frames 内的 18 张目标图。context 内同名短片只展示此前及当前画面，末帧是目标帧。
先保存当前帧的 JSON，再打开下一帧。只在目标 PNG 上画框。每张图是原生像素裁出的 720×266 道路区域。

* 用 shadow 标记可确认由移动车辆产生的阴影，矩形贴近当前帧可见暗区。
* 用 uncertain 标记疑似移动车辆阴影。车辆的亮点、方位向亮拖尾和静态阴影不作为 shadow。
* 目标中心在图内才计入。边缘截断且无法判断的目标可标 uncertain。
* 无目标的帧也保存空 LabelMe JSON。保持文件名对应，例如 review-01.png 对应 review-01.json。
* 请交回 18 个 JSON 的文件夹或 ZIP。两人复标时各自提交。

图像来源为 VSDFormer 发布的 Noisy_Qatar.mp4。帧号清单保存慢放视频帧与独立画面的对应关系。
"""


def decode(source, ffmpeg):
    with subprocess.Popen(decoder_command(ffmpeg, source, 0, None), stdout=subprocess.PIPE) as process:
        width, height, fps = stream_metadata(process.stdout)
        frames = np.stack(list(raw_frames(process.stdout, width, height)))
        if process.wait():
            raise RuntimeError(f"FFmpeg failed to decode {source.name}")
    return frames, fps


def observations(frames, fps, *, minimum_transition=4.0):
    if frames.shape != (313, 720, 720) or fps != 25:
        raise ValueError("Unexpected Qatar playback format")
    starts = (25 * np.arange(25) + 1) // 2
    coarse = frames[:, ::4, ::4].astype(np.int16)
    change = np.abs(np.diff(coarse, axis=0)).mean(axis=(1, 2))
    transitions = starts[1:] - 1
    boundary_min = float(change[transitions].min())
    holds_p95 = float(np.quantile(np.delete(change, transitions), 0.95))
    if boundary_min < minimum_transition or holds_p95 >= 0.1:
        raise ValueError("The expected playback holds and transitions were not reproduced")
    return frames[starts], starts.tolist(), dict(boundary_min=boundary_min, within_hold_p95=holds_p95)


def lossless_video(path, frames, ffmpeg):
    _, height, width = frames.shape
    command = [ffmpeg, "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "gray", "-video_size",
               f"{width}x{height}", "-framerate", "2", "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast",
               "-crf", "0", "-vf", "scale=in_range=full:out_range=full", "-color_range", "pc", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, input=frames.tobytes(), check=True)
    decoded, _ = decode(path, ffmpeg)
    np.testing.assert_array_equal(decoded, frames, err_msg=path.name)


def export_review(root, frames, source_indices, ffmpeg):
    folder = root / "blind-review"
    (folder / "frames").mkdir(parents=True)
    (folder / "context").mkdir()
    x, y, width, height = ROI
    road = frames[:, y:y + height, x:x + width]
    rows, files = [], []
    for sample, index in enumerate(range(HISTORY, len(road)), start=1):
        tag = f"review-{sample:02d}"
        png, clip = folder / "frames" / f"{tag}.png", folder / "context" / f"{tag}.mp4"
        Image.fromarray(road[index]).save(png)
        with Image.open(png) as image:
            np.testing.assert_array_equal(np.asarray(image), road[index])
        lossless_video(clip, road[index - HISTORY:index + 1], ffmpeg)
        rows.append(dict(sample=tag, observation_zero_based=index, playback_frame_zero_based=source_indices[index],
                         target_frame_in_clip=HISTORY + 1, image_sha256=sha_file(png)))
        files.extend((png, clip))
    write_csv(folder / "manifest.csv", rows)
    write_json(folder / "manifest.json", dict(source_url=SOURCE_URL, source_sha256=SOURCE_HASH, crop_xywh=ROI,
               context_policy="past_and_current", frames=rows))
    (folder / "README.txt").write_text(INSTRUCTIONS, encoding="utf-8-sig")
    files.extend(folder / name for name in ("manifest.json", "manifest.csv", "README.txt"))
    with ZipFile(root / "qatar-blind-review.zip", "w", ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(folder))
    with ZipFile(root / "qatar-blind-review.zip") as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(files):
            raise ValueError("Incomplete review archive")
    return rows


def load_boxes(replay):
    with (replay / "frames.csv").open(newline="", encoding="utf-8") as file:
        counts = {int(row["source_frame"]): int(row["detections"]) for row in csv.DictReader(file)}
    boxes = {index: [] for index in counts}
    with (replay / "predictions.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            boxes[int(row["source_frame"])].append([float(row[key]) for key in ("x0_px", "y0_px", "x1_px", "y1_px")])
    if any(len(value) != counts[index] for index, value in boxes.items()):
        raise ValueError("Prediction CSV and frame counts differ")
    return {index: np.asarray(value, dtype=np.float32).reshape(-1, 4) for index, value in boxes.items()}


def in_review_area(boxes):
    x, y, width, height = ROI
    center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    keep = ((center >= (x, y)) & (center < (x + width, y + height))).all(axis=1)
    return np.clip(boxes[keep] - np.array([x, y, x, y], dtype=np.float32), 0, (width, height, width, height))


def prepare(source, output, ffmpeg):
    if sha_file(source) != SOURCE_HASH:
        raise ValueError("Qatar source differs from the pinned VSDFormer release")
    model = read_json(DEFAULT_MODEL)
    if (model["sha256"], model["threshold"], model["nms"], model["low"]) != (PINNED_MODEL, 0.25, 0.35, None):
        raise ValueError("The configured SNL delivery model or operation point differs from the pilot")
    with run_directory(output, "snl-qatar-pilot", {"source": str(source), "source_sha256": SOURCE_HASH, "crop_xywh": ROI}) as root:
        frames, starts, timing = observations(*decode(source, ffmpeg))
        rows = export_review(root, frames, starts, ffmpeg)
        video, replay = root / "qatar-25-observations.mp4", root / "frozen-predictions"
        lossless_video(video, frames, ffmpeg)
        replay.mkdir()
        infer(video, replay, DEFAULT_MODEL, ffmpeg)
        metadata = read_json(replay / "video.json")
        boxes = load_boxes(replay)
        if metadata["frames"] != len(frames) or list(boxes) != list(range(len(frames))):
            raise ValueError("Inference did not replay all observation frames in order")
        reviewed = [row["observation_zero_based"] for row in rows]
        receipt = dict(original_source_sha256=SOURCE_HASH, inference_video_sha256=sha_file(video),
                       representative_playback_frames=starts, hold_transition_check=timing, crop_xywh=ROI,
                       review_manifest_sha256=sha_file(root / "blind-review/manifest.json"),
                       prediction_hashes={name: sha_file(replay / name) for name in REPLAY_FILES},
                       model_sha256=PINNED_MODEL, reviewed_observations=reviewed,
                       all_full_frame_predictions=sum(map(len, boxes.values())),
                       predictions_in_review_area=sum(len(in_review_area(boxes[i])) for i in reviewed))
        write_json(root / "frozen-pilot.json", receipt)
        print(f"{len(frames)} observations, {len(rows)} blind review frames, {receipt['predictions_in_review_area']} predictions in review area")


def score(root, source):
    receipt = read_json(root / "frozen-pilot.json")
    folder, replay = root / "blind-review", root / "frozen-predictions"
    manifest_path = folder / "manifest.json"
    if sha_file(manifest_path) != receipt["review_manifest_sha256"]:
        raise ValueError("Review manifest changed after freezing predictions")
    if any(sha_file(replay / name) != receipt["prediction_hashes"][name] for name in REPLAY_FILES):
        raise ValueError("Frozen prediction artifact changed")
    rows = read_json(manifest_path)["frames"]
    reviewed = read_review(source, [row["sample"] for row in rows])
    boxes = load_boxes(replay)
    predictions, targets, uncertain = {}, {}, {}
    for row in rows:
        index, tag = row["observation_zero_based"], row["sample"]
        png = folder / "frames" / f"{tag}.png"
        if sha_file(png) != row["image_sha256"]:
            raise ValueError(f"Review image changed for {tag}")
        targets[index], uncertain[index] = review_boxes(reviewed[f"{tag}.json"], png, image_size=ROI[2:])
        predictions[index] = in_review_area(boxes[index]), None, None
    iou50, counts = summarize(predictions, targets)
    for row in counts:
        index = row.pop("annotation_id")
        row.pop("video_frame")
        row.update(observation_zero_based=index, playback_frame_zero_based=receipt["representative_playback_frames"][index],
                   uncertain=uncertain[index])
    label_digest = hashlib.sha256(json.dumps(reviewed, sort_keys=True).encode()).hexdigest()
    stats = dict(iou50=iou50, center5=center_metrics(predictions, targets), uncertain=sum(uncertain.values()),
                 source_sha256=SOURCE_HASH, model_sha256=PINNED_MODEL, crop_xywh=ROI,
                 annotations_sha256=label_digest, annotation_source=str(source))
    results = root / "scores" / label_digest[:16]
    results.mkdir(parents=True, exist_ok=True)
    write_json(results / "qatar-score.json", stats)
    write_csv(results / "qatar-score-rows.csv", counts)
    print(results, stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--labels", type=Path, help="Independent LabelMe directory or ZIP to score")
    args = parser.parse_args()
    if args.labels is None:
        torch.set_num_threads(4)
        prepare(args.source.resolve(), args.output.resolve(), find_ffmpeg(args.ffmpeg))
    else:
        score(args.output.resolve(), args.labels.resolve())
