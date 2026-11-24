"""Export and validate blind, continuous SNL shadow-track annotations."""

import argparse
import csv
import io
import json
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image

from snl_independent_review import read_review, review_boxes
from snl_inputs import BLIND, DEFAULT_OUTPUT, HEIGHT, PROFILES, VIDEO, WIDTH, read_json, sha_file, write_csv, write_json


FOLDER = DEFAULT_OUTPUT / "independent-track-review"
SOURCE = "https://www.sandia.gov/app/uploads/sites/124/2021/08/eubankgateandtrafficvideosar.mp4"
CROP = (310, 0, WIDTH, HEIGHT)
CLIPS = (
    ("track-01", "training", 275, 296),
    ("track-02", "development", 332, 353),
    ("track-03", "development", 358, 379),
)
OCCLUSION_COLUMNS = ("sample", "group_id")
LABELME_CONFIG = """labels: [shadow, uncertain]
label_flags:
  '^(shadow|uncertain)$': [occluded]
"""
INSTRUCTIONS = """SNL 视频 SAR 连续轨迹盲标

videos 中有 3 段无叠框短片。可反复查看同一短片的前后画面，确认哪些暗区随车辆移动。
在 LabelMe 中打开 frames 文件夹，按同一段内的帧号逐张画矩形。PNG 与短片使用同一套原生灰度像素。
manifest.csv 的 clip_frame_in_video 从 1 开始，source_frame_zero_based 为原视频帧号，从 0 开始。
短片以每秒 2 帧播放，轨迹时间以原视频连续帧序号为准。
如果已安装 LabelMe，可在解压目录运行 labelme --config labelme-config.yaml frames。

1. 用 shadow 标记可以确认的运动阴影，矩形围住当前帧可见的暗区。车辆亮回波和静态暗区不纳入。
2. 用 uncertain 标记身份、可见范围或跨帧关联仍不确定的暗区。
3. 在 Group ID 填正整数。同一段内，同一目标跨帧保留同一个 ID，不同目标用不同 ID。每帧每个 ID 至多一框，进入下一段可重新编号。
4. 阴影部分被遮挡且仍能画框时，勾选该框的 occluded。若该帧完全看不见，保留不画框，并在 occlusions.csv 填该帧的 sample 和已确认的 group_id。目标刚进入或已离开视野时留空。
5. 每张 PNG 都保存一个同名 LabelMe JSON，空帧保存 shapes 为空的 JSON。交回 JSON 文件夹或 ZIP，并附上填写过的 occlusions.csv。

短片允许查看段内后续画面，所得轨迹作为离线独立参考。请在提交前核对各段轨迹 ID。
来源为 Sandia National Laboratories 的 Eubank Gate and Traffic VideoSAR。
"""


def planned_clips():
    splits = dict(zip(("training", "calibration", "development"), PROFILES["prospective"], strict=True))
    sealed = {i for groups in BLIND.values() for block in groups.values() for i in block}
    used = set()
    for clip, split, first, last in CLIPS:
        ids = tuple(range(first, last + 1))
        if not set(ids) <= set(splits[split]) or set(ids) & (sealed | used):
            raise ValueError(f"{clip}: frames cross a split, overlap another clip, or touch the sealed set")
        used.update(ids)
        yield clip, split, ids


def original_frames(ffmpeg, clips):
    selector = "+".join(rf"between(n\,{indices[0] + 148}\,{indices[-1] + 148})" for _, _, indices in clips)
    x, y, width, height = CROP
    payload = subprocess.check_output([
        ffmpeg, "-nostdin", "-v", "error", "-i", str(VIDEO),
        "-vf", f"select={selector},crop={width}:{height}:{x}:{y},format=gray",
        "-fps_mode", "passthrough", "-threads", "1", "-f", "rawvideo", "pipe:1",
    ])
    frames = np.frombuffer(payload, dtype=np.uint8).reshape(-1, height, width)
    if len(frames) != sum(len(ids) for _, _, ids in clips):
        raise ValueError("The source video did not provide every planned frame")
    return frames


def export(folder, ffmpeg):
    from snl_qatar_pilot import lossless_video

    archive = folder.with_suffix(".zip")
    if folder.exists() or archive.exists():
        raise FileExistsError(f"Review material already exists at {folder} or {archive}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    clips = list(planned_clips())
    frames = original_frames(ffmpeg, clips)
    rows, videos, offset = [], [], 0
    with TemporaryDirectory(prefix=".track-review-", dir=folder.parent) as work:
        staging = Path(work) / folder.name
        (staging / "frames").mkdir(parents=True)
        (staging / "videos").mkdir()
        for clip, split, indices in clips:
            image_stack = frames[offset:offset + len(indices)]
            offset += len(indices)
            video = staging / "videos" / f"{clip}.mp4"
            lossless_video(video, image_stack, ffmpeg)
            videos.append(dict(clip=clip, split=split, first_annotation_id=indices[0],
                               last_annotation_id=indices[-1], frame_count=len(indices),
                               playback_fps=2, path=video.relative_to(staging).as_posix(),
                               video_sha256=sha_file(video)))
            for position, (index, frame) in enumerate(zip(indices, image_stack, strict=True), 1):
                sample = f"{clip}-{position:03d}"
                png = staging / "frames" / f"{sample}.png"
                Image.fromarray(frame).save(png)
                with Image.open(png) as restored:
                    np.testing.assert_array_equal(np.asarray(restored), frame, err_msg=sample)
                rows.append(dict(sample=sample, clip=clip, split=split, clip_frame_in_video=position,
                                 annotation_id=index, source_frame_zero_based=index + 148,
                                 image_sha256=sha_file(png)))
        (staging / "README.txt").write_text(INSTRUCTIONS + SOURCE + "\n", encoding="utf-8-sig")
        (staging / "labelme-config.yaml").write_text(LABELME_CONFIG, encoding="utf-8")
        write_csv(staging / "occlusions.csv", [], fields=OCCLUSION_COLUMNS)
        write_csv(staging / "manifest.csv", rows)
        write_json(staging / "manifest.json", dict(source_url=SOURCE, video_sha256=sha_file(VIDEO),
                                                  crop_xywh=CROP, context_policy="bidirectional_within_clip",
                                                  sample_count=len(rows), clips=videos, frames=rows))
        names = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
        temp_archive = Path(work) / archive.name
        with ZipFile(temp_archive, "w", ZIP_DEFLATED) as output:
            for name in names:
                output.write(staging / name, name)
        with ZipFile(temp_archive) as output:
            if output.namelist() != names or output.testzip() is not None:
                raise ValueError("Review archive did not reproduce the generated files")
        staging.rename(folder)
        temp_archive.rename(archive)
    print(f"Verified {len(rows)} unmarked frames across {len(clips)} lossless clips: {archive}", flush=True)


def returned_occlusions(labels):
    if labels.is_dir():
        path = labels / "occlusions.csv"
        content = path.read_text(encoding="utf-8-sig") if path.is_file() else None
    else:
        with ZipFile(labels) as source:
            matches = [item for item in source.infolist() if not item.is_dir()
                       and PurePosixPath(item.filename).name == "occlusions.csv"]
            if len(matches) > 1:
                raise ValueError("Returned archive contains multiple occlusions.csv files")
            content = source.read(matches[0]).decode("utf-8-sig") if matches else None
    if content is None:
        return []
    records = csv.DictReader(io.StringIO(content))
    if tuple(records.fieldnames or ()) != OCCLUSION_COLUMNS:
        raise ValueError("occlusions.csv requires sample,group_id columns")
    return list(records)


def check_return(folder, labels):
    manifest = read_json(folder / "manifest.json")
    frames = manifest["frames"]
    planned = [(clip, split, index, position) for clip, split, indices in planned_clips()
               for position, index in enumerate(indices, 1)]
    received = [(row["clip"], row["split"], row["annotation_id"], row["clip_frame_in_video"]) for row in frames]
    if (manifest["video_sha256"] != sha_file(VIDEO) or manifest["crop_xywh"] != list(CROP)
            or manifest["sample_count"] != len(frames) or planned != received
            or any(row["sample"] != f"{row['clip']}-{row['clip_frame_in_video']:03d}"
                   or row["source_frame_zero_based"] != row["annotation_id"] + 148 for row in frames)):
        raise ValueError("Review manifest differs from the planned, unsealed SNL clips")
    review = read_review(labels, [row["sample"] for row in frames])
    by_sample = {row["sample"]: row["clip"] for row in frames}
    visible, hidden, tracks = set(), set(), set()
    counts = Counter()
    for row in frames:
        sample, clip = row["sample"], row["clip"]
        png = folder / "frames" / f"{sample}.png"
        if sha_file(png) != row["image_sha256"]:
            raise ValueError(f"Packaged PNG has changed: {sample}")
        data = review[f"{sample}.json"]
        review_boxes(data, png)
        for shape in data["shapes"]:
            group_id = shape.get("group_id")
            if type(group_id) is not int or group_id <= 0:
                raise ValueError(f"{sample}: each rectangle requires a positive integer Group ID")
            key = sample, group_id
            if key in visible:
                raise ValueError(f"{sample}: duplicate Group ID {group_id}")
            flags = shape.get("flags") or {}
            if not isinstance(flags, dict) or type(flags.get("occluded", False)) is not bool:
                raise ValueError(f"{sample}: occluded must be a boolean shape flag")
            visible.add(key)
            tracks.add((clip, group_id))
            counts[shape["label"]] += 1
            counts["partially_occluded"] += flags.get("occluded", False)
    for record in returned_occlusions(labels):
        sample, raw_id = record["sample"], record["group_id"]
        if raw_id is None or not raw_id.isascii() or not raw_id.isdecimal():
            raise ValueError("occlusions.csv Group IDs must be positive integers")
        group_id = int(raw_id)
        key = sample, group_id
        if (sample not in by_sample or (by_sample[sample], group_id) not in tracks
                or key in visible or key in hidden):
            raise ValueError(f"Invalid full occlusion for {sample}, Group ID {group_id}")
        hidden.add(key)
    result = dict(frames=len(frames), tracks=len(tracks), shadow=counts["shadow"], uncertain=counts["uncertain"],
                  partially_occluded=counts["partially_occluded"], fully_occluded=len(hidden))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=FOLDER)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--labels", type=Path, help="Returned LabelMe directory or ZIP archive")
    args = parser.parse_args()
    if args.labels:
        check_return(args.folder.resolve(), args.labels.resolve())
    else:
        from infer_video import find_ffmpeg

        export(args.folder.resolve(), find_ffmpeg(args.ffmpeg))
