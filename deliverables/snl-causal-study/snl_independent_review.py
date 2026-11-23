"""Export unmarked SNL review frames and compare returned LabelMe rectangles."""
import argparse
import base64
import csv
import io
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from snl_inputs import BLIND, DEFAULT_OUTPUT, HEIGHT, PROFILES, ROOT, VIDEO, WIDTH, load_gray, load_labels, read_json, sha_file, write_csv, write_json

sys.path.insert(0, str(ROOT))
from videosar.snl_causal import detection_metrics, matched_boxes


SAMPLE_COUNTS = (6, 4, 8)
CONTEXT = 10
REPLAY = ROOT / "runs/snl-video-prefix-v2"
INSTRUCTIONS = """SNL 视频 SAR 独立复标

请独立查看 frames 内的目标帧。context 中同名 MP4 显示目标前后的短片，已慢速播放。
`manifest.csv` 的 target_frame_in_clip 给出目标帧在短片中的位置，从 1 开始数。
PNG 是原生 660×720 灰度图。请在 LabelMe 打开 PNG 并对该帧绘制矩形。

* 用 shadow 标记能通过连续帧确认的、由移动车辆产生的阴影，框住当前帧可见暗区。
* 用 uncertain 标记不确定是移动车辆阴影还是静态暗区的区域，保留这种不确定性。
* 车辆亮回波、静态建筑阴影和道路本身不作为目标。某帧无目标时也请保存空 JSON。
* 请保持文件同名，如 review-01.png 对应 review-01.json，并交回 JSON 文件夹。

本包没有模型预测和其他标注者的框。数据用于比较不同标注口径。
视频来源 Sandia National Laboratories，Eubank Gate and Traffic VideoSAR。
https://www.sandia.gov/app/uploads/sites/124/2021/08/eubankgateandtrafficvideosar.mp4
"""


def export(output, folder, ffmpeg, *, causal=False):
    gray = load_gray(output, ffmpeg)
    for name in ("frames", "context"):
        (folder / name).mkdir(parents=True, exist_ok=True)
    sealed = {i for split in BLIND.values() for block in split.values() for i in block}
    rows = []
    for split, ids, count in zip(("training", "calibration", "development"), PROFILES["prospective"], SAMPLE_COUNTS, strict=True):
        positions = np.linspace(ids[0] + 1, ids[-1] - 1, count).round().astype(int)
        for index in positions:
            first = max(2, index - (2 * CONTEXT if causal else CONTEXT))
            last = index if causal else index + CONTEXT
            if sealed.intersection(range(first, last + 1)):
                raise ValueError("A review window overlaps a sealed test interval")
            tag = f"review-{len(rows) + 1:02d}"
            png = folder / "frames" / f"{tag}.png"
            mp4 = folder / "context" / f"{tag}.mp4"
            Image.fromarray(gray[index - 2]).save(png)
            frames = gray[first - 2:last - 1]
            quality = ["-crf", "0", "-vf", "scale=in_range=full:out_range=full", "-color_range", "pc"] if causal else ["-crf", "19"]
            subprocess.run([ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pixel_format", "gray", "-video_size",
                            f"{WIDTH}x{HEIGHT}", "-framerate", "8", "-i", "pipe:0", "-c:v", "libx264", "-preset",
                            "fast", *quality, "-pix_fmt", "yuv420p", str(mp4)], input=frames.tobytes(), check=True)
            with Image.open(png) as check:
                np.testing.assert_array_equal(np.asarray(check), gray[index - 2])
            decoded = subprocess.check_output([ffmpeg, "-v", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"])
            restored = np.frombuffer(decoded, dtype=np.uint8).reshape(-1, HEIGHT, WIDTH)
            if causal:
                np.testing.assert_array_equal(restored, frames, err_msg=f"Review context {tag}")
            elif len(restored) != len(frames) or np.mean(np.abs(restored[index - first].astype(np.int16) - gray[index - 2])) > 4:
                raise ValueError(f"Review context did not reproduce frame {index}")
            rows.append(dict(sample=tag, split=split, annotation_id=int(index), source_frame_zero_based=int(index + 148),
                             target_frame_in_clip=int(index - first + 1), image_sha256=sha_file(png)))
    write_csv(folder / "manifest.csv", rows)
    policy = "past_and_current" if causal else "bidirectional"
    write_json(folder / "manifest.json", dict(video_sha256=sha_file(VIDEO), context_policy=policy,
                                             sample_count=len(rows), frames=rows))
    instructions = INSTRUCTIONS
    if causal:
        instructions = instructions.replace("显示目标前后的短片", "只显示目标帧及此前的画面，末帧即为目标帧")
        instructions = instructions.replace("\n\n* 用 shadow", "\n请按文件编号顺序复标。先保存当前帧的结果，再打开下一帧。\n\n* 用 shadow")
    (folder / "README.txt").write_text(instructions, encoding="utf-8-sig")
    archive = folder.with_suffix(".zip")
    files = [folder / name for name in ("README.txt", "manifest.csv", "manifest.json")]
    files.extend(folder / kind / f"{row['sample']}.{extension}" for row in rows
                 for kind, extension in (("frames", "png"), ("context", "mp4")))
    with ZipFile(archive, "w", ZIP_DEFLATED) as output_zip:
        for file in files:
            output_zip.write(file, file.relative_to(folder))
    with ZipFile(archive) as output_zip:
        if output_zip.testzip() is not None or len(output_zip.namelist()) != len(files):
            raise ValueError("Review archive verification failed")
    print(f"{len(rows)} verified review frames with no sealed overlap: {archive}", flush=True)


def read_review(source, samples):
    expected = {f"{sample}.json" for sample in samples}
    if source.is_dir():
        found = {name: read_json(source / name) for name in sorted(expected) if (source / name).is_file()}
    else:
        with ZipFile(source) as archive:
            found = {}
            for member in archive.infolist():
                name = PurePosixPath(member.filename).name
                if member.is_dir() or name not in expected:
                    continue
                if name in found:
                    raise ValueError(f"Duplicate review file in archive: {name}")
                found[name] = json.loads(archive.read(member).decode("utf-8-sig"))
    if missing := sorted(expected - found.keys()):
        raise ValueError(f"Missing review files, including empty annotations: {', '.join(missing)}")
    return found


def review_boxes(data, png):
    if (data["imageWidth"], data["imageHeight"]) != (WIDTH, HEIGHT) or Path(data["imagePath"]).name != png.name:
        raise ValueError(f"LabelMe frame mismatch: {png.name}")
    if data.get("imageData"):
        with Image.open(io.BytesIO(base64.b64decode(data["imageData"]))) as embedded, Image.open(png) as expected:
            if not np.array_equal(np.asarray(embedded.convert("L")), np.asarray(expected)):
                raise ValueError(f"Embedded image does not match original: {png.name}")
    boxes = {"shadow": [], "uncertain": []}
    for shape in data["shapes"]:
        if shape["shape_type"] != "rectangle" or shape["label"] not in boxes:
            raise ValueError(f"Unexpected shape or label in {png.name}")
        points = np.asarray(shape["points"], dtype=np.float32).reshape(2, 2)
        low, high = points.min(axis=0), points.max(axis=0)
        if not np.isfinite(points).all() or np.any(low < 0) or np.any(high > [WIDTH, HEIGHT]) or np.any(high <= low):
            raise ValueError(f"Invalid rectangle in {png.name}")
        boxes[shape["label"]].append(np.concatenate((low, high)))
    return np.asarray(boxes["shadow"], dtype=np.float32).reshape(-1, 4), len(boxes["uncertain"])


def load_predictions(replay, source_hash, frames):
    metadata = read_json(replay / "video.json")
    receipt = read_json(replay / "development-replay.json")
    if metadata["source_sha256"] != source_hash or metadata["crop_xywh"] != [310, 0, WIDTH, HEIGHT]:
        raise ValueError("The model replay used a different video or radar crop")
    if metadata["model_sha256"] != receipt["model_sha256"]:
        raise ValueError("The model replay checkpoint does not match its verification receipt")
    needed = {row["source_frame_zero_based"] for row in frames}
    for name in ("frames.csv", "predictions.csv"):
        if sha_file(replay / name) != receipt["matching_artifact_sha256"][name]:
            raise ValueError(f"Verified replay file changed: {name}")
    with (replay / "frames.csv").open(newline="", encoding="utf-8-sig") as file:
        counts = {int(row["source_frame"]): int(row["detections"]) for row in csv.DictReader(file)}
    if not needed <= counts.keys():
        raise ValueError("The verified replay does not cover every reviewed frame")
    boxes = {i: [] for i in needed}
    with (replay / "predictions.csv").open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            i = int(row["source_frame"])
            if i in boxes:
                boxes[i].append([float(row[key]) for key in ("x0_px", "y0_px", "x1_px", "y1_px")])
    if any(len(found) != counts[i] for i, found in boxes.items()):
        raise ValueError("Reviewed frame prediction counts do not match the verified replay")
    return {i: np.asarray(found, dtype=np.float32).reshape(-1, 4) for i, found in boxes.items()}, metadata["model_sha256"]


def model_counts(predictions, independent, community):
    first, second = [set(matched_boxes(predictions, targets)[:, 0]) for targets in (independent, community)]
    return dict(prediction_count=len(predictions), model_iou50_independent=len(first), model_iou50_community=len(second),
                model_agreed_matches=len(first & second), model_independent_only=len(first - second),
                model_community_only=len(second - first), model_unmatched=len(predictions) - len(first | second))


def summarize_agreement(rows):
    totals = {key: sum(row[key] for row in rows) for key in rows[0] if key not in ("sample", "split", "annotation_id")}
    agreement = {}
    for rule in ("iou50", "center5"):
        hits = totals[f"{rule}_matches"]
        agreement[rule] = detection_metrics((hits, totals["independent"] - hits, totals["community"] - hits), len(rows))
    metrics = {}
    for label in ("independent", "community"):
        hits = totals[f"model_iou50_{label}"]
        metrics[label] = detection_metrics((hits, totals["prediction_count"] - hits, totals[label] - hits), len(rows))
    return dict(totals, annotation_agreement=agreement, model_metrics=metrics)


def compare(output, folder, labels, ffmpeg, *, replay=REPLAY):
    manifest = read_json(folder / "manifest.json")
    frames = manifest["frames"]
    if manifest["video_sha256"] != sha_file(VIDEO):
        raise ValueError("Review package belongs to a different source video")
    planned = set().union(*PROFILES["prospective"])
    ids = [row["annotation_id"] for row in frames]
    if (len(set(ids)) != len(ids) or not set(ids) <= planned
            or any(row["source_frame_zero_based"] != row["annotation_id"] + 148 for row in frames)):
        raise ValueError("Unexpected or sealed frame in review manifest")
    community = load_labels(output, ids, load_gray(output, ffmpeg))[0]
    reviews = read_review(labels, [row["sample"] for row in frames])
    predictions, model = load_predictions(replay, manifest["video_sha256"], frames)
    results = []
    for row in frames:
        tag, index = row["sample"], row["annotation_id"]
        png = folder / "frames" / f"{tag}.png"
        if sha_file(png) != row["image_sha256"]:
            raise ValueError(f"Package image checksum changed: {tag}")
        independent, uncertain = review_boxes(reviews[f"{tag}.json"], png)
        known = community[index]
        tp = len(matched_boxes(independent, known))
        separation = np.linalg.norm(((independent[:, :2] + independent[:, 2:]) / 2)[:, None]
                                    - (known[:, :2] + known[:, 2:]) / 2, axis=2)
        valid = separation <= 5
        a, b = linear_sum_assignment(valid, maximize=True)
        results.append(dict(sample=tag, split=row["split"], annotation_id=index, independent=len(independent),
                            community=len(known), iou50_matches=tp, center5_matches=int(valid[a, b].sum()),
                            independent_only=len(independent) - tp, community_only=len(known) - tp, uncertain=uncertain,
                            **model_counts(predictions[row["source_frame_zero_based"]], independent, known)))
    write_csv(folder / "agreement.csv", results)
    summary = {"model_sha256": model, "context_policy": manifest.get("context_policy", "bidirectional"),
               "overall": summarize_agreement(results)}
    for split in ("training", "calibration", "development"):
        rows = [row for row in results if row["split"] == split]
        summary[split] = summarize_agreement(rows)
    write_json(folder / "agreement.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--labels", type=Path, help="Returned LabelMe directory or ZIP archive")
    parser.add_argument("--replay", type=Path, default=REPLAY, help="Verified frozen-model replay directory")
    parser.add_argument("--causal-context", action="store_true", help="End each review clip on its target frame")
    args = parser.parse_args()
    folder = args.folder or DEFAULT_OUTPUT / ("independent-review-causal" if args.causal_context else "independent-review")
    if args.labels is None:
        export(args.output.resolve(), folder.resolve(), args.ffmpeg, causal=args.causal_context)
    else:
        compare(args.output.resolve(), folder.resolve(), args.labels.resolve(), args.ffmpeg, replay=args.replay.resolve())
