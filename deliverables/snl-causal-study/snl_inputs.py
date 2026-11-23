"""Read, verify, and decode the locked SNL Eubank sequence for this study."""
import base64
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "runs/snl-causal-study"
PUBLIC_INPUTS = ROOT / "runs/snl-eubank-check"
VIDEO = PUBLIC_INPUTS / "eubank-snl.mp4"
COMMIT = "28a6ee2cff7913d817816c01589e46c862927df0"
WEIGHTS_FILENAME = "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
TRAIN = tuple(range(2, 219))
VALIDATE = tuple(range(242, 309))
DEVELOPMENT = tuple(range(332, 380))
PROFILES = {
    "original": (TRAIN, VALIDATE, DEVELOPMENT),
    "prospective": (tuple(range(2, 297)), tuple(range(310, 324)), DEVELOPMENT),
}
BLIND = {
    "first": {"398-439": tuple(range(398, 440)), "638-679": tuple(range(638, 680))},
    "reserve": {"517-547": tuple(range(517, 548)), "691-721": tuple(range(691, 722))},
}
WIDTH, HEIGHT, TOTAL = 660, 720, 899


def sha_file(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_json(path, value):
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.stem}-", suffix=".json.tmp",
                            encoding="utf-8", delete=False) as file:
        file.write(content)
        temporary = Path(file.name)
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_gray(output, ffmpeg):
    path = output / "eubank-gray-660x720.raw"
    if not path.exists() and (DEFAULT_OUTPUT / path.name).is_file():
        path = DEFAULT_OUTPUT / path.name
    if path.is_file():
        if path.stat().st_size != TOTAL * WIDTH * HEIGHT:
            raise ValueError(f"Unexpected grayscale cache size: {path}")
        return np.memmap(path, dtype=np.uint8, mode="r", shape=(TOTAL, HEIGHT, WIDTH))
    temp = path.with_suffix(".partial")
    subprocess.run([
        ffmpeg, "-y", "-v", "error", "-i", str(VIDEO), "-vf",
        rf"select=between(n\,150\,{150 + TOTAL - 1}),crop={WIDTH}:{HEIGHT}:310:0,format=gray",
        "-fps_mode", "passthrough", "-threads", "1", "-f", "rawvideo", str(temp),
    ], check=True)
    if temp.stat().st_size != TOTAL * WIDTH * HEIGHT:
        raise ValueError(f"The SNL video did not decode to the expected {TOTAL} SAR frames")
    temp.replace(path)
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(TOTAL, HEIGHT, WIDTH))


def blob_table():
    tree = read_json(PUBLIC_INPUTS / "jiang-tree.json")
    if tree["sha"] != COMMIT:
        raise ValueError("Unexpected Git commit")
    return {Path(item["path"]).stem: item["sha"] for item in tree["tree"]
            if item["path"].startswith("data/json/") and item["path"].endswith(".json")}


def label_path(output, name):
    locations = (output / "labels", DEFAULT_OUTPUT / "labels", PUBLIC_INPUTS / "factorial-labels",
                 ROOT / "runs/snl-domain-ablation/labels")
    return [location / f"{name}.json" for location in locations] + [PUBLIC_INPUTS / f"recovered-{name}.json"]


def verified_bytes(output, name, blob):
    for path in label_path(output, name):
        if not path.is_file():
            continue
        data = path.read_bytes()
        if hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == blob:
            return data
    return None


def missing_labels(output, ids):
    blobs = blob_table()
    return [index for index in ids if index > 70 and
            verified_bytes(output, f"{index:06}", blobs[f"{index:06}"]) is None]


def thumbnail(image):
    resized = Image.fromarray(image).resize((644, 704), Image.Resampling.BILINEAR)
    return np.asarray(resized.resize((80, 88), Image.Resampling.BILINEAR))


def correlation(left, right):
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return -1.0
    return float(np.corrcoef(left.ravel(), right.ravel())[0, 1])


def format_boxes(rectangles):
    points = np.asarray(rectangles, dtype=np.float32).reshape(-1, 2, 2)
    scale = np.array([WIDTH / 644, HEIGHT / 704], dtype=np.float32)
    boxes = np.column_stack((points.min(axis=1) * scale, points.max(axis=1) * scale))
    if np.any(boxes[:, 2:] <= boxes[:, :2]) or np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2:] > [WIDTH, HEIGHT]):
        raise ValueError("Source rectangles are invalid in the native crop")
    return boxes


def load_labels(output, ids, gray, *, allow_partial=False):
    blobs = blob_table()
    archived = {row["annotation_index"]: row for row in read_json(PUBLIC_INPUTS / "box-labels.json")}
    labels, audits, missing = {}, [], []
    for index in ids:
        key = f"{index:06}"
        blob = blobs[key]
        if index in archived:
            record = archived[index]
            if record["blob_sha1"] != blob or record["video_frame_index"] != index + 148:
                raise ValueError(f"{index}: prior provenance did not match")
            rectangles = record["rectangles"]
            with Image.open(PUBLIC_INPUTS / "label-reference" / f"{key}.png") as reference:
                ref = np.asarray(reference.convert("L"))
        else:
            source = verified_bytes(output, key, blob)
            if source is None:
                missing.append(index)
                continue
            record = json.loads(source.decode("utf-8-sig"))
            if (record["imageWidth"], record["imageHeight"]) != (644, 704):
                raise ValueError(f"{index}: wrong reference size")
            if any(s["shape_type"] != "rectangle" or s["label"] != "car" for s in record["shapes"]):
                raise ValueError(f"{index}: wrong annotation type")
            rectangles = [s["points"] for s in record["shapes"]]
            with Image.open(io.BytesIO(base64.b64decode(record["imageData"]))) as reference:
                ref = np.asarray(reference.convert("L").resize((80, 88), Image.Resampling.BILINEAR))
        current = index - 2
        neighbors = range(max(0, current - 2), min(TOTAL, current + 3))
        correlations = {j: correlation(ref, thumbnail(gray[j])) for j in neighbors}
        gap = max(correlations.values()) - correlations[current]
        if correlations[current] < 0.96 or gap > 0.006:
            raise ValueError(f"{index}: frame alignment failed ({correlations})")
        labels[index] = format_boxes(rectangles)
        audits.append({"annotation_id": index, "source_index": index + 148, "blob_sha1": blob,
                       "correlation": correlations[current], "deficit_to_best_neighbor": gap,
                       "boxes": len(labels[index])})
    if missing and not allow_partial:
        raise FileNotFoundError(f"Missing or invalid annotated frames: {missing}")
    return labels, audits, missing


def feature_cache(output, gray, high, scan):
    """Generate a causal cache through (and including) the supplied annotation ID."""
    variant = "" if scan.profile == "raw" else f"-{scan.profile}"
    path = output / f"temporal{variant}-through-{high}.npy"
    if not path.exists() and (DEFAULT_OUTPUT / path.name).is_file():
        path = DEFAULT_OUTPUT / path.name
    shape = (high - 1, scan.channel_count, HEIGHT, WIDTH)
    if path.exists():
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        if data.shape != shape or data.dtype != np.uint8:
            raise ValueError(f"Bad feature cache: {path}")
        return data
    temp = path.with_suffix(".partial.npy")
    data = np.lib.format.open_memmap(temp, mode="w+", dtype=np.uint8,
                                    shape=shape)
    for offset in range(high - 1):
        channels = scan.network_channels(gray[offset])
        data[offset] = np.rint(channels * 255).astype(np.uint8)
    data.flush()
    del data
    temp.replace(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)
