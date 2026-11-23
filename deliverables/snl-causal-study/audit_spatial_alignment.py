"""Check the spatial mapping of the annotation reference images to the official crop."""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import optimize, signal

from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, WIDTH, blob_table, load_gray, load_labels, read_json, verified_bytes, write_csv, write_json


def local_registration(reference, official, x, y, *, half=17, radius=14):
    patch = reference[y-half:y+half, x-half:x+half].copy()
    patch -= patch.mean()
    energy = float(np.square(patch).sum())
    if energy < 34**2 * 5**2:
        return None
    neighborhood = official[y-half-radius:y+half+radius, x-half-radius:x+half+radius]
    integral = np.pad(neighborhood, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    squared = np.pad(neighborhood**2, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    size = 2 * half
    total = integral[size:, size:] + integral[:-size, :-size] - integral[size:, :-size] - integral[:-size, size:]
    norm = squared[size:, size:] + squared[:-size, :-size] - squared[size:, :-size] - squared[:-size, size:]
    norm = np.sqrt(np.maximum(norm - total**2 / patch.size, 1) * energy)
    correlation = signal.correlate(neighborhood, patch, mode="valid", method="fft") / norm
    row, col = np.unravel_index(correlation.argmax(), correlation.shape)
    best = float(correlation[row, col])
    if best < 0.85 or not (0 < row < 2*radius and 0 < col < 2*radius):
        return None
    origin = np.asarray((row, col), dtype=float)
    for axis in range(2):
        axis_step = np.eye(2, dtype=int)[axis]
        a, b = [float(correlation[tuple(np.asarray((row, col)) + sign * axis_step)]) for sign in (-1, 1)]
        curvature = a - 2*best + b
        if curvature < -1e-8:
            origin[axis] += np.clip(0.5 * (a - b) / curvature, -0.5, 0.5)
    return (*tuple(origin[::-1] - radius), float(correlation[radius, radius]), best)


def reference_image(output, index, blobs):
    tag = f"{index:06}"
    source = verified_bytes(output, tag, blobs[tag])
    if source is None:
        raise FileNotFoundError(f"Missing verified reference image {tag}")
    record = json.loads(source.decode("utf-8-sig"))
    with Image.open(io.BytesIO(base64.b64decode(record["imageData"]))) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def gather(output, gray, ids, blobs):
    records = []
    for index in ids:
        reference = reference_image(output, index, blobs)
        height, width = reference.shape
        official = np.asarray(Image.fromarray(gray[index-2]).resize((width, height), Image.Resampling.BILINEAR),
                              dtype=np.float32)
        for y in range(48, height - 32, 84):
            for x in range(48, width - 32, 84):
                match = local_registration(reference, official, x, y)
                if match is not None:
                    records.append((index, x, y, *match))
    return np.asarray(records, dtype=np.float64).reshape(-1, 7)


def analyze(records):
    displacement = records[:, 3:5]
    coords = (records[:, 1:3] - [322, 352]) / [644, 704]
    design = np.column_stack((np.ones(len(coords)), coords))

    def residual(params):
        return (design @ params.reshape(3, 2) - displacement).ravel()

    best = optimize.least_squares(residual, np.zeros(6), loss="soft_l1").x.reshape(3, 2)
    remaining = np.linalg.norm(design @ best - displacement, axis=1)
    return {"reference_frames": len(set(records[:, 0])), "patches": len(records),
            "nominal_ncc_median": float(np.median(records[:, 5])),
            "local_best_ncc_median": float(np.median(records[:, 6])),
            "local_offset_px_q50_q90": np.percentile(np.linalg.norm(displacement, axis=1), [50, 90]).tolist(),
            "affine_coefficients_bias_normalized_x_normalized_y": best.tolist(),
            "affine_offset_at_corners_px": (np.asarray([[1, x, y] for y in (-0.5, 0.5) for x in (-0.5, 0.5)])
                                             @ best).tolist(),
            "affine_residual_px_q50_q90": np.percentile(remaining, [50, 90]).tolist()}


def audit_targets(output, gray, blobs):
    from experiment import filter_by_config, load_candidates, summarize, with_nms
    from snl_bytetrack import REFERENCE
    from videosar.snl_causal import matched_boxes

    release = read_json(output / REFERENCE)
    config = release["config"]
    _, calibration, development = PROFILES["prospective"]
    rows, report = [], {}
    for split, ids in (("validation", calibration), ("development", development)):
        labels = load_labels(output, ids, gray)[0]
        predicted = filter_by_config(with_nms(load_candidates(output, config, split), config["nms"]), config)
        translated = {}
        offsets = []
        for i, boxes in labels.items():
            reference = reference_image(output, i, blobs)
            h, w = reference.shape
            scale = np.array([WIDTH / w, HEIGHT / h])
            official = np.asarray(Image.fromarray(gray[i - 2]).resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)
            translated[i] = boxes.copy()
            for j, box in enumerate(boxes):
                x, y = np.rint((box[:2] + box[2:]) / (2 * scale)).astype(int)
                match = local_registration(reference, official, x, y) if 31 <= x < w - 31 and 31 <= y < h - 31 else None
                row = dict(split=split, annotation_id=i, box_index=j, dx_px=None, dy_px=None, nominal_ncc=None, best_ncc=None)
                if match is not None:
                    shift = np.asarray(match[:2]) * scale
                    offsets.append(shift)
                    row.update(dx_px=float(shift[0]), dy_px=float(shift[1]), nominal_ncc=match[2], best_ncc=match[3])
                    translated[i][j] = (box + np.tile(shift, 2)).clip(0, [WIDTH, HEIGHT, WIDTH, HEIGHT])
                rows.append(row)

        nominal, _ = summarize(predicted, labels)
        counterfactual, _ = summarize(predicted, translated)
        if any(value != release[split][key] for key, value in nominal.items()):
            raise ValueError("The released model did not replay on the original labels")
        distance = np.linalg.norm(offsets, axis=1)
        report[split] = dict(frames=len(labels), labels=sum(map(len, labels.values())), registered_boxes=len(offsets),
                             offset_native_px_q50_q90_q99=np.percentile(distance, (50, 90, 99)).tolist(),
                             over_two_native_pixels=int(np.count_nonzero(distance > 2)),
                             original_box_overlap_below_half=sum(len(boxes) - len(matched_boxes(boxes, translated[i]))
                                                                for i, boxes in labels.items()),
                             original=nominal, local_shift_sensitivity=counterfactual)
        print(split, report[split], flush=True)
    write_csv(output / "reference-target-alignment.csv", rows)
    write_json(output / "reference-target-alignment.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--audit-targets", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    gray = load_gray(output, args.ffmpeg)
    train, validation, development = PROFILES["prospective"]
    blobs = blob_table()
    if args.audit_targets:
        audit_targets(output, gray, blobs)
        return
    choices = {"train": train[79::17], "validation": validation[::2], "development": development[::4]}
    result = {}
    for split, ids in choices.items():
        records = gather(output, gray, ids, blobs)
        result[split] = analyze(records)
        print(split, result[split], flush=True)
    write_json(output / "reference-spatial-alignment.json", result)


if __name__ == "__main__":
    main()
