"""Audit public annotations and attribute each miss of the shipped SNL detector."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from audit_spatial_alignment import gather
from experiment import filter_by_config, load_candidates, summarize, with_nms
from snl_bytetrack import REFERENCE
from snl_edge_refinement import MOVES, geometry_options
from snl_inputs import DEFAULT_OUTPUT, PROFILES, VIDEO, blob_table, load_gray, load_labels, read_json, sha_file
from snl_inputs import verified_bytes, write_csv, write_json
from videosar.snl_causal import box_iou, matched_boxes


CAUSES = ("detected", "not_selected", "box_geometry", "no_matching_candidate")
COLORS = dict(zip(CAUSES[1:], ("#58d4ff", "#ffce58", "#ff7e83"), strict=True))


def extend_matching(overlaps, required):
    """Maximize coverage while keeping earlier-stage truth identities covered."""
    valid = overlaps >= 0.50
    priority = valid * (1 + (len(required) + 1) * required + overlaps / (len(required) + 1))
    boxes, targets = linear_sum_assignment(priority, maximize=True)
    keep = valid[boxes, targets]
    covered = np.zeros(len(required), dtype=bool)
    covered[targets[keep]] = True
    if np.any(required & ~covered):
        raise ValueError("Candidate matching could not preserve an earlier-stage detection")
    return covered


def explain_frame(index, truth, selected, raw):
    found = np.zeros(len(truth), dtype=bool)
    found[matched_boxes(selected, truth)[:, 1]] = True
    original = box_iou(raw, truth)
    adjusted = box_iou(geometry_options(raw).reshape(-1, 4), truth)
    adjusted = adjusted.reshape(len(raw), len(MOVES), len(truth)).max(axis=1)
    recoverable = extend_matching(original, found)
    bounded = extend_matching(adjusted, recoverable)
    categories = np.select((found, recoverable, bounded), CAUSES[:3], default=CAUSES[3])
    rows = []
    for target, (box, cause) in enumerate(zip(truth, categories, strict=True)):
        x0, y0, x1, y1 = map(float, box)
        rows.append(dict(annotation_id=index, source_index=index + 148, target_index=target, cause=cause,
                         x0=x0, y0=y0, x1=x1, y1=y1, width_px=x1-x0, height_px=y1-y0,
                         max_raw_iou=float(original[:, target].max(initial=0)),
                         max_adjusted_iou=float(adjusted[:, target].max(initial=0))))
    return rows


def frame_sheet(gray, records, path, *, show_detail=False):
    tile_width, tile_height = 330, 360
    gutter, header = 10, 23
    columns = 3
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * (tile_width + gutter) + gutter,
                              rows * (tile_height + header + gutter) + gutter), "#151d22")
    draw = ImageDraw.Draw(sheet)
    for number, (label, index, boxes) in enumerate(records):
        left = gutter + number % columns * (tile_width + gutter)
        top = gutter + number // columns * (tile_height + header + gutter)
        draw.text((left, top + 4), f"{label}   id {index}   MP4 {index + 148}", fill="white")
        panel = Image.fromarray(gray[index - 2]).convert("RGB")
        outlines = ImageDraw.Draw(panel)
        for box, color in boxes:
            outlines.rectangle(tuple(box), outline=color, width=2)
        reduced = panel.resize((tile_width, tile_height), Image.Resampling.BILINEAR)
        if show_detail:
            box, color = boxes[0]
            x, y = (np.asarray(box[:2]) + box[2:]) / 2
            x0 = int(np.clip(x - 36, 0, panel.width - 72))
            y0 = int(np.clip(y - 36, 0, panel.height - 72))
            detail = panel.crop((x0, y0, x0 + 72, y0 + 72)).resize((144, 144), Image.Resampling.NEAREST)
            corner = (tile_width - 149, tile_height - 149)
            reduced.paste(detail, corner)
            ImageDraw.Draw(reduced).rectangle((*corner, corner[0] + 143, corner[1] + 143), outline=color)
        sheet.paste(reduced, (left, top + header))
    sheet.save(path)


def audit_nine(output, gray):
    samples, records = [], []
    blobs = blob_table()
    for name, ids in zip(("train", "validation", "development"), PROFILES["prospective"], strict=True):
        chosen = (ids[0], ids[(len(ids) - 1) // 2], ids[-1])
        labels, frames, _ = load_labels(output, chosen, gray)
        for row in frames:
            index = row["annotation_id"]
            boxes = labels[index]
            tag = f"{index:06}"
            has_reference = verified_bytes(output, tag, blobs[tag]) is not None
            patches = gather(output, gray, (index,), blobs) if has_reference else np.empty((0, 7))
            local = np.linalg.norm(patches[:, 3:5], axis=1)
            widths = boxes[:, 2] - boxes[:, 0]
            samples.append(row | {"split": name, "patches": len(patches),
                                  "full_reference": has_reference,
                                  "local_offset_px_q50_q90": np.percentile(local, (50, 90)).tolist() if len(local) else None,
                                  "box_width_px_range": [float(widths.min()), float(widths.max())] if len(widths) else None})
            records.append((name, index, [(box, "#ffce58") for box in boxes]))
    frame_sheet(gray, records, output / "annotation-audit-nine.png")
    write_json(output / "annotation-audit-nine.json",
               {"video_sha256": sha_file(VIDEO), "crop_xywh": [310, 0, 660, 720],
                "annotation_canvas_wh": [644, 704], "samples": samples})
    return samples


def audit_misses(output, gray):
    reference = read_json(output / REFERENCE)
    config = reference["config"]
    truth = load_labels(output, PROFILES["prospective"][2], gray)[0]
    raw = load_candidates(output, config, "development")
    selected = filter_by_config(with_nms(raw, config["nms"]), config)
    if summarize(selected, truth)[0] != reference["development"]:
        raise ValueError("The candidate cache does not reproduce the shipped detector")
    targets = [row for index, boxes in truth.items()
               for row in explain_frame(index, boxes, selected[index][0], raw[index][0])]
    totals = Counter(row["cause"] for row in targets)
    summary, examples = {}, []
    for cause in CAUSES:
        members = [row for row in targets if row["cause"] == cause]
        if not members:
            continue
        width, height = (np.median([row[key] for row in members]) for key in ("width_px", "height_px"))
        representative = min(members, key=lambda row: (abs(row["width_px"] - width) + abs(row["height_px"] - height),
                                                        row["annotation_id"], row["target_index"]))
        summary[cause] = {"targets": totals[cause], "median_wh_px": [float(width), float(height)],
                          "representative": representative}
        if cause != "detected":
            examples.append((cause, representative["annotation_id"],
                             [([representative[key] for key in ("x0", "y0", "x1", "y1")], COLORS[cause])]))
    frame_sheet(gray, examples, output / "development-miss-examples.png", show_detail=True)
    write_csv(output / "development-miss-attribution.csv", targets)
    write_json(output / "development-miss-attribution.json",
               {"config": config, "match_iou": 0.50, "reference": reference["development"], "groups": summary})
    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    output = args.output.resolve()
    gray = load_gray(output, args.ffmpeg)
    print("Audited annotation IDs:", [row["annotation_id"] for row in audit_nine(output, gray)])
    print("Target counts by outcome:", dict(audit_misses(output, gray)))
