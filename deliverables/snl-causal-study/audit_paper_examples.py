"""Compare the illustrative outlines in DSENet Figure 2 with early public frame labels."""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, ROOT, WIDTH, correlation, load_gray, load_labels, sha_file
from snl_inputs import thumbnail, write_json

sys.path.insert(0, str(ROOT))

from videosar.snl_causal import matched_boxes

PANELS = ((29, 35, 1210, 1326), (1349, 35, 2531, 1326), (2670, 35, 3851, 1326))
FIGURE = "https://www.mdpi.com/2072-4292/18/10/1623"


def outlines(panel):
    r, g, b = np.moveaxis(panel, -1, 0)
    colors = {"yellow": (r > 175) & (g > 155) & (b < 115), "red": (r > 150) & (g < 110) & (b < 110)}
    result = {}
    height, width = panel.shape[:2]
    scale = [WIDTH / width, HEIGHT / height, WIDTH / width, HEIGHT / height]
    for name, mask in colors.items():
        groups, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        rectangles = []
        for index, bounds in enumerate(ndimage.find_objects(groups), 1):
            y, x = bounds
            height_px, width_px = y.stop-y.start, x.stop-x.start
            if min(height_px, width_px) < 12:
                continue
            # Large yellow rectangular traces are distinct from the yellow caption glyphs.
            if name == "yellow" and (height_px < 40 or width_px < 28):
                continue
            component = groups[bounds] == index
            coverage = (component[:6, :].any(axis=0).mean(), component[-6:, :].any(axis=0).mean(),
                        component[:, :6].any(axis=1).mean(), component[:, -6:].any(axis=1).mean())
            if min(coverage) >= 0.70:
                rectangles.append((x.start, y.start, x.stop, y.stop))
        result[name] = np.asarray(rectangles, dtype=np.float32).reshape(-1, 4) * scale
    return result


def align(panel, observations):
    small = np.asarray(Image.fromarray(panel).convert("L").resize((80, 88), Image.Resampling.BILINEAR))
    color = np.ptp(panel, axis=-1) > 22
    downsampled = Image.fromarray(color.astype(np.uint8) * 255).resize((80, 88), Image.Resampling.BILINEAR)
    clear = np.asarray(downsampled) < 12
    ranking = [(index, correlation(small[clear], reference[clear])) for index, reference in observations.items()]
    return sorted(ranking, key=lambda item: item[1], reverse=True)[:4]


def compare(boxes, truth):
    left = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    right = (truth[:, :2] + truth[:, 2:]) * 0.5
    near = np.linalg.norm(left[:, None] - right[None, :], axis=-1) <= 10
    row, column = linear_sum_assignment(near, maximize=True)
    return {"matches_iou50": len(matched_boxes(boxes, truth)),
            "matches_iou25": len(matched_boxes(boxes, truth, threshold=0.25)),
            "centers_matched_within_10_px": int(near[row, column].sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, default=DEFAULT_OUTPUT / "paper-reference/dsenet-figure2.png")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    source = args.figure.resolve()
    output = args.output.resolve()
    gray = load_gray(output, args.ffmpeg)
    with Image.open(source) as image:
        if image.size != (3876, 1521):
            raise ValueError("The expected source is the full-size DSENet Figure 2")
        figure = np.asarray(image.convert("RGB"))
    observations = {i: thumbnail(gray[i-2]) for i in PROFILES["prospective"][0]}
    evidence = []
    for number, (left, top, right, bottom) in zip((1, 90, 150), PANELS, strict=True):
        panel = figure[top:bottom, left:right]
        matches = align(panel, observations)
        index, similarity = matches[0]
        if similarity < 0.92:
            raise ValueError(f"Figure panel {number} could not be aligned to an early training frame")
        indices = [item[0] for item in matches]
        labels, _, _ = load_labels(output, indices, gray)
        baseline = labels[index]
        components = outlines(panel)
        match = {hue: {"count": len(boxes), **compare(boxes, baseline), "outlined_boxes_xyxy": boxes.tolist(),
                       "other_alignment_matches": {candidate: compare(boxes, labels[candidate])
                                                   for candidate in indices[1:]}}
                 for hue, boxes in components.items()}
        row = {"paper_frame": number, "matched_label_id": index, "mp4_zero_based_index": index + 148,
               "top_alignments": matches, "public_label_boxes": len(baseline), "outlines": match}
        evidence.append(row)
        print(number, index, round(similarity, 4), len(baseline),
              {hue: info["matches_iou50"] for hue, info in match.items()}, flush=True)
    write_json(source.with_name("dsenet-figure2-label-agreement.json"),
               {"paper": FIGURE, "figure_sha256": sha_file(source),
                "yellow_minimum_figure_width_height": [28, 40], "panels": evidence})


if __name__ == "__main__":
    main()
