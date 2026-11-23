import csv

import numpy as np
from PIL import Image, ImageDraw


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gray(frame, limits):
    lo, hi = limits
    return Image.fromarray(np.rint(np.clip((frame - lo) / max(float(hi - lo), 1e-8), 0, 1) * 255).astype(np.uint8))


def save_point(path, power, check):
    image = gray(np.log1p(power * 100), (0, np.log1p(power.max() * 100))).convert("RGB")
    image = image.resize((image.width * 4, image.height * 4), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    y, x = check["point_expected_yx"]
    draw.ellipse((x * 4 - 3, y * 4 - 3, x * 4 + 7, y * 4 + 7), outline="#31dbb0", width=1)
    image.save(path)


def save_preview(path, frames, truth, prediction, tracked, indices):
    h, w = frames.shape[1:]
    zoom = min(2, 256 / w, 192 / h)
    panel_w, panel_h = max(1, round(w * zoom)), max(1, round(h * zoom))
    gap, title = 10, 22
    column_width = max(panel_w, 110)
    panels = ["Input", "Ground truth", "Prediction", "Track overlay"]
    image = Image.new("RGB", (len(panels) * (column_width + gap) + gap,
                             title + len(indices) * (panel_h + gap + title)), "#121b22")
    draw = ImageDraw.Draw(image)
    limits = np.percentile(frames, [2, 98])
    if limits[1] <= limits[0]:
        limits[0] = min(0, float(limits[0]) - 1)
    for column, label in enumerate(panels):
        draw.text((gap + column * (column_width + gap), 4), label, fill="white")
    for row, index in enumerate(indices):
        top = title + row * (panel_h + gap + title)
        base = gray(frames[index], limits).convert("RGB")
        overlay = base.copy()
        painter = ImageDraw.Draw(overlay)
        last = None
        for entry in tracked["frames"][:index + 1]:
            point = entry["xy_px"]
            if point is None:
                last = None
                continue
            x, y = point
            color = "#ffca76" if entry["source"] == "imputed" else "#31dbb0"
            if last and last["segment"] == entry["segment"]:
                painter.line([tuple(last["xy_px"]), (x, y)], fill=color, width=1)
            if entry["frame"] == index:
                painter.ellipse((x - 2, y - 2, x + 2, y + 2), outline=color)
            last = entry
        for column, panel in enumerate((base, gray(truth[index], (0, 1)), gray(prediction[index], (0, 1)), overlay)):
            image.paste(panel.resize((panel_w, panel_h), Image.Resampling.NEAREST),
                        (gap + column * (column_width + gap), top))
        draw.text((gap, top + panel_h + 2), f"Frame {index}", fill="white")
    image.save(path)
