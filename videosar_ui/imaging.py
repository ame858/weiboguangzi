"""Display-only radar image enhancement for the Streamlit GUI.

Everything in this module transforms images *for human eyes only*. The arrays
fed to ``videosar detect`` are never modified. Only NumPy and Pillow are used,
so the helpers are unit-testable without Streamlit.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# Match videosar.visuals.save_preview: measured = cyan, imputed = orange.
MEASURED_COLOR = (49, 219, 176)
IMPUTED_COLOR = (255, 202, 118)
TRUTH_COLOR = (64, 235, 120)
PREDICTION_COLOR = (255, 92, 92)
CROP_COLOR = (255, 64, 64)

MODES = ("percentile", "log_db", "linear")
MODE_LABELS = {
    "percentile": "2-98 百分位对比度拉伸（默认）",
    "log_db": "对数 dB 压缩",
    "linear": "原始线性强度",
}
CMAPS = ("gray", "magma", "viridis")

# Approximate matplotlib magma / viridis anchors, so no matplotlib dependency
# is needed. Interpolated to a 256-entry LUT below.
_ANCHORS = {
    "gray": [(0, 0, 0), (255, 255, 255)],
    "magma": [(0, 0, 4), (40, 17, 76), (115, 25, 105), (188, 55, 105),
              (245, 108, 81), (252, 190, 77), (252, 253, 191)],
    "viridis": [(68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
                (38, 130, 142), (53, 183, 121), (181, 222, 43)],
}
_DIVERGING_ANCHORS = np.array([[40, 90, 170], [14, 18, 24], [255, 176, 64]], dtype=np.float64)


def _lut(anchors, size=256):
    points = np.asarray(anchors, dtype=np.float64)
    positions = np.linspace(0, size - 1, len(points))
    grid = np.arange(size)
    channels = [np.interp(grid, positions, points[:, c]) for c in range(3)]
    return np.rint(np.stack(channels, axis=1)).astype(np.uint8)


LUTS = {name: _lut(points) for name, points in _ANCHORS.items()}
DIVERGING = _lut(_DIVERGING_ANCHORS)


def smooth(frame):
    """Mild 3x3 box despeckle with edge padding (preview only)."""
    image = np.asarray(frame, dtype=np.float32)
    padded = np.pad(image, 1, mode="edge")
    total = np.zeros_like(image)
    for dy in range(3):
        for dx in range(3):
            total += padded[dy:dy + image.shape[0], dx:dx + image.shape[1]]
    return total / 9.0


def normalize(frame, mode="percentile"):
    """Map a nonnegative intensity frame to [0, 1] float32."""
    image = np.asarray(frame, dtype=np.float64)
    if mode == "log_db":
        peak = float(image.max()) if image.size else 0.0
        reference = peak if peak > 0 else 1.0
        image = 20.0 * np.log10(np.maximum(image / reference, 1e-4))
        low, high = -40.0, 0.0
    elif mode == "linear":
        low, high = 0.0, float(image.max()) if image.size else 1.0
    else:
        if image.size:
            low, high = (float(value) for value in np.percentile(image, [2, 98]))
        else:
            low, high = 0.0, 1.0
    if high <= low:
        low, high = float(low) - 1.0, float(low) + 1.0
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def colorize(normalized, cmap="gray"):
    """Apply a 256-colour LUT to a [0, 1] frame, returning RGB uint8."""
    index = np.clip(np.rint(np.asarray(normalized, dtype=np.float32) * 255), 0, 255).astype(np.uint8)
    return LUTS.get(cmap, LUTS["gray"])[index].copy()


def enhance(frame, mode="percentile", cmap="gray", denoise=False):
    """One-call display enhancement; never used as detector input."""
    image = np.asarray(frame, dtype=np.float32)
    if denoise:
        image = smooth(image)
    return colorize(normalize(image, mode), cmap)


def temporal_difference(frames, index, reference="median", window=5):
    """Centered diverging RGB view of the change at one frame.

    Static speckle and static dark regions cancel between current and history,
    while a moving shadow produces a strong positive (warm) response.
    """
    current = np.asarray(frames[index], dtype=np.float32)
    if reference == "previous":
        baseline = np.asarray(frames[max(0, index - 1)], dtype=np.float32)
    else:
        start = max(0, index - max(1, int(window)))
        history = np.asarray(frames[start:index], dtype=np.float32)
        baseline = np.median(history, axis=0) if len(history) else current
    delta = baseline - current
    scale = float(np.percentile(np.abs(delta), 98)) if delta.size else 0.0
    scale = max(scale, 1e-6)
    centered = np.clip(delta / (2.0 * scale) + 0.5, 0.0, 1.0)
    lookup = np.clip(np.rint(centered * 255), 0, 255).astype(np.uint8)
    return DIVERGING[lookup].copy()


def boundary(mask):
    """One-pixel outline of a binary mask (four-connected erosion)."""
    binary = np.asarray(mask).astype(bool)
    if not binary.any():
        return np.zeros_like(binary)
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    inner = (binary & padded[:-2, 1:-1] & padded[2:, 1:-1]
             & padded[1:-1, :-2] & padded[1:-1, 2:])
    return binary & ~inner


def overlay_mask(rgb, mask, color, alpha=0.9):
    """Draw a thin coloured contour over an RGB image."""
    if mask is None:
        return rgb
    output = np.asarray(rgb).copy()
    edge = boundary(mask)
    if not edge.any():
        return output
    tint = np.asarray(color, dtype=np.float32)
    output[edge] = np.rint(output[edge] * (1.0 - alpha) + tint * alpha).astype(np.uint8)
    return output


def _point(row):
    x, y = row.get("x_px"), row.get("y_px")
    if x is None or y is None or x == "" or y == "":
        return None
    try:
        return float(x), float(y)
    except (TypeError, ValueError):
        return None


def draw_tracks(rgb, rows, frame_index, radius=2):
    """Draw the track history up to ``frame_index`` using repository colours."""
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    painter = ImageDraw.Draw(image)
    last = None
    for row in rows:
        try:
            index = int(row["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        if index > frame_index:
            break
        point = _point(row)
        if point is None:
            last = None
            continue
        color = IMPUTED_COLOR if row.get("source") == "imputed" else MEASURED_COLOR
        segment = row.get("segment")
        if last is not None and last["segment"] == segment:
            painter.line([last["point"], point], fill=color, width=1)
        if index == frame_index:
            x, y = point
            painter.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color)
        last = {"point": point, "segment": segment}
    return np.asarray(image)


def centroid(mask):
    """Return (x, y) centre of a binary mask, or None when empty."""
    if mask is None:
        return None
    binary = np.asarray(mask).astype(bool)
    if not binary.any():
        return None
    y, x = np.nonzero(binary)
    return float(x.mean()), float(y.mean())


def roi_zoom(rgb, center, factor=4, out_size=(256, 192)):
    """Crop around ``center`` and upscale with nearest-neighbour (4x by default)."""
    image = np.asarray(rgb, dtype=np.uint8)
    height, width = image.shape[:2]
    if center is None:
        center = (width / 2.0, height / 2.0)
    crop_width = max(8, int(round(out_size[0] / max(1, factor))))
    crop_height = max(8, int(round(out_size[1] / max(1, factor))))
    x0 = int(np.clip(round(center[0] - crop_width / 2.0), 0, max(0, width - crop_width)))
    y0 = int(np.clip(round(center[1] - crop_height / 2.0), 0, max(0, height - crop_height)))
    patch = image[y0:y0 + crop_height, x0:x0 + crop_width]
    zoomed = Image.fromarray(patch).resize(out_size, Image.Resampling.NEAREST)
    # Mark the tracked centre inside the magnified view.
    painter = ImageDraw.Draw(zoomed)
    marker_x = (center[0] - x0) * factor
    marker_y = (center[1] - y0) * factor
    if 0 <= marker_x < out_size[0] and 0 <= marker_y < out_size[1]:
        painter.line((marker_x - 5, marker_y, marker_x + 5, marker_y), fill=MEASURED_COLOR)
        painter.line((marker_x, marker_y - 5, marker_x, marker_y + 5), fill=MEASURED_COLOR)
    return np.asarray(zoomed), (x0, y0, crop_width, crop_height)


def count_components(mask):
    """Four-connected component count, mirroring videosar.analysis.components."""
    binary = np.asarray(mask).astype(bool)
    height, width = binary.shape
    seen = np.zeros_like(binary)
    count = 0
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or seen[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return count


PANEL_TITLES = ("1  Enhanced SAR", "2  Temporal difference", "3  Detections + track", "4  ROI zoom x4")


def compose_quad(views, titles=PANEL_TITLES, gap=8, bar=22):
    """Compose four RGB views into one labelled 2x2 panel."""
    panels = [np.asarray(view, dtype=np.uint8) for view in views]
    size = (max(view.shape[1] for view in panels), max(view.shape[0] for view in panels))
    canvas = Image.new("RGB", (2 * size[0] + 3 * gap, 2 * (size[1] + bar) + 3 * gap), (18, 27, 34))
    painter = ImageDraw.Draw(canvas)
    for index, (view, title) in enumerate(zip(panels, titles, strict=True)):
        column, row = index % 2, index // 2
        x = gap + column * (size[0] + gap)
        y = gap + row * (size[1] + bar + gap)
        painter.text((x + 2, y + 4), title, fill=(230, 238, 244))
        panel = Image.fromarray(view).resize(size, Image.Resampling.NEAREST)
        canvas.paste(panel, (x, y + bar))
    return np.asarray(canvas)


def save_gif(frames, path, duration_ms=120):
    """Write an animated GIF using Pillow only."""
    from pathlib import Path

    images = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
    if not images:
        return None
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(destination, save_all=True, append_images=images[1:],
                   duration=duration_ms, loop=0)
    return destination


def save_mp4(frames, path, fps=8):
    """Optional MP4 export; requires imageio-ffmpeg (already in requirements-gui.txt)."""
    from pathlib import Path

    import imageio_ffmpeg

    panels = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    if not panels:
        return None
    height, width = panels[0].shape[:2]
    even = (width // 2 * 2, height // 2 * 2)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(destination), even, fps=fps, codec="libx264",
                                         pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                         output_params=["-movflags", "+faststart"])
    writer.send(None)
    try:
        for panel in panels:
            writer.send(np.ascontiguousarray(panel[:even[1], :even[0]]).tobytes())
    finally:
        writer.close()
    return destination


def draw_crop_box(rgb, crop):
    """Draw a red --crop rectangle on a preview frame."""
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    x, y, width, height = (int(value) for value in crop)
    ImageDraw.Draw(image).rectangle((x, y, x + width, y + height), outline=CROP_COLOR, width=2)
    return np.asarray(image)
