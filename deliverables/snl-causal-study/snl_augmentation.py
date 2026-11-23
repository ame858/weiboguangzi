"""Causal feature copy-paste using only labeled SNL training frames."""
import numpy as np
import torch
from scipy.ndimage import zoom

from snl_detector import target_in_tile
from videosar.snl_causal import STACKED_HISTORY_LAGS


def weaken_shadow(frame, box, fraction):
    """Raise a shadow towards its local surroundings, feathering the change outside the box."""
    x0, y0, x1, y1 = np.rint(box).astype(int)
    left, top = max(0, x0 - 5), max(0, y0 - 5)
    right, bottom = min(frame.shape[1], x1 + 5), min(frame.shape[0], y1 + 5)
    region = frame[top:bottom, left:right]
    x = np.arange(left, right)[None, :]
    y = np.arange(top, bottom)[:, None]
    dx = np.maximum(np.maximum(x0 - x, x - (x1 - 1)), 0)
    dy = np.maximum(np.maximum(y0 - y, y - (y1 - 1)), 0)
    distance = np.hypot(dx, dy)
    outside, core = region[distance >= 3], region[distance == 0]
    if not len(outside) or not len(core):
        return
    difference = float(np.median(outside) - np.median(core))
    if difference > 0:
        opacity = np.clip(1.0 - distance / 3.0, 0, 1)
        lifted = region + opacity * (fraction * difference)
        region[:] = np.rint(lifted).clip(0, 255).astype(np.uint8)


def weak_shadows(labels, seed):
    """Use true positions from the chronological training partition at each observed lag."""
    lags = (0, *STACKED_HISTORY_LAGS)
    empty = np.empty((0, 4), dtype=np.float32)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))

    def transform(image, index, left, top):
        if rng.random() >= 0.72:
            return image
        changed = image.copy()
        fraction = rng.uniform(0.25, 0.70)
        for frame, lag in zip(changed, lags, strict=True):
            source = index - lag if index - lag >= 2 else index
            boxes = target_in_tile(labels.get(source, empty), left, top)
            for box in boxes:
                weaken_shadow(frame, box, fraction)
        return changed

    return transform


def weak_target_weights(features, labels):
    from snl_motion_aug import local_darkness

    contrasts, tiles = {}, {}
    for index, boxes in labels.items():
        frame = features[index-2, 0].astype(np.float32) / 255
        contrasts[index] = np.asarray([local_darkness(frame, box) for box in boxes])
        center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        tiles[index] = np.floor(center / 32).astype(int)
    threshold = float(np.percentile(np.concatenate(list(contrasts.values())), 25))
    cells, counts = np.unique(np.concatenate(list(tiles.values())), return_counts=True, axis=0)
    density = dict(zip(map(tuple, cells), counts, strict=True))
    typical = np.median(counts)
    weights, before, after = {}, [], []
    for index, local in contrasts.items():
        if not len(local):
            continue
        frequency = np.asarray([density[tuple(cell)] for cell in tiles[index]])
        balance = np.clip((typical / frequency)**0.25, 0.65, 1.8)
        weak = local <= threshold
        combined = balance * np.where(weak, 2.0, 1.0)
        weights[index] = combined / combined.sum()
        before.append(float(weak.mean()))
        after.append(float(weights[index][weak].sum()))
    return weights, {"training_contrast_q25": threshold, "weak_crop_probability_before": float(np.mean(before)),
                     "weak_crop_probability_after": float(np.mean(after)), "density_bin_pixels": 32}


def training_chips(features, labels, margin=5):
    chips = []
    _, _, height, width = features.shape
    for index, boxes in labels.items():
        for box in boxes:
            x0, y0, x1, y1 = box
            left, top = max(0, int(np.floor(x0)) - margin), max(0, int(np.floor(y0)) - margin)
            right, bottom = min(width, int(np.ceil(x1)) + margin), min(height, int(np.ceil(y1)) + margin)
            array = features[index-2, :, top:bottom, left:right].astype(np.float32) / 255
            if min(array.shape[-2:]) >= 5:
                chips.append((array, box - np.array([left, top, left, top], dtype=np.float32)))
    if not chips:
        raise ValueError("No verified training patches are available for copy-paste")
    return chips


def paste_chip(image, boxes, source, source_box, rng, *, stacked=False):
    ratios = rng.uniform((0.75, 0.75), (1.50, 1.55))
    frame = zoom(source, (1, *ratios), order=1, prefilter=False)
    h, w = frame.shape[-2:]
    height, width = image.shape[-2:]
    if h >= height or w >= width:
        return boxes
    sx, sy = w / source.shape[-1], h / source.shape[-2]
    local_box = source_box * np.asarray([sx, sy, sx, sy], dtype=np.float32)
    fade = 7 if stacked else 3
    for _ in range(16):
        left, top = rng.integers([0, 0], [width-w+1, height-h+1])
        target = local_box + np.array([left, top, left, top], dtype=np.float32)
        padded = target + np.asarray([-1, -1, 1, 1]) * (fade + 1)
        collision = (np.minimum(boxes[:, 2:], padded[2:]) > np.maximum(boxes[:, :2], padded[:2])).all(axis=1)
        if not collision.any():
            break
    else:
        return boxes

    x = np.arange(w, dtype=np.float32)[None, :]
    y = np.arange(h, dtype=np.float32)[:, None]
    dx = np.maximum.reduce([local_box[0] - x, x - local_box[2], np.zeros_like(x)])
    dy = np.maximum.reduce([local_box[1] - y, y - local_box[3], np.zeros_like(y)])
    distance = np.sqrt(dx**2 + dy**2)
    alpha = np.clip(1 - distance / fade, 0, 1)
    interior = distance == 0
    rim = distance >= fade
    if not rim.any() or not interior.any():
        return boxes
    landing = image[:, top:top+h, left:left+w]
    reference_src = float(np.median(frame[0, rim]))
    reference_dst = float(np.median(landing[0, rim]))
    gain = float(rng.uniform(0.35, 1.15))
    if stacked:
        source_levels = np.median(frame[:, rim], axis=1)[:, None, None]
        destination_levels = np.median(landing[:, rim], axis=1)[:, None, None]
        frame = np.clip(destination_levels + gain * (frame - source_levels), 0, 1)
    else:
        frame[0] = np.clip(reference_dst + gain * (frame[0] - reference_src), 0, 1)
        frame[1:] = np.clip(frame[1:] * rng.uniform(0.60, 1.20), 0, 1)
    landing *= 1 - alpha
    landing += frame * alpha
    return np.concatenate((boxes, target[None]), axis=0)


def copy_paste(batch, targets, chips, rng, *, stacked=False):
    for image, target in zip(batch, targets, strict=True):
        if rng.random() >= 0.58:
            continue
        arr = image.numpy()
        boxes = target["boxes"].numpy()
        for _ in range(int(rng.integers(1, 4))):
            source, source_box = chips[rng.integers(len(chips))]
            boxes = paste_chip(arr, boxes, source, source_box, rng, stacked=stacked)
        target["boxes"] = torch.from_numpy(boxes)
        target["labels"] = torch.ones(len(boxes), dtype=torch.int64)
