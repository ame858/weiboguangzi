import numpy as np


CENTER_OUTPUTS = 6
CENTER_BIAS = -4.6
BOX_WEIGHT = 0.25


def center_loss(torch, output, labels):
    """Focal center heatmap loss and log-distance regression at positive centers."""
    foreground = labels > 0
    height, width = labels.shape[-2:]
    x = torch.arange(width, device=labels.device)
    y = torch.arange(height, device=labels.device)
    columns, rows = foreground.any(dim=1), foreground.any(dim=2)
    valid = columns.any(dim=1)
    left = torch.where(columns, x, width).amin(dim=1)
    right = torch.where(columns, x, -1).amax(dim=1)
    top = torch.where(rows, y, height).amin(dim=1)
    bottom = torch.where(rows, y, -1).amax(dim=1)
    cx, cy = (left + right) // 2, (top + bottom) // 2
    dx = x[None, None, :] - cx[:, None, None]
    dy = y[None, :, None] - cy[:, None, None]
    target = torch.exp(-0.5 * (dx.square() + dy.square())) * valid[:, None, None]

    heatmap = output[:, 1]
    probability = heatmap.sigmoid()
    functional = torch.nn.functional
    positive = -(1 - probability).square() * functional.logsigmoid(heatmap)
    negative = -(1 - target).pow(4) * probability.square() * functional.logsigmoid(-heatmap)
    positives = valid.sum().clamp_min(1)
    heatmap_loss = torch.where(target == 1, positive, negative).sum() / positives

    index = torch.arange(len(labels), device=labels.device)[valid]
    estimated = output[index, 2:, cy[valid], cx[valid]]
    distances = torch.stack((cx - left, cy - top, right - cx, bottom - cy), dim=1)[valid]
    box_loss = functional.smooth_l1_loss(estimated, distances.float().log1p(), reduction="sum") / (4 * positives)
    return heatmap_loss + BOX_WEIGHT * box_loss


def center_masks(heatmaps, log_distances, threshold, min_area):
    """Decode the strongest center per frame into a box on the original grid."""
    count, height, width = heatmaps.shape
    masks = np.zeros_like(heatmaps, dtype=bool)
    limit = np.log1p(max(height, width) - 1)
    for frame, peak in enumerate(heatmaps.reshape(count, height * width).argmax(axis=1)):
        cy, cx = divmod(int(peak), width)
        if heatmaps[frame, cy, cx] < threshold:
            continue
        left, top, right, bottom = np.rint(np.expm1(np.clip(log_distances[frame, :, cy, cx], 0, limit))).astype(int)
        x0, x1 = max(0, cx - left), min(width, cx + right + 1)
        y0, y1 = max(0, cy - top), min(height, cy + bottom + 1)
        if (x1 - x0) * (y1 - y0) >= min_area:
            masks[frame, y0:y1, x0:x1] = True
    return masks
