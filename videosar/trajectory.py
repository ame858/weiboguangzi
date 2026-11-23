import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter
from scipy.special import expit

from .analysis import centroid, components, fit_velocity, track
from .center_box import center_masks
from .study import cleanup
from .unet import predict


def candidate_observations(torch, model, aligned, times, settings, device="cpu"):
    output = predict(torch, model, aligned, settings, device, 32, all_channels=True)
    heatmap = expit(output[:, 1])
    masks = cleanup(expit(output[:, 0]), settings["threshold"], settings["min_area"], settings["closing"])
    boxes = center_masks(heatmap, output[:, 2:], 0, 1)
    count, _, width = heatmap.shape
    peaks = heatmap.reshape(count, -1).argmax(axis=1)
    y, x = np.divmod(peaks, width)
    xy = np.column_stack((x, y))
    centers = [np.array([centroid(region, width) for region in components(frame)]).reshape(-1, 2)
               for frame in masks]
    observed = track(masks, times, None, gate=5, max_gap=3)["frames"]
    expected = predictions_from_history(observed, times)
    return dict(masks=masks, boxes=boxes, scores=heatmap.reshape(count, -1).max(axis=1), heatmap=heatmap,
                xy=xy, mask_centers=centers, expected=expected)


def predictions_from_history(observed, times):
    expected = np.full((len(observed), 2), np.nan)
    for t in range(1, len(observed)):
        recent = [r for r in observed[max(0, t - 3):t] if r["source"] == "measured"]
        if recent:
            last = recent[-1]
            recent = [row for row in recent if row["segment"] == last["segment"]]
            velocity = fit_velocity([row["time_s"] for row in recent], [row["xy_px"] for row in recent])
            expected[t] = last["xy_px"]
            if velocity is not None:
                expected[t] += velocity * (times[t] - last["time_s"])
    return expected


def accumulate_centers(heatmap, gate, minimum_displacement):
    """Best three-frame path ending at the current top peak, using four historical peaks."""
    count, height, width = heatmap.shape
    flat = heatmap.reshape(count, -1)
    top = flat.argmax(axis=1)
    local = (heatmap == maximum_filter(heatmap, size=(1, 3, 3))).reshape(count, -1)
    ranked = np.argsort(-np.where(local, flat, -np.inf), axis=1, kind="stable")[:, :4]
    y, x = np.divmod(ranked, width)
    positions = np.stack((x, y), axis=-1)
    strengths = np.where(np.take_along_axis(local, ranked, axis=1), np.take_along_axis(flat, ranked, axis=1), -np.inf)
    scores = np.full(count, -np.inf)
    for t in range(2, count):
        cy, cx = divmod(int(top[t]), width)
        end = np.array((cx, cy))
        first, second = positions[t - 2:t]
        near = np.linalg.norm(first[:, None] - second[None, :], axis=2) <= gate
        arrives = np.linalg.norm(second - end, axis=1) <= gate
        moves = np.linalg.norm(first - end, axis=1) >= minimum_displacement
        combined = strengths[t - 2, :, None] + strengths[t - 1, None, :] + flat[t, top[t]]
        valid = near & arrives[None, :] & moves[:, None]
        scores[t] = np.where(valid, combined, -np.inf).max() / 3
    return scores


def choose_boxes(observed, config, accumulated=None):
    kind = config["method"]
    gate = config.get("gate", 3)
    anchor = observed["xy"]
    scores = observed["scores"]
    eligible = scores >= config["threshold"]
    if kind == "association":
        anchor = observed["expected"]
        eligible &= np.linalg.norm(observed["xy"] - anchor, axis=1) <= gate
    elif kind == "accumulation":
        if accumulated is None:
            accumulated = accumulate_centers(observed["heatmap"], gate, config["minimum_displacement"])
        eligible = (accumulated >= config["threshold"]) & (scores >= 0.10)
    for i in np.flatnonzero(eligible):
        centers = observed["mask_centers"][i]
        if len(centers):
            eligible[i] = config["replace"] and np.all(np.linalg.norm(centers - anchor[i], axis=1) > gate)
    return eligible


def recover_masks(observed, config):
    masks = observed["masks"].copy()
    selected = choose_boxes(observed, config)
    masks[selected] = observed["boxes"][selected]
    return masks, selected


def low_rank_basis(stack, rank):
    _, vectors = np.linalg.eigh(stack @ stack.T)
    return vectors[:, -rank:]


def low_rank_scores(frames, window, rank):
    """Positive dark residual after causal low-rank reconstruction on the registered grid."""
    power = uniform_filter(np.log1p(frames), size=(1, 3, 3), mode="reflect")
    score = np.zeros_like(power)
    for t in range(rank, len(power)):
        stack = power[max(0, t - window):t + 1].reshape(-1, power.shape[1] * power.shape[2])
        basis = low_rank_basis(stack, rank)
        background = (basis[-1] @ basis.T) @ stack
        score[t] = np.maximum(background.reshape(power.shape[1:]) - power[t], 0)
    return score
