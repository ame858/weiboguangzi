from pathlib import Path

import numpy as np

from .analysis import intensity_transform
from .simulation import box_mean


def features(frames, temporal, network="compact"):
    log_power = intensity_transform(frames)
    small = np.asarray([box_mean(frame, 1) for frame in log_power])
    broad = np.asarray([box_mean(frame, 5) for frame in log_power])
    global_center = np.median(small, axis=(1, 2), keepdims=True)
    channels = [small - global_center, small - broad, broad - global_center]
    if network == "context":
        middle = np.asarray([box_mean(frame, 2) for frame in log_power])
        channels.extend((log_power - small, middle - broad))
    if temporal:
        delta = np.zeros_like(small)
        if network == "context":
            mean_delta, broad_delta, dispersion = (np.zeros_like(small) for _ in range(3))
        for t in range(1, len(small)):
            previous = small[max(0, t - 5):t]
            delta[t] = np.max(previous, axis=0) - small[t]
            if network == "context":
                mean_delta[t] = np.mean(previous, axis=0) - small[t]
                broad_delta[t] = box_mean(mean_delta[t], 2)
                dispersion[t] = np.std(previous, axis=0)
        channels.append(delta)
        if network == "context":
            channels.extend((mean_delta, broad_delta, dispersion))
    return np.stack(channels, axis=-1)


def patches(frames, temporal, network="compact"):
    channels = features(frames, temporal, network)
    padded = np.pad(channels, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))


def sigmoid(scores):
    return 1 / (1 + np.exp(-np.clip(scores, -30, 30)))


def fit(sequences, temporal, epochs, seed, network="compact"):
    rng = np.random.default_rng(seed)
    samples, targets = [], []
    for frames, masks in sequences:
        window = patches(frames, temporal, network)
        foreground = np.flatnonzero(masks)
        background = np.flatnonzero(~masks)
        if foreground.size > 600:
            foreground = rng.choice(foreground, 600, replace=False)
        if network == "context":
            # Sample dark and ordinary negatives so stationary clutter gets training coverage.
            darkness = window[..., 0, 1, 1].ravel()[background]
            cutoff = max(1, background.size // 8)
            dark_pool = background[np.argpartition(darkness, cutoff - 1)[:cutoff]] if background.size else background
            hard = rng.choice(dark_pool, min(1000, dark_pool.size), replace=False)
            remainder = np.setdiff1d(background, hard, assume_unique=True)
            ordinary = rng.choice(remainder, min(2400, remainder.size), replace=False)
            background = np.concatenate((hard, ordinary))
        else:
            background = rng.choice(background, min(3000, background.size), replace=False)
        selected = np.concatenate((foreground, background))
        pixels = np.unravel_index(selected, masks.shape)
        samples.append(window[pixels].reshape(len(selected), -1))
        targets.append(masks.ravel()[selected].astype(np.float32))
    data = np.concatenate(samples)
    labels = np.concatenate(targets)
    if not 0 < labels.sum() < len(labels):
        raise ValueError("train split requires foreground and background pixels")
    mean = data.mean(axis=0)
    std = np.maximum(data.std(axis=0), 1e-4)
    data = (data - mean) / std
    hidden_size = 24 if network == "context" else 12
    kernel = (rng.normal(size=(data.shape[1], hidden_size)) / np.sqrt(data.shape[1])).astype(np.float32)
    bias = np.zeros(hidden_size, dtype=np.float32)
    head = (rng.normal(size=hidden_size) / np.sqrt(hidden_size)).astype(np.float32)
    intercept = np.zeros(1, dtype=np.float32)
    history = []
    for _ in range(epochs):
        loss = 0.0
        order = rng.permutation(len(data))
        for offset in range(0, len(data), 512):
            index = order[offset:offset + 512]
            inputs, expected = data[index], labels[index]
            activation = inputs @ kernel + bias
            hidden = np.maximum(activation, 0)
            logits = hidden @ head + intercept
            weights = 1 + expected * 2
            loss += float(np.sum(weights * (np.logaddexp(0, logits) - expected * logits)))
            gradient = weights * (sigmoid(logits) - expected) / len(index)
            propagated = gradient[:, None] * head * (activation > 0)
            head -= 0.04 * (hidden.T @ gradient)
            intercept -= 0.04 * gradient.sum()
            kernel -= 0.04 * (inputs.T @ propagated)
            bias -= 0.04 * propagated.sum(axis=0)
        history.append(loss / len(data))
    model = {"mean": mean, "std": std, "kernel": kernel, "bias": bias, "head": head,
             "intercept": intercept, "temporal": np.array(int(temporal))}
    if network == "context":
        model["network"] = np.array("context")
    return model, history


def save_model(path, model):
    np.savez_compressed(path, **model)


def load_model(path):
    with np.load(Path(path), allow_pickle=False) as source:
        model = {key: source[key] for key in ("mean", "std", "kernel", "bias", "head", "intercept", "temporal")}
        if "network" in source:
            model["network"] = source["network"]
    network = model.get("network", np.array("compact"))
    if network.shape != () or network.item() not in ("compact", "context") or model["temporal"].shape != ():
        raise ValueError("invalid model network or time context")
    channels = {"compact": (3, 4), "context": (5, 9)}
    size = 9 * channels[network.item()][int(bool(model["temporal"]))]
    hidden = model["head"].size
    expected = {"mean": (size,), "std": (size,), "kernel": (size, hidden), "bias": (hidden,),
                "head": (hidden,), "intercept": (1,), "temporal": ()}
    if (any(model[key].shape != shape or not np.isfinite(model[key]).all() for key, shape in expected.items())
            or int(model["temporal"]) not in (0, 1) or np.any(model["std"] <= 0) or not hidden):
        raise ValueError("invalid model weights or normalization statistics")
    return model


def predict(frames, model):
    windows = patches(frames, bool(model["temporal"]), str(model.get("network", "compact")))
    scores = np.empty(frames.shape, dtype=np.float32)
    for t, window in enumerate(windows):
        pixels = window.reshape(-1, model["mean"].size)
        inputs = (pixels - model["mean"]) / model["std"]
        hidden = np.maximum(inputs @ model["kernel"] + model["bias"], 0)
        scores[t] = sigmoid(hidden @ model["head"] + model["intercept"]).reshape(frames.shape[1:])
    return scores
