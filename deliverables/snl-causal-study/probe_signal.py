"""Compare prospective causal image signals on locked development annotations."""
import argparse
import sys
from collections import deque

import numpy as np
from scipy.ndimage import uniform_filter

from snl_inputs import DEFAULT_OUTPUT, DEVELOPMENT, ROOT, VALIDATE, load_gray, load_labels, write_json

sys.path.insert(0, str(ROOT))

from videosar.snl_causal import box_iou


def means(signal, windows):
    integral = np.pad(signal, ((1, 0), (1, 0))).cumsum(0, dtype=np.float64).cumsum(1)
    x0, y0, x1, y1 = np.rint(windows).astype(int).T
    return (integral[y1,x1] - integral[y1,x0] - integral[y0,x1] + integral[y0,x0]) / ((x1-x0) * (y1-y0))


def paired(boxes, shape):
    h,w = shape
    positives, negatives = [], []
    for box in boxes:
        for dx, dy in ((-21, 0), (21, 0), (0, 24), (0, -24), (-28, 0), (28, 0)):
            candidate = box + [dx, dy, dx, dy]
            if np.any(candidate[:2] < 0) or np.any(candidate[2:] > [w, h]):
                continue
            if not np.any(box_iou(candidate[None], boxes) > 0):
                positives.append(box)
                negatives.append(candidate)
                break
    return np.asarray(positives).reshape(-1, 4), np.asarray(negatives).reshape(-1, 4)


def signals(current, past):
    short = np.stack([past[-i] for i in (6, 12, 18, 24, 30)])
    long = np.stack([past[-i] for i in (10, 20, 30, 40, 50, 60, 75, 90)])
    positive = np.maximum(0, np.quantile(long, 0.75, axis=0) - current).astype(np.float32)
    mad = np.median(np.abs(long - np.median(long, axis=0, keepdims=True)), axis=0)
    local = uniform_filter(current, 13) - current
    short_dark = np.maximum(0, np.quantile(short, 0.75, axis=0) - current)
    temporal_whitened = positive / (0.02 + mad)
    return {
        "gray": -current,
        "local_darkness": local,
        "short_background": short_dark,
        "long_background": positive,
        "long_whitened": temporal_whitened,
        "local_plus_long": local + 0.5 * positive,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    output = DEFAULT_OUTPUT
    gray = load_gray(output, args.ffmpeg)
    val, _, _ = load_labels(output, VALIDATE, gray)
    dev, _, _ = load_labels(output, DEVELOPMENT, gray)
    order = val | dev
    history = deque(maxlen=90)
    groups = {"validation": {}, "development": {}}
    for i in range(2, max(DEVELOPMENT) + 1):
        current = uniform_filter(gray[i-2].astype(np.float32) / 255, size=3)
        if i in order:
            pos, neg = paired(order[i], current.shape)
            group = groups["validation" if i in val else "development"]
            for name, signal in signals(current, history).items():
                group.setdefault(name, []).extend(means(signal, pos)-means(signal, neg))
        history.append(current)
    study = {}
    for split, group in groups.items():
        study[split] = {name: {"pairs": len(margins), "wins": int(np.count_nonzero(np.asarray(margins)>0)),
                              "win_rate": float(np.mean(np.asarray(margins)>0)), "median_margin": float(np.median(margins))}
                        for name, margins in group.items()}
    write_json(output / "causal-signal-probe.json", study)
    print(study, flush=True)


if __name__ == "__main__":
    main()
