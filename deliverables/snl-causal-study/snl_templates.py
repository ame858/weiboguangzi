"""Generate causal multiscale shadow proposals directly on the native radar grid."""
import argparse
import sys
from collections import deque

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

from snl_inputs import DEFAULT_OUTPUT, DEVELOPMENT, ROOT, VALIDATE, load_gray, load_labels, write_json

sys.path.insert(0, str(ROOT))

from videosar.snl_causal import detection_counts, detection_metrics

SIZES = tuple((w, h) for w in (5, 7, 9, 11, 15) for h in (9, 13, 17, 21, 25) if 0.9 <= h/w <= 3.6)
CUES = ("dark", "spatial", "short", "whitened", "combined")


def causal_inputs(gray, history):
    current = gray.astype(np.float32) / 255
    smooth = uniform_filter(current, size=3, mode="nearest")
    short = np.stack([history[-lag] for lag in (6, 12, 18, 24, 30)])
    long = np.stack([history[-lag] for lag in (10, 20, 30, 40, 50, 60, 75, 90)])
    short_motion = np.maximum(0, np.quantile(short, 0.75, axis=0).astype(np.float32) - smooth)
    mid = np.median(long, axis=0)
    dispersion = np.median(np.abs(long-mid), axis=0)
    difference = np.maximum(0, np.quantile(long, 0.75, axis=0).astype(np.float32) - smooth)
    whitened = difference / (0.02 + dispersion)
    return current, short_motion, whitened


def templates(current, short, whitened):
    out = np.full((len(CUES), *current.shape), -np.inf, dtype=np.float32)
    selected = np.zeros(out.shape, dtype=np.int16)
    for index, (w, h) in enumerate(SIZES):
        inside = uniform_filter(current, size=(h,w), mode="nearest")
        around = uniform_filter(current, size=(h+12,w+12), mode="nearest")
        contrast = around - inside
        residual = uniform_filter(short, size=(h,w), mode="nearest")
        white = uniform_filter(whitened, size=(h,w), mode="nearest")
        value = (-inside, contrast, residual, white, contrast + 0.35 * residual + 0.018 * white)
        for j, evidence in enumerate(value):
            border_x, border_y = w//2, h//2
            evidence[:border_y] = -np.inf
            evidence[-border_y:] = -np.inf
            evidence[:, :border_x] = -np.inf
            evidence[:, -border_x:] = -np.inf
            wins = evidence > out[j]
            out[j, wins] = evidence[wins]
            selected[j, wins] = index
    regions = {}
    dimensions = np.asarray(SIZES, dtype=np.float32)
    for cue, confidence, size in zip(CUES, out, selected, strict=True):
        maxima = (confidence == maximum_filter(confidence, size=5, mode="nearest")) & np.isfinite(confidence)
        y,x = np.nonzero(maxima)
        strength = confidence[y,x]
        top = np.argsort(-strength, kind="stable")[:256]
        w,h = dimensions[size[y[top],x[top]]].T
        x, y = x[top] + 0.5, y[top] + 0.5
        boxes = np.column_stack((x-w/2,y-h/2,x+w/2,y+h/2)).astype(np.float32)
        regions[cue] = (boxes, strength[top])
    return regions


def metrics_at(predictions, labels, limit, score_floor=None):
    tally = np.zeros(3, dtype=int)
    for i, targets in labels.items():
        boxes, scores = predictions[i]
        boxes = boxes[:limit]
        if score_floor is not None:
            boxes = boxes[scores[:limit] >= score_floor]
        tally += detection_counts(boxes, targets)
    return detection_metrics(tally, len(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    output = DEFAULT_OUTPUT
    gray = load_gray(output, args.ffmpeg)
    validation, _, _ = load_labels(output, VALIDATE, gray)
    development, _, _ = load_labels(output, DEVELOPMENT, gray)
    labels = validation | development
    history = deque(maxlen=90)
    all_records = {cue: {} for cue in CUES}
    for i in range(2, max(labels) + 1):
        frame = gray[i-2]
        if i in labels:
            out = templates(*causal_inputs(frame, history))
            for cue, values in out.items():
                all_records[cue][i] = values
            if i % 20 == 0:
                print(f"Template candidates through annotation {i}", flush=True)
        history.append(uniform_filter(frame.astype(np.float32)/255, size=3, mode="nearest"))
    scores = {}
    for cue, predictions in all_records.items():
        val_scores = np.concatenate([predictions[i][1] for i in validation])
        grid = np.quantile(val_scores, np.linspace(0.10, 0.98, 30))
        ranking = [(t,metrics_at(predictions,validation,256,t)) for t in grid]
        threshold, val = max(ranking, key=lambda p:p[1]["f1"])
        scores[cue] = {"threshold":float(threshold), "validation":val,
                       "development":metrics_at(predictions,development,256,threshold),
                       "validation_at60":metrics_at(predictions,validation,60),
                       "development_at60":metrics_at(predictions,development,60),
                       "development_at256":metrics_at(predictions,development,256)}
        print(cue, scores[cue], flush=True)
    write_json(output/"templates-development.json", scores)
    write_json(output/"template-candidates.json",
               {cue:{i:[b.tolist(),s.tolist()] for i,(b,s) in rec.items()} for cue,rec in all_records.items()})


if __name__ == "__main__":
    main()
