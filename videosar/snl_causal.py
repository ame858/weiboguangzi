"""Causal VideoSAR features, box matching and multi-target association."""
from collections import deque

import numpy as np
from scipy import fft, ndimage
from scipy.optimize import linear_sum_assignment

STACKED_HISTORY_LAGS = (1, 3, 6, 12, 24, 30)
LONG_HISTORY_LAGS = (1, 6, 15, 30, 60, 90)


def box_iou(boxes, truth):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    truth = np.asarray(truth, dtype=np.float32).reshape(-1, 4)
    low = np.maximum(boxes[:, None, :2], truth[None, :, :2])
    high = np.minimum(boxes[:, None, 2:], truth[None, :, 2:])
    overlap = np.maximum(0, high - low).prod(axis=-1)
    a = (boxes[:, 2:] - boxes[:, :2]).prod(axis=-1)
    b = (truth[:, 2:] - truth[:, :2]).prod(axis=-1)
    union = a[:, None] + b[None, :] - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)


def matched_boxes(boxes, truth, threshold=0.50):
    iou = box_iou(boxes, truth)
    valid = iou >= threshold
    priority = valid * (1 + iou / (1 + min(iou.shape)))
    a, b = linear_sum_assignment(priority, maximize=True)
    return np.column_stack((a[valid[a, b]], b[valid[a, b]]))


def detection_counts(boxes, truth):
    tp = len(matched_boxes(boxes, truth))
    return tp, len(boxes) - tp, len(truth) - tp


def detection_metrics(tally, frames):
    tp, fp, fn = map(int, tally)
    return {
        "tp": tp, "fp": fp, "fn": fn, "frames": int(frames),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "fp_per_frame": fp / frames if frames else None,
    }


class CausalFeatures:
    """Per-frame channels and traditional scores with bounded causal history."""

    def __init__(self, profile="raw"):
        stacked_profiles = ("stacked", "stacked-standardized", "stacked-long", "background-history")
        if profile not in ("raw", "standardized", "registered", "histogram", "histogram-registered", *stacked_profiles):
            raise ValueError("Unknown radar feature profile")
        self.profile = profile
        self.channel_count = 7 if profile in stacked_profiles else 3
        self.history = deque(maxlen=30)
        self.raw_lags = LONG_HISTORY_LAGS if profile == "stacked-long" else STACKED_HISTORY_LAGS
        self.raw_history = deque(maxlen=max(self.raw_lags)) if self.channel_count == 7 else None
        self.registration = HistoryRegistration() if profile.endswith("registered") else None

    def _stack(self, image):
        fallback = image
        if self.profile in ("background-history", "stacked-long") and self.raw_history:
            fallback = self.raw_history[0]
        previous = [self.raw_history[-lag] if len(self.raw_history) >= lag else fallback for lag in self.raw_lags]
        self.raw_history.append(image)
        return np.stack((image, *previous))

    def network_channels(self, gray):
        """Advance the scanner for a consumer that only needs the network input."""
        if self.profile in ("stacked", "stacked-long", "background-history"):
            return self._stack(np.asarray(gray, dtype=np.float32) / np.float32(255))
        return self.update(gray)[0]

    def update(self, gray):
        raw = np.asarray(gray, dtype=np.float32) / np.float32(255)
        image = raw
        if self.profile.startswith("histogram"):
            gray = np.asarray(gray, dtype=np.uint8)
            cdf = np.cumsum(np.bincount(gray.ravel(), minlength=256)) / gray.size
            image = cdf[gray].astype(np.float32)
        if self.profile in ("standardized", "stacked-standardized"):
            center = ndimage.uniform_filter(image, size=25, mode="nearest")
            square = ndimage.uniform_filter(image * image, size=25, mode="nearest")
            scale = np.sqrt(np.maximum(square - center * center, 0.012**2))
            image = np.clip(0.5 + 0.18 * (image - center) / scale, 0, 1)
        smooth = ndimage.uniform_filter(image, size=3, mode="nearest")
        past = self.history
        delays = (6, 12, 15, 18, 24, 30)
        if self.registration:
            current_position = self.registration.estimate(raw)
            history = {
                d: self.registration.align(past[-d], d, current_position) for d in delays if len(past) >= d
            }
            self.registration.positions.append(current_position)
        else:
            history = {d: past[-d] for d in delays if len(past) >= d}
        lags = [history[d] for d in (6, 15, 30) if d in history]
        background = [history[d] for d in (6, 12, 18, 24, 30) if d in history]
        residual = np.zeros_like(image)
        fast = np.zeros_like(image)
        old = np.zeros_like(image)
        if lags:
            history_image = np.stack(background)
            residual = np.maximum(0, np.quantile(history_image, 0.75, axis=0).astype(np.float32) - smooth)
            fast = np.maximum(0, lags[0] - smooth)
            motion = np.maximum(np.maximum.reduce(lags) - smooth, 0)
            fused = np.minimum(residual, motion)
            reference = max(float(np.median(smooth)), 1e-8)
            old = np.minimum((np.max(history_image, axis=0) - smooth) / reference, 1 - smooth / reference)
        else:
            fused = residual
        past.append(smooth)
        if self.raw_history is not None:
            temporal = self._stack(image)
        else:
            temporal = np.stack((image, np.clip(4 * fast, 0, 1), np.clip(4 * residual, 0, 1)))
        return temporal, old, fused


class HistoryRegistration:
    """Estimate drift in each six-frame phase and move history into the current grid."""

    def __init__(self):
        self.transforms = deque(maxlen=6)
        self.positions = deque(maxlen=30)
        self.window = None

    def estimate(self, image):
        power = np.log1p(5 * image)
        if self.window is None:
            h, w = image.shape
            self.window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        spectrum = fft.rfft2((power - power.mean()) * self.window)
        if len(self.transforms) < 6:
            position = np.zeros(2, dtype=np.float32)
        else:
            correlation = fft.irfft2(spectrum * self.transforms[0].conj(), s=image.shape)
            window = np.arange(-3, 4)
            grid = correlation[np.ix_(window, window)]
            best = np.array(np.unravel_index(grid.argmax(), grid.shape)) - 3
            refined = best.astype(np.float32)
            peak = float(correlation[tuple(best)])
            for axis in range(2):
                offset = np.zeros(2, dtype=int)
                offset[axis] = 1
                left, right = (float(correlation[tuple(best + sign*offset)]) for sign in (-1, 1))
                span = left - 2*peak + right
                if span < -1e-8:
                    refined[axis] += np.clip(0.5*(left-right)/span, -0.5, 0.5)
            position = self.positions[-6] + refined
        self.transforms.append(spectrum)
        return position

    def align(self, frame, lag, current):
        if lag % 6:
            # Interpolate adjacent same-phase estimates for the traditional 15-frame cue.
            shift = 0.5 * ((current - self.positions[-12]) + (current - self.positions[-18])) if len(self.positions) >= 18 else current - self.positions[-12]
        else:
            shift = current - self.positions[-lag]
        return ndimage.shift(frame, shift, order=1, mode="nearest", prefilter=False)


def component_boxes(scores, threshold, min_area):
    regions, count = ndimage.label(scores >= threshold)
    area = np.bincount(regions.ravel(), minlength=count + 1)
    total_scores = np.bincount(regions.ravel(), weights=np.asarray(scores).ravel(), minlength=count + 1)
    boxes, strengths = [], []
    for index, (ys, xs) in enumerate(ndimage.find_objects(regions), 1):
        if area[index] >= min_area:
            boxes.append((xs.start, ys.start, xs.stop, ys.stop))
            strengths.append(total_scores[index] / area[index])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(strengths, dtype=np.float32)


class TrackFilter:
    """Keep strong detections and weak current boxes associated with past detections."""

    def __init__(self, high, low, match_iou=0.20, max_age=6, confirmations=1):
        self.high = high
        self.low = low
        self.match_iou = match_iou
        self.max_age = max_age
        self.confirmations = confirmations
        self.boxes = np.empty((0, 4), dtype=np.float32)
        self.seen = np.empty(0, dtype=int)
        self.track_ids = np.empty(0, dtype=int)
        self.next_id = 1
        self.strong_history = deque()

    def update(self, frame_index, boxes, scores):
        eligible = scores >= self.low
        boxes, scores = boxes[eligible], scores[eligible]
        strong = scores >= self.high
        while self.strong_history and frame_index - self.strong_history[0][0] > self.max_age:
            self.strong_history.popleft()
        confirmed = strong.copy()
        if self.confirmations == 1 and self.strong_history:
            predecessors = np.concatenate([value for _, value in self.strong_history])
            confirmed |= box_iou(boxes, predecessors).max(axis=1, initial=0) >= self.match_iou
        elif len(self.strong_history) >= self.confirmations:
            recent = [box_iou(boxes, value).max(axis=1, initial=0) >= self.match_iou for _, value in self.strong_history]
            confirmed |= np.sum(recent, axis=0) >= self.confirmations
        if strong.any():
            self.strong_history.append((frame_index, boxes[strong].copy()))
        boxes, scores = boxes[confirmed], scores[confirmed]
        active = frame_index - self.seen <= self.max_age
        old, times, tracks = self.boxes[active], self.seen[active], self.track_ids[active]
        assignment = matched_boxes(boxes, old, self.match_iou)
        ids = np.full(len(boxes), -1, dtype=int)
        ids[assignment[:, 0]] = tracks[assignment[:, 1]]
        new = ids < 0
        n = int(new.sum())
        ids[new] = np.arange(self.next_id, self.next_id + n)
        self.next_id += n
        chosen = ids > 0
        remaining = ~np.isin(tracks, ids[chosen])
        self.boxes = np.concatenate((old[remaining], boxes[chosen]))
        self.seen = np.concatenate((times[remaining], np.full(chosen.sum(), frame_index)))
        self.track_ids = np.concatenate((tracks[remaining], ids[chosen]))
        return boxes[chosen], scores[chosen], ids[chosen]
