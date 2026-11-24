"""Lightweight ByteTrack-style multi-object tracker & temporal shadow baseline.

Provides an immediate, zero-GPU baseline so the refactored repository can run
end-to-end detection + tracking + FROC/mAP/MOTA/IDF1/HOTA evaluation right out
of the box before plugging in heavy deep detectors (e.g., DINO / YOLO / RT-DETR).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from videosar.metrics import BoxDetection, compute_iou_matrix


@dataclass
class _TrackState:
    track_id: int
    box: np.ndarray          # [x1, y1, x2, y2]
    velocity: np.ndarray     # [vx, vy]
    class_id: int
    score: float
    age: int = 0
    hits: int = 1

    def predict(self) -> np.ndarray:
        shifted = self.box.copy()
        shifted[0] += self.velocity[0]
        shifted[1] += self.velocity[1]
        shifted[2] += self.velocity[0]
        shifted[3] += self.velocity[1]
        return shifted

    def update(self, new_box: np.ndarray, score: float) -> None:
        old_c = np.array(
            [(self.box[0] + self.box[2]) * 0.5, (self.box[1] + self.box[3]) * 0.5]
        )
        new_c = np.array(
            [(new_box[0] + new_box[2]) * 0.5, (new_box[1] + new_box[3]) * 0.5]
        )
        self.velocity = 0.6 * self.velocity + 0.4 * (new_c - old_c)
        self.box = new_box.copy()
        self.score = score
        self.age = 0
        self.hits += 1


class ByteTrackLite:
    """Two-stage (high-score + low-score) motion-assisted IoU tracker."""

    def __init__(
        self,
        high_thresh: float = 0.45,
        low_thresh: float = 0.15,
        match_iou_thresh: float = 0.25,
        max_age: int = 5,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_iou_thresh = match_iou_thresh
        self.max_age = max_age
        self._next_id = 1
        self._tracks: List[_TrackState] = []

    def reset(self) -> None:
        self._next_id = 1
        self._tracks = []

    def _associate(
        self,
        tracks: List[_TrackState],
        dets: List[BoxDetection],
        iou_thresh: float,
    ):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))

        tr_boxes = np.stack([t.predict() for t in tracks], axis=0)
        det_boxes = np.stack([d.as_xyxy() for d in dets], axis=0)
        iou = compute_iou_matrix(tr_boxes, det_boxes)
        row_idx, col_idx = linear_sum_assignment(1.0 - iou)

        matches = []
        matched_t = set()
        matched_d = set()
        for r, c in zip(row_idx, col_idx):
            if iou[r, c] >= iou_thresh:
                matches.append((int(r), int(c)))
                matched_t.add(int(r))
                matched_d.add(int(c))
        unm_t = [i for i in range(len(tracks)) if i not in matched_t]
        unm_d = [j for j in range(len(dets)) if j not in matched_d]
        return matches, unm_t, unm_d

    def update(self, frame_id: int, detections: Sequence[BoxDetection]) -> List[BoxDetection]:
        high_dets = [d for d in detections if d.score >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.score < self.high_thresh]

        # Stage 1: match active tracks to high-confidence detections
        m1, unm_t1, unm_d1 = self._associate(self._tracks, high_dets, self.match_iou_thresh)
        for ti, di in m1:
            self._tracks[ti].update(high_dets[di].as_xyxy(), high_dets[di].score)

        # Stage 2: recover occluded/defocused targets using low-confidence detections
        rem_tracks = [self._tracks[i] for i in unm_t1]
        m2, unm_t2_local, _ = self._associate(rem_tracks, low_dets, max(0.15, self.match_iou_thresh - 0.1))
        for local_ti, di in m2:
            rem_tracks[local_ti].update(low_dets[di].as_xyxy(), low_dets[di].score)

        # Age unmatched tracks
        for local_ti in unm_t2_local:
            tr = rem_tracks[local_ti]
            tr.box = tr.predict()
            tr.age += 1

        # Spawn new tracks from unmatched high-confidence detections
        for di in unm_d1:
            d = high_dets[di]
            self._tracks.append(
                _TrackState(
                    track_id=self._next_id,
                    box=d.as_xyxy(),
                    velocity=np.zeros(2, dtype=np.float64),
                    class_id=d.class_id,
                    score=d.score,
                )
            )
            self._next_id += 1

        # Prune expired tracks
        self._tracks = [t for t in self._tracks if t.age <= self.max_age]

        outputs: List[BoxDetection] = []
        for tr in self._tracks:
            if tr.age == 0:
                outputs.append(
                    BoxDetection(
                        frame_id=frame_id,
                        x1=float(tr.box[0]),
                        y1=float(tr.box[1]),
                        x2=float(tr.box[2]),
                        y2=float(tr.box[3]),
                        score=float(tr.score),
                        class_id=tr.class_id,
                        track_id=tr.track_id,
                    )
                )
        return outputs


def detect_moving_shadows_baseline(
    frames: np.ndarray,
    z_thresh: float = 2.2,
    min_area_px: int = 24,
    max_area_px: int = 1800,
) -> List[BoxDetection]:
    """Temporal median-background shadow detector baseline for smoke testing.

    Because static building shadows persist in the temporal median, subtracting
    ``frame`` from ``median_bg`` selectively highlights moving dark shadows while
    suppressing stationary clutter and static building distractors.
    """
    if frames.ndim != 3 or frames.shape[0] < 2:
        raise ValueError("Expected frames array of shape (T, H, W) with T >= 2.")

    bg_median = np.median(frames, axis=0).astype(np.float32)
    detections: List[BoxDetection] = []

    for t_idx in range(frames.shape[0]):
        # Positive deficit where current frame is significantly darker than median bg
        deficit = bg_median - frames[t_idx]
        smooth = cv2.GaussianBlur(deficit, (5, 5), 1.0)
        std = float(np.std(smooth)) + 1e-6
        z_map = smooth / std
        mask = (z_map >= z_thresh).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for lbl in range(1, n_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area < min_area_px or area > max_area_px:
                continue
            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            w = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            peak_z = float(np.max(z_map[y : y + h, x : x + w]))
            score = float(1.0 / (1.0 + np.exp(-(peak_z - z_thresh))))
            detections.append(
                BoxDetection(
                    frame_id=t_idx + 1,
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + w),
                    y2=float(y + h),
                    score=score,
                    class_id=1,
                    track_id=-1,
                )
            )
    return detections
