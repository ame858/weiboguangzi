"""Unified causal shadow detector interface and registry for Video-SAR v2.0.

Every detector exposes the same ``predict_sequence`` contract: it receives a
``(T, H, W)`` grayscale sequence and returns one ``List[BoxDetection]`` per
frame, in frame order. Implementations MUST be strictly causal -- detections at
index ``t`` may depend only on input frames ``0..t``, never on future pixels or
future global statistics.

The default ``baseline`` detector wraps the existing zero-GPU temporal-median
baseline without changing its behaviour or its public export.
"""

from __future__ import annotations

from typing import Dict, List, Type

import numpy as np

from videosar.metrics import BoxDetection
from videosar.tracker import detect_moving_shadows_baseline


class CausalShadowDetector:
    """Interface for shadow detectors that never peek at future frames."""

    name: str = "base"

    def predict_sequence(self, frames: np.ndarray) -> List[List[BoxDetection]]:
        """Detect shadows in a ``(T, H, W)`` sequence.

        Returns one list of :class:`BoxDetection` per input frame. The result at
        index ``t`` must be computable from frames ``0..t`` alone.
        """
        raise NotImplementedError


class BaselineCausalDetector(CausalShadowDetector):
    """Zero-GPU baseline wrapping :func:`detect_moving_shadows_baseline`.

    The underlying detector builds each frame's background from the per-pixel
    median of frames ``0..t`` only, so it is strictly causal by construction.
    """

    name = "baseline"

    def predict_sequence(self, frames: np.ndarray) -> List[List[BoxDetection]]:
        flat_detections = detect_moving_shadows_baseline(frames)
        per_frame: List[List[BoxDetection]] = [[] for _ in range(frames.shape[0])]
        for det in flat_detections:
            frame_index = det.frame_id - 1  # baseline uses 1-based frame ids
            if 0 <= frame_index < len(per_frame):
                per_frame[frame_index].append(det)
        return per_frame


DETECTOR_REGISTRY: Dict[str, Type[CausalShadowDetector]] = {
    "baseline": BaselineCausalDetector,
}


def list_detectors() -> List[str]:
    """Return the sorted names of all registered detectors."""
    return sorted(DETECTOR_REGISTRY)


def get_detector(name: str, **kwargs) -> CausalShadowDetector:
    """Instantiate a registered detector by name.

    Args:
        name: Key into :data:`DETECTOR_REGISTRY`.
        **kwargs: Forwarded to the detector constructor.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    try:
        detector_cls = DETECTOR_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown detector '{name}'. Available detectors: {', '.join(list_detectors())}"
        ) from exc
    return detector_cls(**kwargs)
