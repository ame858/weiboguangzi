"""Base abstractions for real Video-SAR datasets.

Provides:
- PyTorch ``Dataset`` compatibility (gracefully falls back to pure NumPy when
  ``torch`` is absent in lightweight CI environments).
- Causal sliding temporal windows ``[t - W + 1, ..., t]`` (zero look-ahead leakage).
- Overlapping spatial tiling (default ``512x512`` with ``64px`` overlap) with exact
  bounding-box clipping and coordinate remapping.
- Strict sequence-ID splitting so frames from the same video never leak across
  train/val/test partitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from videosar.metrics import BoxDetection

try:
    import torch
    from torch.utils.data import Dataset as _BaseDataset

    HAS_TORCH = True
except ImportError:  # pragma: no cover - pure NumPy environment
    torch = None
    HAS_TORCH = False

    class _BaseDataset:  # type: ignore[no-redef]
        pass


@dataclass
class SequenceRecord:
    """Metadata and frame-level annotations for a single Video-SAR sequence."""

    sequence_id: str
    frame_paths: List[Path]
    annotations_by_frame: Dict[int, List[BoxDetection]] = field(default_factory=dict)
    source: str = "real"
    width: int = 0
    height: int = 0


@dataclass
class TemporalWindowSample:
    """Single tiled causal temporal window returned by ``VideoSARSequenceDataset``."""

    sequence_id: str
    frame_indices: List[int]
    tile_box: Tuple[int, int, int, int]      # (x0, y0, x1, y1) in original image coords
    frames: Any                              # ndarray or torch.Tensor of shape (W, H, W_tile)
    boxes: np.ndarray                        # (N, 4) xyxy in tile coordinates
    track_ids: np.ndarray                    # (N,) int64
    class_ids: np.ndarray                    # (N,) int64
    source: str = "real"


def split_sequence_ids(
    sequence_ids: Sequence[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Deterministically partition unique sequence IDs into disjoint train/val/test sets."""
    unique_ids = sorted(set(sequence_ids))
    if len(unique_ids) != len(sequence_ids):
        raise ValueError("Duplicate sequence IDs detected before splitting.")

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0.")

    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(unique_ids))
    n = len(perm)

    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        return {"train": perm, "val": [], "test": []}
    if n == 2:
        return {"train": [perm[0]], "val": [], "test": [perm[1]]}

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    n_test = n - n_train - n_val

    splits = {
        "train": sorted(perm[:n_train]),
        "val": sorted(perm[n_train : n_train + n_val]),
        "test": sorted(perm[n_train + n_val : n_train + n_val + n_test]),
    }

    # Explicit zero-leakage assertion
    assert not (set(splits["train"]) & set(splits["val"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["val"]) & set(splits["test"]))
    return splits


def compute_tile_grid(
    height: int,
    width: int,
    tile_size: int = 512,
    overlap: int = 64,
) -> List[Tuple[int, int, int, int]]:
    """Compute overlapping ``(x0, y0, x1, y1)`` tiles covering ``(height, width)``."""
    if tile_size <= 0:
        return [(0, 0, width, height)]
    if height <= tile_size and width <= tile_size:
        return [(0, 0, width, height)]

    stride = max(1, tile_size - overlap)

    def _starts(dim: int) -> List[int]:
        if dim <= tile_size:
            return [0]
        pts = list(range(0, dim - tile_size + 1, stride))
        if pts[-1] + tile_size < dim:
            pts.append(dim - tile_size)
        return pts

    tiles: List[Tuple[int, int, int, int]] = []
    for y0 in _starts(height):
        for x0 in _starts(width):
            tiles.append((x0, y0, min(x0 + tile_size, width), min(y0 + tile_size, height)))
    return tiles


def load_grayscale_frame(path: Path, normalize_db: bool = True) -> np.ndarray:
    """Read a Video-SAR frame as float32 in ``[0, 1]``."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Unable to read image frame: {path}")
    arr = img.astype(np.float32) / 255.0
    if normalize_db:
        # Gentle log-amplitude compression consistent with SAR intensity visualization
        arr = np.log1p(arr * 255.0) / np.log1p(255.0)
    return arr.astype(np.float32)


def clip_boxes_to_tile(
    boxes: Sequence[BoxDetection],
    tile_box: Tuple[int, int, int, int],
    min_visible_frac: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translate and clip frame boxes into local tile coordinates."""
    tx0, ty0, tx1, ty1 = tile_box
    out_boxes: List[List[float]] = []
    out_tids: List[int] = []
    out_cids: List[int] = []

    for b in boxes:
        orig_area = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
        if orig_area <= 0.0:
            continue
        nx1 = max(b.x1, float(tx0)) - float(tx0)
        ny1 = max(b.y1, float(ty0)) - float(ty0)
        nx2 = min(b.x2, float(tx1)) - float(tx0)
        ny2 = min(b.y2, float(ty1)) - float(ty0)
        new_area = max(0.0, nx2 - nx1) * max(0.0, ny2 - ny1)
        if new_area / orig_area >= min_visible_frac:
            out_boxes.append([nx1, ny1, nx2, ny2])
            out_tids.append(int(b.track_id))
            out_cids.append(int(b.class_id))

    if not out_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.asarray(out_boxes, dtype=np.float32),
        np.asarray(out_tids, dtype=np.int64),
        np.asarray(out_cids, dtype=np.int64),
    )


class VideoSARSequenceDataset(_BaseDataset):
    """Unified causal-window, tiled Video-SAR dataset."""

    def __init__(
        self,
        sequences: Sequence[SequenceRecord],
        window_size: int = 5,
        window_stride: int = 1,
        tile_size: int = 512,
        tile_overlap: int = 64,
        normalize_db: bool = True,
        return_tensors: str = "numpy",
        frame_loader: Optional[Callable[[SequenceRecord, int], np.ndarray]] = None,
    ) -> None:
        self.sequences = list(sequences)
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.tile_size = int(tile_size)
        self.tile_overlap = int(tile_overlap)
        self.normalize_db = bool(normalize_db)
        self.return_tensors = return_tensors
        self._frame_loader = frame_loader

        # Build flat index of (seq_idx, target_frame_pos, tile_box)
        self.index: List[Tuple[int, int, Tuple[int, int, int, int]]] = []
        for s_idx, seq in enumerate(self.sequences):
            n_frames = len(seq.frame_paths)
            if n_frames == 0:
                continue
            h, w = seq.height, seq.width
            if h <= 0 or w <= 0:
                sample_img = self._load_frame(seq, 0)
                h, w = sample_img.shape[:2]
                seq.height, seq.width = h, w
            tiles = compute_tile_grid(h, w, tile_size=self.tile_size, overlap=self.tile_overlap)
            start_pos = min(self.window_size - 1, n_frames - 1)
            for pos in range(start_pos, n_frames, self.window_stride):
                for tbox in tiles:
                    self.index.append((s_idx, pos, tbox))

    def _load_frame(self, seq: SequenceRecord, pos: int) -> np.ndarray:
        if self._frame_loader is not None:
            return self._frame_loader(seq, pos)
        return load_grayscale_frame(seq.frame_paths[pos], normalize_db=self.normalize_db)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> TemporalWindowSample:
        s_idx, target_pos, tbox = self.index[idx]
        seq = self.sequences[s_idx]
        tx0, ty0, tx1, ty1 = tbox

        # Strictly causal window: [target_pos - W + 1, ..., target_pos]
        # Pads earliest frame if sequence is shorter than window_size.
        win_positions = [
            max(0, target_pos - (self.window_size - 1 - k))
            for k in range(self.window_size)
        ]
        crops = [
            self._load_frame(seq, p)[ty0:ty1, tx0:tx1]
            for p in win_positions
        ]
        stack = np.stack(crops, axis=0).astype(np.float32)  # (W, tile_h, tile_w)

        # Annotations correspond to the current (latest) causal frame (1-indexed frame_id)
        target_frame_id = target_pos + 1
        frame_gts = seq.annotations_by_frame.get(target_frame_id, [])
        boxes, track_ids, class_ids = clip_boxes_to_tile(frame_gts, tbox)

        frames_out: Any = stack
        if self.return_tensors == "pt":
            if not HAS_TORCH:
                raise ImportError("PyTorch is not installed; set return_tensors='numpy'.")
            frames_out = torch.from_numpy(stack)

        return TemporalWindowSample(
            sequence_id=seq.sequence_id,
            frame_indices=[p + 1 for p in win_positions],
            tile_box=tbox,
            frames=frames_out,
            boxes=boxes,
            track_ids=track_ids,
            class_ids=class_ids,
            source=seq.source,
        )
