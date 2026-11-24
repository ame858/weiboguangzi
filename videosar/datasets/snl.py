"""Sandia National Laboratories (SNL) Eubank Gate Ka-band Video-SAR dataset loader.

Reference:
- Footage: https://www.sandia.gov/app/uploads/sites/124/2021/08/eubankgateandtrafficvideosar.mp4
- Protocol: 660x720, 900 consecutive frames; standard split frames 1-600 train,
  601-900 test (with an explicit temporal guard gap so causal sliding windows
  never cross the train/test boundary).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np

from videosar.datasets.base import SequenceRecord, VideoSARSequenceDataset
from videosar.datasets.vsmb import parse_mot_or_darklabel_gt
from videosar.metrics import BoxDetection

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def extract_snl_mp4_to_frames(mp4_path: str | Path, out_dir: str | Path) -> List[Path]:
    """Utility to unpack SNL Eubank Gate MP4 footage into numbered PNG frames."""
    src = Path(mp4_path)
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open SNL video file: {src}")

    saved: List[Path] = []
    idx = 1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            frame_path = dst / f"frame_{idx:04d}.png"
            cv2.imwrite(str(frame_path), gray)
            saved.append(frame_path)
            idx += 1
    finally:
        cap.release()
    return saved


class SNLDataset(VideoSARSequenceDataset):
    """SNL Eubank Gate Ka-band Video-SAR dataset with temporal guard-gap splitting."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "test",
        gt_path: Optional[str | Path] = None,
        train_end_frame: int = 600,
        guard_gap_frames: int = 16,
        window_size: int = 5,
        window_stride: int = 1,
        tile_size: int = 512,
        tile_overlap: int = 64,
        normalize_db: bool = True,
        return_tensors: str = "numpy",
    ) -> None:
        if split not in {"train", "test", "all"}:
            raise ValueError(f"Unsupported SNL split '{split}'. Expected train/test/all.")

        root = Path(root_dir)
        frames_dir = root / "frames" if (root / "frames").is_dir() else root
        all_frames = sorted(
            p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )
        if not all_frames:
            raise FileNotFoundError(
                f"No image frames found under {frames_dir}. "
                "Use extract_snl_mp4_to_frames() to unpack the SNL MP4 first."
            )

        if gt_path is not None:
            resolved_gt = Path(gt_path)
        else:
            candidates = [root / "gt.txt", root / "gt" / "gt.txt", root / "annotations.csv"]
            resolved_gt = next((p for p in candidates if p.exists()), None)  # type: ignore[assignment]

        raw_ann: Dict[int, List[BoxDetection]] = (
            parse_mot_or_darklabel_gt(resolved_gt) if resolved_gt is not None else {}
        )

        # Temporal block split with guard gap to prevent window-overlap leakage
        n_total = len(all_frames)
        if split == "train":
            selected_indices = list(range(0, min(train_end_frame, n_total)))
        elif split == "test":
            test_start = min(train_end_frame + guard_gap_frames, n_total)
            selected_indices = list(range(test_start, n_total))
        else:
            selected_indices = list(range(0, n_total))

        selected_frames = [all_frames[i] for i in selected_indices]
        remapped_ann: Dict[int, List[BoxDetection]] = {}
        for new_pos, orig_idx in enumerate(selected_indices, start=1):
            orig_fid = orig_idx + 1
            if orig_fid in raw_ann:
                remapped_ann[new_pos] = [
                    BoxDetection(
                        frame_id=new_pos,
                        x1=b.x1,
                        y1=b.y1,
                        x2=b.x2,
                        y2=b.y2,
                        score=b.score,
                        class_id=b.class_id,
                        track_id=b.track_id,
                    )
                    for b in raw_ann[orig_fid]
                ]

        self.split = split
        self.original_frame_indices = [i + 1 for i in selected_indices]
        seq = SequenceRecord(
            sequence_id=f"snl_eubank_gate_{split}",
            frame_paths=selected_frames,
            annotations_by_frame=remapped_ann,
            source="real_snl",
        )
        super().__init__(
            sequences=[seq],
            window_size=window_size,
            window_stride=window_stride,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            normalize_db=normalize_db,
            return_tensors=return_tensors,
        )
