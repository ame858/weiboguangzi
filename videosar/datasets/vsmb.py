"""VSMB (Video SAR MOT Benchmark, 2025) dataset loader.

Reference:
- Paper: Multiple Object Tracking in Video SAR: A Benchmark and Tracking Baseline
  (arXiv:2506.12105, https://arxiv.org/html/2506.12105)
- Repo: https://github.com/softwarePupil/VSMB

Expected directory layouts supported automatically:
1. MOT-style layout:
   ``<root>/<seq_id>/img1/*.png|jpg`` + ``<root>/<seq_id>/gt/gt.txt``
2. Flat DarkLabel layout:
   ``<root>/<seq_id>/*.png|jpg`` + ``<root>/<seq_id>/gt.txt`` (or ``<seq_id>.txt``)
3. Pre-split layout:
   ``<root>/<split>/<seq_id>/...`` where ``split`` in ``{train, val, test}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence
import csv

from videosar.datasets.base import (
    SequenceRecord,
    VideoSARSequenceDataset,
    split_sequence_ids,
)
from videosar.metrics import BoxDetection

VSMB_CLASSES: Dict[int, str] = {
    1: "car",
    2: "ship",
    3: "airplane",
}
_NAME_TO_CLASS_ID: Dict[str, int] = {
    "car": 1,
    "vehicle": 1,
    "ship": 2,
    "vessel": 2,
    "airplane": 3,
    "plane": 3,
    "aircraft": 3,
}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_mot_or_darklabel_gt(gt_path: Path) -> Dict[int, List[BoxDetection]]:
    """Parse MOT Challenge / DarkLabel CSV/TXT annotations into frame-indexed boxes.

    Supported row schemas:
    - MOT standard: ``frame_id, track_id, x, y, w, h, conf, class_id, visibility``
    - DarkLabel xywh: ``frame_id, track_id, x, y, w, h[, class_name_or_id]``
    """
    annotations: Dict[int, List[BoxDetection]] = {}
    if not gt_path.exists():
        return annotations

    with gt_path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            cleaned = [c.strip() for c in row if c.strip() != ""]
            if len(cleaned) < 6:
                continue
            # Skip header rows if present
            try:
                frame_id = int(float(cleaned[0]))
                track_id = int(float(cleaned[1]))
                x = float(cleaned[2])
                y = float(cleaned[3])
                w = float(cleaned[4])
                h = float(cleaned[5])
            except ValueError:
                continue

            if w <= 0.0 or h <= 0.0:
                continue

            class_id = 1
            if len(cleaned) >= 8:
                raw_cls = cleaned[7].lower()
                if raw_cls in _NAME_TO_CLASS_ID:
                    class_id = _NAME_TO_CLASS_ID[raw_cls]
                else:
                    try:
                        class_id = int(float(raw_cls))
                    except ValueError:
                        class_id = 1
            elif len(cleaned) == 7:
                raw_cls = cleaned[6].lower()
                if raw_cls in _NAME_TO_CLASS_ID:
                    class_id = _NAME_TO_CLASS_ID[raw_cls]

            box = BoxDetection(
                frame_id=frame_id,
                x1=x,
                y1=y,
                x2=x + w,
                y2=y + h,
                score=1.0,
                class_id=class_id,
                track_id=track_id,
            )
            annotations.setdefault(frame_id, []).append(box)
    return annotations


def _discover_vsmb_sequences(root_dir: Path, split: Optional[str] = None) -> List[SequenceRecord]:
    search_root = root_dir / split if (split and (root_dir / split).is_dir()) else root_dir
    if not search_root.exists():
        raise FileNotFoundError(f"VSMB root directory not found: {search_root}")

    records: List[SequenceRecord] = []
    for seq_dir in sorted(p for p in search_root.iterdir() if p.is_dir()):
        img_dir = seq_dir / "img1" if (seq_dir / "img1").is_dir() else seq_dir
        frames = sorted(
            p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )
        if not frames:
            continue

        gt_candidates = [
            seq_dir / "gt" / "gt.txt",
            seq_dir / "gt.txt",
            seq_dir / f"{seq_dir.name}.txt",
            seq_dir / "annotations.csv",
        ]
        gt_path = next((p for p in gt_candidates if p.exists()), None)
        ann = parse_mot_or_darklabel_gt(gt_path) if gt_path is not None else {}

        records.append(
            SequenceRecord(
                sequence_id=seq_dir.name,
                frame_paths=frames,
                annotations_by_frame=ann,
                source="real_vsmb",
            )
        )
    return records


class VSMBDataset(VideoSARSequenceDataset):
    """VSMB PyTorch-compatible dataset with strict sequence-ID train/val/test splits."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        window_size: int = 5,
        window_stride: int = 1,
        tile_size: int = 512,
        tile_overlap: int = 64,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split_seed: int = 42,
        normalize_db: bool = True,
        return_tensors: str = "numpy",
        allowed_sequence_ids: Optional[Sequence[str]] = None,
    ) -> None:
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unsupported split '{split}'. Expected train/val/test/all.")

        root = Path(root_dir)
        has_presplit = (root / split).is_dir() and split != "all"
        all_seqs = _discover_vsmb_sequences(root, split=split if has_presplit else None)

        if allowed_sequence_ids is not None:
            allowed = set(allowed_sequence_ids)
            selected = [s for s in all_seqs if s.sequence_id in allowed]
        elif split == "all" or has_presplit:
            selected = all_seqs
        else:
            split_map = split_sequence_ids(
                [s.sequence_id for s in all_seqs],
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=split_seed,
            )
            allowed = set(split_map[split])
            selected = [s for s in all_seqs if s.sequence_id in allowed]

        self.split = split
        self.sequence_ids = [s.sequence_id for s in selected]
        super().__init__(
            sequences=selected,
            window_size=window_size,
            window_stride=window_stride,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            normalize_db=normalize_db,
            return_tensors=return_tensors,
        )
