"""Unified dataset factory and anti-leakage verification utilities."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from videosar.config import DatasetConfig
from videosar.datasets.snl import SNLDataset
from videosar.datasets.vsmb import VSMBDataset


def build_real_dataset(cfg: DatasetConfig, split: str = "test"):
    """Instantiate a real Video-SAR benchmark dataset from ``DatasetConfig``."""
    cfg.validate()
    if cfg.name == "vsmb":
        return VSMBDataset(
            root_dir=cfg.root_dir,
            split=split,
            window_size=cfg.window_size,
            window_stride=cfg.window_stride,
            tile_size=cfg.tile_size,
            tile_overlap=cfg.tile_overlap,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
            split_seed=cfg.split_seed,
            normalize_db=cfg.normalize_db,
            return_tensors=cfg.return_tensors,
        )
    if cfg.name == "snl":
        return SNLDataset(
            root_dir=cfg.root_dir,
            split=split,
            train_end_frame=cfg.snl_train_end_frame,
            guard_gap_frames=cfg.snl_guard_gap_frames,
            window_size=cfg.window_size,
            window_stride=cfg.window_stride,
            tile_size=cfg.tile_size,
            tile_overlap=cfg.tile_overlap,
            normalize_db=cfg.normalize_db,
            return_tensors=cfg.return_tensors,
        )
    raise ValueError(
        f"build_real_dataset() only accepts real benchmarks ('vsmb' or 'snl'), got '{cfg.name}'."
    )


def verify_no_sequence_leakage(
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
) -> Dict[str, int]:
    """Verify train/val/test sequence sets are strictly disjoint."""
    s_train, s_val, s_test = set(train_ids), set(val_ids), set(test_ids)
    overlap_tv = s_train & s_val
    overlap_tt = s_train & s_test
    overlap_vt = s_val & s_test
    if overlap_tv or overlap_tt or overlap_vt:
        raise AssertionError(
            f"Sequence leakage detected! train∩val={overlap_tv}, "
            f"train∩test={overlap_tt}, val∩test={overlap_vt}"
        )
    return {"train": len(s_train), "val": len(s_val), "test": len(s_test)}
