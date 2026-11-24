"""Reproducible experiment runner with strict real/synthetic separation.

Key guarantees:
1. Refuses to pool ``synthetic`` and ``real`` splits into a single aggregate score.
2. Logs exact Git commit hash, random seed, UTC timestamp, config snapshot, and
   dataset source tags into standardized ``metrics.json``, ``summary.csv``, and
   ``froc_curve.csv`` files.
3. Eliminates dead links to uncommitted run folders by writing self-contained
   deliverable directories.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence
import csv
import json
import subprocess

from videosar.config import VideoSARConfig
from videosar.metrics import (
    BoxDetection,
    evaluate_detection_and_froc,
    evaluate_mot,
)


def get_git_commit_hash(cwd: str | Path | None = None) -> str:
    """Return current HEAD commit hash or 'uncommitted' outside a git repository."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "uncommitted"
    except Exception:
        return "uncommitted"


class ExperimentRunner:
    """Evaluates and persists isolated real-benchmark vs. synthetic-ablation results."""

    def __init__(self, config: VideoSARConfig, run_dir: str | Path | None = None) -> None:
        config.validate()
        self.config = config
        self.run_dir = Path(run_dir) if run_dir else Path(config.output_dir) / config.experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.records: Dict[str, Dict[str, object]] = {}

    def evaluate_split(
        self,
        split_name: str,
        domain: str,
        predictions: Sequence[BoxDetection],
        ground_truths: Sequence[BoxDetection],
        num_frames: int,
        tracked_predictions: Sequence[BoxDetection] | None = None,
    ) -> Dict[str, object]:
        """Evaluate one isolated domain split ('real' or 'synthetic')."""
        if domain not in {"real", "synthetic"}:
            raise ValueError(
                f"domain must be 'real' or 'synthetic', got '{domain}'. "
                "Mixed synthetic+real splits are strictly forbidden."
            )

        det_50 = evaluate_detection_and_froc(
            predictions=predictions,
            ground_truths=ground_truths,
            num_frames=num_frames,
            iou_threshold=self.config.evaluation.iou_threshold,
            fa_points=self.config.evaluation.froc_fa_points,
        )
        det_30 = evaluate_detection_and_froc(
            predictions=predictions,
            ground_truths=ground_truths,
            num_frames=num_frames,
            iou_threshold=self.config.evaluation.small_target_iou_threshold,
            fa_points=self.config.evaluation.froc_fa_points,
        )

        mot_preds = tracked_predictions if tracked_predictions is not None else predictions
        mot_res = evaluate_mot(
            predictions=mot_preds,
            ground_truths=ground_truths,
            iou_threshold=self.config.evaluation.iou_threshold,
            hota_iou_thresholds=self.config.evaluation.hota_iou_thresholds,
        )

        entry: Dict[str, object] = {
            "split_name": split_name,
            "domain": domain,
            "num_frames": num_frames,
            "num_gt": det_50["num_gt"],
            "num_pred": det_50["num_pred"],
            "empty_scene": det_50["empty_scene"],
            "map_50": det_50["ap"],
            "map_30": det_30["ap"],
            "precision_50": det_50["precision"],
            "recall_50": det_50["recall"],
            "fa_per_frame_50": det_50["fa_per_frame"],
            "recall_at_fa_50": det_50["recall_at_fa"],
            "froc_curve_50": det_50["froc_curve"],
            "mot": mot_res,
        }
        self.records[split_name] = entry
        return entry

    def assert_no_mixed_pooling(self, split_names: Sequence[str]) -> None:
        """Guard that prevents callers from averaging real and synthetic splits together."""
        domains = {str(self.records[name]["domain"]) for name in split_names if name in self.records}
        if len(domains) > 1:
            raise ValueError(
                f"Cannot aggregate across mixed domains {sorted(domains)}. "
                "Report real blind benchmarks and synthetic ablations separately."
            )

    def save_artifacts(self) -> Dict[str, Path]:
        """Write reproducible config, metrics.json, summary.csv, and froc_curve.csv."""
        cfg_path = self.config.save(self.run_dir / "config.json")

        payload = {
            "experiment_name": self.config.experiment_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit_hash(),
            "seed": self.config.seed,
            "real_blind_eval": {
                k: v for k, v in self.records.items() if v["domain"] == "real"
            },
            "synthetic_ablation": {
                k: v for k, v in self.records.items() if v["domain"] == "synthetic"
            },
        }
        metrics_path = self.run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        summary_path = self.run_dir / "summary.csv"
        fieldnames = [
            "split_name",
            "domain",
            "num_frames",
            "num_gt",
            "num_pred",
            "map_50",
            "map_30",
            "precision_50",
            "recall_50",
            "fa_per_frame_50",
            "recall_at_1_fa",
            "mota",
            "idf1",
            "hota",
            "id_switches",
        ]
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.records.values():
                mot = rec["mot"]
                r_fa = rec["recall_at_fa_50"]
                writer.writerow(
                    {
                        "split_name": rec["split_name"],
                        "domain": rec["domain"],
                        "num_frames": rec["num_frames"],
                        "num_gt": rec["num_gt"],
                        "num_pred": rec["num_pred"],
                        "map_50": f"{float(rec['map_50']):.4f}",
                        "map_30": f"{float(rec['map_30']):.4f}",
                        "precision_50": f"{float(rec['precision_50']):.4f}",
                        "recall_50": f"{float(rec['recall_50']):.4f}",
                        "fa_per_frame_50": f"{float(rec['fa_per_frame_50']):.4f}",
                        "recall_at_1_fa": f"{float(r_fa.get('recall_at_1_fa', 0.0)):.4f}",
                        "mota": f"{float(mot['mota']):.4f}",
                        "idf1": f"{float(mot['idf1']):.4f}",
                        "hota": f"{float(mot['hota']):.4f}",
                        "id_switches": int(mot["id_switches"]),
                    }
                )

        froc_path = self.run_dir / "froc_curve.csv"
        with froc_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["split_name", "domain", "fa_per_frame", "recall", "score"]
            )
            writer.writeheader()
            for rec in self.records.values():
                for pt in rec["froc_curve_50"]:
                    writer.writerow(
                        {
                            "split_name": rec["split_name"],
                            "domain": rec["domain"],
                            "fa_per_frame": f"{pt['fa_per_frame']:.6f}",
                            "recall": f"{pt['recall']:.6f}",
                            "score": f"{pt['score']:.6f}",
                        }
                    )

        return {
            "config": cfg_path,
            "metrics": metrics_path,
            "summary": summary_path,
            "froc": froc_path,
        }
