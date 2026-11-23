"""Plot recall and the axis-aligned rectangle limit on the locked test set."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


BANDS = ("1–11", "12–23", "24–47", "48+")
METHODS = {
    "unet_native_finetuned": ("Frozen U-Net", "#526683"),
    "local_contrast": ("Local contrast U-Net", "#2C7968"),
    "joint_mask": ("Joint model · mask", "#17679A"),
    "center_box": ("Joint model · box", "#BA601E"),
}


def plot(run, output):
    with (run / "recall_by_shadow_area.csv").open(encoding="utf-8-sig", newline="") as stream:
        recall = {(row["method"], row["label_area_px"]): row for row in csv.DictReader(stream)}
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    boxes = {row["group"]: row for row in results["box_diagnostics"]}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), layout="constrained")
    fig.suptitle("Center-based shadow candidates on independent SAR scenes", fontsize=15, weight="bold")
    x = np.arange(len(BANDS))
    for i, (method, (name, color)) in enumerate(METHODS.items()):
        values = [float(recall[method, band]["recall"]) if recall[method, band]["recall"] else np.nan for band in BANDS]
        axes[0].bar(x + 0.19 * (i - 1.5), values, width=0.18, label=name, color=color, zorder=3)
    for i, (key, name, color) in enumerate((
        ("best_rectangle_iou_matches", "Any rectangle could match", "#79689D"),
        ("predicted_box_iou_matches", "Predicted box before threshold", "#6D9EAA"),
        ("accepted_matches", "Accepted predicted box", "#BA601E"),
    )):
        values = [boxes[band][key] / boxes[band]["visible_frames"] if boxes[band]["visible_frames"] else np.nan
                  for band in BANDS]
        axes[1].bar(x + 0.24 * (i - 1), values, width=0.22, label=name, color=color, zorder=3)
    for axis in axes:
        axis.set_xticks(x, [f"{band} px\n{boxes[band]['visible_frames']} frames" for band in BANDS])
        axis.set_ylim(0, 1.04)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set(title="Recall by shadow area", ylabel="Share of visible frames")
    axes[1].set(title="Rectangle shape and score diagnosis", ylabel="Share with IoU ≥ 0.50")
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    axes[1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    output.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.run, args.output)
