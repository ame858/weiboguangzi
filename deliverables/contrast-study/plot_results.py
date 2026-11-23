"""Plot held-out detection results by operating point and shadow size."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


METHODS = {
    "unet_native_finetuned": ("Frozen\nU-Net", "#516478"),
    "retuned": ("Retuned\nU-Net", "#8C8F99"),
    "continued": ("Continued\nU-Net", "#17679A"),
    "local_contrast": ("Local contrast\nU-Net", "#2D7C64"),
    "fpn": ("Frozen\nSMP FPN", "#B75B20"),
    "difference": ("Background\ndifference", "#8060A1"),
}
AREA_METHODS = ("unet_native_finetuned", "continued", "local_contrast", "fpn")


def plot(root, output):
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    totals = {r["method"]: r for r in results["groups"] if r["group"] == "all"}
    with (root / "recall_by_shadow_area.csv").open(encoding="utf-8-sig", newline="") as stream:
        area = {(r["method"], r["label_area_px"]): r for r in csv.DictReader(stream)}
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 5.1), layout="constrained")
    fig.suptitle("Held-out SAR shadow detection at IoU ≥ 0.50", fontsize=15, weight="bold")
    bands = ("1–11", "12–23", "24–47", "48+")
    axis = axes[0]
    for i, method in enumerate(AREA_METHODS):
        name, color = METHODS[method]
        values = [float(area[method, band]["recall"]) if area[method, band]["recall"] else np.nan for band in bands]
        axis.bar(np.arange(len(bands)) + 0.19 * (i - 1.5), values, width=0.18, color=color,
                 label=name.replace("\n", " "), zorder=3)
    counts = [area[AREA_METHODS[0], band]["visible_frames"] for band in bands]
    axis.set_xticks(range(len(bands)), [f"{band} px\n{n} frames" for band, n in zip(bands, counts)])
    axis.set(title="Recall by visible shadow area", ylabel="Recall on visible frames")
    axis.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    axis = axes[1]
    for i, (metric, color) in enumerate((("precision", "#17679A"), ("recall", "#B75B20"))):
        axis.bar(np.arange(len(METHODS)) + 0.3 * (i - 0.5),
                 [totals[method][metric] for method in METHODS], width=0.28, color=color, label=metric.capitalize(), zorder=3)
    axis.axhline(0.8, color="#6A7176", linestyle="--", linewidth=0.9)
    axis.set_xticks(range(len(METHODS)), [label for label, _ in METHODS.values()], fontsize=8)
    axis.set_title("Overall precision and recall")
    axis.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    for axis in axes:
        axis.set_ylim(0, 1.03)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("run", type=Path)
    args.add_argument("output", type=Path)
    options = args.parse_args()
    plot(options.run, options.output)
