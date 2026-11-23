"""Plot the small-target paired aperture gains and their motion interaction."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONTRASTS = {
    "moving_91_minus_61": "Moving target\n91 minus 61 pulses",
    "held_91_minus_61": "Target held in frame\n91 minus 61 pulses",
    "interaction": "Difference of gains\nheld minus moving",
}


def plot(run, output):
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    estimates = {(r["metric"], r["contrast"]): r for r in results["paired"] if r["group"] == "small"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.25), layout="constrained")
    fig.suptitle("Within-aperture motion and small-shadow detection", fontsize=15, weight="bold")
    series = (
        (("local_contrast_db", "Local contrast", "#415867"),),
        (("unet_common_visible_recall", "U-Net", "#17679A"),
         ("fpn_common_visible_recall", "SMP FPN", "#B75B20"),
         ("difference_common_visible_recall", "Background difference", "#327360")),
    )
    for axis, measurements, scale in zip(axes, series, (1, 100)):
        for j, (metric, name, color) in enumerate(measurements):
            x = np.arange(len(CONTRASTS)) + 0.23 * (j - (len(measurements) - 1) / 2)
            rows = [estimates[metric, contrast] for contrast in CONTRASTS]
            point = np.array([r["delta"] for r in rows])
            low = np.array([r["ci95_low"] for r in rows])
            high = np.array([r["ci95_high"] for r in rows])
            axis.errorbar(x, scale * point, yerr=scale * np.stack([point - low, high - point]),
                          label=name, color=color, fmt="o", markersize=5, capsize=3, linewidth=1.5)
        axis.axhline(0, color="#6A7176", linestyle="--", linewidth=0.9)
        axis.set_xticks(range(len(CONTRASTS)), list(CONTRASTS.values()), fontsize=9)
        axis.set_xlim(-0.42, 2.42)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set(title="Observed shadow contrast", ylabel="Change in contrast (dB)")
    axes[1].set(title="Recall against one fixed label", ylabel="Change in recall (percentage points)")
    axes[1].legend(frameon=False, fontsize=8)
    output.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.run, args.output)
