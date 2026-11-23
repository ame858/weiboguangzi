"""Plot the aperture calibration and paired detection results."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


METHODS = {"unet": ("U-Net", "#17679A"), "fpn": ("SMP FPN", "#B75B20"),
           "difference": ("Background difference", "#327360")}
CONTRASTS = {"61_minus_31": "31 → 61", "91_minus_61": "61 → 91", "91_minus_31": "31 → 91"}


def plot(run, output):
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    with (run / "azimuth_response.csv").open(encoding="utf-8-sig", newline="") as stream:
        curves = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    fig.suptitle("Effect of SAR aperture duration at 600 MHz and 25 dB", fontsize=16, weight="bold")
    counts = [r["pulses"] for r in results["point"]]
    point = axes[0, 0]
    for row, color in zip(results["point"], ("#805495", "#17679A", "#B75B20")):
        values = [(float(r["offset_m"]), float(r["normalized_power"])) for r in curves
                  if int(r["pulses"]) == row["pulses"]]
        x, y = np.asarray(values).T
        width = row["azimuth_mainlobe_half_power_width_m"]
        point.plot(x, y, color=color, label=f"{row['pulses']} pulses · {width:.3f} m")
    point.axhline(0.5, linestyle="--", color="#707578", linewidth=0.9)
    point.set(xlim=(-0.9, 0.9), ylim=(0, 1.05), xlabel="Along-track offset (m)", ylabel="Normalized power",
              title="Stationary point response")
    point.legend(frameon=False, fontsize=8, loc="upper right")
    recall = {(r["group"], r["pulses"], r["method"]): r["recall"] for r in results["detection"]}
    for axis, group in zip((axes[0, 1], axes[1, 0]), ("small", "normal")):
        for method, (label, color) in METHODS.items():
            axis.plot(counts, [recall[group, n, method] for n in counts], "o-", color=color, label=label)
        axis.axhline(0.8, linestyle="--", color="#707578", linewidth=0.9)
        axis.set_xticks(counts, [f"{n} pulses\n{r['aperture_span_s']:.2f} s" for n, r in zip(counts, results["point"])])
        axis.set(ylim=(0, 1.05), ylabel="Recall on visible frames", title=f"{group.capitalize()} targets")
        axis.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 1].legend(frameon=False, fontsize=8, loc="upper left")
    axis = axes[1, 1]
    paired = {(r["contrast"], r["metric"]): r for r in results["paired"] if r["group"] == "small"}
    for j, (method, (_, color)) in enumerate(METHODS.items()):
        for i, contrast in enumerate(CONTRASTS):
            row = paired[contrast, f"{method}_common_visible_recall"]
            y = i + 0.19 * (j - 1)
            axis.hlines(y, 100 * row["ci95_low"], 100 * row["ci95_high"], color=color, linewidth=1.6)
            axis.plot(100 * row["delta"], y, "o", color=color, markersize=4)
    axis.axvline(0, color="#707578", linewidth=0.9)
    axis.set_yticks(range(len(CONTRASTS)), list(CONTRASTS.values()))
    axis.invert_yaxis()
    axis.set(xlabel="Recall change (percentage points)", title="Small targets · paired visible frames")
    axis.grid(axis="x", alpha=0.25)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        if axis is not axes[1, 1]:
            axis.grid(axis="y", alpha=0.25)
    output.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.run, args.output)
