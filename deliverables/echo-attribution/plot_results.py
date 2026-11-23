"""Render the paired SAR study figures with NumPy and Matplotlib."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


METHODS = {"unet": ("U-Net", "#17679A"), "fpn": ("SMP FPN", "#BB641F"),
           "difference": ("Background difference", "#467765")}
PAIRS = [
    ("bandwidth_at_25dB", "300 → 600 MHz at 25 dB"),
    ("bandwidth_at_10dB", "300 → 600 MHz at 10 dB"),
    ("noise_at_300MHz", "25 → 10 dB at 300 MHz"),
    ("noise_at_600MHz", "25 → 10 dB at 600 MHz"),
]
CONDITIONS = [(300, 25), (300, 10), (600, 25), (600, 10)]


def finish(fig, target):
    fig.savefig(target, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def overview(results, output):
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.4), layout="constrained")
    fig.suptitle("Paired single-channel SAR echo experiment", fontsize=16, weight="bold")
    lookup = {(r["group"], r["bandwidth_mhz"], r["snr_db"], r["method"]): r for r in results["detection"]}
    x = np.arange(len(CONDITIONS))
    for axis, group in zip(axes[0], ("small", "normal")):
        for i, (method, (label, color)) in enumerate(METHODS.items()):
            values = [lookup[group, bw, snr, method]["recall"] for bw, snr in CONDITIONS]
            axis.bar(x + 0.24 * (i - 1), values, width=0.22, color=color, label=label, zorder=3)
        axis.axhline(0.8, color="#6C7276", linestyle="--", linewidth=1)
        axis.set_xticks(x, [f"{bw} MHz\n{snr} dB" for bw, snr in CONDITIONS])
        axis.set(ylim=(0, 1.02), ylabel="Recall on visible frames", title=f"{group.capitalize()} targets")
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", alpha=0.22)
    axes[0, 0].legend(frameon=False, loc="upper left", ncol=2, fontsize=8)

    estimates = {(r["group"], r["contrast"], r["metric"]): r for r in results["paired"]}
    for axis, group in zip(axes[1], ("small", "normal")):
        for i, (method, (_, color)) in enumerate(METHODS.items()):
            y = np.arange(len(PAIRS)) + 0.19 * (i - 1)
            points = [estimates[group, pair, f"{method}_common_visible_recall"] for pair, _ in PAIRS]
            values = 100 * np.array([r["delta"] for r in points])
            low = 100 * np.array([r["ci95_low"] for r in points])
            high = 100 * np.array([r["ci95_high"] for r in points])
            axis.hlines(y, low, high, color=color, linewidth=1.6)
            axis.plot(values, y, "o", color=color, markersize=4)
        axis.axvline(0, color="#6C7276", linewidth=1)
        axis.set_yticks(np.arange(len(PAIRS)), [label for _, label in PAIRS], fontsize=9)
        axis.invert_yaxis()
        axis.set(xlabel="Paired recall change (percentage points)",
                 title=f"{group.capitalize()} targets · frames visible in both conditions")
        axis.grid(axis="x", alpha=0.22)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    finish(fig, output / "condition-recall.png")


def paired_frame(root, results, output):
    if results["mismatch_examples"]:
        row = results["mismatch_examples"][0]
        family, frame = row["scene_family"], row["frame"]
    else:
        with (root / "per_frame.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        by_frame = {}
        for row in rows:
            if row["group"] == "small" and int(row["visible_pixels"]):
                by_frame.setdefault((row["scene_family"], int(row["frame"])), []).append(row)
        common = [(key, values) for key, values in by_frame.items() if len(values) == len(CONDITIONS)]
        if not common:
            return
        def change(item):
            return sum(len({row[f"{method}_matched"] for row in item[1]}) - 1 for method in METHODS)
        family, frame = max(common, key=change)[0]

    data = []
    for bw, snr in CONDITIONS:
        scene = f"{family}-{bw}MHz-{snr}dB"
        source, prediction = root / "dataset" / scene, root / "predictions" / scene
        image = np.log1p(np.load(prediction / "aligned_frames.npy")[frame])
        geometry = np.load(source / "geometric_masks.npy")[frame]
        visible = np.load(source / "masks.npy")[frame]
        predictions = [np.load(prediction / f"{method}.npy")[frame] for method in METHODS]
        data.append((image, geometry, visible, predictions))
    lower, upper = np.percentile(np.stack([item[0] for item in data]), [2, 98])
    fig, axes = plt.subplots(4, 4, figsize=(11, 8.5), layout="constrained")
    fig.suptitle(f"Paired small-shadow example · {family}, frame {frame}", fontsize=15, weight="bold")
    for i, ((bw, snr), (image, geometry, visible, predictions)) in enumerate(zip(CONDITIONS, data)):
        for axis, mask in zip(axes[i], [None, *predictions]):
            axis.imshow(image, cmap="gray", vmin=lower, vmax=upper, interpolation="nearest")
            axis.contour(visible, levels=[0.5], colors=["#55E6D9"], linewidths=1.2)
            if mask is None and geometry.any():
                axis.contour(geometry, levels=[0.5], colors=["#FCDE71"], linewidths=0.9, linestyles="--")
            elif mask is not None and mask.any():
                axis.contour(mask, levels=[0.5], colors=["#FF827A"], linewidths=1.3)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[i, 0].set_ylabel(f"{bw} MHz\n{snr} dB")
    for axis, title in zip(axes[0], ("Registered input", *(label for label, _ in METHODS.values()))):
        axis.set_title(title, fontsize=11)
    legend = [Line2D([], [], color=color, linestyle=style, label=label) for color, style, label in (
        ("#49AFA7", "-", "Visible label"), ("#B09024", "--", "Geometry (input column)"),
        ("#C65046", "-", "Predicted region"))]
    fig.legend(handles=legend, loc="outside lower center", ncol=3, frameon=False)
    finish(fig, output / "paired-frame.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    results = json.loads((args.run / "results.json").read_text(encoding="utf-8"))
    overview(results, args.output)
    paired_frame(args.run, results, args.output)


if __name__ == "__main__":
    main()
