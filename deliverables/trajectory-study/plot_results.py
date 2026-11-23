"""Plot the paired small-target changes and overall false-positive cost."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = {"low_score": "Lower score threshold", "association": "Prior-track association",
           "accumulation": "Three-frame accumulation"}
COLORS = ("#4D6781", "#2D7865", "#B36127")


def plot(run, output):
    result = json.loads((run / "results.json").read_text(encoding="utf-8"))
    matches = {r["method"]: r for r in result["paired"]
               if r["reference"] == "joint_mask" and r["group"] == "small"}
    overall = {r["method"]: r for r in result["groups"] if r["group"] == "all"}
    baseline = overall["joint_mask"]["false_positive"]
    y = np.arange(len(METHODS))
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 3.55), layout="constrained")
    fig.suptitle("Weak-candidate trade-offs on 36 independent SAR sequences", fontsize=15, weight="bold")
    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        row = matches[method]
        axes[0].hlines(i, 100 * row["ci95_low"], 100 * row["ci95_high"], color=color, linewidth=2)
        axes[0].plot(100 * row["delta"], i, "o", color=color, markersize=6)
        extra = overall[method]["false_positive"] - baseline
        axes[1].barh(i, extra, color=color, height=0.48)
        axes[1].text(extra + 0.15, i, f"{extra:+d}", va="center", fontsize=10)
    axes[0].set(xlabel="Paired recall change (percentage points)", title="Small targets · 95% scene bootstrap interval")
    axes[1].set(xlabel="Additional false-positive regions", title="Overall false-positive cost", xlim=(-0.25, 7))
    for axis in axes:
        axis.axvline(0, color="#6A7176", linewidth=0.9)
        axis.set_yticks(y, list(METHODS.values()) if axis is axes[0] else [])
        axis.set_ylim(len(y) - 0.5, -0.5)
        axis.grid(axis="x", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.run, args.output)
