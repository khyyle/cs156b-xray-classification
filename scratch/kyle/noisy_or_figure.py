"""Conceptual two-panel figure for the noisy-OR slide.

Left: a single linear head defines one half-space; a multimodal positive
set (an umbrella label) cannot be enclosed, so one subtype cluster is
stranded on the wrong side. Right: K latent subtype detectors each lasso
one cluster and OR-combine, covering the union.

Run: python scratch/kyle/noisy_or_figure.py
Writes scratch/kyle/noisy_or_figure.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

RNG = np.random.default_rng(7)
RED = "#d62728"
GREY = "#b0b0b0"

# Three positive subtype clusters (centers chosen so a single line cannot
# enclose all three) plus a diffuse negative background.
CLUSTERS = {
    "thickening": (2.2, 7.3),
    "plaque": (7.2, 7.6),
    "calcification": (7.6, 2.4),
}


def _scatter(ax):
    neg = RNG.uniform([0, 0], [10, 10], size=(320, 2))
    ax.scatter(neg[:, 0], neg[:, 1], s=10, c=GREY, alpha=0.7, linewidths=0, zorder=1)
    centers = {}
    for name, (cx, cy) in CLUSTERS.items():
        pts = RNG.normal([cx, cy], 0.7, size=(38, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=14, c=RED, alpha=0.9, linewidths=0, zorder=3)
        centers[name] = (cx, cy)
    return centers


def main() -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))

    # Left: single linear head 
    _scatter(axL)
    # One boundary; shade the "positive" half-space (upper-left) in red.
    xs = np.array([-0.5, 10.5])
    ys = 1.05 * xs + 1.0  # line: y = 1.05 x + 1
    axL.plot(xs, ys, color="black", lw=2.2, zorder=4)
    axL.fill_between(xs, ys, 11, color=RED, alpha=0.10, zorder=0)
    axL.set_title("Single linear head", fontsize=14)

    # Right: noisy-OR, K detectors (filled red to match left panel) 
    centers = _scatter(axR)
    for cx, cy in centers.values():
        axR.add_patch(Ellipse((cx, cy), 3.0, 2.4, lw=2.0, edgecolor="black",
                              facecolor=RED, alpha=0.18, zorder=2))
        axR.add_patch(Ellipse((cx, cy), 3.0, 2.4, fill=False, lw=2.0,
                              edgecolor="black", zorder=4))
    axR.set_title("Noisy-OR, $K$ subtype detectors", fontsize=14)

    for ax in (axL, axR):
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 11.2)
        ax.set_xticks([])
        ax.set_yticks([])

    # One shared axis label, centered under both panels.
    fig.text(0.5, 0.10, "backbone feature space (2D sketch)",
             ha="center", fontsize=11, color="0.3")
    # Shared legend along the bottom, out of the data area.
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=RED, label="Pleural Other positives"),
        plt.Line2D([], [], marker="o", ls="", color=GREY, label="negatives"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=(0, 0.10, 1, 0.97))
    out = Path(__file__).parent / "noisy_or_figure.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
