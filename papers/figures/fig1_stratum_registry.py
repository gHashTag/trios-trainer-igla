#!/usr/bin/env python3
"""Figure 2 — Stratum enum + mode-string registry flow.

Structural diagram (not data-driven): shows how race::ablation::Stratum +
ModeKind combine via mode_string() to produce the CSV mode tags emitted by
f2_ablation_sweep and consumed by f2_dual_mediation's stratum-registry
lookups.

This is the architecture diagram that explains how adding a new Stratum
variant auto-extends every downstream lookup without touching binary code.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from fig_template import save_figure


STRATA = [
    ("Canonical", "(default)"),
    ("Wd0", "wd0_"),
    ("Warmup0", "warmup0_"),
]

MODE_KINDS = [
    ("Loco", "loco"),
    ("Pairwise", "pairwise"),
    ("Triplet", "triplet"),
]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="papers/figures/fig2_stratum_registry.png")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Left column: Stratum enum.
    ax.text(1.0, 6.4, "race::ablation::Stratum",
            fontsize=11, fontweight="bold", ha="left")
    for i, (name, prefix) in enumerate(STRATA):
        y = 5.5 - i * 0.7
        rect = mpatches.FancyBboxPatch(
            (0.5, y - 0.25), 2.3, 0.55,
            boxstyle="round,pad=0.05",
            facecolor="#dfeaf6", edgecolor="#2c5d8f", linewidth=1.4,
        )
        ax.add_patch(rect)
        ax.text(0.7, y, f"::{name}", fontsize=10, fontweight="bold",
                color="#2c5d8f", va="center")
        ax.text(2.7, y - 0.05, f"prefix={prefix!r}", fontsize=8, color="#555555",
                va="center", ha="right")

    # Middle column: ModeKind enum.
    ax.text(4.5, 6.4, "ModeKind", fontsize=11, fontweight="bold", ha="left")
    for i, (name, base) in enumerate(MODE_KINDS):
        y = 5.5 - i * 0.7
        rect = mpatches.FancyBboxPatch(
            (4.2, y - 0.25), 2.0, 0.55,
            boxstyle="round,pad=0.05",
            facecolor="#e8f0d8", edgecolor="#5a7d2c", linewidth=1.4,
        )
        ax.add_patch(rect)
        ax.text(4.4, y, f"::{name}", fontsize=10, fontweight="bold",
                color="#5a7d2c", va="center")
        ax.text(6.1, y - 0.05, f"base={base!r}", fontsize=8, color="#555555",
                va="center", ha="right")

    # Right column: emitted mode strings (cross-product).
    ax.text(7.7, 6.4, "mode column tag",
            fontsize=11, fontweight="bold", ha="left")
    examples = [
        "loco", "wd0_loco", "warmup0_loco",
        "pairwise", "wd0_pairwise", "warmup0_pairwise",
        "triplet", "wd0_triplet", "warmup0_triplet",
    ]
    for i, tag in enumerate(examples):
        y = 5.7 - i * 0.4
        ax.text(7.7, y, tag, fontsize=8.5, family="monospace",
                color="#222222", va="center")

    # Center: combiner box.
    rect = mpatches.FancyBboxPatch(
        (3.0, 1.6), 4.5, 0.9,
        boxstyle="round,pad=0.08",
        facecolor="#f6e6df", edgecolor="#aa4422", linewidth=1.6,
    )
    ax.add_patch(rect)
    ax.text(5.25, 2.25, "mode_string(kind, stratum)",
            fontsize=11, fontweight="bold", ha="center", color="#aa4422")
    ax.text(5.25, 1.85,
            r"$\rightarrow$ format!(\"{}{}\", stratum.prefix(), kind.base())",
            fontsize=9, ha="center", family="monospace", color="#444444")

    # Arrows from Stratum + ModeKind into combiner.
    ax.annotate("", xy=(4.0, 2.4), xytext=(2.9, 4.0),
                arrowprops=dict(arrowstyle="->", color="#2c5d8f", lw=1.4))
    ax.annotate("", xy=(5.7, 2.4), xytext=(5.2, 4.0),
                arrowprops=dict(arrowstyle="->", color="#5a7d2c", lw=1.4))
    # Arrow from combiner to output column.
    ax.annotate("", xy=(7.5, 4.0), xytext=(6.5, 2.4),
                arrowprops=dict(arrowstyle="->", color="#aa4422", lw=1.4))

    # Caption at bottom.
    caption = (
        "Adding a new Stratum variant (e.g. Smooth0) auto-extends every\n"
        "downstream mode-string consumer (f2_dual_mediation lookups,\n"
        "f2_provenance_check stratum banner, f2_stratum_compare joins)\n"
        "without code edits beyond Stratum::ALL + prefix() arm."
    )
    ax.text(5.0, 0.65, caption, fontsize=9, ha="center", va="center",
            color="#333333", style="italic")

    ax.set_title("F2 stratum registry (Loop 39 fix 2 → Loop 41 Warmup0)",
                 fontsize=12, pad=10)

    save_figure(fig, args.out)


if __name__ == "__main__":
    main()
