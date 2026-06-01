#!/usr/bin/env python3
"""Figure 3 — Canonical 5x4 PSE table heatmap.

Visualizes the canonical dual_mediation output (5 non-mediator fixes ×
4 PSEs) from Loop 36. Each cell is the estimate in BPB; color encodes
sign + magnitude. CI-excludes-zero cells get a bold border.

Use: data/loop36_dual.csv → f2_to_jsonl → this script → PNG.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from fig_template import (
    PSE_NAMES,
    is_real,
    load_jsonl,
    parse_args_with_io,
    safe_float,
    save_figure,
)


def main():
    args = parse_args_with_io(
        default_input="/tmp/loop53_canonical_dual.jsonl",
        default_out="papers/figures/fig3_canonical_pse_heatmap.png",
    )

    records = list(load_jsonl(args.input))
    if not records:
        raise SystemExit(f"# ERROR: no records in {args.input}")

    # Column groups (estimate, ci_lo, ci_hi) for each PSE in f2_dual_mediation schema.
    PSE_COLS = {
        "NDE": ("nde", "ci95_nde_lo", "ci95_nde_hi"),
        "NIE_M1": ("nie_m1", "ci95_nie_m1_lo", "ci95_nie_m1_hi"),
        "NIE_M2": ("nie_m2", "ci95_nie_m2_lo", "ci95_nie_m2_hi"),
        "NIE_chain": ("nie_chain", "ci95_nie_chain_lo", "ci95_nie_chain_hi"),
    }

    # Row order = order of appearance in CSV (already ranked by |Δ_X| desc).
    fix_xs = [r["fix_x"] for r in records]
    n_rows = len(fix_xs)
    n_cols = len(PSE_NAMES)
    estimates = np.full((n_rows, n_cols), np.nan)
    significant = np.zeros((n_rows, n_cols), dtype=bool)

    for i, rec in enumerate(records):
        for j, pse in enumerate(PSE_NAMES):
            est_col, lo_col, hi_col = PSE_COLS[pse]
            est = safe_float(rec.get(est_col))
            lo = safe_float(rec.get(lo_col))
            hi = safe_float(rec.get(hi_col))
            estimates[i, j] = est
            if is_real(est) and is_real(lo) and is_real(hi):
                significant[i, j] = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)

    # Symmetric color scale around zero.
    vmax = float(np.nanmax(np.abs(estimates)))
    if not math.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(estimates, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, label="estimate (BPB)")
    cbar.ax.tick_params(labelsize=9)

    # Tick labels.
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(PSE_NAMES, rotation=20, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(fix_xs)
    ax.set_title("Canonical PSE decomposition\n(F2 dual_mediation, 200-step sandbox, N=5)")

    # Numeric annotations + significance border.
    for i in range(n_rows):
        for j in range(n_cols):
            est = estimates[i, j]
            if not is_real(est):
                continue
            txt = f"{est:+.2f}"
            color = "white" if abs(est) > 0.6 * vmax else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
            if significant[i, j]:
                # Thick rectangle around the cell.
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="black", linewidth=1.6,
                    )
                )

    ax.set_xlabel("Path-specific effect")
    ax.set_ylabel("fix_x")
    save_figure(fig, args.out)


if __name__ == "__main__":
    main()
