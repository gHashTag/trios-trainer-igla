#!/usr/bin/env python3
"""Figure 5 — Swap-parameterization NIE_M1-via-rms across three strata.

Visualizes the Loop 64 Phase 0 cross-stratum picture: for each
non-mediator fix X, the NIE_M1 estimate (i.e., the part of X's effect
mediated through rms) under the swap parameterization
(M_1 = rms, M_2 = warmup), computed at three strata
(canonical, wd0, warmup0).

The headline secondary finding is the wd row: byte-identical −0.751
[−1.325, −0.177] across all three strata. This figure makes that
invariance pattern visually obvious next to the variable estimates
for the other fixes.

Input: data/loop49_swap/3stratum_swap.csv → JSONL (one record per
(fix_x, pse_name) pair). Filters to pse_name = NIE_M1.

Use:
  target/release/f2_to_jsonl data/loop49_swap/3stratum_swap.csv \\
      --out /tmp/3strat_swap.jsonl
  python3 papers/figures/fig5_swap_nie_m1_heatmap.py \\
      --input /tmp/3strat_swap.jsonl
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from fig_template import (
    STRATA_LABELS,
    is_real,
    load_jsonl,
    parse_args_with_io,
    safe_float,
    save_figure,
)


def main():
    args = parse_args_with_io(
        default_input="/tmp/3strat_swap.jsonl",
        default_out="papers/figures/fig5_swap_nie_m1_heatmap.png",
    )

    records = list(load_jsonl(args.input))
    if not records:
        raise SystemExit(f"# ERROR: no records in {args.input}")

    # Filter to NIE_M1 rows.
    nie_m1_records = [r for r in records if r.get("pse_name") == "NIE_M1"]
    if not nie_m1_records:
        raise SystemExit(f"# ERROR: no NIE_M1 records in {args.input}")

    # Fix-x order = the order in which they appear in the CSV.
    fix_xs = []
    for r in nie_m1_records:
        fx = r.get("fix_x")
        if fx not in fix_xs:
            fix_xs.append(fx)

    n_rows = len(fix_xs)
    n_cols = len(STRATA_LABELS)
    estimates = np.full((n_rows, n_cols), np.nan)
    ci_lo = np.full((n_rows, n_cols), np.nan)
    ci_hi = np.full((n_rows, n_cols), np.nan)

    fix_x_idx = {fx: i for i, fx in enumerate(fix_xs)}
    for rec in nie_m1_records:
        i = fix_x_idx[rec["fix_x"]]
        for j, stratum in enumerate(STRATA_LABELS):
            est = safe_float(rec.get(f"estimate_{stratum}"))
            lo = safe_float(rec.get(f"ci95_lo_{stratum}"))
            hi = safe_float(rec.get(f"ci95_hi_{stratum}"))
            estimates[i, j] = est
            ci_lo[i, j] = lo
            ci_hi[i, j] = hi

    # Symmetric color scale around zero. Filter to nonzero estimates for vmax
    # so the lots of structural zeros (mediator-pinned rows) don't compress
    # the scale.
    nonzero = estimates[~np.isnan(estimates) & (np.abs(estimates) > 1e-9)]
    vmax = float(np.max(np.abs(nonzero))) if len(nonzero) else 1.0
    if not math.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    im = ax.imshow(estimates, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, label="NIE_M1 via rms (BPB)")
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(STRATA_LABELS)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(fix_xs)
    ax.set_title(
        "Swap-parameterization NIE_M1 (via rms) across strata\n"
        "(F2 Phase 0, M_1=rms, M_2=warmup, sandbox N=5)"
    )

    # Detect byte-identical-across-all-strata rows.
    for i in range(n_rows):
        row = estimates[i, :]
        if not np.all(np.isfinite(row)):
            continue
        # All three values byte-identical (within float epsilon).
        if np.max(row) - np.min(row) < 1e-6 and np.max(np.abs(row)) > 1e-6:
            # Highlight the row with a left-side bracket.
            ax.add_patch(
                plt.Rectangle(
                    (-0.5, i - 0.5), n_cols, 1,
                    fill=False, edgecolor="black", linewidth=2.4,
                )
            )

    for i in range(n_rows):
        for j in range(n_cols):
            est = estimates[i, j]
            if not is_real(est):
                continue
            # Bold border if CI excludes zero.
            lo, hi = ci_lo[i, j], ci_hi[i, j]
            sig = is_real(lo) and is_real(hi) and (
                (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
            )
            if sig:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="black", linewidth=1.4,
                    )
                )
            # Numeric annotation.
            txt = f"{est:+.2f}"
            color = "white" if abs(est) > 0.6 * vmax else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)

    ax.set_xlabel("Stratum")
    ax.set_ylabel("fix_x")
    # Loop 105 fix: caption was overlapping the x-axis label.
    # Move caption further below and reserve bottom margin in layout.
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    plt.figtext(
        0.5, 0.015,
        "Thick row outline = byte-identical estimate across all 3 strata",
        ha="center", fontsize=8, style="italic",
    )

    save_figure(fig, args.out)


if __name__ == "__main__":
    main()
