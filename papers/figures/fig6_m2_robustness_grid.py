#!/usr/bin/env python3
"""Figure 6 — Full M_2 robustness landscape: 5 M_2 candidates × 3 strata.

Visualizes the Loop 68 swap-parameterization robustness sweep. For the
specific (X=wd, NIE_M1 via rms) row of the four-PSE table, we vary
M_2 across all five candidate mediators (warmup, gradclip, clamp,
smooth, dropout) and re-compute under all three strata (canonical,
wd0, warmup0).

The visual story:
- The wd0 column matches the canonical column in EVERY M_2 row
  (because wd is target X, not a mediator on the rms-M_2 path).
- The warmup0 column matches the others ONLY when M_2 = warmup
  (the stratum's pinned variable aligns with the parameterization).
- For all other M_2 choices, warmup0 diverges, illustrating the
  no-XM-interaction prediction.

Input: a flat CSV with columns `m2, stratum, estimate, ci_lo, ci_hi`
constructed inline from the data/loop49_swap/ files at runtime — the
script directly reads the swap CSVs rather than going through JSONL.

Use:
  python3 papers/figures/fig6_m2_robustness_grid.py \\
      --out papers/figures/fig6_m2_robustness_grid.png
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from fig_template import is_real, safe_float, save_figure


CRATE_ROOT = Path(__file__).resolve().parents[2]
SWAP_DIR = CRATE_ROOT / "data" / "loop49_swap"
STRATA = ["canonical", "wd0", "warmup0"]
M2_CHOICES = ["warmup", "gradclip", "clamp", "smooth", "dropout"]


def read_wd_nie_m1(csv_path: Path) -> tuple[float, float, float] | None:
    """Pull (estimate, ci_lo, ci_hi) for fix_x=wd, NIE_M1 from the file."""
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if row[0] == "rank":  # header
                continue
            if len(row) < 12:
                continue
            if row[1] == "wd":
                # Columns per f2_dual_mediation schema (Loop 68 verified):
                # rank,fix_x,n,delta_x,nde,se_nde,ci95_nde_lo,ci95_nde_hi,
                # nie_m1,se_nie_m1,ci95_nie_m1_lo,ci95_nie_m1_hi, ...
                est = safe_float(row[8])
                lo = safe_float(row[10])
                hi = safe_float(row[11])
                return (est, lo, hi)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out", default="papers/figures/fig6_m2_robustness_grid.png"
    )
    args = p.parse_args()

    estimates = np.full((len(M2_CHOICES), len(STRATA)), np.nan)
    ci_lo = np.full_like(estimates, np.nan)
    ci_hi = np.full_like(estimates, np.nan)

    for i, m2 in enumerate(M2_CHOICES):
        for j, stratum in enumerate(STRATA):
            if m2 == "warmup":
                fname = f"{stratum}_swap_dual.csv"
            else:
                fname = f"{stratum}_swap_m2{m2}.csv"
            row = read_wd_nie_m1(SWAP_DIR / fname)
            if row is None:
                continue
            est, lo, hi = row
            estimates[i, j] = est
            ci_lo[i, j] = lo
            ci_hi[i, j] = hi

    # Symmetric color scale around zero.
    finite = estimates[np.isfinite(estimates)]
    vmax = float(np.max(np.abs(finite))) if len(finite) else 1.0
    if vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(
        estimates, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto"
    )
    cbar = fig.colorbar(im, ax=ax, label="wd × NIE_M1 via rms (BPB)")
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(np.arange(len(STRATA)))
    ax.set_xticklabels(STRATA)
    ax.set_yticks(np.arange(len(M2_CHOICES)))
    ax.set_yticklabels([f"M_2 = {m}" for m in M2_CHOICES])
    ax.set_title(
        "Full M_2 robustness landscape — wd × NIE_M1 via rms\n"
        "(Loop 68 swap-parameterization sweep, sandbox N=5)"
    )

    # Mark fully-invariant rows (warmup) with a thick outline.
    for i in range(len(M2_CHOICES)):
        row = estimates[i, :]
        if np.all(np.isfinite(row)) and np.max(row) - np.min(row) < 1e-6:
            ax.add_patch(
                plt.Rectangle(
                    (-0.5, i - 0.5), len(STRATA), 1,
                    fill=False, edgecolor="black", linewidth=2.6,
                )
            )

    # Mark cells where (canonical, wd0) pair agrees but differs from
    # warmup0 with a side bracket. Annotate each cell with the estimate.
    for i in range(len(M2_CHOICES)):
        for j in range(len(STRATA)):
            est = estimates[i, j]
            if not is_real(est):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="black")
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
                        fill=False, edgecolor="black", linewidth=1.2,
                    )
                )
            txt = f"{est:+.2f}"
            color = "white" if abs(est) > 0.6 * vmax else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)

    ax.set_xlabel("Stratum")
    ax.set_ylabel("Swap-parameterization M_2 choice")
    plt.figtext(
        0.5, 0.005,
        "Thick row outline = byte-identical across all 3 strata. "
        "Pattern: warmup0 only matches when M_2 = warmup.",
        ha="center", fontsize=8, style="italic",
    )

    save_figure(fig, args.out)


if __name__ == "__main__":
    main()
