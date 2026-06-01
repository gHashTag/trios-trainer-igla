#!/usr/bin/env python3
"""Figure 4 — Tipping-point curves Γ_tip(Λ) for rms PSEs.

For each of rms's 4 path-specific effects (NDE, NIE_M1, NIE_M2, NIE_chain),
plot the hyperbola Γ_tip = 1 + |closer_endpoint| / Λ over a grid of Λ.

The shaded region under each curve is the "tipping region": unmeasured
confounding strength below the curve preserves the verdict (CI excludes 0).
Above the curve, the verdict flips.

Input: JSONL emitted by `f2_to_jsonl` on a `f2_mediation_sensitivity
--lambda-sweep` CSV. Filters to the `fix_x = rms` rows by default.

VanderWeele-Ding thresholds (horizontal reference lines):
  - Γ = 1.25 → fragile / moderate boundary
  - Γ = 2.0  → moderate / robust boundary
"""

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


# Visually distinct colors for the 4 PSEs.
PSE_COLORS = {
    "NDE":       "#1f77b4",  # blue
    "NIE_M1":    "#2ca02c",  # green
    "NIE_M2":    "#ff7f0e",  # orange
    "NIE_chain": "#9467bd",  # purple
}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/tmp/loop53_lambda_sweep.jsonl")
    p.add_argument("--out", default="papers/figures/fig4_tipping_curves.png")
    p.add_argument("--fix-x", default="rms",
                   help="filter to this fix_x value (default: rms)")
    args = p.parse_args()

    # Group: pse_name -> [(lambda, gamma_tip), ...] sorted by lambda.
    by_pse = {}
    for rec in load_jsonl(args.input):
        if rec.get("fix_x") != args.fix_x:
            continue
        pse = rec.get("pse_name")
        lam = safe_float(rec.get("lambda"))
        g = safe_float(rec.get("gamma_tip"))
        if pse and is_real(lam) and is_real(g):
            by_pse.setdefault(pse, []).append((lam, g))

    if not by_pse:
        raise SystemExit(f"# ERROR: no rows matched fix_x={args.fix_x!r}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for pse in PSE_NAMES:
        pts = sorted(by_pse.get(pse, []))
        if not pts:
            continue
        lams = np.array([l for l, _ in pts])
        gammas = np.array([g for _, g in pts])
        color = PSE_COLORS[pse]
        ax.plot(lams, gammas, marker="o", color=color, label=pse, linewidth=1.8)
        ax.fill_between(lams, 1.0, gammas, color=color, alpha=0.06)

    # VanderWeele-Ding reference lines.
    ax.axhline(1.25, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
    ax.axhline(2.0, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
    ax.text(
        0.105, 1.25, " fragile/moderate", fontsize=8, va="center", alpha=0.65,
    )
    ax.text(
        0.105, 2.0, " moderate/robust", fontsize=8, va="center", alpha=0.65,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Λ (outcome residual, BPB)")
    ax.set_ylabel("Γ_tip (tipping point)")
    ax.set_title(
        f"Tipping-point curves for {args.fix_x} PSEs\n"
        f"(Ohnishi & Li 2026 Thm 2 + VanderWeele-Ding thresholds)"
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    save_figure(fig, args.out)


if __name__ == "__main__":
    main()
