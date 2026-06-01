#!/usr/bin/env python3
"""Figure 1 — RmsNorm NDE sign flip across strata.

Headline finding from Loops 49-50: under canonical mediation the NDE for rms
is -4.12 BPB (apparently harmful), but under the wd0 Pearl CDE stratum the
same estimand is +0.43 BPB (intrinsically helpful, CI excludes zero).

Input: JSONL emitted by `f2_to_jsonl` from a 3-stratum `f2_stratum_compare`
CSV. Output: PNG bar chart with 95% CI error bars.

Usage:
    python3 fig1_rms_nde_signflip.py \\
        --input /tmp/loop52_3stratum.jsonl \\
        --out papers/figures/fig1_rms_nde_signflip.png

The script reads exactly one row matched by `--fix-x rms --pse-name NDE`
and plots three bars (canonical, wd0, warmup0) with CI95 error bars.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; we only save the PNG.
import matplotlib.pyplot as plt


def load_row(jsonl_path, fix_x, pse_name):
    """Find the (fix_x, pse_name) row in a `f2_stratum_compare` JSONL."""
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("fix_x") == fix_x and rec.get("pse_name") == pse_name:
                return rec
    raise SystemExit(f"# ERROR: no row matched fix_x={fix_x!r}, pse_name={pse_name!r}")


def is_real(x):
    return x is not None and isinstance(x, (int, float)) and not math.isnan(float(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="/tmp/loop52_3stratum.jsonl",
        help="JSONL from f2_to_jsonl on a 3-stratum compare CSV",
    )
    p.add_argument("--out", default="fig1_rms_nde_signflip.png")
    p.add_argument("--fix-x", default="rms")
    p.add_argument("--pse-name", default="NDE")
    args = p.parse_args()

    row = load_row(args.input, args.fix_x, args.pse_name)

    # Per-stratum estimates and CI half-widths.
    strata_labels = ["canonical", "wd0", "warmup0"]
    estimates = []
    err_lo = []
    err_hi = []
    survives = []  # CI excludes zero?
    for s in strata_labels:
        est = row.get(f"estimate_{s}")
        lo = row.get(f"ci95_lo_{s}")
        hi = row.get(f"ci95_hi_{s}")
        if is_real(est) and is_real(lo) and is_real(hi):
            est = float(est)
            lo = float(lo)
            hi = float(hi)
            estimates.append(est)
            err_lo.append(est - lo)
            err_hi.append(hi - est)
            survives.append((lo > 0 and hi > 0) or (lo < 0 and hi < 0))
        else:
            # Missing stratum → placeholder NaN, no bar drawn.
            estimates.append(float("nan"))
            err_lo.append(0.0)
            err_hi.append(0.0)
            survives.append(False)

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    xs = list(range(len(strata_labels)))
    # Color: red for negative (harmful direction), green for positive
    # (helpful direction). Per F2 BPB convention: lower BPB is better, so
    # NDE > 0 means "removing helps → fix is helpful intrinsically".
    colors = []
    for est in estimates:
        if math.isnan(est):
            colors.append("#cccccc")
        elif est < 0:
            colors.append("#c0392b")  # red
        else:
            colors.append("#27ae60")  # green
    bars = ax.bar(
        xs,
        estimates,
        yerr=[err_lo, err_hi],
        color=colors,
        edgecolor="black",
        capsize=6,
        alpha=0.85,
    )
    # Mark CI-excludes-zero (robust) bars with a '*' annotation above.
    for i, (est, ok) in enumerate(zip(estimates, survives)):
        if not math.isnan(est) and ok:
            ax.text(
                i,
                est + (err_hi[i] if est >= 0 else -err_lo[i]) + 0.2,
                "*",
                ha="center",
                va="bottom" if est >= 0 else "top",
                fontsize=14,
                fontweight="bold",
            )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["canonical\n(no strat)", "wd0\n(Pearl CDE)", "warmup0\n(Pearl CDE)"])
    ax.set_ylabel(f"{args.pse_name} for {args.fix_x} (BPB)")
    ax.set_title(
        f"RmsNorm NDE sign flip across strata\n(F2 framework, 200-step sandbox, N=5 seeds)"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    # Add the headline numbers as inline annotations.
    for i, est in enumerate(estimates):
        if not math.isnan(est):
            ax.annotate(
                f"{est:+.2f}",
                xy=(i, est),
                xytext=(0, -14 if est >= 0 else 14),
                textcoords="offset points",
                ha="center",
                va="top" if est >= 0 else "bottom",
                fontsize=10,
                color="black",
            )
    # Loop 104 fix: give the wd0 bar's "*" + error-bar cap clear sky
    # below the title (26th pass caught the collision).
    y_data = [est + err_hi[i] for i, est in enumerate(estimates)
              if not math.isnan(est)] + [0.0]
    y_top = max(y_data) + 0.7   # +0.5 for asterisk, +0.2 padding
    ax.set_ylim(top=y_top)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"# Wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
