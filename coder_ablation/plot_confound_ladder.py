#!/usr/bin/env python3
# IGLA-Coder confound ladder figure (Coder-Loop+10, A).
#
# THE STORY (negative-first)
#   The phi generator was NOT special. It only LOOKED special at Loop+7 because
#   the comparison silently confounded the phi prior with a low learning rate.
#   As each confound was controlled, the phi-vs-standard gap collapsed toward
#   zero. The figure shows the phi-standard delta (BPB) shrinking across three
#   successive controls -- a ladder DOWN to neutrality, not up to a claim.
#
#   x = control stage (increasingly strict)
#   y = phi-standard delta in code_val_bpb (positive = phi WORSE)
#
# Numbers (frozen from the prior loop reports + CSVs):
#   Loop+7 (raw, lr-confounded):  delta = +0.566 BPB   (phi much worse;
#                                 but it was a low-lr axis, not a bad prior)
#   Loop+8 (iso-lr, decay free):  delta = +0.092 BPB   (gap mostly gone; the
#                                 residual was the phi^-3 decay, ~6x standard)
#   Loop+9 (iso-everything):      delta = -0.027 BPB   (CIs overlap; phi
#                                 momentum NEUTRAL once lr AND decay pinned)
#
# HONESTY: this is a falsification ladder. The shrinking gap is the result;
#   phi is not supported as a special optimizer prior. Only phi^2+phi^-2=3 is
#   [Verified]. The method (per-knob isolation) survives; phi does not (yet).

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# stage label, phi-standard delta BPB, half-CI of the delta (approx, for the
# error bar -- pooled from the per-arm CIs in each loop), one-line cause
STAGES = [
    ("Loop+7\nraw\n(lr confounded)", 0.566, 0.108,
     "phi = a low-lr axis,\nnot a bad prior"),
    ("Loop+8\niso-lr\n(decay free)", 0.092, 0.040,
     "residual = phi^-3 decay\n(~6x standard wd)"),
    ("Loop+9\niso-everything\n(lr + wd pinned)", -0.027, 0.041,
     "momentum NEUTRAL;\nCIs overlap"),
]

xs = list(range(len(STAGES)))
labels = [s[0] for s in STAGES]
deltas = [s[1] for s in STAGES]
errs = [s[2] for s in STAGES]
causes = [s[3] for s in STAGES]

fig, ax = plt.subplots(figsize=(9.0, 6.0))

# zero line = neutrality (phi == standard)
ax.axhline(0.0, color="#888888", linewidth=1.2, linestyle="--", zorder=1)
ax.text(0.05, 0.012, "neutral (phi = standard)", color="#666666",
        fontsize=9, ha="left", va="bottom")

# the ladder line + points
ax.plot(xs, deltas, color="#1f3b73", linewidth=2.2, marker="o",
        markersize=11, markerfacecolor="#2e6fdb", markeredgecolor="#13294b",
        markeredgewidth=1.4, zorder=4)
ax.errorbar(xs, deltas, yerr=errs, fmt="none", ecolor="#13294b",
            elinewidth=1.6, capsize=6, capthick=1.6, zorder=3)

# annotate each delta value + cause. For positive deltas, value sits below the
# point and cause above it; for the negative delta both go ABOVE the point so
# nothing collides with the x-axis tick labels at the bottom.
for x, d, e, c in zip(xs, deltas, errs, causes):
    if d >= 0:
        ax.annotate("%+.3f BPB" % d, (x, d), xytext=(x, d + e + 0.045),
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color="#13294b")
        # cause text offset to the RIGHT of the point so it never collides with
        # the title (top) or the descending ladder line.
        ax.annotate(c, (x, d), xytext=(x + 0.18, d),
                    ha="left", va="center", fontsize=8.5, color="#555555")
    else:
        # negative delta: stack both annotations ABOVE the point/zero line
        ax.annotate("%+.3f BPB" % d, (x, d), xytext=(x, 0.035),
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color="#13294b")
        ax.annotate(c, (x, d), xytext=(x, 0.115),
                    ha="center", va="bottom", fontsize=8.5, color="#555555")

ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("phi - standard delta  (code_val_bpb)", fontsize=11)
ax.set_title("IGLA-Coder confound ladder: the phi advantage was a confound\n"
             "as controls tighten, the phi-standard gap collapses to neutral",
             fontsize=12.5, fontweight="bold", pad=14)
ax.set_ylim(-0.18, 0.74)
ax.set_xlim(-0.4, 2.4)
ax.grid(axis="y", color="#e6e6e6", linewidth=0.8, zorder=0)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

# footer anchor
fig.text(0.5, 0.015,
         "negative-first: this is a falsification ladder; phi NOT supported as "
         "an optimizer prior.   Only phi^2 + phi^-2 = 3 is [Verified].",
         ha="center", fontsize=8.5, color="#777777", style="italic")

fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig("confound_ladder.png", dpi=160)
print("wrote confound_ladder.png")
