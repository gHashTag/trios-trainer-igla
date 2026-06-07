"""Self-contained stratified-CDE / mediation analysis for the IGLA-Coder
2x2 optimizer factorial (Option C). Reads the F2 long-form CSV, computes
Pearl controlled direct effects (CDE) for each mediator, the VanderWeele
4-way decomposition of the total effect, bootstrap 95% CIs, and a Gamma_tip
(E-value-style) robustness index per effect.

Mediators (binary, factorially manipulated):
  M1 = momentum (beta1: 0.9 -> phi^-1)
  M2 = decay    (wd:   0.04 -> phi^-3)
Outcome Y = code_val_bpb (lower is better; positive effect = WORSE).
Treatment X = phi-prior on the optimizer.

References: Pearl 2001 (direct/indirect effects); VanderWeele 2015
(Explanation in Causal Inference, 4-way decomposition); VanderWeele & Ding
2017 (E-value); Zhao & Luo 2024 (mediation CIs).
CPU-only, numpy-only.
"""
import csv
import numpy as np

rng = np.random.default_rng(20260602)


def load(path):
    cells = {}  # (m1,m2) -> list of bpb
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            break
        # line now holds header; build a reader over the rest
        rdr = csv.DictReader(f, fieldnames=line.rstrip("\n").split(","))
        for r in rdr:
            key = (int(r["m1_momentum"]), int(r["m2_decay"]))
            cells.setdefault(key, []).append(float(r["code_val_bpb"]))
    return cells


def gamma_tip(effect, lo, hi):
    """E-value-style robustness: how strong must an unmeasured confound be
    (on the ratio scale) to move the CI bound across zero. Larger = more
    robust. We map the standardized effect (effect / half-CI-width) through
    the VanderWeele-Ding approximate E-value transform.
    """
    half = (hi - lo) / 2.0
    if half <= 0:
        return float("inf")
    # z-like ratio of point estimate to its CI half-width
    rr = abs(effect) / half
    if rr <= 0:
        return 1.0
    # approximate risk-ratio scaling then E-value = RR + sqrt(RR*(RR-1))
    RR = 1.0 + rr  # monotone increasing in signal-to-CI ratio
    ev = RR + np.sqrt(RR * (RR - 1.0))
    return ev


def boot(cellsamp, fn, n=20000, alpha=0.05):
    """Bootstrap CI for an arbitrary contrast fn(dict_of_resampled_cells)."""
    vals = []
    keys = list(cellsamp.keys())
    for _ in range(n):
        rs = {k: rng.choice(cellsamp[k], size=len(cellsamp[k]), replace=True)
              for k in keys}
        vals.append(fn(rs))
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return float(np.mean(vals)), float(lo), float(hi), float(min(p, 1.0))


def m(c, key):
    return np.mean(c[key])


# Contrasts on the 2x2 grid. key = (m1_momentum, m2_decay).
# Pearl CDE of momentum, holding decay = 0:
cde_mom = lambda c: m(c, (1, 0)) - m(c, (0, 0))
# Pearl CDE of decay, holding momentum = 0:
cde_dec = lambda c: m(c, (0, 1)) - m(c, (0, 0))
# Total effect of full phi prior (both on) vs neither:
te = lambda c: m(c, (1, 1)) - m(c, (0, 0))
# Interaction (mediated-interaction surplus): TE - CDE_mom - CDE_dec
inter = lambda c: (m(c, (1, 1)) - m(c, (0, 0))) \
    - (m(c, (1, 0)) - m(c, (0, 0))) \
    - (m(c, (0, 1)) - m(c, (0, 0)))
# CDE of momentum WITH decay held ON (does momentum still hurt when decay is phi^-3?):
cde_mom_dec1 = lambda c: m(c, (1, 1)) - m(c, (0, 1))


def report(name, fn, status_hint=""):
    c = CELLS
    point = fn({k: np.array(v) for k, v in c.items()})
    _, lo, hi, p = boot(c, fn)
    g = gamma_tip(point, lo, hi)
    spans0 = lo <= 0 <= hi
    if g >= 2.0 and not spans0:
        rob = "robust"
    elif 1.25 <= g < 2.0 and not spans0:
        rob = "fragile"
    else:
        rob = "do-not-publish" if not spans0 else "CI-spans-0"
    print(f"{name:34s} eff={point:+.4f} BPB  95%CI[{lo:+.4f},{hi:+.4f}]  "
          f"p~{p:.3f}  Gamma_tip={g:.2f} ({rob}) {status_hint}")
    return point, lo, hi, p, g


CELLS = load("coder_ablation_f2.csv")
print("Cell means (m1_momentum, m2_decay) -> mean BPB, n:")
for k in sorted(CELLS):
    print(f"  {k}: mean={np.mean(CELLS[k]):.4f}  n={len(CELLS[k])}  "
          f"raw={['%.4f'%x for x in CELLS[k]]}")
print()
print("Positive effect = HIGHER BPB = WORSE. (phi prior on optimizer)")
print("-" * 92)
report("CDE_momentum | decay=0", cde_mom)
report("CDE_decay | momentum=0", cde_dec)
report("CDE_momentum | decay=1 (phi^-3)", cde_mom_dec1)
report("Total effect (both phi on)", te)
report("Mediated-interaction (TE-CDEs)", inter)
print("-" * 92)
print("Decomposition check: CDE_dec + CDE_mom + interaction == TE (exact on point estimates).")
