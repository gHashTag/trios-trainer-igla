#!/usr/bin/env python3
"""Loop+2 analysis: bootstrap CIs for Option A (power) and Option C (frontier).

Negative-results-first, claim-status discipline. CPU-only experiment.
"""
import csv
import statistics as st

import numpy as np

RNG = np.random.default_rng(0xF2A)
BOOT = 20000


def mean_ci(xs):
    xs = np.asarray(xs, float)
    bs = RNG.choice(xs, size=(BOOT, len(xs)), replace=True).mean(axis=1)
    return float(xs.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_delta_ci(a, b):
    """CI on mean(a) - mean(b) via independent bootstrap of each arm."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    da = RNG.choice(a, size=(BOOT, len(a)), replace=True).mean(axis=1)
    db = RNG.choice(b, size=(BOOT, len(b)), replace=True).mean(axis=1)
    d = da - db
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------- Option A ----------
A = {}
with open("loop2_optionA.csv") as f:
    for r in csv.DictReader(f):
        A.setdefault(r["arm"], {"final": [], "best": []})
        A[r["arm"]]["final"].append(float(r["final_val_bpb"]))
        A[r["arm"]]["best"].append(float(r["best_val_bpb"]))

print("=== OPTION A (power: hidden=128, 1000 steps, lr=0.01, wd=0.02, 6 seeds) ===")
for arm in ("standard", "phi_b1"):
    for k in ("final", "best"):
        m, lo, hi = mean_ci(A[arm][k])
        sd = st.pstdev(A[arm][k])
        print(f"  {arm:8s} {k:5s}: mean={m:.4f} [{lo:.4f}, {hi:.4f}]  sd={sd:.4f}")

for k in ("final", "best"):
    d, lo, hi = paired_delta_ci(A["phi_b1"][k], A["standard"][k])
    verdict = "CI includes 0 -> TIE" if lo <= 0 <= hi else ("phi_b1 WORSE" if d > 0 else "phi_b1 BETTER")
    print(f"  DELTA {k:5s} (phi_b1 - standard) = {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")

# instability: per-seed gap final - best
print("  Instability (final - best per seed):")
for arm in ("standard", "phi_b1"):
    gaps = [fb - bb for fb, bb in zip(A[arm]["final"], A[arm]["best"])]
    print(f"    {arm:8s} max_gap={max(gaps):.4f} mean_gap={st.mean(gaps):.4f}")

# ---------- Option C ----------
C = {}
with open("loop2_optionC.csv") as f:
    for r in csv.DictReader(f):
        C.setdefault(r["config"], []).append(float(r["code_val_bpb"]))

print("\n=== OPTION C (frontier: hidden=64, 300 steps, 3 seeds) ===")
frontier = None
rows = []
for cfg, xs in C.items():
    m, lo, hi = mean_ci(xs)
    rows.append((cfg, m, lo, hi))
    if cfg == "standard_lr03_wd02":
        frontier = m
for cfg, m, lo, hi in rows:
    tag = "  <- STANDARD FRONTIER" if cfg == "standard_lr03_wd02" else ""
    gap = f"  (+{m - frontier:.3f} vs frontier)" if frontier is not None and cfg != "standard_lr03_wd02" else ""
    print(f"  {cfg:28s}: mean={m:.4f} [{lo:.4f}, {hi:.4f}]{gap}{tag}")

# does any phi-derived config reach the frontier?
print("\n  Frontier reach test (vs standard_lr03_wd02):")
for cfg in ("standard_philr_wd02", "allphi", "phi_tempered_philr_wd02"):
    d, lo, hi = paired_delta_ci(C[cfg], C["standard_lr03_wd02"])
    reaches = "REACHES (CI incl 0)" if lo <= 0 <= hi else "DOES NOT reach (worse)"
    print(f"    {cfg:28s} delta={d:+.4f} [{lo:+.4f}, {hi:+.4f}]  -> {reaches}")
