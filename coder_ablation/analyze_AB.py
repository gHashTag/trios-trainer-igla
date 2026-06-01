"""Bootstrap-CI analysis for IGLA-Coder Loop+1 Options A and B.
CPU-only, numpy-only. Honest power-aware reporting.
"""
import csv
import numpy as np

rng = np.random.default_rng(20260601)


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["bpb"] = float(r["bpb"])
            r["wd"] = float(r["wd"])
            rows.append(r)
    return rows


def boot_mean_ci(x, n=20000, alpha=0.05):
    x = np.asarray(x, float)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return x.mean(), lo, hi


def boot_diff_ci(a, b, n=20000, alpha=0.05):
    """CI for mean(a) - mean(b) via independent bootstrap."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    diffs = np.array([
        rng.choice(a, size=len(a), replace=True).mean()
        - rng.choice(b, size=len(b), replace=True).mean()
        for _ in range(n)
    ])
    d = a.mean() - b.mean()
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided bootstrap p for diff != 0
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p = min(p, 1.0)
    return d, lo, hi, p


print("=" * 70)
print("OPTION A: weight-decay isolation, hidden=64, steps=300, lr=0.03")
print("=" * 70)
A = load("optionA_grid.csv")
for wd in [0.04, 0.02, 0.01]:
    print(f"\n-- wd = {wd} --")
    for arm in ["standard", "phi_b1"]:
        x = [r["bpb"] for r in A if r["arm"] == arm and r["wd"] == wd]
        m, lo, hi = boot_mean_ci(x)
        print(f"  {arm:9s} n={len(x)} mean={m:.4f}  95%CI[{lo:.4f}, {hi:.4f}]  raw={['%.4f'%v for v in x]}")
    s = [r["bpb"] for r in A if r["arm"] == "standard" and r["wd"] == wd]
    p = [r["bpb"] for r in A if r["arm"] == "phi_b1" and r["wd"] == wd]
    d, lo, hi, pv = boot_diff_ci(p, s)
    verdict = "DIFFERENT" if (lo > 0 or hi < 0) else "TIE (CI spans 0)"
    print(f"  diff(phi_b1 - standard) = {d:+.4f}  95%CI[{lo:+.4f}, {hi:+.4f}]  p~{pv:.3f}  -> {verdict}")

print()
print("=" * 70)
print("OPTION B: power run, hidden=128, steps=800, lr=0.01, wd=0.02, 5 seeds")
print("=" * 70)
B = load("optionB_grid.csv")
for arm in ["standard", "phi_b1"]:
    x = [r["bpb"] for r in B if r["arm"] == arm]
    m, lo, hi = boot_mean_ci(x)
    sd = np.std(x, ddof=1)
    print(f"  {arm:9s} n={len(x)} mean={m:.4f} sd={sd:.4f}  95%CI[{lo:.4f}, {hi:.4f}]")
    print(f"            raw={['%.4f'%v for v in x]}")
s = [r["bpb"] for r in B if r["arm"] == "standard"]
p = [r["bpb"] for r in B if r["arm"] == "phi_b1"]
d, lo, hi, pv = boot_diff_ci(p, s)
verdict = "DIFFERENT" if (lo > 0 or hi < 0) else "TIE (CI spans 0)"
print(f"\n  diff(phi_b1 - standard) = {d:+.4f}  95%CI[{lo:+.4f}, {hi:+.4f}]  p~{pv:.3f}  -> {verdict}")

# Power note: with n=5 and observed sd, what effect could we even detect?
sd_pool = np.sqrt((np.var(s, ddof=1) + np.var(p, ddof=1)) / 2)
print(f"  pooled sd ~ {sd_pool:.4f}; with n=5/arm, ~80% power needs |effect| >~ {2.0*sd_pool:.3f} BPB")
print(f"  (so any phi advantage smaller than ~{2.0*sd_pool:.2f} BPB is UNDETECTABLE here)")
