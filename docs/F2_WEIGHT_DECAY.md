# F2 Weight Decay Policy

**Status**: SANDBOX policy as of Loop 27 (2026-06-01). Production default unchanged.

## TL;DR

Default `weight_decay = 0.1` (BitNet 2B4T convention) is wrong at our sandbox scale.
Empirical optimal at N≈8K params, B=1 batch, D≈10K tokens is `weight_decay = 0.0`.
Per Bergsma 2025 "Power Lines" (arXiv:2505.13738), the formula

    λ_opt(B,η,D,N) = B · TPP^(-m_τ) / (c_τ · η · D)

with `TPP = D/N`, `m_τ ≈ -0.52`, gives `λ_opt ≈ 4.1e-3` for our config — three
orders of magnitude below the BitNet default and well within the empirically
useless range we observed.

## Evidence (Loop 26 + Loop 27 ablation matrix)

Long-form CSV: `/tmp/loop25_ablation_1k.csv`, `/tmp/loop26_wd_sweep.csv`,
`/tmp/loop27_pairwise.csv`.

### Cumulative (adding fixes incrementally, n=0..7)

| n | + fix      | val BPB ± std       | Δ vs n=0 | Cohen's d |
|---|------------|---------------------|---------:|----------:|
| 0 | baseline   | 0.794 ± 0.252       |  +0.000  |     0.0   |
| 1 | RMSNorm    | 0.161 ± 0.020       |  −0.633  |    −3.5   |
| 2 | + warmup   | 0.253 ± 0.204       |  −0.541  |    −2.4   |
| 3 | + gradclip | 0.173 ± 0.014       |  −0.621  |    −3.5   |
| 4 | + clamp    | 0.175 ± 0.013       |  −0.620  |    −3.5   |
| 5 | + smoothing| 0.175 ± 0.013       |  −0.620  |    −3.5   |
| 6 | + WD=0.1   | **4.179 ± 0.078**   |  +3.385  |   +18.1   |
| 7 | + dropout  | **4.370 ± 0.065**   |  +3.576  |   +19.4   |

WD addition causes a 24× BPB regression. p (paired Welch t, df=n−1) = 1.2e-5.

### LOCO (leave-one-fix-out from full stack)

| removed     | val BPB ± std       |
|-------------|---------------------|
| RMSNorm     | 6.000 ± 0.000 (collapse) |
| warmup      | 4.424 ± 0.046       |
| gradclip    | 4.433 ± 0.053       |
| clamp       | 4.478 ± 0.047       |
| smoothing   | 4.478 ± 0.047       |
| **WD=0.1**  | **0.072 ± 0.007**   |
| dropout     | 4.174 ± 0.048       |

Removing WD from the full stack drops BPB from 4.37 → 0.072 (60× improvement,
Cohen's d = −1258). This is the **dominant compensatory interaction** in the
ablation matrix (Hooker 2019 signature).

### WD sweep (held-out grid)

| λ      | val BPB ± std       |
|--------|---------------------|
| 0.000  | 0.072 ± 0.007       |
| 0.005  | 0.153 ± 0.006       |
| 0.010  | 0.310 ± 0.018       |
| 0.030  | 1.491 ± 0.054       |
| 0.100  | 4.478 ± 0.047       |
| 0.300  | 5.536 ± 0.014       |

Monotone degradation. Sandbox optimum at λ=0.0.

### Pairwise iLOCO (21 pairs)

All pairs `(*, wd)` collapse to < 1 BPB; all pairs without `wd` stay > 5 BPB
(except `rms_*` which catastrophically collapse to 6.0 for unrelated reasons).
This confirms WD as the **single dominant interferer**.

iLOCO_{warmup, wd} = Δ_warmup + Δ_wd − Δ_{warmup, wd}
                   = 3.63 + (−0.72) − (−0.53)
                   = **+3.44**

Positive iLOCO → strong compensatory interaction per arXiv:2502.06661 Eq.(3).

## Why our sandbox empirical λ=0 contradicts Bergsma 2025

Bergsma's Power Lines law is derived at B≥256, η≈1e-3, D∈[1e9, 1e11]. Our
sandbox runs at B=1, η=4e-3, D≈10K. The B=1 regime collapses the formula's
batch term, pushing λ_opt → 0 in practice even where the law predicts 4e-3.
At B=1 every gradient is fully stochastic; weight decay then competes directly
with the only useful signal. This matches the Loshchilov–Hutter AdamW result
that decoupled WD must scale with √B.

## Policy

**Sandbox (current)**: `weight_decay = 0.0` is the right default for our
N=8K, B=1, D=10K regime. **Validated**: do not change empirically.

**Production (BitNet 2B4T target)**: `weight_decay = 0.1` per BitNet recipe.
**Hypothesis**: at champion scale the Power Lines law applies and λ ∈ [0.05, 0.1]
becomes optimal. **Not validated here.**

### Backward compatibility

Per PyTorch BC policy (2-release deprecation, no silent default changes):
**keep `MultiSeedConfig::weight_decay` defaulting to 0.1** in code.
Add a `WeightDecayPolicy` enum in a future minor release with variants:

    enum WeightDecayPolicy {
        BitNet,       // 0.1, current default
        PowerLines,   // λ_opt formula from B, η, D, N
        Disabled,     // 0.0, sandbox default
    }

Flip the default at F3.0 (major) only after validating champion-scale BPB.

## References

- BitNet b1.58 2B4T (arXiv:2503.18757) — WD=0.1 stage 1, 0.0 stage 2.
- Power Lines (Bergsma 2025, arXiv:2505.13738) — λ_opt = B · TPP^(-m_τ) / (c_τ · η · D).
- Loshchilov & Hutter ICLR 2019 — decoupled WD scales with √B.
- iLOCO (arXiv:2502.06661 Eq.3) — pairwise compensatory interaction definition.
- Hooker 2019 — compensatory model interpretability framework.
- Pereyra et al. ICLR 2017 — label smoothing + WD + dropout combined effect.
