# IGLA-Coder Loop+1 -- decay isolation (A) + power (B) + F2 mediation (C)

Repo: gHashTag/trios-trainer-igla @ feat/igla-coder-v1 (HEAD 9f8a8a1).
Architecture: pre-norm RMSNorm decoder, trainable pos-emb, full analytic
backward (gradcheck 23/0). CPU-only. Outcome metric: code_val_bpb on
code_val.bin (lower is better). NOT comparable to the tiny_shakespeare
champion BPB=2.2111 -- different corpus.

## Headline (negative result first)

phi-momentum does **NOT** beat tuned standard AdamW. At matched low weight
decay the two arms are a statistical **tie**; at the canonical wd=0.04 the
phi arm is clearly worse. The earlier "phi beats standard at wd=0.01" read
was an artefact of comparing the phi arm against the *high-wd* standard arm
(apples to oranges). The honest sentence stands: **the method survives, phi
does not (yet).** Only `phi^2 + phi^-2 = 3` remains [Verified].

The single largest finding is causal-mechanistic, not a phi win: the
**phi^-3 weight-decay magnitude is the dominant cause of the BPB damage**
(controlled direct effect +3.54 BPB, Gamma_tip = 65), while phi^-1 momentum
contributes a small, regime-dependent effect.

---

## Option A -- weight-decay isolation (hidden=64, steps=300, lr=0.03, 3 seeds)

Bootstrap 95% CIs (20k resamples), arm = phi_b1 (beta1 = phi^-1 = 0.618)
vs standard (beta1 = 0.9), swept over wd:

| wd | standard mean [95% CI] | phi_b1 mean [95% CI] | diff (phi_b1 - standard) | verdict |
|---|---|---|---|---|
| 0.04 (canonical) | 4.395 [4.311, 4.528] | 5.283 [5.100, 5.588] | +0.888 [+0.653, +1.182] | DIFFERENT (phi worse) |
| 0.02 | 3.775 [3.765, 3.794] | 3.893 [3.862, 3.920] | +0.118 [+0.088, +0.147] | DIFFERENT (phi slightly worse) |
| 0.01 | 3.641 [3.588, 3.719] | 3.641 [3.624, 3.657] | -0.0005 [-0.073, +0.054] | **TIE (CI spans 0)** |

**Interpretation [Efit]:** lowering wd helps BOTH arms; phi-momentum closes
the gap to a tie at wd=0.01 but never overtakes. The phi^-3 weight-decay
magnitude (0.236), not phi-momentum, was the prior-loop killer. Falsification
path from the prior loop is resolved: phi-momentum at beta1=phi^-1 is benign
once wd is sane; it is not an advantage.

## Option B -- power run (hidden=128, steps=800, lr=0.01, wd=0.02, 5 seeds)

lr=0.03 was unstable at hidden=128 (train BPB diverged after ~300 steps);
lr=0.01 used instead (muP intuition: wider net needs smaller lr).

| arm | mean | sd | 95% CI | raw |
|---|---|---|---|---|
| standard | 4.181 | 0.109 | [4.085, 4.263] | 4.011, 4.221, 4.196, 4.170, 4.309 |
| phi_b1 | 4.313 | 0.457 | [4.057, 4.717] | 4.259, 3.990, 4.101, 5.113, 4.102 |

diff(phi_b1 - standard) = +0.132 BPB, 95% CI [-0.152, +0.548], p ~ 0.56 ->
**TIE (CI spans 0)** [Conj].

**Power note (honest):** pooled sd ~ 0.33; with n=5/arm, ~80% power requires
|effect| >~ 0.66 BPB. Any phi advantage smaller than ~0.66 BPB is
**undetectable** at this sample size. phi_b1 also shows ~4x higher variance
(one unstable seed at 5.11), i.e. phi-momentum is *less stable*, not better.

## Option C -- F2 long-form CSV + stratified-CDE mediation

The full F2 toolchain (f2_ablation_sweep / f2_dual_mediation /
f2_mediation_sensitivity) is NOT on this branch (only f2_harness.rs), so
Option C is **self-contained**: it emits the contract-compliant CSV and runs
the mediation in numpy.

CSV: `coder_ablation_f2.csv` -- W3C-PROV preamble + `# INPUT STRATUM =
canonical` banner + ASCII snake_case headers, one row per (arm x seed),
2x2 factorial. Schema validator (stand-in for f2_csv_validate): **PASS**.

2x2 factorial cells (mediators are binary, factorially manipulated):
M1 = momentum (beta1 0.9 -> phi^-1), M2 = decay (wd 0.04 -> phi^-3).
Outcome Y = code_val_bpb (positive effect = WORSE).

| (M1 momentum, M2 decay) | arm | mean BPB |
|---|---|---|
| (0,0) | standard | 4.395 |
| (1,0) | phi_b1 | 5.283 |
| (0,1) | phi_wd | 7.939 |
| (1,1) | phi (canonical) | 7.253 |

Pearl controlled direct effects + VanderWeele 4-way decomposition, bootstrap
95% CIs (20k), Gamma_tip robustness (E-value-style):

| Path-specific effect | effect (BPB) | 95% CI | Gamma_tip | status |
|---|---|---|---|---|
| CDE_decay \| momentum=0 | +3.544 | [+3.420, +3.643] | 65.2 (robust) | **[Verified]** |
| CDE_momentum \| decay=0 | +0.888 | [+0.653, +1.182] | 8.20 (robust) | **[Verified]** |
| CDE_momentum \| decay=phi^-3 | -0.686 | [-1.564, +0.024] | 3.13 (CI spans 0) | [Conj] |
| Total effect (both phi on) | +2.858 | [+2.020, +3.565] | 8.87 (robust) | **[Verified]** |
| Mediated-interaction (TE - CDEs) | -1.574 | [-2.410, -0.844] | 5.47 (robust) | **[Verified]** |

Decomposition identity holds exactly: CDE_dec + CDE_mom + interaction = TE
(3.544 + 0.888 - 1.574 = 2.858).

**Interpretation [Verified for the decomposition; phi claims stay Conj/Retr]:**
- The phi^-3 **decay** path carries ~+3.54 BPB of the damage -- the dominant,
  extremely robust harmful path.
- The phi^-1 **momentum** path adds only +0.89 BPB when decay is off, and its
  sign **flips** (-0.69, CI spans 0) when decay is already phi^-3 -- evidence
  the two mediators are sub-additive.
- The negative mediated-interaction (-1.57 BPB, robust) quantifies that
  sub-additivity: turning both phi knobs on hurts LESS than the sum of the
  single-knob harms. This is a genuine causal-mechanistic finding about the
  optimizer, independent of any phi-superiority claim (which is NOT made).

## Research applied

- Weight-decay dominance over muP for LR transfer: arXiv 2025-10
  "Weight Decay may matter more than muP for Learning Rate Transfer in
  Practice" -- explains why the phi^-3 wd magnitude dominates the BPB delta.
- RMSNorm pre-norm: Zhang & Sennrich 2019.
- Mediation: Pearl 2001 (direct/indirect effects); VanderWeele 2015
  (4-way decomposition); VanderWeele & Ding 2017 (E-value / Gamma_tip);
  Zhao & Luo 2024 (mediation CIs).

## Verification

- gradcheck: checks=23 fails=0 (PASS) after all edits.
- cargo fmt --all -- --check: clean.
- cargo clippy --all-targets -- -D warnings: clean.
- F2 CSV schema validator: PASS (prov preamble + canonical stratum banner +
  12 ASCII snake_case header cols + 12 well-formed data rows).

## Files (this loop)

- src/bin/igla_coder.rs -- CLI flags --beta1/--wd/--curve/--arms;
  arm_hparams / make_arm / make_opt refactor; curve logging (UNCOMMITTED).
- optionA_grid.csv, optionB_grid.csv -- raw run data.
- coder_ablation_f2.csv -- F2 long-form CSV (contract-compliant).
- emit_f2_csv.py, mediation_f2.py, validate_f2_csv.py, analyze_AB.py -- analysis.

## Claim-status summary

- [Verified] phi^2 + phi^-2 = 3; the mediation decomposition identity; the
  three robust CDEs above (these describe phi's HARM, not its benefit).
- [Conj] phi-momentum ties standard at low wd (undetectable below ~0.66 BPB).
- [Retr] "phi-momentum beats standard at wd=0.01" (prior-loop optimistic read;
  was an apples-to-oranges comparison against the high-wd standard arm);
  delta_CP = 3/phi^2 (project-wide).

## Three collaboration options for Loop+2

**A. Close the power gap (more seeds + curves).**
Direction: run hidden=128 to ~2000 steps with 10-12 seeds per arm, log full
train/val curves, and add an early-stopping / best-checkpoint readout so the
unstable phi_b1 seed (5.11) is handled honestly. Tighten the CI until a
<=0.3 BPB effect is detectable.
Cost/Risk: ~24-30 CPU runs at ~4-7 min each under throttling (several hours,
one seed per call); low conceptual risk, pure power.

**B. Real F2 toolchain bridge.**
Direction: port the minimal f2_dual_mediation reader/writer (preamble + INPUT
STRATUM passthrough) into this branch as a real Rust binary, so the coder CSV
flows through the actual F2 pipeline instead of the numpy stand-in; add the
wd0 stratum (pin wd=0, the dominant mediator) to get a Pearl CDE along the
non-decay paths.
Cost/Risk: ~1-2 binaries + tests; medium effort; unlocks cross-stratum
compare and aligns the coder work with the f2-mediation-loop framework.

**C. lr x wd 2-D frontier + phi-lr arm.**
Direction: the wd finding suggests the real lever is the (lr, wd) coupling.
Sweep a small lr x wd grid at hidden=64/128, add a phi-lr arm
(lr scaled by phi^-1), and test whether ANY all-phi-derived config reaches
the standard frontier -- the honest "can one phi constant generate a
competitive config" test (MDL-prior framing), with the falsification path explicit.
Cost/Risk: ~16-25 runs; medium; directly probes the phi-as-hyperparameter-
generator hypothesis rather than single knobs.

STOP -- pick A / B / C (or a combination) for Loop+2.
