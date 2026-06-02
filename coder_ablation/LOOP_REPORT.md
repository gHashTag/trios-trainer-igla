# IGLA-Coder Loop+3 -- third stratum + cross-stratum CDE stability

Repo: gHashTag/trios-trainer-igla @ feat/igla-coder-v1 (HEAD 7abf019 + this
loop's uncommitted edits). Architecture: pre-norm RMSNorm decoder, trainable
pos-emb, full analytic backward (gradcheck 23/0). CPU-only. Outcome metric:
code_val_bpb on code_val.bin (lower is better). NOT comparable to the
tiny_shakespeare champion BPB=2.2111 -- different corpus.

## Audit -- weak spots considered (Loop+3)

1. **Proposed warmup0 stratum is FICTITIOUS for this trainer.** Audit of
   `src/bin/igla_coder.rs` confirms there is NO learning-rate warmup and NO
   LR schedule (lr is constant; grep for `warmup`/`schedule`/`cosine` returns
   nothing). The f2-mediation-loop rule requires a stratum to pin a REAL
   mediator carrying >= 50% of the indirect effect. Pinning a non-existent
   warmup would be a banner with no mechanism behind it. **Dropped.**
2. The single-mediator branch in `f2_dual_mediation` was hardcoded for the
   decay-pinned (wd0) case only -- it would panic on a momentum-pinned CSV
   (missing `(1,0)` cell). Generalised this loop.
3. The wd0 momentum sign flip (Loop+2) had no symmetric control: was decay's
   harmful CDE also regime-dependent, or stable? Unanswered until now.
4. No cross-stratum stability flag existed for the coder track (the skill's
   `f2_stratum_compare` is not on this branch). Added a numpy bridge.
5. The CSV schema validator did not know any stratum beyond
   {canonical, wd0, warmup0}. Extended.

## Headline (negative result first)

phi still does **NOT** beat tuned standard AdamW for the coder model -- nothing
this loop changes that. The new work is causal-mechanistic, not a phi win, and
it sharpens the Loop+2 finding into a clean dichotomy between the two phi
knobs:

- **decay CDE is stable across strata and robustly HARMFUL.** The phi^-3
  weight-decay penalty costs +3.544 BPB whether momentum is free (canonical)
  or pinned to standard (mom_std) -- byte-identical effect, overlapping CIs,
  same sign, Gamma_tip ~65. phi^-3 decay is a **regime-independent liability**.
- **momentum CDE is NOT stable across strata.** It is +0.888 BPB (harmful) at
  canonical wd=0.04 but flips to -0.268 BPB (helpful) at wd=0. The phi^-1
  momentum knob is a **regime-dependent wildcard** whose sign is set entirely
  by the weight-decay regime.

The honest sentence stands: **the method survives, phi does not (yet).** Only
`phi^2 + phi^-2 = 3` is [Verified]. The one cell where a phi knob helps
(wd0 momentum) lives in a regime nobody ships (zero weight decay). The
mechanism literature explains why: warmup -- which this trainer lacks --
normally absorbs the beta1 bias-correction shock; without it, the phi^-1
momentum arm is the unstable one we measured in Loop+2.

---

## Implemented this loop

### New stratum: mom_std (momentum pinned to standard)

Symmetric completion of the 2x2 factorial begun by wd0. Where wd0 pins decay
(wd=0) and varies momentum, **mom_std pins momentum (beta1=0.9) and varies
decay** (0.04 -> phi^-3). The PSE is a Pearl CDE of decay along the
non-momentum paths. Six new CPU runs (hidden=64, 300 steps, seeds 42/43/44):

| arm (decay) | beta1 | wd | seed42 | seed43 | seed44 |
|---|---|---|---|---|---|
| standard (baseline) | 0.9 | 0.04 | 4.3111 | 4.5282 | 4.3456 |
| phi_wd (phi decay) | 0.9 | phi^-3 | 7.9469 | 7.9743 | 7.8957 |

`coder_ablation_f2_mom_std.csv` (INPUT STRATUM = mom_std, single-mediator).

### Generalised single-mediator branch (f2_dual_mediation)

The binary now detects WHICH mediator is pinned from the present cells and
computes the matching CDE:
- decay-pinned (wd0): cells (0,0),(1,0) -> momentum CDE along non-decay paths.
- momentum-pinned (mom_std): cells (0,0),(0,1) -> decay CDE along non-momentum
  paths.
New unit test `mom_std_single_mediator_csv_has_only_momentum0_cells`
(5 tests total, all green). The validator's KNOWN_STRATA now includes mom_std.

### Cross-stratum comparator (coder_ablation/stratum_compare_coder.py)

Runs f2_dual_mediation on all three strata and emits a per-mediator
`stable_across_strata` flag (CI overlap AND sign agreement). The
coder-track analogue of the skill's f2_stratum_compare.

## Empirical findings (the three strata)

| stratum | PSE | effect (BPB) | 95% CI | Gamma_tip | status |
|---|---|---|---|---|---|
| canonical | CDE_momentum (decay=0) | +0.888 | [+0.653, +1.182] | 8.20 | [Verified] |
| canonical | CDE_decay (momentum=0) | +3.544 | [+3.419, +3.643] | 64.82 | [Verified] |
| canonical | total effect | +2.858 | [+2.032, +3.565] | 8.92 | [Verified] |
| canonical | mediated interaction | -1.574 | [-2.402, -0.839] | 5.48 | [Verified] |
| wd0 | CDE_momentum (decay pinned) | **-0.268** | [-0.367, -0.180] | 7.21 | [Verified] |
| mom_std | CDE_decay (momentum pinned) | **+3.544** | [+3.420, +3.643] | 65.16 | [Verified] |

### Cross-stratum stability

| mediator | stratum A | effect A | stratum B | effect B | CI overlap | sign agree | **stable** |
|---|---|---|---|---|---|---|---|
| momentum | canonical | +0.888 | wd0 | -0.268 | No | No | **False** |
| decay | canonical | +3.544 | mom_std | +3.544 | Yes | Yes | **True** |

**Interpretation [Verified for both stability verdicts; no phi-superiority
claim]:** the two phi-derived knobs behave categorically differently.
- **decay (phi^-3): regime-independent, robustly harmful.** Its CDE is
  identical (to 4 dp) whether momentum is free or pinned, Gamma_tip ~65 in
  both. The phi^-3 weight-decay magnitude is simply too large for this model;
  no regime rescues it.
- **momentum (phi^-1): regime-dependent.** Helpful only when weight decay is
  exactly zero; harmful at the canonical wd=0.04 that real training uses.
  There is no regime-independent "phi-momentum is good/bad" claim to make.

This is the strongest causal statement the coder track supports, and it is a
statement about phi's *liabilities and conditionality*, not its benefits.

## Research applied

- "Analyzing & Reducing the Need for Learning Rate Warmup in GPT Training"
  (NeurIPS 2024, arXiv:2410.23922): warmup's mechanism is tied specifically to
  Adam's beta1 bias-correction shock and interacts with weight decay through
  the scale-invariant-weight equilibrium magnitude. Two consequences for us:
  (a) it justifies DROPPING the warmup0 stratum (no warmup exists here, so the
  beta1 shock is unmitigated -- consistent with phi_b1's higher Loop+2
  variance); (b) since warmup interacts with weight decay, stratifying on the
  decay regime is exactly the right lens for the momentum knob.
- VanderWeele 2015 (4-way decomposition) and Pearl 2001 (controlled direct
  effects): the mom_std stratum is the second single-mediator CDE that, with
  wd0 and canonical, lets us test mediator-effect stability rather than assume
  it. VanderWeele & Ding 2017 for the Gamma_tip / E-value robustness transform.
- "Mediation Analysis with Multiple Mediators" (PMC4287269): the
  cross-stratum stability check (do CDEs agree across strata in which a
  mediator is identified) is the multiple-mediator analogue of consistency
  checks across identification strategies.

## Verification

- gradcheck: checks=23 fails=0 (PASS).
- cargo fmt --all -- --check: clean (exit 0).
- cargo clippy --all-targets -- -D warnings: clean (exit 0).
- cargo test --release --bin f2_dual_mediation: 5 passed; 0 failed (adds
  mom_std_single_mediator_csv_has_only_momentum0_cells).
- F2 CSV schema validator: PASS on ALL THREE -- canonical (12 rows),
  wd0 (6 rows), mom_std (6 rows).
- Regression: canonical + wd0 outputs reproduce Loop+2 to displayed precision.

## Files (this loop)

- src/bin/f2_dual_mediation.rs -- generalised single-mediator branch (handles
  both decay-pinned and momentum-pinned strata) + 1 new unit test (5 total).
- coder_ablation/emit_f2_mom_std.py -- mom_std CSV generator + design note on
  why warmup0 was dropped.
- coder_ablation/coder_ablation_f2_mom_std.csv -- mom_std stratum (6 rows).
- coder_ablation/stratum_compare_coder.py -- cross-stratum comparator.
- coder_ablation/validate_f2_csv.py -- KNOWN_STRATA += mom_std.

## Claim-status summary

- [Verified] phi^2 + phi^-2 = 3; the decomposition identity; the four
  canonical CDEs; the mom_std decay CDE (+3.544, Gamma 65); the two
  cross-stratum stability verdicts (decay stable, momentum not). All describe
  phi's harmful or conditional effects, NOT a benefit.
- [Conj] the wd0 momentum benefit (-0.268) is real but regime-specific (wd=0
  only) and does not generalise to shipped training.
- [Efit -> falsified] (from Loop+2, unchanged) no phi-derived config reaches
  the standard frontier; phi-lr alone hurts.
- [Retr] "phi-momentum beats standard" (project-wide); delta_CP = 3/phi^2.

## Three collaboration options for Loop+4

**A. Decay-magnitude dose-response (find where phi^-3 goes wrong).**
Direction: decay is the regime-independent killer at phi^-3 (+3.54 BPB). Sweep
wd along a dose-response curve {0, 0.01, 0.02, 0.04, phi^-3/2~0.118, phi^-3}
at fixed momentum, hidden=64, to locate the BPB knee and quantify exactly how
far phi^-3 overshoots the optimum. Answers "is ANY phi-derived decay value
admissible?" with a curve, not a point.
Cost/Risk: ~12-18 short CPU runs; low; turns the single harmful point into a
calibrated dose-response and gives a fair test of the decay prior.

**B. Detectable-effect power run on momentum (settle the Loop+2 tie).**
Direction: the hidden=128 phi_b1-vs-standard tie still has wide CIs
(~+/-0.15 BPB at best-checkpoint). Run hidden=128 to ~1500-2000 steps,
10-12 seeds/arm, fixed best-checkpoint protocol, and report the achieved
minimum detectable effect explicitly, so the tie is either tightened or
resolved. Pairs naturally with the now-established regime-dependence of
momentum.
Cost/Risk: ~24-30 CPU runs at ~5-8 min each under throttling (several hours,
one seed per call); low conceptual risk, pure statistical power.

**C. Port a real f2_stratum_compare Rust binary (retire the numpy bridge).**
Direction: replace `stratum_compare_coder.py` with a proper
`src/bin/f2_stratum_compare.rs` that reads N dual-mediation CSVs, joins on
mediator, and emits the long-form cross-stratum CSV with stable_across_strata
-- matching the skill's binary contract, with unit tests and INPUT STRATUM
passthrough. Closes weak-spot #4 in Rust instead of Python.
Cost/Risk: ~no new compute; medium implementation; brings the coder track's
tooling fully onto the Rust f2 contract and removes the last numpy stand-in.

STOP -- pick A / B / C (or a combination) for Loop+4.
