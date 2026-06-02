# IGLA-Coder Loop+2 -- power (A) + real F2 bridge (B) + lr x wd frontier (C)

Repo: gHashTag/trios-trainer-igla @ feat/igla-coder-v1 (HEAD bd2b06e + this
loop's uncommitted edits). Architecture: pre-norm RMSNorm decoder, trainable
pos-emb, full analytic backward (gradcheck 23/0). CPU-only. Outcome metric:
code_val_bpb on code_val.bin (lower is better). NOT comparable to the
tiny_shakespeare champion BPB=2.2111 -- different corpus.

## Headline (negative result first)

phi still does **NOT** beat tuned standard AdamW for the coder model. Three
independent angles this loop all return the same verdict:

1. **Power (A):** with a best-checkpoint readout at hidden=128 / 1000 steps /
   6 seeds, phi_b1 and standard are a statistical **tie** (delta_best =
   -0.005 BPB [-0.136, +0.156]). The apparent final-step gap is entirely an
   instability artefact: phi_b1's variance is ~2.7x standard's and its worst
   seed drifts +1.46 BPB between best and final. phi-momentum is **less
   stable, not better**.
2. **Frontier (C):** **no** phi-derived configuration reaches the standard
   frontier (3.775 BPB). phi-lr alone costs +0.31 BPB [+0.27, +0.34]; the
   all-phi config is catastrophic (+4.24 BPB); phi-tempered is +0.46 BPB. All
   CIs lie strictly above zero -- this is a **falsification**, not a tie. The
   "one phi constant generates a competitive config" (MDL-prior) hypothesis
   is rejected at this scale.
3. **Mechanism (B):** the new real-Rust F2 bridge reproduces the numpy
   stand-in exactly, and the wd0 stratum delivers the cleanest causal finding
   of the whole coder track (sign flip, below).

The honest sentence stands: **the method survives, phi does not (yet).** Only
`phi^2 + phi^-2 = 3` remains [Verified].

The single most informative new result is causal-mechanistic, not a phi win:
**phi-momentum's effect is entirely conditional on the weight-decay regime.**
At canonical wd=0.04 it is harmful (CDE +0.89 BPB); when weight decay is
pinned to zero its controlled direct effect **flips sign** to -0.27 BPB
[-0.37, -0.18], Gamma_tip = 7.21, [Verified]. phi-momentum *helps* only in a
regime nobody trains in.

---

## Option A -- power run (hidden=128, 1000 steps, lr=0.01, wd=0.02, 6 seeds)

Best-checkpoint readout added this loop (`eval_every`, `eval_val`, bestval
print line) so the unstable phi_b1 seed is handled honestly instead of being
read at a diverged final step. lr=0.01 (not 0.03) because lr=0.03 was unstable
at hidden=128 in Loop+1 (muP intuition: wider net, smaller lr).

| arm | readout | mean [95% CI] | sd |
|---|---|---|---|
| standard | final | 4.093 [3.944, 4.253] | 0.200 |
| standard | best  | 4.011 [3.933, 4.091] | 0.101 |
| phi_b1   | final | 4.340 [3.969, 4.813] | 0.540 |
| phi_b1   | best  | 4.006 [3.918, 4.146] | 0.153 |

Bootstrap 95% CIs (20k resamples), paired-arm delta:

| readout | delta (phi_b1 - standard) | 95% CI | verdict |
|---|---|---|---|
| final | +0.246 | [-0.166, +0.742] | TIE (CI spans 0), but phi 2.7x variance |
| best  | -0.005 | [-0.136, +0.156] | **TIE (CI spans 0)** |

**Interpretation [Conj]:** the best-checkpoint readout collapses the final-step
"gap" to zero. The only real difference is variance: phi_b1 max best-vs-final
instability gap = 1.46 BPB (seed 43: final 5.39 vs best 3.93) vs standard's
0.33. phi-momentum does not improve the achievable minimum; it widens the
spread. Reporting the diverged final step would have *overstated* a phi
disadvantage -- the honest readout shows a tie with worse stability.

## Option B -- real F2 bridge (Rust binary) + wd0 stratum (Pearl CDE)

The full F2 toolchain was not on this branch (only f2_harness.rs). This loop
ports a minimal real Rust binary `src/bin/f2_dual_mediation.rs` (registered in
Cargo.toml), so the coder CSV now flows through a real F2-contract reader
instead of the numpy stand-in. The binary: SplitMix64 RNG (no external crate),
`#`-preamble + INPUT STRATUM + W3C-PROV passthrough, 2x2 Pearl CDE + 4-way
VanderWeele decomposition, bootstrap CI (20k), Gamma_tip (E-value transform),
claim-status labels, and a single-mediator branch for decay-pinned strata.
4 unit tests, all green.

**Reproduction check (canonical, `--csv coder_ablation_f2.csv`):** the Rust
binary reproduces the Loop+1 numpy stand-in to the displayed precision.

| Path-specific effect | effect (BPB) | 95% CI | Gamma_tip | status |
|---|---|---|---|---|
| CDE_momentum \| decay=0 | +0.888 | [+0.653, +1.182] | 8.20 | **[Verified]** |
| CDE_decay \| momentum=0 | +3.544 | [+3.419, +3.643] | 64.82 | **[Verified]** |
| Total effect (both phi on) | +2.858 | [+2.032, +3.565] | 8.92 | **[Verified]** |
| Mediated-interaction | -1.574 | [-2.402, -0.839] | 5.48 | **[Verified]** |

**New wd0 stratum (`coder_ablation_f2_wd0.csv`, weight_decay pinned to 0,
3 seeds).** Following the f2-mediation-loop stratum rule (pin the dominant
mediator -- decay carries +3.54 BPB, the largest IE -- to isolate the
momentum path as a Pearl CDE along the non-decay paths):

| stratum | PSE | effect (BPB) | 95% CI | Gamma_tip | status |
|---|---|---|---|---|---|
| canonical (wd=0.04) | CDE_momentum | +0.888 | [+0.653, +1.182] | 8.20 | [Verified] |
| **wd0 (wd=0)** | CDE_momentum (decay-pinned) | **-0.268** | [-0.367, -0.180] | 7.21 | **[Verified]** |

**Interpretation [Verified for the sign flip; no phi-superiority claim]:**
this is the cleanest cross-stratum result on the coder track. phi-momentum's
controlled direct effect is **+0.89 BPB (harmful) at canonical wd=0.04** but
**-0.27 BPB (helpful) at wd=0**, both robust (Gamma_tip > 7). The effect of
the phi^-1 momentum knob is **entirely conditional on the weight-decay
regime** -- there is no regime-independent "phi-momentum is good/bad" claim to
make. Since real training uses non-zero weight decay, the operative regime is
the canonical one where phi-momentum hurts. The wd0 benefit is real but lives
in a regime nobody ships.

## Option C -- lr x wd frontier + phi-lr arm (hidden=64, 300 steps, 3 seeds)

Probes the phi-as-hyperparameter-generator (MDL-prior) hypothesis directly:
can ONE phi constant generate a config that reaches the standard frontier?
phi-lr = 0.03 * phi^-1 = 0.0185410197.

| config | beta1 | wd | lr | mean [95% CI] | vs frontier |
|---|---|---|---|---|---|
| standard_lr03_wd04 | 0.9 | 0.04 | 0.03 | 4.395 [4.311, 4.528] | +0.620 |
| **standard_lr03_wd02** | 0.9 | 0.02 | 0.03 | **3.775 [3.765, 3.794]** | FRONTIER |
| standard_philr_wd02 | 0.9 | 0.02 | phi-lr | 4.086 [4.045, 4.109] | +0.311 |
| allphi | phi^-1 | phi^-3 | phi-lr | 8.012 [7.958, 8.039] | +4.237 |
| phi_tempered_philr_wd02 | phi^-1 | 0.02 | phi-lr | 4.232 [4.159, 4.287] | +0.457 |

Frontier-reach test (paired bootstrap delta vs standard_lr03_wd02):

| config | delta | 95% CI | reaches frontier? |
|---|---|---|---|
| standard_philr_wd02 | +0.311 | [+0.270, +0.342] | **NO (worse)** |
| allphi | +4.237 | [+4.184, +4.274] | **NO (worse)** |
| phi_tempered_philr_wd02 | +0.457 | [+0.384, +0.512] | **NO (worse)** |

**Interpretation [Efit -> falsification]:** every phi-derived config is
strictly worse than the standard frontier (all CIs above zero). phi-lr alone
hurts (+0.31 BPB) -- scaling the learning rate by phi^-1 is simply too small a
step here. The all-phi stack (phi momentum + phi^-3 decay + phi-lr) is
dominated by the phi^-3 weight-decay term and lands +4.24 BPB off. The
MDL-prior hypothesis -- "one phi constant generates a competitive config" --
is **rejected at this scale**: no single-constant phi derivation reaches the
hand-tuned frontier. The compact-code advantage of phi as a generator is real
for *description length*, but it does not buy a competitive *config* here.

## Research applied

- Weight-decay dominance over muP for LR transfer: arXiv 2025-10 "Weight Decay
  may matter more than muP for Learning Rate Transfer in Practice" -- explains
  why the phi^-3 wd term dominates the all-phi config and why best-checkpoint
  vs final readout diverges most for the higher-variance arm.
- RMSNorm pre-norm: Zhang & Sennrich 2019.
- Mediation / stratified CDE: Pearl 2001 (direct/indirect effects);
  VanderWeele 2015 (4-way decomposition); VanderWeele & Ding 2017 (E-value /
  Gamma_tip); Zhao & Luo 2024 (mediation CIs). Stratum rule (pin dominant
  mediator -> Pearl CDE) per the f2-mediation-loop framework.

## Verification

- gradcheck: checks=23 fails=0 (PASS) after all edits.
- cargo fmt --all -- --check: clean (exit 0).
- cargo clippy --all-targets -- -D warnings: clean (exit 0).
- cargo test --release --bin f2_dual_mediation: 4 passed; 0 failed
  (reads_csv_with_preamble_and_stratum_banner,
  cde_decay_is_dominant_and_decomposition_holds, wd0_stratum_is_detected,
  wd0_single_mediator_csv_has_only_decay0_cells).
- F2 CSV schema validator: PASS on BOTH coder_ablation_f2.csv (canonical,
  12 rows) AND coder_ablation_f2_wd0.csv (wd0, 6 rows).
- Rust f2_dual_mediation reproduces the numpy stand-in on the canonical CSV to
  displayed precision.

## Files (this loop)

- src/bin/f2_dual_mediation.rs -- NEW minimal real Rust F2 bridge (4 tests).
- Cargo.toml -- registered the new bin.
- src/bin/igla_coder.rs -- eval_val helper, TrainCfg.eval_every,
  best-checkpoint tracking + bestval print, --eval-every CLI flag.
- loop2_optionA.csv -- power run raw data (6 seeds x 2 arms, final + best).
- loop2_optionC.csv -- frontier raw data (5 configs x 3 seeds).
- coder_ablation_f2_wd0.csv -- wd0 stratum F2 long-form CSV.
- emit_f2_wd0.py -- wd0 CSV generator.
- analyze_loop2.py -- bootstrap CI analysis (A + C).

## Claim-status summary

- [Verified] phi^2 + phi^-2 = 3; the mediation decomposition identity; the
  four canonical CDEs; the wd0 CDE_momentum sign flip (-0.27 BPB, Gamma 7.21).
  These describe phi's regime-dependent and mostly-harmful effects, NOT a
  benefit.
- [Conj] phi_b1 ties standard at hidden=128 best-checkpoint (delta -0.005,
  CI spans 0); the tie is "no better, higher variance".
- [Efit -> falsified] no phi-derived config reaches the standard frontier;
  phi-lr alone hurts; all-phi is +4.24 BPB off. MDL-single-constant-config
  hypothesis rejected at this scale.
- [Retr] "phi-momentum beats standard" (project-wide); delta_CP = 3/phi^2.

## Three collaboration options for Loop+3

**A. Cross-stratum compare + warmup0 stratum (close the F2 loop).**
Direction: add the warmup0 stratum (pin warmup=0, the other candidate
mediator) and run a real f2_stratum_compare-style join across
canonical / wd0 / warmup0 for the coder CSV, emitting a stable_across_strata
flag per PSE. The wd0 sign flip is the highest-information result so far;
a third stratum tests whether the momentum CDE is stable or regime-specific
across two different pinned nuisances.
Cost/Risk: ~6-9 short CPU runs + one comparator pass; low; directly extends
the cleanest finding and aligns fully with the f2-mediation-loop pipeline.

**B. Detectable-effect power run (settle the tie honestly).**
Direction: the Option A tie has wide CIs (~+/-0.15 BPB at best-checkpoint).
Run hidden=128 to ~1500-2000 steps with 10-12 seeds per arm and a fixed
best-checkpoint protocol, so a <=0.15 BPB effect becomes detectable and the
tie is either confirmed tight or resolved. Report the achieved minimum
detectable effect explicitly.
Cost/Risk: ~24-30 CPU runs at ~5-8 min each under throttling (several hours,
one seed per call); low conceptual risk, pure statistical power.

**C. lr x wd 2-D surface + phi-on-the-grid test.**
Direction: Option C tested only the phi-lr point. Sweep a denser lr x wd grid
(e.g. lr in {0.01, 0.02, 0.03, phi-lr}, wd in {0, 0.01, 0.02, 0.04, phi^-3})
at hidden=64 to map the BPB surface, then ask the sharper MDL question: does
ANY phi-derived (lr, wd) PAIR land on the empirical frontier, even if no
single phi constant does? Falsification path explicit.
Cost/Risk: ~16-20 runs; medium; turns the single-point frontier rejection
into a real surface and gives a fair test of the multi-constant phi prior.

STOP -- pick A / B / C (or a combination) for Loop+3.
