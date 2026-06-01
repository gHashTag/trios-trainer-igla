# Pearl-Style Multi-Stratum CDE for Transformer Training-Recipe Ablations

**Working title.** Draft outline started Loop 51 (2026-06-01).
Target venue: NeurIPS 2026 ML Reproducibility Workshop OR Causal-ML Workshop
(see §10 for venue calibration).

---

## Abstract (target: 250 words)

The standard practice for evaluating training-recipe interventions (RmsNorm,
weight decay, dropout, label smoothing, gradient clipping, warmup schedules,
latent clamping) in transformer language models is multi-seed mean BPB
comparison, sometimes supplemented with Cohen's *d* or paired t-tests. We
show that this practice systematically misattributes effects when one
intervention (commonly weight decay) acts as a strong mediator for another.

We introduce a Pearl-style controlled-direct-effect (CDE) framework
operationalized through three contributions: (1) a stratification mechanism
(`Stratum::Wd0`, `Stratum::Warmup0`) that pins a candidate mediator to its
disabled value while sweeping the remaining ablation space; (2) a four-path
Zhao-Luo decomposition with multivariate delta-method standard errors,
yielding per-path 95% CIs valid at N=5 seeds; (3) an additive bridge-score
sensitivity envelope (Ohnishi & Li 2026, Thm 2) that translates CI bounds
into VanderWeele-Ding tipping points across an outcome-residual grid.

Applied to a sandbox-scale (~8K params, 200 steps, 5 seeds) ablation matrix,
the framework reveals a **sign flip**: under canonical mediation the RmsNorm
NDE is −4.12 BPB (apparently harmful), but under wd=0 Pearl CDE the same
estimand is **+0.43 BPB [+0.01, +0.84]** (intrinsically helpful, CI excludes
zero). The pattern replicates under an alternative mediator parameterization
where the rms-mediated NIE is −0.75 BPB stable across all three strata
(CI excludes zero universally). We argue this reframes the "rms is the only
intrinsic fix" claim from prior work into a quantitative, sign-corrected
empirical finding.

The framework is open-source in Rust with 11 binaries, 726 unit/integration
tests, and W3C-PROV-tagged CSV provenance preambles. We release it as a
reproducibility artifact for ML methodology research.

---

## 1. Introduction

### 1.1 The problem with seed-mean ablation
- ML practitioners report mean BPB ± std across 3-5 seeds.
- Suppression mediation — where one intervention masks another's intrinsic
  effect — can flip the sign of the apparent effect.
- We document one instance (RmsNorm × WD) and provide infrastructure to
  detect more.

### 1.2 Contributions
1. Stratification framework (3 strata: canonical, wd0, warmup0)
2. Zhao-Luo 4-PSE decomposition with t-CI valid at N=5
3. Bridge-score sensitivity envelope with tipping-point classification
4. Cross-stratum comparator with stability flag
5. Empirical: RmsNorm CDE sign-flip across strata

### 1.3 Roadmap
§2 background, §3 framework, §4 sandbox ablation matrix, §5 results,
§6 sensitivity, §7 limitations, §8 software, §9 related work, §10 conclusion.

---

## 2. Background

### 2.1 Pearl causal mediation
- Natural direct effect (NDE) vs Natural indirect effect (NIE)
- Controlled direct effect (CDE) when mediator is set to a reference level
- Zhao & Luo (2020) four-way decomposition for two ordered mediators

### 2.2 Sensitivity analysis
- E-value (VanderWeele & Ding 2017): Γ such that unmeasured confounding
  of that strength would explain the effect
- Bridge-score additive bound (Ohnishi & Li 2026): same idea on an additive
  scale, parameterized by (Γ, Λ)

### 2.3 ML ablation practice
- AblationBench (arXiv:2507.08038): paired Welch + Cohen's d wide-form
- AblateMe / ablator (PMLR 2023): multi-seed ranking but no mediation
- ROME / activation patching: causal but single forward pass, not training

---

## 3. The F2 framework

### 3.1 Stratification mechanism
- `Stratum::ALL = &[Canonical, Wd0, Warmup0]`
- Mode-string registry: `mode_string(kind, stratum) → "wd0_loco"` etc.
- Implementation: `src/race/ablation.rs`
- **Figure 2**: Stratum enum + registry flow diagram

### 3.2 Zhao-Luo decomposition
- NDE = Δ_{X,M1,M2}
- NIE_chain = (Δ_X − Δ_{X,M1}) − (Δ_{X,M2} − Δ_{X,M1,M2})
- NIE_M1 = Δ_{X,M2} − Δ_{X,M1,M2}; NIE_M2 = Δ_{X,M1} − Δ_{X,M1,M2}
- Per-seed point estimates → sample covariance → delta-method SE
- t_{0.975, N-1} CIs (preferred over BCa at N=5 per arXiv:2508.10083)
- Implementation: `src/bin/f2_dual_mediation.rs`

### 3.3 Sensitivity envelope
- Additive bound: expansion(Γ, Λ) = Λ · (Γ − 1) / Γ
- Tipping point: Γ_tip(Λ) = 1 + closer_endpoint / Λ
- Lambda-sweep: per-PSE × per-Λ table; long-form (CMAverse convention) or
  wide-form pivot
- Implementation: `src/bin/f2_mediation_sensitivity.rs`

### 3.4 Cross-stratum comparator
- Join by (fix_x, pse_name); NaN for missing strata (no silent drops)
- `stable_across_strata = true` iff every pair of present CIs overlaps
- Implementation: `src/bin/f2_stratum_compare.rs`

### 3.5 Provenance and reproducibility
- W3C-PROV / Workflow Run RO-Crate preamble (arXiv:2312.07852)
- `f2_provenance_check` validates schema + git SHA + timestamp
- TRAINER_INTERNALS_SCHEMA constant in `config_fingerprint` detects silent
  drift (e.g., LCG seed changes that flip outputs at identical config hash)

---

## 4. Sandbox ablation matrix

### 4.1 Setup
- 7 fixes: rms, warmup, gradclip, clamp, smooth, wd, dropout
- 5 seeds: [42, 43, 44, 45, 46]
- 200 steps, ~8K params, synthetic counter task (Gros 2025 IARC-Increment
  Eq. 1)
- All 3 strata: canonical (default), wd0 (Pearl CDE on WD), warmup0 (Pearl
  CDE on warmup)
- 80 cells × 5 seeds = 400 training runs per stratum (canonical, wd0,
  warmup0)

### 4.2 Why sandbox-scale
- The methodology is the contribution; the empirical demonstration only
  needs to be detectable, not generalizable
- 200-step sandbox runs in ~25 minutes for each stratum
- Champion-scale validation deferred to a separate pre-registered study
  (see `docs/F2_PRE_REG.md`)

---

## 5. Empirical results

### 5.1 Canonical ablation result (Loop 30 baseline)
- Suppression pattern: every non-mediator fix has +5 BPB indirect effect
  via WD, canceled by −4.5 BPB direct effect
- Net Δ_X ≈ 0 for every fix except rms (Δ_X = +0.87 BPB)
- Sums to total exactly — no residual interaction beyond the model

**Figure 3**: Canonical dual_mediation output: 5×4 PSE table with CIs.

### 5.2 wd0 stratum (Pearl CDE on WD)
- NIE_M1 via WD = 0 by construction (M1 is pinned)
- **Per-fix NDE shifts sign for some fixes**:
  - rms: −4.12 (canonical) → +0.43 (wd0) ← **headline finding**
  - dropout: −4.55 → −4.16 (no flip)
  - gradclip, clamp, smooth: similar magnitudes, no flips

**Figure 1** (headline): bar chart showing rms NDE across 3 strata with CIs.

### 5.3 Cross-stratum stability
- `f2_stratum_compare` on 3-stratum dataset: 4/20 PSEs `stable_across_strata`
- Most "unstable" PSEs are the M2 pathway (mediator differs across strata)
- **Stable cross-stratum**: NIE_M1 via rms (alternative parameterization) =
  −0.75 [−1.32, −0.18] across all 3 strata

### 5.4 Sensitivity
- Canonical NDE for rms (−4.12): Γ_tip(Λ=1.0) = 4.55 → robust per
  VanderWeele-Ding (Γ ≥ 2.0)
- wd0 CDE for rms (+0.43): Γ_tip(Λ=1.0) = 1.43 → moderate-to-fragile range
- Cross-stratum stable NIE (−0.75): Γ_tip ≈ 1.24 → fragile but stable across
  three independent estimates

**Figure 4**: tipping-point curves Γ_tip(Λ) for the four key estimates.

---

## 6. Sensitivity to choices

### 6.1 Mediator pair (M1, M2)
- Loop 50 replication with (M1=rms, M2=warmup) gives the same qualitative
  picture, with the cross-stratum-stable −0.75 BPB NIE_M1
- Sign flip survives parameterization swap

### 6.2 Statistic family
- We use t-CI (Owen 2025) and exact Fisher-Pitman permutation
  (arXiv:2205.01416). Bootstrap-t and BCa give similar but slightly looser
  bounds at N=5
- Sample size N=5 is small; results meant as proof-of-concept, not
  population estimates

### 6.3 Strata
- We chose Wd0 + Warmup0 based on prior mediation analyses identifying these
  as candidate mediators. Other strata (e.g., LabelSmoothing0, ClampZero)
  are pre-defined in `Stratum::ALL` but not run here (see `src/race/ablation.rs`
  doc comment for selection criteria)

---

## 7. Limitations

1. **Sandbox-scale only**: 200 steps × 8K params is a stress test for the
   methodology, not a champion-scale claim. See `docs/F2_PRE_REG.md` for
   the pre-registered champion-scale follow-up.
2. **N=5**: minimum-detectable effect at this N is ~0.1 BPB for paired
   permutation. The +0.43 CDE survives by a narrow margin.
3. **Linearity / no-XM-interaction**: Zhao-Luo's identification rests on
   sequential ignorability and no exposure-mediator interaction. We test
   the no-interaction assumption empirically (Loop 34 lock test:
   `dual_mediation_no_interaction_residual_lock`); residual is < 1e-6 in
   our regime.
4. **Synthetic task**: counter task is an analytical-tractability choice.
   Real BPB on FineWeb is in the pre-registered follow-up.

---

## 8. Software

- Rust 1.82, MIT-licensed, ~726 tests
- 11 binaries documented in `docs/F2_BINARIES.md`
- W3C-PROV preambles for reproducibility
- Anchored at commit hash 19d032e (this PR)

---

## 9. Related work

### 9.1 ML ablation methodology
- ABLATOR (PMLR 2023, Fostiropoulos et al.)
- AblationBench (arXiv:2507.08038)
- Reproducibility in ML (arXiv:2302.04054, Semmelrock 2025)

### 9.2 Causal mediation
- Zhao & Luo (2020), arXiv:2007.16031
- Miles & Shpitser (2017), arXiv:1710.02011
- DoWhy (arXiv:2011.04216), CMAverse (R package)

### 9.3 Sensitivity analysis
- VanderWeele & Ding (2017): E-value
- Ohnishi & Li (2026): bridge-score Theorem 2 (arXiv:2605.18724)
- Alvarez-Bartolo & MacKinnon (2025): tipping-point curves for mediation

### 9.4 Quantization (only as motivation for our ablation study)
- BitNet b1.58 2B4T (arXiv:2402.17764)
- Quantization scaling laws (arXiv:2502.05003)
- The original phi-ladder vs format-zoo question (Issue #1021) is the
  motivating ablation, deferred to champion-scale follow-up

---

## 10. Conclusion + venue calibration

### 10.1 Conclusion
We argue that stratified CDE analysis with sensitivity envelopes is
ready for routine use in ML training-recipe ablation studies. The
methodology costs ~25 minutes of compute per stratum at sandbox scale
and surfaces qualitative effects that vanilla seed-mean comparison
misses. The RmsNorm sign-flip is a single empirical demonstration; we
expect more such findings as the framework sees broader use.

### 10.2 Venue calibration
- **NeurIPS 2026 ML Reproducibility Workshop** (deadline TBD, typically
  Sep-Oct): tight fit, our W3C-PROV + 726 tests + commit-anchored claim
  table is exactly their target
- **NeurIPS 2026 Causal-ML Workshop** (deadline TBD, typically Nov):
  also good fit, more theoretical audience
- **ICML 2027 main track**: requires champion-scale follow-up; current
  paper is workshop-grade

### 10.3 Acknowledgments + funding disclosure
[blank — fill at submission time]

---

## Appendices

### A. Reproducible commands
[Mirror of `docs/F2_RMS_CDE.md` § Reproduce; expand with per-figure command
sequences]

### B. Provenance preamble format
[Mirror of `docs/F2_BINARIES.md` § Cross-binary contract]

### C. Code-to-paper crosswalk
| Section | File | Function/binary |
|---|---|---|
| §3.1 | `src/race/ablation.rs` | `Stratum`, `mode_string` |
| §3.2 | `src/bin/f2_dual_mediation.rs` | `compute_dual_mediation` |
| §3.3 | `src/bin/f2_mediation_sensitivity.rs` | `envelope_expansion`, `tipping_point_gamma` |
| §3.4 | `src/bin/f2_stratum_compare.rs` | `build_comparison`, `cis_overlap` |
| §5.2 | `docs/F2_RMS_CDE.md` | Empirical reproduction |

### D. Test inventory
[List the 726 tests by suite + 6-line summary of each test's purpose]

---

## Author notes for self (delete before submission)

- Headline finding: RmsNorm NDE sign flip across strata
- Sub-finding: NIE_M1 via rms stable across strata (more robust)
- 1-figure pitch: bar chart of rms NDE across 3 strata with CIs
- 1-sentence pitch: "Seed-mean ablation in ML systematically misattributes
  effects when one intervention mediates another; we propose stratified
  CDE analysis and demonstrate a sign flip in the standard RmsNorm
  ablation."
- Estimated paper size: 8 pages workshop, 4 figures, ~15 references
- Anchor commit: 19d032e (f2-methodology branch)
- Empirical replication: also include 47/49/50 sweep CSVs in supplementary

## Next steps to graduate this outline

1. Decide on workshop target (NeurIPS Repro vs Causal-ML) — affects framing
2. Run champion-scale validation per `docs/F2_PRE_REG.md` (optional but
   strengthens claim)
3. Generate the 4 figures (use Loop 49/50 data; matplotlib via f2_to_jsonl)
4. Polish §3 (currently terse) and write §6.2 (sensitivity to statistic
   choice) with more numerical detail
5. Adversarial-review the paper against the checklist in `docs/F2_PRE_REG.md` §10
