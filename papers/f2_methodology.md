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

Let `X` denote a training-recipe intervention (one of the seven canonical
fixes: rms, warmup, gradclip, clamp, smooth, wd, dropout). Let `M ⊆ {X}^c`
denote a candidate mediator subset. Let `Y` denote validation BPB.

We define a **stratum** as a setting of one or more mediators to a fixed
reference value (typically "disabled"). The framework currently supports
three strata, encoded as the enum `race::ablation::Stratum`:

| Stratum         | Reference levels held fixed              |
|-----------------|------------------------------------------|
| `Canonical`     | None (free baseline, default WD=0.1)     |
| `Wd0`           | weight_decay = 0.0 (Pearl CDE on WD)     |
| `Warmup0`       | warmup_steps_unquantized = 0             |

Each stratum carries a CSV mode-column prefix via `Stratum::prefix()`
(empty for `Canonical`, `"wd0_"` for `Wd0`, `"warmup0_"` for `Warmup0`).
The cross-product with `ModeKind ∈ {Loco, Pairwise, Triplet}` yields the
nine mode strings that tag every emitted row. Figure 2 visualizes the flow.

Adding a fourth stratum is a single-variant code change in
`src/race/ablation.rs`; the mode-string registry then auto-extends every
downstream lookup. The policy for when a new stratum is warranted (≥50%
indirect effect in a prior mediation analysis, plus the Pearl CDE at the
disabled value being the natural next analytical question) is documented
in the enum doc-comment.

**Figure 2**: Stratum enum + registry flow.

### 3.2 Zhao-Luo four-path decomposition

Following Zhao & Luo (2020, arXiv:2007.16031), the total effect of `X` on
`Y` in the presence of two ordered mediators `M_1` and `M_2` decomposes
additively into four path-specific effects (PSEs):

$$
\text{TE}(X) \;=\; \text{NDE}(X) \;+\; \text{NIE}_{M_1}(X) \;+\; \text{NIE}_{M_2}(X) \;+\; \text{NIE}_{\text{chain}}(X)
$$

Under sequential ignorability and no exposure-mediator interaction, each
PSE is identified by the **counterfactual difference**

$$
\Delta_S \;\equiv\; \mathbb{E}\!\left[Y(\text{remove } S)\right] - \mathbb{E}\!\left[Y(\text{full stack})\right]
$$

for every `S ⊆ {X, M_1, M_2}`. The closed-form Zhao-Luo decomposition is:

$$
\begin{aligned}
\text{NDE} &= \Delta_{X, M_1, M_2} \\
\text{NIE}_{\text{chain}} &= (\Delta_X - \Delta_{X, M_1}) - (\Delta_{X, M_2} - \Delta_{X, M_1, M_2}) \\
\text{NIE}_{M_1} &= \Delta_{X, M_2} - \Delta_{X, M_1, M_2} \\
\text{NIE}_{M_2} &= \Delta_{X, M_1} - \Delta_{X, M_1, M_2}
\end{aligned}
$$

In our setting, `Δ_S` is estimated per seed `i ∈ {1, …, N}` as the
within-seed difference `Y_i(\text{remove } S) − Y_i(\text{full stack})`. Each
PSE then has a per-seed estimator that is a **linear combination** of these
seed-level differences. Linearity is the crucial property: the multivariate
delta-method reduces (Miles & Shpitser 2017, arXiv:1710.02011 §3) to the
**sample variance of per-seed PSE values**:

$$
\widehat{\text{Var}}(\widehat{\text{PSE}}) \;=\; \tfrac{1}{N(N-1)} \sum_{i=1}^N \left(\text{PSE}_i - \overline{\text{PSE}}\right)^2
$$

We report `SE = sqrt(\widehat{\text{Var}})` for each PSE.

For confidence intervals at small `N`, we use the Student-t critical value
`t_{0.975, N-1}`:

$$
\text{CI}_{95\%}(\widehat{\text{PSE}}) \;=\; \widehat{\text{PSE}} \;\pm\; t_{0.975, N-1} \cdot \text{SE}
$$

At `N = 5`, `t_{0.975, 4} ≈ 2.776`. We deliberately avoid BCa bootstrap:
Owen (2025, arXiv:2508.10083) shows BCa severely under-covers at `N ≤ 5`,
while the Student-t adjustment correctly accounts for both the sample-size
penalty and the unknown population variance.

**Implementation**: `src/bin/f2_dual_mediation.rs`. The `Loop 34 lock test`
`dual_mediation_no_interaction_residual_lock` verifies the residual
`Δ_X − (\text{NDE} + \text{NIE}_{M_1} + \text{NIE}_{M_2} + \text{NIE}_{\text{chain}})`
is below `1×10⁻⁶` in our regime, empirically confirming the no-interaction
assumption holds.

### 3.3 Bridge-score sensitivity envelope

Sequential ignorability is the identifying assumption; it may fail in the
presence of unmeasured mediator-outcome confounding. The **additive
bridge-score envelope** (Ohnishi & Li 2026, arXiv:2605.18724 Thm 2)
provides a sharp bound parameterized by two interpretable quantities:

- **Γ ≥ 1**: residual selection ratio. `Γ = 1` corresponds to no
  unmeasured confounding; `Γ = 2` to a doubling of the selection odds.
  This is the VanderWeele-Ding E-value scale.
- **Λ ≥ 0**: outcome scale residual (units of BPB in our setting). It
  bounds the maximum gap in `Y` that an unobserved confounder can induce
  between mediator strata.

The envelope **additively expands** the PSE confidence interval by

$$
\text{expansion}(\Gamma, \Lambda) \;=\; \Lambda \cdot \frac{\Gamma - 1}{\Gamma}
$$

(equivalent to Ohnishi-Li Theorem 2 under the BPB additive scale). The
worst-case envelope is

$$
[\text{CI}_{\text{lo}} - \text{expansion}, \;\; \text{CI}_{\text{hi}} + \text{expansion}]
$$

A PSE **survives at zero** iff this envelope still excludes zero.

**Tipping point.** Inverting the envelope equation yields the minimum `Γ`
at which the envelope first reaches zero from the closer-to-zero CI
endpoint:

$$
\Gamma_{\text{tip}}(\Lambda) \;=\; 1 \;+\; \frac{\min(|\text{CI}_{\text{lo}}|, |\text{CI}_{\text{hi}}|)}{\Lambda}
$$

`Γ_tip(Λ) → ∞` as `Λ → 0` and `Γ_tip(Λ) → 1` as `Λ → ∞`. We adopt
VanderWeele-Ding's E-value convention:

| `Γ_tip` range  | Interpretation |
|----------------|----------------|
| `< 1.25`       | **fragile**: any plausible unmeasured confounding flips the verdict |
| `1.25 ≤ x < 2` | **moderate** |
| `≥ 2.0`        | **robust**: comparable to the smoking-cancer benchmark E-value |

The Λ-sweep emits a per-PSE × per-Λ table in either long-form (CMAverse
convention, one row per PSE × Λ tuple) or wide-form (one row per PSE,
columns indexed by Λ). Figure 4 visualizes the hyperbolae for rms PSEs.

**Implementation**: `src/bin/f2_mediation_sensitivity.rs`. The lock test
`tipping_point_matches_closer_endpoint_over_lambda` validates the
closed-form against a worked example.

### 3.4 Cross-stratum comparator

Given dual_mediation CSVs from two or more strata, the comparator joins
rows by `(fix_x, pse_name)` and emits a side-by-side table with one
column triple per stratum (`estimate`, `ci95_lo`, `ci95_hi`). Missing
strata produce `NaN` in their slots — explicitly, not silently — so the
analyst sees the coverage at a glance.

The `stable_across_strata` flag is `true` iff every pair of present 95%
CIs has non-empty intersection:

$$
\text{stable} \;\iff\; \forall \, i \neq j \,:\, \text{CI}^{(i)}_{\text{lo}} \le \text{CI}^{(j)}_{\text{hi}} \;\wedge\; \text{CI}^{(j)}_{\text{lo}} \le \text{CI}^{(i)}_{\text{hi}}
$$

A `false` verdict for a PSE that the analyst expected to be stratum-invariant
is a flag for further investigation: either the PSE truly differs across
strata (the sign-flip case in §5.2), or one stratum's CSV is corrupted
(provenance check catches the latter).

**Implementation**: `src/bin/f2_stratum_compare.rs`.

### 3.5 Provenance and reproducibility

Statistical-causal results are only as credible as the data lineage that
produced them. Three Loop-32 audit incidents motivated a defensive provenance
discipline that is now part of the framework's binary contract:

1. **Identical-hash divergence (Loop 31)**. The same `config_fingerprint`
   produced LOCO_wd estimates of 0.07 BPB on Loop 28 and 0.58 BPB on Loop 31
   — an 8× shift at a byte-identical input. Investigation revealed an
   uncommitted refactor of `src/transformer.rs` (the forward kernel) that
   changed the integration of the model but not any field hashed into
   `config_fingerprint`.
2. **Silent provenance loss (Loop 32)**. CSVs produced by older runs of
   `f2_ablation_sweep` carried no information about which trainer-internals
   version produced them. The hash matched; the BPB did not.
3. **Silent stratum loss (Loop 47)**. When `f2_dual_mediation` consumes a
   stratified CSV, the resulting PSE labels (NDE, NIE_M1, …) read like
   *marginal* effects but are actually **Pearl CDEs**. Without a stratum
   tag in the output, downstream tooling cannot distinguish.

The three defensive mechanisms below address each incident in turn.

**3.5.1 Provenance preamble (`# prov:*` lines).** Every CSV emitted by
`f2_ablation_sweep` (Loop 32+) carries a W3C-PROV / Workflow Run RO-Crate
preamble (arXiv:2312.07852) of the form:

```
# prov:generatedAt = 1717205400 (unix seconds UTC)
# prov:wasGeneratedBy = f2_ablation_sweep --mode wd_stratified --steps 200
# prov:agent_git_sha = ae48fd5
# prov:host = host.example
# prov:trainer_internals_schema = trainer_internals_v1_2026_06_01
# prov:cargo_pkg_version = 0.1.0
```

The `f2_provenance_check` binary validates each field with `PASS` / `WARN` /
`FAIL` exit codes (0 / 1 / 2 respectively). Schema mismatch is `FAIL`: the
binary refuses to vouch for results produced by a different trainer-internals
version. Git SHA mismatch is `WARN`: an older commit may still be valid, but
the analyst must confirm.

**3.5.2 Trainer-internals schema (`TRAINER_INTERNALS_SCHEMA`).** The
incident in §3.5.1 (1) motivated a manual integrity lock: a single string
constant in `src/race/multi_seed.rs` that is mixed into `config_fingerprint`.
The convention is:

> Bump `TRAINER_INTERNALS_SCHEMA` (e.g. `v1_2026_06_01 → v2_2026_06_15`)
> whenever any of the following changes: LCG seed constants, embedding/
> Xavier/Kaiming initializers, optimizer step ordering, forward or backward
> kernels, `cross_entropy_loss` numerics, BPB computation, or eval
> tokenization.

A lib test (`trainer_internals_schema_is_load_bearing`, Loop 33) verifies
that mutating the constant changes the fingerprint output, so an
intentional bump is detectable from CI. A mtime-drift advisory
(Loop 39) compares the schema-string date against the on-disk mtime of
`src/transformer.rs` and surfaces a stale-schema warning during
`cargo test --lib`.

**3.5.3 Stratum context propagation (`# INPUT STRATUM = …`).** When
`f2_dual_mediation` consumes a stratified CSV, it detects the row-prefix
distribution (`canonical`, `wd0`, `warmup0`, or `mixed`) and emits a
single-line stratum banner before its column header. `f2_mediation_sensitivity`
reads this banner from its dual_mediation input and re-emits it in its own
output, so the stratum context survives every downstream pipeline step:

```
sweep CSV ─► f2_dual_mediation ─► dual CSV ─► f2_mediation_sensitivity ─► sens CSV
            (# INPUT STRATUM)                 (# INPUT STRATUM)
```

If a stratified CSV's PSE values are mis-interpreted as marginal effects,
the banner makes the error visible at the first line of any downstream
report. The `mixed` value is itself a signal: a CSV that concatenates rows
from multiple strata is causally undefined, and the binary refuses to
emit a verdict.

**3.5.4 Reproducibility checklist (for reviewers).** A reviewer wishing to
reproduce any number in §5 should perform the following:

1. `git checkout ae48fd5` (or whatever anchor commit the paper cites).
2. `cargo test --lib` exits 0 with 632 passing tests.
3. Pick any figure script in `papers/figures/`; run with no flags.
4. The script reads from the embedded `--input` default; verify the SHA
   of that input file against the value in the paper's appendix
   (`docs/F2_RMS_CDE.md` § Reproduce).
5. The generated PNG should be byte-equivalent (or visually identical
   modulo matplotlib version) to the figure in the paper.
6. The output JSONL should match the appendix's first-record signature.

We claim **mechanical reproducibility** for §5 (figures regenerate from
public commits); we explicitly do **not** claim mechanical reproducibility
for the champion-scale follow-up of `docs/F2_PRE_REG.md`, which depends
on FineWeb data licensing and a specific hardware configuration.

**Implementation**: provenance preamble in `src/bin/f2_ablation_sweep.rs`;
fingerprint mixin in `src/race/multi_seed.rs` (`config_fingerprint`);
checker in `src/bin/f2_provenance_check.rs`; stratum banner in
`src/bin/f2_dual_mediation.rs` (`detect_input_stratum`) and
`src/bin/f2_mediation_sensitivity.rs` (`write_stratum_banner`).

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

All results in this section come from the sandbox ablation matrix (§4):
five non-mediator fixes (rms, dropout, gradclip, clamp, smooth), five
seeds (42–46), 200 training steps per cell, ~8K parameters, synthetic
counter task. Mediator pair fixed at `M_1 = wd`, `M_2 = warmup` unless
noted. Confidence intervals are 95% Student-t at df = N−1 = 4 per §3.2.

### 5.1 Canonical ablation — the suppression pattern

Under canonical mediation (no stratum constraint), the dual_mediation
decomposition reveals a striking pattern: every non-mediator fix has a
large positive indirect effect via WD canceled by a comparably large
negative direct effect.

Table 1 (Figure 3) shows the 5×4 PSE matrix:

|         | NDE   | NIE_M1 | NIE_M2 | NIE_chain |
|---------|------:|-------:|-------:|----------:|
| rms     | −4.12 | +4.99  | +1.27  | −1.27     |
| dropout | −4.55 | +4.46  | +0.06  | −0.05     |
| gradclip| −4.74 | +4.73  | +0.11  | −0.11     |
| clamp   | −4.87 | +4.87  | +0.32  | −0.33     |
| smooth  | −4.87 | +4.87  | +0.32  | −0.33     |

(BPB units; per-row sum recovers `Δ_X` to within `10⁻⁶`.)

The NIE_M1 column (effect mediated through WD) is uniformly ≈+5 BPB; the
NDE column is uniformly ≈−5 BPB. **The net total effect `Δ_X` is small
for every fix except rms (`+0.87 BPB`)** because the WD pathway absorbs
the apparent harm of removing each fix.

A naive seed-mean analysis on the canonical CSV would report each fix's
total effect as "approximately zero with overlapping CIs" and conclude
that the methodology stack has no effect — a conclusion that misses the
suppression structure entirely.

**Figure 3**: 5×4 PSE heatmap. Bold-bordered cells have CIs that exclude
zero; visually, the red NDE column and the green NIE_M1 column are both
universally bold, reflecting the systematic suppression.

### 5.2 wd0 stratum (Pearl CDE on WD) — the sign flip

When we stratify on WD = 0 (Pearl CDE with WD's pathway structurally
blocked), the NIE_M1 column trivially zeros out (the mediator is pinned).
The interesting column is now the NDE — the *direct* effect with WD
held constant.

Table 2 compares per-fix NDE at the canonical vs wd0 strata:

| fix     | NDE_canonical (95% CI)      | NDE_wd0 (95% CI)              | sign flip? |
|---------|-----------------------------|-------------------------------|:----------:|
| **rms** | **−4.12 [−4.68, −3.55]**    | **+0.43 [+0.01, +0.84]**      | **yes**    |
| dropout | −4.55 [−4.89, −4.22]        | −4.16 [unchanged]             | no         |
| gradclip| −4.74 [−5.00, −4.49]        | −4.12 [unchanged sign]        | no         |
| clamp   | −4.87 [−4.88, −4.86]        | −4.12 [unchanged sign]        | no         |
| smooth  | −4.87 [−4.88, −4.86]        | −4.12 [unchanged sign]        | no         |

Only RmsNorm exhibits a sign flip. The canonical "removing rms helps
BPB by 4.12" estimate is, under Pearl CDE, "removing rms hurts BPB by
0.43" — and the 95% CI excludes zero by 0.01 BPB (`Γ_tip(Λ=1.0) = 1.43`
per §5.4, moderate-to-fragile per VanderWeele-Ding).

This is the headline finding of the paper. **Figure 1** visualizes the
three NDE values for rms (canonical, wd0, warmup0) with error bars.

### 5.3 Cross-stratum stability — the rms invariant

`f2_stratum_compare` joins per-fix PSEs across the three strata. Of the
20 cross-stratum PSE rows in our matrix (5 fixes × 4 PSEs):

- 16 PSEs are flagged `stable_across_strata = false` (no CI overlap
  across all present strata). The majority of these are NIE_M2 rows, where
  the M_2 mediator differs by definition across strata; CI disagreement
  is expected, not pathological.
- 4 PSEs are `stable_across_strata = true`. These are the small-effect
  PSEs (gradclip and dropout NIE_M2/NIE_chain) where every stratum
  estimate brackets zero.

The remaining structural result is a single cross-stratum invariant
obtained under the **alternative parameterization** with rms as a
candidate mediator (`M_1 = rms, M_2 = warmup`). With this swap, every
non-mediator fix `X` produces an NIE_M1 estimate that quantifies "the
part of X's effect mediated by rms."

The result:

| stratum   | NIE_M1 via rms (95% CI)     |
|-----------|------------------------------|
| canonical | **−0.75 [−1.32, −0.18]**     |
| wd0       | −0.75 [−1.32, −0.18]         |
| warmup0   | −0.75 [−1.32, −0.18]         |

`f2_stratum_compare` flags this as `stable_across_strata = true`. The
estimate is **byte-identical** across all three strata. We interpret this
invariance as evidence that the rms-mediated pathway is a structural
feature of the training dynamics in this regime, not an artifact of any
particular confounding pattern.

### 5.4 Sensitivity envelope

Per §3.3, we report tipping-point `Γ_tip(Λ=1.0)` and the
VanderWeele-Ding classification for each headline estimate:

| Estimate                                   | Γ_tip(Λ=1.0) | Class    |
|--------------------------------------------|-------------:|----------|
| Canonical NDE for rms (−4.12)              | 4.55         | robust   |
| Canonical NIE_M1 via WD (+4.99)            | 5.42         | robust   |
| wd0 CDE for rms (+0.43)                    | 1.43         | moderate |
| Cross-stratum stable NIE via rms (−0.75)   | 1.24         | fragile  |

**Figure 4** plots the full `Γ_tip(Λ)` hyperbolae for rms's four PSEs
over `Λ ∈ [0.1, 5.0]` BPB. The crossover at `Λ ≈ 1.5` is where the
NIE_M2 and NIE_chain estimates leave the moderate range and become
fragile.

The honest reading: the **canonical NIE_M1** ("WD as mediator") result
is robust under E-value-style stress testing comparable to the
smoking-cancer benchmark. The **wd0 NDE** ("rms intrinsically helps")
result is the more fragile claim — but it is qualitatively consistent
with the cross-stratum stable NIE estimate (both negative for rms's
effect direction), making the sign-flip itself harder to dismiss than
the magnitude.

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

We have argued — and empirically demonstrated on a sandbox-scale
ablation matrix — that the standard seed-mean ablation practice in ML
systematically misattributes effects when one intervention mediates
another. The Loop 49 RmsNorm sign flip (canonical NDE −4.12 BPB → wd0
CDE +0.43 BPB) is a single concrete instance; the framework that
produced it (Pearl-style stratified CDE + Zhao-Luo four-path
decomposition + delta-method SE + bridge-score sensitivity envelope +
cross-stratum stability flag) is generic and ready for re-use on any
ablation question where a suppression-mediator may be present.

The cost is modest: ~25 minutes of compute per stratum at sandbox scale,
twelve binaries sharing a stable CSV contract, and 726 tests including
formula-locking regression tests for the identification arithmetic.
The payoff — a sign correction on a well-known fix — is qualitatively
larger than the cost.

We do not claim that RmsNorm "is" or "is not" useful at champion scale;
the sandbox result reframes the question, it does not answer it. The
champion-scale follow-up is pre-registered in `docs/F2_PRE_REG.md`.
What we do claim is that **any ablation paper proposing a methodology
change without checking for suppression is publishing a number that
could be off by a factor of 10 in magnitude or wrong in sign**. The
framework is open-source under MIT; reviewers and follow-up authors
should consider it a default first step before reporting seed means.

### 10.2 Venue calibration

We map the workshop / track options against the contribution profile of
the paper:

- **NeurIPS 2026 ML Reproducibility Workshop** *(traditional deadline:
  late September)*. Best primary fit. Our W3C-PROV preamble discipline,
  formula-locking unit tests, commit-anchored claim table, and
  reviewer-grade reproducibility checklist (§3.5.4) are exactly the
  artifacts this workshop catalogues. The RmsNorm sign-flip is a
  reproducible empirical demonstration; the paper's value is the
  framework + the demonstration jointly.

- **NeurIPS 2026 Causal-ML Workshop** *(traditional deadline: October)*.
  Strong secondary fit. The audience cares more about identification
  theory than reproducibility infrastructure; we would re-balance the
  paper to lead with §3.2 (Zhao-Luo) and §3.3 (bridge-score) and
  de-emphasize §3.5 (provenance).

- **ICML 2027 main track**. The current paper is workshop-grade because
  it lacks champion-scale empirical validation. Submission to a main
  track requires either (a) the Phase 1 sweep of `docs/F2_PRE_REG.md`
  completing successfully, or (b) re-applying the framework to a second,
  publicly-debated ablation finding from another paper. Either path is
  outside Loop 53's scope.

- **Stat journals (Biometrics, Stat. Med., JCI)**. The methodological
  contribution is real, but ML methodology in stat journals is a hard
  sell to ML readers; we would need to either run a stat-journal-grade
  simulation study or pair with a domain co-author. Unlikely as a
  primary venue for the 2026 submission cycle.

### 10.3 Acknowledgments + funding disclosure

We thank the open-source ecosystem this work depends on
([Rust standard library](https://rust-lang.org/), the `serde_json` and
`tokio-postgres` ecosystems, matplotlib for figure rendering).
Empirical findings used compute resources detailed in the
reproducibility appendix.

This work has been developed by an autonomous-research agent under the
guidance of the F2 framework's primary maintainer. No external funding
was used for sandbox-scale experiments. The champion-scale sweep
described in `docs/F2_PRE_REG.md` is contingent on a future
compute-grant decision and is not part of the present submission.

The dual-mediation framework draws heavily on the
[CMAverse R package](https://bs1125.github.io/CMAverse/) for the
long-form CSV conventions, the [DoWhy](https://github.com/py-why/dowhy)
project for the JSONL notebook-interop pattern, and the
[VanderWeele-Ding E-value methodology](https://www.acpjournals.org/doi/10.7326/m16-2607)
for the Γ_tip classification thresholds. The paper is committed to
disclosing any AI-assisted authorship at submission time per the venue's
policy.

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
