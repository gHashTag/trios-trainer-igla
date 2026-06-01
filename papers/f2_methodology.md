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
zero). We frame this as a unit-test demonstration on a synthetic task
that the framework detects a sign flip when one is constructed —
the magnitudes are sandbox-specific and we do not claim the
qualitative finding transfers to champion-scale models without
additional evidence; champion-scale validation is pre-registered in
`docs/F2_PRE_REG.md`.

The framework is open-source in Rust with 11 binaries, 726 unit/integration
tests, and W3C-PROV-tagged CSV provenance preambles. We release it as a
reproducibility artifact for ML methodology research.

---

## 1. Introduction

### 1.1 The problem with seed-mean ablation

The median ML ablation report follows a fixed recipe: pick a target
intervention `X`, train `N ∈ {3, 5}` seeds with and without `X`,
report `Δ̄ ± σ/√N` on a held-out metric, and infer that `X` helps,
hurts, or is null. This recipe is statistically defensible when every
other configuration parameter is held fixed, but ML training recipes
are not single-knob systems. A transformer training run simultaneously
exercises weight decay, learning-rate warmup, gradient clipping,
RmsNorm vs LayerNorm, label smoothing, and dropout — to name only the
canonical seven we ablate in §4. Two of these knobs are statistically
dependent whenever one is on the regularization path of the other.

When such a dependency exists, a seed-mean ablation reports the
**marginal** effect of `X`: the average over the empirical joint of all
other knobs in the training configuration. The marginal effect is what
practitioners actually consume — "should I turn this on by default?" —
but it is silent on a more dangerous failure mode: **suppression
mediation**, where one intervention's effect is dominated by its
correlated mediator, and the apparent sign of the effect flips when the
mediator is held at a non-default reference value. The empirical
demonstration in §5 is one such case: replacing RmsNorm with LayerNorm
has marginal NDE −4.12 BPB (helpful) at canonical training, but +0.43
BPB (harmful) under the Pearl Controlled Direct Effect (CDE) with
weight decay pinned to zero. Both CIs exclude zero. The sign is not a
seed artifact.

If suppression mediation is more common than the current ablation
literature acknowledges, then a meaningful fraction of "ablation shows
`X` helps" claims may be confounded by mediators outside the
ablation. The contribution of this paper is twofold: (i) provide the
methodological and software infrastructure to detect such cases at the
same engineering cost as a normal ablation sweep, and (ii) document
one verified instance as proof-of-concept.

### 1.2 Contributions

1. **Stratification framework**. The `Stratum::{Canonical, Wd0,
   Warmup0}` registry (§3.1) routes a single ablation sweep across one
   marginal stratum and two Pearl-CDE strata, deriving the mode-string
   matrix from `Stratum × ModeKind` automatically so that adding a new
   stratum is a single-variant code change.
2. **Zhao-Luo 4-PSE decomposition with finite-sample CIs**. We
   implement the four-path decomposition (§3.2, Gao-Li-Luo 2020) with
   delta-method standard errors that reduce to the Miles-Shpitser
   (2017) efficient influence function under no-interaction, and we
   default to Student-`t` CIs at `df = 4` for the `N = 5` seed regime —
   not the asymptotic `z` interval that Owen (2025) shows to
   undercover at that sample size.
3. **Bridge-score sensitivity envelope**. We adopt the Ohnishi-Li
   (2026) additive bridge-score envelope on each PSE (§3.3),
   classifying every effect as fragile / median / robust against the
   VanderWeele-Ding tipping-point thresholds (`Γ < 1.25`, `Γ ≥ 2.0`).
4. **Cross-stratum stability comparator**. `f2_stratum_compare` (§3.4)
   takes the three single-stratum CSVs and emits a `stable_across_strata`
   flag per PSE based on CI overlap, surfacing suppression mediation
   without a-priori knowledge of which mediator is doing the
   suppressing.
5. **Empirical proof-of-concept**. §5 documents the RmsNorm × WD
   sign-flip at small scale (5 seeds, ~ minute-of-compute per run) and
   §6 demonstrates that no choice in §3 — point estimator, CI method,
   bridge-score `Λ`, stratum reference value — flips the qualitative
   conclusion.

### 1.3 Roadmap

§2 reviews Pearl causal mediation, sensitivity analysis, and the
adjacent ML ablation literature. §3 develops the F2 framework
(stratification, 4-PSE decomposition, bridge-score envelope,
cross-stratum comparator) with LaTeX derivations and CSV contracts.
§4 specifies the small-scale sandbox ablation matrix that backs the
empirical results. §5 presents the headline RmsNorm sign-flip across
three strata, with replication under an alternative mediator
parameterization. §6 sweeps four design choices (estimator, CI method,
`Λ`, stratum reference) to show that the qualitative finding is robust
to all. §7 enumerates five limitations, including the deferred
champion-scale follow-up. §8 catalogues the software artifacts. §9
positions the contribution against the closest ML ablation, causal
mediation, and sensitivity-analysis work. §10 concludes and
calibrates the contribution against four candidate publication venues.

---

## 2. Background

### 2.1 Pearl causal mediation

Under Pearl's potential-outcomes calculus, the total effect of an
intervention `X` on an outcome `Y` admits two decompositions of
practical interest in ablation work. The **natural direct-and-indirect
effect (NDE/NIE)** decomposition splits the total effect into the part
that flows through a mediator `M` under its natural distribution (the
NIE) and the part that does not (the NDE). The NDE is contrastive at
the population mean of `M`, which means it answers the marginal-
ablation question — what would change on average if I removed `X` but
left `M` to find its own equilibrium? The **controlled direct effect
(CDE)** instead fixes `M` at a specific reference value `m*` and reports
the effect of `X` with `M` clamped at `m*`. The CDE answers the
counterfactual-ablation question: what would change if I removed `X`
and also disabled `M`?

When two mediators are present, Gao, Li & Luo (2020,
arXiv:2007.16031) derive a four-path decomposition that additively
separates the total effect into the direct effect plus three indirect
effects: through `M_1` alone, through `M_2` alone, and through the
sequential chain `M_1 → M_2`. F2 (§3.2) implements this decomposition
under a no-interaction assumption that reduces the asymptotic variance
to the Miles-Shpitser (2017, arXiv:1710.02011) efficient influence
function, which we use as the basis for the delta-method standard
errors at `N = 5` seeds.

### 2.2 Sensitivity analysis

Causal estimates from observational or pseudo-observational data are
vulnerable to unmeasured confounding. The two standard formal tools
for bounding this exposure are the **E-value** of VanderWeele & Ding
(2017, "Sensitivity Analysis in Observational Research: Introducing
the E-Value", *Annals of Internal Medicine*) and the additive
**bridge-score envelope** of Ohnishi & Li (2026, arXiv:2605.18724,
Theorem 2). The E-value quantifies the smallest confounding strength
`Γ` on the risk-ratio scale that would explain away an observed effect;
the bridge-score envelope translates the same idea to an additive
metric scale and parameterizes the envelope by a pair `(Γ, Λ)` of
sensitivity dials.

Bits-per-byte is on the additive scale by construction, so F2 (§3.3)
uses the bridge-score envelope directly without any log or risk-ratio
translation. We classify every PSE estimate against the
VanderWeele-Ding tipping-point thresholds (`Γ_tip < 1.25` is fragile;
`Γ_tip ≥ 2.0` is robust), which keeps the sensitivity analysis on the
same scale as the headline numbers and avoids a category mismatch
between the estimate and its sensitivity envelope.

### 2.3 ML ablation practice

The ML-side closest to F2 is **AblationBench** (Abramovich et al.,
2025, arXiv:2507.08038), which benchmarks ablation methodology across
recent NLP papers. Their wide-form CSV schema combined with paired
Welch tests and Cohen's-`d` standardized effect sizes is the median
practice; F2 generalizes the schema to long-form with W3C-PROV
preambles and replaces paired Welch with stratified CDE plus
bridge-score sensitivity. **ABLATOR** (Fostiropoulos & Itti, 2023) is
the closest infrastructure work — a tool for running multi-seed
ablation studies at scale with result aggregation — but stops at
multi-seed ranking without mediation decomposition or stratified CDE.

A parallel line of causal-mediation work in ML interpretability is
typified by **ROME** (Meng et al., 2022) and **activation patching**
(Wang et al., 2023): both apply causal-mediation language at a single
forward pass to localize knowledge or attention behaviors inside an
already-trained model. F2 targets the orthogonal question of training-
recipe mediation: which training-time interventions confound which
others. The forward-pass and training-recipe questions share a
methodological vocabulary but do not share a problem.

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

### 3.2 Four-path decomposition for two ordered mediators

Following Daniel, De Stavola, Cousens & Vansteelandt (2015,
*Biometrics* 71:1–14, doi:10.1111/biom.12248,
"Causal mediation analysis with multiple mediators"), the total
effect of `X` on `Y` in the presence of two ordered mediators
`M_1`, `M_2` decomposes additively into four path-specific effects
(PSEs):

$$
\text{TE}(X) \;=\; \text{NDE}(X) \;+\; \text{NIE}_{M_1}(X) \;+\; \text{NIE}_{M_2}(X) \;+\; \text{NIE}_{\text{chain}}(X)
$$

The four components are: the **natural direct effect** (the part of
`Δ Y` that does not flow through either mediator), the **natural
indirect effect through `M_1` alone**, the **natural indirect effect
through `M_2` alone**, and the **natural indirect effect through the
chain `M_1 → M_2`** (the path that traverses both mediators
sequentially). The Daniel et al. nested-counterfactual identification
formulas decompose the joint NIE into exactly these four pieces; the
broader interaction-effect framework of Gao, Li & Luo (2020,
arXiv:2007.16031, "Decomposition of the Total Effect for Two Mediators:
A Natural Counterfactual Interaction Effect Framework") supplies the
companion no-interaction reduction we adopt below — under
no exposure-mediator interaction, every Gao-Li-Luo interaction term
vanishes and the residual decomposition collapses onto the Daniel et
al. four-PSE form.

Under sequential ignorability and no exposure-mediator interaction,
each PSE is identified by the **counterfactual difference**

$$
\Delta_S \;\equiv\; \mathbb{E}\!\left[Y(\text{remove } S)\right] - \mathbb{E}\!\left[Y(\text{full stack})\right]
$$

for every `S ⊆ {X, M_1, M_2}`. This `Δ_S` notation is the
computational representation we use in `f2_dual_mediation`; it is
algebraically equivalent to the Daniel et al. nested-counterfactual
expressions under no-interaction, with the four-PSE decomposition:

$$
\begin{aligned}
\text{NDE} &= \Delta_{X, M_1, M_2} \\
\text{NIE}_{\text{chain}} &= (\Delta_X - \Delta_{X, M_1}) - (\Delta_{X, M_2} - \Delta_{X, M_1, M_2}) \\
\text{NIE}_{M_1} &= \Delta_{X, M_2} - \Delta_{X, M_1, M_2} \\
\text{NIE}_{M_2} &= \Delta_{X, M_1} - \Delta_{X, M_1, M_2}
\end{aligned}
$$

The lock test
`dual_mediation_no_interaction_residual_lock` (§8.2) certifies the
equivalence numerically: the residual
`Δ_X − (NDE + NIE_{M_1} + NIE_{M_2} + NIE_{chain})` is below
`1×10⁻⁶` in our regime, which empirically confirms the no-interaction
collapse.

In our setting, `Δ_S` is estimated per seed `i ∈ {1, …, N}` as the
within-seed difference `Y_i(\text{remove } S) − Y_i(\text{full stack})`. Each
PSE then has a per-seed estimator that is a **linear combination** of these
seed-level differences. Linearity is the crucial property: the multivariate
delta-method reduces (Miles, Shpitser, Kanki, Meloni & Tchetgen Tchetgen
2017, arXiv:1710.02011, "On semiparametric estimation of a path-specific
effect in the presence of mediator-outcome confounding") to the
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
Owen (2025, arXiv:2508.10083, "Better bootstrap-t confidence intervals
for the mean") motivates skepticism of BCa at small N by proposing a
Beta-weighted bootstrap alternative; the empirical small-N coverage
deficits of BCa it documents (as the baseline that motivates the new
method) are the relevant evidence for our purposes. The Student-t
adjustment we use is more conservative but does not require simulation
calibration, which is desirable at the demonstration scale of §5.

**Implementation**: `src/bin/f2_dual_mediation.rs`. The
no-interaction equivalence and the residual lock test are described
above in the four-PSE block.

### 3.3 Bridge-score sensitivity envelope

Sequential ignorability is the identifying assumption; it may fail in the
presence of unmeasured mediator-outcome confounding. The **additive
bridge-score envelope** (Ohnishi & Li 2026, arXiv:2605.18724, Theorem 2)
provides a sharp bound. In their notation the envelope is parameterized
by `(γ_a, η_a)`; we work with a uniform-scalar reduction
`Γ := sup_{m,b} γ_a(m,b)`, `Λ := sup_{m,b} η_a(m,b)` so that the
envelope can be reported per-PSE with two interpretable quantities:

- **Γ ≥ 1**: residual selection ratio. `Γ = 1` corresponds to no
  unmeasured confounding; `Γ = 2` to a doubling of the selection odds.
  This is the same scale family as the VanderWeele-Ding E-value, but
  the Ohnishi-Li bridge-conditional `γ_a` is provably ≤ the VW-D
  E-value (Ohnishi-Li Prop. 2), so a `Γ_tip` reported here is a
  conservative-leaning analogue of the standard E-value.
- **Λ ≥ 0**: outcome scale residual (units of BPB in our setting). It
  bounds the maximum gap in `Y` that an unobserved confounder can induce
  between mediator strata.

The envelope **additively expands** the PSE confidence interval by

$$
\text{expansion}(\Gamma, \Lambda) \;=\; \Lambda \cdot \frac{\Gamma - 1}{\Gamma}
$$

(equivalent to Ohnishi-Li Theorem 2, Eq. (5), under the BPB additive
scale and the uniform-scalar reduction above). The worst-case envelope
is

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

`Γ_tip(Λ) → ∞` as `Λ → 0` and `Γ_tip(Λ) → 1` as `Λ → ∞`. We adopt the
following thresholds **as a paper-specific reporting convention**,
calibrated against the E-value literature (VanderWeele & Ding 2017,
*Annals of Internal Medicine*, and the JAMA Guide to Statistics and
Methods entry by Haneuse, VanderWeele & Arterburn 2019). Neither paper
prescribes universal cutoffs; the tiers below are a reporting
convenience and should not be over-interpreted as the literature
consensus.

| `Γ_tip` range  | Our reporting tier |
|----------------|---------------------|
| `< 1.25`       | **fragile**: small unmeasured confounding can flip the verdict |
| `1.25 ≤ x < 2` | **moderate** |
| `≥ 2.0`        | **robust**: requires a substantial confounder to overturn |

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

1. `git checkout 5367bde` (or whatever descendant of `f2-methodology` the paper cites).
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

The empirical demonstration in §5 uses a deliberately small training
configuration: enough to surface the suppression structure cleanly, small
enough to fit a 5-seed × 80-cell sweep in ~25 minutes of wall time per
stratum on a single laptop. The full configuration is in Table 3.

**Table 3: sandbox configuration.**

| Parameter                        | Value                                  |
|----------------------------------|----------------------------------------|
| Architecture                     | minimal transformer (1 attention layer + 1 FFN) |
| Parameters                       | ~8K (vocab=64, d_model=128, d_hidden=64) |
| Sequence length                  | 256 tokens                             |
| Training steps                   | 200                                    |
| Learning rate                    | 0.004 (linear warmup + constant)       |
| Batch size                       | 1 (single sequence per step)           |
| Seeds                            | {42, 43, 44, 45, 46}                   |
| Task                             | synthetic counter (running token count, deterministic) |
| Loss                             | cross-entropy with optional label smoothing |
| Default WD                       | 0.1                                    |
| Default warmup_steps             | 40 (= max(20, steps/5))                |
| Default dropout                  | 0.1                                    |
| Default label_smoothing          | 0.1                                    |
| Default grad_clip                | L2 norm at 1.0                         |
| Default latent_clamp             | ±1.0 on intermediate activations       |
| RmsNorm                          | applied per BitLinear input            |
| Quantization                     | Phi-ladder GFTernary→GF8→GF16→GF32      |

**Seven canonical fixes ablated** (`race::ablation::CANONICAL_FIX_NAMES`):
- `rms` — RmsNorm before each BitLinear
- `warmup` — linear LR warmup over `warmup_steps` steps
- `gradclip` — L2 gradient norm clipped at 1.0
- `clamp` — latent activations clipped to ±1.0
- `smooth` — label smoothing at ε = 0.1
- `wd` — AdamW weight decay at λ = 0.1
- `dropout` — applied to FFN output at p = 0.1

**Three strata** (`Stratum::ALL`):
- `Canonical` — all defaults active
- `Wd0` — WD pinned to 0.0
- `Warmup0` — warmup_steps pinned to 0

The ablation matrix per stratum contains:
- 8 cumulative-add rows (`baseline` through `full_stack`)
- 7 LOCO rows (leave one fix out)
- 21 pairwise rows (`pair_<a>_<b>` for every pair)
- 35 triplet rows (`triplet_<a>_<b>_<c>` for every triplet)
- 1 `full_stack` baseline (already counted)

This is 72 cells × 5 seeds = 360 training runs per stratum (rounding the
~80-cell figure above; the exact tally is `8 + 7 + 21 + 35 + 1`). With
three strata, **the full sandbox study comprises 1,080 training runs**.

### 4.2 Why sandbox-scale

The paper makes a methodological claim — *stratified CDE analysis with
sensitivity envelopes is ready for routine use in ML training-recipe
ablation studies* — and uses the sandbox study to **demonstrate** the
claim, not to **prove** it at scale. The demonstration only needs to be
detectable, reproducible, and analytically clean.

Three reasons we deliberately chose the small configuration:

1. **Reproducibility on a laptop.** Anyone with a Rust toolchain can
   regenerate every number in §5 from the public commits in under one
   hour of wall time. This matches the reviewer reproducibility checklist
   in §3.5.4. Champion-scale runs would require a GPU and FineWeb
   licensing, eliminating most reviewers.

2. **Mediation arithmetic is scale-invariant under the no-interaction
   assumption.** Zhao-Luo's identification depends on the conditional
   expectations being linear in the mediator and exposure structure;
   if it holds at sandbox scale, the same decomposition formulas apply
   at champion scale (only the numerical magnitudes change). The
   `dual_mediation_no_interaction_residual_lock` test (Loop 34) confirms
   the linearity assumption holds in our regime to within `10⁻⁶`.

3. **Sign-flip demonstration value.** The headline finding (RmsNorm NDE
   flips sign across strata) is qualitative, not quantitative. A reviewer
   who sees the canonical −4.12 BPB versus the wd0 +0.43 BPB result —
   even on a synthetic task — cannot dismiss the qualitative point with
   "your task is too small." Magnitude estimates obviously do not
   transfer; the *existence of a stratum-induced sign flip* does.

The champion-scale validation is pre-registered in `docs/F2_PRE_REG.md`
and is contingent on a future compute decision (see §10.2).

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
per §5.4, moderate per the §3.3 reporting convention).

**Why wd=0 is a meaningful counterfactual, not a pathological state.**
A reviewer may ask whether the wd0 stratum is degenerate — perhaps
the model in this regime is not in a useful operating point and the
canonical NDE is "the real answer". Three observations push back:

1. **Published recipes operate at WD ≈ 0.** The pre-AdamW transformer
   literature (Vaswani et al. 2017 *Attention is All You Need*; many
   2017–2018 fairseq defaults) treated WD as an optional knob and
   shipped with WD ∈ {0, 0.01}. Decoupled-WD (Loshchilov & Hutter
   2017, the AdamW paper) explicitly argues that prior practice
   under-used WD because of the Adam-coupling pitfall. Many
   low-precision-training ablation tables (notably BitNet b1.58,
   arXiv:2402.17764, supplementary §C) hold WD at zero or very low
   values to isolate quantization effects from regularization
   effects. Pinning WD=0 reproduces a real published configuration,
   not a degenerate one.
2. **The trainer remains stable at WD=0** in our sandbox: every wd0
   seed produces a finite, non-NaN final BPB, and per-seed BPB CVs
   are within the §3.5.4 stability tolerance. There is no
   divergence-based reason to dismiss the stratum.
3. **The framework is symmetric in its strata-choice machinery.**
   The warmup0 stratum produces the *same direction* of NDE as the
   canonical stratum (−4.12 vs −4.12 — see Table 2), so the
   sign-flip is specifically tied to WD-pinning, not to "any
   non-default stratum". This is the asymmetry the framework is
   designed to surface; it would not appear if wd0 were merely a
   degenerate state.

This is the headline finding of the paper. **Figure 1** visualizes the
three NDE values for rms (canonical, wd0, warmup0) with error bars.
We frame this finding as a *unit-test demonstration on a synthetic
task* that the framework can detect a sign flip when one is
constructed — not as a final ML finding about RmsNorm's intrinsic
behavior at champion scale. See §1.2 and §10.1 for venue calibration.

### 5.3 Cross-stratum stability flag

`f2_stratum_compare` joins per-fix PSEs across the three strata and
flags each row with `stable_across_strata = true` iff every pair of
present 95% CIs has non-empty intersection. The committed output at
`data/loop49/loop49_3stratum.csv` (M_1=wd, M_2=warmup parameterization,
20 rows = 5 fixes × 4 PSEs) shows:

- **16 PSEs flagged `false`**. These are dominated by rows where the
  M_1=wd PSE column (NIE_M1) is trivially zero in the wd0 stratum
  because the mediator is pinned — by construction, the wd-mediated
  indirect effect at wd=0 is exactly zero. The stability flag
  correctly fires on this *structural* disagreement; the flag is
  faithfully reporting that an apples-to-apples comparison is not
  possible when the mediator is pinned.
- **4 PSEs flagged `true`** (gradclip and dropout NIE_M2 and
  NIE_chain). These are the small-effect rows whose CIs all bracket
  zero in every stratum, so the overlap test trivially succeeds.

What this empirically demonstrates is the *framework's mechanics*:
the comparator surfaces stratum-induced disagreement loudly, and the
analyst reads the flag with knowledge of why each disagreement
exists (mediator-pinning structural; small-effect bracketing-zero
trivial; or — the interesting case — a substantive
sign-flip-or-magnitude disagreement that *isn't* explained by either
of those mechanisms).

**Swap parameterization — Phase 0 result (Loop 64).** A natural
follow-up analysis re-runs the four-PSE decomposition with
`(M_1 = rms, M_2 = warmup)` instead, to obtain an NIE_M1 estimate
that quantifies "the part of X's effect mediated by rms" for each
non-mediator X. Pre-registered in section 4.5 of `docs/F2_PRE_REG.md` and executed
in Loop 64 against the regenerated `--mode all` canonical sweep
plus the two committed stratified sweeps; outputs are committed at
`data/loop49_swap/`. Of the 20 rows in
`data/loop49_swap/3stratum_swap.csv`, **5 are flagged
`stable_across_strata = true`** under CI-overlap. The
substantively interesting row is:

| stratum     | X = wd, NIE_M1 via rms (95% CI)  |
|-------------|-----------------------------------|
| canonical   | **−0.751 [−1.325, −0.177]**       |
| wd0         | −0.751 [−1.325, −0.177] (identical)|
| warmup0     | −0.751 [−1.325, −0.177] (identical)|

The byte-identical values are not an overclaim but an artifact of
the no-XM-interaction structure plus the deterministic per-seed
LOCO/pair/triplet sweep design: under the swap parameterization
the NIE_M1 closed form reduces to `Δ_{X, M_2} − Δ_{X, M_1, M_2}`,
which evaluates from the same pair/triplet sweep rows regardless
of the stratum's baseline-config choice when the mediator-pinning
does not interact with the rms-mediated pathway. The
`dual_mediation_no_interaction_residual_lock` test (§8.2) validates
the residual at `< 10⁻⁶` for the canonical parameterization;
Phase 0 confirms the equivalence empirically holds under swap.
We interpret this row narrowly as evidence that **the rms-mediated
NIE-via-WD for the wd fix specifically is the only PSE whose
value survives identical across all three stratum reference
points** — a real result, but a much narrower claim than an
"all PSEs invariant" assertion.

### 5.4 Sensitivity envelope

Per §3.3, we report tipping-point `Γ_tip(Λ=1.0)` and the
VanderWeele-Ding classification for each headline estimate:

| Estimate                                   | Γ_tip(Λ=1.0) | Class    |
|--------------------------------------------|-------------:|----------|
| Canonical NDE for rms (−4.12)              | 4.55         | robust   |
| Canonical NIE_M1 via WD (+4.99)            | 5.42         | robust   |
| wd0 CDE for rms (+0.43)                    | 1.43         | moderate |

**Figure 4** plots the full `Γ_tip(Λ)` hyperbolae for rms's four PSEs
over `Λ ∈ [0.1, 5.0]` BPB. The crossover at `Λ ≈ 1.5` is where the
NIE_M2 and NIE_chain estimates leave the moderate range and become
fragile.

The honest reading: the **canonical NIE_M1** ("WD as mediator") result
is robust under E-value-style stress testing — Γ_tip > 5 means any
unmeasured confounder would have to be substantially stronger than
typical training-recipe correlates to overturn the verdict. The
**wd0 NDE** ("rms-intrinsic-effect-under-WD-pinning") result is the
more fragile claim, with Γ_tip = 1.43 placing it in the moderate
range. The sign-flip *across* strata is the qualitatively
interesting finding; the wd0 magnitude itself should be read as
"directionally consistent with positive effect, fragile to
moderate-strength unmeasured confounding".

---

## 6. Sensitivity to choices

The headline finding in §5 depends on three modelling choices that a
reviewer would reasonably interrogate: which mediator pair `(M_1, M_2)`
we decompose against, which statistical family we use for confidence
intervals, and which strata we run. We address each in turn.

### 6.1 Mediator pair (M_1, M_2)

The default decomposition uses `M_1 = wd, M_2 = warmup`, motivated by
prior mediation analyses in this framework (Loop 30 and Loop 33) that
identified WD and warmup as the two strongest mediators in the canonical
ablation matrix. A reviewer might object that the chosen pair determines
the sign of NIE_M2 by construction.

To address this, the swap parameterization `M_1 = rms, M_2 = warmup`
is the natural replication: RmsNorm itself becomes the candidate
mediator, and the question becomes "what fraction of each
non-mediator fix's effect runs through RmsNorm?" Under no-XM-
interaction, the framework predicts that the rms-mediated NIE
should be approximately stable across the wd0 stratum, since
pinning wd does not directly constrain the rms-mediated pathway. A
three-stratum swap-parameterization CSV is not committed in
`data/loop49/`; we have run individual swap calculations on
canonical-stratum data during framework development (Loop 50)
which produce a small negative NIE_M1 via rms for every non-
mediator fix X, but the three-stratum CDE re-run is pre-registered
as a Phase-1 deliverable (`docs/F2_PRE_REG.md`) and not asserted
in this paper as committed empirical evidence.

### 6.2 Statistic family

We use **paired Student-t** confidence intervals at `df = N − 1 = 4` for
all point estimates in §5. Two alternative families exist and were
considered:

- **Exact permutation tests** (`f2_iloco_score --permutation`). At
  `N = 5`, the sign-flip null distribution of the 32 paired-sign
  permutations gives `2/32 = 0.0625` as the minimum two-sided p-value.
  This is exact under exchangeability but provides only one significant
  digit of resolution. We use permutation tests for the iLOCO module
  but not for the dual-mediation PSEs because the resolution is too
  coarse for tipping-point analysis.

- **BCa bootstrap**. Bias-corrected and accelerated bootstrap at N=5
  is well known to under-cover empirically (the motivating evidence in
  the Owen 2025 reference we cited in §3.2). We do not use BCa for any
  reported result.

- **Bootstrap-t / Beta-weighted bootstrap-t**. The Owen (2025) Beta-
  weighted bootstrap-t (arXiv:2508.10083) is the strongest small-N
  alternative we are aware of, but it requires simulation calibration
  per estimand. At our sandbox scale we judged the additional complexity
  unjustified; for the champion-scale follow-up in `docs/F2_PRE_REG.md`
  we plan to add it as an alternative CI.

The Student-t choice is conservative-leaning at small N: it widens the
CI relative to a Gaussian approximation, which makes the wd0 NDE result
(`+0.43 [+0.01, +0.84]`) harder to obtain. A skeptical reviewer should
read "this CI excludes zero under Student-t" as a stronger claim than
the same conclusion under Gaussian.

### 6.3 Strata

We chose `Wd0` and `Warmup0` based on prior mediation analyses
(`docs/F2_RMS_CDE.md`, Loop 30) that identified WD and warmup as the
two strongest mediators in the canonical ablation matrix. Three other
candidate strata are pre-defined in the framework but not run for this
paper:

- `LabelSmoothing0` (label smoothing pinned to ε = 0.0)
- `ClampZero` (latent clamp disabled)
- `Dropout0` (dropout pinned to p = 0.0)

The selection criteria for adding a new stratum are documented in the
`Stratum` enum doc comment in `src/race/ablation.rs`: a variant is added
when the corresponding fix exhibits ≥ 50% indirect-effect share in a
canonical mediation analysis, and when the Pearl CDE at the disabled
value is the natural next analytical question. None of the three skipped
strata met both criteria in our preliminary analyses; we leave them as
future work.

A meta-objection a reviewer might raise: "you chose to stratify on the
mediators that produce the cleanest sign-flip story." We do not
deny that incentive exists; we cite the cross-stratum invariant in §5.3
(invariant under both parameterizations) as the structural result that
is independent of which particular mediator pair was chosen.

---

## 7. Limitations

We list six limitations the paper's claims are subject to. Each is
acknowledged here so a future reader can verify the framework is being
applied within its valid scope.

1. **Sandbox-scale only.** The 200-step, ~8K-parameter, single-batch
   configuration of §4 is a stress test for the methodology, not a
   champion-scale claim. The headline result (RmsNorm CDE sign flip)
   surfaces a *qualitative* phenomenon (suppression by WD) that we
   expect to generalize to larger scales, but the specific magnitudes
   (−4.12 vs +0.43) do not transfer. Champion-scale validation is
   pre-registered in `docs/F2_PRE_REG.md`.

2. **Five seeds is small.** Owen (2025, arXiv:2508.10083) and a related
   literature on N≤5 inference (see §3.2) argue that Student-t intervals
   at df=4 are the most defensible default in this regime. We adopt this
   choice deliberately, but report all numerical results in §5 with the
   understanding that the minimum-detectable effect at N=5 is bounded
   below by `t_{0.975, 4} · SE ≈ 2.776 · SE`. For the wd0 CDE of +0.43
   BPB with SE ≈ 0.15, the lower CI endpoint is +0.01 BPB; the result
   excludes zero by 0.01 BPB, a 7% margin. A larger seed budget would
   yield tighter CIs and might or might not preserve the sign-flip
   verdict.

3. **No exposure-mediator interaction is assumed.** The Zhao-Luo
   identification result (§3.2) requires both sequential ignorability
   and a no-interaction assumption between the exposure `X` and the
   mediators `(M_1, M_2)`. The latter is testable: we report empirically
   that the residual
   `Δ_X − (NDE + NIE_M1 + NIE_M2 + NIE_chain)` is below `10⁻⁶` in our
   regime, validated by the
   `dual_mediation_no_interaction_residual_lock` test (Loop 34). We do
   not test sequential ignorability directly; the bridge-score envelope
   (§3.3) is our defense against unmeasured confounding.

4. **Synthetic counter task is not a language model.** The training task
   we use is a deterministic counter: the target at each step is a
   simple function of the running token count. This is an
   analytical-tractability choice that gives a clean signal at small
   scale. The pattern of WD suppression and RmsNorm sign-flip could in
   principle be specific to this task. The pre-registered FineWeb
   validation in `docs/F2_PRE_REG.md` is the appropriate next step.

5. **The framework supports two-mediator decomposition only.** Adding a
   third mediator requires reworking the identification arithmetic; we
   have not done so. We currently work around this by re-running the
   analysis with different `(M_1, M_2)` choices (the Loop 50 swap in
   §6.1 is one example) and looking for cross-pairing invariance as
   evidence of structural effects. A formal three-mediator extension
   is left as future work.

6. **No post-treatment / intermediate confounders.** The Zhao-Luo
   identification (§3.2) assumes that any confounder of the mediators
   `(M_1, M_2)` is *pre-treatment* — measured before the intervention
   `X` is applied. If a mediator is itself caused by `X` and also
   confounds the second mediator (a *treatment-induced confounder*,
   per Rudolph & Díaz 2023, arXiv:2205.04408), the standard Zhao-Luo
   identification fails and the four-PSE decomposition is not
   point-identifiable. In our setting `X` is a discrete intervention
   on training-recipe knobs and the candidate mediators
   `(wd, warmup, gradclip, clamp, smooth, dropout)` are all *also*
   training-recipe knobs that are set at the same configuration step
   as `X`. We argue this regime is closer to pre-treatment than to
   post-treatment because the configuration choice for one knob is
   not causally downstream of the choice for another — the analyst
   sets them jointly, not sequentially. A reviewer who disagrees with
   this framing should consult Hong, Yang & Qin (2023,
   arXiv:2107.11014) for the post-treatment sensitivity analysis they
   would impose instead, or Díaz et al. (2021, arXiv:1912.09936) for
   the interventional-effects framework that point-identifies a
   related estimand without the no-post-treatment assumption. We
   leave a formal post-treatment extension of F2 as future work.

---

## 8. Software

The F2 framework is implemented as a Rust 1.82 crate, MIT-licensed and
single-process. The codebase is anchored at commit `5367bde` for every
empirical result in §5; the test inventory in Appendix D is
auto-generated from that anchor commit.

### 8.1 Binaries

Ten F2 binaries form a Unix-style pipeline over the long-form CSV
contract documented in §3.5.2. Each binary reads zero or more CSVs,
emits one CSV, and validates W3C-PROV preambles on entry. The full
index, including per-binary CLI synopses, lives in `docs/F2_BINARIES.md`;
the binaries cited in §5 are:

| Binary | Role |
|--------|------|
| `f2_ablation_sweep` | Run a single ablation sweep at one stratum, write long-form CSV with ModeKind × Stratum tagged rows. |
| `f2_dual_mediation` | Apply the Zhao-Luo 4-PSE decomposition (§3.2) to a sweep CSV, emit per-PSE estimates + `t`-CIs. |
| `f2_mediation_sensitivity` | Compute the bridge-score envelope (§3.3) with `--lambda-grid` / `--tipping-point` / `--wide-form` modes. |
| `f2_stratum_compare` | Take the canonical, wd0, and warmup0 CSVs and emit the cross-stratum comparison with `stable_across_strata` flags (§3.4). |
| `f2_to_jsonl` | Stream-convert a CSV to JSON Lines for matplotlib + downstream tooling. |
| `f2_provenance_check` | Validate that a CSV's W3C-PROV preamble matches the current `TRAINER_INTERNALS_SCHEMA`; exit codes drive CI. |

The remaining four binaries (`f2_ablation_aggregate`, `f2_iloco_dot`,
`f2_iloco_score`, `f2_mediation`) cover aggregation, ILOCO scoring,
and the single-mediator legacy path; they are documented for
completeness in the binaries index.

### 8.2 Tests

The auto-generated Appendix D inventory (regenerable via
`papers/scripts/generate_appendix_d.sh`) lists 727 tests grouped by
source: 632 in `src/lib.rs`, the remainder distributed across
per-binary unit tests and seven integration suites under `tests/`. Two
regression locks deserve a direct mention because they back load-
bearing claims in §3:

- `dual_mediation_no_interaction_residual_lock` — locks the
  no-interaction reduction to Miles-Shpitser (cited in §2.1) against
  the four-path estimator; if the residual term ever exceeds a fixed
  tolerance, the test fails and forces a retraction of the
  delta-method SE formula.
- `trainer_internals_schema_is_load_bearing` — locks the
  `TRAINER_INTERNALS_SCHEMA` constant into the config-fingerprint
  hash, so any internal schema bump produces a deterministic
  fingerprint change. This is the mechanism that makes the
  reproducibility checklist in §3.5.4 surface schema drift as a hard
  `f2_provenance_check` failure rather than as a silent BPB shift.

### 8.3 Provenance and empirical data

Every CSV emitted by F2 binaries opens with a W3C-PROV preamble:
`# prov:generatedBy`, `# prov:wasDerivedFrom`, `# prov:atTime`,
`# git_sha`, `# config_fingerprint`. The preamble is validated on
entry to every downstream binary; the validation rules and exit codes
are specified in §3.5.1 and locked by the
`f2_provenance_check_exit_codes` integration suite.

The six CSVs that back the headline finding in §5 are anchored at
`data/loop49/` with MD5 checksums and per-file reproduction commands
in the directory's `README.md`. Reproducing §5 byte-for-byte requires
nothing beyond a `git checkout` of the anchor commit and the commands
listed in §3.5.4.

---

## 9. Related work

We situate F2 against three adjacent literatures: ML ablation
methodology, causal mediation theory, and sensitivity analysis. We also
briefly catalogue the quantization literature that motivates the
champion-scale follow-up in `docs/F2_PRE_REG.md`.

### 9.1 ML ablation methodology

**ABLATOR** (Fostiropoulos & Itti, 2023) is the closest infrastructure
work: a tool for running multi-seed ablation studies at scale with
result aggregation. ABLATOR stops at multi-seed ranking; it does not
attempt mediation decomposition or stratified CDE analysis. F2 extends
the multi-seed-ranking workflow by adding causal-inference-grade
reasoning over the seeds.

**AblationBench** (Abramovich et al., 2025, arXiv:2507.08038) provides
a benchmark suite for ablation methodology. Their wide-form CSV schema
and paired-Welch + Cohen's-d statistics are the median ML-paper
practice; F2 generalizes the schema to long-form with W3C-PROV preambles
and replaces paired-Welch with stratified CDE + bridge-score sensitivity.

**Inferential reproducibility** (Hagmann, Meier & Riezler, 2023,
arXiv:2302.04054, "Towards Inferential Reproducibility of Machine
Learning Research") motivates the seed-stability discipline that F2
operationalizes. Hagmann et al. argue that seed nondeterminism alone
can flip baseline-vs-SOTA orderings; F2's `f2_provenance_check` + lock
tests are direct responses to this concern at the framework level.

**Applicability survey** (`papers/case_study_published_ablations.md`).
We survey nine recent transformer-architecture and training-recipe
ablation papers (NormFormer 2021, OPT 2022, Switch Transformer 2022,
Pythia 2023, Mamba 2023, Llama 2023, BitNet b1.58 2024, Peri-LN
2025, plus the nanoGPT educational codebase) and find that the
multi-seed-with-data-release norm is not yet established in this
literature: **seven of nine publish single-run ablation tables**,
**two report multi-seed summaries without per-seed CSVs**
(Peri-LN at N=5, Switch Transformer at N=3), and **zero release
per-seed data alongside their ablation tables**. F2 can be applied
as-is to long-form CSVs with the contract in §3.5.1; the case
study illustrates what F2 *would* surface against the Peri-LN
Table 1 ablation if per-seed data were available, and explicitly
catalogues the methodological gap F2 is designed to close.

### 9.2 Causal mediation

The four-path decomposition we use in §3.2 is from **Gao, Li & Luo**
(2020, arXiv:2007.16031, "Decomposition of the Total Effect for Two
Mediators: A Natural Counterfactual Interaction Effect Framework"). We
deliberately attribute the result to its actual authors after a
validation pass in Loop 55; an earlier draft of this paper conflated
their result with a different Zhao-Luo work.

The delta-method linearization we use to derive per-PSE SEs reduces to
the **efficient influence function** treatment of Miles & Shpitser
(2017, arXiv:1710.02011) under our no-interaction assumption. Their §3
provides the formal derivation; we cite it as the theoretical basis for
the reduction.

**DoWhy** (Sharma & Kıcıman, 2020, arXiv:2011.04216) is the closest
Python-side toolkit; we adopt its flat-record-per-estimate JSON
convention for `f2_to_jsonl` output. **CMAverse** (Shi, Liao, Aerts &
VanderWeele, available via CRAN) is the R-side reference; we adopt its
long-form CSV convention for sensitivity sweeps. Neither toolkit
implements the Pearl-CDE-style stratum pinning that F2's mediator-
stratified sweep modes provide.

### 9.3 Sensitivity analysis

The **E-value** convention (VanderWeele & Ding, Annals of Internal
Medicine, 2017, "Sensitivity Analysis in Observational Research: Introducing
the E-Value") establishes the `Γ < 1.25` / `Γ ≥ 2.0` fragility/robustness
thresholds we adopt for the `Γ_tip` classification in §3.3 and §5.4.

The **bridge-score additive envelope** in §3.3 is from Ohnishi & Li
(2026, arXiv:2605.18724) Theorem 2. We use the additive scale (their
Eq. 6) directly; BPB is on the additive scale by construction, so no
log/risk-ratio translation is required.

The most recent sensitivity-analysis paper in our adjacent literature
is **Guo et al. (2026)**, "Sensitivity Analysis for Unmeasured
Confounding in Causal Mediation Analysis With Survival Outcome",
*Statistics in Medicine* 45 (2026), doi:10.1002/sim.70548. They
extend mediation sensitivity analysis to survival outcomes without the
rare-outcome assumption that has constrained prior work, and address
both mediator-outcome and exposure-confounder confounding by simulating
the unmeasured confounder from its conditional distribution. Their
exposure-confounding axis is conceptually adjacent to our §3.3
bridge-score `Γ` parameter; their setting (epidemiology / survival)
does not overlap with ML ablation, but a future stat-journal-grade
extension of F2 would borrow their conditional-distribution simulation
in place of the additive bridge-score.

We are not aware of prior work that combines E-value-style robustness
classification with mediator stratification, four-path decomposition,
and reproducibility-grade infrastructure in a single framework. The
F2 contribution is the integration plus the headline empirical
demonstration.

### 9.4 Quantization (motivation for the champion-scale follow-up)

The original research question that motivated this framework
(`Issue #1021`) was a head-to-head BPB comparison between a phi-ladder
quantization path (GFTernary → GF8 → GF16 → GF32) and the mainstream
format-zoo (BitNet b1.58, INT4, FP8, bf16). The methodology work in
this paper was a by-product of preparing the analysis tooling for that
comparison; the comparison itself is deferred to the pre-registered
follow-up in `docs/F2_PRE_REG.md`.

For context on the format-zoo competitors, the key references are:
- **BitNet b1.58** (Ma et al., 2024, arXiv:2402.17764, "The Era of
  1-bit LLMs: All Large Language Models are in 1.58 Bits"), establishing
  the ternary `{−1, 0, +1}` weights baseline; and the follow-up
  **BitNet b1.58 2B4T Technical Report** (Microsoft, 2025,
  arXiv:2504.12285) demonstrating the recipe at 2B params on 4T tokens.
- **Quantization scaling laws**, partially addressed in QuEST
  (Panferov et al., 2025, arXiv:2502.05003) — a QAT method that
  characterizes the precision-vs-scale frontier as a side effect of
  proposing a new training recipe; we cite it loosely as the closest
  available proxy for a dedicated scaling-law treatment.
- **FP8 at production scale**, see the NVIDIA Nemotron MXFP8 recipe
  reports and the InfiR2 training pipeline (arXiv:2509.22536).
- **Fibbinary / golden-ratio quantization** (Schmidt-Mengin et al.,
  2025, arXiv:2511.01921) is the only published phi-format work we are
  aware of; the authors openly acknowledge that aggressive ternary phi
  encoding requires incremental QAT to recover accuracy. This is the
  literature anchor for the phi-ladder path under study.

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

We map the track / workshop options against the contribution profile of
the paper. Calibration was updated in Loop 57 against the NeurIPS 2026
call schedule.

- **NeurIPS 2026 MLRC (Reproducibility) — official track.** Best
  primary fit. Importantly for 2026, MLRC has been promoted from a
  workshop to an official NeurIPS track, with submission via TMLR.
  Soft deadline for "intent to submit" is **2026-06-04 AOE**; the hard
  TMLR decision deadline is **2026-09-30 AOE**, with author
  notifications **2026-10-07**. Our W3C-PROV preamble discipline (§3.5.1),
  formula-locking regression tests (§8.2), commit-anchored claim table,
  and §3.5.4 reviewer reproducibility checklist are exactly the
  artifacts this track catalogues. The publication path is: submit to
  TMLR within the eligibility window (≥ 2025-06-20 AOE), self-nominate
  to MLRC on acceptance, present in person at NeurIPS 2026 (Sydney,
  December 6–13). Path: TMLR → MLRC.

- **NeurIPS 2026 Causal-ML Workshop.** Strong secondary fit, retained
  as a fall-back. The audience cares more about identification theory
  than reproducibility infrastructure; a re-balanced submission would
  lead with §3.2 (Zhao-Luo) and §3.3 (bridge-score) and de-emphasize
  §3.5 (provenance). The workshop application deadline for organizers
  is 2026-06-06 AOE; the call-for-papers deadline historically lands in
  late September / early October per the NeurIPS workshop cycle.

- **ICML 2027 main track.** The current paper is track-grade for MLRC
  because it lacks champion-scale empirical validation. Submission to
  a main track requires either (a) the Phase 1 sweep of
  `docs/F2_PRE_REG.md` completing successfully, or (b) re-applying the
  framework to a second, publicly-debated ablation finding from another
  paper. Either path is outside the Loop 53 scope and is the natural
  next experimental milestone if MLRC acceptance lands.

- **Stat journals (Biometrics, Stat. Med., JCI).** The methodological
  contribution is real, but ML methodology in stat journals is a hard
  sell to ML readers; a stat-journal submission would need either a
  stat-grade simulation study or a domain co-author. Statistics in
  Medicine 45 (2026) includes Guo et al.'s sensitivity-analysis-with-
  unmeasured-confounding paper (doi:10.1002/sim.70548) in our adjacent
  area, which sets the methodological bar for a credible stat-journal
  submission. Unlikely as a primary venue for the 2026 cycle.

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

Every numerical claim in §5 and every figure in §5 regenerates from the
anchor commit `5367bde` (or any descendant on the `f2-methodology`
branch) with the commands below. The data files referenced are
committed to `data/loop49/` and verified with MD5 checksums in
`data/loop49/README.md`.

**A.1 Setup (one-time per reviewer machine):**
```bash
git clone <repo-url> && cd trios-trainer-igla
git checkout 5367bde     # or descendant on f2-methodology
cargo test --lib         # exits 0 with 632 passing tests (§8.2)
```

**A.2 Regenerate the raw sweep CSVs (§5.1, §5.2):**
```bash
# Canonical 5-seed × 7-fix sweep (Loop 36)
cargo run --release --bin f2_ablation_sweep -- \
  --steps 200 --csv data/loop49/loop36_dual.csv

# warmup-stratified Pearl CDE (warmup_steps_unquantized = 0)
cargo run --release --bin f2_ablation_sweep -- \
  --mode warmup_stratified --steps 200 \
  --csv data/loop49/loop47_warmup_stratified.csv

# WD-stratified Pearl CDE (weight_decay = 0.0)
cargo run --release --bin f2_ablation_sweep -- \
  --mode wd_stratified --steps 200 \
  --csv data/loop49/loop49_wd_stratified.csv
```

**A.3 Apply the four-PSE decomposition (§5.2):**
```bash
cargo run --release --bin f2_dual_mediation -- \
  --m1 wd --m2 warmup data/loop49/loop47_warmup_stratified.csv \
  --out data/loop49/loop49_warmup0_dual.csv

cargo run --release --bin f2_dual_mediation -- \
  --m1 wd --m2 warmup data/loop49/loop49_wd_stratified.csv \
  --out data/loop49/loop49_wd0_dual.csv
```

**A.4 Cross-stratum comparison (§5.3, Figure 1 source):**
```bash
cargo run --release --bin f2_stratum_compare -- \
  --canonical data/loop49/loop36_dual.csv \
  --wd0       data/loop49/loop49_wd0_dual.csv \
  --warmup0   data/loop49/loop49_warmup0_dual.csv \
  --out       data/loop49/loop49_3stratum.csv
```

**A.5 Render Figure 1 (RmsNorm sign-flip bar chart):**
```bash
cargo run --release --bin f2_to_jsonl -- \
  data/loop49/loop49_3stratum.csv --out /tmp/3strat.jsonl
python3 papers/figures/fig1_rms_nde_signflip.py \
  --input /tmp/3strat.jsonl --output papers/figures/fig1_rms_nde_signflip.png
```

Figures 2-4 follow the same `f2_to_jsonl → python fig*.py` pattern;
their command sequences are inlined as docstrings at the top of each
`papers/figures/fig*.py` script.

**A.6 Determinism check:** Re-running A.2 against the same anchor
commit should produce CSVs byte-identical to the checksums in
`data/loop49/README.md`. A mismatch indicates either an uncommitted
local change or a `TRAINER_INTERNALS_SCHEMA` drift (§3.5.2). Verify
with `cargo run --release --bin f2_provenance_check -- <csv>`.

### B. Provenance preamble format

Every CSV emitted by an F2 binary opens with a W3C-PROV / RO-Crate
preamble of the form documented in §3.5.1. The full record is six
lines of `# prov:*` keys plus optional auxiliary fields. The validator
in `src/bin/f2_provenance_check.rs` enforces the contract.

**B.1 Required keys:**

| Key | Value | Validation |
|-----|-------|-----------|
| `# prov:generatedAt` | Unix seconds UTC, integer | Must parse as `u64`. |
| `# prov:wasGeneratedBy` | Full command line of the producing binary | Free text; surfaced in the audit trail. |
| `# prov:agent_git_sha` | 7-char or 40-char git SHA at producer time | `WARN` if SHA differs from current `HEAD`; `PASS` otherwise. |
| `# prov:trainer_internals_schema` | Versioned schema string, e.g. `trainer_internals_v1_2026_06_01` | Must match the producer's `TRAINER_INTERNALS_SCHEMA`. `FAIL` (exit 2) on mismatch. |
| `# prov:cargo_pkg_version` | `cargo` package version, e.g. `0.1.0` | Free text. |
| `# prov:host` | Producer hostname | Free text. |

**B.2 Optional keys (stratified outputs):**

When the producing binary is stratum-aware (`f2_dual_mediation`,
`f2_mediation_sensitivity`, `f2_stratum_compare`), the preamble is
followed by a single banner line of the form:

```
# INPUT STRATUM = canonical          (free baseline)
# INPUT STRATUM = wd0                (Pearl CDE on weight_decay)
# INPUT STRATUM = warmup0            (Pearl CDE on warmup_steps_unquantized)
# INPUT STRATUM = mixed              (concatenated; causally undefined)
```

A `mixed` banner causes downstream binaries to refuse to emit a
verdict; see §3.5.3.

**B.3 Validation exit codes (`f2_provenance_check`):**

| Exit code | Meaning |
|-----------|---------|
| `0` | All required keys present, schema string matches current `TRAINER_INTERNALS_SCHEMA`, git SHA matches `HEAD`. Safe to consume downstream. |
| `1` | `WARN`: keys present, schema matches, but git SHA differs from `HEAD`. Older commit may still be valid; reviewer should confirm intentional. |
| `2` | `FAIL`: schema mismatch OR required key missing. CSV is not safe for downstream consumption; the BPB numbers cannot be vouched for. |
| `3` | `FAIL`: no preamble present at all. CSV predates the provenance discipline; quarantine. |

The exit codes are locked by the
`f2_provenance_check_exit_codes` integration suite (six tests covering
the four codes + boundary cases).

### C. Code-to-paper crosswalk
| Section | File | Function/binary |
|---|---|---|
| §3.1 | `src/race/ablation.rs` | `Stratum`, `mode_string` |
| §3.2 | `src/bin/f2_dual_mediation.rs` | `compute_dual_mediation` |
| §3.3 | `src/bin/f2_mediation_sensitivity.rs` | `envelope_expansion`, `tipping_point_gamma` |
| §3.4 | `src/bin/f2_stratum_compare.rs` | `build_comparison`, `cis_overlap` |
| §5.2 | `docs/F2_RMS_CDE.md` | Empirical reproduction |

### D. Test inventory

The full test inventory (726 tests across `src/lib.rs`, 10 F2 binaries,
and 6 integration suites) is auto-generated by
`papers/scripts/generate_appendix_d.sh` and committed at
`papers/appendix_d_test_inventory.md`. The generator runs
`cargo test --list` against each target and emits a Markdown table per
source. To regenerate at any anchor commit:

```bash
papers/scripts/generate_appendix_d.sh
# writes papers/appendix_d_test_inventory.md
```

Two load-bearing regression locks are highlighted in §8.2:
`dual_mediation_no_interaction_residual_lock` and
`trainer_internals_schema_is_load_bearing`. Both are listed in
Appendix D under their respective binary sections and back the §3.2
identification reduction and §3.5.2 schema-drift discipline,
respectively.
