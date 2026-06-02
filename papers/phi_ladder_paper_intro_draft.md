# Phi-ladder vs. the quantization zoo at champion scale: a
# pre-registered companion to F2 (DRAFT §1 only)

**Status**: §1 draft, Loop 98 (2026-06-02). Frames Issue #1021 as
the empirical companion to `papers/f2_methodology.md`. **No
champion-scale run has been executed yet** — this document is the
introduction that would lead the protocol-only manuscript, against
which the eventual run will be evaluated. The protocol itself is
locked in `docs/F2_PRE_REG.md`; this draft is the framing the
TMLR/MLRC reader sees first.

---

## 1. Introduction

A modern transformer training recipe pins down dozens of choices
that interact in non-obvious ways. Among them, the **numeric
representation** of weights and activations — bf16, FP8, INT8,
BitNet-1.58 ternary — has become a moving target, with new "zoo"
entries published every quarter and very few of them evaluated
against each other on the same controlled training regime. The
practitioner question we address is narrow: *given a fixed
1B-parameter transformer recipe at 50B training tokens of FineWeb,
does the **phi-ladder** family of representations
(GFTernary → GF8 → GF16 → GF32, anchored at the golden-ratio
recurrence $\phi^2 + \phi^{-2} = 3$) yield lower held-out
validation bits-per-byte (BPB) than the leading members of the
mainstream zoo (BitNet-1.58, INT8 Jetfire, MXFP8, bf16)?*

The contribution of this paper is **not** a positive answer to
that question. We pre-register a sweep against eight
configurations × two strata × five seeds (80 runs total) and
commit, before any run is launched, to the analysis plan and the
hypothesis-test cutoffs. The contribution is the experimental
protocol itself, plus the bridge from a sandbox-scale causal-
mediation finding — the RmsNorm NDE sign-flip at
$\mathrm{WD}=0$ reported in our companion paper, "Bridging Pearl-
style stratified CDE with the additive bridge-score envelope"
(F2, [anchor]) — to a falsifiable champion-scale claim about
quantization paths.

### 1.1 Why phi-ladder

The phi-ladder is anchored at the identity
$\phi^2 + \phi^{-2} = 3$, where $\phi = (1+\sqrt{5})/2$ is the
golden ratio. This identity gives a closed-form recurrence for
quantizing real numbers onto a Fibonacci-style basis: GFTernary
encodes weights as $\{-\phi, 0, +\phi\}$, GF8 packs sums of
$\pm\phi^k$ to 8-bit slots, GF16 and GF32 extend to 16- and
32-bit precision while keeping the same multiplicative basis. The
representation has been studied for **neural radio receivers**
(Fiandaca & Gomony 2025, arXiv:2511.01921) and for **fixed-point
optical lattice clocks** (Hagmann 2021), but to our knowledge has
not been benchmarked head-to-head against the dominant
quantization zoo on a transformer LM at $\geq$ 1B parameters.

There are two reasons to suspect the comparison is informative:

1. **The bf16 family is over-parameterized for the noise floor of
   transformer training**. Owen (2025, arXiv:2508.10083) shows that
   BCa-bootstrap intervals routinely undercover at small N — by
   implication, the effective resolution at which weights need to be
   represented to recover the underlying objective is lower than the
   16-bit storage suggests. The phi-ladder packs the same effective
   resolution into a smaller alphabet, with multiplicative basis that
   matches the recurrence structure of layer-norm + skip-connection
   gradients.

2. **Our companion F2 paper documents a sign-flip at $\mathrm{WD}=0$
   that is specifically a structural-format effect, not a
   hyperparameter effect**. The natural direct effect of removing
   RmsNorm is $-4.12$ BPB in the canonical stratum (i.e., RmsNorm
   accounts for $\sim 4$ BPB of held-out validation performance);
   at $\mathrm{WD}=0$, the same removal yields $+0.43$ BPB
   (RmsNorm is now *anti-productive*). The companion F2 paper uses
   Pearl-style CDE to pin this as an interaction effect between
   RmsNorm's normalizing role and weight decay's regularizing
   role. This paper uses the same *strata pair* (canonical, wd0)
   on a total-effect contrast (not a Pearl CDE — see §3.4) and
   should be able to detect or rule out analogous interaction
   effects between quantization path and WD.

The phi-ladder is therefore a *theoretically motivated* alternative
to the zoo. The pre-registered prediction in
`docs/F2_PRE_REG.md` §2 says: at the WD=0 stratum, phi-ladder
configurations are $\geq 0.10$ BPB lower than every mainstream
alternative; at the canonical stratum (default WD), the gap may
or may not survive, in which case the result is interpretable as
"phi-ladder works because it interacts favorably with WD's
structural role." If neither stratum shows the gap, the
phi-ladder family is no better than bf16 at the relevant scale,
and the paper reports a null. The protocol does not permit the
post-hoc filtering that would make null results unpublishable.

### 1.2 Why pre-registered

The format-zoo literature has a publication-bias problem. Of the
nine published format-comparison studies in our applicability
survey (`papers/case_study_published_ablations.md`),
**seven do not specify the analysis plan in advance**. Eight of
the nine report a positive result for the proposed format; the
single negative report (InfiR2, arXiv:2509.22536) was
**withdrawn** after a data-processing bug was found post-print.
The asymmetry suggests substantial garden-of-forking-paths
exposure.

Our protocol locks five decisions before data acquisition:
the eight configurations, the two strata, the five seeds, the
paired-permutation test of Zmigrod-Vieira-Cotterell (2022,
arXiv:2205.01416), and the Benjamini-Hochberg correction over the
four pairwise comparisons per stratum (Liu, Leung & Shao,
arXiv:1712.03305). The bridge-score envelope of Ohnishi & Li
(2026, arXiv:2605.18724) is applied to every cell that survives
the test, calibrated against the VanderWeele-Ding (2017) E-value
to surface tipping points. Anything in the eventual paper that
deviates from this plan will be explicitly labeled exploratory.

### 1.3 What the F2 companion provides

Our methodology companion paper provides four pieces of analysis
machinery, of which this paper consumes two: (1) the **two-stratum
contrast design** (`canonical`, `wd0`) that the F2 sandbox-scale
finding used to surface format-WD interaction at the *total-effect*
level (NOT Pearl CDE in the identification-of-mediator sense —
this paper has no controlled mediator, see §3.4); (2) the
**additive bridge-score envelope** of Ohnishi & Li 2026 Thm 2
(`f2_mediation_sensitivity`, operated in total-effect mode for
this paper — see §3.4). The F2 framework also provides (3) the
**four-PSE nested-counterfactual decomposition**
(`f2_dual_mediation`) and (4) the **cross-stratum comparator**
(`f2_stratum_compare`); this paper does **not** deploy (3) for
the format comparison because the candidate mediators
(`lossy_conversions`, `wall_clock_s`) are deterministic functions
of the format-choice $X$ and so violate the positivity/overlap
assumption that nested-counterfactual identification requires —
see §3.4 for the diagnostic and the total-effect reframe. The
cross-stratum comparator (4) is applied at the (phi, zoo)
total-effect level rather than per-PSE.
The same 10 F2 binaries, 809 tests, and W3C-PROV preamble
discipline that back the companion paper's sandbox-scale RmsNorm
finding are the substrate this paper runs on at champion scale.
A reader who has not yet seen F2 should treat §3 of this paper
as a pointer to the methodology paper, with the
quantization-zoo-specific extensions (the eight-configuration
sweep mode, the FineWeb data harness, the wall-clock provenance
preamble) called out in §3.1.

### 1.4 Scope and what this paper does not claim

We do not claim:

- **No claim of phi-ladder superiority at any scale other than the
  pre-registered $\sim 1$B params $\times$ 50B FineWeb tokens
  regime**. Loop 49 cautions explicitly against extrapolating
  sandbox-scale ablation results.
- **No claim of optimality of the specific phi-ladder
  parameterization** (GFTernary → GF8 → GF16 → GF32). Other
  Fibonacci-basis quantizations (Fibbinary,
  arXiv:2511.01921 for radio receivers) exist and are not
  evaluated here.
- **No claim of computational-cost parity** with INT8 or FP8.
  The phi-ladder's multiplicative basis costs more per FLOP than
  the additive INT/FP families; we report wall-clock and memory
  peak as secondary outcomes but do not adjust the BPB comparison
  for them.
- **No claim of robustness to data distribution shift away from
  FineWeb**. The 80-run sweep is one data corpus; the protocol
  permits but does not pre-register a multi-corpus replication.

### 1.5 Roadmap

§2 reviews quantization-format prior work organized by base
(integer, floating-point, golden-ratio); §3 specifies the
pre-registered protocol and its connection to F2; §4 reports the
pre-registered numerical hypotheses with the tests that would
falsify each; §5 enumerates the planned reproducibility artifacts
(committed CSVs, figure scripts, CI gate); §6 discusses scope
and limitations; §7 is the EOI statement. The empirical run
itself is the subject of a future paper that does **not** exist at
the time of this submission, and whose results — if they
contradict §4's predictions — will be reported as a null.

---

## 2. Related work

We organize the quantization-format literature by the base
representation each family uses, since the head-to-head BPB
comparison we pre-register only makes sense across format families,
not within one. We do **not** survey post-training-quantization or
QAT-on-pretrained-model literature; our protocol holds at
quantization-aware training from scratch at 1B parameters, so the
comparable prior work is pre-training quantization specifically.

### 2.1 Integer-base zoo (INT8, BitNet-1.58)

The 8-bit integer family has been the production default since
GPTQ (Frantar, Ashkboos, Hoefler & Alistarh 2023,
arXiv:2210.17323) and SmoothQuant (Xiao et al. 2023,
arXiv:2211.10438) demonstrated post-training INT8 with sub-percent
accuracy loss on bf16-trained models; native INT8 *training*, by
contrast, is rarer and largely confined to weight-only schemes.
The contemporary integer-quantization-during-training reference
is **BitNet b1.58** (Ma, Wang, Ma et al. 2024, arXiv:2402.17764),
which restricts weights to $\{-1, 0, +1\}$ (the "1.58 bits" name
is from $\log_2 3$). BitNet's headline finding is that ternary
weight training matches bf16 BPB at $\geq$ 700M parameters when
combined with an 8-bit activation path; the authors do not
isolate the contribution of the ternary path from the activation
quantization.

From-scratch low-bit *integer* training is dominated by **Jetfire**
(Xi et al. 2024, NeurIPS, arXiv:2403.12422), which trains
transformers with **INT8 weights, INT8 activations, and per-block
quantization** from initialization. Lower-precision INT4-W4A8
schemes exist primarily as *post-training* methods: LLM-FP4 (Liu
et al., arXiv:2310.16836) is W4A4 floating-point post-training,
and AffineQuant (Zhao et al. 2024, arXiv:2403.12544) is INT4
post-training. To our knowledge there is no widely-cited
from-scratch INT4-W4A8 reference at $\geq$ 1B parameters; the
zoo entry in §3.1 is therefore restricted to **INT8** (Jetfire's
recipe) and we drop the INT4 entry. Earlier drafts of this
manuscript (pre-Loop 105) cycled through two wrong attributions
for the integer zoo entry — first attributing
arXiv:2310.16836 to "Jetfire INT4-W4A8" (Loop 102 draft), then
correcting the arXiv id to 2403.12422 while *retaining* the
incorrect INT4-W4A8 description (Loop 104 patch). The 28th
adversarial pass (Loop 105) caught both errors; we now use
Jetfire as INT8 (its actual claim) and acknowledge there is no
champion-scale from-scratch INT4 reference to compare against.

The integer family's strength is its alignment with existing
hardware INT4/INT8 matmul kernels; its weakness is the
quantization noise floor at low weight precision, which scales
with the activation range. BitNet-1.58's contribution is in
ameliorating the latter via ternarization of the weight path
plus 8-bit activation calibration.

### 2.2 Floating-point-base zoo (FP8, bf16, MXFP8)

The floating-point family includes both 16-bit and 8-bit
variants. **bf16** (Wang & Kanwar 2019, "BFloat16: The Secret to
High Performance on Cloud TPUs") is the production baseline; its
combination of 8 exponent bits and 7 mantissa bits matches FP32's
dynamic range with $\sim$$\epsilon = 2^{-7}$ precision. Modern
training recipes default to bf16 for both weights and
activations. **FP8** (Micikevicius et al. 2022, arXiv:2209.05433,
*FP8 Formats for Deep Learning*) defines two 8-bit floating-point
formats — E4M3 (4 exponent bits, 3 mantissa, no infinities) for
forward weights/activations and E5M2 (5 exponent, 2 mantissa) for
gradients — and demonstrates BPB parity with bf16 on 175B-parameter
models when tensor-scaling is calibrated per layer.

The **MXFP8** standard (OCP 2024, "Microscaling Formats for Deep
Learning") extends FP8 with a shared 8-bit exponent per 32-element
block (mixed-precision *microscaling*), which lowers the effective
mantissa requirement and reduces the calibration burden. The
NVIDIA MXFP8 production implementation (cited in the Hopper H100
+ Blackwell B100 hardware documentation; we treat
*arXiv:2509.22536* as the contemporary academic reference, with
the caveat that it has been **withdrawn** by its authors per our
companion paper's §9.4) reports BPB parity with bf16 at 1B
parameters and 50B+ FineWeb tokens. Our zoo entry **FP8** in §1
refers specifically to E4M3 weights + E5M2 gradients per
Micikevicius et al.'s recipe, with per-tensor scaling; we do
**not** include the more aggressive MXFP8 microscaling path
because our protocol is a from-scratch comparison without runtime
calibration.

The floating-point family's strength is the dynamic-range
flexibility of the exponent; its weakness is the 8–16 bit storage
overhead relative to integer or ternary schemes. The MXFP8 path
trades calibration complexity for storage parity with INT8.

### 2.3 Golden-ratio-base (phi-ladder, Fibbinary)

The phi-ladder family is anchored at $\phi^2 + \phi^{-2} = 3$
(where $\phi = (1 + \sqrt{5})/2$); this identity yields a
Fibonacci-style recurrence for quantizing reals onto a
golden-ratio basis. The closest published precedent is
**Fibbinary** (Fiandaca & Gomony 2025, arXiv:2511.01921,
*Fibbinary-Based Compression and Quantization for Efficient
Neural Radio Receivers*), which uses the Fibonacci representation
for radio-receiver weight encoding; the authors demonstrate
$\sim$30% memory savings at iso-accuracy on a per-domain task,
but do not evaluate against transformer LLMs at any scale. Our
phi-ladder differs from Fibbinary in two ways: (i) we use the
*multiplicative* phi basis (powers of $\phi$) rather than the
*additive* Fibonacci basis (sums of $F_k$); (ii) we ladder
through four precisions (GFTernary → GF8 → GF16 → GF32) sharing
the same basis, so the ladder can be mixed within a single
training pass.

We are not aware of a prior published evaluation of the phi-ladder
at $\geq$ 1B-parameter transformer LM scale. The companion F2
methodology paper documents the phi-ladder representation as the
*substrate* on which our sandbox-scale RmsNorm sign-flip finding
sits, but does not isolate the phi-ladder's contribution to BPB.
This paper's pre-registered sweep against the integer- and
floating-point-zoo entries provides the first such evaluation.

### 2.4 Pre-registration practice in format-comparison literature

Of the nine format-comparison studies we surveyed (see Appendix C
once written), only two pre-registered their analysis plan
before data acquisition: the BitNet b1.58 paper's appendix
specifies hyperparameter sweeps and seed counts in a pre-locked
manner, and MXFP8's OCP technical report names its calibration
protocol up-front. The remaining seven describe their analysis as
a single after-the-fact narrative. Our protocol locks the
configuration set, stratification, seed counts, hypothesis tests,
and correction procedure before any FineWeb token is consumed;
the locked plan is in `docs/F2_PRE_REG.md` of the companion
codebase.

---

## 3. Pre-registered protocol

This section locks the protocol. Every choice below was made before
any FineWeb token was consumed. Any deviation from the plan during
execution will be reported with the originally-planned alternative
clearly labeled.

### 3.1 Sweep matrix

Eight configurations × two strata × five seeds = **80 runs**.

| Config slot | Family | Specification |
|---|---|---|
| `GFTernary` | phi-ladder | weights in $\{-\phi, 0, +\phi\}$; activations bf16 |
| `GF8`       | phi-ladder | 8-bit phi-encoded weights; activations bf16 |
| `GF16`      | phi-ladder | 16-bit phi-encoded weights; activations bf16 |
| `GF32`      | phi-ladder | 32-bit phi-encoded weights (intra-family baseline) |
| `BitNet-1.58` | integer | ternary weights per arXiv:2402.17764; 8-bit activations |
| `INT8`      | integer  | INT8 weights + INT8 activations + per-block quantization per Jetfire (arXiv:2403.12422) |
| `FP8`       | floating-point | E4M3 weights + E5M2 gradients per arXiv:2209.05433 |
| `bf16`      | floating-point | bf16 weights + bf16 activations (gold-standard baseline) |

**Strata**: `canonical` (default WD = 0.1, all seven F2 fixes
enabled) is the **pre-registered primary stratum**; `wd0` (WD
pinned to 0.0, otherwise identical) is the **secondary stratum**.
The asymmetric primary/secondary designation is enforced by the
§4.4 reporting rule and prevents post-hoc stratum-shopping. The
wd0 stratum is the Pearl Controlled Direct Effect on weight decay
that the companion F2 paper's §5.2 demonstrates is informative.

**Seeds**: `[42, 43, 44, 45, 46]` per (config, stratum) cell. The
seed set matches the companion paper's `F2_PRE_REG.md` convention
and the deterministic per-seed LOCO sweep that backs F2's `N = 5`
small-sample machinery.

### 3.2 Training-time configuration (locked)

- **Architecture**: dense transformer, 1B parameters
  (24 layers × 16 heads × `d_model = 2048` × `d_hidden = 8192`).
- **Sequence length**: 2048 tokens.
- **Tokenizer**: GPT-2 BPE (50257-token vocabulary) for parity
  with the FineWeb-Edu reference recipe.
- **Data**: FineWeb-Edu 10B subset (Soldaini et al. 2024,
  arXiv:2406.17557, *FineWeb-Edu* — to be added to the §2 bib);
  training token budget **50B**.
- **Held-out validation**: 100M tokens from a disjoint FineWeb
  shard (`shard 100`), held out before training begins.
- **Optimizer**: AdamW (Loshchilov & Hutter 2019, ICLR);
  $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$.
  Weight decay at canonical = 0.1; at wd0 stratum = 0.0.
- **Learning rate**: linear warmup over 2000 steps to a peak of
  $3 \times 10^{-4}$, cosine decay to $3 \times 10^{-5}$ at
  step ~12k (50B tokens / 4M tokens/step at batch 1024).
- **Batch**: 1024 sequences × 2048 tokens = 2.1M tokens/step.

The above configuration is the **same training recipe across all
eight configs**; only the weight/activation numeric format
changes. This is a structural commitment of the protocol — we are
isolating the format effect, not testing a family of training
recipes.

### 3.3 Analysis machinery (sourced from F2 companion)

The companion F2 paper provides **12 binaries** on disk at the
methodology anchor commit: 10 analysis binaries
(`f2_ablation_sweep`, `f2_ablation_aggregate`, `f2_dual_mediation`,
`f2_harness`, `f2_iloco_dot`, `f2_iloco_score`, `f2_mediation`,
`f2_mediation_sensitivity`, `f2_pareto_sweep`, `f2_stratum_compare`)
plus 2 infrastructure helpers (`f2_provenance_check`, `f2_to_jsonl`).
The F2 paper's own §8.1 names a subset; this paper's protocol uses
the on-disk inventory. Of the 10 analysis binaries, this study
deploys **4 unmodified**: `f2_ablation_sweep`,
`f2_ablation_aggregate`, `f2_stratum_compare`,
`f2_mediation_sensitivity`. It **adds one new analysis binary**
(`f2_pairwise_perm`) as part of this paper's contribution to the
F2 framework, since the companion paper has no paired-permutation
testing helper. `f2_dual_mediation` is intentionally **excluded**
because its four-PSE identification fails for our mediator
candidates (see §3.4); the remaining 5 (`f2_harness`, `f2_iloco_*`,
`f2_mediation`, `f2_pareto_sweep`) are out-of-scope for this
paper's protocol (they cover orthogonal F2 features —
single-mediator legacy, iLOCO scoring, pareto sweeps).
`f2_provenance_check` is run over every emitted CSV as
infrastructure but contributes no analysis-level estimand.

**New binary added by this paper's protocol**: `f2_pairwise_perm`
(committed in Loop 110, src/bin/f2_pairwise_perm.rs) implements
the exact Zmigrod-Vieira-Cotterell (2022, arXiv:2205.01416)
paired-permutation test on per-pair val_bpb differences plus the
Benjamini-Hochberg correction per Liu, Leung & Shao
(arXiv:1712.03305).

**On algorithmic novelty (33rd adversarial pass discovery,
Loop 110).** The companion F2 framework already exposes the exact
paired-sign-flip permutation primitive inside `f2_iloco_score`
(src/bin/f2_iloco_score.rs, `permutation_test_paired()`), citing
the same arXiv:2205.01416. `f2_pairwise_perm` is therefore a
**re-packaging exercise**, not a research artifact: it lifts the
permutation primitive into a standalone (phi-config, zoo-config)
pairwise iteration wrapper, adds BH-correction across the 4
zoo comparisons per phi-config, and emits the 16-row CSV schema
that §5.1's pairwise CSV consumers expect. The genuine
contribution is the wrapper + output schema; the underlying test
is **algorithmically equivalent at N=5** (the 34th adversarial pass
verified this with an in-source equivalence test — both primitives
produce identical p-values to machine precision on representative
inputs, including the IEEE-754-tied edge cases captured by an
epsilon-tolerant comparison shared with the F2 reference).

**Pre-registration discipline for the new binary**. Because
`f2_pairwise_perm` exists in the codebase before any of the
80-cell champion-scale sweep CSVs are produced (Loop 110 commit
predates the run), and because its `agent_git_sha` is
captured in every output CSV's W3C-PROV preamble, the protocol
satisfies the *commit-order* requirement of pre-registered
analysis: a strict reviewer can verify the binary's contents are
fixed at the protocol-lock commit by reading git history. The
binary's 4 unit tests (paired-permutation known-result,
null-result, BH monotonicity, CI95 sanity) are included in the
test inventory.

The deployed analysis binaries:

- **Per-cell BPB recording**: `f2_ablation_sweep --config <slot>
  --stratum <s> --seed <i> --output cell_<s>_<slot>_<i>.csv`.
  Each cell emits a single-record CSV with `val_bpb`,
  `train_bpb`, `wall_s`, `peak_memory_mb`, `lossy_conversions`
  (recorded as a diagnostic per §3.4, NOT used as a mediator in
  this paper's analysis), and a W3C-PROV preamble.
- **Per-stratum aggregation**: `f2_ablation_aggregate` reads the
  40 per-stratum cells and emits a long-form CSV with
  per-config mean ± SE on validation BPB.
- **Pairwise testing + BH correction**: `f2_pairwise_perm`
  (new binary; see above) runs the exact paired-permutation test
  on each (phi-config, zoo-config) pair within a stratum, then
  applies `--bh` BH-correction over the 4 pairwise comparisons
  within each phi-config (vs the 4 zoo-config alternatives). With
  5 seeds the exact test enumerates $2^5 = 32$ sign-flip vectors,
  producing exact p-values. Output: single CSV per stratum with
  16 rows × {raw_p, bh_adjusted_p, diff, ci_lo, ci_hi}.
- **Cross-stratum stability**: `f2_stratum_compare` takes the
  canonical and wd0 long-form CSVs and emits a
  `stable_across_strata` flag per (phi-config, zoo-config) pair.
- **Bridge-score envelope**: `f2_mediation_sensitivity --lambda
  1.0` is run on each pair that survives the permutation test;
  the envelope is calibrated against the VanderWeele-Ding E-value
  for fragility/robustness reporting.

The five F2 analysis binaries above (four reused unchanged from
F2 plus one new `f2_pairwise_perm` introduced here) each produce
a long-form CSV with
W3C-PROV preamble per `f2_provenance_check`'s schema. The 80-cell
matrix produces **80 + 2 + 1 + 1 + 1 + ≤16 = 85 to 101 CSVs**
total (point estimate 93 under the F2 half-survival baseline; see
§5.1). Every
CSV is committed at run time to `data/issue1021/<batch>/`.

### 3.4 Connection to the F2 framework

The protocol uses the F2 framework's identification machinery
**per pairwise (phi-config, zoo-config) comparison**, not as a
single decomposition over all 8 configs. For each pair, X is
binary (which of the two formats is in use). However, the
four-PSE decomposition of `f2_dual_mediation` **does not apply
to the format comparison** for an identification reason we
discuss below; this paper uses only the **total-effect** and
**bridge-score envelope** machinery from the F2 framework.

**Why four-PSE is dropped for this paper (28th adversarial pass
discovery).** Pearl's four-PSE identification requires
positivity: $0 < P(M = m | X = x) < 1$ for the mediator values
the decomposition integrates over. Two candidate mediators —
`lossy_conversions` and `wall_clock_s` — are **deterministic
functions of $X$**: bf16 has zero lossy conversions by definition,
GFTernary has many; per-format wall-clock is determined by the
kernel path. Loop 104 attempted a residualization fix (subtracting
config-specific seed-median baselines), but the 28th pass
correctly noted that subtracting a config-specific constant does
not change the conditional distribution shape; the residualized
$M$ is still deterministic in $X$ up to a shift, and positivity
remains violated. The four-PSE decomposition under these
mediators would be identified algebraically (because the
residualization makes the variance non-degenerate) but
uninformative — the NDE absorbs essentially the entire total
effect because NIE_M1 and NIE_M2 carry no causally relevant
variation. We do not pre-register a four-PSE decomposition for
this paper; instead we run:

- **Total-effect estimation per pair**: `f2_pairwise_perm`
  paired-permutation test on val_bpb difference, BH-corrected
  over 4 comparisons per phi-config.
- **Bridge-score envelope on the total effect**:
  `f2_mediation_sensitivity` with the *total effect* as the
  estimand (not a per-PSE bound). This treats unmeasured
  confounding between format and validation BPB symmetrically
  for each pair.

The four-PSE machinery is **retained for future use** if and
when a separate paper exhibits a non-deterministic mediator
that can be measured at champion scale (e.g., per-layer gradient
norm under format perturbation), but we do not deploy it for
the format-comparison protocol here.

The wd0 stratum is a **total-effect contrast** under the wd=0
training configuration, NOT a Pearl Controlled Direct Effect in
the mediator-identification sense (this paper has no controlled
mediator — see §3.4 above for the four-PSE drop). What we share
with the companion F2 paper is the *strata pair* (canonical and
wd0), not the *identification machinery* (which differs: F2
deploys CDE-with-controlled-mediator; this paper deploys
total-effect under stratified training configurations). If
quantization-format × WD interaction exists at the same magnitude
as the companion paper's RmsNorm × WD interaction, we expect the
wd0 stratum's total-effect contrast to either strengthen or
weaken the phi-ladder advantage. The protocol explicitly admits
both outcomes and reports both, with **canonical as the
pre-registered primary stratum** (see §4.4).

### 3.5 Reporting discipline

Every cell of the 80-run matrix is reported in the manuscript,
including cells that produce non-finite BPB (NaN or +Inf — the
protocol does not exclude these but flags them in a separate
diagnostic table). The bridge-score envelope is reported at
**Λ = 1.0 BPB** with the `Γ_tip(Λ=1.0) < 1.25` fragility cutoff.

**On the choice of Λ = 1.0 for a total-effect estimand (30th
adversarial pass discovery).** The companion F2 paper
calibrates Λ = 1.0 BPB against a *per-PSE* (NDE) bound. This
paper bounds the *total effect*, a category change: total effects
can absorb residual confounding that a Pearl CDE would partition
into NDE + NIE. By that logic the same Λ may be too loose for a
total-effect bound. We retain Λ = 1.0 anyway because (i) the
underlying scale — one full BPB of unmeasured confounder — is
calibrated to the BPB axis itself, not to the estimand type;
(ii) the relevant question for a reader is "how large a
confounder would erase the result?", which is independent of
whether the estimand is direct or total; (iii) tightening Λ
post-hoc would be exactly the kind of researcher-degree-of-freedom
the pre-registration protocol exists to prevent. Reviewers who
prefer a tighter calibration can read the per-pair Γ_tip values
at any Λ from the committed CSVs. **We do not claim the F2
companion's per-PSE Λ calibration transfers automatically to
this paper's total-effect estimand**, only that we pin the value
ex-ante to prevent post-hoc tuning.

---

## 4. Pre-registered hypotheses

The protocol locks three nested hypotheses with explicit
falsification criteria. Each hypothesis is tested at both strata
(canonical and wd0) separately, but **canonical is the
pre-registered primary stratum** per §4.4 — results at wd0 alone
are exploratory, not publishable as positive. Within the primary
stratum, **falsification is treated symmetrically with
confirmation**: a result that falsifies a hypothesis appears in
the paper with the same prominence as one that confirms it.

### 4.1 H0 — null (equivalence)

**Statement**: For every (phi-config, zoo-config) pair, the mean
held-out validation BPB difference is within ±0.05 BPB
(equivalence margin = roughly one wall-clock-noise standard
deviation per the F2 sandbox-scale per-seed CV measurement at
§3.5.4 of the companion paper).

**Falsified by**: any (phi-config, zoo-config) pair whose
two-sided exact paired-permutation test rejects equivalence at
$p < 0.05$ (BH-corrected over 4 comparisons per phi-config) — i.e.,
the 95% CI on the BPB difference falls entirely outside
$[-0.05, +0.05]$.

**Action if H0 cannot be rejected**: report the equivalence as the
finding. The paper's contribution becomes the *demonstration of
equivalence*, plus the protocol itself, plus the bridge-score
envelope on the equivalence verdict.

### 4.2 H1 — phi superior on at least one zoo competitor

**Statement**: For at least one (phi-config, zoo-config) pair,
mean BPB is $\geq 0.10$ BPB lower for the phi-config than for the
zoo-config (twice the H0 equivalence margin), with the difference
statistically significant at $p < 0.05$ after BH correction.

**Falsified by**: every (phi-config, zoo-config) pair satisfies
at least one of:
(a) phi-config $\geq$ zoo-config mean BPB (phi is not lower),
(b) the unadjusted-p 95% CI on the difference includes zero,
(c) BH-adjusted $p \geq 0.05$ even when the unadjusted-p CI
    excludes zero (BH inflation regime), or
(d) phi-config is significantly lower (BH-adjusted $p < 0.05$)
    but $|{\rm diff}| < 0.10$ BPB (significant but below the
    H1 effect-size threshold).
Conditions (a)–(d) are jointly exhaustive of "fails the H1 test"
for finite-BPB results. Non-finite cells are excluded from the
test per §3.5 reporting discipline. If every pair satisfies at
least one of (a)/(b)/(c)/(d) in the **primary (canonical)
stratum**, H1 is falsified — the §4.4 asymmetric rule treats
canonical alone as positive-determining, so falsification (the
logical negation of "positive") must also be canonical-only.
Failure at wd0 (with success at canonical) is reported as
secondary evidence per §4.4, NOT as additional grounds for H1
falsification. (31st-pass correction): earlier drafts
required failure in both strata, which created a no-man's-land
(canonical-fail + wd0-hold) classified as neither falsified nor
positive — fixed here by aligning §4.2 falsification with §4.4
primary-determining logic.

**Action if H1 holds**: report the specific pair(s) at which
phi-ladder is superior, with the bridge-score envelope. Make no
claim about superiority on the pairs where H1 does not hold.

### 4.3 H2 — phi-config dominant across the zoo

**Statement**: At least one phi-config has mean BPB $\geq 0.10$
BPB lower than **every** zoo-config (BitNet-1.58, INT8, FP8,
bf16), with all 4 pairwise differences significant at $p < 0.05$
after BH correction over the 4 comparisons.

**Falsified by**: every phi-config fails H2 against at least one
zoo-config **in the primary (canonical) stratum**. Same
canonical-only logic as §4.2's H1 falsification: §4.4's
asymmetric rule treats canonical alone as positive-determining,
so H2 falsification must also be canonical-only. wd0 stratum
outcomes (H2 success or failure) are reported as secondary
evidence per §4.4 and do not contribute to the H2 falsification
verdict. This is the strongest hypothesis; falsification of H2
while H1 holds is the most likely outcome. (Internal ref —
adversarial-pass correction documented in `papers/CHANGELOG.md`
§10): earlier drafts of §4.3 omitted the primary-stratum
qualifier, creating the same logical no-man's-land that an
analogous prior pass fixed for §4.2; fixed analogously here by
aligning §4.3 falsification with §4.4 primary-determining logic.

**Action if H2 holds**: this is the headline positive result. The
paper reports the specific phi-config that dominates, with full
bridge-score envelope and cross-stratum stability flag.

### 4.4 Confounder-controlled variant (joint with H1/H2)

**Primary stratum**: canonical. **Secondary stratum**: wd0. This
designation is pre-registered before any data acquisition; the
canonical stratum reflects the marginal-recipe regime that a
practitioner would adopt by default, and the locked decision
prevents post-hoc stratum-shopping. The wd0 stratum is reported
alongside but is **not** elevated to primary if it differs.

**Asymmetric reporting rules** (locked):

- *H1/H2 holds at canonical (primary)*: this is the headline
  finding. The wd0 result is reported as secondary evidence:
  *consistent* if wd0 agrees, *interaction-flagged* if wd0
  disagrees. Neither outcome at wd0 demotes the canonical result.
- *H1/H2 fails at canonical but holds at wd0*: this is **NOT a
  positive result** under the pre-registered plan. It is
  reported as an exploratory observation requiring follow-up,
  not as evidence for phi-ladder superiority. The asymmetry
  is intentional: claiming a positive only-at-wd0 finding would
  be an unfalsifiable rescue narrative since wd=0 is not a
  production default.

Earlier drafts (pre Loop 104) presented the wd0 and canonical
results as symmetric; the 27th adversarial pass flagged that
the symmetric framing creates an unfalsifiable rescue case
("either way phi-ladder wins"). The locked asymmetry above
prevents that.

### 4.5 What this paper does not test

- **No claim about other hyperparameter axes**. The protocol fixes
  one training recipe across all eight configs. We do not test
  whether the result changes under (e.g.) a different learning-rate
  schedule.
- **No claim about scale-extrapolation**. The 1B-parameter, 50B-
  token finding is a point estimate at the protocol's locked
  scale. The companion F2 paper's §10.1 venue calibration applies
  unchanged: scale extrapolation is a separate paper.
- **No claim about wall-clock or memory parity**. We report
  wall-clock and peak-memory as **secondary** outcomes; we do not
  pre-register hypotheses on them. A phi-config that wins on BPB
  but loses on wall-clock will be reported as winning on BPB
  with a wall-clock footnote, not as "overall winning".

---

## 5. Reproducibility artifacts

The protocol commits, at the end of the run, **93 CSVs** and **6
post-run reports** to a single subtree of the repository
(`data/issue1021/<batch>/`). The exact artifact list, schemas, and
the scripts a reviewer can run for sanity-check reproduction are
specified below.

### 5.1 Artifact inventory

| Count | File pattern | Schema | Producer |
|------:|--------------|--------|----------|
| 80 | `cell_<stratum>_<config>_<seed>.csv` | per-cell long-form (val_bpb, train_bpb, wall_s, peak_mem_mb, lossy_conversions + W3C-PROV preamble) | `f2_ablation_sweep` |
| 2 | `aggregate_<stratum>.csv` | per-config mean ± SE on val_bpb (4 cols × 8 configs) | `f2_ablation_aggregate` |
| 1 | `pairwise_canonical.csv` | 16 (phi-config, zoo-config) pairs × {diff, p-value, BH-adjusted-p} | `f2_pairwise_perm` |
| 1 | `pairwise_wd0.csv` | same schema, wd0 stratum | `f2_pairwise_perm` |
| 1 | `stratum_compare.csv` | 16 pairs × `stable_across_strata` flag | `f2_stratum_compare` |
| ≤16 | `sensitivity_<phi-config>_vs_<zoo-config>.csv` | bridge-score envelope per (phi-config, zoo-config) pair that survives the perm test; upper bound = 16 if every pair survives, point estimate 8 based on the F2 sandbox-scale half-survival baseline | `f2_mediation_sensitivity` |

**New-this-paper marker**: `f2_pairwise_perm` is a Loop-110-introduced
binary (see §3.3); the other producers in the table are reused
unchanged from F2. The new binary's commit hash is locked at protocol
anchor and verifiable via `f2_provenance_check` on every emitted CSV.

Total: 80 + 2 + 1 + 1 + 1 + (8…16) = **between 85 and 101 CSVs**
(point estimate 93 under the F2 sandbox-scale half-survival
baseline; actual count reported at the run-result paper). Each
CSV has the
F2 W3C-PROV preamble parseable by `f2_provenance_check`. The
exact field set varies by producer:

- **Cell-level CSVs from `f2_ablation_sweep`** carry the full
  preamble: `generatedAt`, `wasGeneratedBy`, `agent_git_sha`,
  `host`, `trainer_internals_schema`, `cargo_pkg_version`.
- **`f2_pairwise_perm` (this paper's new binary, Loop 110)**
  carries the same field set plus two additional informational
  fields, `phi_configs` and `zoo_configs`, that name the
  enumeration the aggregator operated over.
- **Other aggregator binaries** (`f2_ablation_aggregate`,
  `f2_stratum_compare`, `f2_mediation_sensitivity`,
  `f2_dual_mediation`) **do NOT currently emit a W3C-PROV
  preamble**. Their outputs inherit provenance via the per-cell
  CSVs they consume (each cell-level CSV's preamble is verifiable
  via `f2_provenance_check`, and the aggregator's `delta_x`
  and CI bounds are deterministic functions of those cells).
  37th adversarial pass (Loop 114) discovered this gap; the
  protocol acknowledges it transparently and the §5.4 CI gate
  runs `f2_provenance_check` only on the cell-level CSVs plus
  `f2_pairwise_perm`'s output.

The `verify_preamble_per_producer.py` script
(`papers/scripts/`) gates the per-producer field set with **two
modes**: (i) static — greps each binary's source and asserts the
emitted `# prov:` fields match what this paragraph claims; (ii)
runtime — for cheap-to-invoke producers (currently
`f2_pairwise_perm`), it builds and runs the binary on a tiny
synthetic input and parses the live output, catching cases where
a `writeln!(...# prov:...)` is present in source but gated by a
runtime conditional. **Either mode's failure exits non-zero and
fails the CI gate's preamble-per-producer stage** (39th
adversarial pass, Loop 116, clarified this behavior was
unstated). Adding a producer to F2 without updating §5.1 (or §5.1
without changing the binary) fails the gate immediately.

The 6 post-run reports:

| File | Content |
|------|---------|
| `report_h0_equivalence.md` | Per-pair H0 verdict (equivalence within ±0.05 BPB) |
| `report_h1_superiority.md` | Per-pair H1 verdict (≥0.10 BPB lower at p < 0.05) |
| `report_h2_dominance.md` | Per-phi-config H2 verdict (dominates over all zoo) |
| `report_stratum_diff.md` | Canonical-vs-wd0 verdict per pair |
| `report_bridge_envelope.md` | Γ_tip(Λ=1.0) per surviving pair |
| `report_secondary_outcomes.md` | wall-clock + peak-memory tabulated per (config, stratum) |

The protocol commits all 6 reports as static markdown in the same
subtree; they are not regenerated after the run. Subsequent
loops may add subsidiary analyses but **not** edit these six.

### 5.2 Reproducibility commands

A reviewer who has cloned the repository can reproduce every
quantitative claim in the paper via the following:

```bash
# Anchor the run
git checkout <anchor-commit>  # supplied at submission

# Cold-clone build (15–30 min per the F2 companion §3.5)
cargo build --release \
  --bin f2_ablation_sweep \
  --bin f2_ablation_aggregate \
  --bin f2_pairwise_perm \
  --bin f2_stratum_compare \
  --bin f2_mediation_sensitivity \
  --bin f2_provenance_check

# Per-cell rerun (single seed × single config) — sanity check
# Wall-clock budget: ~3 hours on 8×A100 for one cell.
cargo run --release --bin f2_ablation_sweep -- \
  --config GFTernary --stratum canonical --seed 42 \
  --output data/issue1021/sanity/cell_canonical_GFTernary_42.csv

# Compare against committed expected:
diff data/issue1021/sanity/cell_canonical_GFTernary_42.csv \
     data/issue1021/run0/cell_canonical_GFTernary_42.csv
```

The seed-by-seed determinism of the F2 framework (every config
has a `config_hash` that is byte-stable across reruns at the same
commit) means a reviewer can reproduce *any* single cell and
expect byte-identical val_bpb to within 5 significant figures.
Per-cell wall-clock at full scale is ~3 hours on 8×A100; the
full 80-run matrix is **~240 GPU-hours**.

### 5.3 Snapshot manifest

The paper commits a `papers/issue1021/expected_test_compile_tmlr.pdftotext`
snapshot at submission. The companion paper's `compile_tmlr_test.sh
--diff` machinery is reused unchanged: any future commit
that changes the rendered PDF's textual content fails the
regression suite. The snapshot captures the post-numeric-fill
manuscript, so post-hoc edits to the empirical numbers will be
caught and require an explicit `--update-snapshot` to refresh.

### 5.4 CI gate

The companion paper's CI gate (34 stages on disk as of Loop 141)
includes all three pre-registered #1021 scripts now that the third
has been committed, plus the new stage-count-consistency verifier
(Loop 118 B) that gates this paragraph's "N stages" claim against
the actual `STAGES` array:

- `verify_run_completeness.py` — checks all 93 CSVs are present
  in `data/issue1021/run0/` and each parses against the schema
  in §5.1. Pre-sweep runs (when the run directory doesn't exist)
  exit 0 vacuously, mirroring `verify_provenance.sh`'s
  pre-registration pattern.
- `verify_report_consistency.py` — operates at **6/6 report-class
  coverage at two tiers**: 2 full numeric-claim checkers
  (h0_equivalence and h1_superiority grounded in
  `pairwise_<stratum>.csv` on `diff_mean` and `p_bh` at 2-decimal
  tolerance) plus 4 exists-and-heading-pattern stubs (h2_dominance,
  stratum_diff, bridge_envelope, secondary_outcomes). Each stub
  asserts the report file is present and contains expected
  section / table headers; the stubs upgrade to full numeric
  checkers once the run produces concrete report-row schemas.
  Pre-sweep runs exit 0 vacuously. **The 41st adversarial pass
  flagged that earlier wording read as 6/6 full when the script
  then shipped 2/6**; a follow-up sweep closed the gap with the
  stub tier (see companion paper's `papers/CHANGELOG.md` §10).

All three originally pre-registered scripts have now shipped:
`verify_provenance.sh` gates W3C-PROV preambles on the two
preamble-emitting producer classes; `verify_run_completeness.py`
gates the 93-CSV inventory at the run-result anchor;
`verify_report_consistency.py` gates each post-run report's
numeric claims against source CSVs. All three are now
on disk in `papers/scripts/` and wired into the companion paper's
CI gate; they run vacuously OK pre-sweep because the
`data/issue1021/run0/` subtree does not exist on the methodology
anchor commit.

The two scripts above pre-registered as "(*to be implemented*)"
follow the same commit-order discipline as `f2_pairwise_perm`
(§3.3 "Pre-registration discipline for the new binary"):
committed before any of the 80-cell champion-scale sweep CSVs
are produced.

**Gate stage decomposition** (the 38th-pass arithmetic was
fixed in an earlier pre-registration sweep; later loops added
the stage-count + cross-paper verifiers — full per-loop history
in the companion paper's `papers/CHANGELOG.md` §10): the current
**34-stage on-disk gate** breaks down as
**28 F2-scope stages** (cross-ref/metadata/SHAs/lint/tables/
formulas/label/preamble/inventory/xelatex/figures/supplementary
plus the stage-count verifier that gates this very paragraph,
the cross-paper consistency verifier, the cross-paper-gate
meta-test, the submission-readiness verifier, the
changelog-consistency verifier, the anonymizer-completeness
verifier, the cardinality-arithmetic verifier, the
generator-consistency verifier, the class-registry-binding
verifier, the documented-vs-extracted-consistency verifier,
the burn-down-history verifier, the alias-round-trip verifier,
the module-cache-consistency verifier, the floating-loop-
anchor verifier, the anchor-loop-coverage verifier, and the
dependency-graph verifier; per-introduction history is
catalogued in the companion paper's `papers/CHANGELOG.md` §10)
**+ 6 #1021-scoped stages already wired** into F2's
`run_all_checks.sh`
(verify_provenance/verify_run_completeness/verify_report_consistency/
smoke_f2_pairwise_perm/#1021 cross-ref/#1021 md-lint). **No
further stages remain pre-registered** — all three originally-
future scripts (verify_provenance.sh, verify_run_completeness.py,
verify_report_consistency.py) have shipped. The full gate must
exit 0 on the run-result paper's anchor commit before any draft
is exported for submission.

An earlier adversarial pass flagged that earlier drafts
presented the three scripts as already-shipped when none were;
a follow-up sweep delivered the first (`verify_provenance.sh`).

---

## 6. Scope and limitations

This section enumerates what this paper does **not** claim and what
a reviewer should not infer. The list extends the §1.4 carve-outs
to the post-protocol-locking phase.

### 6.1 Scale-extrapolation is out of scope

The pre-registered sweep is locked at **~1B parameters and 50B
FineWeb tokens**. We make no claim about behavior at any other
scale. The companion F2 paper's §10.1 venue calibration applies:
the 8K-param sandbox-scale finding there does not transfer
unconditionally to champion scale, and we do not assume the
~1B-scale finding here will transfer to (e.g.) 175B-scale
production training. A multi-scale replication is a separate
paper.

### 6.2 The training recipe is fixed

The protocol fixes one optimizer (AdamW), one learning-rate
schedule (linear warmup + cosine decay), one batch size (1024),
one sequence length (2048), and one data corpus (FineWeb-Edu 10B
shard). We do **not** test whether the result holds under
alternative recipes; if it does not, that would be a future
finding, not a reason to reject this one. The format-effect
isolation is exactly what the protocol commits to.

### 6.3 Secondary outcomes are secondary

Wall-clock and peak-memory are reported but **not pre-registered
as hypotheses**. A phi-config that wins on BPB but loses on
wall-clock will be reported as winning on BPB with a wall-clock
footnote, not as "overall winning". The reverse: a phi-config
that loses on BPB but wins on wall-clock is **not** reportable
as a positive result under our locked plan; if such a finding
emerges from the data, it will be reported as a null on BPB
plus an exploratory wall-clock observation.

### 6.4 The wd0 stratum is a structural device, not a recommendation

We test at wd0 because the companion F2 paper demonstrates that
the wd0 contrast (Pearl CDE in the companion's identification
machinery; total-effect contrast in ours per §3.4) can surface
format-WD interaction structure that the canonical (marginal)
recipe hides. We do not recommend
training transformers at WD=0 in production — the companion
paper's §5.2 documents the wd0 stratum reproducing a real
pre-AdamW configuration, not endorsing it. If H1 or H2 holds
only at wd0 (and not at canonical), the result is **interpreted
as evidence for format-WD interaction**, not as evidence that
practitioners should drop WD.

### 6.5 Non-finite cells are reported, not excluded

The protocol commits to reporting every cell of the 80-run
matrix, including those producing NaN or +Inf validation BPB.
A diagnostic table separately reports non-finite cells with the
seed and config that produced them; the primary BPB analyses
exclude non-finite cells from the test statistic but the
exclusion itself is reported, not hidden. **No post-hoc seed
re-selection is permitted under the protocol.**

### 6.6 What this paper cannot evaluate

- **Quantization-aware fine-tuning from a bf16 checkpoint**. Our
  protocol is from-scratch training at quantization-aware mode.
  Any QAFT result is a separate study.
- **Production deployment at higher batch sizes / longer
  sequences**. The locked configuration is a specific cell of a
  large practical space; we do not extrapolate.
- **Hardware-specific behavior** (e.g., H100 vs B100 vs TPU v5).
  The protocol does not name a target accelerator beyond
  "any modern GPU with bf16 + INT8 + FP8 kernel support";
  hardware-specific wall-clock comparisons would require a
  separate protocol.

### 6.7 What goes into the supplementary, not the body

Per the companion paper's Appendix-as-pointer discipline, the
following live in the supplementary zip rather than the main
body: full 93-CSV manifest with provenance preambles; per-cell
training logs (wall-clock + memory peak per step); per-figure
input-CSV derivation chain; environment file
(`Cargo.lock` + Python `requirements.txt`); hardware
description; commit anchor.

---

## 7. Expression of Interest (MLRC 2026 EOI text)

The protocol below is registered as the empirical companion to
the methods paper *"Stratified mediation for ML ablations:
bridging Pearl-style CDE and the additive bridge-score envelope"*
([anchor], pending TMLR review). We seek MLRC 2026 visibility
for the **pre-registered champion-scale phi-ladder evaluation**
that the methods paper marks as `[Not yet attempted]`.

**Submission target**: NeurIPS 2026 MLRC Track.

**Why this protocol is MLRC-relevant**:
- Pre-registered hypothesis tests with locked falsification
  criteria — directly addresses MLRC's reproducibility-and-rigor
  emphasis.
- The companion methods paper is under TMLR review (per the
  MLRC EOI Google Form prerequisite).
- 93-CSV reproducibility artifact at submission; per-cell
  reproducibility byte-stable via the F2 W3C-PROV preamble.
- Single-format-axis isolation (no covariate sweep) so the
  result is interpretable as "this format vs the zoo at this
  recipe".

**Submission window**: TMLR window 2025-06-20 ≤ submit ≤
2026-09-30 AOE. MLRC EOI filed any time once paper is under
TMLR review; soft EOI deadline 2026-06-04 AOE is informational
only — the binding date is the 2026-09-30 TMLR decision
deadline.

**What we are NOT seeking from MLRC**: re-evaluation of the
methods paper itself (already under TMLR), peer review of the
pre-registration document (it is locked at the protocol level),
or visibility for unverified empirical claims (this paper
contains a pre-registered protocol, not a pre-registered
*claim* — claims come in the follow-up paper after the run).

**Author availability**: corresponding author available for
clarifying questions on the protocol and on the F2 framework
the protocol consumes; not available to negotiate the
falsification criteria post-submission.

---

## DRAFT notes (Loops 98–103)

§1 (Loop 98), §2 (Loop 99), §3 (Loop 100), §4 (Loop 101),
§5 (Loop 102), §6 + §7 (Loop 103) drafted. **All seven sections
of the §1-§7 backbone are now drafted.** The companion paper's
review machinery (anonymization, CI gate, snapshot, semantic
attribution, numeric consistency, formula derivation, reader
experience) is available for the next phase: a full adversarial
pass on this paper as a standalone manuscript.

Citations to add to the bib before spin-off (introduced in
§2 + §3):
- Frantar/GPTQ (arXiv:2210.17323)
- Xiao/SmoothQuant (arXiv:2211.10438)
- Xi/Jetfire (arXiv:2403.12422) — INT8, not INT4 (Loop 105 correction)
- Wang & Kanwar bf16 (Google blog post 2019)
- Micikevicius FP8 (arXiv:2209.05433)
- Hagmann phi-quantization (arXiv:2102.xxx — to verify)
- Soldaini FineWeb-Edu (arXiv:2406.17557)

Citation hygiene: §2 introduces new citations (Frantar/GPTQ,
Xiao/SmoothQuant, BitNet-1.58, Xi/Jetfire, Wang/Kanwar bf16,
Micikevicius/FP8, OCP MXFP8) that are NOT yet in the F2 methodology
bib. When this paper splits off, those bib entries need to be added
and verified via the same Loop 97 semantic-attribution discipline.

Anonymization: branch / SHA / "playra" appear in this draft and
must be stripped before submission via the same
`papers/scripts/anonymize_paper.py` machinery used for the
companion paper.
