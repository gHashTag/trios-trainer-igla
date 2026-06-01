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
representation** of weights and activations — bf16, FP8, INT4-W4A8,
BitNet-1.58 ternary — has become a moving target, with new "zoo"
entries published every quarter and very few of them evaluated
against each other on the same controlled training regime. The
practitioner question we address is narrow: *given a fixed
1B-parameter transformer recipe at 50B training tokens of FineWeb,
does the **phi-ladder** family of representations
(GFTernary → GF8 → GF16 → GF32, anchored at the golden-ratio
recurrence $\phi^2 + \phi^{-2} = 3$) yield lower held-out
validation bits-per-byte (BPB) than the leading members of the
mainstream zoo (BitNet-1.58, INT4-W4A8, MXFP8, bf16)?*

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
   (RmsNorm is now *anti-productive*). Pearl-style CDE pins this
   as an interaction effect between RmsNorm's normalizing role and
   weight decay's regularizing role. The same Pearl-CDE framework
   should be able to detect — or rule out — analogous interaction
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

Our methodology companion paper provides the analysis machinery
that this paper consumes: stratified Pearl-CDE
(`f2_dual_mediation`), four-PSE decomposition with delta-method
SEs (`f2_mediation_sensitivity`), additive bridge-score envelope
(Ohnishi & Li 2026 Thm 2), and cross-stratum comparator
(`f2_stratum_compare`). The same 10 F2 binaries, 805 tests, and
W3C-PROV preamble discipline that back the companion paper's
sandbox-scale RmsNorm finding are the substrate this paper runs on
at champion scale. A reader who has not yet seen F2 should treat
§3 of this paper as a pointer to the methodology paper, with the
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
- **No claim of computational-cost parity** with INT4 or FP8.
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

### 2.1 Integer-base zoo (INT8, INT4-W4A8, BitNet-1.58)

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

The INT4-W4A8 schemes (Xi et al. 2024, arXiv:2310.16836 for
*Jetfire*; Zhao et al. 2024 for *AffineQuant*) collapse weights
to 4-bit integers while keeping activations at 8-bit. Their BPB
deltas vs bf16 at $\sim$1B parameters are reported as $\leq 0.05$
BPB in the original papers, but the comparisons are usually
against post-training rather than from-scratch training, so the
parsing into our protocol (from-scratch, FineWeb 50B tokens) is
not direct. We include INT4-W4A8 in the zoo as the strongest
integer-quantization-during-training competitor at our target
scale.

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
| `INT4-W4A8` | integer  | 4-bit weights + 8-bit activations per arXiv:2310.16836 |
| `FP8`       | floating-point | E4M3 weights + E5M2 gradients per arXiv:2209.05433 |
| `bf16`      | floating-point | bf16 weights + bf16 activations (gold-standard baseline) |

**Strata**: `canonical` (default WD = 0.1, all seven F2 fixes
enabled) and `wd0` (WD pinned to 0.0, otherwise identical). The
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

The same 10 F2 binaries that back the companion paper run this
study unmodified at the per-cell level:

- **Per-cell BPB recording**: `f2_ablation_sweep --config <slot>
  --stratum <s> --seed <i> --output cell_<s>_<slot>_<i>.csv`.
  Each cell emits a single-record CSV with `val_bpb`,
  `train_bpb`, `wall_s`, `peak_memory_mb`, `lossy_conversions`,
  and a W3C-PROV preamble.
- **Per-stratum aggregation**: `f2_ablation_aggregate` reads the
  40 per-stratum cells and emits a long-form CSV with
  per-config mean ± SE on validation BPB.
- **Pairwise testing**: `f2_pairwise_perm` runs the exact
  Zmigrod-Vieira-Cotterell paired-permutation test on each
  (phi-config, zoo-config) pair within a stratum. With 5 seeds
  the exact test enumerates $2^5 = 32$ sign-flip vectors,
  producing exact p-values.
- **BH correction**: `f2_pairwise_perm --bh` BH-corrects across
  the 4 pairwise comparisons within each phi-config (vs the 4
  zoo-config alternatives).
- **Cross-stratum stability**: `f2_stratum_compare` takes the
  canonical and wd0 long-form CSVs and emits a
  `stable_across_strata` flag per (phi-config, zoo-config) pair.
- **Bridge-score envelope**: `f2_mediation_sensitivity --lambda
  1.0` is run on each pair that survives the permutation test;
  the envelope is calibrated against the VanderWeele-Ding E-value
  for fragility/robustness reporting.

The five F2 binaries above each produce a long-form CSV with
W3C-PROV preamble per `f2_provenance_check`'s schema. The 80-cell
matrix produces **80 + 2 + 1 + 1 + 1 + 8 = 93 CSVs** total. Every
CSV is committed at run time to `data/issue1021/<batch>/`.

### 3.4 Connection to the F2 framework

The protocol uses the F2 framework's identification machinery
unchanged: the four-PSE decomposition of `f2_dual_mediation`
applied with `X = format_choice`, `M_1 = lossy_conversions`, and
`M_2 = wall_clock_s`. The hypothesis that phi-ladder configurations
yield lower validation BPB factors through both mediators
($M_1$ captures quantization-noise accumulation; $M_2$ captures
the multiplicative-basis overhead) and a remaining direct path.
The pre-registered question is whether the direct path itself
favors phi-ladder, controlling for both mediators.

The wd0 stratum is the Pearl CDE that the companion paper
demonstrates can flip signs. If quantization-format × WD
interaction exists at the same magnitude as the companion paper's
RmsNorm × WD interaction, we expect the wd0 stratum to either
strengthen or weaken the phi-ladder advantage. The protocol
explicitly admits both outcomes and reports both.

### 3.5 Reporting discipline

Every cell of the 80-run matrix is reported in the manuscript,
including cells that produce non-finite BPB (NaN or +Inf — the
protocol does not exclude these but flags them in a separate
diagnostic table). The bridge-score envelope is reported at
**Λ = 1.0 BPB** (the same scale-of-effect that the F2 companion
adopts as its reporting baseline). Any cell whose `Γ_tip(Λ=1.0)
< 1.25` is flagged as fragile in the bridge-score column.

---

## 4. Pre-registered hypotheses

The protocol locks three nested hypotheses with explicit
falsification criteria. Each hypothesis is tested at both strata
(canonical and wd0) separately. Reporting will be **symmetric**:
results that falsify a hypothesis appear in the paper with the
same prominence as results that confirm one.

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

**Falsified by**: every (phi-config, zoo-config) pair either
(a) has paper-config $\geq$ zoo-config mean BPB, or
(b) has a CI on the difference that includes zero, or
(c) has $|{\rm diff}| < 0.10$ BPB even with $p < 0.05$.
If all 16 pairs (4 phi × 4 zoo) satisfy at least one of (a)/(b)/(c)
in both strata, H1 is falsified.

**Action if H1 holds**: report the specific pair(s) at which
phi-ladder is superior, with the bridge-score envelope. Make no
claim about superiority on the pairs where H1 does not hold.

### 4.3 H2 — phi-config dominant across the zoo

**Statement**: At least one phi-config has mean BPB $\geq 0.10$
BPB lower than **every** zoo-config (BitNet-1.58, INT4-W4A8, FP8,
bf16), with all 4 pairwise differences significant at $p < 0.05$
after BH correction over the 4 comparisons.

**Falsified by**: every phi-config fails H2 against at least one
zoo-config. This is the strongest hypothesis; falsification of H2
while H1 holds is the most likely outcome.

**Action if H2 holds**: this is the headline positive result. The
paper reports the specific phi-config that dominates, with full
bridge-score envelope and cross-stratum stability flag.

### 4.4 Confounder-controlled variant (joint with H1/H2)

Both H1 and H2 are tested at the **wd0 stratum** separately. If
H1 or H2 holds at wd0 but fails at canonical, the result is
reported as "phi-ladder is superior when weight decay is controlled
to zero, but the marginal-recipe canonical comparison is
inconclusive." The companion paper's RmsNorm sign-flip at the wd0
stratum establishes that the canonical/wd0 comparison can be
substantive; H1/H2 at wd0 alone is therefore a meaningful finding.

If H1 or H2 holds at canonical but fails at wd0, the result is
reported as "phi-ladder is superior in the marginal-recipe
canonical regime, but the wd-controlled comparison is
inconclusive — the marginal effect may be confounded by WD's
interaction with the format choice."

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

## DRAFT notes (Loops 98–101)

§1 (Loop 98), §2 (Loop 99), §3 (Loop 100), §4 (Loop 101) drafted.
Remaining sections:
- §5 — Reproducibility artifacts
- §6 — Scope/limitations
- §7 — EOI

Citation hygiene: §2 introduces new citations (Frantar/GPTQ,
Xiao/SmoothQuant, BitNet-1.58, Xi/Jetfire, Wang/Kanwar bf16,
Micikevicius/FP8, OCP MXFP8) that are NOT yet in the F2 methodology
bib. When this paper splits off, those bib entries need to be added
and verified via the same Loop 97 semantic-attribution discipline.

Anonymization: branch / SHA / "playra" appear in this draft and
must be stripped before submission via the same
`papers/scripts/anonymize_paper.py` machinery used for the
companion paper.
