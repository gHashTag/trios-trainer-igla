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

## DRAFT notes (Loop 98)

This is a §1-only draft. Sections to write in subsequent loops:
- §2 — Related work (3 sub-sections, one per format family)
- §3 — Protocol (sources from `docs/F2_PRE_REG.md` §3-§4, expanded
  with the F2-companion connection points)
- §4 — Pre-registered hypotheses with falsification tests
- §5 — Reproducibility artifacts
- §6 — Scope/limitations
- §7 — EOI

Citation hygiene: all citations above already appear in the F2
methodology paper's bib at
`papers/tmlr_submission_kit/f2_methodology.bib`; the future
follow-up paper will share that bib + add Issue #1021 setup-
specific entries.

Anonymization: branch / SHA / "playra" appear in this draft and
must be stripped before submission via the same
`papers/scripts/anonymize_paper.py` machinery used for the
companion paper.
