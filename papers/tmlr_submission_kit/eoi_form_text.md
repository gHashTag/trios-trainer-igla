# MLRC 2026 "Intent to submit" — paste-ready form text

**Submission target**: OpenReview MLRC 2026 EOI form
(https://reproml.org/call_for_papers/), **soft deadline 2026-06-04
AOE**.

**How to use this file**: open the EOI OpenReview form, paste each
field below into the matching form field. The Abstract has been
revised in Loop 63 to reflect the §5 reframing as a unit-test
demonstration (per Loop 61 reviewer screen Top-1 closest-fix).

Final paper anchor commit: descendant of `5367bde` on
`f2-methodology` branch (currently `3077b0d` per Loop 62 push).

---

**Title**: F2: A Stratified Pearl-CDE Framework for Transformer
Training-Recipe Ablations, with a Sandbox Demonstration of
Suppression Mediation

**Track**: Reproducibility (MLRC 2026 official NeurIPS track)

**Abstract** (250 words target, revised Loop 63):
The standard ML ablation report tabulates seed-mean validation
metric across three to five seeds for each toggle of a training-
recipe fix, then infers the fix's effect. We argue this practice
systematically misattributes effects when the ablation target is
causally entangled with another training-recipe knob — a phenomenon
we call *suppression mediation*. We operationalize the Pearl
Controlled Direct Effect (CDE) at the mediator-disabled level as a
counterfactual ablation, complementing the marginal Natural Direct
Effect that seed-mean reporting tracks. We implement a two-mediator
framework over the Daniel et al. (2015) four-path decomposition
with delta-method standard errors valid at N=5, an additive
bridge-score envelope (Ohnishi-Li Theorem 2) for sensitivity, and
a cross-stratum stability flag. We demonstrate the framework on a
synthetic-counter sandbox: the canonical Natural Direct Effect of
replacing RmsNorm with LayerNorm is −4.12 BPB (helpful), but the
wd0 Pearl CDE is +0.43 BPB (harmful), with both CIs excluding
zero. We frame this as a unit-test demonstration that the
framework detects a sign flip when one is constructed — not as a
final claim about RmsNorm at champion scale, which is pre-
registered as Phase-1 follow-up. We additionally survey three
recent transformer-architecture ablation papers (NormFormer 2021,
BitNet b1.58 2024, Peri-LN 2025) and find that the
multi-seed-with-data-release norm is not yet established in this
literature, illustrating the methodological gap F2 is designed
to close. The framework is open-source under MIT (727 tests, 10
binaries sharing a long-form CSV contract); the six empirical
CSVs that back the sign-flip are committed in-repo with MD5
checksums and per-file reproduction commands.

**Reproducibility claim**: "Mechanical reproducibility for §5":
every numerical claim and every figure regenerates from the anchor
commit `5367bde` (or any descendant of `f2-methodology` branch) plus
the committed `data/loop49/` empirical CSVs. The reviewer-grade
reproducibility checklist is documented in §3.5.4 of the paper.
Champion-scale validation is explicitly pre-registered (`docs/F2_PRE_REG.md`)
and not part of this submission.

**TMLR action editor preferences**: any editor with prior experience
in causal mediation analysis OR ML reproducibility methodology.

**Conflicts of interest**: [to be filled by author at submission time]

**Dual submission status**: not submitted to NeurIPS 2026 Journal-to-
Conference track (per MLRC 2026 dual-submission policy).

---

## Reviewer-facing reproducibility statement (excerpted from §3.5.4)

> A reviewer wishing to reproduce any number in §5 should:
> 1. `git checkout 5367bde` (or any descendant on `f2-methodology`).
> 2. `cargo test --lib` exits 0 with 632 passing tests.
> 3. Run any figure script in `papers/figures/`; no flags needed.
> 4. Verify the input file's checksum against `data/loop49/README.md`.
> 5. The generated PNG should be visually identical to the figure in
>    the paper modulo matplotlib version.
> 6. The output JSONL first-record signature should match `Appendix B`.

---

## What this kit does NOT cover

- The actual TMLR submission flow on OpenReview.net (manual step).
- LaTeX conversion of `papers/f2_methodology.md` body (manual; the
  math is already in dollar-delimited LaTeX, but the prose, tables,
  and figure references need template-specific markup).
- Anonymization sweep: see `anonymization_checklist.md` for the items
  to strip / replace before upload.
