# MLRC 2026 "Intent to submit" — draft form text

**Submission to OpenReview MLRC 2026 EOI form (soft deadline 2026-06-04 AOE).**

This is draft text only. Final form text should be reviewed against
the actual OpenReview form fields when the submitter is ready to
file. Form fields below are inferred from the MLRC 2026 call for
papers (https://reproml.org/call_for_papers/).

---

**Title**: F2: Stratified Pearl-CDE Mediation Analysis for Transformer
Training-Recipe Ablations

**Track**: Reproducibility

**Abstract** (250 words target):
The standard ML ablation report tabulates seed-mean BPB ± standard
deviation across three to five seeds for each fix toggle, then infers
the fix's effect. We argue this practice systematically misattributes
effects when the ablation target is causally entangled with another
training-recipe knob — a phenomenon we call suppression mediation. We
operationalize the Pearl Controlled Direct Effect (CDE) at the
mediator-disabled level as a counterfactual ablation, complementing
the marginal NDE that seed-mean reporting tracks. We implement a
two-mediator framework over the Gao-Li-Luo four-path decomposition
with delta-method standard errors valid at N=5, an additive
bridge-score envelope (Ohnishi-Li Theorem 2) for sensitivity, and a
cross-stratum stability flag. We demonstrate the framework on a small
sandbox ablation matrix: the canonical Natural Direct Effect of
replacing RmsNorm with LayerNorm is −4.12 BPB (helpful), but the
wd0 Pearl CDE is +0.43 BPB (harmful) — both CIs exclude zero. The
result is replicated under an alternative mediator parameterization.
The framework is open-source under MIT (727 tests, 10 binaries
sharing a long-form CSV contract), and the six empirical CSVs that
back the sign-flip are committed in-repo with MD5 checksums and
per-file reproduction commands.

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
