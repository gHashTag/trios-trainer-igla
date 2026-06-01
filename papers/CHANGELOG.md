# F2 paper changelog — Loops 28–73

This file curates the meaningful contributions across the
~30 development loops of the F2 methodology paper, grouped by theme.
Intended for reviewers, future maintainers, and camera-ready
disclosure.

The commits themselves live on the `f2-methodology` branch
(PR #185). This document is a higher-level reading guide — not a
substitute for `git log`.

## Themes

### 1. Framework + binaries (Loops 28–50, foundation)

The framework's load-bearing infrastructure was built across the
early loops:

- Pearl-style stratification mechanism (`race::ablation::Stratum`
  enum: `Canonical`, `Wd0`, `Warmup0`) with `ModeKind` × `Stratum`
  registry that auto-derives the nine mode strings tagging every
  emitted row.
- Daniel et al. 2015 / Gao-Li-Luo 2020 four-PSE decomposition with
  delta-method standard errors valid at N=5 seeds.
- Ohnishi-Li 2026 additive bridge-score sensitivity envelope.
- 10 F2 binaries sharing a stable long-form CSV contract
  (`docs/F2_BINARIES.md`).
- Two named regression locks:
  `dual_mediation_no_interaction_residual_lock` and
  `trainer_internals_schema_is_load_bearing`.

### 2. Empirical findings (Loops 49, 64, 68)

Two sandbox-scale findings anchor §5 of the paper:

- **Headline (Loop 49)**: RmsNorm × WD sign-flip. Under canonical
  mediation the NDE for replacing RmsNorm with LayerNorm is
  −4.12 BPB (apparently helpful); under wd=0 Pearl CDE the same
  estimand is +0.43 BPB with CI excluding zero (intrinsically
  harmful to remove). Framed as a unit-test demonstration that the
  framework detects a sign flip when one is constructed.
- **Secondary (Loop 64)**: swap-parameterization invariant.
  Under (M_1 = rms, M_2 = warmup), the rms-mediated NIE for
  X = wd is byte-identical −0.751 [−1.325, −0.177] across all
  three strata.
- **Robustness (Loop 68)**: full 5 × 3 (M_2 × stratum) sweep
  confirms the invariance is conditional on M_2 matching the
  warmup0 stratum's pinned variable. Under the four alternative
  M_2 choices, the weaker canonical = wd0 invariance still holds
  in every case — a structural consequence of wd being target X.

### 3. Paper sections (Loops 51–62, 66, 70, 71)

- Loop 51: initial outline + `docs/F2_PRE_REG.md` pre-registration.
- Loop 55–56: citation hygiene pass + §1/§2/§8 polish from bullets
  to prose.
- Loop 58: §7 limitation #6 on post-treatment confounding (Rudolph
  & Díaz 2023, Hong et al. 2023, Díaz et al. 2021).
- Loop 60: §3.2 primary citation switched to Daniel et al. 2015
  Biometrics (foundational two-mediator decomposition); Gao-Li-Luo
  retained as no-interaction-reduction companion.
- Loop 62: honest §5.3 rewrite (removed unanchored "byte-identical
  across all 3 strata" claim) + §5.2 three-point defense of wd=0
  as a meaningful counterfactual.
- Loop 66: Figure 5 swap-parameterization heatmap.
- Loop 70: title polish, threshold-tier typo fix, test-count refresh.
- Loop 71: §5.3 robustness paragraph promoted to §6.4 alongside
  the other sensitivity subsections.

### 4. Figures (Loops 51–69)

Six figures, all regenerable in one shot from committed CSVs:

- Figure 1 — RmsNorm NDE sign flip across strata (Loop 51).
- Figure 2 — Stratum × ModeKind architecture diagram (Loop 53).
- Figure 3 — Canonical 5 × 4 PSE heatmap (Loop 52).
- Figure 4 — Γ_tip(Λ) hyperbolae for rms PSEs (Loop 53).
- Figure 5 — Phase 0 swap NIE_M1 heatmap (Loop 66).
- Figure 6 — Full M_2 robustness grid (Loop 69).

### 5. Reproducibility tooling (Loops 56–73)

Five auxiliary scripts under `papers/scripts/`:

- `generate_appendix_d.sh` (Loop 56) — enumerate every test.
- `cross_reference_audit.py` (Loop 59) — paper-internal ref check.
- `compile_tmlr_test.sh` (Loop 61) — xelatex build verify;
  extended to 3 variants (non-anon / anon / real TMLR class) in
  Loop 73.
- `figure_regen.sh` (Loop 65) — 6-figure regen from CSVs.
- `verify_paper_metadata.py` (Loop 71) — 4-check drift gate.
- `run_all_checks.sh` (Loop 73) — single-shot CI gate chaining
  all five plus `pack_supplementary.sh`. ~30 s wall.

### 6. Submission machinery (Loops 60–67)

- Loop 60: initial TMLR kit (`papers/tmlr_submission_kit/`) with
  `template.tex` skeleton, EOI form text, manifest, anonymization
  checklist, `pack_supplementary.sh`.
- Loop 61: 22-entry BibTeX file (`f2_methodology.bib`).
- Loop 67: `anonymize_paper.py` produces
  `papers/f2_methodology_anonymized.md`; sanity-checks zero
  identifying strings remain.
- Loop 68: real TMLR class compile via official `tmlr.sty` +
  `tmlr.bst` from `github.com/JmlrOrg/tmlr-style-file`.
- Loop 72: `pack_supplementary.sh` 3-stage pre-flight (figures →
  provenance → metadata).
- Loop 73: 3-variant compile + 6-stage CI gate.

### 7. Reviewer-screen feedback loop (Loops 59, 61, 75–86)

Independent adversarial reviews surfaced load-bearing issues
caught before reviewers saw them. **Eleven** independent passes:

- **Loop 59** derivation audit: re-attributed Miles-Shpitser
  citation (5 authors, not 2), explicit Γ/Λ uniform-scalar
  reduction, removed unsupported smoking-cancer benchmark
  comparison.
- **Loop 61** hostile-reviewer screen: 3 desk-reject risks
  (synthetic counter framing, wd=0 pathological-state
  counter-reading, MLRC reproducibility-track fit). All three
  closed by Loops 62, 62, 63 respectively.
- **Loop 75** anchor SHA cleanup: replaced `5367bde` (Loop 55, no
  empirical CSVs committed) with `583b417` (Loop 68, full data
  anchor).
- **Loop 76** 8-issue sweep: §6.1 ↔ §5.3 self-contradiction
  (closest single fix); Zhao-Luo → Daniel et al. bulk rename;
  EOI count drift; abstract acronym expansion.
- **Loop 77** Vaswani/Loshchilov verification; "Daniel et al.'s
  identification" possessive cleanup.
- **Loop 79** AblationBench mis-description caught (paper is about
  LM-agent ablation *planning*, not Welch/Cohen's-d analysis).
- **Loop 80** four citation hygiene catches in one pass: RO-Crate
  authorship, InfiR2 paper withdrawn, Fibbinary author
  fabrication, QuEST title correction.
- **Loop 82** submission-flow correction: MLRC EOI is a Google
  Form (not OpenReview); requires prior TMLR submission.
- **Loop 83** 9 additional EOI/OpenReview leaks across docs.
- **Loop 84** SUBMISSION_CHECKLIST decision-tree fix.
- **Loop 85** PDF visual inspection: TWO SEV-5 rendering bugs
  invisible in 84 prior loops because the PDF was never opened.
  HTML anonymizer banner rendering as prose in abstract; math
  symbols Γ/Λ/Δ rendering as literal `\{}Gamma` text. Both fixed
  in the converter; `compile_tmlr_test.sh` extended with a
  pdftotext-grep stage to catch this class permanently (Loop 86).

### 7.5 Aesthetic finalization arc (Loops 84–89)

The Loop 84-89 cycle was an intensive 6-loop PDF rendering /
aesthetic / structural cleanup phase. Each loop ran adversarial
review subagents on the COMPILED PDF (not just the markdown
source) and acted on the findings. Catches by loop:

- **Loop 84** — pre-PDF: tightened EOI flow + added action editor
  candidates + SUBMISSION_CHECKLIST decision-tree fix.
- **Loop 85** — first PDF visual catch (SEV 5): HTML banner +
  math literal text. Root-caused as `unicode_to_latex` running
  before backtick capture; fixed by reordering.
- **Loop 86** — second PDF visual sweep (SEV 5×2): multi-line
  bold leaking, bibtex never run for non-anon variants. Plus
  SEV-4 issues: `\widehat{}` inside backticks, unicode `∈`/`γ`/`η`
  rendering as U+FFFD. Added pdftotext-grep stage with 11
  patterns; added BT_UNICODE_FALLBACK with 40+ entries.
- **Loop 87** — PDF aesthetic catch (SEV 4): §3.5.5 + §8.1 tables
  truncating at right margin. Converted both to bulleted prose.
  Added `lint_paper_md.py` upstream lint (~170 LoC, 6 checks).
- **Loop 88** — §7 cluster lead-ins (SEV 2 cleanup): 6 flat
  limitations grouped into 4 cluster subsections. lint_paper_md
  wired into GH Actions workflow.
- **Loop 89** — Loop 88 regression catch (SEV 4): §7 renumbering
  caused intro/rendering mismatch. Converted to bold paragraph
  leads (no list numbering). §10.2 venue calibration cleaned —
  rewrote 4 long bullets (URLs, dates, city/date strings) as a
  4-row table + 2 sentences.
- **Loop 90** — 15th pass surfaced FOUR more SEV-5/SEV-4 blockers
  that would have failed desk-screen: §10.2 table truncation,
  Appendix C structural orphan (table appeared under wrong
  heading), §B.3 exit-codes table truncation, and **7 body
  citations had no rendered bib entry** (Daniel et al. 2015 the
  primary attribution, VanderWeele-Ding cited 8+ times, Vaswani,
  Loshchilov, Haneuse, Guo, Fostiropoulos). Fixed by converting
  the 3 tables to bullets and adding `\nocite{}` directive to
  test_compile_tmlr.tex. Bib gained 2 new entries (meng2022rome,
  wang2023activation).
- **Loop 91** — Loop 90 regression catch (SEV 2): the converted
  bullets in §10.2 / §B.3 / Appendix C broke prose continuation
  into separate paragraphs because the converter's list-
  continuation detection required 3-space indent (`lj.startswith
  ("   ")`) but Markdown spec is 2-space. Relaxed to 2-space;
  bullets now flow inline. Table renumber 3→1 to restore reading
  order. README updated with preflight_submission.sh as the
  one-command entry point.

Net effect: 7-stage CI gate → 8-stage CI gate (added markdown
lint). pdftotext-grep added to xelatex stage with ~20 patterns.
PDF renders cleanly with math symbols, multi-line bold, tables
without truncation, four-cluster §7, bullet flow, complete
bibliography. **From 16 adversarial reviews.**

### 8. Case study survey (Loops 63–64)

- Loop 63: 3-paper applicability survey (NormFormer, BitNet b1.58,
  Peri-LN). Detailed F2 walkthrough on Peri-LN Table 1.
- Loop 64: expansion to 9 papers (adding OPT, Switch Transformer,
  Pythia, Mamba, Llama, nanoGPT). Headline finding: 7 of 9 publish
  single-run ablation tables; 0 of 9 release per-seed CSVs.
  Documents the methodological gap F2 is designed to close.

### 9. Github Actions integration (Loop 74)

`.github/workflows/paper-checks.yml` wires `run_all_checks.sh`
into CI on every push touching `papers/`. PR #185 turns from
"Draft, locally-verified" → "Draft, CI-verified".

---

## Submission status (at the time of this changelog)

- Branch: `f2-methodology` (PR #185 Draft)
- Target venue: NeurIPS 2026 MLRC official track (TMLR-routed)
- Soft EOI deadline: 2026-06-04 AOE
- Hard TMLR decision: 2026-09-30 AOE
- Paste-ready EOI text:
  `papers/tmlr_submission_kit/eoi_form_text.md`
- Paste-ready Issue #1021 update:
  `papers/tmlr_submission_kit/issue_1021_comment.md`
- Submission action items:
  - User: submit paper to TMLR via OpenReview, then fill MLRC EOI
    Google Form ([forms.gle/bvYxagcRjKSmYhUM7](https://forms.gle/bvYxagcRjKSmYhUM7))
  - User: post #1021 comment
  - System: PDF + supplementary zip auto-built by
    `.github/workflows/paper-checks.yml` artifacts on push

---

## How to reproduce every paper result

```bash
git checkout <anchor commit on f2-methodology>
papers/scripts/run_all_checks.sh  # 6-stage CI gate, ~30 s
# Outputs:
#   papers/tmlr_submission_kit/test_compile.pdf       (37 pp non-anon)
#   papers/tmlr_submission_kit/test_compile_anon.pdf  (34 pp anon)
#   papers/tmlr_submission_kit/test_compile_tmlr.pdf  (23 pp TMLR class)
#   papers/tmlr_submission_kit/f2_methodology_supp.zip
#   papers/figures/fig{1,2,3,4,5,6}_*.png             (regenerated)
#   papers/appendix_d_test_inventory.md               (regenerated)
```
