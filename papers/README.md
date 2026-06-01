# F2 methodology paper

This directory is the home of the F2 methodology paper: a stratified
Pearl-CDE framework for transformer training-recipe ablations with a
sandbox demonstration of suppression mediation. Target venue: NeurIPS
2026 MLRC official track (TMLR-routed).

Anchor commit: any descendant of `583b417` on the `f2-methodology`
branch. Soft EOI deadline: **2026-06-04 AOE**.

## Where to look first

| If you want to … | Read |
|---|---|
| **Submit the paper** (decision-tree, all gates) | [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) |
| **See the paper itself** | [`f2_methodology.md`](f2_methodology.md) (~1500 lines), or the compiled PDF after running the CI gate below |
| **See the headline empirical finding** | §5.2 of `f2_methodology.md` (RmsNorm sign-flip), or `figures/fig1_rms_nde_signflip.png` |
| **See the change history at a high level** | [`CHANGELOG.md`](CHANGELOG.md) (Loops 28-81 by theme) |
| **Check that every cited paper is real** | [`CITATIONS.md`](CITATIONS.md) (24 VERIFIED + 1 VERIFIED-WITHDRAWN of 25 entries) |
| **Survey what other ablation papers do** | [`case_study_published_ablations.md`](case_study_published_ablations.md) (9 papers, 0 release per-seed CSVs) |
| **See the pre-registered champion-scale follow-up** | [`../docs/F2_PRE_REG.md`](../docs/F2_PRE_REG.md) |
| **Inspect the binaries** | [`../docs/F2_BINARIES.md`](../docs/F2_BINARIES.md) |
| **Reproduce the figures** | `scripts/figure_regen.sh` (regenerates all 6 from committed CSVs) |
| **See the empirical CSVs** | [`../data/loop49/README.md`](../data/loop49/README.md) (headline §5 evidence) + [`../data/loop49_swap/README.md`](../data/loop49_swap/README.md) (§6.4 robustness sweep) |

## Verify the paper end-to-end

```bash
papers/scripts/run_all_checks.sh
# 7-stage CI gate, ~46 s on warm caches
```

If any stage fails, the failure log is on stderr and a `/tmp/run_all_checks_N.log`
file remains for inspection.

## Submit the paper

```bash
# 1. Open the checklist
$EDITOR papers/SUBMISSION_CHECKLIST.md

# 2. Run the gate
papers/scripts/run_all_checks.sh

# 3. Anonymize and recompile
python3 papers/scripts/anonymize_paper.py
papers/scripts/compile_tmlr_test.sh

# 4. Copy paste-ready texts
cat papers/tmlr_submission_kit/eoi_form_text.md       # → MLRC EOI Google Form (after TMLR submission)
cat papers/tmlr_submission_kit/issue_1021_comment.md  # → GitHub issue #1021
```

## Directory structure

```
papers/
├── README.md                         ← you are here
├── SUBMISSION_CHECKLIST.md           ← go/no-go
├── CHANGELOG.md                      ← Loops 28-81 by theme
├── CITATIONS.md                      ← per-entry verification ledger
├── case_study_published_ablations.md ← 9-paper survey of ablation reporting
├── f2_methodology.md                 ← the paper body (~1500 lines)
├── f2_methodology_anonymized.md      ← anonymized variant (auto-gen)
├── appendix_d_test_inventory.md      ← auto-gen test inventory
├── cross_reference_report.md         ← auto-gen audit output
├── scripts/
│   ├── anonymize_paper.py            ← strip identifying strings
│   ├── compile_tmlr_test.sh          ← 3-variant xelatex compile
│   ├── cross_reference_audit.py      ← §X.Y / arXiv / file ref check
│   ├── figure_regen.sh               ← 6-figure regen
│   ├── generate_appendix_d.sh        ← cargo test --list inventory
│   ├── md_to_tmlr_tex.py             ← Markdown → TMLR LaTeX
│   ├── pre_commit_paper.sh           ← fast drift check
│   ├── run_all_checks.sh             ← 7-stage CI gate
│   ├── verify_paper_metadata.py      ← title/test/bib/figures parity
│   └── check_no_fabricated_shas.py   ← git cat-file -e per SHA
├── figures/
│   ├── fig1_rms_nde_signflip.{png,py}     ← headline sign flip
│   ├── fig2_stratum_registry.{png,py}     ← architecture
│   ├── fig3_canonical_pse_heatmap.{png,py} ← 5×4 PSE table
│   ├── fig4_tipping_curves.{png,py}       ← Γ_tip(Λ) hyperbolae
│   ├── fig5_swap_nie_m1_heatmap.{png,py}  ← Phase 0 swap
│   ├── fig6_m2_robustness_grid.{png,py}   ← M_2 robustness
│   └── fig_template.py                    ← shared helpers
└── tmlr_submission_kit/
    ├── README.md                            ← kit overview + deadlines
    ├── manifest.md                          ← supplementary layout
    ├── eoi_form_text.md                     ← paste into MLRC Google Form (after TMLR submission)
    ├── issue_1021_comment.md                ← paste into GitHub #1021
    ├── anonymization_checklist.md           ← pre-upload check
    ├── derivation_audit_loop59.md           ← Loop 59 audit findings
    ├── adversarial_review_loop61.md         ← Loop 61 reviewer screen
    ├── f2_methodology.bib                   ← 25 verified entries
    ├── tmlr.sty + tmlr.bst                  ← JmlrOrg TMLR class
    ├── template.tex                         ← skeleton wrapper
    ├── test_compile.tex                     ← article wrapper (non-anon)
    ├── test_compile_anon.tex                ← article wrapper (anon)
    ├── test_compile_tmlr.tex                ← real TMLR class wrapper
    └── pack_supplementary.sh                ← bundle the zip (3-stage pre-flight)
```

## Adversarial review history

The paper has survived **7 independent adversarial review passes** and
**6 citation-hygiene catches** (every entry in `CITATIONS.md` notes
the catching loop). See `CHANGELOG.md` § "Reviewer-screen feedback
loop" for details.
