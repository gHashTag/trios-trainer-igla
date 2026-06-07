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

### 7. Reviewer-screen feedback loop (Loops 59, 61, 75–161)

Independent adversarial reviews surfaced load-bearing issues
caught before reviewers saw them. **Eighty** independent
passes total across Loops 59–161 (50-pass milestone reached at
Loop 127; passes 51–80 dispatched at Loops 128–161; Loops 157,
158, 159 were rehearsal / anchor-pin / camera-ready-prep loops
with no adversarial passes dispatched).
The first 11 (Loops 59, 61, 75–85) targeted the original paper
drafts and submission flow; Loops 86–161 extended the discipline
to round-N audits where each substantive patch is independently
re-audited the loop after it lands. Detail on passes 1–11 below;
passes 12–80 drove the gate-evolution loops summarized in §10
(per-loop CI additions Loops 87–161; per-pass detail lives in
the per-loop commit messages, not §10). The
combined breadcrumb is `git log --oneline --grep="adversarial
pass" --grep="round-"` which surfaces ≥35 commits across Loops
90–161 (49th pass flagged the un-widened grep covered only
~35 of 80 passes; the two-pattern form widens reach).

#### Adversarial review retrospective (frozen at 50-pass milestone, Loop 127)

Snapshot at Loop 127 (50 passes); see §7 lead for cumulative count.
The 27→50 audit cycle (rounds N=1..23 across Loops 104–127)
followed a productive pattern: each substantive patch was
re-audited the next loop; round-N audits at N≥5 still found
new SEV classes the prior N-1 had missed. Bug-class
retrospective:

- **Citation drift** (Loops 79, 80, 93, 97): 5 distinct error
  modes — fabricated authors, withdrawn papers, mis-categorized
  semantic claims, bib title paraphrasing, primary-source
  attribution.
- **Numeric / arithmetic drift** (Loops 98, 102, 117, 124): 4
  modes — CSV-vs-table mismatch, formula-vs-table inconsistency,
  cross-paper test count drift, stage-count partition arithmetic.
- **Code-vs-text contradiction** (Loops 110, 112, 114, 115): 4
  modes — claimed binary doesn't exist, claimed preamble fields
  not emitted, primitive equivalence claimed but not enforced,
  CI-gate scripts named but not implemented.
- **Label / framing residue** (Loops 95, 100, 107): 3 modes —
  Pearl-CDE label survival after §3.4 drop, "originally drafted"
  anchor ambiguity, asymmetric reporting framing leftover.
- **Gate-design fragility** (Loops 122, 125, 127): 3 modes —
  ACKN regex typo making gate a no-op, shim-vs-dispatcher tuple
  arity drift, CommonMark fence parity violation.

**Bug classes static gates CANNOT catch** (require adversarial
reads): pre-registration honesty (does §5.4 actually pre-register
what it claims?), framing emphasis (does §1 over-promise relative
to §5 delivery?), conceptual coherence (does the Pearl-CDE
identification machinery actually apply to the new dataset?),
reader-experience flow (does §3→§5 traversal land where promised?).
Estimated ~30% of the (then-)50 passes caught issues in this
category that no static check would surface (Loop 127 anchor; see
lead paragraph for current cumulative count). The recurring
meta-finding: **static gates catch what passes catch on the loop
they're introduced, but every patch needs the round-after audit**.

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

### 10. CI gate evolution (Loops 87–161)

The 39th and 40th adversarial passes both surfaced that the
F2 paper's §E catalogue, whose 8-script composition crystallized
at Loop 96 (both `check_no_fabricated_shas.py` and `lint_paper_md.py`
added to §E together at commit `c8e8707`; the gate had grown earlier
but §E was synced then) and remained authoritative as a complete
enumeration through Loop 98,
diverged from the on-disk gate after subsequent loops added new
stages. This section documents the trajectory per-loop so that
the §E paragraph can point here for the additional gates rather
than reciting them inline.

Stage additions since the original 8-script catalogue:

- **Loop 87** — Markdown lint (`lint_paper_md.py`) **first added as a
  stage in `run_all_checks.sh`** (§E catalogue itself wasn't synced
  until Loop 96 `c8e8707`; see also the Loop 87 entry in §7.5 above
  for the conversion-tooling context). Upstream of
  the LaTeX render to catch SEV-4 table-truncation and structural
  issues before xelatex parses the body.
- **Loop 99** — `verify_tables_against_csv.py` added: numerical
  table claims gated against source CSVs at 2-decimal tolerance.
  Operationalized the Loop 98 22nd-adversarial-pass catch.
- **Loop 103** — `verify_formulas_vs_tables.py` added: four-PSE
  algebraic closure + §3.3 envelope Γ_tip claims re-derived from
  CSV CIs. Operationalized Loop 102's 25th-pass catch on the
  manuscript Γ_tip = 1.43 vs derived 1.01 discrepancy.
- **Loop 108** — `verify_label_consistency.py` added: deprecated-
  term grep with allowed-context regex. Catches retired
  terminology (Pearl-CDE post-§3.4-drop, Zhao-Luo pre-Loop-55
  correction, etc.) drifting back into the body.
- **Loop 111** — `smoke_f2_pairwise_perm.sh` added: end-to-end
  test of the #1021 paper's new binary, exact-match assertions
  on the synthetic 5-seed input's expected p-value.
- **Loop 114** — `verify_preamble_per_producer.py` added with
  static + runtime modes. Gates per-producer W3C-PROV preamble
  schema against the actual binary source.
- **Loop 115** — `verify_provenance.sh` (first of three #1021
  pre-registered scripts) added. Walks `data/issue1021/run0/`
  and pipes preamble-emitting CSVs through `f2_provenance_check`.
- **Loop 116** — `verify_run_completeness.py` added. Asserts
  §5.1 artifact inventory at the run-result anchor (80 cell + 2
  aggregate + 2 pairwise + 1 stratum_compare + 8-16 sensitivity
  CSVs).
- **Loop 117** — `verify_report_consistency.py` added. Gates
  numeric claims in the 6 post-run reports against source CSVs
  (2/6 reports registered at draft time; remaining 4 deferred
  until the run produces concrete outputs).
- **Loop 118** — `verify_stage_count_consistency.py` added. Gates
  paper claims of "N stages" against the actual `STAGES` array
  count in `run_all_checks.sh`. Closes the SEV-2 prose-drift
  class that the 39th + 40th adversarial passes both surfaced.
- **Loop 121** — `verify_cross_paper_consistency.py` added with 3
  claim classes (EXACT_MATCH, SCOPED_DIFF, ACKNOWLEDGES). Gates
  cross-paper numeric claims (F2 ↔ #1021) for self-consistency
  and explicit acknowledgement of intentional scope differences.
- **Loop 124** — `meta_test_cross_paper_gates.py` added. Synthetic-
  failure tests for each class of cross-paper gate. Closes the
  46th-pass "gate ships with no enforcement" class (originally a
  `\\bf12` regex typo that made the ACKN gate a no-op).
- **Loop 129 C** — `verify_stage_count_consistency.py` registry
  extended with `SUBMISSION_CHECKLIST.md` §1 "exits 0 with **N/N
  PASS**" pattern. Closes the 52nd-pass submission-readiness audit
  finding that §1's `13/13` count drifted silently as new stages
  were added.
- **Loop 130 C** — `verify_submission_readiness.py` added (22nd
  stage). Parses SUBMISSION_CHECKLIST.md §1 sub-bullet enumeration
  `(k/M) <name> — <desc>` and asserts: (1) sub-bullet count equals
  STAGES count, (2) every `M` denominator matches the actual count,
  (3) `k` numerators are exactly `[1..N]` monotonic, (4) per-position
  name overlap (Jaccard ≥ 0.30, case-folded, stop-word filtered)
  between sub-bullet and STAGES entry. Operationalizes the audit
  finding fully: the §1 N/N gate added in Loop 129 only caught the
  top-line; this gate catches every sub-list drift class.
- **Loop 130 B** — `verify_cross_paper_consistency.py` extended with
  `RELATIONAL_CLAIMS` class. First entry: non-anon ≥ anon page
  count (anonymization can only replace content with an
  equal-or-shorter `[OMITTED FOR DOUBLE-BLIND REVIEW]` placeholder,
  so the non-anon page count must dominate). `meta_test_cross_paper_gates.py`
  extended in lock-step with a synthetic break-test, keeping the
  Loop 126 B inventory-completeness check green (4/4 classes
  covered → 5/5 break-tests pass).
- **Loop 131 B** — `RELATIONAL_CLAIMS` grown from 1 → 4 entries:
  TMLR-class ≤ anon page count (article-wrapper wider than tmlr.sty
  single-column tight layout); F2 §8.2 grouped total ≥ §3.5.4
  lib-only (lib tests are a subset of cumulative); #1021 §3.3
  binary count ≥ F2 §D 10-binary inventory (#1021 superset). Total
  cross-paper claims now 11 across 4 classes (1 EXACT + 5 SCOPED +
  1 ACKN + 4 RELATIONAL) — *as of Loop 131 B; superseded by Loop
  133 A.iii below which reclassified 3 SCOPED → EXACT_PIN.*
- **Loop 131 C** — `verify_changelog_consistency.py` added (23rd
  stage). Binds the adversarial-pass count + loop-range terminal
  cursor across three sites: CHANGELOG §7 lead paragraph (uses
  English word like "Fifty-three"), SUBMISSION_CHECKLIST §2
  (digits), ADVERSARIAL_REVIEW_LOG headline (digits). Closes the
  53rd-pass catch #1 (CHANGELOG §7 was 20 loops stale at "Fifty"
  while the other two sites had been refreshed to "52" at Loop
  129) — operationalizes "no single doc owns the canonical count;
  the gate is the only canonical source".
- **Loop 132 B** — `regen_changelog_section7.py` added as an
  informational generator (NOT in the CI gate). Runs `git log
  --grep="adversarial pass" --grep="round-"` on the f2-methodology
  branch and emits a per-loop Markdown table at
  `papers/CHANGELOG_section7_generated.md`. The hand-maintained §7
  prose remains authoritative; this artifact makes drift visible
  at-a-glance and prepares for an eventual retire-the-prose
  transition (54th-pass catch #7+#8 *class* mitigation).
- **Loop 132 C** — `verify_anonymizer_completeness.py` added
  (24th stage). Scans the two TMLR-bound papers
  (`f2_methodology.md`, `phi_ladder_paper_intro_draft.md`) for
  bare `\bLoop \d+\b` anchors outside allowed contexts (section
  headers, `(internal ref)` parentheticals, HTML comments, fenced
  code). Operates in **ratchet mode**: legacy debt (28 + 42 = 70
  bare anchors at Loop 132 baseline) is allowed; new additions
  fail the gate. Closes the 54th-pass catch #9 *class*
  (anonymization-leak through bare Loop-N anchors).
- **Loop 133 A.iii** — `EXACT_PIN_CLAIMS` class extracted from the
  three [N, N] SCOPED_DIFFs that were exact-pins disguised as
  ranges (the 48th-pass A5 comment explicitly acknowledged the
  disguise). 11 cross-paper claims now span 5 classes (1 EXACT +
  2 SCOPED + 1 ACKN + 4 RELATIONAL + 3 EXACT_PIN). `meta_test_*`
  extended with EXACT_PIN break-test → 7/7 synthetic-break tests.
- **Loop 133 A.iv** — Bare-anchor burn-down in F2 §E catalogue:
  the paragraph at `f2_methodology.md:1730+` was rewritten to drop
  7 inline Loop-N attributions (Loop 96, 118, 121, 124, 130 C,
  131 C, 132 C); the per-loop introduction history is now
  centralized in this §10 alone. `verify_anonymizer_completeness.py`
  baseline lowered 29 → 22 for f2_methodology.md (total 65).
- **Loop 133 B** — `verify_cardinality_arithmetic.py` added (25th
  stage). Generalizes the 55th-pass #14 catch (§1 claim-class
  enum specifically) to a registry of "N items (a + b + c)"
  claims with class-count-agnostic + order-agnostic two-stage
  parsing. Closes the 56th-pass #4 rigidity catch.
- **Loop 133 C** — `verify_generator_consistency.py` added (26th
  stage). Runs `regen_changelog_section7.py` and asserts agreement
  on commit floor + range terminal with the §7 lead breadcrumb.
  Operationalizes the "parallel generator vs hand-maintained
  drift" pattern (preserves Loop 132 B's generator as a binding
  artifact rather than informational only).
- **Loop 134 A.iii** — phi_ladder §5.4 partition burn-down (−7
  inline Loop-N attributions); anonymizer baseline 46 → 39.
- **Loop 134 A.iv** — `verify_class_registry_binding.py` added
  (27th stage). Binds §1 (13/M) class-class labels + counts to
  live `*_CLAIMS` list lengths via CLASS_LABEL_ALIASES (single
  source of truth in verify_cross_paper_consistency.py).
- **Loop 134 B** — Anonymizer baselines migrated from in-code
  SCAN_TARGETS tuple to JSON sidecar
  `papers/scripts/anonymizer_baseline.json`. Ratchet semantics
  preserved; FALLBACK_BASELINES rescue on parse failure.
- **Loop 134 C** — `verify_committed_state_consistency.py` added
  as manual pre-flight tool (NOT a CI stage — would fail every
  mid-loop edit). Walks `git status --porcelain papers/ docs/`
  and fails on tracked-file modifications.
- **Loop 135 A.iii** — phi_ladder §5.4 second-pass burn-down
  (−7 attributions in pre-registered-scripts bullets);
  anonymizer baseline 39 → 32.
- **Loop 135 B** — `verify_cardinality_arithmetic.py` CLAIMS
  registry supports `frozen=True` 4-tuples for historical
  snapshots. §10 Loop 131 B entry marked frozen with "(as of
  Loop 131 B; superseded by Loop 133 A.iii)" annotation.
- **Loop 135 C** — `verify_documented_vs_extracted_consistency.py`
  added (28th stage). Binds §1 sub-bullet description metadata
  to live verifier source state via importlib + regex extraction.
  Initial bindings: (11/N) report consistency 6=2+4 vs
  REPORT_SPECS + EXISTS_STUBS; (14/N) meta-test 7 = count of
  `def test_*`.
- **Loop 136 A.iii** — F2 §E remaining-anchors burn-down (−6
  inline attributions across §3.2 / §3.5 / §4 / §6.1);
  anonymizer baseline 22 → 16. Total legacy debt 54 → 48.
- **Loop 136 A.iv** — doc-vs-extracted registry grown 2 → 4
  bindings: (12/N) stage-count "5+2+1" → CLAIMS / DECOMPOSITION /
  DERIVED list lengths; (24/N) anonymizer "48 at Loop 136" →
  sum of sidecar baselines.
- **Loop 136 B** — `verify_anonymizer_completeness.py
  --update-baseline` flag. Atomically rewrites
  anonymizer_baseline.json with current per-file counts (refused
  on regression — cannot ratchet up). Closes 58th-pass #6.
- **Loop 136 C** — `verify_committed_state_consistency.py
  --staged-only` mode. Asserts every working-tree-modified
  scope file is in the git index; passes during mid-loop edits
  if staging is complete. Kept as manual pre-commit tool (NOT a
  CI stage) because wiring into run_all_checks.sh would fail
  every dev cycle that hasn't yet `git add`-ed all edits.
- **Loop 137 A.iii** — phi_ladder §5.4 third-pass burn-down: dropped
  5 inline Loop-N attributions (Gate-decomp prelude Loops 116/117/
  118+121, plus 37th-pass refs Loops 114/115 A, plus Loop 98 --diff
  machinery). Anonymizer baseline 32 → 27. Total legacy debt 48 → 43.
- **Loop 137 A.iv** — `papers/scripts/_gate_utils.py` shared helper
  module added. Currently exposes `import_gate(name)`. Used by
  verify_class_registry_binding.py and verify_documented_vs_extracted_
  consistency.py — closes the 59th-pass #1 silent-swallow class
  permanently (any future closure to import_gate lands in one place).
- **Loop 137 B** — `verify_burn_down_history.py` added (29th stage).
  Parses the FALLBACK_BASELINES breadcrumb in verify_anonymizer_
  completeness.py and asserts (a) every "(Loop N): A + B = C." tuple
  satisfies A+B=C arithmetic, and (b) the most-recent entry matches
  the live sidecar (per-file counts + total). Operationalizes
  "documented history vs live state" as a permanent class.
- **Loop 137 C** — `verify_src_unchanged_during_paper_loop.py` added
  as manual pre-flight tool (NOT a CI stage — parallel agents
  routinely modify src/ during paper loops). Walks `git status
  --porcelain src/` and reports tracked + untracked modifications
  with `--stat` option for line-count detail.
- **Loop 137** 60th-pass SEV-3/4 closures: anonymizer (24/N) regex
  anchored on "ratchet," to prevent multi-match silent-pick;
  doc-vs-extracted multi-line lookahead extended with `^##\s` to
  stop at markdown headings; FROZEN tag uppercased for visual
  consistency; EXCLUDE_PATHS extended with xelatex compile logs.
- **Loop 138 A.iii** — F2 trailing-attributions burn-down: dropped
  5 inline Loop-N anchors across §3.5 / §4 / §6.1 (Loop 32+ ablation
  sweep, Loop 102 wd0/25th-pass refs, Loop 30 RMS-CDE doc anchor,
  Loop 60 validation pass). Anonymizer baseline 16 → 11. Total
  legacy debt 43 → 38.
- **Loop 138 A.iv** — `_to_int` English-numeral helper extracted
  from verify_changelog_consistency.py (was `_to_int`) AND
  verify_stage_count_consistency.py (was `words_to_int`) into
  `papers/scripts/_gate_utils.py` as `to_int`. Both prior callers
  now delegate; closes 60th-pass SEV-4 #6 helper-extraction-
  conventions class permanently.
- **Loop 138 B** — `verify_alias_round_trip.py` added (30th stage).
  Asserts CLASS_LABEL_ALIASES is bijective on the gate's actual
  class names: every alias key resolves to a `*_CLAIMS` list AND
  every `*_CLAIMS` list has at least one alias mapping to its
  canonical name. Currently 6 aliases cover 5 classes.
- **Loop 138 C** — `verify_pre_commit_hook.py` added as manual
  tool. Detects whether `.git/hooks/pre-commit` or
  `.husky/pre-commit` is installed and references the staged-only
  gate. Supports `--install` flag to write a minimal hook script.
- **Loop 139 A.iii** — F2 §3.5 burn-down: dropped 5 inline Loop-N
  anchors from the three "Identical-hash divergence / Silent
  provenance loss / Silent stratum loss" bullets (Loops 28/31/32/47).
  Anonymizer baseline 11 → 6. Total legacy debt 38 → 33.
- **Loop 139 B** — `_gate_utils.import_gate` extended with optional
  `_MODULE_CACHE`. Default cache=True; pass cache=False to bypass.
  Regression protection against future in-process orchestrators
  (each stage today is a separate subprocess, so the cache is
  exercised within `verify_module_cache_consistency.py` itself —
  the earlier "4× speedup for run_all_checks.sh sweeps" claim was
  inaccurate and corrected at the 62nd-pass closure).
  `verify_module_cache_consistency.py` added (31st stage) asserting
  the cache contract: cache=True same instance, cache=False distinct,
  cache=True after cache=False not poisoned.
- **Loop 139 C** — `verify_loop_floating_anchors.py` added (32nd
  stage). Scans TMLR-bound papers for `(... as of Loop N)`
  patterns and asserts N matches the §7 lead loop within
  1-loop in-flight tolerance. Closes 61st-pass #7 SEV-3:
  phi_ladder line 831 "30 stages on disk as of Loop 138" stage-count
  was enforced but Loop tag floated silently — now bound. Frozen
  historical "as of Loop X" entries (italicized `*as of Loop N*` form
  used in CHANGELOG §10) are excluded by the regex.
- **Loop 140 A.iv** — `meta_test_cross_paper_gates.py` extended
  with tempdir+subprocess break-tests for verify_burn_down_history
  (arithmetic 5+5=11 break) and verify_alias_round_trip (dangling
  alias BOGUS → NONEXISTENT_CLASS break). Total break-tests
  7 → 9 covering 5 cross-paper claim classes + burn-down +
  alias bijection. Closes 61st-pass #12 SEV-3 (partial — 5+
  newer gates still uncovered, but the highest-value two have
  break-tests now).
- **Loop 140 B** — `verify_anchor_loop_coverage.py` added (33rd
  stage). Parses CHANGELOG §10 for `**Loop N <suffix>**` entries
  and asserts each one has ≥1 commit on HEAD (the current branch)
  via `git log --grep="Loop N\b"`. Most-recent loop exempted
  for in-flight lag. Catches "narrative is ahead of history"
  drift. Currently 23 historical entries all with ≥1 commit.
- **Loop 140 C** — `test_gate_composition.sh` added as a minimal
  shell smoke wrapper. Runs `run_all_checks.sh` and asserts
  exit 0. NOT a CI stage (recursive invocation would loop).
  Documented as top-level entry point for shell-only CI runners.
- **Loop 141 A.iii** — phi_ladder §4-§5 pass-attribution burn-down:
  dropped 3 inline Loop-N anchors (28th-pass Loop 105 four-PSE,
  30th-pass Loop 107 Λ choice, 31st-pass Loop 108 H2 falsification).
  Anonymizer baseline 27 → 24. Total legacy debt 33 → 30.
- **Loop 141 A.iv** — `meta_test_cross_paper_gates.py` extended
  9 → 12 break-tests via the established tempdir+subprocess pattern.
  Added: documented-vs-extracted drift, changelog-consistency
  disagreement, stage-count drift. Closes 61st-pass #12 SEV-3
  for the highest-value newer gates.
- **Loop 141 B** — `verify_dependency_graph.py` added (34th
  stage). Parses each gate's `_gate_utils.import_gate(name)`
  calls, builds a DAG, asserts (a) no cycles via DFS coloring,
  (b) STAGES execution order respects dependency direction.
  At Loop 141 introduction: 25 gates, 5 import edges; subsequent
  loops grow the graph — current count surfaces in the gate's own
  output (Loop 143 reports 28 gates, 6 edges).
- **Loop 141 C** — `run_all_checks.sh` STAGES annotated with
  parallel `STAGE_TIERS` array (`submission` vs `discipline`).
  Per-stage tier badge in output (e.g., `[submission]`); per-tier
  PASS/FAIL counts in summary. Helps contributors see at-a-glance
  which class fired without re-reading 34 stage descriptions.
- **Loop 142 A.iii** — phi_ladder §1/§2.3/§3.1 attribution drops:
  removed `Loop 98` status anchor, `Loop 49` sandbox-caution
  anchor, and Loop 102/104/105 integer-zoo attribution sequence
  (6 anchors total dropped via 3 rewrites). Anonymizer baseline
  24 → 18. Total legacy debt 30 → 24.
- **Loop 142 A.iv** — `verify_tier_classification.py` added
  (35th stage). Asserts STAGE_TIERS length matches STAGES + every
  tier ∈ {submission, discipline}. Promotes the runtime WARN
  (Loop 141 64th-pass fix #4) to a hard FAIL via this gate stage.
- **Loop 142 B** — `verify_burn_down_trajectory.py` added (36th
  stage). Asserts FALLBACK_BASELINES breadcrumb is (a) strictly
  loop-monotonic and (b) totals are monotonically non-increasing.
  Catches a regression where someone appends a higher-total entry
  (e.g., misclick on --update-baseline) or an out-of-order Loop N.
  Currently 10 entries: Loop 132 → 142, total 72 → 24.
- **Loop 142 C** — `papers/scripts/GATE_AUTHORING_GUIDE.md` added
  as internal methodology distillation (NOT a CI stage). Covers
  when to add a gate, _gate_utils helpers, tempdir+subprocess
  break-test pattern, tier classification, cascade discipline,
  dependency-graph hygiene, breadcrumb maintenance. 6 stages +
  bottom-line discipline rules.
- **Loop 143 A.iii** — `meta_test_cross_paper_gates.py` extracted
  shared helpers `_copy_to_tmp` + `_run_gate` + `_assert_fires` at
  module level. Closes 64th-pass #8 SEV-4 (boilerplate refactor).
  Existing 5 break-tests not yet migrated to keep diff focused;
  future loops can refactor.
- **Loop 143 A.iv** — Two new break-tests in
  `meta_test_cross_paper_gates.py`: `test_tier_classification_parity_break`
  (drops a STAGE_TIERS entry → asserts FAIL) and
  `test_burn_down_trajectory_monotonicity_break` (appends a
  synthetic Loop 999 with 50+50=100 → asserts FAIL on total >
  previous). Meta-test 12 → 14 break-tests. Closes 65th-pass #5
  (stages 35/36 missing break-tests).
- **Loop 143 B** — Re-classified stages 22-24 from discipline →
  submission. STAGE_TIERS split is now 24/13 (was 21/15; Loop 144 added the
  38th stage to discipline making it 24/14). Submission
  tier: anonymization-breaking drift (24), §1↔STAGES alignment (22),
  CHANGELOG/§2/log agreement (23). Extended `verify_tier_classification.py`
  with contiguity check (all submission entries must precede any
  discipline entry). Closes 65th-pass #7 SEV-3 + #8 SEV-3.
- **Loop 143 C** — `verify_changelog_section10_authority.py` added
  (37th stage). Asserts every `verify_*.py` stage in STAGES has a
  matching CHANGELOG §10 entry (legacy gates exempted via
  LEGACY_ALLOWLIST). Closes the "is §10 actually authoritative?"
  question — every gate addition must land with a §10 entry in
  the same commit.
- **Loop 144 A.iii** — Migrated 5 pre-Loop-143 break-tests to
  shared helpers (`_copy_to_tmp` / `_run_gate` / `_assert_fires`).
  ~80 LOC reduction; 14/14 break-tests still pass. Closes
  66th-pass #10.
- **Loop 144 A.iv** — Breadcrumb regex hardening + re-baseline
  escape hatch: (1) label class extended with brackets/
  semicolons/pipes (closes 66th-pass #7); (2) silent-drop
  detector emits WARN on `# Loop N` lines that miss `_ENTRY_RE`;
  (3) `# RE-BASELINE: Loop N <reason>` annotation skips
  monotonicity check for legitimate up-baselines (closes 66th-
  pass #8).
- **Loop 144 B** — `verify_gate_authoring_guide_drift.py` added
  (38th stage). Asserts every `verify_*.py` cited in
  `GATE_AUTHORING_GUIDE.md` exists + breadcrumb label-class regex
  matches live code's `_ENTRY_RE`. Caught real drift on first
  run (guide vs code disagreement); now permanent class closure.
- **Loop 144 C** — `gate_dashboard.sh` added as manual project-
  health snapshot tool (NOT a CI stage). Reports stages + tier
  split + pass count + anonymizer total + deadlines.
- **Loop 145 A** — anonymized PDF body + all three TMLR variants
  recompiled after parallel-agent content drift in
  `f2_methodology_anonymized.md`. Page counts refreshed:
  non-anon 43 → 44, anon 42 → 43, real-TMLR 27 → 28. Closes
  67th-pass SEV-2 #2 (stale body.tex) + SEV-1 #3 (PDFs behind
  source).
- **Loop 145 B** — `verify_changelog_section10_authority.py`
  gains MANUAL_TOOL_ALLOWLIST inverse check: every allowlisted
  manual tool must exist on disk AND be mentioned in §10. Closes
  67th-pass SEV-3 #4 (allowlist silent-drift after rename/delete).
- **Loop 145 C** — `verify_deadline_freshness.py` added
  (39th stage). Asserts every `YYYY-MM-DD AOE` token in
  `SUBMISSION_CHECKLIST.md` §4 + `Anchor / version` is today-or-
  future (or carries `(passed)` / `(historical)` / `has passed`
  annotation). Soft EOI 2026-06-04 has now aged out; the gate
  enforces the annotation discipline going forward. Closes 67th-
  pass SEV-4 #6.
- **Loop 146** — Posit16 (Gustafson 2017, es=1) codec landed at
  `src/phi_numbers/posit16.rs` (16 unit tests). Wired into
  `src/race/format_ladder.rs` with `apply_posit16` quantizer and
  `convert_f32_to_posit16` instrumented method. Microbenchmark
  binary `src/bin/format_microbench.rs` produces 5-seed L2-error
  comparison: Posit16 vs GF16 = −74.0% to −74.3% rel L2 at
  Xavier-init magnitudes (std ≈ 0.13%); GF16 underflows 55-70
  values per seed, Posit16 has zero underflows. Posit16 + GF16
  comparison is the F2 §9.4 format-zoo arm's headline number.
- **Loop 147 A** — citation ledger grew **27 → 32 VERIFIED**
  (+1 VERIFIED-WITHDRAWN unchanged). Independent WebFetch
  verification of 5 candidate IDs caught two fabricated
  attributions: arxiv:2506.20752 first author is Huangyuan Su,
  not "Mishra"; arxiv:2605.09825 first author is Musa Cim,
  not "AMD/MI355X group". Both corrected before adding.
  Sixth addition `gustafson2017posit` (foundational posit paper)
  for the Loop 146 codec. New entries: Semenov-Pagliardini-Jaggi
  2025 (§2.3 multi-seed optimizer race), Hochlehnert et al. 2025
  COLM (§2.3 methodology critique), NVIDIA NVFP4 2025 (§9.4),
  Su et al. 2025 microscaling-format instabilities (§9.4 ablation
  methodology), Cim et al. 2026 MXFP4 on native FP4 hardware
  (§9.4 MXFP4 anchor), Gustafson & Yonemoto 2017 (§9.4 Posit16
  ref).
- **Loop 147 B** — `verify_tex_anonymization.py` added
  (40th stage, submission tier). Scans
  `papers/tmlr_submission_kit/f2_methodology_anonymized_body.tex`
  for converter-side leaks across 8 leak classes (bare Loop-N,
  branch name `f2-methodology`, 4 PII identifiers, `@anthropic.com`
  email, SHA-like hex tokens). Closes 67th-pass SEV-3 #5 (markdown-
  source-only anonymizer ratchet missed converter-side leaks).
  Synthetic break-test verified 5/5 leak classes fire correctly.
- **Loop 147 C** — 68th-pass audit follow-up: Posit16 gains
  `Ord`/`PartialOrd` impls (i16 cast preserves Posit Standard 2022
  §5.2 total order); `ConversionCounter` Display extended to emit
  every tracked field (was silently dropping `f32_to_posit16`,
  `f32_to_int4`, `f32_to_paretoq`, `f32_to_fp8_*`, `f32_to_int8`).
  Two new posit16 unit tests verify total-order claims (NaR < all,
  positive monotone). All 42 phi_numbers + format_ladder tests
  green.
- **Loop 161** — bridge bench v4: MLP → **single-head self-attention
  block** (embed → Q/K/V → softmax-scaled-dot-product → V-aggregation
  → output projection → softmax). HIDDEN = 64, HIDDEN_HEAD = 64,
  SEQ_LEN = 8. **Five** weight matrices (embed + W_Q + W_K + W_V +
  W_O) round-tripped through the chosen format after every SGD step.
  New §9.4.3 numbers (5 seeds × 200 steps): f32 = Posit16 = 4.8281
  ± 0.0142 BPB (gap < 1e-5); GF16 = 4.8378 ± 0.0127 (+0.0097, 0.76×
  std, t ≈ 1.7 — **NOT statistically significant** at N=5). The
  bridge-bench iteration history: bigram +0.030, MLP +0.058,
  attention +0.010 BPB GF16 delta; the GF16 penalty is model-
  dependent and does not monotonically grow with the number of
  quantized matrices alone (attention has 5 matrices but smaller
  delta than 3-matrix MLP). What IS regime-stable across all three
  iterations is Posit16's equivalence with f32. 80th adversarial
  pass folded inline (SEV-2 gap-to-floor explanation reworded from
  "because" to "consistent with"). PDF page counts: TMLR-class
  32 → 33; non-anon and anon unchanged at 52 / 52.
- **Loop 160** — bridge bench v3: bigram → **2-layer MLP**
  (embed → linear → ReLU → linear → softmax). HIDDEN 64 → 128,
  HIDDEN_MLP = 128. All three weight matrices (embed, W1, W2)
  round-tripped through the chosen format after every SGD step
  (was: embed only). Same 5 seeds × 200 SGD steps. New §9.4.3
  numbers: f32 = 4.3086 ± 0.0109; Posit16 = 4.3087 ± 0.0109
  (essentially identical to f32 — gap is 1e-4 BPB); GF16 = 4.3668
  ± 0.0138 (+0.0582 BPB, **4.2× per-seed std, 9.4× MC SE at N=5**,
  t-statistic ≈ 9.4 → decisively above α = 0.001). The GF16
  delta widened from the bigram's +0.030 BPB as expected — three
  format round-trips per step compound the per-step quantization
  noise where the bigram only had one. Posit16's tapered-precision
  encoding remains regime-stable across the two model upgrades.
  79th adversarial pass dispatched on the model upgrade + the new
  prose. PDF page counts: anon 51 → 52 (non-anon and TMLR-class
  unchanged at 52 / 32).
- **Loop 159** — camera-ready prep (post-submission, pre-acceptance):
  added `papers/tmlr_submission_kit/camera_ready_checklist.md`
  as the single-page execution checklist for converting the
  submission to camera-ready *after* TMLR acceptance. Smoke-tested
  the `\usepackage[accepted]{tmlr}` option locally: 30-page output
  with "Published in Transactions on Machine Learning Research"
  banner replacing "Under review". Documents the 8-step flow:
  acceptance verification → §10.3 un-strip → tmlr.sty flip →
  bibliography update → artifact rebuild → CHANGELOG + checklist
  update → upload → tag camera-ready. No CI behavior change.
- **Loop 158** — submission anchor pinned at SHA `5f2bc78` in
  SUBMISSION_CHECKLIST.md §1. All 43/43 CI green; 78 adversarial
  passes complete; 87/87 SHAs reachable; PDFs 52/51/32 pp; supp
  zip ~1.0 MB. The submission package is ready for the user's
  OpenReview upload + MLRC EOI form + #1021 status comment
  actions.
- **Loop 157** — cold-clone submission-day rehearsal. Three
  findings, all surfaced + closed in the same loop:
   1. **Shallow clone breaks 3 gates** (`--depth 1` truncates
      history; the no-fabricated-SHAs / generator-consistency /
      anchor-loop-coverage gates all need full history). Documented
      in SUBMISSION_CHECKLIST.md §1: use full `git clone`.
   2. **Cold-clone wall time** measured at ~3:30 min on M-series
      macOS (8s clone + 3:28 pipeline-with-cargo-build). The
      earlier 15-30 min estimate was the slower CI runner.
      Documented in SUBMISSION_CHECKLIST.md §1.
   3. **10 orphan SHAs** referenced in CHANGELOG.md / anonymize_paper.py
      / body.tex / data/loop49/README.md but not reachable from
      f2-methodology branch (5367bde, 05f37cd, 19d032e, 2969bdf,
      76048b5, a092d5e, ae48fd5, ccbf52b, 6ac812d, afc71ba). Closed
      by adding lightweight `historical/<sha7>` tags + pushing them
      to remote. Cold clone now sees all 87 referenced SHAs as
      reachable (was 77 reachable + 10 orphaned).
  Stage count unchanged at 43. No new adversarial pass — this is
  a discipline / rehearsal loop, not a paper-claim loop.
- **Loop 156** — bridge bench v2 hardening: STEPS 50 → 200
  (4× longer training); default seeds expanded 3 → 5. New
  budget puts the val BPB into the converged regime (≈ 4.55
  vs the Loop 155 random-byte plateau at ≈ 6.78). The result:
  Posit16 = f32 still indistinguishable (means agree to 4
  decimals, rank flips arbitrarily); GF16's penalty is now
  **+0.0297 BPB** = **1.7× per-seed std** and **≈ 3.9× MC SE
  at N=5** — statistically meaningful at α = 0.05. The 77th-pass
  SEV-2 #1 concern (N=3 borderline) is closed by Loop 156's
  hardened budget. F2 §9.4.3 table and narrative updated; PDF
  page counts shift 52/51/32 → no change (table is the same
  shape; only the numbers and N moved). Footnote (iv) extended
  to document why 50 steps was a random-byte plateau.
- **Loop 155** — **bridge bench**: actual training-time comparison
  of f32 / GF16 / Posit16 quantization at the embedding gate on a
  one-layer bigram LM trained on byte-level `tiny_shakespeare` for
  50 SGD steps × batch 64. 3 seeds × 3 formats = 9 cells. New
  binary `src/bin/bridge_bench` + 43rd CI stage
  (`verify_bridge_bench_freshness.py`, discipline tier).
  F2 §9.4.3 added as a new sub-subsection with the inline table.
  **Headline**: f32 = Posit16 = 6.7833 ± 0.0332 BPB; GF16 = 6.7900
  ± 0.0323 BPB. The Posit16 advantage over GF16 predicted by
  §9.4.1's encode-time grid (small-magnitude underflow protection)
  survives a trip through a real optimizer loop at sandbox scale.
  PDF page counts shift 50/50/31 → 51/51/32.
- **Loop 154** — three 75th-pass deferred closures, no new
  stages:
   - SEV-3: `dot_naive` in `src/bin/quire_microbench.rs` gets a
     long doc-comment walking through the f64 round-trip step-by-
     step + contrasting with the f32-accum and quire methods.
     Implementation unchanged.
   - SEV-4: rationale paragraph added to
     `verify_quire_microbench_freshness.py` docstring documenting
     why stage 42 stays in discipline tier per the catch-class
     taxonomy ("drift catcher", not direct submission-blocker —
     a missing JSON is not visible in the submission PDF).
   - SEV-5: F2 §9.4.2 gains a "Scope note" footnote explaining
     why d_model=384 was pinned (matches §9.4.1's xavier column;
     §9.4.1 owns the d_model sweep; §9.4.2 isolates the
     accumulator-vs-storage-format axis at fixed encoding).
- **Loop 153** — `verify_quire_microbench_freshness.py` added
  (42nd stage, discipline tier). Asserts the §9.4.2 dot-product
  regime-map summary
  (`.trinity/results/quire_microbench_summary_seeds_<lo>-<hi>.json`)
  plus per-seed JSONs exist with the expected schema
  ({xavier, cancellation} × {64, 256, 1024, 4096} = 40 cells).
  New binary `src/bin/quire_microbench` exercises the Loop 152
  quire on real dot products and reports {naive Posit16,
  f32-accumulator, PositQuire} vs f64-ground-truth relative
  error. F2 §9.4.2 added as a new sub-subsection with the
  regime-map table inline. Headline: at the F2 §9.4.1 scale
  (L ≤ 4096, Xavier-init magnitudes), the quire and f32
  accumulator are indistinguishable to four significant figures
  — both ~10× lower error than naive Posit16 sum. The quire's
  theoretical advantage materializes only at L ≫ 10⁶ or under
  adversarial cancellation, outside F2's regime; honest reporting
  rather than overclaiming. PDF page counts shift 48/47/30 →
  49/48/31.
- **Loop 152** — Posit16 quire-bit accumulator landed at
  `src/phi_numbers/posit16_quire.rs` (19 unit tests, all green).
  `PositQuire` is an i128-fixed-point accumulator with 2⁻⁵⁶
  resolution and 71 bits of integer headroom — exact for any
  Posit16 dot-product up to ~32 768 elements at maximal product
  magnitude, which covers every neural-network vector dimension
  in the F2 §9.4.1 regime. Public API: `PositQuire::new()`,
  `add_product(a, b)`, `add(x)`, `clear()`, `to_posit16()` +
  helper `posit16_dot(a, b) -> Posit16`. NaR is a sticky bit;
  zero-multiplication is a no-op; saturating add on overflow.
  `Posit16::decode_extended` extracted as a public accessor.
  Total lib test count 735 → 754 (+19); inventory 830 → 849.
- **Loop 151** — closes 71st-pass deferred SEV-3 + SEV-4:
   - **Shared grid config**
     (`papers/scripts/format_microbench_grid_config.json`) is
     now the single source of truth for `d_models` (4 values),
     `inits` (3 schemes), `seeds` (5 values). Both the binary
     (`load_grid_config` in `src/bin/format_microbench.rs`) and
     the freshness gate (`_load_config` in
     `verify_format_microbench_freshness.py`) read it; both fall
     back to documented defaults if the file is missing. CLI
     `--seeds=` still overrides for ad-hoc subset runs.
   - **Per-cell JSON schema validation**: freshness gate now
     parses each cell file and checks top-level keys + nested
     `cell.headline.*` numerics — closes the "existence-only"
     gap the 71st-pass flagged. Synthetic break-test verifies
     6 schema violations are surfaced on a deliberately-corrupted
     cell.
   - 73rd adversarial pass dispatched; this is a discipline-only
     loop with no paper-body changes (PDF page counts unchanged
     at 48/47/30). Pass count cascade 72 → 73.
- **Loop 150** — F2 §9.4.1 added as new sub-subsection: inline
  4×3 grid table reporting Δ(Posit16 vs GF16) % rel L2 across
  (d_model × init) cells with explicit encode-vs-train framing.
  PDF page counts shift 46/46/29 → 48/47/30 (non-anon picks up
  one extra page from the table + methodological-footnote block).
  72nd adversarial pass — 3 catches folded:
   - SEV-1: MC-error claim "≈ 0.5× the values shown" was ambiguous
     between sample-std and delta; rewritten with explicit
     500× MC-error ratio and "no significance threshold issue at
     this N".
   - SEV-2: 5-seed budget rationale + sample-std heterogeneity
     explanation added as methodological footnote (i)+(ii).
   - SEV-3: encode-vs-train distinction made explicit *inside*
     the "what the table says" paragraph (was only outside it
     in the closing paragraph). LCG-vs-real-RNG rationale added
     as footnote (iii).
- **Loop 149** — `verify_format_microbench_freshness.py` added
  (41st stage, discipline tier). Asserts the format-zoo grid
  summary JSON at
  `.trinity/results/format_microbench_grid/format_microbench_grid_summary_seeds_<lo>-<hi>.json`
  exists with the expected schema, and 60 per-cell JSONs
  (3 inits × 4 d_models × 5 seeds) are present. Closes 68th-pass
  audit SEV-3 #4 (microbench result freshness was not validated
  by CI). Format-microbench binary extended to grid mode:
  {d_model ∈ {128, 384, 768, 1024}} × {init ∈ {xavier, he,
  normal_002}} = 12 cells × N seeds. **Headline grid table**
  (5 seeds, mean ± std of Δ posit16 vs gf16 % rel L2):
  - xavier:     d128 −83.2±0.08, d384 −74.1±0.14, d768 −72.8±0.12, d1024 −70.6±0.07
  - he:         d128 −88.6±0.06, d384 −85.4±0.06, d768 −82.9±0.09, d1024 −81.5±0.02
  - normal_002: d128 −72.2±0.26, d384 −72.1±0.17, d768 −72.1±0.08, d1024 −72.2±0.01
  Posit16 dominates GF16 in every cell of the 4×3 grid; He init
  shows the largest win, normal_002 is regime-independent.
  (± = sample-std across N=5 seeds, not SEM; MC error = std/√5
  ≈ 0.5× the std values shown. All deltas are ≥1000× the MC
  error so significance is not a concern; the notation is
  flagged here for reader clarity. 71st-pass DESIGN-1 closure.)
  Note: "he" variant uses σ = √(2/d_model), not the strict
  He-2015 σ = √(2/fan_in) — d_model-scaled form keeps the
  d_model sweep informative (a fan_in-scaled init would be
  regime-independent of d_model, defeating the point of the
  sweep). 71st-pass SEV-2 disambiguation in code docs +
  CHANGELOG.
- **Loop 148** — 6 verified citations from Loop 147 A folded into
  paper body (no longer orphan bib entries):
   - §2.3: Semenov-Pagliardini-Jaggi 2025 (multi-seed optimizer-
     race anchor); Hochlehnert et al. 2025 COLM (sober-look
     methodology critique).
   - §9.4: NVFP4 2025; Su et al. 2025 (microscaling instabilities);
     Cim et al. 2026 (MXFP4 on native FP4); Gustafson & Yonemoto
     2017 (posit standards ref). PDF page counts shift 44/43/28 →
     46/46/29 across the three variants. 70th adversarial pass
     caught and folded three SEV-2/3 overclaim risks: Posit16
     −74% framed as encode-time only (not training-time); Su et
     al. shared-premise softened from "strongly aligned"; NVFP4
     hybrid framed as motivation, not as validation of F2.

The on-disk gate now runs **43 stages** (verified by the new
stage-count gate above). The #1021 follow-up paper's §5.4 names
the 6 stages it contributes; the F2 §E paragraph references this
CHANGELOG section for the full enumeration.

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
papers/scripts/run_all_checks.sh  # 34-stage CI gate, ~30–60 s warm
# Outputs:
#   papers/tmlr_submission_kit/test_compile.pdf       (43 pp non-anon)
#   papers/tmlr_submission_kit/test_compile_anon.pdf  (42 pp anon)
#   papers/tmlr_submission_kit/test_compile_tmlr.pdf  (27 pp TMLR class)
#   papers/tmlr_submission_kit/f2_methodology_supp.zip
#   papers/figures/fig{1,2,3,4,5,6}_*.png             (regenerated)
#   papers/appendix_d_test_inventory.md               (regenerated)
```
