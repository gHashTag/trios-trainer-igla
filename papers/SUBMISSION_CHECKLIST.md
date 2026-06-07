# F2 paper — submission go/no-go checklist

Single-page pre-submission gate. Designed for zero-thinking
submission-day execution. If every box below is checked and every
gate is green, the paper is ready.

**Empirical anchor**: `583b417` (earliest commit at which every
empirical CSV referenced in the paper is committed).
**Methodology anchor**: a descendant on `f2-methodology` branch
(pinned to a specific SHA at submission day; pre-submission
placeholder below).

**Pre-submission anchor**:
`5f2bc78` (full SHA `5f2bc782bddbcfb015b067027dab50f5d5571d70`) —
branch HEAD commit on submission day (Loop 158, 2026-06-07).

---

## 1. CI gates (all must PASS)

Run from the crate root. Total wall: ~30–60 s warm; **~3:30 min cold**
(measured Loop 157 rehearsal on M-series macOS — full clone + full
pipeline including cargo build + xelatex compile + figure regen +
supplementary pack). The earlier 15-30 min cold-clone estimate was
based on the slower CI runner of `paper-checks.yml`; local cold runs
are substantially faster.

**Clone command** — use full clone, NOT `--depth N`:
```bash
git clone --branch f2-methodology git@github.com:gHashTag/trios-trainer-igla.git
```
Shallow clones break three gates (no-fabricated-SHAs / generator-
consistency / anchor-loop-coverage), all of which depend on full git
history. The orphan SHAs referenced by anonymize_paper.py /
CHANGELOG.md / data/loop49/README.md are preserved across operations
by lightweight `historical/<sha7>` tags (Loop 157 hardening).

- [ ] `papers/scripts/run_all_checks.sh` — exits 0 with **43/43 PASS**
  - [ ] (1/43) cross-reference audit (F2/main paper) — 0 dangling refs
  - [ ] (2/43) metadata verifier — title/tests/bib/figures parity
  - [ ] (3/43) no fabricated SHAs — `git cat-file -e` per SHA-like token
  - [ ] (4/43) markdown lint — 0 SEV-≥4 issues
  - [ ] (5/43) tables vs CSVs — 6 tables × 119 numeric assertions verified
  - [ ] (6/43) formulas vs tables — 4 algebraic identities (closure +
        Γ_tip bullets + headline-table + #1021 arithmetic) — 44 assertions
  - [ ] (7/43) label consistency — 7 deprecated-term rules, 0 leaks
  - [ ] (8/43) preamble per producer — static + runtime W3C-PROV gate
  - [ ] (9/43) provenance gate — `f2_provenance_check` on cell + pairwise CSVs
  - [ ] (10/43) run completeness — 93-CSV inventory check (vacuous pre-sweep)
  - [ ] (11/43) report consistency — 6 reports (2 full + 4 stub coverage)
  - [ ] (12/43) stage count consistency — 5 counts + 2 decompositions + 1 derived
  - [ ] (13/43) cross-paper consistency — 11 claims (1 EXACT + 2 SCOPED + 1 ACKN + 4 RELATIONAL + 3 EXACT_PIN)
  - [ ] (14/43) cross-paper gate meta-test — 14 synthetic-break tests PASS (5 cross-paper classes + burn-down + alias + doc-vs-extracted + changelog + stage-count + tier parity + trajectory monotonicity)
  - [ ] (15/43) f2_pairwise_perm smoke — end-to-end binary test + prov check
  - [ ] (16/43) #1021 cross-ref audit — informational
  - [ ] (17/43) #1021 markdown lint — 0 SEV-≥4 issues
  - [ ] (18/43) test inventory regen — `cargo test --list` re-emits 849
  - [ ] (19/43) xelatex 3-variant compile — non-anon, anon, real TMLR class
  - [ ] (20/43) figure regen — all 6 figures rebuild from CSVs
  - [ ] (21/43) supplementary pack — zip builds with 3-stage pre-flight
  - [ ] (22/43) submission readiness — §1 ↔ STAGES alignment gate
        (count + sub-bullet renumbering + per-position nearest-neighbor)
  - [ ] (23/43) changelog consistency — CHANGELOG §7 ↔ §2 ↔
        ADVERSARIAL_REVIEW_LOG agree on (pass_count, last_loop)
  - [ ] (24/43) anonymizer completeness — bare `Loop N` anchor count
        per-file ≤ baseline (legacy debt ratchet, 24 at Loop 144)
  - [ ] (25/43) cardinality arithmetic — N items (a+b+...) sum-equality
        with frozen-historical-entry distinction
  - [ ] (26/43) generator consistency — regen_changelog_section7.py
        output agrees with §7 lead commit floor + range terminal
  - [ ] (27/43) class registry binding — §1 (13/M) class-class
        labels + counts agree with live `_CLAIMS` lists in
        verify_cross_paper_consistency.py
  - [ ] (28/43) documented vs extracted — §1 sub-bullet description
        metadata (e.g., '6 reports (2 full + 4 stub)') agrees with
        live verifier-source state
  - [ ] (29/43) burn-down history — FALLBACK_BASELINES breadcrumb
        most-recent (loop, A+B=C) tuple agrees with live sidecar
  - [ ] (30/43) alias round-trip — CLASS_LABEL_ALIASES bijection
        (every alias → class; every class ← ≥1 alias)
  - [ ] (31/43) module cache — import_gate cache contract (cache=True
        same instance; cache=False distinct; no poisoning)
  - [ ] (32/43) floating loop anchors — "as of Loop N" anchors track
        §7 lead within 1-loop in-flight tolerance
  - [ ] (33/43) anchor loop coverage — every §10 Loop-N entry has
        ≥1 matching commit on HEAD (most-recent exempt)
  - [ ] (34/43) dependency graph — inter-gate import topology is
        acyclic + respects STAGES execution order
  - [ ] (35/43) tier classification — STAGES/STAGE_TIERS parity + tier
        names ∈ {submission, discipline}
  - [ ] (36/43) burn-down trajectory — FALLBACK_BASELINES breadcrumb
        loop-monotonic + total non-increasing
  - [ ] (37/43) §10 authority — every `verify_*.py` STAGES gate has a
        matching CHANGELOG §10 entry (legacy gates exempted)
  - [ ] (38/43) guide drift — GATE_AUTHORING_GUIDE.md and live code
        agree on cited filenames + breadcrumb label-class regex
  - [ ] (39/43) deadline freshness — every dated AOE deadline in §4
        is today-or-future (or annotated as historical/passed)
  - [ ] (40/43) tex anonymization — anonymized .tex artifact clean of
        bare Loop-N, branch, PII identifiers, internal-email leaks, SHA
  - [ ] (41/43) format microbench freshness — grid summary JSON exists,
        schema OK, 60 per-cell JSONs present (3 inits × 4 d_models × 5 seeds)
  - [ ] (42/43) quire microbench freshness — §9.4.2 dot-product accuracy
        summary + 5 per-seed JSONs (2 regimes × 4 lengths × 5 seeds)
  - [ ] (43/43) bridge bench freshness — §9.4.3 sandbox training summary
        + 5 per-seed JSONs (3 formats × 5 seeds = 15 cells)
- [ ] (recommended) `papers/scripts/run_all_checks.sh --check-prereqs` —
      9/9 OK on submission machine (xelatex, bibtex, pdftotext,
      python3 + matplotlib + numpy, zip, cargo, git)
- [ ] (recommended) `papers/scripts/compile_tmlr_test.sh --diff` —
      3 pdftotext snapshots match (regression check on rendered PDF)
- [ ] GitHub Actions workflow `paper-checks.yml` green on latest commit

## 2. Paper state

- [ ] Anchor commit pinned (see top of file).
- [ ] Non-anonymized PDF: **52 pages**, ~239 KB
      (`papers/tmlr_submission_kit/test_compile.pdf`)
- [ ] Anonymized PDF: **51 pages**, ~237 KB
      (`papers/tmlr_submission_kit/test_compile_anon.pdf`)
- [ ] Real-TMLR-class PDF: **32 pages**, ~221 KB
      (`papers/tmlr_submission_kit/test_compile_tmlr.pdf`)
- [ ] Supplementary zip: **45 files**, ~1.25 MB
      (`papers/tmlr_submission_kit/f2_methodology_supp.zip`)
- [ ] `papers/CITATIONS.md` ledger: 32 VERIFIED + 1 VERIFIED-WITHDRAWN
      (97%); 0 UNVERIFIED
- [ ] **Adversarial review**: 78 passes across Loops 59-157
      (`docs/ADVERSARIAL_REVIEW_LOG.md` covers passes 1-18 in detail;
      passes 19-78 documented in `papers/CHANGELOG.md` §7 50-pass
      milestone retrospective and per-loop commit messages). All
      passes addressed; gate-design SEV catches surface in round-N
      audits at N≥5.

## 3. Anonymization (TMLR double-blind)

Per `papers/tmlr_submission_kit/anonymization_checklist.md`:

- [ ] `papers/f2_methodology_anonymized.md` regenerated by
      `papers/scripts/anonymize_paper.py`
- [ ] Sanity check: zero matches for
      `grep -nE '(gHashTag|playra|trios-railway|@anthropic\.com)'`
- [ ] §10.3 Acknowledgments stripped (replaced with
      `[OMITTED FOR DOUBLE-BLIND REVIEW]`)
- [ ] All Loop-N references stripped → `(internal ref)`
- [ ] Known git SHAs replaced → `<anchor commit>`
- [ ] Branch name `f2-methodology` replaced → `<branch>`

## 4. Submission deliverables

For OpenReview TMLR submission:

- [ ] Anonymized PDF (use `test_compile_tmlr.pdf`)
- [ ] Anonymized supplementary zip (rebuild after anonymization)
- [ ] LaTeX source bundle (optional but recommended): `tmlr.sty`,
      `tmlr.bst`, `f2_methodology.bib`, `f2_methodology_anonymized_body.tex`,
      `test_compile_tmlr.tex`, `figures/*.png`
- [ ] OpenReview author profile complete (anonymous during review)

For MLRC 2026 EOI Google Form (loop 82 correction — NOT OpenReview):

- [ ] Paper submitted to TMLR via OpenReview FIRST (within window
      2025-06-20 ≤ submit ≤ 2026-09-30 AOE) — see
      [openreview.net/group?id=TMLR](https://openreview.net/group?id=TMLR)
- [ ] EOI Google Form filled at
      [forms.gle/bvYxagcRjKSmYhUM7](https://forms.gle/bvYxagcRjKSmYhUM7)
      using the text from `papers/tmlr_submission_kit/eoi_form_text.md`
- [ ] EOI soft deadline: **2026-06-04 AOE** — has passed; soft / non-blocking per Loops 82-84 framing — file EOI Google Form after TMLR submission
- [ ] TMLR hard decision deadline: **2026-09-30 AOE**

For GitHub Issue #1021:

- [ ] Status comment from
      `papers/tmlr_submission_kit/issue_1021_comment.md` posted

## 5. Decision tree

```
              ┌─────────────────────────────┐
              │ Did run_all_checks.sh PASS? │
              └──────┬──────────────────────┘
                     │
            ┌────────┴────────┐
            │NO              YES
            ▼                 ▼
   Fix until PASS    Did the GH Actions
   then re-check    workflow also PASS?
                            │
                  ┌─────────┴─────────┐
                  │NO                YES
                  ▼                   ▼
        Debug locally vs CI    Has the HARD TMLR
                                decision deadline
                                (2026-09-30 AOE) passed?
                                       │
                              ┌────────┴────────┐
                              │YES             NO
                              ▼                 ▼
                  Skip MLRC 2026 cycle;   Submit paper to TMLR
                  pivot to Causal-ML      (openreview.net/group?id=TMLR);
                  workshop (Oct deadline) THEN fill MLRC EOI Google
                                          Form (forms.gle/bvYxagcRjKSmYhUM7);
                                          THEN post #1021 status comment
```

**Note on the soft 2026-06-04 EOI deadline**: this is a soft date.
The EOI is filed *after* the paper enters TMLR review, so missing
this specific date does not require skipping MLRC. The binding date
is the hard TMLR decision deadline 2026-09-30 AOE. Loops 82-84
corrected the earlier framing that treated the soft date as
blocking.

## 6. Post-submission

If accepted to MLRC:

- [ ] Restore §10.3 Acknowledgments + funding disclosure
- [ ] Add author block to `test_compile_tmlr.tex`
- [ ] Switch tmlr.sty option: `\usepackage{tmlr}` → `\usepackage[accepted]{tmlr}`
- [ ] Update `papers/CHANGELOG.md` with the camera-ready loop

If rejected:

- [ ] Address reviewer comments
- [ ] Re-run `run_all_checks.sh` end-to-end after revisions
- [ ] Resubmit to alternative venue per `papers/f2_methodology.md` §10.2

## Anchor / version

- Checklist version: **Loop 157 (2026-06-07)**
- Branch HEAD at checklist update: refreshed in lock-step with the
  Loop 157 commits on `f2-methodology`
- Next deadline: hard TMLR 2026-09-30 AOE
  (EOI soft 2026-06-04 AOE has passed — non-blocking per Loops
  82-84 framing; file EOI Google Form *after* TMLR submission)

### Checklist change log

- Loops 81-84: initial checklist (7-stage CI gate).
- Loop 96: documented cold-clone wall-clock (15-30 min) + prereq probe.
- Loop 99: stage count 8 → 9 (added tables-vs-CSVs).
- Loop 103: stage count 9 → 10 (added formulas-vs-tables).
- Loop 106: stage count 10 → 12 (added #1021 paper to CI gate).
- Loop 108: stage count 12 → 13 (added label-consistency gate).
- Loop 109: page counts refreshed (27pp TMLR target was 23pp; non-anon
  was 37 → 42), supp zip 29 → 45 files, adversarial pass count
  18 → 31, anchor placeholders added.
- Loops 110-117: stage count 13 → 18 (added f2_pairwise_perm smoke,
  preamble verifier, and three #1021 pre-registered scripts
  verify_provenance.sh / verify_run_completeness.py /
  verify_report_consistency.py).
- Loops 118-124: stage count 18 → 20 (added stage-count consistency
  verifier, cross-paper consistency verifier, and cross-paper gate
  meta-test).
- Loops 125-127: 50-pass adversarial-review milestone reached;
  CommonMark fence parity gate added.
- Loop 128-129: page-count drift catches (non-anon 42 → 43);
  adversarial pass count 31 → 52; SUBMISSION_CHECKLIST §1
  exit-message stage count wired into the stage-count gate.
- Loop 130: stage count 21 → 22 (added
  `verify_submission_readiness.py`); §1 sub-bullet enumeration
  refreshed in lock-step with the new stage; cross-paper gate
  extended with RELATIONAL class (non-anon ≥ anon); adversarial
  pass count 52 → 53; sub-bullet `(1/22)` mis-label "companion
  paper" → "F2/main paper" fixed.
- Loop 131: stage count 22 → 23 (added
  `verify_changelog_consistency.py` that binds the three sites
  reporting cumulative pass count); RELATIONAL class grown
  1 → 4 entries (TMLR ≤ anon chain; lib ≤ grouped subset;
  #1021 binary ≥ F2 binary superset).
- Loop 132: stage count 23 → 24 (added
  `verify_anonymizer_completeness.py` ratchet gate against legacy
  70-bare-anchor baseline); `regen_changelog_section7.py` added as
  informational tool producing the auto-generated §7 breadcrumb;
  nearest-neighbor + stop-word extension + `[x]` regex + cold-clone
  errors landed in submission-readiness gate (54th-pass closures).
- Loop 133: stage count 24 → 26 (added
  `verify_cardinality_arithmetic.py` and `verify_generator_consistency.py`);
  EXACT_PIN_CLAIMS class extracted from 3 disguised SCOPED_DIFFs
  (cross-paper claims now 11 across 5 classes); F2 §E
  catalogue bare-anchor burn-down −7; 56th-pass UX/regex polish
  on changelog-consistency and submission-readiness gates.
- Loop 134: stage count 26 → 27 (added
  `verify_class_registry_binding.py` as 27th stage; class label
  drift between §1 and live registry now caught); #1021 §5.4
  partition bare-anchor burn-down −7 (anonymizer baseline 46
  → 39); anonymizer baselines migrated to JSON sidecar; manual
  pre-flight `verify_committed_state_consistency.py` added.
- Loop 135: stage count 27 → 28 (added
  `verify_documented_vs_extracted_consistency.py` binding §1
  description metadata to live verifier source); phi_ladder §5.4
  second-pass burn-down −7 (anonymizer baseline 39 → 32);
  cardinality registry gains `frozen=True` for historical
  snapshots; 6 quick SEV-3/4 closures (sidecar validation,
  traceback emission, dead flag, EXCLUDE_PATHS, "as of Loop X"
  annotation, docstring clarity).
- Loop 136: stage count stays 28 (committed-state
  staged-only kept as manual pre-commit tool, not CI stage);
  F2 §E remaining-anchors burn-down −6 (anonymizer baseline
  22 → 16); doc-vs-extracted registry grown 2 → 4 bindings
  (added (12) stage-count + (24) anonymizer); --update-baseline
  flag added to anonymizer for automated ratchet re-arm.
- Loop 137: stage count 28 → 29 (added verify_burn_down_history.py
  asserting breadcrumb trajectory matches sidecar); phi_ladder §5.4
  third-pass burn-down −5 (anonymizer baseline 32 → 27, total 48 → 43);
  _gate_utils.py shared helper extracted (closes import-helper
  duplication class); 60th-pass quick closures (regex anchors,
  lookahead boundary, FROZEN tag, EXCLUDE_PATHS extension).
- Loop 138: stage count 29 → 30 (added verify_alias_round_trip.py
  asserting CLASS_LABEL_ALIASES is bijective); F2 trailing-
  attributions burn-down −5 (anonymizer baseline 16 → 11,
  total 43 → 38); _to_int helper extracted to _gate_utils.py
  (consolidates _to_int + words_to_int duplicates);
  verify_pre_commit_hook.py manual tool added for hook
  installation detection.
- Loop 139: stage count 30 → 32 (added module-cache +
  floating-loop-anchor gates); F2 §3.5 burn-down −5
  (anonymizer baseline 11 → 6, total 38 → 33); import_gate
  module caching added with cache=True default; 61st-pass #7
  floating-loop-anchor drift class closed permanently.
- Loop 140: stage count 32 → 33 (added anchor-loop-coverage gate);
  meta-test extended 7 → 9 break-tests (burn-down arithmetic +
  alias bijection); test_gate_composition.sh shell smoke wrapper
  added; 62nd-pass #5/#13 closed in Loop 139 follow-up.
- Loop 141: stage count 33 → 34 (added dependency-graph gate);
  phi_ladder §4-§5 burn-down −3 (anonymizer baseline 27 → 24,
  total 33 → 30); meta-test extended 9 → 12 break-tests
  (doc-vs-extracted, changelog, stage-count); tier classification
  annotation added to STAGES.
- Loop 142: stage count 34 → 36 (added tier-classification + burn-
  down-trajectory gates); phi_ladder §1/§2.3/§3.1 attribution drops
  −6 (anonymizer baseline 24 → 18, total 30 → 24);
  GATE_AUTHORING_GUIDE.md added as internal methodology distillation.
- Loop 143: stage count 36 → 37 (added §10-authority gate);
  meta-test refactored with shared helpers + 2 new break-tests
  for stages 35/36 (12 → 14 total); tier semantics re-classified
  21/15 → 24/12 (submission/discipline) with contiguity check;
  verify_tier_classification gains contiguity assertion.
- Loop 144: stage count 37 → 38 (added guide-drift gate);
  RE-BASELINE escape hatch added to burn-down-trajectory gate;
  5 tempdir break-tests migrated to shared helpers (12 → 14
  total break-tests, 7 distinct classes covered); tier split
  24/12 → 24/14 after promoting stages 22-24 from discipline
  to submission (their fail modes ARE submission-blocking);
  `gate_dashboard.sh` manual project-health tool added.
- Loop 145: page counts refreshed (43/42/27 → 44/43/28 after
  anonymized.md content drift); §10-authority gate gains
  MANUAL_TOOL_ALLOWLIST inverse check; stage count 38 → 39
  (deadline-freshness gate). 67th-pass items #2/#3/#4/#6 closed.
- Loop 146: Posit16 (Gustafson 2017, es=1) codec lands at
  `src/phi_numbers/posit16.rs` (16 tests). Wired into
  `format_ladder.rs`; microbench produces 5-seed −74% rel L2
  vs GF16 at Xavier-init magnitudes. Format-zoo §9.4 arm gains
  a real number.
- Loop 147: citation ledger 27 → 32 VERIFIED
  (Semenov/Hochlehnert/NVFP4/Su/Cim/Gustafson, two attributions
  corrected from Loop 145 quick-check fabrications); stage count
  39 → 40 (`verify_tex_anonymization.py` scans anonymized .tex
  artifact for converter-side leaks — closes 67th-pass SEV-3 #5);
  68th-pass audit follow-up: Posit16 gains `Ord`/`PartialOrd`,
  ConversionCounter Display now emits all 14 tracked fields
  (was silently dropping posit16/int4/paretoq/fp8/int8).
- Loop 148: 6 Loop-147 citations folded into F2
  body prose (§2.3 Semenov + Hochlehnert; §9.4 NVFP4, Su, Cim,
  Gustafson). PDF page counts shift 44/43/28 → 46/46/29. 70th
  adversarial pass softened 3 overclaim risks (Posit16 framed
  as encode-time only; Su "strongly aligned" → "shared premise";
  NVFP4 framed as motivation not as validation of F2).
- Loop 149: format_microbench extended to 60-cell
  grid (4 d_model × 3 init × 5 seeds); stage count 40 → 41 added
  freshness gate asserting summary + per-cell JSONs exist. 71st-
  pass folded: Box-Muller singularity guard documented; "he"
  variant disambiguated from strict He-fan_in (uses d_model
  scaling for regime informativeness); ±std notation footnoted
  (sample std, not SEM; MC error ≈ 0.5× shown). Headline:
  Posit16 dominates GF16 in every grid cell (−70.6% to −88.6%
  rel L2, std ≤ 0.26%).
- Loop 150: F2 §9.4.1 added as new sub-subsection
  with the 4×3 grid table inline. PDF page counts shift
  46/46/29 → 48/47/30. 72nd-pass folded MC-error claim ambiguity,
  added 5-seed budget rationale + LCG-vs-RNG footnotes, made the
  encode-vs-train distinction explicit inside the "what the table
  says" paragraph (not only outside).
- Loop 151: closes 71st-pass deferred SEV-3 + SEV-4
  via shared grid config (`papers/scripts/format_microbench_grid_config.json`)
  + per-cell JSON schema validation. 73rd-pass folded inline:
  WARN-on-unknown-init in binary, filename↔content consistency
  check in gate, empty-grid-config FAIL, --grid --seed= behavior
  documented. Stage count unchanged at 41.
- Loop 152: Posit16 quire-bit accumulator landed at
  `src/phi_numbers/posit16_quire.rs`. PositQuire + posit16_dot
  public API; 19 unit tests; i128 fixed-point with 2⁻⁵⁶
  resolution + 71 bits integer headroom (exact for up to ~32k
  product accumulation). Total lib test count 735 → 754
  (+19); inventory 830 → 849.
- Loop 153: F2 §9.4.2 new sub-subsection with quire-
  microbench regime map (xavier × cancellation × {64,256,1024,4096});
  stage count 41 → 42 (`verify_quire_microbench_freshness.py`).
  Honest finding: quire ≈ f32-accum at F2's scale; both ~10× better
  than naive Posit16 sum. PDF page counts 48/47/30 → 50/49/31.
- Loop 154: 75th-pass SEV-3/4/5 deferred closures
  (dot_naive doc comment + stage 42 tier rationale + §9.4.2 scope
  note). 76th-pass folded inline: §9.4.2 prose reworded to match
  the gate's drift-catcher framing (was "gated for freshness ...
  alongside the §9.4.1 grid by a dedicated CI stage" — read as
  load-bearing; now framed explicitly as hygiene check on
  reproducibility provenance). PDF 50/49/31 → 50/50/31.
- Loop 155: new bridge_bench binary + F2 §9.4.3 (the
  training-time bridge from §9.4.1's encode-time grid); stage count
  42 → 43 (`verify_bridge_bench_freshness.py`). Sandbox training of
  f32 / GF16 / Posit16 at the embed gate: f32 = Posit16 to four
  decimal places; GF16 +0.0067 BPB worse. PDF 50/50/31 → 51/51/32.
- Loop 156: bridge_bench hardened to STEPS=200,
  N_seeds=5 to push past the random-byte plateau. New §9.4.3
  numbers: f32 = Posit16 = 4.5548 ± 0.0171 BPB; GF16 = 4.5845
  ± 0.0166 (+0.0297, 1.7× std, 3.9× MC SE at N=5 — statistically
  meaningful). 77th-pass SEV-2 #1 closed.
- Loop 157: this update — cold-clone submission-day rehearsal.
  Documented 3:30 min cold time + full-clone requirement (shallow
  breaks 3 gates) + 10 orphan SHAs hardened via `historical/<sha7>`
  lightweight tags pushed to remote. Stage count unchanged at 43.
