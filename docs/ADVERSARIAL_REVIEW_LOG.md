# Adversarial review log — F2 methodology paper

**Update (Loop 144)**: this file documents the **first 18 passes
(Loops 59-94) in detail**. The discipline continued across Loops
95-144 to total **67 adversarial passes**; the post-94 passes
are tracked in `papers/CHANGELOG.md` §7 (Reviewer-screen feedback
loop) and §7.5 + §10 (per-loop CI gate evolution arcs). See also
per-loop commit messages on the `f2-methodology` branch
(run from the crate root: `cd crates/trios-trainer-igla && git log
--oneline --grep="adversarial pass" --grep="round-"`) for passes
19-67 individually.

This document consolidates the **first 18 adversarial review passes**
that were dispatched across Loops 59-94 of the F2 paper
development. Each pass was an independent subagent invocation
with a focused prompt; together they form the methodology that
took the paper from a Loop 49 sandbox-scale finding to a
submission-ready manuscript.

The catches are also reflected in `papers/CHANGELOG.md` §7.5
("Aesthetic finalization arc") and `papers/CITATIONS.md` audit
history. This file is the **methodology-first** view: how the
review pattern itself was structured.

## Headline statistics

- **67 adversarial passes total** (Loops 59-144); detail below for
  passes 1-18 covering Loops 59-94.
- **18 adversarial passes (Loops 59-94)** documented in this file
- **17 caught at least one issue** (Loops 59-93)
- **1 clean pass** (Loop 93 = first clean verdict)
- **~50 distinct catches** spanning citation errors, derivation
  issues, framing, anchor SHAs, submission flow, PDF rendering,
  and aesthetic structure
- Per-pass time budget: 15-30 minutes subagent + 15-30 minutes of
  follow-up edits

## Pass-by-pass headline

| # | Loop | Prompt focus | Headline catch |
|---|------|--------------|----------------|
| 1 | 59 | Math derivation audit | 8 issues incl. Miles-Shpitser citation (2→5 authors), Γ/Λ symbol reduction, removed unsupported smoking-cancer benchmark |
| 2 | 61 | TMLR hostile-reviewer screen | 3 desk-reject risks (synthetic counter framing, wd=0 pathological reading, MLRC track fit). All 3 closed in Loops 62-63 |
| 3 | 75 | Anchor SHA hygiene | `5367bde` (Loop 55) pre-dated empirical CSVs; switched to `583b417` (Loop 68) |
| 4 | 76 | Broad SEV sweep | 8 issues incl. §6.1 ↔ §5.3 self-contradiction, Zhao-Luo → Daniel et al. bulk rename, EOI count drift |
| 5 | 77 | Citation verification | Vaswani + Loshchilov bib entries verified; corrected one "Daniel et al.'s" possessive |
| 6 | 79 | Citation audit | AblationBench MIS-DESCRIBED (is about LM-agent ablation *planning*, not Welch/Cohen's-d). §2.3 + §9.1 rewritten |
| 7 | 80 | Citation audit on remaining CONFIRMED-VENUE | 4 catches: RO-Crate "Sefton" fabricated, InfiR2 paper withdrawn, Fibbinary attribution fabricated, QuEST title wrong |
| 8 | 82 | Submission flow | MLRC EOI is Google Form, not OpenReview; requires prior TMLR submission |
| 9 | 83 | EOI/OpenReview leak sweep | 9 more places still mis-stated submission flow |
| 10 | 84 | SUBMISSION_CHECKLIST consistency | Decision-tree still treated soft EOI deadline as binding |
| 11 | 85 | First PDF visual inspection | 2 SEV-5 catches: HTML banner in abstract, math symbols as literal `\{}Gamma` |
| 12 | 86 | PDF rendering bug hunt | 7 more catches: multi-line bold leaking, bibtex missing for non-anon, `\widehat` inside backticks, unicode `∈`/`γ`/`η` as U+FFFD |
| 13 | 87 | PDF aesthetic review | SEV-4 table truncation in §3.5.5 + §8.1 (highest desk-reject impact). Tables → bullets |
| 14 | 88 | §6.3/§7/§10.2 SEV-2 cleanup | §7 highest ROI; added 4 cluster subsections |
| 15 | 89 | §7 restructure verify | Item numbering mismatch caught (intro says "items 1, 4" but each subsection restarted at "1.") |
| 16 | 90 | Full PDF final sweep | **4 SEV-5/SEV-4 blockers** incl. table truncation + 7 bibliography orphans + Appendix C orphan |
| 17 | 91 | Bullet flow regression sweep | Loop 91 fix verified clean across all 10 lists in paper |
| 18 | 93 | Citation re-verify | Meng 2022 (ROME) confirmed; Wang 2023 author "Kevin Ro" → "Kevin" |
| 19 | 94 | Reader simulation (narrative quality) | §3.5.5 belongs in appendix — paper spends 3 pages telling reviewers how to verify before convincing them why result matters |

## Methodology patterns

### 1. Verify before publish

Every pass found something. The base-rate of "we're done" is
demonstrably wrong; the only way to know if a paper is
submission-ready is to run an adversarial pass that actively
looks for problems. Loop 93 was the first clean verdict — but
even that pass didn't make the paper better than it was; it
confirmed that prior passes had caught everything they could find.

### 2. Different prompts catch different things

Loops 59 (math), 61 (TMLR action-editor), 79-80 (citation
hygiene), 85-86 (PDF visual), 87 (aesthetic), 90 (final sweep),
93 (reader simulation) each found things the prior pass did not.
A single "review this paper" prompt would miss class-specific
issues that focused prompts catch.

### 3. The first PDF visual inspection happens at Loop 85

In 84 prior loops, the markdown source was rebuilt and the CI
gate ran, but nobody actually opened the rendered PDF. When Loop
85 finally did, two SEV-5 rendering bugs appeared instantly. The
inference: any pipeline involving Markdown → LaTeX → PDF must
include a visual PDF inspection as a CI stage, not just a
"compile cleanly" check. Loop 86 added pdftotext-grep to the CI
gate; Loop 87 added markdown lint upstream.

### 4. Loops cluster into themed arcs

- **Loops 59-61**: math + framing audit
- **Loops 75-77**: citation hygiene + anchor SHA
- **Loops 79-80**: deeper citation audit
- **Loops 82-84**: submission flow correction
- **Loops 85-91**: aesthetic finalization (PDF rendering + visual)
- **Loop 93**: first clean pass
- **Loop 94**: narrative quality (reader simulation, not bug hunt)

Each arc was triggered by a finding in the prior arc that
suggested a new class of issue worth scanning for. The catches
across arcs are independent — fixing all citation issues did
not prevent the PDF rendering bugs (because nobody had opened
the PDF), and vice versa.

### 5. Subagent prompts must be specific

Generic "review this paper" prompts produce generic "looks fine"
verdicts. The most productive prompts named a specific failure
class to look for ("look for tables that truncate at the right
margin", "look for fabricated SHAs"). The most productive single
prompt was Loop 90's "full PDF final sweep" with a 10-item
checklist — it caught 4 SEV-5/SEV-4 blockers in one pass.

## Recommended reading order

For a new contributor or future-maintainer wanting to understand
the paper's review history:

1. Read `papers/CHANGELOG.md` §7.5 for the per-loop catches
   summary.
2. Read `papers/CITATIONS.md` audit history for the citation-
   hygiene pattern (Loops 55, 58, 59, 60, 77, 79, 80, 93).
3. Read this file for the meta-methodology.
4. Read individual commit messages on the `f2-methodology` branch
   for the line-by-line fixes (`git log --oneline --grep="(Loop "
   <starting commit>..HEAD`).

## Anchor

This log was compiled at Loop 94 (2026-06-02), branch HEAD at
the time of writing.
