# Case study — F2 applicability survey across recent published ablations

Loop 63 deliverable. This document closes the Loop 61 reviewer-flag
"MLRC reproducibility-track fit asserted but not earned" by surveying
whether the published transformer-ablation literature releases the
seed-level data F2 needs, and demonstrating what F2 *would* analyze
on one paper where the published summary statistics are sufficient.

This is a **forward-looking demonstration**, not a re-run of source
data. The original ablation authors did not release per-seed CSVs,
which is itself one of the methodological gaps F2 is designed to
close.

## Survey: 3 recent transformer-ablation papers

| Paper | Year | Ablation focus | Seeds / cell | Variance reported | Per-seed data released | F2-applicable as-is |
|---|---|---|---|---|---|---|
| NormFormer (Shleifer et al., arXiv:2110.09456) | 2021 | Pre-LN/Post-LN/extra LayerNorms | 1 | No | No | **No** (single-run) |
| BitNet b1.58 (Ma et al., arXiv:2402.17764) | 2024 | FP16 vs INT8 vs ternary 1.58-bit | 1 | No | No | **No** (single-run) |
| Peri-LN (Kim et al., arXiv:2502.02732) | 2025 | Pre-LN/Post-LN/Peri-LN | 5 | Yes (std reported per benchmark) | No | **Partial** (summary stats only) |

**Headline result of the survey**: of three recent, well-cited
transformer-architecture ablation papers, **two** publish single-run
tables and **one** publishes multi-seed summary statistics but does
not release per-seed CSVs. The F2 framework cannot be applied to the
single-run papers (no variance information at all); it can be
applied to Peri-LN's reported summaries via shape-matched
simulation, but a true F2 re-analysis would require the per-seed
data the authors did not release.

This survey illustrates the methodological gap F2 is designed to
close: even in 2025, multi-seed ablation reporting with per-seed
data release is **not the norm** in transformer architecture papers.

## Detailed walkthrough — Peri-LN Table 1

[Web search at 2026-06-01 surfaces:]

> Peri-LN paper Table 1, 400M parameters, 5 seeds, average benchmark
> score across {ARC-Easy, HellaSwag, PIQA, SIQA, Winogrande}:
> - Post-LN: 42.45 (per-benchmark std 1.09–2.35)
> - Pre-LN: 49.69 (per-benchmark std 1.63–2.35)
> - Peri-LN: 51.57 (per-benchmark std 0.67–0.81)
>
> Note: "The standard deviation of the benchmark results across
> different training seeds is reduced by more than half with Peri-LN."

### What F2 would ask

A standard seed-mean analysis on this table concludes:
1. Peri-LN > Pre-LN > Post-LN on average benchmark score.
2. Peri-LN has roughly half the per-seed std of the alternatives.

F2 would additionally surface:
- **Is the gap between Pre-LN and Peri-LN robust to mediator
  confounding?** The three normalization choices in the Peri-LN
  paper are run with *different default learning rates and warmup
  schedules* (Pre-LN classically tolerates higher LRs and shorter
  warmup; Post-LN typically needs longer warmup; Peri-LN's authors
  may have tuned these per-norm to give each its best shot). If the
  ablation does not stratify on (LR, warmup), then the per-norm
  comparison is the *marginal* effect of normalization choice,
  averaged over the joint of all other recipe knobs.
- **What is the Pearl CDE of normalization choice at a single fixed
  LR**? This is the apples-to-apples comparison that controls for
  the LR-mediator confound. F2 would compute this if a sweep over
  (norm, LR, warmup, seed) were available.

### What F2 cannot do without the missing data

The Peri-LN paper reports averages and per-benchmark stds, not
per-seed values, and does not include LR-stratified runs in Table
1. Without:
- per-seed benchmark scores, and
- a (norm × LR) grid sweep,

F2 cannot empirically compute the Pearl CDE on LR for this paper.
The F2 framework's *prediction* — that the per-norm comparison
might survive LR pinning, given the apparent Peri-LN std reduction
— is testable by the original authors and not by us. We frame this
as a forward-looking application of F2's machinery, not as a
re-analysis claim.

## What this case study supports for the F2 paper

1. **F2 fills a real methodological gap.** Survey shows the
   multi-seed-with-data-release norm is not yet established in the
   published transformer-ablation literature, even in 2025. F2's
   long-form CSV contract + W3C-PROV preamble + reproducibility-
   checklist discipline are concrete defaults that future ablation
   reports could adopt.
2. **F2 produces testable predictions on third-party ablations.**
   The Peri-LN walkthrough demonstrates that F2's stratification-
   over-mediator question is well-posed against any published
   ablation table, even if the original authors did not run the
   stratified comparison.
3. **F2 surfaces what the published comparison cannot answer.**
   The "Pre-LN tolerates higher LR" folklore is exactly the kind
   of mediator-confounding pattern F2 is designed to handle.

## Limitations of this case study

- **No re-analysis with original data.** This is a forward-looking
  walkthrough; the F2 framework cannot be tested against Peri-LN's
  data without access to per-seed runs the authors did not release.
- **Single detailed case.** A stronger paper would include 2-3
  detailed walkthroughs. We have time-budgeted one for the present
  submission and pre-register expanded case studies as Phase-2
  follow-up.
- **Selection bias of papers surveyed.** The three papers above are
  high-visibility ablations; the underlying methodological gap
  applies more broadly, but a systematic survey of, say, the top-100
  cited transformer ablation papers since 2017 is outside the scope
  of the present submission.

## Pointer from the main paper

Referenced from `papers/f2_methodology.md` §9.1 (ML ablation
methodology) as the bridge between the F2 framework and
third-party published claims.

## Anchor

- Survey date: 2026-06-01 (Loop 63)
- Peri-LN Table 1 data quoted from WebFetch of arXiv:2502.02732v3
- NormFormer single-run reporting confirmed via WebSearch
- BitNet b1.58 single-run reporting confirmed via WebSearch
