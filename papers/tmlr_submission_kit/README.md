# TMLR / MLRC 2026 submission kit

This directory holds the artifacts needed to submit
`papers/f2_methodology.md` to TMLR via OpenReview, with the goal of
acceptance into the MLRC 2026 NeurIPS official track.

**Status (Loop 59)**: skeleton only. No decision-to-submit has been
made by the user. This kit is built to be reusable if MLRC 2026 is
declined and the paper is redirected to Causal-ML Workshop or ICML
2027 main track per `papers/f2_methodology.md` §10.2.

## Deadline reminders

| Date | Event | Notes |
|------|-------|-------|
| **2026-06-04 AOE** | MLRC "intent to submit" soft deadline | OpenReview EOI form opens earlier |
| **2026-09-30 AOE** | Hard TMLR decision deadline | Paper must be TMLR-accepted by this date to be MLRC-eligible |
| **2026-10-07** | MLRC author notifications | Acceptance / rejection at this point |
| **2026-12-06 – 12-13** | NeurIPS 2026 Sydney | In-person presentation if accepted |

## Submission path: TMLR → MLRC

1. Submit anonymized paper PDF + supplementary materials to TMLR via
   OpenReview (TMLR LaTeX template, see `template.tex`).
2. Once accepted (with no or minor revisions), submit the OpenReview
   self-nomination form to MLRC 2026.
3. On MLRC acceptance, update camera-ready info and prepare a Sydney
   in-person presentation.

## Kit contents

- `README.md` (this file) — kit overview + deadlines
- `manifest.md` — list of supplementary materials to upload (zip them
  via `pack_supplementary.sh` when ready)
- `pack_supplementary.sh` — script that produces
  `f2_methodology_supp.zip` from `data/loop49/`, `papers/figures/`, and
  `papers/appendix_d_test_inventory.md`
- `template.tex` — TMLR LaTeX skeleton ready to receive the body of
  `papers/f2_methodology.md`. Not auto-generated; the conversion is
  manual (Markdown → LaTeX) and tracked separately.
- `eoi_form_text.md` — draft "intent to submit" text for the MLRC EOI
  OpenReview form
- `anonymization_checklist.md` — items to verify before upload

## What this kit does NOT do

- Does not auto-convert Markdown → LaTeX (manual step; the body of
  `papers/f2_methodology.md` is ~1300 lines of structured Markdown,
  and the body math is already in dollar-delimited LaTeX so the lift
  is modest but not zero).
- Does not auto-anonymize the paper. See `anonymization_checklist.md`.
- Does not submit on the user's behalf. Submission is an explicit
  decision step.

## Reusability if MLRC is declined

If the user declines MLRC 2026, this kit drops `eoi_form_text.md` and
the TMLR-specific items, keeping the supplementary-pack script and the
LaTeX skeleton intact. The skeleton is venue-agnostic; only the
preamble would change.
