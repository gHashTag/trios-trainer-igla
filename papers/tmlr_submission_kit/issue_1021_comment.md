# Draft comment for Issue #1021 — F2 framework + pre-reg status update

This file is the draft text of a GitHub comment to be posted on
gHashTag/trios#1021 ("phi-ladder vs format-zoo BPB head-to-head").
The comment closes the GitHub-side communication loop on the F2
methodology work that's been progressing on PR #185 since Loop 28.

**Posting**: when ready, paste the body of the fenced block below
into the issue comment composer.

---

```markdown
## F2 methodology framework — ready for the compute window when it lands

This issue has been silent while methodology infra has been built up
on PR #185 (`f2-methodology` branch). Short status:

**What's ready**
- 10 F2 binaries with stable long-form CSV contract
  (`docs/F2_BINARIES.md`)
- Pearl-style stratification framework: canonical / wd0 / warmup0
- Zhao-Luo / Daniel et al. 2015 four-PSE decomposition with
  delta-method SE valid at N=5 seeds
- Ohnishi-Li bridge-score sensitivity envelope (additive scale, BPB)
- Cross-stratum stability flag via CI overlap
- 727 tests including two named regression locks
  (`dual_mediation_no_interaction_residual_lock`,
  `trainer_internals_schema_is_load_bearing`)
- W3C-PROV preamble + `TRAINER_INTERNALS_SCHEMA` integrity lock
- Workshop-grade paper draft at `papers/f2_methodology.md` (1300+
  lines, end-to-end prose with 4 figures)
- TMLR submission kit at `papers/tmlr_submission_kit/` (template,
  EOI form text, anonymization checklist, supplementary-bundle script)
- Empirical anchor: `data/loop49/` (6 CSVs, MD5-checksummed)
  documenting the sandbox-scale RmsNorm × WD sign-flip
  (canonical NDE −4.12 BPB → wd0 CDE +0.43 BPB)

**Pre-registration** (`docs/F2_PRE_REG.md`)
- 8 configs × 2 strata × 5 seeds = 80 runs
- ~1B params, FineWeb 10B tokens, 50B training tokens, seq 2048
- Primary metric: held-out FineWeb BPB
- Analysis plan locked: aggregate → paired permutation (Zmigrod
  exact 2^5=32 enum) → BH-correct over 4 zoo comparisons within each
  stratum → cross-stratum stability flag

**What this issue needs to unblock**
1. **Compute target chosen** (Railway / personal GPU / cloud)
2. **Budget approved** (rough $ ceiling)
3. **Decision**: run only wd0 stratum (half compute, ~40 runs) or
   both strata (full, 80 runs)

Until those land, the pre-reg stays *draft-protocol, non-binding*.

**MLRC 2026 venue**
- Soft EOI deadline: **2026-06-04 AOE** (3 days from this comment)
- Hard TMLR decision deadline: **2026-09-30 AOE**
- Whether or not the champion-scale run happens, the methodology
  paper itself is publishable as-is at MLRC 2026 (sandbox-scale
  proof-of-concept). Submission decision is a separate question
  from compute decision.

PR #185 stays Draft until either champion-scale data lands OR the
methodology-paper-only path is taken explicitly.
```

---

## When to post

- After the user signs off on the wording, OR
- When MLRC EOI deadline approaches and a public commitment is needed

## How to post (when authenticated)

```bash
gh issue comment 1021 --repo gHashTag/trios \
    --body-file papers/tmlr_submission_kit/issue_1021_comment.md
```

(Strip the markdown fence wrapper; the file's outer scaffold is for
local reference only.)
