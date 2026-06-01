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

**What's ready (as of Loop 66, 2026-06-01)**
- 10 F2 binaries with stable long-form CSV contract
  (`docs/F2_BINARIES.md`)
- Pearl-style stratification framework: canonical / wd0 / warmup0
- Daniel et al. 2015 / Gao-Li-Luo 2020 four-PSE decomposition with
  delta-method SE valid at N=5 seeds
- Ohnishi-Li bridge-score sensitivity envelope (additive scale, BPB)
- Cross-stratum stability flag via CI overlap
- 727 tests including two named regression locks
  (`dual_mediation_no_interaction_residual_lock`,
  `trainer_internals_schema_is_load_bearing`)
- W3C-PROV preamble + `TRAINER_INTERNALS_SCHEMA` integrity lock
- Paper draft at `papers/f2_methodology.md` (~1500 lines, end-to-end
  prose with 5 figures, compiles cleanly to 35-page xelatex PDF)
- TMLR submission kit at `papers/tmlr_submission_kit/`
  (template.tex, eoi_form_text.md, anonymization checklist,
  pack_supplementary.sh, f2_methodology.bib with 22 verified entries)
- Empirical anchor: `data/loop49/` (6 CSVs, MD5-checksummed) +
  `data/loop49_swap/` (4 additional swap-parameterization CSVs from
  Loop 64 Phase 0)
- **Two findings, both sandbox-scale (5 seeds, ~8K params, 200 steps)**:
  - RmsNorm × WD sign-flip: canonical NDE −4.12 BPB → wd0 Pearl CDE
    +0.43 BPB, both CIs excluding zero. Framed as unit-test
    demonstration that the framework detects a sign flip when one
    is constructed (per Loop 61 hostile-reviewer screen).
  - Swap-parameterization invariant: the rms-mediated indirect effect
    of weight decay is byte-identical −0.751 [−1.325, −0.177] across
    all three strata — the only PSE row in our matrix that survives
    intact across canonical, wd0, and warmup0 reference points
    (Loop 64 Phase 0 result).
- **9-paper survey** of recent transformer ablation papers
  (NormFormer, OPT, Switch Transformer, Pythia, Mamba, Llama,
  BitNet b1.58, Peri-LN, nanoGPT): 7 single-run / 2 multi-seed
  summary / 0 per-seed CSVs released. Illustrates the gap F2 closes.

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

**Citation hygiene history** (6 catches across the paper's loops)
- Loop 55 — corrected Gao-Li-Luo attribution (paper had said "Zhao &
  Luo"); corrected Hagmann/Riezler (paper had said "Semmelrock");
  corrected BitNet 2402.17764 vs 2504.12285 conflation; removed
  unverifiable Alvarez-Bartolo & MacKinnon entry
- Loop 59 — corrected Miles & Shpitser to the full 5-author list
  (Miles, Shpitser, Kanki, Meloni, Tchetgen Tchetgen); corrected the
  Γ_tip < 1.25 / ≥ 2.0 threshold attribution from VanderWeele-Ding
  to Haneuse-VanderWeele-Arterburn (JAMA 2019)
- Loop 60 — switched primary attribution Gao-Li-Luo → Daniel et al.
  2015 (foundational two-mediator decomposition)
- Loop 77 — verified Vaswani 2017 NeurIPS + Loshchilov ICLR 2019
  citations for the §5.2 wd=0 defense
- Loop 79 — corrected AblationBench mis-description (paper is about
  LM-agent ablation *planning*, not wide-form / paired Welch / Cohen's
  d analysis); corrected author "Abramovich et al." → "Abramovich &
  Chechik"
- Loop 80 — caught FOUR additional issues in one pass: (i) RO-Crate
  first author "Sefton" was fabricated (actually Leo + Soiland-Reyes,
  PLoS ONE 2024); (ii) **InfiR2 FP8 paper (arXiv:2509.22536) was
  withdrawn 2025-10-17 by authors due to a data-processing pipeline
  bug** — §9.4 now explicitly flags this; (iii) Fibbinary paper
  attribution "Schmidt-Mengin et al." was fabricated (actually
  Fiandaca & Gomony, and the paper is about neural radio receivers,
  not transformer LLMs); (iv) QuEST title was wrong

`papers/CITATIONS.md` documents each entry's verification status.
Current ledger: **24 VERIFIED + 1 VERIFIED-WITHDRAWN (96%), 0
CONFIRMED-VENUE**.

**MLRC 2026 venue**
- Soft EOI deadline: **2026-06-04 AOE** (single-digit hours from
  Loop 81 push)
- Hard TMLR decision deadline: **2026-09-30 AOE**
- Author notifications: 2026-10-07; in-person presentation
  NeurIPS Sydney 2026-12-06–13
- Whether or not the champion-scale run happens, the methodology
  paper itself is publishable as-is at MLRC 2026 (sandbox-scale
  proof-of-concept + 9-paper applicability survey). Submission
  decision is a separate question from compute decision.

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
