# Anonymization checklist — pre-submission

TMLR enforces double-blind review. Submissions that reveal authorship
are rejected without review. Walk this checklist before uploading.

## Mandatory strips

- [ ] **Author block** in LaTeX template: replace with `\author{Anonymous Authors}`
- [ ] **Acknowledgments section** (`§10.3` in current paper): remove
      entirely for submission; restore at camera-ready
- [ ] **GitHub URL / repo name**: replace any reference to
      `gHashTag/trios-trainer-igla` with `[anonymous-repo]`
- [ ] **Git SHAs**: the anchor commit `583b417` is identifying if
      checked against the public repo. Two options:
  - Option A (preferred): keep the SHA — it's load-bearing for
    reproducibility, and TMLR explicitly allows code/data links if
    anonymized
  - Option B: strip the SHA, reference "anchor commit (provided to
    reviewers upon request via OpenReview comment)"
- [ ] **Repo-local file paths**: `crates/trios-trainer-igla/...` →
      `[code-root]/...`
- [ ] **`prov:agent_git_sha`** field in CSV preambles: matches a
      public commit SHA. Either strip from supplementary CSVs or
      retain with Option A above.
- [ ] **`prov:host`**: contains hostname; strip from supplementary.

## Mandatory replaces

- [ ] **Email addresses**: none currently in paper. Confirm.
- [ ] **ORCID IDs**: none. Confirm.
- [ ] **Institutional affiliations**: none in body. Affiliations
      go in TMLR template author block only.

## Discretionary review (judgement calls)

- [ ] **Loop numbers** in commit messages and `docs/F2_PRE_REG.md`:
      these reference an internal numbering convention. They are not
      identifying per se but might be unusual enough to flag. Decision:
      keep — they trace to the loop-numbering convention doc.
- [ ] **`Co-Authored-By: Claude` trailers** in commit messages: if a
      reviewer clones the anonymized repo, they would see these. TMLR
      AI-assisted authorship disclosure policy applies; check the
      current TMLR policy before submission.

## Self-check

```bash
# Search the paper for identifying strings before upload:
grep -nE '(gHashTag|playra|trios-railway|@anthropic\.com|@claude\.com)' \
    papers/f2_methodology.md \
    papers/tmlr_submission_kit/*.md \
    data/loop49/README.md 2>&1 | head -20
# Expected: zero matches (after anonymization passes)
```

## Camera-ready restore

If the paper is accepted, restore (in order):
1. Author block in LaTeX template
2. `§10.3` Acknowledgments + funding disclosure
3. GitHub URL in repo references
4. Any stripped git SHAs
5. `prov:agent_git_sha` and `prov:host` fields in supplementary CSVs

The pre-anonymization diff should be kept as a single commit on the
camera-ready branch so the restore is a one-shot revert.
