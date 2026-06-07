# Camera-ready checklist — F2 paper post-acceptance

This file is the single-page execution checklist for converting the
TMLR submission to camera-ready *after* the paper is accepted. The
submission anchor commit is documented in
`papers/SUBMISSION_CHECKLIST.md` §1.

**Loop 159 (2026-06-07)**: this checklist was produced + smoke-tested
before submission. The smoke test compiled a `[accepted]` variant of
`test_compile_tmlr.tex`, verified the "Under review" banner is
replaced with "Published in Transactions on Machine Learning Research",
and confirmed the page count is 30 pp (vs the 32 pp under-review
variant — the [accepted] mode drops the submission boilerplate).

## Step 1: Verify acceptance

- [ ] TMLR Action Editor sent acceptance email with the OpenReview
      submission ID.
- [ ] Record the OpenReview submission ID in
      `papers/SUBMISSION_CHECKLIST.md` §6 ("If accepted to MLRC")
      change log.

## Step 2: Restore §10.3 Acknowledgments

The anonymized variant of §10.3 currently reads:
```
[OMITTED FOR DOUBLE-BLIND REVIEW. Restored at camera-ready.]
```

The full Acknowledgments block lives in the non-anonymized
`papers/f2_methodology.md` §10.3 (~line 1822-1834). To restore:

- [ ] Open `papers/scripts/anonymize_paper.py` and confirm the §10.3
      stripping rule is the only one that needs to be disabled.
- [ ] Either (a) edit `anonymize_paper.py` to skip the §10.3 strip
      when the env var `F2_CAMERA_READY=1` is set, or (b) manually
      replace the stub in `papers/f2_methodology_anonymized.md` §10.3
      with the body from `papers/f2_methodology.md` §10.3.
- [ ] Add author block to the top of `papers/f2_methodology.md` §1
      (currently anonymous-author placeholder).
- [ ] Add funding disclosure if any compute-grant decision lands
      before camera-ready deadline.

## Step 3: Flip tmlr.sty option

In `papers/tmlr_submission_kit/test_compile_tmlr.tex`:

```diff
-\usepackage{tmlr}        % Under-review (anonymous) mode.
-% \usepackage[accepted]{tmlr}  % Camera-ready mode.
+% \usepackage{tmlr}        % Under-review (anonymous) mode.
+\usepackage[accepted]{tmlr}  % Camera-ready mode.
```

- [ ] Recompile via `papers/scripts/compile_tmlr_test.sh`.
- [ ] Verify the rendered banner says "Published in Transactions on
      Machine Learning Research" instead of "Under review".

## Step 4: Update the bibliography

- [ ] Add the official TMLR citation (volume / number / pages / DOI)
      once known.
- [ ] Remove the `arXiv:NNNN.NNNNN` placeholder for the camera-ready
      bibliography if TMLR's bibtex style prefers the journal form.

## Step 5: Rebuild artifacts

- [ ] `papers/scripts/compile_tmlr_test.sh` exits 0 with all three
      variants green.
- [ ] `papers/tmlr_submission_kit/pack_supplementary.sh` rebuilds
      the supp zip (the encryption / hash invariants remain valid).
- [ ] `papers/scripts/run_all_checks.sh` exits 0 with 43/43 PASS.

## Step 6: Update CHANGELOG + SUBMISSION_CHECKLIST

- [ ] CHANGELOG §10 gains a "Camera-ready" entry naming the loop
      that flipped tmlr.sty.
- [ ] CHANGELOG §11 "Submission status" updated with the TMLR
      acceptance date + OpenReview submission ID.
- [ ] SUBMISSION_CHECKLIST §6 ("If accepted to MLRC") boxes ticked.

## Step 7: Upload camera-ready

- [ ] Upload the new `test_compile_tmlr.pdf` to OpenReview's
      camera-ready endpoint.
- [ ] Submit the LaTeX source bundle if TMLR requests it (`tmlr.sty`,
      `tmlr.bst`, `f2_methodology.bib`, body.tex, figures).

## Step 8: Post-camera-ready cleanup

- [ ] Tag the camera-ready commit: `git tag camera-ready/f2 HEAD`.
- [ ] Push the tag: `git push origin camera-ready/f2`.
- [ ] Branch HEAD at this point is the canonical "F2 paper as
      published in TMLR" snapshot.

---

**Pre-submission smoke test result** (Loop 159, 2026-06-07):
- `[accepted]` option compiled cleanly with `xelatex` 3-pass + bibtex.
- 30-page output (vs 32-page under-review variant).
- "Published in Transactions on Machine Learning Research" banner
  rendered correctly.
- No pdftotext rendering bugs (the same 4-stage sanity grep that
  `compile_tmlr_test.sh` runs passed on the [accepted] variant).
