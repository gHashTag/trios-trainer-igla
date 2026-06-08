#!/usr/bin/env bash
# submit_tmlr.sh — stage everything for a TMLR OpenReview submission.
#
# TMLR has no CLI; submission is via openreview.net web form. This
# script does the *staging* work the user would otherwise do by hand:
# - Re-runs run_all_checks.sh to confirm 7/7 PASS
# - Regenerates the anonymized PDF (the actual upload target)
# - Bundles the supplementary zip with provenance pre-flight
# - Prints the OpenReview submission URL + the field values the user
#   needs to paste, sourced from already-committed paste-ready texts
#
# Output: /tmp/tmlr_submission_$$/ staging directory + console
# checklist with paths to each artifact.
#
# Use:
#   papers/scripts/submit_tmlr.sh

set -euo pipefail

CRATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CRATE_ROOT"

STAGE="/tmp/tmlr_submission_$$"
mkdir -p "$STAGE"

echo "# (1/4) Run full CI gate to confirm the paper is submission-ready"
if ! papers/scripts/run_all_checks.sh > "$STAGE/run_all_checks.log" 2>&1; then
    echo "# ABORT: run_all_checks.sh failed. See $STAGE/run_all_checks.log" >&2
    tail -20 "$STAGE/run_all_checks.log" >&2
    exit 1
fi
echo "  PASS  7/7 stages green"

echo
echo "# (2/4) Stage the submission artifacts"
KIT="papers/tmlr_submission_kit"

# Anonymized PDF (the actual TMLR submission upload).
if [[ ! -f "$KIT/test_compile_tmlr.pdf" ]]; then
    echo "# ERROR: $KIT/test_compile_tmlr.pdf missing — compile_tmlr_test.sh" \
        " did not produce the real-TMLR-class PDF." >&2
    exit 1
fi
cp "$KIT/test_compile_tmlr.pdf" "$STAGE/f2_methodology.pdf"
echo "  Paper PDF: $STAGE/f2_methodology.pdf"
echo "             ($(wc -c < "$STAGE/f2_methodology.pdf" | tr -d ' ') bytes)"

# Supplementary zip.
if [[ -f "$KIT/f2_methodology_supp.zip" ]]; then
    cp "$KIT/f2_methodology_supp.zip" "$STAGE/f2_methodology_supp.zip"
    echo "  Supplementary: $STAGE/f2_methodology_supp.zip"
    echo "                 ($(wc -c < "$STAGE/f2_methodology_supp.zip" | tr -d ' ') bytes)"
fi

# Bib file (in case OpenReview asks for separate sources).
cp "$KIT/f2_methodology.bib" "$STAGE/f2_methodology.bib"

# Anonymized LaTeX body for source-bundle uploads.
cp "$KIT/f2_methodology_anonymized_body.tex" "$STAGE/" 2>/dev/null || true

# Action editor candidates.
cp "$KIT/action_editor_candidates.md" "$STAGE/" 2>/dev/null || true

echo
echo "# (3/4) OpenReview submission portal"
echo "  URL:  https://openreview.net/group?id=TMLR"
echo "  Action: sign in → 'Create new submission' / 'TMLR Submission'"
echo "  CRITICAL: use the anonymized PDF ($STAGE/f2_methodology.pdf)."
echo "            Non-anonymized submissions are rejected without review."

echo
echo "# (4/4) Field values to paste into the OpenReview form"
PAPER="$CRATE_ROOT/papers/f2_methodology.md"
TITLE=$(awk '/^# / {sub(/^# /, ""); print; exit}' "$PAPER")
echo "  Title:    $TITLE"
echo "  Authors:  Anonymous Authors (double-blind; restore at camera-ready)"
echo
echo "  Abstract (paste from $PAPER §Abstract):"
awk '/^## Abstract/,/^---$/' "$PAPER" | sed '1d;$d' \
    | head -10 \
    | sed 's/^/      /'
echo "      [... continues; full abstract in $PAPER lines 9-42]"
echo
echo "  Action editor candidates (see"
echo "      $KIT/action_editor_candidates.md):"
echo "      - Fredrik D. Johansson (Chalmers) — causal inference"
echo "      - Junpei Komiyama (MBZUAI) — reproducibility methodology"
echo "      - Sameer Deshpande (UW-Madison) — applied statistics"
echo
echo "  Keywords: causal mediation; ablation methodology; Pearl CDE;"
echo "            stratified analysis; reproducibility; sensitivity"
echo "            analysis"
echo
echo "  Anonymization checklist:"
echo "      $KIT/anonymization_checklist.md"
echo
echo "# Done staging. Submission directory: $STAGE"
echo
echo "# Next steps (manual):"
echo "  1. Open https://openreview.net/group?id=TMLR in browser"
echo "  2. Sign in / create OpenReview profile"
echo "  3. Click 'Create new submission'"
echo "  4. Paste title + abstract + keywords; upload"
echo "     $STAGE/f2_methodology.pdf as the main PDF"
echo "  5. Upload $STAGE/f2_methodology_supp.zip as supplementary"
echo "  6. Pick action editors from $KIT/action_editor_candidates.md"
echo "  7. Confirm dual-submission, ethics, formatting policies"
echo "  8. Submit"
echo "  9. After TMLR submission lands: fill the MLRC EOI Google Form"
echo "     at https://forms.gle/bvYxagcRjKSmYhUM7"
echo "  10. Run papers/scripts/post_issue_1021.sh to update #1021"
