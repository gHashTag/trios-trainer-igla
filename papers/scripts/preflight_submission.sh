#!/usr/bin/env bash
# preflight_submission.sh — submission-day one-shot orchestrator.
#
# Reduces the 4-command submission workflow to a single invocation:
#   1. run_all_checks.sh (full 8-stage CI gate)
#   2. submit_tmlr.sh (stage anonymized PDF + supplementary + console
#      checklist)
#   3. echo paste-ready MLRC EOI Google Form text
#   4. echo paste-ready Issue #1021 comment text + post command
#
# Output: console summary + staging path. No actual submission
# occurs — TMLR and the EOI Google Form are web-only and the
# Issue #1021 post is delegated to papers/scripts/post_issue_1021.sh
# after gh authentication.
#
# Usage:
#   papers/scripts/preflight_submission.sh
#   papers/scripts/preflight_submission.sh --skip-checks  (skip the
#       8-stage CI gate; useful for re-staging after a successful
#       prior preflight)

set -euo pipefail

CRATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CRATE_ROOT"

SKIP_CHECKS=""
for a in "$@"; do
    case "$a" in
        --skip-checks) SKIP_CHECKS=1 ;;
        *) echo "# WARN: unknown flag: $a" >&2 ;;
    esac
done

echo "============================================================"
echo "  F2 paper — submission-day preflight orchestrator"
echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo

# (1/4) Run the full CI gate.
if [[ -z "$SKIP_CHECKS" ]]; then
    echo "# (1/4) Full CI gate (run_all_checks.sh, ~46 s)"
    if ! papers/scripts/run_all_checks.sh > /tmp/preflight_checks.log 2>&1; then
        echo "# ABORT: run_all_checks.sh failed. Last 20 lines:" >&2
        tail -20 /tmp/preflight_checks.log >&2
        exit 1
    fi
    tail -3 /tmp/preflight_checks.log
else
    echo "# (1/4) Full CI gate — SKIPPED via --skip-checks"
fi
echo

# (2/4) Stage TMLR submission artifacts.
echo "# (2/4) Staging TMLR submission artifacts"
papers/scripts/submit_tmlr.sh > /tmp/preflight_submit.log 2>&1
STAGE=$(grep "Submission directory:" /tmp/preflight_submit.log | tail -1 | awk '{print $NF}')
if [[ -z "$STAGE" || ! -d "$STAGE" ]]; then
    echo "# ABORT: submit_tmlr.sh did not produce a staging dir" >&2
    tail -10 /tmp/preflight_submit.log >&2
    exit 1
fi
echo "  Staged at: $STAGE"
ls -la "$STAGE" | tail -7 | head -7
echo

# (3/4) MLRC EOI Google Form text — copy to clipboard if pbcopy
# available, else echo.
echo "# (3/4) MLRC EOI Google Form text"
echo "  Source: papers/tmlr_submission_kit/eoi_form_text.md"
echo "  Target: https://forms.gle/bvYxagcRjKSmYhUM7"
echo "  Note:   Submit ONLY AFTER the paper is under TMLR review."
echo "          The EOI is filed within the 2025-06-20 to"
echo "          2026-09-30 AOE TMLR submission window."
echo
echo "  First 5 lines of EOI text:"
sed -n '23,28p' papers/tmlr_submission_kit/eoi_form_text.md \
    | sed 's/^/      /'
echo "      [...full text in source file...]"
if command -v pbcopy >/dev/null 2>&1; then
    sed -n '23,$p' papers/tmlr_submission_kit/eoi_form_text.md | pbcopy
    echo "  EOI text copied to clipboard (pbcopy)."
fi
echo

# (4/4) Issue #1021 status comment.
echo "# (4/4) Issue gHashTag/trios#1021 status comment"
echo "  Posting command:"
echo "    papers/scripts/post_issue_1021.sh           # post if authed"
echo "    papers/scripts/post_issue_1021.sh --dry-run # extract only"
echo
echo "  Body preview (first 5 lines):"
awk '/^```markdown/,/^```$/' \
    papers/tmlr_submission_kit/issue_1021_comment.md \
    | sed '1d;$d' | head -5 | sed 's/^/      /'
echo "      [...full body extractable via post_issue_1021.sh --dry-run]"
echo

echo "============================================================"
echo "  PREFLIGHT COMPLETE."
echo
echo "  Next steps (manual):"
echo "    1. Open https://openreview.net/group?id=TMLR in browser"
echo "    2. Click 'Create new submission'; upload:"
echo "       Main PDF:      $STAGE/f2_methodology.pdf"
echo "       Supplementary: $STAGE/f2_methodology_supp.zip"
echo "    3. Pick action editors from \\"
echo "       papers/tmlr_submission_kit/action_editor_candidates.md"
echo "    4. After TMLR submission lands: open"
echo "       https://forms.gle/bvYxagcRjKSmYhUM7"
echo "       and paste the EOI text already copied to clipboard"
echo "    5. After EOI lands: run"
echo "       papers/scripts/post_issue_1021.sh"
echo "============================================================"
