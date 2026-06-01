#!/usr/bin/env bash
# post_issue_1021.sh — extract and post the F2 status update comment
# to GitHub issue gHashTag/trios#1021.
#
# The comment body lives inside a ```markdown fence in
# papers/tmlr_submission_kit/issue_1021_comment.md (the surrounding
# file has local scaffolding). This script extracts the body,
# verifies gh auth, posts, and confirms.
#
# Usage:
#   papers/scripts/post_issue_1021.sh           # auth-check + post
#   papers/scripts/post_issue_1021.sh --dry-run # extract + show, no post
#
# Requires: gh CLI with authenticated session.

set -euo pipefail

CRATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CRATE_ROOT"

DRY_RUN=""
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "# WARN: unknown flag: $a" >&2 ;;
    esac
done

SRC="papers/tmlr_submission_kit/issue_1021_comment.md"
BODY="/tmp/issue_1021_body.md"

# Extract the body between ```markdown ... ``` fences.
awk '/^```markdown/,/^```$/' "$SRC" | sed '1d;$d' > "$BODY"

if [[ ! -s "$BODY" ]]; then
    echo "# ERROR: extraction produced empty body from $SRC" >&2
    exit 1
fi

lines=$(wc -l < "$BODY")
echo "# Extracted body: $lines lines from $SRC"
echo "# First line: $(head -1 "$BODY")"

if [[ -n "$DRY_RUN" ]]; then
    echo "# --dry-run: not posting. Body at $BODY"
    exit 0
fi

# Verify gh is authenticated.
if ! gh auth status > /tmp/gh_auth.log 2>&1; then
    echo "# ERROR: gh CLI is not authenticated. Run: gh auth login" >&2
    tail -5 /tmp/gh_auth.log >&2
    exit 1
fi
echo "# gh auth: OK ($(gh auth status 2>&1 | head -3 | tail -1))"

# Post the comment.
gh issue comment 1021 --repo gHashTag/trios --body-file "$BODY"

# Verify it landed.
echo
echo "# Verifying — most recent comment:"
gh issue view 1021 --repo gHashTag/trios --comments \
    | tail -10
