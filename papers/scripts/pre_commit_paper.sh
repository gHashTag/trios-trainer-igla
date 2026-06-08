#!/usr/bin/env bash
# pre_commit_paper.sh — fast pre-commit drift check (Loop 80).
#
# This is the sub-second subset of run_all_checks.sh: only the
# Python-only verifiers (cross-ref audit + metadata + SHA check)
# that complete in ~750 ms on warm caches. Stages 4-7 of the full
# run_all_checks.sh require xelatex / cargo and are reserved for
# the CI gate (.github/workflows/paper-checks.yml).
#
# Install as a pre-commit hook:
#   ln -sf ../../papers/scripts/pre_commit_paper.sh \
#       .git/hooks/pre-commit
#
# The hook is *opt-in* (manual symlink) rather than auto-installed
# so contributors who don't touch paper assets are unaffected.
#
# Only runs the checks if the staged diff touches paper / docs /
# data files. If the staged diff is purely Rust code or unrelated,
# exits 0 immediately.
#
# Usage (manual):
#   papers/scripts/pre_commit_paper.sh

set -euo pipefail

CRATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CRATE_ROOT"

# Inspect the staged diff. If git isn't available (rare), fall through
# and run all checks anyway.
if command -v git >/dev/null 2>&1; then
    staged=$(git diff --cached --name-only 2>/dev/null || echo "")
    if [[ -n "$staged" ]]; then
        touches_paper=0
        while IFS= read -r f; do
            case "$f" in
                papers/*|docs/F2_*|data/*) touches_paper=1; break ;;
            esac
        done <<< "$staged"
        if [[ $touches_paper -eq 0 ]]; then
            # No paper-side changes. Skip silently.
            exit 0
        fi
    fi
fi

echo "# pre-commit: paper drift check (fast subset)"

FAIL=0

if ! python3 papers/scripts/cross_reference_audit.py \
        > /tmp/pre_commit_xref.$$.log 2>&1; then
    echo "  FAIL  cross_reference_audit.py" >&2
    cat /tmp/pre_commit_xref.$$.log >&2
    FAIL=1
fi

if ! python3 papers/scripts/verify_paper_metadata.py \
        > /tmp/pre_commit_meta.$$.log 2>&1; then
    echo "  FAIL  verify_paper_metadata.py" >&2
    grep FAIL /tmp/pre_commit_meta.$$.log >&2 || true
    FAIL=1
fi

if ! python3 papers/scripts/check_no_fabricated_shas.py \
        > /tmp/pre_commit_sha.$$.log 2>&1; then
    echo "  FAIL  check_no_fabricated_shas.py" >&2
    grep FAIL /tmp/pre_commit_sha.$$.log >&2 || true
    FAIL=1
fi

rm -f /tmp/pre_commit_*.$$.log

if [[ $FAIL -eq 1 ]]; then
    echo "" >&2
    echo "# Commit blocked: paper drift detected." >&2
    echo "# Fix the failures above, re-stage, and re-commit." >&2
    echo "# To bypass (NOT RECOMMENDED): git commit --no-verify" >&2
    exit 1
fi

echo "  PASS  paper drift check (3 stages, ~750 ms)"
exit 0
