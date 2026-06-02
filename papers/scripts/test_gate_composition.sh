#!/usr/bin/env bash
# test_gate_composition.sh — minimal smoke wrapper that runs the full
# CI gate composition (`run_all_checks.sh`) and asserts exit 0.
#
# Loop 140 C operationalizes "for non-Python contributors / CI/CD
# integration": a single-shell-invocation smoke test that surfaces
# whether the 33-stage gate composition still holds.
#
# Usage:
#   papers/scripts/test_gate_composition.sh
#     (runs every stage; ~30s warm, 15-30min cold)
#
# Exit 0 if every stage passes; exit code from run_all_checks.sh
# otherwise.
#
# This is NOT a CI gate stage itself (recursive invocation would
# loop). Treat as a top-level entry point for shell-only CI runners.

set -euo pipefail

CRATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CRATE_ROOT"

echo "# test_gate_composition.sh — invoking papers/scripts/run_all_checks.sh"
echo "# (CRATE_ROOT=$CRATE_ROOT)"
echo

if papers/scripts/run_all_checks.sh; then
    echo
    echo "# test_gate_composition.sh — PASS (every stage exited 0)"
    exit 0
else
    rc=$?
    echo
    echo "# test_gate_composition.sh — FAIL (run_all_checks.sh exited $rc)" >&2
    exit "$rc"
fi
