#!/usr/bin/env bash
# run_all_checks.sh — single-shot CI gate that runs every paper-side
# verification. Intended use: before any submission, run this script
# and only proceed if it exits 0.
#
# Stages (each must PASS):
#   (1) cross_reference_audit.py — §X.Y refs + arXiv format + file paths
#   (2) verify_paper_metadata.py — title parity, test count, BibTeX,
#       figure files
#   (3) generate_appendix_d.sh   — rebuild test inventory (Appendix D)
#   (4) compile_tmlr_test.sh     — xelatex compile all 3 variants
#       (non-anon, anon, real TMLR class)
#   (5) figure_regen.sh          — regenerate all 6 figures
#   (6) pack_supplementary.sh    — bundle supplementary zip (which
#       itself runs the 3-stage pre-flight from Loop 72)
#
# Output: PASS/FAIL summary on stdout. Exit 0 if every stage passes,
# 1 if any stage fails.
#
# Use:
#   papers/scripts/run_all_checks.sh
#
# Approx total wall time: ~30 s (no trainer invocation; figure regen
# is the largest component at ~10 s).

set -euo pipefail

CRATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CRATE_ROOT"

STAGES=(
    "cross-ref audit:papers/scripts/cross_reference_audit.py"
    "metadata verify:python3 papers/scripts/verify_paper_metadata.py"
    "test inventory regen:papers/scripts/generate_appendix_d.sh"
    "xelatex 3-variant compile:papers/scripts/compile_tmlr_test.sh"
    "figure regen:papers/scripts/figure_regen.sh"
    "supplementary pack:papers/tmlr_submission_kit/pack_supplementary.sh --skip-regen"
)

PASSED=0
FAILED=0
declare -a FAIL_NAMES=()

echo "# run_all_checks.sh — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

i=0
for stage in "${STAGES[@]}"; do
    i=$((i + 1))
    name="${stage%%:*}"
    cmd="${stage#*:}"
    echo "# (${i}/${#STAGES[@]}) ${name}"
    logfile="/tmp/run_all_checks_${i}.log"
    if eval "$cmd" > "$logfile" 2>&1; then
        # Last line of stdout (often shows result summary).
        last=$(tail -1 "$logfile" 2>/dev/null | head -c 100)
        echo "  PASS  — ${last}"
        PASSED=$((PASSED + 1))
    else
        echo "  FAIL  — see $logfile"
        FAIL_NAMES+=("$name")
        FAILED=$((FAILED + 1))
        tail -5 "$logfile" >&2
    fi
done

echo
echo "# Summary: $PASSED PASS / $FAILED FAIL of ${#STAGES[@]} stages"
if [[ $FAILED -gt 0 ]]; then
    echo "# FAILED stages: ${FAIL_NAMES[*]}" >&2
    exit 1
fi
echo "# All stages green. Paper + supplementary verified for submission."
exit 0
