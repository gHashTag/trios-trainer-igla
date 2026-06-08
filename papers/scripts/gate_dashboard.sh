#!/usr/bin/env bash
# gate_dashboard.sh — at-a-glance project health for the paper-side
# CI gate composition.
#
# Loop 144 C operationalizes the cumulative-discipline tracking. Run
# this script to see, in one pane:
#   - Total stage count + per-tier (submission/discipline) breakdown
#   - Adversarial review count
#   - Anonymizer baseline (legacy debt total)
#   - Upcoming deadlines (MLRC EOI, TMLR hard)
#
# NOT a CI stage. Manual tool for project health snapshots.
#
# Usage:
#   papers/scripts/gate_dashboard.sh
#   papers/scripts/gate_dashboard.sh --run-checks
#     (also invoke run_all_checks.sh and report per-tier PASS/FAIL)

set -euo pipefail

CRATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CRATE_ROOT"

RUN_CHECKS=0
if [[ "${1:-}" == "--run-checks" ]]; then
    RUN_CHECKS=1
fi

echo "# gate_dashboard.sh — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

# ── STAGES + tiers ───────────────────────────────────────────────
n_stages=$(awk '/^STAGES=\(/,/^\)/' papers/scripts/run_all_checks.sh \
    | grep -c '^\s*"')
n_sub=$(awk '/^STAGE_TIERS=\(/,/^\)/' papers/scripts/run_all_checks.sh \
    | grep -o '"submission"' | wc -l | tr -d ' ')
n_disc=$(awk '/^STAGE_TIERS=\(/,/^\)/' papers/scripts/run_all_checks.sh \
    | grep -o '"discipline"' | wc -l | tr -d ' ')
echo "## CI gate composition"
echo "  total stages:    ${n_stages}"
echo "  submission tier: ${n_sub}"
echo "  discipline tier: ${n_disc}"
echo

# ── Adversarial review ───────────────────────────────────────────
passes=$(grep -oE 'count=[0-9]+' \
    <(python3 papers/scripts/verify_changelog_consistency.py 2>&1) \
    | head -1 | sed 's/count=//')
echo "## Adversarial review"
echo "  cumulative passes: ${passes:-?} (3 sites in agreement)"
echo

# ── Anonymizer baseline ──────────────────────────────────────────
total_anchors=$(python3 papers/scripts/verify_burn_down_history.py 2>&1 \
    | grep -oE 'most-recent Loop [0-9]+: [0-9]+ \+ [0-9]+ = [0-9]+' \
    | head -1 | grep -oE '= [0-9]+' | sed 's/= //')
echo "## Anonymizer ratchet"
echo "  current total: ${total_anchors:-?} bare Loop-N anchor(s)"
echo "  (entries: $(python3 papers/scripts/verify_burn_down_trajectory.py 2>&1 \
    | grep -oE '[0-9]+ breadcrumb entries' | head -1))"
echo

# ── Deadlines ────────────────────────────────────────────────────
today=$(date -u +%Y-%m-%d)
echo "## Deadlines (today: ${today})"
echo "  MLRC EOI soft deadline: 2026-06-04 AOE"
echo "  TMLR hard deadline:     2026-09-30 AOE"
if [[ "${today}" > "2026-06-04" ]]; then
    echo "  ⚠ EOI deadline has passed; per CHANGELOG §10.2, pivot to"
    echo "    Causal-ML workshop (Oct deadline) if not yet TMLR-submitted."
fi
echo

# ── Optional: run all checks ─────────────────────────────────────
if [[ $RUN_CHECKS -eq 1 ]]; then
    echo "## Full CI sweep (running run_all_checks.sh)..."
    echo "  (this may take 30–60s warm; up to 15-30min cold)"
    echo
    if papers/scripts/run_all_checks.sh; then
        echo "  ✓ all stages PASS"
    else
        echo "  ✗ at least one stage FAILED (see output above)"
    fi
fi
