#!/usr/bin/env bash
# run_all_checks.sh — single-shot CI gate that runs every paper-side
# verification. Intended use: before any submission, run this script
# and only proceed if it exits 0.
#
# Stages (each must PASS): full ordered list is the STAGES=( ... )
# array below; the count is verified against paper claims by
# `verify_stage_count_consistency.py` (stage 12) and against the
# SUBMISSION_CHECKLIST.md §1 sub-bullet enumeration by
# `verify_submission_readiness.py` (stage 22). The §E catalogue in
# `papers/f2_methodology.md` references this script as the
# orchestrator; see `papers/CHANGELOG.md` §10 for the per-loop
# additions since the 8-script §E catalogue crystallized at Loop 96.
#
# Output: PASS/FAIL summary on stdout. Exit 0 if every stage passes,
# 1 if any stage fails.
#
# Use:
#   papers/scripts/run_all_checks.sh
#
# Approx total wall time:
#   - WARM (prior cargo build of f2_to_jsonl + f2_mediation_sensitivity
#     + f2_provenance_check; prior cargo test --no-run for lib + 10 F2
#     bins + 7 integration suites; TeX Live + Python deps installed):
#     ~30–60 s.
#   - COLD CLONE (no cargo cache, no warm target/): 15–30 minutes,
#     dominated by Rust compilation of a 38-binary workspace.
#
# Required external tools: xelatex, bibtex, pdftotext (poppler);
# python3 with matplotlib + numpy; zip; cargo (Rust toolchain).
#
# Pass --check-prereqs to run the dependency probe only (no stages).
# The probe runs by default before stage 1; missing deps cause an early
# exit 2 with platform-aware install hints.

set -euo pipefail

CRATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CRATE_ROOT"

# ---------------------------------------------------------------------------
# Prerequisite probe (Loop 97)
#
# Run before any stage so the failure mode is "missing dep with install hint"
# instead of an opaque eval error 30 seconds in.
#
# Use `--check-prereqs` to run the probe only and exit (useful in CI to
# decide whether to install dependencies before running the gate).
# ---------------------------------------------------------------------------
PROBE_ONLY=0
if [[ "${1:-}" == "--check-prereqs" ]]; then
    PROBE_ONLY=1
fi

case "$(uname -s)" in
    Darwin) INSTALL_HINT_TEX="brew install --cask mactex-no-gui"
            INSTALL_HINT_POPPLER="brew install poppler"
            INSTALL_HINT_PY="brew install python && python3 -m pip install matplotlib numpy"
            INSTALL_HINT_ZIP="(zip ships with macOS — should not be missing)"
            INSTALL_HINT_CARGO="curl https://sh.rustup.rs -sSf | sh" ;;
    Linux)  INSTALL_HINT_TEX="apt-get install -y texlive-xetex texlive-bibtex-extra"
            INSTALL_HINT_POPPLER="apt-get install -y poppler-utils"
            INSTALL_HINT_PY="apt-get install -y python3-pip && pip3 install matplotlib numpy"
            INSTALL_HINT_ZIP="apt-get install -y zip"
            INSTALL_HINT_CARGO="curl https://sh.rustup.rs -sSf | sh" ;;
    *)      INSTALL_HINT_TEX="install TeX Live with xelatex + bibtex"
            INSTALL_HINT_POPPLER="install poppler (pdftotext)"
            INSTALL_HINT_PY="install python3 with matplotlib + numpy"
            INSTALL_HINT_ZIP="install zip"
            INSTALL_HINT_CARGO="install Rust toolchain" ;;
esac

probe_miss=0
probe_one() {
    local name="$1"; local check="$2"; local hint="$3"
    if eval "$check" > /dev/null 2>&1; then
        printf "  OK    %s\n" "$name"
    else
        printf "  MISS  %s  — install: %s\n" "$name" "$hint" >&2
        probe_miss=$((probe_miss + 1))
    fi
}

echo "# prereq probe — $(uname -s)"
probe_one "xelatex"   "command -v xelatex"   "$INSTALL_HINT_TEX"
probe_one "bibtex"    "command -v bibtex"    "$INSTALL_HINT_TEX"
probe_one "pdftotext" "command -v pdftotext" "$INSTALL_HINT_POPPLER"
probe_one "python3"   "command -v python3"   "$INSTALL_HINT_PY"
probe_one "matplotlib (python3)" "python3 -c 'import matplotlib'" "$INSTALL_HINT_PY"
probe_one "numpy (python3)"      "python3 -c 'import numpy'"      "$INSTALL_HINT_PY"
probe_one "zip"     "command -v zip"   "$INSTALL_HINT_ZIP"
probe_one "cargo"   "command -v cargo" "$INSTALL_HINT_CARGO"
probe_one "git"     "command -v git"   "(git is required — should already be installed)"

if [[ $probe_miss -gt 0 ]]; then
    echo "" >&2
    echo "# prereq probe: $probe_miss missing tool(s). Install them and re-run." >&2
    exit 2
fi
echo "# prereq probe: OK"
echo

if [[ $PROBE_ONLY -eq 1 ]]; then
    echo "# --check-prereqs requested; exiting without running stages."
    exit 0
fi

STAGES=(
    "cross-ref audit:papers/scripts/cross_reference_audit.py"
    "metadata verify:python3 papers/scripts/verify_paper_metadata.py"
    "no fabricated SHAs:python3 papers/scripts/check_no_fabricated_shas.py"
    "markdown lint:python3 papers/scripts/lint_paper_md.py"
    "tables vs CSVs:python3 papers/scripts/verify_tables_against_csv.py"
    "formulas vs tables:python3 papers/scripts/verify_formulas_vs_tables.py"
    "label consistency:python3 papers/scripts/verify_label_consistency.py"
    "preamble per producer:python3 papers/scripts/verify_preamble_per_producer.py"
    "provenance gate:papers/scripts/verify_provenance.sh"
    "run completeness:papers/scripts/verify_run_completeness.py"
    "report consistency:papers/scripts/verify_report_consistency.py"
    "stage count consistency:papers/scripts/verify_stage_count_consistency.py"
    "cross-paper consistency:papers/scripts/verify_cross_paper_consistency.py"
    "cross-paper gate meta-test:papers/scripts/meta_test_cross_paper_gates.py"
    "f2_pairwise_perm smoke:papers/scripts/smoke_f2_pairwise_perm.sh"
    "#1021 cross-ref audit:papers/scripts/cross_reference_audit.py papers/phi_ladder_paper_intro_draft.md"
    "#1021 markdown lint:python3 papers/scripts/lint_paper_md.py papers/phi_ladder_paper_intro_draft.md"
    "test inventory regen:papers/scripts/generate_appendix_d.sh"
    "xelatex 3-variant compile:papers/scripts/compile_tmlr_test.sh"
    "figure regen:papers/scripts/figure_regen.sh"
    "supplementary pack:papers/tmlr_submission_kit/pack_supplementary.sh --skip-regen"
    "submission readiness:python3 papers/scripts/verify_submission_readiness.py"
    "changelog consistency:python3 papers/scripts/verify_changelog_consistency.py"
    "anonymizer completeness:python3 papers/scripts/verify_anonymizer_completeness.py"
    "cardinality arithmetic:python3 papers/scripts/verify_cardinality_arithmetic.py"
    "generator consistency:python3 papers/scripts/verify_generator_consistency.py"
    "class registry binding:python3 papers/scripts/verify_class_registry_binding.py"
    "documented vs extracted:python3 papers/scripts/verify_documented_vs_extracted_consistency.py"
    "burn-down history:python3 papers/scripts/verify_burn_down_history.py"
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
