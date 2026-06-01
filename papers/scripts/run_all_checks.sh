#!/usr/bin/env bash
# run_all_checks.sh — single-shot CI gate that runs every paper-side
# verification. Intended use: before any submission, run this script
# and only proceed if it exits 0.
#
# Stages (each must PASS):
#   (1) cross_reference_audit.py — §X.Y refs + arXiv format + file paths
#   (2) verify_paper_metadata.py — title parity, test count, BibTeX,
#       figure files
#   (3) check_no_fabricated_shas.py — git cat-file -e per SHA-like token
#   (4) lint_paper_md.py         — Markdown lint (6 checks, upstream of
#       LaTeX render)
#   (5) verify_tables_against_csv.py — Loop 99: every numeric row in a
#       registered table matches its source CSV at 2-decimal tolerance
#   (6) verify_formulas_vs_tables.py — Loop 103: four-PSE closure +
#       §3.3 envelope Γ_tip claims re-derived from CIs
#   (7) #1021 cross-ref audit — Loop 106: cross_reference_audit.py
#       run on phi_ladder_paper_intro_draft.md (multi-paper mode)
#   (8) #1021 markdown lint — Loop 106: lint_paper_md.py on the
#       same draft (multi-paper mode)
#   (9) generate_appendix_d.sh   — rebuild test inventory (Appendix D)
#   (10) compile_tmlr_test.sh    — xelatex compile all 3 variants
#       (non-anon, anon, real TMLR class)
#   (11) figure_regen.sh         — regenerate all 6 figures
#   (12) pack_supplementary.sh   — bundle supplementary zip (which
#       itself runs the 3-stage pre-flight from Loop 72)
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
    "#1021 cross-ref audit:papers/scripts/cross_reference_audit.py papers/phi_ladder_paper_intro_draft.md"
    "#1021 markdown lint:python3 papers/scripts/lint_paper_md.py papers/phi_ladder_paper_intro_draft.md"
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
