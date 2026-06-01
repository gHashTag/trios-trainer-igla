#!/usr/bin/env bash
# compile_tmlr_test.sh — verify the paper compiles cleanly under all
# three submission-relevant LaTeX configurations:
#
#   (1) Non-anonymized variant against the vanilla article wrapper
#       (papers/tmlr_submission_kit/test_compile.tex). Maximum
#       portability sanity check.
#   (2) Anonymized variant against the same vanilla article wrapper
#       (papers/tmlr_submission_kit/test_compile_anon.tex). Verifies
#       the anonymizer + converter produce LaTeX that still compiles.
#   (3) Anonymized variant against the REAL TMLR class
#       (papers/tmlr_submission_kit/test_compile_tmlr.tex, using the
#       JmlrOrg tmlr.sty + tmlr.bst). The actual submission target.
#
# Each variant requires xelatex (TeX Live ships it). On macOS:
#   brew install --cask mactex
#
# Output: papers/tmlr_submission_kit/test_compile{,_anon,_tmlr}.pdf
# Exit: 0 if every variant succeeds, 1 otherwise.

set -euo pipefail

CRATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CRATE_ROOT"

KIT="papers/tmlr_submission_kit"
SCRIPT="papers/scripts/md_to_tmlr_tex.py"

run_xelatex() {
    # $1 = root name (no .tex). Two passes for cross-refs.
    local name="$1"
    local logfile="${name}.xelatex.log"
    if ! xelatex -interaction=nonstopmode -halt-on-error "${name}.tex" \
            > "$logfile" 2>&1; then
        echo "# xelatex failed on ${name}.tex — last errors:" >&2
        grep -E "^!|l\.[0-9]+" "$logfile" | head -20 >&2
        return 1
    fi
    xelatex -interaction=nonstopmode "${name}.tex" > "$logfile" 2>&1 || true
}

extract_pages() {
    local name="$1"
    local logfile="${name}.xelatex.log"
    grep -oE "Output written on ${name}\.pdf \([0-9]+ pages" "$logfile" \
        | grep -oE "[0-9]+ pages" || echo "?"
}

# --- Variant 1: non-anonymized + article wrapper --------------------
echo "# (1/3) Non-anonymized variant (article wrapper)"
python3 "$SCRIPT" \
    --output "$KIT/f2_methodology_body.tex"
( cd "$KIT" && run_xelatex test_compile )
# Loop 86 fix — bibtex was never running for non-tmlr variants, so
# 25 citations were stale at `(?)`. Run bibtex + 2 more xelatex passes.
( cd "$KIT" && bibtex test_compile > /dev/null 2>&1 || true )
( cd "$KIT" && run_xelatex test_compile )
( cd "$KIT" && run_xelatex test_compile )
pages1=$(cd "$KIT" && extract_pages test_compile)
bytes1=$(wc -c < "$KIT/test_compile.pdf" | tr -d ' ')
echo "  PDF: $pages1, $bytes1 bytes"

# --- Variant 2: anonymized + article wrapper ------------------------
echo "# (2/3) Anonymized variant (article wrapper)"
python3 papers/scripts/anonymize_paper.py > /dev/null
python3 "$SCRIPT" \
    --input papers/f2_methodology_anonymized.md \
    --output "$KIT/f2_methodology_anonymized_body.tex"
( cd "$KIT" && run_xelatex test_compile_anon )
( cd "$KIT" && bibtex test_compile_anon > /dev/null 2>&1 || true )
( cd "$KIT" && run_xelatex test_compile_anon )
( cd "$KIT" && run_xelatex test_compile_anon )
pages2=$(cd "$KIT" && extract_pages test_compile_anon)
bytes2=$(wc -c < "$KIT/test_compile_anon.pdf" | tr -d ' ')
echo "  PDF: $pages2, $bytes2 bytes"

# --- Variant 3: anonymized + REAL TMLR class ------------------------
echo "# (3/3) Anonymized variant (real TMLR class — actual submission target)"
( cd "$KIT" && run_xelatex test_compile_tmlr )
# Real TMLR class wants natbib + bibtex.
( cd "$KIT" && bibtex test_compile_tmlr > /dev/null 2>&1 || true )
( cd "$KIT" && run_xelatex test_compile_tmlr )
pages3=$(cd "$KIT" && extract_pages test_compile_tmlr)
bytes3=$(wc -c < "$KIT/test_compile_tmlr.pdf" | tr -d ' ')
echo "  PDF: $pages3, $bytes3 bytes"

echo "# All three variants compiled."

# --- PDF rendering sanity check (Loop 86) ---------------------------
# Loop 85 caught two SEV-5 bugs that were invisible in raw LaTeX:
# - HTML comment "<!-- ANONYMIZED VARIANT -->" rendering as prose
# - Math symbols Γ/Λ/Δ rendering as literal "\{}Gamma" text
# Stage 4 below greps the pdftotext output for these and other
# telltale rendering-bug strings. Any match aborts the compile step.
echo "# (4/4) pdftotext sanity check across all 3 PDF variants"
if ! command -v pdftotext >/dev/null 2>&1; then
    echo "# WARN: pdftotext not installed; skipping the sanity grep" >&2
else
    ANY_FAIL=0
    for pdf in test_compile test_compile_anon test_compile_tmlr; do
        pdfpath="$KIT/$pdf.pdf"
        if [[ ! -f "$pdfpath" ]]; then
            continue
        fi
        text_file="/tmp/${pdf}_text.$$.txt"
        pdftotext "$pdfpath" "$text_file" 2>/dev/null
        FAIL=0
        # Patterns that should NEVER appear in a clean PDF body.
        # Loop 86 — patterns for `grep -F` (fixed-string match) to avoid
        # regex misinterpretation of `*`, `\`, `(`, etc.
        for pattern in \
            '<!-- ' \
            'ANONYMIZED VARIANT' \
            '\{}Gamma' \
            '\{}Lambda' \
            '\{}Delta' \
            '\{}widehat' \
            '\{}text{' \
            'textbackslash' \
            '**' \
            '(?)' ; do
            if grep -q -F -- "$pattern" "$text_file" 2>/dev/null; then
                echo "  FAIL  $pdf.pdf contains rendering bug: $pattern" >&2
                FAIL=1
                ANY_FAIL=1
            fi
        done
        rm -f "$text_file"
        if [[ $FAIL -eq 0 ]]; then
            echo "  PASS  $pdf.pdf rendered cleanly"
        fi
    done
    if [[ $ANY_FAIL -eq 1 ]]; then
        echo "# ABORT: PDF rendering bugs detected. Fix the converter " \
            "or the source markdown and re-run compile_tmlr_test.sh." >&2
        exit 1
    fi
fi
echo "# All three variants verified."
