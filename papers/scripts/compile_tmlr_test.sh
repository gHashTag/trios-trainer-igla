#!/usr/bin/env bash
# compile_tmlr_test.sh — regenerate body.tex from Markdown, then xelatex
# compile the test wrapper to verify no structural breakage.
#
# Requires xelatex (TeX Live ships it). On macOS:
#   brew install --cask mactex
#
# Output: papers/tmlr_submission_kit/test_compile.pdf
# Exit: 0 on successful PDF generation, 1 otherwise.

set -euo pipefail

CRATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CRATE_ROOT"

KIT="papers/tmlr_submission_kit"
SCRIPT="papers/scripts/md_to_tmlr_tex.py"

echo "# (1/3) Regenerating $KIT/f2_methodology_body.tex …"
python3 "$SCRIPT"

echo "# (2/3) Running xelatex …"
cd "$KIT"
# Single pass — \cref references show as "??" without a 2nd pass, but
# any structural breakage will surface on the first pass.
if ! xelatex -interaction=nonstopmode -halt-on-error test_compile.tex \
        > test_compile.xelatex.log 2>&1; then
    echo "# xelatex failed — see test_compile.xelatex.log:" >&2
    grep -E "^!|l\.[0-9]+" test_compile.xelatex.log | head -20 >&2
    exit 1
fi

# Second pass for cross-references.
xelatex -interaction=nonstopmode test_compile.tex \
        > test_compile.xelatex.log 2>&1 || true

if [[ -f test_compile.pdf ]]; then
    pages=$(grep -oE "Output written on test_compile\.pdf \([0-9]+ pages" \
            test_compile.xelatex.log | grep -oE "[0-9]+ pages" || echo "?")
    bytes=$(wc -c < test_compile.pdf | tr -d ' ')
    echo "# (3/3) PDF: $pages, $bytes bytes"
    exit 0
else
    echo "# xelatex did not produce PDF; check test_compile.xelatex.log" >&2
    exit 1
fi
