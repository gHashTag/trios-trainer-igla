#!/usr/bin/env bash
# pack_supplementary.sh — bundle TMLR supplementary materials.
#
# Reads from the repo (data/loop49/, papers/figures/, papers/appendix_d_*)
# and produces papers/tmlr_submission_kit/f2_methodology_supp.zip.
#
# Layout matches papers/tmlr_submission_kit/manifest.md.
#
# Usage:
#   papers/tmlr_submission_kit/pack_supplementary.sh
#
# No arguments. Idempotent — re-running rebuilds the zip from scratch.

set -euo pipefail

CRATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CRATE_ROOT"

KIT="papers/tmlr_submission_kit"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

OUT="$KIT/f2_methodology_supp.zip"

# Stage tree per manifest.md
mkdir -p \
    "$STAGE/f2_methodology_supp/reproducibility" \
    "$STAGE/f2_methodology_supp/data" \
    "$STAGE/f2_methodology_supp/figures" \
    "$STAGE/f2_methodology_supp/audit"

cat > "$STAGE/f2_methodology_supp/README.md" <<'README'
# F2 methodology — supplementary materials

This zip accompanies a TMLR submission of "F2: stratified Pearl-CDE
ablation methodology with Λ-sweep envelope". It contains:

- `reproducibility/` — paper §Appendix A commands, §Appendix D test
  inventory, MD5 manifest of empirical CSVs
- `data/` — six committed CSVs that back §5
- `figures/` — four headline figures + matplotlib template
- `audit/` — cross-reference + derivation audit reports

Every file is self-contained. No git checkout needed to reproduce
Figure 1 from the data; see `reproducibility/appendix_a_commands.md`.

Anchor commit: `5367bde` (descendant on `f2-methodology` branch).
README

# Reproducibility section
if [[ -f "papers/appendix_d_test_inventory.md" ]]; then
    cp "papers/appendix_d_test_inventory.md" \
       "$STAGE/f2_methodology_supp/reproducibility/appendix_d_test_inventory.md"
fi

# Extract Appendix A from the paper (lines from "### A. Reproducible" to next ###)
python3 - <<'PY'
import re
from pathlib import Path
src = Path("papers/f2_methodology.md").read_text()
m = re.search(r"### A\. Reproducible commands.*?(?=^### [BC]\.)",
              src, flags=re.MULTILINE | re.DOTALL)
if m:
    out = Path("/tmp/pack_supp_app_a.md")
    out.write_text("# Appendix A — Reproducible commands\n\n" + m.group(0).split("\n", 1)[1])
    print(f"# extracted Appendix A: {len(m.group(0))} chars")
PY
if [[ -f /tmp/pack_supp_app_a.md ]]; then
    mv /tmp/pack_supp_app_a.md \
       "$STAGE/f2_methodology_supp/reproducibility/appendix_a_commands.md"
fi

if [[ -f "data/loop49/README.md" ]]; then
    cp "data/loop49/README.md" \
       "$STAGE/f2_methodology_supp/reproducibility/data_loop49_checksums.md"
fi

# Data section
if [[ -d "data/loop49" ]]; then
    for csv in data/loop49/*.csv; do
        [[ -f "$csv" ]] && cp "$csv" "$STAGE/f2_methodology_supp/data/"
    done
fi

# Figures section
if [[ -d "papers/figures" ]]; then
    for f in papers/figures/*.png papers/figures/fig_template.py; do
        [[ -f "$f" ]] && cp "$f" "$STAGE/f2_methodology_supp/figures/"
    done
fi

# Audit section
if [[ -f "papers/cross_reference_report.md" ]]; then
    cp "papers/cross_reference_report.md" \
       "$STAGE/f2_methodology_supp/audit/cross_reference_report.md"
fi

# Derivation audit (Loop 59 — see /tmp; kit captures the report verbatim
# if it's present at pack-time, else skips). At anchor time the audit
# lives in the PR or in /tmp; the kit-included copy is what reviewers
# see.
if [[ -f "$KIT/derivation_audit_loop59.md" ]]; then
    cp "$KIT/derivation_audit_loop59.md" \
       "$STAGE/f2_methodology_supp/audit/derivation_audit_loop59.md"
fi

# Zip
cd "$STAGE"
zip -r "$CRATE_ROOT/$OUT" f2_methodology_supp > /dev/null
cd "$CRATE_ROOT"

echo "# Wrote $OUT ($(wc -c < "$OUT") bytes)"
echo "# Verify: unzip -l $OUT"
