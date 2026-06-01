#!/usr/bin/env bash
# figure_regen.sh — regenerate all 4 paper figures from committed data
# in data/loop49/.
#
# Each figure script (papers/figures/figN_*.py) has its own --input
# default pointing at /tmp/ paths from earlier loops. This wrapper
# pipes the committed CSVs through f2_to_jsonl into temp JSONL files,
# then invokes each figure script with the correct --input pointer.
#
# Output: papers/figures/figN_*.png (4 files), overwritten in place.
# Exit: 0 on success, 1 on any figure failure.
#
# Required: python3 with matplotlib + numpy; cargo (release builds of
# f2_to_jsonl and f2_mediation_sensitivity).

set -euo pipefail

CRATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CRATE_ROOT"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "# (0/5) staging committed CSVs → JSONL ..."

# Required binaries.
cargo build --release --bin f2_to_jsonl --bin f2_mediation_sensitivity \
    > /dev/null 2>&1

# fig1 input: loop49_3stratum.csv (Loop 49 cross-stratum table)
target/release/f2_to_jsonl \
    data/loop49/loop49_3stratum.csv --out "$TMP/loop49_3stratum.jsonl"

# fig3 input: loop36_dual.csv (canonical-stratum dual mediation)
target/release/f2_to_jsonl \
    data/loop49/loop36_dual.csv --out "$TMP/loop36_dual.jsonl"

# fig4 input: requires a lambda-sweep CSV.
# Produce one from the canonical dual via f2_mediation_sensitivity.
target/release/f2_mediation_sensitivity \
    --lambda-sweep --tipping-point \
    data/loop49/loop36_dual.csv --out "$TMP/lambda_sweep.csv"
target/release/f2_to_jsonl \
    "$TMP/lambda_sweep.csv" --out "$TMP/lambda_sweep.jsonl"

# fig5 input: 3stratum_swap.csv (Loop 64 Phase 0 swap-parameterization).
target/release/f2_to_jsonl \
    data/loop49_swap/3stratum_swap.csv --out "$TMP/3strat_swap.jsonl"

echo "# (1/6) Figure 1 — RmsNorm NDE sign flip across strata"
python3 papers/figures/fig1_rms_nde_signflip.py \
    --input "$TMP/loop49_3stratum.jsonl" \
    --out papers/figures/fig1_rms_nde_signflip.png

echo "# (2/6) Figure 2 — Stratum × ModeKind architecture diagram"
python3 papers/figures/fig2_stratum_registry.py \
    --out papers/figures/fig2_stratum_registry.png

echo "# (3/6) Figure 3 — canonical 5x4 PSE heatmap"
python3 papers/figures/fig3_canonical_pse_heatmap.py \
    --input "$TMP/loop36_dual.jsonl"

echo "# (4/6) Figure 4 — Γ_tip(Λ) hyperbolae"
python3 papers/figures/fig4_tipping_curves.py \
    --input "$TMP/lambda_sweep.jsonl" \
    --out papers/figures/fig4_tipping_curves.png

echo "# (5/6) Figure 5 — Phase 0 swap NIE_M1 heatmap (Loop 64)"
python3 papers/figures/fig5_swap_nie_m1_heatmap.py \
    --input "$TMP/3strat_swap.jsonl" \
    --out papers/figures/fig5_swap_nie_m1_heatmap.png

echo "# (6/6) verify all 5 PNGs landed"
for fig in fig1_rms_nde_signflip fig2_stratum_registry \
           fig3_canonical_pse_heatmap fig4_tipping_curves \
           fig5_swap_nie_m1_heatmap; do
    if [[ ! -f "papers/figures/${fig}.png" ]]; then
        echo "# ERROR: papers/figures/${fig}.png missing" >&2
        exit 1
    fi
    bytes=$(wc -c < "papers/figures/${fig}.png" | tr -d ' ')
    echo "  $fig: $bytes bytes"
done

echo "# Done. All 5 figures regenerated from data/loop49/ + data/loop49_swap/ + binaries."
