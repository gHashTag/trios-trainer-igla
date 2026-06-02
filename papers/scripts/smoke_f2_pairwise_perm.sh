#!/usr/bin/env bash
# smoke_f2_pairwise_perm.sh — end-to-end smoke test for the Loop 110 binary.
#
# Builds f2_pairwise_perm if needed, runs it against a tiny synthetic input,
# and asserts the output schema + p-values match expected exact values.
#
# Catches regressions in the binary's I/O parsing, permutation enumeration,
# BH correction, or CSV output ordering. The 4 unit tests cover the
# primitive math; this test covers the end-to-end CLI path.
#
# Exit: 0 if PASS, 1 on any assertion failure or build error.

set -euo pipefail

CRATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CRATE_ROOT"

INPUT="$(mktemp -t f2_pairwise_smoke_input.XXXXXX.csv)"
OUTPUT="$(mktemp -t f2_pairwise_smoke_output.XXXXXX.csv)"
trap 'rm -f "$INPUT" "$OUTPUT"' EXIT

# Tiny synthetic input: 5 seeds × 2 configs × 1 stratum.
# Choose values so that phi - zoo = (-0.2, -0.2, -0.2, -0.2, -0.2)
# constant negative diff → exact paired-perm p_two = 2/32 = 0.0625.
cat > "$INPUT" <<EOF
# synthetic smoke input — see smoke_f2_pairwise_perm.sh
config,stratum,seed,val_bpb
GFTernary,canonical,42,1.20
GFTernary,canonical,43,1.18
GFTernary,canonical,44,1.22
GFTernary,canonical,45,1.19
GFTernary,canonical,46,1.21
bf16,canonical,42,1.40
bf16,canonical,43,1.38
bf16,canonical,44,1.42
bf16,canonical,45,1.39
bf16,canonical,46,1.41
EOF

# Ensure the binary is built.
cargo build --release --bin f2_pairwise_perm > /dev/null 2>&1

./target/release/f2_pairwise_perm \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --phi-configs GFTernary \
    --zoo-configs bf16 > /dev/null 2>&1

# Header sanity.
if ! grep -q "^stratum,phi_config,zoo_config,n,diff_mean,ci_lo,ci_hi,p_raw,p_bh$" "$OUTPUT"; then
    echo "FAIL: smoke output missing expected header" >&2
    cat "$OUTPUT" >&2
    exit 1
fi

# Single data row sanity.
DATA_LINE=$(grep -v "^#" "$OUTPUT" | grep -v "^stratum" | head -1)
if [[ -z "$DATA_LINE" ]]; then
    echo "FAIL: smoke output has no data rows" >&2
    cat "$OUTPUT" >&2
    exit 1
fi

# Verify p_raw value (should be 2/32 = 0.0625 to 4 decimal places).
# Format: stratum,phi,zoo,n,diff_mean,ci_lo,ci_hi,p_raw,p_bh
P_RAW=$(echo "$DATA_LINE" | awk -F, '{print $8}')
EXPECTED="0.062500"
if [[ "$P_RAW" != "$EXPECTED" ]]; then
    echo "FAIL: smoke p_raw mismatch — got $P_RAW, expected $EXPECTED" >&2
    echo "  data row: $DATA_LINE" >&2
    exit 1
fi

# Verify diff_mean = -0.20 (phi mean 1.20, zoo mean 1.40, diff -0.20).
DIFF_MEAN=$(echo "$DATA_LINE" | awk -F, '{print $5}')
if [[ "$DIFF_MEAN" != "-0.200000" ]]; then
    echo "FAIL: smoke diff_mean mismatch — got $DIFF_MEAN, expected -0.200000" >&2
    echo "  data row: $DATA_LINE" >&2
    exit 1
fi

# With only 1 phi × 1 zoo, BH-adjusted p = raw p.
P_BH=$(echo "$DATA_LINE" | awk -F, '{print $9}')
if [[ "$P_BH" != "$EXPECTED" ]]; then
    echo "FAIL: smoke p_bh mismatch (1-comparison case) — got $P_BH, expected $EXPECTED" >&2
    exit 1
fi

echo "# smoke_f2_pairwise_perm.sh — primitive PASS (diff=-0.2, p_raw=p_bh=$EXPECTED)"

# Loop 112 B: pipe output through f2_provenance_check.
# Build the verifier if needed (release profile to match the binary).
cargo build --release --bin f2_provenance_check > /dev/null 2>&1

# Run with a known git SHA so agent_git_sha doesn't WARN. We accept
# exit codes 0 (all PASS) and 1 (only WARN, e.g. unknown host). Reject
# exit 2 (FAIL on a required field) and exit 3 (malformed preamble).
F2_GIT_SHA="$(git rev-parse --short=7 HEAD 2>/dev/null || echo unknown)" \
HOST="$(hostname 2>/dev/null || echo unknown)" \
./target/release/f2_pairwise_perm \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --phi-configs GFTernary \
    --zoo-configs bf16 > /dev/null 2>&1

set +e
./target/release/f2_provenance_check "$OUTPUT" > /tmp/prov_check_output.log 2>&1
PROV_EXIT=$?
set -e

if [[ $PROV_EXIT -ge 2 ]]; then
    echo "FAIL: f2_provenance_check exited $PROV_EXIT on smoke output" >&2
    cat /tmp/prov_check_output.log >&2
    exit 1
fi

echo "# smoke_f2_pairwise_perm.sh — provenance PASS (f2_provenance_check exit=$PROV_EXIT)"
