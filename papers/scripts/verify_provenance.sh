#!/usr/bin/env bash
# verify_provenance.sh — gate W3C-PROV preambles on every CSV emitted by
# a producer that claims to emit one (per §5.1 in the #1021 paper).
#
# Loop 115 A: honors the §5.4 pre-registration ("(*to be implemented*)").
# Walks the two preamble-emitting producer classes:
#   - cell-level CSVs from f2_ablation_sweep
#   - aggregator CSVs from f2_pairwise_perm
# Other aggregators (f2_ablation_aggregate / f2_stratum_compare /
# f2_mediation_sensitivity) emit no preamble by design — §5.1 acknowledges
# this transparently and they are NOT gated here.
#
# Usage:
#   verify_provenance.sh <root>
#     where <root> is the run-result paper's data subtree (e.g.
#     data/issue1021/run0/). Each *.csv under <root> is classified by
#     filename prefix and gated as appropriate.
#
# Pass-through to f2_provenance_check:
#   - exit 0 (all PASS) and exit 1 (WARN, e.g. older git SHA): accepted
#   - exit 2 (FAIL on required field): rejected
#   - exit 3 (malformed preamble): rejected
#
# Exit: 0 if every gated CSV passes; 1 on any FAIL or malformed CSV.

set -euo pipefail

ROOT="${1:-data/issue1021/run0}"
CRATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CRATE_ROOT"

if [[ ! -d "$ROOT" ]]; then
    echo "# verify_provenance.sh: root $ROOT does not exist; nothing to gate." >&2
    echo "# (The 80-cell sweep produces it; pre-sweep runs of this script" >&2
    echo "# are vacuously OK.)"
    exit 0
fi

# Build the verifier (idempotent at release profile).
cargo build --release --bin f2_provenance_check > /dev/null 2>&1

n_gated=0
n_fail=0
for csv in "$ROOT"/cell_*.csv "$ROOT"/pairwise_*.csv; do
    [[ -f "$csv" ]] || continue
    n_gated=$((n_gated + 1))
    set +e
    ./target/release/f2_provenance_check "$csv" \
        > /tmp/prov_check.$$.log 2>&1
    rc=$?
    set -e
    if [[ $rc -ge 2 ]]; then
        n_fail=$((n_fail + 1))
        echo "FAIL  $csv (f2_provenance_check exit=$rc)" >&2
        tail -5 /tmp/prov_check.$$.log >&2
    fi
done
rm -f /tmp/prov_check.$$.log

if [[ $n_gated -eq 0 ]]; then
    echo "# verify_provenance.sh: no gateable CSVs found under $ROOT" >&2
    echo "# (expected cell_*.csv and/or pairwise_*.csv prefixes)" >&2
    echo "# vacuously OK (no sweep output yet)"
    exit 0
fi

if [[ $n_fail -gt 0 ]]; then
    echo "# verify_provenance.sh: $n_fail FAIL / $n_gated gated" >&2
    exit 1
fi
echo "# verify_provenance.sh: $n_gated CSVs gated, 0 FAIL"
