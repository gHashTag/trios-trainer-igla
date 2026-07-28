#!/usr/bin/env bash
# scripts/wave29_smoke_test.sh — Wave 29 PR-A smoke test
#
# Falsification criteria (R7):
#   1. cargo test --release seed_canon → all tests GREEN
#   2. SEED=43 cargo run --release --bin trios-train → must exit non-zero
#
# Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877
# Author: Dmitrii Vasilev <admin@t27.ai>

set -euo pipefail

echo "=== Wave 29 smoke test starting ==="

# 1. Run seed_canon tests
echo "[1/2] Running seed_canon tests..."
cargo test --release seed_canon -- --nocapture
echo "[1/2] seed_canon tests: PASS"

# 2. Verify that SEED=43 causes non-zero exit
echo "[2/2] Verifying SEED=43 is rejected by trios-train..."
if SEED=43 cargo run --release --bin trios-train -- --steps 1 2>/dev/null; then
    echo "FAIL: SEED=43 was accepted (expected non-zero exit)"
    exit 1
fi
echo "[2/2] SEED=43 correctly rejected: PASS"

echo "=== OK: Wave 29 smoke test passed ==="
