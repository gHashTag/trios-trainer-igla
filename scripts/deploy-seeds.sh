#!/usr/bin/env bash
#
# Quick Railway Seed Deployment for trios-trainer-igla
# Deploys 3 separate services for the NEW seed fleet 100, 101, 102.
#
# History (R5-honest):
#   - Champion seed: 43 (BPB=2.2393 @ 27K)
#   - Old gate-2 fleet: 43, 44, 45 (no row < 1.85, see seed_results.jsonl)
#   - New gate-2 fleet (this deploy): 100, 101, 102
#
# Pass NEW_SEEDS env var to override the default sequence.

set -euo pipefail

# Default: NEW seeds (use a new range (fresh seeds outside 43-45)). Override with NEW_SEEDS.
SEEDS=(${NEW_SEEDS:-100 101 102})
SERVICE_PREFIX="igla-trainer-seed"

echo "Railway Seed Deployment"
echo "========================"
echo "Seeds: ${SEEDS[*]}"
echo "Project: gHashTag/trios-trainer-igla"
echo ""

for seed in "${SEEDS[@]}"; do
    service_name="$SERVICE_PREFIX-$seed"
    echo "------------------------------------------------"
    echo "Seed: $seed | Service: $service_name"
    echo "------------------------------------------------"
    echo "Run in TTY:"
    echo "  railway add --service $service_name --variables \"TRIOS_SEED=$seed\""
    echo "  railway up --service $service_name"
    echo ""
done

echo "================================================"
echo "Or one-liner in TTY:"
echo "for seed in ${SEEDS[*]}; do"
echo "  railway add --service igla-trainer-seed-\$seed --variables \"TRIOS_SEED=\$seed\""
echo "  railway up --service igla-trainer-seed-\$seed"
echo "done"
