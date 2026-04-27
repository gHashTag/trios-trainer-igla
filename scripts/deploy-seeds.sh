#!/usr/bin/env bash
#
# Quick Railway Seed Deployment for trios-trainer-igla
# Deploys 3 separate services for seeds 43, 44, 45
#

set -e

SEEDS=(43 44 45)
SERVICE_PREFIX="igla-trainer-seed"

echo "🚀 Railway Seed Deployment"
echo "========================"
echo "Seeds: ${SEEDS[*]}"
echo ""

for seed in "${SEEDS[@]}"; do
    service_name="$SERVICE_PREFIX-$seed"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Seed: $seed | Service: $service_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Create service (railway add with --service, but needs TTY for interactive prompts)
    # Using railway service create command instead
    echo "Creating service..."

    # Note: This will require interactive TTY
    # Run manually in terminal: railway add --service $service_name
    echo ""
    echo "⚠️  Run this command in TTY terminal:"
    echo "   railway add --service $service_name --variables \"TRIOS_SEED=$seed\""
    echo ""
    echo "Then deploy:"
    echo "   railway up --service $service_name"
    echo ""

done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All deployment commands generated!"
echo ""
echo "Or use one-liner in TTY:"
echo "for seed in 43 44 45; do railway add --service igla-trainer-seed-\$seed --variables \"TRIOS_SEED=\$seed\" && railway up --service igla-trainer-seed-\$seed; done"
