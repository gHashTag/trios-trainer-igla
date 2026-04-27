#!/usr/bin/env bash
#
# P1 Optimizer Lab Railway Deployment
# Deploys 3 services: AdamW (control), Muon, Muon+CWD
# Each runs 12K steps on seed 43
#

set -euo pipefail

echo "========================================"
echo "P1 Optimizer Lab - Railway Deployment"
echo "========================================"
echo ""
echo "Services to deploy:"
echo "  1. p1-adamw (control)"
echo "  2. p1-muon (η2D=0.008, η1D=0.007)"
echo "  3. p1-muon-cwd (μ + CWD)"
echo ""

# Check if railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ railway CLI not found!"
    echo "   Install: npm install -g @railway/cli"
    exit 1
fi

# Check login status
if ! railway whoami &> /dev/null; then
    echo "❌ Not logged in to Railway!"
    echo "   Run: railway login"
    exit 1
fi

PROJECT_ID="e4fe33bb-3b09-4842-9782-7d2dea1abc9b"
BASE_CONFIG="configs/lab"

# Function to deploy a single optimizer configuration
deploy_optimizer() {
    local name=$1
    local config=$2
    local service_name="p1-${name}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Deploying: $service_name"
    echo "Config: $config"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Set TRIOS_CONFIG to use the TOML config
    export TRIOS_CONFIG="$config"

    # Create or update service
    railway add --service "$service_name" --variables "TRIOS_CONFIG=$config" || echo "Service may already exist"

    # Trigger deployment
    railway up --service "$service_name"

    echo "✅ $service_name deployment started"
    echo ""
}

# Deploy all three optimizer configurations
deploy_optimizer "adamw" "$BASE_CONFIG/p1-adamw.toml"
deploy_optimizer "muon" "$BASE_CONFIG/p1-muon.toml"
deploy_optimizer "muon-cwd" "$BASE_CONFIG/p1-muon-cwd.toml"

echo "========================================"
echo "✅ All P1 experiments deployed!"
echo ""
echo "Monitor at: https://railway.com/project/$PROJECT_ID"
echo ""
echo "Expected results (~30 min each):"
echo "  - assertions/lab/p1_leaderboard.jsonl"
echo ""
echo "Winner determined by lowest val_bpb @ 12K steps"
echo "Margin required: ≥0.05 BPB over AdamW"
