#!/usr/bin/env bash
#
# Railway Multi-Seed Deployment Script for trios-trainer-igla
# Creates 3 separate Railway services for seeds 43, 44, 45
#

set -e

# Railway Project ID (obtain from: railway project list)
RAILWAY_PROJECT_ID="${RAILWAY_PROJECT_ID:-}"

# Seeds to deploy
SEEDS=(43 44 45)
SERVICE_PREFIX="igla-trainer-seed"

echo "🚀 Railway Multi-Seed Deployment"
echo "==================================="
echo "Project: gHashTag/trios-trainer-igla"
echo "Seeds: ${SEEDS[*]}"
echo ""

# Check if railway token is set
if [[ -z "$RAILWAY_TOKEN" ]]; then
    echo "❌ RAILWAY_TOKEN environment variable not set!"
    echo "   Get token: https://build.railway.app/settings/tokens"
    exit 1
fi

# GraphQL API endpoint
RAILWAY_API="https://backboard.railway.app/graphql/v2"

# Function to create a service via Railway API
create_service() {
    local seed=$1
    local service_name="$SERVICE_PREFIX-$seed"

    echo "📦 Creating service: $service_name (seed: $seed)"

    # GraphQL mutation to create service
    local query=$(cat <<EOF
mutation {
  serviceCreate(input: {projectId: "$RAILWAY_PROJECT_ID", name: "$service_name"}) {
    id
    name
    projectId
  }
}
EOF
)

    local response=$(curl -s -X POST \
        -H "Authorization: Bearer $RAILWAY_TOKEN" \
        -H "Content-Type: application/json" \
        -d "$query" \
        "$RAILWAY_API" 2>&1)

    echo "$response" | jq -r '.data.serviceCreate // .errors'
}

# Function to set environment variable for a service
set_env_var() {
    local service_name=$1
    local seed=$2
    local var_name="TRIOS_SEED"
    local var_value="$seed"

    echo "🔧 Setting $var_name=$var_value on $service_name"

    # First, get service ID
    local get_query='{"query": "{ project(id: \"'$RAILWAY_PROJECT_ID'\") { services { id name } }"}'
    local services=$(curl -s -X POST \
        -H "Authorization: Bearer $RAILWAY_TOKEN" \
        -H "Content-Type: application/json" \
        -d "$get_query" \
        "$RAILWAY_API")

    local service_id=$(echo "$services" | jq -r --arg name "$service_name" '.data.project.services[] | select(.name == $name) | .id')

    if [[ -z "$service_id" ]]; then
        echo "❌ Could not find service ID for $service_name"
        return 1
    fi

    # Then set environment variable
    local var_query=$(cat <<EOF
mutation {
  serviceVariableUpsert(input: {serviceId: "$service_id", name: "$var_name", value: "$var_value"}) {
    name
    value
  }
}
EOF
)

    local var_response=$(curl -s -X POST \
        -H "Authorization: Bearer $RAILWAY_TOKEN" \
        -H "Content-Type: application/json" \
        -d "$var_query" \
        "$RAILWAY_API" 2>&1)

    echo "$var_response" | jq -r '.data.serviceVariableUpsert // .errors'
}

# Function to trigger deployment
trigger_deploy() {
    local service_name=$1

    echo "🚀 Triggering deployment: $service_name"

    # Get service ID first
    local get_query='{"query": "{ project(id: \"'$RAILWAY_PROJECT_ID'\") { services { id name } }"}'
    local services=$(curl -s -X POST \
        -H "Authorization: Bearer $RAILWAY_TOKEN" \
        -H "Content-Type: application/json" \
        -d "$get_query" \
        "$RAILWAY_API")

    local service_id=$(echo "$services" | jq -r --arg name "$service_name" '.data.project.services[] | select(.name == $name) | .id')

    if [[ -z "$service_id" ]]; then
        echo "❌ Could not find service ID for $service_name"
        return 1
    fi

    # Trigger deployment
    local deploy_query=$(cat <<EOF
mutation {
  deployBranch(input: {serviceId: "$service_id", branch: "main"}) {
    id
    domain
    status
  }
}
EOF
)

    local deploy_response=$(curl -s -X POST \
        -H "Authorization: Bearer $RAILWAY_TOKEN" \
        -H "Content-Type: application/json" \
        -d "$deploy_query" \
        "$RAILWAY_API" 2>&1)

    echo "$deploy_response" | jq -r '.data.deployBranch // .errors'
}

# Main deployment loop
for seed in "${SEEDS[@]}"; do
    service_name="$SERVICE_PREFIX-$seed"

    # 1. Create service (if doesn't exist)
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Deploying seed: $seed"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    create_service "$seed"
    set_env_var "$service_name" "$seed"
    trigger_deploy "$service_name"

    echo "✅ Seed $seed deployment initiated"
done

echo ""
echo "==================================="
echo "📋 All seeds deployed!"
echo ""
echo "Monitor at: https://railway.app/project/$RAILWAY_PROJECT_ID"
