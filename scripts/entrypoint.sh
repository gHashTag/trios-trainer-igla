#!/bin/bash
set -euo pipefail

SEED="${TRIOS_SEED:-43}"
STEPS="${TRIOS_STEPS:-27000}"
CONFIG="${TRIOS_CONFIG:-configs/gate2-final.toml}"

echo "[entrypoint] trios-train seed=$SEED steps=$STEPS config=$CONFIG"

# If config is specified, use config mode
if [ -n "$CONFIG" ]; then
    exec /usr/local/bin/trios-train \
        --config="$CONFIG"
else
    exec /usr/local/bin/trios-train \
        --seed="$SEED" \
        --steps="$STEPS"
fi
