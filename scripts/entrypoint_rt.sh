#!/bin/bash
set -euo pipefail

SEED="${TRIOS_SEED:-43}"
STEPS="${TRIOS_STEPS:-50000}"
LR="${TRIOS_LR:-0.003}"
HIDDEN="${TRIOS_HIDDEN:-2048}"

echo "[entrypoint] trinity_3k_tinyshakespeare seed=$SEED steps=$STEPS lr=$LR hidden=$HIDDEN"

exec /usr/local/bin/trinity_3k_tinyshakespeare \
    --seed="$SEED" \
    --steps="$STEPS" \
    --lr="$LR" \
    --hidden="$HIDDEN"
