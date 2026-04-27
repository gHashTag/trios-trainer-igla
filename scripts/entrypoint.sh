#!/bin/bash
set -euo pipefail

SEED="${TRIOS_SEED:-43}"
STEPS="${TRIOS_STEPS:-81000}"
LR="${TRIOS_LR:-0.003}"
HIDDEN="${TRIOS_HIDDEN:-828}"
EVAL_EVERY="${TRIOS_EVAL_EVERY:-1000}"

echo "[entrypoint] hybrid_train seed=$SEED steps=$STEPS lr=$LR hidden=$HIDDEN"

exec /usr/local/bin/hybrid_train \
    --seed="$SEED" \
    --steps="$STEPS" \
    --lr="$LR" \
    --hidden="$HIDDEN" \
    --eval-every="$EVAL_EVERY"
