#!/bin/bash
set -euo pipefail

SEED="${TRIOS_SEED:-43}"
STEPS="${TRIOS_STEPS:-27000}"
LR="${TRIOS_LR:-0.003}"
HIDDEN="${TRIOS_HIDDEN:-384}"
OPT="${TRIOS_OPTIMIZER:-adamw}"

echo "[entrypoint] trios-train seed=$SEED steps=$STEPS lr=$LR hidden=$HIDDEN opt=$OPT"

exec /usr/local/bin/trios-train \
    --seed="$SEED" \
    --steps="$STEPS" \
    --lr="$LR" \
    --hidden="$HIDDEN" \
    --optimizer="$OPT" \
    --train-data=data/tiny_shakespeare.txt \
    --val-data=data/tiny_shakespeare_val.txt
