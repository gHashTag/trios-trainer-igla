#!/bin/bash
set -euo pipefail

SEED="${TRIOS_SEED:-43}"
STEPS="${TRIOS_STEPS:-81000}"
LR="${TRIOS_LR:-0.003}"
HIDDEN="${TRIOS_HIDDEN:-384}"
OPT="${TRIOS_OPTIMIZER:-adamw}"

TRAIN_DATA="${TRIOS_TRAIN_DATA:-/work/data/tiny_shakespeare.txt}"
VAL_DATA="${TRIOS_VAL_DATA:-/work/data/tiny_shakespeare_val.txt}"

echo "[entrypoint] trios-train seed=$SEED steps=$STEPS lr=$LR hidden=$HIDDEN opt=$OPT"
echo "[entrypoint] train=$TRAIN_DATA val=$VAL_DATA"

exec /usr/local/bin/trios-train \
    --seed="$SEED" \
    --steps="$STEPS" \
    --lr="$LR" \
    --hidden="$HIDDEN" \
    --optimizer="$OPT" \
    --train-data="$TRAIN_DATA" \
    --val-data="$VAL_DATA"
