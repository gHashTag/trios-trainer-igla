#!/bin/bash
set -euo pipefail

SEED="${TRIOS_SEED:-43}"
STEPS="${TRIOS_STEPS:-27000}"
LR="${TRIOS_LR:-0.003}"
HIDDEN="${TRIOS_HIDDEN:-384}"
OPT="${TRIOS_OPTIMIZER:-adamw}"
FALLBACK="${L_R8_SYNTHETIC_FALLBACK:-ALLOW}"

TRAIN_DATA="/work/data/fineweb_train.bin"
VAL_DATA="/work/data/fineweb_val.bin"

# R8 guard: hard-fail if corpus missing
if [ ! -s "$TRAIN_DATA" ]; then
    echo "[entrypoint] FATAL: train corpus missing: $TRAIN_DATA"
    echo "[entrypoint] R5/R8 VIOLATION: cannot produce honest BPB without corpus"
    exit 1
fi
if [ ! -s "$VAL_DATA" ]; then
    echo "[entrypoint] FATAL: val corpus missing: $VAL_DATA"
    exit 1
fi

# R8 guard: if FORBID mode, abort on any synthetic/stale data path
if [ "$FALLBACK" = "FORBID" ]; then
    if echo "$TRAIN_DATA" | grep -q "tiny_shakespeare"; then
        echo "[entrypoint] FATAL: L_R8_SYNTHETIC_FALLBACK=FORBID but tinyshakespeare path detected"
        exit 1
    fi
fi

echo "[entrypoint] trios-train seed=$SEED steps=$STEPS lr=$LR hidden=$HIDDEN opt=$OPT"
echo "[entrypoint] train=$TRAIN_DATA val=$VAL_DATA fallback=$FALLBACK"

exec /usr/local/bin/trios-train \
    --seed="$SEED" \
    --steps="$STEPS" \
    --lr="$LR" \
    --hidden="$HIDDEN" \
    --optimizer="$OPT" \
    --train-data="$TRAIN_DATA" \
    --val-data="$VAL_DATA"
