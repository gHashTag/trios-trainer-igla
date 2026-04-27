#!/bin/bash
set -e

echo "=== IGLA Trainer Container seed=${TRIOS_SEED} steps=${TRIOS_STEPS} ==="

exec trios-train \
  --seed "${TRIOS_SEED:-43}" \
  --steps "${TRIOS_STEPS:-27000}" \
  --hidden "${TRIOS_HIDDEN:-384}" \
  --lr "${TRIOS_LR:-0.004}" \
  --attn-layers "${TRIOS_ATTN_LAYERS:-2}" \
  --eval-every "${TRIOS_EVAL_EVERY:-1000}" \
  --optimizer "${TRIOS_OPTIMIZER:-adamw}" \
  --train-data "${TRIOS_TRAIN_PATH:-data/tinyshakespeare.txt}" \
  --val-data "${TRIOS_VAL_PATH:-data/tinyshakespeare.txt}"
