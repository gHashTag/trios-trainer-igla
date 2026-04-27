#!/usr/bin/env bash
# trios-trainer Railway entrypoint — runs tjepa_train (champion binary).
#
# Each Railway service sets TRIOS_SEED. This script passes it to tjepa_train
# with champion-default hyperparams (hidden=384, encoder-lr=0.003, ntp-lr=0.003).
set -euo pipefail

SEED="${TRIOS_SEED:-100}"
STEPS="${TRIOS_STEPS:-81000}"
ENCODER_LR="${TRIOS_ENCODER_LR:-0.003}"
NTP_LR="${TRIOS_NTP_LR:-0.003}"

echo "[entrypoint] trios-trainer-igla (tjepa_train champion)"
echo "[entrypoint]   seed        = $SEED"
echo "[entrypoint]   steps       = $STEPS"
echo "[entrypoint]   encoder-lr  = $ENCODER_LR"
echo "[entrypoint]   ntp-lr      = $NTP_LR"

exec /usr/local/bin/tjepa_train \
    --seed="$SEED" \
    --steps="$STEPS" \
    --encoder-lr="$ENCODER_LR" \
    --ntp-lr="$NTP_LR" \
    --no-jepa \
    --no-nca
