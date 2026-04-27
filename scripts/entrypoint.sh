#!/bin/bash
set -euo pipefail

SEED="${TRIOS_SEED:-100}"
STEPS="${TRIOS_STEPS:-81000}"
ENCODER_LR="${TRIOS_ENCODER_LR:-0.003}"
NTP_LR="${TRIOS_NTP_LR:-0.003}"

echo "[entrypoint] tjepa_train champion seed=$SEED steps=$STEPS enc_lr=$ENCODER_LR ntp_lr=$NTP_LR"

exec /usr/local/bin/tjepa_train \
    --seed="$SEED" \
    --steps="$STEPS" \
    --encoder-lr="$ENCODER_LR" \
    --ntp-lr="$NTP_LR" \
    --no-jepa \
    --no-nca
