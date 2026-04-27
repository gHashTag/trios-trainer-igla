#!/usr/bin/env bash
# trios-trainer Railway entrypoint — expands TRIOS_SEED at runtime.
#
# Each Railway service sets its own TRIOS_SEED variable. This script reads it
# and passes it to trios-train so the same image runs every seed without
# rebuilds.
#
# R5-honest triplet emit: BPB=<v> @ step=<N> seed=$TRIOS_SEED sha=<7c> ...
set -euo pipefail

SEED="${TRIOS_SEED:-46}"
CONFIG="${TRIOS_CONFIG:-/configs/gate2-attempt.toml}"
STEPS="${TRIOS_STEPS:-81000}"
LR="${TRIOS_LR:-0.003}"
TARGET_BPB="${TRIOS_TARGET_BPB:-1.50}"

echo "[entrypoint] trios-trainer-igla starting"
echo "[entrypoint]   TRIOS_SEED       = $SEED"
echo "[entrypoint]   TRIOS_CONFIG     = $CONFIG"
echo "[entrypoint]   TRIOS_STEPS      = $STEPS"
echo "[entrypoint]   TRIOS_LR         = $LR"
echo "[entrypoint]   TRIOS_TARGET_BPB = $TARGET_BPB"

exec /usr/local/bin/trios-train \
    --config "$CONFIG" \
    --seed "$SEED" \
    "$@"
