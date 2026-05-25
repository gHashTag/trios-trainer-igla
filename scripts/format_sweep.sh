#!/bin/bash
# format_sweep.sh — QAT sweep across numeric formats for IGLA RACE
# Usage: ./scripts/format_sweep.sh
# Results: .trinity/results/format_sweep_<fmt>_seed<seed>.json

cd "$(dirname "$0")/.."
mkdir -p .trinity/results

SEED=43
STEPS=5000
HIDDEN=384
LR=0.003
OPT=adamw
EVAL_EVERY=500

FORMATS=(
    f32
    gf16
    fp16
    bf16
    gf8
    fp8_e4m3
    fp8_e5m2
    int8
    int4
    nf4
)

echo "=== FORMAT SWEEP ==="
echo "seed=$SEED steps=$STEPS hidden=$HIDDEN lr=$LR opt=$OPT"
echo "formats: ${FORMATS[@]}"
echo ""

for fmt in "${FORMATS[@]}"; do
    echo "--- Format: $fmt ---"
    ./target/release/trios-train \
        --seed=$SEED \
        --steps=$STEPS \
        --hidden=$HIDDEN \
        --lr=$LR \
        --optimizer=$OPT \
        --eval-every=$EVAL_EVERY \
        --format=$fmt \
        2>&1 | tee ".trinity/results/format_sweep_${fmt}_seed${SEED}.log" | grep -E "Initial val_bpb|step=|DONE:|Training Complete"
    echo ""
done

echo "=== SWEEP COMPLETE ==="
for fmt in "${FORMATS[@]}"; do
    log=".trinity/results/format_sweep_${fmt}_seed${SEED}.log"
    best=$(grep "DONE:" "$log" | sed 's/.*bpb=//' | awk '{print $1}')
    echo "$fmt: $best"
done
