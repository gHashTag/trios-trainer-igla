#!/bin/bash
# auto_launch.sh — continuously launch new runs as old ones complete
# Usage: nohup ./scripts/auto_launch.sh > .trinity/auto_launch.log 2>&1 &

cd "$(dirname "$0")/.."
mkdir -p .trinity/results

# Counter for unique run IDs
COUNTER_FILE=".trinity/auto_launch_counter"
COUNTER=$(cat "$COUNTER_FILE" 2>/dev/null || echo 0)

# Launch configurations — cycled through as slots free up
FORMATS=("gf16" "f32" "posit8" "gf8" "mxfp8" "nf4" "fp16" "bf16")
SEEDS=(43 44 45 46 47 48 49 50)
STEPS=(81000 50000 20000)

# Round-robin index
INDEX_FILE=".trinity/auto_launch_index"
INDEX=$(cat "$INDEX_FILE" 2>/dev/null || echo 0)

# Check free CPU capacity
check_free_slots() {
    local running=$(ps aux | grep "trios-train" | grep -v grep | wc -l)
    local max_slots=80
    echo $((max_slots - running))
}

# Launch a single run
launch_run() {
    local wave="v6"
    local counter=$1
    local idx=$2
    
    local fmt=${FORMATS[$((idx % ${#FORMATS[@]}))]}
    local seed=${SEEDS[$((idx % ${#SEEDS[@]}))]}
    local step=${STEPS[$((idx % ${#STEPS[@]}))]}
    
    local eval_every=$((step / 10))
    [ $eval_every -lt 500 ] && eval_every=500
    [ $eval_every -gt 10000 ] && eval_every=10000
    
    local name="${wave}_${fmt}_seed${seed}_${step}"
    local log=".trinity/results/${name}.log"
    
    echo "[$(date)] Launching $name (counter=$counter)"
    nohup nice -n 19 ./target/release/trios-train \
        --seed=$seed \
        --steps=$step \
        --hidden=384 \
        --lr=0.003 \
        --optimizer=adamw \
        --eval-every=$eval_every \
        --format=$fmt \
        > "$log" 2>&1 &
echo ""
}

# Main loop
while true; do
    FREE=$(check_free_slots)
    if [ "$FREE" -gt 0 ]; then
        echo "[$(date)] Free slots: $FREE, launching..."
        for i in $(seq 1 $FREE); do
            COUNTER=$((COUNTER + 1))
            launch_run $COUNTER $INDEX
            INDEX=$(((INDEX + 1) % 1000))
            sleep 2
        done
        echo $COUNTER > "$COUNTER_FILE"
        echo $INDEX > "$INDEX_FILE"
    else
        echo "[$(date)] No free slots (running: $(ps aux | grep trios-train | grep -v grep | wc -l)), waiting..."
    fi
    sleep 60
done
