#!/bin/bash
# auto_launch.sh — continuously launch new runs as old ones complete
# Usage: nohup ./scripts/auto_launch.sh > .trinity/auto_launch.log 2>&1 &

cd "$(dirname "$0")/.."
mkdir -p .trinity/results

# Counter for unique run IDs
COUNTER_FILE=".trinity/auto_launch_counter"
COUNTER=$(cat "$COUNTER_FILE" 2>/dev/null || echo 0)

# Launch configurations — cycled through as slots free up
# All 8 GF family formats + key baselines
FORMATS=("gf4" "gf8" "gf12" "gf16" "gf20" "gf24" "gf32" "gf64" "f32" "f64" "posit8" "posit16" "posit32" "fp16" "bf16" "tf32" "fp8_e4m3" "fp8_e5m2" "fp6_e2m3" "fp6_e3m2" "fp4_e2m1" "int8" "int16" "int4" "mxfp8" "mxfp6" "mxfp4" "nf4" "lns8" "uint8")
SEEDS=(43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180)
# Completable step counts — fast (5K) to ultra-deep (10M) for maximum data density
STEPS=(2000 5000 10000 20000 50000 81000 100000 200000 300000 500000 750000 1000000 2000000 3000000 5000000 7500000 10000000 15000000)
HIDDENS=(256 384 512 768 1024)
LRS=(0.01 0.003 0.001 0.0003 0.0001)
OPTIMIZERS=("adamw" "muon" "muon-cwd")

# Round-robin index
INDEX_FILE=".trinity/auto_launch_index"
INDEX=$(cat "$INDEX_FILE" 2>/dev/null || echo 0)

# Check free CPU capacity
check_free_slots() {
    local running=$(ps aux | grep "trios-train" | grep -v grep | wc -l)
    local max_slots=110
    echo $((max_slots - running))
}

# Launch a single run
launch_run() {
    local wave="v6"
    local counter=$1
    local idx=$2

    local n_fmt=${#FORMATS[@]}
    local n_seed=${#SEEDS[@]}
    local n_step=${#STEPS[@]}
    local n_hidden=${#HIDDENS[@]}
    local n_lr=${#LRS[@]}
    local n_opt=${#OPTIMIZERS[@]}
    local total=$((n_fmt * n_seed * n_step * n_hidden * n_lr * n_opt))

    # Combinatorial index decomposition
    local opt_idx=$((idx % n_opt))
    local remaining=$((idx / n_opt))
    local lr_idx=$((remaining % n_lr))
    remaining=$((remaining / n_lr))
    local hidden_idx=$((remaining % n_hidden))
    remaining=$((remaining / n_hidden))
    local step_idx=$((remaining % n_step))
    remaining=$((remaining / n_step))
    local seed_idx=$((remaining % n_seed))
    remaining=$((remaining / n_seed))
    local fmt_idx=$((remaining % n_fmt))

    local fmt=${FORMATS[$fmt_idx]}
    local seed=${SEEDS[$seed_idx]}
    local step=${STEPS[$step_idx]}
    local hidden=${HIDDENS[$hidden_idx]}
    local lr=${LRS[$lr_idx]}
    local opt=${OPTIMIZERS[$opt_idx]}

    # Ultra-high-density evals: maximum data points per run
    local eval_every=$((step / 200000))
    [ $eval_every -lt 200 ] && eval_every=200
    [ $eval_every -gt 10000 ] && eval_every=10000

    local name="${wave}_${fmt}_h${hidden}_lr${lr}_seed${seed}_${step}_${opt}"
    local log=".trinity/results/${name}.log"

    echo "[$(date)] Launching $name (counter=$counter, idx=$idx)"
    nohup nice -n 19 ./target/release/trios-train \
        --seed=$seed \
        --steps=$step \
        --hidden=$hidden \
        --lr=$lr \
        --optimizer=$opt \
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
            INDEX=$(((INDEX + 1) % (30 * 138 * 18 * 5 * 5 * 3)))
            sleep 0.2
        done
        echo $COUNTER > "$COUNTER_FILE"
        echo $INDEX > "$INDEX_FILE"
    else
        echo "[$(date)] No free slots (running: $(ps aux | grep trios-train | grep -v grep | wc -l)), waiting..."
    fi
    sleep 5
done
