#!/bin/bash
# E41-E45 experiments - Push beyond Gate-2 target (BPB < 2.03)
# Based on E36-E40 results, try:
# - Higher model capacity (d_model=128 vs 64)
# - Different JEPA/NCA weight combinations
# - Different learning rates
# - Different warmup periods

SEEDS=(42 43 44 45 46)
BASE_LR=0.004

# E41: Higher capacity (d_model=128, n_layers=3, n_heads=8)
./target/release/tjepa_train \
  --seed=42 \
  --steps=200000 \
  --encoder-lr=0.0035 \
  --ntp-lr=0.000875 \
  --ntp-weight=1.0 \
  --jepa-weight=1.0 \
  --nca-weight=0.30 \
  --optimizer=adamw \
  --jepa-warmup=2000 \
  --trial-id=E41-HigherCapacity-200K \
  --agent-id=ALFA &

# E42: Higher JEPA weight (w_jepa=0.15)
./target/release/tjepa_train \
  --seed=43 \
  --steps=200000 \
  --encoder-lr=0.004 \
  --ntp-lr=0.001 \
  --ntp-weight=1.0 \
  --jepa-weight=1.15 \
  --nca-weight=0.25 \
  --optimizer=adamw \
  --jepa-warmup=2000 \
  --trial-id=E42-HigherJEPA-200K \
  --agent-id=BRAVO &

# E43: Higher NCA weight (w_nca=0.35)
./target/release/tjepa_train \
  --seed=44 \
  --steps=200000 \
  --encoder-lr=0.004 \
  --ntp-lr=0.001 \
  --ntp-weight=1.0 \
  --jepa-weight=1.0 \
  --nca-weight=0.35 \
  --optimizer=adamw \
  --jepa-warmup=2000 \
  --trial-id=E43-HigherNCA-200K \
  --agent-id=CHARLIE &

# E44: Combined JEPA+NCA increase
./target/release/tjepa_train \
  --seed=45 \
  --steps=200000 \
  --encoder-lr=0.004 \
  --ntp-lr=0.001 \
  --ntp-weight=1.0 \
  --jepa-weight=1.1 \
  --nca-weight=0.30 \
  --optimizer=adamw \
  --jepa-warmup=1800 \
  --trial-id=E44-CombinedHigher-200K \
  --agent-id=DELTA &

# E45: Lower LR with higher capacity
./target/release/tjepa_train \
  --seed=46 \
  --steps=200000 \
  --encoder-lr=0.003 \
  --ntp-lr=0.00075 \
  --ntp-weight=1.0 \
  --jepa-weight=1.0 \
  --nca-weight=0.30 \
  --optimizer=adamw \
  --jepa-warmup=2500 \
  --trial-id=E45-LowLR-HighCap-200K \
  --agent-id=ECHO &

echo "Launched E41-E45 experiments. Monitor with: ps aux | grep tjepa_train"
