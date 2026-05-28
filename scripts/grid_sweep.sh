#!/usr/bin/env bash
# Grid sweep generator for IGLA RACE
# Usage: ./scripts/grid_sweep.sh | head -N
#
# Focused grid:
#   formats: gf16 gf20 gf12 f32 bf16 fp8
#   hidden:  256 384 512
#   lr:      0.001 0.003
#   opt:     adamw muon
#   seeds:   42 43 44

FORMATS=(gf16 gf20 gf12 f32 bf16 fp8)
HIDDENS=(256 384 512)
LRS=(0.001 0.003)
OPTS=(adamw muon)
SEEDS=(42 43 44)

idx=0
for fmt in "${FORMATS[@]}"; do
  for hid in "${HIDDENS[@]}"; do
    for lr in "${LRS[@]}"; do
      for opt in "${OPTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          echo "${idx}|${fmt}|${hid}|${lr}|${opt}|${seed}"
          ((idx++))
        done
      done
    done
  done
done
