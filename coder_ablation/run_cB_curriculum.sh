#!/usr/bin/env bash
# IGLA-Coder C-only curriculum driver (Coder-Loop+10, B).
#
# Accumulates a long hidden=128 training run in resumable chunks using the new
# --resume flag. Each chunk warm-starts WEIGHTS from the prior checkpoint
# (fresh optimizer state -- the format stores weights only) and trains CHUNK
# more steps, then overwrites the checkpoint. Progress is appended to the log
# after every chunk so we can watch BPB descend across the curriculum.
#
# C-only: lang_id=1, fim-loss=middle (infill -- the natural code-completion
# objective). hidden=128 (493K params) lifts the hidden=64 capacity ceiling
# (296K) that pinned compile@1=0 in Loop+7/8. standard optimizer (phi is
# neutral per Loop+9, so no reason to handicap the capacity probe with it).
set -u
source "$HOME/.cargo/env"
BIN=../target/release/igla_coder
TRAIN=../data/code_train.bin
VAL=../data/code_val.bin
CKPT=ckpt_cB_h128.bin
LOG=loop10_cB_curriculum.log
CHUNK=3000
TOTAL=12000
COMMON="--train $TRAIN --val $VAL --optimizer standard --beta1 0.9 --wd 0.04 \
  --lr 0.002 --hidden 128 --lang-id 1 --fim-loss middle --max-new 0 \
  --steps $CHUNK --save $CKPT"

echo "anchor: phi^2 + phi^-2 = 3" > "$LOG"
echo "C-only curriculum: hidden=128 fim=middle lang=C standard-opt; chunk=$CHUNK total=$TOTAL" >> "$LOG"

done_steps=0
# the ckpt already holds 200 steps from the timing probe; treat that as step 0
# baseline and accumulate from there.
echo "baseline ckpt = 200 warmup steps (timing probe)" >> "$LOG"
while [ "$done_steps" -lt "$TOTAL" ]; do
  $BIN generate $COMMON --seed 42 --resume "$CKPT" 2>&1 \
    | grep -E "pre_resume_val_bpb|trained code_val_bpb|saved checkpoint" >> "$LOG"
  done_steps=$((done_steps + CHUNK))
  echo "=== cumulative additional steps: $done_steps / $TOTAL ===" >> "$LOG"
done
echo "CURRICULUM COMPLETE: $done_steps additional steps on top of 200 warmup" >> "$LOG"
echo "final checkpoint: $CKPT" >> "$LOG"
