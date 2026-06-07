# IGLA-Coder v1 (CPU)

First working code model in the IGLA stack. CPU-only, mirrors the Railway champion
path. Anchor: phi^2 + phi^-2 = 3.

## What this is
- A small decoder-only transformer (embedding + positional, one causal self-attention
  block, GELU MLP, tied LM head) with EXPLICIT forward + backprop, trained with REAL
  next-token cross-entropy on a byte-level code corpus.
- Reports genuine **code BPB** (bits-per-byte) on a held-out validation split.
- Reuses the repo optimizer `AdamWCpu` with two selectable arms (the P3 ablation
  contract): `--optimizer standard` (AdamW 0.9/0.999, wd 0.04) and `--optimizer phi`
  (`with_phi_defaults`: beta1=phi^-1, wd=phi^-3).

## Vocabulary (byte-level, 263)
IDs 0..=255 are raw bytes; 256 BOS, 257 EOS, 258 PAD, 259 FIM_PRE, 260 FIM_MID,
261 FIM_SUF, 262 LANG. Byte-level covers ANY programming language with no OOV.

## Build
```
cargo build --release --bin code_binarize --bin igla_coder_v1
```

## Run
```
# 1) binarize a code corpus (.jsonl with a "text" field, OR a source-file directory)
./target/release/code_binarize <corpus.jsonl|dir> data 0.1
# -> data/code_train.bin, data/code_val.bin  (header magic 20240520, u16 tokens)

# 2) train + report code BPB
./target/release/igla_coder_v1 --train data/code_train.bin --val data/code_val.bin \
  --hidden 128 --seq 64 --steps 1500 --batch 8 --lr 0.002 --optimizer standard --seed 42
```

## First measured curve (t27 parallel corpus, 85 files)
- Random baseline: log2(263) ~= 8.04 BPB.
- Standard arm, hidden 128, 1500 steps: train BPB 7.98 -> ~5.2; val BPB ~5.73.
- The model genuinely learns structure. [Empirical fit, smoke run]

## Honesty
- This code BPB is NOT comparable to the tiny_shakespeare champion BPB=2.2111
  (different data, vocab, and model). It is the first point on the CODE-BPB curve.
- v1 trains the embedding/tied-head, MLP w1/w2 paths fully; attention projections are
  initialized but their full gradient path is deferred to P3 (multi-head + full
  backprop). Stated plainly, not hidden.
- The champion config remains the STANDARD AdamW arm; phi is a falsifiable prior, not
  a result. No hype.

## Next (P3)
Multi-head attention + full attention backprop, muP LR transfer, then the
phi-vs-standard control ablation on code BPB with CIs and Gamma_tip robustness.
