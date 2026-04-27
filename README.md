# trios-trainer-igla

> IGLA RACE trainer — single-binary Rust, runs everywhere.
> Anchor: `φ² + φ⁻² = 3`

[![CI](https://github.com/gHashTag/trios-trainer-igla/actions/workflows/ci.yml/badge.svg)](https://github.com/gHashTag/trios-trainer-igla/actions/workflows/ci.yml)

## Quick start

```bash
git clone https://github.com/gHashTag/trios-trainer-igla.git
cd trios-trainer-igla

# Download training data
mkdir -p data
curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt > data/tiny_shakespeare.txt
head -c 100000 data/tiny_shakespeare.txt > data/tiny_shakespeare_val.txt

# Train (single seed)
cargo run --release --bin hybrid_train -- --seed=43 --steps=27000

# Train (3-seed Gate-2 sweep)
cargo run --release --bin hybrid_train -- --seed=0 --steps=54000
```

## Binaries

| Binary | Purpose |
|--------|---------|
| `hybrid_train` | **Main trainer** — N-gram + HybridAttn (causal, RoPE, QK-Gain=φ²) + ReLU² + Muon + NCA |
| `trios-train` | Config-driven trainer (TOML configs in `configs/`) |
| `trios-igla` | Ledger query tool — search, gate-2 verdict, embargo check |

## hybrid_train flags

```
--seed=N        Seed (0 = 3-seed sweep {42,43,44}). Default: 0
--steps=N       Training steps. Default: 54000
--lr=F          Learning rate. Default: 0.003
--hidden=N      Hidden dim. Default: 828
--eval-every=N  Eval interval. Default: 1000
--accum=N       Gradient accumulation. Default: 4
```

## Deploy to Railway (3 seeds in parallel)

```bash
# Prerequisites: Railway CLI installed + logged in
railway login

# Link project
railway link  # select "trios-trainer"

# Create services (once)
for s in 42 43 44; do
  railway add --service "igla-seed-$s"  # choose "Empty Service"
  railway variables set --service "igla-seed-$s" TRIOS_SEED=$s
  railway variables set --service "igla-seed-$s" TRIOS_STEPS=54000
done

# Deploy all 3 seeds
for s in 42 43 44; do
  railway up --service "igla-seed-$s" --detach
done

# Check logs
railway logs --service igla-seed-42
railway logs --service igla-seed-43
railway logs --service igla-seed-44

# Check status
railway status
```

## Via `tri` CLI (t27)

```bash
# From t27 repo
cargo build --release --bin tri -p trios-cli

# Dry-run
tri railway deploy --seeds 3 --start-seed 42 --dry-run

# Deploy
tri railway deploy --seeds 3 --start-seed 42
```

## Docker on any VPS

```bash
docker build -t trios-trainer .
docker run --rm -e TRIOS_SEED=42 trios-trainer
docker run --rm -e TRIOS_SEED=43 trios-trainer
docker run --rm -e TRIOS_SEED=44 trios-trainer
```

## Architecture

```
hybrid_train (main training binary)
├── N-gram base: NGRAM=8, DIM=64, VOCAB=128, NUM_CTX=6
├── HybridAttn: 2-layer causal attention, RoPE, QK-Gain=φ² (INV-9)
├── ReLU² activation
├── Muon optimizer (NS5 orthogonalization) for proj weights
├── NCA auxiliary loss (entropy band [1.5, 2.8], INV-4)
├── AdamW for embeddings, attention projections, lm_head
├── Cosine LR schedule with φ-warmup
├── GF16 weight flooring at 70% of training
└── EMA val BPB (β=φ⁻¹)
```

## Gate-2 Target

- **BPB < 1.50** on 3 seeds {42, 43, 44}
- Deadline: 2026-04-30 23:59 UTC
- Champion: BPB=2.2393 @ 27K steps, seed=43, sha `2446855`

## BPB Roadmap

| Step | Technique | Target BPB |
|------|-----------|------------|
| T1-01 | JEPA-T real backward | ≤2.23 |
| T1-02 | Attention + ReLU² | ≤2.00 |
| T2-01 | Muon optimizer | ≤1.85 |
| T2-02 | NCA auxiliary (INV-4) | ≤1.70 |
| T2-04 | QK-Gain φ² (INV-9) | ≤1.60 |
| T2-07 | ReLU² activation | ≤1.52 |
| T2-07b | GF16 d_model=384 | ≤1.47 |

## Environment variables

| Var | Default | Effect |
|-----|---------|--------|
| `TRIOS_SEED` | 43 | Training seed |
| `TRIOS_STEPS` | 54000 | Max training steps |
| `TRIOS_LR` | 0.003 | Learning rate |
| `TRIOS_HIDDEN` | 828 | Hidden dimension |
| `TRIOS_EVAL_EVERY` | 1000 | Eval every N steps |

## Configs

| File | Purpose |
|------|---------|
| `configs/champion.toml` | Champion baseline reproduction (BPB=2.2393) |
| `configs/gate2-attempt.toml` | HybridAttn + JEPA push |
| `configs/gate2-final.toml` | Gate-2 final config |

## Invariants

- **INV-3**: d_model ≥ 256 for GF16
- **INV-4**: NCA entropy band [1.5, 2.8], K=9
- **INV-8**: lr ∈ [0.001, 0.01]
- **INV-9**: QK-Gain = φ² ≈ 2.618

Anchor: `φ² + φ⁻² = 3` ([Zenodo 10.5281/zenodo.19227877](https://doi.org/10.5281/zenodo.19227877)).

## License

MIT
