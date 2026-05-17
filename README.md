# trios-trainer-igla

[![CI](https://github.com/gHashTag/trios-trainer-igla/actions/workflows/ci.yml/badge.svg)](https://github.com/gHashTag/trios-trainer-igla/actions/workflows/ci.yml)
[![Anchor](https://img.shields.io/badge/anchor-%CF%86%C2%B2%2B%CF%86%E2%81%BB%C2%B2%3D3-black)](https://doi.org/10.5281/zenodo.19227877)

IGLA RACE trainer. Tracks [gHashTag/trios#143](https://github.com/gHashTag/trios/issues/143).
Anchor: `phi^2 + phi^-2 = 3`.

> **Canonical Zenodo SOT:** [zenodo.org/communities/trinity-s3ai](https://zenodo.org/communities/trinity-s3ai/). The anchor badge resolves to record [19227877 (VSA Operations v5.0, B007)](https://doi.org/10.5281/zenodo.19227877), which is canonical inside the SOT community.

**Champion: BPB=2.2111** (seed=43, 81K steps, AdamW, hidden=384, Railway).

## Quick start

```bash
git clone https://github.com/gHashTag/trios-trainer-igla.git
cd trios-trainer-igla

# Download data
mkdir -p data
curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
    > data/tiny_shakespeare.txt
head -c 100000 data/tiny_shakespeare.txt > data/tiny_shakespeare_val.txt

# Build
cargo build --release

# Train single seed (best config)
./target/release/trios-train --seed=43 --steps=81000 --hidden=384 --lr=0.003 --optimizer=adamw

# Train all 3 Gate-2 seeds
for s in 42 43 44; do
  ./target/release/trios-train --seed=$s --steps=81000 --hidden=384 --lr=0.003 --optimizer=adamw
done
```

## Railway deploy (3 seeds in parallel)

```bash
# Prerequisites
brew install railway # or: npm i -g @railway/cli
railway login

# Link project (first time)
railway link  # select "trios-trainer"

# Create services (once)
for s in 42 43 44; do
  railway add --service "igla-seed-$s"  # choose "Empty Service"
  railway variables set --service "igla-seed-$s" TRIOS_SEED=$s
done

# Deploy all 3 seeds
for s in 42 43 44; do
  railway up --service "igla-seed-$s" --detach
done

# Watch logs
railway logs --service igla-seed-42
railway logs --service igla-seed-43
railway logs --service igla-seed-44
```

## Binaries

### `trios-train` — main trainer

```bash
./target/release/trios-train [OPTIONS]

Options:
      --seed <SEED>            Seed. 0 = 3-seed sweep {43,44,45} [env: TRIOS_SEED=] [default: 43]
      --steps <STEPS>          Training steps [env: TRIOS_STEPS=] [default: 54000]
      --hidden <HIDDEN>        Hidden dim [env: TRIOS_HIDDEN=] [default: 828]
      --lr <LR>                Learning rate [env: TRIOS_LR=] [default: 0.003]
      --attn-layers <N>        Attention layers [default: 2]
      --eval-every <N>         Eval interval [default: 1000]
      --optimizer <OPT>        adamw | muon | muon-cwd [env: TRIOS_OPTIMIZER=] [default: adamw]
      --train-data <PATH>      [default: data/tiny_shakespeare.txt]
      --val-data <PATH>        [default: data/tiny_shakespeare_val.txt]
      --config <TOML>          Config file (overrides flags)
      --sweep                  3-seed sweep {43,44,45}
```

### `trios-igla` — ledger query tool

```bash
./target/release/trios-igla <COMMAND>

Commands:
  search   Filter ledger rows (--seed, --bpb-max, --step-min, --sha)
  list     Last N rows (--last N)
  gate     Gate-2 quorum check (--target BPB)
  check    Embargo refusal for SHA
  triplet  Print R7 triplet for row index
```

### Other binaries

| Binary | Purpose |
|--------|---------|
| `hybrid_train` | N-gram + HybridAttn + ReLU² + Muon trainer |
| `seed_emit` | Emit a ledger row (--seed, --bpb, --step, --sha) |
| `ledger_check` | Validate ledger format |
| `qk_gain_check` | Check QK-gain against INV-13 (--lr, --gain) |

## Results (Railway, 2026-04-27)

| Config | Seed 42 | Seed 43 | Seed 44 | Avg |
|--------|---------|---------|---------|-----|
| **trios-train 81K AdamW h=384** | **2.222** | **2.211** | **2.218** | **2.217** |
| trios-train 27K AdamW h=384 | 2.362 | 2.359 | 2.387 | 2.369 |
| trios-train 54K Muon h=384 | 2.410 | 2.419 | 2.403 | 2.411 |
| hybrid_train 81K Muon+NCA h=828 | 2.686 | 2.681 | 2.678 | 2.682 |

P1 conclusion: **AdamW wins over Muon** on this architecture. Stick with AdamW.

## Environment variables

| Var | Default | Used by |
|-----|---------|---------|
| `TRIOS_SEED` | 43 | trios-train, entrypoint.sh |
| `TRIOS_STEPS` | 81000 | trios-train, entrypoint.sh |
| `TRIOS_LR` | 0.003 | trios-train, entrypoint.sh |
| `TRIOS_HIDDEN` | 384 | trios-train, entrypoint.sh |
| `TRIOS_OPTIMIZER` | adamw | trios-train, entrypoint.sh |
| `TRIOS_TRAIN_DATA` | tiny_shakespeare.txt | entrypoint.sh |
| `TRIOS_VAL_DATA` | tiny_shakespeare_val.txt | entrypoint.sh |
| `RUST_LOG` | info | all binaries |

## Docker

```bash
docker build -t trios-trainer .
docker run --rm -e TRIOS_SEED=42 -e TRIOS_STEPS=81000 trios-trainer
```

## Tests

```bash
cargo test --release          # unit + integration (9 tests)
cargo test --release -- --ignored  # champion reproduction (long)
```

## Gate-2 target

- **BPB < 1.85** on 3 seeds {42, 43, 44}, step >= 4000
- Deadline: 2026-04-30 23:59 UTC
- Current gap: +0.36 (BPB=2.21 → target 1.85)

## Roadmap

| Phase | Status | Result |
|-------|--------|--------|
| P0 Audit | DONE | Champion reproduced BPB=2.24 |
| P1 Optimizer Lab | DONE | NULL — AdamW wins |
| P2 muP Transfer | NEXT | Transfer LR to larger model |
| P3 Schedule-Free | Pending | SF/WSD vs cosine |
| P4 Multi-Obj + EMA | Pending | JEPA+NCA+EMA sweep |
| P5 Gate-2 Push | Running | BPB=2.21, need 1.85 |

See [`docs/TRAINING_FLOW_V2.md`](docs/TRAINING_FLOW_V2.md) for full plan.

## License

MIT

---

## Repo boundary

This is the **model plane** of the IGLA marathon. The control plane (Railway
client, `tri-railway` CLI, `tri-gardener` autonomous orchestrator,
`gardener_runs` ledger) lives in [`gHashTag/trios-railway`](https://github.com/gHashTag/trios-railway).
See [`docs/adr/0001-repo-boundaries.md`](docs/adr/0001-repo-boundaries.md) for
the binding contract.

`phi^2 + phi^-2 = 3 · TRINITY · NEVER STOP`
