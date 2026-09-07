# trios-trainer-igla

[![CI](https://github.com/gHashTag/trios-trainer-igla/actions/workflows/ci.yml/badge.svg)](https://github.com/gHashTag/trios-trainer-igla/actions/workflows/ci.yml)
[![Anchor](https://img.shields.io/badge/anchor-%CF%86%C2%B2%2B%CF%86%E2%81%BB%C2%B2%3D3-black)](https://doi.org/10.5281/zenodo.19227877)

IGLA RACE trainer. Tracks [gHashTag/trios#143](https://github.com/gHashTag/trios/issues/143).
Anchor: `phi^2 + phi^-2 = 3`.

> **Canonical Zenodo SOT:** [zenodo.org/communities/trinity-s3ai](https://zenodo.org/communities/trinity-s3ai/). The anchor badge resolves to record [19227877 (VSA Operations v5.0, B007)](https://doi.org/10.5281/zenodo.19227877), which is canonical inside the SOT community.

> ### The BPB figures in this README are withdrawn
>
> They are left visible below rather than deleted. A number that quietly
> disappears is indistinguishable from one that was never wrong, and anyone who
> quoted these should be able to find out why they should not.
>
> **No model artifact exists for any of them.** `checkpoint::save` on this branch
> is a stub that returns `Ok(())` and writes nothing
> ([`src/checkpoint.rs:116`](src/checkpoint.rs#L116)). There is nothing on disk to
> re-evaluate, so no figure here can be reproduced, even in principle.
>
> **The quick start below contradicted this repository's own leak guard.** It
> built the validation file as a verbatim prefix of the training file --- which
> is the exact condition `assert_train_val_disjoint`
> ([`src/train_loop.rs:114`](src/train_loop.rs#L114)) aborts on, and whose panic
> message prescribes the correct split. Following the old instructions did not
> produce a quiet leak; it produced an immediate assertion failure. The split
> below is now the one the guard asks for.
>
> An atomic checkpoint save whose sha256 is taken over bytes re-read from disk,
> with the provenance record and uncertainty budget that belong beside it, is on
> branch `fix/509-qat-v2` and **is not merged here**. Nothing in this repository
> has been re-measured under that code.

## Quick start

```bash
git clone https://github.com/gHashTag/trios-trainer-igla.git
cd trios-trainer-igla

# Download data
mkdir -p data
curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
    > data/tiny_shakespeare_full.txt

# Byte-disjoint split, as assert_train_val_disjoint requires: validation is the
# LAST 100000 bytes, training is everything before it. Concatenating the two
# reconstructs the original file byte for byte.
FULL=$(wc -c < data/tiny_shakespeare_full.txt)
head -c $((FULL - 100000)) data/tiny_shakespeare_full.txt > data/tiny_shakespeare.txt
tail -c 100000            data/tiny_shakespeare_full.txt > data/tiny_shakespeare_val.txt

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

## Results (Railway, 2026-04-27) --- withdrawn

Every figure in this table is covered by the withdrawal note at the top of this
file: it was produced by a build that persisted no checkpoint.

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

## Gate-2 target --- historical

The deadline below passed on 2026-04-30 and the figures quoted in it are
withdrawn. Kept for provenance.

- **BPB < 1.85** on 3 seeds {42, 43, 44}, step >= 4000
- Deadline: 2026-04-30 23:59 UTC
- Current gap: +0.36 (BPB=2.21 → target 1.85)

## Roadmap --- historical

The result column quotes withdrawn figures. Kept for provenance.

| Phase | Status | Result |
|-------|--------|--------|
| P0 Audit | DONE | Champion reproduced BPB=2.24 |
| P1 Optimizer Lab | DONE | NULL — AdamW wins |
| P2 muP Transfer | NEXT | Transfer LR to larger model |
| P3 Schedule-Free | Pending | SF/WSD vs cosine |
| P4 Multi-Obj + EMA | Pending | JEPA+NCA+EMA sweep |
| P5 Gate-2 Push | Running | BPB=2.21, need 1.85 |

See [`docs/TRAINING_FLOW_V2.md`](docs/TRAINING_FLOW_V2.md) for full plan.

## Railway Deployment Status (2026-04-27)

3-seed cloud fleet **deployed and live** in Railway project `IGLA`
([`e4fe33bb-3b09-4842-9782-7d2dea1abc9b`](https://railway.com/project/e4fe33bb-3b09-4842-9782-7d2dea1abc9b)),
env `production` (`54e293b9-00a9-4102-814d-db151636d96e`):

| Service | serviceId | Image | Deploy |
|---|---|---|---|
| `trios-train-seed-100` | `0f0a948f-c457-4f4c-b5c7-a5ef96fcf9e9` | `ghcr.io/ghashtag/trios-trainer-igla:latest` | SUCCESS |
| `trios-train-seed-101` | `8e1c7858-5c38-43bc-8015-23c46aaa1ee2` | `ghcr.io/ghashtag/trios-trainer-igla:latest` | SUCCESS |
| `trios-train-seed-102` | `20b0fcef-b6da-4853-94b7-b1cc27cbd406` | `ghcr.io/ghashtag/trios-trainer-igla:latest` | SUCCESS |

E2E test of `tri` + `trios-igla` against [`ee7771f`](https://github.com/gHashTag/trios-trainer-igla/commit/ee7771f7) is recorded in [gHashTag/trios#143](https://github.com/gHashTag/trios/issues/143#issuecomment-4324652513).

### R5 honesty — Gate-2 status: **NOT DONE**

Containers **start and train**, but the training corpus is missing from `ghcr.io/ghashtag/trios-trainer-igla:latest`:

```
Failed to load data/tiny_shakespeare.txt: No such file or directory (os error 2). Using fallback.
step=4000 ntp=0.0001 ... val_bpb=0.0001
step=7000 ntp=0.0000 ... val_bpb=0.0000
```

`val_bpb=0.0000` is degenerate output of the synthetic fallback in `train_loop.rs:57`, **not** a Gate-2 win. No valid R7 ledger row will be emitted from these services until the corpus is baked into the image (or a Railway volume is mounted). Same root cause on legacy seed-43/44/45 (`val_bpb=3.4e38` = `f32::MAX` overflow).

### Open follow-up issues

- **P0 — bake corpus into Docker image**: add `COPY data/tinyshakespeare.txt /app/data/` (or `ADD <url>`) to the `Dockerfile`; flip `train_loop.rs` from silent fallback to `bail!` on missing data file.
- **P1 — `tri.rs` infra mismatch**:
  - `RAILWAY_PROJECT_ID = "abdf752c-..."` -> `e4fe33bb-3b09-4842-9782-7d2dea1abc9b` (live IGLA project)
  - `service_name(seed) = "trainer-seed-{}"` -> `"trios-train-seed-{}"` (live convention)
  - `GATE_SEEDS = &[42, 43, 44]` -> `&[100, 101, 102]` (or env-driven)
- **P2 — clean stale fleet**: legacy `trios-train-seed-43/44/45` show `f32::MAX` overflow; remove or diagnose.

## License

MIT
