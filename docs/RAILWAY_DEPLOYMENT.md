# Railway Seed Deployment Guide

## Overview

Deploy trios-trainer-igla with **per-seed Railway containers**. Each Railway
service spawns one training run with its own seed via the `TRIOS_SEED`
environment variable, and the same Docker image runs every seed without
rebuilds.

### Seed history (R5-honest)

| Wave | Seeds | Status | Notes |
|---|---|---|---|
| Champion | 43 | Reproduces BPB=2.2393 @ 27K | reference run, do not touch |
| Attempt-1 | 43, 44, 45 | 0 rows < BPB 1.85 | old fleet, tried previously |
| **Attempt-2 (this deploy)** | **46, 47, 48** | **NEW** | new seeds — continue sequence after 45 |

The new fleet keeps the same numeric series (one number after the previous
attempt) so we can A/B against attempt-1 cleanly.

## Prerequisites

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login (if not already logged in)
railway login

# Verify connection to the IGLA project
railway list
```

## Quick Deploy (TTY Terminal Required)

### Method 1: One-liner (Fastest)

```bash
cd trios-trainer-igla
for seed in 46 47 48; do
  railway add --service "igla-trainer-seed-$seed" --variables "TRIOS_SEED=$seed"
  railway up --service "igla-trainer-seed-$seed"
done
```

### Method 2: Individual Commands

```bash
# Seed 46
railway add --service igla-trainer-seed-46 --variables "TRIOS_SEED=46"
railway up --service igla-trainer-seed-46

# Seed 47
railway add --service igla-trainer-seed-47 --variables "TRIOS_SEED=47"
railway up --service igla-trainer-seed-47

# Seed 48
railway add --service igla-trainer-seed-48 --variables "TRIOS_SEED=48"
railway up --service igla-trainer-seed-48
```

### Method 3: Cleanup attempt-1 services first (recommended)

The old `igla-trainer-seed-43/44/45` services completed with no Gate-2 row.
Delete them before the new deploy to avoid name collisions and idle cost:

```bash
railway delete --service igla-trainer-seed-43
railway delete --service igla-trainer-seed-44
railway delete --service igla-trainer-seed-45

# Then deploy the NEW fleet
for seed in 46 47 48; do
  railway add --service "igla-trainer-seed-$seed" --variables "TRIOS_SEED=$seed"
  railway up --service "igla-trainer-seed-$seed"
done
```

### Method 4: Future waves (override seeds)

To launch any future wave (e.g. 49, 50, 51) without editing scripts:

```bash
NEW_SEEDS="49 50 51" bash scripts/deploy-seeds.sh

# Or directly:
for seed in 49 50 51; do
  railway add --service "igla-trainer-seed-$seed" --variables "TRIOS_SEED=$seed"
  railway up --service "igla-trainer-seed-$seed"
done
```

## Using Deployment Scripts

```bash
cd trios-trainer-igla

# Show commands without executing (default: 46 47 48)
bash scripts/deploy-seeds.sh

# Override seeds via env var
NEW_SEEDS="100 101 102" bash scripts/deploy-seeds.sh

# Railway API deployment (requires RAILWAY_TOKEN)
# Get token: https://build.railway.app/settings/tokens
export RAILWAY_TOKEN=your_token_here
bash scripts/railway-seed-deploy.sh
```

## Monitoring

After deployment, monitor at:
```
https://railway.app/project/IGLA
```

Or check logs:
```bash
railway logs --service igla-trainer-seed-46
railway logs --service igla-trainer-seed-47
railway logs --service igla-trainer-seed-48
```

## Training Configuration

Each seed service uses:

- **TRIOS_SEED**: DYNAMIC — set per-service via Railway `--variables` at deploy
  time. New fleet uses 46/47/48.
- Default seed in `Dockerfile`: `46` (overridden by Railway env var).
- **TRIOS_CONFIG**: `/configs/gate2-attempt.toml` (from Dockerfile).
- **TRIOS_STEPS**: `81000` (from Dockerfile).
- **TRIOS_LR**: `0.003` (from Dockerfile).
- **TRIOS_TARGET_BPB**: `1.50` (from Dockerfile).

### Dynamic Seed Handling

The image now runs `scripts/entrypoint.sh`, which expands `$TRIOS_SEED` at
runtime (Docker `CMD` exec-form does not expand env vars, so a shell wrapper
is required):

```bash
exec /usr/local/bin/trios-train --config "$CONFIG" --seed "$TRIOS_SEED" "$@"
```

`Dockerfile`:

```dockerfile
ENV TRIOS_SEED=46  # Default for the new fleet, overridden by Railway env var
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
```

Each Railway service injects its own `TRIOS_SEED` at deploy time and the
entrypoint forwards it to `trios-train`.

## Notes

- Railway CLI requires TTY for interactive commands (service creation).
- All services share the same Docker image — only the `TRIOS_SEED` env var
  differs between them.
- Each ledger emit must carry the full triplet:
  `BPB=<v> @ step=<N> seed=<S> sha=<7c> jsonl_row=<L> gate_status=<g>` (R7).
- Embargo (`assertions/embargo.txt`, 8 SHAs) is checked before every
  `ledger::emit_row`. Do not bypass.
- Gate-2 deadline: **2026-04-30 23:59 UTC**.
