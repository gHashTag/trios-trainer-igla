# Railway Seed Deployment Guide

## Overview

Deploy trios-trainer-igla with 3 parallel seed services on Railway:
- `igla-trainer-seed-43`
- `igla-trainer-seed-44`
- `igla-trainer-seed-45`

## Prerequisites

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login (if not already logged in)
railway login

# Verify connection to IGLA project
railway list
```

## Quick Deploy (TTY Terminal Required)

### Method 1: One-liner (Fastest)

```bash
cd trios-trainer-igla
for seed in 43 44 45; do
  railway add --service "igla-trainer-seed-$seed" --variables "TRIOS_SEED=$seed"
  railway up --service "igla-trainer-seed-$seed"
done
```

### Method 2: Individual Commands

```bash
# Seed 43
railway add --service igla-trainer-seed-43 --variables "TRIOS_SEED=43"
railway up --service igla-trainer-seed-43

# Seed 44
railway add --service igla-trainer-seed-44 --variables "TRIOS_SEED=44"
railway up --service igla-trainer-seed-44

# Seed 45
railway add --service igla-trainer-seed-45 --variables "TRIOS_SEED=45"
railway up --service igla-trainer-seed-45
```

## Using Deployment Scripts

```bash
cd trios-trainer-igla

# Show commands without executing
bash scripts/deploy-seeds.sh

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
railway logs --service igla-trainer-seed-43
railway logs --service igla-trainer-seed-44
railway logs --service igla-trainer-seed-45
```

## Cleanup

To remove services:
```bash
railway delete --service igla-trainer-seed-43
railway delete --service igla-trainer-seed-44
railway delete --service igla-trainer-seed-45
```

## Training Configuration

Each seed service will use:
- **TRIOS_SEED**: 43, 44, or 45
- **TRIOS_CONFIG**: `/configs/gate2-attempt.toml` (from Dockerfile)
- **TRIOS_STEPS**: 81000 (from Dockerfile)
- **TRIOS_LR**: 0.003 (from Dockerfile)
- **TRIOS_TARGET_BPB**: 1.50 (from Dockerfile)

## Notes

- Railway CLI requires TTY for interactive commands (service creation)
- L-R1: MAX 4 Railway instances allowed per trios law #143
- Each seed runs independently for Gate-2 verification (P5 phase)
