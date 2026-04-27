# Deployment Blocker Summary

**Date:** 2026-04-27
**Issue:** Railway deployment blocked by Rust toolchain version incompatibility
**Deadline:** 2026-04-30 23:59 UTC (~3 days remaining)
**Status:** CRITICAL - BUT Railway IS working for other services

## NEW CHAMPION (2026-04-27 20:20 UTC)

- **BPB = 2.2111** (seed 43, 81K steps)
- Previous: 2.2393
- **Improvement: -0.028 BPB (-1.25%)**
- Gap to target (1.85): 0.3611 BPB
- Architecture: hidden=384, lr=0.003, adamw, 2 attn layers, tiny_shakespeare
- Agent: `railway-trios-train-81k` (ALPHA)

## Problem

Railway's build environment for the `igla-trainer-seed-101` service uses Rust 1.82.0 which does NOT support the `edition2024` feature required by `hashbrown-0.17.0` dependency.

### Error
```
error: failed to parse manifest at `/usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/hashbrown-0.17.0/Cargo.toml`
feature `edition2024` is required
The package requires the Cargo feature called `edition2024`, but that feature is not stabilized in this version of Cargo (1.82.0 (8f40fc59f 2024-08-21)).
```

## Working Services (Evidence Railway CAN work)

The following services are working on Railway (from environment variables):
- `trios-dwagent-production`
- `trios-mcp-public-production`
- `trios-train-seed-44-production`

The champion results were achieved on `railway-trios-train-81k`, proving that Railway deployment IS possible with the correct configuration.

## Root Cause

The `igla-trainer-seed-101` service appears to have a **cached build configuration** that references an old package (`igla-trainer`) instead of the current Dockerfile. The build command shown is:
```
RUN cargo build --release -p igla-trainer
```

But Dockerfile specifies:
```
RUN cargo build --release --bin trios-train -p trios-trainer
```

## Attempted Fixes

1. ✗ **Dockerfile with explicit Rust 1.91** - `igla-trainer-seed-101` ignores Dockerfile
2. ✗ **nixpacks.toml with Rust v1.91 provider** - Service still uses 1.82.0
3. ✗ **railway.json with builder: DOCKERFILE** - Service uses wrong build command
4. ✗ **Pin clap to 4.5.4** - Still fails on hashbrown
5. ✗ **Railway up --detach** - Upload timeout (11MB)
6. ✗ **railway link to different service** - Requires TTY (not available in autonomous mode)

## Alternative Deployment Options

### Option 1: Use Existing Working Service ⭐ RECOMMENDED
- Link to `trios-train-seed-44-production` or create similar service
- Requires Railway dashboard access (TTY not available)
- Copy configuration from working service

### Option 2: Create New Railway Service
- Delete `igla-trainer-seed-101` and create fresh one
- Requires Railway dashboard access
- Risk: May use same cached toolchain

### Option 3: Fly.io
- CLI available: `/opt/homebrew/bin/fly`
- Requires: `flyctl auth login` (needs user action)
- Better Dockerfile support than Railway

### Option 4: Render
- Alternative PaaS platform
- Unknown CLI status
- May need user signup

## Current Status

- **P1 null result:** ✅ Documented (Muon worse than AdamW)
- **gate2-final.toml:** ✅ Updated with AdamW
- **Build:** ✅ Local build succeeds
- **Deployment (igla-trainer-seed-101):** ❌ BLOCKED by cached config
- **Deployment (other services):** ✅ WORKING (champion achieved)
- **Best BPB:** 2.2111 (seed 43, 81K steps) - NEW CHAMPION!
- **Target BPB:** 1.85
- **Gap:** 0.3611 BPB

## Immediate Action Required

**USER INTERVENTION NEEDED:** One of the following:
1. Link to working Railway service (`trios-train-seed-44-production`) via dashboard
2. Run `flyctl auth login` to enable Fly.io deployment
3. Delete and recreate `igla-trainer-seed-101` service via dashboard

## Fallback Plan

If deployment cannot be resolved by 2026-04-28 12:00 UTC:
1. Document post-mortem of IGLA RACE #143
2. Publish findings (P1 null result, best achieved BPB: 2.2111)
3. Exit with dignity (no false claims)

## Configuration Ready

When deployment becomes possible, these configs are ready:
- `configs/gate2-final.toml` - Final configuration
- `Dockerfile` - Multi-stage build with Rust 1.91
- `scripts/entrypoint.sh` - Entrypoint script
- `railway.json` - Railway build config

## Contact

For deployment assistance or to provide credentials, coordinate via:
- GitHub issue: #143
- Experience log: `.trinity/experience/trios_20260427_pt2.trinity`
