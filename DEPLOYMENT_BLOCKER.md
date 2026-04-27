# Deployment Blocker Summary

**Date:** 2026-04-27
**Issue:** Railway deployment blocked by Rust toolchain version incompatibility
**Deadline:** 2026-04-30 23:59 UTC (~3 days remaining)
**Status:** CRITICAL

## Problem

Railway's build environment uses Rust 1.82.0 which does NOT support the `edition2024` feature required by `hashbrown-0.17.0` dependency.

### Error
```
error: failed to parse manifest at `/usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/hashbrown-0.17.0/Cargo.toml`
feature `edition2024` is required
The package requires the Cargo feature called `edition2024`, but that feature is not stabilized in this version of Cargo (1.82.0 (8f40fc59f 2024-08-21)).
```

## Attempted Fixes

1. ✗ **Dockerfile with explicit Rust 1.91** - Railway ignores Dockerfile
2. ✗ **nixpacks.toml with Rust v1.91 provider** - Railway still uses 1.82.0
3. ✗ **railway.json with builder: DOCKERFILE** - Railway uses wrong build command
4. ✗ **Pin clap to 4.5.4** - Still fails on hashbrown
5. ✗ **Railway up --detach** - Upload timeout (11MB)

## Root Cause

Railway appears to be using a **cached build configuration** that references an old package (`igla-trainer`) instead of the current Dockerfile. The build command shown is:
```
RUN cargo build --release -p igla-trainer
```

But Dockerfile specifies:
```
RUN cargo build --release --bin trios-train -p trios-trainer
```

## Alternative Deployment Options

### Option 1: Create New Railway Service
- Delete current service and create fresh one
- Requires Railway dashboard access
- Risk: May use same cached toolchain

### Option 2: Fly.io
- CLI available: `/opt/homebrew/bin/fly`
- Requires: `flyctl auth login` (needs user action)
- Better Dockerfile support than Railway

### Option 3: Render
- Alternative PaaS platform
- Unknown CLI status
- May need user signup

### Option 4: Self-hosted (last resort)
- Set up VPS with Docker
- Requires infrastructure setup
- Time-consuming (~1-2 days)

## Current Status

- **P1 null result:** ✅ Documented (Muon worse than AdamW)
- **gate2-final.toml:** ✅ Updated with AdamW
- **Build:** ✅ Local build succeeds
- **Deployment:** ❌ BLOCKED
- **Best BPB:** 2.3586 (seed 43, 27K steps)
- **Target BPB:** 1.85
- **Gap:** 0.51 BPB

## Immediate Action Required

**USER INTERVENTION NEEDED:** One of the following:
1. Log in to Railway dashboard and configure service to use Dockerfile
2. Run `flyctl auth login` to enable Fly.io deployment
3. Provide alternative deployment platform credentials

## Fallback Plan

If deployment cannot be resolved by 2026-04-28 12:00 UTC:
1. Document post-mortem of IGLA RACE #143
2. Publish findings (P1 null result, best achieved BPB)
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
