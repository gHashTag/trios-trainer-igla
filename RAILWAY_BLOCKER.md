# Railway Deployment Blocker — Rust 1.82.0 Issue

**Date:** 2026-04-27
**Status:** Blocked — Railway NOT using Dockerfile

## Problem

Railway deployment fails with:
```
error: failed to parse manifest at `/usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/hashbrown-0.17.0/Cargo.toml`
feature `edition2024` is required
The package requires the Cargo feature called `edition2024`, but that feature is not stabilized in this version of Cargo (1.82.0)
```

## Root Cause

Railway's build system is **NOT using the Dockerfile** despite `railway.json` specifying:
```json
{
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  }
}
```

Evidence: Build logs show `RUN cargo build --release -p igla-trainer` which differs from Dockerfile's `RUN cargo build --release --bin trios-train -p trios-trainer`

## Attempted Fixes

1. **nixpacks.toml with Rust v1.91 provider** — Failed, Railway still uses 1.82.0
2. **Remove nixpacks files** — Failed, Railway still uses cached build
3. **Update Dockerfile to build trios-train** — Failed, Railway uses wrong command
4. **Update entrypoint.sh** — Irrelevant, Dockerfile not used
5. **Explicit Rust 1.91 via rustup** — Failed, Railway still uses 1.82.0

## Workarounds to Try

1. **Manual Railway dashboard configuration** — Service may need to be manually reconfigured
2. **Create service from Railway dashboard** — Bypass CLI entirely
3. **Pin problematic dependencies** — Not ideal but may work (hashbrown, block-buffer, time-macros)
4. **Alternative deployment platform** — Consider Fly.io, Render, or other platforms

## Current Status

- P1 null result: ✅ Documented
- gate2-final.toml: ✅ Updated to AdamW
- Railway deployment: ❌ Blocked
- Local training: ⚠️ Works but uses synthetic data
- Best BPB: 2.3586 (seed 43, 27K steps)
- Target BPB: 1.85 (gap: 0.51 BPB)
- Deadline: 2026-04-30 23:59 UTC (~3 days)

## Next Steps

1. User intervention needed: Configure Railway via dashboard or provide correct RAILWAY_PROJECT_ID
2. Or: Try alternative deployment method
3. Continue local experiments with real data (if available)
