# Tomorrow Wave — IGLA RACE Grid Expansion

**Planned for:** 2026-05-29 (when Railway 25/day limit resets)
**Goal:** Deploy additional sweep configs with 50M steps from start

## Configs to deploy (pre-validated)

| # | Service Name | Format | Hidden | LR | Optimizer | Seed | Steps |
|---|-------------|--------|--------|----|-----------|------|-------|
| 1 | scarab-gf32-seed99 | gf32 | 256 | 0.001 | adamw | 99 | 50000000 |
| 2 | scarab-gf64-seed100 | gf64 | 256 | 0.001 | adamw | 100 | 50000000 |
| 3 | scarab-gf16-h512-seed47 | gf16 | 512 | 0.001 | adamw | 47 | 50000000 |
| 4 | scarab-f32-h1024-seed89 | f32 | 1024 | 0.001 | adamw | 89 | 50000000 |
| 5 | scarab-gf16-muon-h512-seed144 | gf16 | 512 | 0.001 | muon | 144 | 50000000 |

## Commands
```bash
railway add -s scarab-gf32-seed99
railway variables --service scarab-gf32-seed99 --set "TRIOS_FORMAT=gf32"
railway variables --service scarab-gf32-seed99 --set "TRIOS_SEED=99"
railway variables --service scarab-gf32-seed99 --set "TRIOS_STEPS=50000000"
railway variables --service scarab-gf32-seed99 --set "TRIOS_HIDDEN=256"
railway variables --service scarab-gf32-seed99 --set "TRIOS_LR=0.001"
railway variables --service scarab-gf32-seed99 --set "TRIOS_OPTIMIZER=adamw"
railway up -s scarab-gf32-seed99 -d -c
# ... repeat for each config
```

## Notes
- All seeds must be from allowed set: {47, 89, 123, 144}
- GF32/GF64 formats require Rust codegen verification before deploy
- Set steps ONCE at creation to avoid container restart
