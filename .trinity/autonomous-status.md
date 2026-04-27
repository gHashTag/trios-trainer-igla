# IGLA RACE Autonomous Status
# Last updated: 2026-04-27T17:30:00Z

## Current Progress

### trios-trainer-igla (IGLA RACE repo)

- **P0 (Audit)**: Champion reproduction tested
  - Best 27K-step result: BPB=2.3586 (seed 43, AdamW, d=384, 2L)
  - Champion target: 2.2393 (original baseline)

- **P1 (Optimizer Lab)**: Configs ready for Railway deployment
  - `configs/lab/p1-adamw.toml` - Control (AdamW, LR=0.004)
  - `configs/lab/p1-muon.toml` - Muon (η2D=0.008, η1D=0.007)
  - `configs/lab/p1-muon-cwd.toml` - Muon+CWD (LR=0.008, CWD enabled)
  - All configs: 12K steps, seed 43, d_model=256, n_layers=2
  - **nixpacks-p1.toml** available for P1 lab deployments
  - **Next**: Deploy to Railway (requires RAILWAY_TOKEN)

- **P2-P5**: Pending P1 completion

### Code Changes

- Fixed `igla::LedgerRow` deserialization to support both legacy and new schema
  - Added `#[serde(alias = "steps")]` for step field
  - Added optional legacy fields: `val_bpb_24k`, `val_bpb_27k`, `ema_bpb`, `optimizer`, `hidden`, `lr`, `attn_layers`, `time_s`
- Added `[workspace]` table to Cargo.toml for isolation from parent trios workspace
- Built and tested `trios-igla` binary successfully

## Next Steps (Priority)

1. **Set RAILWAY_TOKEN** - Required for Railway deployment
2. **Deploy P1 to Railway** - Use `scripts/railway-seed-deploy.sh` or Railway CLI
3. **Monitor P1 results** - Collect results, determine winner (>= 0.05 BPB margin)
4. **Proceed to P2-P5** - Follow TRAINING_FLOW_V2.md phases

## Gate-2 Status

- **Target**: BPB ≤ 1.85 on 3 seeds (step ≥ 4000)
- **Current best**: 2.3586 (seed 43, step 27K, AdamW, d=384, 2L)
- **Gap**: 0.5086 BPB
- **Deadline**: 2026-04-30 23:59 UTC
- **Time remaining**: ~3 days

## Known Issues

- **RAILWAY_TOKEN not set** - Need to set environment variable for Railway deployment
- **Local data insufficient** - `data/fineweb_train.bin` is only 60KB (text data)
  - Real training must happen on Railway with proper data
