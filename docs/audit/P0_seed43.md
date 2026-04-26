# P0 Audit — Champion Reproduction Snapshot

## Reference
- **Champion SHA**: `gHashTag/trios@2446855` → BPB=2.2393 @ 27K steps, seed=43
- **Current HEAD**: `a12bf4f` (from PR #25 merge)
- **Date**: 2026-04-27
- **Issue**: [gHashTag/trios-trainer-igla#24](https://github.com/gHashTag/trios-trainer-igla/issues/24)
- **PR**: [gHashTag/trios-trainer-igla#25](https://github.com/gHashTag/trios-trainer-igla/pull/25)

## Hypothesis
`configs/champion.toml --seed 43` reproduces `BPB = 2.2393 +/- 0.01 @ step 27000`

## Drift Analysis: `gHashTag/trios@2446855` → `trios-trainer-igla`

### Architecture Comparison

| Aspect | Reference (2446855) | Current (train_loop.rs) | Note |
|---------|------------------------|------------------------|------|
| **File** | `crates/trios-train-cpu/src/bin/tjepa_train.rs` | `src/train_loop.rs` | File relocated & modularized |
| **AdamW beta1** | 0.9 | `1.0/phi` ≈ 0.618 | **DRIFT** — phi-based beta1 |
| **AdamW beta2** | 0.999 | 0.95 | **DRIFT** — beta2 changed |
| **Cosine LR** | `base_lr * 0.5 * (1 + cos(pi*p))` | Same formula, wrapped in module | Same |
| **Layer norm** | Simple inline implementation | Re-exported from module | Same math |
| **Softmax** | Simple inline implementation | Re-exported from module | Same math |
| **Config source** | Hardcoded constants | TOML-based `TrainConfig` | **REFACTOR** — flexibility gain |

### Critical Drift Assessment

**MAJOR DRIFT: AdamW hyperparameters**

```
Reference (2446855):
  beta1 = 0.9
  beta2 = 0.999

Current (champion.toml):
  beta1 = 0.9   [overridden in config]
  beta2 = 0.95   [CHANGED in train_loop.rs]
```

The `champion.toml` specifies `beta1 = 0.9, beta2 = 0.95`, but **reference code** uses `beta2 = 0.999`.

**Action Required**: This drift MUST be resolved before P0 can claim reproduction.

## Test Status

| Test | Status | Notes |
|------|--------|-------|
| `champion_config_loads_and_validates` | ✅ PASS | Config loads correctly |
| `champion_model_config_matches_spec` | ✅ PASS | dim=256, layers=2, heads=4 |
| `champion_optimizer_is_adamw_phi` | ✅ PASS | lr=0.004, schedule=phi |
| `champion_objective_pure_ce` | ✅ PASS | w_ce=1.0, w_jepa=0.0, w_nca=0.0 |
| `champion_inv8_lr_in_phi_band` | ✅ PASS | lr=0.004 ∈ [0.001, 0.01] |
| `champion_bpb_reproduction_full_run` | ⏸️ BLOCKED | FineWeb data not available |

## Full Run Requirements

To complete P0:

1. **FineWeb data required** at:
   - `/data/fineweb_train.bin`
   - `/data/fineweb_val.bin`

2. **Resolve AdamW beta2 drift**:
   - Option A: Update `champion.toml` to `beta2 = 0.999`
   - Option B: Verify reference actually used `beta2 = 0.95`

3. **Run**:
   ```bash
   cargo test --release champion_bpb_reproduction_full_run -- --ignored
   ```

4. **Capture metrics**:
   - Wall-clock time
   - Memory profile
   - Final BPB ∈ [2.229, 2.249]

## Exit Criterion

When BPB ∈ [2.229, 2.249] @ step=27000 with seed=43:
- Emit ledger row to `assertions/seed_results.jsonl`
- Row passes R8 (step ≥ 4000)
- Row passes R9 (embargo check)

## Falsification

If BPB drift > 0.05 (i.e., BPB ∉ [2.214, 2.264]):
- Bisect against `gHashTag/trios@2446855`
- Identify source of divergence
- Fix before proceeding to P1
