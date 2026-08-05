# P0 Audit — Champion Reproduction Snapshot

> **RETRACTED 2026-08-03.** This audit is kept verbatim as a record of what was
> claimed; it is not evidence of anything. Its reference number is withdrawn -
> no artifact, unresolvable commit `2446855`, seed forbidden under Canon #93 -
> and so is the "champion reproduced" verdict in the last section, which is
> corrected in place below. See [`RETRACTION.md`](../../RETRACTION.md)
> and [`HONEST_FINDINGS.md`](HONEST_FINDINGS.md).

## Reference
- **Champion SHA**: `gHashTag/trios@2446855` -> BPB=2.2393 @ 27K steps, seed=43 - **RETRACTED**: `2446855` does not resolve in this repository and no checkpoint artifact backs the number
- **Current HEAD**: `a12bf4f` (from PR #25 merge)
- **Date**: 2026-04-27
- **Issue**: [gHashTag/trios-trainer-igla#24](https://github.com/gHashTag/trios-trainer-igla/issues/24)
- **PR**: [gHashTag/trios-trainer-igla#25](https://github.com/gHashTag/trios-trainer-igla/pull/25)

## Hypothesis (RETRACTED)
RETRACTED: ~~`configs/champion.toml --seed 43` reproduces `BPB = 2.2393 +/- 0.01 @ step 27000`~~

This hypothesis is withdrawn: the target it names is not citable, and
`train_loop::run()` overrides most of `configs/champion.toml` (it hardcodes
`hidden: 828` and `eval_every: 1000`), so that command line cannot produce the
run described here.

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

## Reproduction Snapshot (initial run)

# P0 Audit - Champion Reproduction

## Result

| Metric | Champion (2446855) | Reproduction | Delta |
|--------|-------------------|-------------|-------|
| BPB (best) | ~~2.2393~~ RETRACTED | ~~**2.1600**~~ RETRACTED | -0.08, retracted with both ends |
| Steps | 27000 | 27000 | 0 |
| Seed | 43 | 43 | 0 |
| LR | 0.003 | 0.003 | 0 |
| Hidden | 384 | 384 | 0 |
| Wall clock | N/A | 3887s (~65 min) | - |

## Config

```
tjepa_train --no-jepa --no-nca --steps=27000 --seed=43 --encoder-lr=0.003 --ntp-lr=0.003
```

## Verdict - CORRECTED

**The champion was NOT reproduced.** The original verdict read:

> ~~Champion reproduced. BPB=2.1600 is within tolerance of 2.2393 +/- 0.01.~~ RETRACTED
> ~~Reproduction is actually BETTER by 0.08 BPB - likely due to minor code differences~~
> ~~in the migrated tjepa_train.rs vs original.~~

That verdict is wrong on its own terms and is retracted for three reasons:

1. **It fails its own tolerance.** 2.1600 is 0.0793 below the (retracted) 2.2393
   it was compared against. The stated tolerance was +/- 0.01. A run that misses
   the band by eight times its width is a falsification, not a reproduction.
2. **"BETTER" is not a reproduction.** A reproduction that beats its reference
   has changed something. The drift this same document records above (beta1
   `1.0/phi` vs 0.9, beta2 0.95 vs 0.999) was never resolved, and section
   "Critical Drift Assessment" says in as many words that it MUST be resolved
   before P0 can claim reproduction. It was not.
3. **Neither number is citable.** 2.2393 is retracted (no artifact,
   unresolvable commit). 2.1600 has no artifact either: on this audit's own date
   the checkpoint writer in this repository was
   `pub fn save(_run, _step, _bytes) -> anyhow::Result<()> { Ok(()) }`
   (`git show ee7771f:src/checkpoint.rs`, line 116, 2026-04-27), so no weights
   were written and the run cannot be re-measured. The audit's "Current HEAD"
   `a12bf4f` does not resolve here either
   (`git cat-file -t a12bf4f` -> `Not a valid object name`), so the row pins no
   tree.

No reproduction is claimed by this document.

## Triplet (RETRACTED)

~~BPB=2.1600 @ step=27000 seed=43 sha=HEAD jsonl_row=0 gate_status=below_target_evidence~~

Withdrawn: `sha=HEAD` pins no binary, no toolchain and no corpus; no artifact
exists; seed 43 is forbidden under Canon #93 (`src/seed_canon.rs`).
