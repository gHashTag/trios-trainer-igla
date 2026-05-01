# PR #71 was a REGRESSION of PR #70 — Full Revert + R1 Cleanup Needed

## Summary

PR #71 (commit 956c6c9) claimed to fix Bug E/F/G but actually:
1. **REVERTED** Bug A/B/C fixes from PR #70 (commit 9efdd9a)
2. **DID NOT IMPLEMENTE** Bug E fix (final_bpb/final_step writeback)
3. **ADDED** R1 violation .sh files (later removed, but shows intent drift)

This is a deploy regression P0. Current HEAD state has ALL of Bug A/B/C/E present.

## Root Cause Analysis

### What PR #70 (9efdd9a) Added ✅

**Bug A**: `use tokio_postgres_rustls::MakeRustlsConnect` (TLS fix for Neon)
```rust
// src/bin/scarab.rs:17
use tokio_postgres_rustls::MakeRustlsConnect;
```

**Bug B**: `train_path` and `val_path` fields in TrainerSpec
```rust
// src/bin/scarab.rs:35-38
/// Override train data path (default: /work/data/tiny_shakespeare.txt)
train_path: Option<String>,
/// Override val data path (default: /work/data/tiny_shakespeare_val.txt)
val_path: Option<String>,
```

**Bug C**: --ctx, --train-data, --val-data args passed to trios-train
```rust
// src/bin/scarab.rs:176-189
cmd.args([
    "--ctx",
    &ctx,
    "--train-data",
    &train_path,
    "--val-data",
    &val_path,
    // ... other args
])
```

### What PR #71 (956c6c9) Did ❌

**Bug A REVERTED**:
```rust
// -use tokio_postgres_rustls::MakeRustlsConnect;
+use tokio_postgres::NoTls;
```

**Bug B REVERTED**:
```diff
-    /// Override train data path (default: /work/data/tiny_shakespeare.txt)
-    train_path: Option<String>,
-    /// Override val data path (default: /work/data/tiny_shakespeare_val.txt)
-    val_path: Option<String>,
```

**Bug C REVERTED**: --train-data, --val-data, --ctx removed from args

**Bug E NOT IMPLEMENTED**:
PR #71 commit message claimed:
> Bug E (P0): scarab run_strategy() never wrote final_bpb/final_step back to strategy_queue.

But `git grep -E "final_bpb|final_step" 956c6c9:src/bin/scarab.rs` returns 0 matches. The writeback logic is NOT in the code.

### R1 Violation

PR #71 added two .sh files (violating Rust-only policy):
- `entrypoint.sh` (15 lines)
- `scripts/entrypoint_rt.sh` (15 lines)

These were later removed in a follow-up commit, but they should never have been added. f60ebb1 already purged .sh files.

## Impact

- **Fleet**: Any container built from 956c6c9 or later has NoTls (can't connect to Neon)
- **Data**: No corpus paths passed to trios-train → hardcoded defaults used
- **Metrics**: final_bpb/final_step never written to strategy_queue
- **Debugging**: Harder to diagnose issues due to missing args visibility

## Required Action

1. **Revert PR #71** completely (restores PR #70 state)
2. **Implement clean Bug E/F/G fix** WITHOUT removing Bug A/B/C
3. **Add contract tests** to prevent future regression:
   - `tests/scarab_args_pass_through.rs` (Bug X1)
   - `tests/scarab_writeback_contract.rs` (Bug X2/X3)
   - `tests/neon_writer_failure_visibility.rs` (Bug F)
   - `tests/entrypoint_corpus_args.rs` (Bug B root)
   - `tests/canon_name_seed_range.rs` (Tripwire #98)
4. **Migration 0003**: Add final_bpb/final_step columns (prepared)

## Test Evidence

Probe 2033 confirmed: final_step=NULL, final_bpb=NULL despite status='done'.
This directly matches Bug E (writeback missing) and explains why bpb_samples query can't work.

## References

- PR #70: 9efdd9a "fix(scarab): Bug A+B+C+D"
- PR #71: 956c6c9 "fix(scarab+neon_writer): Bug E/F/G"
- L1 (Rust-only): forbids .sh files in repository
- Probe 2033: evidence of final_bpb/final_step missing
