# attic/ - quarantined source that was never compiled

**Nothing in this directory is built, tested, linted, or referenced by any claim
made by this repository.** It is not part of the `trios_trainer` library, it is
not part of any binary target, and no number printed by any file here has ever
been produced by a run of this crate.

The files were moved here (`git mv`, so history is preserved) on 2026-08-03 from
`src/`, where they had been tracked in the public repository and were
indistinguishable from live code to anyone who cloned it.


## The measurement that proved they never compiled

Three independent facts, each checkable:

1. **Not declared.** `src/lib.rs` declared 24 modules on 2026-08-03 (25 on
   2026-08-05; the crate is under active development and this number moves).
   None of them is any of these 16 files, at either date. A `.rs` file in `src/`
   that no `mod` statement names is not part of the crate; rustc never sees it.
2. **Not picked up as binaries.** `Cargo.toml` sets `autobins = false`, so cargo
   does not auto-discover targets, and no `[[bin]]` entry - live or commented -
   points at any of these files.
3. **Zero tests registered.** The 16 files carry 83 `#[test]` functions between
   them. Before the move:

   ```
   $ cargo test --release --lib -- --list > /tmp/list.txt
   $ wc -l < /tmp/list.txt
   649                       # last line reads: 647 tests, 0 benchmarks

   $ for t in test_cosine_lr_warmup test_kill_threshold test_config_default \
              test_bpb_from_loss test_phase_display; do
   >   printf '%-24s %s\n' "$t" "$(grep -c $t /tmp/list.txt)"
   > done
   test_cosine_lr_warmup    0
   test_kill_threshold      0
   test_config_default      0
   test_bpb_from_loss       0
   test_phase_display       0
   ```

   Every one of the 83 test names matches 0 lines of the test listing. The
   assertions in these files have never been evaluated.

The move was verified to be behaviour-preserving: `cargo build --release`
succeeds and `cargo test --release --lib` reports the same 647 tests after the
move as before it. If that count had changed, something would have referenced
these files and the move would have been wrong.

Total quarantined: 6221 lines across 16 files.


### Re-verified 2026-08-05

The absolute test count is not a stable check - other work in this tree adds
tests, and the same command now reports 686. The count is therefore the weak
form of the argument. The strong form, re-run on 2026-08-05 and independent of
how many tests the crate has:

```
$ cargo build --release                              # Finished `release` profile
$ cargo test --release --lib | tail -1
test result: ok. 686 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

# extract every #[test] fn name from the 16 files here (83 attributes,
# 74 distinct names) and look for each one in the live test listing:
$ cargo test --release --lib -- --list > /tmp/list.txt   # 686 tests, 0 benchmarks
$ ... grep each name ...
distinct test fn names extracted: 74
quarantined test names appearing in the live test listing: 0
```

Also re-confirmed on that date: no `mod <name>;`, `#[path = ...]` or `include!`
anywhere in `src/` or `tests/` names any of these 16 files (the only `#[path]`
in the tree points at `src/bin/ckpt_replay.rs`, which is live), and no `path =`
line in `Cargo.toml` - live or commented - points at a file that is not there.


## Note: this directory holds more than these 16 files

A second quarantine pass on 2026-08-03 moved additional never-built files into
this same directory from `src/bin/`: `attn_train.rs`, `bench_cpu.rs`,
`lr_calibration.rs`, `ptq_eval.rs`, `train_cpu.rs`, `transformer_train.rs`,
`tjepa_modules/`, `trinity_3k_fineweb_train.rs`, `trinity_3k_simple_train.rs`,
`trinity_3k_tinyshakespeare.rs` and `trinity_tournament.rs`. Each is documented
where its target used to be declared, in the `[[bin]]` section of `Cargo.toml`.
Most of them failed to build precisely because they depended on the 16 library
files quarantined here. The file table below covers only the 16; the opening
sentence of this README - nothing here is built, tested or referenced by any
claim - covers everything in the directory.


## WARNING - fabricated output. Never quote a number from this directory.

These files print strings that read exactly like measurements but are literal
constants. Two are named explicitly because they are the worst, and because a
reader skimming `src/` would have had no way to tell:

### `attic/pipeline.rs` - a loss of 0.000000 from zero steps

```rust
println!("[Phase 1] Loading NCA checkpoint: {}", config.nca_checkpoint);
let nca_loaded = false;                                    // nothing was loaded
...
println!("[Phase 2] Starting JEPA training (20K steps)");
let jepa_final_loss = 0.0f64;                              // hardcoded
println!("[Phase 2] JEPA complete: loss={:.6}", jepa_final_loss);
```

No checkpoint is read. No JEPA step is run. `jepa_final_loss` is the literal
`0.0` - the best attainable loss - announced as the outcome of "20K steps", and
then stored into the returned result struct as if it were an observation.

### `attic/transformer_trainer.rs` - an "IMPROVEMENT" percentage against a constant

The file states its own nature in a comment inside the training loop:

```rust
// Simplified: no actual gradient updates in this minimal version
```

It trains on uniform-random synthetic tokens, then prints:

```rust
let baseline_bpb = 2.5329;                                 // hardcoded constant
let improvement = ((baseline_bpb - best_bpb) / baseline_bpb) * 100.0;
println!("IMPROVEMENT: {:.2}% better than N-gram baseline ({:.4} vs {:.4})", ...);
```

Three separate defects compound here:

- there are no gradient updates, so `best_bpb` cannot reflect learning;
- the data is random, so there is nothing to learn from;
- `bpb_from_loss(loss) = loss / LN_2` converts a **token-level** cross-entropy
  into bits per **token**, then compares it to a **byte-level** baseline number.
  The two quantities have different denominators and are not comparable. This is
  not a rounding disagreement; the ratio is the average bytes per token.

Neither number is traceable to a computation. Both must be treated as void.

The same caution applies to every other file here. These are the two documented
cases, not an exhaustive audit - the remaining 14 files were not reviewed
line-by-line, because unbuilt code does not need to be correct, it needs to be
unmistakably marked as unbuilt. That is what this directory is for.


## If you want one of these files back

Do not un-comment a `[[bin]]` entry and do not copy code out of here into `src/`
by hand. The correct move is:

1. `git mv attic/<file>.rs src/<file>.rs`
2. add `pub mod <file>;` to `src/lib.rs`
3. run `cargo build --release && cargo test --release --lib`

and let the compiler and the file's own 83 previously-unevaluated tests speak.
Expect failures: this code has never been type-checked against the current tree.
Any print statement that reports a metric must be re-derived from a real
computation before the file is allowed back, or deleted.


## Files

| File | Lines | `#[test]` fns |
| --- | ---: | ---: |
| attention.rs | 1215 | 15 |
| trinity_3k.rs | 1095 | 6 |
| bench.rs | 566 | 7 |
| pipeline.rs | 552 | 8 |
| transformer.rs | 503 | 9 |
| backward.rs | 420 | 9 |
| real_igla_model.rs | 394 | 4 |
| forward.rs | 325 | 8 |
| train_model.rs | 302 | 2 |
| transformer_trainer.rs | 289 | 4 |
| real_igla_trainer.rs | 159 | 0 |
| phi_ortho_init.rs | 126 | 2 |
| swa_phi.rs | 108 | 3 |
| ortho_init_baseline.rs | 68 | 1 |
| sliding_eval.rs | 56 | 3 |
| residual_mix.rs | 43 | 2 |
| **total** | **6221** | **83** |
