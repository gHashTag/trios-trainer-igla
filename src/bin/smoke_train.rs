//! `smoke_train` — minimal end-to-end proof that the IGLA RACE pipeline is
//! alive: trainer subprocess emits JSONL on stdout, writes one row to
//! `public.bpb_samples`, and exits cleanly with code 0.
//!
//! Designed to run in <60 s on a single CPU thread inside CI and inside the
//! seed-agent runtime image. **Synthetic data only** — does not require
//! `data/tiny_shakespeare.txt`. **One step** — does not exercise the optimizer.
//!
//! ## Why this exists
//!
//! The 60/60 IGLA-RAILWAY-CPU-T-7H wave failed with
//! `prune_reason='trainer produced zero steps (exited without JSONL output)'`
//! because (a) `trios-train` did not accept `--ctx 12` (fixed in
//! [trios-trainer-igla#56](https://github.com/gHashTag/trios-trainer-igla/pull/56))
//! and (b) `println!` lines were buffered inside the trainer's stdout
//! BufWriter and never flushed before the process exited (fixed in this PR).
//!
//! `smoke_train` proves the fix end-to-end **before** queueing real training
//! waves on Railway. Green CI ⇒ live cycle.
//!
//! ## The BPB here is typed, not measured
//!
//! `BPB` is an operator-supplied env var with a 2.5 default; this binary runs
//! no optimizer and evaluates nothing. Under an unconstrained `CANON_NAME` that
//! number landed in the same namespace as a trained result. Every canon is now
//! forced through `neon_writer::smoke_canon_name`, which pins the reserved
//! `SMOKE-` prefix, and `bpb_sample` refuses to route a `SMOKE-` canon to
//! `ssot.bpb_samples`. The stdout step line carries `synthetic=1` and the DONE
//! line keeps `opt=smoke`.
//!
//! ## CLI
//!
//! ```bash
//! CANON_NAME=IGLA-SMOKE-PROOF-T0 SEED=1597 BPB=2.5 smoke_train
//! ```
//!
//! ## Output (verbatim, parseable by seed-agent)
//!
//! ```text
//! [smoke_train] start canon=SMOKE-IGLA-SMOKE-PROOF-T0 seed=1597
//! seed=1597 step=1 val_bpb=2.5000 ema_bpb=2.5000 best=2.5000 nca_h=0.000 t=0.0s synthetic=1
//! DONE: seed=1597 bpb=2.5000 steps=1 opt=smoke
//! [smoke_train] ledger=skipped-no-dsn
//! ```
//!
//! Anchor: `phi^2 + phi^-2 = 3`. Refs: trios-trainer-igla#57, trios#445.

use std::io::Write;

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    // Parse minimal args (no clap to keep dependency footprint zero).
    //
    // S9: whatever the operator supplies, the reserved SMOKE- prefix is
    // applied. There is no spelling of CANON_NAME or TRIOS_CANON that puts this
    // hand-typed BPB into the trained-result namespace.
    let raw = std::env::var("CANON_NAME")
        .or_else(|_| std::env::var("TRIOS_CANON"))
        .unwrap_or_else(|_| "IGLA-SMOKE-PROOF-T0".to_string());
    let canon = trios_trainer::neon_writer::smoke_canon_name(&raw);
    let seed: i32 = env_or("SEED", env_or("TRIOS_SEED", 1597));
    let bpb: f64 = env_or("BPB", 2.5);
    let step: i32 = 1;

    eprintln!("[smoke_train] start canon={canon} seed={seed}");
    if canon != raw {
        eprintln!("[smoke_train] canon '{raw}' pinned to the reserved smoke namespace");
    }
    eprintln!("[smoke_train] bpb={bpb:.4} is OPERATOR-SUPPLIED, not measured");
    let _ = std::io::stderr().flush();

    // Emit the JSONL line that seed-agent's parse_step_output() expects.
    // `synthetic=1` is appended, not inserted: the seed-agent regex and
    // .github/workflows/smoke.yml both anchor on the prefix of this line.
    println!(
        "seed={seed} step={step} val_bpb={bpb:.4} ema_bpb={bpb:.4} best={bpb:.4} nca_h=0.000 t=0.0s synthetic=1"
    );
    let _ = std::io::stdout().flush();

    // Emit the DONE line that seed-agent's parse_done_output() expects. It is
    // anchored at both ends, so `opt=smoke` stays the marker there.
    println!("DONE: seed={seed} bpb={bpb:.4} steps={step} opt=smoke");
    let _ = std::io::stdout().flush();

    // Write to bpb_samples if a Neon DSN is configured (R5: never panic on Neon
    // errors; CI runs without Neon). The outcome printed is the writer's own,
    // so this can no longer announce a row it did not write.
    trios_trainer::neon_writer::ensure_schema();
    let outcome = trios_trainer::neon_writer::bpb_sample(&canon, seed, step, bpb as f32, None);
    eprintln!("[smoke_train] ledger={}", outcome.as_str());

    let code = trios_trainer::neon_writer::ledger_exit_code();
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    std::process::exit(code);
}
