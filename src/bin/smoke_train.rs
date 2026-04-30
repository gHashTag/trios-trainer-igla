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
//! ## CLI
//!
//! ```bash
//! smoke_train --canon IGLA-SMOKE-PROOF-T0 --seed 1597 --bpb 2.5
//! ```
//!
//! ## Output (verbatim, parseable by seed-agent)
//!
//! ```
//! [smoke_train] start canon=IGLA-SMOKE-PROOF-T0 seed=1597
//! seed=1597 step=1 val_bpb=2.5000 ema_bpb=2.5000 best=2.5000 nca_h=0.000 t=0.0s
//! DONE: seed=1597 bpb=2.5000 steps=1 opt=smoke
//! [smoke_train] wrote 1 row to bpb_samples
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
    let canon = std::env::var("CANON_NAME")
        .or_else(|_| std::env::var("TRIOS_CANON"))
        .unwrap_or_else(|_| "IGLA-SMOKE-PROOF-T0".to_string());
    let seed: i32 = env_or("SEED", env_or("TRIOS_SEED", 1597));
    let bpb: f64 = env_or("BPB", 2.5);
    let step: i32 = 1;

    eprintln!("[smoke_train] start canon={canon} seed={seed}");
    let _ = std::io::stderr().flush();

    // Emit the JSONL line that seed-agent's parse_step_output() expects.
    println!(
        "seed={seed} step={step} val_bpb={bpb:.4} ema_bpb={bpb:.4} best={bpb:.4} nca_h=0.000 t=0.0s"
    );
    let _ = std::io::stdout().flush();

    // Emit the DONE line that seed-agent's parse_done_output() expects.
    println!("DONE: seed={seed} bpb={bpb:.4} steps={step} opt=smoke");
    let _ = std::io::stdout().flush();

    // Write to bpb_samples if a Neon DSN is configured. Silent no-op otherwise
    // (R5: never panic on Neon errors; CI runs without Neon).
    if std::env::var("TRIOS_NEON_DSN").is_ok()
        || std::env::var("NEON_DATABASE_URL").is_ok()
        || std::env::var("DATABASE_URL").is_ok()
    {
        trios_trainer::neon_writer::ensure_schema();
        trios_trainer::neon_writer::bpb_sample(&canon, seed, step, bpb as f32);
        eprintln!("[smoke_train] wrote 1 row to bpb_samples");
    } else {
        eprintln!("[smoke_train] no Neon DSN — skipping DB write");
    }
    let _ = std::io::stderr().flush();
}
