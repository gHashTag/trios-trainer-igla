//! `bpb_smoke` — minimum reproducible NEON-write probe for trios#444.
//!
//! Reads `NEON_DATABASE_URL` (or `TRIOS_NEON_DSN` / `DATABASE_URL` alias),
//! writes one row to `public.bpb_samples`, prints the writer's own verdict,
//! exits.
//!
//! Acceptance for trios#444: this binary, given a working DSN, MUST produce
//! exactly one new row in NEON within 90 seconds, with no panics.
//!
//! ## The BPB here is typed, not measured
//!
//! `BPB` is an operator-supplied env var with a 2.5 default; nothing in this
//! binary trains or evaluates anything. Under an unconstrained `CANON_NAME`
//! that number was indistinguishable from a trained result in the leaderboard's
//! own table. Every canon this binary can emit is now forced through
//! `neon_writer::smoke_canon_name`, which pins the reserved `SMOKE-` prefix,
//! and `bpb_sample` refuses to route a `SMOKE-` canon to `ssot.bpb_samples`.
//!
//! Anchor: phi^2 + phi^-2 = 3.

use std::io::Write;

fn main() {
    // S9: whatever the operator supplies, the reserved prefix is applied. There
    // is no spelling of CANON_NAME that reaches the trained-result namespace.
    let raw = std::env::var("CANON_NAME").unwrap_or_else(|_| "bpb_smoke_test".to_string());
    let canon = trios_trainer::neon_writer::smoke_canon_name(&raw);
    if canon != raw {
        eprintln!("[bpb_smoke] canon '{raw}' pinned to the reserved smoke namespace: '{canon}'");
    }
    let seed: i32 = std::env::var("SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1597);
    let step: i32 = std::env::var("STEP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let bpb: f32 = std::env::var("BPB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2.5);

    // This binary's entire acceptance contract is "given a working DSN, produce
    // exactly one new row". With no DSN it proves nothing, so it must not exit
    // 0 and let a supervisor record the probe as green.
    if trios_trainer::neon_writer::db_status() == trios_trainer::neon_writer::DbStatus::NoDsn {
        eprintln!(
            "[bpb_smoke] no DSN configured (DATABASE_URL / NEON_DATABASE_URL / \
             TRIOS_NEON_DSN / TRIOS_DATABASE_URL all unset). This probe exists to \
             prove a write reaches the ledger; without a DSN it proves nothing."
        );
        let _ = std::io::stderr().flush();
        std::process::exit(2);
    }

    eprintln!(
        "[bpb_smoke] writing canon={canon} seed={seed} step={step} \
         value=OPERATOR-SUPPLIED-NOT-MEASURED"
    );
    trios_trainer::neon_writer::ensure_schema();
    let outcome = trios_trainer::neon_writer::bpb_sample(&canon, seed, step, bpb, None);
    eprintln!(
        "[bpb_smoke] ledger={} — verify with: SELECT * FROM public.bpb_samples \
         WHERE canon_name='{canon}' ORDER BY ts DESC LIMIT 1;",
        outcome.as_str()
    );

    let code = trios_trainer::neon_writer::ledger_exit_code();
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    std::process::exit(code);
}
