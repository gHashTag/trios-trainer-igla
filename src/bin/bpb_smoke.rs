//! `bpb_smoke` — minimum reproducible NEON-write probe for trios#444.
//!
//! Reads `NEON_DATABASE_URL` (or `TRIOS_NEON_DSN` / `DATABASE_URL` alias),
//! writes one row to `public.bpb_samples`, prints success/failure, exits.
//!
//! Acceptance for trios#444: this binary, given a working DSN, MUST produce
//! exactly one new row in NEON within 90 seconds, with no panics.
//!
//! Anchor: phi^2 + phi^-2 = 3.

fn main() {
    let canon = std::env::var("CANON_NAME").unwrap_or_else(|_| "bpb_smoke_test".to_string());
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

    eprintln!("[bpb_smoke] writing canon={canon} seed={seed} step={step} bpb={bpb}");
    trios_trainer::neon_writer::ensure_schema();
    trios_trainer::neon_writer::bpb_sample(&canon, seed, step, bpb, None);
    eprintln!("[bpb_smoke] done — verify with: SELECT * FROM bpb_samples WHERE canon_name='{canon}' ORDER BY ts DESC LIMIT 1;");
}
