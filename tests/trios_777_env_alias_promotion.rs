//! Regression test for gHashTag/trios#777 — fake-fallback bug where 47
//! distinct canon_name configs (ranger/adafactor/adamw/adopt/demo on
//! binary16) collapsed to bit-identical bpb=2.9504425525665283 at step
//! 80000 in `ssot.bpb_samples` because every trainer silently fell back to
//! the clap defaults (adamw / None→f32 / hidden=828).
//!
//! Root cause: `gHashTag/trios-railway/.github/workflows/wave-a-dispatch.yml`
//! sets env vars with the un-prefixed names `OPTIMIZER`, `FORMAT`, `HIDDEN`,
//! but `src/bin/trios-train.rs::Cli` only reads `TRIOS_OPTIMIZER`,
//! `TRIOS_FORMAT_TYPE`, `TRIOS_HIDDEN` via `#[arg(env = "...")]`.
//!
//! Fix: `main()` promotes the un-prefixed aliases into the canonical
//! `TRIOS_*` names BEFORE `Cli::parse()` so clap picks them up. Resolution
//! order is `TRIOS_*` > alias > clap default, mirroring the Wave-33
//! `entrypoint_env::resolve_env_alias` pattern (`STEPS`/`SEED`).
//!
//! This test asserts the promotion logic in isolation: given a fresh env
//! where only the un-prefixed aliases are set, the canonical names must end
//! up populated with the alias values.
//!
//! Anchor: phi^2 + phi^-2 = 3 · DOI 10.5281/zenodo.19227877.

use std::sync::Mutex;

/// Env mutations across tests would race; serialize via a global mutex.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Mirror of the production promotion loop in `trios-train.rs::main()`.
/// Kept verbatim so the test exercises the exact same algorithm; if the
/// production loop changes, this copy must be updated in lock-step.
fn promote_unprefixed_aliases() {
    for (canonical, alias) in [
        ("TRIOS_OPTIMIZER", "OPTIMIZER"),
        ("TRIOS_FORMAT_TYPE", "FORMAT"),
        ("TRIOS_HIDDEN", "HIDDEN"),
    ] {
        if std::env::var(canonical).is_err() {
            if let Ok(v) = std::env::var(alias) {
                if !v.is_empty() {
                    std::env::set_var(canonical, v);
                }
            }
        }
    }
}

fn clear_all() {
    for k in [
        "OPTIMIZER",
        "FORMAT",
        "HIDDEN",
        "TRIOS_OPTIMIZER",
        "TRIOS_FORMAT_TYPE",
        "TRIOS_HIDDEN",
    ] {
        std::env::remove_var(k);
    }
}

#[test]
fn alias_only_gets_promoted_to_canonical() {
    let _g = ENV_LOCK.lock().unwrap();
    clear_all();

    // Wave-A dispatch shape: only un-prefixed aliases set, no TRIOS_*.
    std::env::set_var("OPTIMIZER", "muon");
    std::env::set_var("FORMAT", "gf16");
    std::env::set_var("HIDDEN", "128");

    promote_unprefixed_aliases();

    assert_eq!(std::env::var("TRIOS_OPTIMIZER").as_deref(), Ok("muon"));
    assert_eq!(std::env::var("TRIOS_FORMAT_TYPE").as_deref(), Ok("gf16"));
    assert_eq!(std::env::var("TRIOS_HIDDEN").as_deref(), Ok("128"));

    clear_all();
}

#[test]
fn canonical_wins_over_alias() {
    let _g = ENV_LOCK.lock().unwrap();
    clear_all();

    std::env::set_var("OPTIMIZER", "muon");
    std::env::set_var("TRIOS_OPTIMIZER", "adamw");
    std::env::set_var("FORMAT", "gf16");
    std::env::set_var("TRIOS_FORMAT_TYPE", "fp16");
    std::env::set_var("HIDDEN", "128");
    std::env::set_var("TRIOS_HIDDEN", "828");

    promote_unprefixed_aliases();

    // Existing TRIOS_* values must not be clobbered by aliases.
    assert_eq!(std::env::var("TRIOS_OPTIMIZER").as_deref(), Ok("adamw"));
    assert_eq!(std::env::var("TRIOS_FORMAT_TYPE").as_deref(), Ok("fp16"));
    assert_eq!(std::env::var("TRIOS_HIDDEN").as_deref(), Ok("828"));

    clear_all();
}

#[test]
fn empty_alias_does_not_overwrite() {
    let _g = ENV_LOCK.lock().unwrap();
    clear_all();

    std::env::set_var("OPTIMIZER", "");
    std::env::set_var("FORMAT", "");
    std::env::set_var("HIDDEN", "");

    promote_unprefixed_aliases();

    // Empty alias values must NOT promote — otherwise we'd set
    // `TRIOS_OPTIMIZER=""` and clap might pick the empty string instead
    // of its declared default.
    assert!(std::env::var("TRIOS_OPTIMIZER").is_err());
    assert!(std::env::var("TRIOS_FORMAT_TYPE").is_err());
    assert!(std::env::var("TRIOS_HIDDEN").is_err());

    clear_all();
}

#[test]
fn no_env_no_promotion() {
    let _g = ENV_LOCK.lock().unwrap();
    clear_all();

    promote_unprefixed_aliases();

    assert!(std::env::var("TRIOS_OPTIMIZER").is_err());
    assert!(std::env::var("TRIOS_FORMAT_TYPE").is_err());
    assert!(std::env::var("TRIOS_HIDDEN").is_err());
}
