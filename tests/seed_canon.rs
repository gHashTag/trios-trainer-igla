//! tests/seed_canon.rs — Wave 29 PR-A: Canon #93 seed parser unit tests.
//!
//! Falsification criteria (R7):
//!   - SEED=47  → Ok(47)        [allowed canon seed]
//!   - SEED=43  → Err("forbidden") [forbidden under Canon #93]
//!   - SEED unset → Err("unset")
//!   - SEED=foobar → Err("parse")
//!
//! Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877

use std::sync::Mutex;
use trios_trainer::seed_canon::parse_seed;

/// Serialize SEED env mutation across parallel cargo test threads in this
/// integration-test binary. Without this, allowed_seed_NNN tests race on
/// the shared SEED env var (one set_var("144") collides with another
/// test's read of "123"). std-only, no extra deps.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// SEED=47 is in the allowed canon set → Ok(47).
#[test]
fn seed_47_ok() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::set_var("SEED", "47");
    assert_eq!(parse_seed(), Ok(47));
    std::env::remove_var("SEED");
}

/// SEED=89 is in the allowed canon set → Ok(89).
#[test]
fn seed_89_ok() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::set_var("SEED", "89");
    assert_eq!(parse_seed(), Ok(89));
    std::env::remove_var("SEED");
}

/// SEED=123 is in the allowed canon set → Ok(123).
#[test]
fn seed_123_ok() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::set_var("SEED", "123");
    assert_eq!(parse_seed(), Ok(123));
    std::env::remove_var("SEED");
}

/// SEED=144 is in the allowed canon set → Ok(144).
#[test]
fn seed_144_ok() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::set_var("SEED", "144");
    assert_eq!(parse_seed(), Ok(144));
    std::env::remove_var("SEED");
}

/// SEED=43 is forbidden under Canon #93 → Err containing "forbidden".
#[test]
fn seed_43_forbidden() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::set_var("SEED", "43");
    let err = parse_seed().expect_err("seed 43 should be rejected");
    assert!(
        err.contains("forbidden"),
        "expected error to contain 'forbidden', got: {err}"
    );
    std::env::remove_var("SEED");
}

/// SEED=42 is forbidden under Canon #93 → Err containing "forbidden".
#[test]
fn seed_42_forbidden() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::set_var("SEED", "42");
    let err = parse_seed().expect_err("seed 42 should be rejected");
    assert!(
        err.contains("forbidden"),
        "expected error to contain 'forbidden', got: {err}"
    );
    std::env::remove_var("SEED");
}

/// SEED=44 is forbidden under Canon #93 → Err containing "forbidden".
#[test]
fn seed_44_forbidden() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::set_var("SEED", "44");
    let err = parse_seed().expect_err("seed 44 should be rejected");
    assert!(err.contains("forbidden"));
    std::env::remove_var("SEED");
}

/// SEED=45 is forbidden under Canon #93 → Err containing "forbidden".
#[test]
fn seed_45_forbidden() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::set_var("SEED", "45");
    let err = parse_seed().expect_err("seed 45 should be rejected");
    assert!(err.contains("forbidden"));
    std::env::remove_var("SEED");
}

/// SEED env var unset → Err containing "unset".
#[test]
fn seed_unset_error() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::remove_var("SEED");
    let err = parse_seed().expect_err("unset SEED should return error");
    assert!(
        err.contains("unset"),
        "expected error to contain 'unset', got: {err}"
    );
}

/// SEED=foobar (non-numeric) → Err containing "parse".
#[test]
fn seed_parse_error() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::set_var("SEED", "foobar");
    let err = parse_seed().expect_err("non-numeric SEED should return error");
    assert!(
        err.contains("parse"),
        "expected error to contain 'parse', got: {err}"
    );
    std::env::remove_var("SEED");
}

/// SEED=0 (not in forbidden set) → Ok(0).
#[test]
fn seed_zero_ok() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::set_var("SEED", "0");
    assert_eq!(parse_seed(), Ok(0));
    std::env::remove_var("SEED");
}

// --- Wave-29 PR-A.1: GATE_FINAL_SEEDS sanity check ----------------------------
//
// The --sweep CLI flag in src/bin/trios-train.rs feeds
// `train_loop::GATE_FINAL_SEEDS` directly into the trainer loop. PR-A.1
// changed the constant from {43, 44, 45} (entirely forbidden) to the
// Canon #93 triple {47, 89, 123}. This regression guard prevents anyone
// from re-introducing a forbidden seed into the sweep set.

/// `GATE_FINAL_SEEDS` must contain ZERO members of the Canon #93
/// forbidden set {42, 43, 44, 45}.
#[test]
fn gate_final_seeds_no_forbidden_canon() {
    use trios_trainer::train_loop::GATE_FINAL_SEEDS;
    const FORBIDDEN: &[u64] = &[42, 43, 44, 45];
    for &s in GATE_FINAL_SEEDS {
        assert!(
            !FORBIDDEN.contains(&s),
            "GATE_FINAL_SEEDS contains forbidden Canon #93 seed {s}; \
             allowed canon: {{47, 89, 123, 144}}"
        );
    }
}

/// Every member of `GATE_FINAL_SEEDS` must be in the Canon #93 allowed
/// set `{47, 89, 123, 144}`. This is a stricter guard than the
/// not-forbidden test above — it forbids any operator-override seed
/// from sneaking into the sweep set.
#[test]
fn gate_final_seeds_only_allowed_canon() {
    use trios_trainer::train_loop::GATE_FINAL_SEEDS;
    const ALLOWED: &[u64] = &[47, 89, 123, 144];
    for &s in GATE_FINAL_SEEDS {
        assert!(
            ALLOWED.contains(&s),
            "GATE_FINAL_SEEDS seed {s} is not in Canon #93 allowed set \
             {{47, 89, 123, 144}}"
        );
    }
}

/// Sweep cardinality: 3-seed sweep, not 1 or 4. (PR-A.1 chose
/// {47, 89, 123} because 144 is reserved for the bridge canon.)
#[test]
fn gate_final_seeds_cardinality_three() {
    use trios_trainer::train_loop::GATE_FINAL_SEEDS;
    assert_eq!(
        GATE_FINAL_SEEDS.len(),
        3,
        "GATE_FINAL_SEEDS is a 3-seed sweep set; got {} entries",
        GATE_FINAL_SEEDS.len()
    );
}
