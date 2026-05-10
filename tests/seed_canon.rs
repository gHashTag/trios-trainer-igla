//! tests/seed_canon.rs — Wave 29 PR-A: Canon #93 seed parser unit tests.
//!
//! Falsification criteria (R7):
//!   - SEED=47  → Ok(47)        [allowed canon seed]
//!   - SEED=43  → Err("forbidden") [forbidden under Canon #93]
//!   - SEED unset → Err("unset")
//!   - SEED=foobar → Err("parse")
//!
//! Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877

use trios_trainer::seed_canon::parse_seed;

/// SEED=47 is in the allowed canon set → Ok(47).
#[test]
fn seed_47_ok() {
    std::env::set_var("SEED", "47");
    assert_eq!(parse_seed(), Ok(47));
    std::env::remove_var("SEED");
}

/// SEED=89 is in the allowed canon set → Ok(89).
#[test]
fn seed_89_ok() {
    std::env::set_var("SEED", "89");
    assert_eq!(parse_seed(), Ok(89));
    std::env::remove_var("SEED");
}

/// SEED=123 is in the allowed canon set → Ok(123).
#[test]
fn seed_123_ok() {
    std::env::set_var("SEED", "123");
    assert_eq!(parse_seed(), Ok(123));
    std::env::remove_var("SEED");
}

/// SEED=144 is in the allowed canon set → Ok(144).
#[test]
fn seed_144_ok() {
    std::env::set_var("SEED", "144");
    assert_eq!(parse_seed(), Ok(144));
    std::env::remove_var("SEED");
}

/// SEED=43 is forbidden under Canon #93 → Err containing "forbidden".
#[test]
fn seed_43_forbidden() {
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
    std::env::set_var("SEED", "44");
    let err = parse_seed().expect_err("seed 44 should be rejected");
    assert!(err.contains("forbidden"));
    std::env::remove_var("SEED");
}

/// SEED=45 is forbidden under Canon #93 → Err containing "forbidden".
#[test]
fn seed_45_forbidden() {
    std::env::set_var("SEED", "45");
    let err = parse_seed().expect_err("seed 45 should be rejected");
    assert!(err.contains("forbidden"));
    std::env::remove_var("SEED");
}

/// SEED env var unset → Err containing "unset".
#[test]
fn seed_unset_error() {
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
    std::env::set_var("SEED", "0");
    assert_eq!(parse_seed(), Ok(0));
    std::env::remove_var("SEED");
}
