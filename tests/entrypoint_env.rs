//! Wave 33 — entrypoint env-var alias tests.
//! Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877
//!
//! Regression test for the Wave-29 silent failure: cron set `STEPS=200000`
//! on 52 services, but `entrypoint` only read `TRIOS_STEPS`, so every
//! trainer defaulted to `--steps=81000` and finished two cycles short
//! without any error log.
//!
//! 6 cases covering the 3 × 2 grid of (TRIOS_*, alias, both, neither) ×
//! (precedence, source-tag correctness).

use std::sync::Mutex;
use trios_trainer::entrypoint_env::{resolve_env_alias, ResolveSrc};

/// Serialise env-var mutations; std::env is process-global. Same lock
/// pattern as `tests/arch_config.rs`.
static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn trios_prefixed_takes_precedence() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::set_var("TRIOS_STEPS_TEST_A", "200000");
    std::env::set_var("STEPS_TEST_A", "81000");
    let (v, src) = resolve_env_alias("TRIOS_STEPS_TEST_A", "STEPS_TEST_A", "5000");
    assert_eq!(v, "200000", "TRIOS_* must win over alias");
    assert_eq!(src, ResolveSrc::TriosPrefixed);
    std::env::remove_var("TRIOS_STEPS_TEST_A");
    std::env::remove_var("STEPS_TEST_A");
}

#[test]
fn alias_used_when_trios_unset() {
    // Wave-29 regression: cron set the alias (STEPS), TRIOS_* was unset,
    // and the old code silently fell back to the default.
    let _g = ENV_LOCK.lock().unwrap();
    std::env::remove_var("TRIOS_STEPS_TEST_B");
    std::env::set_var("STEPS_TEST_B", "200000");
    let (v, src) = resolve_env_alias("TRIOS_STEPS_TEST_B", "STEPS_TEST_B", "81000");
    assert_eq!(v, "200000", "alias must be honoured when TRIOS_* unset");
    assert_eq!(src, ResolveSrc::Alias);
    std::env::remove_var("STEPS_TEST_B");
}

#[test]
fn default_used_when_neither_set() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::remove_var("TRIOS_STEPS_TEST_C");
    std::env::remove_var("STEPS_TEST_C");
    let (v, src) = resolve_env_alias("TRIOS_STEPS_TEST_C", "STEPS_TEST_C", "81000");
    assert_eq!(v, "81000");
    assert_eq!(src, ResolveSrc::Default);
}

#[test]
fn empty_string_in_trios_is_still_a_value() {
    // env::var returns Ok("") for KEY= (set-but-empty). We accept that as
    // the value so operators can deliberately blank out a knob.
    let _g = ENV_LOCK.lock().unwrap();
    std::env::set_var("TRIOS_STEPS_TEST_D", "");
    std::env::remove_var("STEPS_TEST_D");
    let (v, src) = resolve_env_alias("TRIOS_STEPS_TEST_D", "STEPS_TEST_D", "81000");
    assert_eq!(v, "");
    assert_eq!(src, ResolveSrc::TriosPrefixed);
    std::env::remove_var("TRIOS_STEPS_TEST_D");
}

#[test]
fn hidden_dim_alias_matches_arch_config_knob_name() {
    // Compatibility witness with the arch_config knob: HIDDEN_DIM is the
    // un-prefixed name that train_loop::run_single also reads. If we ever
    // rename this alias, train_loop must rename in lockstep.
    let _g = ENV_LOCK.lock().unwrap();
    std::env::remove_var("TRIOS_HIDDEN");
    std::env::set_var("HIDDEN_DIM", "1024");
    let (v, src) = resolve_env_alias("TRIOS_HIDDEN", "HIDDEN_DIM", "384");
    assert_eq!(v, "1024");
    assert_eq!(src, ResolveSrc::Alias);
    std::env::remove_var("HIDDEN_DIM");
}

#[test]
fn resolve_src_as_str_is_grep_friendly() {
    // The trace line uses these strings; CI greps them. Pin the values.
    assert_eq!(ResolveSrc::TriosPrefixed.as_str(), "TRIOS_*");
    assert_eq!(ResolveSrc::Alias.as_str(), "alias");
    assert_eq!(ResolveSrc::Default.as_str(), "default");
}
