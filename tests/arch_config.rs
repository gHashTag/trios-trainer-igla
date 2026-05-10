//! Wave 31 PR-B — arch_config tests.
//! Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877
//!
//! 9 cases:
//!   HIDDEN_DIM=1024      → Ok(1024)
//!   HIDDEN_DIM=128       → Err containing "forbidden"
//!   HIDDEN_DIM unset     → Ok(384)   (default)
//!   NUM_ATTN_LAYERS=4    → Ok(4)
//!   NUM_ATTN_LAYERS=0    → Err
//!   GF16_ENABLED=true    → Ok(true)
//!   GF16_ENABLED=false   → Ok(false)
//!   GF16_ENABLED=garbage → Err
//!   all-defaults         → (h=384, n=1, gf16=false)

use std::sync::Mutex;
use trios_trainer::arch_config::{
    parse_arch_config, parse_gf16_enabled, parse_hidden_dim, parse_num_attn_layers, ArchConfig,
};

/// Serialise env-var mutations; std::env is process-global.
static ENV_LOCK: Mutex<()> = Mutex::new(());

// ── HIDDEN_DIM ────────────────────────────────────────────────────────────────

#[test]
fn hidden_dim_1024_allowed() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::set_var("HIDDEN_DIM", "1024");
    assert_eq!(parse_hidden_dim(), Ok(1024));
    std::env::remove_var("HIDDEN_DIM");
}

#[test]
fn hidden_dim_128_forbidden() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::set_var("HIDDEN_DIM", "128");
    let err = parse_hidden_dim().unwrap_err();
    assert!(
        err.contains("forbidden"),
        "expected 'forbidden' in error: {err}"
    );
    std::env::remove_var("HIDDEN_DIM");
}

#[test]
fn hidden_dim_default_384() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::remove_var("HIDDEN_DIM");
    assert_eq!(parse_hidden_dim(), Ok(384));
}

// ── NUM_ATTN_LAYERS ───────────────────────────────────────────────────────────

#[test]
fn num_attn_layers_4_allowed() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::set_var("NUM_ATTN_LAYERS", "4");
    assert_eq!(parse_num_attn_layers(), Ok(4));
    std::env::remove_var("NUM_ATTN_LAYERS");
}

#[test]
fn num_attn_layers_0_forbidden() {
    let _g = ENV_LOCK.lock().unwrap();
    // usize can't be negative; test 0 which is < 1.
    std::env::set_var("NUM_ATTN_LAYERS", "0");
    let err = parse_num_attn_layers().unwrap_err();
    assert!(
        err.contains("forbidden") || err.contains("< 1"),
        "expected forbidden/< 1 in error: {err}"
    );
    std::env::remove_var("NUM_ATTN_LAYERS");
}

// ── GF16_ENABLED ─────────────────────────────────────────────────────────────

#[test]
fn gf16_enabled_true() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::set_var("GF16_ENABLED", "true");
    assert_eq!(parse_gf16_enabled(), Ok(true));
    std::env::remove_var("GF16_ENABLED");
}

#[test]
fn gf16_enabled_false() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::set_var("GF16_ENABLED", "false");
    assert_eq!(parse_gf16_enabled(), Ok(false));
    std::env::remove_var("GF16_ENABLED");
}

#[test]
fn gf16_enabled_garbage_err() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::set_var("GF16_ENABLED", "garbage");
    let err = parse_gf16_enabled().unwrap_err();
    assert!(
        err.contains("invalid value"),
        "expected 'invalid value' in error: {err}"
    );
    std::env::remove_var("GF16_ENABLED");
}

// ── all-defaults ──────────────────────────────────────────────────────────────

#[test]
fn all_defaults_wave30_baseline() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::remove_var("HIDDEN_DIM");
    std::env::remove_var("NUM_ATTN_LAYERS");
    std::env::remove_var("GF16_ENABLED");
    let cfg = parse_arch_config().expect("all-defaults must not error");
    assert_eq!(
        cfg,
        ArchConfig {
            hidden_dim: 384,
            num_attn_layers: 1,
            gf16_enabled: false,
        },
        "all-defaults must be the Wave-30 baseline (h=384, 1L, no GF16)"
    );
}
