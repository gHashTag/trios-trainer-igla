//! Integration test enforcing the **pre-registration §4 seed-lock**:
//! while Gate-2 of [trios#143](https://github.com/gHashTag/trios/issues/143)
//! is open, the live `assertions/seed_results.jsonl` ledger must contain
//! **no** rows for seeds other than 43.  Seeds 42 and 44 are explicitly
//! frozen out — appending them before Gate-2 closes is test-set leakage.
//!
//! ## How the lock is enforced
//!
//! The pre-registration introduces a sentinel file
//! `assertions/.gate2_done`.  When that file is missing, this test
//! refuses any non-43 seed in the ledger.  When the file is present,
//! the lock is lifted and the test passes vacuously (Gate-final
//! enforcement happens in a follow-up test wired to its own sentinel).
//!
//! ## Pointers
//!
//! * Pre-registration comment: trios#143:4320342032 (immutable).
//! * Producer: `crates/trios-igla-race/src/bin/seed_emit.rs` (lane L-h3).
//! * Consumer: `crates/trios-igla-race/src/bin/ledger_check.rs` (lane L14).
//! * This test is the *additional* belt that complements the producer's
//!   own per-row validation: the producer cannot reject a row that has
//!   already been written by an out-of-band edit, but this test can.
//!
//! Refs: trios#143 lane L-h1 · INV-7 · R6 · R7 · R10.

use std::fs;
use std::path::PathBuf;

/// Locate the crate root regardless of where Cargo invokes the test from.
fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn ledger_path() -> PathBuf {
    crate_root().join("assertions/seed_results.jsonl")
}

fn gate_2_done_marker() -> PathBuf {
    crate_root().join("assertions/.gate2_done")
}

#[test]
fn falsify_no_extra_seeds_during_gate_2() {
    let marker = gate_2_done_marker();
    if marker.exists() {
        // Lock lifted — the Gate-final follow-up test takes over.
        return;
    }
    let path = ledger_path();
    let raw = match fs::read_to_string(&path) {
        Ok(s) => s,
        // The ledger may not exist yet on a fresh checkout; absence is
        // not a violation.
        Err(_) => return,
    };
    for (line_no, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // The header row carries `"schema": "1.0.0"` and no `seed` key.
        // We deliberately don't pull `serde_json` into this test (it is
        // a workspace member, but the test is intentionally
        // dependency-free) — we parse via substring.
        if !trimmed.contains("\"seed\"") {
            continue;
        }
        // Extract the integer seed via a minimal scanner.  This avoids
        // pulling in a JSON parser for a single-purpose test.
        let after = match trimmed.find("\"seed\"") {
            Some(i) => &trimmed[i + 6..],
            None => continue,
        };
        // skip whitespace + ':' + whitespace
        let after = after
            .trim_start_matches(|c: char| c.is_whitespace() || c == ':');
        let digits: String = after
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if digits.is_empty() {
            panic!(
                "ledger row {line_no} has malformed `seed` field: {trimmed}",
            );
        }
        let seed: u64 = digits.parse().unwrap();
        assert_eq!(
            seed, 43,
            "pre-registration §4 violation: ledger row {line_no} has seed={seed}; \
             only seed=43 is allowed before assertions/.gate2_done exists. \
             Row: {trimmed}",
        );
    }
}

#[test]
fn marker_path_is_under_crate() {
    // Sanity: the path resolution helpers point inside the crate.
    // A failure here means the test would silently no-op on every
    // machine — this catches a misconfigured CARGO_MANIFEST_DIR.
    let root = crate_root();
    assert!(
        root.join("Cargo.toml").exists(),
        "crate_root() resolved to {} which has no Cargo.toml",
        root.display(),
    );
    assert!(ledger_path().starts_with(&root));
    assert!(gate_2_done_marker().starts_with(&root));
}
