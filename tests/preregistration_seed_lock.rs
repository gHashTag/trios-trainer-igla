//! Post-retraction seed-lock falsifiers.
//!
//! The pre-registration section-4 seed-lock was written against a *live* ledger
//! at `assertions/seed_results.jsonl`.  That ledger has been retracted: it
//! published rows with no artifact, no corpus hash, no trainer hash and no eval
//! coverage, written outside `ledger::emit_row`, on seeds that
//! `src/seed_canon.rs` forbids.  It now lives at
//! `assertions/RETRACTED-seed_results.jsonl.txt` behind a header naming those
//! defects.  See `RETRACTION.md` at the repository root.
//!
//! What these falsifiers enforce now:
//!
//! 1. The retracted ledger must still be on disk, under its retracted name,
//!    with its retraction header intact.  A falsifier whose evidence file has
//!    vanished must FAIL, not pass quietly.
//! 2. The old live path must not come back.  Any producer that re-creates
//!    `assertions/seed_results.jsonl` resurrects retracted evidence under the
//!    name five documents still remember as the SSOT.
//! 3. Every seed in the retracted ledger must still be one the canon rejects.
//!    If someone edits the rows to look canon-compliant, the file starts
//!    looking citable again while its artifacts remain non-existent.
//!
//! Refs: trios#143 lane L-h1 - INV-7 - R6 - R7 - R10 - Canon #93.

use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use trios_trainer::seed_canon;

/// `seed_canon::parse_seed` reads the process-global `SEED` env var, so every
/// probe through it must be serialised against the other tests in this binary.
static SEED_ENV_LOCK: Mutex<()> = Mutex::new(());

/// Locate the crate root regardless of where Cargo invokes the test from.
fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The retracted ledger, preserved verbatim behind a plain-text header.
fn retracted_ledger_path() -> PathBuf {
    crate_root().join("assertions/RETRACTED-seed_results.jsonl.txt")
}

/// The path the ledger used to occupy.  Must stay empty.
fn live_ledger_path() -> PathBuf {
    crate_root().join("assertions/seed_results.jsonl")
}

fn retraction_note_path() -> PathBuf {
    crate_root().join("RETRACTION.md")
}

/// Read the retracted ledger, or fail loudly.  This is the single place where a
/// missing evidence file is turned into a failure instead of a vacuous pass.
fn read_retracted_ledger() -> String {
    let path = retracted_ledger_path();
    match fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => panic!(
            "evidence file missing: this falsifier cannot pass vacuously ({}: {e})",
            path.display()
        ),
    }
}

/// Extract every integer `seed` field from JSONL-ish text.
///
/// Deliberately dependency-free: the header lines carry no `"seed"` key and are
/// skipped by the same filter that skips blanks and non-object lines.
fn extract_seeds(text: &str) -> Vec<u64> {
    let mut seeds = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || !trimmed.starts_with('{') || !trimmed.contains("\"seed\"") {
            continue;
        }
        let after = match trimmed.find("\"seed\"") {
            Some(i) => &trimmed[i + 6..],
            None => continue,
        };
        let after = after.trim_start_matches(|c: char| c.is_whitespace() || c == ':');
        let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            panic!("retracted ledger row has a malformed `seed` field: {trimmed}");
        }
        seeds.push(digits.parse::<u64>().expect("digits parse as u64"));
    }
    seeds
}

/// Ask the source law, rather than a local copy of it, whether a seed is legal.
fn canon_rejects(seed: u64) -> bool {
    let _g = SEED_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("SEED", seed.to_string());
    let verdict = seed_canon::parse_seed();
    std::env::remove_var("SEED");
    verdict.is_err()
}

#[test]
fn falsify_retracted_ledger_disappeared() {
    let raw = read_retracted_ledger();
    assert!(
        raw.contains("RETRACTED EVIDENCE - DO NOT CITE"),
        "the retracted ledger lost its retraction header; without it the rows \
         read as a live SSOT again"
    );
    assert!(
        retraction_note_path().exists(),
        "RETRACTION.md is missing from the repository root; the retracted \
         ledger points at an accounting that does not ship"
    );
}

#[test]
fn falsify_live_ledger_resurrected() {
    let live = live_ledger_path();
    assert!(
        !live.exists(),
        "assertions/seed_results.jsonl exists again at {}. That path is \
         retracted (see RETRACTION.md); re-creating it republishes rows with \
         no artifact, no corpus hash and no trainer hash under the name five \
         documents still call the source of truth.",
        live.display(),
    );
}

#[test]
fn falsify_retracted_rows_look_canon_compliant() {
    let raw = read_retracted_ledger();
    let seeds = extract_seeds(&raw);
    assert!(
        !seeds.is_empty(),
        "no rows found in the retracted ledger: this falsifier cannot pass vacuously"
    );
    for seed in seeds {
        assert!(
            canon_rejects(seed),
            "retracted ledger row carries seed={seed}, which src/seed_canon.rs \
             ACCEPTS. Every row in this file used a forbidden seed; a row that \
             now looks canon-compliant has been edited, and the file is still \
             backed by no artifact."
        );
    }
}

#[test]
fn marker_path_is_under_crate() {
    // Sanity: the path resolution helpers point inside the crate.
    // A failure here means the tests would silently no-op on every
    // machine - this catches a misconfigured CARGO_MANIFEST_DIR.
    let root = crate_root();
    assert!(
        root.join("Cargo.toml").exists(),
        "crate_root() resolved to {} which has no Cargo.toml",
        root.display(),
    );
    assert!(retracted_ledger_path().starts_with(&root));
    assert!(live_ledger_path().starts_with(&root));
    assert!(retraction_note_path().starts_with(&root));
}
