//! Gate-final seed-lock, reconciled with the source law.
//!
//! This file used to hardcode `GATE_FINAL_ALLOWED_SEEDS = [42, 43, 44]` and
//! assert that every row in `assertions/seed_results.jsonl` used one of them.
//! `src/seed_canon.rs` declares {42, 43, 44, 45} FORBIDDEN under Canon #93 and
//! the allowed set to be {47, 89, 123, 144}.  The test therefore did not merely
//! miss the violation - it certified the violation as the only legal state.
//!
//! The source law wins.  The canon is now imported rather than copied: every
//! verdict below comes from `seed_canon::parse_seed`, so a change to Canon #93
//! propagates here instead of being silently contradicted.
//!
//! The Gate-final DRAFT's quorum condition ("3 distinct seeds") is unreachable
//! on canon-legal evidence as long as the only ledger this repo ever had is
//! retracted, so the `.gate_final_done` sentinel must stay absent.  See
//! `RETRACTION.md`, section 1.
//!
//! Refs: trios#143 Gate-final DRAFT section 4 - section 8 - L-f3 - INV-7 -
//! Canon #93.

use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use trios_trainer::seed_canon;

/// The DRAFT's allowed set, kept ONLY so the test below can prove that the
/// canon rejects all of it.  Nothing here asserts that a seed *is* in this set.
const GATE_FINAL_DRAFT_SEEDS: [u64; 3] = [42, 43, 44];

/// `seed_canon::parse_seed` reads the process-global `SEED` env var.
static SEED_ENV_LOCK: Mutex<()> = Mutex::new(());

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn retracted_ledger_path() -> PathBuf {
    crate_root().join("assertions/RETRACTED-seed_results.jsonl.txt")
}

fn live_ledger_path() -> PathBuf {
    crate_root().join("assertions/seed_results.jsonl")
}

fn gate_final_done_marker() -> PathBuf {
    crate_root().join("assertions/.gate_final_done")
}

/// A missing evidence file is a failure, never a silent pass.
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
        if !digits.is_empty() {
            if let Ok(seed) = digits.parse::<u64>() {
                seeds.push(seed);
            }
        }
    }
    seeds
}

/// Ask `src/seed_canon.rs` directly instead of trusting a local constant.
fn canon_verdict(seed: u64) -> Result<u64, String> {
    let _g = SEED_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("SEED", seed.to_string());
    let verdict = seed_canon::parse_seed();
    std::env::remove_var("SEED");
    verdict
}

/// The reconciliation itself: the DRAFT's allowed set is entirely forbidden.
///
/// If Canon #93 is ever relaxed to admit 42/43/44, this test fails and forces
/// whoever relaxed it to revisit the retraction, rather than letting the old
/// hardcoded lock quietly become true again.
#[test]
fn gate_final_draft_seeds_are_all_forbidden_by_canon() {
    for seed in GATE_FINAL_DRAFT_SEEDS {
        match canon_verdict(seed) {
            Err(e) => assert!(
                e.contains("forbidden"),
                "seed_canon rejected seed={seed} for the wrong reason: {e}"
            ),
            Ok(v) => panic!(
                "Canon #93 now ACCEPTS seed={seed} (parse_seed returned {v}). The \
                 Gate-final DRAFT's allowed set {{42, 43, 44}} was reconciled \
                 against src/seed_canon.rs, which forbade all three. If the canon \
                 changed, re-read RETRACTION.md before re-admitting the retracted \
                 ledger's seeds."
            ),
        }
    }
}

/// Every seed a producer might copy forward out of the retracted ledger must
/// still be rejected by the canon.
#[test]
fn falsify_skew_seeds() {
    let raw = read_retracted_ledger();
    let seeds = extract_seeds(&raw);
    assert!(
        !seeds.is_empty(),
        "no rows parsed out of the retracted ledger: this falsifier cannot pass vacuously"
    );
    for &seed in &seeds {
        assert!(
            canon_verdict(seed).is_err(),
            "retracted ledger row carries seed={seed}, which src/seed_canon.rs \
             accepts. Canon-legal seeds are {{47, 89, 123, 144}}; every row in \
             this file used a forbidden one, and none of them has an artifact."
        );
    }
}

/// Gate-final cannot be declared done on retracted evidence.
///
/// The old version of this test returned early when the sentinel was absent -
/// which is always - so it never ran. It now asserts the sentinel's absence,
/// which is the actual post-retraction invariant: the only ledger this repo
/// ever had is retracted, so no quorum of canon-legal seeds exists to close the
/// gate with.
#[test]
fn falsify_gate_final_declared_done_on_retracted_evidence() {
    let marker = gate_final_done_marker();
    assert!(
        !marker.exists(),
        "assertions/.gate_final_done exists at {}, but the only ledger in this \
         repository is retracted (RETRACTION.md) and the live path {} must stay \
         absent. Gate-final cannot be closed on evidence with no artifact, no \
         corpus hash and no canon-legal seed.",
        marker.display(),
        live_ledger_path().display(),
    );
}
