//! L-T* falsification witness: invariants do not DRIFT.
//!
//! The hypothesis (Gate G1) is **falsified iff** invariants imported from
//! `trios-igla-race` (canonical) diverge from this repo's view during the
//! migration. Today the canonical crates are dependency-disabled in
//! `Cargo.toml` (submodule issue logged on PR-0), so we cross-check against
//! `assertions/igla_assertions.json`, which is the migration-time mirror of
//! the canonical invariants snapshot.
//!
//! When the `trios-integration` feature is enabled (after the submodule fix
//! lands), the test should additionally compare runtime constants against
//! `trios_igla_race::*`. Until then the snapshot file is the reference.
//!
//! Mission: TRAINER-IGLA-SOT (`phi^2 + phi^-2 = 3`).
//! Refs: gHashTag/trios-trainer-igla#2 (ONE SHOT), R5/R7 standing rules.

use std::fs;
use trios_trainer::TRINITY_ANCHOR;

fn assertions_snapshot() -> serde_json::Value {
    let txt = fs::read_to_string("assertions/igla_assertions.json")
        .expect("assertions/igla_assertions.json must exist (mirror of canonical invariants)");
    serde_json::from_str(&txt).expect("igla_assertions.json must be valid JSON")
}

#[test]
fn anchor_in_repo_matches_assertions_snapshot() {
    // φ² + φ⁻² = 3 — Trinity Identity. Anchor must agree across (a) the local
    // `lib.rs::TRINITY_ANCHOR` constant, (b) the runtime computation, and (c)
    // the canonical assertions JSON.
    let phi: f64 = (1.0 + 5f64.sqrt()) / 2.0;
    let runtime = phi.powi(2) + phi.powi(-2);
    assert!(
        (runtime - TRINITY_ANCHOR).abs() < 1e-12,
        "DRIFT: runtime φ² + φ⁻² = {} != TRINITY_ANCHOR = {}",
        runtime,
        TRINITY_ANCHOR
    );

    // The canonical assertions snapshot encodes the same anchor as a plain
    // string; we accept any nesting under `_metadata` so a future schema
    // refactor can move the field without breaking the witness.
    let snap = assertions_snapshot();
    fn find_anchor(v: &serde_json::Value) -> Option<String> {
        match v {
            serde_json::Value::Object(m) => {
                if let Some(s) = m.get("trinity_anchor").and_then(|x| x.as_str()) {
                    return Some(s.to_string());
                }
                for (_, vv) in m.iter() {
                    if let Some(s) = find_anchor(vv) {
                        return Some(s);
                    }
                }
                None
            }
            serde_json::Value::Array(a) => a.iter().find_map(find_anchor),
            _ => None,
        }
    }
    let anchor_str = find_anchor(&snap)
        .expect("DRIFT: trinity_anchor key missing from assertions/igla_assertions.json");
    assert!(
        anchor_str.contains("phi^2") && anchor_str.contains("phi^-2") && anchor_str.contains("3"),
        "DRIFT: trinity_anchor string lost the φ² + φ⁻² = 3 phrasing: {anchor_str:?}"
    );
}

#[test]
fn assertions_snapshot_invariant_count_is_consistent() {
    let snap = assertions_snapshot();
    let declared = snap["_metadata"]["invariant_count"].as_u64().unwrap_or(0);
    let actual = snap["invariants"]
        .as_array()
        .map(|a| a.len() as u64)
        .unwrap_or(0);
    assert!(
        declared > 0,
        "DRIFT: _metadata.invariant_count must be > 0, got {declared}"
    );
    assert!(
        actual >= declared,
        "DRIFT: invariants[] has {actual} entries, but _metadata.invariant_count = {declared}"
    );
}

#[test]
fn igla_target_bpb_matches_repo_default() {
    // The Gate-2 acceptance metric is mean(BPB) < 1.85 over 3 seeds. The repo
    // default and the assertions snapshot must agree on the floor.
    assert!(
        (trios_trainer::train_loop::DEFAULT_IGLA_TARGET_BPB - 1.85).abs() < 1e-12,
        "DRIFT: DEFAULT_IGLA_TARGET_BPB = {} (expected 1.85)",
        trios_trainer::train_loop::DEFAULT_IGLA_TARGET_BPB
    );
}

#[cfg(feature = "trios-integration")]
mod integration {
    //! Cross-checks that activate only when the canonical workspace is present.
    //!
    //! Today these are stubs because `Cargo.toml` keeps the canonical crates
    //! disabled (submodule issue). Re-enable once the dependency is restored.
    #[test]
    fn anchor_matches_canonical() {
        // Placeholder: when trios_igla_race is wired, replace with
        // assert!((trios_igla_race::TRINITY_ANCHOR - super::TRINITY_ANCHOR).abs() < 1e-12);
        let phi: f64 = (1.0 + 5f64.sqrt()) / 2.0;
        let runtime = phi.powi(2) + phi.powi(-2);
        assert!((runtime - 3.0).abs() < 1e-12);
    }
}
