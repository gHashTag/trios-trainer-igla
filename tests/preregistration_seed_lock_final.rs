//! Integration test enforcing the **Gate-final pre-registration §4 seed-lock**:
//! while Gate-final is open, the live `assertions/seed_results.jsonl` ledger
//! must contain rows **only** for seeds in {42, 43, 44}.  Seeds 41 and 45
//! are explicitly frozen out — appending them before Gate-final closes is
//! test-set leakage (per DRAFT §8).
//!
//! Refs: trios#143 Gate-final DRAFT §4 · §8 · L-f3 · INV-7.

use std::fs;
use std::path::PathBuf;

const GATE_FINAL_ALLOWED_SEEDS: [u64; 3] = [42, 43, 44];

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn ledger_path() -> PathBuf {
    crate_root().join("assertions/seed_results.jsonl")
}

fn gate_final_done_marker() -> PathBuf {
    crate_root().join("assertions/.gate_final_done")
}

fn extract_seeds(text: &str) -> Vec<u64> {
    let mut seeds = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || !trimmed.contains("\"seed\"") {
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

#[test]
fn falsify_skew_seeds() {
    if gate_final_done_marker().exists() {
        return;
    }
    let path = ledger_path();
    let raw = match fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return,
    };
    let seeds = extract_seeds(&raw);
    for &seed in &seeds {
        assert!(
            GATE_FINAL_ALLOWED_SEEDS.contains(&seed),
            "Gate-final §8 violation: seed={} not in allowed set {{42, 43, 44}}",
            seed,
        );
    }
}

#[test]
fn falsify_fewer_than_3_distinct_seeds_at_done() {
    if !gate_final_done_marker().exists() {
        return;
    }
    let path = ledger_path();
    let raw = fs::read_to_string(&path).expect("ledger must exist at gate-final done");
    let seeds = extract_seeds(&raw);
    let distinct: std::collections::HashSet<u64> = seeds.into_iter().collect();
    assert!(
        distinct.len() >= 3,
        "Gate-final §2 falsifier 3: fewer than 3 distinct seeds in ledger (found {})",
        distinct.len(),
    );
    for &s in &distinct {
        assert!(
            GATE_FINAL_ALLOWED_SEEDS.contains(&s),
            "Gate-final §8: seed={} outside allowed set",
            s,
        );
    }
}

#[test]
fn gate_final_seeds_constant_matches_draft() {
    assert_eq!(&GATE_FINAL_ALLOWED_SEEDS, &[42u64, 43, 44]);
}
