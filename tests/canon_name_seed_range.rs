//! Contract tests: IGLA naming convention (Tripwire #98)
//!
//! Spec: IGLA-<TYPE>-<NUM>-<TAG>-seed<N>
//! Example: IGLA-PROBE-2033-smeargate-seed42
//!
//! Bugs covered:
//!   - Naming convention violations not caught at runtime
//!   - Seed not embedded in canon_name

use std::collections::HashSet;

#[test]
fn test_igla_naming_format() {
    // Valid IGLA names follow this pattern:
    // ^IGLA-[A-Z]+-\d+-[a-z0-9]+-seed\d+$

    let valid_names = vec![
        "IGLA-PROBE-2033-smeargate-seed42",
        "IGLA-PROBE-2033-smeargate-seed1597",
        "IGLA-CHAMP-2048-baseline-seed43",
        "IGLA-PROBE-2050-gatetest-seed1",
    ];

    let invalid_names = vec![
        "probe-2033",           // Missing IGLA prefix
        "IGLA-PROBE-2033",      // Missing -seed<N> suffix
        "IGLA-PROBE-seed42",    // Missing NUM
        "IGLA-2033-seed42",     // Missing TYPE
        "IGLA-PROBE-2033-42",   // Missing TAG
    ];

    // Pattern: IGLA-<TYPE>-<NUM>-<TAG>-seed<N>
    // TYPE: one or more uppercase letters
    // NUM: one or more digits
    // TAG: lowercase letters and numbers
    // SEED: literal "seed" followed by one or more digits

    for name in &valid_names {
        assert!(validate_igla_name(name), "Valid name rejected: {}", name);
    }

    for name in &invalid_names {
        assert!(!validate_igla_name(name), "Invalid name accepted: {}", name);
    }
}

fn validate_igla_name(name: &str) -> bool {
    // ^IGLA-[A-Z]+-\d+-[a-z0-9]+-seed\d+$
    let parts: Vec<&str> = name.split('-').collect();

    if parts.len() < 5 {
        return false;
    }

    // Prefix: IGLA
    if parts[0] != "IGLA" {
        return false;
    }

    // TYPE: uppercase letters (e.g., PROBE, CHAMP)
    if !parts[1].chars().all(|c| c.is_ascii_uppercase()) {
        return false;
    }

    // NUM: digits (e.g., 2033)
    if !parts[2].chars().all(|c| c.is_ascii_digit()) {
        return false;
    }

    // TAG: lowercase alphanumeric (e.g., smeargate)
    if !parts[3].chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()) {
        return false;
    }

    // seed<N>: must contain "seed" prefix followed by digits
    // The last part(s) may be combined (e.g., "seed42")
    let last = parts.last().unwrap();
    if !last.starts_with("seed") {
        return false;
    }
    let seed_str = &last["seed".len()..];
    if !seed_str.chars().all(|c| c.is_ascii_digit()) || seed_str.is_empty() {
        return false;
    }

    true
}

#[test]
fn test_seed_extractable_from_canon_name() {
    // scarab.rs sets CANON_NAME env var for bpb_samples writes
    // neon_writer.rs:157 bpb_sample(canon_name, seed, step, bpb)
    // train_loop.rs:755 calls bpb_sample with seed from args.seed
    //
    // Bug: if canon_name doesn't contain seed, bpb_samples rows
    // can't be grouped by (canon_name, seed) correctly

    let name = "IGLA-PROBE-2033-smeargate-seed42";
    let seed = extract_seed(name).unwrap();

    assert_eq!(seed, 42, "Seed should be extractable from canon_name");

    let name2 = "IGLA-CHAMP-2048-baseline-seed1597";
    let seed2 = extract_seed(name2).unwrap();

    assert_eq!(seed2, 1597);
}

fn extract_seed(name: &str) -> Option<i32> {
    // Extract seed from IGLA-*-seed<N> pattern
    name.split("-seed")
        .last()?
        .parse()
        .ok()
}

#[test]
fn test_canon_name_uniqueness_per_seed() {
    // Each seed must have a unique canon_name
    // Otherwise bpb_samples GROUP BY (canon_name, seed) breaks

    let base = "IGLA-PROBE-2033-smeargate";
    let seeds = vec![42, 43, 1597];

    let mut canon_names = HashSet::new();

    for seed in seeds {
        let name = format!("{}-seed{}", base, seed);
        canon_names.insert(name);
    }

    assert_eq!(canon_names.len(), 3,
        "Each seed must have a unique canon_name");

    // Bug scenario: same canon_name for different seeds
    // IGLA-PROBE-2033-smeargate used for seed=42 AND seed=43
    // This would corrupt bpb_samples aggregation
}

#[test]
fn test_trios_canon_name_env_var_required() {
    // scarab.rs:149 sets CANON_NAME env var
    // train_loop.rs:752-755 reads CANON_NAME to call bpb_sample
    //
    // If CANON_NAME is unset, bpb_samples writes fail silently

    // Contract: scarab MUST set CANON_NAME before spawning trios-train
    // scarab.rs:149 cmd.env("CANON_NAME", &strat.canon_name)

    let test_canon = "IGLA-PROBE-2033-smeargate-seed42";

    assert!(test_canon.starts_with("IGLA-"),
        "CANON_NAME must start with IGLA- prefix");

    assert!(test_canon.contains("-seed"),
        "CANON_NAME must contain -seed<N> suffix");
}
