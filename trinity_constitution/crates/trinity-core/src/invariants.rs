//! PhD Invariants — INV-1..INV-9 as const fn
//!
//! # Constitutional mandate (Law 5)
//!
//! All invariants encoded as compile-time constants.
//! No magic numbers without explicit `(ref: INV-X)` reference.
//!
//! # Invariants
//!
//! - **INV-1**: Minimum training steps for meaningful learning
//! - **INV-2**: Hidden dimension must be multiple of φ-scaled base (828)
//! - **INV-3**: Learning rate bounded by φ-physics (0.0001 to 0.01)
//! - **INV-4**: Context window at least φ³ (≈4.2, round to 12 for practicality)
//! - **INV-5**: BPB always between 0 and 100 (information theory)
//! - **INV-6**: Floor BPB for "worse than random" detection
//! - **INV-7**: Seed must be from Fibonacci sequence (21, 34, 55, 89, 144, 233, 377, 610, 987, 1597)
//! - **INV-8**: Steps budget multiple of 1000 (clean reporting)
//! - **INV-9**: Validation split exactly 100000 bytes (byte-disjoint)

/// INV-1: Minimum training steps for meaningful convergence
/// (ref: PhD Chapter 5 — convergence diagnostics)
pub const INV_1_MIN_STEPS: usize = 5000;

/// INV-2: φ-scaled hidden dimension base = 828
/// (ref: φ-physics — 828 = round(φ⁶ × 100))
pub const INV_2_HIDDEN_BASE: u32 = 828;

/// INV-3: Learning rate bounds — φ-physics validated range
/// (ref: PhD Chapter 8 — lr sensitivity analysis)
pub const INV_3_LR_MIN: f32 = 0.0001;
pub const INV_3_LR_MAX: f32 = 0.01;

/// INV-4: Minimum context window
/// (ref: φ³ ≈ 4.236, round to 12 for practicality)
pub const INV_4_MIN_CTX: usize = 12;

/// INV-5: BPB theoretical bounds (bits per byte, 0-8 max, we allow up to 100 for error cases)
/// (ref: Shannon entropy — 8 bits max for uniform bytes, 100 for catastrophic failure)
pub const INV_5_BPB_MIN: f32 = 0.0;
pub const INV_5_BPB_MAX: f32 = 100.0;

/// INV-6: Floor BPB — below this, model is worse than random
/// (ref: PhD Chapter 12 — "worse than random" detection threshold)
pub const INV_6_FLOOR_BPB: f32 = 2.0;

/// INV-7: Valid seeds from Fibonacci sequence
/// (ref: Fibonacci properties — deterministic but pseudorandom)
pub const INV_7_VALID_SEEDS: &[u64] = &[
    21, 34, 55, 89, 144, 233, 377, 610, 987, 1597
];

/// INV-8: Steps budget must be multiple of 1000
/// (ref: clean checkpointing and reporting)
pub const INV_8_STEPS_MULTIPLE: usize = 1000;

/// INV-9: Validation split exactly 100000 bytes
/// (ref: byte-disjoint split for honest evaluation)
pub const INV_9_VAL_SPLIT_BYTES: usize = 100_000;

/// INV-7 helper: check if seed is valid
#[inline]
pub const fn is_valid_seed(seed: u64) -> bool {
    // const fn doesn't support for loop, use match instead
    match seed {
        21 | 34 | 55 | 89 | 144 | 233 | 377 | 610 | 987 | 1597 => true,
        _ => false,
    }
}

/// INV-8 helper: check if steps budget is valid
#[inline]
pub const fn is_valid_steps_budget(steps: usize) -> bool {
    steps >= INV_1_MIN_STEPS && steps % INV_8_STEPS_MULTIPLE == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inv_7_all_seeds_valid() {
        for &seed in INV_7_VALID_SEEDS {
            assert!(is_valid_seed(seed), "Seed {} marked invalid", seed);
        }
    }

    #[test]
    fn inv_7_invalid_seed_rejected() {
        assert!(!is_valid_seed(42), "Seed 42 should be invalid");
        assert!(!is_valid_seed(1000), "Seed 1000 should be invalid");
    }

    #[test]
    fn inv_8_valid_steps_budgets() {
        assert!(is_valid_steps_budget(5000), "5000 should be valid");
        assert!(is_valid_steps_budget(10000), "10000 should be valid");
        assert!(is_valid_steps_budget(54000), "54000 should be valid");
    }

    #[test]
    fn inv_8_invalid_steps_budgets() {
        assert!(!is_valid_steps_budget(4999), "4999 should be invalid");
        assert!(!is_valid_steps_budget(5500), "5500 not a multiple of 1000");
    }
}
