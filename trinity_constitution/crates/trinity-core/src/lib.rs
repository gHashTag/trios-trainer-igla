//! Trinity Core — L0+L1: PhD invariants, φ-physics, BPB calculation
//!
//! # Constitutional mandates
//!
//! - **INV-1..INV-9** encoded as `const fn` — invariant violation = test failure
//! - φ-physics constants — φ² + φ⁻² = 3 (golden ratio identity)
//! - Honest BPB calculation — no magic numbers without `(ref: INV-X)`
//!
//! # PR-O2 status
//!
//! - [ ] invariants.rs — INV-1..INV-9 as const fn
//! - [ ] phi.rs — φ, φ², φ⁻², Fibonacci
//! - [ ] bpb.rs — honest BPB calculation
//! - [ ] 9 unit tests (all green)
//!
//! 🌻 φ² + φ⁻² = 3 · TRINITY · O(1) FOREVER

pub mod invariants;
pub mod phi;
pub mod bpb;

// Re-exports for convenience
pub use phi::{PHI, PHI_SQUARED, PHI_INVERSE_SQUARED};
pub use invariants::{INV_1_MIN_STEPS, INV_6_FLOOR_BPB};
pub use bpb::calculate_bpb;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_ratio_identity() {
        // φ² + φ⁻² = 3 (golden ratio identity)
        let left = PHI_SQUARED + PHI_INVERSE_SQUARED;
        let diff = (left - 3.0).abs();
        assert!(diff < 1e-10, "φ² + φ⁻² ≠ 3: left={}, diff={}", left, diff);
    }

    #[test]
    fn inv_1_min_steps_positive() {
        assert!(INV_1_MIN_STEPS > 0, "INV-1: min_steps must be positive");
    }

    #[test]
    fn inv_6_floor_bpb_reasonable() {
        assert!(INV_6_FLOOR_BPB > 0.0 && INV_6_FLOOR_BPB < 10.0,
                "INV-6: floor_bpb must be between 0 and 10");
    }
}
