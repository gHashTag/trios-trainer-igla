//! L9 — QK head shape guard for IGLA attention layers.
//!
//! Type-safe configuration for the query-key (QK) head used by the IGLA
//! attention block.  This module is **shape & scale only** — no actual
//! tensor math lives here.  The job is to refuse malformed
//! configurations *before* a worker burns minutes of training time on a
//! geometry that violates either INV-3 (`d_model >= 256`) or the
//! Trinity head-dim φ-floor (`head_dim >= φ⁴ ≈ 6.854 ⇒ head_dim ≥ 7`).
//!
//! ## Why a dedicated guard
//!
//! Past races have wasted entire seeds on:
//!
//! 1. `num_heads` that does not divide `d_model` — silent slicing bug
//!    that produces NaN gradients.
//! 2. `head_dim < 7` — below the GF16 representational floor; QK dots
//!    collapse into a single bucket.
//! 3. `scale` set to a hard-coded `0.125` instead of `1/sqrt(d_k)` —
//!    miscalibrated softmax temperature.
//! 4. `scale = 0` after a refactor — flat attention, model never
//!    learns.
//!
//! `QkHead::new()` refuses each case with a typed `QkHeadError`.
//!
//! ## Coq anchor
//!
//! No new invariant introduced — L9 consumes INV-3 (`d_model_min`) and
//! the Trinity φ⁴ head-dim floor.  The φ⁴ floor is presented as a
//! `pub const` derived from `crate::invariants::PHI` so L-R14 holds.
//!
//! ## Falsification witnesses (R8)
//!
//! Each `falsify_*` test demonstrates the guard rejects a known-bad
//! configuration.
//!
//! Refs: trios#143 lane L9 · INV-3 (consumer) · L-R14 · R8.

use crate::invariants::{INV3_D_MODEL_MIN, PHI};

// ----------------------------------------------------------------------
// Anchors (L-R14)
// ----------------------------------------------------------------------

/// φ⁴ ≈ 6.854.  Trinity head-dim floor: `head_dim ≥ ceil(φ⁴) = 7`.
/// Derivation: `φ⁴ = (φ²)² = (φ + 1)² = φ² + 2φ + 1 = 3φ + 2`.
/// We compute it from the canonical `PHI` so any drift in the φ
/// constant surfaces in `anchor_phi_4_matches_derivation`.
pub const PHI_4: f64 = 3.0 * PHI + 2.0;

/// Concrete head-dim floor used by the guard: `ceil(φ⁴) = 7`.
/// Sourced symbolically as `PHI_4.ceil() as u32`, but we keep it as a
/// pub const so callers can pattern-match against it directly without
/// depending on a runtime ceil.
pub const HEAD_DIM_PHI_FLOOR: u32 = 7;

/// Maximum number of heads: defensive upper bound to catch obvious
/// misconfiguration (a `usize::MAX / sizeof(f32)` typo would otherwise
/// allocate gigabytes of attention weights).  Sourced from the largest
/// historically-useful value, 64.
pub const NUM_HEADS_MAX: u32 = 64;

// ----------------------------------------------------------------------
// Errors
// ----------------------------------------------------------------------

/// Reasons a QK head config can be rejected.  Closed enum — pattern
/// match exhaustively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkHeadError {
    /// `d_model` below `INV3_D_MODEL_MIN`.
    DModelBelowInv3,
    /// `num_heads == 0`.
    ZeroHeads,
    /// `num_heads > NUM_HEADS_MAX`.
    TooManyHeads,
    /// `d_model % num_heads != 0`.
    DModelNotDivisible,
    /// `head_dim < HEAD_DIM_PHI_FLOOR`.
    HeadDimBelowPhiFloor,
    /// Computed `scale` is non-finite or non-positive.
    ScaleInvalid,
}

impl core::fmt::Display for QkHeadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            QkHeadError::DModelBelowInv3 => {
                write!(f, "d_model below INV-3 floor (256) — reject")
            }
            QkHeadError::ZeroHeads => write!(f, "num_heads = 0 — reject"),
            QkHeadError::TooManyHeads => {
                write!(f, "num_heads > NUM_HEADS_MAX (64) — reject")
            }
            QkHeadError::DModelNotDivisible => write!(
                f,
                "d_model not divisible by num_heads — silent slicing bug, reject"
            ),
            QkHeadError::HeadDimBelowPhiFloor => write!(
                f,
                "head_dim below φ⁴ floor (7) — GF16 representational collapse, reject"
            ),
            QkHeadError::ScaleInvalid => write!(
                f,
                "scale non-finite or ≤ 0 — softmax temperature undefined, reject"
            ),
        }
    }
}

impl std::error::Error for QkHeadError {}

// ----------------------------------------------------------------------
// QkHead config
// ----------------------------------------------------------------------

/// QK-head configuration.  Once you hold a `QkHead`, the geometry has
/// passed every static check; no runtime-NaN attention weights from a
/// malformed config.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QkHead {
    d_model: u32,
    num_heads: u32,
    head_dim: u32,
    /// Pre-computed `1 / sqrt(head_dim)`.
    scale: f64,
}

impl QkHead {
    /// Construct & validate a QK head.  Computes `head_dim` and
    /// `scale = 1/sqrt(head_dim)` internally; callers may not override
    /// these — that is precisely the bug class this guard exists to
    /// prevent (cf. doc-comment item 3).
    pub fn new(d_model: u32, num_heads: u32) -> Result<Self, QkHeadError> {
        if (d_model as usize) < INV3_D_MODEL_MIN {
            return Err(QkHeadError::DModelBelowInv3);
        }
        if num_heads == 0 {
            return Err(QkHeadError::ZeroHeads);
        }
        if num_heads > NUM_HEADS_MAX {
            return Err(QkHeadError::TooManyHeads);
        }
        if !d_model.is_multiple_of(num_heads) {
            return Err(QkHeadError::DModelNotDivisible);
        }
        let head_dim = d_model / num_heads;
        if head_dim < HEAD_DIM_PHI_FLOOR {
            return Err(QkHeadError::HeadDimBelowPhiFloor);
        }
        let scale = 1.0 / (head_dim as f64).sqrt();
        if !scale.is_finite() || scale <= 0.0 {
            return Err(QkHeadError::ScaleInvalid);
        }
        Ok(Self {
            d_model,
            num_heads,
            head_dim,
            scale,
        })
    }

    #[inline]
    pub fn d_model(&self) -> u32 {
        self.d_model
    }
    #[inline]
    pub fn num_heads(&self) -> u32 {
        self.num_heads
    }
    #[inline]
    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }
    #[inline]
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ----- admit (positive) cases ----------------------------------

    #[test]
    fn admit_canonical_256_8() {
        let h = QkHead::new(256, 8).unwrap();
        assert_eq!(h.head_dim(), 32);
        assert!((h.scale() - 1.0 / (32.0_f64).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn admit_384_8_works() {
        let h = QkHead::new(384, 8).unwrap();
        assert_eq!(h.head_dim(), 48);
    }

    #[test]
    fn admit_minimum_d_model_works() {
        // INV-3 boundary: d_model = 256 must succeed.
        let h = QkHead::new(INV3_D_MODEL_MIN as u32, 8).unwrap();
        assert_eq!(h.d_model() as usize, INV3_D_MODEL_MIN);
    }

    #[test]
    fn admit_single_head_with_large_d_model_works() {
        // Edge: a single head over the whole d_model is geometrically
        // valid (head_dim = d_model). Useful as a smoke test.
        let h = QkHead::new(256, 1).unwrap();
        assert_eq!(h.head_dim(), 256);
    }

    // ----- falsification witnesses (R8) ----------------------------

    #[test]
    fn falsify_d_model_below_inv3_rejected() {
        // 128 violates INV-3 (`d_model_min = 256`).
        assert_eq!(
            QkHead::new(128, 8).unwrap_err(),
            QkHeadError::DModelBelowInv3
        );
    }

    #[test]
    fn falsify_d_model_one_below_inv3_boundary() {
        // Boundary: 255 must be rejected.
        assert_eq!(
            QkHead::new((INV3_D_MODEL_MIN - 1) as u32, 8).unwrap_err(),
            QkHeadError::DModelBelowInv3
        );
    }

    #[test]
    fn falsify_zero_heads_rejected() {
        assert_eq!(QkHead::new(256, 0).unwrap_err(), QkHeadError::ZeroHeads);
    }

    #[test]
    fn falsify_too_many_heads_rejected() {
        assert_eq!(
            QkHead::new(256 * 65, NUM_HEADS_MAX + 1).unwrap_err(),
            QkHeadError::TooManyHeads
        );
    }

    #[test]
    fn falsify_non_divisible_rejected() {
        // 257 prime → no nontrivial num_heads divides it ≤ 64.
        assert_eq!(
            QkHead::new(257, 8).unwrap_err(),
            QkHeadError::DModelNotDivisible
        );
    }

    #[test]
    fn falsify_head_dim_below_phi_floor_rejected() {
        // d_model = 256, num_heads = 64 → head_dim = 4 < φ⁴ floor (7).
        assert_eq!(
            QkHead::new(256, 64).unwrap_err(),
            QkHeadError::HeadDimBelowPhiFloor
        );
    }

    #[test]
    fn falsify_head_dim_six_rejected() {
        // 384 / 64 = 6 < 7 (φ⁴ floor). Boundary one below the floor.
        assert_eq!(
            QkHead::new(384, 64).unwrap_err(),
            QkHeadError::HeadDimBelowPhiFloor
        );
    }

    #[test]
    fn admit_head_dim_seven_at_phi_floor() {
        // Cannot construct head_dim = 7 at INV-3 floor (256 / 7 isn't
        // integral). Use d_model = 448 to get head_dim = 7 exactly.
        let h = QkHead::new(448, 64).unwrap();
        assert_eq!(h.head_dim(), HEAD_DIM_PHI_FLOOR);
    }

    // ----- L-R14 anchor guards -------------------------------------

    #[test]
    fn anchor_phi_4_matches_derivation() {
        // φ⁴ = 3φ + 2 algebraic identity.
        let derived = 3.0 * PHI + 2.0;
        assert!((PHI_4 - derived).abs() < 1e-15);
        // Numerical sanity: φ⁴ ≈ 6.8541
        assert!((PHI_4 - 6.854_101_966_249_685).abs() < 1e-10);
    }

    #[test]
    fn anchor_head_dim_floor_matches_phi_4_ceil() {
        // HEAD_DIM_PHI_FLOOR = ceil(φ⁴) = 7.
        let derived = PHI_4.ceil() as u32;
        assert_eq!(HEAD_DIM_PHI_FLOOR, derived);
    }

    #[test]
    fn anchor_inv3_floor_pinned_to_256() {
        // Pin the INV-3 anchor we depend on.  If L5 ever moves it,
        // this test will catch the implicit contract break.
        assert_eq!(INV3_D_MODEL_MIN, 256);
    }

    #[test]
    fn anchor_num_heads_max_is_64() {
        assert_eq!(NUM_HEADS_MAX, 64);
    }

    // ----- numerical scale guard ----------------------------------

    #[test]
    fn admit_scale_is_one_over_sqrt_head_dim() {
        let h = QkHead::new(256, 8).unwrap();
        let expected = 1.0 / (32.0_f64).sqrt();
        assert!((h.scale() - expected).abs() < 1e-15);
        // And it is finite and > 0 (the guard's invariant).
        assert!(h.scale().is_finite());
        assert!(h.scale() > 0.0);
    }
}
