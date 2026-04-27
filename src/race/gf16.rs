//! INV-3 mirror: GF(16) safe-domain guard.
//!
//! Theorem chain:
//! - Coq: `trinity-clara/proofs/igla/gf16_precision.v::gf16_safe_domain` (Admitted —
//!   awaits coq-interval for the φ⁻⁶ numeric bound).
//! - Falsification (R8): `gf16_falsification_witness: gf16_safe 255 true = false`.
//! - JSON: `assertions/igla_assertions.json::INV-3` (`runtime_check.action="abort"`).
//!
//! L-R14: every literal in this file traces back to `invariants.rs` (which holds
//! the L5-owned anchors `INV3_D_MODEL_MIN`, `INV3_ERROR_BOUND`, `PHI`) and via
//! `assertions/igla_assertions.json::INV-3.numeric_anchor` to the .v file.
//!
//! R6 (lane ownership): this lane owns ONLY `gf16.rs`. We do **not** add a new
//! variant to `invariants.rs::InvError`; we expose our own closed enum
//! [`Gf16Error`] and let callers convert at the boundary if/when they need to.
//!
//! R8 falsification witnesses (≥3, mirrored as `#[test]` below):
//! 1. `d_model = 255` → `Err(DModelBelowFloor { d_model: 255 })`
//! 2. `error = PHI_INV_5_PLUS_EPS` (≈ 0.09017, > φ⁻⁶) → `Err(ErrorAboveBound { .. })`
//! 3. `error = f64::NAN` → `Err(NonFiniteError)`
//!
//! Dual-band policy (per JSON `bands` field, INV-3):
//! - `empirical_band`: BENCH-004b (97.67% MNIST = f32 baseline, guarantee_ratio 55).
//! - `certified_band`: φ⁻⁶ ≈ 0.0557 (Admitted, pending coq-interval).
//!
//! These bands MUST NOT be merged. We surface only the certified bound at
//! runtime — the empirical band is exposed only via JSON for offline audit.

use crate::invariants::{INV3_D_MODEL_MIN, INV3_ERROR_BOUND};
use std::fmt;

// ---------------------------------------------------------------------------
// Algebraic anchors (all literals carry a Coq trail)
// ---------------------------------------------------------------------------

/// GF(16) field — `2^4` elements (one zero + 15 nonzero generators).
/// Coq: `lucas_closure_gf16.v::gf16_field_size` (Proven).
pub const GF16_FIELD_BITS: u32 = 4;

/// `2^GF16_FIELD_BITS = 16` total elements in the field, including zero.
/// Coq: `lucas_closure_gf16.v::gf16_field_size` (Proven).
pub const GF16_FIELD_SIZE: usize = 1usize << GF16_FIELD_BITS;

/// The Lucas-closure floor on `d_model` for safe quantization.
/// Coq: `gf16_precision.v::gf16_safe_domain` (Admitted).
/// Mirrors `invariants.rs::INV3_D_MODEL_MIN = 256` so a single edit on the L5
/// anchor propagates here without duplicating the literal.
pub const D_MODEL_MIN: usize = INV3_D_MODEL_MIN;

/// Certified upper bound on quantization error: φ⁻⁶ ≈ 0.0557.
/// Coq: `gf16_precision.v::gf16_end_to_end_error_bound` (Admitted, pending
/// coq-interval).
/// Mirrors `invariants.rs::INV3_ERROR_BOUND` (= `PHI_INV_6`).
pub const ERROR_BOUND_CERTIFIED: f64 = INV3_ERROR_BOUND;

// ---------------------------------------------------------------------------
// Closed error enum
// ---------------------------------------------------------------------------

/// Closed enum of GF(16) safety violations.
///
/// Per R6 we do **not** extend `invariants.rs::InvError`; callers that want a
/// uniform error type over the whole IGLA stack should add `From<Gf16Error>`
/// implementations in their own module.
#[derive(Debug, Clone, PartialEq)]
pub enum Gf16Error {
    /// `d_model` is below the Lucas-closure floor (`< 256`).
    /// Action per JSON INV-3: **abort**.
    DModelBelowFloor { d_model: usize, floor: usize },
    /// Observed quantization error exceeds the certified φ⁻⁶ bound.
    /// Action per JSON INV-3: **abort**.
    ErrorAboveBound { observed: f64, certified_bound: f64 },
    /// Error is NaN, +∞, or -∞ — meaningless for the band check.
    /// Action: **abort** (caller should re-instrument upstream).
    NonFiniteError,
}

impl fmt::Display for Gf16Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Gf16Error::DModelBelowFloor { d_model, floor } => write!(
                f,
                "INV-3: d_model={d_model} < {floor} (Lucas-closure floor); GF16 quantization unsafe"
            ),
            Gf16Error::ErrorAboveBound {
                observed,
                certified_bound,
            } => write!(
                f,
                "INV-3: GF16 error={observed:.6} > φ⁻⁶ ≈ {certified_bound:.6} (certified bound)"
            ),
            Gf16Error::NonFiniteError => {
                write!(f, "INV-3: GF16 error is NaN/±∞ — invalid measurement")
            }
        }
    }
}

impl std::error::Error for Gf16Error {}

// ---------------------------------------------------------------------------
// Public guards
// ---------------------------------------------------------------------------

/// Validate that the model dimension is ≥ the Lucas-closure floor for GF(16).
///
/// Coq: `gf16_precision.v::gf16_safe_domain` (Admitted).
/// Falsification: `gf16_falsification_witness: gf16_safe 255 true = false`.
pub fn check_d_model(d_model: usize) -> Result<(), Gf16Error> {
    if d_model < D_MODEL_MIN {
        Err(Gf16Error::DModelBelowFloor {
            d_model,
            floor: D_MODEL_MIN,
        })
    } else {
        Ok(())
    }
}

/// Validate that an observed quantization error is finite and below the
/// certified φ⁻⁶ bound.
///
/// Coq: `gf16_precision.v::gf16_end_to_end_error_bound` (Admitted, pending
/// coq-interval).
pub fn check_error(error: f64) -> Result<(), Gf16Error> {
    if !error.is_finite() {
        return Err(Gf16Error::NonFiniteError);
    }
    if error > ERROR_BOUND_CERTIFIED {
        return Err(Gf16Error::ErrorAboveBound {
            observed: error,
            certified_bound: ERROR_BOUND_CERTIFIED,
        });
    }
    Ok(())
}

/// Composite guard — both `d_model` and `error` must satisfy INV-3.
///
/// This is the function callers should use at every checkpoint commit when
/// running with `use_gf16=true`. It returns `Ok(())` only when both predicates
/// hold; otherwise the **first** failing condition is returned.
pub fn gf16_safe(d_model: usize, error: f64) -> Result<(), Gf16Error> {
    check_d_model(d_model)?;
    check_error(error)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests — falsification witnesses + anchor guards
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ----- L-R14 anchor guards -----

    #[test]
    fn anchor_d_model_min_matches_invariants() {
        // Trail: gf16.rs::D_MODEL_MIN ↔ invariants.rs::INV3_D_MODEL_MIN
        // ↔ assertions/igla_assertions.json::INV-3.numeric_anchor.d_model_min = 256
        // ↔ trinity-clara/proofs/igla/gf16_precision.v::gf16_safe_domain
        assert_eq!(D_MODEL_MIN, 256);
    }

    #[test]
    fn anchor_error_bound_is_phi_inv_6() {
        // φ⁻⁶ = 1/φ⁶; verify within numerical tolerance using the L5-owned PHI.
        let phi_pow_6 = crate::invariants::PHI.powi(6);
        let phi_inv_6 = 1.0 / phi_pow_6;
        assert!(
            (ERROR_BOUND_CERTIFIED - phi_inv_6).abs() < 1e-3,
            "ERROR_BOUND_CERTIFIED={ERROR_BOUND_CERTIFIED} != φ⁻⁶ ≈ {phi_inv_6}"
        );
    }

    #[test]
    fn anchor_field_size_is_16() {
        // Coq: lucas_closure_gf16.v::gf16_field_size
        assert_eq!(GF16_FIELD_SIZE, 16);
        assert_eq!(GF16_FIELD_BITS, 4);
    }

    // ----- Falsification witnesses (R8) -----

    #[test]
    fn falsification_d_model_255_rejected() {
        // Mirrors Coq: gf16_falsification_witness: gf16_safe 255 true = false.
        let r = check_d_model(255);
        assert_eq!(
            r,
            Err(Gf16Error::DModelBelowFloor {
                d_model: 255,
                floor: 256
            })
        );
    }

    #[test]
    fn falsification_error_above_phi_inv_5_rejected() {
        // φ⁻⁵ ≈ 0.09017 > φ⁻⁶ ≈ 0.0557 — must reject.
        let above = 1.0 / crate::invariants::PHI.powi(5);
        let r = check_error(above);
        match r {
            Err(Gf16Error::ErrorAboveBound {
                observed,
                certified_bound,
            }) => {
                assert!((observed - above).abs() < 1e-12);
                assert!((certified_bound - ERROR_BOUND_CERTIFIED).abs() < 1e-12);
            }
            other => panic!("expected ErrorAboveBound, got {other:?}"),
        }
    }

    #[test]
    fn falsification_nan_error_rejected() {
        assert_eq!(check_error(f64::NAN), Err(Gf16Error::NonFiniteError));
    }

    #[test]
    fn falsification_pos_inf_error_rejected() {
        assert_eq!(check_error(f64::INFINITY), Err(Gf16Error::NonFiniteError));
    }

    #[test]
    fn falsification_neg_inf_error_rejected() {
        assert_eq!(
            check_error(f64::NEG_INFINITY),
            Err(Gf16Error::NonFiniteError)
        );
    }

    #[test]
    fn falsification_d_model_zero_rejected() {
        assert!(matches!(
            check_d_model(0),
            Err(Gf16Error::DModelBelowFloor { d_model: 0, .. })
        ));
    }

    // ----- Boundary acceptance -----

    #[test]
    fn boundary_d_model_at_floor_accepted() {
        assert!(check_d_model(D_MODEL_MIN).is_ok());
    }

    #[test]
    fn boundary_d_model_above_floor_accepted() {
        assert!(check_d_model(D_MODEL_MIN + 1).is_ok());
        assert!(check_d_model(384).is_ok());
        assert!(check_d_model(1024).is_ok());
    }

    #[test]
    fn boundary_error_at_certified_bound_accepted() {
        // Strict `>` in check_error means equal-to-bound is accepted.
        assert!(check_error(ERROR_BOUND_CERTIFIED).is_ok());
    }

    #[test]
    fn boundary_error_below_certified_bound_accepted() {
        assert!(check_error(0.0).is_ok());
        assert!(check_error(ERROR_BOUND_CERTIFIED - 1e-6).is_ok());
        assert!(check_error(0.01).is_ok());
    }

    #[test]
    fn boundary_negative_error_accepted_as_nonsense_but_finite() {
        // Negative quantization error is physically nonsense but finite, so
        // we let it through (caller's responsibility to bound below). This
        // documents the intentional asymmetry: the guard protects the upper
        // tail (where INV-3 lives), not the lower one.
        assert!(check_error(-1.0).is_ok());
    }

    // ----- Composite gf16_safe -----

    #[test]
    fn composite_safe_champion_config() {
        // Trail: INV3_D_MODEL_MIN=256, error well below φ⁻⁶.
        assert!(gf16_safe(256, 0.01).is_ok());
        assert!(gf16_safe(384, 0.05).is_ok());
    }

    #[test]
    fn composite_rejects_d_model_first() {
        // When BOTH predicates fail, the d_model error wins (returned first).
        let r = gf16_safe(128, 1.0);
        assert!(matches!(
            r,
            Err(Gf16Error::DModelBelowFloor { d_model: 128, .. })
        ));
    }

    #[test]
    fn composite_rejects_error_when_d_model_ok() {
        let r = gf16_safe(512, 0.5);
        assert!(matches!(r, Err(Gf16Error::ErrorAboveBound { .. })));
    }

    // ----- Display -----

    #[test]
    fn error_display_d_model_floor() {
        let s = format!(
            "{}",
            Gf16Error::DModelBelowFloor {
                d_model: 64,
                floor: 256
            }
        );
        assert!(s.contains("d_model=64"));
        assert!(s.contains("256"));
        assert!(s.contains("INV-3"));
    }

    #[test]
    fn error_display_above_bound() {
        let s = format!(
            "{}",
            Gf16Error::ErrorAboveBound {
                observed: 0.1,
                certified_bound: ERROR_BOUND_CERTIFIED
            }
        );
        assert!(s.contains("INV-3"));
        assert!(s.contains("φ⁻⁶"));
    }

    #[test]
    fn error_display_non_finite() {
        let s = format!("{}", Gf16Error::NonFiniteError);
        assert!(s.contains("INV-3"));
        assert!(s.contains("NaN"));
    }
}
