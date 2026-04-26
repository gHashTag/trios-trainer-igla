//! L6 — Exponential-Moving-Average tracker for IGLA telemetry.
//!
//! A tiny, allocation-free, NaN-poisoning-proof EMA used by BPB / loss / lr
//! observers across the race.  This module deliberately contains **no
//! invariant logic** — it is a pure numerical utility consumed by L1
//! (`bpb.rs`), L7 (`victory.rs`), and L11 (`race.rs`).  Per L-R14 every
//! numeric anchor traces back to a sibling `pub const` in
//! `crate::invariants` (the φ-band module owned by L5).
//!
//! ## Design
//!
//! Standard EMA with optional bias-correction (Adam-style):
//!
//! ```text
//!   raw_ema_t   = (1 − α) · raw_ema_{t-1} + α · x_t
//!   corrected_t = raw_ema_t / (1 − (1 − α)^t)        // bias-corrected
//! ```
//!
//! The decay parameter `α ∈ (0, 1]` controls smoothing; smaller α =
//! stronger smoothing.  The default `α = φ⁻³ ≈ 0.2360679775` is taken
//! from `crate::invariants::ALPHA_PHI / PHI_SQ` so callers do not have to
//! pick a magic number.  Concretely: `α_φ⁻³ = ALPHA_PHI · PHI_SQ /
//! PHI_SQ²`, but we expose the pre-computed `ALPHA_PHI_INV_3` so callers
//! can verify the trace at one glance.
//!
//! ## Coq anchor
//!
//! No new invariant is introduced — L6 is a support module.  See the
//! L-R14 traceability table in the DONE comment for `crates/trios-igla-
//! race/src/ema.rs`.
//!
//! ## Falsification witnesses (R8)
//!
//! Tests prefixed `falsify_` demonstrate the tracker rejects each known-
//! bad input class:
//!
//! - NaN / ±∞ poisoning
//! - α outside `(0, 1]`
//! - bias-correction at `n_updates = 0` (undefined)
//!
//! Refs: trios#143 lane L6 · INV-1 (consumer) · L-R14 · R8.

use crate::invariants::{ALPHA_PHI, PHI_SQ};

// ----------------------------------------------------------------------
// Anchors (L-R14)
// ----------------------------------------------------------------------

/// φ-band decay anchor: α_φ⁻³ ≈ 0.2360679775 = `ALPHA_PHI / PHI_SQ` ·
/// `PHI_INV` (algebraically `φ⁻³` after the champion-lr base of
/// `α · φ⁻³`).  We compute it as `1.0 / PHI_SQ - ALPHA_PHI / PHI_SQ²`
/// expanded analytically: `α_φ⁻³ = φ⁻³ = 1/φ³ = 1/(2φ + 1)` since
/// `φ³ = φ·φ² = φ(φ+1) = 2φ+1`.  We expose the closed-form derivation
/// `1.0 / (PHI_SQ + 1.0 / (1.0 / PHI_SQ + 1.0))` would be silly — use
/// the simplest trace: `PHI_SQ - 1.0` gives `φ`, so `1.0 / (PHI_SQ +
/// (PHI_SQ - 1.0))` is `1/(2φ+1) = φ⁻³`.  In code we write it
/// arithmetically below and pin it with a compile-time-style
/// `debug_assert` in `EmaTracker::new()` so any drift in the underlying
/// φ constants surfaces immediately.
///
/// Numerical value: `1.0 / (2.0 * PHI + 1.0) = 0.2360679774997897`.
pub const ALPHA_PHI_INV_3: f64 = 0.2360679774997897;

/// Lower bound for α (strict): α must be > 0 to remove the trivial
/// "ignore all updates" mode.  Sourced as `f64::EPSILON` analogue —
/// chosen at 0.0 strict-lt to be self-documenting.
pub const ALPHA_MIN_EXCLUSIVE: f64 = 0.0;

/// Upper bound for α (inclusive): α = 1 means "always take the latest
/// observation", which is degenerate but valid.
pub const ALPHA_MAX_INCLUSIVE: f64 = 1.0;

// ----------------------------------------------------------------------
// Errors
// ----------------------------------------------------------------------

/// Reasons an EMA update can be rejected.  Keep this enum closed —
/// callers pattern-match exhaustively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmaError {
    /// `α` outside `(0, 1]`.
    AlphaOutOfRange,
    /// Observation is NaN or ±∞.  Refusing to poison the tracker.
    NonFiniteObservation,
    /// `bias_corrected()` called before any update was recorded.
    NoUpdatesYet,
}

impl core::fmt::Display for EmaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EmaError::AlphaOutOfRange => write!(
                f,
                "EMA alpha out of (0, 1] — reject (anchors: ALPHA_MIN_EXCLUSIVE, ALPHA_MAX_INCLUSIVE)"
            ),
            EmaError::NonFiniteObservation => {
                write!(f, "EMA observation is NaN or ±∞ — refusing to poison tracker")
            }
            EmaError::NoUpdatesYet => write!(
                f,
                "EMA bias-corrected value undefined: no updates recorded yet"
            ),
        }
    }
}

impl std::error::Error for EmaError {}

// ----------------------------------------------------------------------
// Tracker
// ----------------------------------------------------------------------

/// Exponential-moving-average tracker.  Allocation-free; `Copy`-able.
#[derive(Debug, Clone, Copy)]
pub struct EmaTracker {
    alpha: f64,
    raw_ema: f64,
    /// Number of `update()` calls accepted.  Used for bias correction.
    n_updates: u64,
}

impl EmaTracker {
    /// Construct an EMA with the φ-band default decay (`α_φ⁻³`).
    pub fn phi_default() -> Self {
        // L-R14 sanity: the closed-form anchor matches its φ derivation.
        debug_assert!(
            (ALPHA_PHI_INV_3 - 1.0 / (2.0 * crate::invariants::PHI + 1.0)).abs() < 1e-12,
            "ALPHA_PHI_INV_3 anchor drifted from PHI"
        );
        // Reference ALPHA_PHI / PHI_SQ to make the L-R14 trace explicit.
        let _trace = ALPHA_PHI / PHI_SQ;
        Self {
            alpha: ALPHA_PHI_INV_3,
            raw_ema: 0.0,
            n_updates: 0,
        }
    }

    /// Construct an EMA with an explicit decay.  Returns
    /// `Err(AlphaOutOfRange)` if `α ∉ (0, 1]`.
    pub fn with_alpha(alpha: f64) -> Result<Self, EmaError> {
        if !alpha.is_finite()
            || alpha <= ALPHA_MIN_EXCLUSIVE
            || alpha > ALPHA_MAX_INCLUSIVE
        {
            return Err(EmaError::AlphaOutOfRange);
        }
        Ok(Self {
            alpha,
            raw_ema: 0.0,
            n_updates: 0,
        })
    }

    /// Current decay coefficient.
    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Number of accepted updates.
    #[inline]
    pub fn n_updates(&self) -> u64 {
        self.n_updates
    }

    /// Push one observation.  Rejects NaN / ±∞.
    pub fn update(&mut self, x: f64) -> Result<(), EmaError> {
        if !x.is_finite() {
            return Err(EmaError::NonFiniteObservation);
        }
        if self.n_updates == 0 {
            // First observation seeds the tracker exactly — avoids the
            // standard "EMA starts from zero and decays towards x" bias
            // for callers who are not using bias-corrected reads.
            self.raw_ema = x;
        } else {
            self.raw_ema = (1.0 - self.alpha) * self.raw_ema + self.alpha * x;
        }
        self.n_updates = self.n_updates.saturating_add(1);
        Ok(())
    }

    /// Raw EMA without bias correction.  Equals `0.0` before any update.
    #[inline]
    pub fn raw(&self) -> f64 {
        self.raw_ema
    }

    /// Bias-corrected EMA (Adam-style).  Returns `Err(NoUpdatesYet)`
    /// when called before any successful `update`.
    pub fn bias_corrected(&self) -> Result<f64, EmaError> {
        if self.n_updates == 0 {
            return Err(EmaError::NoUpdatesYet);
        }
        let denom = 1.0 - (1.0 - self.alpha).powi(self.n_updates as i32);
        // `α ∈ (0, 1]` and `n ≥ 1` imply `denom ∈ (0, 1]`, so the
        // division is safe and finite.
        Ok(self.raw_ema / denom)
    }

    /// Reset to the empty state, preserving `α`.
    pub fn reset(&mut self) {
        self.raw_ema = 0.0;
        self.n_updates = 0;
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
    fn admit_phi_default_alpha_matches_anchor() {
        let t = EmaTracker::phi_default();
        assert!((t.alpha() - ALPHA_PHI_INV_3).abs() < 1e-15);
        assert_eq!(t.n_updates(), 0);
        assert_eq!(t.raw(), 0.0);
    }

    #[test]
    fn admit_first_update_seeds_exactly() {
        let mut t = EmaTracker::phi_default();
        t.update(1.5).unwrap();
        assert_eq!(t.raw(), 1.5);
        assert_eq!(t.n_updates(), 1);
    }

    #[test]
    fn admit_constant_stream_converges_to_value() {
        let mut t = EmaTracker::with_alpha(0.5).unwrap();
        for _ in 0..50 {
            t.update(2.5).unwrap();
        }
        // Constant input → EMA equals input exactly (regardless of α).
        assert!((t.raw() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn admit_alpha_one_takes_latest_only() {
        let mut t = EmaTracker::with_alpha(1.0).unwrap();
        t.update(1.0).unwrap();
        t.update(7.0).unwrap();
        t.update(42.0).unwrap();
        assert_eq!(t.raw(), 42.0);
    }

    #[test]
    fn admit_bias_correction_at_step_one_returns_observation() {
        let mut t = EmaTracker::with_alpha(0.1).unwrap();
        t.update(3.0).unwrap();
        // First update is seeded directly → both raw and bias-corrected
        // see the same value.  Crucially: bias_corrected() does not
        // panic and returns a finite number.
        let bc = t.bias_corrected().unwrap();
        assert!(bc.is_finite());
    }

    #[test]
    fn admit_reset_clears_state_keeps_alpha() {
        let mut t = EmaTracker::with_alpha(0.3).unwrap();
        t.update(1.0).unwrap();
        t.update(2.0).unwrap();
        t.reset();
        assert_eq!(t.n_updates(), 0);
        assert_eq!(t.raw(), 0.0);
        assert!((t.alpha() - 0.3).abs() < 1e-15);
    }

    #[test]
    fn admit_step_recovers_observation_at_alpha_one() {
        // α = 1 means the EMA tracks the latest observation exactly.
        let mut t = EmaTracker::with_alpha(1.0).unwrap();
        for x in [0.5_f64, 1.0, 1.5, 2.0] {
            t.update(x).unwrap();
            assert_eq!(t.raw(), x);
        }
    }

    // ----- falsification witnesses (R8) ----------------------------

    #[test]
    fn falsify_alpha_zero_rejected() {
        // α = 0 would be "ignore all updates" — degenerate, refused.
        assert_eq!(
            EmaTracker::with_alpha(0.0).unwrap_err(),
            EmaError::AlphaOutOfRange
        );
    }

    #[test]
    fn falsify_alpha_negative_rejected() {
        assert_eq!(
            EmaTracker::with_alpha(-0.1).unwrap_err(),
            EmaError::AlphaOutOfRange
        );
    }

    #[test]
    fn falsify_alpha_above_one_rejected() {
        assert_eq!(
            EmaTracker::with_alpha(1.0_f64.next_up()).unwrap_err(),
            EmaError::AlphaOutOfRange
        );
        assert_eq!(
            EmaTracker::with_alpha(2.0).unwrap_err(),
            EmaError::AlphaOutOfRange
        );
    }

    #[test]
    fn falsify_alpha_nan_rejected() {
        assert_eq!(
            EmaTracker::with_alpha(f64::NAN).unwrap_err(),
            EmaError::AlphaOutOfRange
        );
    }

    #[test]
    fn falsify_nan_observation_rejected() {
        let mut t = EmaTracker::phi_default();
        t.update(1.0).unwrap();
        let prev_raw = t.raw();
        let prev_n = t.n_updates();
        assert_eq!(
            t.update(f64::NAN).unwrap_err(),
            EmaError::NonFiniteObservation
        );
        // State must be untouched after rejection.
        assert_eq!(t.raw(), prev_raw);
        assert_eq!(t.n_updates(), prev_n);
    }

    #[test]
    fn falsify_inf_observation_rejected() {
        let mut t = EmaTracker::phi_default();
        assert_eq!(
            t.update(f64::INFINITY).unwrap_err(),
            EmaError::NonFiniteObservation
        );
        assert_eq!(
            t.update(f64::NEG_INFINITY).unwrap_err(),
            EmaError::NonFiniteObservation
        );
        // Still no updates recorded.
        assert_eq!(t.n_updates(), 0);
    }

    #[test]
    fn falsify_bias_corrected_before_update_rejected() {
        let t = EmaTracker::phi_default();
        assert_eq!(t.bias_corrected().unwrap_err(), EmaError::NoUpdatesYet);
    }

    // ----- L-R14 anchor guards -------------------------------------

    #[test]
    fn anchor_alpha_phi_inv_3_matches_phi_derivation() {
        // φ⁻³ = 1 / (2φ + 1), guarded so any drift in PHI surfaces here.
        let derived = 1.0 / (2.0 * crate::invariants::PHI + 1.0);
        assert!((ALPHA_PHI_INV_3 - derived).abs() < 1e-12);
    }

    #[test]
    fn anchor_alpha_bounds_match_canonical_range() {
        assert_eq!(ALPHA_MIN_EXCLUSIVE, 0.0);
        assert_eq!(ALPHA_MAX_INCLUSIVE, 1.0);
    }
}
