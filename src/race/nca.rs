//! L4 — NCA dual-band entropy validator (INV-4 mirror)
//!
//! Stable entropy `H` of an NCA on the canonical Trinity lattice
//! (`K = 9` states on a `9 × 9 = 81 = 3⁴` grid) lies in the
//! certified band `[φ, φ²]` (width = 1 exactly) per
//! `nca_entropy_band.v::entropy_band_width`.
//!
//! The empirical band `[1.5, 2.8]` (width = 1.3) is the legacy
//! Wave-8.5 G1–G8 sweep range — strictly wider than the certified
//! band, never silently merged with it (proof:
//! `nca_entropy_band.v::empirical_wider_than_certified`).
//!
//! ## Why a dedicated module
//!
//! Past failures the gate exists to refuse:
//!
//! 1. **Band conflation** — code that quietly accepted both
//!    `[1.5, 2.8]` and `[φ, φ²]` as "the band" silently allowed
//!    `H ∈ [1.5, φ)` to count as certified, polluting NCA loss
//!    weighting downstream.
//! 2. **Wrong topology** — entropy reported on a non-`9×9` grid is
//!    not on a Trinity lattice, so the band theorem does not apply
//!    (INV-4 only holds for `K=9, grid=81`).
//! 3. **Non-finite H** — numeric pipeline corruption that previously
//!    fell through to the loss term.
//!
//! ## Coq anchor
//!
//! - `trinity-clara/proofs/igla/nca_entropy_band.v::entropy_band_width`
//!   (Proven, `H_upper - H_lower = 1`)
//! - `entropy_band_numeric_lower` (Proven, `H_lower > 1.6180`)
//! - `entropy_band_numeric_upper` (Proven, `H_upper < 2.6182`)
//! - `k_is_trinity_squared`, `grid_is_trinity_4th` (Proven)
//!
//! ## L-R14 anchors
//!
//! Every numeric below routes through `crate::invariants::INV4_*` —
//! this module declares **zero** new constants.
//!
//! Refs: trios#143 lane L4 · INV-4 · L-R14 · R6 · R8.

use crate::invariants::{
    INV4_ENTROPY_CERTIFIED_HI, INV4_ENTROPY_CERTIFIED_LO, INV4_ENTROPY_EMPIRICAL_HI,
    INV4_ENTROPY_EMPIRICAL_LO, INV4_NCA_GRID, INV4_NCA_K_STATES,
};

/// Which entropy band the caller wants validated.
///
/// `Certified` is the theory-first band `[φ, φ²]` — what the Coq
/// theorem actually proves.  `Empirical` is the wider Wave-8.5 sweep
/// band `[1.5, 2.8]` — kept for backwards-compatibility with legacy
/// runs.  These bands are deliberately exposed as distinct enum
/// variants so callers cannot silently widen "certified" to
/// "empirical" by accident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NcaBandMode {
    /// Theory-first band `[φ, φ²]` — width exactly 1.  Default for
    /// all fresh runs.
    Certified,
    /// Legacy band `[1.5, 2.8]` — width 1.3.  Use only when
    /// reproducing pre-2026 sweeps.
    Empirical,
}

impl NcaBandMode {
    /// Lower edge of this mode's band (inclusive).
    pub fn lower(self) -> f64 {
        match self {
            NcaBandMode::Certified => INV4_ENTROPY_CERTIFIED_LO,
            NcaBandMode::Empirical => INV4_ENTROPY_EMPIRICAL_LO,
        }
    }

    /// Upper edge of this mode's band (inclusive).
    pub fn upper(self) -> f64 {
        match self {
            NcaBandMode::Certified => INV4_ENTROPY_CERTIFIED_HI,
            NcaBandMode::Empirical => INV4_ENTROPY_EMPIRICAL_HI,
        }
    }

    /// Width of this mode's band (`upper - lower`).
    pub fn width(self) -> f64 {
        self.upper() - self.lower()
    }
}

/// Successful validation report.  Constructible only by
/// [`validate_nca_entropy`].
#[derive(Debug, Clone, PartialEq)]
pub struct NcaReport {
    /// The entropy value that was admitted.
    pub entropy: f64,
    /// The band mode that admitted it.
    pub mode: NcaBandMode,
    /// Lower edge of the admitting band (provenance).
    pub band_lower: f64,
    /// Upper edge of the admitting band (provenance).
    pub band_upper: f64,
}

/// Reasons the NCA entropy gate refuses an input.
///
/// `EntropyOutOfBand` and `InvalidGrid` / `InvalidKStates` /
/// `NonFiniteEntropy` correspond to the four falsification clauses in
/// `nca_entropy_band.v`.  `BandsConflated` is a defence-in-depth
/// guard surfaced when a caller passes a synthesized band whose
/// edges no longer match the Coq-derived ones.
#[derive(Debug, Clone, PartialEq)]
pub enum NcaError {
    /// `entropy` is outside the band of the requested mode.
    EntropyOutOfBand {
        entropy: f64,
        mode: NcaBandMode,
        lower: f64,
        upper: f64,
    },
    /// Grid size is not the canonical `81 = 3⁴`.
    InvalidGrid { grid: usize, expected: usize },
    /// `K` (state count) is not the canonical `9 = 3²`.
    InvalidKStates { k: usize, expected: usize },
    /// Entropy is `NaN` or `±∞` — refused before any comparison.
    NonFiniteEntropy { entropy: f64 },
    /// Caller-supplied band edges drift from the Coq-derived ones —
    /// returned by [`assert_bands_distinct`].
    BandsConflated {
        certified_lower: f64,
        empirical_lower: f64,
    },
}

/// Validate a single observed NCA entropy `h` against the requested
/// `mode`'s band, on the canonical Trinity lattice (`K=9, grid=81`).
///
/// Returns `Ok(NcaReport)` only when **all** of the following hold:
///
/// * `h.is_finite()`,
/// * `grid == INV4_NCA_GRID` and `k == INV4_NCA_K_STATES`,
/// * `band_lower ≤ h ≤ band_upper` for the requested mode.
///
/// On the first violation we return the corresponding [`NcaError`]
/// — there is no "score" / soft penalty here; the caller (e.g.
/// the loss term) is free to convert the typed error into a
/// gradient penalty downstream.
pub fn validate_nca_entropy(
    h: f64,
    mode: NcaBandMode,
    grid: usize,
    k: usize,
) -> Result<NcaReport, NcaError> {
    if !h.is_finite() {
        return Err(NcaError::NonFiniteEntropy { entropy: h });
    }
    if grid != INV4_NCA_GRID {
        return Err(NcaError::InvalidGrid {
            grid,
            expected: INV4_NCA_GRID,
        });
    }
    if k != INV4_NCA_K_STATES {
        return Err(NcaError::InvalidKStates {
            k,
            expected: INV4_NCA_K_STATES,
        });
    }
    let lower = mode.lower();
    let upper = mode.upper();
    if h < lower || h > upper {
        return Err(NcaError::EntropyOutOfBand {
            entropy: h,
            mode,
            lower,
            upper,
        });
    }
    Ok(NcaReport {
        entropy: h,
        mode,
        band_lower: lower,
        band_upper: upper,
    })
}

/// Compatibility alias: validate on the canonical Trinity lattice
/// without forcing the caller to repeat `INV4_NCA_GRID` / `_K_STATES`
/// at every call site.  Equivalent to
/// `validate_nca_entropy(h, mode, INV4_NCA_GRID, INV4_NCA_K_STATES)`.
pub fn validate_nca_entropy_canonical(
    h: f64,
    mode: NcaBandMode,
) -> Result<NcaReport, NcaError> {
    validate_nca_entropy(h, mode, INV4_NCA_GRID, INV4_NCA_K_STATES)
}

/// Defence-in-depth: assert the empirical and certified bands are
/// not silently merged.  Mirrors `nca_entropy_band.v::bands_are_distinct`.
///
/// Returns `Err(BandsConflated)` if the lower edges have drifted to
/// equality (which would be the start of an L-R14 violation).
pub fn assert_bands_distinct() -> Result<(), NcaError> {
    if (NcaBandMode::Certified.lower() - NcaBandMode::Empirical.lower()).abs() < f64::EPSILON {
        return Err(NcaError::BandsConflated {
            certified_lower: NcaBandMode::Certified.lower(),
            empirical_lower: NcaBandMode::Empirical.lower(),
        });
    }
    Ok(())
}

// ----------------------------------------------------------------------
// Tests — every #[test] is either a positive admission case or a
// **falsification witness** (R8).
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- admission cases --------------------------------------------

    #[test]
    fn admit_certified_midpoint() {
        let h = (NcaBandMode::Certified.lower() + NcaBandMode::Certified.upper()) / 2.0;
        let rep = validate_nca_entropy_canonical(h, NcaBandMode::Certified).unwrap();
        assert_eq!(rep.mode, NcaBandMode::Certified);
        assert!(rep.entropy >= rep.band_lower && rep.entropy <= rep.band_upper);
    }

    #[test]
    fn admit_certified_lower_edge() {
        let h = NcaBandMode::Certified.lower();
        let _ = validate_nca_entropy_canonical(h, NcaBandMode::Certified)
            .expect("certified lower edge must be admitted");
    }

    #[test]
    fn admit_certified_upper_edge() {
        let h = NcaBandMode::Certified.upper();
        let _ = validate_nca_entropy_canonical(h, NcaBandMode::Certified)
            .expect("certified upper edge must be admitted");
    }

    #[test]
    fn admit_empirical_midpoint() {
        let h = (NcaBandMode::Empirical.lower() + NcaBandMode::Empirical.upper()) / 2.0;
        let _ = validate_nca_entropy_canonical(h, NcaBandMode::Empirical).unwrap();
    }

    // --- falsification: outside band --------------------------------

    #[test]
    fn falsify_below_certified_band() {
        let h = NcaBandMode::Certified.lower() - 1e-3;
        match validate_nca_entropy_canonical(h, NcaBandMode::Certified) {
            Err(NcaError::EntropyOutOfBand { entropy, mode, .. }) => {
                assert_eq!(mode, NcaBandMode::Certified);
                assert!(entropy < NcaBandMode::Certified.lower());
            }
            other => panic!("expected EntropyOutOfBand below, got {other:?}"),
        }
    }

    #[test]
    fn falsify_above_certified_band() {
        let h = NcaBandMode::Certified.upper() + 1e-3;
        assert!(matches!(
            validate_nca_entropy_canonical(h, NcaBandMode::Certified),
            Err(NcaError::EntropyOutOfBand { .. })
        ));
    }

    #[test]
    fn falsify_below_empirical_band() {
        let h = NcaBandMode::Empirical.lower() - 1e-3;
        assert!(matches!(
            validate_nca_entropy_canonical(h, NcaBandMode::Empirical),
            Err(NcaError::EntropyOutOfBand { .. })
        ));
    }

    #[test]
    fn falsify_above_empirical_band() {
        let h = NcaBandMode::Empirical.upper() + 1e-3;
        assert!(matches!(
            validate_nca_entropy_canonical(h, NcaBandMode::Empirical),
            Err(NcaError::EntropyOutOfBand { .. })
        ));
    }

    // --- falsification: wrong topology ------------------------------

    #[test]
    fn falsify_wrong_grid_rejected() {
        let h = (NcaBandMode::Certified.lower() + NcaBandMode::Certified.upper()) / 2.0;
        match validate_nca_entropy(h, NcaBandMode::Certified, 64, INV4_NCA_K_STATES) {
            Err(NcaError::InvalidGrid { grid, expected }) => {
                assert_eq!(grid, 64);
                assert_eq!(expected, INV4_NCA_GRID);
            }
            other => panic!("expected InvalidGrid, got {other:?}"),
        }
    }

    #[test]
    fn falsify_wrong_k_rejected() {
        let h = (NcaBandMode::Certified.lower() + NcaBandMode::Certified.upper()) / 2.0;
        match validate_nca_entropy(h, NcaBandMode::Certified, INV4_NCA_GRID, 8) {
            Err(NcaError::InvalidKStates { k, expected }) => {
                assert_eq!(k, 8);
                assert_eq!(expected, INV4_NCA_K_STATES);
            }
            other => panic!("expected InvalidKStates, got {other:?}"),
        }
    }

    // --- falsification: non-finite ----------------------------------

    #[test]
    fn falsify_non_finite_entropy_rejected() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            match validate_nca_entropy_canonical(bad, NcaBandMode::Certified) {
                Err(NcaError::NonFiniteEntropy { entropy }) => assert!(!entropy.is_finite()),
                other => panic!("expected NonFiniteEntropy for {bad}, got {other:?}"),
            }
        }
    }

    // --- L-R14 anchor pinning ---------------------------------------

    #[test]
    fn certified_width_is_exactly_one() {
        // Coq: nca_entropy_band.v::entropy_band_width  →  φ² - φ = 1
        let w = NcaBandMode::Certified.width();
        assert!((w - 1.0).abs() < 1e-9, "certified width drifted: {w}");
    }

    #[test]
    fn empirical_width_is_one_point_three() {
        // L-R14: legacy sweep band, width derived from invariants.rs
        // empirical-LO=1.5 / HI=2.8.
        let w = NcaBandMode::Empirical.width();
        assert!((w - 1.3).abs() < 1e-9, "empirical width drifted: {w}");
    }

    #[test]
    fn certified_band_inside_empirical_band_lower_distinct() {
        // Empirical band MUST be wider on both ends — never silently
        // merged.  Mirrors Coq theorem `empirical_wider_than_certified`.
        assert!(NcaBandMode::Empirical.lower() < NcaBandMode::Certified.lower());
        assert!(NcaBandMode::Empirical.upper() > NcaBandMode::Certified.upper());
    }

    #[test]
    fn assert_bands_distinct_passes_for_current_constants() {
        assert!(assert_bands_distinct().is_ok());
    }

    #[test]
    fn nca_grid_is_trinity_fourth_power() {
        // Coq: grid_is_trinity_4th — 81 = 3⁴
        assert_eq!(INV4_NCA_GRID, 81);
        assert_eq!(81usize, 3usize.pow(4));
    }

    #[test]
    fn nca_k_states_is_trinity_squared() {
        // Coq: k_is_trinity_squared — 9 = 3²
        assert_eq!(INV4_NCA_K_STATES, 9);
        assert_eq!(9usize, 3usize.pow(2));
    }

    // --- composition with `is_finite` and PartialEq -----------------

    #[test]
    fn report_is_partial_eq_for_diagnostics() {
        let h = NcaBandMode::Certified.lower();
        let r1 = validate_nca_entropy_canonical(h, NcaBandMode::Certified).unwrap();
        let r2 = validate_nca_entropy_canonical(h, NcaBandMode::Certified).unwrap();
        assert_eq!(r1, r2);
    }
}
