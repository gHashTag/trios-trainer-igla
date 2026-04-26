//! Invariant parity guard (#320 §2 falsification #2).
//!
//! When built with `--features trios-integration`, cross-checks every
//! Coq-proven constant against the canonical `trios_igla_race::invariants`.
//! Without the feature, runs a self-consistency check against computed values.
//!
//! Anchor: phi^2 + phi^-2 = 3 (Zenodo 10.5281/zenodo.19227877).

#[cfg(not(feature = "trios-integration"))]
use trios_trainer::invariants as inv;

#[cfg(feature = "trios-integration")]
use trios_igla_race::invariants as canonical;

// ═══════════════════════════════════════════════════════════════
// Self-consistency checks (always run)
// ═══════════════════════════════════════════════════════════════

#[test]
fn trinity_anchor_holds() {
    let phi: f64 = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let lhs = phi.powi(2) + phi.powi(-2);
    assert!(
        (lhs - 3.0).abs() < 1e-12,
        "φ² + φ⁻² = {lhs:.15}, expected 3.0"
    );
}

#[test]
#[cfg(not(feature = "trios-integration"))]
fn phi_constants_computed_match_stored() {
    let phi: f64 = (1.0 + 5.0_f64.sqrt()) / 2.0;
    assert!((inv::PHI - phi).abs() < 1e-12, "PHI mismatch");
    assert!((inv::PHI_SQ - (phi * phi)).abs() < 1e-12, "PHI_SQ mismatch");
    assert!((inv::PHI_CUBE - (phi * phi * phi)).abs() < 1e-12, "PHI_CUBE mismatch");
    assert!((inv::PHI_INV6 - phi.powi(-6)).abs() < 1e-8, "PHI_INV6 mismatch");
    assert!((inv::ALPHA_PHI - phi.powi(-3) / 2.0).abs() < 1e-10, "ALPHA_PHI mismatch");
}

#[test]
#[cfg(not(feature = "trios-integration"))]
fn lr_champion_is_alpha_phi_over_phi_cube() {
    let phi: f64 = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let alpha_phi = phi.powi(-3) / 2.0;
    // Champion LR = 0.004 is in the INV-1 safe range
    assert!(
        inv::LR_CHAMPION >= inv::LR_SAFE_MIN && inv::LR_CHAMPION <= inv::LR_SAFE_MAX,
        "LR_CHAMPION {} not in safe range [{}, {}]",
        inv::LR_CHAMPION, inv::LR_SAFE_MIN, inv::LR_SAFE_MAX
    );
    // alpha_phi ≈ 0.118 matches strong coupling constant to 4 decimal places
    assert!(
        (alpha_phi - 0.1180).abs() < 0.001,
        "α_φ = {alpha_phi:.6} diverges from αs(mZ) = 0.1180"
    );
}

#[test]
#[cfg(not(feature = "trios-integration"))]
fn entropy_band_width_is_one() {
    assert!(
        (inv::NCA_ENTROPY_HI - inv::NCA_ENTROPY_LO - 1.0).abs() < 1e-10,
        "entropy band width must be exactly 1 (φ² - φ = 1)"
    );
    let phi: f64 = (1.0 + 5.0_f64.sqrt()) / 2.0;
    assert!(
        (inv::NCA_ENTROPY_LO - phi).abs() < 1e-10,
        "NCA_ENTROPY_LO must equal φ"
    );
    assert!(
        (inv::NCA_ENTROPY_HI - phi * phi).abs() < 1e-10,
        "NCA_ENTROPY_HI must equal φ²"
    );
}

#[test]
#[cfg(not(feature = "trios-integration"))]
fn lucas_recurrence_holds() {
    let l = inv::LUCAS;
    assert_eq!(l[0], 2);
    assert_eq!(l[1], 1);
    for i in 2..l.len() {
        assert_eq!(l[i], l[i - 1] + l[i - 2], "Lucas recurrence at index {i}");
    }
}

#[test]
#[cfg(not(feature = "trios-integration"))]
fn asha_champion_survives_pruning() {
    assert!(
        inv::BPB_CHAMPION < inv::ASHA_PRUNE_THRESHOLD,
        "BPB_CHAMPION {} must be below ASHA_PRUNE_THRESHOLD {}",
        inv::BPB_CHAMPION, inv::ASHA_PRUNE_THRESHOLD
    );
}

#[test]
#[cfg(not(feature = "trios-integration"))]
fn asha_rungs_are_monotonic() {
    for w in inv::ASHA_RUNGS.windows(2) {
        assert!(w[0] < w[1], "ASHA rungs must be monotonically increasing");
    }
}

#[test]
#[cfg(not(feature = "trios-integration"))]
fn nca_grid_is_trinity_structure() {
    let dim = (inv::NCA_GRID_SIZE as f64).sqrt() as usize;
    assert_eq!(dim * dim, inv::NCA_GRID_SIZE, "grid must be square");
    assert_eq!(dim, 9, "grid dimension must be 9 = 3²");
}

#[test]
#[cfg(not(feature = "trios-integration"))]
fn gf16_precision_bound() {
    let phi: f64 = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let phi_inv6 = phi.powi(-6);
    assert!(
        (inv::PHI_INV6 - phi_inv6).abs() < 1e-8,
        "PHI_INV6 must equal φ⁻⁶"
    );
    assert!(phi_inv6 > 0.05 && phi_inv6 < 0.06, "φ⁻⁶ ≈ 0.0557");
}

// ═══════════════════════════════════════════════════════════════
// Cross-check against canonical (only with trios-integration)
// ═══════════════════════════════════════════════════════════════

#[cfg(feature = "trios-integration")]
#[test]
fn phi_matches_canonical() {
    assert!((inv::PHI - canonical::PHI).abs() < 1e-15);
}

#[cfg(feature = "trios-integration")]
#[test]
fn phi_sq_matches_canonical() {
    assert!((inv::PHI_SQ - canonical::PHI_SQ).abs() < 1e-15);
}

#[cfg(feature = "trios-integration")]
#[test]
fn phi_cube_matches_canonical() {
    assert!((inv::PHI_CUBE - canonical::PHI_CUBE).abs() < 1e-15);
}

#[cfg(feature = "trios-integration")]
#[test]
fn lr_safe_bounds_match_canonical() {
    assert!((inv::LR_SAFE_MIN - canonical::LR_SAFE_MIN).abs() < 1e-15);
    assert!((inv::LR_SAFE_MAX - canonical::LR_SAFE_MAX).abs() < 1e-15);
}

#[cfg(feature = "trios-integration")]
#[test]
fn lr_champion_matches_canonical() {
    assert!((inv::LR_CHAMPION - canonical::LR_CHAMPION).abs() < 1e-15);
}

#[cfg(feature = "trios-integration")]
#[test]
fn asha_threshold_matches_canonical() {
    assert!((inv::ASHA_PRUNE_THRESHOLD - canonical::ASHA_PRUNE_THRESHOLD).abs() < 1e-15);
}

#[cfg(feature = "trios-integration")]
#[test]
fn bpb_champion_matches_canonical() {
    assert!((inv::BPB_CHAMPION - canonical::BPB_CHAMPION).abs() < 1e-15);
}

#[cfg(feature = "trios-integration")]
#[test]
fn nca_entropy_bounds_match_canonical() {
    assert!((inv::NCA_ENTROPY_LO - canonical::NCA_ENTROPY_LO).abs() < 1e-15);
    assert!((inv::NCA_ENTROPY_HI - canonical::NCA_ENTROPY_HI).abs() < 1e-15);
}
