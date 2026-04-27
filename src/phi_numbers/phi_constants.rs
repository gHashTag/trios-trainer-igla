//! Core Golden Ratio (φ) Constants
//!
//! All constants derived from φ = (1 + √5) / 2 ≈ 1.6180339887498948482...

/// The Golden Ratio φ = (1 + √5) / 2
pub const PHI: f64 = 1.6180339887498948482045868343656381177203091798057628621354486227;

/// Conjugate φ = 1/φ = φ - 1 ≈ 0.6180339887498948482
pub const PHI_CONJUGATE: f64 = 0.6180339887498948482045868343656381177203091798057628621354486227;

/// φ² = φ + 1 ≈ 2.6180339887498948482
pub const PHI_SQUARED: f64 = 2.6180339887498948482045868343656381177203091798057628621354486227;

/// 1/φ² = 2 - φ ≈ 0.3819660112501051518
pub const PHI_INVERSE_SQUARED: f64 =
    0.3819660112501051517954131656343618822826908201942371378645513773;

/// φ - 1/φ = 1.0 (exact identity)
pub const PHI_DIFFERENCE: f64 = 1.0;

/// φ³ = 2φ + 1 ≈ 4.2360679774997896964
pub const PHI_CUBED: f64 = 4.2360679774997896964091736687312762354406183596115257242708972454;

/// √φ ≈ 1.2720196495140689643
pub const PHI_SQRT: f64 = 1.2720196495140689643253831498140330639650285407750715066475066493;

/// ln(φ) ≈ 0.4812118250596034474
pub const PHI_LN: f64 = 0.4812118250596034474344251845978255064174482898100857955306613267;

/// Main Trinity identity: φ² + 1/φ² = 3
pub const TRINITY_IDENTITY: f64 = 3.0;

/// φ-exponent for GF16: 6 bits (derived from optimal 6:9 split)
pub const GF16_EXP_BITS: u8 = 6;

/// φ-mantissa for GF16: 9 bits
pub const GF16_MANT_BITS: u8 = 9;

/// GF16 split ratio: 6/9 ≈ 0.667 (engineering approximation to 1/φ ≈ 0.618)
pub const GF16_SPLIT_RATIO: f64 = 6.0 / 9.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_value() {
        // Verify φ = (1 + √5) / 2
        let computed = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((PHI - computed).abs() < 1e-10, "PHI mismatch");
    }

    #[test]
    fn test_phi_conjugate() {
        // Verify 1/φ = φ - 1
        assert!((PHI_CONJUGATE - (1.0 / PHI)).abs() < 1e-10);
        assert!((PHI_CONJUGATE - (PHI - 1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_phi_squared() {
        // Verify φ² = φ + 1
        assert!((PHI_SQUARED - (PHI + 1.0)).abs() < 1e-10);
        assert!((PHI_SQUARED - (PHI * PHI)).abs() < 1e-10);
    }

    #[test]
    fn test_phi_inverse_squared() {
        // Verify 1/φ² = 2 - φ
        assert!((PHI_INVERSE_SQUARED - (1.0 / (PHI * PHI))).abs() < 1e-10);
        assert!((PHI_INVERSE_SQUARED - (2.0 - PHI)).abs() < 1e-10);
    }

    #[test]
    fn test_phi_difference() {
        // Verify φ - 1/φ = 1
        assert!((PHI_DIFFERENCE - (PHI - PHI_CONJUGATE)).abs() < 1e-10);
    }

    #[test]
    fn test_trinity_identity() {
        // Verify φ² + 1/φ² = 3
        let computed = PHI_SQUARED + PHI_INVERSE_SQUARED;
        assert!(
            (computed - TRINITY_IDENTITY).abs() < 1e-10,
            "Trinity identity failed: {} != 3",
            computed
        );
    }

    #[test]
    fn test_phi_cubed() {
        // Verify φ³ = 2φ + 1
        assert!((PHI_CUBED - (PHI * PHI * PHI)).abs() < 1e-10);
        assert!((PHI_CUBED - (2.0 * PHI + 1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_phi_sqrt() {
        // Verify (√φ)² = φ
        assert!((PHI_SQRT * PHI_SQRT - PHI).abs() < 1e-10);
    }

    #[test]
    fn test_phi_ln() {
        // Verify exp(ln(φ)) = φ
        assert!((PHI_LN.exp() - PHI).abs() < 1e-10);
    }

    #[test]
    fn test_gf16_split_ratio() {
        // Verify 6:9 split ≈ 2/3 (engineering approximation to 1/φ)
        let ratio = GF16_SPLIT_RATIO;
        assert!((ratio - 2.0 / 3.0).abs() < 1e-10);
        // Should be close to but not equal to 1/φ
        let diff = (ratio - PHI_CONJUGATE).abs();
        assert!(diff > 1e-2, "Ratio too close to 1/φ: {}", diff);
        assert!(diff < 1e-1, "Ratio too far from 1/φ: {}", diff);
    }

    #[test]
    fn test_golden_power_series() {
        // Test φ^n follows Fibonacci pattern: φ^n = F_n * φ + F_{n-1}
        // φ^1 = 1*φ + 0 = φ
        assert!((PHI - (1.0 * PHI + 0.0)).abs() < 1e-10);
        // φ^2 = 1*φ + 1 = φ + 1
        assert!((PHI_SQUARED - (1.0 * PHI + 1.0)).abs() < 1e-10);
        // φ^3 = 2*φ + 1
        assert!((PHI_CUBED - (2.0 * PHI + 1.0)).abs() < 1e-10);
    }
}
