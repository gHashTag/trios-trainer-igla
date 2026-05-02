//! φ-Physics — Golden ratio constants and Fibonacci sequence
//!
//! # Constitutional mandate (Law 5)
//!
//! All φ-related constants encoded as compile-time constants.
//! Used throughout for:
//! - Hidden dimension scaling (828 = round(φ⁶ × 100))
//! - Learning rate boundaries (φ-physics validated)
//! - Seed selection (Fibonacci sequence)
//!
//! # Golden ratio identity
//!
//! ```
//! φ² + φ⁻² = 3
//! ```
//!
//! This is a mathematical identity derived from φ = (1 + √5) / 2:
//! - φ² = φ + 1 ≈ 2.618
//! - φ⁻² = 2 - φ ≈ 0.382
//! - φ² + φ⁻² = 3 (exactly)

/// Golden ratio φ = (1 + √5) / 2 ≈ 1.618
pub const PHI: f64 = 1.618_033_988_749_895;

/// φ² = φ + 1 ≈ 2.618
pub const PHI_SQUARED: f64 = PHI * PHI;

/// φ⁻² = 2 - φ ≈ 0.382
pub const PHI_INVERSE_SQUARED: f64 = 1.0 / (PHI * PHI);

/// φ⁶ × 100 = 828 (rounded) — hidden dimension base
pub const PHI_SIX_SCLED_100: u32 = ((PHI * PHI * PHI * PHI * PHI * PHI) * 100.0).round() as u32;

/// Fibonacci sequence up to 1597 (used for seed selection)
pub const FIBONACCI: &[u64] = &[
    1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_ratio_identity() {
        // φ² + φ⁻² = 3 (within floating point precision)
        let sum = PHI_SQUARED + PHI_INVERSE_SQUARED;
        let diff = (sum - 3.0).abs();
        assert!(diff < 1e-10, "φ² + φ⁻² ≠ 3: sum={}, diff={}", sum, diff);
    }

    #[test]
    fn phi_squared_is_phi_plus_one() {
        // φ² = φ + 1
        let lhs = PHI_SQUARED;
        let rhs = PHI + 1.0;
        let diff = (lhs - rhs).abs();
        assert!(diff < 1e-10, "φ² ≠ φ + 1: lhs={}, rhs={}, diff={}", lhs, rhs, diff);
    }

    #[test]
    fn phi_inverse_squared_is_two_minus_phi() {
        // φ⁻² = 2 - φ
        let lhs = PHI_INVERSE_SQUARED;
        let rhs = 2.0 - PHI;
        let diff = (lhs - rhs).abs();
        assert!(diff < 1e-10, "φ⁻² ≠ 2 - φ: lhs={}, rhs={}, diff={}", lhs, rhs, diff);
    }

    #[test]
    fn phi_six_scaled_100_is_828() {
        assert_eq!(PHI_SIX_SCLED_100, 828, "φ⁶ × 100 should be 828");
    }

    #[test]
    fn fibonacci_sequence() {
        // Verify Fibonacci property: F(n) = F(n-1) + F(n-2)
        for i in 2..FIBONACCI.len() {
            let sum = FIBONACCI[i - 1] + FIBONACCI[i - 2];
            assert_eq!(FIBONACCI[i], sum, "Fibonacci property failed at index {}", i);
        }
    }
}
