//! GF24 — Golden Float 24-bit format
//!
//! 8 exponent bits + 15 mantissa bits (8:15 split)
//! Medium precision between GF16 and GF32
//! φ-optimized: 8:15 ≈ 0.533 (balanced for 24 bits)

use super::phi_constants::*;

/// GF24 format: 1 sign bit, 8 exp bits, 15 mantissa bits
/// Total: 24 bits = 32768 mantissa values × 255 exponent values × 2 signs ≈ 16M representable values
pub struct GF24 {
    bits: u32,
}

impl GF24 {
    const SIGN_BIT: u32 = 0x00800000; // bit 23
    const EXP_MASK: u32 = 0x007F8000; // bits 22:15
    const MANT_MASK: u32 = 0x00007FFF; // bits 14:0

    const EXP_BITS: u8 = 8;
    const MANT_BITS: u8 = 15;

    const EXP_BIAS: i16 = 127; // 2^(8-1) - 1 = 127

    /// Create GF24 from f32 (quantization)
    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Self { bits: 0 };
        }

        let sign = if value < 0.0 { 1u32 } else { 0u32 };
        let abs_val = value.abs();

        // Handle infinity and NaN
        if !abs_val.is_finite() {
            if abs_val.is_nan() {
                return Self {
                    bits: sign << 23 | 0x007FFFFF,
                };
            }
            return Self {
                bits: sign << 23 | 0x007F8000,
            };
        }

        // Extract exponent and mantissa from f32
        let bits = abs_val.to_bits();
        let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let f32_mant = bits & 0x007FFFFF;

        // Scale to GF24 format
        let mut gf24_exp = f32_exp + Self::EXP_BIAS as i32;

        // Clamp exponent
        if gf24_exp < 0 {
            return Self { bits: 0 };
        }
        if gf24_exp >= 255 {
            return Self {
                bits: sign << 23 | (254 << 15) | 32767,
            };
        }

        // Round mantissa: 23 bits → 15 bits (8 bits removed)
        let shift = 23 - Self::MANT_BITS;
        let mut mant_rounded = (f32_mant >> shift) as u32;
        let remainder = (f32_mant >> (shift - 1)) & 1;

        // φ-weighted rounding: bias = φ * 0.5 ≈ 0.809
        if remainder == 1 {
            let lower_bits = f32_mant & ((1u32 << shift) - 1);
            let phi_threshold = (0.809 * (1u32 << shift) as f64) as u32;
            if lower_bits >= phi_threshold {
                mant_rounded += 1;
            }
        }

        // Handle mantissa overflow
        if mant_rounded >= 32768 {
            mant_rounded = 0;
            gf24_exp += 1;
            if gf24_exp >= 255 {
                return Self {
                    bits: sign << 23 | (254 << 15) | 32767,
                };
            }
        }

        Self {
            bits: sign << 23 | ((gf24_exp as u32) << 15) | mant_rounded,
        }
    }

    /// Convert GF24 to f32
    pub fn to_f32(self) -> f32 {
        if self.bits == 0 {
            return 0.0;
        }

        let sign = if (self.bits & Self::SIGN_BIT) != 0 {
            -1.0f32
        } else {
            1.0f32
        };
        let exp = ((self.bits & Self::EXP_MASK) >> 15) as i32;
        let mant = (self.bits & Self::MANT_MASK) as u32;

        // Check for infinity/NaN
        if exp == 255 {
            return if mant == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            };
        }

        // Reconstruct value: exp == 0 represents 2^(-127), not zero!
        let exp_val = 2.0_f32.powi(exp - Self::EXP_BIAS as i32);
        let mant_val = 1.0 + (mant as f32) / 32768.0;

        sign * exp_val * mant_val
    }

    /// Get raw bits
    pub fn bits(self) -> u32 {
        self.bits
    }

    /// Create from raw bits
    pub fn from_bits(bits: u32) -> Self {
        Self { bits: bits & 0x00FFFFFF }
    }

    /// Get the quantization error vs f32
    pub fn quant_error_f32(self, original: f32) -> f32 {
        let decoded = self.to_f32();
        (decoded - original).abs()
    }

    /// Get relative error
    pub fn relative_error_f32(self, original: f32) -> f32 {
        if original.abs() < f32::MIN_POSITIVE {
            return self.quant_error_f32(original);
        }
        self.quant_error_f32(original) / original.abs()
    }

    /// Range of representable values
    pub const MIN_POSITIVE: f32 = 1.1754943508222875e-38; // 2^(-126) (same as f32)
    pub const MAX: f32 = 65535.5; // (1 + 32767/32768) × 2^127 (clamped by f32)
}

impl Clone for GF24 {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for GF24 {}

impl std::fmt::Debug for GF24 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GF24(0b{:024b} -> {})", self.bits, self.to_f32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let gf24 = GF24::from_f32(0.0);
        assert_eq!(gf24.bits(), 0);
        assert_eq!(gf24.to_f32(), 0.0);
    }

    #[test]
    fn test_positive_one() {
        let gf24 = GF24::from_f32(1.0);
        let decoded = gf24.to_f32();
        assert!((decoded - 1.0).abs() < f32::EPSILON);
        assert!(!decoded.is_nan());
    }

    #[test]
    fn test_negative_one() {
        let gf24 = GF24::from_f32(-1.0);
        let decoded = gf24.to_f32();
        assert!((decoded + 1.0).abs() < f32::EPSILON);
        assert!(!decoded.is_nan());
    }

    #[test]
    fn test_phi_high_precision() {
        let gf24 = GF24::from_f32(PHI as f32);
        let decoded = gf24.to_f32();
        let err = gf24.quant_error_f32(PHI as f32);
        // GF24 has 15 mantissa bits, should be very close to f32
        assert!(err < 0.001, "GF24 phi error too high: {}", err);
        assert!(!decoded.is_nan());
    }

    #[test]
    fn test_phi_squared_high_precision() {
        let gf24 = GF24::from_f32(PHI_SQUARED as f32);
        let decoded = gf24.to_f32();
        let err = gf24.quant_error_f32(PHI_SQUARED as f32);
        assert!(err < 0.001);
        assert!(!decoded.is_nan());
    }

    #[test]
    fn test_phi_cubed_high_precision() {
        let gf24 = GF24::from_f32(PHI_CUBED as f32);
        let decoded = gf24.to_f32();
        let err = gf24.quant_error_f32(PHI_CUBED as f32);
        assert!(err < 0.001);
        assert!(!decoded.is_nan());
    }

    #[test]
    fn test_phi_sqrt_high_precision() {
        let gf24 = GF24::from_f32(PHI_SQRT as f32);
        let decoded = gf24.to_f32();
        let err = gf24.quant_error_f32(PHI_SQRT as f32);
        assert!(err < 0.001);
        assert!(!decoded.is_nan());
    }

    #[test]
    fn test_trinity_identity_preservation() {
        let phi_sq = GF24::from_f32(PHI_SQUARED as f32).to_f32() as f64;
        let phi_inv_sq = GF24::from_f32(PHI_INVERSE_SQUARED as f32).to_f32() as f64;

        // Check for NaN
        assert!(!phi_sq.is_nan());
        assert!(!phi_inv_sq.is_nan());

        let sum = phi_sq + phi_inv_sq;
        let error = (sum - 3.0).abs();

        // GF24 has high precision, should be very close
        assert!(error < 0.001, "Trinity identity error in GF24: {}", error);
    }

    #[test]
    fn test_fibonacci_dims_high_precision() {
        // GF24 preserves integers exactly when mantissa bits suffice. For 15 mantissa bits,
        // precision is 2^E/32768. To preserve integers, need 2^E < 32768, i.e., E < 15, i.e., values < 16384
        let fib_nums = [
            1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584,
            4181, 6765, 10946,
        ];
        for f in fib_nums {
            let gf24 = GF24::from_f32(f as f32);
            let decoded = gf24.to_f32();
            // GF24 should preserve small integers exactly
            if f < 16384 {
                assert_eq!(decoded as i32, f);
            }
            let err = gf24.quant_error_f32(f as f32);
            assert!(err < 0.001);
            assert!(!decoded.is_nan());
        }
    }

    #[test]
    fn test_split_ratio() {
        // GF24 uses 8:15 split ≈ 0.533
        let ratio = GF24::EXP_BITS as f64 / GF24::MANT_BITS as f64;
        let expected = 8.0 / 15.0;
        assert!((ratio - expected).abs() < 1e-10);
        // Should be reasonably close to 1/φ
        let phi_diff = (ratio - PHI_CONJUGATE).abs();
        assert!(phi_diff < 0.1, "Ratio too far from 1/φ: {}", phi_diff);
    }

    #[test]
    fn test_round_trip_accuracy() {
        let values = [
            0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, PHI as f32, 10.0, 100.0, 1000.0,
            10000.0,
        ];
        for v in values {
            let gf24 = GF24::from_f32(v);
            let decoded = gf24.to_f32();
            let rel_err = gf24.relative_error_f32(v);
            assert!(!decoded.is_nan(), "NaN for value {}", v);
            assert!(rel_err < 0.001, "Round-trip error for {}: {}", v, rel_err);
        }
    }

    #[test]
    fn test_exp_bias() {
        assert_eq!(GF24::EXP_BIAS, 127);
    }

    #[test]
    fn test_bits_layout() {
        let gf24 = GF24::from_f32(-2.0);
        let bits = gf24.bits();
        assert_ne!(bits & GF24::SIGN_BIT, 0, "Sign bit should be set");

        let positive = GF24::from_f32(2.0);
        assert_eq!(positive.bits() & GF24::SIGN_BIT, 0);
    }

    #[test]
    fn test_comparison_with_f32() {
        let test_values = [
            0.001, 0.01, 0.1, 1.0, 10.0, PHI as f32, PHI_SQUARED as f32, 100.0,
            1000.0, 10000.0,
        ];
        for v in test_values {
            let gf24 = GF24::from_f32(v);
            let decoded = gf24.to_f32();
            let abs_err = gf24.quant_error_f32(v);
            let rel_err = gf24.relative_error_f32(v);
            assert!(!decoded.is_nan(), "NaN for value {}", v);
            assert!(
                abs_err < v.abs() * 0.001,
                "Absolute error too high for {}: {}",
                v,
                abs_err
            );
            assert!(
                rel_err < 0.001,
                "Relative error too high for {}: {}",
                v,
                rel_err
            );
        }
    }

    #[test]
    fn test_mantissa_precision() {
        assert_eq!(GF24::MANT_BITS, 15);
        assert_eq!(GF24::EXP_BITS, 8);
        assert_eq!(GF24::MANT_BITS + GF24::EXP_BITS + 1, 24); // +1 for sign
    }

    #[test]
    fn test_special_values() {
        let inf = GF24::from_f32(f32::INFINITY);
        assert!(inf.to_f32().is_infinite());

        let neg_inf = GF24::from_f32(f32::NEG_INFINITY);
        assert!(neg_inf.to_f32().is_infinite());
        assert!(neg_inf.to_f32() < 0.0);

        let nan = GF24::from_f32(f32::NAN);
        assert!(nan.to_f32().is_nan());
    }

    #[test]
    fn test_phi_family_consistency() {
        let constants = [
            (PHI, "PHI"),
            (PHI_CONJUGATE, "PHI_CONJUGATE"),
            (PHI_SQUARED, "PHI_SQUARED"),
            (PHI_INVERSE_SQUARED, "PHI_INVERSE_SQUARED"),
            (PHI_CUBED, "PHI_CUBED"),
            (PHI_SQRT, "PHI_SQRT"),
            (PHI_LN, "PHI_LN"),
        ];

        for (val, name) in constants {
            let gf24 = GF24::from_f32(val as f32);
            let decoded = gf24.to_f32();
            assert!(!decoded.is_nan(), "{} produced NaN", name);
            let rel_err = gf24.relative_error_f32(val as f32);
            assert!(rel_err < 0.001, "{} error too high: {}", name, rel_err);
        }
    }

    #[test]
    fn test_phi_power_series() {
        // Verify φ^n = F_n * φ + F_{n-1} in GF24
        let cases = [
            (1, 1.0, 0.0), // φ^1 = 1*φ + 0
            (2, 1.0, 1.0), // φ^2 = 1*φ + 1
            (3, 2.0, 1.0), // φ^3 = 2*φ + 1
            (4, 3.0, 2.0), // φ^4 = 3*φ + 2
            (5, 5.0, 3.0), // φ^5 = 5*φ + 3
            (6, 8.0, 5.0), // φ^6 = 8*φ + 5
        ];

        for (n, f_n, f_n_minus_1) in cases {
            let phi_n = PHI.powf(n as f64);
            let gf24_n = GF24::from_f32(phi_n as f32).to_f32() as f64;

            let expected = f_n * PHI + f_n_minus_1;
            let error = (gf24_n - expected).abs();
            assert!(error < 0.001, "φ^{} power series error: {}", n, error);
        }
    }

    #[test]
    fn test_24_bit_layout() {
        // Verify we use exactly 24 bits
        let gf24 = GF24::from_f32(1.0);
        assert!(gf24.bits() < 0x01000000, "GF24 should use only 24 bits");
    }

    #[test]
    fn test_no_nan_in_normal_range() {
        // Ensure no NaN for values in normal range
        for i in 1..=65535 {
            let val = (i as f32).ln().exp(); // Use exponential for wider range
            if val.is_finite() {
                let gf24 = GF24::from_f32(val);
                let decoded = gf24.to_f32();
                assert!(!decoded.is_nan(), "NaN for value {}", val);
            }
        }
    }

    #[test]
    fn test_golden_ratio_identities() {
        // Test all φ-related identities in GF24
        let gf_phi = GF24::from_f32(PHI as f32);
        let gf_phi_conj = GF24::from_f32(PHI_CONJUGATE as f32);
        let gf_phi_sq = GF24::from_f32(PHI_SQUARED as f32);
        let gf_phi_inv_sq = GF24::from_f32(PHI_INVERSE_SQUARED as f32);

        // φ * (1/φ) = 1
        let prod = gf_phi.to_f32() * gf_phi_conj.to_f32();
        assert!((prod - 1.0).abs() < 0.001, "φ * 1/φ ≠ 1: {}", prod);

        // φ² - φ - 1 = 0
        let identity = gf_phi_sq.to_f32() - gf_phi.to_f32() - 1.0;
        assert!(identity.abs() < 0.001, "φ² - φ - 1 ≠ 0: {}", identity);

        // φ² + 1/φ² = 3
        let trinity = gf_phi_sq.to_f32() + gf_phi_inv_sq.to_f32();
        assert!(
            (trinity - 3.0).abs() < 0.001,
            "Trinity identity failed: {}",
            trinity
        );

        // φ - 1/φ = 1
        let diff = gf_phi.to_f32() - gf_phi_conj.to_f32();
        assert!((diff - 1.0).abs() < 0.001, "φ - 1/φ ≠ 1: {}", diff);
    }
}
