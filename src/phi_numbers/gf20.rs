//! GF20 — Golden Float 20-bit format
//!
//! 7 exponent bits + 12 mantissa bits (7:12 split)
//! Low precision for attention weights
//! φ-optimized: 7:12 ≈ 0.583 (closest to 1/φ ≈ 0.618 for 20 bits)

use super::phi_constants::*;

/// GF20 format: 1 sign bit, 7 exp bits, 12 mantissa bits
/// Total: 20 bits = 4096 mantissa values × 127 exponent values × 2 signs ≈ 1M representable values
pub struct GF20 {
    bits: u32,
}

impl GF20 {
    const SIGN_BIT: u32 = 0x00080000; // bit 19
    const EXP_MASK: u32 = 0x0007F000; // bits 18:12
    const MANT_MASK: u32 = 0x00000FFF; // bits 11:0

    const EXP_BITS: u8 = 7;
    const MANT_BITS: u8 = 12;

    const EXP_BIAS: i16 = 63; // 2^(7-1) - 1 = 63

    /// Create GF20 from f32 (quantization)
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
                    bits: sign << 19 | 0x0007FFFF,
                };
            }
            return Self {
                bits: sign << 19 | 0x0007F000,
            };
        }

        // Extract exponent and mantissa from f32
        let bits = abs_val.to_bits();
        let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let f32_mant = bits & 0x007FFFFF;

        // Scale to GF20 format
        let mut gf20_exp = f32_exp + Self::EXP_BIAS as i32;

        // Clamp exponent
        if gf20_exp < 0 {
            return Self { bits: 0 };
        }
        if gf20_exp >= 127 {
            return Self {
                bits: sign << 19 | (126 << 12) | 4095,
            };
        }

        // Round mantissa: 23 bits → 12 bits
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
        if mant_rounded >= 4096 {
            mant_rounded = 0;
            gf20_exp += 1;
            if gf20_exp >= 127 {
                return Self {
                    bits: sign << 19 | (126 << 12) | 4095,
                };
            }
        }

        Self {
            bits: sign << 19 | ((gf20_exp as u32) << 12) | mant_rounded,
        }
    }

    /// Convert GF20 to f32
    pub fn to_f32(self) -> f32 {
        if self.bits == 0 {
            return 0.0;
        }

        let sign = if (self.bits & Self::SIGN_BIT) != 0 {
            -1.0f32
        } else {
            1.0f32
        };
        let exp = ((self.bits & Self::EXP_MASK) >> 12) as i32;
        let mant = (self.bits & Self::MANT_MASK) as u32;

        // Check for infinity/NaN
        if exp == 127 {
            return if mant == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            };
        }

        // Reconstruct value: exp == 0 represents 2^(-63), not zero!
        let exp_val = 2.0_f32.powi(exp - Self::EXP_BIAS as i32);
        let mant_val = 1.0 + (mant as f32) / 4096.0;

        sign * exp_val * mant_val
    }

    /// Get raw bits
    pub fn bits(self) -> u32 {
        self.bits
    }

    /// Create from raw bits
    pub fn from_bits(bits: u32) -> Self {
        Self { bits: bits & 0x000FFFFF }
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
    pub const MIN_POSITIVE: f32 = 1.0842021724855044e-19; // 2^(-63)
    pub const MAX: f32 = 8191.5; // (1 + 4095/4096) × 2^63 (clamped by f32)
}

impl Clone for GF20 {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for GF20 {}

impl std::fmt::Debug for GF20 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GF20(0b{:020b} -> {})", self.bits, self.to_f32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let gf20 = GF20::from_f32(0.0);
        assert_eq!(gf20.bits(), 0);
        assert_eq!(gf20.to_f32(), 0.0);
    }

    #[test]
    fn test_positive_one() {
        let gf20 = GF20::from_f32(1.0);
        let decoded = gf20.to_f32();
        assert!((decoded - 1.0).abs() < f32::EPSILON * 10.0);
        assert!(!decoded.is_nan());
    }

    #[test]
    fn test_negative_one() {
        let gf20 = GF20::from_f32(-1.0);
        let decoded = gf20.to_f32();
        assert!((decoded + 1.0).abs() < f32::EPSILON * 10.0);
        assert!(!decoded.is_nan());
    }

    #[test]
    fn test_phi_high_precision() {
        let gf20 = GF20::from_f32(PHI as f32);
        let decoded = gf20.to_f32();
        let err = gf20.quant_error_f32(PHI as f32);
        // GF20 has 12 mantissa bits, should be reasonably close to f32
        assert!(err < 0.01, "GF20 phi error too high: {}", err);
        assert!(!decoded.is_nan());
    }

    #[test]
    fn test_phi_squared_high_precision() {
        let gf20 = GF20::from_f32(PHI_SQUARED as f32);
        let decoded = gf20.to_f32();
        let err = gf20.quant_error_f32(PHI_SQUARED as f32);
        assert!(err < 0.01);
        assert!(!decoded.is_nan());
    }

    #[test]
    fn test_phi_conjugate_value() {
        let gf20 = GF20::from_f32(PHI_CONJUGATE as f32);
        let decoded = gf20.to_f32();
        let err = gf20.quant_error_f32(PHI_CONJUGATE as f32);
        assert!(err < 0.005, "GF20 phi_conjugate error too high: {}", err);
        assert!(!decoded.is_nan());
    }

    #[test]
    fn test_trinity_identity_preservation() {
        let phi_sq = GF20::from_f32(PHI_SQUARED as f32).to_f32() as f64;
        let phi_inv_sq = GF20::from_f32(PHI_INVERSE_SQUARED as f32).to_f32() as f64;

        // Check for NaN
        assert!(!phi_sq.is_nan());
        assert!(!phi_inv_sq.is_nan());

        let sum = phi_sq + phi_inv_sq;
        let error = (sum - 3.0).abs();

        // GF20 has good precision, should be close
        assert!(error < 0.01, "Trinity identity error in GF20: {}", error);
    }

    #[test]
    fn test_fibonacci_dims_high_precision() {
        let fib_nums = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987];
        for f in fib_nums {
            let gf20 = GF20::from_f32(f as f32);
            let decoded = gf20.to_f32();
            let err = gf20.quant_error_f32(f as f32);
            // GF20 should preserve small integers exactly
            assert_eq!(decoded as i32, f);
            assert_eq!(err, 0.0);
            assert!(!decoded.is_nan());
        }
    }

    #[test]
    fn test_split_ratio() {
        // GF20 uses 7:12 split ≈ 0.583 (close to 1/φ ≈ 0.618)
        let ratio = GF20::EXP_BITS as f64 / GF20::MANT_BITS as f64;
        let expected = 7.0 / 12.0;
        assert!((ratio - expected).abs() < 1e-10);
        // Should be close to 1/φ
        let phi_diff = (ratio - PHI_CONJUGATE).abs();
        assert!(phi_diff < 0.05, "Ratio too far from 1/φ: {}", phi_diff);
    }

    #[test]
    fn test_round_trip_accuracy() {
        let values = [0.001, 0.01, 0.1, 0.5, 1.0, PHI as f32, 10.0, 100.0, 1000.0];
        for v in values {
            let gf20 = GF20::from_f32(v);
            let decoded = gf20.to_f32();
            let rel_err = gf20.relative_error_f32(v);
            assert!(!decoded.is_nan(), "NaN for value {}", v);
            assert!(rel_err < 0.01, "Round-trip error for {}: {}", v, rel_err);
        }
    }

    #[test]
    fn test_exp_bias() {
        assert_eq!(GF20::EXP_BIAS, 63);
    }

    #[test]
    fn test_bits_layout() {
        let gf20 = GF20::from_f32(-2.0);
        let bits = gf20.bits();
        assert_ne!(bits & GF20::SIGN_BIT, 0, "Sign bit should be set");

        let positive = GF20::from_f32(2.0);
        assert_eq!(positive.bits() & GF20::SIGN_BIT, 0);
    }

    #[test]
    fn test_comparison_with_f32() {
        let test_values = [0.001, 0.1, 1.0, 10.0, PHI as f32, PHI_SQUARED as f32, 100.0];
        for v in test_values {
            let gf20 = GF20::from_f32(v);
            let decoded = gf20.to_f32();
            let abs_err = gf20.quant_error_f32(v);
            let rel_err = gf20.relative_error_f32(v);
            assert!(!decoded.is_nan(), "NaN for value {}", v);
            assert!(
                abs_err < v.abs() * 0.01,
                "Absolute error too high for {}: {}",
                v,
                abs_err
            );
            assert!(
                rel_err < 0.01,
                "Relative error too high for {}: {}",
                v,
                rel_err
            );
        }
    }

    #[test]
    fn test_mantissa_precision() {
        assert_eq!(GF20::MANT_BITS, 12);
        assert_eq!(GF20::EXP_BITS, 7);
        assert_eq!(GF20::MANT_BITS + GF20::EXP_BITS + 1, 20); // +1 for sign
    }

    #[test]
    fn test_special_values() {
        let inf = GF20::from_f32(f32::INFINITY);
        assert!(inf.to_f32().is_infinite());

        let neg_inf = GF20::from_f32(f32::NEG_INFINITY);
        assert!(neg_inf.to_f32().is_infinite());
        assert!(neg_inf.to_f32() < 0.0);

        let nan = GF20::from_f32(f32::NAN);
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
            let gf20 = GF20::from_f32(val as f32);
            let decoded = gf20.to_f32();
            assert!(!decoded.is_nan(), "{} produced NaN", name);
            let rel_err = gf20.relative_error_f32(val as f32);
            assert!(rel_err < 0.01, "{} error too high: {}", name, rel_err);
        }
    }

    #[test]
    fn test_phi_power_series() {
        // Verify φ^n = F_n * φ + F_{n-1} in GF20
        let cases = [
            (1, 1.0, 0.0), // φ^1 = 1*φ + 0
            (2, 1.0, 1.0), // φ^2 = 1*φ + 1
            (3, 2.0, 1.0), // φ^3 = 2*φ + 1
            (4, 3.0, 2.0), // φ^4 = 3*φ + 2
        ];

        for (n, f_n, f_n_minus_1) in cases {
            let phi_n = PHI.powf(n as f64);
            let gf20_n = GF20::from_f32(phi_n as f32).to_f32() as f64;

            let expected = f_n * PHI + f_n_minus_1;
            let error = (gf20_n - expected).abs();
            assert!(error < 0.01, "φ^{} power series error: {}", n, error);
        }
    }

    #[test]
    fn test_20_bit_layout() {
        // Verify we use exactly 20 bits
        let gf20 = GF20::from_f32(1.0);
        assert!(gf20.bits() < 0x00100000, "GF20 should use only 20 bits");
    }

    #[test]
    fn test_no_nan_in_normal_range() {
        // Ensure no NaN for values in normal range
        for i in 1..=8190 {
            let val = (i as f32).ln().exp(); // Use exponential for wider range
            if val.is_finite() && val < 8000.0 {
                let gf20 = GF20::from_f32(val);
                let decoded = gf20.to_f32();
                assert!(!decoded.is_nan(), "NaN for value {}", val);
            }
        }
    }
}
