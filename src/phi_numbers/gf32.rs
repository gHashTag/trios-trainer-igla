//! GF32 — Golden Float 32-bit format
//!
//! 13 exponent bits + 18 mantissa bits (13:18 split)
//! φ-optimized replacement for IEEE 754 single precision

use super::phi_constants::*;

/// GF32 format: 1 sign bit, 13 exp bits, 18 mantissa bits
pub struct GF32 {
    bits: u32,
}

impl GF32 {
    const SIGN_BIT: u32 = 0x80000000;
    const EXP_MASK: u32 = 0x7FFE0000; // 13 bits
    const MANT_MASK: u32 = 0x0003FFFF; // 18 bits

    const EXP_BITS: u8 = 13;
    const MANT_BITS: u8 = 18;

    const EXP_BIAS: i16 = 4095; // 2^(13-1) - 1

    /// Create GF32 from f32
    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Self { bits: 0 };
        }

        let sign = if value < 0.0 { 1u32 } else { 0u32 };
        let abs_val = value.abs();

        if !abs_val.is_finite() {
            return Self {
                bits: sign << 31 | Self::EXP_MASK,
            };
        }

        let bits = abs_val.to_bits();
        let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let f32_mant = bits & 0x007FFFFF;

        // Scale to GF32 format
        let mut gf32_exp = f32_exp + Self::EXP_BIAS as i32;

        if gf32_exp < 0 {
            return Self { bits: 0 };
        }
        if gf32_exp > 0x1FFF {
            gf32_exp = 0x1FFF;
        }

        // Round mantissa: 23 bits → 18 bits
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
        if mant_rounded >= (1u32 << Self::MANT_BITS) {
            mant_rounded = 0;
            gf32_exp += 1;
            if gf32_exp > 0x1FFF {
                gf32_exp = 0x1FFF;
                mant_rounded = (1u32 << Self::MANT_BITS) - 1;
            }
        }

        Self {
            bits: sign << 31 | ((gf32_exp as u32) << Self::MANT_BITS) | mant_rounded,
        }
    }

    /// Convert GF32 to f32
    pub fn to_f32(self) -> f32 {
        if self.bits == 0 {
            return 0.0;
        }

        let sign = if (self.bits & Self::SIGN_BIT) != 0 {
            -1.0f32
        } else {
            1.0f32
        };
        let exp = ((self.bits & Self::EXP_MASK) >> Self::MANT_BITS) as i32;
        let mant = (self.bits & Self::MANT_MASK) as u32;

        if exp == 0x1FFF {
            return if mant == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            };
        }

        let exp_val = if exp > 0 {
            2.0_f32.powi(exp - Self::EXP_BIAS as i32)
        } else {
            0.0
        };

        let mant_val = 1.0 + (mant as f32) / ((1u32 << Self::MANT_BITS) as f32);

        sign * exp_val * mant_val
    }

    /// Get raw bits
    pub fn bits(self) -> u32 {
        self.bits
    }

    /// Create from raw bits
    pub fn from_bits(bits: u32) -> Self {
        Self { bits }
    }

    /// Quantization error vs f32
    pub fn quant_error_f32(self, original: f32) -> f32 {
        (self.to_f32() - original).abs()
    }

    /// Relative error
    pub fn relative_error_f32(self, original: f32) -> f32 {
        if original.abs() < f32::MIN_POSITIVE {
            return self.quant_error_f32(original);
        }
        self.quant_error_f32(original) / original.abs()
    }

    /// Precision comparison with standard f32
    pub fn compare_with_f32(value: f32) -> (f32, f32, f32) {
        let gf32 = GF32::from_f32(value);
        let decoded = gf32.to_f32();
        let abs_err = gf32.quant_error_f32(value);
        let rel_err = gf32.relative_error_f32(value);
        (decoded, abs_err, rel_err)
    }

    /// Range constants
    pub const MIN_POSITIVE: f32 = 0.0;
    pub const MAX: f32 = f32::MAX;
}

impl Clone for GF32 {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for GF32 {}

impl std::fmt::Debug for GF32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GF32({} -> {})", self.bits, self.to_f32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let gf32 = GF32::from_f32(0.0);
        assert_eq!(gf32.bits(), 0);
        assert_eq!(gf32.to_f32(), 0.0);
    }

    #[test]
    fn test_positive_one() {
        let gf32 = GF32::from_f32(1.0);
        let decoded = gf32.to_f32();
        assert!((decoded - 1.0).abs() < f32::EPSILON);
        assert_eq!(gf32.relative_error_f32(1.0), 0.0);
    }

    #[test]
    fn test_negative_one() {
        let gf32 = GF32::from_f32(-1.0);
        let decoded = gf32.to_f32();
        assert!((decoded + 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_phi_high_precision() {
        let gf32 = GF32::from_f32(PHI as f32);
        let decoded = gf32.to_f32();
        let err = gf32.quant_error_f32(PHI as f32);
        // GF32 has 18 mantissa bits, should be very close to f32
        assert!(err < 1e-5, "GF32 phi error too high: {}", err);
    }

    #[test]
    fn test_phi_squared_high_precision() {
        let gf32 = GF32::from_f32(PHI_SQUARED as f32);
        let decoded = gf32.to_f32();
        let err = gf32.quant_error_f32(PHI_SQUARED as f32);
        assert!(err < 1e-5);
    }

    #[test]
    fn test_phi_cubed_high_precision() {
        let gf32 = GF32::from_f32(PHI_CUBED as f32);
        let decoded = gf32.to_f32();
        let err = gf32.quant_error_f32(PHI_CUBED as f32);
        assert!(err < 1e-5);
    }

    #[test]
    fn test_phi_sqrt_high_precision() {
        let gf32 = GF32::from_f32(PHI_SQRT as f32);
        let decoded = gf32.to_f32();
        let err = gf32.quant_error_f32(PHI_SQRT as f32);
        assert!(err < 1e-5);
    }

    #[test]
    fn test_trinity_identity_preservation() {
        // φ² + 1/φ² = 3 should hold well in GF32
        let phi_sq = GF32::from_f32(PHI_SQUARED as f32).to_f32() as f64;
        let phi_inv_sq = GF32::from_f32(PHI_INVERSE_SQUARED as f32).to_f32() as f64;

        let sum = phi_sq + phi_inv_sq;
        let error = (sum - 3.0).abs();

        // GF32 has 18 mantissa bits, should be very close
        assert!(error < 1e-10, "Trinity identity error in GF32: {}", error);
    }

    #[test]
    fn test_fibonacci_dims_high_precision() {
        let fib_nums = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
        for f in fib_nums {
            let gf32 = GF32::from_f32(f as f32);
            let decoded = gf32.to_f32();
            let err = gf32.quant_error_f32(f as f32);
            // GF32 should preserve small integers exactly
            assert_eq!(decoded as i32, f);
            assert_eq!(err, 0.0);
        }
    }

    #[test]
    fn test_split_ratio() {
        // GF32 uses 13:18 split ≈ 0.722 (close to 1/φ ≈ 0.618)
        let ratio = GF32::EXP_BITS as f64 / GF32::MANT_BITS as f64;
        let expected = 13.0 / 18.0;
        assert!((ratio - expected).abs() < 1e-10);
        // Should be closer to 1/φ than GF16's 6:9 split
        let phi_diff = (ratio - PHI_CONJUGATE).abs();
        assert!(phi_diff < 0.2, "Ratio too far from 1/φ: {}", phi_diff);
    }

    #[test]
    fn test_round_trip_accuracy() {
        let values = [0.1, 0.5, 1.0, PHI as f32, 10.0, 100.0];
        for v in values {
            let gf32 = GF32::from_f32(v);
            let decoded = gf32.to_f32();
            let rel_err = gf32.relative_error_f32(v);
            // GF32 should be very accurate
            assert!(rel_err < 1e-4, "Round-trip error for {}: {}", v, rel_err);
        }
    }

    #[test]
    fn test_exp_bias() {
        assert_eq!(GF32::EXP_BIAS, 4095);
    }

    #[test]
    fn test_bits_layout() {
        let gf32 = GF32::from_f32(-2.0);
        let bits = gf32.bits();
        assert_ne!(bits & GF32::SIGN_BIT, 0, "Sign bit should be set");

        let positive = GF32::from_f32(2.0);
        assert_eq!(positive.bits() & GF32::SIGN_BIT, 0);
    }

    #[test]
    fn test_comparison_with_f32() {
        let test_values = [0.001, 0.1, 1.0, 10.0, PHI as f32, PHI_SQUARED as f32];
        for v in test_values {
            let (decoded, abs_err, rel_err) = GF32::compare_with_f32(v);
            assert!(
                abs_err < v.abs() * 1e-4,
                "Absolute error too high for {}: {}",
                v,
                abs_err
            );
            assert!(
                rel_err < 1e-4,
                "Relative error too high for {}: {}",
                v,
                rel_err
            );
        }
    }

    #[test]
    fn test_mantissa_precision() {
        assert_eq!(GF32::MANT_BITS, 18);
        assert_eq!(GF32::EXP_BITS, 13);
        assert_eq!(GF32::MANT_BITS + GF32::EXP_BITS + 1, 32); // +1 for sign
    }

    #[test]
    fn test_special_values() {
        let inf = GF32::from_f32(f32::INFINITY);
        assert!(inf.to_f32().is_infinite());

        let neg_inf = GF32::from_f32(f32::NEG_INFINITY);
        assert!(neg_inf.to_f32().is_infinite());
        assert!(neg_inf.to_f32() < 0.0);
    }

    #[test]
    fn test_phi_family_consistency() {
        // All φ-family constants should be well-preserved in GF32
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
            let gf32 = GF32::from_f32(val as f32);
            let decoded = gf32.to_f32();
            let rel_err = gf32.relative_error_f32(val as f32);
            assert!(rel_err < 1e-5, "{} error too high: {}", name, rel_err);
        }
    }

    #[test]
    fn test_phi_power_series() {
        // Verify φ^n = F_n * φ + F_{n-1} in GF32
        let cases = [
            (1, 1.0, 0.0), // φ^1 = 1*φ + 0
            (2, 1.0, 1.0), // φ^2 = 1*φ + 1
            (3, 2.0, 1.0), // φ^3 = 2*φ + 1
            (4, 3.0, 2.0), // φ^4 = 3*φ + 2
        ];

        for (n, f_n, f_n_minus_1) in cases {
            let phi_n = PHI.powf(n as f64);
            let gf32_n = GF32::from_f32(phi_n as f32).to_f32() as f64;

            let expected = f_n * PHI + f_n_minus_1;
            let error = (gf32_n - expected).abs();
            assert!(error < 1e-5, "φ^{} power series error: {}", n, error);
        }
    }
}
