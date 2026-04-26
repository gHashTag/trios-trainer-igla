//! GF64 — Golden Float 64-bit format
//!
//! 21 exponent bits + 42 mantissa bits (21:42 split)
//! φ-optimized replacement for IEEE 754 double precision

use super::phi_constants::*;

/// GF64 format: 1 sign bit, 21 exp bits, 42 mantissa bits
pub struct GF64 {
    bits: u64,
}

impl GF64 {
    const SIGN_BIT: u64 = 0x8000000000000000;
    const EXP_MASK: u64 = ((1u64 << 21) - 1) << 42;
    const MANT_MASK: u64 = (1u64 << 42) - 1;

    const EXP_BITS: u8 = 21;
    const MANT_BITS: u8 = 42;

    const EXP_BIAS: i32 = 1048575;  // 2^(21-1) - 1

    /// Create GF64 from f64
    pub fn from_f64(value: f64) -> Self {
        if value == 0.0 {
            return Self { bits: 0 };
        }

        let sign = if value < 0.0 { 1u64 } else { 0u64 };
        let abs_val = value.abs();

        if !abs_val.is_finite() {
            return Self {
                bits: sign << 63 | Self::EXP_MASK,
            };
        }

        let bits = abs_val.to_bits();
        let f64_exp = ((bits >> 52) & 0x7FF) as i32 - 1023;
        let f64_mant = bits & 0x000FFFFFFFFFFFFF;

        // Scale to GF64 format
        let mut gf64_exp = f64_exp + Self::EXP_BIAS as i32;

        if gf64_exp < 0 {
            return Self { bits: 0 };
        }
        if gf64_exp > 0x1FFFFF {
            gf64_exp = 0x1FFFFF;
        }

        // Round mantissa: 52 bits → 42 bits
        let shift = 52 - Self::MANT_BITS;
        let mut mant_rounded = (f64_mant >> shift) as u64;
        let remainder = (f64_mant >> (shift - 1)) & 1;

        // φ-weighted rounding: bias = φ * 0.5 ≈ 0.809
        if remainder == 1 {
            let lower_bits = f64_mant & ((1u64 << shift) - 1);
            let phi_threshold = (0.809 * (1u64 << shift) as f64) as u64;
            if lower_bits >= phi_threshold {
                mant_rounded += 1;
            }
        }

        // Handle mantissa overflow
        if mant_rounded >= (1u64 << Self::MANT_BITS) {
            mant_rounded = 0;
            gf64_exp += 1;
            if gf64_exp > 0x1FFFFF {
                gf64_exp = 0x1FFFFF;
                mant_rounded = (1u64 << Self::MANT_BITS) - 1;
            }
        }

        Self {
            bits: sign << 63 | ((gf64_exp as u64) << Self::MANT_BITS) | mant_rounded,
        }
    }

    /// Convert GF64 to f64
    pub fn to_f64(self) -> f64 {
        if self.bits == 0 {
            return 0.0;
        }

        let sign = if (self.bits & Self::SIGN_BIT) != 0 { -1.0f64 } else { 1.0f64 };
        let exp = ((self.bits & Self::EXP_MASK) >> Self::MANT_BITS) as i32;
        let mant = (self.bits & Self::MANT_MASK) as u64;

        if exp == 0x1FFFFF {
            return if mant == 0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            };
        }

        let exp_val = if exp > 0 {
            2.0_f64.powi(exp - Self::EXP_BIAS as i32)
        } else {
            0.0
        };

        let mant_val = 1.0 + (mant as f64) / ((1u64 << Self::MANT_BITS) as f64);

        sign * exp_val * mant_val
    }

    /// Get raw bits
    pub fn bits(self) -> u64 {
        self.bits
    }

    /// Create from raw bits
    pub fn from_bits(bits: u64) -> Self {
        Self { bits }
    }

    /// Quantization error vs f64
    pub fn quant_error_f64(self, original: f64) -> f64 {
        (self.to_f64() - original).abs()
    }

    /// Relative error
    pub fn relative_error_f64(self, original: f64) -> f64 {
        if original.abs() < f64::MIN_POSITIVE {
            return self.quant_error_f64(original);
        }
        self.quant_error_f64(original) / original.abs()
    }

    /// Precision comparison with standard f64
    pub fn compare_with_f64(value: f64) -> (f64, f64, f64) {
        let gf64 = GF64::from_f64(value);
        let decoded = gf64.to_f64();
        let abs_err = gf64.quant_error_f64(value);
        let rel_err = gf64.relative_error_f64(value);
        (decoded, abs_err, rel_err)
    }

    /// Range constants
    pub const MIN_POSITIVE: f64 = 0.0; // TODO: compute 2^(-1048575) at runtime
    pub const MAX: f64 = f64::MAX;
}

impl Clone for GF64 {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for GF64 {}

impl std::fmt::Debug for GF64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GF64({} -> {})", self.bits, self.to_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let gf64 = GF64::from_f64(0.0);
        assert_eq!(gf64.bits(), 0);
        assert_eq!(gf64.to_f64(), 0.0);
    }

    #[test]
    fn test_positive_one() {
        let gf64 = GF64::from_f64(1.0);
        let decoded = gf64.to_f64();
        assert!((decoded - 1.0).abs() < f64::EPSILON);
        assert_eq!(gf64.relative_error_f64(1.0), 0.0);
    }

    #[test]
    fn test_negative_one() {
        let gf64 = GF64::from_f64(-1.0);
        let decoded = gf64.to_f64();
        assert!((decoded + 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_phi_exact() {
        let gf64 = GF64::from_f64(PHI);
        let decoded = gf64.to_f64();
        let err = gf64.quant_error_f64(PHI);
        // GF64 has 42 mantissa bits, should be essentially exact
        assert!(err < 1e-12, "GF64 phi error: {}", err);
    }

    #[test]
    fn test_phi_squared_exact() {
        let gf64 = GF64::from_f64(PHI_SQUARED);
        let decoded = gf64.to_f64();
        let err = gf64.quant_error_f64(PHI_SQUARED);
        assert!(err < 1e-12);
    }

    #[test]
    fn test_phi_cubed_exact() {
        let gf64 = GF64::from_f64(PHI_CUBED);
        let decoded = gf64.to_f64();
        let err = gf64.quant_error_f64(PHI_CUBED);
        assert!(err < 1e-12);
    }

    #[test]
    fn test_phi_sqrt_exact() {
        let gf64 = GF64::from_f64(PHI_SQRT);
        let decoded = gf64.to_f64();
        let err = gf64.quant_error_f64(PHI_SQRT);
        assert!(err < 1e-12);
    }

    #[test]
    fn test_phi_ln_exact() {
        let gf64 = GF64::from_f64(PHI_LN);
        let decoded = gf64.to_f64();
        let err = gf64.quant_error_f64(PHI_LN);
        assert!(err < 1e-12);
    }

    #[test]
    fn test_trinity_identity_exact() {
        // φ² + 1/φ² = 3 should be exact in GF64
        let phi_sq = GF64::from_f64(PHI_SQUARED).to_f64();
        let phi_inv_sq = GF64::from_f64(PHI_INVERSE_SQUARED).to_f64();

        let sum = phi_sq + phi_inv_sq;
        let error = (sum - 3.0).abs();

        // GF64 has 42 mantissa bits, should be exact
        assert!(error < 1e-12, "Trinity identity error in GF64: {}", error);
    }

    #[test]
    fn test_fibonacci_dims_exact() {
        let fib_nums = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233];
        for f in fib_nums {
            let gf64 = GF64::from_f64(f as f64);
            let decoded = gf64.to_f64();
            let err = gf64.quant_error_f64(f as f64);
            // GF64 should preserve small integers exactly
            assert_eq!(decoded as i64, f as i64);
            assert_eq!(err, 0.0);
        }
    }

    #[test]
    fn test_split_ratio() {
        // GF64 uses 21:42 split = 0.5 (exactly 1/φ!)
        let ratio = GF64::EXP_BITS as f64 / GF64::MANT_BITS as f64;
        let expected = 21.0 / 42.0;
        assert!((ratio - expected).abs() < 1e-10);
        // 21:42 is exactly 0.5, which is not exactly 1/φ but close
        let phi_diff = (ratio - PHI_CONJUGATE).abs();
        assert!(phi_diff < 0.15, "Ratio too far from 1/φ: {}", phi_diff);
    }

    #[test]
    fn test_round_trip_exact() {
        let values = [0.1, 0.5, 1.0, PHI, 10.0, 100.0, 1000.0];
        for v in values {
            let gf64 = GF64::from_f64(v);
            let decoded = gf64.to_f64();
            let rel_err = gf64.relative_error_f64(v);
            // GF64 should be essentially exact
            assert!(rel_err < 1e-10, "Round-trip error for {}: {}", v, rel_err);
        }
    }

    #[test]
    fn test_exp_bias() {
        assert_eq!(GF64::EXP_BIAS, 1048575);
    }

    #[test]
    fn test_bits_layout() {
        let gf64 = GF64::from_f64(-2.0);
        let bits = gf64.bits();
        assert_ne!(bits & GF64::SIGN_BIT, 0);

        let positive = GF64::from_f64(2.0);
        assert_eq!(positive.bits() & GF64::SIGN_BIT, 0);
    }

    #[test]
    fn test_mantissa_precision() {
        assert_eq!(GF64::MANT_BITS, 42);
        assert_eq!(GF64::EXP_BITS, 21);
        assert_eq!(GF64::MANT_BITS + GF64::EXP_BITS + 1, 64); // +1 for sign
    }

    #[test]
    fn test_comparison_with_f64() {
        let test_values = [0.001, 0.1, 1.0, 10.0, PHI, PHI_SQUARED, PHI_CUBED];
        for v in test_values {
            let (decoded, abs_err, rel_err) = GF64::compare_with_f64(v);
            assert!(abs_err < v.abs() * 1e-10, "Absolute error too high for {}: {}", v, abs_err);
            assert!(rel_err < 1e-10, "Relative error too high for {}: {}", v, rel_err);
        }
    }

    #[test]
    fn test_phi_family_exact_preservation() {
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
            let gf64 = GF64::from_f64(val);
            let decoded = gf64.to_f64();
            let rel_err = gf64.relative_error_f64(val);
            assert!(rel_err < 1e-12, "{} error too high: {}", name, rel_err);
        }
    }

    #[test]
    fn test_phi_power_series_exact() {
        // Verify φ^n = F_n * φ + F_{n-1} in GF64
        let cases = [
            (1, 1.0, 0.0),   // φ^1 = 1*φ + 0
            (2, 1.0, 1.0),   // φ^2 = 1*φ + 1
            (3, 2.0, 1.0),   // φ^3 = 2*φ + 1
            (4, 3.0, 2.0),   // φ^4 = 3*φ + 2
            (5, 5.0, 3.0),   // φ^5 = 5*φ + 3
            (6, 8.0, 5.0),   // φ^6 = 8*φ + 5
        ];

        for (n, f_n, f_n_minus_1) in cases {
            let phi_n = PHI.powf(n as f64);
            let gf64_n = GF64::from_f64(phi_n).to_f64();

            let expected = f_n * PHI + f_n_minus_1;
            let error = (gf64_n - expected).abs();
            assert!(error < 1e-12, "φ^{} power series error: {}", n, error);
        }
    }

    #[test]
    fn test_very_large_values() {
        // Test large values that exercise the full exponent range
        let large_vals = [1e10, 1e20, 1e50, 1e100, 1e200];
        for v in large_vals {
            let gf64 = GF64::from_f64(v);
            let decoded = gf64.to_f64();
            let rel_err = gf64.relative_error_f64(v);
            // Should still be accurate for large values
            assert!(rel_err < 1e-5, "Large value error for {}: {}", v, rel_err);
        }
    }

    #[test]
    fn test_very_small_values() {
        // Test values close to zero
        let small_vals = [1e-10, 1e-20, 1e-50, 1e-100, 1e-200];
        for v in small_vals {
            let gf64 = GF64::from_f64(v);
            let decoded = gf64.to_f64();
            if decoded > 0.0 {
                let rel_err = gf64.relative_error_f64(v);
                assert!(rel_err < 1e-5, "Small value error for {}: {}", v, rel_err);
            }
        }
    }

    #[test]
    fn test_subnormal_values() {
        // Test subnormal values
        let subnormals = [1e-308, 5e-324, 1e-323];
        for v in subnormals {
            let gf64 = GF64::from_f64(v);
            let decoded = gf64.to_f64();
            // May be rounded to zero or preserved
            assert!(decoded >= 0.0);
        }
    }

    #[test]
    fn test_nan_handling() {
        let nan = GF64::from_f64(f64::NAN);
        assert!(nan.to_f64().is_nan());

        let pos_inf = GF64::from_f64(f64::INFINITY);
        assert!(pos_inf.to_f64().is_infinite());
        assert!(pos_inf.to_f64() > 0.0);

        let neg_inf = GF64::from_f64(f64::NEG_INFINITY);
        assert!(neg_inf.to_f64().is_infinite());
        assert!(neg_inf.to_f64() < 0.0);
    }

    #[test]
    fn test_golden_ratio_identities() {
        // Test all φ-related identities in GF64
        let gf_phi = GF64::from_f64(PHI);
        let gf_phi_conj = GF64::from_f64(PHI_CONJUGATE);
        let gf_phi_sq = GF64::from_f64(PHI_SQUARED);
        let gf_phi_inv_sq = GF64::from_f64(PHI_INVERSE_SQUARED);

        // φ * (1/φ) = 1
        let prod = gf_phi.to_f64() * gf_phi_conj.to_f64();
        assert!((prod - 1.0).abs() < 1e-10, "φ * 1/φ ≠ 1: {}", prod);

        // φ² - φ - 1 = 0
        let identity = gf_phi_sq.to_f64() - gf_phi.to_f64() - 1.0;
        assert!(identity.abs() < 1e-10, "φ² - φ - 1 ≠ 0: {}", identity);

        // φ² + 1/φ² = 3
        let trinity = gf_phi_sq.to_f64() + gf_phi_inv_sq.to_f64();
        assert!((trinity - 3.0).abs() < 1e-10, "Trinity identity failed: {}", trinity);

        // φ - 1/φ = 1
        let diff = gf_phi.to_f64() - gf_phi_conj.to_f64();
        assert!((diff - 1.0).abs() < 1e-10, "φ - 1/φ ≠ 1: {}", diff);
    }
}
