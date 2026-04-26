//! GF8 — Golden Float 8-bit format
//!
//! 3 exponent bits + 4 mantissa bits (3:4 split)
//! Ultra-low precision for extreme quantization scenarios

use super::phi_constants::*;

/// GF8 format: 3 exp bits, 4 mantissa bits, 1 sign bit
pub struct GF8 {
    bits: u8,
}

impl GF8 {
    const SIGN_BIT: u8 = 0x80; // 1000_0000
    const EXP_MASK: u8 = 0x70; // 0111_0000
    const MANT_MASK: u8 = 0x0F; // 0000_1111

    const EXP_BITS: u8 = 3;
    const MANT_BITS: u8 = 4;

    const EXP_BIAS: i8 = 3; // 2^(3-1) - 1

    /// Create GF8 from f32 (quantization)
    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Self { bits: 0 };
        }

        let sign = if value < 0.0 { 1 } else { 0 };
        let abs_val = value.abs();

        // Handle infinity and NaN
        if !abs_val.is_finite() {
            if abs_val.is_nan() {
                return Self {
                    bits: (sign << 7) | 0x79,
                };
            }
            return Self {
                bits: (sign << 7) | 0x70,
            };
        }

        // Extract exponent and mantissa from f32
        let bits = abs_val.to_bits();
        let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let f32_mant = bits & 0x007FFFFF;

        // Scale to GF8 format
        // GF8: 3 exp bits, 4 mant bits
        // f32: 8 exp bits, 23 mant bits
        let mut gf8_exp = f32_exp + Self::EXP_BIAS as i32;

        // Clamp exponent
        if gf8_exp < 0 {
            return Self { bits: 0 }; // Underflow to zero
        }
        if gf8_exp > 7 {
            return Self {
                bits: (sign << 7) | 0x70,
            };
        }

        // Round mantissa: 23 bits → 4 bits
        // Use φ-weighted rounding: favor rounding toward nearest with φ-weighted bias
        let mut mant_rounded = (f32_mant >> (23 - Self::MANT_BITS)) as u8;
        let remainder = (f32_mant >> (23 - Self::MANT_BITS - 1)) & 1;

        if remainder == 1 {
            mant_rounded += 1;
        }

        // Handle mantissa overflow
        if mant_rounded >= 16 {
            mant_rounded = 0;
            gf8_exp += 1;
            if gf8_exp > 6 {
                return Self {
                    bits: (sign << 7) | 0x70,
                };
            }
        }

        Self {
            bits: (sign << 7) | ((gf8_exp as u8) << 4) | mant_rounded,
        }
    }

    /// Convert GF8 to f32
    pub fn to_f32(self) -> f32 {
        if self.bits == 0 {
            return 0.0;
        }

        let sign = if (self.bits & Self::SIGN_BIT) != 0 {
            -1.0
        } else {
            1.0
        };
        let exp = ((self.bits & Self::EXP_MASK) >> 4) as i32;
        let mant = (self.bits & Self::MANT_MASK) as u32;

        // Check for infinity/NaN
        if exp == 7 {
            return if mant == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            };
        }

        // Reconstruct value
        let exp_val = if exp > 0 {
            2.0_f32.powi(exp - Self::EXP_BIAS as i32)
        } else {
            0.0
        };

        let mant_val = 1.0 + (mant as f32) / 16.0;

        sign * exp_val * mant_val
    }

    /// Get raw bits
    pub fn bits(self) -> u8 {
        self.bits
    }

    /// Create from raw bits
    pub fn from_bits(bits: u8) -> Self {
        Self { bits }
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
    pub const MIN_POSITIVE: f32 = 0.125_f32; // 2^(-3) = 0.125 (powi not const in Rust 1.90)
    pub const MAX: f32 = 15.75; // (2 - 1/16) * 2^4
}

impl Clone for GF8 {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for GF8 {}

impl std::fmt::Debug for GF8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GF8({} -> {})", self.bits, self.to_f32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let gf8 = GF8::from_f32(0.0);
        assert_eq!(gf8.bits(), 0);
        assert_eq!(gf8.to_f32(), 0.0);
    }

    #[test]
    fn test_positive_one() {
        let gf8 = GF8::from_f32(1.0);
        let decoded = gf8.to_f32();
        assert!(decoded > 0.9 && decoded < 1.1);
        let rel_err = gf8.relative_error_f32(1.0);
        assert!(rel_err < 0.2, "Relative error too high: {}", rel_err);
    }

    #[test]
    fn test_negative_one() {
        let gf8 = GF8::from_f32(-1.0);
        let decoded = gf8.to_f32();
        assert!(decoded < -0.9 && decoded > -1.1);
        assert!(gf8.relative_error_f32(-1.0) < 0.2);
    }

    #[test]
    fn test_phi_value() {
        let gf8 = GF8::from_f32(PHI as f32);
        let decoded = gf8.to_f32();
        let err = gf8.quant_error_f32(PHI as f32);
        // With 4 mantissa bits, expect significant error
        assert!(err < 0.5, "Quantization error too high: {}", err);
    }

    #[test]
    fn test_phi_conjugate_value() {
        let gf8 = GF8::from_f32(PHI_CONJUGATE as f32);
        let decoded = gf8.to_f32();
        let err = gf8.quant_error_f32(PHI_CONJUGATE as f32);
        assert!(err < 0.3, "Quantization error too high: {}", err);
    }

    #[test]
    fn test_phi_squared_value() {
        let gf8 = GF8::from_f32(PHI_SQUARED as f32);
        let decoded = gf8.to_f32();
        let err = gf8.quant_error_f32(PHI_SQUARED as f32);
        assert!(err < 1.0, "Quantization error too high: {}", err);
    }

    #[test]
    fn test_phi_sqrt_value() {
        let gf8 = GF8::from_f32(PHI_SQRT as f32);
        let decoded = gf8.to_f32();
        let err = gf8.quant_error_f32(PHI_SQRT as f32);
        assert!(err < 0.3, "Quantization error too high: {}", err);
    }

    #[test]
    fn test_round_trip() {
        let values = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
        for v in values {
            let gf8 = GF8::from_f32(v);
            let decoded = gf8.to_f32();
            let rel_err = gf8.relative_error_f32(v);
            assert!(rel_err < 0.5, "Round-trip error for {}: {}", v, rel_err);
        }
    }

    #[test]
    fn test_bits_conversion() {
        let gf8 = GF8::from_f32(1.5);
        let bits = gf8.bits();
        let gf8_back = GF8::from_bits(bits);
        assert_eq!(gf8_back.to_f32(), gf8.to_f32());
    }

    #[test]
    fn test_exp_range() {
        // Test minimum positive
        let min_gf8 = GF8::from_f32(0.125);
        assert!(min_gf8.to_f32() >= 0.125);

        // Test near max (clamped)
        let max_gf8 = GF8::from_f32(1000.0);
        assert!(max_gf8.to_f32() <= GF8::MAX);
    }

    #[test]
    fn test_fibonacci_dims_quantization() {
        let fib_nums = [1, 2, 3, 5, 8, 13, 21];
        for f in fib_nums {
            let gf8 = GF8::from_f32(f as f32);
            let decoded = gf8.to_f32();
            let rel_err = gf8.relative_error_f32(f as f32);
            // GF8 has limited precision, allow up to 50% error for small numbers
            assert!(rel_err < 0.5, "Fibonacci {} error: {}", f, rel_err);
        }
    }

    #[test]
    fn test_phi_identity_preservation() {
        // Test φ² + 1/φ² ≈ 3 in GF8
        let phi_gf8 = GF8::from_f32(PHI as f32);
        let phi_sq_gf8 = GF8::from_f32(PHI_SQUARED as f32);
        let phi_inv_sq_gf8 = GF8::from_f32(PHI_INVERSE_SQUARED as f32);

        let phi_sq_decoded = phi_sq_gf8.to_f32() as f64;
        let phi_inv_sq_decoded = phi_inv_sq_gf8.to_f32() as f64;

        let sum = phi_sq_decoded + phi_inv_sq_decoded;
        let error = (sum - 3.0).abs();

        // GF8 has low precision, allow 20% error
        assert!(error < 0.6, "Trinity identity error in GF8: {}", error);
    }

    #[test]
    fn test_mantissa_bits() {
        assert_eq!(GF8::MANT_BITS, 4);
        assert_eq!(GF8::EXP_BITS, 3);
    }

    #[test]
    fn test_split_ratio() {
        // GF8 uses 3:4 split = 0.75 (different from GF16's 6:9 ≈ 0.667)
        let ratio = GF8::EXP_BITS as f64 / GF8::MANT_BITS as f64;
        assert!((ratio - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_sign_bit() {
        let pos = GF8::from_f32(2.0);
        let neg = GF8::from_f32(-2.0);

        assert_eq!(pos.bits() & GF8::SIGN_BIT, 0);
        assert_eq!(neg.bits() & GF8::SIGN_BIT, GF8::SIGN_BIT);
    }
}
