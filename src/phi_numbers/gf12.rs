//! GF12 — Golden Float 12-bit format
//!
//! 4 exponent bits + 7 mantissa bits (4:7 split)
//! Ultra-low precision for quantized embeddings
//! φ-optimized: 4:7 ≈ 0.571 (close to 1/φ ≈ 0.618)

use super::phi_constants::*;

/// GF12 format: 1 sign bit, 4 exp bits, 7 mantissa bits
/// Total: 12 bits = 128 mantissa values × 15 exponent values × 2 signs ≈ 3840 representable values
pub struct GF12 {
    bits: u16,
}

impl GF12 {
    const SIGN_BIT: u16 = 0x0800; // 1000_0000_0000 (bit 11)
    const EXP_MASK: u16 = 0x0780; // 0111_1000_0000 (bits 10:7)
    const MANT_MASK: u16 = 0x007F; // 0000_0111_1111 (bits 6:0)

    const EXP_BITS: u8 = 4;
    const MANT_BITS: u8 = 7;

    const EXP_BIAS: i8 = 7; // 2^(4-1) - 1 = 7

    /// Create GF12 from f32 (quantization)
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
                    bits: (sign << 11) | 0x07FF,
                };
            }
            return Self {
                bits: (sign << 11) | 0x0780,
            };
        }

        // Extract exponent and mantissa from f32
        let bits = abs_val.to_bits();
        let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let f32_mant = bits & 0x007FFFFF;

        // Scale to GF12 format
        let mut gf12_exp = f32_exp + Self::EXP_BIAS as i32;

        // Clamp exponent
        if gf12_exp < 0 {
            return Self { bits: 0 };
        }
        if gf12_exp >= 15 {
            return Self {
                bits: (sign << 11) | (14 << 7) | 127,
            };
        }

        // Round mantissa: 23 bits → 7 bits
        let shift = 23 - Self::MANT_BITS;
        let mut mant_rounded = (f32_mant >> shift) as u16;
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
        if mant_rounded >= 128 {
            mant_rounded = 0;
            gf12_exp += 1;
            if gf12_exp >= 15 {
                return Self {
                    bits: (sign << 11) | (14 << 7) | 127,
                };
            }
        }

        Self {
            bits: (sign << 11) | ((gf12_exp as u16) << 7) | mant_rounded,
        }
    }

    /// Convert GF12 to f32
    pub fn to_f32(self) -> f32 {
        if self.bits == 0 {
            return 0.0;
        }

        let sign = if (self.bits & Self::SIGN_BIT) != 0 {
            -1.0
        } else {
            1.0
        };
        let exp = ((self.bits & Self::EXP_MASK) >> 7) as i32;
        let mant = (self.bits & Self::MANT_MASK) as u32;

        // Check for infinity/NaN
        if exp == 15 {
            return if mant == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            };
        }

        // Reconstruct value: exp == 0 represents 2^(-7), not zero!
        let exp_val = 2.0_f32.powi(exp - Self::EXP_BIAS as i32);
        let mant_val = 1.0 + (mant as f32) / 128.0;

        sign * exp_val * mant_val
    }

    /// Get raw bits
    pub fn bits(self) -> u16 {
        self.bits
    }

    /// Create from raw bits
    pub fn from_bits(bits: u16) -> Self {
        Self { bits: bits & 0x0FFF }
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
    pub const MIN_POSITIVE: f32 = 0.0078125; // 2^(-7) = 1/128
    pub const MAX: f32 = (1.0 + 127.0 / 128.0) * 128.0; // (1 + 127/128) × 2^7 ≈ 255
}

impl Clone for GF12 {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for GF12 {}

impl std::fmt::Debug for GF12 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GF12(0b{:012b} -> {})", self.bits, self.to_f32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let gf12 = GF12::from_f32(0.0);
        assert_eq!(gf12.bits(), 0);
        assert_eq!(gf12.to_f32(), 0.0);
    }

    #[test]
    fn test_positive_one() {
        let gf12 = GF12::from_f32(1.0);
        let decoded = gf12.to_f32();
        assert!((decoded - 1.0).abs() < 0.1);
        assert!(!decoded.is_nan());
    }

    #[test]
    fn test_negative_one() {
        let gf12 = GF12::from_f32(-1.0);
        let decoded = gf12.to_f32();
        assert!((decoded + 1.0).abs() < 0.1);
        assert!(!decoded.is_nan());
    }

    #[test]
    fn test_phi_value() {
        let gf12 = GF12::from_f32(PHI as f32);
        let decoded = gf12.to_f32();
        let err = gf12.quant_error_f32(PHI as f32);
        // GF12 has 7 mantissa bits, expect moderate error
        assert!(err < 0.1, "GF12 phi error too high: {}", err);
        assert!(!decoded.is_nan());
    }

    #[test]
    fn test_phi_conjugate_value() {
        let gf12 = GF12::from_f32(PHI_CONJUGATE as f32);
        let decoded = gf12.to_f32();
        let err = gf12.quant_error_f32(PHI_CONJUGATE as f32);
        assert!(err < 0.05, "GF12 phi_conjugate error too high: {}", err);
        assert!(!decoded.is_nan());
    }

    #[test]
    fn test_phi_squared_value() {
        let gf12 = GF12::from_f32(PHI_SQUARED as f32);
        let decoded = gf12.to_f32();
        let err = gf12.quant_error_f32(PHI_SQUARED as f32);
        assert!(err < 0.2, "GF12 phi_squared error too high: {}", err);
        assert!(!decoded.is_nan());
    }

    #[test]
    fn test_round_trip() {
        let values = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
        for v in values {
            let gf12 = GF12::from_f32(v);
            let decoded = gf12.to_f32();
            let rel_err = gf12.relative_error_f32(v);
            assert!(!decoded.is_nan(), "NaN for value {}", v);
            assert!(rel_err < 0.1, "Round-trip error for {}: {}", v, rel_err);
        }
    }

    #[test]
    fn test_bits_conversion() {
        let gf12 = GF12::from_f32(1.5);
        let bits = gf12.bits();
        let gf12_back = GF12::from_bits(bits);
        assert_eq!(gf12_back.to_f32(), gf12.to_f32());
    }

    #[test]
    fn test_exp_range() {
        // Test minimum positive
        let min_gf12 = GF12::from_f32(0.01);
        let decoded = min_gf12.to_f32();
        assert!(decoded >= 0.0);
        assert!(!decoded.is_nan());

        // Test near max (clamped) - adjust expectation
        let max_gf12 = GF12::from_f32(1000.0);
        let decoded = max_gf12.to_f32();
        // GF12 max is ~127.5, values larger are clamped
        // Allow small floating point margin
        assert!(decoded <= GF12::MAX + 0.1);
        assert!(!decoded.is_nan());
    }

    #[test]
    fn test_fibonacci_dims_quantization() {
        let fib_nums = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
        for f in fib_nums {
            if f < 128 {
                let gf12 = GF12::from_f32(f as f32);
                let decoded = gf12.to_f32();
                let rel_err = gf12.relative_error_f32(f as f32);
                assert!(!decoded.is_nan(), "Fibonacci {} produced NaN", f);
                assert!(rel_err < 0.1, "Fibonacci {} error: {}", f, rel_err);
            }
        }
    }

    #[test]
    fn test_phi_identity_preservation() {
        let phi_gf12 = GF12::from_f32(PHI as f32);
        let phi_sq_gf12 = GF12::from_f32(PHI_SQUARED as f32);
        let phi_inv_sq_gf12 = GF12::from_f32(PHI_INVERSE_SQUARED as f32);

        let phi_sq_decoded = phi_sq_gf12.to_f32() as f64;
        let phi_inv_sq_decoded = phi_inv_sq_gf12.to_f32() as f64;

        // Check for NaN
        assert!(!phi_sq_decoded.is_nan());
        assert!(!phi_inv_sq_decoded.is_nan());

        let sum = phi_sq_decoded + phi_inv_sq_decoded;
        let error = (sum - 3.0).abs();

        // GF12 has low precision, allow 10% error
        assert!(error < 0.3, "Trinity identity error in GF12: {}", error);
    }

    #[test]
    fn test_mantissa_bits() {
        assert_eq!(GF12::MANT_BITS, 7);
        assert_eq!(GF12::EXP_BITS, 4);
    }

    #[test]
    fn test_split_ratio() {
        let ratio = GF12::EXP_BITS as f64 / GF12::MANT_BITS as f64;
        assert!((ratio - (4.0 / 7.0)).abs() < 1e-10);
        // 4:7 ≈ 0.571 is close to 1/φ ≈ 0.618
        let phi_diff = (ratio - PHI_CONJUGATE).abs();
        assert!(phi_diff < 0.1, "Ratio too far from 1/φ: {}", phi_diff);
    }

    #[test]
    fn test_sign_bit() {
        let pos = GF12::from_f32(2.0);
        let neg = GF12::from_f32(-2.0);

        assert_eq!(pos.bits() & GF12::SIGN_BIT, 0);
        assert_ne!(neg.bits() & GF12::SIGN_BIT, 0);
    }

    #[test]
    fn test_special_values() {
        let inf = GF12::from_f32(f32::INFINITY);
        assert!(inf.to_f32().is_infinite());

        let neg_inf = GF12::from_f32(f32::NEG_INFINITY);
        assert!(neg_inf.to_f32().is_infinite());
        assert!(neg_inf.to_f32() < 0.0);

        let nan = GF12::from_f32(f32::NAN);
        assert!(nan.to_f32().is_nan());
    }

    #[test]
    fn test_no_nan_in_normal_range() {
        // Ensure no NaN for values in normal range
        for i in 1..=127 {
            let val = i as f32;
            let gf12 = GF12::from_f32(val);
            let decoded = gf12.to_f32();
            assert!(!decoded.is_nan(), "NaN for value {}", val);
        }
    }

    #[test]
    fn test_phi_family_consistency() {
        let constants = [
            (PHI, "PHI"),
            (PHI_CONJUGATE, "PHI_CONJUGATE"),
            (PHI_SQUARED, "PHI_SQUARED"),
            (PHI_INVERSE_SQUARED, "PHI_INVERSE_SQUARED"),
        ];

        for (val, name) in constants {
            let gf12 = GF12::from_f32(val as f32);
            let decoded = gf12.to_f32();
            assert!(!decoded.is_nan(), "{} produced NaN", name);
            let rel_err = gf12.relative_error_f32(val as f32);
            assert!(rel_err < 0.2, "{} error too high: {}", name, rel_err);
        }
    }

    #[test]
    fn test_12_bit_layout() {
        // Verify we use exactly 12 bits
        let gf12 = GF12::from_f32(1.0);
        assert!(gf12.bits() < 4096, "GF12 should use only 12 bits");
    }
}
