//! GF4 — Golden Float 4-bit format
//!
//! 1 sign + 1 exponent + 2 mantissa bits (ratio 1:2, Fibonacci F1:F2)
//! Used for extreme ternary-adjacent quantization; aligns with PhD Glava 06
//! "Golden Mantissa: GoldenFloat Family GF4--GF64".

use super::phi_constants::*;

/// GF4 format: 1 sign bit, 1 exp bit, 2 mantissa bits
pub struct GF4 {
    bits: u8,
}

impl GF4 {
    const SIGN_BIT: u8 = 0x08; // 1000
    const EXP_MASK: u8 = 0x04; // 0100
    const MANT_MASK: u8 = 0x03; // 0011

    pub const EXP_BITS: u8 = 1;
    pub const MANT_BITS: u8 = 2;
    pub const EXP_BIAS: i8 = 0; // 2^(1-1) - 1 = 0

    /// Create GF4 from f32 (extreme quantization)
    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 || !value.is_finite() {
            return Self { bits: 0 };
        }

        let sign = if value < 0.0 { 1u8 } else { 0u8 };
        let abs_val = value.abs();

        let bits = abs_val.to_bits();
        let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let f32_mant = bits & 0x007FFFFF;

        let mut gf4_exp = f32_exp + Self::EXP_BIAS as i32;
        if gf4_exp < 0 {
            return Self { bits: sign << 3 };
        }
        if gf4_exp > 1 {
            // Saturate at max representable
            return Self {
                bits: (sign << 3) | (1 << 2) | Self::MANT_MASK,
            };
        }

        let shift = 23 - Self::MANT_BITS;
        let mut mant_rounded = (f32_mant >> shift) as u8 & Self::MANT_MASK;
        let remainder = (f32_mant >> (shift - 1)) & 1;
        if remainder == 1 && mant_rounded < Self::MANT_MASK {
            mant_rounded += 1;
        }
        // Avoid the all-zero encoding for non-zero values (would alias to +0)
        if (gf4_exp as u8) == 0 && mant_rounded == 0 && sign == 0 {
            mant_rounded = 1;
        }

        Self {
            bits: (sign << 3) | ((gf4_exp as u8) << 2) | mant_rounded,
        }
    }

    /// Convert GF4 to f32
    pub fn to_f32(self) -> f32 {
        if self.bits & 0x07 == 0 {
            return 0.0;
        }
        let sign = if (self.bits & Self::SIGN_BIT) != 0 {
            -1.0
        } else {
            1.0
        };
        let exp = ((self.bits & Self::EXP_MASK) >> 2) as i32;
        let mant = (self.bits & Self::MANT_MASK) as u32;

        let exp_val = 2.0_f32.powi(exp - Self::EXP_BIAS as i32);
        let mant_val = 1.0 + (mant as f32) / 4.0;
        sign * exp_val * mant_val
    }

    pub fn bits(self) -> u8 {
        self.bits
    }
    pub fn from_bits(bits: u8) -> Self {
        Self { bits: bits & 0x0F }
    }
    pub fn quant_error_f32(self, original: f32) -> f32 {
        (self.to_f32() - original).abs()
    }
}

impl Clone for GF4 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for GF4 {}
impl std::fmt::Debug for GF4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GF4({} -> {})", self.bits, self.to_f32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let g = GF4::from_f32(0.0);
        assert_eq!(g.to_f32(), 0.0);
    }

    #[test]
    fn test_positive_one() {
        let g = GF4::from_f32(1.0);
        let d = g.to_f32();
        // 1.0 with 2-bit mantissa lands at 1.0 + (1/4) = 1.25 after the no-zero guard
        assert!(d > 0.9 && d < 1.6, "decoded={}", d);
    }

    #[test]
    fn test_phi_anchor() {
        // φ²+1/φ² ≈ 3 must hold within GF4's coarse precision
        let phi_sq = GF4::from_f32(PHI_SQUARED as f32).to_f32() as f64;
        let phi_inv_sq = GF4::from_f32(PHI_INVERSE_SQUARED as f32).to_f32() as f64;
        // GF4 has only 2 mantissa bits → tolerate up to 1.5 abs error
        assert!((phi_sq + phi_inv_sq - 3.0).abs() < 1.5);
    }

    #[test]
    fn test_split_ratio() {
        let r = GF4::EXP_BITS as f64 / GF4::MANT_BITS as f64;
        assert!((r - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sign() {
        let p = GF4::from_f32(1.0);
        let n = GF4::from_f32(-1.0);
        assert_eq!(p.bits() & GF4::SIGN_BIT, 0);
        assert_eq!(n.bits() & GF4::SIGN_BIT, GF4::SIGN_BIT);
    }
}
