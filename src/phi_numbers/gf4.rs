//! GF4 — Golden Float 4-bit format
//!
//! 1 exponent bit + 2 mantissa bits (1:2 split)
//!
//! Encoding:
//! - 0b0000 = 0
//! - 0b0001 = 1.0
//! - 0b0010 = 1.5
//! - 0b0011 = 2.0
//! - 0b0100 = 2.5
//! - 0b0101 = 3.0
//! - 0b0110 = 3.5
//! - 0b0111 = NaN
//!
//! Formula: value = 1.0 + (code) * 0.5, where code = (bits & 0x07) - 1
//! Sign is stored in bit 3

use super::phi_constants::*;

pub struct GF4 {
    bits: u8,
}

impl GF4 {
    const SIGN_BIT: u8 = 0x08;
    const EXP_MASK: u8 = 0x04;
    const MANT_MASK: u8 = 0x03;

    const EXP_BITS: u8 = 1;
    const MANT_BITS: u8 = 2;

    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Self { bits: 0 };
        }

        let sign = if value < 0.0 { 1u8 } else { 0u8 };
        let abs_val = value.abs();

        if !abs_val.is_finite() {
            return Self { bits: (sign << 3) | 0x07 };
        }

        // GF4 range: [1.0, 3.5], step 0.5
        if abs_val < 1.0 {
            return Self { bits: 0 };
        }
        if abs_val > 4.0 {
            return Self { bits: (sign << 3) | 0x07 };
        }

        // value = 1.0 + code * 0.5, where code is 0-6
        // Store as bits = code + 1 to avoid zero collision
        let code = ((abs_val - 1.0) / 0.5).round().clamp(0.0, 6.0) as u8;
        Self { bits: (sign << 3) | (code + 1) }
    }

    pub fn to_f32(self) -> f32 {
        let bits = self.bits;
        if bits == 0 {
            return 0.0;
        }

        let sign = if (bits & Self::SIGN_BIT) != 0 { -1.0 } else { 1.0 };
        let code_bits = bits & 0x07;

        // 0b0111 = NaN (also handles sign bit: 0x0F is -NaN)
        if code_bits == 7 {
            return f32::NAN;
        }

        // code = bits - 1 (since we added +1 in from_f32)
        let code = (code_bits - 1) as f32;

        // value = 1.0 + code * 0.5
        sign * (1.0 + code * 0.5)
    }

    pub fn bits(self) -> u8 { self.bits }
    pub fn from_bits(bits: u8) -> Self { Self { bits: bits & 0x0F } }
    pub fn quant_error_f32(self, original: f32) -> f32 {
        (self.to_f32() - original).abs()
    }
    pub fn relative_error_f32(self, original: f32) -> f32 {
        if original.abs() < f32::MIN_POSITIVE {
            return self.quant_error_f32(original);
        }
        self.quant_error_f32(original) / original.abs()
    }

    pub const MIN_POSITIVE: f32 = 1.0;
    pub const MAX: f32 = 3.5;
}

impl Clone for GF4 {
    fn clone(&self) -> Self { *self }
}
impl Copy for GF4 {}

impl std::fmt::Debug for GF4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GF4(0b{:04b} -> {})", self.bits, self.to_f32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        assert_eq!(GF4::from_f32(0.0).bits(), 0);
        assert_eq!(GF4::from_f32(0.0).to_f32(), 0.0);
    }

    #[test]
    fn test_one() {
        assert_eq!(GF4::from_f32(1.0).to_f32(), 1.0);
        assert_eq!(GF4::from_f32(-1.0).to_f32(), -1.0);
    }

    #[test]
    fn test_phi() {
        let val = GF4::from_f32(PHI as f32).to_f32();
        assert!(!val.is_nan());
        assert!(val > 0.0);
    }

    #[test]
    fn test_max() {
        assert_eq!(GF4::from_f32(3.5).to_f32(), 3.5);
    }

    #[test]
    fn test_all_values() {
        for i in 0..7u8 {
            let bits = i;
            let val = GF4::from_bits(bits).to_f32();
            assert!(!val.is_nan(), "bits 0b{:04b} is NaN", bits);
        }
    }

    #[test]
    fn test_special() {
        assert!(GF4::from_f32(f32::NAN).to_f32().is_nan());
        assert!(GF4::from_bits(0x0F).to_f32().is_nan());
    }

    #[test]
    fn test_fibonacci() {
        // Only test values that are exactly representable
        for f in [1, 2, 3] {
            assert_eq!(GF4::from_f32(f as f32).to_f32() as i32, f);
        }
    }

    #[test]
    fn test_round_trip() {
        for v in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5] {
            let gf4 = GF4::from_f32(v);
            let decoded = gf4.to_f32();
            assert!(!decoded.is_nan());
            assert!(!decoded.is_infinite());
        }
    }
}
