//! GF24 — Golden Float 24-bit format
//!
//! 1 sign + 9 exp + 14 mantissa bits (ratio 9/14 ≈ 0.643, between GF16 and GF32).
//! PhD Glava 06.

use super::phi_constants::*;

pub struct GF24 {
    bits: u32,
}

impl GF24 {
    const SIGN_BIT: u32 = 1 << 23;
    const EXP_MASK: u32 = ((1u32 << 9) - 1) << 14;
    const MANT_MASK: u32 = (1u32 << 14) - 1;

    pub const EXP_BITS: u8 = 9;
    pub const MANT_BITS: u8 = 14;
    pub const EXP_BIAS: i16 = 255; // 2^(9-1)-1

    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Self { bits: 0 };
        }
        let sign = if value < 0.0 { 1u32 } else { 0u32 };
        let abs_val = value.abs();
        if !abs_val.is_finite() {
            let nan_bit = if abs_val.is_nan() { 1 } else { 0 };
            return Self {
                bits: (sign << 23) | Self::EXP_MASK | nan_bit,
            };
        }
        let bits = abs_val.to_bits();
        let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let f32_mant = bits & 0x007FFFFF;

        let mut g_exp = f32_exp + Self::EXP_BIAS as i32;
        if g_exp < 0 {
            return Self { bits: 0 };
        }
        if g_exp >= 0x1FF {
            return Self {
                bits: (sign << 23) | (0x1FE << 14) | Self::MANT_MASK,
            };
        }

        let shift = 23 - Self::MANT_BITS;
        let mut mant_rounded = (f32_mant >> shift) as u32;
        let remainder = (f32_mant >> (shift - 1)) & 1;
        if remainder == 1 {
            mant_rounded += 1;
        }
        if mant_rounded >= (1u32 << Self::MANT_BITS) {
            mant_rounded = 0;
            g_exp += 1;
            if g_exp >= 0x1FF {
                return Self {
                    bits: (sign << 23) | (0x1FE << 14) | Self::MANT_MASK,
                };
            }
        }

        Self {
            bits: (sign << 23) | ((g_exp as u32) << 14) | mant_rounded,
        }
    }

    pub fn to_f32(self) -> f32 {
        if self.bits == 0 {
            return 0.0;
        }
        let sign = if (self.bits & Self::SIGN_BIT) != 0 {
            -1.0
        } else {
            1.0
        };
        let exp = ((self.bits & Self::EXP_MASK) >> 14) as i32;
        let mant = self.bits & Self::MANT_MASK;
        if exp == 0x1FF {
            return if mant == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            };
        }
        let exp_val = 2.0_f32.powi(exp - Self::EXP_BIAS as i32);
        let mant_val = 1.0 + (mant as f32) / 16384.0;
        sign * exp_val * mant_val
    }

    pub fn bits(self) -> u32 {
        self.bits
    }
    pub fn from_bits(bits: u32) -> Self {
        Self {
            bits: bits & 0x00FF_FFFF,
        }
    }
    pub fn quant_error_f32(self, original: f32) -> f32 {
        (self.to_f32() - original).abs()
    }
    pub fn relative_error_f32(self, original: f32) -> f32 {
        if original.abs() < f32::MIN_POSITIVE {
            return self.quant_error_f32(original);
        }
        self.quant_error_f32(original) / original.abs()
    }
}

impl Clone for GF24 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for GF24 {}
impl std::fmt::Debug for GF24 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GF24({} -> {})", self.bits, self.to_f32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        assert_eq!(GF24::from_f32(0.0).to_f32(), 0.0);
    }

    #[test]
    fn test_phi_precision() {
        let g = GF24::from_f32(PHI as f32);
        assert!(g.relative_error_f32(PHI as f32) < 1e-4);
    }

    #[test]
    fn test_trinity_identity() {
        let s = GF24::from_f32(PHI_SQUARED as f32).to_f32() as f64
            + GF24::from_f32(PHI_INVERSE_SQUARED as f32).to_f32() as f64;
        assert!((s - 3.0).abs() < 1e-4, "trinity={}", s);
    }

    #[test]
    fn test_split_ratio() {
        let r = GF24::EXP_BITS as f64 / GF24::MANT_BITS as f64;
        assert!((r - 9.0 / 14.0).abs() < 1e-10);
    }
}
