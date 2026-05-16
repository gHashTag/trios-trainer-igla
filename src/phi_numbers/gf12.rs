//! GF12 — Golden Float 12-bit format
//!
//! 1 sign + 4 exp + 7 mantissa bits (ratio 4/7 ≈ 0.571, Fibonacci F4:F5).
//! Bridge between GF8 (3:4) and GF16 (6:9). Aligns with PhD Glava 06.

use super::phi_constants::*;

/// GF12 format: 1 sign, 4 exp, 7 mantissa bits (stored in u16, low 12 bits)
pub struct GF12 {
    bits: u16,
}

impl GF12 {
    const SIGN_BIT: u16 = 0x0800; // bit 11
    const EXP_MASK: u16 = 0x0780; // bits 10-7
    const MANT_MASK: u16 = 0x007F; // bits 6-0

    pub const EXP_BITS: u8 = 4;
    pub const MANT_BITS: u8 = 7;
    pub const EXP_BIAS: i16 = 7; // 2^(4-1) - 1

    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Self { bits: 0 };
        }
        let sign = if value < 0.0 { 1u16 } else { 0u16 };
        let abs_val = value.abs();

        if !abs_val.is_finite() {
            let nan_bit = if abs_val.is_nan() { 1 } else { 0 };
            return Self {
                bits: (sign << 11) | Self::EXP_MASK | nan_bit,
            };
        }

        let bits = abs_val.to_bits();
        let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let f32_mant = bits & 0x007FFFFF;

        let mut g_exp = f32_exp + Self::EXP_BIAS as i32;
        if g_exp < 0 {
            return Self { bits: 0 };
        }
        if g_exp >= 0x0F {
            return Self {
                bits: (sign << 11) | ((0x0E) << 7) | Self::MANT_MASK,
            };
        }

        let shift = 23 - Self::MANT_BITS;
        let mut mant_rounded = (f32_mant >> shift) as u16;
        let remainder = (f32_mant >> (shift - 1)) & 1;
        if remainder == 1 {
            mant_rounded += 1;
        }
        if mant_rounded >= (1u16 << Self::MANT_BITS) {
            mant_rounded = 0;
            g_exp += 1;
            if g_exp >= 0x0F {
                return Self {
                    bits: (sign << 11) | ((0x0E) << 7) | Self::MANT_MASK,
                };
            }
        }

        Self {
            bits: (sign << 11) | ((g_exp as u16) << 7) | mant_rounded,
        }
    }

    pub fn to_f32(self) -> f32 {
        if self.bits == 0 {
            return 0.0;
        }
        let sign = if (self.bits & Self::SIGN_BIT) != 0 { -1.0 } else { 1.0 };
        let exp = ((self.bits & Self::EXP_MASK) >> 7) as i32;
        let mant = (self.bits & Self::MANT_MASK) as u32;

        if exp == 0x0F {
            return if mant == 0 { sign * f32::INFINITY } else { f32::NAN };
        }

        let exp_val = 2.0_f32.powi(exp - Self::EXP_BIAS as i32);
        let mant_val = 1.0 + (mant as f32) / 128.0;
        sign * exp_val * mant_val
    }

    pub fn bits(self) -> u16 { self.bits }
    pub fn from_bits(bits: u16) -> Self { Self { bits: bits & 0x0FFF } }
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

impl Clone for GF12 { fn clone(&self) -> Self { *self } }
impl Copy for GF12 {}
impl std::fmt::Debug for GF12 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GF12({} -> {})", self.bits, self.to_f32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() { assert_eq!(GF12::from_f32(0.0).to_f32(), 0.0); }

    #[test]
    fn test_one() {
        let g = GF12::from_f32(1.0);
        assert!((g.to_f32() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_phi() {
        let g = GF12::from_f32(PHI as f32);
        assert!(g.relative_error_f32(PHI as f32) < 0.01);
    }

    #[test]
    fn test_trinity_identity() {
        let s = GF12::from_f32(PHI_SQUARED as f32).to_f32() as f64
              + GF12::from_f32(PHI_INVERSE_SQUARED as f32).to_f32() as f64;
        assert!((s - 3.0).abs() < 0.05, "trinity={}", s);
    }

    #[test]
    fn test_split_ratio() {
        let r = GF12::EXP_BITS as f64 / GF12::MANT_BITS as f64;
        assert!((r - 4.0/7.0).abs() < 1e-10);
    }

    #[test]
    fn test_round_trip() {
        for v in [0.5_f32, 1.0, PHI as f32, 2.0, 8.0, 100.0] {
            let g = GF12::from_f32(v);
            assert!(g.relative_error_f32(v) < 0.02, "v={} err={}", v, g.relative_error_f32(v));
        }
    }
}
