//! Experimental Golden Float Variants — Alternative φ-splits
//!
//! This module provides alternative split ratios for GF formats to explore
//! different trade-offs between range and precision. Each variant targets
//! specific use cases with φ-optimized bit allocations.

use super::phi_constants::*;

// ============================================================================
// GF12 Variants (Alternative 4:7 split)
// ============================================================================

/// GF12Alt — 12-bit format with 5:6 split (aggressive exponent)
///
/// 5 exp bits, 6 mant bits → wider range, lower precision
/// Split ratio: 5:6 = 0.833 (good for wide-dynamic-range activations)
///
/// Range: ~±2047 (vs ±127 for standard GF12)
/// Precision: 6-bit mantissa → 64 values (vs 128 for standard GF12)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GF12Alt {
    bits: u16,
}

impl GF12Alt {
    const SIGN_BIT: u16 = 0x0800;
    const EXP_MASK: u16 = 0x07C0; // bits 10:6 (5 bits)
    const MANT_MASK: u16 = 0x003F; // bits 5:0 (6 bits)
    const EXP_BITS: u8 = 5;
    const MANT_BITS: u8 = 6;
    const EXP_BIAS: i8 = 15; // 2^(5-1) - 1 = 15

    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Self { bits: 0 };
        }
        let sign = if value < 0.0 { 1u16 } else { 0u16 };
        let abs_val = value.abs();
        if !abs_val.is_finite() {
            if abs_val.is_nan() {
                return Self { bits: (sign << 11) | 0x07FF };
            }
            return Self { bits: (sign << 11) | 0x07C0 };
        }
        let bits = abs_val.to_bits();
        let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let f32_mant = bits & 0x007FFFFF;
        let mut exp = f32_exp + Self::EXP_BIAS as i32;
        if exp < 0 {
            return Self { bits: 0 };
        }
        if exp >= 31 {
            return Self { bits: (sign << 11) | (30 << 6) | 63 };
        }
        let shift = 23 - Self::MANT_BITS;
        let mut mant = (f32_mant >> shift) as u16;
        if (f32_mant >> (shift - 1)) & 1 == 1 {
            mant += 1;
        }
        if mant >= 64 {
            mant = 0;
            exp += 1;
            if exp >= 31 {
                return Self { bits: (sign << 11) | (30 << 6) | 63 };
            }
        }
        Self { bits: (sign << 11) | ((exp as u16) << 6) | mant }
    }

    pub fn to_f32(self) -> f32 {
        if self.bits == 0 {
            return 0.0;
        }
        let sign = if (self.bits & Self::SIGN_BIT) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = ((self.bits & Self::EXP_MASK) >> 6) as i32;
        let mant = (self.bits & Self::MANT_MASK) as u32;
        if exp == 31 {
            return if mant == 0 { sign * f32::INFINITY } else { f32::NAN };
        }
        let exp_val = 2.0_f32.powi(exp - Self::EXP_BIAS as i32);
        sign * exp_val * (1.0 + (mant as f32) / 64.0)
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

/// GF12Alt2 — 12-bit format with 3:8 split (aggressive mantissa)
///
/// 3 exp bits, 8 mant bits → narrow range, high precision
/// Split ratio: 3:8 = 0.375 (good for quantized embeddings)
///
/// Range: ~±7.9 (vs ±127 for standard GF12)
/// Precision: 8-bit mantissa → 256 values (vs 128 for standard GF12)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GF12Alt2 {
    bits: u16,
}

impl GF12Alt2 {
    const SIGN_BIT: u16 = 0x0800;
    const EXP_MASK: u16 = 0x0700; // bits 10:8 (3 bits)
    const MANT_MASK: u16 = 0x00FF; // bits 7:0 (8 bits)
    const EXP_BITS: u8 = 3;
    const MANT_BITS: u8 = 8;
    const EXP_BIAS: i8 = 3; // 2^(3-1) - 1 = 3

    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Self { bits: 0 };
        }
        let sign = if value < 0.0 { 1u16 } else { 0u16 };
        let abs_val = value.abs();
        if !abs_val.is_finite() {
            if abs_val.is_nan() {
                return Self { bits: (sign << 11) | 0x07FF };
            }
            return Self { bits: (sign << 11) | 0x0700 };
        }
        let bits = abs_val.to_bits();
        let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let f32_mant = bits & 0x007FFFFF;
        let mut exp = f32_exp + Self::EXP_BIAS as i32;
        if exp < 0 {
            return Self { bits: 0 };
        }
        if exp >= 7 {
            return Self { bits: (sign << 11) | (6 << 8) | 255 };
        }
        let shift = 23 - Self::MANT_BITS;
        let mut mant = (f32_mant >> shift) as u16;
        if (f32_mant >> (shift - 1)) & 1 == 1 {
            mant += 1;
        }
        if mant >= 256 {
            mant = 0;
            exp += 1;
            if exp >= 7 {
                return Self { bits: (sign << 11) | (6 << 8) | 255 };
            }
        }
        Self { bits: (sign << 11) | ((exp as u16) << 8) | mant }
    }

    pub fn to_f32(self) -> f32 {
        if self.bits == 0 {
            return 0.0;
        }
        let sign = if (self.bits & Self::SIGN_BIT) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = ((self.bits & Self::EXP_MASK) >> 8) as i32;
        let mant = (self.bits & Self::MANT_MASK) as u32;
        if exp == 7 {
            return if mant == 0 { sign * f32::INFINITY } else { f32::NAN };
        }
        let exp_val = 2.0_f32.powi(exp - Self::EXP_BIAS as i32);
        sign * exp_val * (1.0 + (mant as f32) / 256.0)
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

// ============================================================================
// GF20 Variants (Alternative 7:12 split)
// ============================================================================

/// GF20Alt — 20-bit format with 8:11 split (better range)
///
/// 8 exp bits, 11 mant bits → wider range for small models
/// Split ratio: 8:11 = 0.727 (good for attention weights with wide dynamic range)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GF20Alt {
    bits: u32,
}

impl GF20Alt {
    const SIGN_BIT: u32 = 0x00080000;
    const EXP_MASK: u32 = 0x0007F800; // bits 18:11 (8 bits)
    const MANT_MASK: u32 = 0x000007FF; // bits 10:0 (11 bits)
    const EXP_BITS: u8 = 8;
    const MANT_BITS: u8 = 11;
    const EXP_BIAS: i16 = 127;

    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Self { bits: 0 };
        }
        let sign = if value < 0.0 { 1u32 } else { 0u32 };
        let abs_val = value.abs();
        if !abs_val.is_finite() {
            if abs_val.is_nan() {
                return Self { bits: sign << 19 | 0x0007FFFF };
            }
            return Self { bits: sign << 19 | 0x0007F800 };
        }
        let bits = abs_val.to_bits();
        let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let f32_mant = bits & 0x007FFFFF;
        let mut exp = f32_exp + Self::EXP_BIAS as i32;
        if exp < 0 {
            return Self { bits: 0 };
        }
        if exp >= 255 {
            return Self { bits: sign << 19 | (254 << 11) | 2047 };
        }
        let shift = 23 - Self::MANT_BITS;
        let mut mant = (f32_mant >> shift) as u32;
        if (f32_mant >> (shift - 1)) & 1 == 1 {
            mant += 1;
        }
        if mant >= 2048 {
            mant = 0;
            exp += 1;
            if exp >= 255 {
                return Self { bits: sign << 19 | (254 << 11) | 2047 };
            }
        }
        Self { bits: sign << 19 | ((exp as u32) << 11) | mant }
    }

    pub fn to_f32(self) -> f32 {
        if self.bits == 0 {
            return 0.0;
        }
        let sign = if (self.bits & Self::SIGN_BIT) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = ((self.bits & Self::EXP_MASK) >> 11) as i32;
        let mant = (self.bits & Self::MANT_MASK) as u32;
        if exp == 255 {
            return if mant == 0 { sign * f32::INFINITY } else { f32::NAN };
        }
        let exp_val = 2.0_f32.powi(exp - Self::EXP_BIAS as i32);
        sign * exp_val * (1.0 + (mant as f32) / 2048.0)
    }

    pub fn bits(self) -> u32 { self.bits }
    pub fn from_bits(bits: u32) -> Self { Self { bits: bits & 0x000FFFFF } }
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

// ============================================================================
// GF24 Variants (Alternative 8:15 split)
// ============================================================================

/// GF24Alt — 24-bit format with 9:14 split (closer to 1/φ)
///
/// 9 exp bits, 14 mant bits → split ratio ≈ 0.643 (very close to 1/φ ≈ 0.618)
/// This variant is optimally φ-balanced for 24 bits
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GF24Alt {
    bits: u32,
}

impl GF24Alt {
    const SIGN_BIT: u32 = 0x00800000;
    const EXP_MASK: u32 = 0x007FC000; // bits 22:14 (9 bits)
    const MANT_MASK: u32 = 0x00003FFF; // bits 13:0 (14 bits)
    const EXP_BITS: u8 = 9;
    const MANT_BITS: u8 = 14;
    const EXP_BIAS: i16 = 255;

    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Self { bits: 0 };
        }
        let sign = if value < 0.0 { 1u32 } else { 0u32 };
        let abs_val = value.abs();
        if !abs_val.is_finite() {
            if abs_val.is_nan() {
                return Self { bits: sign << 23 | 0x007FFFFF };
            }
            return Self { bits: sign << 23 | 0x007FC000 };
        }
        let bits = abs_val.to_bits();
        let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let f32_mant = bits & 0x007FFFFF;
        let mut exp = f32_exp + Self::EXP_BIAS as i32;
        if exp < 0 {
            return Self { bits: 0 };
        }
        if exp >= 511 {
            return Self { bits: sign << 23 | (510 << 14) | 16383 };
        }
        let shift = 23 - Self::MANT_BITS;
        let mut mant = (f32_mant >> shift) as u32;
        if (f32_mant >> (shift - 1)) & 1 == 1 {
            mant += 1;
        }
        if mant >= 16384 {
            mant = 0;
            exp += 1;
            if exp >= 511 {
                return Self { bits: sign << 23 | (510 << 14) | 16383 };
            }
        }
        Self { bits: sign << 23 | ((exp as u32) << 14) | mant }
    }

    pub fn to_f32(self) -> f32 {
        if self.bits == 0 {
            return 0.0;
        }
        let sign = if (self.bits & Self::SIGN_BIT) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = ((self.bits & Self::EXP_MASK) >> 14) as i32;
        let mant = (self.bits & Self::MANT_MASK) as u32;
        if exp == 511 {
            return if mant == 0 { sign * f32::INFINITY } else { f32::NAN };
        }
        let exp_val = 2.0_f32.powi(exp - Self::EXP_BIAS as i32);
        sign * exp_val * (1.0 + (mant as f32) / 16384.0)
    }

    pub fn bits(self) -> u32 { self.bits }
    pub fn from_bits(bits: u32) -> Self { Self { bits: bits & 0x00FFFFFF } }
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

/// GF24Alt2 — 24-bit format with 10:13 split (aggressive exponent)
///
/// 10 exp bits, 13 mant bits → split ratio ≈ 0.769
/// Very wide range for models with extreme dynamic range
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GF24Alt2 {
    bits: u32,
}

impl GF24Alt2 {
    const SIGN_BIT: u32 = 0x00800000;
    const EXP_MASK: u32 = 0x007FE000; // bits 22:13 (10 bits)
    const MANT_MASK: u32 = 0x00001FFF; // bits 12:0 (13 bits)
    const EXP_BITS: u8 = 10;
    const MANT_BITS: u8 = 13;
    const EXP_BIAS: i16 = 511;

    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Self { bits: 0 };
        }
        let sign = if value < 0.0 { 1u32 } else { 0u32 };
        let abs_val = value.abs();
        if !abs_val.is_finite() {
            if abs_val.is_nan() {
                return Self { bits: sign << 23 | 0x007FFFFF };
            }
            return Self { bits: sign << 23 | 0x007FE000 };
        }
        let bits = abs_val.to_bits();
        let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let f32_mant = bits & 0x007FFFFF;
        let mut exp = f32_exp + Self::EXP_BIAS as i32;
        if exp < 0 {
            return Self { bits: 0 };
        }
        if exp >= 1023 {
            return Self { bits: sign << 23 | (1022 << 13) | 8191 };
        }
        let shift = 23 - Self::MANT_BITS;
        let mut mant = (f32_mant >> shift) as u32;
        if (f32_mant >> (shift - 1)) & 1 == 1 {
            mant += 1;
        }
        if mant >= 8192 {
            mant = 0;
            exp += 1;
            if exp >= 1023 {
                return Self { bits: sign << 23 | (1022 << 13) | 8191 };
            }
        }
        Self { bits: sign << 23 | ((exp as u32) << 13) | mant }
    }

    pub fn to_f32(self) -> f32 {
        if self.bits == 0 {
            return 0.0;
        }
        let sign = if (self.bits & Self::SIGN_BIT) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = ((self.bits & Self::EXP_MASK) >> 13) as i32;
        let mant = (self.bits & Self::MANT_MASK) as u32;
        if exp == 1023 {
            return if mant == 0 { sign * f32::INFINITY } else { f32::NAN };
        }
        let exp_val = 2.0_f32.powi(exp - Self::EXP_BIAS as i32);
        sign * exp_val * (1.0 + (mant as f32) / 8192.0)
    }

    pub fn bits(self) -> u32 { self.bits }
    pub fn from_bits(bits: u32) -> Self { Self { bits: bits & 0x00FFFFFF } }
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gf12alt_no_nan() {
        let gf = GF12Alt::from_f32(PHI as f32);
        assert!(!gf.to_f32().is_nan());
    }

    #[test]
    fn test_gf12alt2_no_nan() {
        let gf = GF12Alt2::from_f32(PHI as f32);
        assert!(!gf.to_f32().is_nan());
    }

    #[test]
    fn test_gf20alt_no_nan() {
        let gf = GF20Alt::from_f32(PHI as f32);
        assert!(!gf.to_f32().is_nan());
    }

    #[test]
    fn test_gf24alt_no_nan() {
        let gf = GF24Alt::from_f32(PHI as f32);
        assert!(!gf.to_f32().is_nan());
    }

    #[test]
    fn test_gf24alt2_no_nan() {
        let gf = GF24Alt2::from_f32(PHI as f32);
        assert!(!gf.to_f32().is_nan());
    }

    #[test]
    fn test_gf24alt_phi_optimized() {
        // GF24Alt has 9:14 split ≈ 0.643, closest to 1/φ = 0.618
        assert_eq!(GF24Alt::EXP_BITS as f64 / GF24Alt::MANT_BITS as f64, 9.0 / 14.0);
        let phi_diff = (9.0 / 14.0 - PHI_CONJUGATE).abs();
        assert!(phi_diff < 0.03, "Should be very close to 1/φ");
    }

    #[test]
    fn test_all_varants_round_trip() {
        let test_val = 1.618_f32;
        let _ = GF12Alt::from_f32(test_val).to_f32();
        let _ = GF12Alt2::from_f32(test_val).to_f32();
        let _ = GF20Alt::from_f32(test_val).to_f32();
        let _ = GF24Alt::from_f32(test_val).to_f32();
        let _ = GF24Alt2::from_f32(test_val).to_f32();
    }
}
