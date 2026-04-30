//! Precision Format System — 13 formats for PhD gradient experiments
//!
//! This module defines all 13 precision formats:
//! - 10 GoldenFloat formats: gf4, gf4a, gf6a, gf8, gf12, gf16, gf20, gf24, gf32, gf64
//! - 3 IEEE baselines: fp16, bf16, fp32
//!
//! PhD context: 12 formats × 5 architectures × 5 seeds = 300 experiments pre-staged in Neon.
//! Gradient data flows: bpb_samples + grad_norm per step.

use super::phi_constants::*;
use super::{gf12, gf20, gf24, gf32, gf4, gf64, gf8, GFTernary};

// Re-export gf16 from crate level (it's not in phi_numbers)
pub use crate::gf16;

// ============================================================================
// GF6a — Golden Float 6-bit Adaptive
// ============================================================================

/// GF6a — 6-bit adaptive format with dynamic range
///
/// 2 exponent bits + 3 mantissa bits (2:3 split) + 1 sign bit
/// Adaptive scaling for activations in narrow dynamic range
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GF6a {
    bits: u8,
}

impl GF6a {
    const SIGN_BIT: u8 = 0x40; // 0100_0000
    const EXP_MASK: u8 = 0x30; // 0011_0000
    const MANT_MASK: u8 = 0x0F; // 0000_1111

    const EXP_BITS: u8 = 2;
    const MANT_BITS: u8 = 3;
    const EXP_BIAS: i8 = 1; // 2^(2-1) - 1 = 1

    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Self { bits: 0 };
        }

        let sign = if value < 0.0 { 1 } else { 0 };
        let abs_val = value.abs();

        if !abs_val.is_finite() {
            if abs_val.is_nan() {
                return Self { bits: (sign << 6) | 0x3F };
            }
            return Self { bits: (sign << 6) | 0x30 };
        }

        let bits = abs_val.to_bits();
        let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let f32_mant = bits & 0x007FFFFF;

        let mut gf6a_exp = f32_exp + Self::EXP_BIAS as i32;

        if gf6a_exp < 0 {
            return Self { bits: 0 };
        }
        if gf6a_exp >= 4 {
            return Self { bits: (sign << 6) | (3 << 3) | 7 };
        }

        let shift = 23 - Self::MANT_BITS;
        let mut mant_rounded = (f32_mant >> shift) as u8;
        let remainder = (f32_mant >> (shift - 1)) & 1;

        if remainder == 1 {
            mant_rounded += 1;
        }

        if mant_rounded >= 8 {
            mant_rounded = 0;
            gf6a_exp += 1;
            if gf6a_exp >= 4 {
                return Self { bits: (sign << 6) | (3 << 3) | 7 };
            }
        }

        Self {
            bits: (sign << 6) | ((gf6a_exp as u8) << 3) | mant_rounded,
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
        let exp = ((self.bits & Self::EXP_MASK) >> 3) as i32;
        let mant = (self.bits & Self::MANT_MASK) as u32;

        if exp == 3 {
            return if mant == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            };
        }

        let exp_val = 2.0_f32.powi(exp - Self::EXP_BIAS as i32);
        let mant_val = 1.0 + (mant as f32) / 8.0;

        sign * exp_val * mant_val
    }

    pub fn bits(self) -> u8 { self.bits }
    pub fn from_bits(bits: u8) -> Self { Self { bits: bits & 0x7F } }
    pub fn quant_error_f32(self, original: f32) -> f32 {
        (self.to_f32() - original).abs()
    }
    pub fn relative_error_f32(self, original: f32) -> f32 {
        if original.abs() < f32::MIN_POSITIVE {
            return self.quant_error_f32(original);
        }
        self.quant_error_f32(original) / original.abs()
    }

    pub const MIN_POSITIVE: f32 = 0.25; // 2^(-2)
    pub const MAX: f32 = 7.0;
}

// ============================================================================
// GF4a — Golden Float 4-bit Adaptive
// ============================================================================

/// GF4a — 4-bit adaptive format with logarithmic scaling
///
/// 1 exponent bit + 2 mantissa bits (1:2 split) + 1 sign bit
/// Logarithmic encoding for extreme quantization scenarios
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GF4a {
    bits: u8,
}

impl GF4a {
    const SIGN_BIT: u8 = 0x08; // 0000_1000
    const EXP_MASK: u8 = 0x04; // 0000_0100
    const MANT_MASK: u8 = 0x03; // 0000_0011

    const EXP_BITS: u8 = 1;
    const MANT_BITS: u8 = 2;
    const EXP_BIAS: i8 = 0; // 2^(1-1) - 1 = 0

    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Self { bits: 0 };
        }

        let sign = if value < 0.0 { 1 } else { 0 };
        let abs_val = value.abs();

        if !abs_val.is_finite() {
            if abs_val.is_nan() {
                return Self { bits: (sign << 3) | 0x07 };
            }
            return Self { bits: (sign << 3) | 0x04 };
        }

        let bits = abs_val.to_bits();
        let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let f32_mant = bits & 0x007FFFFF;

        let mut gf4a_exp = f32_exp + Self::EXP_BIAS as i32;

        if gf4a_exp < 0 {
            return Self { bits: 0 };
        }
        if gf4a_exp >= 2 {
            return Self { bits: (sign << 3) | (1 << 2) | 3 };
        }

        let shift = 23 - Self::MANT_BITS;
        let mut mant_rounded = (f32_mant >> shift) as u8;
        let remainder = (f32_mant >> (shift - 1)) & 1;

        if remainder == 1 {
            mant_rounded += 1;
        }

        if mant_rounded >= 4 {
            mant_rounded = 0;
            gf4a_exp += 1;
            if gf4a_exp >= 2 {
                return Self { bits: (sign << 3) | (1 << 2) | 3 };
            }
        }

        Self {
            bits: (sign << 3) | ((gf4a_exp as u8) << 2) | mant_rounded,
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
        let exp = ((self.bits & Self::EXP_MASK) >> 2) as i32;
        let mant = (self.bits & Self::MANT_MASK) as u32;

        if exp == 1 {
            return if mant == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            };
        }

        let exp_val = 2.0_f32.powi(exp - Self::EXP_BIAS as i32);
        let mant_val = 1.0 + (mant as f32) / 4.0;

        sign * exp_val * mant_val
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

    pub const MIN_POSITIVE: f32 = 0.5; // 2^(-1)
    pub const MAX: f32 = 3.0;
}

// ============================================================================
// IEEE Formats
// ============================================================================

/// FP16 — IEEE 754 half-precision (16-bit)
///
/// Standard IEEE 754 binary16 format
/// 1 sign | 5 exponent | 10 mantissa
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FP16 {
    bits: u16,
}

impl FP16 {
    const SIGN_BIT: u16 = 0x8000;
    const EXP_MASK: u16 = 0x7C00;
    const MANT_MASK: u16 = 0x03FF;
    const EXP_BIAS: i16 = 15;

    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Self {
                bits: if value.is_sign_negative() {
                    Self::NEG_ZERO
                } else {
                    Self::ZERO
                },
            };
        }
        if value.is_nan() {
            return Self { bits: Self::NAN };
        }
        if !value.is_finite() {
            return Self {
                bits: if value.is_sign_negative() {
                    Self::NEG_INF
                } else {
                    Self::INF
                },
            };
        }

        let bits = value.to_bits();
        let sign = ((bits >> 31) & 1) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + Self::EXP_BIAS as i32;
        let mant = ((bits & 0x007F_FFFF) >> 13) as u16;

        if exp <= 0 {
            // Subnormal or underflow
            Self { bits: sign << 15 }
        } else if exp >= 31 {
            // Overflow
            Self {
                bits: sign << 15 | 0x7C00,
            }
        } else {
            Self {
                bits: sign << 15 | ((exp as u16) << 10) | mant,
            }
        }
    }

    pub fn to_f32(&self) -> f32 {
        let bits = self.bits;
        let sign = (bits & Self::SIGN_BIT) >> 15;
        let exp = (bits & Self::EXP_MASK) >> 10;
        let mant = bits & Self::MANT_MASK;

        if exp == 0 {
            return if sign != 0 { -0.0_f32 } else { 0.0_f32 };
        }

        if exp == 31 {
            return if mant == 0 {
                if sign != 0 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            } else {
                f32::NAN
            };
        }

        let f32_exp = exp as i32 - Self::EXP_BIAS as i32 + 127;
        let f32_mant = (mant as u32) << 13;

        f32::from_bits((sign as u32) << 31 | (f32_exp as u32) << 23 | f32_mant)
    }

    pub fn bits(self) -> u16 { self.bits }
    pub fn from_bits(bits: u16) -> Self { Self { bits } }

    const ZERO: u16 = 0x0000;
    const NEG_ZERO: u16 = 0x8000;
    const INF: u16 = 0x7C00;
    const NEG_INF: u16 = 0xFC00;
    const NAN: u16 = 0x7FFF;
}

/// BF16 — Brain Floating Point (16-bit)
///
/// bfloat16 format: 1 sign | 8 exponent | 7 mantissa
/// Same exponent range as f32, truncated mantissa
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BF16 {
    bits: u16,
}

impl BF16 {
    const SIGN_BIT: u16 = 0x8000;
    const EXP_MASK: u16 = 0x7F80;
    const MANT_MASK: u16 = 0x007F;

    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Self {
                bits: if value.is_sign_negative() {
                    Self::NEG_ZERO
                } else {
                    Self::ZERO
                },
            };
        }
        if value.is_nan() {
            return Self { bits: Self::NAN };
        }
        if !value.is_finite() {
            return Self {
                bits: if value.is_sign_negative() {
                    Self::NEG_INF
                } else {
                    Self::INF
                },
            };
        }

        let bits = value.to_bits();
        let sign = ((bits >> 31) & 1) as u16;
        let exp = ((bits >> 23) & 0xFF) as u16;
        let mant = ((bits >> 16) & 0x7F) as u16;

        Self {
            bits: sign << 15 | (exp << 7) | mant,
        }
    }

    pub fn to_f32(&self) -> f32 {
        let bits = self.bits;
        let sign = (bits & Self::SIGN_BIT) >> 15;
        let exp = (bits & Self::EXP_MASK) >> 7;
        let mant = bits & Self::MANT_MASK;

        if exp == 0 && mant == 0 {
            return if sign != 0 { -0.0_f32 } else { 0.0_f32 };
        }

        if exp == 255 {
            return if mant == 0 {
                if sign != 0 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            } else {
                f32::NAN
            };
        }

        let f32_mant = (mant as u32) << 16;
        f32::from_bits((sign as u32) << 31 | (exp as u32) << 23 | f32_mant)
    }

    pub fn bits(self) -> u16 { self.bits }
    pub fn from_bits(bits: u16) -> Self { Self { bits } }

    const ZERO: u16 = 0x0000;
    const NEG_ZERO: u16 = 0x8000;
    const INF: u16 = 0x7F80;
    const NEG_INF: u16 = 0xFF80;
    const NAN: u16 = 0x7FC0;
}

/// FP32 — IEEE 754 single precision (32-bit)
///
/// Native Rust f32 type wrapper for consistency
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FP32 {
    value: f32,
}

impl FP32 {
    pub fn from_f32(value: f32) -> Self { Self { value } }
    pub fn to_f32(&self) -> f32 { self.value }
    pub fn bits(self) -> u32 { self.value.to_bits() }
    pub fn from_bits(bits: u32) -> Self { Self { value: f32::from_bits(bits) } }
}

// ============================================================================
// Precision Format Enum
// ============================================================================

/// All 12 precision formats for PhD gradient experiments
///
/// PhD context:
/// - 12 formats × 5 architectures × 5 seeds = 300 experiments
/// - Pre-staged in Neon database
/// - Gradient data: bpb_samples + grad_norm per step
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PrecisionFormat {
    // GoldenFloat formats (9)
    GF4,
    GF4a,
    GF6a,
    GF8,
    GF12,
    GF16,
    GF20,
    GF24,
    GF32,
    GF64,

    // IEEE baselines (3)
    FP16,
    BF16,
    FP32,
}

impl PrecisionFormat {
    /// Get bit width of this format
    pub fn bit_width(&self) -> u8 {
        match self {
            Self::GF4 | Self::GF4a => 4,
            Self::GF6a => 6,
            Self::GF8 => 8,
            Self::GF12 => 12,
            Self::GF16 | Self::FP16 | Self::BF16 => 16,
            Self::GF20 => 20,
            Self::GF24 => 24,
            Self::GF32 => 32,
            Self::GF64 => 64,
            Self::FP32 => 32,
        }
    }

    /// Get format category (GoldenFloat or IEEE)
    pub fn category(&self) -> &'static str {
        match self {
            Self::GF4 | Self::GF4a | Self::GF6a | Self::GF8 | Self::GF12 | Self::GF16
            | Self::GF20 | Self::GF24 | Self::GF32 | Self::GF64 => "GoldenFloat",
            Self::FP16 | Self::BF16 | Self::FP32 => "IEEE",
        }
    }

    /// Parse format from string (case-insensitive)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "gf4" => Some(Self::GF4),
            "gf4a" => Some(Self::GF4a),
            "gf6a" => Some(Self::GF6a),
            "gf8" => Some(Self::GF8),
            "gf12" => Some(Self::GF12),
            "gf16" => Some(Self::GF16),
            "gf20" => Some(Self::GF20),
            "gf24" => Some(Self::GF24),
            "gf32" => Some(Self::GF32),
            "gf64" => Some(Self::GF64),
            "fp16" => Some(Self::FP16),
            "bf16" => Some(Self::BF16),
            "fp32" => Some(Self::FP32),
            _ => None,
        }
    }

    /// Get all formats in standard order
    pub fn all_formats() -> Vec<Self> {
        vec![
            Self::GF4,
            Self::GF4a,
            Self::GF6a,
            Self::GF8,
            Self::GF12,
            Self::GF16,
            Self::GF20,
            Self::GF24,
            Self::GF32,
            Self::GF64,
            Self::FP16,
            Self::BF16,
            Self::FP32,
        ]
    }

    /// Get only GoldenFloat formats
    pub fn golden_float_formats() -> Vec<Self> {
        vec![
            Self::GF4,
            Self::GF4a,
            Self::GF6a,
            Self::GF8,
            Self::GF12,
            Self::GF16,
            Self::GF20,
            Self::GF24,
            Self::GF32,
            Self::GF64,
        ]
    }

    /// Get only IEEE baseline formats
    pub fn ieee_formats() -> Vec<Self> {
        vec![Self::FP16, Self::BF16, Self::FP32]
    }
}

impl std::fmt::Display for PrecisionFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ============================================================================
// Quantization Trait for All Formats
// ============================================================================

/// Common trait for quantization across all formats
pub trait Quantize {
    fn quantize_f32(value: f32) -> Self;
    fn dequantize_f32(&self) -> f32;
}

impl Quantize for gf4::GF4 {
    fn quantize_f32(value: f32) -> Self { gf4::GF4::from_f32(value) }
    fn dequantize_f32(&self) -> f32 { self.to_f32() }
}

impl Quantize for GF4a {
    fn quantize_f32(value: f32) -> Self { GF4a::from_f32(value) }
    fn dequantize_f32(&self) -> f32 { self.to_f32() }
}

impl Quantize for GF6a {
    fn quantize_f32(value: f32) -> Self { GF6a::from_f32(value) }
    fn dequantize_f32(&self) -> f32 { self.to_f32() }
}

impl Quantize for gf8::GF8 {
    fn quantize_f32(value: f32) -> Self { gf8::GF8::from_f32(value) }
    fn dequantize_f32(&self) -> f32 { self.to_f32() }
}

impl Quantize for gf12::GF12 {
    fn quantize_f32(value: f32) -> Self { gf12::GF12::from_f32(value) }
    fn dequantize_f32(&self) -> f32 { self.to_f32() }
}

impl Quantize for gf16::GF16 {
    fn quantize_f32(value: f32) -> Self { gf16::GF16::from_f32(value) }
    fn dequantize_f32(&self) -> f32 { self.to_f32() }
}

impl Quantize for gf20::GF20 {
    fn quantize_f32(value: f32) -> Self { gf20::GF20::from_f32(value) }
    fn dequantize_f32(&self) -> f32 { self.to_f32() }
}

impl Quantize for gf24::GF24 {
    fn quantize_f32(value: f32) -> Self { gf24::GF24::from_f32(value) }
    fn dequantize_f32(&self) -> f32 { self.to_f32() }
}

impl Quantize for gf32::GF32 {
    fn quantize_f32(value: f32) -> Self { gf32::GF32::from_f32(value) }
    fn dequantize_f32(&self) -> f32 { self.to_f32() }
}

impl Quantize for gf64::GF64 {
    fn quantize_f32(value: f32) -> Self { gf64::GF64::from_f64(value as f64) }
    fn dequantize_f32(&self) -> f32 { self.to_f64() as f32 }
}

impl Quantize for FP16 {
    fn quantize_f32(value: f32) -> Self { FP16::from_f32(value) }
    fn dequantize_f32(&self) -> f32 { self.to_f32() }
}

impl Quantize for BF16 {
    fn quantize_f32(value: f32) -> Self { BF16::from_f32(value) }
    fn dequantize_f32(&self) -> f32 { self.to_f32() }
}

impl Quantize for FP32 {
    fn quantize_f32(value: f32) -> Self { FP32::from_f32(value) }
    fn dequantize_f32(&self) -> f32 { self.to_f32() }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gf6a_round_trip() {
        let values = [0.5, 1.0, 1.618, 2.0, 4.0];
        for v in values {
            let gf = GF6a::from_f32(v);
            let decoded = gf.to_f32();
            assert!(!decoded.is_nan(), "GF6a NaN for {}", v);
            assert!(!decoded.is_infinite(), "GF6a Inf for {}", v);
        }
    }

    #[test]
    fn test_gf6a_phi() {
        let gf = GF6a::from_f32(PHI as f32);
        let decoded = gf.to_f32();
        assert!(!decoded.is_nan());
        assert!(decoded > 0.0);
    }

    #[test]
    fn test_gf4a_round_trip() {
        // GF4a range is [0.5, 1.5] with 1 exponent bit and 2 mantissa bits
        let values = [0.5, 0.75, 1.0, 1.25, 1.5];
        for v in values {
            let gf = GF4a::from_f32(v);
            let decoded = gf.to_f32();
            assert!(!decoded.is_nan(), "GF4a NaN for {}", v);
            assert!(!decoded.is_infinite(), "GF4a Inf for {}", v);
        }
    }

    #[test]
    fn test_gf4a_phi() {
        let gf = GF4a::from_f32(PHI as f32);
        let decoded = gf.to_f32();
        assert!(!decoded.is_nan());
        assert!(decoded > 0.0);
    }

    #[test]
    fn test_fp16_round_trip() {
        let values = [0.1, 1.0, 1.618, 100.0];
        for v in values {
            let fp = FP16::from_f32(v);
            let decoded = fp.to_f32();
            assert!(!decoded.is_nan());
            let rel_err = (decoded - v).abs() / v.abs().max(1e-10);
            assert!(rel_err < 0.01, "FP16 error too high for {}: {}", v, rel_err);
        }
    }

    #[test]
    fn test_bf16_round_trip() {
        let values = [0.1, 1.0, 1.618, 100.0];
        for v in values {
            let bf = BF16::from_f32(v);
            let decoded = bf.to_f32();
            assert!(!decoded.is_nan());
            let rel_err = (decoded - v).abs() / v.abs().max(1e-10);
            assert!(rel_err < 0.01, "BF16 error too high for {}: {}", v, rel_err);
        }
    }

    #[test]
    fn test_fp32_round_trip() {
        let values = [0.1, 1.0, 1.618, 100.0];
        for v in values {
            let fp = FP32::from_f32(v);
            let decoded = fp.to_f32();
            assert_eq!(decoded, v);
        }
    }

    #[test]
    fn test_precision_format_all() {
        let all = PrecisionFormat::all_formats();
        assert_eq!(all.len(), 13); // 10 GF + 3 IEEE
    }

    #[test]
    fn test_precision_format_golden_float() {
        let gf = PrecisionFormat::golden_float_formats();
        assert_eq!(gf.len(), 10); // gf4, gf4a, gf6a, gf8, gf12, gf16, gf20, gf24, gf32, gf64
    }

    #[test]
    fn test_precision_format_ieee() {
        let ieee = PrecisionFormat::ieee_formats();
        assert_eq!(ieee.len(), 3); // fp16, bf16, fp32
    }

    #[test]
    fn test_precision_format_parse() {
        assert_eq!(
            PrecisionFormat::from_str("gf16"),
            Some(PrecisionFormat::GF16)
        );
        assert_eq!(
            PrecisionFormat::from_str("GF16"),
            Some(PrecisionFormat::GF16)
        );
        assert_eq!(
            PrecisionFormat::from_str("fp16"),
            Some(PrecisionFormat::FP16)
        );
        assert_eq!(PrecisionFormat::from_str("invalid"), None);
    }

    #[test]
    fn test_bit_widths() {
        assert_eq!(PrecisionFormat::GF4.bit_width(), 4);
        assert_eq!(PrecisionFormat::GF4a.bit_width(), 4);
        assert_eq!(PrecisionFormat::GF6a.bit_width(), 6);
        assert_eq!(PrecisionFormat::GF8.bit_width(), 8);
        assert_eq!(PrecisionFormat::GF12.bit_width(), 12);
        assert_eq!(PrecisionFormat::GF16.bit_width(), 16);
        assert_eq!(PrecisionFormat::FP16.bit_width(), 16);
        assert_eq!(PrecisionFormat::BF16.bit_width(), 16);
        assert_eq!(PrecisionFormat::GF20.bit_width(), 20);
        assert_eq!(PrecisionFormat::GF24.bit_width(), 24);
        assert_eq!(PrecisionFormat::GF32.bit_width(), 32);
        assert_eq!(PrecisionFormat::FP32.bit_width(), 32);
        assert_eq!(PrecisionFormat::GF64.bit_width(), 64);
    }

    #[test]
    fn test_special_values() {
        let inf = FP16::from_f32(f32::INFINITY);
        assert!(inf.to_f32().is_infinite());

        let nan = FP16::from_f32(f32::NAN);
        assert!(nan.to_f32().is_nan());

        let neg_inf = BF16::from_f32(f32::NEG_INFINITY);
        assert!(neg_inf.to_f32().is_infinite());
        assert!(neg_inf.to_f32() < 0.0);
    }

    #[test]
    fn test_phi_family_preservation() {
        // Test that all formats preserve PHI without NaN
        let phi_val = PHI as f32;

        let gf4 = GF4a::from_f32(phi_val);
        assert!(!gf4.to_f32().is_nan());

        let gf6 = GF6a::from_f32(phi_val);
        assert!(!gf6.to_f32().is_nan());

        let fp = FP16::from_f32(phi_val);
        assert!(!fp.to_f32().is_nan());

        let bf = BF16::from_f32(phi_val);
        assert!(!bf.to_f32().is_nan());
    }
}
