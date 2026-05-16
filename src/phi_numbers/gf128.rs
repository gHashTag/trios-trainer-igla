//! GF128 — Golden Float 128-bit format
//!
//! 1 sign + 28 exp + 99 mantissa bits (ratio 28/99 ≈ 0.283, extended dynamic range).
//! Stored as two u64 halves (high = sign + exp + top mantissa, low = bottom mantissa).
//! Aligned with PhD Glava 06 / 09 (extended-precision φ-anchor for long chains).

use super::phi_constants::*;

/// GF128: 1 sign + 28 exp + 99 mantissa bits. Stored as (hi, lo) u64 pair.
/// Layout:
///   hi[63]       = sign
///   hi[62..35]   = 28 exp bits
///   hi[34..0]    = top 35 mantissa bits
///   lo[63..0]    = bottom 64 mantissa bits
pub struct GF128 {
    hi: u64,
    lo: u64,
}

impl GF128 {
    const SIGN_BIT: u64 = 1u64 << 63;
    pub const EXP_BITS: u8 = 28;
    pub const MANT_BITS: u8 = 99;
    pub const EXP_BIAS: i64 = (1i64 << 27) - 1; // 2^(28-1)-1 = 134217727

    /// Encode an f64 value (we don't have f128 natively; f64 is the highest-precision source).
    pub fn from_f64(value: f64) -> Self {
        if value == 0.0 { return Self { hi: 0, lo: 0 }; }
        let sign = if value < 0.0 { 1u64 } else { 0u64 };
        let abs_val = value.abs();

        if !abs_val.is_finite() {
            let nan_bit = if abs_val.is_nan() { 1u64 } else { 0u64 };
            let exp_max: u64 = (1u64 << Self::EXP_BITS) - 1;
            return Self {
                hi: (sign << 63) | (exp_max << 35),
                lo: nan_bit,
            };
        }

        let f64_bits = abs_val.to_bits();
        let f64_exp = ((f64_bits >> 52) & 0x7FF) as i64 - 1023;
        let f64_mant = f64_bits & 0x000F_FFFF_FFFF_FFFF; // 52 bits

        let mut g_exp = f64_exp + Self::EXP_BIAS;
        if g_exp < 0 { return Self { hi: sign << 63, lo: 0 }; }
        let exp_max: i64 = (1i64 << Self::EXP_BITS) - 1;
        if g_exp >= exp_max {
            return Self {
                hi: (sign << 63) | (((exp_max - 1) as u64) << 35) | ((1u64 << 35) - 1),
                lo: u64::MAX,
            };
        }

        // 99 mantissa bits, source = 52 → left-shift by 47.
        // Top 35 bits go in hi, bottom 64 in lo.
        let mant128_lo: u64 = f64_mant << 47;     // bits 47..98 occupied → bottom 64 of mant
        let mant128_hi: u64 = f64_mant >> (64 - 47); // bits 99-64=35 top bits → top 35 of mant
        // mantissa-high mask = 35 bits
        let mant_hi_mask: u64 = (1u64 << 35) - 1;
        let mant_hi = mant128_hi & mant_hi_mask;

        let hi = (sign << 63) | ((g_exp as u64) << 35) | mant_hi;
        Self { hi, lo: mant128_lo }
    }

    pub fn to_f64(self) -> f64 {
        if self.hi == 0 && self.lo == 0 { return 0.0; }
        let sign = if (self.hi & Self::SIGN_BIT) != 0 { -1.0f64 } else { 1.0f64 };
        let exp_mask: u64 = ((1u64 << Self::EXP_BITS) - 1) << 35;
        let exp = ((self.hi & exp_mask) >> 35) as i64;
        let exp_max: i64 = (1i64 << Self::EXP_BITS) - 1;
        if exp == exp_max {
            // Infinity / NaN
            let mant_hi = self.hi & ((1u64 << 35) - 1);
            return if mant_hi == 0 && self.lo == 0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            };
        }

        let exp_val = 2.0_f64.powi((exp - Self::EXP_BIAS) as i32);
        // Reconstruct mantissa fraction from top 35 bits only (matches f64 source precision).
        let mant_hi = self.hi & ((1u64 << 35) - 1);
        // Combine top 35 bits + top 17 bits from lo to reach 52-bit f64 precision.
        let mant52 = (mant_hi << 17) | (self.lo >> (64 - 17));
        let mant_val = 1.0 + (mant52 as f64) / ((1u64 << 52) as f64);
        sign * exp_val * mant_val
    }

    pub fn hi(self) -> u64 { self.hi }
    pub fn lo(self) -> u64 { self.lo }
    pub fn from_parts(hi: u64, lo: u64) -> Self { Self { hi, lo } }

    pub fn quant_error_f64(self, original: f64) -> f64 { (self.to_f64() - original).abs() }
    pub fn relative_error_f64(self, original: f64) -> f64 {
        if original.abs() < f64::MIN_POSITIVE { return self.quant_error_f64(original); }
        self.quant_error_f64(original) / original.abs()
    }
}

impl Clone for GF128 { fn clone(&self) -> Self { *self } }
impl Copy for GF128 {}
impl std::fmt::Debug for GF128 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GF128(hi={:016x} lo={:016x} -> {})", self.hi, self.lo, self.to_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_zero() { assert_eq!(GF128::from_f64(0.0).to_f64(), 0.0); }

    #[test]
    fn test_one() {
        let g = GF128::from_f64(1.0);
        assert!((g.to_f64() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_phi_full_precision() {
        let g = GF128::from_f64(PHI);
        // GF128 carries f64 precision losslessly through the encode → ≤ f64 epsilon
        assert!(g.relative_error_f64(PHI) < 1e-15);
    }

    #[test]
    fn test_trinity_identity_exact() {
        let s = GF128::from_f64(PHI_SQUARED).to_f64()
              + GF128::from_f64(PHI_INVERSE_SQUARED).to_f64();
        assert!((s - 3.0).abs() < 1e-14, "trinity={}", s);
    }

    #[test]
    fn test_extended_range() {
        // GF128 exponent range >> f64 → 1e200 and 1e-200 stay finite
        let g_big = GF128::from_f64(1e200);
        assert!(g_big.to_f64().is_finite());
        let g_small = GF128::from_f64(1e-200);
        assert!(g_small.to_f64() > 0.0);
    }

    #[test]
    fn test_sign() {
        let p = GF128::from_f64(2.0);
        let n = GF128::from_f64(-2.0);
        assert_eq!(p.hi() & GF128::SIGN_BIT, 0);
        assert_eq!(n.hi() & GF128::SIGN_BIT, GF128::SIGN_BIT);
    }
}
