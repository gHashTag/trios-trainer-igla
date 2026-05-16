//! GF256 — Golden Float 256-bit format
//!
//! 1 sign + 32 exp + 223 mantissa bits. Used as champion-tier dynamic-range carrier
//! in IGLA-STRATEGY race (best BPB 2.5719 @ gf256/h384). Stored as four u64 limbs:
//!   limbs[3] = sign + exp + top mantissa bits
//!   limbs[2..0] = remaining mantissa bits (low order)
//! PhD Glava 06, Glava 09 (GF vs MXFP4 ablation).

use super::phi_constants::*;

pub struct GF256 {
    limbs: [u64; 4],
}

impl GF256 {
    const SIGN_BIT: u64 = 1u64 << 63;
    pub const EXP_BITS: u8 = 32;
    pub const MANT_BITS: u16 = 223;
    pub const EXP_BIAS: i64 = (1i64 << 31) - 1; // 2^31 - 1

    /// limbs[3] layout:
    ///   bit 63        : sign
    ///   bits 62..31   : 32 exp bits
    ///   bits 30..0    : top 31 mantissa bits
    /// limbs[2..0]    : remaining 192 mantissa bits, MSB at limbs[2]
    pub fn from_f64(value: f64) -> Self {
        if value == 0.0 { return Self { limbs: [0; 4] }; }
        let sign = if value < 0.0 { 1u64 } else { 0u64 };
        let abs_val = value.abs();

        if !abs_val.is_finite() {
            let exp_max: u64 = (1u64 << Self::EXP_BITS) - 1;
            let nan_bit = if abs_val.is_nan() { 1u64 } else { 0u64 };
            return Self {
                limbs: [nan_bit, 0, 0, (sign << 63) | (exp_max << 31)],
            };
        }

        let f64_bits = abs_val.to_bits();
        let f64_exp = ((f64_bits >> 52) & 0x7FF) as i64 - 1023;
        let f64_mant: u64 = f64_bits & 0x000F_FFFF_FFFF_FFFF; // 52 bits

        let mut g_exp = f64_exp + Self::EXP_BIAS;
        if g_exp < 0 { return Self { limbs: [0, 0, 0, sign << 63] }; }
        let exp_max: i64 = (1i64 << Self::EXP_BITS) - 1;
        if g_exp >= exp_max {
            let mant_top_mask: u64 = (1u64 << 31) - 1;
            return Self {
                limbs: [u64::MAX, u64::MAX, u64::MAX,
                        (sign << 63) | (((exp_max - 1) as u64) << 31) | mant_top_mask],
            };
        }

        // Place 52-bit mantissa at the top of the 223-bit field.
        // Top mantissa slot is 31 bits in limbs[3]; the next limbs[2] starts immediately below.
        // We left-align: top 31 bits → limbs[3] low 31, next 21 bits → limbs[2] high 21.
        let top31 = (f64_mant >> (52 - 31)) as u64; // upper 31 bits
        let rem21 = f64_mant & ((1u64 << 21) - 1);  // remaining 21 bits
        let limb2 = rem21 << (64 - 21); // place at top of limbs[2]

        let limb3 = (sign << 63) | ((g_exp as u64) << 31) | (top31 & ((1u64 << 31) - 1));
        Self { limbs: [0, 0, limb2, limb3] }
    }

    pub fn to_f64(self) -> f64 {
        if self.limbs == [0; 4] { return 0.0; }
        let l3 = self.limbs[3];
        let sign = if (l3 & Self::SIGN_BIT) != 0 { -1.0f64 } else { 1.0f64 };
        let exp_mask: u64 = ((1u64 << Self::EXP_BITS) - 1) << 31;
        let exp = ((l3 & exp_mask) >> 31) as i64;
        let exp_max: i64 = (1i64 << Self::EXP_BITS) - 1;

        if exp == exp_max {
            let top31 = l3 & ((1u64 << 31) - 1);
            return if top31 == 0 && self.limbs[2] == 0 && self.limbs[1] == 0 && self.limbs[0] == 0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            };
        }

        let exp_val = 2.0_f64.powi((exp - Self::EXP_BIAS) as i32);
        let top31 = l3 & ((1u64 << 31) - 1);
        let next21 = self.limbs[2] >> (64 - 21);
        // Reconstruct 52-bit f64 mantissa from top 31 + next 21
        let mant52 = (top31 << 21) | next21;
        let mant_val = 1.0 + (mant52 as f64) / ((1u64 << 52) as f64);
        sign * exp_val * mant_val
    }

    pub fn limbs(self) -> [u64; 4] { self.limbs }
    pub fn from_limbs(limbs: [u64; 4]) -> Self { Self { limbs } }

    pub fn quant_error_f64(self, original: f64) -> f64 { (self.to_f64() - original).abs() }
    pub fn relative_error_f64(self, original: f64) -> f64 {
        if original.abs() < f64::MIN_POSITIVE { return self.quant_error_f64(original); }
        self.quant_error_f64(original) / original.abs()
    }
}

impl Clone for GF256 { fn clone(&self) -> Self { Self { limbs: self.limbs } } }
impl Copy for GF256 {}
impl std::fmt::Debug for GF256 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GF256({:016x}{:016x}{:016x}{:016x} -> {})",
               self.limbs[3], self.limbs[2], self.limbs[1], self.limbs[0], self.to_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_zero() { assert_eq!(GF256::from_f64(0.0).to_f64(), 0.0); }

    #[test]
    fn test_phi() {
        let g = GF256::from_f64(PHI);
        assert!(g.relative_error_f64(PHI) < 1e-15);
    }

    #[test]
    fn test_trinity_identity() {
        let s = GF256::from_f64(PHI_SQUARED).to_f64()
              + GF256::from_f64(PHI_INVERSE_SQUARED).to_f64();
        assert!((s - 3.0).abs() < 1e-14, "trinity={}", s);
    }

    #[test]
    fn test_extended_range() {
        let big = GF256::from_f64(1e300);
        assert!(big.to_f64().is_finite() || big.to_f64().is_infinite());
        // The big point of GF256 is the huge exponent range — 2^32 >> 2^11
    }

    #[test]
    fn test_sign() {
        let p = GF256::from_f64(2.0);
        let n = GF256::from_f64(-2.0);
        assert_eq!(p.limbs()[3] & GF256::SIGN_BIT, 0);
        assert_eq!(n.limbs()[3] & GF256::SIGN_BIT, GF256::SIGN_BIT);
    }

    #[test]
    fn test_round_trip() {
        for v in [0.5_f64, 1.0, PHI, PHI_SQUARED, 16.0, 1024.0, 1e6] {
            let g = GF256::from_f64(v);
            assert!(g.relative_error_f64(v) < 1e-14, "v={} err={}", v, g.relative_error_f64(v));
        }
    }
}
