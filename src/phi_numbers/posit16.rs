//! Posit16 (es=1) — Gustafson 2017 posit number system, 16-bit total
//!
//! Standards reference: J. Gustafson, "Beating Floating Point at its Own
//! Game: Posit Arithmetic" (2017), Posit Standard 2022.
//!
//! Bit layout (after un-twos-complementing for negative posits):
//!   [sign:1] [regime:variable] [exponent:es=1] [mantissa:variable]
//!
//! - useed = 2^(2^es) = 2^2 = 4 (for es=1)
//! - Regime field: run-length encoding. If the first body bit is 1, count
//!   leading 1s; regime_value = count - 1. If 0, count leading 0s;
//!   regime_value = -count. A terminator bit of opposite value ends the
//!   run (terminator is absent when the regime fills the entire body).
//! - Total exponent (scale) = regime_value * 2^es + exponent_bits
//!   = 2 * regime_value + exp_bit
//! - Value = sign * 2^scale * (1 + frac), where frac is the mantissa
//!   fraction in [0, 1).
//!
//! Special values:
//!   ZERO  = 0x0000  (only one zero — no negative zero in posit)
//!   NAR   = 0x8000  ("Not a Real" — only one exceptional value)
//!
//! Representable magnitude range for Posit16 with es=1:
//!   MIN_POS = 4^-14 = 2^-28 ≈ 3.73e-9
//!   MAX_POS = 4^+14 = 2^+28 ≈ 2.68e8

use std::fmt;

/// Posit16 with es=1 (per Posit Standard 2022 default for n=16).
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Posit16(pub u16);

impl Posit16 {
    const ES: u32 = 1;
    const NBITS: u32 = 16;

    /// Zero. The only posit with bit pattern of all zeros.
    pub const ZERO: Posit16 = Posit16(0x0000);
    /// Not-a-Real. The only exceptional posit (combines f32's NaN + Inf roles).
    pub const NAR: Posit16 = Posit16(0x8000);
    /// Smallest representable positive value: 0x0001 → useed^-14 × 1.0 = 2^-28.
    pub const MIN_POS: Posit16 = Posit16(0x0001);
    /// Largest representable positive value: 0x7FFF → useed^+14 × 1.0 = 2^+28.
    pub const MAX_POS: Posit16 = Posit16(0x7FFF);
    /// Largest-magnitude negative: 0x8001 → -MAX_POS.
    pub const MAX_NEG: Posit16 = Posit16(0x8001);
    /// Smallest-magnitude negative: 0xFFFF → -MIN_POS.
    pub const MIN_NEG: Posit16 = Posit16(0xFFFF);

    /// Convert f32 → Posit16 with round-to-nearest-even, saturating on overflow/underflow.
    #[must_use]
    pub fn from_f32(val: f32) -> Self {
        if val == 0.0 {
            return Self::ZERO;
        }
        if !val.is_finite() {
            return Self::NAR;
        }

        let neg = val.is_sign_negative();
        let abs = val.abs();

        // Extract f32 fields: scale = unbiased exponent, mant23 = 23-bit fraction.
        let f32_bits = abs.to_bits();
        let f32_exp: i32 = (((f32_bits >> 23) & 0xFF) as i32) - 127;
        let mant23: u32 = f32_bits & 0x007F_FFFF;

        // Subnormal f32 (f32_exp == -127) → too small for Posit16 anyway; saturate.
        if f32_exp <= -29 {
            return if neg { Self::MIN_NEG } else { Self::MIN_POS };
        }
        // Magnitude exceeds Posit16 max (scale > 28) → saturate to MAX.
        if f32_exp >= 29 {
            return if neg { Self::MAX_NEG } else { Self::MAX_POS };
        }

        let scale: i32 = f32_exp;
        // Decompose scale = 2*regime + exp_bit using arithmetic shift (signed).
        let regime_val: i32 = scale >> 1;
        let exp_bit: u32 = (scale & 1) as u32;

        // Build body MSB-first into a u64 with body MSB at bit 47.
        // This leaves bits 32..0 as rounding/sticky headroom.
        let mut buf: u64 = 0;
        let mut top: i32 = 47;

        // Regime field.
        if regime_val >= 0 {
            // (regime_val + 1) one-bits followed by terminator 0.
            let n_ones = (regime_val + 1) as u32;
            // If n_ones >= 15, the regime alone fills (or overflows) the body.
            // For n_ones == 15 exactly: 15 ones fit in body, no terminator. That's MAX_POS.
            // For n_ones > 15: saturate.
            if n_ones >= 15 {
                return if neg { Self::MAX_NEG } else { Self::MAX_POS };
            }
            let shift = top + 1 - n_ones as i32;
            buf |= ((1u64 << n_ones) - 1) << shift;
            top -= n_ones as i32;
            // Terminator 0 at `top` (already zero); just advance.
            top -= 1;
        } else {
            let n_zeros = (-regime_val) as i32;
            // n_zeros == 14 + terminator 1 = MIN_POS body. n_zeros > 14 → saturate.
            if n_zeros >= 15 {
                return if neg { Self::MIN_NEG } else { Self::MIN_POS };
            }
            top -= n_zeros;
            // Terminator 1.
            buf |= 1u64 << top;
            top -= 1;
        }

        // Exponent bit (es=1).
        if top >= 33 {
            // Still inside body LSB range (body LSB = bit 33).
            if exp_bit != 0 {
                buf |= 1u64 << top;
            }
        } else if exp_bit != 0 && top >= 0 {
            // exp falls into rounding region — bit value still influences rounding.
            buf |= 1u64 << top;
        }
        top -= 1;

        // Mantissa bits: place mant23 MSB-first starting at `top`.
        // mant23 has MSB at bit 22 (within the 23-bit field).
        let mant_shift: i32 = top - 22;
        if mant_shift >= 0 {
            buf |= (mant23 as u64) << mant_shift;
        } else {
            let drop = (-mant_shift) as u32;
            if drop < 32 {
                buf |= (mant23 as u64) >> drop;
                let sticky_mask: u32 = (1u32 << drop) - 1;
                if (mant23 & sticky_mask) != 0 {
                    buf |= 1; // sticky parked in bit 0
                }
            } else if mant23 != 0 {
                buf |= 1;
            }
        }

        // Round-half-to-even at body LSB (bit 33).
        // Round bit = bit 32; sticky = OR of bits 31..0.
        let lsb_pos: i32 = 33;
        let round_pos: i32 = 32;
        let lsb = ((buf >> lsb_pos) & 1) as u32;
        let round = ((buf >> round_pos) & 1) as u32;
        let sticky_mask: u64 = (1u64 << round_pos) - 1;
        let sticky = (buf & sticky_mask) != 0;

        let mut body15: u32 = ((buf >> lsb_pos) & 0x7FFF) as u32;
        if round == 1 && (sticky || lsb == 1) {
            body15 += 1;
            // Overflow into sign bit: this means we exceeded MAX_POS magnitude.
            if body15 > 0x7FFF {
                return if neg { Self::MAX_NEG } else { Self::MAX_POS };
            }
        }

        // If rounding underflows body to zero, return MIN_POS / MIN_NEG (posit never
        // underflows to ZERO from a non-zero input — that would lose information).
        if body15 == 0 {
            return if neg { Self::MIN_NEG } else { Self::MIN_POS };
        }

        // Apply sign via two's-complement on the full 16-bit pattern.
        let bits = if neg {
            (body15 as u32).wrapping_neg() as u16
        } else {
            body15 as u16
        };
        Posit16(bits)
    }

    /// Convert Posit16 → f32 by exact decode of regime / exponent / mantissa.
    #[must_use]
    pub fn to_f32(self) -> f32 {
        let raw = self.0;
        if raw == 0x0000 {
            return 0.0;
        }
        if raw == 0x8000 {
            return f32::NAN;
        }
        let neg = (raw & 0x8000) != 0;
        // Take 2's-complement magnitude form for negative values.
        let body: u16 = if neg { raw.wrapping_neg() & 0x7FFF } else { raw & 0x7FFF };

        // Decode regime starting from bit 14.
        let first = (body >> 14) & 1;
        let mut pos: i32 = 14;
        let mut count: i32 = 0;
        while pos >= 0 && ((body >> pos) & 1) == first {
            count += 1;
            pos -= 1;
        }
        // pos points to the terminator (or -1 if regime fills body — implicit terminator).
        let regime_val: i32 = if first == 1 { count - 1 } else { -count };
        pos -= 1; // skip terminator

        // Exponent bit (es=1).
        let exp_bit: i32 = if pos >= 0 { ((body >> pos) & 1) as i32 } else { 0 };
        pos -= 1;

        let scale: i32 = 2 * regime_val + exp_bit;

        // Remaining bits = mantissa fraction. pos+1 is the bit-width available.
        let mant_bits: i32 = if pos >= 0 { pos + 1 } else { 0 };
        let frac: f64 = if mant_bits > 0 {
            let mant_raw = (body as u32) & ((1u32 << mant_bits) - 1);
            (mant_raw as f64) / ((1u64 << mant_bits) as f64)
        } else {
            0.0
        };

        let mag = (1.0 + frac) * (2.0_f64).powi(scale);
        let val = if neg { -mag } else { mag };
        val as f32
    }

    /// Raw 16-bit pattern.
    #[must_use]
    pub fn to_bits(self) -> u16 {
        self.0
    }

    /// Construct from raw 16-bit pattern.
    #[must_use]
    pub fn from_bits(bits: u16) -> Self {
        Posit16(bits)
    }

    /// True iff this is the NaR (Not-a-Real) exceptional value.
    #[must_use]
    pub fn is_nar(self) -> bool {
        self.0 == 0x8000
    }

    /// True iff this is +0 (the only zero).
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.0 == 0x0000
    }

    /// Number of exponent bits (compile-time constant; useful for tests).
    #[must_use]
    pub fn es() -> u32 {
        Self::ES
    }

    /// Total bit-width (compile-time constant).
    #[must_use]
    pub fn nbits() -> u32 {
        Self::NBITS
    }
}

impl fmt::Debug for Posit16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nar() {
            write!(f, "Posit16(NaR)")
        } else if self.is_zero() {
            write!(f, "Posit16(0)")
        } else {
            write!(f, "Posit16({})", self.to_f32())
        }
    }
}

impl fmt::Display for Posit16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nar() {
            write!(f, "NaR")
        } else {
            write!(f, "{}", self.to_f32())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_round_trip() {
        assert_eq!(Posit16::from_f32(0.0).to_bits(), 0x0000);
        assert_eq!(Posit16::from_f32(-0.0).to_bits(), 0x0000);
        assert_eq!(Posit16::ZERO.to_f32(), 0.0);
    }

    #[test]
    fn nar_round_trip() {
        assert_eq!(Posit16::from_f32(f32::NAN).to_bits(), 0x8000);
        assert_eq!(Posit16::from_f32(f32::INFINITY).to_bits(), 0x8000);
        assert_eq!(Posit16::from_f32(f32::NEG_INFINITY).to_bits(), 0x8000);
        assert!(Posit16::NAR.to_f32().is_nan());
    }

    #[test]
    fn one_encodes_as_0x4000() {
        // 1.0 → scale=0, regime=0, exp=0, mant=0.
        // Body bits: regime "10" + exp "0" + mant "000000000000" = "100000000000000"
        // = 0x4000.
        assert_eq!(Posit16::from_f32(1.0).to_bits(), 0x4000);
        assert_eq!(Posit16(0x4000).to_f32(), 1.0);
    }

    #[test]
    fn two_encodes_as_0x5000() {
        // 2.0 → scale=1, regime=0, exp=1.
        // Body: "10" + "1" + zeros = "101000000000000" = 0x5000.
        assert_eq!(Posit16::from_f32(2.0).to_bits(), 0x5000);
        assert_eq!(Posit16(0x5000).to_f32(), 2.0);
    }

    #[test]
    fn four_encodes_as_0x6000() {
        // 4.0 → scale=2, regime=1, exp=0.
        // Body: "110" + "0" + zeros = "110000000000000" = 0x6000.
        assert_eq!(Posit16::from_f32(4.0).to_bits(), 0x6000);
        assert_eq!(Posit16(0x6000).to_f32(), 4.0);
    }

    #[test]
    fn half_encodes_as_0x3000() {
        // 0.5 → scale=-1, regime=-1, exp=1.
        // Body: "01" + "1" + zeros = "011000000000000" = 0x3000.
        assert_eq!(Posit16::from_f32(0.5).to_bits(), 0x3000);
        assert_eq!(Posit16(0x3000).to_f32(), 0.5);
    }

    #[test]
    fn quarter_encodes_as_0x2000() {
        // 0.25 → scale=-2, regime=-1, exp=0.
        assert_eq!(Posit16::from_f32(0.25).to_bits(), 0x2000);
        assert_eq!(Posit16(0x2000).to_f32(), 0.25);
    }

    #[test]
    fn negation_is_twos_complement() {
        let plus_one = Posit16::from_f32(1.0);
        let minus_one = Posit16::from_f32(-1.0);
        // Two's-complement of 0x4000 = 0xC000.
        assert_eq!(plus_one.to_bits(), 0x4000);
        assert_eq!(minus_one.to_bits(), 0xC000);
        assert_eq!(minus_one.to_f32(), -1.0);
    }

    #[test]
    fn max_pos_decodes_to_4_pow_14() {
        // 0x7FFF = 15 ones in body → regime=14, scale=28, value = 2^28.
        let mp = Posit16::MAX_POS;
        let v = mp.to_f32();
        assert!((v - (2.0_f32).powi(28)).abs() / v < 1e-6);
    }

    #[test]
    fn min_pos_decodes_to_2_pow_neg_28() {
        // 0x0001 = 14 zeros + 1 terminator → regime=-14, scale=-28, value=2^-28.
        let mp = Posit16::MIN_POS;
        let v = mp.to_f32();
        let expected = (2.0_f32).powi(-28);
        assert!((v - expected).abs() / v < 1e-3);
    }

    #[test]
    fn saturates_on_overflow() {
        assert_eq!(Posit16::from_f32(1e30).to_bits(), Posit16::MAX_POS.to_bits());
        assert_eq!(Posit16::from_f32(-1e30).to_bits(), Posit16::MAX_NEG.to_bits());
    }

    #[test]
    fn saturates_on_underflow_to_min_pos_not_zero() {
        // Posit explicitly does NOT underflow to zero — it saturates at MIN_POS.
        let tiny = Posit16::from_f32(1e-30);
        assert_eq!(tiny.to_bits(), Posit16::MIN_POS.to_bits());
        let tiny_neg = Posit16::from_f32(-1e-30);
        assert_eq!(tiny_neg.to_bits(), Posit16::MIN_NEG.to_bits());
    }

    #[test]
    fn round_trip_typical_range() {
        // Posit16 (es=1) has highest precision near magnitude 1; relative error
        // grows toward both extremes. The "golden tape" of well-conditioned values:
        let cases: [f32; 14] = [
            1.0, 2.0, 4.0, 0.5, 0.25, 3.0, 1.5, -1.0, -2.0, 100.0, 0.01,
            std::f32::consts::PI, std::f32::consts::E, 1.618,
        ];
        for &x in &cases {
            let p = Posit16::from_f32(x);
            let back = p.to_f32();
            // Posit16 is ~10-12 bits of precision near 1.0; allow 1% relative err.
            let rel = (back - x).abs() / x.abs().max(1e-12);
            assert!(rel < 1e-2, "round-trip lost too much: {x} → {back} ({p:?})");
        }
    }

    #[test]
    fn negation_symmetric_for_typical_values() {
        for x in [1.0_f32, 0.5, 3.0, 100.0, 0.01, std::f32::consts::PI] {
            let pos = Posit16::from_f32(x).to_f32();
            let neg = Posit16::from_f32(-x).to_f32();
            assert!((pos + neg).abs() / pos.abs() < 1e-6,
                    "asymmetric negation at {x}");
        }
    }

    #[test]
    fn monotone_in_a_dense_range() {
        // For increasing inputs in a tight range, encoded value should be
        // monotonically non-decreasing (Posit is a totally-ordered numeric type).
        let mut prev = Posit16::from_f32(0.5).to_f32();
        for i in 1..=200 {
            let x = 0.5 + (i as f32) * 0.01;
            let now = Posit16::from_f32(x).to_f32();
            assert!(now >= prev - 1e-6,
                    "non-monotone at i={i}, x={x}: prev={prev}, now={now}");
            prev = now;
        }
    }

    #[test]
    fn exact_known_useed_powers() {
        // useed^k for small k must encode exactly with mantissa = 0.
        for k in -10..=10_i32 {
            let v = (4.0_f64).powi(k) as f32;
            let p = Posit16::from_f32(v);
            let back = p.to_f32();
            let rel = ((back - v) / v).abs();
            assert!(rel < 1e-5, "useed^{k} round-trip drift: {v} → {back}");
        }
    }
}
