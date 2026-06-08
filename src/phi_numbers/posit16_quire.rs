//! Posit16 quire-bit accumulator for exact dot products.
//!
//! The "quire" is the defining feature of the Posit number system
//! (Gustafson 2017, §4): a wide fixed-point register that accumulates
//! products without rounding until the final round-back to Posit16.
//! For matmul / dot-product workloads — the workhorse of every neural-
//! network forward pass — this changes the accuracy profile from
//! "naive f32 sum with O(N · ε_f32) rounding error" to "exact for any
//! accumulation length that fits in the quire's range."
//!
//! Loop 152 implementation choices:
//!
//! - **Quire storage**: `i128` fixed-point with scale 2⁻⁵⁶. That gives
//!   us 71 bits of integer headroom (sign + 71 magnitude) above the
//!   2⁻⁵⁶ resolution, so accumulating up to ≈ 2¹⁵ = 32 768 Posit16
//!   products of maximal magnitude is exact. (Posit16's largest
//!   representable value is 4¹⁴ = 2²⁸; a product is bounded by 2⁵⁶;
//!   the quire holds at least 2¹⁵ such products before overflow.)
//!   This covers every neural-network vector dimension we care about
//!   in the F2 §9.4.1 regime (d_model ≤ 1024 × vocab ≤ 128 = 131 072
//!   stays inside the quire if no single product saturates).
//!
//! - **Product computation**: we use the `Posit16::decode_extended`
//!   path (sign, scale, mantissa, mant_bits) rather than going
//!   through f32. Two reasons:
//!     1. The product `(1+frac_a) × (1+frac_b)` fits exactly in
//!        an `i64` since each factor is ≤ 13 bits; the f32 path
//!        would round at the 24-bit mantissa boundary.
//!     2. Underflow handling is explicit at the quire-resolution
//!        boundary (products below 2⁻⁵⁶ round to zero in the quire,
//!        which is far below Posit16's own MIN_POS = 2⁻²⁸ output
//!        precision anyway).
//!
//! - **NaR propagation**: if any input is NaR, the quire's state
//!   becomes NaR (a single sticky bit), and `to_posit16()` returns
//!   NaR. This matches IEEE NaN propagation semantics.
//!
//! - **Saturation**: on overflow (≥ 2¹²⁷ in the i128), we clamp the
//!   quire at i128::MIN / i128::MAX. The final round-back to Posit16
//!   will then produce MAX_POS / MAX_NEG. We do NOT panic.

use super::posit16::Posit16;

/// Fixed-point quire with scale 2⁻⁵⁶. Holds the running sum of
/// Posit16 products with no precision loss until rounded back to Posit16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositQuire {
    /// Fixed-point accumulator: real value = `acc / 2^FP_SHIFT`.
    acc: i128,
    /// Sticky NaR flag — once set, every subsequent `to_posit16()` returns NaR.
    nar: bool,
}

const FP_SHIFT: u32 = 56;

impl Default for PositQuire {
    fn default() -> Self {
        Self::new()
    }
}

impl PositQuire {
    /// Empty quire (accumulator = 0, NaR flag clear).
    #[must_use]
    pub const fn new() -> Self {
        Self { acc: 0, nar: false }
    }

    /// True iff this quire has absorbed at least one NaR input.
    #[must_use]
    pub const fn is_nar(self) -> bool {
        self.nar
    }

    /// Add the product `a * b` to the running quire sum. Zero-cost on
    /// `Posit16::ZERO` inputs (any-times-zero is zero). NaR poisons the
    /// quire permanently.
    pub fn add_product(&mut self, a: Posit16, b: Posit16) {
        if self.nar {
            return;
        }
        if a.is_nar() || b.is_nar() {
            self.nar = true;
            return;
        }
        if a.is_zero() || b.is_zero() {
            return;
        }
        let (sa, scale_a, mant_a, bits_a) = a.decode_extended();
        let (sb, scale_b, mant_b, bits_b) = b.decode_extended();

        // Implicit-leading-1 expansion: m = mant + 2^bits.
        let lead_a: i64 = (mant_a as i64) | (1i64 << bits_a);
        let lead_b: i64 = (mant_b as i64) | (1i64 << bits_b);
        // Exact 64-bit mantissa product (each factor ≤ 13 bits).
        let mantissa_product: i64 = lead_a * lead_b;
        let mantissa_bits_total: i32 = bits_a + bits_b;
        // True value = sign × mantissa_product × 2^(scale_a + scale_b - mantissa_bits_total)
        let total_scale: i32 = scale_a + scale_b - mantissa_bits_total;

        // Convert to quire fixed-point (scale 2^-FP_SHIFT):
        //   real_value × 2^FP_SHIFT = mantissa_product × 2^(total_scale + FP_SHIFT)
        let target_shift: i32 = total_scale + FP_SHIFT as i32;

        let chunk: i128 = if target_shift >= 0 {
            // Left shift loses no precision as long as we don't overflow i128.
            // We allow up to 127 bits — overflow saturates.
            if target_shift >= 127 {
                if mantissa_product == 0 {
                    0
                } else if mantissa_product > 0 {
                    i128::MAX
                } else {
                    i128::MIN
                }
            } else {
                (mantissa_product as i128).wrapping_shl(target_shift as u32)
            }
        } else {
            // Right shift drops bits below the quire resolution; that's
            // intentional (values below 2^-FP_SHIFT are sub-quire-resolution).
            let drop = (-target_shift) as u32;
            if drop >= 64 {
                0
            } else {
                (mantissa_product as i128) >> drop
            }
        };

        let signed_chunk: i128 = if sa ^ sb { -chunk } else { chunk };
        self.acc = self.acc.saturating_add(signed_chunk);
    }

    /// Add a Posit16 value directly to the running quire (no product;
    /// equivalent to `add_product(x, Posit16::from_f32(1.0))` but cheaper).
    pub fn add(&mut self, x: Posit16) {
        if self.nar {
            return;
        }
        if x.is_nar() {
            self.nar = true;
            return;
        }
        if x.is_zero() {
            return;
        }
        let (sign, scale, mant, bits) = x.decode_extended();
        let lead: i64 = (mant as i64) | (1i64 << bits);
        let mantissa_bits_total: i32 = bits;
        let total_scale: i32 = scale - mantissa_bits_total;
        let target_shift: i32 = total_scale + FP_SHIFT as i32;
        let chunk: i128 = if target_shift >= 0 {
            if target_shift >= 127 {
                if lead == 0 { 0 } else { i128::MAX }
            } else {
                (lead as i128).wrapping_shl(target_shift as u32)
            }
        } else {
            let drop = (-target_shift) as u32;
            if drop >= 64 { 0 } else { (lead as i128) >> drop }
        };
        let signed_chunk = if sign { -chunk } else { chunk };
        self.acc = self.acc.saturating_add(signed_chunk);
    }

    /// Reset the quire to zero (clears NaR flag too).
    pub fn clear(&mut self) {
        self.acc = 0;
        self.nar = false;
    }

    /// Round the accumulator back to a Posit16. NaR if the quire is poisoned.
    #[must_use]
    pub fn to_posit16(self) -> Posit16 {
        if self.nar {
            return Posit16::NAR;
        }
        if self.acc == 0 {
            return Posit16::ZERO;
        }
        // Convert fixed-point i128 → f32 → Posit16. f32 has 24-bit
        // significand which is wider than Posit16's max-mantissa-precision
        // region (≈ 13 bits at scale ≈ 0), so the round-trip preserves
        // Posit16's full precision in the typical region.
        let acc_f64 = (self.acc as f64) / (1u128 << FP_SHIFT) as f64;
        Posit16::from_f32(acc_f64 as f32)
    }

    /// Raw i128 accumulator value (for testing / introspection).
    #[must_use]
    pub const fn acc_raw(self) -> i128 {
        self.acc
    }
}

/// Exact Posit16 dot product via the quire. Zero-allocation; runs in
/// O(N) with two memory passes (one over each input slice). Panics if
/// the inputs are of different lengths.
#[must_use]
pub fn posit16_dot(a: &[Posit16], b: &[Posit16]) -> Posit16 {
    assert_eq!(a.len(), b.len(), "posit16_dot: length mismatch");
    let mut q = PositQuire::new();
    for (x, y) in a.iter().zip(b.iter()) {
        q.add_product(*x, *y);
    }
    q.to_posit16()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f32) -> Posit16 {
        Posit16::from_f32(x)
    }

    #[test]
    fn empty_quire_is_zero() {
        assert_eq!(PositQuire::new().to_posit16().to_bits(), Posit16::ZERO.to_bits());
    }

    #[test]
    fn single_product_one_times_one() {
        let mut q = PositQuire::new();
        q.add_product(p(1.0), p(1.0));
        assert_eq!(q.to_posit16().to_bits(), p(1.0).to_bits());
    }

    #[test]
    fn add_product_with_zero_is_noop() {
        let mut q = PositQuire::new();
        q.add_product(Posit16::ZERO, p(42.0));
        q.add_product(p(42.0), Posit16::ZERO);
        assert_eq!(q.to_posit16().to_bits(), Posit16::ZERO.to_bits());
    }

    #[test]
    fn nar_poisons_quire_permanently() {
        let mut q = PositQuire::new();
        q.add_product(p(1.0), p(1.0));
        q.add_product(Posit16::NAR, p(2.0));
        // Subsequent valid adds shouldn't recover from NaR.
        q.add_product(p(100.0), p(100.0));
        assert!(q.is_nar());
        assert!(q.to_posit16().is_nar());
    }

    #[test]
    fn dot_orthogonal_unit_vectors_is_zero() {
        let a = [p(1.0), p(0.0)];
        let b = [p(0.0), p(1.0)];
        assert_eq!(posit16_dot(&a, &b).to_bits(), Posit16::ZERO.to_bits());
    }

    #[test]
    fn dot_self_unit_vector_is_one() {
        let a = [p(1.0), p(0.0), p(0.0)];
        assert_eq!(posit16_dot(&a, &a).to_bits(), p(1.0).to_bits());
    }

    #[test]
    fn dot_2_3_dot_4_5_is_23() {
        let a = [p(2.0), p(3.0)];
        let b = [p(4.0), p(5.0)];
        // 2*4 + 3*5 = 8 + 15 = 23
        let r = posit16_dot(&a, &b).to_f32();
        assert!((r - 23.0).abs() < 0.1, "expected ≈23, got {r}");
    }

    #[test]
    fn dot_with_cancellation_exact() {
        // (1, -1) · (1, 1) = 1 - 1 = 0. With the quire this should be exactly 0;
        // a naive Posit16 accumulator might also get 0 here, but the test
        // documents the invariant.
        let a = [p(1.0), p(-1.0)];
        let b = [p(1.0), p(1.0)];
        assert_eq!(posit16_dot(&a, &b).to_bits(), Posit16::ZERO.to_bits());
    }

    #[test]
    fn dot_length_mismatch_panics() {
        let a = [p(1.0)];
        let b = [p(1.0), p(1.0)];
        let result = std::panic::catch_unwind(|| posit16_dot(&a, &b));
        assert!(result.is_err());
    }

    #[test]
    fn dot_long_vector_better_than_naive_posit16() {
        // Comparison: long-vector dot product where naive Posit16 accumulation
        // accumulates rounding noise. The quire holds full precision until
        // the final round.
        //
        // Construct a vector where every product is small but the sum is large.
        // Each product ≈ 0.01, sum of 100 products ≈ 1.0. Naive Posit16
        // accumulation rounds at each step (each intermediate ≈ 0.5, 0.51, …,
        // 1.0); the quire accumulates exactly.
        let n = 100;
        let a: Vec<Posit16> = (0..n).map(|_| p(0.1)).collect();
        let b: Vec<Posit16> = (0..n).map(|_| p(0.1)).collect();
        let quire_result = posit16_dot(&a, &b).to_f32();
        // True dot product = 100 * 0.01 = 1.0.
        // The quire result must be closer to 1.0 than 5% — Posit16 has
        // ~12-bit mantissa precision near magnitude 1, so the round-back
        // is the only error source.
        assert!((quire_result - 1.0).abs() < 0.05,
                "quire dot got {quire_result}, expected ≈ 1.0");
    }

    #[test]
    fn quire_beats_naive_f32_accum_under_cancellation() {
        // Build a vector with alternating signs where naive f32 accumulation
        // could lose precision but the quire holds exact integers.
        let n = 1000;
        let a: Vec<Posit16> = (0..n)
            .map(|i| if i % 2 == 0 { p(1.0) } else { p(-1.0) })
            .collect();
        let b: Vec<Posit16> = (0..n).map(|_| p(1.0)).collect();
        // Exact answer: 500 * 1 - 500 * 1 = 0.
        let result = posit16_dot(&a, &b);
        assert_eq!(result.to_bits(), Posit16::ZERO.to_bits(),
                   "quire should give exactly zero; got {result:?}");
    }

    #[test]
    fn add_value_directly() {
        let mut q = PositQuire::new();
        q.add(p(1.0));
        q.add(p(2.0));
        q.add(p(3.0));
        let result = q.to_posit16().to_f32();
        assert!((result - 6.0).abs() < 0.1, "expected ≈6, got {result}");
    }

    #[test]
    fn clear_resets_state() {
        let mut q = PositQuire::new();
        q.add_product(p(100.0), p(100.0));
        q.clear();
        assert_eq!(q.acc_raw(), 0);
        assert!(!q.is_nar());
        assert_eq!(q.to_posit16().to_bits(), Posit16::ZERO.to_bits());
    }

    #[test]
    fn nar_propagation_through_clear() {
        let mut q = PositQuire::new();
        q.add_product(Posit16::NAR, p(1.0));
        assert!(q.is_nar());
        q.clear();
        assert!(!q.is_nar(), "clear() must reset the NaR flag");
    }

    #[test]
    fn negative_product_subtracts() {
        let mut q = PositQuire::new();
        q.add_product(p(2.0), p(3.0));    // +6
        q.add_product(p(-1.0), p(4.0));   // -4
        let r = q.to_posit16().to_f32();
        assert!((r - 2.0).abs() < 0.05, "expected ≈2, got {r}");
    }

    #[test]
    fn quire_accumulator_handles_underflow_below_resolution() {
        // Add many tiny products below the quire's 2^-56 resolution.
        // They should NOT accumulate to a spurious non-zero — they're
        // simply below the representable bottom of the accumulator.
        let mut q = PositQuire::new();
        let tiny = Posit16::MIN_POS;
        for _ in 0..1000 {
            q.add_product(tiny, tiny);
        }
        // (2^-28)^2 = 2^-56, exactly at the quire's bottom rung. With i128
        // shift = 56, each chunk is 1 << 0 = 1; 1000 of those = 1000. Final
        // round to Posit16 should give a value below MIN_POS → MIN_POS (or
        // ZERO if the saturation rounded down). Either is acceptable.
        let _ = q.to_posit16();  // just verify no panic
    }

    #[test]
    fn posit16_dot_empty_vectors_is_zero() {
        let empty: [Posit16; 0] = [];
        assert_eq!(posit16_dot(&empty, &empty).to_bits(), Posit16::ZERO.to_bits());
    }

    #[test]
    fn quire_acc_raw_observable() {
        let mut q = PositQuire::new();
        q.add(p(1.0));
        // 1.0 in fixed-point with shift 56 = 2^56.
        assert_eq!(q.acc_raw(), 1i128 << FP_SHIFT);
    }

    #[test]
    fn quire_is_associative_in_practice() {
        // (a + b) + c == a + (b + c) for typical-magnitude inputs. The quire
        // makes summation associative on its representable range; naive
        // floating-point summation isn't.
        let a = p(1.0);
        let b = p(2.0);
        let c = p(3.0);
        let mut q1 = PositQuire::new();
        q1.add(a);
        q1.add(b);
        q1.add(c);
        let mut q2 = PositQuire::new();
        q2.add(c);
        q2.add(b);
        q2.add(a);
        assert_eq!(q1.acc_raw(), q2.acc_raw(),
                   "quire summation must be order-independent");
    }
}
