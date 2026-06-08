//! F2 instrumentation: lossy cross-format conversion counter.
//!
//! Tracks every quantization step in phi-ladder (GFTernary→GF8→GF16→GF32)
//! and format-zoo (INT8/FP8/bf16) paths. Metric (2) of the F2 protocol.

use crate::gf16::GF16;
use crate::phi_numbers::{GFTernary, Posit16, GF32, GF8};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LadderKind {
    PhiLadder,
    FormatZoo,
}

#[derive(Debug, Clone, Default)]
pub struct ConversionCounter {
    pub gf32_to_gf16: u64,
    pub gf16_to_gf8: u64,
    pub gf8_to_ternary: u64,
    pub gf16_to_gf32: u64,
    pub gf8_to_gf16: u64,
    pub ternary_to_gf8: u64,
    pub f32_to_int8: u64,
    pub f32_to_int4: u64,
    pub f32_to_fp8: u64,
    pub f32_to_fp8_e4m3: u64,
    pub f32_to_fp8_e5m2: u64,
    pub f32_to_bf16: u64,
    pub f32_to_paretoq: u64,
    pub f32_to_posit16: u64,
    pub lossy_total: u64,
}

// Loop 147 — 68th-pass audit SEV-5: the Display impl below previously omitted
// `f32_to_posit16`, `f32_to_int4`, `f32_to_paretoq`. A counter print at the
// end of a training run silently dropped them, hiding undercount bugs. The
// updated impl emits every tracked field. The field-mention order is the
// same as the struct declaration so a future reader can ctrl-F to verify.

impl ConversionCounter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn total(&self) -> u64 {
        self.lossy_total
    }

    pub fn convert_gf32_to_gf16(&mut self, val: GF32) -> GF16 {
        let f = val.to_f32();
        let q = GF16::from_f32(f);
        self.gf32_to_gf16 += 1;
        if (f - q.to_f32()).abs() > f32::EPSILON {
            self.lossy_total += 1;
        }
        q
    }

    pub fn convert_gf16_to_gf8(&mut self, val: GF16) -> GF8 {
        let f = val.to_f32();
        let q = GF8::from_f32(f);
        self.gf16_to_gf8 += 1;
        if (f - q.to_f32()).abs() > f32::EPSILON {
            self.lossy_total += 1;
        }
        q
    }

    pub fn convert_gf8_to_ternary(&mut self, val: GF8) -> GFTernary {
        let f = val.to_f32();
        let q = GFTernary::from_f32(f);
        self.gf8_to_ternary += 1;
        if (f - q.to_f32()).abs() > f32::EPSILON {
            self.lossy_total += 1;
        }
        q
    }

    pub fn convert_gf16_to_gf32(&mut self, val: GF16) -> GF32 {
        let f = val.to_f32();
        let q = GF32::from_f32(f);
        self.gf16_to_gf32 += 1;
        if (f - q.to_f32()).abs() > f32::EPSILON {
            self.lossy_total += 1;
        }
        q
    }

    pub fn convert_gf8_to_gf16(&mut self, val: GF8) -> GF16 {
        let f = val.to_f32();
        let q = GF16::from_f32(f);
        self.gf8_to_gf16 += 1;
        if (f - q.to_f32()).abs() > f32::EPSILON {
            self.lossy_total += 1;
        }
        q
    }

    pub fn convert_ternary_to_gf8(&mut self, val: GFTernary) -> GF8 {
        let f = val.to_f32();
        let q = GF8::from_f32(f);
        self.ternary_to_gf8 += 1;
        if (f - q.to_f32()).abs() > f32::EPSILON {
            self.lossy_total += 1;
        }
        q
    }

    pub fn convert_f32_to_bf16(&mut self, val: f32) -> u16 {
        let bits = val.to_bits();
        let bf16 = (bits >> 16) as u16;
        self.f32_to_bf16 += 1;
        let reconstructed = f32::from_bits((bf16 as u32) << 16);
        if (val - reconstructed).abs() > f32::EPSILON {
            self.lossy_total += 1;
        }
        bf16
    }

    /// Convert f32 → Posit16 (Gustafson 2017, es=1, useed=4), counting the
    /// conversion and bumping `lossy_total` if precision was lost.
    pub fn convert_f32_to_posit16(&mut self, val: f32) -> Posit16 {
        let p = Posit16::from_f32(val);
        self.f32_to_posit16 += 1;
        let reconstructed = p.to_f32();
        if (val - reconstructed).abs() > f32::EPSILON {
            self.lossy_total += 1;
        }
        p
    }
}

/// Apply Posit16 (es=1) round-trip quantization in place.
///
/// Master copy stays f32 in the caller (shadow-weight pattern). Each value
/// is encoded to Posit16, decoded back to f32, and the result replaces the
/// input. Used as the "Posit16 arm" in the F2 format-zoo comparison
/// against bf16 / E4M3 / GF16 at the same nominal bit-width.
///
/// Underflow semantics differ from f16: Posit16 saturates to MIN_POS rather
/// than rounding to zero, which is the load-bearing distinction for
/// gradient-update preservation in late-training low-precision regimes.
pub fn apply_posit16(values: &mut [f32], counter: &mut ConversionCounter) {
    for v in values.iter_mut() {
        let original = *v;
        let p = Posit16::from_f32(original);
        let dequant = p.to_f32();
        counter.f32_to_posit16 += 1;
        if (original - dequant).abs() > f32::EPSILON {
            counter.lossy_total += 1;
        }
        *v = dequant;
    }
}

/// Quantize a single f32 to FP8 E4M3 representable grid (3 mantissa bits, ±448 range).
/// NVIDIA TE forward weights/activations format (Format::HYBRID).
pub fn quantize_e4m3(x: f32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return if x.is_nan() { f32::NAN } else { 0.0 };
    }
    let max_e4m3 = 448.0_f32;
    let min_e4m3 = 2.0_f32.powi(-9);
    let abs = x.abs();
    let sign = x.signum();
    if abs > max_e4m3 {
        return sign * max_e4m3;
    }
    if abs < min_e4m3 {
        return 0.0;
    }
    let exp = abs.log2().floor();
    let step = 2.0_f32.powf(exp - 3.0);
    sign * ((abs / step).round() * step)
}

/// Quantize a single f32 to FP8 E5M2 representable grid (2 mantissa bits, ±57344 range).
/// NVIDIA TE backward gradients format (Format::HYBRID).
pub fn quantize_e5m2(x: f32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return if x.is_nan() { f32::NAN } else { 0.0 };
    }
    let max_e5m2 = 57344.0_f32;
    let min_e5m2 = 2.0_f32.powi(-16);
    let abs = x.abs();
    let sign = x.signum();
    if abs > max_e5m2 {
        return sign * max_e5m2;
    }
    if abs < min_e5m2 {
        return 0.0;
    }
    let exp = abs.log2().floor();
    let step = 2.0_f32.powf(exp - 2.0);
    sign * ((abs / step).round() * step)
}

/// ParetoQ unified low-bit quantizer (arXiv:2502.02631).
///
/// Per-bit grids and scale init (per ParetoQ §3.2-3.3, NeurIPS 2025):
///   P=1.58 ternary (SEQ symmetric): {−1, 0, +1},      α = max(|W|)
///   P=2.0  4-level  (SEQ symmetric): {−1.5, −0.5, +0.5, +1.5}, α = max(|W|)
///   P=3.0  8-level  (LSQ asymmetric): {−4..+3}, α = 2·‖W‖₂/√Q_p
///   P=4.0  16-level (LSQ asymmetric INT4): {−8..+7}, α = 2·‖W‖₂/√Q_p
///
/// IMPORTANT (loop 13 fix): P=2.0 uses SEQ symmetric grid with half-integer offsets,
/// NOT the previous LSQ asymmetric {−2,−1,0,+1}. The latter undersaled in absmean
/// scaling → most values collapsed to ±1 → degenerated to ternary.
#[derive(Debug, Clone, Copy)]
pub enum ParetoQFamily {
    /// SEQ symmetric — half-integer offsets for 1.58/2-bit ({−1, 0, +1} or {−1.5..+1.5}).
    SeqSymmetric { half_levels: f32 },
    /// LSQ asymmetric — integer levels with explicit 0 ({−4..+3} or {−8..+7}).
    LsqAsymmetric { q_n: i32, q_p: i32 },
}

pub fn paretoq_family(p_w: f64) -> ParetoQFamily {
    if p_w <= 1.6 {
        ParetoQFamily::SeqSymmetric { half_levels: 1.0 } // {-1, 0, 1}
    } else if p_w <= 2.5 {
        ParetoQFamily::SeqSymmetric { half_levels: 1.5 } // {-1.5, -0.5, +0.5, +1.5}
    } else if p_w <= 3.5 {
        ParetoQFamily::LsqAsymmetric { q_n: -4, q_p: 3 }
    } else {
        ParetoQFamily::LsqAsymmetric { q_n: -8, q_p: 7 }
    }
}

/// Legacy integer-grid accessor (kept for backward compat with sweep CSV).
/// Maps SEQ grids to the closest integer (Q_n, Q_p) for reporting purposes only.
pub fn paretoq_grid(p_w: f64) -> (i32, i32) {
    match paretoq_family(p_w) {
        ParetoQFamily::SeqSymmetric { half_levels } => {
            if half_levels <= 1.0 {
                (-1, 1)
            } else {
                (-2, 1) // 4 SEQ levels in symmetric grid summarized as 4-cell asymmetric
            }
        }
        ParetoQFamily::LsqAsymmetric { q_n, q_p } => (q_n, q_p),
    }
}

pub fn paretoq_scale(values: &[f32], family: ParetoQFamily) -> f32 {
    match family {
        ParetoQFamily::SeqSymmetric { half_levels } => {
            // SEQ: α = max(|W|) / outermost_level
            // Outermost level is half_levels (1.0 for ternary, 1.5 for 4-cell SEQ).
            // This makes max|W| map exactly to ±outermost·α, avoiding tie-induced collapse.
            let max_abs: f32 = values.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
            (max_abs / half_levels.max(1e-9)).max(f32::EPSILON)
        }
        ParetoQFamily::LsqAsymmetric { q_p, .. } => {
            // LSQ: α = 2·‖W‖₂ / √Q_p (Esser et al. arXiv:1902.08153)
            let l2_sq: f32 = values.iter().map(|x| x * x).sum();
            let l2_norm = l2_sq.sqrt();
            let q_p_f = q_p.max(1) as f32;
            (2.0 * l2_norm / q_p_f.sqrt() / (values.len().max(1) as f32).sqrt()).max(f32::EPSILON)
        }
    }
}

/// Apply ParetoQ quantization round-trip in-place for a given precision P_w.
pub fn apply_paretoq(values: &mut [f32], p_w: f64, counter: &mut ConversionCounter) {
    if values.is_empty() {
        return;
    }
    let family = paretoq_family(p_w);
    let s = paretoq_scale(values, family);

    for v in values.iter_mut() {
        let original = *v;
        let dequant = match family {
            ParetoQFamily::SeqSymmetric { half_levels } => {
                // Map x/s to nearest of {-half, ..., -0.5, +0.5, ..., +half} if half_levels=1.5
                // OR {-1, 0, +1} if half_levels=1.0.
                let x_scaled = original / s;
                if (half_levels - 1.0).abs() < 1e-6 {
                    // Ternary: round to {-1, 0, +1}
                    x_scaled.round().clamp(-1.0, 1.0) * s
                } else {
                    // 4-level SEQ: snap to {-1.5, -0.5, +0.5, +1.5}
                    let levels = [-half_levels, -0.5_f32, 0.5_f32, half_levels];
                    let mut best = levels[0];
                    let mut best_err = (x_scaled - best).abs();
                    for &lv in &levels[1..] {
                        let err = (x_scaled - lv).abs();
                        if err < best_err {
                            best_err = err;
                            best = lv;
                        }
                    }
                    best * s
                }
            }
            ParetoQFamily::LsqAsymmetric { q_n, q_p } => {
                let scaled = (original / s).round().clamp(q_n as f32, q_p as f32);
                scaled * s
            }
        };
        counter.f32_to_paretoq += 1;
        if (original - dequant).abs() > f32::EPSILON {
            counter.lossy_total += 1;
        }
        *v = dequant;
    }
}

/// Compute BitNet b1.58 per-tensor absmean scale: α = mean(|W|).
/// Reference: BitNet b1.58 paper (arXiv:2402.17764) §2.1 "Quantization Function".
/// Returns max(α, ε) to avoid division by zero on all-zero tensors.
pub fn absmean_scale(values: &[f32]) -> f32 {
    if values.is_empty() {
        return f32::EPSILON;
    }
    let sum_abs: f32 = values.iter().map(|x| x.abs()).sum();
    (sum_abs / values.len() as f32).max(f32::EPSILON)
}

/// Apply phi-ladder quantization round-trip with BitNet absmean scaling.
///
/// Formula (BitNet b1.58, arXiv:2402.17764):
///   α = mean(|W|)
///   W̃ = RoundClip(W / (α + ε), -1, +1) * α
///
/// At each rung the tensor is rescaled by its own absmean before quantization,
/// then multiplied back. Without this scaling, raw embeddings (~0.01 magnitude)
/// collapse to zero under ternary {-1, 0, +1} — destroying signal at log₂(vocab)
/// saturation. With scaling, the ternary grid spans the actual data range.
///
/// Shadow-weight pattern: master stays f32 between forward passes (caller responsibility).
pub fn apply_phi_ladder(values: &mut [f32], counter: &mut ConversionCounter) {
    if values.is_empty() {
        return;
    }
    let alpha = absmean_scale(values);
    let inv_alpha = 1.0 / alpha;

    for v in values.iter_mut() {
        let original = *v;

        // Scale into quantization domain [−1, +1] (BitNet b1.58 §2.1)
        let scaled = original * inv_alpha;

        let gf16 = GF16::from_f32(scaled);
        counter.gf32_to_gf16 += 1;
        if (scaled - gf16.to_f32()).abs() > f32::EPSILON {
            counter.lossy_total += 1;
        }

        let gf8 = GF8::from_f32(gf16.to_f32());
        counter.gf16_to_gf8 += 1;
        if (gf16.to_f32() - gf8.to_f32()).abs() > f32::EPSILON {
            counter.lossy_total += 1;
        }

        let ternary = GFTernary::from_f32(gf8.to_f32());
        counter.gf8_to_ternary += 1;
        if (gf8.to_f32() - ternary.to_f32()).abs() > f32::EPSILON {
            counter.lossy_total += 1;
        }

        // Ascent path
        let gf8_up = GF8::from_f32(ternary.to_f32());
        counter.ternary_to_gf8 += 1;
        let gf16_up = GF16::from_f32(gf8_up.to_f32());
        counter.gf8_to_gf16 += 1;

        // Dequantize back to original scale (multiply by α)
        *v = gf16_up.to_f32() * alpha;
    }
}

/// Boundary between INT4 RTN and bf16+E4M3 HYBRID for zoo arm precision dispatch.
/// Below this P_w, zoo uses INT4 RTN (GPTQ baseline). Above, bf16/E4M3 HYBRID.
/// Loop 13 exposed as const for grep-ability and tuning.
pub const ZOO_INT4_FP8_BOUNDARY: f64 = 4.5;

/// GPTQ-style INT4 round-to-nearest, per-group symmetric (group_size=128 default).
/// Formula (Frantar et al. arXiv:2210.17323 §3.1, simplest baseline without Hessian):
///   s_g = max(|W_g|) / 7        (qmax = 2^(4-1) - 1 = 7)
///   q   = clip(round(w/s), -8, 7)
///   W̃   = q · s
///
/// This is the canonical RTN-INT4 used as baseline before GPTQ's error correction.
pub fn apply_int4_rtn(values: &mut [f32], group_size: usize, counter: &mut ConversionCounter) {
    let group_size = group_size.max(1);
    let n = values.len();
    let mut i = 0;
    while i < n {
        let end = (i + group_size).min(n);
        let group = &values[i..end];
        let max_abs: f32 = group.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let s = (max_abs / 7.0).max(f32::EPSILON);
        for v in &mut values[i..end] {
            let original = *v;
            let q = (original / s).round().clamp(-8.0, 7.0);
            let dq = q * s;
            counter.f32_to_int4 += 1;
            if (original - dq).abs() > f32::EPSILON {
                counter.lossy_total += 1;
            }
            *v = dq;
        }
        i = end;
    }
}

/// Per-precision zoo dispatch — returns correct quantizer for given P_w.
/// Loop 12 fix: zoo arm at P=4.0 uses INT4 RTN (not bf16 truncation).
///   P ≤ 4.5 → INT4 RTN (GPTQ baseline)
///   P > 4.5 → bf16/E4M3 HYBRID (NVIDIA TE)
pub fn apply_zoo_at_precision(values: &mut [f32], p_w: f64, counter: &mut ConversionCounter) {
    if p_w <= ZOO_INT4_FP8_BOUNDARY {
        // Loop 14 fix: group_size=32 (was 128, degenerate at d_model=128).
        // At small models, group_size=128 = whole tensor → single scale → no per-group benefit.
        // 32 keeps 4 groups at d_model=128 — honest GPTQ behavior.
        apply_int4_rtn(values, 32, counter);
    } else {
        apply_format_zoo(values, counter);
    }
}

/// Apply format-zoo HYBRID quantization — NVIDIA TE Format::HYBRID-style baseline.
///
/// Forward path uses E4M3 (3 mantissa bits, ±448, finer precision in small-magnitude range).
/// Reference: NVIDIA Transformer Engine FP8 primer, FP8-LM (arXiv:2310.18313).
/// Backward gradient quantization uses E5M2 — see `apply_format_zoo_grads`.
/// Master copy stays f32 in the caller (shadow-weight pattern).
pub fn apply_format_zoo(values: &mut [f32], counter: &mut ConversionCounter) {
    for v in values.iter_mut() {
        let original = *v;
        let q = quantize_e4m3(original);
        counter.f32_to_fp8_e4m3 += 1;
        counter.f32_to_fp8 += 1;
        if (original - q).abs() > f32::EPSILON {
            counter.lossy_total += 1;
        }
        *v = q;
    }
}

/// Apply E5M2 quantization to gradients — NVIDIA TE HYBRID backward format.
pub fn apply_format_zoo_grads(grads: &mut [f32], counter: &mut ConversionCounter) {
    for g in grads.iter_mut() {
        let original = *g;
        let q = quantize_e5m2(original);
        counter.f32_to_fp8_e5m2 += 1;
        counter.f32_to_fp8 += 1;
        if (original - q).abs() > f32::EPSILON {
            counter.lossy_total += 1;
        }
        *g = q;
    }
}

impl core::fmt::Display for ConversionCounter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "lossy={} (gf32→16:{} gf16→8:{} gf8→t:{} ↑16→32:{} ↑8→16:{} \
             ↑t→8:{} bf16:{} int8:{} int4:{} fp8:{}(e4m3:{} e5m2:{}) \
             paretoq:{} posit16:{})",
            self.lossy_total,
            self.gf32_to_gf16,
            self.gf16_to_gf8,
            self.gf8_to_ternary,
            self.gf16_to_gf32,
            self.gf8_to_gf16,
            self.ternary_to_gf8,
            self.f32_to_bf16,
            self.f32_to_int8,
            self.f32_to_int4,
            self.f32_to_fp8,
            self.f32_to_fp8_e4m3,
            self.f32_to_fp8_e5m2,
            self.f32_to_paretoq,
            self.f32_to_posit16,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_starts_at_zero() {
        let c = ConversionCounter::new();
        assert_eq!(c.total(), 0);
        assert_eq!(c.gf32_to_gf16, 0);
    }

    #[test]
    fn phi_ladder_downscale_counts() {
        let mut c = ConversionCounter::new();
        let gf32 = GF32::from_f32(0.75);
        let gf16 = c.convert_gf32_to_gf16(gf32);
        let gf8 = c.convert_gf16_to_gf8(gf16);
        let _ternary = c.convert_gf8_to_ternary(gf8);

        assert_eq!(c.gf32_to_gf16, 1);
        assert_eq!(c.gf16_to_gf8, 1);
        assert_eq!(c.gf8_to_ternary, 1);
    }

    #[test]
    fn ternary_round_trip_is_lossy() {
        let mut c = ConversionCounter::new();
        let gf8 = GF8::from_f32(0.42);
        let _t = c.convert_gf8_to_ternary(gf8);
        assert!(
            c.lossy_total > 0,
            "ternary quantization of 0.42 should be lossy"
        );
    }

    #[test]
    fn bf16_truncation_counts() {
        let mut c = ConversionCounter::new();
        let _bf16 = c.convert_f32_to_bf16(std::f32::consts::PI);
        assert_eq!(c.f32_to_bf16, 1);
        assert!(c.lossy_total > 0, "bf16 truncation of PI should be lossy");
    }

    #[test]
    fn display_format() {
        let c = ConversionCounter::new();
        let s = format!("{}", c);
        assert!(s.starts_with("lossy=0"));
    }

    #[test]
    fn e4m3_zero_round_trips() {
        assert_eq!(quantize_e4m3(0.0), 0.0);
    }

    #[test]
    fn e4m3_clamps_to_max() {
        assert_eq!(quantize_e4m3(1000.0), 448.0);
        assert_eq!(quantize_e4m3(-1000.0), -448.0);
    }

    #[test]
    fn e4m3_underflows_small_to_zero() {
        // Below 2^-9 ≈ 0.00195
        assert_eq!(quantize_e4m3(1e-6), 0.0);
    }

    #[test]
    fn e5m2_has_wider_range_than_e4m3() {
        // E5M2 can represent magnitudes E4M3 cannot
        let big = 10_000.0_f32;
        assert!(quantize_e5m2(big).abs() > quantize_e4m3(big).abs());
    }

    #[test]
    fn e4m3_more_precise_than_e5m2_in_unit_band() {
        // E4M3 has 3 mantissa bits vs E5M2's 2 — more precision near 1.0
        let x = 1.1_f32;
        let err_e4m3 = (x - quantize_e4m3(x)).abs();
        let err_e5m2 = (x - quantize_e5m2(x)).abs();
        assert!(err_e4m3 <= err_e5m2);
    }

    #[test]
    fn apply_format_zoo_uses_e4m3() {
        let mut c = ConversionCounter::new();
        let mut v = vec![0.5_f32, 1.0, 2.0];
        apply_format_zoo(&mut v, &mut c);
        assert_eq!(c.f32_to_fp8_e4m3, 3);
        assert_eq!(c.f32_to_fp8, 3);
    }

    #[test]
    fn paretoq_grid_levels_per_paper() {
        assert_eq!(paretoq_grid(1.58), (-1, 1));
        assert_eq!(paretoq_grid(2.0), (-2, 1));
        assert_eq!(paretoq_grid(3.0), (-4, 3));
        assert_eq!(paretoq_grid(4.0), (-8, 7));
    }

    #[test]
    fn paretoq_scale_is_positive() {
        let v = [0.1_f32, -0.2, 0.3, -0.4];
        let s = paretoq_scale(&v, paretoq_family(4.0));
        assert!(s > 0.0 && s.is_finite());
    }

    #[test]
    fn paretoq_p2_gives_four_levels_not_ternary() {
        // Loop 13 fix: at P=2.0 with SEQ symmetric grid, values should occupy
        // 4 distinct cells {-1.5, -0.5, +0.5, +1.5}·s, not collapse to 3 ternary levels.
        let mut v: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.02).collect();
        let mut c = ConversionCounter::new();
        apply_paretoq(&mut v, 2.0, &mut c);
        let unique_count = {
            let mut sorted: Vec<f32> = v.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted.dedup_by(|a, b| (*a - *b).abs() < 1e-5);
            sorted.len()
        };
        // SEQ 4-level should give exactly 4 unique values, not 3 (ternary collapse).
        assert_eq!(
            unique_count, 4,
            "P=2.0 SEQ should yield 4 distinct values, got {}",
            unique_count
        );
    }

    #[test]
    fn paretoq_ternary_collapses_to_three_levels() {
        let mut v = vec![0.5_f32, -0.3, 0.1, -0.8, 0.6];
        let mut c = ConversionCounter::new();
        apply_paretoq(&mut v, 1.58, &mut c);
        // After ternary, absolute values are 0 or s (single nonzero magnitude).
        let mut abs_vals: Vec<f32> = v.iter().map(|x| x.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        abs_vals.dedup_by(|a, b| (*a - *b).abs() < f32::EPSILON);
        assert!(
            abs_vals.len() <= 2,
            "ternary should yield ≤2 distinct |x|, got {:?}",
            v
        );
    }

    #[test]
    fn paretoq_int4_has_more_distinct_values_than_ternary() {
        let v: Vec<f32> = (0..50).map(|i| (i as f32 * 0.01) - 0.25).collect();
        let mut v_ternary = v.clone();
        let mut v_int4 = v.clone();
        let mut c1 = ConversionCounter::new();
        let mut c2 = ConversionCounter::new();
        apply_paretoq(&mut v_ternary, 1.58, &mut c1);
        apply_paretoq(&mut v_int4, 4.0, &mut c2);
        let u_ternary: std::collections::HashSet<u32> =
            v_ternary.iter().map(|x| x.to_bits()).collect();
        let u_int4: std::collections::HashSet<u32> = v_int4.iter().map(|x| x.to_bits()).collect();
        assert!(u_int4.len() > u_ternary.len());
    }

    #[test]
    fn absmean_scale_matches_bitnet_formula() {
        let v = [1.0_f32, -2.0, 3.0, -4.0];
        // mean(|v|) = (1+2+3+4)/4 = 2.5
        assert!((absmean_scale(&v) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn absmean_scale_handles_empty_and_zero() {
        assert_eq!(absmean_scale(&[]), f32::EPSILON);
        assert_eq!(absmean_scale(&[0.0, 0.0]), f32::EPSILON);
    }

    #[test]
    fn scaled_phi_ladder_preserves_signal_magnitude() {
        // Small-magnitude embeddings (~0.01 — typical Xavier init at d_model=384)
        // should survive ternary cascade with absmean scaling.
        let mut v = vec![0.01_f32, -0.02, 0.015, -0.008, 0.025];
        let original = v.clone();
        let mut c = ConversionCounter::new();
        apply_phi_ladder(&mut v, &mut c);

        // After scaling, the output should NOT collapse to all-zeros (the old bug).
        let nonzero_count = v.iter().filter(|x| x.abs() > f32::EPSILON).count();
        assert!(
            nonzero_count >= 3,
            "scaled phi-ladder must preserve at least 3/5 nonzero values; got {:?}",
            v
        );

        // Sign should be preserved on most entries (ternary keeps sign).
        let signs_preserved = original
            .iter()
            .zip(v.iter())
            .filter(|(a, b)| a.signum() == b.signum() || b.abs() < f32::EPSILON)
            .count();
        assert!(
            signs_preserved >= 4,
            "ternary should preserve sign; got {:?}",
            v
        );
    }

    #[test]
    fn apply_format_zoo_grads_uses_e5m2() {
        let mut c = ConversionCounter::new();
        let mut g = vec![0.01_f32, 0.1, 1.0];
        apply_format_zoo_grads(&mut g, &mut c);
        assert_eq!(c.f32_to_fp8_e5m2, 3);
        assert_eq!(c.f32_to_fp8, 3);
    }

    #[test]
    fn upscale_gf8_to_gf16_lossless_for_representable() {
        let mut c = ConversionCounter::new();
        let gf8 = GF8::from_f32(0.0);
        let _gf16 = c.convert_gf8_to_gf16(gf8);
        assert_eq!(c.gf8_to_gf16, 1);
    }

    #[test]
    fn posit16_round_trip_counts() {
        let mut c = ConversionCounter::new();
        let _ = c.convert_f32_to_posit16(std::f32::consts::PI);
        assert_eq!(c.f32_to_posit16, 1);
        assert!(c.lossy_total > 0, "Posit16(π) should be lossy");
    }

    #[test]
    fn apply_posit16_round_trip_in_place() {
        let mut c = ConversionCounter::new();
        let mut v = vec![0.5_f32, 1.0, 2.0, std::f32::consts::PI];
        apply_posit16(&mut v, &mut c);
        assert_eq!(c.f32_to_posit16, 4);
        assert_eq!(v[0], 0.5);
        assert_eq!(v[1], 1.0);
        assert_eq!(v[2], 2.0);
    }

    #[test]
    fn posit16_does_not_underflow_tiny_to_zero() {
        // Posit16's MIN_POS = 2^-28; values smaller than that saturate to MIN_POS,
        // unlike f16 which rounds to zero. This is the F2-relevant distinction.
        let mut c = ConversionCounter::new();
        let mut v = vec![1e-30_f32];
        apply_posit16(&mut v, &mut c);
        assert!(v[0] > 0.0, "Posit16 must not underflow to zero; got {:?}", v[0]);
        assert!(v[0] <= 2.0_f32.powi(-27),
                "Posit16 MIN_POS expected, got {:?}", v[0]);
    }
}
