//! Fake-quantization (QAT) for the cpu_train pipeline.
//!
//! Closes the methodology gap surfaced in
//! [trios#509](https://github.com/gHashTag/trios/issues/509):
//! Wave-3 canary showed BPB collapsing to a single per-seed value across
//! 31 declared formats, because the trainer never invoked any
//! quantize/dequantize path — `TRIOS_FORMAT_TYPE` was used only for the
//! result-file suffix.
//!
//! This module introduces a minimal `FakeQuant` operator that provides
//! the canonical QAT contract:
//!
//! ```text
//!     forward:  y = dequantize(quantize(x))
//!     backward: dy/dx = 1   (Straight-Through Estimator)
//! ```
//!
//! The forward pass injects a format-specific rounding error into the
//! activations / weights, while the backward pass is identity — exactly
//! the recipe used by `Bengio (2013)`, `PACT (Choi 2018)` and
//! `LSQ (Esser 2020)`. Because the `cpu_train` backward path is hand-coded
//! against `f32` parameters, the STE means we can apply the quant error
//! during forward without any change to existing gradient flow.
//!
//! ## Supported formats (initial 8)
//!
//! | Token            | Family              | Round mode                     |
//! |------------------|---------------------|--------------------------------|
//! | `fp32` (default) | identity            | none — scalar pass-through     |
//! | `bf16`           | bfloat16            | round-to-nearest mantissa-7    |
//! | `fp16`           | binary16            | round-to-nearest mantissa-10   |
//! | `fp8_e4m3`       | OFP8 E4M3           | scale + 4-bit-mantissa round   |
//! | `fp8_e5m2`       | OFP8 E5M2           | scale + 2-bit-mantissa round   |
//! | `int8`           | symmetric int8      | per-tensor scale, clamp [-127,127] |
//! | `int4`           | symmetric int4      | per-tensor scale, clamp [-7,7]  |
//! | `nf4`            | NormalFloat4        | per-tensor scale, NF4 codebook  |
//!
//! Additional formats from `phd_format_registry` will be wired up in
//! follow-up PRs after this lands and Wave-5 canary confirms divergence.

#![allow(clippy::module_name_repetitions)]

/// Quantization mode controlled at runtime by `TRIOS_FORMAT_TYPE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormatKind {
    /// No quantization — full f32 baseline (the historical Wave-1..Wave-3 path).
    Fp32,
    Bf16,
    Fp16,
    Fp8E4M3,
    Fp8E5M2,
    Int8,
    Int4,
    Nf4,
}

impl FormatKind {
    /// Parse `TRIOS_FORMAT_TYPE`. Unknown / empty / `f32` → [`FormatKind::Fp32`].
    pub fn from_env() -> Self {
        match std::env::var("TRIOS_FORMAT_TYPE")
            .ok()
            .as_deref()
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("bf16") | Some("bfloat16") => Self::Bf16,
            Some("fp16") | Some("binary16") => Self::Fp16,
            Some("fp8_e4m3") | Some("fp8e4m3") => Self::Fp8E4M3,
            Some("fp8_e5m2") | Some("fp8e5m2") => Self::Fp8E5M2,
            Some("int8") => Self::Int8,
            Some("int4") => Self::Int4,
            Some("nf4") => Self::Nf4,
            _ => Self::Fp32,
        }
    }

    /// Stable lowercase token for filenames / DB rows.
    pub fn token(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Bf16 => "bf16",
            Self::Fp16 => "fp16",
            Self::Fp8E4M3 => "fp8_e4m3",
            Self::Fp8E5M2 => "fp8_e5m2",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
            Self::Nf4 => "nf4",
        }
    }
}

/// NF4 codebook (16 levels) — quantiles of a standard normal, normalized.
/// Source: `Dettmers et al. 2023 (QLoRA)` table 3 / reference impl.
const NF4_CODEBOOK: [f32; 16] = [
    -1.0, -0.6961928, -0.5250730, -0.3949175, -0.2844967, -0.1848188, -0.0917970, 0.0,
    0.0795803, 0.1609302, 0.2461123, 0.3379152, 0.4407410, 0.5626170, 0.7229568, 1.0,
];

/// Round f32 → bfloat16 → f32 by truncating mantissa to 7 bits with
/// round-to-nearest-even.
fn round_bf16(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    // Round-to-nearest-even on the 16 bits we are about to drop.
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    f32::from_bits(rounded & 0xFFFF_0000)
}

/// Round f32 → IEEE binary16 → f32. We re-use the rounding logic from
/// `f32 → f16 → f32` by manipulating bit patterns directly to avoid
/// pulling in a `half` crate dependency.
fn round_fp16(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    // f32: 1 sign | 8 exp | 23 mantissa
    // f16: 1 sign | 5 exp | 10 mantissa
    let f = x;
    let bits = f.to_bits();
    let sign = (bits >> 31) & 0x1;
    let exp32 = ((bits >> 23) & 0xFF) as i32;
    let mant32 = bits & 0x7F_FFFF;

    if exp32 == 0 {
        // Subnormal in f32 — flushes to zero in f16 with our config.
        return if sign == 1 { -0.0 } else { 0.0 };
    }

    let exp16 = exp32 - 127 + 15;
    if exp16 >= 31 {
        // Overflow — saturate to the largest representable f16 value
        // (no inf/nan to keep training stable).
        let max_f16: f32 = 65504.0;
        return if sign == 1 { -max_f16 } else { max_f16 };
    }
    if exp16 <= 0 {
        // Underflow — flush to zero (matches the GF16 contract).
        return if sign == 1 { -0.0 } else { 0.0 };
    }

    // Round-to-nearest-even on the bottom 13 mantissa bits.
    let mant16 = (mant32 + (1 << 12) + ((mant32 >> 13) & 1)) >> 13;
    // If rounding overflowed the 10-bit mantissa, bump the exponent.
    let (final_exp, final_mant) = if mant16 >= 1 << 10 {
        (exp16 + 1, 0)
    } else {
        (exp16, mant16)
    };
    if final_exp >= 31 {
        let max_f16: f32 = 65504.0;
        return if sign == 1 { -max_f16 } else { max_f16 };
    }

    // Reconstruct as f32 by re-biasing the exponent.
    let new_exp32 = (final_exp - 15 + 127) as u32;
    let new_bits = (sign << 31) | (new_exp32 << 23) | (final_mant << 13);
    f32::from_bits(new_bits)
}

/// Generic OFP8 round: keep `mantissa_bits` mantissa bits and clamp to
/// `[-max_abs, max_abs]`. Subnormals are flushed.
fn round_fp8(x: f32, mantissa_bits: u32, max_abs: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    if x.abs() < f32::EPSILON {
        return 0.0;
    }
    let clamped = x.clamp(-max_abs, max_abs);
    // Round mantissa to `mantissa_bits` bits using round-to-nearest-even.
    let bits = clamped.to_bits();
    let drop = 23 - mantissa_bits;
    let half = 1u32 << (drop - 1);
    let lsb = (bits >> drop) & 1;
    let rounded = bits.wrapping_add(half - 1 + lsb);
    let mask = !((1u32 << drop) - 1);
    f32::from_bits(rounded & mask)
}

/// Symmetric per-tensor INT-Q round with straight-through error.
fn round_int_symmetric(x: f32, levels: i32, scale: f32) -> f32 {
    if scale.abs() < f32::EPSILON {
        return 0.0;
    }
    let q = (x / scale).round().clamp(-(levels as f32), levels as f32);
    q * scale
}

/// NF4 round — find the nearest entry in the codebook on the normalized
/// scale, then re-scale back.
fn round_nf4(x: f32, scale: f32) -> f32 {
    if scale.abs() < f32::EPSILON {
        return 0.0;
    }
    let normalized = (x / scale).clamp(-1.0, 1.0);
    let mut best = NF4_CODEBOOK[0];
    let mut best_d = (best - normalized).abs();
    for &v in &NF4_CODEBOOK[1..] {
        let d = (v - normalized).abs();
        if d < best_d {
            best = v;
            best_d = d;
        }
    }
    best * scale
}

/// Quantize-dequantize a slice in-place. The backward pass is the
/// straight-through estimator (identity), so callers only need to apply
/// this on the forward side; gradient code is unchanged.
pub fn fake_quant_in_place(buf: &mut [f32], kind: FormatKind) {
    if matches!(kind, FormatKind::Fp32) || buf.is_empty() {
        return;
    }

    // Per-tensor scale derived from |x|_max for INT / NF4 paths.
    let abs_max = buf
        .iter()
        .copied()
        .fold(0.0_f32, |m, v| if v.is_finite() { m.max(v.abs()) } else { m });

    match kind {
        FormatKind::Fp32 => {}
        FormatKind::Bf16 => {
            for v in buf.iter_mut() {
                *v = round_bf16(*v);
            }
        }
        FormatKind::Fp16 => {
            for v in buf.iter_mut() {
                *v = round_fp16(*v);
            }
        }
        FormatKind::Fp8E4M3 => {
            // OFP8 E4M3 max representable ≈ 448.
            for v in buf.iter_mut() {
                *v = round_fp8(*v, 3, 448.0);
            }
        }
        FormatKind::Fp8E5M2 => {
            // OFP8 E5M2 max representable ≈ 57344.
            for v in buf.iter_mut() {
                *v = round_fp8(*v, 2, 57344.0);
            }
        }
        FormatKind::Int8 => {
            let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
            for v in buf.iter_mut() {
                *v = round_int_symmetric(*v, 127, scale);
            }
        }
        FormatKind::Int4 => {
            let scale = if abs_max > 0.0 { abs_max / 7.0 } else { 1.0 };
            for v in buf.iter_mut() {
                *v = round_int_symmetric(*v, 7, scale);
            }
        }
        FormatKind::Nf4 => {
            let scale = if abs_max > 0.0 { abs_max } else { 1.0 };
            for v in buf.iter_mut() {
                *v = round_nf4(*v, scale);
            }
        }
    }
}

/// Convenience wrapper for nested `Vec<Vec<f32>>` activations.
pub fn fake_quant_nested(activations: &mut [Vec<f32>], kind: FormatKind) {
    if matches!(kind, FormatKind::Fp32) {
        return;
    }
    for row in activations.iter_mut() {
        fake_quant_in_place(row, kind);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp32_is_identity() {
        let mut a = [1.0_f32, -2.5, 0.125, 1e-9];
        let original = a;
        fake_quant_in_place(&mut a, FormatKind::Fp32);
        assert_eq!(a, original);
    }

    #[test]
    fn bf16_drops_low_mantissa_bits() {
        // 1.0 + tiny noise should round back to 1.0 in bf16.
        let mut a = [1.0_f32 + (1e-5)];
        fake_quant_in_place(&mut a, FormatKind::Bf16);
        assert_eq!(a[0], 1.0_f32);
    }

    #[test]
    fn fp16_overflow_saturates_not_inf() {
        let mut a = [1.0e30_f32];
        fake_quant_in_place(&mut a, FormatKind::Fp16);
        assert!(a[0].is_finite());
        assert!(a[0] <= 65504.0 && a[0] > 0.0);
    }

    #[test]
    fn int8_changes_values() {
        let mut a = [0.001_f32, 0.5, -0.7];
        let before = a;
        fake_quant_in_place(&mut a, FormatKind::Int8);
        // Values must change (proves the kernel actually fires).
        assert_ne!(a, before);
        // But scale should keep them in roughly the same range.
        for (b, q) in before.iter().zip(a.iter()) {
            assert!((b - q).abs() < 0.7_f32);
        }
    }

    #[test]
    fn int4_quantizes_more_aggressively_than_int8() {
        let buf = [-0.9_f32, -0.3, 0.1, 0.4, 0.8];
        let mut a = buf;
        let mut b = buf;
        fake_quant_in_place(&mut a, FormatKind::Int4);
        fake_quant_in_place(&mut b, FormatKind::Int8);
        let err4: f32 = a.iter().zip(buf.iter()).map(|(x, y)| (x - y).abs()).sum();
        let err8: f32 = b.iter().zip(buf.iter()).map(|(x, y)| (x - y).abs()).sum();
        assert!(err4 >= err8);
    }

    #[test]
    fn nf4_picks_codebook_levels() {
        let mut a = [0.5_f32, -0.7];
        fake_quant_in_place(&mut a, FormatKind::Nf4);
        // Each output must equal one of the codebook levels times the scale.
        for v in a.iter() {
            let normalized = v / 0.7_f32;
            assert!(NF4_CODEBOOK.iter().any(|c| (c - normalized).abs() < 0.05));
        }
    }

    #[test]
    fn different_formats_diverge() {
        // The whole point of #509: declaring different formats must
        // produce different post-quant tensors when |x| is non-trivial.
        let buf = [0.123_f32, 0.456, -0.789, 1.234];
        let mut a = buf;
        let mut b = buf;
        let mut c = buf;
        fake_quant_in_place(&mut a, FormatKind::Bf16);
        fake_quant_in_place(&mut b, FormatKind::Int8);
        fake_quant_in_place(&mut c, FormatKind::Nf4);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn from_env_fp32_default() {
        let prev = std::env::var("TRIOS_FORMAT_TYPE").ok();
        std::env::remove_var("TRIOS_FORMAT_TYPE");
        assert_eq!(FormatKind::from_env(), FormatKind::Fp32);
        if let Some(v) = prev {
            std::env::set_var("TRIOS_FORMAT_TYPE", v);
        }
    }
}
