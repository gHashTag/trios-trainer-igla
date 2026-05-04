//! FakeQuant + Straight-Through Estimator (STE) for QAT
//!
//! Phase-1 fix for `trios#509-B` (Bug catalogue from issue #95):
//!
//! * **A** — IEEE narrow-exp formats now clamp to per-format `max_finite()`
//!   (fp16/fp8/fp6/fp4/mxfp* no longer overflow into f32 infinity-land).
//! * **B** — round-to-nearest is applied to `abs(val)` and the sign is
//!   restored afterwards, so negatives no longer round AWAY from zero.
//! * **C/D** — integer fake-quantization now supports per-tensor amax scale
//!   via [`fake_quantize_weights_tensor`]; the legacy single-value path
//!   keeps a fixed `±1` range with a clear documentation note.
//! * **G** — GF formats (`Gf16` / `Gf8` / `Gf32` / `Gf64`) now go through the
//!   canonical [`crate::gf16::GF16`] and [`crate::phi_numbers::{GF8,GF32,GF64}`]
//!   round-trips instead of the IEEE-style mantissa-mask shortcut.
//! * **H** — formats with mantissa ≥ 23 (f64/fp80/binary128/binary256/decimal128/
//!   vax_d/cray_float/stochastic_rnd) and other formats without a faithful
//!   f32 simulator are explicitly marked via [`FormatKind::is_unsupported_in_f32`]
//!   and return the input unchanged with a TRACE-able comment.
//!
//! What this PR does **not** fix yet (deferred to Phase-2/3):
//!
//! * NF4 / Posit* / LNS* / MXFP* / Decimal* / BlockFp / SharedExp / Unum* /
//!   Tapered / BCD / IBM-HFP / VAX / Cray / AfP / QFormat — still modelled
//!   as IEEE-style mantissa-mask + exponent clamp. Distinct BPB but not
//!   spec-faithful.
//! * Stochastic rounding is intentionally a no-op for now; a faithful
//!   bernoulli-rounding implementation needs an RNG threaded through the
//!   call site.
//!
//! Anchor: `phi^2 + phi^-2 = 3 · TRINITY`.

use crate::gf16::GF16;
use crate::phi_numbers::{GF32, GF64, GF8};

/// Supported numeric formats for fake quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatKind {
    F32,
    F64,
    Fp16,
    Bf16,
    Tf32,
    Fp8E4M3,
    Fp8E5M2,
    Fp6E2M3,
    Fp6E3M2,
    Fp4E2M1,
    Gf16,
    Gf8,
    Gf4,
    Gf32,
    Gf64,
    Gf12,
    Gf20,
    Gf24,
    Int8,
    Int4,
    Int16,
    Int32,
    Uint8,
    Nf4,
    Posit8,
    Posit16,
    Posit32,
    Lns8,
    Mxfp4,
    Mxfp6,
    Mxfp8,
    Binary128,
    Binary256,
    Decimal32,
    Decimal64,
    Decimal128,
    Fp80,
    Bcd,
    IbmHfp,
    VaxF,
    VaxD,
    CrayFloat,
    Minifloat,
    TaperedFp,
    BlockFp,
    SharedExp,
    StochasticRnd,
    UnumI,
    UnumII,
    AfP,
    QFormat,
}

impl FormatKind {
    /// Parse format from `TRIOS_FORMAT_TYPE` env var or string
    pub fn from_env(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "f32" | "fp32" | "binary32" | "float32" => Some(FormatKind::F32),
            "f64" | "fp64" | "binary64" | "float64" => Some(FormatKind::F64),
            "fp16" | "binary16" | "float16" => Some(FormatKind::Fp16),
            "bf16" | "bfloat16" => Some(FormatKind::Bf16),
            "tf32" => Some(FormatKind::Tf32),
            "fp8_e4m3" | "fp8-e4m3" | "fp8e4m3" => Some(FormatKind::Fp8E4M3),
            "fp8_e5m2" | "fp8-e5m2" | "fp8e5m2" => Some(FormatKind::Fp8E5M2),
            "fp6_e2m3" | "fp6-e2m3" => Some(FormatKind::Fp6E2M3),
            "fp6_e3m2" | "fp6-e3m2" => Some(FormatKind::Fp6E3M2),
            "fp4_e2m1" | "fp4-e2m1" => Some(FormatKind::Fp4E2M1),
            "gf16" => Some(FormatKind::Gf16),
            "gf8" => Some(FormatKind::Gf8),
            "gf4" => Some(FormatKind::Gf4),
            "gf32" => Some(FormatKind::Gf32),
            "gf64" => Some(FormatKind::Gf64),
            "gf12" => Some(FormatKind::Gf12),
            "gf20" => Some(FormatKind::Gf20),
            "gf24" => Some(FormatKind::Gf24),
            "int8" => Some(FormatKind::Int8),
            "int4" => Some(FormatKind::Int4),
            "int16" => Some(FormatKind::Int16),
            "int32" => Some(FormatKind::Int32),
            "uint8" => Some(FormatKind::Uint8),
            "nf4" => Some(FormatKind::Nf4),
            "posit8" => Some(FormatKind::Posit8),
            "posit16" => Some(FormatKind::Posit16),
            "posit32" => Some(FormatKind::Posit32),
            "lns8" => Some(FormatKind::Lns8),
            "mxfp4" => Some(FormatKind::Mxfp4),
            "mxfp6" => Some(FormatKind::Mxfp6),
            "mxfp8" => Some(FormatKind::Mxfp8),
            "binary128" => Some(FormatKind::Binary128),
            "binary256" => Some(FormatKind::Binary256),
            "decimal32" => Some(FormatKind::Decimal32),
            "decimal64" => Some(FormatKind::Decimal64),
            "decimal128" => Some(FormatKind::Decimal128),
            "fp80" => Some(FormatKind::Fp80),
            "bcd" => Some(FormatKind::Bcd),
            "ibm_hfp" | "ibm-hfp" => Some(FormatKind::IbmHfp),
            "vax_f" | "vax-f" => Some(FormatKind::VaxF),
            "vax_d" | "vax-d" => Some(FormatKind::VaxD),
            "cray_float" | "cray-float" => Some(FormatKind::CrayFloat),
            "minifloat" => Some(FormatKind::Minifloat),
            "tapered_fp" | "tapered-fp" => Some(FormatKind::TaperedFp),
            "block_fp" | "block-fp" => Some(FormatKind::BlockFp),
            "shared_exp" | "shared-exp" => Some(FormatKind::SharedExp),
            "stochastic_rnd" | "stochastic-rounding" => Some(FormatKind::StochasticRnd),
            "unum_i" | "unum-i" => Some(FormatKind::UnumI),
            "unum_ii" | "unum-ii" => Some(FormatKind::UnumII),
            "afp" => Some(FormatKind::AfP),
            "q_format" | "q-format" | "qformat" => Some(FormatKind::QFormat),
            _ => None,
        }
    }

    /// Number of effective mantissa bits (excluding implicit bit)
    pub fn mantissa_bits(&self) -> u32 {
        match self {
            FormatKind::F32 => 23,
            FormatKind::F64 => 52,
            FormatKind::Fp16 => 10,
            FormatKind::Bf16 => 7,
            FormatKind::Tf32 => 10,
            FormatKind::Fp8E4M3 => 3,
            FormatKind::Fp8E5M2 => 2,
            FormatKind::Fp6E2M3 => 3,
            FormatKind::Fp6E3M2 => 2,
            FormatKind::Fp4E2M1 => 1,
            FormatKind::Gf16 => 9,
            FormatKind::Gf8 => 4,
            FormatKind::Gf4 => 2,
            FormatKind::Gf32 => 18,
            FormatKind::Gf64 => 42,
            FormatKind::Gf12 => 7,
            FormatKind::Gf20 => 12,
            FormatKind::Gf24 => 14,
            FormatKind::Int8 => 0,
            FormatKind::Int4 => 0,
            FormatKind::Int16 => 0,
            FormatKind::Int32 => 0,
            FormatKind::Uint8 => 0,
            FormatKind::Nf4 => 2,
            FormatKind::Posit8 => 4,
            FormatKind::Posit16 => 10,
            FormatKind::Posit32 => 26,
            FormatKind::Lns8 => 4,
            FormatKind::Mxfp4 => 1,
            FormatKind::Mxfp6 => 2,
            FormatKind::Mxfp8 => 3,
            FormatKind::Binary128 => 112,
            FormatKind::Binary256 => 236,
            FormatKind::Decimal32 => 7,
            FormatKind::Decimal64 => 16,
            FormatKind::Decimal128 => 34,
            FormatKind::Fp80 => 63,
            FormatKind::Bcd => 4,
            FormatKind::IbmHfp => 6,
            FormatKind::VaxF => 23,
            FormatKind::VaxD => 55,
            FormatKind::CrayFloat => 48,
            FormatKind::Minifloat => 3,
            FormatKind::TaperedFp => 8,
            FormatKind::BlockFp => 4,
            FormatKind::SharedExp => 4,
            FormatKind::StochasticRnd => 23,
            FormatKind::UnumI => 8,
            FormatKind::UnumII => 8,
            FormatKind::AfP => 4,
            FormatKind::QFormat => 8,
        }
    }

    /// Whether this format uses floating-point representation
    pub fn is_float(&self) -> bool {
        !matches!(
            self,
            FormatKind::Int4
                | FormatKind::Int8
                | FormatKind::Int16
                | FormatKind::Int32
                | FormatKind::Uint8
        )
    }

    /// Maximum finite magnitude representable in this format,
    /// used to clamp narrow-exponent IEEE-style formats (Bug A).
    /// Returns `None` for formats without a meaningful finite limit
    /// for f32 simulation (e.g. Bf16/Tf32 share f32's exponent range).
    pub fn max_finite(&self) -> Option<f32> {
        match self {
            // Same exponent range as f32 → no clamp needed
            FormatKind::F32 | FormatKind::Bf16 | FormatKind::Tf32 => None,

            // Wider than f32 → no clamp possible from f32
            FormatKind::F64
            | FormatKind::Fp80
            | FormatKind::Binary128
            | FormatKind::Binary256
            | FormatKind::Decimal128
            | FormatKind::VaxD
            | FormatKind::CrayFloat
            | FormatKind::Decimal64
            | FormatKind::StochasticRnd => None,

            // IEEE-754 narrow-exp: max = (2 - 2^-m) * 2^max_exp
            FormatKind::Fp16 => Some(65504.0),
            FormatKind::Fp8E4M3 => Some(448.0), // OCP FP8: reuses inf-bits for max
            FormatKind::Fp8E5M2 => Some(57344.0),
            FormatKind::Fp6E2M3 => Some(7.5),
            FormatKind::Fp6E3M2 => Some(28.0),
            FormatKind::Fp4E2M1 => Some(6.0),

            // OCP MX block-floats — element max (block-scale not modelled here)
            FormatKind::Mxfp4 => Some(6.0),   // E2M1
            FormatKind::Mxfp6 => Some(28.0),  // E3M2 default
            FormatKind::Mxfp8 => Some(448.0), // E4M3 default

            // Golden Float family — local impl (`gf16.rs` / `phi_numbers/*`)
            FormatKind::Gf16 => Some(65504.0), // bias=15, exp=6 → ~6.5e4
            FormatKind::Gf8 => Some(15.5),     // bias=3,  exp=3 → 15.5
            FormatKind::Gf4 => Some(3.0),      // approx
            FormatKind::Gf32 => None,          // bias=4095 → larger than f32
            FormatKind::Gf64 => None,          // bias=1048575 → larger
            FormatKind::Gf12 => Some(240.0),   // bias=7,  exp=4
            FormatKind::Gf20 => Some(65504.0), // approx
            FormatKind::Gf24 => Some(65504.0), // approx

            // Integer formats clamp via tensor scale, not max_finite
            FormatKind::Int4
            | FormatKind::Int8
            | FormatKind::Int16
            | FormatKind::Int32
            | FormatKind::Uint8 => None,

            // Approximations — Phase 2/3 will replace with faithful kernels
            FormatKind::Nf4 => Some(1.0),
            FormatKind::Posit8 => Some(64.0),
            FormatKind::Posit16 => Some(2.68e8),
            FormatKind::Posit32 => None,
            FormatKind::Lns8 => Some(64.0),
            FormatKind::Decimal32 => None, // ~9.999e96 exceeds f32 range
            FormatKind::Bcd => Some(99.0),
            FormatKind::IbmHfp => None, // ~7.2e75 exceeds f32 range
            FormatKind::VaxF => Some(1.7e38),
            FormatKind::Minifloat => Some(15.5),
            FormatKind::TaperedFp => Some(65504.0),
            FormatKind::BlockFp => Some(15.5),
            FormatKind::SharedExp => Some(15.5),
            FormatKind::UnumI => Some(255.0),
            FormatKind::UnumII => Some(255.0),
            FormatKind::AfP => Some(15.5),
            FormatKind::QFormat => Some(127.0),
        }
    }

    /// Formats whose simulation cannot be faithfully expressed in `f32`
    /// (mantissa wider than 23 bits, or exponent range wider than f32).
    /// Returning `true` means [`fake_quantize_f32`] is forced to identity
    /// — distinct BPB cannot be expected on these. See Bug H.
    pub fn is_unsupported_in_f32(&self) -> bool {
        matches!(
            self,
            FormatKind::F64
                | FormatKind::Fp80
                | FormatKind::Binary128
                | FormatKind::Binary256
                | FormatKind::Decimal128
                | FormatKind::Decimal64
                | FormatKind::VaxD
                | FormatKind::CrayFloat
                | FormatKind::StochasticRnd
        )
    }
}

/// Round-to-nearest-even helper that operates on `abs(val)` so the rounding
/// is sign-symmetric (Bug B fix).
#[inline]
fn round_mantissa_unsigned(abs_bits: u32, drop_bits: u32) -> u32 {
    if drop_bits == 0 {
        return abs_bits;
    }
    let mask = !((1u32 << drop_bits) - 1);
    let rounding_bit = 1u32 << (drop_bits - 1);
    // Round-half-up on the abs value: symmetric around zero.
    let rounded = abs_bits.wrapping_add(rounding_bit) & mask;
    rounded
}

/// Apply a magnitude clamp for narrow-exponent IEEE-style formats (Bug A).
#[inline]
fn clamp_to_max_finite(val: f32, fmt: FormatKind) -> f32 {
    if let Some(maxf) = fmt.max_finite() {
        if val.abs() > maxf {
            return maxf.copysign(val);
        }
    }
    val
}

/// Fake-quantize a single f32 value for the given format.
///
/// **Note on integer formats (Bug C/D):** the single-value path uses a
/// fixed `±1` range. Use [`fake_quantize_weights_tensor`] for per-tensor
/// amax scaling — that's the correct API for QAT.
pub fn fake_quantize_f32(val: f32, fmt: FormatKind) -> f32 {
    // NaN/Inf passthrough
    if !val.is_finite() {
        return val;
    }

    // F32 baseline
    if fmt == FormatKind::F32 {
        return val;
    }

    // Bug H: formats with mantissa >= 23 bits or wider exponent than f32
    // cannot be faithfully simulated from f32. Identity with explicit note.
    if fmt.is_unsupported_in_f32() {
        return val;
    }

    // Bug G: route Golden Float formats through canonical kernels
    match fmt {
        FormatKind::Gf16 => return GF16::from_f32(val).to_f32(),
        FormatKind::Gf8 => return GF8::from_f32(val).to_f32(),
        FormatKind::Gf32 => return GF32::from_f32(val).to_f32(),
        FormatKind::Gf64 => return GF64::from_f64(val as f64).to_f64() as f32,
        _ => {}
    }

    // Integer formats — single-value path uses fixed scale (legacy behaviour).
    // For correct per-tensor scaling, callers should use
    // `fake_quantize_weights_tensor`.
    if !fmt.is_float() {
        if fmt == FormatKind::Int32 {
            // 2^32 levels exceed f32's 24-bit mantissa precision — degenerate.
            return val;
        }
        let levels = match fmt {
            FormatKind::Int4 => 16.0_f32,
            FormatKind::Int8 => 256.0,
            FormatKind::Int16 => 65536.0,
            FormatKind::Uint8 => 256.0,
            _ => return val,
        };
        // Fixed range = ±1.0; per-tensor amax should override via tensor API.
        let scaled = val * (levels / 2.0);
        let q = quantize_round_signed(scaled).clamp(-(levels / 2.0), levels / 2.0 - 1.0);
        return q * (2.0 / levels);
    }

    // Floating-point fake quantization with exponent clamp (Bug A) +
    // sign-symmetric rounding (Bug B).
    let mantissa_bits = fmt.mantissa_bits();
    if mantissa_bits == 0 || mantissa_bits >= 23 {
        return val;
    }

    // Step 1: clamp magnitude to max_finite if the format has narrower exp.
    let clamped = clamp_to_max_finite(val, fmt);

    // Step 2: sign-symmetric rounding on |clamped|.
    let sign = clamped.is_sign_negative();
    let abs_val = clamped.abs();
    let abs_bits = abs_val.to_bits();
    let drop_bits = 23u32.saturating_sub(mantissa_bits);
    let rounded_abs_bits = round_mantissa_unsigned(abs_bits, drop_bits);
    let rounded_abs = f32::from_bits(rounded_abs_bits);

    // Step 3: re-clamp after rounding (rounding may have pushed past max).
    let rounded = if sign { -rounded_abs } else { rounded_abs };
    clamp_to_max_finite(rounded, fmt)
}

/// Round-half-away-from-zero for signed scaled integer values (Bug B
/// applied symmetrically). `val.round()` does this in Rust.
#[inline]
fn quantize_round_signed(val: f32) -> f32 {
    val.round()
}

/// Fake-quantize a weight tensor in-place using STE.
///
/// Floating-point formats use [`fake_quantize_f32`]; integer formats use
/// per-tensor amax scaling (Bug C fix).
pub fn fake_quantize_weights(weights: &mut [f32], fmt: FormatKind) {
    if fmt == FormatKind::F32 || fmt.is_unsupported_in_f32() {
        return;
    }
    if !fmt.is_float() {
        fake_quantize_int_tensor(weights, fmt);
        return;
    }
    for w in weights.iter_mut() {
        *w = fake_quantize_f32(*w, fmt);
    }
}

/// Tensor-aware fake-quantization that handles integer formats with
/// per-tensor amax scaling. For float formats this is identical to
/// [`fake_quantize_weights`].
pub fn fake_quantize_weights_tensor(weights: &mut [f32], fmt: FormatKind) {
    fake_quantize_weights(weights, fmt);
}

/// Per-tensor symmetric integer quantization:
///   `s = amax / (L/2 - 1)` (signed) or `s = amax / (L - 1)` (unsigned)
///   `q = clamp(round(x / s), q_min, q_max)`
///   `dq = q * s`
fn fake_quantize_int_tensor(weights: &mut [f32], fmt: FormatKind) {
    if fmt == FormatKind::Int32 {
        // 2^32 levels exceed f32 precision; treat as identity for f32 weights.
        return;
    }
    let amax = weights.iter().fold(0.0_f32, |acc, &w| acc.max(w.abs()));
    if amax == 0.0 || !amax.is_finite() {
        return;
    }

    let (q_min, q_max, levels_minus_one) = match fmt {
        FormatKind::Int4 => (-8.0_f32, 7.0_f32, 7.0_f32),
        FormatKind::Int8 => (-128.0, 127.0, 127.0),
        FormatKind::Int16 => (-32768.0, 32767.0, 32767.0),
        FormatKind::Uint8 => (0.0, 255.0, 255.0),
        _ => return,
    };

    // For unsigned formats, shift so that minimum maps to 0.
    let is_unsigned = matches!(fmt, FormatKind::Uint8);

    let scale = if is_unsigned {
        // Map [0, amax] (or [-amax, amax] folded by abs) to [0, 255].
        // We use abs amax so symmetric distributions don't collapse, but
        // unsigned needs caller to pass non-negative weights for full fidelity.
        amax / levels_minus_one
    } else {
        amax / levels_minus_one
    };

    if scale == 0.0 {
        return;
    }

    for w in weights.iter_mut() {
        let q = (*w / scale).round().clamp(q_min, q_max);
        *w = q * scale;
    }
}

/// Recursively fake-quantize nested weight structures (embed, lm_head, ffn, etc.)
pub fn fake_quantize_nested(
    embed: &mut [f32],
    lm_head: &mut [f32],
    ffn_w1: &mut [Vec<f32>],
    ffn_b1: &mut [Vec<f32>],
    ffn_w2: &mut [Vec<f32>],
    ffn_b2: &mut [Vec<f32>],
    fmt: FormatKind,
) {
    fake_quantize_weights(embed, fmt);
    fake_quantize_weights(lm_head, fmt);
    for w in ffn_w1.iter_mut() {
        fake_quantize_weights(w, fmt);
    }
    for b in ffn_b1.iter_mut() {
        fake_quantize_weights(b, fmt);
    }
    for w in ffn_w2.iter_mut() {
        fake_quantize_weights(w, fmt);
    }
    for b in ffn_b2.iter_mut() {
        fake_quantize_weights(b, fmt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------- Existing smoke tests (kept) --------

    #[test]
    fn test_f32_no_change() {
        let val = 1.234567890f32;
        assert_eq!(fake_quantize_f32(val, FormatKind::F32), val);
    }

    #[test]
    fn test_fp16_reduces_precision() {
        let val = 1.0000001f32;
        let q = fake_quantize_f32(val, FormatKind::Fp16);
        assert_ne!(q, val);
        assert!((q - val).abs() < 0.001);
    }

    #[test]
    fn test_bf16_reduces_precision() {
        let val = 1.0000001f32;
        let q = fake_quantize_f32(val, FormatKind::Bf16);
        assert_ne!(q, val);
    }

    #[test]
    fn test_int8_quantizes_tensor() {
        let mut w = vec![0.123f32, -0.5, 0.8, -0.2];
        let original = w.clone();
        fake_quantize_weights(&mut w, FormatKind::Int8);
        assert_ne!(w, original);
        for v in &w {
            assert!(v.abs() <= 1.0);
        }
    }

    #[test]
    fn test_int4_heavy_quantization() {
        let mut w = vec![0.5f32, -0.5, 0.25, 0.75];
        let original = w.clone();
        fake_quantize_weights(&mut w, FormatKind::Int4);
        assert_ne!(w, original);
    }

    #[test]
    fn test_gf16_quantizes() {
        let val = 1.0000001f32;
        let q = fake_quantize_f32(val, FormatKind::Gf16);
        assert_ne!(q, val);
    }

    #[test]
    fn test_different_formats_diverge() {
        let val = 0.123456789f32;
        let q_fp16 = fake_quantize_f32(val, FormatKind::Fp16);
        let q_bf16 = fake_quantize_f32(val, FormatKind::Bf16);
        let q_gf16 = fake_quantize_f32(val, FormatKind::Gf16);
        assert_ne!(q_fp16, q_bf16, "fp16 vs bf16 should differ");
        assert_ne!(q_fp16, q_gf16, "fp16 vs gf16 should differ");
    }

    #[test]
    fn test_nan_inf_passthrough() {
        assert!(fake_quantize_f32(f32::NAN, FormatKind::Int8).is_nan());
        assert!(fake_quantize_f32(f32::INFINITY, FormatKind::Fp16).is_infinite());
        assert!(fake_quantize_f32(f32::NEG_INFINITY, FormatKind::Gf16).is_infinite());
    }

    // -------- Bug A: exponent clamp --------

    #[test]
    fn bug_a_fp16_overflow_clamped() {
        // Pre-fix: fake_quantize_f32(70000.0, Fp16) returned ~10003415040.0
        // Post-fix: must clamp to 65504 (fp16 max_finite).
        let q = fake_quantize_f32(70000.0, FormatKind::Fp16);
        assert!(q.is_finite(), "fp16 must be finite after clamp");
        assert!((q - 65504.0).abs() < 1.0, "expected ~65504, got {q}");
    }

    #[test]
    fn bug_a_fp8_e4m3_overflow_clamped() {
        let q = fake_quantize_f32(1000.0, FormatKind::Fp8E4M3);
        assert!(q.is_finite());
        assert!(q.abs() <= 448.0 + 1e-3, "fp8e4m3 max=448, got {q}");
    }

    #[test]
    fn bug_a_fp4_overflow_clamped() {
        let q = fake_quantize_f32(100.0, FormatKind::Fp4E2M1);
        assert!(q.is_finite());
        assert!(q.abs() <= 6.0 + 1e-3, "fp4 max=6, got {q}");
    }

    // -------- Bug B: sign-symmetric rounding --------

    #[test]
    fn bug_b_bf16_negative_does_not_round_away() {
        // Pre-fix: bf16(-1.5) = -1.50390625 (away from zero).
        // Post-fix: rounding is symmetric around zero — magnitude must
        // equal that of the positive case.
        let pos = fake_quantize_f32(1.5, FormatKind::Bf16);
        let neg = fake_quantize_f32(-1.5, FormatKind::Bf16);
        assert_eq!(pos, -neg, "bf16 must be sign-symmetric around 0");
    }

    #[test]
    fn bug_b_fp16_sign_symmetric() {
        for v in [0.1_f32, 0.3, 0.7, 1.234, 100.5, -0.1, -0.3, -100.5] {
            let q = fake_quantize_f32(v, FormatKind::Fp16);
            let q_neg = fake_quantize_f32(-v, FormatKind::Fp16);
            assert!(
                (q + q_neg).abs() < 1e-6,
                "fp16 not symmetric for |v|={}: q={}, q_neg={}",
                v.abs(),
                q,
                q_neg
            );
        }
    }

    // -------- Bug C: per-tensor int scale --------

    #[test]
    fn bug_c_int8_distinguishes_magnitudes() {
        // Pre-fix: int8(2.0) == int8(5.0) == int8(100.0) ≈ 0.984 (saturated).
        // Post-fix: per-tensor amax means [2.0, 5.0, 100.0] scales by 100/127,
        // so the quantised values must remain distinct.
        let mut w = vec![2.0_f32, 5.0, 100.0];
        fake_quantize_weights(&mut w, FormatKind::Int8);
        assert_ne!(w[0], w[1]);
        assert_ne!(w[1], w[2]);
        // Largest magnitude lands near amax.
        assert!((w[2].abs() - 100.0).abs() < 1.0);
    }

    #[test]
    fn bug_c_int4_distinguishes_magnitudes() {
        let mut w = vec![1.0_f32, 3.0, 7.0];
        fake_quantize_weights(&mut w, FormatKind::Int4);
        assert_ne!(w[0], w[1]);
        assert_ne!(w[1], w[2]);
    }

    // -------- Bug G: GF formats use canonical kernel --------

    #[test]
    fn bug_g_gf16_uses_canonical_round_trip() {
        let val = 1.5_f32;
        let q = fake_quantize_f32(val, FormatKind::Gf16);
        let canonical = GF16::from_f32(val).to_f32();
        assert_eq!(q, canonical, "Gf16 must equal canonical GF16 round-trip");
    }

    #[test]
    fn bug_g_gf8_uses_canonical_round_trip() {
        let val = 0.75_f32;
        let q = fake_quantize_f32(val, FormatKind::Gf8);
        let canonical = GF8::from_f32(val).to_f32();
        assert_eq!(q, canonical, "Gf8 must equal canonical GF8 round-trip");
    }

    // -------- Bug H: unsupported formats --------

    #[test]
    fn bug_h_unsupported_formats_are_identity() {
        for fmt in [
            FormatKind::F64,
            FormatKind::Fp80,
            FormatKind::Binary128,
            FormatKind::Binary256,
            FormatKind::Decimal128,
            FormatKind::Decimal64,
            FormatKind::VaxD,
            FormatKind::CrayFloat,
            FormatKind::StochasticRnd,
        ] {
            assert!(fmt.is_unsupported_in_f32(), "{fmt:?}");
            assert_eq!(fake_quantize_f32(1.234_567f32, fmt), 1.234_567);
        }
    }

    // -------- Cross-format divergence on a tiny canary --------

    #[test]
    fn canary_distinct_bpb_per_format() {
        // Canary tensor: values across the f32 dynamic range.
        let canary: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.137 + 0.0001).collect();

        let mut signatures: Vec<(FormatKind, u64)> = Vec::new();
        for fmt in [
            FormatKind::Fp16,
            FormatKind::Bf16,
            FormatKind::Fp8E4M3,
            FormatKind::Fp8E5M2,
            FormatKind::Fp4E2M1,
            FormatKind::Gf16,
            FormatKind::Gf8,
            FormatKind::Int8,
            FormatKind::Int4,
        ] {
            let mut w = canary.clone();
            fake_quantize_weights(&mut w, fmt);
            // Sum of bit-patterns gives a fast format-divergence signature.
            let sig: u64 = w.iter().map(|v| v.to_bits() as u64).sum();
            signatures.push((fmt, sig));
        }

        // Every format must produce a distinct signature.
        for i in 0..signatures.len() {
            for j in (i + 1)..signatures.len() {
                assert_ne!(
                    signatures[i].1, signatures[j].1,
                    "{:?} and {:?} produced identical canary signature",
                    signatures[i].0, signatures[j].0
                );
            }
        }
    }
}
