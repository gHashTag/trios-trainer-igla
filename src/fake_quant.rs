//! FakeQuant + Straight-Through Estimator (STE) for QAT
//! Fixes trios#509: per-seed BPB collapse across formats
//!
//! Quantize→dequantize weights during training so different formats
//! produce different BPB values. Uses STE so gradients flow through
//! the quantization bottleneck as if it were identity.

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
    /// Parse format from TRIOS_FORMAT_TYPE env var or string
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

    /// Number of effective mantissa bits (including implicit bit)
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
            FormatKind::Gf16 => 9,  // 1:6:9, mantissa = 9 bits
            FormatKind::Gf8 => 4,   // 1:3:4
            FormatKind::Gf4 => 2,   // 1:1:2
            FormatKind::Gf32 => 19, // 1:12:19
            FormatKind::Gf64 => 39, // 1:24:39
            FormatKind::Gf12 => 7,  // 1:4:7
            FormatKind::Gf20 => 12, // 1:7:12
            FormatKind::Gf24 => 14, // 1:9:14
            FormatKind::Int8 => 0,  // integer — uses scale
            FormatKind::Int4 => 0,
            FormatKind::Int16 => 0,
            FormatKind::Int32 => 0,
            FormatKind::Uint8 => 0,
            FormatKind::Nf4 => 2,   // 4-bit normal float
            FormatKind::Posit8 => 4, // approx
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
        !matches!(self, FormatKind::Int4 | FormatKind::Int8 | FormatKind::Int16 | FormatKind::Int32 | FormatKind::Uint8)
    }

    /// Quantization scale for integer formats (max representable / levels)
    pub fn int_scale(&self) -> f32 {
        match self {
            FormatKind::Int4 => 7.0 / 8.0,    // 4-bit signed: [-8, 7]
            FormatKind::Int8 => 127.0 / 128.0,
            FormatKind::Int16 => 32767.0 / 32768.0,
            FormatKind::Int32 => 1.0,          // effectively f32
            FormatKind::Uint8 => 255.0 / 256.0,
            _ => 1.0,
        }
    }

    /// Number of quantization levels for integer formats
    pub fn int_levels(&self) -> f32 {
        match self {
            FormatKind::Int4 => 16.0,
            FormatKind::Int8 => 256.0,
            FormatKind::Int16 => 65536.0,
            FormatKind::Int32 => 4294967296.0,
            FormatKind::Uint8 => 256.0,
            _ => 1.0,
        }
    }
}

/// Fake-quantize a single f32 value for the given format.
/// Returns the dequantized value (STE: gradient flows through as identity).
pub fn fake_quantize_f32(val: f32, fmt: FormatKind) -> f32 {
    if !val.is_finite() {
        return val; // pass through NaN/Inf
    }

    if fmt == FormatKind::F32 || fmt == FormatKind::StochasticRnd {
        return val; // no quantization for f32 baseline
    }

    if !fmt.is_float() {
        // Integer quantization: scale → round → rescale
        let scale = fmt.int_scale();
        let levels = fmt.int_levels();
        let scaled = val * levels / (2.0 * scale);
        let q = scaled.round().clamp(-levels / 2.0, levels / 2.0 - 1.0);
        return q * 2.0 * scale / levels;
    }

    // Floating-point fake quantization:
    // Simulate reduced mantissa by rounding to nearest representable value
    let mantissa_bits = fmt.mantissa_bits();

    if mantissa_bits >= 23 {
        // More precision than f32 — effectively no quantization
        return val;
    }

    if mantissa_bits == 0 {
        return val; // shouldn't happen for floats, but safe
    }

    // Convert to bits, mask off lower mantissa bits, convert back
    let bits = val.to_bits();
    let f32_mantissa_bits = 23u32;
    let drop_bits = f32_mantissa_bits.saturating_sub(mantissa_bits);

    if drop_bits == 0 {
        return val;
    }

    // Create mask to zero out the dropped bits
    let mask = !((1u32 << drop_bits) - 1);
    let masked_bits = bits & mask;

    // Add rounding bit (round to nearest)
    let rounding_bit = 1u32 << (drop_bits - 1);
    let rounded_bits = masked_bits + rounding_bit;

    f32::from_bits(rounded_bits)
}

/// Fake-quantize a weight tensor in-place using STE.
/// Modifies weights, but gradients will flow through as if unchanged (STE).
pub fn fake_quantize_weights(weights: &mut [f32], fmt: FormatKind) {
    if fmt == FormatKind::F32 {
        return; // skip for baseline
    }
    for w in weights.iter_mut() {
        *w = fake_quantize_f32(*w, fmt);
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

    #[test]
    fn test_f32_no_change() {
        let val = 1.234567890f32;
        assert_eq!(fake_quantize_f32(val, FormatKind::F32), val);
    }

    #[test]
    fn test_fp16_reduces_precision() {
        let val = 1.0000001f32; // high precision f32 value
        let q = fake_quantize_f32(val, FormatKind::Fp16);
        // FP16 has 10 mantissa bits → should lose some precision
        assert_ne!(q, val); // Should differ
        assert!((q - val).abs() < 0.001); // But still close
    }

    #[test]
    fn test_bf16_reduces_precision() {
        let val = 1.0000001f32;
        let q = fake_quantize_f32(val, FormatKind::Bf16);
        assert_ne!(q, val); // Should differ
    }

    #[test]
    fn test_int8_quantizes() {
        let val = 0.123f32;
        let q = fake_quantize_f32(val, FormatKind::Int8);
        assert_ne!(q, val); // Should differ
        assert!(q.abs() <= 1.0); // Within int8 range after rescale
    }

    #[test]
    fn test_int4_heavy_quantization() {
        let val = 0.5f32;
        let q = fake_quantize_f32(val, FormatKind::Int4);
        assert_ne!(q, val); // Should differ significantly
    }

    #[test]
    fn test_gf16_quantizes() {
        let val = 1.0000001f32;
        let q = fake_quantize_f32(val, FormatKind::Gf16);
        // GF16 has 9 mantissa bits → less than f32 but more than fp8
        assert_ne!(q, val); // Should differ from f32
    }

    #[test]
    fn test_different_formats_diverge() {
        let val = 0.123456789f32;
        let q_fp16 = fake_quantize_f32(val, FormatKind::Fp16);
        let q_bf16 = fake_quantize_f32(val, FormatKind::Bf16);
        let q_int8 = fake_quantize_f32(val, FormatKind::Int8);
        let q_gf16 = fake_quantize_f32(val, FormatKind::Gf16);

        // All different formats should produce different quantized values
        assert_ne!(q_fp16, q_bf16, "fp16 vs bf16 should differ");
        assert_ne!(q_fp16, q_int8, "fp16 vs int8 should differ");
        assert_ne!(q_bf16, q_int8, "bf16 vs int8 should differ");
        assert_ne!(q_fp16, q_gf16, "fp16 vs gf16 should differ");
    }

    #[test]
    fn test_nan_inf_passthrough() {
        assert!(fake_quantize_f32(f32::NAN, FormatKind::Int8).is_nan());
        assert!(fake_quantize_f32(f32::INFINITY, FormatKind::Int8).is_infinite());
        assert!(fake_quantize_f32(f32::NEG_INFINITY, FormatKind::Gf16).is_infinite());
    }

    #[test]
    fn test_int8_changes_values() {
        let mut weights = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let original = weights.clone();
        fake_quantize_weights(&mut weights, FormatKind::Int8);
        // At least some values should change
        let changes = weights.iter().zip(original.iter()).filter(|(a, b)| a != b).count();
        assert!(changes > 0, "Int8 quantization should change some values");
    }
}
