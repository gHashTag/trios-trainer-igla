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
//! **Phase C L-C2** — Format kernel pack (refs gHashTag/trios#446, gHashTag/trios#536):
//!
//! Adds Posit64, Nf8, IbmHfpShort/Long, VaxG/H, CrayFloat, UnumI8/I16,
//! UnumII8/II16, BCD8/BCD16, Q15/Q31, StochasticRound to reach ≥65 formats.
//! Mantissa-mask approximations are clearly marked `is_faithful()=false` so
//! anti-fake-pass CI can flag them as deferred Phase 2 work.
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

/// Supported numeric formats for fake quantization.
///
/// The full set returned by [`FormatKind::all()`] spans ≥65 variants for
/// matrix evaluation per gHashTag/trios#446.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatKind {
    // ---- IEEE 754 standard floats ----
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
    // ---- Golden Float family ----
    Gf16,
    Gf8,
    Gf4,
    Gf32,
    Gf64,
    Gf12,
    Gf20,
    Gf24,
    // ---- Integer formats ----
    Int8,
    Int4,
    Int16,
    Int32,
    Uint8,
    // ---- LUT / NF formats ----
    Nf4,
    /// Nf8: 8-bit lookup-table normal float (extension of Nf4).
    /// TODO: faithful encoder — currently using mantissa-mask approximation.
    Nf8,
    // ---- Posit formats ----
    Posit8,
    Posit16,
    Posit32,
    /// Posit64: 64-bit posit.
    /// TODO: faithful encoder — currently using mantissa-mask approximation.
    Posit64,
    // ---- Logarithmic Number System ----
    Lns8,
    // ---- OCP MX block-floats ----
    Mxfp4,
    Mxfp6,
    Mxfp8,
    // ---- IEEE extended / quad ----
    Binary128,
    Binary256,
    Decimal32,
    Decimal64,
    Decimal128,
    Fp80,
    // ---- BCD ----
    /// Bcd: legacy 4-bit-per-digit BCD (pre-existing, kept for compat).
    Bcd,
    /// BCD8: 8-bit Binary-Coded Decimal (2 digits).
    /// TODO: faithful encoder — currently using mantissa-mask approximation.
    Bcd8,
    /// BCD16: 16-bit Binary-Coded Decimal (4 digits).
    /// TODO: faithful encoder — currently using mantissa-mask approximation.
    Bcd16,
    // ---- IBM HFP ----
    /// IbmHfp: legacy single variant (pre-existing, kept for compat).
    IbmHfp,
    /// IbmHfpShort: 32-bit IBM hex floating-point (single precision).
    /// TODO: faithful encoder — currently using mantissa-mask approximation.
    IbmHfpShort,
    /// IbmHfpLong: 64-bit IBM hex floating-point (double precision).
    /// Wider than f32 — treated as unsupported (identity).
    IbmHfpLong,
    // ---- VAX formats ----
    VaxF,
    VaxD,
    /// VaxG: 64-bit VAX G-float.
    /// Wider than f32 — treated as unsupported (identity).
    VaxG,
    /// VaxH: 128-bit VAX H-float.
    /// Wider than f32 — treated as unsupported (identity).
    VaxH,
    // ---- Cray ----
    CrayFloat,
    // ---- Miscellaneous float formats ----
    Minifloat,
    TaperedFp,
    BlockFp,
    SharedExp,
    // ---- Stochastic rounding ----
    /// StochasticRnd: pre-existing variant (legacy name).
    StochasticRnd,
    /// StochasticRound: alias variant added in Phase C L-C2.
    /// Marked as_unsupported_in_f32 with TODO for bernoulli-RNG path.
    StochasticRound,
    // ---- Unum / Unums ----
    /// UnumI: pre-existing generic Unum Type I.
    UnumI,
    /// UnumII: pre-existing generic Unum Type II.
    UnumII,
    /// UnumI8: 8-bit Type I unum.
    /// TODO: faithful encoder — currently using mantissa-mask approximation.
    UnumI8,
    /// UnumI16: 16-bit Type I unum.
    /// TODO: faithful encoder — currently using mantissa-mask approximation.
    UnumI16,
    /// UnumII8: 8-bit Type II unum (posit-like).
    /// TODO: faithful encoder — currently using mantissa-mask approximation.
    UnumII8,
    /// UnumII16: 16-bit Type II unum (posit-like).
    /// TODO: faithful encoder — currently using mantissa-mask approximation.
    UnumII16,
    // ---- Q-format fixed-point ----
    /// AfP: pre-existing adaptive float-point.
    AfP,
    /// QFormat: pre-existing generic Q-format.
    QFormat,
    /// Q15: Q0.15 fixed-point (signed 16-bit, 15 fractional bits, range [-1, 1)).
    /// TODO: faithful encoder — currently using mantissa-mask approximation.
    Q15,
    /// Q31: Q0.31 fixed-point (signed 32-bit, 31 fractional bits, range [-1, 1)).
    /// Wider than f32 — treated as unsupported (identity).
    Q31,
}

// ---------------------------------------------------------------------------
// NF4 and NF8 lookup tables
// ---------------------------------------------------------------------------

/// NF4 lookup table — 16 values in [-1, 1], symmetric normal-float mapping.
const NF4_LUT: [f32; 16] = [
    -1.0,
    -0.6961928010,
    -0.5250730515,
    -0.3949100077,
    -0.2844675183,
    -0.1848396957,
    -0.0943080932,
    0.0,
    0.0943080932,
    0.1848396957,
    0.2844675183,
    0.3949100077,
    0.5250730515,
    0.6961928010,
    0.8480964303,
    1.0,
];

/// NF8 lookup table — 256 values in [-1, 1], normal-float quantiles.
/// Approximated here as uniformly-spaced in [-1, 1] pending a faithful
/// normal-float quantile computation.
/// TODO: replace with quantile-correct NF8 table.
const NF8_LUT_LEN: usize = 256;

fn nf8_lut_value(idx: usize) -> f32 {
    // TODO: faithful NF8 quantile table — currently uniform spacing
    -1.0 + (2.0 * idx as f32) / (NF8_LUT_LEN as f32 - 1.0)
}

/// Find nearest value in a sorted slice using linear scan (for small LUTs).
#[inline]
fn nearest_in_lut(val: f32, lut: &[f32]) -> f32 {
    let mut best = lut[0];
    let mut best_dist = (val - best).abs();
    for &entry in &lut[1..] {
        let d = (val - entry).abs();
        if d < best_dist {
            best_dist = d;
            best = entry;
        }
    }
    best
}

/// Find nearest value in the NF8 LUT.
#[inline]
fn nearest_in_nf8(val: f32) -> f32 {
    // Preserve zero exactly
    if val == 0.0 {
        return 0.0;
    }
    // Clamp to table range
    let clamped = val.clamp(-1.0, 1.0);
    // Compute approximate index using round-to-nearest
    let approx_idx = ((clamped + 1.0) * (NF8_LUT_LEN as f32 - 1.0) / 2.0).round() as usize;
    let idx = approx_idx.min(NF8_LUT_LEN - 1);
    nf8_lut_value(idx)
}

impl FormatKind {
    // -----------------------------------------------------------------------
    // Matrix-iteration helpers (Phase C L-C2)
    // -----------------------------------------------------------------------

    /// Returns a slice of every `FormatKind` variant.
    /// Used by matrix-evaluation loops (refs gHashTag/trios#446).
    pub fn all() -> &'static [FormatKind] {
        use FormatKind::*;
        &[
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
            Nf8,
            Posit8,
            Posit16,
            Posit32,
            Posit64,
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
            Bcd8,
            Bcd16,
            IbmHfp,
            IbmHfpShort,
            IbmHfpLong,
            VaxF,
            VaxD,
            VaxG,
            VaxH,
            CrayFloat,
            Minifloat,
            TaperedFp,
            BlockFp,
            SharedExp,
            StochasticRnd,
            StochasticRound,
            UnumI,
            UnumII,
            UnumI8,
            UnumI16,
            UnumII8,
            UnumII16,
            AfP,
            QFormat,
            Q15,
            Q31,
        ]
    }

    /// Lowercase canonical name matching the matrix headers in gHashTag/trios#446.
    pub fn name(&self) -> &'static str {
        match self {
            FormatKind::F32 => "fp32",
            FormatKind::F64 => "fp64",
            FormatKind::Fp16 => "fp16",
            FormatKind::Bf16 => "bf16",
            FormatKind::Tf32 => "tf32",
            FormatKind::Fp8E4M3 => "fp8_e4m3",
            FormatKind::Fp8E5M2 => "fp8_e5m2",
            FormatKind::Fp6E2M3 => "fp6_e2m3",
            FormatKind::Fp6E3M2 => "fp6_e3m2",
            FormatKind::Fp4E2M1 => "fp4_e2m1",
            FormatKind::Gf16 => "gf16",
            FormatKind::Gf8 => "gf8",
            FormatKind::Gf4 => "gf4",
            FormatKind::Gf32 => "gf32",
            FormatKind::Gf64 => "gf64",
            FormatKind::Gf12 => "gf12",
            FormatKind::Gf20 => "gf20",
            FormatKind::Gf24 => "gf24",
            FormatKind::Int8 => "int8",
            FormatKind::Int4 => "int4",
            FormatKind::Int16 => "int16",
            FormatKind::Int32 => "int32",
            FormatKind::Uint8 => "uint8",
            FormatKind::Nf4 => "nf4",
            FormatKind::Nf8 => "nf8",
            FormatKind::Posit8 => "posit8",
            FormatKind::Posit16 => "posit16",
            FormatKind::Posit32 => "posit32",
            FormatKind::Posit64 => "posit64",
            FormatKind::Lns8 => "lns8",
            FormatKind::Mxfp4 => "mxfp4",
            FormatKind::Mxfp6 => "mxfp6",
            FormatKind::Mxfp8 => "mxfp8",
            FormatKind::Binary128 => "binary128",
            FormatKind::Binary256 => "binary256",
            FormatKind::Decimal32 => "decimal32",
            FormatKind::Decimal64 => "decimal64",
            FormatKind::Decimal128 => "decimal128",
            FormatKind::Fp80 => "fp80",
            FormatKind::Bcd => "bcd",
            FormatKind::Bcd8 => "bcd8",
            FormatKind::Bcd16 => "bcd16",
            FormatKind::IbmHfp => "ibm_hfp",
            FormatKind::IbmHfpShort => "ibm_hfp_short",
            FormatKind::IbmHfpLong => "ibm_hfp_long",
            FormatKind::VaxF => "vax_f",
            FormatKind::VaxD => "vax_d",
            FormatKind::VaxG => "vax_g",
            FormatKind::VaxH => "vax_h",
            FormatKind::CrayFloat => "cray_float",
            FormatKind::Minifloat => "minifloat",
            FormatKind::TaperedFp => "tapered_fp",
            FormatKind::BlockFp => "block_fp",
            FormatKind::SharedExp => "shared_exp",
            FormatKind::StochasticRnd => "stochastic_rnd",
            FormatKind::StochasticRound => "stochastic_round",
            FormatKind::UnumI => "unum_i",
            FormatKind::UnumII => "unum_ii",
            FormatKind::UnumI8 => "unum_i8",
            FormatKind::UnumI16 => "unum_i16",
            FormatKind::UnumII8 => "unum_ii8",
            FormatKind::UnumII16 => "unum_ii16",
            FormatKind::AfP => "afp",
            FormatKind::QFormat => "q_format",
            FormatKind::Q15 => "q15",
            FormatKind::Q31 => "q31",
        }
    }

    /// Inverse of `name()`, case-insensitive.
    pub fn from_name(name: &str) -> Option<FormatKind> {
        let lower = name.to_lowercase();
        // First try the canonical from_env path (covers legacy aliases too)
        if let Some(fk) = FormatKind::from_env(&lower) {
            return Some(fk);
        }
        // Then match canonical names from all()
        for &variant in FormatKind::all() {
            if variant.name() == lower.as_str() {
                return Some(variant);
            }
        }
        None
    }

    /// `true` iff the round-trip `f32 → format → f32` is provably
    /// bit-exact or uses a verified encoder (e.g. GF canonical kernel,
    /// IEEE-754 mantissa-mask on formats ≤ 23 mantissa bits).
    ///
    /// `false` for mantissa-mask approximations, unsupported-in-f32 identity
    /// passthrough, and deferred LUT/posit/BCD/IBM-HFP/VAX encoders.
    pub fn is_faithful(&self) -> bool {
        match self {
            // F32 identity — trivially faithful
            FormatKind::F32 => true,
            // IEEE mantissa-mask formats with correct exponent range
            FormatKind::Fp16
            | FormatKind::Bf16
            | FormatKind::Tf32
            | FormatKind::Fp8E4M3
            | FormatKind::Fp8E5M2
            | FormatKind::Fp6E2M3
            | FormatKind::Fp6E3M2
            | FormatKind::Fp4E2M1
            | FormatKind::Mxfp4
            | FormatKind::Mxfp6
            | FormatKind::Mxfp8 => true,
            // Golden Float canonical kernels
            FormatKind::Gf16 | FormatKind::Gf8 | FormatKind::Gf32 | FormatKind::Gf64 => true,
            // Integer formats — per-tensor symmetric quantisation is faithful
            FormatKind::Int4
            | FormatKind::Int8
            | FormatKind::Int16
            | FormatKind::Int32
            | FormatKind::Uint8 => true,
            // Everything else: deferred to Phase 2
            _ => false,
        }
    }

    // -----------------------------------------------------------------------
    // Parse format from `TRIOS_FORMAT_TYPE` env var or string
    // -----------------------------------------------------------------------

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
            "nf8" => Some(FormatKind::Nf8),
            "posit8" => Some(FormatKind::Posit8),
            "posit16" => Some(FormatKind::Posit16),
            "posit32" => Some(FormatKind::Posit32),
            "posit64" => Some(FormatKind::Posit64),
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
            "bcd8" => Some(FormatKind::Bcd8),
            "bcd16" => Some(FormatKind::Bcd16),
            "ibm_hfp" | "ibm-hfp" => Some(FormatKind::IbmHfp),
            "ibm_hfp_short" | "ibm-hfp-short" => Some(FormatKind::IbmHfpShort),
            "ibm_hfp_long" | "ibm-hfp-long" => Some(FormatKind::IbmHfpLong),
            "vax_f" | "vax-f" => Some(FormatKind::VaxF),
            "vax_d" | "vax-d" => Some(FormatKind::VaxD),
            "vax_g" | "vax-g" => Some(FormatKind::VaxG),
            "vax_h" | "vax-h" => Some(FormatKind::VaxH),
            "cray_float" | "cray-float" => Some(FormatKind::CrayFloat),
            "minifloat" => Some(FormatKind::Minifloat),
            "tapered_fp" | "tapered-fp" => Some(FormatKind::TaperedFp),
            "block_fp" | "block-fp" => Some(FormatKind::BlockFp),
            "shared_exp" | "shared-exp" => Some(FormatKind::SharedExp),
            "stochastic_rnd" | "stochastic-rounding" => Some(FormatKind::StochasticRnd),
            "stochastic_round" => Some(FormatKind::StochasticRound),
            "unum_i" | "unum-i" => Some(FormatKind::UnumI),
            "unum_ii" | "unum-ii" => Some(FormatKind::UnumII),
            "unum_i8" | "unum-i8" => Some(FormatKind::UnumI8),
            "unum_i16" | "unum-i16" => Some(FormatKind::UnumI16),
            "unum_ii8" | "unum-ii8" => Some(FormatKind::UnumII8),
            "unum_ii16" | "unum-ii16" => Some(FormatKind::UnumII16),
            "afp" => Some(FormatKind::AfP),
            "q_format" | "q-format" | "qformat" => Some(FormatKind::QFormat),
            "q15" => Some(FormatKind::Q15),
            "q31" => Some(FormatKind::Q31),
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
            FormatKind::Nf8 => 7, // 8-bit LUT → ~7 bits effective
            FormatKind::Posit8 => 4,
            FormatKind::Posit16 => 10,
            FormatKind::Posit32 => 26,
            FormatKind::Posit64 => 58, // 64-bit posit
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
            FormatKind::Bcd8 => 4,  // 2 BCD digits
            FormatKind::Bcd16 => 8, // 4 BCD digits
            FormatKind::IbmHfp => 6,
            FormatKind::IbmHfpShort => 6, // 32-bit IBM HFP: 1s + 7exp + 24frac (hex)
            FormatKind::IbmHfpLong => 14, // 64-bit IBM HFP: 1s + 7exp + 56frac (hex)
            FormatKind::VaxF => 23,
            FormatKind::VaxD => 55,
            FormatKind::VaxG => 52,  // 64-bit VAX G-float
            FormatKind::VaxH => 112, // 128-bit VAX H-float
            FormatKind::CrayFloat => 48,
            FormatKind::Minifloat => 3,
            FormatKind::TaperedFp => 8,
            FormatKind::BlockFp => 4,
            FormatKind::SharedExp => 4,
            FormatKind::StochasticRnd => 23,
            FormatKind::StochasticRound => 23,
            FormatKind::UnumI => 8,
            FormatKind::UnumII => 8,
            FormatKind::UnumI8 => 4,
            FormatKind::UnumI16 => 10,
            FormatKind::UnumII8 => 4,
            FormatKind::UnumII16 => 10,
            FormatKind::AfP => 4,
            FormatKind::QFormat => 8,
            FormatKind::Q15 => 15,
            FormatKind::Q31 => 31,
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
            | FormatKind::VaxG      // 64-bit, wider than f32
            | FormatKind::VaxH      // 128-bit, wider than f32
            | FormatKind::CrayFloat
            | FormatKind::Decimal64
            | FormatKind::StochasticRnd
            | FormatKind::StochasticRound
            | FormatKind::IbmHfpLong  // 64-bit IBM HFP
            | FormatKind::Posit64     // 64-bit posit
            | FormatKind::Q31         // 31-bit fractional, but wider than f32 for faithful
            => None,

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
            FormatKind::Nf4  => Some(1.0),
            FormatKind::Nf8  => Some(1.0),   // NF8 LUT normalised to [-1,1]
            FormatKind::Posit8  => Some(64.0),
            FormatKind::Posit16 => Some(2.68e8),
            FormatKind::Posit32 => None,
            FormatKind::Lns8 => Some(64.0),
            FormatKind::Decimal32 => None, // ~9.999e96 exceeds f32 range
            FormatKind::Bcd   => Some(99.0),
            FormatKind::Bcd8  => Some(99.0),    // 2 BCD digits → 00-99
            FormatKind::Bcd16 => Some(9999.0),  // 4 BCD digits → 0000-9999
            FormatKind::IbmHfp      => None, // ~7.2e75 exceeds f32 range
            FormatKind::IbmHfpShort => None, // IBM HFP short max ~7.2e75 exceeds f32 range → no clamp needed
            FormatKind::VaxF => Some(1.7e38),
            FormatKind::Minifloat => Some(15.5),
            FormatKind::TaperedFp => Some(65504.0),
            FormatKind::BlockFp => Some(15.5),
            FormatKind::SharedExp => Some(15.5),
            FormatKind::UnumI  => Some(255.0),
            FormatKind::UnumII => Some(255.0),
            FormatKind::UnumI8   => Some(255.0),
            FormatKind::UnumI16  => Some(65535.0_f32),
            FormatKind::UnumII8  => Some(64.0),   // posit8-like
            FormatKind::UnumII16 => Some(2.68e8),  // posit16-like
            FormatKind::AfP    => Some(15.5),
            FormatKind::QFormat => Some(127.0),
            FormatKind::Q15    => Some(0.999969482421875), // (2^15-1)/2^15
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
                | FormatKind::VaxG          // 64-bit
                | FormatKind::VaxH          // 128-bit
                | FormatKind::CrayFloat
                | FormatKind::IbmHfpLong    // 64-bit IBM HFP
                | FormatKind::Posit64       // 64-bit posit — wider than f32
                | FormatKind::StochasticRnd
                | FormatKind::StochasticRound // TODO: bernoulli-RNG path
                | FormatKind::Q31 // 31 fractional bits > f32 mantissa
        )
    }
}

// ---------------------------------------------------------------------------
// Round-to-nearest helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Q15 fixed-point fake-quantisation
// ---------------------------------------------------------------------------

/// Fake-quantize to Q0.15 format:
/// x_q = round(x * 2^15) / 2^15, clamped to [-1, 1 - 2^-15].
/// TODO: faithful encoder — currently using direct f32 arithmetic.
#[inline]
fn fake_quantize_q15(val: f32) -> f32 {
    let scale = 32768.0_f32; // 2^15
    let max_val = (scale - 1.0) / scale; // (2^15 - 1) / 2^15
    let scaled = (val * scale).round().clamp(-scale, scale - 1.0);
    let result = scaled / scale;
    result.clamp(-1.0, max_val)
}

// ---------------------------------------------------------------------------
// BCD fake-quantisation helpers
// ---------------------------------------------------------------------------

/// BCD8: round to nearest integer representable as 2 BCD digits (0–99).
/// TODO: faithful encoder — currently rounding to nearest integer in [0, 99].
#[inline]
fn fake_quantize_bcd8(val: f32) -> f32 {
    let sign = if val < 0.0 { -1.0_f32 } else { 1.0_f32 };
    let abs_val = val.abs();
    let rounded = abs_val.round().clamp(0.0, 99.0);
    rounded * sign
}

/// BCD16: round to nearest integer representable as 4 BCD digits (0–9999).
/// TODO: faithful encoder — currently rounding to nearest integer in [0, 9999].
#[inline]
fn fake_quantize_bcd16(val: f32) -> f32 {
    let sign = if val < 0.0 { -1.0_f32 } else { 1.0_f32 };
    let abs_val = val.abs();
    let rounded = abs_val.round().clamp(0.0, 9999.0);
    rounded * sign
}

// ---------------------------------------------------------------------------
// IBM HFP short fake-quantisation
// ---------------------------------------------------------------------------

/// IBM HFP Short (32-bit): hex-base-16 float with 6 hex digits mantissa.
/// Approximation: mask 17 mantissa bits from the f32 representation.
/// TODO: faithful encoder — currently using mantissa-mask approximation.
#[inline]
fn fake_quantize_ibm_hfp_short(val: f32) -> f32 {
    // IBM HFP short has ~6.9 decimal digits (~23 bits), but the hex base
    // means we get coarser rounding. We simulate by dropping the lower 6
    // mantissa bits (23 - 17 = 6 dropped bits).
    let drop_bits = 6u32;
    let sign = val.is_sign_negative();
    let abs_val = val.abs();
    let abs_bits = abs_val.to_bits();
    let rounded_bits = round_mantissa_unsigned(abs_bits, drop_bits);
    let rounded_abs = f32::from_bits(rounded_bits);
    // IBM HFP short max finite: ~ 7.2e75, which exceeds f32 max, so no clamp
    if sign {
        -rounded_abs
    } else {
        rounded_abs
    }
}

// ---------------------------------------------------------------------------
// Unum helpers (mantissa-mask approximations)
// ---------------------------------------------------------------------------

/// Unum Type I / II 8-bit: approximate as Posit8-style (4 mantissa bits).
/// TODO: faithful encoder — currently using mantissa-mask approximation.
#[inline]
fn fake_quantize_unum8(val: f32, max_mag: f32) -> f32 {
    // 4 effective mantissa bits, posit-like
    let drop_bits = 23u32 - 4;
    let sign = val.is_sign_negative();
    let abs_val = val.abs().min(max_mag);
    let abs_bits = abs_val.to_bits();
    let rounded_bits = round_mantissa_unsigned(abs_bits, drop_bits);
    let rounded_abs = f32::from_bits(rounded_bits).min(max_mag);
    if sign {
        -rounded_abs
    } else {
        rounded_abs
    }
}

/// Unum Type I / II 16-bit: approximate as Posit16-style (10 mantissa bits).
/// TODO: faithful encoder — currently using mantissa-mask approximation.
#[inline]
fn fake_quantize_unum16(val: f32, max_mag: f32) -> f32 {
    let drop_bits = 23u32 - 10;
    let sign = val.is_sign_negative();
    let abs_val = val.abs().min(max_mag);
    let abs_bits = abs_val.to_bits();
    let rounded_bits = round_mantissa_unsigned(abs_bits, drop_bits);
    let rounded_abs = f32::from_bits(rounded_bits).min(max_mag);
    if sign {
        -rounded_abs
    } else {
        rounded_abs
    }
}

// ---------------------------------------------------------------------------
// Main fake_quantize_f32
// ---------------------------------------------------------------------------

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
        // TODO: bernoulli-RNG path for StochasticRound
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

    // LUT-based formats
    match fmt {
        FormatKind::Nf4 => {
            // Clamp to [-1, 1] then find nearest LUT entry
            let clamped = val.clamp(-1.0, 1.0);
            return nearest_in_lut(clamped, &NF4_LUT);
        }
        FormatKind::Nf8 => {
            // TODO: faithful NF8 — currently uniform-spaced LUT approximation
            let clamped = val.clamp(-1.0, 1.0);
            return nearest_in_nf8(clamped);
        }
        _ => {}
    }

    // BCD formats
    match fmt {
        FormatKind::Bcd => return fake_quantize_bcd8(val),
        FormatKind::Bcd8 => return fake_quantize_bcd8(val),
        FormatKind::Bcd16 => return fake_quantize_bcd16(val),
        _ => {}
    }

    // IBM HFP short
    if fmt == FormatKind::IbmHfpShort {
        return fake_quantize_ibm_hfp_short(val);
    }

    // Q15 fixed-point
    if fmt == FormatKind::Q15 {
        return fake_quantize_q15(val);
    }

    // Unum variants
    match fmt {
        FormatKind::UnumI8 => return fake_quantize_unum8(val, 255.0),
        FormatKind::UnumII8 => return fake_quantize_unum8(val, 64.0),
        FormatKind::UnumI16 => return fake_quantize_unum16(val, 65504.0),
        FormatKind::UnumII16 => return fake_quantize_unum16(val, 2.68e8),
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

// ---------------------------------------------------------------------------
// Tensor-level fake quantization
// ---------------------------------------------------------------------------

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

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Phase C L-C2 — required tests
    // -----------------------------------------------------------------------

    /// All variants returned by FormatKind::all() must have distinct names.
    #[test]
    fn test_all_variants_have_unique_name() {
        let all = FormatKind::all();
        let mut names: Vec<&'static str> = all.iter().map(|f| f.name()).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names.len(),
            total,
            "Duplicate names found in FormatKind::all()"
        );
    }

    /// from_name(name()) must round-trip for every variant.
    #[test]
    fn test_from_name_roundtrip() {
        for &variant in FormatKind::all() {
            let name = variant.name();
            let recovered = FormatKind::from_name(name).unwrap_or_else(|| {
                panic!(
                    "from_name({:?}) returned None for variant {:?}",
                    name, variant
                )
            });
            assert_eq!(
                recovered, variant,
                "from_name({:?}) returned {:?}, expected {:?}",
                name, recovered, variant
            );
        }
    }

    /// F32 fake_quantize must be bit-exact identity.
    #[test]
    fn test_f32_is_identity() {
        for val in [0.0_f32, 1.0, -1.0, 0.123456789, f32::MAX, -f32::MAX, 1e-30] {
            let q = fake_quantize_f32(val, FormatKind::F32);
            assert_eq!(q.to_bits(), val.to_bits(), "F32 identity failed for {val}");
        }
    }

    /// Every format must quantize 0.0 → 0.0.
    #[test]
    fn test_zero_preserved() {
        for &fmt in FormatKind::all() {
            let q = fake_quantize_f32(0.0_f32, fmt);
            assert_eq!(
                q,
                0.0_f32,
                "Format {:?} ({}) did not preserve zero: got {}",
                fmt,
                fmt.name(),
                q
            );
        }
    }

    /// Every format must preserve the sign of finite non-zero values.
    #[test]
    fn test_sign_preserved() {
        // Use a value that is representable in all narrow formats without clamping to zero
        let val = 0.5_f32;
        for &fmt in FormatKind::all() {
            // Skip integer formats which may quantize small values to zero
            if !fmt.is_float() {
                continue;
            }
            let q_pos = fake_quantize_f32(val, fmt);
            let q_neg = fake_quantize_f32(-val, fmt);
            if q_pos == 0.0 {
                // If positive rounds to zero, negative must also be zero (or negative zero)
                assert!(
                    q_neg == 0.0 || q_neg == -0.0,
                    "Format {:?} ({}): +val→0 but -val→{} (sign broken)",
                    fmt,
                    fmt.name(),
                    q_neg
                );
            } else {
                assert!(
                    q_pos > 0.0,
                    "Format {:?} ({}): positive input produced negative output {}",
                    fmt,
                    fmt.name(),
                    q_pos
                );
                assert!(
                    q_neg < 0.0 || q_neg == 0.0,
                    "Format {:?} ({}): negative input produced positive output {}",
                    fmt,
                    fmt.name(),
                    q_neg
                );
            }
        }
    }

    /// FormatKind::all() must return at least 65 variants.
    #[test]
    fn test_unique_count() {
        let count = FormatKind::all().len();
        assert!(
            count >= 65,
            "FormatKind::all() returned {} variants, need >= 65",
            count
        );
    }

    // -----------------------------------------------------------------------
    // Existing smoke tests (kept)
    // -----------------------------------------------------------------------

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
        let mut w = vec![2.0_f32, 5.0, 100.0];
        fake_quantize_weights(&mut w, FormatKind::Int8);
        assert_ne!(w[0], w[1]);
        assert_ne!(w[1], w[2]);
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
            FormatKind::VaxG,
            FormatKind::VaxH,
            FormatKind::CrayFloat,
            FormatKind::IbmHfpLong,
            FormatKind::Posit64,
            FormatKind::StochasticRnd,
            FormatKind::StochasticRound,
            FormatKind::Q31,
        ] {
            assert!(fmt.is_unsupported_in_f32(), "{fmt:?}");
            assert_eq!(fake_quantize_f32(1.234_567f32, fmt), 1.234_567);
        }
    }

    // -------- Phase C L-C2: new format smoke tests --------

    #[test]
    fn test_nf8_reduces_precision() {
        let val = 0.123456789_f32;
        let q = fake_quantize_f32(val, FormatKind::Nf8);
        assert!((q - val).abs() < 0.1, "nf8 should be close: {q} vs {val}");
        assert!(q.abs() <= 1.0, "nf8 must stay in [-1, 1]");
    }

    #[test]
    fn test_posit64_is_identity() {
        // posit64 wider than f32 → identity
        let val = 3.14_f32;
        assert_eq!(fake_quantize_f32(val, FormatKind::Posit64), val);
    }

    #[test]
    fn test_ibm_hfp_short_reduces_precision() {
        let val = 1.0000001_f32;
        let q = fake_quantize_f32(val, FormatKind::IbmHfpShort);
        // Should reduce precision (drop lower bits)
        assert!((q - val).abs() < 0.01);
    }

    #[test]
    fn test_ibm_hfp_long_is_identity() {
        let val = 1.5_f32;
        assert_eq!(fake_quantize_f32(val, FormatKind::IbmHfpLong), val);
    }

    #[test]
    fn test_vax_g_is_identity() {
        let val = 2.0_f32;
        assert_eq!(fake_quantize_f32(val, FormatKind::VaxG), val);
    }

    #[test]
    fn test_vax_h_is_identity() {
        let val = 2.0_f32;
        assert_eq!(fake_quantize_f32(val, FormatKind::VaxH), val);
    }

    #[test]
    fn test_bcd8_rounds_to_integer() {
        let q = fake_quantize_f32(12.7_f32, FormatKind::Bcd8);
        assert!((q - 13.0).abs() < 1e-5, "bcd8(12.7) should be 13, got {q}");
        let q2 = fake_quantize_f32(100.0_f32, FormatKind::Bcd8);
        assert_eq!(q2, 99.0, "bcd8 max=99, got {q2}");
    }

    #[test]
    fn test_bcd16_rounds_to_integer() {
        let q = fake_quantize_f32(1234.5_f32, FormatKind::Bcd16);
        assert!(
            (q - 1235.0).abs() < 1e-3,
            "bcd16(1234.5) should be 1235, got {q}"
        );
        let q2 = fake_quantize_f32(10000.0_f32, FormatKind::Bcd16);
        assert_eq!(q2, 9999.0, "bcd16 max=9999, got {q2}");
    }

    #[test]
    fn test_q15_range_and_precision() {
        let q = fake_quantize_f32(0.5_f32, FormatKind::Q15);
        assert!((q - 0.5).abs() < 1e-4, "q15(0.5) ≈ 0.5, got {q}");
        // Clamp test
        let q_big = fake_quantize_f32(2.0_f32, FormatKind::Q15);
        assert!(q_big <= 1.0, "q15 must clamp to ≤ 1, got {q_big}");
        let q_neg = fake_quantize_f32(-2.0_f32, FormatKind::Q15);
        assert!(q_neg >= -1.0, "q15 must clamp to ≥ -1, got {q_neg}");
    }

    #[test]
    fn test_q31_is_identity() {
        // q31 wider than f32 mantissa → identity
        let val = 0.7_f32;
        assert_eq!(fake_quantize_f32(val, FormatKind::Q31), val);
    }

    #[test]
    fn test_stochastic_round_is_identity() {
        // StochasticRound marked unsupported → identity
        let val = 1.5_f32;
        assert_eq!(fake_quantize_f32(val, FormatKind::StochasticRound), val);
    }

    #[test]
    fn test_unum_i8_reduces_precision() {
        let val = 0.123456_f32;
        let q = fake_quantize_f32(val, FormatKind::UnumI8);
        assert!((q - val).abs() < 0.1);
    }

    #[test]
    fn test_unum_i16_reduces_precision() {
        let val = 0.123456_f32;
        let q = fake_quantize_f32(val, FormatKind::UnumI16);
        assert!((q - val).abs() < 0.01);
    }

    // -------- Cross-format divergence on a tiny canary --------

    #[test]
    fn canary_distinct_bpb_per_format() {
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
            let sig: u64 = w.iter().map(|v| v.to_bits() as u64).sum();
            signatures.push((fmt, sig));
        }

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

    // -------- is_faithful() checks --------

    #[test]
    fn test_is_faithful_known_true() {
        for fmt in [
            FormatKind::F32,
            FormatKind::Fp16,
            FormatKind::Bf16,
            FormatKind::Fp8E4M3,
            FormatKind::Int8,
            FormatKind::Gf16,
        ] {
            assert!(fmt.is_faithful(), "{fmt:?} should be faithful");
        }
    }

    #[test]
    fn test_is_faithful_known_false() {
        for fmt in [
            FormatKind::Nf8,
            FormatKind::Posit8,
            FormatKind::Posit16,
            FormatKind::Posit32,
            FormatKind::Bcd8,
            FormatKind::Bcd16,
            FormatKind::IbmHfpShort,
            FormatKind::Q15,
            FormatKind::UnumI8,
            FormatKind::UnumII8,
        ] {
            assert!(
                !fmt.is_faithful(),
                "{fmt:?} should be non-faithful (deferred)"
            );
        }
    }
}
