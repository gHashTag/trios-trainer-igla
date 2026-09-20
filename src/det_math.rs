//! Bit-exact elementary functions, portable by construction.
//!
//! WHY THIS EXISTS
//!
//! The cross-architecture checkpoint mismatch was localised, not explained, by
//! `scripts/local_isa_probe.py`: on ONE macOS host, ONE pinned rustc and ONE
//! working tree, with the `--target` triple as the only moving variable, the
//! initial weights (`0.bin`) are byte-identical across the ISA change and the
//! step-10 weights (`10.bin`) are not. So the divergence is introduced by the
//! arithmetic of the training loop, within ten optimizer steps.
//!
//! Two candidate mechanisms were then eliminated by measurement rather than by
//! argument:
//!
//!   * FMA contraction is EXCLUDED BY DISASSEMBLY. Neither the aarch64 nor the
//!     x86_64 binary contains a fused multiply-add on the training path.
//!   * Reduction order is EXCLUDED BY CONSTRUCTION. Building the same tree at
//!     `-O0` and at `-O3 + LTO` produces identical weights; a compiler free to
//!     re-associate the summations would not.
//!
//! What was left standing is `libm`. `src/bin/ulp_census.rs` measures it
//! directly: over the census inputs, `expf` disagrees between the two arms on
//! 246 of 40010 inputs, and every disagreement is EXACTLY 1 ULP, while `sqrtf`
//! and `powi` agree everywhere (`sqrtf` is required to be correctly rounded by
//! IEEE 754, and `powi` lowers to a compiler intrinsic, not a libm call).
//!
//! A 1-ULP disagreement in `expf` is not a bug in either libm. `expf` is not
//! required by IEEE 754 to be correctly rounded, so two conforming
//! implementations may legitimately differ in the last bit, and Apple ships a
//! different code path for each instruction set. NOTHING IN THE TRAINER CAN
//! FIX THAT BY CALLING `expf` MORE CAREFULLY. The only way to remove the
//! dependency is to stop calling it.
//!
//! WHY EACH CHOICE BELOW IS FORCED
//!
//! `exp_det` therefore uses ONLY `+`, `-`, `*`, `/` and integer shifts. IEEE
//! 754 requires each of those five to be correctly rounded, which means their
//! results are a function of the input bits alone - not of the instruction set,
//! not of the vendor, not of the optimisation level. Every other choice in the
//! body follows from that single constraint:
//!
//!   * NO `mul_add`. `f32::mul_add` IS correctly rounded and IS portable in its
//!     result, but on a target without a hardware FMA it lowers to a `fmaf`
//!     library call whose availability and code path are again the platform's
//!     business. Avoiding it costs a few ULP of headroom in the polynomial and
//!     buys a body that cannot call into any vendor at all.
//!   * NO libm call of any kind, transitively: no `exp`, no `ln`, no `powf`,
//!     no `powi`, no `round`, no `floor`. `round`/`floor` are correctly rounded
//!     but are still function calls into the platform on some targets, so the
//!     rounding is done with an add-and-truncate that the compiler lowers to an
//!     ISA instruction.
//!   * Cody-Waite TWO-PART ln2. The reduction `r = x - k*ln2` must be computed
//!     without losing the low bits of `k*ln2`. `LN2_HI` is chosen with nine
//!     zero trailing mantissa bits, so `kf * LN2_HI` is EXACT for every `|k|`
//!     this function can produce (|k| <= 128 << 512), and the subtraction
//!     `x - kf*LN2_HI` is exact by Sterbenz. Only then is the correction
//!     `kf * LN2_LO` applied. A one-part ln2 would leave ~1e-8 of absolute
//!     error in `r`, which is 4 ULP after the `2^k` scaling.
//!   * HORNER, written out. Horner fixes the association order at the source
//!     level: there is exactly one legal parse of the nest, so no compiler and
//!     no ISA can choose a different summation. An Estrin or pairwise form
//!     would be faster and would reintroduce exactly the freedom this module
//!     exists to remove.
//!   * `2^k` FROM EXPONENT BITS. Constructing the power of two by writing the
//!     biased exponent field is exact and involves no floating-point operation
//!     at all. It is split into two factors when `k > 127` so the field never
//!     leaves the normal range `[1, 254]`.
//!
//! WHAT IT IS NOT
//!
//! `exp_det` is NOT a drop-in replacement for `f32::exp`. It is up to 3 ULP
//! away from Apple's libm (measured, see `tests/det_math_bit_exact.rs`), so
//! enabling the `det-math` feature MOVES EVERY PUBLISHED HASH. That is why the
//! feature is default-OFF. See `docs/DET-MATH.md`.
//!
//! It also has a DELIBERATE UNDERFLOW DEVIATION, measured rather than assumed.
//! `exp_det` returns `+0.0` once the argument reduction yields `k < -126`, and
//! that boundary is sharp and was read off the implementation, not derived:
//!
//! ```text
//!     x = -87.683113098 (0xc2af5dc1)  exp_det 8.312046e-39, libm 8.312044e-39
//!     x = -87.683120728 (0xc2af5dc2)  exp_det 0.0,          libm 8.31198e-39
//!     x = -103.972084045 (0xc2cff1b5) is where libm itself first returns 0.0
//! ```
//!
//! So on `[-103.972084045, -87.683120728]` this function returns zero and
//! Apple's libm returns a subnormal. The window is asserted in
//! `tests/det_math_bit_exact.rs` so it cannot drift silently.
//!
//! The deviation is deliberate. Reaching further down means scaling a normal
//! result into the subnormal range with a SECOND rounding after the polynomial
//! has already rounded once, and a double rounding is the one place where this
//! construction could quietly become platform-dependent again - which would
//! defeat the whole point. Returning zero is exact, cheap and identical
//! everywhere. The training path calls `exp` only inside softmax on
//! `x - max <= 0`, where a term below 8e-39 contributes nothing to a sum that
//! already contains a 1.0.
//!
//! THE SECOND SITE: `cos`
//!
//! Pinning `exp` was measured to close the cross-ISA gap at TEN steps and to
//! leave the 12000-step headline open, because the learning-rate schedule
//! diverges on a path the forward pass never touches. `train_loop::cosine_lr`
//! calls `f32::cos`, and `src/bin/lr_schedule_dump.rs` measured the
//! consequence: 58 of the 12000 headline learning rates differ between the two
//! instruction sets, the first at step 4041 (aarch64 `3b25032e`, x86_64
//! `3b25032d`). No amount of determinism in `exp` can reach that.
//!
//! `cos_det` is the answer, built under the SAME constraints as `exp_det` and
//! none of them relaxed. What is FORCED, and by what:
//!
//!   * ONLY `+`, `-`, `*`, `/` and integer shifts, for the same reason: those
//!     are the operations IEEE 754 requires to be correctly rounded, so their
//!     results are a function of the input bits alone.
//!   * NO `mul_add`, NO libm call transitively - and that includes `abs`. The
//!     magnitude test below masks the sign bit with an integer AND rather than
//!     calling `f32::abs`, so no path in this function can reach a vendor.
//!   * CODY-WAITE THREE-PART `pi/2`. `PIO2_HI` carries TWELVE significant bits
//!     (its low twelve mantissa bits are zero), so `kf * PIO2_HI` is exact for
//!     every `|k| <= 2^11` this function can produce, and `x - kf * PIO2_HI` is
//!     exact too: for `|x| <= COS_MAX_ARG` both operands are multiples of
//!     `2^-12` and the difference is below `pi/4`, so it needs at most twelve
//!     bits. `PIO2_MID` carries twelve significant bits for the same reason.
//!     `PIO2_LO` is the remainder; `PIO2_HI + PIO2_MID + PIO2_LO` reproduces
//!     `pi/2` to the last bit of a double. A one-part or two-part split would
//!     leave error in `r` that the polynomial cannot recover.
//!   * A DOMAIN LIMIT, `COS_MAX_ARG = 2048.0`, and it is a HARD one. Above it
//!     the exactness argument in the previous bullet stops holding, so the
//!     function returns `NaN` rather than a plausible wrong number. An
//!     unbounded reduction needs Payne-Hanek, which is a different piece of
//!     work; a silent approximation past the point where the construction is
//!     provable is exactly the defect class this module exists to remove.
//!   * QUADRANT SELECT ON `k & 3`, with sin and cos minimax nests written out
//!     longhand. Rust's `&` is two's-complement, so `k & 3` is already in
//!     `0..=3` for negative `k` and no separate sign fixup can disagree with
//!     itself.
//!   * HORNER, written out, for both nests. Same argument as `exp_det`: there
//!     is exactly one legal parse, so no compiler and no ISA can choose a
//!     different summation.
//!
//! What is FREE, and was chosen rather than forced: the polynomial DEGREES
//! (cos to `r^10`, sin to `r^9`) and the split point of `pi/2` (twelve bits
//! rather than eleven or thirteen). Both were picked to put the truncation
//! error below the rounding error, and both could be moved without breaking
//! portability - they would only move the accuracy, and the frozen digest in
//! `tests/det_math_bit_exact.rs` would name the move.
//!
//! Nothing UPSTREAM of the call needs replacing. `cosine_lr` passes
//! `std::f32::consts::PI * p`, and that is a single correctly-rounded f32
//! multiplication of two f32 values - portable already, by the same IEEE 754
//! guarantee this module is built on. The libm dependence began at `.cos()`
//! and ends there.
//!
//! A TRAP, recorded because it produced a wrong number once. Anyone
//! re-deriving the 58/12000 baseline in C must pass `-ffp-contract=off`.
//! clang defaults to `-ffp-contract=on`, and a contracted build of the same
//! schedule expression reports 3413 of 12000 differing, first at step 1203 -
//! a ~59x larger effect than libm's, and not a measurement of libm at all.
//! rustc never contracts, so the Rust numbers do not carry this hazard.

/// `log2(e)`, rounded to f32. Used only to pick the integer `k`; an error here
/// shifts `k` by one at a boundary, which the Cody-Waite reduction absorbs.
const LOG2E: f32 = 1.442_695_04;

/// High part of `ln 2`. Chosen with the low mantissa bits zeroed so that
/// `kf * LN2_HI` is exact for every `k` this function produces.
const LN2_HI: f32 = 0.693_145_75;

/// `ln 2 - LN2_HI`, the Cody-Waite correction term.
const LN2_LO: f32 = 1.428_606_8e-6;

/// Degree-6 Taylor coefficients `1/2!`..`1/6!`, each rounded to f32 once, here,
/// so the constant folding is done at source level and not by a compiler whose
/// intermediate precision is its own business.
const C2: f32 = 0.5;
const C3: f32 = 0.166_666_67;
const C4: f32 = 0.041_666_67;
const C5: f32 = 0.008_333_33;
const C6: f32 = 0.001_388_89;

/// Above this argument `exp(x)` exceeds `f32::MAX` and the answer is `+inf`.
/// `ln(f32::MAX) = 88.7228390...`; this is the next f32 at or above it.
pub const EXP_MAX_ARG: f32 = 88.722_84;

/// Smallest `k` for which `2^k` is a normal f32. Below it the true result is
/// subnormal and this function returns `+0.0` - see the module note.
const MIN_NORMAL_K: i32 = -126;

/// `exp(x)` computed with correctly-rounded IEEE-754 operations only.
///
/// The result is a function of the input bits alone. It does not depend on the
/// instruction set, the operating system, the libm vendor or the optimisation
/// level. It is up to 3 ULP from `f32::exp` on this platform; that difference
/// is measured, not bounded by argument, in `tests/det_math_bit_exact.rs`.
///
/// Documented edges:
///   * `NaN`   -> the same `NaN` bits, returned without arithmetic.
///   * `x > EXP_MAX_ARG`, including `+inf` -> `f32::INFINITY`.
///   * arguments whose reduction yields `k < -126` (roughly `x < -87.7`),
///     including `-inf` -> `+0.0`.
#[inline]
pub fn exp_det(x: f32) -> f32 {
    // NaN first: every comparison below is false for NaN, so it would otherwise
    // fall through into the polynomial and come out as an arbitrary NaN. Return
    // the argument itself, so no arithmetic touches the payload bits.
    if x.is_nan() {
        return x;
    }
    // Overflow, and `+inf` with it.
    if x > EXP_MAX_ARG {
        return f32::INFINITY;
    }

    // k = round(x * log2 e), by add-and-truncate. `as i32` truncates toward
    // zero in Rust and saturates on out-of-range input, so `-inf` lands on
    // `i32::MIN` and is caught by the MIN_NORMAL_K guard below.
    let half = if x >= 0.0 { 0.5 } else { -0.5 };
    let k = (x * LOG2E + half) as i32;
    if k < MIN_NORMAL_K {
        // Underflow to zero, and `-inf` with it.
        return 0.0;
    }
    let kf = k as f32;

    // Cody-Waite: r = x - k*ln2, in [-ln2/2, ln2/2]. Both products are exact.
    let r = (x - kf * LN2_HI) - kf * LN2_LO;

    // Horner. The nesting is the association order; there is no other parse.
    let p = 1.0 + r * (1.0 + r * (C2 + r * (C3 + r * (C4 + r * (C5 + r * C6)))));

    // 2^k from the exponent field. Split when k > 127 so the biased exponent
    // stays inside [1, 254]; k <= 128 here, so k2 is 0 or 1 and both factors
    // are normal. The multiplication order is fixed by the parentheses.
    let (k1, k2) = if k > 127 { (127, k - 127) } else { (k, 0) };
    let s1 = f32::from_bits(((k1 + 127) as u32) << 23);
    let s2 = f32::from_bits(((k2 + 127) as u32) << 23);
    (p * s1) * s2
}

/// `2/pi`, rounded to f32. Used only to pick the integer quadrant `k`; an error
/// here shifts `k` by one at a boundary, which the Cody-Waite reduction absorbs
/// into a slightly wider `r`.
const TWO_OVER_PI: f32 = 0.636_619_747;

/// High part of `pi/2`, with the low TWELVE mantissa bits zeroed so that
/// `kf * PIO2_HI` is exact for every `k` this function produces. The value is
/// `1.57080078125` exactly - a dyadic rational, written out in full.
const PIO2_HI: f32 = 1.570_800_781_25;

/// Middle Cody-Waite term, also twelve significant bits. Negative because
/// `PIO2_HI` rounded UP; the three parts still sum to `pi/2`.
const PIO2_MID: f32 = -4.453_584_55e-6;

/// Remainder: `pi/2 - PIO2_HI - PIO2_MID`, rounded to f32 once, here.
const PIO2_LO: f32 = -8.705_516_31e-10;

/// Even Taylor coefficients `-1/2!`, `1/4!`, `-1/6!`, `1/8!`, `-1/10!`, each
/// rounded to f32 once at source level. Truncating the cosine nest after
/// `r^10` leaves `r^12/12! < 1.6e-10` on `|r| <= pi/4`, which is three orders
/// of magnitude below an f32 ULP at 1.0.
const COS_C2: f32 = -0.5;
const COS_C4: f32 = 0.041_666_667_9;
const COS_C6: f32 = -0.001_388_888_92;
const COS_C8: f32 = 2.480_158_76e-5;
const COS_C10: f32 = -2.755_732_00e-7;

/// Odd Taylor coefficients `-1/3!`, `1/5!`, `-1/7!`, `1/9!`. Truncating the
/// sine nest after `r^9` leaves `r^11/11! < 1.9e-9` on `|r| <= pi/4`.
const SIN_S3: f32 = -0.166_666_672;
const SIN_S5: f32 = 0.008_333_333_77;
const SIN_S7: f32 = -1.984_127_01e-4;
const SIN_S9: f32 = 2.755_731_88e-6;

/// The hard domain limit. Above `|x| = 2048` the Cody-Waite reduction below is
/// no longer provably exact, and this function returns `NaN` rather than an
/// approximation it cannot justify. See the module header.
pub const COS_MAX_ARG: f32 = 2048.0;

/// `cos(x)` computed with correctly-rounded IEEE-754 operations only.
///
/// The result is a function of the input bits alone. It does not depend on the
/// instruction set, the operating system, the libm vendor or the optimisation
/// level. It is up to 2 ULP from `f32::cos` on this platform; that difference
/// is measured, not bounded by argument, in `tests/det_math_bit_exact.rs`.
///
/// Documented edges:
///   * `NaN`   -> the same `NaN` bits, returned without arithmetic.
///   * `|x| > COS_MAX_ARG`, including both infinities -> `NaN`. This is a
///     refusal, not a result: outside that range the argument reduction is not
///     exact and no claim about portability would survive.
///   * `cos_det(0.0)` and `cos_det(-0.0)` are exactly `1.0`, which is what
///     makes `cosine_lr` agree with the libm branch at `p == 0`.
#[inline]
pub fn cos_det(x: f32) -> f32 {
    // NaN first: every comparison below is false for NaN, so it would otherwise
    // fall through into the polynomial. Return the argument itself, so no
    // arithmetic touches the payload bits.
    if x.is_nan() {
        return x;
    }
    // Magnitude by masking the sign bit. `f32::abs` would be a second thing to
    // trust; an integer AND is not.
    let mag = f32::from_bits(x.to_bits() & 0x7fff_ffff);
    if mag > COS_MAX_ARG {
        return f32::NAN;
    }

    // k = round(x * 2/pi), by add-and-truncate. `as i32` truncates toward zero
    // in Rust. |x| <= 2048 gives |k| <= 1305, well inside the 2^11 for which
    // the products below are exact.
    let half = if x >= 0.0 { 0.5 } else { -0.5 };
    let k = (x * TWO_OVER_PI + half) as i32;
    let kf = k as f32;

    // Cody-Waite: r = x - k*(pi/2), in about [-pi/4, pi/4]. The first product
    // and the first subtraction are both exact; the other two terms are
    // corrections small enough that their rounding stays below an ULP of the
    // result.
    let r = ((x - kf * PIO2_HI) - kf * PIO2_MID) - kf * PIO2_LO;
    let r2 = r * r;

    // Horner, both nests. The nesting is the association order; there is no
    // other parse.
    let c = 1.0 + r2 * (COS_C2 + r2 * (COS_C4 + r2 * (COS_C6 + r2 * (COS_C8 + r2 * COS_C10))));
    let s = r * (1.0 + r2 * (SIN_S3 + r2 * (SIN_S5 + r2 * (SIN_S7 + r2 * SIN_S9))));

    // Quadrant. Rust's `&` is two's-complement, so this is already in 0..=3 for
    // negative k and there is no sign fixup to get wrong.
    match k & 3 {
        0 => c,
        1 => -s,
        2 => -c,
        _ => s,
    }
}
