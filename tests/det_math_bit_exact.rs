//! Guards on `trios_trainer::det_math::exp_det` and `::cos_det`.
//!
//! This test runs under DEFAULT FEATURES. `src/det_math.rs` is compiled
//! unconditionally and only the two training-path call sites are behind
//! `--features det-math`, precisely so that a regression here cannot hide
//! behind a disabled flag - a guard that only runs when someone remembers to
//! pass a feature is a guard nobody runs.
//!
//! Eight separate claims, eight separate tests, because they fail for different
//! reasons and a single test would report only the first. The first four guard
//! `exp_det`:
//!
//!   1. `frozen_vector_is_bit_identical` - the function's OUTPUT BITS, frozen.
//!      This is the guard that matters. `exp_det` exists to be a fixed function
//!      of its input bits; any change to a coefficient, to the reduction, to
//!      the association order or to the scaling changes these numbers, and this
//!      test is what makes that visible instead of silent. The pairs were
//!      generated from the implementation itself and then frozen - they are a
//!      record of what it DOES, not an independent check of what it SHOULD do.
//!      Test 2 is what supplies the second opinion.
//!   2. `within_three_ulp_of_libm` - an independent accuracy bound. The frozen
//!      vector alone would happily freeze a broken function.
//!   3. `documented_edges_hold` - the overflow, underflow and NaN behaviour
//!      that `src/det_math.rs` promises in prose, asserted as numbers,
//!      including the measured window where this function deliberately
//!      disagrees with libm.
//!   4. `sweep_digest_is_frozen` - a checksum over the output bits of 400001
//!      points. It exists because tests 1 and 2 were MEASURED to be blind to
//!      changes they ought to catch, and the measurement is in
//!      `docs/DET-MATH.md`: a one-ULP change to `C3`, `C4` or `C5` moves 775,
//!      55 and 5 of those 400001 outputs respectively and moves NONE of the 28
//!      frozen pairs, while staying inside test 2's 3-ULP bound. Twenty-eight
//!      points cannot guard a function; a digest over the sweep can.
//!
//! What NO test here can catch, also measured: a one-ULP change to `C6`,
//! `LOG2E` or `LN2_LO` changes not one bit of output anywhere in the sweep.
//! Those constants are below the resolution of the result at f32 precision -
//! `C6 * r^6` contributes at most 2.4e-6 of the polynomial, so perturbing it by
//! its own last bit moves the answer by ~1e-13 relative against an f32 ULP of
//! 6e-8. There is nothing to detect, and a test claiming to detect it would be
//! lying. See docs/DET-MATH.md, "Breaking the guard".
//!
//! `cos_det` adds four more, and they guard a DIFFERENT thing. `exp_det` is
//! tested on its own outputs because softmax's use of it is diffuse; `cos_det`
//! has exactly one caller, `train_loop::cosine_lr`, and one argument set that
//! anybody cites - the 12000 steps of the headline run. So the cosine tests are
//! anchored on that schedule rather than on an abstract sweep:
//!
//!   5. `cos_det_matches_libm_within_measured_bound_on_the_headline` - the
//!      accuracy second opinion, taken at the arguments that matter, and it
//!      also proves the argument replica this file uses is faithful to
//!      `cosine_lr` by reconstructing every learning rate from it and
//!      comparing bit for bit against the real function, under BOTH feature
//!      states.
//!   6. `headline_schedule_endpoints_are_pinned` - the schedule's contract at
//!      its two ends, asserted under BOTH feature states so that swapping the
//!      cosine cannot silently move the learning rate the trainer starts and
//!      finishes on.
//!   7. `cos_det_headline_digest_is_frozen` - the tripwire, always compiled.
//!      One FNV-1a over the output bits of all 10801 cosine-branch arguments.
//!   8. `headline_schedule_digest_is_frozen` (det-math builds) /
//!      `default_schedule_is_not_the_det_math_schedule` (default builds) - the
//!      frozen det-math schedule is the SAME number both ISA arms of
//!      `scripts/lr_schedule_isa_probe.py` printed, so this test is what ties
//!      the published 0-of-12000 result to the code in the tree. Its default
//!      counterpart asserts the feature is not inert.

use trios_trainer::det_math::{cos_det, exp_det, COS_MAX_ARG, EXP_MAX_ARG};
use trios_trainer::train_loop::cosine_lr;

/// The headline run's schedule parameters: `--steps 12000 --lr 0.003`, with
/// `warmup = steps / 10` as `run_single` computes it. The same three numbers
/// `src/bin/lr_schedule_dump.rs` dumps with.
const HEADLINE_STEPS: usize = 12_000;
const HEADLINE_WARMUP: usize = 1_200;
const HEADLINE_BASE_LR: f32 = 0.003;

/// The argument `cosine_lr` hands to the cosine at `step`, for the steps that
/// take the cosine branch (`step >= warmup`).
///
/// This is a two-line replica of one line of `cosine_lr`, and a replica is the
/// thing this repository has been burned by before. So it is not trusted: test
/// 5 reconstructs the whole learning rate from it and compares against
/// `cosine_lr` itself, bit for bit, at all 10801 steps and under both feature
/// states. If the replica ever drifts, that comparison fails.
fn headline_cos_arg(step: usize) -> f32 {
    let p = (step - HEADLINE_WARMUP) as f32 / (HEADLINE_STEPS - HEADLINE_WARMUP).max(1) as f32;
    std::f32::consts::PI * p
}

/// FNV-1a over a stream of f32 bit patterns. Pure integer arithmetic, so the
/// digest itself cannot become the platform-dependent thing. Same constants and
/// same byte order as `sweep_digest` below, so the two are comparable by eye.
struct Fnv1a(u64);

impl Fnv1a {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn push(&mut self, value: f32) {
        for byte in value.to_bits().to_le_bytes() {
            self.0 ^= byte as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// 28 (input bits -> output bits) pairs. Generated from `exp_det` on
/// 2026-08-06 and frozen. If a pair moves, `exp_det` is a different function
/// than the one every measurement under `evidence/det-math-isa/` was taken
/// with, and those measurements no longer describe the code.
///
/// The inputs are chosen to reach every branch and every regime: zero and
/// negative zero, |x| < 1 where `k == 0` and the polynomial carries the whole
/// result, exactly +-ln2 and +-ln2/2 (the ends of the reduced interval, where
/// the answer must land on an exact power of two or its square root), large
/// positive values near the overflow edge, and large negative values in the
/// subnormal-output regime.
const FROZEN: &[(u32, u32)] = &[
    (0x0000_0000, 0x3f80_0000), // 0
    (0x8000_0000, 0x3f80_0000), // -0
    (0x3f80_0000, 0x402d_f854), // 1
    (0xbf80_0000, 0x3ebc_5ab2), // -1
    (0x3f00_0000, 0x3fd3_094c), // 0.5
    (0xbf00_0000, 0x3f1b_4598), // -0.5
    (0x4000_0000, 0x40ec_7326), // 2
    (0xc000_0000, 0x3e0a_9555), // -2
    (0x40e0_0000, 0x4489_1443), // 7
    (0xc0e0_0000, 0x3a6f_0b5d), // -7
    (0x3f31_7218, 0x4000_0000), // ln2      -> exactly 2.0
    (0xbf31_7218, 0x3f00_0000), // -ln2     -> exactly 0.5
    (0x322b_cc77, 0x3f80_0000), // 1e-8
    (0xb22b_cc77, 0x3f80_0000), // -1e-8
    (0x3dcc_cccd, 0x3f8d_763e), // 0.1
    (0xbdcc_cccd, 0x3f67_a36d), // -0.1
    (0x4049_0fdb, 0x41b9_2026), // pi
    (0xc049_0fdb, 0x3d31_0112), // -pi
    (0x4138_0000, 0x47c0_cde3), // 11.5
    (0xc138_0000, 0x3729_f46c), // -11.5
    (0x42ae_0000, 0x7e36_d80a), // 87
    (0xc2ae_0000, 0x00b3_3686), // -87
    (0x3eb1_7218, 0x3fb5_04f5), // ln2/2    -> exactly sqrt(2)
    (0xbeb1_7218, 0x3f35_04f2), // -ln2/2
    (0x41a2_0000, 0x4e14_86bc), // 20.25
    (0xc1a2_0000, 0x30dc_9ef1), // -20.25
    (0x0da2_4260, 0x3f80_0000), // 1e-30
    (0x8da2_4260, 0x3f80_0000), // -1e-30
];

#[test]
fn frozen_vector_is_bit_identical() {
    assert!(
        FROZEN.len() >= 24,
        "the frozen vector must carry at least 24 pairs; it has {}",
        FROZEN.len()
    );
    let mut failures: Vec<String> = Vec::new();
    for &(inp, expected) in FROZEN {
        let x = f32::from_bits(inp);
        let got = exp_det(x).to_bits();
        if got != expected {
            failures.push(format!(
                "exp_det({x:e}) [input bits 0x{inp:08x}] gave 0x{got:08x}, frozen value is 0x{expected:08x}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "exp_det is no longer the function these bits were frozen from.\n\
         first differing input: {}\n\
         {} of {} pairs moved:\n  {}",
        failures[0],
        failures.len(),
        FROZEN.len(),
        failures.join("\n  ")
    );
}

/// The census range from `docs/DIVERGENCE-MECHANISM.md`, swept densely.
///
/// This is the second opinion on the frozen vector: it compares against the
/// platform libm, which was written by somebody else. The bound is 3 ULP and
/// the MEASURED maximum is printed on every run, so a slow drift toward the
/// bound is visible before it becomes a failure.
#[test]
fn within_three_ulp_of_libm() {
    const SAMPLES: u32 = 400_000;
    const BOUND: i64 = 3;

    let mut worst: i64 = 0;
    let mut worst_at = f32::NAN;
    let mut differing: u64 = 0;

    for i in 0..=SAMPLES {
        let x = -8.0 + 16.0 * (i as f32) / (SAMPLES as f32);
        let det = exp_det(x).to_bits() as i64;
        let libm = x.exp().to_bits() as i64;
        if det != libm {
            differing += 1;
            let ulp = (det - libm).abs();
            if ulp > worst {
                worst = ulp;
                worst_at = x;
            }
        }
    }

    println!(
        "exp_det vs f32::exp over [-8.0, 8.0]: n={}, differing={}, max |ULP|={} at x={worst_at}",
        SAMPLES + 1,
        differing,
        worst
    );
    assert!(
        worst <= BOUND,
        "exp_det drifted to {worst} ULP from libm at x={worst_at}; the documented bound is {BOUND}"
    );
    // A max of zero would mean exp_det had somehow become libm, which would
    // mean the whole module is not doing what it claims. Say so rather than
    // pass quietly.
    assert!(
        differing > 0,
        "exp_det agreed with libm on every one of {} samples. It is not a \
         reimplementation then, and docs/DET-MATH.md's premise is wrong.",
        SAMPLES + 1
    );
}

/// FNV-1a over the output bits of the whole sweep. Pure integer arithmetic, so
/// the digest itself cannot become the platform-dependent thing.
fn sweep_digest() -> (u64, u32) {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut n: u32 = 0;
    for i in 0..=SWEEP_SAMPLES {
        let x = -8.0 + 16.0 * (i as f32) / (SWEEP_SAMPLES as f32);
        for byte in exp_det(x).to_bits().to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        n += 1;
    }
    (h, n)
}

/// One number that changes if ANY of 400001 outputs changes.
///
/// Frozen from the implementation on 2026-08-06, on the same run that produced
/// the frozen vector above. It is a tripwire, not a diagnosis: when it fires it
/// says only that `exp_det` is a different function, and
/// `frozen_vector_is_bit_identical` is what names an input if the change is
/// large enough to reach one of those 28 points.
const SWEEP_DIGEST: u64 = 0x0e01_1a25_8691_4fb2;
const SWEEP_SAMPLES: u32 = 400_000;

#[test]
fn sweep_digest_is_frozen() {
    let (got, n) = sweep_digest();
    assert_eq!(n, SWEEP_SAMPLES + 1, "the sweep changed size");
    assert_eq!(
        got, SWEEP_DIGEST,
        "exp_det produced different output bits somewhere in [-8.0, 8.0].\n\
         digest 0x{got:016x}, frozen 0x{SWEEP_DIGEST:016x}.\n\
         If frozen_vector_is_bit_identical still passes, the change is smaller \
         than the 28 sampled points can see - a one-ULP move of C3, C4 or C5 \
         does exactly that. Find it before assuming it is harmless."
    );
}

#[test]
fn documented_edges_hold() {
    // NaN in, NaN out, with the payload bits untouched.
    let nan = f32::from_bits(0x7fc0_1234);
    assert!(exp_det(f32::NAN).is_nan(), "exp_det(NaN) is not NaN");
    assert_eq!(
        exp_det(nan).to_bits(),
        nan.to_bits(),
        "exp_det changed a NaN's payload bits"
    );

    // Overflow. EXP_MAX_ARG is the first f32 at or above ln(f32::MAX); at it
    // and above it the answer is +inf, and libm agrees at that point.
    assert_eq!(EXP_MAX_ARG.to_bits(), 0x42b1_7218, "EXP_MAX_ARG moved");
    assert!(exp_det(EXP_MAX_ARG).is_infinite());
    assert!(exp_det(f32::INFINITY).is_infinite());
    assert!(
        exp_det(f32::from_bits(EXP_MAX_ARG.to_bits() + 1)).is_infinite(),
        "an argument above the overflow cutoff did not saturate"
    );

    // Just below the cutoff the result must still be FINITE. A cutoff placed
    // one ULP too low would return +inf here while the true answer is
    // representable, and no other assertion in this file would notice.
    let below = f32::from_bits(EXP_MAX_ARG.to_bits() - 1);
    let big = exp_det(below);
    assert!(
        big.is_finite() && big > 3.0e38,
        "exp_det({below}) = {big:e}; the true value is ~3.4028e38 and finite"
    );

    // Underflow, and the DELIBERATE deviation from libm. Measured boundary:
    // 0xc2af5dc1 is the last argument with a nonzero result, 0xc2af5dc2 the
    // first with zero, and libm keeps returning subnormals until 0xc2cff1b5.
    let last_nonzero = f32::from_bits(0xc2af_5dc1);
    let first_zero = f32::from_bits(0xc2af_5dc2);
    assert!(
        exp_det(last_nonzero) > 0.0,
        "the underflow boundary moved: exp_det({last_nonzero}) is no longer positive"
    );
    assert_eq!(
        exp_det(first_zero),
        0.0,
        "the underflow boundary moved: exp_det({first_zero}) is no longer zero"
    );
    assert!(
        first_zero.exp() > 0.0,
        "libm no longer returns a subnormal at {first_zero}; the documented \
         disagreement window in docs/DET-MATH.md needs re-measuring"
    );
    assert_eq!(exp_det(f32::NEG_INFINITY), 0.0);
    assert!(
        exp_det(0.0) == 1.0 && exp_det(-0.0) == 1.0,
        "exp_det(+-0.0) must be exactly 1.0"
    );
}

// ------------------------------------------------------------ cos_det ------

/// The measured maximum, in ULP, between `cos_det` and Apple's `f32::cos` over
/// the 10801 cosine-branch arguments of the headline schedule.
///
/// READ OFF THE IMPLEMENTATION, NOT DERIVED. It is what this code does on this
/// platform, in the same spirit as the underflow window asserted for `exp_det`
/// above. A truncation-error argument would give a much smaller number and a
/// rounding-error argument a vaguer one; neither would be this.
const COS_HEADLINE_ULP_BOUND: i64 = 2;

/// One number that changes if ANY of the 10801 cosine outputs changes.
///
/// Frozen from the implementation on 2026-08-07. A tripwire, not a diagnosis:
/// when it fires it says only that `cos_det` is a different function.
const COS_HEADLINE_DIGEST: u64 = 0x0c4a_426e_5af2_4b0e;

/// The 12000-entry headline schedule under `--features det-math`, as one
/// number.
///
/// This is not merely frozen from the implementation. It is the digest of the
/// dump that BOTH ISA arms of `scripts/lr_schedule_isa_probe.py` produced -
/// `evidence/cos-det-isa/lr-schedule-det-math-aarch64-apple-darwin.txt` and
/// `...-x86_64-apple-darwin.txt` hash to the same value, and this constant was
/// computed from those files by a script outside this crate as well as by the
/// code below. So the assertion below is what ties the published
/// "0 of 12000 differing" result to the function in this tree: if `cosine_lr`
/// changes, the evidence files stop describing it and this test says so.
/// It is compiled in BOTH feature states on purpose: the det-math build
/// asserts the schedule IS this number, and the default build asserts it is
/// NOT, which is what proves the gate is wired.
const DET_MATH_SCHEDULE_DIGEST: u64 = 0x73e1_00bc_f8e6_69fd;

/// Accuracy second opinion, at the arguments that are actually cited - and the
/// faithfulness check on `headline_cos_arg`.
#[test]
fn cos_det_matches_libm_within_measured_bound_on_the_headline() {
    let mut worst: i64 = 0;
    let mut worst_at = f32::NAN;
    let mut differing: u64 = 0;
    let mut n: u64 = 0;

    for step in HEADLINE_WARMUP..=HEADLINE_STEPS {
        let arg = headline_cos_arg(step);

        // The replica is not trusted: rebuild the learning rate from it with
        // whichever cosine this build uses, and demand the real function's
        // bits. This is what licenses every other use of `headline_cos_arg`.
        #[cfg(feature = "det-math")]
        let cos_p = cos_det(arg);
        #[cfg(not(feature = "det-math"))]
        let cos_p = arg.cos();
        let rebuilt = 1e-5 + (HEADLINE_BASE_LR - 1e-5) * 0.5 * (1.0 + cos_p);
        let real = cosine_lr(step, HEADLINE_STEPS, HEADLINE_BASE_LR, HEADLINE_WARMUP);
        assert_eq!(
            rebuilt.to_bits(),
            real.to_bits(),
            "headline_cos_arg has drifted from cosine_lr at step {step}: \
             rebuilt 0x{:08x}, cosine_lr 0x{:08x}",
            rebuilt.to_bits(),
            real.to_bits()
        );

        let det = cos_det(arg).to_bits() as i64;
        let libm = arg.cos().to_bits() as i64;
        n += 1;
        if det != libm {
            differing += 1;
            let ulp = (det - libm).abs();
            if ulp > worst {
                worst = ulp;
                worst_at = arg;
            }
        }
    }

    println!(
        "cos_det vs f32::cos over the {n} headline arguments: differing={differing}, \
         max |ULP|={worst} at x={worst_at}"
    );
    assert_eq!(n, 10_801, "the headline cosine branch changed length");
    assert!(
        worst <= COS_HEADLINE_ULP_BOUND,
        "cos_det drifted to {worst} ULP from libm at x={worst_at}; the measured \
         bound is {COS_HEADLINE_ULP_BOUND}"
    );
    // A max of zero would mean cos_det had somehow become libm, which would
    // mean the whole module is not doing what it claims. Say so rather than
    // pass quietly.
    assert!(
        differing > 0,
        "cos_det agreed with libm on every one of {n} headline arguments. It is \
         not a reimplementation then, and docs/DET-MATH.md's premise is wrong."
    );
}

/// The schedule's contract at both ends, under BOTH feature states.
///
/// Swapping the cosine must not move the learning rate the trainer starts the
/// cosine phase on or the one it finishes on. Both hold exactly because
/// `cos_det(0.0)` is exactly `1.0` and `cos_det(PI)` is exactly `-1.0`, the
/// same two values libm returns; the bit patterns are asserted rather than the
/// decimals so a one-ULP move cannot hide behind a rounded printout.
#[test]
fn headline_schedule_endpoints_are_pinned() {
    let first_cosine = cosine_lr(
        HEADLINE_WARMUP,
        HEADLINE_STEPS,
        HEADLINE_BASE_LR,
        HEADLINE_WARMUP,
    );
    assert_eq!(
        first_cosine.to_bits(),
        HEADLINE_BASE_LR.to_bits(),
        "at step == warmup the schedule must hand back base_lr exactly; got \
         {first_cosine:e} (0x{:08x}) against 0x{:08x}",
        first_cosine.to_bits(),
        HEADLINE_BASE_LR.to_bits()
    );

    let last = cosine_lr(
        HEADLINE_STEPS,
        HEADLINE_STEPS,
        HEADLINE_BASE_LR,
        HEADLINE_WARMUP,
    );
    assert_eq!(
        last.to_bits(),
        1e-5f32.to_bits(),
        "the final learning rate must be exactly 1e-5; got {last:e} (0x{:08x})",
        last.to_bits()
    );

    // The warmup branch calls no cosine at all, so these must be identical in
    // both builds by construction. Asserted anyway: "by construction" is the
    // phrase that precedes most of this repository's wrong sentences.
    assert_eq!(
        cosine_lr(0, HEADLINE_STEPS, HEADLINE_BASE_LR, HEADLINE_WARMUP),
        0.0,
        "the ramp must start at zero"
    );
    assert_eq!(
        cosine_lr(
            HEADLINE_WARMUP / 2,
            HEADLINE_STEPS,
            HEADLINE_BASE_LR,
            HEADLINE_WARMUP
        )
        .to_bits(),
        (HEADLINE_BASE_LR * 0.5).to_bits(),
        "the ramp must be linear through its midpoint"
    );
}

/// The tripwire on `cos_det` itself. Compiled in every build, feature or not.
#[test]
fn cos_det_headline_digest_is_frozen() {
    let mut digest = Fnv1a::new();
    let mut n = 0u32;
    for step in HEADLINE_WARMUP..=HEADLINE_STEPS {
        digest.push(cos_det(headline_cos_arg(step)));
        n += 1;
    }
    assert_eq!(n, 10_801, "the headline cosine branch changed length");
    assert_eq!(
        digest.0, COS_HEADLINE_DIGEST,
        "cos_det produced different output bits somewhere on the headline \
         arguments.\ndigest 0x{:016x}, frozen 0x{COS_HEADLINE_DIGEST:016x}.\n\
         Every number in evidence/cos-det-isa/ was taken with the other \
         function. Find what moved before assuming it is harmless.",
        digest.0
    );
}

/// The det-math schedule, pinned to the digest of the published evidence.
#[cfg(feature = "det-math")]
#[test]
fn headline_schedule_digest_is_frozen() {
    let mut digest = Fnv1a::new();
    for step in 1..=HEADLINE_STEPS {
        digest.push(cosine_lr(
            step,
            HEADLINE_STEPS,
            HEADLINE_BASE_LR,
            HEADLINE_WARMUP,
        ));
    }
    assert_eq!(
        digest.0, DET_MATH_SCHEDULE_DIGEST,
        "the det-math headline schedule moved.\ndigest 0x{:016x}, frozen \
         0x{DET_MATH_SCHEDULE_DIGEST:016x}.\nThat frozen value is the digest of \
         BOTH dumps under evidence/cos-det-isa/, so those files no longer \
         describe this code and the 0-of-12000 result must be re-measured.",
        digest.0
    );
}

/// The default schedule must NOT be the det-math schedule.
///
/// Without this, a `det-math` feature that failed to reach `cosine_lr` at all
/// would leave every other test in this file green: the tripwire above tests
/// `cos_det` directly, and the endpoint test is satisfied by either cosine.
/// This is the assertion that the gate is wired.
#[cfg(not(feature = "det-math"))]
#[test]
fn default_schedule_is_not_the_det_math_schedule() {
    let mut digest = Fnv1a::new();
    for step in 1..=HEADLINE_STEPS {
        digest.push(cosine_lr(
            step,
            HEADLINE_STEPS,
            HEADLINE_BASE_LR,
            HEADLINE_WARMUP,
        ));
    }
    println!("default-build headline schedule digest 0x{:016x}", digest.0);
    assert_ne!(
        digest.0, DET_MATH_SCHEDULE_DIGEST,
        "the default build produced the det-math schedule. Either the feature \
         gate in cosine_lr is not a gate, or cos_det has become libm's cosf. \
         Either way the 58-vs-0 comparison in evidence/cos-det-isa/ is not \
         measuring what it says it measures."
    );
}

/// The edges `src/det_math.rs` promises for `cos_det`, asserted as numbers.
#[test]
fn cos_det_documented_edges_hold() {
    // NaN in, NaN out, with the payload bits untouched.
    let nan = f32::from_bits(0x7fc0_1234);
    assert!(cos_det(f32::NAN).is_nan(), "cos_det(NaN) is not NaN");
    assert_eq!(
        cos_det(nan).to_bits(),
        nan.to_bits(),
        "cos_det changed a NaN's payload bits"
    );

    // Exact at zero, which is what makes the schedule endpoint hold.
    assert_eq!(cos_det(0.0).to_bits(), 1.0f32.to_bits());
    assert_eq!(cos_det(-0.0).to_bits(), 1.0f32.to_bits());

    // The domain limit is a REFUSAL, not a result. Inside it the function must
    // still answer; one ULP outside it must be NaN.
    assert_eq!(COS_MAX_ARG.to_bits(), 0x4500_0000, "COS_MAX_ARG moved");
    assert!(
        cos_det(COS_MAX_ARG).is_finite(),
        "cos_det refused at COS_MAX_ARG itself, which is inside the domain"
    );
    assert!(
        cos_det(-COS_MAX_ARG).is_finite(),
        "cos_det refused at -COS_MAX_ARG, which is inside the domain"
    );
    let just_outside = f32::from_bits(COS_MAX_ARG.to_bits() + 1);
    assert!(
        cos_det(just_outside).is_nan(),
        "cos_det({just_outside}) is outside the provable reduction range and \
         must refuse, not approximate"
    );
    assert!(cos_det(f32::INFINITY).is_nan());
    assert!(cos_det(f32::NEG_INFINITY).is_nan());

    // Quadrant select, including the negative-k path that `k & 3` handles
    // without a sign fixup. cos is even, so these must agree exactly.
    for step in [1usize, 2, 37, 4041, 11966, HEADLINE_STEPS] {
        let x = headline_cos_arg(step.max(HEADLINE_WARMUP));
        assert_eq!(
            cos_det(x).to_bits(),
            cos_det(-x).to_bits(),
            "cos_det is not even at x={x}"
        );
    }
    assert_eq!(
        cos_det(std::f32::consts::PI).to_bits(),
        (-1.0f32).to_bits(),
        "cos_det(PI) must be exactly -1.0; the final learning rate depends on it"
    );
}
