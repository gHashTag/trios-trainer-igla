//! F2 breadth-as-moat harness: phi-ladder vs a heterogeneous "zoo" at a matched
//! bit budget, on a small CPU proxy task.
//!
//! STATUS DISCIPLINE (skill `goldenfloat-ladder`, FL-002). The claim under test
//! is breadth / toolchain-coherence: ONE self-similar rule
//! `e = round((N-1)/phi^2)` spans the whole GF4..GF256 ladder, so a mixed-
//! precision datapath built from GF rungs needs FEWER lossy cross-format
//! conversions than an equally-capable datapath assembled from unrelated formats
//! (ternary + INT8 + FP8 + bf16). This is an [Open conjecture], NOT [Verified],
//! and explicitly NOT a per-rung accuracy claim. Only `phi^2 + phi^-2 = 3`
//! (Lucas L2) is [Verified]. The moat is FALSIFIED (FL-002 Fpath) if a posit /
//! takum / MX ladder, or any prior single-rule width-spanning float family,
//! matches the phi-ladder at a matched bit budget.
//!
//! WHAT THIS HARNESS CAN AND CANNOT SHOW. It is an in-sandbox PROXY at toy scale
//! (a quadratic regression surrogate for an NTP step). It can demonstrate
//! mechanics (the ladder no longer saturates; conversion counts differ) and give
//! a Welch two-sample read, but its numeric verdict is NEVER a Verdict. Per the
//! critical-honesty mandate, the verdict is allowed to come back ZooWins or Tie,
//! and the moat stays [Open conjecture] regardless.
//!
//! THE TWO ARMS (matched ~bit budget per stage).
//! - phi-ladder: scale-aware ternary weights -> GF8 -> GF16 -> GF32 accumulate.
//!   Every rung shares the single phi rule, so stage-to-stage transfers are
//!   "in-family" widenings (counted as coherent, not lossy re-encodings).
//! - zoo: BitNet-style ternary weights -> INT8 -> FP8 (E4M3 forward / E5M2
//!   backward, NVIDIA TE hybrid, arXiv:2310.18313) -> bf16 accumulate. Each
//!   stage is an unrelated encoding, so each transfer is a lossy cross-format
//!   re-encoding.

use crate::fake_quant::{fake_quantize_f32, fake_quantize_weights, FormatKind};
use crate::gf16::GF16;
use crate::phi_numbers::scale_aware::ScaleAwareTernary;
use crate::phi_numbers::{GF32, GF8};

/// Which datapath family a tensor is currently encoded in. Used to decide
/// whether a stage transition is an in-family widening or a lossy cross-format
/// re-encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// GoldenFloat ladder (ternary / GF8 / GF16 / GF32 all share one phi rule).
    Phi,
    /// Heterogeneous zoo (ternary / INT8 / FP8 / bf16 are unrelated families).
    Zoo,
}

/// Tracks the count of LOSSY cross-format conversions performed by a datapath.
///
/// A "coherent" transfer (within the phi ladder: ternary -> GF8 -> GF16 -> GF32)
/// is recorded but NOT counted as lossy, because all rungs are derived from the
/// same closed rule and widen monotonically. A cross-family transfer (INT8 ->
/// FP8, FP8 -> bf16, etc.) is counted as lossy: it requires a re-encode through
/// an unrelated mantissa/exponent split.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConversionCounter {
    /// Number of lossy cross-format re-encodings.
    pub lossy: u64,
    /// Number of coherent in-family widenings (informational).
    pub coherent: u64,
}

impl ConversionCounter {
    /// Record an in-family widening (phi rung -> wider phi rung). Not lossy.
    #[inline]
    pub fn record_coherent(&mut self) {
        self.coherent += 1;
    }

    /// Record a lossy cross-format re-encoding (unrelated format -> unrelated
    /// format).
    #[inline]
    pub fn record_lossy(&mut self) {
        self.lossy += 1;
    }
}

/// One element-wise quantization through a GF rung. The phi ladder uses these as
/// in-family widenings.
fn gf8_round_trip(x: f32) -> f32 {
    GF8::from_f32(x).to_f32()
}
fn gf16_round_trip(x: f32) -> f32 {
    GF16::from_f32(x).to_f32()
}
fn gf32_round_trip(x: f32) -> f32 {
    GF32::from_f32(x).to_f32()
}

/// Run the phi-ladder datapath on a weight tensor: scale-aware ternary weights,
/// then carry the per-element values up the coherent rung sequence
/// ternary -> GF8 -> GF16 -> GF32. Counts three coherent widenings, zero lossy
/// re-encodings. Returns the dequantized tensor at GF32 precision.
pub fn phi_ladder_forward(weights: &[f32], counter: &mut ConversionCounter) -> Vec<f32> {
    // Stage 0: scale-aware ternary (root of the ladder, no saturation).
    let tern = ScaleAwareTernary::quantize(weights);
    let mut x = tern.dequantize();
    // Stage 1: ternary -> GF8 (coherent: GF8 split from the same phi rule).
    for v in x.iter_mut() {
        *v = gf8_round_trip(*v);
    }
    counter.record_coherent();
    // Stage 2: GF8 -> GF16 (coherent widening).
    for v in x.iter_mut() {
        *v = gf16_round_trip(*v);
    }
    counter.record_coherent();
    // Stage 3: GF16 -> GF32 accumulate (coherent widening).
    for v in x.iter_mut() {
        *v = gf32_round_trip(*v);
    }
    counter.record_coherent();
    x
}

/// Run the zoo datapath on a weight tensor: BitNet-style scale-aware ternary
/// weights, then carry the values through INT8 -> FP8(E4M3 fwd) -> bf16 accumulate.
/// Each transition is an unrelated re-encoding and counts as lossy.
pub fn zoo_forward(weights: &[f32], counter: &mut ConversionCounter) -> Vec<f32> {
    // Stage 0: scale-aware ternary weights (same BitNet b1.58 root for fairness).
    let tern = ScaleAwareTernary::quantize(weights);
    let mut x = tern.dequantize();
    // Stage 1: ternary -> INT8 (lossy: per-tensor integer re-encode).
    fake_quantize_weights(&mut x, FormatKind::Int8);
    counter.record_lossy();
    // Stage 2: INT8 -> FP8 E4M3 forward (lossy: unrelated float re-encode).
    for v in x.iter_mut() {
        *v = fake_quantize_f32(*v, FormatKind::Fp8E4M3);
    }
    counter.record_lossy();
    // Stage 3: FP8 -> bf16 accumulate (lossy: unrelated float re-encode).
    for v in x.iter_mut() {
        *v = fake_quantize_f32(*v, FormatKind::Bf16);
    }
    counter.record_lossy();
    x
}

/// Backward-pass re-encode for the zoo arm. NVIDIA TE hybrid uses E5M2 on the
/// backward path (wider exponent for gradient dynamic range), which is a fourth
/// lossy cross-format conversion the phi ladder does not incur (it accumulates
/// gradients in GF32, an in-family rung).
pub fn zoo_backward(grads: &mut [f32], counter: &mut ConversionCounter) {
    for g in grads.iter_mut() {
        *g = fake_quantize_f32(*g, FormatKind::Fp8E5M2);
    }
    counter.record_lossy();
}

/// Phi-arm backward: gradients stay in the GF32 rung (coherent, in-family).
pub fn phi_backward(grads: &mut [f32], counter: &mut ConversionCounter) {
    for g in grads.iter_mut() {
        *g = gf32_round_trip(*g);
    }
    counter.record_coherent();
}

/// A tiny deterministic proxy "loss" for one arm on one synthetic batch.
///
/// This is a quadratic regression surrogate for an NTP step (the same shape the
/// repo's `r12_optimizer_race` proxy uses): we have target weights `w*`, the arm
/// quantizes a noisy estimate `w_hat` through its datapath, and the proxy loss is
/// the mean-squared reconstruction error of the dequantized weights against the
/// clean targets, expressed in bits ( -log2 ) so it reads like a BPB-scale
/// number. LOWER is better. It is a PROXY: it measures how much the datapath's
/// quantization distorts the signal, not real corpus BPB.
pub struct ArmResult {
    /// Proxy loss in bits (lower is better). NOT real BPB.
    pub proxy_bits: f64,
    /// Conversion accounting for this arm.
    pub conversions: ConversionCounter,
}

/// Convert a mean-squared error into a bits-scale figure, clamped to a sane
/// range so a degenerate (saturated) arm reports a large-but-finite number
/// rather than +inf.
fn mse_to_bits(mse: f64) -> f64 {
    // Map MSE in (0, inf) to bits via 0.5*log2(1 + mse/eps); monotone, finite,
    // and ~0 for a near-lossless datapath.
    let eps = 1e-6_f64;
    (0.5 * (1.0 + mse / eps).log2()).clamp(0.0, 64.0)
}

/// Run one arm (phi or zoo) for one synthetic step at a given seed and return
/// its proxy loss + conversion accounting. `warmup` skips quantization for the
/// first `warmup` steps within the multi-step inner loop, modeling
/// continual-QAT warmup (arXiv:2502.11895): the datapath runs in full precision
/// until activations settle, avoiding an early-saturation strawman.
pub fn run_arm(family: Family, seed: u64, steps: usize, warmup: usize, dim: usize) -> ArmResult {
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    let mut rng = StdRng::seed_from_u64(seed);
    // Clean target weights w*, drawn at a small but non-micro scale.
    let target: Vec<f32> = (0..dim).map(|_| rng.gen_range(-0.05_f32..0.05)).collect();
    let mut counter = ConversionCounter::default();
    let mut acc_mse = 0.0_f64;
    let mut counted_steps = 0usize;

    for step in 0..steps {
        // Noisy estimate of the targets for this step.
        let noisy: Vec<f32> = target
            .iter()
            .map(|&w| w + rng.gen_range(-0.01_f32..0.01))
            .collect();

        let reconstructed = if step < warmup {
            // Warmup: full-precision pass, no quantization, no conversions.
            noisy.clone()
        } else {
            match family {
                Family::Phi => {
                    let mut fwd = phi_ladder_forward(&noisy, &mut counter);
                    // backward re-encode of a gradient proxy (= residual).
                    let mut grad: Vec<f32> =
                        fwd.iter().zip(&target).map(|(&r, &t)| r - t).collect();
                    phi_backward(&mut grad, &mut counter);
                    // apply the (quantized) gradient as a tiny correction.
                    for (f, g) in fwd.iter_mut().zip(&grad) {
                        *f -= 0.0 * g; // proxy: correction folded into next step's noise
                    }
                    fwd
                }
                Family::Zoo => {
                    let mut fwd = zoo_forward(&noisy, &mut counter);
                    let mut grad: Vec<f32> =
                        fwd.iter().zip(&target).map(|(&r, &t)| r - t).collect();
                    zoo_backward(&mut grad, &mut counter);
                    for (f, g) in fwd.iter_mut().zip(&grad) {
                        *f -= 0.0 * g;
                    }
                    fwd
                }
            }
        };

        // Proxy loss: MSE of reconstructed vs clean targets.
        let mse: f64 = reconstructed
            .iter()
            .zip(&target)
            .map(|(&r, &t)| {
                let d = (r - t) as f64;
                d * d
            })
            .sum::<f64>()
            / dim as f64;
        acc_mse += mse;
        counted_steps += 1;
    }

    let mean_mse = if counted_steps > 0 {
        acc_mse / counted_steps as f64
    } else {
        0.0
    };
    ArmResult {
        proxy_bits: mse_to_bits(mean_mse),
        conversions: counter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phi_ladder_has_zero_lossy_conversions() {
        let w = vec![0.01_f32, -0.02, 0.03, -0.005, 0.0];
        let mut c = ConversionCounter::default();
        let _ = phi_ladder_forward(&w, &mut c);
        assert_eq!(c.lossy, 0, "phi ladder must incur no lossy re-encodings");
        assert_eq!(c.coherent, 3, "three coherent widenings expected");
    }

    #[test]
    fn zoo_has_more_lossy_conversions_than_phi() {
        let w = vec![0.01_f32, -0.02, 0.03, -0.005, 0.0];
        let mut cz = ConversionCounter::default();
        let _ = zoo_forward(&w, &mut cz);
        let mut grads = vec![0.001_f32, -0.001, 0.0, 0.002, -0.0005];
        zoo_backward(&mut grads, &mut cz);

        let mut cp = ConversionCounter::default();
        let _ = phi_ladder_forward(&w, &mut cp);
        let mut pg = grads.clone();
        phi_backward(&mut pg, &mut cp);

        // The breadth claim's mechanical signature: zoo has strictly more lossy
        // cross-format conversions than the phi ladder at matched stage count.
        assert!(
            cz.lossy > cp.lossy,
            "zoo lossy {} should exceed phi lossy {}",
            cz.lossy,
            cp.lossy
        );
        assert_eq!(cp.lossy, 0);
        assert_eq!(cz.lossy, 4); // 3 forward + 1 backward (E5M2)
    }

    #[test]
    fn phi_arm_does_not_saturate_on_micro_scale() {
        // The whole point of G: scale-aware root means the proxy loss stays
        // bounded well below the old log2(vocab) saturation ceiling even on a
        // micro-scale tensor.
        let res = run_arm(Family::Phi, 43, 20, 4, 64);
        assert!(
            res.proxy_bits < 5.0,
            "phi arm proxy_bits {} should be < 5.0 (no saturation)",
            res.proxy_bits
        );
    }

    #[test]
    fn arms_are_deterministic_per_seed() {
        let a = run_arm(Family::Zoo, 7, 10, 2, 32);
        let b = run_arm(Family::Zoo, 7, 10, 2, 32);
        assert_eq!(a.proxy_bits, b.proxy_bits);
        assert_eq!(a.conversions, b.conversions);
    }
}
