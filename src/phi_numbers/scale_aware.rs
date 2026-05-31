//! Scale-aware ternary quantization (F2 breadth-as-moat harness).
//!
//! STATUS DISCIPLINE (see skill `goldenfloat-ladder`):
//! Only `phi^2 + phi^-2 = 3` (Lucas L2, 1878) is [Verified]. The breadth /
//! toolchain-coherence advantage exercised by this module is an
//! [Open conjecture] with a falsification path (FL-002): it is falsified if a
//! posit / takum / MX ladder, or any prior width-spanning float family,
//! matches the phi-ladder at a matched bit budget. Nothing here promotes the
//! moat to [Verified], and any in-sandbox proxy verdict is NOT a Verdict.
//!
//! WHY THIS EXISTS. The existing enum `super::gfternary::GFTernary` snaps every
//! input through a FIXED threshold `phi/2 ~= 0.809` onto `{-phi, 0, +phi}`. With
//! no per-tensor scaling, micro-scale tensors (all entries << 0.809) collapse to
//! Zero, and anything above the threshold jumps to the +-phi ~= +-1.618 scale.
//! That scale mismatch is the root cause of saturation at log2(vocab) BPB.
//!
//! This module instead does standard absmax-normalized ternary quantization
//! (BitNet b1.58 style, arXiv:2402.17764): per tensor compute alpha = absmax,
//! quantize each element with q(x) = round(clip(x/alpha, -1, +1)) in {-1, 0, +1},
//! then dequantize as q * alpha. The ternary *levels* are scale-free integers;
//! the recovered magnitude rides on the per-tensor scale, so the representable
//! range tracks the data and saturation no longer occurs by construction.
//!
//! The existing enum is left untouched (its tests depend on the fixed-threshold
//! behaviour). This is an additive scale-aware path, not a replacement.

/// Per-tensor absolute-maximum scale. Returns `0.0` for an empty or all-zero
/// tensor (the quantizer then maps everything to the zero level, which is the
/// correct lossless behaviour for an all-zero tensor).
#[inline]
pub fn absmax_scale(tensor: &[f32]) -> f32 {
    tensor.iter().fold(0.0_f32, |acc, &x| acc.max(x.abs()))
}

/// Quantize a single value to a ternary level in `{-1, 0, +1}` given a
/// precomputed per-tensor scale `alpha`. `round(clip(x/alpha, -1, +1))`.
///
/// `round_ties_even` is used so the harness is deterministic and matches the
/// `fake_quant` integer path; with a single-magnitude ternary code the only
/// tie that can occur is at `|x/alpha| = 0.5`.
#[inline]
pub fn quantize_level(x: f32, alpha: f32) -> i8 {
    if alpha <= 0.0 || !x.is_finite() {
        return 0;
    }
    let r = (x / alpha).clamp(-1.0, 1.0);
    // round half to even for determinism
    let q = r.round_ties_even();
    q as i8
}

/// Dequantize a ternary level back to f32 at the given per-tensor scale.
#[inline]
pub fn dequantize_level(level: i8, alpha: f32) -> f32 {
    (level as f32) * alpha
}

/// Result of a scale-aware ternary round-trip on a whole tensor: the integer
/// ternary levels, the per-tensor scale, and the dequantized values.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleAwareTernary {
    /// Per-element ternary levels in `{-1, 0, +1}`.
    pub levels: Vec<i8>,
    /// Per-tensor absmax scale used for (de)quantization.
    pub alpha: f32,
}

impl ScaleAwareTernary {
    /// Quantize a tensor: compute `alpha = absmax`, then map each element to a
    /// ternary level.
    pub fn quantize(tensor: &[f32]) -> Self {
        let alpha = absmax_scale(tensor);
        let levels = tensor.iter().map(|&x| quantize_level(x, alpha)).collect();
        Self { levels, alpha }
    }

    /// Dequantize back to f32 magnitudes at the stored scale.
    pub fn dequantize(&self) -> Vec<f32> {
        self.levels
            .iter()
            .map(|&l| dequantize_level(l, self.alpha))
            .collect()
    }

    /// Number of nonzero ternary levels (a sparsity / collapse diagnostic).
    pub fn nonzero_count(&self) -> usize {
        self.levels.iter().filter(|&&l| l != 0).count()
    }
}

/// Convenience: scale-aware ternary round-trip (quantize then dequantize) for a
/// whole tensor in one call.
pub fn fake_quantize_ternary_scale_aware(tensor: &[f32]) -> Vec<f32> {
    ScaleAwareTernary::quantize(tensor).dequantize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn micro_scale_does_not_collapse_to_zero() {
        // All entries are << phi/2 ~= 0.809; the fixed-threshold enum would
        // snap these to Zero. Scale-aware must preserve sign structure.
        let t = vec![0.001_f32, -0.002, 0.0015, -0.0005];
        let q = ScaleAwareTernary::quantize(&t);
        // absmax is 0.002 -> the largest-magnitude entries must be +-1, not 0.
        assert!(q.nonzero_count() >= 2, "micro-scale tensor collapsed: {q:?}");
        // The -0.002 entry is the absmax -> level -1.
        let idx = 1;
        assert_eq!(q.levels[idx], -1);
    }

    #[test]
    fn dequant_magnitude_tracks_scale() {
        // Large-scale tensor: dequantized peak must ride the data scale, not a
        // fixed +-phi.
        let t = vec![100.0_f32, -50.0, 0.0, 25.0];
        let q = ScaleAwareTernary::quantize(&t);
        let dq = q.dequantize();
        assert_eq!(q.alpha, 100.0);
        assert_eq!(dq[0], 100.0); // +1 * 100
        assert_eq!(dq[2], 0.0); // 0 * 100
    }

    #[test]
    fn all_zero_tensor_is_lossless() {
        let t = vec![0.0_f32; 8];
        let q = ScaleAwareTernary::quantize(&t);
        assert_eq!(q.alpha, 0.0);
        assert_eq!(q.nonzero_count(), 0);
        assert_eq!(q.dequantize(), t);
    }

    #[test]
    fn levels_are_strictly_ternary() {
        let t = vec![0.9_f32, -0.4, 0.1, -0.95, 0.5, -0.5];
        let q = ScaleAwareTernary::quantize(&t);
        for &l in &q.levels {
            assert!(l == -1 || l == 0 || l == 1, "non-ternary level {l}");
        }
    }

    #[test]
    fn nonfinite_input_maps_to_zero_level() {
        assert_eq!(quantize_level(f32::NAN, 1.0), 0);
        assert_eq!(quantize_level(f32::INFINITY, 1.0), 0);
    }
}
