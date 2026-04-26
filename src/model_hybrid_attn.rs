<<<<<<< HEAD
#![allow(
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::field_reassign_with_default,
    clippy::doc_overindented_list_items,
    dead_code
)]
//! # Hybrid Attention Block — Gate-2 + Gate-final Architecture (L-h2 → L-f1)
//!
//! Causal self-attention layers used by the hybrid ngram+attn trainer
//! ([`crate::bin::hybrid_train`]).  Supports 1 or 2 attention layers
//! behind `cfg.num_attn_layers` (default 2 per Gate-final pre-reg DRAFT).
//!
//! ## Pre-registration
//!
//! This module is authored against the **immutable** Gate-2 pre-registration
//! comment on [trios#143](https://github.com/gHashTag/trios/issues/143#issuecomment-4320342032)
//! (lane L-h5 DONE) and extended by the Gate-final DRAFT (L-f1).
//! Any deviation from the published values below must appear as a *new*
//! comment on #143 **cited from the deviating commit before** the data is
//! collected (Rule R5).
//!
//! ## Constants (Coq-grounded, L-R14)
//!
//! | Constant              | Value                        | Source                                          |
//! |-----------------------|------------------------------|-------------------------------------------------|
//! | `PHI_SQ`              | `2.618033988749895`          | [`crate::invariants::PHI_SQ`] (`lr_convergence.v::phi_cube`) |
//! | `PHI_CUBE`            | `4.23606797749979`           | [`crate::invariants::PHI_CUBE`]                  |
//! | `LR_SAFE_MIN`         | `0.002`                      | [`crate::invariants::LR_SAFE_MIN`] (INV-1)       |
//! | `LR_SAFE_MAX`         | `0.007`                      | [`crate::invariants::LR_SAFE_MAX`] (INV-1)       |
//! | `ALLOWED_QK_GAINS`    | `{PHI_SQ, PHI_CUBE}`         | INV-13 (this module)                             |
//!
//! ## Falsification (R7)
//!
//! The block refuses to construct itself when any of the following hold:
//!
//! 1. `lr ∉ [LR_SAFE_MIN, LR_SAFE_MAX]` → [`HybridAttnError::LrOutOfBand`]
//! 2. `qk_gain ∉ {PHI_SQ, PHI_CUBE}`    → [`HybridAttnError::QkGainOutsidePhi`]
//! 3. `d_model == 0` or `num_heads == 0` or `d_model % num_heads != 0`
//!                                      → [`HybridAttnError::Shape`]
//! 4. Non-finite input in the forward pass → [`HybridAttnError::NonFinite`]
//!
//! Each of these corresponds to a named falsifier test at the bottom of this
//! file.  Deleting or weakening a test is a pre-registration deviation and
//! must be filed as described above.
//!
//! ## Scope
//!
//! This file is the **single** file owned by L-h2.  It is called by
//! `hybrid_train.rs` (L-h1) but owns **no** pre-existing module.  Per R6
//! (lane discipline), the only out-of-file touch is a one-line
//! `pub mod hybrid_attn;` re-export in [`crate::lib`].

#![allow(clippy::too_many_arguments)]

// use crate::invariants::{LR_SAFE_MAX, LR_SAFE_MIN, PHI_CUBE, PHI_SQ};
// Placeholder phi constants for L-T1 (TODO: replace with invariants import)
const PHI: f64 = 1.618033988749895;
const PHI_SQ: f64 = PHI * PHI;
const PHI_CUBE: f64 = PHI * PHI * PHI;
const LR_SAFE_MIN: f64 = 0.002;
const LR_SAFE_MAX: f64 = 0.007;

// use crate::invariants::{LR_SAFE_MAX, LR_SAFE_MIN, PHI_CUBE, PHI_SQ};
// TODO: Uncomment above when `trios-igla-race` is available via `--features trios-integration`

// ═══════════════════════════════════════════════════════════════════════
// INV-13 — Allowed qk_gain values
// Pre-registered: qk_gain ∈ {φ², φ³}.
// Coq lemma (L-h4): trinity-clara/proofs/igla/hybrid_qk_gain.v
//     ::counter_qk_gain_outside_phi_sq
// ═══════════════════════════════════════════════════════════════════

/// Allowed quarks-gain values for the causal attention block.
///
/// Pre-registered as `{φ², φ³}`.  Any other value is refused at construction.
pub const ALLOWED_QK_GAINS: [f64; 2] = [PHI_SQ, PHI_CUBE];

/// Pre-registered default qk_gain for Gate-2: φ².
pub const DEFAULT_QK_GAIN: f64 = PHI_SQ;

/// Pre-registered default learning rate for Gate-2: 0.0035 (inside the
/// INV-1 band `[0.002, 0.007]`).
pub const DEFAULT_LR: f64 = 0.0035;

// ═══════════════════════════════════════════════════════════════════
// Error type
// ═══════════════════════════════════════════════════════════════════

/// Construction / forward-pass refusals.
///
/// Every variant has a corresponding falsifier test.  Never silence a
/// variant — surface it as `Result::Err` so the trainer lane (L-h1) can
/// record the refusal in the race ledger.
#[derive(Debug, Clone, PartialEq)]
pub enum HybridAttnError {
    /// `lr ∉ [LR_SAFE_MIN, LR_SAFE_MAX]` — INV-1 violation.
    LrOutOfBand { lr: f64 },
    /// `qk_gain ∉ {PHI_SQ, PHI_CUBE}` — INV-13 violation (pre-registered).
    QkGainOutsidePhi { qk_gain: f64 },
    /// Shape invariants failed (zero dimension or indivisible head split).
    Shape { d_model: usize, num_heads: usize },
    /// Non-finite tensor detected in forward pass.
=======
//! # Hybrid Attention Block — Gate-2 + Gate-final Architecture (L-h2 → L-f1)
//!
//! Causal self-attention layers used by the hybrid ngram+attn trainer.
//! Supports 1 or 2 attention layers behind `cfg.num_attn_layers` (default 2).

#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::doc_overindented_list_items)]

use crate::invariants::{LR_SAFE_MAX, LR_SAFE_MIN, PHI_CUBE, PHI_SQ};

pub const ALLOWED_QK_GAINS: [f64; 2] = [PHI_SQ, PHI_CUBE];

pub const DEFAULT_QK_GAIN: f64 = PHI_SQ;

pub const DEFAULT_LR: f64 = 0.0035;

#[derive(Debug, Clone, PartialEq)]
pub enum HybridAttnError {
    LrOutOfBand { lr: f64 },
    QkGainOutsidePhi { qk_gain: f64 },
    Shape { d_model: usize, num_heads: usize },
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    NonFinite,
}

impl std::fmt::Display for HybridAttnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LrOutOfBand { lr } => write!(
                f,
                "INV-1 violation: lr={lr} outside φ-safe band [{LR_SAFE_MIN}, {LR_SAFE_MAX}]",
            ),
            Self::QkGainOutsidePhi { qk_gain } => write!(
                f,
                "INV-13 violation: qk_gain={qk_gain} not in pre-registered \
                 set {{φ²={PHI_SQ}, φ³={PHI_CUBE}}}",
            ),
<<<<<<< HEAD
            Self::Shape { d_model, num_heads } => write!(
=======
            Self::Shape {
                d_model,
                num_heads,
            } => write!(
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
                f,
                "shape invariant failed: d_model={d_model}, num_heads={num_heads} \
                 (both must be > 0 and d_model % num_heads == 0)",
            ),
            Self::NonFinite => write!(f, "non-finite tensor in forward pass"),
        }
    }
}

impl std::error::Error for HybridAttnError {}

<<<<<<< HEAD
// ═══════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════

/// Pre-registered Gate-2 shape: `d_model=64`, `num_heads=4`, `seq_len=8`.
///
/// These are the numbers published in the pre-registration comment §2.
/// Gate-final DRAFT adds `num_attn_layers: u8` (default 2, L-f1).
#[derive(Debug, Clone, Copy)]
pub struct HybridAttnConfig {
    /// Model dimension (must be a multiple of `num_heads`).
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Maximum sequence length handled by RoPE.
    pub seq_len: usize,
    /// Query/key scaling gain — **must** be in [`ALLOWED_QK_GAINS`].
    pub qk_gain: f64,
    /// Learning rate — **must** be in `[LR_SAFE_MIN, LR_SAFE_MAX]`.
    pub lr: f64,
    /// Number of attention layers — **must** be in `{1, 2}` (Gate-final §8).
=======
#[derive(Debug, Clone, Copy)]
pub struct HybridAttnConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub seq_len: usize,
    pub qk_gain: f64,
    pub lr: f64,
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    pub num_attn_layers: u8,
}

impl Default for HybridAttnConfig {
    fn default() -> Self {
        Self {
            d_model: 64,
            num_heads: 4,
            seq_len: 8,
            qk_gain: DEFAULT_QK_GAIN,
            lr: DEFAULT_LR,
            num_attn_layers: 2,
        }
    }
}

impl HybridAttnConfig {
<<<<<<< HEAD
    /// Validate this config against INV-1, INV-13, and the shape invariants.
    ///
    /// This is the central chokepoint: every public constructor routes
    /// through here so a single inspection audits all refusal paths.
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    pub fn validate(&self) -> Result<(), HybridAttnError> {
        if !(LR_SAFE_MIN..=LR_SAFE_MAX).contains(&self.lr) {
            return Err(HybridAttnError::LrOutOfBand { lr: self.lr });
        }
        if !ALLOWED_QK_GAINS
            .iter()
            .any(|g| (g - self.qk_gain).abs() < 1e-9)
        {
            return Err(HybridAttnError::QkGainOutsidePhi {
                qk_gain: self.qk_gain,
            });
        }
<<<<<<< HEAD
        if self.d_model == 0 || self.num_heads == 0 || self.d_model % self.num_heads != 0 {
=======
        if self.d_model == 0
            || self.num_heads == 0
            || !self.d_model.is_multiple_of(self.num_heads)
        {
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
            return Err(HybridAttnError::Shape {
                d_model: self.d_model,
                num_heads: self.num_heads,
            });
        }
        if !(self.num_attn_layers == 1 || self.num_attn_layers == 2) {
            return Err(HybridAttnError::Shape {
                d_model: self.num_attn_layers as usize,
                num_heads: 0,
            });
        }
        Ok(())
    }
}

<<<<<<< HEAD
// ═══════════════════════════════════════════════════════════════════
// The block itself
// ═══════════════════════════════════════════════════════════════════

/// Weights are stored row-major.  Supports 1 or 2 attention layers.
/// Layer 2 shares RoPE with layer 1 (per Gate-final DRAFT §6 lever 1).
/// Residual + LayerNorm between layers.
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
#[derive(Debug, Clone)]
pub struct HybridAttn {
    cfg: HybridAttnConfig,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    wq2: Vec<f32>,
    wk2: Vec<f32>,
    wv2: Vec<f32>,
    wo2: Vec<f32>,
}

impl HybridAttn {
<<<<<<< HEAD
    /// Construct with the pre-registered defaults (`φ²`, `lr=0.0035`,
    /// `d_model=64`, `num_heads=4`).
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    pub fn new() -> Result<Self, HybridAttnError> {
        Self::with_config(HybridAttnConfig::default())
    }

<<<<<<< HEAD
    /// Construct with an explicit learning rate (all other values default).
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    pub fn new_with_lr(lr: f64) -> Result<Self, HybridAttnError> {
        let mut cfg = HybridAttnConfig::default();
        cfg.lr = lr;
        Self::with_config(cfg)
    }

<<<<<<< HEAD
    /// Construct with an explicit qk_gain (all other values default).
    ///
    /// This refuses at construction time, **not** inside the forward pass —
    /// silent acceptance of a bad gain is a pre-registration violation.
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    pub fn new_with_qk_gain(qk_gain: f64) -> Result<Self, HybridAttnError> {
        let mut cfg = HybridAttnConfig::default();
        cfg.qk_gain = qk_gain;
        Self::with_config(cfg)
    }

<<<<<<< HEAD
    /// Construct with a full config.
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    pub fn with_config(cfg: HybridAttnConfig) -> Result<Self, HybridAttnError> {
        cfg.validate()?;
        let d = cfg.d_model;
        let dd = d * d;
        Ok(Self {
            cfg,
            wq: vec![0.0_f32; dd],
            wk: vec![0.0_f32; dd],
            wv: vec![0.0_f32; dd],
            wo: vec![0.0_f32; dd],
            wq2: vec![0.0_f32; dd],
            wk2: vec![0.0_f32; dd],
            wv2: vec![0.0_f32; dd],
            wo2: vec![0.0_f32; dd],
        })
    }

<<<<<<< HEAD
    /// The pre-registered config.  Callers that need to re-assert
    /// invariants (e.g. the CI gate in L-h1) should use this accessor
    /// instead of clone-unwrapping internal fields.
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    pub fn config(&self) -> &HybridAttnConfig {
        &self.cfg
    }

<<<<<<< HEAD
    /// Re-assert INV-1 + INV-13 + shape at any later point.  This is
    /// cheap and idempotent, and the trainer calls it once per step as
    /// an online invariant check.
=======
    pub fn wq_mut(&mut self) -> &mut [f32] {
        &mut self.wq
    }

    pub fn wk_mut(&mut self) -> &mut [f32] {
        &mut self.wk
    }

    pub fn wv_mut(&mut self) -> &mut [f32] {
        &mut self.wv
    }

    pub fn wo_mut(&mut self) -> &mut [f32] {
        &mut self.wo
    }

    pub fn weights_flat_mut(&mut self) -> Vec<&mut [f32]> {
        vec![&mut self.wq, &mut self.wk, &mut self.wv, &mut self.wo]
    }

    pub fn total_weights(&self) -> usize {
        self.wq.len() + self.wk.len() + self.wv.len() + self.wo.len()
    }

>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    pub fn reassert(&self) -> Result<(), HybridAttnError> {
        self.cfg.validate()
    }

<<<<<<< HEAD
    // --- RoPE -----------------------------------------------------------

    /// RoPE angle for position `p` and head-dim index `i` (`0 ≤ i < d_head/2`).
    ///
    /// We use the classical formula `θ = p / 10000^{2i / d_head}`, which
    /// has the φ-periodicity property required by INV-9 (see the
    /// `hybrid_attn_rope_periodicity` test for the concrete bound).
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    pub fn rope_angle(position: usize, head_dim_idx: usize, d_head: usize) -> f32 {
        assert!(d_head > 0, "INV: d_head must be positive");
        assert!(
            head_dim_idx < d_head / 2,
            "INV: head_dim_idx {head_dim_idx} must be < d_head/2 = {}",
            d_head / 2,
        );
        let exp = (2.0 * head_dim_idx as f32) / (d_head as f32);
        (position as f32) / 10_000.0_f32.powf(exp)
    }

<<<<<<< HEAD
    // --- Forward pass ---------------------------------------------------

    /// Single-step causal attention forward pass on a batch of
    /// `seq_len × d_model` tokens.  Returns the post-output-projection
    /// activations of the same shape, flattened row-major.
    ///
    /// The pass is written straightforwardly: clarity beats speed in the
    /// pre-registered block, because the measured quantity is the
    /// learning dynamic (`val_bpb_at_step_54000`) not wall-clock.
    /// Optimisation lives downstream in `hybrid_train.rs` (L-h1).
    pub fn forward(&self, tokens: &[f32], seq_len: usize) -> Result<Vec<f32>, HybridAttnError> {
=======
    pub fn forward(
        &self,
        tokens: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>, HybridAttnError> {
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
        if tokens.iter().any(|x| !x.is_finite()) {
            return Err(HybridAttnError::NonFinite);
        }
        let d = self.cfg.d_model;
        assert_eq!(
            tokens.len(),
            seq_len * d,
            "forward: tokens.len() = {} but expected seq_len * d_model = {}",
            tokens.len(),
            seq_len * d,
        );

<<<<<<< HEAD
        let layer1_out =
            self.forward_single_layer(tokens, seq_len, &self.wq, &self.wk, &self.wv, &self.wo)?;
=======
        let layer1_out = self.forward_single_layer(tokens, seq_len, &self.wq, &self.wk, &self.wv, &self.wo)?;
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
        let residual1 = add_residual(tokens, &layer1_out);
        let normed1 = layer_norm_rows(&residual1, seq_len, d);

        if self.cfg.num_attn_layers == 1 {
            if normed1.iter().any(|x| !x.is_finite()) {
                return Err(HybridAttnError::NonFinite);
            }
            return Ok(normed1);
        }

<<<<<<< HEAD
        let layer2_out = self.forward_single_layer(
            &normed1, seq_len, &self.wq2, &self.wk2, &self.wv2, &self.wo2,
        )?;
=======
        let layer2_out = self.forward_single_layer(&normed1, seq_len, &self.wq2, &self.wk2, &self.wv2, &self.wo2)?;
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
        let residual2 = add_residual(&normed1, &layer2_out);
        let out = layer_norm_rows(&residual2, seq_len, d);

        if out.iter().any(|x| !x.is_finite()) {
            return Err(HybridAttnError::NonFinite);
        }
        Ok(out)
    }

    fn forward_single_layer(
        &self,
        tokens: &[f32],
        seq_len: usize,
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
        wo: &[f32],
    ) -> Result<Vec<f32>, HybridAttnError> {
        let d = self.cfg.d_model;
        let h = self.cfg.num_heads;
        let d_head = d / h;

        let q = matmul(tokens, wq, seq_len, d, d);
        let k = matmul(tokens, wk, seq_len, d, d);
        let v = matmul(tokens, wv, seq_len, d, d);

        let scale = (d_head as f32).sqrt();
        let mut attn_out = vec![0.0_f32; seq_len * d];
        for head in 0..h {
            let head_offset = head * d_head;
            for i in 0..seq_len {
                let mut scores = vec![0.0_f32; i + 1];
                for (j, score) in scores.iter_mut().enumerate() {
                    let mut s = 0.0_f32;
                    for k_idx in 0..d_head {
                        let qv = q[i * d + head_offset + k_idx];
                        let kv = k[j * d + head_offset + k_idx];
                        s += qv * kv;
                    }
                    *score = (self.cfg.qk_gain as f32) * s / scale;
                }
                softmax_inplace(&mut scores);
                for j in 0..=i {
                    let w = scores[j];
                    for k_idx in 0..d_head {
<<<<<<< HEAD
                        attn_out[i * d + head_offset + k_idx] += w * v[j * d + head_offset + k_idx];
=======
                        attn_out[i * d + head_offset + k_idx] +=
                            w * v[j * d + head_offset + k_idx];
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
                    }
                }
            }
        }

        let out = matmul(&attn_out, wo, seq_len, d, d);
        if out.iter().any(|x| !x.is_finite()) {
            return Err(HybridAttnError::NonFinite);
        }
        Ok(out)
    }
}

<<<<<<< HEAD
// ═══════════════════════════════════════════════════════════════════
// Helpers (kept private; test-visible via the `HybridAttn::forward` call)
// ═══════════════════════════════════════════════════════════════════

=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "matmul lhs shape");
    assert_eq!(b.len(), k * n, "matmul rhs shape");
    let mut out = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f32;
            for l in 0..k {
                s += a[i * k + l] * b[l * n + j];
            }
            out[i * n + j] = s;
        }
    }
    out
}

fn add_residual(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "add_residual shape mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn layer_norm_rows(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * cols, "layer_norm_rows shape");
    let eps = 1e-5_f32;
    let mut out = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let n = cols as f32;
        let mean = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let std_inv = 1.0 / (var + eps).sqrt();
        for c in 0..cols {
            out[r * cols + c] = (row[c] - mean) * std_inv;
        }
    }
    out
}

fn softmax_inplace(v: &mut [f32]) {
    let max_val = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in v.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

<<<<<<< HEAD
// ═══════════════════════════════════════════════════════════════════
// Falsifier tests — R7 witnesses for INV-1, INV-13, shape, and forward
// ═══════════════════════════════════════════════════════════════════

=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
#[cfg(test)]
mod falsifiers {
    use super::*;
    use crate::invariants::PHI;

<<<<<<< HEAD
    /// R7 / INV-1: a learning rate outside the Coq-proven φ-band must
    /// refuse at construction time.  This is the deterministic sibling
    /// of the earlier pure-attention plateau (BPB ≈ 4.74 @ lr=0.01).
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    #[test]
    fn falsify_hybrid_diverges_bad_lr() {
        let err = HybridAttn::new_with_lr(0.02).unwrap_err();
        assert!(
            matches!(err, HybridAttnError::LrOutOfBand { .. }),
            "expected LrOutOfBand, got {err:?}",
        );
<<<<<<< HEAD
        // Lower-side witness.
        let err = HybridAttn::new_with_lr(0.0005).unwrap_err();
        assert!(matches!(err, HybridAttnError::LrOutOfBand { .. }));
        // And the inside-band default must succeed.
        HybridAttn::new_with_lr(0.0035).expect("0.0035 is inside the band");
    }

    /// R7 / INV-13: any qk_gain outside `{φ², φ³}` must refuse.  This is
    /// the Rust mirror of the pre-registered Coq lemma
    /// `counter_qk_gain_outside_phi_sq` (L-h4).
=======
        let err = HybridAttn::new_with_lr(0.0005).unwrap_err();
        assert!(matches!(err, HybridAttnError::LrOutOfBand { .. }));
        HybridAttn::new_with_lr(0.0035).expect("0.0035 is inside the band");
    }

>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    #[test]
    fn falsify_hybrid_qk_gain_not_phi_sq_or_phi_cube() {
        let err = HybridAttn::new_with_qk_gain(PHI).unwrap_err();
        assert!(
            matches!(err, HybridAttnError::QkGainOutsidePhi { .. }),
            "qk_gain=PHI must be refused, got {err:?}",
        );
        let err = HybridAttn::new_with_qk_gain(1.0).unwrap_err();
        assert!(matches!(err, HybridAttnError::QkGainOutsidePhi { .. }));
<<<<<<< HEAD
        // Both pre-registered gains must succeed.
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
        HybridAttn::new_with_qk_gain(PHI_SQ).expect("φ² is allowed");
        HybridAttn::new_with_qk_gain(PHI_CUBE).expect("φ³ is allowed");
    }

<<<<<<< HEAD
    /// Shape invariant: `d_model % num_heads != 0` must refuse.
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    #[test]
    fn falsify_hybrid_shape_invariant() {
        let cfg = HybridAttnConfig {
            d_model: 64,
<<<<<<< HEAD
            num_heads: 5, // 64 % 5 = 4 ≠ 0
=======
            num_heads: 5,
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
            ..HybridAttnConfig::default()
        };
        let err = HybridAttn::with_config(cfg).unwrap_err();
        assert!(matches!(err, HybridAttnError::Shape { .. }));
    }

<<<<<<< HEAD
    /// Deterministic forward pass: zero weights on zero tokens must
    /// return zeros (no NaN, no Inf).  The goal is to exercise the
    /// non-finite detector on a known-good input.
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    #[test]
    fn hybrid_attn_forward_roundtrip() {
        let block = HybridAttn::new().expect("defaults are valid");
        let seq_len = 4;
        let d = block.config().d_model;
        let tokens = vec![0.0_f32; seq_len * d];
        let out = block.forward(&tokens, seq_len).unwrap();
        assert_eq!(out.len(), seq_len * d);
        assert!(out.iter().all(|x| x.is_finite()));
    }

<<<<<<< HEAD
    /// Non-finite input must be surfaced as `Err(NonFinite)`, not
    /// propagated silently.  R5: honest refusal.
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    #[test]
    fn hybrid_attn_non_finite_refused() {
        let block = HybridAttn::new().expect("defaults are valid");
        let seq_len = 2;
        let d = block.config().d_model;
        let mut tokens = vec![0.0_f32; seq_len * d];
        tokens[0] = f32::NAN;
        let err = block.forward(&tokens, seq_len).unwrap_err();
        assert_eq!(err, HybridAttnError::NonFinite);
    }

<<<<<<< HEAD
    /// RoPE periodicity: for `d_head = 16`, the ratio between the
    /// frequency at index 0 and index 7 is exactly `10_000^{14/16}`.
    /// This property is the INV-9 φ-anchor hook — the actual φ-relation
    /// is proven in the Coq lemma, not re-asserted here.
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    #[test]
    fn hybrid_attn_rope_periodicity() {
        let d_head = 16;
        let a0 = HybridAttn::rope_angle(1, 0, d_head);
        let a7 = HybridAttn::rope_angle(1, 7, d_head);
        let ratio = a0 / a7;
        let expected = 10_000.0_f32.powf(14.0 / 16.0);
        assert!(
            (ratio - expected).abs() < 1e-2,
            "RoPE frequency ratio drifted: got {ratio}, expected {expected}",
        );
    }

<<<<<<< HEAD
    /// `reassert()` must stay green for the default config.  This is
    /// called inside L-h1's training loop; regressing it breaks the
    /// online invariant sweep.
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    #[test]
    fn hybrid_attn_reassert_stable() {
        let block = HybridAttn::new().expect("defaults are valid");
        for _ in 0..8 {
            block.reassert().expect("online reassertion must hold");
        }
    }

<<<<<<< HEAD
    /// L-f1 Gate-final: 2-layer forward pass with residual + LayerNorm
    /// must produce finite output on zero-initialized weights.
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    #[test]
    fn twin_attn_2layer_forward_roundtrip() {
        let block = HybridAttn::new().expect("defaults are valid (num_attn_layers=2)");
        assert_eq!(block.config().num_attn_layers, 2);
        let seq_len = 4;
        let d = block.config().d_model;
        let tokens = vec![0.0_f32; seq_len * d];
        let out = block.forward(&tokens, seq_len).unwrap();
        assert_eq!(out.len(), seq_len * d);
<<<<<<< HEAD
        assert!(
            out.iter().all(|x| x.is_finite()),
            "2-layer output must be finite"
        );
    }

    /// L-f1 Gate-final: 1-layer mode must still work (backward compat).
=======
        assert!(out.iter().all(|x| x.is_finite()), "2-layer output must be finite");
    }

>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    #[test]
    fn twin_attn_1layer_forward_roundtrip() {
        let cfg = HybridAttnConfig {
            num_attn_layers: 1,
            ..HybridAttnConfig::default()
        };
        let block = HybridAttn::with_config(cfg).expect("1-layer config valid");
        assert_eq!(block.config().num_attn_layers, 1);
        let seq_len = 4;
        let d = block.config().d_model;
        let tokens = vec![0.0_f32; seq_len * d];
        let out = block.forward(&tokens, seq_len).unwrap();
        assert_eq!(out.len(), seq_len * d);
        assert!(out.iter().all(|x| x.is_finite()));
    }

<<<<<<< HEAD
    /// L-f1 Gate-final: num_attn_layers > 2 is forbidden (§8).
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    #[test]
    fn falsify_invalid_num_attn_layers() {
        let cfg = HybridAttnConfig {
            num_attn_layers: 3,
            ..HybridAttnConfig::default()
        };
        let err = HybridAttn::with_config(cfg).unwrap_err();
        assert!(
            matches!(err, HybridAttnError::Shape { .. }),
            "num_attn_layers=3 must be refused, got {err:?}"
        );
        let cfg0 = HybridAttnConfig {
            num_attn_layers: 0,
            ..HybridAttnConfig::default()
        };
        let err0 = HybridAttn::with_config(cfg0).unwrap_err();
        assert!(matches!(err0, HybridAttnError::Shape { .. }));
    }

<<<<<<< HEAD
    /// L-f1 Gate-final: non-finite input rejected in 2-layer mode.
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    #[test]
    fn twin_attn_2layer_nonfinite_refused() {
        let block = HybridAttn::new().expect("defaults valid");
        let seq_len = 2;
        let d = block.config().d_model;
        let mut tokens = vec![0.0_f32; seq_len * d];
        tokens[0] = f32::NAN;
        let err = block.forward(&tokens, seq_len).unwrap_err();
        assert_eq!(err, HybridAttnError::NonFinite);
    }

<<<<<<< HEAD
    /// L-f1 Gate-final witness: qk_gain outside φ-band refused
    /// (Gate-final §2 falsifier 4).  Re-asserts for the DRAFT context.
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    #[test]
    fn falsify_invalid_qk_gain() {
        for bad in [1.0, 1.5, 2.0, 3.0, 5.0] {
            let err = HybridAttn::new_with_qk_gain(bad).unwrap_err();
            assert!(
                matches!(err, HybridAttnError::QkGainOutsidePhi { .. }),
                "qk_gain={bad} must be refused"
            );
        }
    }
}
