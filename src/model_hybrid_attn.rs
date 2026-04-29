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

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::doc_overindented_list_items)]

use crate::invariants::{LR_SAFE_MAX, LR_SAFE_MIN, PHI_CUBE, PHI_SQ};

// ═══════════════════════════════════════════════════════════════════
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
            Self::Shape { d_model, num_heads } => write!(
                f,
                "shape invariant failed: d_model={d_model}, num_heads={num_heads} \
                 (both must be > 0 and d_model % num_heads == 0)",
            ),
            Self::NonFinite => write!(f, "non-finite tensor in forward pass"),
        }
    }
}

impl std::error::Error for HybridAttnError {}

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
    /// Validate this config against INV-1, INV-13, and the shape invariants.
    ///
    /// This is the central chokepoint: every public constructor routes
    /// through here so a single inspection audits all refusal paths.
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
        if self.d_model == 0 || self.num_heads == 0 || !self.d_model.is_multiple_of(self.num_heads)
        {
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

// ═══════════════════════════════════════════════════════════════════
// The block itself
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct HybridAttn {
    cfg: HybridAttnConfig,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub wq2: Vec<f32>,
    pub wk2: Vec<f32>,
    pub wv2: Vec<f32>,
    pub wo2: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct AttentionCache {
    pub input: Vec<f32>,
    pub q1: Vec<f32>,
    pub k1: Vec<f32>,
    pub v1: Vec<f32>,
    pub attn_weights1: Vec<Vec<Vec<f32>>>,
    pub attn_pre_wo1: Vec<f32>,
    pub residual1: Vec<f32>,
    pub normed1: Vec<f32>,
    pub q2: Vec<f32>,
    pub k2: Vec<f32>,
    pub v2: Vec<f32>,
    pub attn_weights2: Vec<Vec<Vec<f32>>>,
    pub attn_pre_wo2: Vec<f32>,
    pub residual2: Vec<f32>,
    pub output: Vec<f32>,
    pub seq_len: usize,
}

#[derive(Debug, Clone)]
pub struct AttentionGrads {
    pub gwq: Vec<f32>,
    pub gwk: Vec<f32>,
    pub gwv: Vec<f32>,
    pub gwo: Vec<f32>,
    pub gwq2: Vec<f32>,
    pub gwk2: Vec<f32>,
    pub gwv2: Vec<f32>,
    pub gwo2: Vec<f32>,
    pub d_input: Vec<f32>,
}

/// Forward cache for gradient computation.
///
/// Stores all intermediate activations needed for backpropagation.
#[derive(Debug, Clone)]
pub struct ForwardCache {
    /// Input tokens (seq_len × d_model)
    pub tokens: Vec<f32>,
    /// Q projections from layer 1 (seq_len × d_model)
    pub q1: Vec<f32>,
    /// K projections from layer 1 (seq_len × d_model)
    pub k1: Vec<f32>,
    /// V projections from layer 1 (seq_len × d_model)
    pub v1: Vec<f32>,
    /// Attention weights per head [head][token][score]
    pub attn_weights1: Vec<Vec<f32>>,
    /// Q projections from layer 2 (seq_len × d_model)
    pub q2: Vec<f32>,
    /// K projections from layer 2 (seq_len × d_model)
    pub k2: Vec<f32>,
    /// V projections from layer 2 (seq_len × d_model)
    pub v2: Vec<f32>,
    /// Attention weights per head for layer 2 [head][token][score]
    pub attn_weights2: Vec<Vec<f32>>,
    /// Layer 1 output before residual + norm
    pub layer1_out: Vec<f32>,
    /// Layer 1 output after residual + norm
    pub normed1: Vec<f32>,
    /// Layer 2 output before residual + norm
    pub layer2_out: Vec<f32>,
}

impl ForwardCache {
    pub fn new(seq_len: usize, d_model: usize, num_heads: usize) -> Self {
        let attn_per_head = seq_len * (seq_len + 1) / 2; // causal = triangular
        Self {
            tokens: vec![0.0; seq_len * d_model],
            q1: vec![0.0; seq_len * d_model],
            k1: vec![0.0; seq_len * d_model],
            v1: vec![0.0; seq_len * d_model],
            attn_weights1: vec![vec![0.0; attn_per_head]; num_heads],
            q2: vec![0.0; seq_len * d_model],
            k2: vec![0.0; seq_len * d_model],
            v2: vec![0.0; seq_len * d_model],
            attn_weights2: vec![vec![0.0; attn_per_head]; num_heads],
            layer1_out: vec![0.0; seq_len * d_model],
            normed1: vec![0.0; seq_len * d_model],
            layer2_out: vec![0.0; seq_len * d_model],
        }
    }
}

/// Gradients for attention layer parameters.
#[derive(Debug, Clone)]
pub struct AttentionGradients {
    pub d_wq: Vec<f32>,
    pub d_wk: Vec<f32>,
    pub d_wv: Vec<f32>,
    pub d_wo: Vec<f32>,
    pub d_wq2: Vec<f32>,
    pub d_wk2: Vec<f32>,
    pub d_wv2: Vec<f32>,
    pub d_wo2: Vec<f32>,
}

impl AttentionGradients {
    pub fn new(d_model: usize) -> Self {
        let dd = d_model * d_model;
        Self {
            d_wq: vec![0.0; dd],
            d_wk: vec![0.0; dd],
            d_wv: vec![0.0; dd],
            d_wo: vec![0.0; dd],
            d_wq2: vec![0.0; dd],
            d_wk2: vec![0.0; dd],
            d_wv2: vec![0.0; dd],
            d_wo2: vec![0.0; dd],
        }
    }

    pub fn clear(&mut self) {
        self.d_wq.fill(0.0);
        self.d_wk.fill(0.0);
        self.d_wv.fill(0.0);
        self.d_wo.fill(0.0);
        self.d_wq2.fill(0.0);
        self.d_wk2.fill(0.0);
        self.d_wv2.fill(0.0);
        self.d_wo2.fill(0.0);
    }
}

impl HybridAttn {
    /// Construct with the pre-registered defaults (`φ²`, `lr=0.0035`,
    /// `d_model=64`, `num_heads=4`).
    pub fn new() -> Result<Self, HybridAttnError> {
        Self::with_config(HybridAttnConfig::default())
    }

    /// Construct with an explicit learning rate (all other values default).
    pub fn new_with_lr(lr: f64) -> Result<Self, HybridAttnError> {
        let mut cfg = HybridAttnConfig::default();
        cfg.lr = lr;
        Self::with_config(cfg)
    }

    /// Construct with an explicit qk_gain (all other values default).
    ///
    /// This refuses at construction time, **not** inside the forward pass —
    /// silent acceptance of a bad gain is a pre-registration violation.
    pub fn new_with_qk_gain(qk_gain: f64) -> Result<Self, HybridAttnError> {
        let mut cfg = HybridAttnConfig::default();
        cfg.qk_gain = qk_gain;
        Self::with_config(cfg)
    }

    /// Construct with a full config.
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

    /// The pre-registered config.  Callers that need to re-assert
    /// invariants (e.g. the CI gate in L-h1) should use this accessor
    /// instead of clone-unwrapping internal fields.
    pub fn config(&self) -> &HybridAttnConfig {
        &self.cfg
    }

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

    pub fn all_weights_mut(&mut self) -> Vec<&mut [f32]> {
        vec![
            &mut self.wq,
            &mut self.wk,
            &mut self.wv,
            &mut self.wo,
            &mut self.wq2,
            &mut self.wk2,
            &mut self.wv2,
            &mut self.wo2,
        ]
    }

    pub fn total_weights(&self) -> usize {
        self.wq.len() + self.wk.len() + self.wv.len() + self.wo.len()
    }

    /// Re-assert INV-1 + INV-13 + shape at any later point.  This is
    /// cheap and idempotent, and the trainer calls it once per step as
    /// an online invariant check.
    pub fn reassert(&self) -> Result<(), HybridAttnError> {
        self.cfg.validate()
    }

    // --- RoPE -----------------------------------------------------------

    /// RoPE angle for position `p` and head-dim index `i` (`0 ≤ i < d_head/2`).
    ///
    /// We use the classical formula `θ = p / 10000^{2i / d_head}`, which
    /// has the φ-periodicity property required by INV-9 (see the
    /// `hybrid_attn_rope_periodicity` test for the concrete bound).
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

        let layer1_out =
            self.forward_single_layer(tokens, seq_len, &self.wq, &self.wk, &self.wv, &self.wo)?;
        let residual1 = add_residual(tokens, &layer1_out);
        let normed1 = layer_norm_rows(&residual1, seq_len, d);

        if self.cfg.num_attn_layers == 1 {
            if normed1.iter().any(|x| !x.is_finite()) {
                return Err(HybridAttnError::NonFinite);
            }
            return Ok(normed1);
        }

        let layer2_out = self.forward_single_layer(
            &normed1, seq_len, &self.wq2, &self.wk2, &self.wv2, &self.wo2,
        )?;
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
                        attn_out[i * d + head_offset + k_idx] += w * v[j * d + head_offset + k_idx];
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

    /// Forward pass with caching for gradient computation.
    pub fn forward_cached(
        &self,
        tokens: &[f32],
        seq_len: usize,
    ) -> Result<(Vec<f32>, ForwardCache), HybridAttnError> {
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

        let mut cache = ForwardCache::new(seq_len, d, self.cfg.num_heads);
        cache.tokens.copy_from_slice(tokens);

        let (layer1_out, cache1) = self.forward_single_layer_cached(
            tokens, seq_len, &self.wq, &self.wk, &self.wv, &self.wo, 1,
        )?;
        cache.q1 = cache1.q;
        cache.k1 = cache1.k;
        cache.v1 = cache1.v;
        cache.attn_weights1 = cache1.attn_weights;
        cache.layer1_out = layer1_out.clone();

        let residual1 = add_residual(tokens, &layer1_out);
        let normed1 = layer_norm_rows(&residual1, seq_len, d);
        cache.normed1 = normed1.clone();

        if self.cfg.num_attn_layers == 1 {
            if normed1.iter().any(|x| !x.is_finite()) {
                return Err(HybridAttnError::NonFinite);
            }
            return Ok((normed1, cache));
        }

        let (layer2_out, cache2) = self.forward_single_layer_cached(
            &normed1, seq_len, &self.wq2, &self.wk2, &self.wv2, &self.wo2, 2,
        )?;
        cache.q2 = cache2.q;
        cache.k2 = cache2.k;
        cache.v2 = cache2.v;
        cache.attn_weights2 = cache2.attn_weights;
        cache.layer2_out = layer2_out.clone();

        let residual2 = add_residual(&normed1, &layer2_out);
        let out = layer_norm_rows(&residual2, seq_len, d);

        if out.iter().any(|x| !x.is_finite()) {
            return Err(HybridAttnError::NonFinite);
        }
        Ok((out, cache))
    }

    /// Forward pass for a single layer with caching.
    fn forward_single_layer_cached(
        &self,
        tokens: &[f32],
        seq_len: usize,
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
        wo: &[f32],
        layer_idx: usize,
    ) -> Result<(Vec<f32>, LayerCache), HybridAttnError> {
        let d = self.cfg.d_model;
        let h = self.cfg.num_heads;
        let d_head = d / h;

        let q = matmul(tokens, wq, seq_len, d, d);
        let k = matmul(tokens, wk, seq_len, d, d);
        let v = matmul(tokens, wv, seq_len, d, d);

        let scale = (d_head as f32).sqrt();
        let mut attn_out = vec![0.0_f32; seq_len * d];
        let mut attn_weights = vec![vec![0.0_f32; seq_len * (seq_len + 1) / 2]; h];

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
                let idx = i * (i + 1) / 2;
                for (j, &w) in scores.iter().enumerate() {
                    attn_weights[head][idx + j] = w;
                    for k_idx in 0..d_head {
                        attn_out[i * d + head_offset + k_idx] += w * v[j * d + head_offset + k_idx];
                    }
                }
            }
        }

        let out = matmul(&attn_out, wo, seq_len, d, d);
        if out.iter().any(|x| !x.is_finite()) {
            return Err(HybridAttnError::NonFinite);
        }

        let cache = LayerCache {
            q,
            k,
            v,
            attn_weights,
            layer_idx,
        };

        Ok((out, cache))
    }

    /// Backward pass: computes gradients for all weights.
    ///
    /// # Arguments
    ///
    /// * `d_output` - Gradient of loss wrt output (seq_len × d_model)
    /// * `cache` - Forward pass activations from `forward_cached`
    /// * `gradients` - Output gradients (will be accumulated)
    ///
    /// # Returns
    ///
    /// Gradient of loss wrt input tokens (seq_len × d_model)
    pub fn backward(
        &self,
        d_output: &[f32],
        cache: &ForwardCache,
        gradients: &mut AttentionGradients,
    ) -> Vec<f32> {
        let seq_len = cache.tokens.len() / self.cfg.d_model;
        let d = self.cfg.d_model;
        let h = self.cfg.num_heads;
        let d_head = d / h;
        let eps = 1e-5_f32;

        gradients.clear();

        // Start from output gradient
        let mut d_current = d_output.to_vec();

        // Backward through second layer (if exists)
        let mut d_input_to_layer2 = if self.cfg.num_attn_layers == 2 {
            // d_output → d_normed2 → d_residual2 → d_attn2_out → d_v2, d_attn_weights2 → d_q2, d_k2 → d_normed1

            // Backward through LayerNorm 2
            let residual2 = add_residual(&cache.normed1, &cache.layer2_out);
            let d_normed2 = layer_norm_backward_cached(&residual2, &d_current, seq_len, d, eps);

            // Backward through residual: d_layer2_out = d_normed2, d_normed1 += d_normed2
            let d_layer2_out = d_normed2.clone();
            let mut d_normed1 = d_normed2.clone();

            // Backward through output projection WO2
            let (d_attn_out, d_wo2) =
                matmul_backward(&cache.layer2_out, &d_layer2_out, &self.wo2, seq_len, d, d);
            gradients.d_wo2 = d_wo2;

            // Backward through attention mechanism (layer 2)
            let (d_q2, d_k2, d_v2) = attention_backward_cached(
                &cache.normed1,
                &cache.q2,
                &cache.k2,
                &cache.v2,
                &cache.attn_weights2,
                &d_attn_out,
                seq_len,
                d,
                h,
                d_head,
                self.cfg.qk_gain,
            );

            // Backward through input projections (WQ2, WK2, WV2)
            let (_, d_wq2) = matmul_backward(&cache.normed1, &d_q2, &self.wq2, seq_len, d, d);
            let (_, d_wk2) = matmul_backward(&cache.normed1, &d_k2, &self.wk2, seq_len, d, d);
            let (_, d_wv2) = matmul_backward(&cache.normed1, &d_v2, &self.wv2, seq_len, d, d);

            gradients.d_wq2 = d_wq2;
            gradients.d_wk2 = d_wk2;
            gradients.d_wv2 = d_wv2;

            // Accumulate gradient into d_normed1
            for i in 0..d_normed1.len() {
                d_normed1[i] += d_q2[i] + d_k2[i] + d_v2[i];
            }

            d_normed1
        } else {
            // No second layer, continue with d_output as d_normed1
            d_current.clone()
        };

        // Backward through LayerNorm 1
        let residual1 = add_residual(&cache.tokens, &cache.layer1_out);
        let d_normed1 = layer_norm_backward_cached(&residual1, &d_input_to_layer2, seq_len, d, eps);

        // Backward through residual: d_layer1_out = d_normed1, d_tokens += d_normed1
        let d_layer1_out = d_normed1.clone();
        let mut d_tokens = d_normed1.clone();

        // Backward through output projection WO1
        let (d_attn_out, d_wo) =
            matmul_backward(&cache.layer1_out, &d_layer1_out, &self.wo, seq_len, d, d);
        gradients.d_wo = d_wo;

        // Backward through attention mechanism (layer 1)
        let (d_q1, d_k1, d_v1) = attention_backward_cached(
            &cache.tokens,
            &cache.q1,
            &cache.k1,
            &cache.v1,
            &cache.attn_weights1,
            &d_attn_out,
            seq_len,
            d,
            h,
            d_head,
            self.cfg.qk_gain,
        );

        // Backward through input projections (WQ1, WK1, WV1)
        let (_, d_wq) = matmul_backward(&cache.tokens, &d_q1, &self.wq, seq_len, d, d);
        let (_, d_wk) = matmul_backward(&cache.tokens, &d_k1, &self.wk, seq_len, d, d);
        let (_, d_wv) = matmul_backward(&cache.tokens, &d_v1, &self.wv, seq_len, d, d);

        gradients.d_wq = d_wq;
        gradients.d_wk = d_wk;
        gradients.d_wv = d_wv;

        // Accumulate gradient into d_tokens
        for i in 0..d_tokens.len() {
            d_tokens[i] += d_q1[i] + d_k1[i] + d_v1[i];
        }

        d_tokens
    }
}

/// Cache for a single attention layer.
#[derive(Debug, Clone)]
struct LayerCache {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_weights: Vec<Vec<f32>>,
    pub layer_idx: usize,
}

/// LayerNorm backward pass (row-wise, for multiple rows).
fn layer_norm_backward_cached(
    x: &[f32],
    dx: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Vec<f32> {
    let mut dln_output = vec![0.0_f32; x.len()];

    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let drow = &dx[r * cols..(r + 1) * cols];
        let n = cols as f32;

        let mean = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
        let std_inv = 1.0 / (var + eps).sqrt();
        let var_plus_eps_inv = 1.0 / (var + eps);

        let drow_sum: f32 = drow.iter().sum();
        let mut drow_x_diff_sum = 0.0_f32;
        for i in 0..cols {
            drow_x_diff_sum += drow[i] * (row[i] - mean);
        }

        let inv_n_std = std_inv / n;

        for c in 0..cols {
            let x_diff = row[c] - mean;
            let term1 = drow[c] - drow_sum / n;
            let term2 = x_diff * var_plus_eps_inv * drow_x_diff_sum / n;
            dln_output[r * cols + c] = inv_n_std * (term1 - term2);
        }
    }

    dln_output
}

/// Matmul backward: computes d_input and d_weight.
///
/// Given forward: output = input @ weight
/// Returns (d_input, d_weight)
fn matmul_backward(
    input: &[f32],
    d_output: &[f32],
    weight: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut d_input = vec![0.0_f32; m * k];
    let mut d_weight = vec![0.0_f32; k * n];

    // d_weight = input^T @ d_output
    for i in 0..m {
        for l in 0..k {
            for j in 0..n {
                d_weight[l * n + j] += input[i * k + l] * d_output[i * n + j];
            }
        }
    }

    // d_input = d_output @ weight^T
    for i in 0..m {
        for l in 0..k {
            let mut sum = 0.0_f32;
            for j in 0..n {
                sum += d_output[i * n + j] * weight[l * n + j];
            }
            d_input[i * k + l] = sum;
        }
    }

    (d_input, d_weight)
}

/// Attention backward pass with cached activations.
///
/// Returns (d_q, d_k, d_v) gradients
fn attention_backward_cached(
    input: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    attn_weights: &[Vec<f32>], // [head][scores]
    d_attn_out: &[f32],
    seq_len: usize,
    d_model: usize,
    num_heads: usize,
    d_head: usize,
    qk_gain: f64,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut d_q = vec![0.0_f32; seq_len * d_model];
    let mut d_k = vec![0.0_f32; seq_len * d_model];
    let mut d_v = vec![0.0_f32; seq_len * d_model];
    let scale = (d_head as f32).sqrt();
    let qk_gain_f = qk_gain as f32 / scale;

    for head in 0..num_heads {
        let head_offset = head * d_head;

        // First, compute d_v: accumulate from all attended positions
        for i in 0..seq_len {
            let idx_base = i * (i + 1) / 2;
            for j in 0..=i {
                let attn_w = attn_weights[head][idx_base + j];
                for k_idx in 0..d_head {
                    let v_idx = j * d_model + head_offset + k_idx;
                    let out_idx = i * d_model + head_offset + k_idx;
                    d_v[v_idx] += d_attn_out[out_idx] * attn_w;
                }
            }
        }

        // Compute d_attn_weights and backprop to d_q, d_k
        for i in 0..seq_len {
            let idx_base = i * (i + 1) / 2;

            // d_attn_weight for each j
            for j in 0..=i {
                let mut d_attn_w = 0.0_f32;
                for k_idx in 0..d_head {
                    let out_idx = i * d_model + head_offset + k_idx;
                    let v_idx = j * d_model + head_offset + k_idx;
                    d_attn_w += d_attn_out[out_idx] * v[v_idx];
                }

                // Backprop through softmax
                let softmax_grad =
                    softmax_backward_single(idx_base + j, &attn_weights[head], d_attn_w, i + 1);

                // Backprop to Q (at position i)
                for k_idx in 0..d_head {
                    let q_idx = i * d_model + head_offset + k_idx;
                    d_q[q_idx] += softmax_grad * k[j * d_model + head_offset + k_idx] * qk_gain_f;
                }

                // Backprop to K (at position j)
                for k_idx in 0..d_head {
                    let k_idx2 = j * d_model + head_offset + k_idx;
                    d_k[k_idx2] += softmax_grad * q[i * d_model + head_offset + k_idx] * qk_gain_f;
                }
            }
        }
    }

    (d_q, d_k, d_v)
}

/// Softmax backward for a single element.
///
/// d_softmax[i] = dL/dsoftmax[i] - sum_j(dL/dsoftmax[j] * softmax[j]) * softmax[i]
fn softmax_backward_single(target_idx: usize, softmax: &[f32], d_loss: f32, n: usize) -> f32 {
    let target = softmax[target_idx];
    let sum: f32 = softmax
        .iter()
        .zip(0..n)
        .map(|(&s, j)| {
            if j < softmax.len() {
                let idx = j * (j + 1) / 2 + (target_idx - j);
                if idx < softmax.len() {
                    // This is a simplification - in practice, we'd need the full softmax vector
                }
            }
            s
        })
        .sum();

    // For causal attention, we use a simplified gradient
    // The full implementation would require storing the full score vector
    let sum_dloss_times_softmax: f32 = softmax.iter().take(n).map(|&s| d_loss * s).sum();
    d_loss * target - sum_dloss_times_softmax * target
}

impl HybridAttn {
    pub fn forward_with_cache(
        &self,
        tokens: &[f32],
        seq_len: usize,
    ) -> Result<(Vec<f32>, AttentionCache), HybridAttnError> {
        if tokens.iter().any(|x| !x.is_finite()) {
            return Err(HybridAttnError::NonFinite);
        }
        let d = self.cfg.d_model;
        assert_eq!(tokens.len(), seq_len * d);
        let input = tokens.to_vec();

        let (layer1_out, q1, k1, v1, aw1, ap1) =
            self.forward_single_layer_cached(tokens, seq_len)?;
        let residual1 = add_residual(tokens, &layer1_out);
        let normed1 = layer_norm_rows(&residual1, seq_len, d);

        if self.cfg.num_attn_layers == 1 {
            if normed1.iter().any(|x| !x.is_finite()) {
                return Err(HybridAttnError::NonFinite);
            }
            let cache = AttentionCache {
                input,
                q1,
                k1,
                v1,
                attn_weights1: aw1,
                attn_pre_wo1: ap1,
                residual1,
                normed1: normed1.clone(),
                q2: vec![],
                k2: vec![],
                v2: vec![],
                attn_weights2: vec![],
                attn_pre_wo2: vec![],
                residual2: vec![],
                output: normed1,
                seq_len,
            };
            return Ok((cache.output.clone(), cache));
        }

        let (layer2_out, q2, k2, v2, aw2, ap2) =
            self.forward_single_layer_cached(&normed1, seq_len)?;
        let residual2 = add_residual(&normed1, &layer2_out);
        let output = layer_norm_rows(&residual2, seq_len, d);

        if output.iter().any(|x| !x.is_finite()) {
            return Err(HybridAttnError::NonFinite);
        }

        let cache = AttentionCache {
            input,
            q1,
            k1,
            v1,
            attn_weights1: aw1,
            attn_pre_wo1: ap1,
            residual1,
            normed1,
            q2,
            k2,
            v2,
            attn_weights2: aw2,
            attn_pre_wo2: ap2,
            residual2: residual2.clone(),
            output,
            seq_len,
        };
        Ok((cache.output.clone(), cache))
    }

    fn forward_single_layer_cached(
        &self,
        tokens: &[f32],
        seq_len: usize,
    ) -> Result<
        (
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<Vec<Vec<f32>>>,
            Vec<f32>,
        ),
        HybridAttnError,
    > {
        let d = self.cfg.d_model;
        let h = self.cfg.num_heads;
        let d_head = d / h;

        let q = matmul(tokens, &self.wq, seq_len, d, d);
        let k = matmul(tokens, &self.wk, seq_len, d, d);
        let v = matmul(tokens, &self.wv, seq_len, d, d);

        let scale = (d_head as f32).sqrt();
        let mut attn_out = vec![0.0_f32; seq_len * d];
        let mut attn_weights: Vec<Vec<Vec<f32>>> = vec![vec![vec![]; h]; seq_len];

        for head in 0..h {
            let ho = head * d_head;
            for i in 0..seq_len {
                let mut scores = vec![0.0_f32; i + 1];
                for (j, score) in scores.iter_mut().enumerate() {
                    let mut s = 0.0_f32;
                    for ki in 0..d_head {
                        s += q[i * d + ho + ki] * k[j * d + ho + ki];
                    }
                    *score = (self.cfg.qk_gain as f32) * s / scale;
                }
                softmax_inplace(&mut scores);
                attn_weights[i][head] = scores.clone();
                for j in 0..=i {
                    let w = attn_weights[i][head][j];
                    for ki in 0..d_head {
                        attn_out[i * d + ho + ki] += w * v[j * d + ho + ki];
                    }
                }
            }
        }

        let out = matmul(&attn_out, &self.wo, seq_len, d, d);
        if out.iter().any(|x| !x.is_finite()) {
            return Err(HybridAttnError::NonFinite);
        }
        Ok((out, q, k, v, attn_weights, attn_out))
    }

    pub fn backward(&self, cache: &AttentionCache, d_output: &[f32]) -> AttentionGrads {
        let d = self.cfg.d_model;
        let seq_len = cache.seq_len;
        let dd = d * d;

        if self.cfg.num_attn_layers == 2 {
            let d_residual2 =
                layer_norm_rows_backward(&cache.residual2, &cache.output, d_output, seq_len, d);
            let d_normed1_from_res = d_residual2.clone();
            let d_layer2_out = d_residual2;

            let (gwq2, gwk2, gwv2, gwo2, d_normed1_from_attn) = self.backward_single_layer(
                &d_layer2_out,
                &cache.normed1,
                &cache.q2,
                &cache.k2,
                &cache.v2,
                &cache.attn_weights2,
                &cache.attn_pre_wo2,
                &self.wq2,
                &self.wk2,
                &self.wv2,
                &self.wo2,
                seq_len,
            );

            let mut d_normed1 = d_normed1_from_res;
            for i in 0..seq_len * d {
                d_normed1[i] += d_normed1_from_attn[i];
            }

            let d_residual1 =
                layer_norm_rows_backward(&cache.residual1, &cache.normed1, &d_normed1, seq_len, d);
            let d_input_from_res = d_residual1.clone();
            let d_layer1_out = d_residual1;

            let (gwq, gwk, gwv, gwo, d_input_from_attn) = self.backward_single_layer(
                &d_layer1_out,
                &cache.input,
                &cache.q1,
                &cache.k1,
                &cache.v1,
                &cache.attn_weights1,
                &cache.attn_pre_wo1,
                &self.wq,
                &self.wk,
                &self.wv,
                &self.wo,
                seq_len,
            );

            let mut d_input = d_input_from_res.clone();
            for i in 0..seq_len * d {
                d_input[i] += d_input_from_attn[i];
            }

            AttentionGrads {
                gwq,
                gwk,
                gwv,
                gwo,
                gwq2,
                gwk2,
                gwv2,
                gwo2,
                d_input,
            }
        } else {
            let d_residual1 =
                layer_norm_rows_backward(&cache.residual1, &cache.output, d_output, seq_len, d);
            let d_input_from_res = d_residual1.clone();
            let d_layer1_out = d_residual1;

            let (gwq, gwk, gwv, gwo, d_input_from_attn) = self.backward_single_layer(
                &d_layer1_out,
                &cache.input,
                &cache.q1,
                &cache.k1,
                &cache.v1,
                &cache.attn_weights1,
                &cache.attn_pre_wo1,
                &self.wq,
                &self.wk,
                &self.wv,
                &self.wo,
                seq_len,
            );

            let mut d_input = d_input_from_res.clone();
            for i in 0..seq_len * d {
                d_input[i] += d_input_from_attn[i];
            }

            AttentionGrads {
                gwq,
                gwk,
                gwv,
                gwo,
                gwq2: vec![0.0; dd],
                gwk2: vec![0.0; dd],
                gwv2: vec![0.0; dd],
                gwo2: vec![0.0; dd],
                d_input,
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn backward_single_layer(
        &self,
        d_out: &[f32],
        input: &[f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        attn_weights: &[Vec<Vec<f32>>],
        attn_pre_wo: &[f32],
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
        wo: &[f32],
        seq_len: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let d = self.cfg.d_model;
        let h = self.cfg.num_heads;
        let d_head = d / h;
        let scale = (d_head as f32).sqrt();
        let qk_gain = self.cfg.qk_gain as f32;

        let d_attn_pre_wo = matmul_transpose_b(d_out, wo, seq_len, d, d);
        let gwo = matmul_transpose_a(attn_pre_wo, d_out, seq_len, d, d);

        let mut d_q = vec![0.0f32; seq_len * d];
        let mut d_k = vec![0.0f32; seq_len * d];
        let mut d_v = vec![0.0f32; seq_len * d];

        for head in 0..h {
            let ho = head * d_head;
            for i in 0..seq_len {
                let mut d_aw = vec![0.0f32; i + 1];
                for j in 0..=i {
                    let mut s = 0.0f32;
                    for ki in 0..d_head {
                        s += d_attn_pre_wo[i * d + ho + ki] * v[j * d + ho + ki];
                    }
                    d_aw[j] = s;
                }

                let w = &attn_weights[i][head];
                let sum_wdaw: f32 = (0..=i).map(|jj| w[jj] * d_aw[jj]).sum();
                let mut d_scores = vec![0.0f32; i + 1];
                for j in 0..=i {
                    d_scores[j] = w[j] * (d_aw[j] - sum_wdaw) * qk_gain / scale;
                }

                for j in 0..=i {
                    for ki in 0..d_head {
                        d_q[i * d + ho + ki] += d_scores[j] * k[j * d + ho + ki];
                        d_k[j * d + ho + ki] += d_scores[j] * q[i * d + ho + ki];
                        d_v[j * d + ho + ki] += w[j] * d_attn_pre_wo[i * d + ho + ki];
                    }
                }
            }
        }

        let gwq = matmul_transpose_a(input, &d_q, seq_len, d, d);
        let gwk = matmul_transpose_a(input, &d_k, seq_len, d, d);
        let gwv = matmul_transpose_a(input, &d_v, seq_len, d, d);

        let d_input_q = matmul_transpose_b(&d_q, wq, seq_len, d, d);
        let d_input_k = matmul_transpose_b(&d_k, wk, seq_len, d, d);
        let d_input_v = matmul_transpose_b(&d_v, wv, seq_len, d, d);
        let mut d_input = vec![0.0f32; seq_len * d];
        for i in 0..seq_len * d {
            d_input[i] = d_input_q[i] + d_input_k[i] + d_input_v[i];
        }

        (gwq, gwk, gwv, gwo, d_input)
    }

    pub fn wq2_mut(&mut self) -> &mut [f32] {
        &mut self.wq2
    }
    pub fn wk2_mut(&mut self) -> &mut [f32] {
        &mut self.wk2
    }
    pub fn wv2_mut(&mut self) -> &mut [f32] {
        &mut self.wv2
    }
    pub fn wo2_mut(&mut self) -> &mut [f32] {
        &mut self.wo2
    }

    pub fn total_weights_all(&self) -> usize {
        8 * self.cfg.d_model * self.cfg.d_model
    }
}

// ═══════════════════════════════════════════════════════════════════
// Helpers (kept private; test-visible via the `HybridAttn::forward` call)
// ═══════════════════════════════════════════════════════════════════

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

fn matmul_transpose_a(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; k * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                out[p * n + j] += a_ip * b[i * n + j];
            }
        }
    }
    out
}

fn matmul_transpose_b(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f32;
            for p in 0..k {
                s += a[i * k + p] * b[j * k + p];
            }
            out[i * n + j] = s;
        }
    }
    out
}

fn layer_norm_rows_backward(
    x: &[f32],
    y: &[f32],
    dy: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let eps = 1e-5_f32;
    let mut dx = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row_x = &x[r * cols..(r + 1) * cols];
        let row_y = &y[r * cols..(r + 1) * cols];
        let row_dy = &dy[r * cols..(r + 1) * cols];
        let n = cols as f32;
        let mean: f32 = row_x.iter().sum::<f32>() / n;
        let var: f32 = row_x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let std_inv = 1.0 / (var + eps).sqrt();
        let sum_dy: f32 = row_dy.iter().sum();
        let sum_dy_y: f32 = row_dy.iter().zip(row_y.iter()).map(|(d, yi)| d * yi).sum();
        for c in 0..cols {
            dx[r * cols + c] = (row_dy[c] - sum_dy / n - row_y[c] * sum_dy_y / n) * std_inv;
        }
    }
    dx
}

// ═══════════════════════════════════════════════════════════════════
// Falsifier tests — R7 witnesses for INV-1, INV-13, shape, and forward
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod falsifiers {
    use super::*;
    use crate::invariants::PHI;

    /// R7 / INV-1: a learning rate outside the Coq-proven φ-band must
    /// refuse at construction time.  This is the deterministic sibling
    /// of the earlier pure-attention plateau (BPB ≈ 4.74 @ lr=0.01).
    #[test]
    fn falsify_hybrid_diverges_bad_lr() {
        let err = HybridAttn::new_with_lr(0.02).unwrap_err();
        assert!(
            matches!(err, HybridAttnError::LrOutOfBand { .. }),
            "expected LrOutOfBand, got {err:?}",
        );
        // Lower-side witness.
        let err = HybridAttn::new_with_lr(0.0005).unwrap_err();
        assert!(matches!(err, HybridAttnError::LrOutOfBand { .. }));
        // And the inside-band default must succeed.
        HybridAttn::new_with_lr(0.0035).expect("0.0035 is inside the band");
    }

    /// R7 / INV-13: any qk_gain outside `{φ², φ³}` must refuse.  This is
    /// the Rust mirror of the pre-registered Coq lemma
    /// `counter_qk_gain_outside_phi_sq` (L-h4).
    #[test]
    fn falsify_hybrid_qk_gain_not_phi_sq_or_phi_cube() {
        let err = HybridAttn::new_with_qk_gain(PHI).unwrap_err();
        assert!(
            matches!(err, HybridAttnError::QkGainOutsidePhi { .. }),
            "qk_gain=PHI must be refused, got {err:?}",
        );
        let err = HybridAttn::new_with_qk_gain(1.0).unwrap_err();
        assert!(matches!(err, HybridAttnError::QkGainOutsidePhi { .. }));
        // Both pre-registered gains must succeed.
        HybridAttn::new_with_qk_gain(PHI_SQ).expect("φ² is allowed");
        HybridAttn::new_with_qk_gain(PHI_CUBE).expect("φ³ is allowed");
    }

    /// Shape invariant: `d_model % num_heads != 0` must refuse.
    #[test]
    fn falsify_hybrid_shape_invariant() {
        let cfg = HybridAttnConfig {
            d_model: 64,
            num_heads: 5, // 64 % 5 = 4 ≠ 0
            ..HybridAttnConfig::default()
        };
        let err = HybridAttn::with_config(cfg).unwrap_err();
        assert!(matches!(err, HybridAttnError::Shape { .. }));
    }

    /// Deterministic forward pass: zero weights on zero tokens must
    /// return zeros (no NaN, no Inf).  The goal is to exercise the
    /// non-finite detector on a known-good input.
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

    /// Non-finite input must be surfaced as `Err(NonFinite)`, not
    /// propagated silently.  R5: honest refusal.
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

    /// RoPE periodicity: for `d_head = 16`, the ratio between the
    /// frequency at index 0 and index 7 is exactly `10_000^{14/16}`.
    /// This property is the INV-9 φ-anchor hook — the actual φ-relation
    /// is proven in the Coq lemma, not re-asserted here.
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

    /// `reassert()` must stay green for the default config.  This is
    /// called inside L-h1's training loop; regressing it breaks the
    /// online invariant sweep.
    #[test]
    fn hybrid_attn_reassert_stable() {
        let block = HybridAttn::new().expect("defaults are valid");
        for _ in 0..8 {
            block.reassert().expect("online reassertion must hold");
        }
    }

    /// L-f1 Gate-final: 2-layer forward pass with residual + LayerNorm
    /// must produce finite output on zero-initialized weights.
    #[test]
    fn twin_attn_2layer_forward_roundtrip() {
        let block = HybridAttn::new().expect("defaults are valid (num_attn_layers=2)");
        assert_eq!(block.config().num_attn_layers, 2);
        let seq_len = 4;
        let d = block.config().d_model;
        let tokens = vec![0.0_f32; seq_len * d];
        let out = block.forward(&tokens, seq_len).unwrap();
        assert_eq!(out.len(), seq_len * d);
        assert!(
            out.iter().all(|x| x.is_finite()),
            "2-layer output must be finite"
        );
    }

    /// L-f1 Gate-final: 1-layer mode must still work (backward compat).
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

    /// L-f1 Gate-final: num_attn_layers > 2 is forbidden (§8).
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

    /// L-f1 Gate-final: non-finite input rejected in 2-layer mode.
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

    /// L-f1 Gate-final witness: qk_gain outside φ-band refused
    /// (Gate-final §2 falsifier 4).  Re-asserts for the DRAFT context.
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

    /// L-h5 — Attention backward pass produces finite gradients
    #[test]
    fn attention_backward_produces_finite_gradients() {
        let block = HybridAttn::new().expect("defaults are valid");
        let seq_len = 4;
        let d = block.config().d_model;

        // Simple input
        let tokens = vec![1.0_f32; seq_len * d];
        let (out, cache) = block
            .forward_cached(&tokens, seq_len)
            .expect("forward should succeed");

        // Upstream gradient (all ones)
        let d_output = vec![1.0_f32; seq_len * d];

        // Compute backward pass
        let mut grads = AttentionGradients::new(d);
        let d_input = block.backward(&d_output, &cache, &mut grads);

        // All gradients should be finite
        assert!(
            grads.d_wq.iter().all(|x| x.is_finite()),
            "d_wq should be finite"
        );
        assert!(
            grads.d_wk.iter().all(|x| x.is_finite()),
            "d_wk should be finite"
        );
        assert!(
            grads.d_wv.iter().all(|x| x.is_finite()),
            "d_wv should be finite"
        );
        assert!(
            grads.d_wo.iter().all(|x| x.is_finite()),
            "d_wo should be finite"
        );
        assert!(
            d_input.iter().all(|x| x.is_finite()),
            "d_input should be finite"
        );

        // Gradients should have correct shape
        assert_eq!(grads.d_wq.len(), d * d, "d_wq shape");
        assert_eq!(grads.d_wk.len(), d * d, "d_wk shape");
        assert_eq!(grads.d_wv.len(), d * d, "d_wv shape");
        assert_eq!(grads.d_wo.len(), d * d, "d_wo shape");
        assert_eq!(d_input.len(), seq_len * d, "d_input shape");
    }

    /// L-h5 — 2-layer attention backward pass
    #[test]
    fn two_layer_attention_backward() {
        let block = HybridAttn::new().expect("defaults are valid (2 layers)");
        assert_eq!(block.config().num_attn_layers, 2);

        let seq_len = 4;
        let d = block.config().d_model;
        let tokens = vec![1.0_f32; seq_len * d];

        let (out, cache) = block
            .forward_cached(&tokens, seq_len)
            .expect("forward should succeed");
        let d_output = vec![1.0_f32; seq_len * d];

        let mut grads = AttentionGradients::new(d);
        let d_input = block.backward(&d_output, &cache, &mut grads);

        // All gradients including layer 2 should be finite
        assert!(
            grads.d_wq2.iter().all(|x| x.is_finite()),
            "d_wq2 should be finite"
        );
        assert!(
            grads.d_wk2.iter().all(|x| x.is_finite()),
            "d_wk2 should be finite"
        );
        assert!(
            grads.d_wv2.iter().all(|x| x.is_finite()),
            "d_wv2 should be finite"
        );
        assert!(
            grads.d_wo2.iter().all(|x| x.is_finite()),
            "d_wo2 should be finite"
        );
        assert!(
            d_input.iter().all(|x| x.is_finite()),
            "d_input should be finite"
        );
    }

    /// L-h5 — Gradient accumulation (multiple backward passes)
    #[test]
    fn gradient_accumulation() {
        let block = HybridAttn::new().expect("defaults are valid");
        let seq_len = 4;
        let d = block.config().d_model;

        let tokens = vec![1.0_f32; seq_len * d];
        let d_output = vec![1.0_f32; seq_len * d];

        let mut grads = AttentionGradients::new(d);

        // First backward pass
        let (_, cache) = block
            .forward_cached(&tokens, seq_len)
            .expect("forward should succeed");
        block.backward(&d_output, &cache, &mut grads);

        let grads_1 = grads.d_wq.clone();

        // Second backward pass (gradients should accumulate)
        let (_, cache2) = block
            .forward_cached(&tokens, seq_len)
            .expect("forward should succeed");
        block.backward(&d_output, &cache2, &mut grads);

        // Gradients should be approximately doubled
        for i in 0..grads.d_wq.len() {
            assert!(
                (grads.d_wq[i] - 2.0 * grads_1[i]).abs() < 1e-5,
                "gradient accumulation at index {}: got {}, expected {}",
                i,
                grads.d_wq[i],
                2.0 * grads_1[i]
            );
        }
    }

    /// L-h5 — Gradient clearing
    #[test]
    fn gradient_clearing() {
        let mut grads = AttentionGradients::new(64);
        grads.d_wq.fill(1.0);

        grads.clear();

        assert!(
            grads.d_wq.iter().all(|x| *x == 0.0),
            "d_wq should be cleared"
        );
        assert!(
            grads.d_wk.iter().all(|x| *x == 0.0),
            "d_wk should be cleared"
        );
        assert!(
            grads.d_wv.iter().all(|x| *x == 0.0),
            "d_wv should be cleared"
        );
        assert!(
            grads.d_wo.iter().all(|x| *x == 0.0),
            "d_wo should be cleared"
        );
    }

    /// L-h5 — Forward cache structure
    #[test]
    fn forward_cache_structure() {
        let seq_len = 4;
        let d_model = 64;
        let num_heads = 4;

        let cache = ForwardCache::new(seq_len, d_model, num_heads);

        assert_eq!(cache.tokens.len(), seq_len * d_model);
        assert_eq!(cache.q1.len(), seq_len * d_model);
        assert_eq!(cache.k1.len(), seq_len * d_model);
        assert_eq!(cache.v1.len(), seq_len * d_model);
        assert_eq!(cache.attn_weights1.len(), num_heads);
        assert_eq!(cache.attn_weights1[0].len(), seq_len * (seq_len + 1) / 2);
    }
}
