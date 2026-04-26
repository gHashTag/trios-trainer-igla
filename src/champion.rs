#![allow(clippy::needless_range_loop, dead_code, unused_variables)]
//! Champion NgramModel — proven architecture from commit 2446855
//!
//! Architecture: dim=64, hidden=384, layer norm, projection, separate ctx
//! Champion: BPB=2.2393 @ 27K steps, seed=43, lr=0.003, no-JEPA, no-NCA
//!
//! This reproduces the exact model that achieved the champion result.

use crate::config::ModelConfig;
use anyhow::Result;

const DIM: usize = 64;
const HIDDEN: usize = 384;
const NUM_CTX: usize = 4;

/// Champion NgramModel — reproduces BPB=2.2393 @ 27K seed=43
#[derive(Debug, Clone)]
pub struct NgramModel {
    pub embed: Vec<f32>,
    pub ctx: Vec<Vec<f32>>,
    pub ctx_weights: Vec<f32>,
    pub proj: Vec<f32>,
    pub lm_head: Vec<f32>,
}

impl NgramModel {
    /// Create a new NgramModel with champion architecture
    pub fn new(seed: u64, vocab: usize) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let lim = (6.0f32 / (3 * DIM) as f32).sqrt();
        let lim_h = (6.0f32 / (DIM + HIDDEN) as f32).sqrt();
        let lim_o = (6.0f32 / (HIDDEN + vocab) as f32).sqrt();
        let ctx_weights: Vec<f32> = vec![0.7, 0.3, 0.2, 0.15];
        assert_eq!(ctx_weights.len(), NUM_CTX, "ctx_weights count mismatch");
        Self {
            embed: (0..vocab * DIM).map(|_| rng() * lim).collect(),
            ctx: (0..NUM_CTX)
                .map(|_| (0..vocab * DIM).map(|_| rng() * lim).collect())
                .collect(),
            ctx_weights,
            proj: (0..HIDDEN * DIM).map(|_| rng() * lim_h).collect(),
            lm_head: (0..vocab * HIDDEN).map(|_| rng() * lim_o).collect(),
        }
    }

    /// Create from ModelConfig
    pub fn from_config(cfg: &ModelConfig, seed: u64) -> Self {
        Self::new(seed, cfg.vocab_size)
    }

    /// Compute hidden representation for given context
    pub fn compute_hidden(&self, context: &[usize], vocab: usize) -> Vec<f32> {
        assert!(context.len() >= 2, "context too short for hidden");
        let t0 = context[context.len() - 1].min(vocab - 1);
        let mut combined = self.embed[t0 * DIM..(t0 + 1) * DIM].to_vec();
        for (ci, cw) in self.ctx_weights.iter().enumerate() {
            let ctx_idx = context.len() - 2 - ci;
            let t = context[ctx_idx].min(vocab - 1);
            let cv = &self.ctx[ci][t * DIM..(t + 1) * DIM];
            for j in 0..DIM {
                combined[j] += cv[j] * cw;
            }
        }
        let ln = self.layer_norm(&combined, 1e-5);
        let mut hidden = vec![0.0f32; HIDDEN];
        for hi in 0..HIDDEN {
            for (j, l) in ln.iter().enumerate() {
                hidden[hi] += self.proj[hi * DIM + j] * l;
            }
            hidden[hi] = hidden[hi].max(0.0);
        }
        hidden
    }

    /// Predict logits from hidden representation
    pub fn predict(&self, hidden: &[f32], vocab: usize) -> Vec<f32> {
        assert_eq!(hidden.len(), HIDDEN, "hidden dim mismatch");
        let mut logits = vec![0.0f32; vocab];
        for (vi, logit) in logits.iter_mut().enumerate() {
            for (hi, hn) in hidden.iter().enumerate() {
                *logit += self.lm_head[vi * HIDDEN + hi] * hn;
            }
        }
        logits
    }

    fn layer_norm(&self, x: &[f32], eps: f32) -> Vec<f32> {
        assert!(!x.is_empty(), "layer_norm: empty input");
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let std = (var + eps).sqrt();
        x.iter().map(|v| (v - mean) / std).collect()
    }

    /// Count parameters
    pub fn param_count(&self, vocab: usize) -> usize {
        self.embed.len()
            + self.ctx.iter().map(|c| c.len()).sum::<usize>()
            + self.proj.len()
            + self.lm_head.len()
    }
}

/// Build champion model from configuration (L-T1 entry point)
pub fn build(cfg: &ModelConfig, seed: u64) -> Result<NgramModel> {
    Ok(NgramModel::from_config(cfg, seed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let model = NgramModel::new(42, 128);
        assert_eq!(model.embed.len(), 128 * DIM);
        assert_eq!(model.ctx.len(), NUM_CTX);
        assert_eq!(model.proj.len(), HIDDEN * DIM);
        assert_eq!(model.lm_head.len(), 128 * HIDDEN);
    }

    #[test]
    fn test_hidden_computation() {
        let model = NgramModel::new(42, 128);
        // NUM_CTX = 4 ctx_weights; need context.len() >= NUM_CTX + 1 = 5
        let context = vec![10, 20, 30, 40, 50];
        let hidden = model.compute_hidden(&context, 128);
        assert_eq!(hidden.len(), HIDDEN);
        assert!(hidden.iter().all(|h| h.is_finite()));
    }

    #[test]
    fn test_prediction() {
        let model = NgramModel::new(42, 128);
        let hidden = vec![0.1f32; HIDDEN];
        let logits = model.predict(&hidden, 128);
        assert_eq!(logits.len(), 128);
        assert!(logits.iter().all(|l| l.is_finite()));
    }

    #[test]
    fn test_param_count() {
        let model = NgramModel::new(42, 128);
        let count = model.param_count(128);
        assert!(count > 0);
    }
}
