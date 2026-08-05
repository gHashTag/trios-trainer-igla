//! T-JEPA Encoder Module

use std::f32::consts::LN_2;

pub struct NgramEncoder {
    pub embed: Vec<f32>,  // accessible for GF16 round
    ctx_weights: Vec<f32>,
    d_model: usize,
    vocab: usize,
    #[allow(dead_code)]
    num_ctx: usize,
}

impl NgramEncoder {
    pub fn new(vocab: usize, d_model: usize, num_ctx: usize, seed: u64) -> Self {
        assert!(vocab > 0, "vocab must be positive");
        assert!(d_model > 0, "d_model must be positive");
        assert!(num_ctx > 0, "num_ctx must be positive");
        assert!(num_ctx <= 6, "num_ctx must be ≤ 6");

        let mut s = seed;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let lim = (6.0f32 / (3 * d_model) as f32).sqrt();
        let embed = (0..vocab * d_model).map(|_| rng() * lim).collect();
        let base_weights: Vec<f32> = vec![0.7, 0.3, 0.2, 0.15, 0.12, 0.1];
        let ctx_weights: Vec<f32> = base_weights.iter().take(num_ctx).cloned().collect();

        assert!(embed.len() == vocab * d_model, "embed size mismatch");
        assert!(ctx_weights.len() == num_ctx, "ctx_weights size mismatch");

        Self { embed, ctx_weights, d_model, vocab, num_ctx }
    }

    pub fn encode(&self, tokens: &[usize]) -> Vec<Vec<f32>> {
        assert!(!tokens.is_empty(), "tokens must not be empty");
        let d = self.d_model;
        let v = self.vocab;
        tokens.iter().enumerate().map(|(pos, &t)| {
            let t_idx = t.min(v - 1);
            let e = &self.embed[t_idx * d..(t_idx + 1) * d];
            let mut combined = e.to_vec();
            for (ci, cw) in self.ctx_weights.iter().enumerate() {
                let ctx_pos = if ci < pos { pos - ci - 1 } else { 0 };
                let t_ctx = tokens.get(ctx_pos).copied().unwrap_or(0).min(v - 1);
                let cv = &self.embed[t_ctx * d..(t_ctx + 1) * d];
                for j in 0..d { combined[j] += cv[j] * cw; }
            }
            combined.iter().map(|&x| x.max(0.0)).collect()
        }).collect()
    }

    pub fn encode_positions(&self, tokens: &[usize], positions: &[usize]) -> Vec<Vec<f32>> {
        assert!(!tokens.is_empty(), "tokens must not be empty");
        assert!(!positions.is_empty(), "positions must not be empty");
        let full = self.encode(tokens);
        positions.iter().map(|&pos| {
            full.get(pos).cloned().unwrap_or_else(|| vec![0.0f32; self.d_model])
        }).collect()
    }
}
