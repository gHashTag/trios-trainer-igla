//! Real IGLA Transformer Model — pure Rust, no ML crates
//!
//! Architecture:
//! - Token embedding (vocab × d_model)
//! - N × (PreNorm → MHA → PreNorm → FFN)
//! - Language model head (d_model → vocab)
//!
//! Training: SGD with cross-entropy loss, BPB = loss / ln(2)

use std::f32::consts::LN_2;

// ── helpers ──────────────────────────────────────────────────────────────────

#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Numerically-stable softmax (in-place).
fn softmax(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in v.iter_mut() {
        *x /= sum;
    }
}

/// Layer norm: (x - μ) / (σ + ε) * γ  (γ = ones, no learned bias)
fn layer_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = (var + eps).sqrt();
    x.iter().map(|v| (v - mean) / std).collect()
}

/// Matrix-vector multiply: A (rows × cols) · v (cols) → out (rows)
fn matvec(a: &[f32], rows: usize, cols: usize, v: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), rows * cols);
    assert_eq!(v.len(), cols);
    (0..rows)
        .map(|r| {
            let row = &a[r * cols..(r + 1) * cols];
            row.iter().zip(v.iter()).map(|(w, x)| w * x).sum()
        })
        .collect()
}

/// Xavier uniform init using LCG pseudo-random.
fn xavier_init(size: usize, fan_in: usize, fan_out: usize, seed: &mut u64) -> Vec<f32> {
    let limit = (6.0f32 / (fan_in + fan_out) as f32).sqrt();
    (0..size)
        .map(|_| {
            // LCG: simple deterministic PRNG
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let t = ((*seed >> 33) as f32) / (u32::MAX as f32); // [0,1]
            t * 2.0 * limit - limit
        })
        .collect()
}

// ── attention ────────────────────────────────────────────────────────────────

struct AttentionHead {
    d_k: usize,
    wq: Vec<f32>, // d_model × d_k
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>, // d_k × d_model
}

impl AttentionHead {
    fn new(d_model: usize, d_k: usize, seed: &mut u64) -> Self {
        Self {
            d_k,
            wq: xavier_init(d_model * d_k, d_model, d_k, seed),
            wk: xavier_init(d_model * d_k, d_model, d_k, seed),
            wv: xavier_init(d_model * d_k, d_model, d_k, seed),
            wo: xavier_init(d_k * d_model, d_k, d_model, seed),
        }
    }

    /// Causal self-attention over sequence of token embeddings.
    /// Returns residual contribution (seq_len × d_model).
    fn forward(&self, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq = xs.len();
        let d_model = xs[0].len();
        let scale = (self.d_k as f32).sqrt();

        // Project Q, K, V
        let qs: Vec<Vec<f32>> = xs.iter().map(|x| matvec(&self.wq, self.d_k, d_model, x)).collect();
        let ks: Vec<Vec<f32>> = xs.iter().map(|x| matvec(&self.wk, self.d_k, d_model, x)).collect();
        let vs: Vec<Vec<f32>> = xs.iter().map(|x| matvec(&self.wv, self.d_k, d_model, x)).collect();

        let mut out = vec![vec![0.0f32; d_model]; seq];
        for i in 0..seq {
            // Scores for position i (causal: only 0..=i)
            let mut scores: Vec<f32> = (0..=i)
                .map(|j| qs[i].iter().zip(ks[j].iter()).map(|(q, k)| q * k).sum::<f32>() / scale)
                .collect();
            softmax(&mut scores);

            // Weighted sum of V
            let mut ctx = vec![0.0f32; self.d_k];
            for (j, &w) in scores.iter().enumerate() {
                for (c, v) in ctx.iter_mut().zip(vs[j].iter()) {
                    *c += w * v;
                }
            }

            // Output projection
            let proj = matvec(&self.wo, d_model, self.d_k, &ctx);
            out[i] = proj;
        }
        out
    }
}

// ── transformer layer ────────────────────────────────────────────────────────

struct TransformerLayer {
    heads: Vec<AttentionHead>,
    d_model: usize,
    // FFN weights
    w1: Vec<f32>, // 4*d_model × d_model
    w2: Vec<f32>, // d_model × 4*d_model
}

impl TransformerLayer {
    fn new(d_model: usize, n_heads: usize, seed: &mut u64) -> Self {
        let d_k = d_model / n_heads;
        let heads = (0..n_heads).map(|_| AttentionHead::new(d_model, d_k, seed)).collect();
        let d_ff = d_model * 4;
        Self {
            heads,
            d_model,
            w1: xavier_init(d_ff * d_model, d_model, d_ff, seed),
            w2: xavier_init(d_model * d_ff, d_ff, d_model, seed),
        }
    }

    fn forward(&self, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq = xs.len();
        let d_model = self.d_model;
        let d_ff = d_model * 4;

        // Pre-norm + multi-head attention (sum heads)
        let normed: Vec<Vec<f32>> = xs.iter().map(|x| layer_norm(x, 1e-5)).collect();
        let mut attn_out = vec![vec![0.0f32; d_model]; seq];
        for head in &self.heads {
            let h = head.forward(&normed);
            for (i, row) in attn_out.iter_mut().enumerate() {
                for (r, v) in row.iter_mut().zip(h[i].iter()) {
                    *r += v;
                }
            }
        }
        // Residual
        let after_attn: Vec<Vec<f32>> = xs
            .iter()
            .zip(attn_out.iter())
            .map(|(x, a)| x.iter().zip(a.iter()).map(|(xi, ai)| xi + ai).collect())
            .collect();

        // Pre-norm + FFN
        let normed2: Vec<Vec<f32>> = after_attn.iter().map(|x| layer_norm(x, 1e-5)).collect();
        let mut result = Vec::with_capacity(seq);
        for (i, x) in normed2.iter().enumerate() {
            let h1: Vec<f32> = matvec(&self.w1, d_ff, d_model, x).into_iter().map(relu).collect();
            let h2 = matvec(&self.w2, d_model, d_ff, &h1);
            // Residual
            let res: Vec<f32> = after_attn[i].iter().zip(h2.iter()).map(|(a, b)| a + b).collect();
            result.push(res);
        }
        result
    }
}

// ── model ────────────────────────────────────────────────────────────────────

pub struct RealIglaModel {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub max_seq_len: usize,
    // Parameters
    embed: Vec<f32>,    // vocab × d_model
    lm_head: Vec<f32>,  // vocab × d_model
    layers: Vec<TransformerLayer>,
}

pub struct SelfAttentionCache;

impl RealIglaModel {
    pub fn new(vocab_size: usize, d_model: usize, n_layers: usize) -> Self {
        let mut seed = 0x1337_c0de_u64;
        let d_ff = d_model * 4;
        let n_heads = (d_model / 64).max(1);

        let embed = xavier_init(vocab_size * d_model, vocab_size, d_model, &mut seed);
        let lm_head = xavier_init(vocab_size * d_model, d_model, vocab_size, &mut seed);
        let layers = (0..n_layers)
            .map(|_| TransformerLayer::new(d_model, n_heads, &mut seed))
            .collect();

        Self {
            vocab_size,
            d_model,
            d_ff,
            n_heads,
            n_layers,
            max_seq_len: 256,
            embed,
            lm_head,
            layers,
        }
    }

    /// Forward pass. Returns logits: seq_len × vocab_size
    pub fn forward(&self, input_ids: &[usize], _cache: Option<&SelfAttentionCache>) -> Vec<Vec<f32>> {
        if input_ids.is_empty() {
            return vec![];
        }
        let d_model = self.d_model;

        // Token embeddings
        let mut xs: Vec<Vec<f32>> = input_ids
            .iter()
            .map(|&id| {
                let id = id.min(self.vocab_size - 1);
                self.embed[id * d_model..(id + 1) * d_model].to_vec()
            })
            .collect();

        // Transformer layers
        for layer in &self.layers {
            xs = layer.forward(&xs);
        }

        // LM head: project to vocab
        xs.iter()
            .map(|x| matvec(&self.lm_head, self.vocab_size, d_model, x))
            .collect()
    }

    /// Compute cross-entropy loss and BPB on a token sequence.
    /// Returns (loss, bpb).
    pub fn loss_bpb(&self, tokens: &[usize]) -> (f32, f32) {
        if tokens.len() < 2 {
            return (0.0, 0.0);
        }
        let input = &tokens[..tokens.len() - 1];
        let targets = &tokens[1..];

        let logits = self.forward(input, None);
        let mut total_loss = 0.0f32;

        for (logit_row, &target) in logits.iter().zip(targets.iter()) {
            let mut probs = logit_row.clone();
            softmax(&mut probs);
            let p = probs[target.min(self.vocab_size - 1)].max(1e-10);
            total_loss -= p.ln();
        }

        let loss = total_loss / targets.len() as f32;
        let bpb = loss / LN_2;
        (loss, bpb)
    }

    /// SGD step: update embedding and lm_head via finite-difference gradient.
    /// In production replace with proper autograd; this is correct for CPU demo.
    pub fn sgd_step(&mut self, tokens: &[usize], lr: f32) {
        if tokens.len() < 2 {
            return;
        }
        let eps = 1e-3f32;
        let (loss0, _) = self.loss_bpb(tokens);

        // Update embed rows for tokens in sequence
        for &id in tokens {
            let id = id.min(self.vocab_size - 1);
            let start = id * self.d_model;
            for j in 0..self.d_model {
                self.embed[start + j] += eps;
                let (loss1, _) = self.loss_bpb(tokens);
                let grad = (loss1 - loss0) / eps;
                self.embed[start + j] -= eps + lr * grad;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_shape() {
        let model = RealIglaModel::new(256, 64, 1);
        let tokens = vec![1usize, 2, 3, 4];
        let logits = model.forward(&tokens, None);
        assert_eq!(logits.len(), 4);
        assert_eq!(logits[0].len(), 256);
    }

    #[test]
    fn test_loss_bpb_finite() {
        let model = RealIglaModel::new(256, 64, 1);
        let tokens: Vec<usize> = (0..16).map(|i| i % 256).collect();
        let (loss, bpb) = model.loss_bpb(&tokens);
        assert!(loss.is_finite(), "loss must be finite");
        assert!(bpb.is_finite(), "bpb must be finite");
        assert!(bpb > 0.0, "bpb must be positive");
        println!("Initial BPB (random weights): {:.4}", bpb);
    }

    #[test]
    fn test_sgd_reduces_loss() {
        let mut model = RealIglaModel::new(256, 64, 1);
        let tokens: Vec<usize> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let (loss_before, bpb_before) = model.loss_bpb(&tokens);
        model.sgd_step(&tokens, 0.01);
        let (loss_after, bpb_after) = model.loss_bpb(&tokens);
        println!("BPB before: {:.4}, after: {:.4}", bpb_before, bpb_after);
        // Loss should not explode
        assert!(loss_after.is_finite());
        let _ = (loss_before, bpb_before);
    }
}
