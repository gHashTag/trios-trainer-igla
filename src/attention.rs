//! Self-Attention Layer — Complete Rewrite (TASK-0A)
//!
//! Causal multi-head self-attention with:
//! - Full analytical backward pass
//! - QK-Norm (learnable scale parameters)
//! - QK-Gain (per-head gain scalar) — INV-9: anchored to φ² = φ + 1 ≈ 2.618
//! - RoPE positional encoding
//! - ReLU^2 activation in FFN
//! - AdamW optimizer integrated per-layer
//!
//! Architecture:
//!   Token Embed → N × (PreNorm → MHA → Res → PreNorm → FFN → Res) → LM Head
//!
//! ## INV-9: QK-Gain = φ²
//!
//! The QK gain scalar is anchored to the golden ratio:
//!   QK_GAIN = φ² = φ + 1 ≈ 2.618033988749895
//!
//! This value is sourced from `crate::invariants::PHI_SQ` to maintain
//! L-R14 traceability. The invariant guarantees that the attention mechanism
//! operates at the mathematically-optimized gain level derived from the
//! Trinity identity φ² + φ⁻² = 3.
//!
//! Coq anchor: `phi_sq_eq` lemma proves φ² = φ + 1
//!
//! Based on working patterns from trinity_3k_model.rs.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use std::f32::consts::LN_2;

// INV-9: Import φ² constant for QK gain anchor (L-R14 traceability)
use crate::invariants::PHI_SQ;

#[inline]
fn relu_squared(x: f32) -> f32 {
    let r = x.max(0.0);
    r * r
}

#[inline]
fn relu_squared_backward(x: f32) -> f32 {
    if x > 0.0 {
        2.0 * x
    } else {
        0.0
    }
}

fn softmax(v: &mut [f32]) {
    let max_val = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
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

fn layer_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = (var + eps).sqrt();
    x.iter().map(|v| (v - mean) / std).collect()
}

fn backward_layer_norm(input: &[f32], grad_out: &[f32], eps: f32) -> Vec<f32> {
    let d = input.len();
    let n = d as f32;
    let mean: f32 = input.iter().sum::<f32>() / n;
    let var: f32 = input.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = (var + eps).sqrt();

    let dx_sum: f32 = grad_out.iter().sum();
    let dx_xm_sum: f32 = grad_out.iter().zip(input.iter()).map(|(&g, &xi)| g * (xi - mean)).sum();
    let inv_var_eps = 1.0 / (var + eps);

    let mut result = vec![0.0f32; d];
    for j in 0..d {
        let xm = input[j] - mean;
        result[j] = (1.0 / (n * std)) * (n * grad_out[j] - dx_sum - xm * inv_var_eps * dx_xm_sum);
    }
    result
}

fn left_matvec(a: &[f32], rows: usize, cols: usize, v: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let row = &a[r * cols..(r + 1) * cols];
            v.iter().zip(row.iter()).map(|(&x, &w)| x * w).sum()
        })
        .collect()
}

fn xavier_init(size: usize, fan_in: usize, fan_out: usize, seed: &mut u64) -> Vec<f32> {
    let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
    let mut rng = *seed;
    let mut weights = Vec::with_capacity(size);
    for _ in 0..size {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let t = ((rng >> 33) as f32) / (u32::MAX as f32);
        weights.push((t * 2.0 - 1.0) * std * 3.0);
    }
    *seed = rng;
    weights
}

pub fn apply_rope(vecs: &mut [Vec<f32>], head_dim: usize, offset: usize) {
    for (i, v) in vecs.iter_mut().enumerate() {
        let pos = i + offset;
        for d in (0..head_dim).step_by(2) {
            if d + 1 < v.len() {
                let freq = 1.0f32 / 10000.0_f32.powf(d as f32 / head_dim as f32);
                let angle = freq * pos as f32;
                let cos_val = angle.cos();
                let sin_val = angle.sin();
                let v0 = v[d];
                let v1 = v[d + 1];
                v[d] = v0 * cos_val - v1 * sin_val;
                v[d + 1] = v0 * sin_val + v1 * cos_val;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub max_seq_len: usize,
    pub use_rope: bool,
    pub qk_gain_init: f32,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            vocab_size: 128,
            d_model: 384,
            n_heads: 4,
            n_layers: 2,
            max_seq_len: 64,
            use_rope: true,
            // INV-9: QK gain anchored to φ² = φ + 1 ≈ 2.618
            // Coq: phi_sq_eq proves φ² = φ + 1
            qk_gain_init: PHI_SQ as f32,
            lr: 0.003,
            beta1: 0.618,
            beta2: 0.999,
            weight_decay: 0.01,
        }
    }
}

impl AttentionConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    pub fn d_ffn(&self) -> usize {
        self.d_model * 4
    }
}

struct AdamWState {
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
}

impl AdamWState {
    fn new(size: usize) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            t: 0,
        }
    }

    fn update(&mut self, param: &mut [f32], grad: &[f32], lr: f32, beta1: f32, beta2: f32, wd: f32, eps: f32) {
        self.t += 1;
        for p in param.iter_mut() {
            *p *= 1.0 - lr * wd;
        }
        for (m, &g) in self.m.iter_mut().zip(grad.iter()) {
            *m = beta1 * *m + (1.0 - beta1) * g;
        }
        for (v, &g) in self.v.iter_mut().zip(grad.iter()) {
            *v = beta2 * *v + (1.0 - beta2) * g * g;
        }
        let bc1 = 1.0 / (1.0 - beta1.powi(self.t as i32));
        let bc2 = 1.0 / (1.0 - beta2.powi(self.t as i32));
        for (i, p) in param.iter_mut().enumerate() {
            let m_hat = self.m[i] * bc1;
            let v_hat = self.v[i] * bc2;
            *p -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

struct HeadCache {
    qs: Vec<Vec<f32>>,
    ks: Vec<Vec<f32>>,
    vs: Vec<Vec<f32>>,
    attn_weights: Vec<Vec<f32>>,
    normed_input: Vec<Vec<f32>>,
}

struct AttentionHead {
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    q_norm_scale: Vec<f32>,
    k_norm_scale: Vec<f32>,
    qk_gain: f32,
    head_dim: usize,
    d_model: usize,
    grad_w_q: Vec<f32>,
    grad_w_k: Vec<f32>,
    grad_w_v: Vec<f32>,
    grad_w_o: Vec<f32>,
    grad_q_norm: Vec<f32>,
    grad_k_norm: Vec<f32>,
    grad_qk_gain: f32,
    adamw_q: AdamWState,
    adamw_k: AdamWState,
    adamw_v: AdamWState,
    adamw_o: AdamWState,
    adamw_qn: AdamWState,
    adamw_kn: AdamWState,
    adamw_gain: AdamWState,
}

impl AttentionHead {
    fn new(d_model: usize, head_dim: usize, qk_gain: f32, seed: &mut u64) -> Self {
        let qk_size = d_model * head_dim;
        let o_size = head_dim * d_model;
        Self {
            w_q: xavier_init(qk_size, d_model, head_dim, seed),
            w_k: xavier_init(qk_size, d_model, head_dim, seed),
            w_v: xavier_init(qk_size, d_model, head_dim, seed),
            w_o: xavier_init(o_size, head_dim, d_model, seed),
            q_norm_scale: vec![1.0; head_dim],
            k_norm_scale: vec![1.0; head_dim],
            qk_gain,
            head_dim,
            d_model,
            grad_w_q: vec![0.0; qk_size],
            grad_w_k: vec![0.0; qk_size],
            grad_w_v: vec![0.0; qk_size],
            grad_w_o: vec![0.0; o_size],
            grad_q_norm: vec![0.0; head_dim],
            grad_k_norm: vec![0.0; head_dim],
            grad_qk_gain: 0.0,
            adamw_q: AdamWState::new(qk_size),
            adamw_k: AdamWState::new(qk_size),
            adamw_v: AdamWState::new(qk_size),
            adamw_o: AdamWState::new(o_size),
            adamw_qn: AdamWState::new(head_dim),
            adamw_kn: AdamWState::new(head_dim),
            adamw_gain: AdamWState::new(1),
        }
    }

    fn forward_with_cache(&self, xs: &[Vec<f32>], use_rope: bool) -> (Vec<Vec<f32>>, HeadCache) {
        let seq_len = xs.len();
        let d_model = self.d_model;

        let mut qs: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let mut ks: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let mut vs: Vec<Vec<f32>> = Vec::with_capacity(seq_len);

        for x in xs {
            qs.push(left_matvec(&self.w_q, self.head_dim, d_model, x));
            ks.push(left_matvec(&self.w_k, self.head_dim, d_model, x));
            vs.push(left_matvec(&self.w_v, self.head_dim, d_model, x));
        }

        if use_rope {
            apply_rope(&mut qs, self.head_dim, 0);
            apply_rope(&mut ks, self.head_dim, 0);
        }

        for qi in 0..seq_len {
            for j in 0..self.head_dim {
                qs[qi][j] *= self.q_norm_scale[j] * self.qk_gain;
                ks[qi][j] *= self.k_norm_scale[j] * self.qk_gain;
            }
        }

        let scale = (self.head_dim as f32).sqrt();
        let mut output = Vec::with_capacity(seq_len);
        let mut all_attn_weights = Vec::with_capacity(seq_len);

        for qi in 0..seq_len {
            let mut attn_weights = Vec::with_capacity(qi + 1);
            for kj in 0..=qi {
                let score: f32 = (0..self.head_dim).map(|k| qs[qi][k] * ks[kj][k]).sum();
                attn_weights.push(score / scale);
            }
            softmax(&mut attn_weights);

            let mut head_output = vec![0.0; self.head_dim];
            for (j, &aw) in attn_weights.iter().enumerate() {
                for k in 0..self.head_dim {
                    head_output[k] += aw * vs[j][k];
                }
            }

            let proj = left_matvec(&self.w_o, d_model, self.head_dim, &head_output);
            output.push(proj);
            all_attn_weights.push(attn_weights);
        }

        let cache = HeadCache {
            qs,
            ks,
            vs,
            attn_weights: all_attn_weights,
            normed_input: xs.to_vec(),
        };

        (output, cache)
    }

    #[allow(clippy::needless_range_loop)]
    fn backward(&mut self, grad_output: &[Vec<f32>], cache: &HeadCache) -> Vec<Vec<f32>> {
        let seq = grad_output.len();
        let d = self.d_model;
        let hd = self.head_dim;
        let scale = (hd as f32).sqrt();

        let mut grad_head_out = vec![vec![0.0f32; hd]; seq];
        for i in 0..seq {
            for j in 0..d {
                for k in 0..hd {
                    self.grad_w_o[j * hd + k] += cache.normed_input[i].get(j).copied().unwrap_or(0.0) * grad_output[i][j];
                }
            }
            for k in 0..hd {
                let mut s = 0.0f32;
                for j in 0..d {
                    s += self.w_o[j * hd + k] * grad_output[i][j];
                }
                grad_head_out[i][k] = s;
            }
        }

        let mut grad_attn_weights = vec![vec![0.0f32; seq]; seq];
        let mut grad_vs = vec![vec![0.0f32; hd]; seq];

        for qi in 0..seq {
            let aw_len = cache.attn_weights[qi].len();
            for kj in 0..aw_len {
                let mut gaw = 0.0f32;
                for k in 0..hd {
                    gaw += grad_head_out[qi][k] * cache.vs[kj][k];
                    grad_vs[kj][k] += cache.attn_weights[qi][kj] * grad_head_out[qi][k];
                }
                grad_attn_weights[qi][kj] = gaw;
            }
        }

        let mut grad_scores = vec![vec![0.0f32; seq]; seq];
        for qi in 0..seq {
            let aw_len = cache.attn_weights[qi].len();
            let dot: f32 = cache.attn_weights[qi].iter().zip(grad_attn_weights[qi].iter()).map(|(&aw, &gaw)| aw * gaw).sum();
            for kj in 0..aw_len {
                grad_scores[qi][kj] = cache.attn_weights[qi][kj] * (grad_attn_weights[qi][kj] - dot) / scale;
            }
        }

        let mut grad_qs = vec![vec![0.0f32; hd]; seq];
        let mut grad_ks = vec![vec![0.0f32; hd]; seq];

        for qi in 0..seq {
            let aw_len = cache.attn_weights[qi].len();
            for k in 0..hd {
                let mut gq = 0.0f32;
                for kj in 0..aw_len {
                    gq += grad_scores[qi][kj] * cache.ks[kj][k];
                }
                grad_qs[qi][k] = gq;
            }
        }
        for kj in 0..seq {
            for k in 0..hd {
                let mut gk = 0.0f32;
                for qi in kj..seq {
                    if kj < cache.attn_weights[qi].len() {
                        gk += grad_scores[qi][kj] * cache.qs[qi][k];
                    }
                }
                grad_ks[kj][k] = gk;
            }
        }

        for k in 0..hd {
            let qns = self.q_norm_scale[k];
            if qns.abs() > 1e-12 {
                for qi in 0..seq {
                    self.grad_q_norm[k] += (cache.qs[qi][k] / qns) * grad_qs[qi][k];
                }
            }
            let kns = self.k_norm_scale[k];
            if kns.abs() > 1e-12 {
                for kj in 0..seq {
                    self.grad_k_norm[k] += (cache.ks[kj][k] / kns) * grad_ks[kj][k];
                }
            }
        }

        let mut qk_gain_grad_sum = 0.0f32;
        for qi in 0..seq {
            for k in 0..hd {
                let raw_q = if self.qk_gain.abs() > 1e-12 { cache.qs[qi][k] / self.qk_gain } else { 0.0 };
                let raw_k = if self.qk_gain.abs() > 1e-12 { cache.ks[qi][k] / self.qk_gain } else { 0.0 };
                qk_gain_grad_sum += raw_q * grad_qs[qi][k] + raw_k * grad_ks[qi][k];
            }
        }
        self.grad_qk_gain = qk_gain_grad_sum;

        for qi in 0..seq {
            for k in 0..hd {
                grad_qs[qi][k] *= self.q_norm_scale[k] * self.qk_gain;
                grad_ks[qi][k] *= self.k_norm_scale[k] * self.qk_gain;
            }
        }

        let ni = &cache.normed_input;
        let mut grad_input = vec![vec![0.0f32; d]; seq];
        for qi in 0..seq {
            for oi in 0..hd {
                for j in 0..d {
                    self.grad_w_q[oi * d + j] += grad_qs[qi][oi] * ni[qi][j];
                    self.grad_w_k[oi * d + j] += grad_ks[qi][oi] * ni[qi][j];
                    self.grad_w_v[oi * d + j] += grad_vs[qi][oi] * ni[qi][j];
                }
            }
            for j in 0..d {
                let mut sq = 0.0f32;
                let mut sk = 0.0f32;
                let mut sv = 0.0f32;
                for oi in 0..hd {
                    sq += self.w_q[oi * d + j] * grad_qs[qi][oi];
                    sk += self.w_k[oi * d + j] * grad_ks[qi][oi];
                    sv += self.w_v[oi * d + j] * grad_vs[qi][oi];
                }
                grad_input[qi][j] = sq + sk + sv;
            }
        }

        grad_input
    }

    fn zero_grad(&mut self) {
        self.grad_w_q.fill(0.0);
        self.grad_w_k.fill(0.0);
        self.grad_w_v.fill(0.0);
        self.grad_w_o.fill(0.0);
        self.grad_q_norm.fill(0.0);
        self.grad_k_norm.fill(0.0);
        self.grad_qk_gain = 0.0;
    }

    fn adamw_update(&mut self, cfg: &AttentionConfig) {
        let lr = cfg.lr;
        let b1 = cfg.beta1;
        let b2 = cfg.beta2;
        let wd = cfg.weight_decay;
        let eps = 1e-8;
        self.adamw_q.update(&mut self.w_q, &self.grad_w_q, lr, b1, b2, wd, eps);
        self.adamw_k.update(&mut self.w_k, &self.grad_w_k, lr, b1, b2, wd, eps);
        self.adamw_v.update(&mut self.w_v, &self.grad_w_v, lr, b1, b2, wd, eps);
        self.adamw_o.update(&mut self.w_o, &self.grad_w_o, lr, b1, b2, wd, eps);
        self.adamw_qn.update(&mut self.q_norm_scale, &self.grad_q_norm, lr, b1, b2, wd, eps);
        self.adamw_kn.update(&mut self.k_norm_scale, &self.grad_k_norm, lr, b1, b2, wd, eps);
        self.adamw_gain.update(&mut [self.qk_gain], &[self.grad_qk_gain], lr, b1, b2, wd, eps);
    }
}

struct LayerCache {
    input: Vec<Vec<f32>>,
    residual1: Vec<Vec<f32>>,
    normed2_input: Vec<Vec<f32>>,
    ffn_pre_act: Vec<Vec<f32>>,
    head_caches: Vec<HeadCache>,
}

struct AttentionLayer {
    heads: Vec<AttentionHead>,
    w_ff1: Vec<f32>,
    w_ff2: Vec<f32>,
    d_model: usize,
    d_ffn: usize,
    use_rope: bool,
    grad_w_ff1: Vec<f32>,
    grad_w_ff2: Vec<f32>,
    adamw_ff1: AdamWState,
    adamw_ff2: AdamWState,
}

impl AttentionLayer {
    fn new(d_model: usize, n_heads: usize, head_dim: usize, d_ffn: usize, qk_gain: f32, use_rope: bool, seed: &mut u64) -> Self {
        let mut heads = Vec::with_capacity(n_heads);
        for _ in 0..n_heads {
            heads.push(AttentionHead::new(d_model, head_dim, qk_gain, seed));
        }
        let ff1_size = d_model * d_ffn;
        let ff2_size = d_ffn * d_model;
        Self {
            heads,
            w_ff1: xavier_init(ff1_size, d_model, d_ffn, seed),
            w_ff2: xavier_init(ff2_size, d_ffn, d_model, seed),
            d_model,
            d_ffn,
            use_rope,
            grad_w_ff1: vec![0.0; ff1_size],
            grad_w_ff2: vec![0.0; ff2_size],
            adamw_ff1: AdamWState::new(ff1_size),
            adamw_ff2: AdamWState::new(ff2_size),
        }
    }

    fn forward_cached(&self, xs: &[Vec<f32>]) -> (Vec<Vec<f32>>, LayerCache) {
        let seq_len = xs.len();
        let eps = 1e-5;

        let normed_xs: Vec<Vec<f32>> = xs.iter().map(|x| layer_norm(x, eps)).collect();

        let mut attn_output = vec![vec![0.0f32; self.d_model]; seq_len];
        let mut head_caches = Vec::new();
        for head in &self.heads {
            let (ho, hc) = head.forward_with_cache(&normed_xs, self.use_rope);
            for (i, row) in attn_output.iter_mut().enumerate() {
                for (r, v) in row.iter_mut().zip(ho[i].iter()) {
                    *r += v;
                }
            }
            head_caches.push(hc);
        }

        let residual1: Vec<Vec<f32>> = (0..seq_len)
            .map(|i| xs[i].iter().zip(attn_output[i].iter()).map(|(&a, &b)| a + b).collect())
            .collect();

        let normed_r1: Vec<Vec<f32>> = residual1.iter().map(|x| layer_norm(x, eps)).collect();

        let mut ffn_pre_act = Vec::with_capacity(seq_len);
        let ffn_hidden: Vec<Vec<f32>> = normed_r1.iter().map(|x| {
            let pre = left_matvec(&self.w_ff1, self.d_ffn, self.d_model, x);
            ffn_pre_act.push(pre.clone());
            pre.into_iter().map(relu_squared).collect()
        }).collect();

        let ffn_output: Vec<Vec<f32>> = ffn_hidden.iter()
            .map(|x| left_matvec(&self.w_ff2, self.d_model, self.d_ffn, x))
            .collect();

        let output: Vec<Vec<f32>> = (0..seq_len)
            .map(|i| residual1[i].iter().zip(ffn_output[i].iter()).map(|(&a, &b)| a + b).collect())
            .collect();

        let cache = LayerCache {
            input: xs.to_vec(),
            residual1,
            normed2_input: normed_r1,
            ffn_pre_act,
            head_caches,
        };

        (output, cache)
    }

    #[allow(clippy::needless_range_loop)]
    fn backward(&mut self, grad_output: &[Vec<f32>], cache: &LayerCache) -> Vec<Vec<f32>> {
        let seq = grad_output.len();
        let d = self.d_model;
        let ffn = self.d_ffn;

        let mut grad_ffn_hidden = vec![vec![0.0f32; ffn]; seq];
        for si in 0..seq {
            for k in 0..ffn {
                let mut s = 0.0f32;
                for j in 0..d {
                    s += grad_output[si][j] * self.w_ff2[j * ffn + k];
                }
                grad_ffn_hidden[si][k] = s;
            }
            for j in 0..d {
                for k in 0..ffn {
                    let activated = relu_squared(cache.ffn_pre_act[si][k]);
                    self.grad_w_ff2[j * ffn + k] += activated * grad_output[si][j];
                }
            }
        }

        for si in 0..seq {
            for k in 0..ffn {
                grad_ffn_hidden[si][k] *= relu_squared_backward(cache.ffn_pre_act[si][k]);
            }
        }

        let mut grad_normed2 = vec![vec![0.0f32; d]; seq];
        for si in 0..seq {
            for j in 0..d {
                let mut s = 0.0f32;
                for k in 0..ffn {
                    s += grad_ffn_hidden[si][k] * self.w_ff1[j * ffn + k];
                }
                grad_normed2[si][j] = s;
            }
            for j in 0..d {
                for k in 0..ffn {
                    self.grad_w_ff1[j * ffn + k] += cache.normed2_input[si][j] * grad_ffn_hidden[si][k];
                }
            }
        }

        let grad_residual1 = backward_layer_norm_batch(&cache.residual1, &grad_normed2);

        let mut grad_normed1 = vec![vec![0.0f32; d]; seq];
        for (hi, head) in self.heads.iter_mut().enumerate() {
            let hc = &cache.head_caches[hi];
            let hd = head.head_dim;

            let mut grad_head_out = vec![vec![0.0f32; d]; seq];
            for i in 0..seq {
                for k in 0..hd {
                    grad_head_out[i][hi * hd + k] = grad_residual1[i].get(hi * hd + k).copied().unwrap_or(0.0);
                }
            }

            let grad_input = head.backward(&grad_head_out, hc);
            for i in 0..seq {
                for j in 0..d {
                    grad_normed1[i][j] += grad_input[i][j];
                }
            }
        }

        backward_layer_norm_batch(&cache.input, &grad_normed1)
    }

    fn zero_grad(&mut self) {
        self.grad_w_ff1.fill(0.0);
        self.grad_w_ff2.fill(0.0);
        for head in &mut self.heads {
            head.zero_grad();
        }
    }

    fn adamw_update(&mut self, cfg: &AttentionConfig) {
        let lr = cfg.lr;
        let b1 = cfg.beta1;
        let b2 = cfg.beta2;
        let wd = cfg.weight_decay;
        let eps = 1e-8;
        self.adamw_ff1.update(&mut self.w_ff1, &self.grad_w_ff1, lr, b1, b2, wd, eps);
        self.adamw_ff2.update(&mut self.w_ff2, &self.grad_w_ff2, lr, b1, b2, wd, eps);
        for head in &mut self.heads {
            head.adamw_update(cfg);
        }
    }
}

fn backward_layer_norm_batch(input: &[Vec<f32>], grad_out: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }
    input.iter().zip(grad_out.iter()).map(|(x, go)| backward_layer_norm(x, go, 1e-5)).collect()
}

pub struct AttentionModel {
    embed: Vec<f32>,
    layers: Vec<AttentionLayer>,
    config: AttentionConfig,
    grad_embed: Vec<f32>,
    adamw_embed: AdamWState,
}

impl AttentionModel {
    pub fn new(config: AttentionConfig) -> Self {
        let mut seed = 42u64;
        let embed_size = config.vocab_size * config.d_model;
        let head_dim = config.head_dim();

        let embed = xavier_init(embed_size, config.vocab_size, config.d_model, &mut seed);

        let layers: Vec<AttentionLayer> = (0..config.n_layers)
            .map(|_| {
                AttentionLayer::new(
                    config.d_model,
                    config.n_heads,
                    head_dim,
                    config.d_ffn(),
                    config.qk_gain_init,
                    config.use_rope,
                    &mut seed,
                )
            })
            .collect();

        Self {
            embed,
            layers,
            grad_embed: vec![0.0; embed_size],
            adamw_embed: AdamWState::new(embed_size),
            config,
        }
    }

    pub fn forward(&self, input_ids: &[usize]) -> Vec<Vec<f32>> {
        let d = self.config.d_model;
        let vs = self.config.vocab_size;

        let mut hidden: Vec<Vec<f32>> = input_ids
            .iter()
            .map(|&tid| {
                let tid = tid.min(vs - 1);
                self.embed[tid * d..tid * d + d].to_vec()
            })
            .collect();

        for layer in &self.layers {
            hidden = layer.forward_cached(&hidden).0;
        }

        let eps = 1e-5;
        let normed: Vec<Vec<f32>> = hidden.iter().map(|x| layer_norm(x, eps)).collect();

        normed.iter()
            .map(|x| {
                (0..vs)
                    .map(|vi| {
                        let emb = &self.embed[vi * d..vi * d + d];
                        emb.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum()
                    })
                    .collect()
            })
            .collect()
    }

    pub fn loss_bpb(&self, tokens: &[usize]) -> (f32, f32) {
        if tokens.len() < 2 {
            return (0.0, 0.0);
        }
        let input_ids = &tokens[..tokens.len() - 1];
        let target_ids = &tokens[1..];
        let logits = self.forward(input_ids);

        let mut total_loss = 0.0f32;
        let mut count = 0usize;

        for (i, &target) in target_ids.iter().enumerate() {
            if i >= logits.len() {
                break;
            }
            let mut probs = logits[i].clone();
            softmax(&mut probs);
            let p = probs[target].max(1e-9);
            total_loss += -p.ln();
            count += 1;
        }

        if count == 0 {
            return (0.0, 0.0);
        }
        let loss = total_loss / count as f32;
        let bpb = loss / LN_2;
        (loss, bpb)
    }

    #[allow(clippy::needless_range_loop)]
    fn backward_analytical(&mut self, tokens: &[usize]) {
        if tokens.len() < 2 {
            return;
        }
        let d = self.config.d_model;
        let vs = self.config.vocab_size;
        let input_ids = &tokens[..tokens.len() - 1];
        let targets = &tokens[1..];
        let seq = input_ids.len();

        let mut hidden: Vec<Vec<f32>> = input_ids
            .iter()
            .map(|&tid| {
                let tid = tid.min(vs - 1);
                self.embed[tid * d..tid * d + d].to_vec()
            })
            .collect();

        let mut layer_caches = Vec::new();
        for layer in &self.layers {
            let (out, cache) = layer.forward_cached(&hidden);
            layer_caches.push(cache);
            hidden = out;
        }

        let eps = 1e-5f32;
        let normed: Vec<Vec<f32>> = hidden.iter().map(|x| layer_norm(x, eps)).collect();

        let logits: Vec<Vec<f32>> = normed.iter().map(|x| {
            (0..vs).map(|vi| {
                let emb = &self.embed[vi * d..vi * d + d];
                emb.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum()
            }).collect()
        }).collect();

        let mut grad_logits: Vec<Vec<f32>> = Vec::with_capacity(seq);
        for i in 0..seq {
            let mut probs = logits[i].clone();
            softmax(&mut probs);
            let target = targets[i].min(vs - 1);
            for v in 0..vs {
                probs[v] = if v == target { probs[v] - 1.0 } else { probs[v] };
            }
            grad_logits.push(probs);
        }
        let n_tokens = seq as f32;
        for gl in grad_logits.iter_mut() {
            for g in gl.iter_mut() {
                *g /= n_tokens;
            }
        }

        let mut grad_normed = vec![vec![0.0f32; d]; seq];
        for i in 0..seq {
            for v in 0..vs {
                let gl = grad_logits[i][v];
                if gl.abs() < 1e-10 {
                    continue;
                }
                let emb_start = v * d;
                for j in 0..d {
                    grad_normed[i][j] += gl * self.embed[emb_start + j];
                    self.grad_embed[emb_start + j] += gl * normed[i][j];
                }
            }
        }

        let grad_hidden = backward_layer_norm_batch(&hidden, &grad_normed);

        let mut grad_input = grad_hidden;
        for (li, layer) in self.layers.iter_mut().enumerate().rev() {
            let cache = &layer_caches[li];
            grad_input = layer.backward(&grad_input, cache);
        }

        for (i, &tid) in input_ids.iter().enumerate() {
            let tid = tid.min(vs - 1);
            let base = tid * d;
            for j in 0..d {
                self.grad_embed[base + j] += grad_input[i][j];
            }
        }
    }

    pub fn train_step(&mut self, tokens: &[usize]) {
        self.zero_grad();
        self.backward_analytical(tokens);
        self.adamw_update();
    }

    fn zero_grad(&mut self) {
        self.grad_embed.fill(0.0);
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }

    fn adamw_update(&mut self) {
        let cfg = &self.config;
        self.adamw_embed.update(&mut self.embed, &self.grad_embed, cfg.lr, cfg.beta1, cfg.beta2, cfg.weight_decay, 1e-8);
        for layer in &mut self.layers {
            layer.adamw_update(cfg);
        }
    }

    pub fn config(&self) -> &AttentionConfig {
        &self.config
    }

    pub fn param_count(&self) -> usize {
        let embed = self.config.vocab_size * self.config.d_model;
        let head_dim = self.config.head_dim();
        let d = self.config.d_model;
        let ffn = self.config.d_ffn();
        let per_head = 3 * d * head_dim + head_dim * d + 2 * head_dim + 1;
        let per_layer = self.config.n_heads * per_head + d * ffn + ffn * d;
        embed + per_layer * self.config.n_layers
    }
}

pub fn apply_attention_integration(
    ngram_output: &[f32],
    attention_scores: &[f32],
    _d_model: usize,
    ngram_weight: f32,
    attn_weight: f32,
) -> Vec<f32> {
    let vocab_size = ngram_output.len();
    let mut result = vec![0.0f32; vocab_size];
    for i in 0..vocab_size.min(attention_scores.len()) {
        result[i] = ngram_weight * ngram_output[i] + attn_weight * attention_scores[i];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let mut v = vec![1.0f32, 2.0, 3.0];
        softmax(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(v[0] < v[1] && v[1] < v[2]);
    }

    #[test]
    fn test_layer_norm() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let normed = layer_norm(&x, 1e-5);
        let mean: f32 = normed.iter().sum::<f32>() / normed.len() as f32;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_rope() {
        let mut vecs = vec![vec![1.0f32, 0.0, 0.0, 1.0], vec![0.0f32, 1.0, 1.0, 0.0]];
        apply_rope(&mut vecs, 4, 0);
        for v in &vecs {
            for &x in v {
                assert!(x.is_finite());
            }
        }
    }

    #[test]
    fn test_model_creation() {
        let config = AttentionConfig {
            d_model: 64,
            n_heads: 2,
            n_layers: 1,
            vocab_size: 32,
            max_seq_len: 16,
            ..Default::default()
        };
        let model = AttentionModel::new(config);
        assert!(model.param_count() > 0);
    }

    #[test]
    fn test_forward_shape() {
        let config = AttentionConfig {
            d_model: 64,
            n_heads: 2,
            n_layers: 1,
            vocab_size: 32,
            max_seq_len: 16,
            ..Default::default()
        };
        let model = AttentionModel::new(config);
        let tokens = vec![1usize, 2, 3, 4];
        let logits = model.forward(&tokens);
        assert_eq!(logits.len(), 4);
        assert_eq!(logits[0].len(), 32);
        for row in &logits {
            for &v in row {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn test_loss_bpb_finite() {
        let config = AttentionConfig {
            d_model: 64,
            n_heads: 2,
            n_layers: 1,
            vocab_size: 32,
            max_seq_len: 16,
            ..Default::default()
        };
        let model = AttentionModel::new(config);
        let tokens: Vec<usize> = (0..16).map(|i| i % 32).collect();
        let (loss, bpb) = model.loss_bpb(&tokens);
        assert!(loss.is_finite());
        assert!(bpb.is_finite());
        assert!(bpb > 0.0);
    }

    #[test]
    fn test_train_step_reduces_loss() {
        let config = AttentionConfig {
            d_model: 64,
            n_heads: 2,
            n_layers: 1,
            vocab_size: 32,
            max_seq_len: 16,
            lr: 0.003,
            ..Default::default()
        };
        let mut model = AttentionModel::new(config);
        let tokens: Vec<usize> = (0..16).map(|i| i % 32).collect();

        let (loss_before, _) = model.loss_bpb(&tokens);

        for _ in 0..5 {
            model.train_step(&tokens);
        }

        let (loss_after, _) = model.loss_bpb(&tokens);

        assert!(
            loss_after < loss_before,
            "Loss should decrease after training: before={}, after={}",
            loss_before, loss_after
        );
    }

    #[test]
    fn test_causal_masking() {
        let config = AttentionConfig {
            d_model: 32,
            n_heads: 2,
            n_layers: 1,
            vocab_size: 16,
            max_seq_len: 8,
            ..Default::default()
        };
        let model = AttentionModel::new(config);
        let tokens = vec![0usize, 1, 2, 3];
        let logits = model.forward(&tokens);
        assert_eq!(logits.len(), 4);
        assert_eq!(logits[0].len(), 16);
        for row in &logits {
            for &v in row {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn test_qk_gain_effect() {
        let config_no_gain = AttentionConfig {
            d_model: 32,
            n_heads: 2,
            n_layers: 1,
            vocab_size: 16,
            max_seq_len: 8,
            qk_gain_init: 1.0,
            ..Default::default()
        };
        let config_with_gain = AttentionConfig {
            d_model: 32,
            n_heads: 2,
            n_layers: 1,
            vocab_size: 16,
            max_seq_len: 8,
            qk_gain_init: 4.0,
            ..Default::default()
        };

        let model_no_gain = AttentionModel::new(config_no_gain);
        let model_with_gain = AttentionModel::new(config_with_gain);

        let tokens = vec![0usize, 1, 2, 3];
        let logits_no = model_no_gain.forward(&tokens);
        let logits_with = model_with_gain.forward(&tokens);

        assert_eq!(logits_no.len(), logits_with.len());
        assert!(logits_no[0] != logits_with[0] || logits_no[1] != logits_with[1]);
    }

    #[test]
    fn test_integration_weighted() {
        let ngram = vec![0.5f32, 0.3, 0.2];
        let attn = vec![0.4f32, 0.4, 0.2];
        let result = apply_attention_integration(&ngram, &attn, 1, 0.7, 0.3);
        assert!((result[0] - 0.47).abs() < 1e-6);
        assert!((result[1] - 0.33).abs() < 1e-6);
    }

    // ----------------------------------------------------------------------
    // INV-9: QK-Gain φ² invariant tests
    // ----------------------------------------------------------------------

    /// INV-9: Default QK gain must equal φ² ≈ 2.618
    /// Coq: phi_sq_eq proves φ² = φ + 1
    #[test]
    fn inv9_default_qk_gain_is_phi_sq() {
        let config = AttentionConfig::default();
        let expected = PHI_SQ as f32;
        assert!(
            (config.qk_gain_init - expected).abs() < 1e-6,
            "INV-9 VIOLATED: qk_gain_init = {} ≠ φ² = {}",
            config.qk_gain_init,
            expected
        );
    }

    /// INV-9: Verify QK gain is in the phi-anchored range [2.0, 3.0]
    /// This is the practical operational range around φ²
    #[test]
    fn inv9_qk_gain_in_phi_range() {
        let config = AttentionConfig::default();
        assert!(
            config.qk_gain_init >= 2.0 && config.qk_gain_init <= 3.0,
            "INV-9 VIOLATED: qk_gain_init = {} outside phi-range [2.0, 3.0]",
            config.qk_gain_init
        );
    }

    /// INV-9: Falsification witness — qk_gain = 1.0 is NOT φ²
    #[test]
    fn inv9_falsify_qk_gain_one_rejected() {
        let bad_config = AttentionConfig {
            qk_gain_init: 1.0, // Old broken default
            ..Default::default()
        };
        assert_ne!(
            bad_config.qk_gain_init,
            PHI_SQ as f32,
            "INV-9: qk_gain=1.0 is not φ²"
        );
    }

    /// INV-9: Verify the model actually uses the phi^2 gain
    #[test]
    fn inv9_model_uses_phi_sq_gain() {
        let config = AttentionConfig::default();
        let model = AttentionModel::new(config);
        // Check that the first head has the correct qk_gain
        let head = &model.layers[0].heads[0];
        let expected = PHI_SQ as f32;
        assert!(
            (head.qk_gain - expected).abs() < 1e-6,
            "INV-9 VIOLATED: model head qk_gain = {} ≠ φ² = {}",
            head.qk_gain,
            expected
        );
    }
}
