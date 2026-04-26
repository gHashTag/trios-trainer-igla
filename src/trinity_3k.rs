//! Trinity 3k Model for Parameter Golf #110 - Pure Rust Implementation
//!
//! Byte-level Trinity 3^k transformer:
//! - vocab_size: 729 (3^6)
//! - hidden_dim: 243 (3^5)
//! - n_heads: 27 (3^3)
//! - head_dim: 9 (3^2)
//! - activation: ReLU^2
//! - normalization: QK-Norm + LayerNorm

use std::f32::consts::LN_2;

pub struct AdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
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

    fn update(&mut self, param: &mut [f32], grad: &[f32], cfg: &AdamWConfig) {
        self.t += 1;

        for p in param.iter_mut() {
            *p *= 1.0 - cfg.lr * cfg.weight_decay;
        }

        for (m, &g) in self.m.iter_mut().zip(grad.iter()) {
            *m = cfg.beta1 * *m + (1.0 - cfg.beta1) * g;
        }

        for (v, &g) in self.v.iter_mut().zip(grad.iter()) {
            *v = cfg.beta2 * *v + (1.0 - cfg.beta2) * g * g;
        }

        let bc1 = 1.0 / (1.0 - cfg.beta1.powi(self.t as i32));
        let bc2 = 1.0 / (1.0 - cfg.beta2.powi(self.t as i32));

        for (i, p) in param.iter_mut().enumerate() {
            let m_hat = self.m[i] * bc1;
            let v_hat = self.v[i] * bc2;
            *p -= cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps);
        }
    }
}

#[inline]
fn relu_squared(x: f32) -> f32 {
    let r = x.max(0.0);
    r * r
}

fn softmax(v: &mut [f32]) {
    let max_val = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    for x in v.iter_mut() {
        *x /= sum;
    }
}

fn layer_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = (var + eps).sqrt();
    x.iter().map(|v| (v - mean) / std).collect()
}

fn left_matvec(a: &[f32], rows: usize, cols: usize, v: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), rows * cols);
    assert_eq!(v.len(), cols);
    (0..rows)
        .map(|r| {
            let row = &a[r * cols..(r + 1) * cols];
            v.iter().zip(row.iter()).map(|(&x, &w)| x * w).sum()
        })
        .collect()
}

fn left_matvec_single(a: &[f32], _rows: usize, cols: usize, v: &[f32], out_idx: usize) -> f32 {
    let row = &a[out_idx * cols..(out_idx + 1) * cols];
    v.iter().zip(row.iter()).map(|(&x, &w)| x * w).sum()
}

fn xavier_phi_init(size: usize, fan_in: usize, fan_out: usize, layer_idx: usize, total_layers: usize, seed: &mut u64) -> Vec<f32> {
    let phi: f64 = 1.618033988749895;
    let phi_scale = phi.powf(-(layer_idx as f64 / total_layers as f64));
    let std = (2.0 / (fan_in + fan_out) as f32).sqrt() * phi_scale as f32;

    let mut rng = *seed;
    let mut weights = Vec::with_capacity(size);

    for _ in 0..size {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let uniform = (rng & 0x7fffffff) as f32 / 2147483648.0;
        weights.push((uniform - 0.5) * 2.0 * std * 3.0);
    }

    *seed = rng;
    weights
}

struct Trinity3kAttentionHead {
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    q_norm_scale: Vec<f32>,
    k_norm_scale: Vec<f32>,
    grad_w_q: Vec<f32>,
    grad_w_k: Vec<f32>,
    grad_w_v: Vec<f32>,
    grad_w_o: Vec<f32>,
    grad_q_norm: Vec<f32>,
    grad_k_norm: Vec<f32>,
    adamw_w_q: AdamWState,
    adamw_w_k: AdamWState,
    adamw_w_v: AdamWState,
    adamw_w_o: AdamWState,
    adamw_q_norm: AdamWState,
    adamw_k_norm: AdamWState,
    head_dim: usize,
}

impl Trinity3kAttentionHead {
    fn new(d_model: usize, head_dim: usize, layer_idx: usize, total_layers: usize, seed: &mut u64) -> Self {
        let qk_size = d_model * head_dim;
        let o_size = head_dim * head_dim;
        Self {
            w_q: xavier_phi_init(qk_size, d_model, head_dim, layer_idx, total_layers, seed),
            w_k: xavier_phi_init(qk_size, d_model, head_dim, layer_idx, total_layers, seed),
            w_v: xavier_phi_init(qk_size, d_model, head_dim, layer_idx, total_layers, seed),
            w_o: xavier_phi_init(o_size, head_dim, head_dim, layer_idx, total_layers, seed),
            q_norm_scale: vec![1.0; head_dim],
            k_norm_scale: vec![1.0; head_dim],
            grad_w_q: vec![0.0; qk_size],
            grad_w_k: vec![0.0; qk_size],
            grad_w_v: vec![0.0; qk_size],
            grad_w_o: vec![0.0; o_size],
            grad_q_norm: vec![0.0; head_dim],
            grad_k_norm: vec![0.0; head_dim],
            adamw_w_q: AdamWState::new(qk_size),
            adamw_w_k: AdamWState::new(qk_size),
            adamw_w_v: AdamWState::new(qk_size),
            adamw_w_o: AdamWState::new(o_size),
            adamw_q_norm: AdamWState::new(head_dim),
            adamw_k_norm: AdamWState::new(head_dim),
            head_dim,
        }
    }

    fn forward_with_cache(&self, xs: &[Vec<f32>]) -> (Vec<Vec<f32>>, HeadCache) {
        let seq_len = xs.len();
        let d_model = xs[0].len();

        let mut qs: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let mut ks: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let mut vs: Vec<Vec<f32>> = Vec::with_capacity(seq_len);

        for x in xs {
            qs.push(left_matvec(&self.w_q, self.head_dim, d_model, x));
            ks.push(left_matvec(&self.w_k, self.head_dim, d_model, x));
            vs.push(left_matvec(&self.w_v, self.head_dim, d_model, x));
        }

        #[allow(clippy::needless_range_loop)]
        for qi in 0..seq_len {
            for j in 0..self.head_dim {
                qs[qi][j] *= self.q_norm_scale[j];
                ks[qi][j] *= self.k_norm_scale[j];
            }
        }

        let scale = (self.head_dim as f32).sqrt();
        let mut output = Vec::with_capacity(seq_len);
        let mut all_attn_weights = Vec::with_capacity(seq_len);
        #[allow(clippy::needless_range_loop)]
        for qi in 0..seq_len {
            let mut attn_weights = Vec::with_capacity(seq_len);
            #[allow(clippy::needless_range_loop)]
            for kj in 0..seq_len {
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
            output.push(head_output);
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

    fn zero_grad(&mut self) {
        fill_zero(&mut self.grad_w_q);
        fill_zero(&mut self.grad_w_k);
        fill_zero(&mut self.grad_w_v);
        fill_zero(&mut self.grad_w_o);
        fill_zero(&mut self.grad_q_norm);
        fill_zero(&mut self.grad_k_norm);
    }

    fn adamw_update(&mut self, cfg: &AdamWConfig) {
        self.adamw_w_q.update(&mut self.w_q, &self.grad_w_q.clone(), cfg);
        self.adamw_w_k.update(&mut self.w_k, &self.grad_w_k.clone(), cfg);
        self.adamw_w_v.update(&mut self.w_v, &self.grad_w_v.clone(), cfg);
        self.adamw_w_o.update(&mut self.w_o, &self.grad_w_o.clone(), cfg);
        self.adamw_q_norm.update(&mut self.q_norm_scale, &self.grad_q_norm.clone(), cfg);
        self.adamw_k_norm.update(&mut self.k_norm_scale, &self.grad_k_norm.clone(), cfg);
    }
}

fn fill_zero(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = 0.0;
    }
}

fn backward_layer_norm_batch(input: &[Vec<f32>], grad_out: &[Vec<f32>], eps: f32) -> Vec<Vec<f32>> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }
    let d = input[0].len();
    let mut result = vec![vec![0.0; d]; n];

    for i in 0..n {
        let x = &input[i];
        let go = &grad_out[i];
        let mean: f32 = x.iter().sum::<f32>() / d as f32;
        let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
        let std = (var + eps).sqrt();
        let inv_n_std = 1.0 / (d as f32 * std);

        let dx_sum: f32 = go.iter().sum();
        let dx_xm_sum: f32 = go.iter().zip(x.iter()).map(|(&g, &xi)| g * (xi - mean)).sum();
        let inv_var_eps = 1.0 / (var + eps);

        for j in 0..d {
            let xm = x[j] - mean;
            result[i][j] = inv_n_std * (d as f32 * go[j] - dx_sum - xm * inv_var_eps * dx_xm_sum);
        }
    }
    result
}

struct HeadCache {
    qs: Vec<Vec<f32>>,
    ks: Vec<Vec<f32>>,
    vs: Vec<Vec<f32>>,
    attn_weights: Vec<Vec<f32>>,
    normed_input: Vec<Vec<f32>>,
}

struct LayerCache {
    input: Vec<Vec<f32>>,
    #[allow(dead_code)]
    normed1: Vec<Vec<f32>>,
    #[allow(dead_code)]
    attn_out: Vec<Vec<f32>>,
    residual1: Vec<Vec<f32>>,
    normed2: Vec<Vec<f32>>,
    ffn_hidden_activated: Vec<Vec<f32>>,
    head_caches: Vec<HeadCache>,
}

struct Trinity3kLayer {
    attention_heads: Vec<Trinity3kAttentionHead>,
    w_ff1: Vec<f32>,
    w_ff2: Vec<f32>,
    norm1_scale: Vec<f32>,
    norm2_scale: Vec<f32>,
    grad_w_ff1: Vec<f32>,
    grad_w_ff2: Vec<f32>,
    grad_norm1: Vec<f32>,
    grad_norm2: Vec<f32>,
    adamw_w_ff1: AdamWState,
    adamw_w_ff2: AdamWState,
    adamw_norm1: AdamWState,
    adamw_norm2: AdamWState,
    d_model: usize,
    #[allow(dead_code)]
    n_heads: usize,
    ffn_dim: usize,
}

impl Trinity3kLayer {
    fn new(d_model: usize, n_heads: usize, layer_idx: usize, total_layers: usize, seed: &mut u64) -> Self {
        let head_dim = d_model / n_heads;
        let ffn_dim = d_model * 4;

        let mut attention_heads = Vec::with_capacity(n_heads);
        for _ in 0..n_heads {
            attention_heads.push(Trinity3kAttentionHead::new(d_model, head_dim, layer_idx, total_layers, seed));
        }

        let ff1_size = d_model * ffn_dim;
        let ff2_size = ffn_dim * d_model;

        Self {
            attention_heads,
            w_ff1: xavier_phi_init(ff1_size, d_model, ffn_dim, layer_idx, total_layers, seed),
            w_ff2: xavier_phi_init(ff2_size, ffn_dim, d_model, layer_idx, total_layers, seed),
            norm1_scale: vec![1.0; d_model],
            norm2_scale: vec![1.0; d_model],
            grad_w_ff1: vec![0.0; ff1_size],
            grad_w_ff2: vec![0.0; ff2_size],
            grad_norm1: vec![0.0; d_model],
            grad_norm2: vec![0.0; d_model],
            adamw_w_ff1: AdamWState::new(ff1_size),
            adamw_w_ff2: AdamWState::new(ff2_size),
            adamw_norm1: AdamWState::new(d_model),
            adamw_norm2: AdamWState::new(d_model),
            d_model,
            n_heads,
            ffn_dim,
        }
    }

    fn forward(&self, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let (out, _) = self.forward_cached(xs);
        out
    }

    fn forward_cached(&self, xs: &[Vec<f32>]) -> (Vec<Vec<f32>>, LayerCache) {
        let seq_len = xs.len();
        let eps = 1e-5;

        let normed_xs: Vec<Vec<f32>> = xs.iter().map(|x| layer_norm(x, eps)).collect();

        let mut head_outputs: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut head_caches: Vec<HeadCache> = Vec::new();
        for head in &self.attention_heads {
            let (ho, hc) = head.forward_with_cache(&normed_xs);
            head_outputs.push(ho);
            head_caches.push(hc);
        }

        let mut attn_output = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let mut concatenated: Vec<f32> = Vec::with_capacity(self.d_model);
            for ho in &head_outputs {
                concatenated.extend(&ho[i]);
            }
            attn_output.push(concatenated);
        }

        let residual1: Vec<Vec<f32>> = (0..seq_len)
            .map(|i| {
                (0..self.d_model)
                    .map(|j| xs[i][j] + attn_output[i][j])
                    .collect()
            })
            .collect();

        let normed_r1: Vec<Vec<f32>> = residual1.iter().map(|x| layer_norm(x, eps)).collect();

        let ffn_hidden_activated: Vec<Vec<f32>> = normed_r1
            .iter()
            .map(|x| {
                left_matvec(&self.w_ff1, self.ffn_dim, self.d_model, x)
                    .into_iter()
                    .map(relu_squared)
                    .collect()
            })
            .collect();

        let ffn_output: Vec<Vec<f32>> = ffn_hidden_activated
            .iter()
            .map(|x| left_matvec(&self.w_ff2, self.d_model, self.ffn_dim, x))
            .collect();

        let output: Vec<Vec<f32>> = (0..seq_len)
            .map(|i| {
                (0..self.d_model)
                    .map(|j| residual1[i][j] + ffn_output[i][j])
                    .collect()
            })
            .collect();

        let cache = LayerCache {
            input: xs.to_vec(),
            normed1: normed_xs,
            attn_out: attn_output,
            residual1,
            normed2: normed_r1,
            ffn_hidden_activated,
            head_caches,
        };

        (output, cache)
    }

    #[allow(clippy::needless_range_loop)]
    fn backward(&mut self, grad_output: &[Vec<f32>], cache: &LayerCache) -> Vec<Vec<f32>> {
        let seq = grad_output.len();
        let d = self.d_model;
        let ffn = self.ffn_dim;
        let n_heads = self.attention_heads.len();
        let _head_dim = d.checked_div(n_heads).unwrap_or(1);

        // ── Residual2: grad flows to both residual1 and ffn_output ──
        // output = residual1 + ffn_output
        // grad_residual1 = grad_output, grad_ffn_output = grad_output

        // ── FFN backward ──
        let mut grad_ffn_hidden = vec![vec![0.0f32; ffn]; seq];
        for (si, go) in grad_output.iter().enumerate() {
            for k in 0..ffn {
                let mut s = 0.0f32;
                for j in 0..d {
                    s += go[j] * self.w_ff2[j * ffn + k];
                }
                grad_ffn_hidden[si][k] = s;
            }
            for j in 0..d {
                for k in 0..ffn {
                    self.grad_w_ff2[j * ffn + k] += cache.ffn_hidden_activated[si][k] * go[j];
                }
            }
        }

        // ReLU^2 backward: derivative = 2 * max(0, x)
        for si in 0..seq {
            for k in 0..ffn {
                let pre_act = left_matvec_single(&self.w_ff1, ffn, d, &cache.normed2[si], k);
                grad_ffn_hidden[si][k] *= 2.0 * pre_act.max(0.0);
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
                    self.grad_w_ff1[j * ffn + k] += cache.normed2[si][j] * grad_ffn_hidden[si][k];
                }
            }
        }

        // ── LayerNorm2 backward ──
        let grad_residual1 = backward_layer_norm_batch(&cache.residual1, &grad_normed2, 1e-5);

        // ── Residual1 backward ──
        // residual1 = input + attn_out  →  grad flows to both
        // grad_attn_out = grad_residual1  (copy)
        let grad_attn_out = grad_residual1.clone();

        // ── Attention backward ──
        // attn_out is concatenation of all head outputs along head_dim
        // grad for each head h, position i: grad_h[i] = grad_attn_out[i][h*head_dim..(h+1)*head_dim]
        let mut grad_normed1 = vec![vec![0.0f32; d]; seq];

        for hi in 0..n_heads {
            let head = &mut self.attention_heads[hi];
            let hc = &cache.head_caches[hi];
            let hd = head.head_dim;
            let scale = (hd as f32).sqrt();

            // Extract grad for this head from concatenated grad_attn_out
            // attn_out[i][hi*hd..(hi+1)*hd] = head_output[i]
            // But we have no w_o projection yet — head output IS the contribution
            // So grad_head_output[i][k] = grad_attn_out[i][hi*hd + k]
            let mut grad_head_out = vec![vec![0.0f32; hd]; seq];
            for i in 0..seq {
                for k in 0..hd {
                    grad_head_out[i][k] = grad_attn_out[i][hi * hd + k];
                }
            }

            // ── Attention weights backward ──
            // head_output[i][k] = sum_j attn_weights[i][j] * vs[j][k]
            // grad_attn_weights[i][j] = sum_k grad_head_out[i][k] * vs[j][k]
            // grad_vs[j][k] += sum_i attn_weights[i][j] * grad_head_out[i][k]  (over all query positions)
            let mut grad_attn_weights = vec![vec![0.0f32; seq]; seq];
            let mut grad_vs = vec![vec![0.0f32; hd]; seq];

            for qi in 0..seq {
                for kj in 0..seq {
                    let mut gaw = 0.0f32;
                    for k in 0..hd {
                        gaw += grad_head_out[qi][k] * hc.vs[kj][k];
                        grad_vs[kj][k] += hc.attn_weights[qi][kj] * grad_head_out[qi][k];
                    }
                    grad_attn_weights[qi][kj] = gaw;
                }
            }

            // ── Softmax backward ──
            // grad_scores[i][j] = attn_weights[i] * (grad_attn_weights[i] - sum_j(attn_weights[i][j] * grad_attn_weights[i][j]))
            let mut grad_scores = vec![vec![0.0f32; seq]; seq];
            for qi in 0..seq {
                let dot: f32 = hc.attn_weights[qi]
                    .iter()
                    .zip(grad_attn_weights[qi].iter())
                    .map(|(&aw, &gaw)| aw * gaw)
                    .sum();
                for kj in 0..seq {
                    grad_scores[qi][kj] =
                        hc.attn_weights[qi][kj] * (grad_attn_weights[qi][kj] - dot);
                }
            }

            // ── Scale backward ──
            // scores = QK^T / sqrt(head_dim)
            // grad_qk_scores /= scale
            for qi in 0..seq {
                for kj in 0..seq {
                    grad_scores[qi][kj] /= scale;
                }
            }

            // ── QK^T backward ──
            // score[qi][kj] = sum_k Q[qi][k] * K[kj][k]
            // grad_Q[qi][k] += sum_j grad_scores[qi][j] * K[j][k]
            // grad_K[kj][k] += sum_q grad_scores[q][kj] * Q[q][k]
            let mut grad_qs = vec![vec![0.0f32; hd]; seq];
            let mut grad_ks = vec![vec![0.0f32; hd]; seq];

            for qi in 0..seq {
                for k in 0..hd {
                    let mut gq = 0.0f32;
                    for kj in 0..seq {
                        gq += grad_scores[qi][kj] * hc.ks[kj][k];
                    }
                    grad_qs[qi][k] = gq;
                }
            }
            for kj in 0..seq {
                for k in 0..hd {
                    let mut gk = 0.0f32;
                    for qi in 0..seq {
                        gk += grad_scores[qi][kj] * hc.qs[qi][k];
                    }
                    grad_ks[kj][k] = gk;
                }
            }

            // ── QK-Norm scale backward ──
            // Q_scaled[qi][k] = Q[qi][k] * q_norm_scale[k]
            // grad_q_norm_scale[k] += sum_qi Q_raw[qi][k] * grad_Q_scaled[qi][k]
            // But we stored qs AFTER scaling. Need to undo:
            // Q_raw[qi][k] = qs[qi][k] / q_norm_scale[k]  (if q_norm_scale != 0)
            // Actually: qs stored = Q_raw * q_norm_scale, so Q_raw = qs / q_norm_scale
            // grad_q_norm[k] += sum_i Q_raw[i][k] * grad_Q_scaled[i][k]
            //                   = sum_i (qs[i][k] / q_norm_scale[k]) * grad_qs[i][k]
            // And grad_Q_raw = grad_Q_scaled * q_norm_scale
            // Similarly for K.

            for k in 0..hd {
                let qns = head.q_norm_scale[k];
                if qns.abs() > 1e-12 {
                    for qi in 0..seq {
                        head.grad_q_norm[k] += (hc.qs[qi][k] / qns) * grad_qs[qi][k];
                    }
                }
                let kns = head.k_norm_scale[k];
                if kns.abs() > 1e-12 {
                    for kj in 0..seq {
                        head.grad_k_norm[k] += (hc.ks[kj][k] / kns) * grad_ks[kj][k];
                    }
                }
            }

            // Propagate through norm scale: grad_Q_raw = grad_Q_scaled * q_norm_scale
            for qi in 0..seq {
                for k in 0..hd {
                    grad_qs[qi][k] *= head.q_norm_scale[k];
                    grad_ks[qi][k] *= head.k_norm_scale[k];
                }
            }

            // ── Linear projection backward ──
            // Q = W_q @ input  (left_matvec: out = W @ in)
            // grad_W_q += outer(grad_Q, input)
            // grad_input += W_q^T @ grad_Q
            let ni = &hc.normed_input;
            for qi in 0..seq {
                for oi in 0..hd {
                    for j in 0..d {
                        head.grad_w_q[oi * d + j] += grad_qs[qi][oi] * ni[qi][j];
                    }
                }
                for j in 0..d {
                    let mut s = 0.0f32;
                    for oi in 0..hd {
                        s += head.w_q[oi * d + j] * grad_qs[qi][oi];
                    }
                    grad_normed1[qi][j] += s;
                }
            }

            for qi in 0..seq {
                for oi in 0..hd {
                    for j in 0..d {
                        head.grad_w_k[oi * d + j] += grad_ks[qi][oi] * ni[qi][j];
                    }
                }
                for j in 0..d {
                    let mut s = 0.0f32;
                    for oi in 0..hd {
                        s += head.w_k[oi * d + j] * grad_ks[qi][oi];
                    }
                    grad_normed1[qi][j] += s;
                }
            }

            for qi in 0..seq {
                for oi in 0..hd {
                    for j in 0..d {
                        head.grad_w_v[oi * d + j] += grad_vs[qi][oi] * ni[qi][j];
                    }
                }
                for j in 0..d {
                    let mut s = 0.0f32;
                    for oi in 0..hd {
                        s += head.w_v[oi * d + j] * grad_vs[qi][oi];
                    }
                    grad_normed1[qi][j] += s;
                }
            }
        }

        // ── LayerNorm1 backward ──
        backward_layer_norm_batch(&cache.input, &grad_normed1, 1e-5)
    }

    fn zero_grad(&mut self) {
        fill_zero(&mut self.grad_w_ff1);
        fill_zero(&mut self.grad_w_ff2);
        fill_zero(&mut self.grad_norm1);
        fill_zero(&mut self.grad_norm2);
        for head in &mut self.attention_heads {
            head.zero_grad();
        }
    }

    fn adamw_update(&mut self, cfg: &AdamWConfig) {
        self.adamw_w_ff1.update(&mut self.w_ff1, &self.grad_w_ff1.clone(), cfg);
        self.adamw_w_ff2.update(&mut self.w_ff2, &self.grad_w_ff2.clone(), cfg);
        self.adamw_norm1.update(&mut self.norm1_scale, &self.grad_norm1.clone(), cfg);
        self.adamw_norm2.update(&mut self.norm2_scale, &self.grad_norm2.clone(), cfg);
        for head in &mut self.attention_heads {
            head.adamw_update(cfg);
        }
    }
}

pub struct Trinity3kConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub n_layers: usize,
    pub max_seq_len: usize,
}

impl Default for Trinity3kConfig {
    fn default() -> Self {
        Self {
            vocab_size: 729,
            hidden_dim: 243,
            n_heads: 27,
            head_dim: 9,
            n_layers: 11,
            max_seq_len: 1024,
        }
    }
}

impl Trinity3kConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_dim != self.n_heads * self.head_dim {
            return Err(format!(
                "hidden_dim ({}) must equal n_heads ({}) * head_dim ({}), got {}",
                self.hidden_dim, self.n_heads, self.head_dim, self.n_heads * self.head_dim
            ));
        }
        Ok(())
    }

    pub fn total_params(&self) -> usize {
        let emb = self.vocab_size * self.hidden_dim;
        let per_layer = 4 * self.head_dim * self.hidden_dim * self.n_heads
            + 2 * self.head_dim * self.n_heads
            + 2 * self.hidden_dim * (self.hidden_dim * 4)
            + 2 * self.hidden_dim;
        emb + per_layer * self.n_layers + self.hidden_dim
    }
}

pub struct Trinity3kModel {
    token_embeddings: Vec<f32>,
    layers: Vec<Trinity3kLayer>,
    final_norm_scale: Vec<f32>,
    grad_token_embeddings: Vec<f32>,
    grad_final_norm: Vec<f32>,
    adamw_emb: AdamWState,
    adamw_final_norm: AdamWState,
    config: Trinity3kConfig,
}

impl Trinity3kModel {
    pub fn new(config: Trinity3kConfig) -> Result<Self, String> {
        config.validate()?;
        let mut seed = 42u64;
        let emb_size = config.vocab_size * config.hidden_dim;

        let token_embeddings = xavier_phi_init(
            emb_size, config.vocab_size, config.hidden_dim, 0, config.n_layers, &mut seed,
        );

        let layers: Vec<Trinity3kLayer> = (0..config.n_layers)
            .map(|i| Trinity3kLayer::new(config.hidden_dim, config.n_heads, i, config.n_layers, &mut seed))
            .collect();

        Ok(Self {
            token_embeddings,
            layers,
            final_norm_scale: vec![1.0; config.hidden_dim],
            grad_token_embeddings: vec![0.0; emb_size],
            grad_final_norm: vec![0.0; config.hidden_dim],
            adamw_emb: AdamWState::new(emb_size),
            adamw_final_norm: AdamWState::new(config.hidden_dim),
            config,
        })
    }

    pub fn forward(&self, input_ids: &[usize]) -> Vec<Vec<f32>> {
        let d = self.config.hidden_dim;

        let mut hidden: Vec<Vec<f32>> = input_ids
            .iter()
            .map(|&tid| self.token_embeddings[tid * d..tid * d + d].to_vec())
            .collect();

        for layer in &self.layers {
            hidden = layer.forward(&hidden);
        }

        let eps = 1e-5;
        let normed: Vec<Vec<f32>> = hidden.iter().map(|x| layer_norm(x, eps)).collect();

        normed
            .iter()
            .map(|x| {
                (0..self.config.vocab_size)
                    .map(|vi| {
                        let emb = &self.token_embeddings[vi * d..vi * d + d];
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

    fn zero_grad(&mut self) {
        fill_zero(&mut self.grad_token_embeddings);
        fill_zero(&mut self.grad_final_norm);
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }

    /// Analytical backward pass: compute all gradients in one forward+backward
    #[allow(clippy::needless_range_loop)]
    fn backward_analytical(&mut self, tokens: &[usize]) {
        if tokens.len() < 2 {
            return;
        }
        let d = self.config.hidden_dim;
        let vs = self.config.vocab_size;
        let input_ids = &tokens[..tokens.len() - 1];
        let targets = &tokens[1..];
        let seq = input_ids.len();

        // ── Forward pass, caching activations ──
        let mut embed_activations: Vec<Vec<f32>> = Vec::with_capacity(seq);
        for &tid in input_ids {
            let tid = tid.min(vs - 1);
            embed_activations.push(self.token_embeddings[tid * d..tid * d + d].to_vec());
        }

        // Run through layers, saving intermediate activations
        let mut layer_cache: Vec<LayerCache> = Vec::with_capacity(self.config.n_layers);
        let mut hidden = embed_activations.clone();
        for layer in &self.layers {
            let (out, cache) = layer.forward_cached(&hidden);
            layer_cache.push(cache);
            hidden = out;
        }

        // Final layer norm
        let eps = 1e-5f32;
        let normed: Vec<Vec<f32>> = hidden.iter().map(|x| layer_norm(x, eps)).collect();

        // Logits via tied embeddings
        let logits: Vec<Vec<f32>> = normed.iter().map(|x| {
            (0..vs).map(|vi| {
                let emb = &self.token_embeddings[vi * d..vi * d + d];
                emb.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum()
            }).collect()
        }).collect();

        // ── Backward pass ──

        // 1. Cross-entropy + softmax gradient: d_logits = softmax(logits) - one_hot(target)
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

        // 2. Through tied LM head: grad_normed[i][j] += sum_v grad_logits[i][v] * emb[v][j]
        //    and grad_emb[v][j] += sum_i grad_logits[i][v] * normed[i][j]
        let mut grad_normed = vec![vec![0.0f32; d]; seq];
        for i in 0..seq {
            for v in 0..vs {
                let gl = grad_logits[i][v];
                if gl.abs() < 1e-10 {
                    continue;
                }
                let emb_start = v * d;
                for j in 0..d {
                    grad_normed[i][j] += gl * self.token_embeddings[emb_start + j];
                    self.grad_token_embeddings[emb_start + j] += gl * normed[i][j];
                }
            }
        }

        // 3. Through final layer norm
        let grad_hidden = backward_layer_norm_batch(&hidden, &grad_normed, eps);

        // 4. Through layers in reverse
        let mut grad_input = grad_hidden;
        for (li, layer) in self.layers.iter_mut().enumerate().rev() {
            let cache = &layer_cache[li];
            grad_input = layer.backward(&grad_input, cache);
        }

        // 5. Through embedding lookup
        for (i, &tid) in input_ids.iter().enumerate() {
            let tid = tid.min(vs - 1);
            let base = tid * d;
            for j in 0..d {
                self.grad_token_embeddings[base + j] += grad_input[i][j];
            }
        }
    }

    pub fn train_step(&mut self, tokens: &[usize], cfg: &AdamWConfig) {
        self.zero_grad();
        self.backward_analytical(tokens);

        self.adamw_emb.update(&mut self.token_embeddings, &self.grad_token_embeddings.clone(), cfg);
        self.adamw_final_norm.update(&mut self.final_norm_scale, &self.grad_final_norm.clone(), cfg);
        for layer in &mut self.layers {
            layer.adamw_update(cfg);
        }
    }

    /// Backward-compatible alias
    pub fn sgd_step(&mut self, tokens: &[usize], _lr: f32) {
        let cfg = AdamWConfig::default();
        self.train_step(tokens, &cfg);
    }

    /// Backward-compatible alias  
    pub fn adamw_step(&mut self, tokens: &[usize], _lr: f32) {
        let cfg = AdamWConfig::default();
        self.train_step(tokens, &cfg);
    }

    pub fn config(&self) -> &Trinity3kConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let c = Trinity3kConfig::default();
        assert!(c.validate().is_ok());
        assert_eq!(c.vocab_size, 729);
        assert_eq!(c.hidden_dim, 243);
        assert_eq!(c.n_heads, 27);
        assert_eq!(c.head_dim, 9);
    }

    #[test]
    fn test_model_creation() {
        let c = Trinity3kConfig::default();
        assert!(Trinity3kModel::new(c).is_ok());
    }

    #[test]
    fn test_forward_pass() {
        let c = Trinity3kConfig::default();
        let m = Trinity3kModel::new(c).unwrap();
        let tokens = vec![1, 2, 3, 4];
        let logits = m.forward(&tokens);
        assert_eq!(logits.len(), 4);
        assert_eq!(logits[0].len(), 729);
    }

    #[test]
    fn test_loss_finite() {
        let c = Trinity3kConfig::default();
        let m = Trinity3kModel::new(c).unwrap();
        let tokens: Vec<usize> = (0..16).map(|i| i % 729).collect();
        let (loss, bpb) = m.loss_bpb(&tokens);
        assert!(loss.is_finite());
        assert!(bpb.is_finite());
        assert!(bpb > 0.0);
    }

    #[test]
    fn test_attention_backward_reduces_loss() {
        let mut c = Trinity3kConfig::default();
        c.n_layers = 1;
        c.vocab_size = 32;
        c.hidden_dim = 27;
        c.n_heads = 3;
        c.head_dim = 9;
        c.max_seq_len = 16;

        let mut model = Trinity3kModel::new(c).unwrap();
        let tokens: Vec<usize> = (0..32).collect();

        let (loss_before, _) = model.loss_bpb(&tokens);

        let cfg = AdamWConfig {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        };

        for _ in 0..5 {
            model.train_step(&tokens, &cfg);
        }

        let (loss_after, _) = model.loss_bpb(&tokens);

        assert!(
            loss_after < loss_before,
            "Loss should decrease after training: before={}, after={}",
            loss_before, loss_after
        );
    }
}
