//! Complete CPU Transformer for FineWeb training (IGLA RACE).
//!
//! Pre-LN architecture with full forward + backward passes.
//! No external dependencies beyond `std`.

// ============================================================================
// RNG & Init
// ============================================================================

fn lcg_next(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 32) as u32) as f32 / (u32::MAX as f32)
}

fn xavier_init(size: usize, fan_in: usize, fan_out: usize, seed: &mut u64) -> Vec<f32> {
    let scale = (6.0f32 / (fan_in + fan_out) as f32).sqrt();
    (0..size)
        .map(|_| {
            let t = lcg_next(seed);
            t * 2.0 * scale - scale
        })
        .collect()
}

// ============================================================================
// Basic ops
// ============================================================================

/// C = A @ B
/// A: (m, k), B: (k, n), C: (m, n)
pub fn matmul(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for l in 0..k {
            let a_il = a[i * k + l];
            for j in 0..n {
                c[i * n + j] += a_il * b[l * n + j];
            }
        }
    }
    c
}

/// C = A^T @ B
/// A: (m, k)  → A^T is (k, m)
/// B: (m, n)
/// C: (k, n)
pub fn matmul_transpose_a(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), m * n);
    let mut c = vec![0.0f32; k * n];
    for l in 0..k {
        for i in 0..m {
            let a_il = a[i * k + l];
            for j in 0..n {
                c[l * n + j] += a_il * b[i * n + j];
            }
        }
    }
    c
}

/// C = A @ B^T
/// A: (m, k)
/// B: (n, k)  → B^T is (k, n)
/// C: (m, n)
pub fn matmul_transpose_b(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[j * k + l];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// In-place softmax over a 1-D slice.
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    } else {
        let inv_len = 1.0 / x.len() as f32;
        for v in x.iter_mut() {
            *v = inv_len;
        }
    }
}

/// Row-wise LayerNorm in-place.
/// `x` is (rows, cols) stored row-major.
pub fn layer_norm_rows(x: &mut [f32], rows: usize, cols: usize, eps: f32) {
    assert_eq!(x.len(), rows * cols);
    for i in 0..rows {
        let row = &mut x[i * cols..(i + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / cols as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for v in row.iter_mut() {
            *v = (*v - mean) * inv_std;
        }
    }
}

/// Row-wise LayerNorm backward.
/// `dout` and `x` are both (rows, cols) row-major.
/// Returns `dx` of the same shape.
pub fn layer_norm_backward_rows(
    dout: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Vec<f32> {
    assert_eq!(dout.len(), rows * cols);
    assert_eq!(x.len(), rows * cols);
    let mut dx = vec![0.0f32; rows * cols];
    for i in 0..rows {
        let x_row = &x[i * cols..(i + 1) * cols];
        let dout_row = &dout[i * cols..(i + 1) * cols];
        let mean = x_row.iter().sum::<f32>() / cols as f32;
        let var = x_row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / cols as f32;
        let std = (var + eps).sqrt();
        let inv_std = 1.0 / std;

        let y: Vec<f32> = x_row.iter().map(|v| (*v - mean) * inv_std).collect();
        let mean_dout = dout_row.iter().sum::<f32>() / cols as f32;
        let mean_dout_y = dout_row.iter().zip(&y).map(|(&d, &yi)| d * yi).sum::<f32>() / cols as f32;

        for j in 0..cols {
            dx[i * cols + j] = (dout_row[j] - mean_dout - y[j] * mean_dout_y) * inv_std;
        }
    }
    dx
}

// ============================================================================
// Linear
// ============================================================================

#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Vec<f32>, // (in_features, out_features)
    pub bias: Vec<f32>,   // (out_features,)
    pub in_features: usize,
    pub out_features: usize,
}

#[derive(Debug, Clone)]
pub struct LinearCache {
    pub input: Vec<f32>, // (batch, in_features)
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, seed: &mut u64) -> Self {
        let weight = xavier_init(in_features * out_features, in_features, out_features, seed);
        let bias = vec![0.0f32; out_features];
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Forward: y = x @ W + b
    pub fn forward(&self, input: &[f32]) -> (Vec<f32>, LinearCache) {
        let batch = input.len() / self.in_features;
        assert_eq!(input.len(), batch * self.in_features);

        let mut output = matmul(input, batch, self.in_features, &self.weight, self.out_features);
        for i in 0..batch {
            for j in 0..self.out_features {
                output[i * self.out_features + j] += self.bias[j];
            }
        }

        let cache = LinearCache {
            input: input.to_vec(),
        };
        (output, cache)
    }

    /// Backward: returns (d_input, d_weight, d_bias)
    pub fn backward(&self, d_output: &[f32], cache: &LinearCache) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let batch = cache.input.len() / self.in_features;
        assert_eq!(d_output.len(), batch * self.out_features);

        let d_input = matmul_transpose_b(d_output, batch, self.out_features, &self.weight, self.in_features);
        let d_weight = matmul_transpose_a(&cache.input, batch, self.in_features, d_output, self.out_features);

        let mut d_bias = vec![0.0f32; self.out_features];
        for i in 0..batch {
            for j in 0..self.out_features {
                d_bias[j] += d_output[i * self.out_features + j];
            }
        }

        (d_input, d_weight, d_bias)
    }
}

// ============================================================================
// CausalSelfAttention
// ============================================================================

#[derive(Debug, Clone)]
pub struct CausalSelfAttention {
    pub w_qkv: Linear,
    pub w_o: Linear,
    pub n_heads: usize,
    pub d_model: usize,
    pub d_head: usize,
}

#[derive(Debug, Clone)]
pub struct AttentionCache {
    pub x: Vec<f32>,           // (seq_len, d_model)
    pub qkv_cache: LinearCache,
    pub q: Vec<f32>,           // (n_heads, seq_len, d_head)
    pub k: Vec<f32>,           // (n_heads, seq_len, d_head)
    pub v: Vec<f32>,           // (n_heads, seq_len, d_head)
    pub attn_weights: Vec<f32>, // (n_heads, seq_len, seq_len)
    pub wo_cache: LinearCache,
}

#[derive(Debug, Clone)]
pub struct AttentionBackwardResult {
    pub d_input: Vec<f32>, // (seq_len, d_model)
    pub d_wqkv: Vec<f32>,  // (d_model, 3*d_model)
    pub d_bqkv: Vec<f32>,  // (3*d_model,)
    pub d_wo: Vec<f32>,    // (d_model, d_model)
    pub d_bo: Vec<f32>,    // (d_model,)
}

impl CausalSelfAttention {
    pub fn new(d_model: usize, n_heads: usize, seed: &mut u64) -> Self {
        assert_eq!(d_model % n_heads, 0);
        let d_head = d_model / n_heads;
        let w_qkv = Linear::new(d_model, 3 * d_model, seed);
        let w_o = Linear::new(d_model, d_model, seed);
        Self {
            w_qkv,
            w_o,
            n_heads,
            d_model,
            d_head,
        }
    }

    pub fn forward(&self, x: &[f32]) -> (Vec<f32>, AttentionCache) {
        let seq_len = x.len() / self.d_model;

        // QKV projection
        let (qkv, qkv_cache) = self.w_qkv.forward(x);

        // Split into q, k, v and reshape to (n_heads, seq_len, d_head)
        let mut q = vec![0.0f32; self.n_heads * seq_len * self.d_head];
        let mut k = vec![0.0f32; self.n_heads * seq_len * self.d_head];
        let mut v = vec![0.0f32; self.n_heads * seq_len * self.d_head];

        for t in 0..seq_len {
            for h in 0..self.n_heads {
                for d in 0..self.d_head {
                    let src_q = t * 3 * self.d_model + h * self.d_head + d;
                    let src_k = t * 3 * self.d_model + self.d_model + h * self.d_head + d;
                    let src_v = t * 3 * self.d_model + 2 * self.d_model + h * self.d_head + d;
                    let dst = h * seq_len * self.d_head + t * self.d_head + d;
                    q[dst] = qkv[src_q];
                    k[dst] = qkv[src_k];
                    v[dst] = qkv[src_v];
                }
            }
        }

        // Attention scores & softmax
        let scale = 1.0 / (self.d_head as f32).sqrt();
        let mut attn_weights = vec![0.0f32; self.n_heads * seq_len * seq_len];

        for h in 0..self.n_heads {
            for i in 0..seq_len {
                let mut scores_row = vec![0.0f32; seq_len];
                for j in 0..seq_len {
                    if j > i {
                        scores_row[j] = f32::NEG_INFINITY;
                    } else {
                        let mut dot = 0.0f32;
                        for d in 0..self.d_head {
                            let q_idx = h * seq_len * self.d_head + i * self.d_head + d;
                            let k_idx = h * seq_len * self.d_head + j * self.d_head + d;
                            dot += q[q_idx] * k[k_idx];
                        }
                        scores_row[j] = dot * scale;
                    }
                }
                softmax_inplace(&mut scores_row);
                for j in 0..seq_len {
                    let idx = h * seq_len * seq_len + i * seq_len + j;
                    attn_weights[idx] = scores_row[j];
                }
            }
        }

        // Weighted sum of values
        let mut pre_proj = vec![0.0f32; seq_len * self.d_model];
        for h in 0..self.n_heads {
            for i in 0..seq_len {
                for d in 0..self.d_head {
                    let mut sum = 0.0f32;
                    for j in 0..=i {
                        let attn_idx = h * seq_len * seq_len + i * seq_len + j;
                        let v_idx = h * seq_len * self.d_head + j * self.d_head + d;
                        sum += attn_weights[attn_idx] * v[v_idx];
                    }
                    let dst = i * self.d_model + h * self.d_head + d;
                    pre_proj[dst] = sum;
                }
            }
        }

        // Output projection
        let (output, wo_cache) = self.w_o.forward(&pre_proj);

        let cache = AttentionCache {
            x: x.to_vec(),
            qkv_cache,
            q,
            k,
            v,
            attn_weights,
            wo_cache,
        };

        (output, cache)
    }

    pub fn backward(&self, d_output: &[f32], cache: &AttentionCache) -> AttentionBackwardResult {
        let seq_len = cache.x.len() / self.d_model;

        // 1. Backprop through output projection
        let (d_pre_proj, d_wo, d_bo) = self.w_o.backward(d_output, &cache.wo_cache);

        // 2. Backprop through reshape / merge of heads
        let mut d_v = vec![0.0f32; self.n_heads * seq_len * self.d_head];
        let mut d_attn_weights = vec![0.0f32; self.n_heads * seq_len * seq_len];

        for h in 0..self.n_heads {
            for i in 0..seq_len {
                for d in 0..self.d_head {
                    let d_pre = d_pre_proj[i * self.d_model + h * self.d_head + d];
                    for j in 0..=i {
                        let attn_idx = h * seq_len * seq_len + i * seq_len + j;
                        let v_idx = h * seq_len * self.d_head + j * self.d_head + d;
                        d_attn_weights[attn_idx] += d_pre * cache.v[v_idx];
                        d_v[v_idx] += d_pre * cache.attn_weights[attn_idx];
                    }
                }
            }
        }

        // 3. Backprop through softmax
        let mut d_scores = vec![0.0f32; self.n_heads * seq_len * seq_len];
        for h in 0..self.n_heads {
            for i in 0..seq_len {
                let mut sum = 0.0f32;
                for j in 0..=i {
                    let idx = h * seq_len * seq_len + i * seq_len + j;
                    sum += d_attn_weights[idx] * cache.attn_weights[idx];
                }
                for j in 0..=i {
                    let idx = h * seq_len * seq_len + i * seq_len + j;
                    d_scores[idx] = cache.attn_weights[idx] * (d_attn_weights[idx] - sum);
                }
            }
        }

        // 4. Backprop through scale and Q @ K^T
        let scale = 1.0 / (self.d_head as f32).sqrt();
        let mut d_q = vec![0.0f32; self.n_heads * seq_len * self.d_head];
        let mut d_k = vec![0.0f32; self.n_heads * seq_len * self.d_head];

        for h in 0..self.n_heads {
            for i in 0..seq_len {
                for j in 0..=i {
                    let d_s = d_scores[h * seq_len * seq_len + i * seq_len + j] * scale;
                    for d in 0..self.d_head {
                        let q_idx = h * seq_len * self.d_head + i * self.d_head + d;
                        let k_idx = h * seq_len * self.d_head + j * self.d_head + d;
                        d_q[q_idx] += d_s * cache.k[k_idx];
                        d_k[k_idx] += d_s * cache.q[q_idx];
                    }
                }
            }
        }

        // 5. Merge d_q, d_k, d_v back into d_qkv
        let mut d_qkv = vec![0.0f32; seq_len * 3 * self.d_model];
        for t in 0..seq_len {
            for h in 0..self.n_heads {
                for d in 0..self.d_head {
                    let dst_q = t * 3 * self.d_model + h * self.d_head + d;
                    let dst_k = t * 3 * self.d_model + self.d_model + h * self.d_head + d;
                    let dst_v = t * 3 * self.d_model + 2 * self.d_model + h * self.d_head + d;
                    let src = h * seq_len * self.d_head + t * self.d_head + d;
                    d_qkv[dst_q] = d_q[src];
                    d_qkv[dst_k] = d_k[src];
                    d_qkv[dst_v] = d_v[src];
                }
            }
        }

        // 6. Backprop through QKV projection
        let (d_input, d_wqkv, d_bqkv) = self.w_qkv.backward(&d_qkv, &cache.qkv_cache);

        AttentionBackwardResult {
            d_input,
            d_wqkv,
            d_bqkv,
            d_wo,
            d_bo,
        }
    }
}

// ============================================================================
// FFN
// ============================================================================

#[derive(Debug, Clone)]
pub struct Ffn {
    pub linear1: Linear,
    pub linear2: Linear,
}

#[derive(Debug, Clone)]
pub struct FfnCache {
    pub linear1_cache: LinearCache,
    pub hidden: Vec<f32>, // post-ReLU
    pub linear2_cache: LinearCache,
}

#[derive(Debug, Clone)]
pub struct FfnBackwardResult {
    pub d_input: Vec<f32>,
    pub d_w1: Vec<f32>,
    pub d_b1: Vec<f32>,
    pub d_w2: Vec<f32>,
    pub d_b2: Vec<f32>,
}

impl Ffn {
    pub fn new(d_model: usize, d_ffn: usize, seed: &mut u64) -> Self {
        Self {
            linear1: Linear::new(d_model, d_ffn, seed),
            linear2: Linear::new(d_ffn, d_model, seed),
        }
    }

    pub fn forward(&self, x: &[f32]) -> (Vec<f32>, FfnCache) {
        let (hidden_pre, l1_cache) = self.linear1.forward(x);
        let mut hidden = hidden_pre;
        for v in hidden.iter_mut() {
            *v = v.max(0.0);
        }
        let (output, l2_cache) = self.linear2.forward(&hidden);

        let cache = FfnCache {
            linear1_cache: l1_cache,
            hidden: hidden.clone(),
            linear2_cache: l2_cache,
        };
        (output, cache)
    }

    pub fn backward(&self, d_output: &[f32], cache: &FfnCache) -> (Vec<f32>, FfnBackwardResult) {
        let (d_hidden_post, d_w2, d_b2) = self.linear2.backward(d_output, &cache.linear2_cache);

        let mut d_hidden_pre = vec![0.0f32; d_hidden_post.len()];
        for i in 0..d_hidden_post.len() {
            d_hidden_pre[i] = if cache.hidden[i] > 0.0 {
                d_hidden_post[i]
            } else {
                0.0
            };
        }

        let (d_input, d_w1, d_b1) = self.linear1.backward(&d_hidden_pre, &cache.linear1_cache);

        let grads = FfnBackwardResult {
            d_input: d_input.clone(),
            d_w1,
            d_b1,
            d_w2,
            d_b2,
        };
        (d_input, grads)
    }
}

// ============================================================================
// TransformerBlock
// ============================================================================

#[derive(Debug, Clone)]
pub struct TransformerBlock {
    pub ln1_eps: f32,
    pub ln2_eps: f32,
    pub attn: CausalSelfAttention,
    pub ffn: Ffn,
    pub d_model: usize,
}

#[derive(Debug, Clone)]
pub struct TransformerBlockCache {
    pub ln1_input: Vec<f32>,
    pub ln1_output: Vec<f32>,
    pub attn_cache: AttentionCache,
    pub attn_output: Vec<f32>,
    pub ln2_input: Vec<f32>,
    pub ln2_output: Vec<f32>,
    pub ffn_cache: FfnCache,
}

#[derive(Debug, Clone)]
pub struct BlockGradients {
    pub attn: AttentionBackwardResult,
    pub ffn: FfnBackwardResult,
}

impl TransformerBlock {
    pub fn new(d_model: usize, d_ffn: usize, n_heads: usize, seed: &mut u64) -> Self {
        Self {
            ln1_eps: 1e-5,
            ln2_eps: 1e-5,
            attn: CausalSelfAttention::new(d_model, n_heads, seed),
            ffn: Ffn::new(d_model, d_ffn, seed),
            d_model,
        }
    }

    /// Pre-LN forward:
    ///   x = x + attn(layer_norm1(x))
    ///   x = x + ffn(layer_norm2(x))
    pub fn forward(&self, x: &[f32]) -> (Vec<f32>, TransformerBlockCache) {
        let seq_len = x.len() / self.d_model;

        // LN1
        let mut ln1_out = x.to_vec();
        layer_norm_rows(&mut ln1_out, seq_len, self.d_model, self.ln1_eps);

        // Attention + residual
        let (attn_out, attn_cache) = self.attn.forward(&ln1_out);
        let mut attn_output = vec![0.0f32; x.len()];
        for i in 0..x.len() {
            attn_output[i] = x[i] + attn_out[i];
        }

        // LN2
        let mut ln2_out = attn_output.clone();
        layer_norm_rows(&mut ln2_out, seq_len, self.d_model, self.ln2_eps);

        // FFN + residual
        let (ffn_out, ffn_cache) = self.ffn.forward(&ln2_out);
        let mut output = vec![0.0f32; x.len()];
        for i in 0..output.len() {
            output[i] = attn_output[i] + ffn_out[i];
        }

        let cache = TransformerBlockCache {
            ln1_input: x.to_vec(),
            ln1_output: ln1_out,
            attn_cache,
            attn_output: attn_output.clone(),
            ln2_input: attn_output,
            ln2_output: ln2_out,
            ffn_cache,
        };

        (output, cache)
    }

    pub fn backward(&self, d_output: &[f32], cache: &TransformerBlockCache) -> (Vec<f32>, BlockGradients) {
        let seq_len = d_output.len() / self.d_model;

        // output = attn_output + ffn_out
        let mut d_attn_output = d_output.to_vec();
        let d_ffn_out = d_output.to_vec();

        // Backprop FFN
        let (d_ln2_out, d_ffn) = self.ffn.backward(&d_ffn_out, &cache.ffn_cache);

        // Backprop LN2
        let d_ln2 = layer_norm_backward_rows(&d_ln2_out, &cache.ln2_input, seq_len, self.d_model, self.ln2_eps);
        for i in 0..d_attn_output.len() {
            d_attn_output[i] += d_ln2[i];
        }

        // attn_output = x + attn_out
        let mut d_x = d_attn_output.clone();
        let d_attn_out = d_attn_output;

        // Backprop Attention
        let d_attn = self.attn.backward(&d_attn_out, &cache.attn_cache);

        // Backprop LN1
        let d_ln1 = layer_norm_backward_rows(&d_attn.d_input, &cache.ln1_input, seq_len, self.d_model, self.ln1_eps);
        for i in 0..d_x.len() {
            d_x[i] += d_ln1[i];
        }

        let grads = BlockGradients {
            attn: d_attn,
            ffn: d_ffn,
        };

        (d_x, grads)
    }
}

// ============================================================================
// TransformerModel
// ============================================================================

#[derive(Debug, Clone)]
pub struct TransformerModel {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub max_seq_len: usize,
    pub token_embedding: Vec<f32>, // (vocab_size, d_model)
    pub pos_embedding: Vec<f32>,   // (max_seq_len, d_model)
    pub blocks: Vec<TransformerBlock>,
    pub final_ln_eps: f32,
    pub lm_head: Linear,
}

#[derive(Debug, Clone)]
pub struct TransformerCache {
    pub tokens: Vec<usize>,
    pub embeddings: Vec<f32>,          // (seq_len, d_model)
    pub block_caches: Vec<TransformerBlockCache>,
    pub final_ln_input: Vec<f32>,
    pub final_ln_output: Vec<f32>,
    pub lm_head_cache: LinearCache,
}

#[derive(Debug, Clone)]
pub struct ModelGradients {
    pub d_token_embedding: Vec<f32>,
    pub d_pos_embedding: Vec<f32>,
    pub block_grads: Vec<BlockGradients>,
    pub d_lm_head_w: Vec<f32>,
    pub d_lm_head_b: Vec<f32>,
}

impl TransformerModel {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        d_ffn: usize,
        n_heads: usize,
        n_layers: usize,
        max_seq_len: usize,
        seed: u64,
    ) -> Self {
        let mut rng = seed;
        let token_embedding = xavier_init(vocab_size * d_model, vocab_size, d_model, &mut rng);
        let pos_embedding = xavier_init(max_seq_len * d_model, max_seq_len, d_model, &mut rng);
        let blocks: Vec<_> = (0..n_layers)
            .map(|_| TransformerBlock::new(d_model, d_ffn, n_heads, &mut rng))
            .collect();
        let lm_head = Linear::new(d_model, vocab_size, &mut rng);
        Self {
            vocab_size,
            d_model,
            n_layers,
            max_seq_len,
            token_embedding,
            pos_embedding,
            blocks,
            final_ln_eps: 1e-5,
            lm_head,
        }
    }

    pub fn forward(&self, tokens: &[usize]) -> (Vec<Vec<f32>>, TransformerCache) {
        let seq_len = tokens.len();
        assert!(
            seq_len <= self.max_seq_len,
            "sequence length {} exceeds max_seq_len {}",
            seq_len,
            self.max_seq_len
        );

        // Token + positional embeddings
        let mut embeddings = vec![0.0f32; seq_len * self.d_model];
        for (pos, &tok) in tokens.iter().enumerate() {
            let tok_src = tok * self.d_model;
            let pos_src = pos * self.d_model;
            let dst = pos * self.d_model;
            for d in 0..self.d_model {
                embeddings[dst + d] = self.token_embedding[tok_src + d] + self.pos_embedding[pos_src + d];
            }
        }

        let mut x = embeddings.clone();
        let mut block_caches = Vec::with_capacity(self.n_layers);
        for block in &self.blocks {
            let (out, cache) = block.forward(&x);
            block_caches.push(cache);
            x = out;
        }

        // Final layer norm
        let final_ln_input = x.clone();
        let mut final_ln_output = x;
        layer_norm_rows(&mut final_ln_output, seq_len, self.d_model, self.final_ln_eps);

        // LM head
        let (logits_flat, lm_head_cache) = self.lm_head.forward(&final_ln_output);

        // Reshape to Vec<Vec<f32>>
        let mut logits = vec![vec![0.0f32; self.vocab_size]; seq_len];
        for i in 0..seq_len {
            logits[i].copy_from_slice(&logits_flat[i * self.vocab_size..(i + 1) * self.vocab_size]);
        }

        let cache = TransformerCache {
            tokens: tokens.to_vec(),
            embeddings,
            block_caches,
            final_ln_input,
            final_ln_output: final_ln_output.clone(),
            lm_head_cache,
        };

        (logits, cache)
    }

    pub fn backward(&self, d_logits: &[Vec<f32>], cache: &TransformerCache) -> ModelGradients {
        let seq_len = cache.tokens.len();

        // Flatten d_logits
        let mut d_logits_flat = vec![0.0f32; seq_len * self.vocab_size];
        for i in 0..seq_len {
            d_logits_flat[i * self.vocab_size..(i + 1) * self.vocab_size].copy_from_slice(&d_logits[i]);
        }

        // Backprop LM head
        let (d_final_ln, d_lm_head_w, d_lm_head_b) = self.lm_head.backward(&d_logits_flat, &cache.lm_head_cache);

        // Backprop final LN
        let mut d_x = layer_norm_backward_rows(
            &d_final_ln,
            &cache.final_ln_input,
            seq_len,
            self.d_model,
            self.final_ln_eps,
        );

        // Backprop blocks in reverse
        let mut block_grads = Vec::with_capacity(self.n_layers);
        for i in (0..self.n_layers).rev() {
            let (d_x_new, grads) = self.blocks[i].backward(&d_x, &cache.block_caches[i]);
            block_grads.push(grads);
            d_x = d_x_new;
        }
        block_grads.reverse();

        // Backprop embeddings -> token + pos embedding grads
        let mut d_token_embedding = vec![0.0f32; self.vocab_size * self.d_model];
        let mut d_pos_embedding = vec![0.0f32; self.max_seq_len * self.d_model];
        for (pos, &tok) in cache.tokens.iter().enumerate() {
            let src = pos * self.d_model;
            let tok_dst = tok * self.d_model;
            let pos_dst = pos * self.d_model;
            for d in 0..self.d_model {
                let grad = d_x[src + d];
                d_token_embedding[tok_dst + d] += grad;
                d_pos_embedding[pos_dst + d] += grad;
            }
        }

        ModelGradients {
            d_token_embedding,
            d_pos_embedding,
            block_grads,
            d_lm_head_w,
            d_lm_head_b,
        }
    }
}

// ============================================================================
// Loss helper
// ============================================================================

/// Cross-entropy loss over a sequence of logits.
///
/// Returns `(loss, d_logits)` where `d_logits` has the same shape as `logits`.
pub fn cross_entropy_loss(logits: &[Vec<f32>], targets: &[usize]) -> (f32, Vec<Vec<f32>>) {
    assert_eq!(logits.len(), targets.len());
    let mut loss = 0.0f32;
    let mut d_logits = Vec::with_capacity(logits.len());

    for (logits_row, &target) in logits.iter().zip(targets.iter()) {
        let mut probs = logits_row.to_vec();
        softmax_inplace(&mut probs);
        loss += -probs[target].ln();

        let mut d_row = probs;
        d_row[target] -= 1.0;
        d_logits.push(d_row);
    }

    (loss, d_logits)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // (2,2)
        let b = vec![1.0f32, 0.0, 0.0, 1.0]; // (2,2)
        let c = matmul(&a, 2, 2, &b, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_transpose_a() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // (2,3)
        let b = vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0]; // (2,3) - but wait, b should be (2, something)
        // Let's use b as (2,2) identity
        let b = vec![1.0f32, 0.0, 0.0, 1.0]; // (2,2)
        let c = matmul_transpose_a(&a, 2, 3, &b, 2); // A^T(3,2) @ B(2,2) = C(3,2)
        // A = [[1,2,3],[4,5,6]]
        // A^T = [[1,4],[2,5],[3,6]]
        // A^T @ I = A^T
        assert_eq!(c, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_matmul_transpose_b() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]; // (2,3)
        let b = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]; // (2,3)
        let c = matmul_transpose_b(&a, 2, 3, &b, 2); // A(2,3) @ B^T(3,2) = C(2,2)
        assert_eq!(c, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_softmax_inplace() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn test_layer_norm_rows() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // (2,3)
        layer_norm_rows(&mut x, 2, 3, 1e-5);
        let mean0 = (x[0] + x[1] + x[2]) / 3.0;
        let mean1 = (x[3] + x[4] + x[5]) / 3.0;
        assert!(mean0.abs() < 1e-5);
        assert!(mean1.abs() < 1e-5);
    }

    #[test]
    fn test_linear_forward_backward() {
        let mut seed = 42u64;
        let linear = Linear::new(3, 2, &mut seed);
        let input = vec![1.0f32, 2.0, 3.0];
        let (output, cache) = linear.forward(&input);
        assert_eq!(output.len(), 2);

        let d_output = vec![0.5f32, -0.5];
        let (d_input, d_weight, d_bias) = linear.backward(&d_output, &cache);
        assert_eq!(d_input.len(), 3);
        assert_eq!(d_weight.len(), 3 * 2);
        assert_eq!(d_bias.len(), 2);
    }

    #[test]
    fn test_attention_forward_backward() {
        let mut seed = 42u64;
        let attn = CausalSelfAttention::new(64, 4, &mut seed);
        let seq_len = 4;
        let x = vec![0.1f32; seq_len * 64];
        let (output, cache) = attn.forward(&x);
        assert_eq!(output.len(), seq_len * 64);

        let d_output = vec![0.01f32; output.len()];
        let grads = attn.backward(&d_output, &cache);
        assert_eq!(grads.d_input.len(), x.len());
        assert_eq!(grads.d_wqkv.len(), 64 * 192);
        assert_eq!(grads.d_wo.len(), 64 * 64);
    }

    #[test]
    fn test_ffn_forward_backward() {
        let mut seed = 42u64;
        let ffn = Ffn::new(64, 256, &mut seed);
        let x = vec![0.1f32; 64];
        let (output, cache) = ffn.forward(&x);
        assert_eq!(output.len(), 64);

        let d_output = vec![0.01f32; 64];
        let (d_input, grads) = ffn.backward(&d_output, &cache);
        assert_eq!(d_input.len(), 64);
        assert_eq!(grads.d_w1.len(), 64 * 256);
        assert_eq!(grads.d_w2.len(), 256 * 64);
    }

    #[test]
    fn test_transformer_block_forward_backward() {
        let mut seed = 42u64;
        let block = TransformerBlock::new(64, 256, 4, &mut seed);
        let seq_len = 4;
        let x = vec![0.1f32; seq_len * 64];
        let (output, cache) = block.forward(&x);
        assert_eq!(output.len(), x.len());

        let d_output = vec![0.01f32; output.len()];
        let (d_input, grads) = block.backward(&d_output, &cache);
        assert_eq!(d_input.len(), x.len());
        assert_eq!(grads.attn.d_wqkv.len(), 64 * 192);
        assert_eq!(grads.ffn.d_w1.len(), 64 * 256);
    }

    #[test]
    fn test_transformer_model_forward_backward() {
        let model = TransformerModel::new(100, 64, 256, 4, 2, 16, 42);
        let tokens = vec![1usize, 2, 3, 4];
        let (logits, cache) = model.forward(&tokens);
        assert_eq!(logits.len(), 4);
        assert_eq!(logits[0].len(), 100);

        let targets = vec![2usize, 3, 4, 5];
        let (loss, d_logits) = cross_entropy_loss(&logits, &targets);
        assert!(loss.is_finite());

        let grads = model.backward(&d_logits, &cache);
        assert_eq!(grads.d_token_embedding.len(), 100 * 64);
        assert_eq!(grads.block_grads.len(), 2);
        assert_eq!(grads.d_lm_head_w.len(), 64 * 100);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let logits = vec![
            vec![1.0f32, 2.0, 3.0],
            vec![1.0f32, 2.0, 3.0],
        ];
        let targets = vec![2usize, 0];
        let (loss, d_logits) = cross_entropy_loss(&logits, &targets);
        assert!(loss.is_finite());
        assert_eq!(d_logits.len(), 2);
        assert_eq!(d_logits[0].len(), 3);

        // For the first row, target=2 => d_logits[2] = prob_2 - 1.0 < 0
        assert!(d_logits[0][2] < 0.0);
    }
}
