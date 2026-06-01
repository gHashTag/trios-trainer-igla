//! Loop 16 PP + Loop 19 YY: 2-layer BitLinear FFN forward + backward with RMSNorm.
//!
//! Architecture: y = BitLinear2( ReLU( BitLinear1( emb[token] ) ) )
//!   where BitLinear(x) = (Quant_w(W) · RMSNorm(x)) — LN INSIDE each BitLinear per
//!   BitNet b1.58 §2 (arXiv:2402.17764) and BitNet Reloaded (arXiv:2407.09527).
//!
//! Total params: emb (V×D) + W1 (D×H) + W2 (H×V); RMSNorm uses γ=1 (no learnable scale at
//! sandbox-scale to keep parameter count grep-able).

const RMS_EPS: f32 = 1e-5;

/// RMSNorm forward: y_i = x_i / sqrt(mean(x²) + eps).
/// Returns (normalized output, rms scale r for backward use).
pub fn rms_norm(x: &[f32]) -> (Vec<f32>, f32) {
    let n = x.len().max(1) as f32;
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
    let r = (mean_sq + RMS_EPS).sqrt();
    let inv_r = 1.0 / r;
    let out: Vec<f32> = x.iter().map(|v| v * inv_r).collect();
    (out, r)
}

/// RMSNorm backward.
/// Given d_y (incoming gradient) and the forward (x, r), compute d_x:
///   d_x_i = d_y_i / r − x_i / (r³ · n) · Σ_j (d_y_j · x_j)
pub fn rms_norm_backward(x: &[f32], d_y: &[f32], r: f32) -> Vec<f32> {
    let n = x.len().max(1) as f32;
    let r3n = r * r * r * n;
    let dot: f32 = d_y.iter().zip(x.iter()).map(|(g, v)| g * v).sum();
    d_y.iter()
        .zip(x.iter())
        .map(|(g, v)| g / r - v / r3n * dot)
        .collect()
}

/// Dropout probability for FFN hidden activations (Loop 21 EEE, Pereyra ICLR 2017 recipe complete).
/// p=0.1 per Vaswani 2017 + nanoGPT convention. Inverted-dropout: forward h' = mask/(1-p) · h.
pub const DROPOUT_P: f32 = 0.1;

/// FFN forward state: caches all intermediate tensors needed for backward through RMSNorm + ReLU.
pub struct FfnCache {
    pub hidden_pre_relu: Vec<f32>,   // [seq_len × d_hidden] BEFORE ReLU
    pub hidden_post_relu: Vec<f32>,  // [seq_len × d_hidden] AFTER ReLU + Dropout (RMSNorm input for W2)
    pub norm_emb: Vec<f32>,          // [seq_len × d_model]  RMSNorm output of emb[tok]
    pub norm_emb_r: Vec<f32>,        // [seq_len] RMSNorm scale for emb per position
    pub norm_hidden: Vec<f32>,       // [seq_len × d_hidden] RMSNorm output of hidden
    pub norm_hidden_r: Vec<f32>,     // [seq_len] RMSNorm scale for hidden per position
    /// Loop 21 EEE: scaled dropout mask (mask/(1-p)). Empty if dropout disabled.
    pub dropout_scale: Vec<f32>,     // [seq_len × d_hidden] same shape as hidden_post_relu
}

/// Generate a deterministic dropout mask using a u64 LCG state.
/// Returns the SCALED mask: 0 (dropped) or 1/(1-p) (survivor).
fn make_dropout_mask(n: usize, p: f32, rng_state: &mut u64) -> Vec<f32> {
    let inv_keep = 1.0 / (1.0 - p);
    (0..n)
        .map(|_| {
            *rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((*rng_state >> 33) as f32) / (u32::MAX as f32);
            if u < p { 0.0 } else { inv_keep }
        })
        .collect()
}

/// Forward through 2-layer BitLinear FFN with RMSNorm per BitNet b1.58 §2.
/// Pipeline: emb[tok] → RMSNorm → W1 → ReLU → RMSNorm → W2 → logits.
pub fn forward_ffn(
    embeddings: &[f32],
    w1: &[f32],
    w2: &[f32],
    input: &[f32],
    vocab_size: usize,
    d_model: usize,
    d_hidden: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (logits, cache) = forward_ffn_with_cache(
        embeddings, w1, w2, input, vocab_size, d_model, d_hidden,
    );
    // Legacy signature: return logits + post-ReLU hidden (used by old backward path).
    (logits, cache.hidden_post_relu)
}

/// Forward with full cache + dropout + apply_rmsnorm bypass (loop 19 + 21 + 25 PPP).
/// If `dropout_rng` is Some, applies inverted dropout after ReLU (training mode).
/// If `apply_rmsnorm` is false, skips RMSNorm inside each BitLinear (sentinel: norm_*_r = 0).
pub fn forward_ffn_with_cache_dropout(
    embeddings: &[f32],
    w1: &[f32],
    w2: &[f32],
    input: &[f32],
    vocab_size: usize,
    d_model: usize,
    d_hidden: usize,
    dropout_rng: Option<&mut u64>,
) -> (Vec<f32>, FfnCache) {
    forward_ffn_with_options(
        embeddings, w1, w2, input, vocab_size, d_model, d_hidden, dropout_rng, true,
    )
}

/// Internal forward with full ablation options. apply_rmsnorm=false bypasses LN.
pub fn forward_ffn_with_options(
    embeddings: &[f32],
    w1: &[f32],
    w2: &[f32],
    input: &[f32],
    vocab_size: usize,
    d_model: usize,
    d_hidden: usize,
    dropout_rng: Option<&mut u64>,
    apply_rmsnorm: bool,
) -> (Vec<f32>, FfnCache) {
    let seq_len = input.len();
    let vocab_avail = embeddings.len() / d_model.max(1);

    let mut logits = vec![0.0f32; seq_len * vocab_size];
    let mut hidden_pre = vec![0.0f32; seq_len * d_hidden];
    let mut hidden_post = vec![0.0f32; seq_len * d_hidden];
    let mut norm_emb = vec![0.0f32; seq_len * d_model];
    let mut norm_emb_r = vec![0.0f32; seq_len];
    let mut norm_hidden = vec![0.0f32; seq_len * d_hidden];
    let mut norm_hidden_r = vec![0.0f32; seq_len];

    // Loop 21 EEE: generate dropout mask if training mode.
    let dropout_scale: Vec<f32> = if let Some(rng) = dropout_rng {
        make_dropout_mask(seq_len * d_hidden, DROPOUT_P, rng)
    } else {
        // Eval mode: no scaling (identity = 1.0 everywhere).
        vec![1.0_f32; seq_len * d_hidden]
    };

    for (i, &token) in input.iter().enumerate() {
        let tok_idx = (token.abs() as usize) % vocab_avail.max(1);
        let emb_offset = tok_idx * d_model;

        // RMSNorm(emb[tok]) — input LN per BitNet b1.58 §2 BitLinear spec.
        let emb_slice: Vec<f32> = (0..d_model)
            .map(|d| {
                if emb_offset + d < embeddings.len() {
                    embeddings[emb_offset + d]
                } else {
                    0.0
                }
            })
            .collect();
        // Loop 25 PPP: branch-in-forward, no clone on bypass path. Sentinel r=0 for backward.
        let (xn, r1) = if apply_rmsnorm {
            rms_norm(&emb_slice)
        } else {
            (emb_slice.clone(), 0.0)
        };
        for d in 0..d_model {
            norm_emb[i * d_model + d] = xn[d];
        }
        norm_emb_r[i] = r1;

        // BitLinear 1: W1 · RMSNorm(emb), then ReLU, then dropout (training only).
        for h in 0..d_hidden {
            let mut acc = 0.0f32;
            for d in 0..d_model {
                acc += w1[h * d_model + d] * xn[d];
            }
            hidden_pre[i * d_hidden + h] = acc;
            let relu = acc.max(0.0);
            hidden_post[i * d_hidden + h] = relu * dropout_scale[i * d_hidden + h];
        }

        // RMSNorm(hidden_post) — per BitLinear spec.
        let hidden_slice: Vec<f32> = (0..d_hidden)
            .map(|h| hidden_post[i * d_hidden + h])
            .collect();
        let (hn, r2) = if apply_rmsnorm {
            rms_norm(&hidden_slice)
        } else {
            (hidden_slice.clone(), 0.0)
        };
        for h in 0..d_hidden {
            norm_hidden[i * d_hidden + h] = hn[h];
        }
        norm_hidden_r[i] = r2;

        // BitLinear 2: W2 · RMSNorm(hidden)
        for v in 0..vocab_size {
            let mut acc = 0.0f32;
            for h in 0..d_hidden {
                acc += w2[v * d_hidden + h] * hn[h];
            }
            logits[i * vocab_size + v] = acc;
        }
    }

    let cache = FfnCache {
        hidden_pre_relu: hidden_pre,
        hidden_post_relu: hidden_post,
        norm_emb,
        norm_emb_r,
        norm_hidden,
        norm_hidden_r,
        dropout_scale,
    };
    (logits, cache)
}

/// Eval forward (no dropout). Convenience wrapper used by callers expecting old signature.
pub fn forward_ffn_with_cache(
    embeddings: &[f32],
    w1: &[f32],
    w2: &[f32],
    input: &[f32],
    vocab_size: usize,
    d_model: usize,
    d_hidden: usize,
) -> (Vec<f32>, FfnCache) {
    forward_ffn_with_cache_dropout(
        embeddings, w1, w2, input, vocab_size, d_model, d_hidden, None,
    )
}

/// Backward with RMSNorm-aware gradient flow (loop 19 YY + critical fix).
/// Pipeline: CE → softmax → W2^T → RMSNorm⁻¹ → ReLU mask → W1^T → RMSNorm⁻¹ → emb.
/// Scale = 1 / (number of unmasked positions), NOT 1/seq_len (loop 19 fix for last-only loss).
pub fn backward_ffn(
    embeddings: &[f32],
    w1: &[f32],
    w2: &[f32],
    logits: &[f32],
    hidden: &[f32], // post-ReLU (legacy back-compat with loop 16 callers)
    input: &[f32],
    targets: &[usize],
    vocab_size: usize,
    d_model: usize,
    d_hidden: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // For RMSNorm-aware backward we need the FULL cache. Re-run forward to populate it.
    // (Legacy signature passes only post-ReLU hidden; we recompute the cache here.)
    let (_, cache) = forward_ffn_with_cache(
        embeddings, w1, w2, input, vocab_size, d_model, d_hidden,
    );
    let _ = hidden; // legacy parameter, superseded by cache.hidden_post_relu
    backward_ffn_with_cache(
        embeddings, w1, w2, logits, &cache, input, targets,
        vocab_size, d_model, d_hidden,
    )
}

/// Single shared label-smoothing strength (Loop 22 critical fix).
/// Used by BOTH forward (last_position_ce_loss) and backward (backward_ffn_with_cache)
/// to guarantee gradient ↔ loss consistency.
/// Default: 0.1 per Pereyra ICLR 2017. Override via MultiSeedConfig.label_smoothing.
pub const LABEL_SMOOTHING_EPS: f32 = 0.1;

pub fn backward_ffn_with_cache(
    embeddings: &[f32],
    w1: &[f32],
    w2: &[f32],
    logits: &[f32],
    cache: &FfnCache,
    input: &[f32],
    targets: &[usize],
    vocab_size: usize,
    d_model: usize,
    d_hidden: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let seq_len = input.len();
    let mut d_emb = vec![0.0f32; embeddings.len()];
    let mut d_w1 = vec![0.0f32; w1.len()];
    let mut d_w2 = vec![0.0f32; w2.len()];
    let vocab_avail = embeddings.len() / d_model.max(1);

    // Loop 19 fix: scale by valid (unmasked) positions, not seq_len.
    let n_valid = targets.iter().filter(|&&t| t < vocab_size).count().max(1) as f32;
    let scale = 1.0 / n_valid;

    for i in 0..seq_len {
        if targets[i] >= vocab_size {
            continue;
        }
        // Softmax grad
        let offset = i * vocab_size;
        let max_logit = logits[offset..offset + vocab_size]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum_exp = 0.0f32;
        let mut exp_logits = vec![0.0f32; vocab_size];
        for v in 0..vocab_size {
            exp_logits[v] = (logits[offset + v] - max_logit).exp();
            sum_exp += exp_logits[v];
        }
        // Loop 21 critical fix: label-smoothed gradient consistent with forward loss.
        // Soft target q_v = (1-ε)·1[v=t] + ε/V; gradient = (softmax_v - q_v).
        let eps = LABEL_SMOOTHING_EPS;
        let v_f = vocab_size as f32;
        let eps_uniform = eps / v_f;
        let mut grad_logits = vec![0.0f32; vocab_size];
        for v in 0..vocab_size {
            let softmax_v = exp_logits[v] / sum_exp.max(1e-30);
            grad_logits[v] = (softmax_v - eps_uniform) * scale;
        }
        grad_logits[targets[i]] -= (1.0 - eps) * scale;

        // BitLinear 2: dW2 = RMSNorm(hidden) ⊗ grad_logits
        //              d_norm_hidden = W2^T · grad_logits
        let mut d_norm_hidden = vec![0.0f32; d_hidden];
        for v in 0..vocab_size {
            let g_v = grad_logits[v];
            for h in 0..d_hidden {
                d_w2[v * d_hidden + h] += cache.norm_hidden[i * d_hidden + h] * g_v;
                d_norm_hidden[h] += w2[v * d_hidden + h] * g_v;
            }
        }

        // RMSNorm backward (for hidden_post_relu → norm_hidden)
        let hidden_slice: Vec<f32> = (0..d_hidden)
            .map(|h| cache.hidden_post_relu[i * d_hidden + h])
            .collect();
        // Loop 25 PPP: r=0 sentinel signals RMSNorm bypass — d_hidden_post = d_norm_hidden.
        let d_hidden_post = if cache.norm_hidden_r[i] == 0.0 {
            d_norm_hidden.clone()
        } else {
            rms_norm_backward(&hidden_slice, &d_norm_hidden, cache.norm_hidden_r[i])
        };

        // Dropout backward: d_h_post_pre_dropout = dropout_scale ⊙ d_h_post.
        // Then ReLU mask backward.
        let mut d_hidden_pre = vec![0.0f32; d_hidden];
        for h in 0..d_hidden {
            let d_relu = d_hidden_post[h] * cache.dropout_scale[i * d_hidden + h];
            if cache.hidden_pre_relu[i * d_hidden + h] > 0.0 {
                d_hidden_pre[h] = d_relu;
            }
        }

        // BitLinear 1: dW1 = RMSNorm(emb) ⊗ d_hidden_pre
        //              d_norm_emb = W1^T · d_hidden_pre
        let mut d_norm_emb = vec![0.0f32; d_model];
        for h in 0..d_hidden {
            let g_h = d_hidden_pre[h];
            if g_h == 0.0 {
                continue;
            }
            for d in 0..d_model {
                d_w1[h * d_model + d] += cache.norm_emb[i * d_model + d] * g_h;
                d_norm_emb[d] += w1[h * d_model + d] * g_h;
            }
        }

        // RMSNorm backward (emb[tok] → norm_emb)
        let token = input[i];
        let tok_idx = (token.abs() as usize) % vocab_avail.max(1);
        let emb_offset = tok_idx * d_model;
        let emb_slice: Vec<f32> = (0..d_model)
            .map(|d| {
                if emb_offset + d < embeddings.len() {
                    embeddings[emb_offset + d]
                } else {
                    0.0
                }
            })
            .collect();
        let d_emb_tok = if cache.norm_emb_r[i] == 0.0 {
            d_norm_emb.clone()
        } else {
            rms_norm_backward(&emb_slice, &d_norm_emb, cache.norm_emb_r[i])
        };
        for d in 0..d_model {
            if emb_offset + d < embeddings.len() {
                d_emb[emb_offset + d] += d_emb_tok[d];
            }
        }
    }

    (d_emb, d_w1, d_w2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Numerical gradient check (Loop 20): verify rms_norm_backward via finite differences.
    /// Mirrors torch.autograd.gradcheck convention (Paszke 2019).
    #[test]
    fn rms_norm_backward_matches_finite_difference() {
        let x = vec![0.3_f32, -0.5, 0.8, 1.2, -0.7, 0.1, -0.2, 0.9];
        let (y, r) = rms_norm(&x);
        // Loss = Σ y_i² / 2  →  dL/dy = y
        let dy: Vec<f32> = y.clone();
        let dx_analytical = rms_norm_backward(&x, &dy, r);

        let eps = 1e-3_f32;
        for i in 0..x.len() {
            let mut x_plus = x.clone();
            x_plus[i] += eps;
            let (y_plus, _) = rms_norm(&x_plus);
            let l_plus: f32 = y_plus.iter().map(|v| 0.5 * v * v).sum();

            let mut x_minus = x.clone();
            x_minus[i] -= eps;
            let (y_minus, _) = rms_norm(&x_minus);
            let l_minus: f32 = y_minus.iter().map(|v| 0.5 * v * v).sum();

            let dx_numerical = (l_plus - l_minus) / (2.0 * eps);
            let err = (dx_analytical[i] - dx_numerical).abs();
            assert!(
                err < 0.01,
                "rms_norm gradient mismatch at i={}: analytical={:.4}, numerical={:.4}, err={:.4}",
                i, dx_analytical[i], dx_numerical, err
            );
        }
    }

    #[test]
    fn apply_rmsnorm_false_produces_different_logits() {
        // Loop 26: verify the apply_rmsnorm bypass actually changes forward output.
        let v = 4;
        let d = 3;
        let h = 4;
        let emb: Vec<f32> = (0..v * d).map(|i| 0.3 + 0.07 * i as f32).collect();
        let w1: Vec<f32> = (0..h * d).map(|i| -0.2 + 0.13 * i as f32).collect();
        let w2: Vec<f32> = (0..v * h).map(|i| 0.05 - 0.09 * i as f32).collect();
        let input = vec![1.0_f32, 2.0];

        let (logits_with_norm, _) = forward_ffn_with_options(
            &emb, &w1, &w2, &input, v, d, h, None, true,
        );
        let (logits_without_norm, _) = forward_ffn_with_options(
            &emb, &w1, &w2, &input, v, d, h, None, false,
        );
        // RMSNorm renormalizes embeddings before W1 → different downstream logits
        let total_diff: f32 = logits_with_norm
            .iter()
            .zip(logits_without_norm.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            total_diff > 0.01,
            "apply_rmsnorm=false should yield measurably different logits; diff={}",
            total_diff
        );
    }

    #[test]
    fn apply_rmsnorm_false_zeroes_cache_r_sentinel() {
        let v = 4;
        let d = 3;
        let h = 4;
        let emb = vec![0.5_f32; v * d];
        let w1 = vec![0.1_f32; h * d];
        let w2 = vec![0.2_f32; v * h];
        let input = vec![0.0_f32];

        let (_, cache) = forward_ffn_with_options(&emb, &w1, &w2, &input, v, d, h, None, false);
        // Sentinel: norm_emb_r and norm_hidden_r should be 0.0 when RMSNorm bypassed.
        assert_eq!(cache.norm_emb_r[0], 0.0, "norm_emb_r sentinel should be 0");
        assert_eq!(cache.norm_hidden_r[0], 0.0, "norm_hidden_r sentinel should be 0");
    }

    #[test]
    fn dropout_mask_zero_blocks_gradient() {
        // Loop 22: verify dropout backward gates gradient correctly.
        // Asymmetric weights ensure gradients don't cancel by structure.
        let v = 4;
        let d = 3;
        let h = 4;
        let emb: Vec<f32> = (0..v * d).map(|i| 0.1 + 0.07 * i as f32).collect();
        let w1: Vec<f32> = (0..h * d).map(|i| -0.2 + 0.13 * i as f32).collect();
        let w2: Vec<f32> = (0..v * h).map(|i| 0.05 - 0.09 * i as f32).collect();
        let input = vec![0.0_f32];
        let targets = vec![1_usize];

        let (logits, mut cache) = forward_ffn_with_cache(&emb, &w1, &w2, &input, v, d, h);

        // Test 1: all-dropped → backward should produce zero W1 gradients
        // (no signal flows through hidden layer back to W1 or embeddings).
        for h_idx in 0..h {
            cache.dropout_scale[h_idx] = 0.0;
        }
        let (_, d_w1_all_dropped, _) = backward_ffn_with_cache(
            &emb, &w1, &w2, &logits, &cache, &input, &targets, v, d, h,
        );
        let max_w1_zero: f32 = d_w1_all_dropped.iter().map(|g| g.abs()).fold(0.0, f32::max);
        assert!(
            max_w1_zero < 1e-5,
            "all-dropped should zero d_w1 (gradient blocked by dropout); got max={}",
            max_w1_zero
        );

        // Test 2: all-survivors with scale=2 → should produce non-zero W1 gradients
        for h_idx in 0..h {
            cache.dropout_scale[h_idx] = 2.0;
        }
        let (_, d_w1_survivors, _) = backward_ffn_with_cache(
            &emb, &w1, &w2, &logits, &cache, &input, &targets, v, d, h,
        );
        let max_w1_survivor: f32 = d_w1_survivors.iter().map(|g| g.abs()).fold(0.0, f32::max);
        assert!(
            max_w1_survivor > 1e-4,
            "all-survivors should produce non-zero d_w1; got max={}",
            max_w1_survivor
        );
    }

    #[test]
    fn rms_norm_zero_input_safe() {
        let x = vec![0.0_f32; 4];
        let (y, r) = rms_norm(&x);
        // With eps, r = sqrt(eps) > 0; y_i = 0/r = 0
        for v in &y {
            assert_eq!(*v, 0.0);
        }
        assert!(r > 0.0);
    }

    #[test]
    fn forward_ffn_shape_correct() {
        let v = 8;
        let d = 4;
        let h = 6;
        let emb = vec![0.1_f32; v * d];
        let w1 = vec![0.1_f32; h * d];
        let w2 = vec![0.1_f32; v * h];
        let input = vec![0.0_f32, 1.0, 2.0];
        let (logits, hidden) = forward_ffn(&emb, &w1, &w2, &input, v, d, h);
        assert_eq!(logits.len(), 3 * v);
        assert_eq!(hidden.len(), 3 * h);
    }

    #[test]
    fn backward_ffn_shape_correct() {
        let v = 8;
        let d = 4;
        let h = 6;
        let emb = vec![0.1_f32; v * d];
        let w1 = vec![0.1_f32; h * d];
        let w2 = vec![0.1_f32; v * h];
        let input = vec![0.0_f32, 1.0];
        let targets = vec![1, 2];
        let (logits, hidden) = forward_ffn(&emb, &w1, &w2, &input, v, d, h);
        let (d_emb, d_w1, d_w2) = backward_ffn(
            &emb, &w1, &w2, &logits, &hidden, &input, &targets, v, d, h,
        );
        assert_eq!(d_emb.len(), emb.len());
        assert_eq!(d_w1.len(), w1.len());
        assert_eq!(d_w2.len(), w2.len());
    }

    #[test]
    fn relu_masks_negative_hidden() {
        let v = 4;
        let d = 2;
        let h = 2;
        // Set W1 so that hidden[0] = positive, hidden[1] = negative pre-ReLU
        let emb = vec![1.0_f32, 1.0, /* tok 0 */];
        let emb_full = vec![1.0_f32; v * d];
        let w1 = vec![1.0, 1.0,    // h=0: positive sum
                      -1.0, -1.0]; // h=1: negative sum → ReLU zeros
        let w2 = vec![0.0_f32; v * h];
        let _ = emb;
        let input = vec![0.0_f32];
        let (_, hidden) = forward_ffn(&emb_full, &w1, &w2, &input, v, d, h);
        assert!(hidden[0] > 0.0);
        assert_eq!(hidden[1], 0.0); // ReLU zero
    }
}
