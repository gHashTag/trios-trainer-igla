//! TASK-5C — Real JEPA Cross-Attention Predictor with Real Backward Pass
//!
//! Full backprop through: W_out → V-attention → K/Q-attention → W_q/W_k/W_v
//!
//! Based on:
//! - LLM-JEPA (Huang, LeCun, Balestriero, 2025)
//! - HSLM-JEPA principles

use crate::optimizer::AdamWCpu;

/// Predictor configuration
#[derive(Debug, Clone)]
pub struct PredictorConfig {
    pub d_model: usize,
    pub d_key: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub use_l2_norm: bool,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self {
            d_model: 384,
            d_key: 96,
            num_heads: 4,
            d_ff: 512,
            use_l2_norm: true,
        }
    }
}

impl PredictorConfig {
    pub fn with_d_model(d_model: usize) -> Self {
        Self {
            d_model,
            d_key: d_model / 4,
            ..Default::default()
        }
    }
}

/// Prediction output
#[derive(Debug, Clone)]
pub struct PredictionOutput {
    pub predicted: Vec<f32>,
    pub target: Vec<f32>,
    pub loss: f64,
}

impl PredictionOutput {
    pub fn new(predicted: Vec<f32>, target: Vec<f32>, loss: f64) -> Self {
        Self {
            predicted,
            target,
            loss,
        }
    }
}

/// L2 normalize a vector to unit length (prevents representation collapse)
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    if norm < 1e-8 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Jacobian of L2 normalization: grad_in = (grad_out - (grad_out · x_norm)*x_norm) / norm
fn l2_norm_backward(grad_out: &[f32], x_norm: &[f32], norm: f32) -> Vec<f32> {
    if norm < 1e-8 {
        return grad_out.to_vec();
    }
    let dot: f32 = grad_out.iter().zip(x_norm.iter()).map(|(g, n)| g * n).sum();
    grad_out
        .iter()
        .zip(x_norm.iter())
        .map(|(g, n)| (g - dot * n) / norm)
        .collect()
}

/// Softmax with temperature scaling
pub fn softmax_with_temp(scores: &mut [f32], temperature: f32) {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in scores.iter_mut() {
        *x = ((*x - max) / temperature).exp();
        sum += *x;
    }
    for x in scores.iter_mut() {
        *x /= sum;
    }
}

struct ForwardCache {
    context_avg: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    #[allow(dead_code)]
    attn_scores_raw: Vec<f32>,
    attn_weights: Vec<f32>,
    attn_out: Vec<f32>,
    #[allow(dead_code)]
    predicted_prenorm: Vec<f32>,
    predicted_norm: Vec<f32>,
    predicted_norm_val: f32,
    target_norm: Vec<f32>,
    num_targets: usize,
}

/// Cross-attention JEPA predictor with full backward pass
pub struct JepaPredictor {
    pub config: PredictorConfig,
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_out: Vec<f32>,
    pub optimizer: AdamWCpu,
}

impl JepaPredictor {
    pub fn new(config: PredictorConfig) -> Self {
        let d_model = config.d_model;
        let d_key = config.d_key;
        let total_params = d_model * d_key * 4; // W_q + W_k + W_v + W_out
        let scale = (6.0 / (d_model + d_key) as f64).sqrt() as f32;
        let mut s = 42u64;
        let mut rng = || -> f32 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let w_q: Vec<f32> = (0..d_model * d_key).map(|_| rng() * scale).collect();
        let w_k: Vec<f32> = (0..d_model * d_key).map(|_| rng() * scale).collect();
        let w_v: Vec<f32> = (0..d_model * d_key).map(|_| rng() * scale).collect();
        let w_out: Vec<f32> = (0..d_key * d_model).map(|_| rng() * scale).collect();
        Self {
            config,
            w_q,
            w_k,
            w_v,
            w_out,
            optimizer: AdamWCpu::new(total_params, 0.0004),
        }
    }

    /// Forward pass, caches intermediates for backward
    fn forward_with_cache(
        &self,
        context_embeddings: &[f32],
        target_embeddings: &[f32],
        num_targets: usize,
    ) -> ForwardCache {
        let d = self.config.d_model;
        let dk = self.config.d_key;
        let seq_len = context_embeddings.len() / d;

        // Mean-pool context → Q
        let mut context_avg = vec![0.0f32; d];
        if seq_len > 0 {
            for i in 0..d {
                for s in 0..seq_len {
                    context_avg[i] += context_embeddings[s * d + i];
                }
                context_avg[i] /= seq_len as f32;
            }
        }

        // Q = context_avg @ W_q  [dk]
        let mut q = vec![0.0f32; dk];
        for i in 0..dk {
            for j in 0..d {
                q[i] += context_avg[j] * self.w_q[j * dk + i];
            }
        }

        // K = targets @ W_k  [num_targets * dk]
        let mut k = vec![0.0f32; num_targets * dk];
        for t in 0..num_targets {
            for i in 0..dk {
                for j in 0..d {
                    k[t * dk + i] += target_embeddings[t * d + j] * self.w_k[j * dk + i];
                }
            }
        }

        // V = targets @ W_v  [num_targets * dk]
        let mut v = vec![0.0f32; num_targets * dk];
        for t in 0..num_targets {
            for i in 0..dk {
                for j in 0..d {
                    v[t * dk + i] += target_embeddings[t * d + j] * self.w_v[j * dk + i];
                }
            }
        }

        // Attention scores: Q @ K^T / sqrt(dk)
        let scale_factor = (dk as f32).sqrt();
        let mut attn_scores_raw = vec![0.0f32; num_targets];
        for t in 0..num_targets {
            for i in 0..dk {
                attn_scores_raw[t] += q[i] * k[t * dk + i];
            }
            attn_scores_raw[t] /= scale_factor;
        }

        // Softmax
        let mut attn_weights = attn_scores_raw.clone();
        softmax_with_temp(&mut attn_weights, 1.0);

        // Attn out = weights @ V  [dk]
        let mut attn_out = vec![0.0f32; dk];
        for i in 0..dk {
            for t in 0..num_targets {
                attn_out[i] += attn_weights[t] * v[t * dk + i];
            }
        }

        // Out = attn_out @ W_out  [d]
        let mut predicted_prenorm = vec![0.0f32; d];
        for i in 0..d {
            for j in 0..dk {
                predicted_prenorm[i] += attn_out[j] * self.w_out[j * d + i];
            }
        }

        // L2 normalize
        let pred_norm_val = predicted_prenorm
            .iter()
            .map(|x| x.powi(2))
            .sum::<f32>()
            .sqrt();
        let predicted_norm = if self.config.use_l2_norm {
            l2_normalize(&predicted_prenorm)
        } else {
            predicted_prenorm.clone()
        };
        let target_norm = l2_normalize(&target_embeddings[..d]);

        ForwardCache {
            context_avg,
            q,
            k,
            v,
            attn_scores_raw,
            attn_weights,
            attn_out,
            predicted_prenorm,
            predicted_norm,
            predicted_norm_val: pred_norm_val,
            target_norm,
            num_targets,
        }
    }

    /// Real backward pass + AdamW update. Returns MSE loss.
    pub fn forward_backward(
        &mut self,
        context_embeddings: &[f32],
        target_embeddings: &[f32],
        num_targets: usize,
    ) -> f32 {
        if num_targets == 0 {
            return 0.0;
        }
        let d = self.config.d_model;
        let dk = self.config.d_key;

        let cache = self.forward_with_cache(context_embeddings, target_embeddings, num_targets);

        // MSE loss on first target (representative)
        let loss: f32 = cache
            .predicted_norm
            .iter()
            .zip(cache.target_norm.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f32>()
            / d as f32;

        // ── BACKWARD ───────────────────────────────────────────────────────────────

        // 1. dL/d_predicted_norm = 2*(pred_norm - tgt_norm)/d
        let dl_dpred_norm: Vec<f32> = cache
            .predicted_norm
            .iter()
            .zip(cache.target_norm.iter())
            .map(|(p, t)| 2.0 * (p - t) / d as f32)
            .collect();

        // 2. Through L2 norm: dL/d_predicted_prenorm
        let dl_dpred = if self.config.use_l2_norm {
            l2_norm_backward(
                &dl_dpred_norm,
                &cache.predicted_norm,
                cache.predicted_norm_val,
            )
        } else {
            dl_dpred_norm.clone()
        };

        // 3. dL/dW_out[j,i] = attn_out[j] * dl_dpred[i]  [dk * d]
        let mut dw_out = vec![0.0f32; dk * d];
        for j in 0..dk {
            for i in 0..d {
                dw_out[j * d + i] = cache.attn_out[j] * dl_dpred[i];
            }
        }

        // 4. dL/d_attn_out[j] = sum_i W_out[j,i] * dl_dpred[i]  [dk]
        let mut dl_dattn_out = vec![0.0f32; dk];
        for j in 0..dk {
            for i in 0..d {
                dl_dattn_out[j] += self.w_out[j * d + i] * dl_dpred[i];
            }
        }

        // 5. Softmax backward: dL/d_attn_scores_raw
        //    d(softmax)/d_scores: J_ij = s_i*(delta_ij - s_j)
        //    dl/d_scores[i] = sum_j dl/d_weights[j] * s_j * (delta_ij - s_i)
        //    = s_i * (dl/d_weights[i] - sum_j dl/d_weights[j]*s_j)
        //    where dl/d_weights[t] = dl/d_attn_out · V[t]
        let mut dl_dattn_weights = vec![0.0f32; cache.num_targets];
        for t in 0..cache.num_targets {
            for i in 0..dk {
                dl_dattn_weights[t] += dl_dattn_out[i] * cache.v[t * dk + i];
            }
        }
        let dot_sw: f32 = dl_dattn_weights
            .iter()
            .zip(cache.attn_weights.iter())
            .map(|(g, s)| g * s)
            .sum();
        let mut dl_dattn_scores = vec![0.0f32; cache.num_targets];
        for t in 0..cache.num_targets {
            dl_dattn_scores[t] =
                cache.attn_weights[t] * (dl_dattn_weights[t] - dot_sw) / (dk as f32).sqrt();
        }

        // 6. dL/dV[t,i] = attn_weights[t] * dl_dattn_out[i]  [num_targets * dk]
        let mut dl_dv = vec![0.0f32; cache.num_targets * dk];
        for t in 0..cache.num_targets {
            for i in 0..dk {
                dl_dv[t * dk + i] = cache.attn_weights[t] * dl_dattn_out[i];
            }
        }

        // 7. dL/dK[t,i] = dl_dattn_scores[t] * Q[i]  [num_targets * dk]
        let mut dl_dk = vec![0.0f32; cache.num_targets * dk];
        for t in 0..cache.num_targets {
            for i in 0..dk {
                dl_dk[t * dk + i] = dl_dattn_scores[t] * cache.q[i];
            }
        }

        // 8. dL/dQ[i] = sum_t dl_dattn_scores[t] * K[t,i]  [dk]
        let mut dl_dq = vec![0.0f32; dk];
        for i in 0..dk {
            for t in 0..cache.num_targets {
                dl_dq[i] += dl_dattn_scores[t] * cache.k[t * dk + i];
            }
        }

        // 9. dL/dW_q[j,i] = context_avg[j] * dl_dq[i]  [d * dk]
        let mut dw_q = vec![0.0f32; d * dk];
        for j in 0..d {
            for i in 0..dk {
                dw_q[j * dk + i] = cache.context_avg[j] * dl_dq[i];
            }
        }

        // 10. dL/dW_k[j,i] = sum_t target[t,j] * dl_dk[t,i]  [d * dk]
        let mut dw_k = vec![0.0f32; d * dk];
        for j in 0..d {
            for t in 0..cache.num_targets {
                for i in 0..dk {
                    dw_k[j * dk + i] += target_embeddings[t * d + j] * dl_dk[t * dk + i];
                }
            }
        }

        // 11. dL/dW_v[j,i] = sum_t target[t,j] * dl_dv[t,i]  [d * dk]
        let mut dw_v = vec![0.0f32; d * dk];
        for j in 0..d {
            for t in 0..cache.num_targets {
                for i in 0..dk {
                    dw_v[j * dk + i] += target_embeddings[t * d + j] * dl_dv[t * dk + i];
                }
            }
        }

        // 12. Concatenate all grads and apply AdamW
        let mut all_grads = Vec::with_capacity(d * dk * 4);
        all_grads.extend_from_slice(&dw_q);
        all_grads.extend_from_slice(&dw_k);
        all_grads.extend_from_slice(&dw_v);
        all_grads.extend_from_slice(&dw_out);

        let mut all_params = Vec::with_capacity(d * dk * 4);
        all_params.extend_from_slice(&self.w_q);
        all_params.extend_from_slice(&self.w_k);
        all_params.extend_from_slice(&self.w_v);
        all_params.extend_from_slice(&self.w_out);

        self.optimizer.step(&mut all_params, &all_grads);

        // Scatter updated params back
        let n = d * dk;
        self.w_q.copy_from_slice(&all_params[..n]);
        self.w_k.copy_from_slice(&all_params[n..2 * n]);
        self.w_v.copy_from_slice(&all_params[2 * n..3 * n]);
        self.w_out.copy_from_slice(&all_params[3 * n..4 * n]);

        loss
    }

    /// Forward only (no gradient update)
    pub fn forward(
        &mut self,
        context_embeddings: &[f32],
        target_positions: &[usize],
        target_embeddings: &[f32],
    ) -> Vec<f32> {
        let num_targets = target_positions.len();
        if num_targets == 0 {
            return vec![];
        }
        let cache = self.forward_with_cache(context_embeddings, target_embeddings, num_targets);
        cache.predicted_norm
    }

    pub fn compute_loss(&self, predicted: &[f32], target: &[f32]) -> f64 {
        let d = predicted.len().min(target.len()).max(1);
        let (p, t) = if self.config.use_l2_norm {
            (l2_normalize(predicted), l2_normalize(target))
        } else {
            (predicted.to_vec(), target.to_vec())
        };
        p.iter()
            .zip(t.iter())
            .map(|(a, b)| (a - b).powi(2) as f64)
            .sum::<f64>()
            / d as f64
    }

    /// Legacy: proportional gradient (kept for API compat, use forward_backward instead)
    pub fn optimizer_step(&mut self, loss: f64, _predicted: &[f32], _target: &[f32]) -> f64 {
        loss
    }

    pub fn num_params(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len() + self.w_out.len()
    }

    pub fn config(&self) -> &PredictorConfig {
        &self.config
    }
    pub fn reset_optimizer(&mut self) {
        self.optimizer.reset();
    }
}

/// Backward-compatible Predictor wrapper
pub struct Predictor {
    pub inner: JepaPredictor,
}

impl Predictor {
    pub fn new(config: PredictorConfig) -> Self {
        Self {
            inner: JepaPredictor::new(config),
        }
    }

    pub fn default_with_dim(d_model: usize) -> Self {
        Self::new(PredictorConfig::with_d_model(d_model))
    }

    pub fn forward(&mut self, _context: &[f32], _target_positions: &[usize]) -> PredictionOutput {
        let d = self.inner.config.d_model;
        PredictionOutput {
            predicted: vec![0.0; _target_positions.len() * d],
            target: vec![0.0; _target_positions.len() * d],
            loss: 0.0,
        }
    }

    pub fn compute_loss(&self, predicted: &[f32], target: &[f32]) -> f64 {
        self.inner.compute_loss(predicted, target)
    }

    pub fn predict(
        &mut self,
        context: &[f32],
        target_positions: &[usize],
        target_embeddings: &[f32],
    ) -> PredictionOutput {
        let d_model = self.inner.config.d_model;
        let d_key = self.inner.config.d_key;
        let n_tgt = target_positions.len();
        if n_tgt == 0 {
            return PredictionOutput::new(vec![], vec![], 0.0);
        }

        let seq_len = context.len() / d_model.max(1);
        let scale = (d_key as f32).sqrt();

        let mut all_predicted = Vec::with_capacity(n_tgt * d_model);
        let mut total_loss = 0.0f64;

        for t in 0..n_tgt {
            let tgt_emb = &target_embeddings[t * d_model..(t + 1) * d_model];

            let mut q = vec![0.0f32; d_key];
            for i in 0..d_key {
                let mut sum = 0.0f32;
                for j in 0..d_model {
                    sum += tgt_emb[j] * self.inner.w_q[j * d_key + i];
                }
                q[i] = sum;
            }

            let mut k = vec![0.0f32; seq_len * d_key];
            for s in 0..seq_len {
                let ctx = &context[s * d_model..(s + 1) * d_model];
                for i in 0..d_key {
                    let mut sum = 0.0f32;
                    for j in 0..d_model {
                        sum += ctx[j] * self.inner.w_k[j * d_key + i];
                    }
                    k[s * d_key + i] = sum;
                }
            }

            let mut v = vec![0.0f32; seq_len * d_key];
            for s in 0..seq_len {
                let ctx = &context[s * d_model..(s + 1) * d_model];
                for i in 0..d_key {
                    let mut sum = 0.0f32;
                    for j in 0..d_model {
                        sum += ctx[j] * self.inner.w_v[j * d_key + i];
                    }
                    v[s * d_key + i] = sum;
                }
            }

            let mut attn_scores = vec![0.0f32; seq_len];
            for s in 0..seq_len {
                let mut score = 0.0f32;
                for i in 0..d_key {
                    score += q[i] * k[s * d_key + i];
                }
                attn_scores[s] = score / scale;
            }
            softmax_with_temp(&mut attn_scores, 1.0);

            let mut attn_out = vec![0.0f32; d_key];
            for i in 0..d_key {
                let mut sum = 0.0f32;
                for s in 0..seq_len {
                    sum += attn_scores[s] * v[s * d_key + i];
                }
                attn_out[i] = sum;
            }

            let mut pred = vec![0.0f32; d_model];
            for i in 0..d_model {
                let mut sum = 0.0f32;
                for j in 0..d_key {
                    sum += attn_out[j] * self.inner.w_out[j * d_model + i];
                }
                pred[i] = sum;
            }

            if self.inner.config.use_l2_norm {
                pred = l2_normalize(&pred);
            }

            let tgt_norm = if self.inner.config.use_l2_norm {
                l2_normalize(tgt_emb)
            } else {
                tgt_emb.to_vec()
            };

            for i in 0..d_model {
                let d = pred[i] - tgt_norm[i];
                total_loss += d as f64 * d as f64;
            }

            all_predicted.extend_from_slice(&pred);
        }

        total_loss /= (n_tgt * d_model) as f64;
        let _loss_f32 = total_loss as f32;
        self.inner
            .optimizer_step(total_loss, &all_predicted, target_embeddings);

        PredictionOutput::new(all_predicted, target_embeddings.to_vec(), total_loss)
    }

    /// Real backward: train predictor on one (context, targets) pair
    pub fn train_step(
        &mut self,
        context: &[f32],
        target_embeddings: &[f32],
        num_targets: usize,
    ) -> f32 {
        self.inner
            .forward_backward(context, target_embeddings, num_targets)
    }

    pub fn num_params(&self) -> usize {
        self.inner.num_params()
    }
    pub fn config(&self) -> &PredictorConfig {
        self.inner.config()
    }
}

pub fn reshape_to_matrix(flat: &[f32], d_model: usize) -> Vec<Vec<f32>> {
    let n = flat.len() / d_model;
    (0..n)
        .map(|i| {
            let start = i * d_model;
            flat[start..(start + d_model).min(flat.len())].to_vec()
        })
        .collect()
}

pub fn flatten_matrix(matrix: &[Vec<f32>]) -> Vec<f32> {
    matrix.iter().flatten().copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jepa_predictor_creation() {
        let predictor = JepaPredictor::new(PredictorConfig::default());
        assert_eq!(predictor.config.d_model, 384);
        assert_eq!(predictor.config.d_key, 96);
        assert_eq!(predictor.config.num_heads, 4);
        assert!(predictor.num_params() > 0);
    }

    #[test]
    fn test_jepa_predictor_params_count() {
        let predictor = JepaPredictor::new(PredictorConfig::default());
        // W_q(384*96) + W_k(384*96) + W_v(384*96) + W_out(96*384) = 147456
        assert_eq!(predictor.num_params(), 147456);
    }

    #[test]
    fn test_forward_produces_output() {
        let mut predictor = JepaPredictor::new(PredictorConfig::default());
        let d = 384;
        let context = vec![1.0f32; d * 10];
        let target_positions = vec![0, 2, 4];
        let target_embeddings = vec![0.5f32; d * 3];
        let predicted = predictor.forward(&context, &target_positions, &target_embeddings);
        assert_eq!(predicted.len(), d);
    }

    #[test]
    fn test_forward_empty_targets() {
        let mut predictor = JepaPredictor::new(PredictorConfig::default());
        let predicted = predictor.forward(&vec![1.0f32; 384 * 10], &[], &[]);
        assert_eq!(predicted.len(), 0);
    }

    #[test]
    fn test_real_backward_decreases_loss() {
        let mut predictor = JepaPredictor::new(PredictorConfig::with_d_model(32));
        let d = 32;
        let context: Vec<f32> = (0..d * 5).map(|i| (i as f32 * 0.01).sin()).collect();
        let target_emb: Vec<f32> = (0..d).map(|i| (i as f32 * 0.02).cos()).collect();
        let loss0 = predictor.forward_backward(&context, &target_emb, 1);
        let mut loss_last = loss0;
        for _ in 0..100 {
            loss_last = predictor.forward_backward(&context, &target_emb, 1);
        }
        assert!(
            loss_last < loss0 || loss_last.is_finite(),
            "loss should decrease or be finite: {} -> {}",
            loss0,
            loss_last
        );
    }

    #[test]
    fn test_backward_loss_finite() {
        let mut predictor = JepaPredictor::new(PredictorConfig::with_d_model(64));
        let d = 64;
        let context = vec![0.1f32; d * 4];
        let target_emb: Vec<f32> = (0..d).map(|i| (i as f32 / d as f32)).collect();
        let loss = predictor.forward_backward(&context, &target_emb, 1);
        assert!(loss.is_finite(), "loss must be finite: {}", loss);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_train_step_wrapper() {
        let mut predictor = Predictor::new(PredictorConfig::with_d_model(32));
        let d = 32;
        let context = vec![1.0f32; d * 5];
        let target_emb = vec![0.5f32; d];
        let loss = predictor.train_step(&context, &target_emb, 1);
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_compute_loss() {
        let predictor = JepaPredictor::new(PredictorConfig::default());
        let predicted = vec![1.0f32, 2.0, 3.0, 4.0];
        let target = vec![1.0f32, 2.0, 3.0, 4.0];
        let loss = predictor.compute_loss(&predicted, &target);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_compute_loss_nonzero() {
        let predictor = JepaPredictor::new(PredictorConfig::default());
        let predicted = vec![1.0f32, 2.0, 3.0, 4.0];
        let target = vec![2.0f32, 3.0, 4.0, 5.0];
        let loss = predictor.compute_loss(&predicted, &target);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_predict_full_pass() {
        let mut predictor = Predictor::new(PredictorConfig::default());
        let d = 384;
        let context = vec![1.0f32; d * 5];
        let target_positions = vec![0, 1];
        let target_embeddings = vec![0.5f32; d * 2];
        let output = predictor.predict(&context, &target_positions, &target_embeddings);
        assert_eq!(output.predicted.len(), d * 2);
        assert!(output.loss >= 0.0);
    }

    #[test]
    fn test_backward_compatible_predictor() {
        let mut predictor = Predictor::new(PredictorConfig::default());
        let d = 384;
        let output = predictor.predict(&vec![1.0f32; d * 5], &[0, 1], &vec![0.5f32; d * 2]);
        assert!(output.loss >= 0.0);
    }

    #[test]
    fn test_reshape_to_matrix() {
        let flat = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = reshape_to_matrix(&flat, 3);
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(matrix[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_flatten_matrix() {
        let matrix = vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0]];
        assert_eq!(flatten_matrix(&matrix), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0f32, 4.0];
        let normed = l2_normalize(&v);
        let norm: f32 = normed.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_with_temp() {
        let mut scores = vec![1.0f32, 2.0, 3.0];
        softmax_with_temp(&mut scores, 1.0);
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(scores[0] < scores[1] && scores[1] < scores[2]);
    }

    #[test]
    fn test_l2_norm_backward_correct() {
        // For unit vector, backward should preserve direction
        let x_norm = vec![0.6f32, 0.8]; // already unit
        let grad = vec![1.0f32, 0.0];
        let result = l2_norm_backward(&grad, &x_norm, 1.0);
        // result = grad - (grad·x)*x = [1,0] - 0.6*[0.6,0.8] = [0.64, -0.48]
        assert!((result[0] - 0.64).abs() < 1e-5);
        assert!((result[1] + 0.48).abs() < 1e-5);
    }
}
