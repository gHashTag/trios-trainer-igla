//! GF16-based N-Gram Language Model Training
//!
//! Uses pure Rust GF16 implementation (no FFI, no C library)
//! Tests if φ-optimized quantization improves BPB over f32 baseline

#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]
#![allow(unused_variables)]

use std::fs;
use std::io::Write;
use std::time::Instant;

use trios_trainer::gf16::{QuantizationMetrics, GF16};
use trios_trainer::neon_writer as nw;

const VOCAB: usize = 128;
const SEQ: usize = 64;
const LN_2: f32 = std::f32::consts::LN_2;

fn gelu(x: f32) -> f32 {
    let x3 = x * x * x;
    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
    let tanh_val = tanh_arg.tanh();
    0.5 * x * (1.0 + tanh_val)
}

/// Load a byte corpus as VOCAB-folded tokens, or fail.
///
/// This function used to substitute a 52-byte string for any corpus it could
/// not read, behind an `eprintln!`. The hardcoded path it was called with
/// misspelled the corpus in `data/` (it dropped an underscore), so the file has
/// never existed in this repo, EVERY run took the fallback, and every BPB it
/// published measured how well the model had memorised 52 bytes. Measured:
/// BPB=2.5648, within 0.05 of the crate's own BPB_CHAMPION=2.5193.
///
/// There is no fallback now, opt-in or otherwise: unlike `train_loop::load_data`
/// this binary has no way to stamp a `data_synthetic` bit into an artifact, so a
/// synthetic run here would be indistinguishable after the fact.
fn load_data(path: &str) -> Result<Vec<usize>, String> {
    let raw = fs::read(path).map_err(|e| {
        format!(
            "corpus '{path}' could not be read: {e}. Refusing the synthetic fallback: \
             substituting a placeholder string for a missing corpus is the documented \
             cause of the leak-tainted BPB rows (trios-trainer-igla#60)."
        )
    })?;
    if raw.is_empty() {
        return Err(format!("corpus '{path}' is empty"));
    }
    Ok(raw.into_iter().map(|b| (b as usize) % VOCAB).collect())
}

/// Refuse to measure against a val stream that is not held out.
///
/// Returns the reason the eval would be meaningless, or `None` when the split
/// can carry a measurement. Mirrors `train_loop::assert_train_val_disjoint`,
/// which is `pub(crate)` and so not reachable from a binary target.
fn val_not_held_out(train: &[usize], val: &[usize]) -> Option<String> {
    if val.is_empty() {
        return Some("val corpus is empty".to_string());
    }
    if train.is_empty() {
        return Some("train corpus is empty".to_string());
    }
    if train.as_ptr() == val.as_ptr() {
        return Some("train and val are the same allocation".to_string());
    }
    if train == val {
        return Some("train and val are byte-identical".to_string());
    }

    use std::collections::HashSet;

    // Full-coverage overlap probe. Every train window of the probe length is
    // hashed once and the val probe is looked up in that set; a `step_by`
    // sampled scan detects a uniformly placed overlap with probability
    // 1/step, and a guard that is blind most of the time is worse than none
    // because it gets cited as evidence.
    let probe_len = 1024.min(val.len()).min(train.len());
    let val_probe = &val[..probe_len];
    let train_windows: HashSet<&[usize]> = train.windows(probe_len).collect();
    if train_windows.contains(val_probe) {
        return Some(format!(
            "TRAIN/VAL OVERLAP: first {probe_len} val tokens occur in train \
             (trios-trainer-igla#60)"
        ));
    }

    // A disjoint corpus is not yet an informative one. A periodic or heavily
    // duplicated eval stream drives BPB toward zero honestly, which is exactly
    // the signature that got ledger rows misfiled as leaks.
    if val.len() >= 8 {
        let distinct: HashSet<&[usize]> = val.windows(8).collect();
        let total = val.len() - 7;
        let ratio = distinct.len() as f64 / total as f64;
        if ratio < 0.05 {
            return Some(format!(
                "DEGENERATE EVAL CORPUS: only {:.3}% of val 8-grams are distinct \
                 ({} of {})",
                ratio * 100.0,
                distinct.len(),
                total
            ));
        }
    }
    None
}

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

fn layer_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = (var + eps).sqrt();
    x.iter().map(|v| (v - mean) / std).collect()
}

/// GF16 parameter storage - all weights stored as GF16
struct Gf16Parameters {
    data: Vec<GF16>,
}

impl Gf16Parameters {
    fn new(size: usize) -> Self {
        Self {
            data: vec![GF16::ZERO; size],
        }
    }

    fn from_f32(values: &[f32]) -> Self {
        Self {
            data: values.iter().map(|&v| GF16::from_f32(v)).collect(),
        }
    }

    fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&v| v.to_f32()).collect()
    }

    fn get(&self, idx: usize) -> f32 {
        self.data[idx].to_f32()
    }

    fn set(&mut self, idx: usize, val: f32) {
        self.data[idx] = GF16::from_f32(val);
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    /// Compute quantization error metrics for this parameter block
    fn quantization_metrics(&self, original_f32: &[f32]) -> QuantizationMetrics {
        let dequantized = self.to_f32();
        QuantizationMetrics::compute(original_f32, &dequantized)
    }
}

struct Optimizers {
    e: AdamW,
    c1: AdamW,
    c2: AdamW,
    c3: AdamW,
    c4: AdamW,
    c5: AdamW,
    p: AdamW,
    h: AdamW,
}

struct AdamW {
    m: Vec<f32>,
    v: Vec<f32>,
    step: usize,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
}

impl AdamW {
    fn new(size: usize, wd: f32) -> Self {
        let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            step: 0,
            beta1: 1.0 / phi as f32,
            beta2: 0.999,
            eps: 1e-8,
            wd,
        }
    }

    fn update(&mut self, params: &mut Gf16Parameters, grads: &[f32], lr: f32) {
        self.step += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);

        // First, dequantize params to f32, apply updates, then requantize
        let mut params_f32: Vec<f32> = params.to_f32();

        for i in 0..params_f32.len() {
            // Weight decay
            params_f32[i] -= self.wd * lr * params_f32[i];

            // Adam update
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            let update = lr * (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + self.eps);
            params_f32[i] -= update;
        }

        // Re-quantize back to GF16
        params.data = params_f32.into_iter().map(GF16::from_f32).collect();
    }
}

struct NgramModelGF16 {
    embed: Gf16Parameters,
    ctx1: Gf16Parameters,
    ctx2: Gf16Parameters,
    ctx3: Gf16Parameters,
    ctx4: Gf16Parameters,
    ctx5: Gf16Parameters,
    proj: Gf16Parameters,
    lm_head: Gf16Parameters,
    vocab: usize,
    dim: usize,
    hidden: usize,
    activation: String,
    dropout: f32,
    use_ctx3: bool,
    use_ctx4: bool,
    use_ctx5: bool,
    label_smoothing: f32,
}

impl NgramModelGF16 {
    #[allow(clippy::too_many_arguments)]
    fn new(
        vocab: usize,
        dim: usize,
        hidden: usize,
        activation: String,
        seed: u64,
        dropout: f32,
        use_ctx3: bool,
        use_ctx4: bool,
        use_ctx5: bool,
        label_smoothing: f32,
    ) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let n_ctx = if use_ctx5 {
            6
        } else if use_ctx4 {
            5
        } else if use_ctx3 {
            4
        } else {
            3
        };
        let lim = (6.0f32 / (n_ctx * dim) as f32).sqrt();
        let lim_h = (6.0f32 / (dim + hidden) as f32).sqrt();
        let lim_o = (6.0f32 / (hidden + dim) as f32).sqrt();

        Self {
            embed: Gf16Parameters::from_f32(
                &(0..vocab * dim).map(|_| rng() * lim).collect::<Vec<_>>(),
            ),
            ctx1: Gf16Parameters::from_f32(
                &(0..vocab * dim).map(|_| rng() * lim).collect::<Vec<_>>(),
            ),
            ctx2: Gf16Parameters::from_f32(
                &(0..vocab * dim).map(|_| rng() * lim).collect::<Vec<_>>(),
            ),
            ctx3: Gf16Parameters::from_f32(
                &(0..vocab * dim).map(|_| rng() * lim).collect::<Vec<_>>(),
            ),
            ctx4: Gf16Parameters::from_f32(
                &(0..vocab * dim).map(|_| rng() * lim).collect::<Vec<_>>(),
            ),
            ctx5: Gf16Parameters::from_f32(
                &(0..vocab * dim).map(|_| rng() * lim).collect::<Vec<_>>(),
            ),
            proj: Gf16Parameters::from_f32(
                &(0..hidden * dim).map(|_| rng() * lim_h).collect::<Vec<_>>(),
            ),
            lm_head: Gf16Parameters::from_f32(
                &(0..vocab * hidden)
                    .map(|_| rng() * lim_o)
                    .collect::<Vec<_>>(),
            ),
            vocab,
            dim,
            hidden,
            activation,
            dropout,
            use_ctx3,
            use_ctx4,
            use_ctx5,
            label_smoothing,
        }
    }

    fn get_hidden(
        &self,
        t5: usize,
        t4: usize,
        t3: usize,
        t2: usize,
        t1: usize,
        t0: usize,
    ) -> Vec<f32> {
        let _v = self.vocab;
        let d = self.dim;
        let h = self.hidden;

        let e0: Vec<f32> = (0..d).map(|j| self.embed.get(t0 * d + j)).collect();
        let c1: Vec<f32> = (0..d).map(|j| self.ctx1.get(t1 * d + j)).collect();
        let c2: Vec<f32> = (0..d).map(|j| self.ctx2.get(t2 * d + j)).collect();

        let mut combined = vec![0.0f32; d];
        if self.use_ctx5 {
            let c3: Vec<f32> = (0..d).map(|j| self.ctx3.get(t3 * d + j)).collect();
            let c4: Vec<f32> = (0..d).map(|j| self.ctx4.get(t4 * d + j)).collect();
            let c5: Vec<f32> = (0..d).map(|j| self.ctx5.get(t5 * d + j)).collect();
            for j in 0..d {
                combined[j] = e0[j]
                    + c1[j] * 0.35
                    + c2[j] * 0.22
                    + c3[j] * 0.15
                    + c4[j] * 0.11
                    + c5[j] * 0.17;
            }
        } else if self.use_ctx4 {
            let c3: Vec<f32> = (0..d).map(|j| self.ctx3.get(t3 * d + j)).collect();
            let c4: Vec<f32> = (0..d).map(|j| self.ctx4.get(t4 * d + j)).collect();
            for j in 0..d {
                combined[j] = e0[j] + c1[j] * 0.4 + c2[j] * 0.25 + c3[j] * 0.2 + c4[j] * 0.15;
            }
        } else if self.use_ctx3 {
            let c3: Vec<f32> = (0..d).map(|j| self.ctx3.get(t3 * d + j)).collect();
            for j in 0..d {
                combined[j] = e0[j] + c1[j] * 0.5 + c2[j] * 0.3 + c3[j] * 0.2;
            }
        } else {
            for j in 0..d {
                combined[j] = e0[j] + c1[j] * 0.7 + c2[j] * 0.3;
            }
        }

        let ln = layer_norm(&combined, 1e-5);

        let mut hidden = vec![0.0f32; h];
        for hi in 0..h {
            for j in 0..d {
                hidden[hi] += self.proj.get(hi * d + j) * ln[j];
            }
            // Activation
            hidden[hi] = if self.activation == "gelu" {
                gelu(hidden[hi])
            } else {
                hidden[hi].max(0.0)
            };
            // Dropout
            if self.dropout > 0.0 {
                let mask = ((hi as u64).wrapping_mul(6364136223846793005u64) >> 33) as f32
                    / u32::MAX as f32;
                if mask < self.dropout {
                    hidden[hi] = 0.0;
                } else {
                    hidden[hi] /= 1.0 - self.dropout;
                }
            }
        }
        hidden
    }

    fn loss_on_seq(&self, tokens: &[usize]) -> f32 {
        let start = if self.use_ctx5 {
            5
        } else if self.use_ctx4 {
            4
        } else if self.use_ctx3 {
            3
        } else {
            2
        };
        if tokens.len() < start + 2 {
            return 0.0;
        }
        let v = self.vocab;
        let mut total = 0.0f32;
        for i in start..tokens.len() - 1 {
            let t5 = if self.use_ctx5 {
                tokens[i - 5].min(v - 1)
            } else {
                0
            };
            let t4 = if self.use_ctx5 || self.use_ctx4 {
                tokens[i - 4].min(v - 1)
            } else {
                0
            };
            let t3 = if self.use_ctx5 || self.use_ctx4 || self.use_ctx3 {
                tokens[i - 3].min(v - 1)
            } else {
                0
            };
            let t2 = tokens[i - 2].min(v - 1);
            let t1 = tokens[i - 1].min(v - 1);
            let t0 = tokens[i].min(v - 1);
            let hidden_vec = self.get_hidden(t5, t4, t3, t2, t1, t0);
            let target = tokens[i + 1].min(v - 1);
            let mut logits = vec![0.0f32; v];
            for vi in 0..v {
                for hi in 0..self.hidden {
                    logits[vi] += self.lm_head.get(vi * self.hidden + hi) * hidden_vec[hi];
                }
            }
            softmax(&mut logits);
            total -= logits[target].max(1e-10).ln();
        }
        total / (tokens.len() - start - 1) as f32
    }

    #[allow(clippy::needless_range_loop)]
    fn train_step(&mut self, tokens: &[usize], lr: f32, opts: &mut Optimizers) {
        let start = if self.use_ctx5 {
            5
        } else if self.use_ctx4 {
            4
        } else if self.use_ctx3 {
            3
        } else {
            2
        };
        if tokens.len() < start + 2 {
            return;
        }
        let v = self.vocab;
        let d = self.dim;
        let h = self.hidden;

        let mut g_embed = vec![0.0f32; v * d];
        let mut g_ctx1 = vec![0.0f32; v * d];
        let mut g_ctx2 = vec![0.0f32; v * d];
        let mut g_ctx3 = vec![0.0f32; v * d];
        let mut g_ctx4 = vec![0.0f32; v * d];
        let mut g_ctx5 = vec![0.0f32; v * d];
        let mut g_proj = vec![0.0f32; h * d];
        let mut g_head = vec![0.0f32; v * h];

        for i in start..tokens.len() - 1 {
            let t5 = if self.use_ctx5 {
                tokens[i - 5].min(v - 1)
            } else {
                0
            };
            let t4 = if self.use_ctx5 || self.use_ctx4 {
                tokens[i - 4].min(v - 1)
            } else {
                0
            };
            let t3 = if self.use_ctx5 || self.use_ctx4 || self.use_ctx3 {
                tokens[i - 3].min(v - 1)
            } else {
                0
            };
            let t2 = tokens[i - 2].min(v - 1);
            let t1 = tokens[i - 1].min(v - 1);
            let t0 = tokens[i].min(v - 1);
            let tgt = tokens[i + 1].min(v - 1);

            let hidden = self.get_hidden(t5, t4, t3, t2, t1, t0);

            let mut logits = vec![0.0f32; v];
            for vi in 0..v {
                for hi in 0..h {
                    logits[vi] += self.lm_head.get(vi * h + hi) * hidden[hi];
                }
            }
            softmax(&mut logits);

            let mut d_hidden = vec![0.0f32; h];
            let smooth = self.label_smoothing;
            for (vi, &prob) in logits.iter().enumerate() {
                let target_val = if vi == tgt {
                    1.0 - smooth + smooth / v as f32
                } else {
                    smooth / v as f32
                };
                let grad = prob - target_val;
                for hi in 0..h {
                    g_head[vi * h + hi] += grad * hidden[hi];
                    d_hidden[hi] += grad * self.lm_head.get(vi * h + hi);
                }
            }

            // Backprop through proj (simplified, no full LN backward)
            let e0: Vec<f32> = (0..d).map(|j| self.embed.get(t0 * d + j)).collect();
            let c1: Vec<f32> = (0..d).map(|j| self.ctx1.get(t1 * d + j)).collect();
            let c2: Vec<f32> = (0..d).map(|j| self.ctx2.get(t2 * d + j)).collect();
            let mut combined = vec![0.0f32; d];
            if self.use_ctx5 {
                let c3: Vec<f32> = (0..d).map(|j| self.ctx3.get(t3 * d + j)).collect();
                let c4: Vec<f32> = (0..d).map(|j| self.ctx4.get(t4 * d + j)).collect();
                let c5: Vec<f32> = (0..d).map(|j| self.ctx5.get(t5 * d + j)).collect();
                for j in 0..d {
                    combined[j] = e0[j]
                        + c1[j] * 0.35
                        + c2[j] * 0.22
                        + c3[j] * 0.15
                        + c4[j] * 0.11
                        + c5[j] * 0.17;
                }
            } else if self.use_ctx4 {
                let c3: Vec<f32> = (0..d).map(|j| self.ctx3.get(t3 * d + j)).collect();
                let c4: Vec<f32> = (0..d).map(|j| self.ctx4.get(t4 * d + j)).collect();
                for j in 0..d {
                    combined[j] = e0[j] + c1[j] * 0.4 + c2[j] * 0.25 + c3[j] * 0.2 + c4[j] * 0.15;
                }
            } else if self.use_ctx3 {
                let c3: Vec<f32> = (0..d).map(|j| self.ctx3.get(t3 * d + j)).collect();
                for j in 0..d {
                    combined[j] = e0[j] + c1[j] * 0.5 + c2[j] * 0.3 + c3[j] * 0.2;
                }
            } else {
                for j in 0..d {
                    combined[j] = e0[j] + c1[j] * 0.7 + c2[j] * 0.3;
                }
            }
            let ln = layer_norm(&combined, 1e-5);

            for hi in 0..h {
                let activation_grad = if self.activation == "gelu" {
                    // GELU derivative approximation
                    let x = hidden[hi];
                    0.5 * (1.0 + (0.7978846 * (x + 0.044715 * x * x * x)).tanh())
                        + 0.5
                            * x
                            * 0.7978846
                            * (1.0 + 0.134145 * x * x * x)
                            * (1.0 - (0.7978846 * (x + 0.044715 * x * x * x)).tanh().powi(2))
                } else {
                    if hidden[hi] > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                };
                for j in 0..d {
                    g_proj[hi * d + j] += d_hidden[hi] * activation_grad * ln[j];
                }
            }
            for j in 0..d {
                let grad_j = d_hidden
                    .iter()
                    .enumerate()
                    .map(|(hi, dh)| {
                        self.proj.get(hi * d + j)
                            * if self.activation == "gelu" {
                                let x = hidden[hi];
                                0.5 * (1.0 + (0.7978846 * (x + 0.044715 * x * x * x)).tanh())
                                    + 0.5
                                        * x
                                        * 0.7978846
                                        * (1.0 + 0.134145 * x * x * x)
                                        * (1.0
                                            - (0.7978846 * (x + 0.044715 * x * x * x))
                                                .tanh()
                                                .powi(2))
                            } else {
                                if hidden[hi] > 0.0 {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            * dh
                    })
                    .sum::<f32>();
                g_embed[t0 * d + j] += grad_j;
                if self.use_ctx5 {
                    g_ctx1[t1 * d + j] += 0.35 * grad_j;
                    g_ctx2[t2 * d + j] += 0.22 * grad_j;
                    g_ctx3[t3 * d + j] += 0.15 * grad_j;
                    g_ctx4[t4 * d + j] += 0.11 * grad_j;
                    g_ctx5[t5 * d + j] += 0.17 * grad_j;
                } else if self.use_ctx4 {
                    g_ctx1[t1 * d + j] += 0.4 * grad_j;
                    g_ctx2[t2 * d + j] += 0.25 * grad_j;
                    g_ctx3[t3 * d + j] += 0.2 * grad_j;
                    g_ctx4[t4 * d + j] += 0.15 * grad_j;
                } else if self.use_ctx3 {
                    g_ctx1[t1 * d + j] += 0.5 * grad_j;
                    g_ctx2[t2 * d + j] += 0.3 * grad_j;
                    g_ctx3[t3 * d + j] += 0.2 * grad_j;
                } else {
                    g_ctx1[t1 * d + j] += 0.7 * grad_j;
                    g_ctx2[t2 * d + j] += 0.3 * grad_j;
                }
            }
        }

        let n = (tokens.len() - start - 1) as f32;
        for g in [
            &mut g_embed,
            &mut g_ctx1,
            &mut g_ctx2,
            &mut g_ctx3,
            &mut g_ctx4,
            &mut g_ctx5,
            &mut g_proj,
            &mut g_head,
        ] {
            for x in g.iter_mut() {
                *x /= n;
            }
        }

        opts.e.update(&mut self.embed, &g_embed, lr);
        opts.c1.update(&mut self.ctx1, &g_ctx1, lr);
        opts.c2.update(&mut self.ctx2, &g_ctx2, lr);
        if self.use_ctx3 || self.use_ctx4 || self.use_ctx5 {
            opts.c3.update(&mut self.ctx3, &g_ctx3, lr);
        }
        if self.use_ctx4 || self.use_ctx5 {
            opts.c4.update(&mut self.ctx4, &g_ctx4, lr);
        }
        if self.use_ctx5 {
            opts.c5.update(&mut self.ctx5, &g_ctx5, lr);
        }
        opts.p.update(&mut self.proj, &g_proj, lr);
        opts.h.update(&mut self.lm_head, &g_head, lr);
    }

    /// Get overall quantization metrics across all parameters
    fn quantization_report(&self) -> QuantizationMetrics {
        // Collect all f32 values
        let all_f32: Vec<f32> = [
            self.embed.to_f32(),
            self.ctx1.to_f32(),
            self.ctx2.to_f32(),
            self.ctx3.to_f32(),
            self.ctx4.to_f32(),
            self.ctx5.to_f32(),
            self.proj.to_f32(),
            self.lm_head.to_f32(),
        ]
        .concat();

        // Quantize and compare
        let quantized: Vec<f32> = all_f32
            .iter()
            .map(|&x| GF16::from_f32(x).to_f32())
            .collect();
        QuantizationMetrics::compute(&all_f32, &quantized)
    }
}

/// Mean bits-per-byte over evenly spaced chunks of `tokens`.
///
/// `None` means the evaluation could not be performed. It used to return
/// `(f32::MAX, f32::MAX)` for that case, and every downstream guard tested
/// `is_finite()` — which `f32::MAX` passes. The observed consequence was a
/// published `bpb=340282346638528859811704183484516925440.0000` followed by a
/// divide-by-zero panic. A sentinel that is indistinguishable from a
/// measurement is not a sentinel.
fn evaluate(model: &NgramModelGF16, tokens: &[usize], seq_len: usize) -> Option<f32> {
    let mut total = 0.0f32;
    let mut n = 0usize;
    for c in (0..tokens.len()).step_by(seq_len + 1) {
        let end = (c + seq_len + 1).min(tokens.len());
        // `loss_on_seq` returns the 0.0 sentinel when the chunk is shorter
        // than `start + 2`, and `start` is 5 under `use_ctx5`. Chunks of
        // length 5 and 6 therefore passed the old `< 5` guard and averaged a
        // fake perfect score into the result (#62). Require the longest
        // context this model can be configured with, plus the target token.
        if end - c < 7 {
            continue;
        }
        let loss = model.loss_on_seq(&tokens[c..end]);
        if loss.is_finite() && loss > 0.0 {
            total += loss / LN_2;
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    let bpb = total / n as f32;
    if bpb.is_finite() {
        Some(bpb)
    } else {
        None
    }
}

/// Publish one BPB reading, or say plainly that it was not published.
///
/// Every ledger write in this binary goes through here. The value guard lives
/// in `neon_writer::bpb_sample_with_algo` so no caller can bypass it; this
/// wrapper exists so the printed line is the writer's own verdict rather than
/// an unconditional success announcement on the line after `- skipping`.
fn publish_bpb(canon_name: &str, seed: u64, step: usize, bpb: f32, algo: &str) {
    let outcome = nw::bpb_sample_with_algo(canon_name, seed as i32, step as i32, bpb, None, algo);
    eprintln!("[neon] step={step} ledger={}", outcome.as_str());
}

fn cosine_lr(step: usize, max_steps: usize, base_lr: f32, warmup: usize) -> f32 {
    if step < warmup {
        return base_lr * step as f32 / warmup as f32;
    }
    let p = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
    1e-5 + (base_lr - 1e-5) * 0.5 * (1.0 + (std::f32::consts::PI * p).cos())
}

/// Value of `--name=VALUE`, or of the first of `env_keys` that is set.
fn arg_or_env(flags: &[&str], env_keys: &[&str]) -> Option<String> {
    for flag in flags {
        let prefix = format!("{flag}=");
        if let Some(v) = std::env::args().find_map(|a| {
            a.strip_prefix(prefix.as_str())
                .map(std::string::ToString::to_string)
        }) {
            return Some(v);
        }
    }
    env_keys.iter().find_map(|k| std::env::var(k).ok())
}

fn main() {
    let code = match run() {
        Ok(()) => nw::ledger_exit_code(),
        Err(e) => {
            eprintln!("[gf16] FATAL: {e}");
            1
        }
    };
    // `process::exit` skips the runtime's stdout flush.
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    std::process::exit(code);
}

fn run() -> Result<(), String> {
    let seed = std::env::args()
        .find(|a| a.starts_with("--seed="))
        .map(|a| a[7..].parse::<u64>().unwrap_or(42))
        .unwrap_or_else(|| {
            std::env::var("SEED")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(42)
        });
    let steps = std::env::args()
        .find(|a| a.starts_with("--steps="))
        .map(|a| a[8..].parse::<usize>().unwrap_or(10000))
        .unwrap_or_else(|| {
            std::env::var("STEPS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10000)
        });
    let base_lr = std::env::args()
        .find(|a| a.starts_with("--lr="))
        .map(|a| a[5..].parse::<f32>().unwrap_or(0.004))
        .unwrap_or(0.004);
    let hidden = std::env::args()
        .find(|a| a.starts_with("--hidden="))
        .map(|a| a[9..].parse::<usize>().unwrap_or(128))
        .unwrap_or(128);
    let dim = std::env::args()
        .find(|a| a.starts_with("--dim="))
        .map(|a| a[6..].parse::<usize>().unwrap_or(64))
        .unwrap_or(64);
    let activation = std::env::args()
        .find(|a| a.starts_with("--activation="))
        .map(|a| a[13..].to_string())
        .unwrap_or_else(|| "relu".to_string());
    let wd = std::env::args()
        .find(|a| a.starts_with("--wd="))
        .map(|a| a[5..].parse::<f32>().unwrap_or(0.04))
        .unwrap_or(0.04);
    let warmup = std::env::args()
        .find(|a| a.starts_with("--warmup="))
        .map(|a| a[9..].parse::<usize>().unwrap_or(500))
        .unwrap_or(500);
    let dropout = std::env::args()
        .find(|a| a.starts_with("--dropout="))
        .map(|a| a[10..].parse::<f32>().unwrap_or(0.1))
        .unwrap_or(0.1);
    let use_ctx3 = std::env::args().any(|a| a == "--ctx3");
    let use_ctx4 = std::env::args().any(|a| a == "--ctx4");
    let use_ctx5 = std::env::args().any(|a| a == "--ctx5");
    let label_smoothing = std::env::args()
        .find(|a| a.starts_with("--label-smoothing="))
        .map(|a| a[17..].parse::<f32>().unwrap_or(0.0))
        .unwrap_or(0.0);
    let patience: usize = std::env::args()
        .find(|a| a.starts_with("--patience="))
        .map(|a| a[11..].parse::<usize>().unwrap_or(500))
        .unwrap_or(500);

    let checkpoint_interval = nw::checkpoint_interval();

    let ngram = if use_ctx5 {
        "7-Gram"
    } else if use_ctx4 {
        "6-Gram"
    } else if use_ctx3 {
        "5-Gram"
    } else {
        "4-Gram"
    };
    let activation_name = if activation == "gelu" { "GELU" } else { "ReLU" };

    // canon_name: prefer env var, fall back to deterministic name
    let canon_name =
        std::env::var("CANON_NAME").unwrap_or_else(|_| format!("IGLA-GF16-{}-rng{}", ngram, seed));

    // S10: the whitelist must gate the optimizer that EXECUTES, not a string
    // parsed out of the canon name. This binary runs AdamW and nothing else, so
    // it declares that up front; a canon whose suffix disagrees is a hard error
    // here, at process start, rather than a silent `return` inside the first
    // ledger write two hours into a run.
    const EXECUTED_OPTIMIZER: &str = "adamw";
    if let Some(requested) = arg_or_env(&["--optimizer", "--opt"], &["TRIOS_OPTIMIZER"]) {
        if requested != EXECUTED_OPTIMIZER {
            return Err(format!(
                "--optimizer={requested} requested, but this binary only implements \
                 {EXECUTED_OPTIMIZER}. Refusing to run and label the result {requested}."
            ));
        }
    }
    nw::bind_executed_optimizer(&canon_name, EXECUTED_OPTIMIZER)?;

    println!("=== GF16 {} Context Model + {} ===", ngram, activation_name);
    println!("vocab={} dim={} hidden={} seq={} steps={} seed={} lr={} activation={} wd={} warmup={} dropout={} checkpoint_interval={}",
        VOCAB, dim, hidden, SEQ, steps, seed, base_lr, activation, wd, warmup, dropout, checkpoint_interval);
    println!("GF16 φ-distance: {:.6}", GF16::phi_distance());
    println!("canon_name={} optimizer={}", canon_name, EXECUTED_OPTIMIZER);

    // S1: explicit corpus paths, no default and no fallback. The old code
    // derived val by slicing the last 10% off the SAME buffer it trained on,
    // after loading that buffer from a path that did not exist.
    // `--train-data` / `--val-data` are the spellings `entrypoint.rs` already
    // passes; `--train` / `--val` are accepted as aliases.
    let train_path = arg_or_env(&["--train-data", "--train"], &["TRIOS_TRAIN_PATH"]).ok_or_else(
        || {
            "no training corpus given. Pass --train-data=PATH (or --train=PATH, or set \
             TRIOS_TRAIN_PATH). There is no default: the previous hardcoded default \
             misspelled the corpus in data/, so it never existed and every run silently \
             trained on a 52-byte placeholder."
                .to_string()
        },
    )?;
    let val_path = arg_or_env(&["--val-data", "--val"], &["TRIOS_VAL_PATH"]).ok_or_else(|| {
        "no validation corpus given. Pass --val-data=PATH (or --val=PATH, or set \
         TRIOS_VAL_PATH). There is no default: a val stream sliced out of the train \
         buffer measures memorisation, not generalisation."
            .to_string()
    })?;

    let train = load_data(&train_path)?;
    let val = load_data(&val_path)?;
    println!(
        "Corpus: train={} ({} tokens) val={} ({} tokens)",
        train_path,
        train.len(),
        val_path,
        val.len()
    );

    if let Some(reason) = val_not_held_out(&train, &val) {
        return Err(format!(
            "{reason}. Refusing to train: no BPB measured against this split is a \
             model result."
        ));
    }
    let train: &[usize] = &train;
    let val: &[usize] = &val;

    let mut model = NgramModelGF16::new(
        VOCAB,
        dim,
        hidden,
        activation.clone(),
        seed,
        dropout,
        use_ctx3,
        use_ctx4,
        use_ctx5,
        label_smoothing,
    );
    let ps = VOCAB * dim;
    let mut opts = Optimizers {
        e: AdamW::new(ps, wd),
        c1: AdamW::new(ps, wd),
        c2: AdamW::new(ps, wd),
        c3: AdamW::new(ps, wd),
        c4: AdamW::new(ps, wd),
        c5: AdamW::new(ps, wd),
        p: AdamW::new(hidden * dim, wd),
        h: AdamW::new(VOCAB * hidden, wd),
    };

    let init_bpb = evaluate(&model, val, SEQ)
        .ok_or_else(|| format!("initial evaluation over '{val_path}' produced no measurable chunk"))?;
    println!(
        "Initial val: loss={:.4} bpb={:.4}",
        init_bpb * LN_2,
        init_bpb
    );

    // Step=0 ping so the queue knows the trainer started. Guarded like every
    // other write: the old unconditional call published whatever `evaluate`
    // had returned, sentinel included.
    publish_bpb(&canon_name, seed, 0, init_bpb, EXECUTED_OPTIMIZER);

    println!();
    println!(
        "{:>6} | {:>10} | {:>10} | {:>10} | {:>12}",
        "step", "val_loss", "val_bpb", "best_bpb", "max_q_err%"
    );
    println!("{}", "-".repeat(65));

    let t0 = Instant::now();
    let mut best_bpb = init_bpb;
    let mut max_q_error = 0.0f64;
    let mut steps_without_improvement = 0usize;
    let mut best_step = 0usize;
    let dl = train.len();
    let mut results: Vec<(usize, f32, f32)> = Vec::new();

    for step in 1..=steps {
        let lr = cosine_lr(step, steps, base_lr, warmup);
        let off = (step * 97 + seed as usize) % (dl.saturating_sub(SEQ + 1));
        model.train_step(&train[off..off + SEQ + 1], lr, &mut opts);

        if step % checkpoint_interval == 0 || step == steps {
            let _ms = t0.elapsed().as_millis();
            // An unmeasurable eval invalidates the run; it is not a step to
            // skip past while the loop keeps reporting the previous best.
            let vb = evaluate(&model, val, SEQ).ok_or_else(|| {
                format!("evaluation over '{val_path}' produced no measurable chunk at step {step}")
            })?;
            let vl = vb * LN_2;
            let improved = vb < best_bpb;
            if improved {
                best_bpb = vb;
                best_step = step;
                steps_without_improvement = 0;
            } else {
                steps_without_improvement += checkpoint_interval;
            }

            // Early stopping
            if steps_without_improvement >= patience {
                println!(
                    "\n>>> Early stopping at step {} (patience={} exceeded)",
                    step, patience
                );
                publish_bpb(&canon_name, seed, step, vb, EXECUTED_OPTIMIZER);
                break;
            }

            // Measure quantization error
            let q_metrics = model.quantization_report();
            max_q_error = max_q_error.max(q_metrics.max_error_pct);

            println!(
                "{:>6} | {:>10.4} | {:>10.4} | {:>10.4} | {:>12.4}",
                step, vl, vb, best_bpb, max_q_error
            );
            results.push((step, vl, vb));

            publish_bpb(&canon_name, seed, step, vb, EXECUTED_OPTIMIZER);
        }
    }

    let total = t0.elapsed();
    let q_final = model.quantization_report();
    println!("\n=== GF16 Training Complete ===");
    println!(
        "Time: {:.1}s | BPB: {:.4} → {:.4} | Delta: {:.4} | Best step: {}",
        total.as_secs_f64(),
        init_bpb,
        best_bpb,
        best_bpb - init_bpb,
        best_step
    );
    println!("\nFinal Quantization Metrics:");
    println!("  φ-distance:    {:.6}", q_final.phi_error);
    println!("  Max error:     {:.4}%", q_final.max_error_pct);
    println!("  Avg error:     {:.4}%", q_final.avg_error_pct);
    println!("  MSE:           {:.8}", q_final.mse);
    println!("  MAE:           {:.8}", q_final.mae);

    // Record result
    let _ = fs::create_dir_all(".trinity/results");
    let exp_name = format!("gf16-{}gram-{}", ngram.to_lowercase(), activation);
    let rj = serde_json::json!({
        "experiment": exp_name,
        "train_corpus": train_path,
        "val_corpus": val_path,
        "optimizer": EXECUTED_OPTIMIZER,
        "model": format!("GF16 {} + {}", ngram, activation_name),
        "seed": seed,
        "steps": steps,
        "base_lr": base_lr,
        "hidden_size": hidden,
        "activation": activation,
        "train_tokens": train.len(),
        "val_tokens": val.len(),
        "initial_val_bpb": init_bpb,
        "final_val_bpb": best_bpb,
        "delta_bpb": best_bpb - init_bpb,
        "duration_seconds": total.as_secs_f64(),
        "quantization": {
            "phi_distance": q_final.phi_error,
            "max_error_pct": q_final.max_error_pct,
            "avg_error_pct": q_final.avg_error_pct,
            "mse": q_final.mse,
            "mae": q_final.mae,
        },
        "results": results.iter().map(|(s, l, b)| serde_json::json!({"step":*s,"loss":*l,"bpb":*b})).collect::<Vec<_>>(),
    });
    let rp = format!(
        ".trinity/results/gf16_{}gram_{}_seed{}.json",
        ngram.to_lowercase(),
        activation,
        seed
    );
    let rj_text = serde_json::to_string_pretty(&rj).map_err(|e| e.to_string())?;
    fs::File::create(&rp)
        .and_then(|mut f| f.write_all(rj_text.as_bytes()))
        .map_err(|e| format!("failed to write results '{rp}': {e}"))?;
    println!("\nResults: {}", rp);

    // Experience log
    let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
    let ep = format!(
        ".trinity/experience/trios_{}.trinity",
        chrono::Utc::now().format("%Y%m%d")
    );
    let _ = fs::create_dir_all(".trinity/experience");
    let _ = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&ep)
        .and_then(|mut f| {
            f.write_all(
                format!(
                    "[{}] TASK: GF16 {} training | seed={} | steps={} | val_bpb={:.4}->{:.4} | {:.1}s | q_err={:.2}%\n",
                    ts, ngram, seed, steps, init_bpb, best_bpb, total.as_secs_f64(), max_q_error
                )
                .as_bytes(),
            )
        });
    println!("Experience: {}", ep);

    // L-R8: stdout must end with BPB=X.XXXX for ASHA worker parsing
    println!("BPB={:.4}", best_bpb);
    Ok(())
}
