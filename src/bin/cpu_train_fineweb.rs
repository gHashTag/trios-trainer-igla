use std::fs;
use std::io::Write;
use std::time::Instant;

use trios_trainer::model_hybrid_attn::{HybridAttn, HybridAttnConfig, AttentionCache};

const LN_2: f32 = std::f32::consts::LN_2;

fn load_data(path: &str) -> Vec<usize> {
    if path.ends_with(".bin") && std::path::Path::new(path).exists() {
        return load_fineweb_bin(path);
    }
    let raw = fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to load {}: {}. Using fallback.", path, e);
        b"Hello world this is a tiny training dataset for IGLA".to_vec()
    });
    raw.into_iter().map(|b| b as usize).collect()
}

fn load_fineweb_bin(path: &str) -> Vec<usize> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| {
        panic!("open {}: {}", path, e);
    });
    let mut header = [0u8; 1024];
    file.read_exact(&mut header).expect("read header");
    let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    let num_tokens = u64::from_le_bytes([
        header[8], header[9], header[10], header[11],
        header[12], header[13], header[14], header[15],
    ]);
    eprintln!("FineWeb bin: path={} magic={} version={} num_tokens={}", path, magic, version, num_tokens);
    assert_eq!(magic, 20240520, "FineWeb magic mismatch");
    assert_eq!(version, 1, "FineWeb version mismatch");
    let mut tokens = vec![0u16; num_tokens as usize];
    let mut buf = vec![0u8; num_tokens as usize * 2];
    file.read_exact(&mut buf).expect("read tokens");
    for i in 0..tokens.len() {
        tokens[i] = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
    }
    tokens.into_iter().map(|t| t as usize).collect()
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

fn rng_next(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let t = ((*s >> 33) as f32) / (u32::MAX as f32);
    t * 2.0 - 1.0
}

struct BigramHash {
    embed: Vec<f32>,
    vocab: usize,
    dim: usize,
}

impl BigramHash {
    fn new(vocab: usize, dim: usize, seed: &mut u64) -> Self {
        let embed: Vec<f32> = (0..vocab * dim).map(|_| rng_next(seed) * 0.02).collect();
        Self { embed, vocab, dim }
    }

    fn hash(&self, curr: usize, prev: usize) -> usize {
        ((36313u32.wrapping_mul(curr as u32)) ^ (27191u32.wrapping_mul(prev as u32))) as usize
            % (self.vocab - 1)
    }

    fn forward(&self, tokens: &[usize]) -> Vec<Vec<f32>> {
        let d = self.dim;
        let mut out = Vec::with_capacity(tokens.len());
        for (i, &t) in tokens.iter().enumerate() {
            let prev = if i > 0 { tokens[i - 1] } else { 0 };
            let h = self.hash(t, prev);
            out.push(self.embed[h * d..(h + 1) * d].to_vec());
        }
        out
    }

    fn grad_step(&mut self, tokens: &[usize], grad: &[Vec<f32>], lr: f32) {
        let d = self.dim;
        for (i, &t) in tokens.iter().enumerate() {
            let prev = if i > 0 { tokens[i - 1] } else { 0 };
            let h = self.hash(t, prev);
            for (j, g) in grad[i].iter().enumerate().take(d) {
                self.embed[h * d + j] -= lr * g;
            }
        }
    }
}

struct SmearGate {
    gate: Vec<f32>,
}

impl SmearGate {
    fn new(dim: usize) -> Self {
        Self {
            gate: vec![0.0f32; dim],
        }
    }

    fn forward(&self, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(xs.len());
        for (i, x) in xs.iter().enumerate() {
            let g: Vec<f32> = self
                .gate
                .iter()
                .map(|&g| 1.0 / (1.0 + (-g).exp()))
                .collect();
            if i == 0 {
                out.push(
                    x.iter()
                        .zip(g.iter())
                        .map(|(xi, gi)| xi * (1.0 - gi))
                        .collect(),
                );
            } else {
                out.push(
                    x.iter()
                        .zip(g.iter())
                        .zip(xs[i - 1].iter())
                        .map(|((xi, gi), pi)| xi * (1.0 - gi) + pi * gi)
                        .collect(),
                );
            }
        }
        out
    }

    fn grad_step(&mut self, grad: &[Vec<f32>], lr: f32) {
        for (i, g) in self.gate.iter_mut().enumerate() {
            let mut total = 0.0f32;
            for g_vec in grad {
                total += g_vec[i];
            }
            *g -= lr * total;
        }
    }
}

struct FFNLayer {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    d_model: usize,
    d_ff: usize,
}

impl FFNLayer {
    fn new(d_model: usize, d_ff: usize, seed: &mut u64) -> Self {
        let std = (2.0 / (d_model + d_ff) as f32).sqrt();
        Self {
            w1: (0..d_ff * d_model).map(|_| rng_next(seed) * std).collect(),
            b1: vec![0.0; d_ff],
            w2: (0..d_model * d_ff).map(|_| rng_next(seed) * std).collect(),
            b2: vec![0.0; d_model],
            d_model,
            d_ff,
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let hidden: Vec<f32> = (0..self.d_ff)
            .map(|r| {
                let row = &self.w1[r * self.d_model..(r + 1) * self.d_model];
                let sum: f32 = row.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum();
                (sum + self.b1[r]).max(0.0)
            })
            .collect();
        (0..self.d_model)
            .map(|r| {
                let row = &self.w2[r * self.d_ff..(r + 1) * self.d_ff];
                let sum: f32 = row.iter().zip(hidden.iter()).map(|(&w, &h)| w * h).sum();
                sum + self.b2[r]
            })
            .collect()
    }

    #[allow(clippy::needless_range_loop)]
    fn backward(&self, x: &[f32], grad_out: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let d = self.d_model;
        let ff = self.d_ff;
        let mut hidden = vec![0.0f32; ff];
        for r in 0..ff {
            let row = &self.w1[r * d..(r + 1) * d];
            hidden[r] = row.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum();
        }
        let activated: Vec<f32> = hidden.iter().map(|&h| h.max(0.0)).collect();
        let relu_mask: Vec<f32> = hidden
            .iter()
            .map(|&h| if h > 0.0 { 1.0 } else { 0.0 })
            .collect();

        let mut d_w2 = vec![0.0f32; d * ff];
        let mut d_b2 = vec![0.0f32; d];
        let mut d_hidden = vec![0.0f32; ff];

        for r in 0..d {
            for k in 0..ff {
                d_w2[r * ff + k] += grad_out[r] * activated[k];
                d_hidden[k] += grad_out[r] * self.w2[r * ff + k];
            }
            d_b2[r] += grad_out[r];
        }

        for k in 0..ff {
            d_hidden[k] *= relu_mask[k];
        }

        let mut d_w1 = vec![0.0f32; ff * d];
        let mut d_b1 = vec![0.0f32; ff];
        let mut d_input = vec![0.0f32; d];

        for k in 0..ff {
            for j in 0..d {
                d_w1[k * d + j] += d_hidden[k] * x[j];
                d_input[j] += d_hidden[k] * self.w1[k * d + j];
            }
            d_b1[k] += d_hidden[k];
        }

        (d_w1, d_b1, d_w2, d_b2)
    }
}

struct AdamW {
    m: Vec<f32>,
    v: Vec<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    wd: f32,
    step: usize,
}

impl AdamW {
    fn new(size: usize, lr: f32) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            lr,
            beta1: 0.9,
            beta2: 0.999,
            wd: 0.01,
            step: 0,
        }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.step += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);
        for i in 0..params.len() {
            let g = grads[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * (m_hat / (v_hat.sqrt() + 1e-8) + self.wd * params[i]);
        }
    }
}

struct CpuModel {
    embed: Vec<f32>,
    lm_head: Vec<f32>,
    bigram: BigramHash,
    smear: SmearGate,
    ffn_layers: Vec<FFNLayer>,
    attn: Option<HybridAttn>,
    attn_cache: Option<AttentionCache>,
    bigram_scale: f32,
    vocab: usize,
    dim: usize,
}

fn flatten_2d(xs: &[Vec<f32>]) -> Vec<f32> {
    let mut out = Vec::with_capacity(xs.len() * xs[0].len());
    for x in xs {
        out.extend_from_slice(x);
    }
    out
}

fn unflatten_2d(flat: &[f32], seq_len: usize, d: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        out.push(flat[i * d..(i + 1) * d].to_vec());
    }
    out
}

impl CpuModel {
    fn new(vocab: usize, dim: usize, seed: u64) -> Self {
        let mut s = seed;
        let embed: Vec<f32> = (0..vocab * dim).map(|_| rng_next(&mut s) * 0.02).collect();
        let lm_head: Vec<f32> = (0..vocab * dim).map(|_| rng_next(&mut s) * 0.02).collect();
        let bigram = BigramHash::new(vocab, dim, &mut s);
        let smear = SmearGate::new(dim);

        let use_attn = std::env::args().any(|a| a == "--attn");

        let ffn_layers = if std::env::args().any(|a| a == "--ffn") {
            let mut layers = Vec::new();
            // Read --ffn-layers argument, default to 2 if not present
            let ffn_layers_str = arg_or("ffn-layers", "2");
            let n_layers = ffn_layers_str.parse::<usize>().unwrap_or(2);
            for _ in 0..n_layers {
                layers.push(FFNLayer::new(dim, dim * 4, &mut s));
            }
            layers
        } else {
            Vec::new()
        };

        let attn = if use_attn {
            let mut cfg = HybridAttnConfig::default();
            cfg.d_model = dim;
            cfg.num_heads = 4;
            cfg.seq_len = 64; // max seq_len for cache
            cfg.num_attn_layers = 2;
            Some(HybridAttn::with_config(cfg).expect("valid attn config"))
        } else {
            None
        };

        Self {
            embed,
            lm_head,
            bigram,
            smear,
            ffn_layers,
            attn,
            attn_cache: None,
            bigram_scale: 0.1,
            vocab,
            dim,
        }
    }

    #[allow(dead_code)]
    fn forward_logits(&self, tokens: &[usize]) -> Vec<Vec<f32>> {
        let d = self.dim;
        let v = self.vocab;

        let tok_emb: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&id| self.embed[(id % v) * d..((id % v) + 1) * d].to_vec())
            .collect();

        let bigram_emb = self.bigram.forward(tokens);
        let mut xs: Vec<Vec<f32>> = tok_emb
            .iter()
            .zip(bigram_emb.iter())
            .map(|(t, b)| {
                t.iter()
                    .zip(b.iter())
                    .map(|(ti, bi)| ti + bi * self.bigram_scale)
                    .collect()
            })
            .collect();

        xs = self.smear.forward(&xs);

        // Apply attention if present
        let xs = if let Some(ref attn) = self.attn {
            let flat = flatten_2d(&xs);
            let (attn_out, _) = attn.forward_with_cache(&flat, tokens.len()).expect("attn forward");
            unflatten_2d(&attn_out, tokens.len(), d)
        } else {
            xs
        };

        let mut logits = Vec::with_capacity(tokens.len());
        for x in &xs {
            let mut row = vec![0.0f32; v];
            for (vi, r) in row.iter_mut().enumerate() {
                for (j, xj) in x.iter().enumerate() {
                    *r += self.lm_head[vi * d + j] * xj;
                }
            }
            logits.push(row);
        }
        logits
    }

    fn loss_and_grad(&self, tokens: &[usize]) -> (f32, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let d = self.dim;
        let v = self.vocab;
        let n = tokens.len();

        let tok_emb: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&id| self.embed[(id % v) * d..((id % v) + 1) * d].to_vec())
            .collect();
        let bigram_emb = self.bigram.forward(tokens);
        let xs: Vec<Vec<f32>> = tok_emb
            .iter()
            .zip(bigram_emb.iter())
            .map(|(t, b)| {
                t.iter()
                    .zip(b.iter())
                    .map(|(ti, bi)| ti + bi * self.bigram_scale)
                    .collect()
            })
            .collect();
        let xs_smeared = self.smear.forward(&xs);

        let xs_after_attn: Vec<Vec<f32>> = if let Some(ref attn) = self.attn {
            let flat = flatten_2d(&xs_smeared);
            let (out, _) = attn.forward_with_cache(&flat, n).expect("attn forward");
            unflatten_2d(&out, n, d)
        } else {
            xs_smeared.clone()
        };

        let xs_final: Vec<Vec<f32>> = if !self.ffn_layers.is_empty() {
            let mut current = xs_after_attn;
            for ffn_layer in &self.ffn_layers {
                let normed: Vec<Vec<f32>> = current
                    .iter()
                    .map(|x| {
                        let mean = x.iter().sum::<f32>() / d as f32;
                        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
                        let std = (var + 1e-5).sqrt();
                        x.iter().map(|v| (v - mean) / std).collect()
                    })
                    .collect();
                let ffn_out: Vec<Vec<f32>> = normed.iter().map(|x| ffn_layer.forward(x)).collect();
                current = (0..n)
                    .map(|i| {
                        current[i]
                            .iter()
                            .zip(ffn_out[i].iter())
                            .map(|(&a, &b)| a + b)
                            .collect()
                    })
                    .collect();
            }
            current
        } else {
            xs_after_attn
        };

        let mut total_loss = 0.0f32;
        let mut d_logits = vec![vec![0.0f32; v]; n - 1];

        for i in 0..n - 1 {
            let x = &xs_final[i];
            let target = tokens[i + 1] % v;
            let mut logits = vec![0.0f32; v];
            for (vi, l) in logits.iter_mut().enumerate() {
                for (j, xj) in x.iter().enumerate() {
                    *l += self.lm_head[vi * d + j] * xj;
                }
            }
            softmax(&mut logits);
            let p_target = logits[target].max(1e-10);
            total_loss -= p_target.ln();
            for (vi, dl) in d_logits[i].iter_mut().enumerate() {
                *dl = logits[vi] - if vi == target { 1.0 } else { 0.0 };
            }
        }

        let loss = total_loss / (n - 1) as f32;

        let mut d_hidden = vec![vec![0.0f32; d]; n];
        for i in 0..n - 1 {
            for (vi, dl) in d_logits[i].iter().enumerate() {
                for (j, dh) in d_hidden[i].iter_mut().enumerate() {
                    *dh += dl * self.lm_head[vi * d + j];
                }
            }
        }

        (loss, d_logits, d_hidden)
    }

    fn train_step(
        &mut self,
        tokens: &[usize],
        opt_embed: &mut AdamW,
        opt_head: &mut AdamW,
        lr: f32,
    ) -> f32 {
        let d = self.dim;
        let v = self.vocab;
        let n = tokens.len();

        let (loss, d_logits, _d_hidden) = self.loss_and_grad(tokens);

        // Recompute forward activations
        let tok_emb: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&id| self.embed[(id % v) * d..((id % v) + 1) * d].to_vec())
            .collect();
        let bigram_emb = self.bigram.forward(tokens);
        let xs: Vec<Vec<f32>> = tok_emb
            .iter()
            .zip(bigram_emb.iter())
            .map(|(t, b)| {
                t.iter()
                    .zip(b.iter())
                    .map(|(ti, bi)| ti + bi * self.bigram_scale)
                    .collect()
            })
            .collect();
        let xs_smeared = self.smear.forward(&xs);

        // Attention forward with cache
        let (xs_after_attn, attn_cache) = if let Some(ref attn) = self.attn {
            let flat = flatten_2d(&xs_smeared);
            let (out, cache) = attn.forward_with_cache(&flat, n).expect("attn forward");
            (unflatten_2d(&out, n, d), Some(cache))
        } else {
            (xs_smeared.clone(), None)
        };

        // d_from_logits[i][j] = sum over vocab of d_logits[i][v] * lm_head[v][j]
        // This is the gradient flowing back from the loss through the LM head
        let mut d_from_logits = vec![vec![0.0f32; d]; n];
        for (i, dl_row) in d_logits.iter().enumerate() {
            for (vi, &dl) in dl_row.iter().enumerate() {
                for (j, df) in d_from_logits[i].iter_mut().enumerate() {
                    *df += dl * self.lm_head[vi * d + j];
                }
            }
        }

        // Compute d_lm_head
        let xs_for_head = if !self.ffn_layers.is_empty() {
            // Recompute FFN forward on xs_after_attn
            let mut current = xs_after_attn.clone();
            for ffn_layer in &self.ffn_layers {
                let normed: Vec<Vec<f32>> = current
                    .iter()
                    .map(|x| {
                        let mean = x.iter().sum::<f32>() / d as f32;
                        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
                        let std = (var + 1e-5).sqrt();
                        x.iter().map(|v| (v - mean) / std).collect()
                    })
                    .collect();
                let ffn_out: Vec<Vec<f32>> = normed.iter().map(|x| ffn_layer.forward(x)).collect();
                current = (0..n)
                    .map(|i| {
                        current[i]
                            .iter()
                            .zip(ffn_out[i].iter())
                            .map(|(&a, &b)| a + b)
                            .collect()
                    })
                    .collect();
            }
            current
        } else {
            xs_after_attn.clone()
        };

        let mut d_lm_head = vec![0.0f32; v * d];
        for i in 0..n - 1 {
            for (vi, &dl) in d_logits[i].iter().enumerate() {
                for (j, xf) in xs_for_head[i].iter().enumerate() {
                    d_lm_head[vi * d + j] += dl * xf;
                }
            }
        }

        // d_from_logits[i][j] = sum over vocab of d_logits[i][v] * lm_head[v][j]
        let mut d_from_logits = vec![vec![0.0f32; d]; n];
        for (i, dl_row) in d_logits.iter().enumerate() {
            for (vi, &dl) in dl_row.iter().enumerate() {
                for (j, df) in d_from_logits[i].iter_mut().enumerate() {
                    *df += dl * self.lm_head[vi * d + j];
                }
            }
        }

        // Backward through FFN layers (applied to xs_after_attn)
        let mut d_after_ffn = d_from_logits.clone();
        if !self.ffn_layers.is_empty() {
            // Recompute forward activations for FFN
            let mut current = xs_after_attn.clone();
            let mut normed_activations: Vec<Vec<Vec<f32>>> = Vec::new();

            for ffn_layer in &self.ffn_layers {
                let normed: Vec<Vec<f32>> = current
                    .iter()
                    .map(|x| {
                        let mean = x.iter().sum::<f32>() / d as f32;
                        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
                        let std = (var + 1e-5).sqrt();
                        x.iter().map(|v| (v - mean) / std).collect()
                    })
                    .collect();
                let ffn_out: Vec<Vec<f32>> = normed.iter().map(|x| ffn_layer.forward(x)).collect();
                normed_activations.push(normed);
                current = (0..n)
                    .map(|i| {
                        current[i]
                            .iter()
                            .zip(ffn_out[i].iter())
                            .map(|(&a, &b)| a + b)
                            .collect()
                    })
                    .collect();
            }

            // Backward through FFN layers in reverse
            let mut current_grad = d_from_logits.clone();
            let mut all_layer_grads = Vec::with_capacity(self.ffn_layers.len());

            for (layer_idx, ffn_layer) in self.ffn_layers.iter().rev().enumerate() {
                let normed = &normed_activations[self.ffn_layers.len() - 1 - layer_idx];
                let ffn_input = if layer_idx == self.ffn_layers.len() - 1 {
                    &xs_after_attn
                } else {
                    // For intermediate layers, we'd need to recompute; for now assume single FFN layer
                    &xs_after_attn
                };

                let mut layer_grads = Vec::with_capacity(n);
                for i in 0..n {
                    let (dw1, db1, dw2, db2) = ffn_layer.backward(&normed[i], &current_grad[i]);
                    layer_grads.push((dw1, db1, dw2, db2));
                }
                all_layer_grads.push(layer_grads);

                // Backward through LayerNorm
                let mut d_ln = vec![vec![0.0f32; d]; n];
                for i in 0..n {
                    let x = &ffn_input[i];
                    let mean = x.iter().sum::<f32>() / d as f32;
                    let var = x.iter().map(|vv| (vv - mean).powi(2)).sum::<f32>() / d as f32;
                    let std = (var + 1e-5).sqrt();
                    let nn = d as f32;
                    let dx_sum: f32 = current_grad[i].iter().sum();
                    let dx_xm_sum: f32 = current_grad[i]
                        .iter()
                        .zip(x.iter())
                        .map(|(&g, &xi)| g * (xi - mean))
                        .sum();
                    let inv_n_std = 1.0 / (nn * std);
                    let inv_var_eps = 1.0 / (var + 1e-5);

                    for j in 0..d {
                        let xm = x[j] - mean;
                        d_ln[i][j] = inv_n_std
                            * (nn * current_grad[i][j]
                                - dx_sum
                                - xm * inv_var_eps * dx_xm_sum);
                    }
                }

                current_grad = d_ln;
            }

            // Residual connection: gradient flows directly too
            d_after_ffn = current_grad;
            for i in 0..n {
                for j in 0..d {
                    d_after_ffn[i][j] += d_from_logits[i][j];
                }
            }

            // Apply FFN parameter updates
            let n_layers = self.ffn_layers.len();
            for (layer_idx, layer_grads) in all_layer_grads.into_iter().enumerate() {
                let layer_mut = &mut self.ffn_layers[n_layers - 1 - layer_idx];
                for (dw1, db1, dw2, db2) in layer_grads.into_iter() {
                    for (k, &g) in dw1.iter().enumerate() {
                        layer_mut.w1[k] -= lr * g;
                    }
                    for (k, &g) in db1.iter().enumerate() {
                        layer_mut.b1[k] -= lr * g;
                    }
                    for (k, &g) in dw2.iter().enumerate() {
                        layer_mut.w2[k] -= lr * g;
                    }
                    for (k, &g) in db2.iter().enumerate() {
                        layer_mut.b2[k] -= lr * g;
                    }
                }
            }
        }

        // Attention backward
        let d_to_embed = if let (Some(ref mut attn), Some(cache)) = (&mut self.attn, attn_cache) {
            let flat_grad = flatten_2d(&d_after_ffn);
            let grads = attn.backward(&cache, &flat_grad);
            let dd = d * d;
            for i in 0..dd { attn.wq_mut()[i] -= lr * grads.gwq[i]; }
            for i in 0..dd { attn.wk_mut()[i] -= lr * grads.gwk[i]; }
            for i in 0..dd { attn.wv_mut()[i] -= lr * grads.gwv[i]; }
            for i in 0..dd { attn.wo_mut()[i] -= lr * grads.gwo[i]; }
            if !grads.gwq2.is_empty() {
                for i in 0..dd { attn.wq2_mut()[i] -= lr * grads.gwq2[i]; }
                for i in 0..dd { attn.wk2_mut()[i] -= lr * grads.gwk2[i]; }
                for i in 0..dd { attn.wv2_mut()[i] -= lr * grads.gwv2[i]; }
                for i in 0..dd { attn.wo2_mut()[i] -= lr * grads.gwo2[i]; }
            }
            unflatten_2d(&grads.d_input, n, d)
        } else {
            d_after_ffn
        };

        // Update embeddings
        let mut d_embed = vec![0.0f32; v * d];
        for (i, &tid) in tokens.iter().enumerate() {
            let id = tid % v;
            let gi = i.min(n - 2);
            for (j, &dh) in d_to_embed[gi].iter().enumerate().take(d) {
                d_embed[id * d + j] += dh;
            }
        }

        opt_head.step(&mut self.lm_head, &d_lm_head);
        opt_embed.step(&mut self.embed, &d_embed);

        self.bigram.grad_step(tokens, &d_to_embed, lr);
        self.smear.grad_step(&d_to_embed, lr);

        loss
    }

    fn eval_bpb(&self, tokens: &[usize], seq_len: usize) -> f32 {
        let max_eval = 5000.min(tokens.len());
        let eval_tokens = &tokens[..max_eval];
        let mut total_bpb = 0.0f32;
        let mut n = 0usize;
        for c in (0..eval_tokens.len()).step_by(seq_len + 1) {
            let end = (c + seq_len + 1).min(eval_tokens.len());
            if end - c < 3 {
                continue;
            }
            let seq = &eval_tokens[c..end];
            let (loss, _, _) = self.loss_and_grad(seq);
            if loss.is_finite() {
                total_bpb += loss / LN_2;
                n += 1;
            }
        }
        if n == 0 {
            return f32::MAX;
        }
        total_bpb / n as f32
    }
}

fn main() {
    let format_type = std::env::var("TRIOS_FORMAT_TYPE").ok();
    let seed = arg_or("seed", "42").parse::<u64>().unwrap_or(42);
    let steps = arg_or("steps", "3000").parse::<usize>().unwrap_or(3000);
    let lr = arg_or("lr", "0.003").parse::<f32>().unwrap_or(0.003);
    let vocab: usize = arg_or("vocab", "128").parse().unwrap_or(128);
    let dim: usize = arg_or("dim", "96").parse().unwrap_or(96);
    let seq: usize = arg_or("seq", "32").parse().unwrap_or(32);

    // For Phase E.GF v2, use format type in result filename
    let default_format = "f32".to_string();
    let format_suffix = format_type.as_ref().unwrap_or(&default_format);

    let train_path = arg_or("train", "data/tinyshakespeare.txt");
    let val_path = arg_or("val", "");
    let raw_tokens = load_data(&train_path);
    let tokens: Vec<usize> = raw_tokens.iter().map(|&t| t % vocab).collect();
    
    let val_tokens: Vec<usize> = if !val_path.is_empty() {
        load_data(&val_path).iter().map(|&t| t % vocab).collect()
    } else {
        let train_end = (tokens.len() as f64 * 0.9) as usize;
        tokens[train_end..].to_vec()
    };

    println!("=== trios CPU Training (Analytical Backprop) ===");
    println!(
        "vocab={} dim={} seq={} steps={} seed={} lr={}",
        vocab, dim, seq, steps, seed, lr
    );

    let train_tokens: Vec<usize>;
    let val_tokens_ref: &[usize];
    if !val_path.is_empty() {
        train_tokens = tokens;
        val_tokens_ref = &val_tokens;
    } else {
        let train_end = (tokens.len() as f64 * 0.9) as usize;
        train_tokens = tokens[..train_end].to_vec();
        val_tokens_ref = &tokens[train_end..];
    }
    println!(
        "Dataset: {} train / {} val tokens",
        train_tokens.len(),
        val_tokens_ref.len()
    );

    let mut model = CpuModel::new(vocab, dim, seed);
    let mut opt_embed = AdamW::new(vocab * dim, lr);
    let mut opt_head = AdamW::new(vocab * dim, lr);

    let init_bpb = model.eval_bpb(&val_tokens, seq);
    println!("Initial val BPB: {:.4}", init_bpb);
    println!();
    println!(
        "{:>6} | {:>10} | {:>10} | {:>10} | {:>8}",
        "step", "train_loss", "val_bpb", "best_bpb", "ms"
    );
    println!("{}", "-".repeat(60));

    let t0 = Instant::now();
    let mut best_bpb = init_bpb;
    let data_len = train_tokens.len();
    let mut rng_state = seed;

    for step in 1..=steps {
        let progress = step as f32 / steps as f32;
        let warmup = 0.05;
        let current_lr = if progress < warmup {
            lr * progress / warmup
        } else {
            let decay_progress = (progress - warmup) / (1.0 - warmup);
            lr * 0.5 * (1.0 + (std::f32::consts::PI * decay_progress).cos())
        };

        let offset = {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng_state as usize) % (data_len.saturating_sub(seq + 1))
        };
        let batch = &train_tokens[offset..offset + seq + 1];
        let train_loss = model.train_step(batch, &mut opt_embed, &mut opt_head, current_lr);

        if step % 500 == 0 || step == steps {
            let ms = t0.elapsed().as_millis();
            let val_bpb = model.eval_bpb(val_tokens_ref, seq);
            if val_bpb < best_bpb && val_bpb.is_finite() {
                best_bpb = val_bpb;
            }
            println!(
                "{:>6} | {:>10.4} | {:>10.4} | {:>10.4} | {:>6}ms",
                step, train_loss, val_bpb, best_bpb, ms
            );
        }
    }

    let total = t0.elapsed();
    println!();
    println!("=== Training Complete ===");
    println!(
        "Time: {:.1}s | Init BPB: {:.4} | Best BPB: {:.4} | Delta: {:.4}",
        total.as_secs_f64(),
        init_bpb,
        best_bpb,
        init_bpb - best_bpb
    );

    let _ = fs::create_dir_all(".trinity/results");
    let result_json = serde_json::json!({
        "experiment": "cpu-backprop-scalable",
        "model": "embed+bigram+smear+lm_head",
        "seed": seed,
        "vocab_size": vocab,
        "dim": dim,
        "seq_len": seq,
        "steps": steps,
        "lr": lr,
        "initial_bpb": init_bpb,
        "final_bpb": best_bpb,
        "delta_bpb": init_bpb - best_bpb,
        "duration_seconds": total.as_secs_f64(),
    });

    let rpath = format!(
        ".trinity/results/cpu_train_{}_seed{}.json",
        format_suffix, seed
    );
    fs::File::create(&rpath)
        .unwrap()
        .write_all(
            serde_json::to_string_pretty(&result_json)
                .unwrap()
                .as_bytes(),
        )
        .unwrap();
    println!("Results: {}", rpath);
}

fn arg_or(name: &str, default: &str) -> String {
    let prefix = format!("--{}=", name);
    std::env::args()
        .find(|a| a.starts_with(&prefix))
        .map(|a| a[prefix.len()..].to_string())
        .unwrap_or_else(|| default.to_string())
}
