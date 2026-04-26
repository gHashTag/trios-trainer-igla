#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use std::fs;
use std::io::Write;
use std::time::Instant;

use trios_trainer::optimizer::MuonOptimizer;

const VOCAB: usize = 128;
const DIM: usize = 64;
const SEQ: usize = 64;
const LN_2: f32 = std::f32::consts::LN_2;

fn gelu(x: f32) -> f32 {
    let x3 = x * x * x;
    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
    0.5 * x * (1.0 + tanh_arg.tanh())
}

fn activate(x: f32, name: &str) -> f32 {
    match name {
        "gelu" => gelu(x),
        "relu2" => { let r = x.max(0.0); r * r },
        _ => x.max(0.0),
    }
}

fn activate_grad(x: f32, name: &str) -> f32 {
    match name {
        "gelu" => gelu(x),
        "relu2" => { let r = x.max(0.0); 2.0 * r },
        _ => if x > 0.0 { 1.0 } else { 0.0 },
    }
}

fn load_data(path: &str) -> Vec<usize> {
    let raw = fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to load {}: {}. Using fallback.", path, e);
        b"Hello world this is a tiny training dataset for IGLA".to_vec()
    });
    raw.into_iter().map(|b| (b as usize) % VOCAB).collect()
}

fn softmax(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() { *x = (*x - max).exp(); sum += *x; }
    for x in v.iter_mut() { *x /= sum; }
}

fn layer_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = (var + eps).sqrt();
    x.iter().map(|v| (v - mean) / std).collect()
}

struct LocalAdamW {
    m: Vec<f32>, v: Vec<f32>, step: usize,
    beta1: f32, beta2: f32, eps: f32, wd: f32,
}

impl LocalAdamW {
    fn new(size: usize, wd: f32) -> Self {
        let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
        Self { m: vec![0.0; size], v: vec![0.0; size], step: 0,
            beta1: 1.0 / phi as f32, beta2: 0.999, eps: 1e-8, wd }
    }
    fn update(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        self.step += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);
        for i in 0..params.len() {
            params[i] -= self.wd * lr * params[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            params[i] -= lr * (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + self.eps);
        }
    }
}

trait Optimizer {
	fn update(&mut self, params: &mut [f32], grads: &[f32], lr: f32);
}
impl Optimizer for LocalAdamW {
	fn update(&mut self, params: &mut [f32], grads: &[f32], lr: f32) { LocalAdamW::update(self, params, grads, lr) }
}
impl Optimizer for MuonOptimizer {
	fn update(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
		self.step(params, &grads.iter().map(|&g| g * lr).collect::<Vec<_>>());
	}
}

struct NgramModel {
    embed: Vec<f32>,
    ctx: Vec<Vec<f32>>,
    ctx_weights: Vec<f32>,
    proj: Vec<f32>,
    lm_head: Vec<f32>,
    attn_query: Vec<f32>,
    attn_key: Vec<f32>,
    attn_value: Vec<f32>,
    vocab: usize,
    dim: usize,
    hidden: usize,
    activation: String,
    use_attention: bool,
    attn_dim: usize,
}

impl NgramModel {
    fn new(vocab: usize, dim: usize, hidden: usize, activation: String, seed: u64, num_ctx: usize, use_attention: bool) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let lim = (6.0f32 / (3 * dim) as f32).sqrt();
        let lim_h = (6.0f32 / (dim + hidden) as f32).sqrt();
        let lim_o = (6.0f32 / (hidden + dim) as f32).sqrt();

        let ctx = (0..num_ctx).map(|_| {
            (0..vocab * dim).map(|_| rng() * lim).collect()
        }).collect();

        let base_weights: Vec<f32> = vec![0.7, 0.3, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06];
        let ctx_weights: Vec<f32> = base_weights.iter().take(num_ctx).cloned().collect();

        let attn_dim = hidden / 4;

        Self {
            embed: (0..vocab * dim).map(|_| rng() * lim).collect(),
            ctx,
            ctx_weights,
            proj: (0..hidden * dim).map(|_| rng() * lim_h).collect(),
            lm_head: (0..vocab * hidden).map(|_| rng() * lim_o).collect(),
            attn_query: if use_attention { (0..attn_dim).map(|_| rng() * 0.1).collect() } else { vec![] },
            attn_key: if use_attention { (0..hidden * attn_dim).map(|_| rng() * 0.1).collect() } else { vec![] },
            attn_value: if use_attention { (0..hidden * attn_dim).map(|_| rng() * 0.1).collect() } else { vec![] },
            vocab, dim, hidden, activation, use_attention, attn_dim,
        }
    }

    fn compute_hidden(&self, tokens_context: &[usize]) -> Vec<f32> {
        let d = self.dim;
        let h = self.hidden;
        let v = self.vocab;
        let t0 = tokens_context.last().unwrap().min(&(v - 1)).to_owned();

        let e0 = &self.embed[t0 * d..(t0 + 1) * d];
        let mut combined = e0.to_vec();

        for (ci, cw) in self.ctx_weights.iter().enumerate() {
            let ctx_idx = tokens_context.len() - 2 - ci;
            if ctx_idx == 0 && ci > 0 { break; }
            let t = tokens_context[ctx_idx].min(v - 1);
            let cv = &self.ctx[ci][t * d..(t + 1) * d];
            for j in 0..d { combined[j] += cv[j] * cw; }
        }

        let ln = layer_norm(&combined, 1e-5);

        let mut hidden = vec![0.0f32; h];
        for hi in 0..h {
            let w = &self.proj[hi * d..(hi + 1) * d];
            for (j, l) in ln.iter().enumerate() { hidden[hi] += w[j] * l; }
            hidden[hi] = activate(hidden[hi], &self.activation);
        }
        hidden
    }

    fn attention_pool(&self, hidden_states: &[Vec<f32>]) -> Vec<f32> {
        let h = self.hidden;
        let ad = self.attn_dim;
        let n = hidden_states.len();

        let mut keys = vec![vec![0.0f32; ad]; n];
        let mut values = vec![vec![0.0f32; ad]; n];
        for (pos, hid) in hidden_states.iter().enumerate() {
            for j in 0..ad {
                for k in 0..h {
                    keys[pos][j] += self.attn_key[j * h + k] * hid[k];
                    values[pos][j] += self.attn_value[j * h + k] * hid[k];
                }
            }
        }

        let mut scores = vec![0.0f32; n];
        for pos in 0..n {
            for j in 0..ad {
                scores[pos] += self.attn_query[j] * keys[pos][j];
            }
            scores[pos] /= (ad as f32).sqrt();
        }
        softmax(&mut scores);

        let mut pooled = vec![0.0f32; ad];
        for pos in 0..n {
            for j in 0..ad {
                pooled[j] += scores[pos] * values[pos][j];
            }
        }
        pooled
    }

    fn predict(&self, hidden_seq: &[Vec<f32>], current_hidden: &[f32]) -> Vec<f32> {
        let v = self.vocab;
        let h = self.hidden;

        let final_hidden = if self.use_attention && hidden_seq.len() >= 3 {
            let attn_vec = self.attention_pool(hidden_seq);
            let mut combined = current_hidden.to_vec();
            let ad = self.attn_dim;
            for j in 0..ad.min(h) {
                combined[j % h] += attn_vec[j];
            }
            combined
        } else {
            current_hidden.to_vec()
        };

        let mut logits = vec![0.0f32; v];
        for (vi, logit) in logits.iter_mut().enumerate() {
            let w = &self.lm_head[vi * h..(vi + 1) * h];
            for (hi, hn) in final_hidden.iter().enumerate() { *logit += w[hi] * hn; }
        }
        logits
    }

    fn loss_on_seq(&self, tokens: &[usize]) -> f32 {
        let ngram = self.ctx.len() + 2;
        if tokens.len() < ngram + 1 { return 0.0; }
        let v = self.vocab;
        let h = self.hidden;
        let count = tokens.len() - ngram;
        let mut total = 0.0f32;

        if self.use_attention {
            let mut hidden_seq: Vec<Vec<f32>> = Vec::with_capacity(count);
            for i in 0..count {
                let context = &tokens[i..i + ngram];
                hidden_seq.push(self.compute_hidden(context));
            }

            for i in 0..count {
                let start = if i >= ngram { i - ngram + 1 } else { 0 };
                let window = &hidden_seq[start..=i];
                let logits = self.predict(window, &hidden_seq[i]);
                let target = tokens[i + ngram].min(v - 1);
                let mut logits = logits;
                softmax(&mut logits);
                total -= logits[target].max(1e-10).ln();
            }
        } else {
            for i in 0..count {
                let context = &tokens[i..i + ngram];
                let hidden = self.compute_hidden(context);
                let target = tokens[i + ngram].min(v - 1);
                let mut logits = vec![0.0f32; v];
                for (vi, logit) in logits.iter_mut().enumerate() {
                    let w = &self.lm_head[vi * h..(vi + 1) * h];
                    for (hi, hn) in hidden.iter().enumerate() { *logit += w[hi] * hn; }
                }
                softmax(&mut logits);
                total -= logits[target].max(1e-10).ln();
            }
        }
        total / count as f32
    }

    #[allow(clippy::needless_range_loop)]
    fn train_step(&mut self, tokens: &[usize], lr: f32,
        opt_embed: &mut dyn Optimizer, opt_ctx: &mut [Box<dyn Optimizer>], opt_proj: &mut dyn Optimizer, opt_head: &mut dyn Optimizer,
        opt_aq: &mut dyn Optimizer, opt_ak: &mut dyn Optimizer, opt_av: &mut dyn Optimizer) {
        let ngram = self.ctx.len() + 2;
        if tokens.len() < ngram + 1 { return; }
        let v = self.vocab;
        let d = self.dim;
        let h = self.hidden;
        let count = tokens.len() - ngram;
        let ad = self.attn_dim;

        let mut g_embed = vec![0.0f32; v * d];
        let num_ctx = self.ctx.len();
        let mut g_ctx: Vec<Vec<f32>> = (0..num_ctx).map(|_| vec![0.0f32; v * d]).collect();
        let mut g_proj = vec![0.0f32; h * d];
        let mut g_head = vec![0.0f32; v * h];
        let mut g_aq = if self.use_attention { vec![0.0f32; ad] } else { vec![] };
        let mut g_ak = if self.use_attention { vec![0.0f32; ad * h] } else { vec![] };
        let mut g_av = if self.use_attention { vec![0.0f32; ad * h] } else { vec![] };

        let mut all_hidden: Vec<Vec<f32>> = Vec::with_capacity(count);
        let mut all_ln: Vec<Vec<f32>> = Vec::with_capacity(count);
        let mut all_contexts: Vec<Vec<usize>> = Vec::with_capacity(count);
        let mut all_pre_act: Vec<Vec<f32>> = Vec::with_capacity(count);

        for i in 0..count {
            let context: Vec<usize> = tokens[i..i + ngram].to_vec();
            let t0 = context[ngram - 1].min(v - 1);
            let e0 = &self.embed[t0 * d..(t0 + 1) * d];
            let mut combined = e0.to_vec();
            for (ci, cw) in self.ctx_weights.iter().enumerate() {
                let ctx_idx = ngram - 2 - ci;
                let t = context[ctx_idx].min(v - 1);
                let cv = &self.ctx[ci][t * d..(t + 1) * d];
                for j in 0..d { combined[j] += cv[j] * cw; }
            }
            let ln = layer_norm(&combined, 1e-5);
            let mut pre_act = vec![0.0f32; h];
            let mut hidden = vec![0.0f32; h];
            for hi in 0..h {
                let w = &self.proj[hi * d..(hi + 1) * d];
                for (j, l) in ln.iter().enumerate() { pre_act[hi] += w[j] * l; }
                hidden[hi] = activate(pre_act[hi], &self.activation);
            }
            all_hidden.push(hidden);
            all_ln.push(ln);
            all_contexts.push(context);
            all_pre_act.push(pre_act);
        }

        for i in 0..count {
            let target = tokens[i + ngram].min(v - 1);
            let mut d_hidden_final = vec![0.0f32; h];
            let mut d_attn_vec = vec![0.0f32; ad];

            if self.use_attention && i >= 3 {
                let start = if i >= ngram { i - ngram + 1 } else { 0 };
                let window_len = i - start + 1;

                let mut keys = vec![vec![0.0f32; ad]; window_len];
                let mut values = vec![vec![0.0f32; ad]; window_len];
                for (wi, idx) in (start..=i).enumerate() {
                    for j in 0..ad {
                        for k in 0..h {
                            keys[wi][j] += self.attn_key[j * h + k] * all_hidden[idx][k];
                            values[wi][j] += self.attn_value[j * h + k] * all_hidden[idx][k];
                        }
                    }
                }

                let mut scores = vec![0.0f32; window_len];
                for wi in 0..window_len {
                    for j in 0..ad {
                        scores[wi] += self.attn_query[j] * keys[wi][j];
                    }
                    scores[wi] /= (ad as f32).sqrt();
                }
                softmax(&mut scores);

                let mut attn_vec = vec![0.0f32; ad];
                for wi in 0..window_len {
                    for j in 0..ad {
                        attn_vec[j] += scores[wi] * values[wi][j];
                    }
                }

                let mut combined = all_hidden[i].clone();
                for j in 0..ad.min(h) {
                    combined[j % h] += attn_vec[j];
                }

                let mut logits = vec![0.0f32; v];
                for (vi, logit) in logits.iter_mut().enumerate() {
                    let w = &self.lm_head[vi * h..(vi + 1) * h];
                    for (hi, hn) in combined.iter().enumerate() { *logit += w[hi] * hn; }
                }
                softmax(&mut logits);

                for (vi, prob) in logits.iter().enumerate() {
                    let grad = prob - if vi == target { 1.0 } else { 0.0 };
                    for hi in 0..h {
                        g_head[vi * h + hi] += grad * combined[hi];
                        d_hidden_final[hi] += grad * self.lm_head[vi * h + hi];
                    }
                }

                for j in 0..ad.min(h) {
                    d_attn_vec[j] += d_hidden_final[j % h];
                }

                for wi in 0..window_len {
                    for j in 0..ad {
                        g_av[j * h..(j + 1) * h].iter_mut()
                            .zip(all_hidden[start + wi].iter())
                            .for_each(|(g, &h_val)| *g += scores[wi] * d_attn_vec[j] * h_val);
                    }
                }

                let mut d_scores = vec![0.0f32; window_len];
                for wi in 0..window_len {
                    for j in 0..ad {
                        d_scores[wi] += d_attn_vec[j] * values[wi][j] * scores[wi];
                        d_scores[wi] += d_attn_vec[j] * values[wi][j];
                    }
                    d_scores[wi] -= d_attn_vec.iter().zip(values[wi].iter()).map(|(&da, &vl)| da * vl * scores[wi]).sum::<f32>();
                }

                let mut d_scores_clean = vec![0.0f32; window_len];
                for wi in 0..window_len {
                    for j in 0..ad {
                        d_scores_clean[wi] += d_attn_vec[j] * values[wi][j];
                    }
                    for wj in 0..window_len {
                        let dot: f32 = d_attn_vec.iter().zip(values[wj].iter()).map(|(&da, &vl)| da * vl).sum();
                        d_scores_clean[wi] -= scores[wi] * scores[wj] * dot;
                    }
                }

                for j in 0..ad {
                    for wi in 0..window_len {
                        g_aq[j] += d_scores_clean[wi] * keys[wi][j] / (ad as f32).sqrt();
                    }
                }

                let mut d_keys = vec![vec![0.0f32; ad]; window_len];
                for wi in 0..window_len {
                    for j in 0..ad {
                        d_keys[wi][j] = d_scores_clean[wi] * self.attn_query[j] / (ad as f32).sqrt();
                    }
                }

                let mut d_values = vec![vec![0.0f32; ad]; window_len];
                for wi in 0..window_len {
                    for j in 0..ad {
                        d_values[wi][j] = scores[wi] * d_attn_vec[j];
                    }
                }

                for (wi, idx) in (start..=i).enumerate() {
                    for j in 0..ad {
                        for k in 0..h {
                            g_ak[j * h + k] += d_keys[wi][j] * all_hidden[idx][k];
                        }
                        for k in 0..h {
                            g_av[j * h + k] += d_values[wi][j] * all_hidden[idx][k];
                        }
                    }
                }
            } else {
                let hidden = &all_hidden[i];
                let mut logits = vec![0.0f32; v];
                for (vi, logit) in logits.iter_mut().enumerate() {
                    let w = &self.lm_head[vi * h..(vi + 1) * h];
                    for (hi, hn) in hidden.iter().enumerate() { *logit += w[hi] * hn; }
                }
                softmax(&mut logits);

                for (vi, prob) in logits.iter().enumerate() {
                    let grad = prob - if vi == target { 1.0 } else { 0.0 };
                    for hi in 0..h {
                        g_head[vi * h + hi] += grad * hidden[hi];
                        d_hidden_final[hi] += grad * self.lm_head[vi * h + hi];
                    }
                }
            }

            let act_grads: Vec<f32> = all_pre_act[i].iter().map(|&pv| activate_grad(pv, &self.activation)).collect();
            for hi in 0..h {
                for j in 0..d {
                    g_proj[hi * d + j] += d_hidden_final[hi] * act_grads[hi] * all_ln[i][j];
                }
            }

            let t0 = all_contexts[i][ngram - 1].min(v - 1);
            for j in 0..d {
                let mut grad_sum = 0.0f32;
                for hi in 0..h {
                    grad_sum += self.proj[hi * d + j] * act_grads[hi] * d_hidden_final[hi];
                }
                g_embed[t0 * d + j] += grad_sum;
                for (ci, cw) in self.ctx_weights.iter().enumerate() {
                    let ctx_idx = ngram - 2 - ci;
                    let t = all_contexts[i][ctx_idx].min(v - 1);
                    g_ctx[ci][t * d + j] += cw * grad_sum;
                }
            }
        }

        let n = count as f32;
        for x in g_embed.iter_mut() { *x /= n; }
        for gc in g_ctx.iter_mut() { for x in gc.iter_mut() { *x /= n; } }
        for x in g_proj.iter_mut() { *x /= n; }
        for x in g_head.iter_mut() { *x /= n; }
        if self.use_attention {
            for x in g_aq.iter_mut() { *x /= n; }
            for x in g_ak.iter_mut() { *x /= n; }
            for x in g_av.iter_mut() { *x /= n; }
        }

        opt_embed.update(&mut self.embed, &g_embed, lr);
        for (ci, oc) in opt_ctx.iter_mut().enumerate() {
            oc.update(&mut self.ctx[ci], &g_ctx[ci], lr);
        }
        opt_proj.update(&mut self.proj, &g_proj, lr);
        opt_head.update(&mut self.lm_head, &g_head, lr);
        if self.use_attention {
            opt_aq.update(&mut self.attn_query, &g_aq, lr);
            opt_ak.update(&mut self.attn_key, &g_ak, lr);
            opt_av.update(&mut self.attn_value, &g_av, lr);
        }
    }
}

fn evaluate(model: &NgramModel, tokens: &[usize], seq_len: usize) -> (f32, f32) {
    let eval_step = if model.use_attention { (seq_len + 1) * 8 } else { seq_len + 1 };
    let mut total = 0.0f32;
    let mut n = 0usize;
    for c in (0..tokens.len()).step_by(eval_step) {
        let end = (c + seq_len + 1).min(tokens.len());
        if end - c < model.ctx.len() + 3 { continue; }
        let loss = model.loss_on_seq(&tokens[c..end]);
        if loss.is_finite() { total += loss / LN_2; n += 1; }
    }
    if n == 0 { return (f32::MAX, f32::MAX); }
    let bpb = total / n as f32;
    (bpb * LN_2, bpb)
}

fn cosine_lr(step: usize, max_steps: usize, base_lr: f32, warmup: usize) -> f32 {
    if step < warmup { return base_lr * step as f32 / warmup as f32; }
    let p = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
    1e-5 + (base_lr - 1e-5) * 0.5 * (1.0 + (std::f32::consts::PI * p).cos())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.iter().find(|a| a.starts_with("--seed="))
        .map(|a| a[7..].parse::<u64>().unwrap_or(42)).unwrap_or(42);
    let steps: usize = args.iter().find(|a| a.starts_with("--steps="))
        .map(|a| a[8..].parse::<usize>().unwrap_or(10000)).unwrap_or(10000);
    let base_lr: f32 = args.iter().find(|a| a.starts_with("--lr="))
        .map(|a| a[5..].parse::<f32>().unwrap_or(0.003)).unwrap_or(0.003);
    let hidden: usize = args.iter().find(|a| a.starts_with("--hidden="))
        .map(|a| a[9..].parse::<usize>().unwrap_or(128)).unwrap_or(128);
    let wd: f32 = args.iter().find(|a| a.starts_with("--wd="))
        .map(|a| a[5..].parse::<f32>().unwrap_or(0.04)).unwrap_or(0.04);
    let activation = args.iter().find(|a| a.starts_with("--activation="))
        .map(|a| a[13..].to_string()).unwrap_or_else(|| "relu".to_string());
    let has_ctx5 = args.iter().any(|a| a == "--ctx5");
    let has_ctx4 = args.iter().any(|a| a == "--ctx4");
    let has_ctx3 = args.iter().any(|a| a == "--ctx3");
    let num_ctx = if has_ctx5 { 5 } else if has_ctx4 { 4 } else if has_ctx3 { 3 } else { 2 };
    let ngram_order = num_ctx + 2;
    let use_attention = args.iter().any(|a| a == "--attention");

    let ngram_name = format!("{}-Gram", ngram_order);
    let act_name = match activation.as_str() { "gelu" => "GELU", "relu2" => "ReLU²", _ => "ReLU" };
    let attn_str = if use_attention { " + AttentionPool" } else { "" };
    println!("=== {} Context Model + {} Hidden{} ===", ngram_name, act_name, attn_str);
    println!("vocab={} dim={} hidden={} seq={} steps={} seed={} lr={} wd={} ctx={} activation={} attention={}",
        VOCAB, DIM, hidden, SEQ, steps, seed, base_lr, wd, num_ctx, activation, use_attention);

    let tokens = load_data("data/tinyshakespeare.txt");
    println!("Dataset: {} tokens", tokens.len());

    let train_end = (tokens.len() as f64 * 0.9) as usize;
    let train = &tokens[..train_end];
    let val = &tokens[train_end..];
    println!("Split: {} train / {} val", train.len(), val.len());

    let mut model = NgramModel::new(VOCAB, DIM, hidden, activation.clone(), seed, num_ctx, use_attention);
    let ps = VOCAB * DIM;
    let mut optimizer = args.iter().find(|a| a.starts_with("--optimizer="))
        .map(|a| a[11..].to_string()).unwrap_or_else(|| "adamw".to_string());
    let use_muon = optimizer == "muon";

    fn make_opt(size: usize, wd: f32, muon: bool) -> Box<dyn Optimizer> {
		if muon { Box::new(MuonOptimizer::new(size, 0.004, 0.95, wd as f64)) } else { Box::new(LocalAdamW::new(size, wd)) }
    }
    let mut opt_embed: Box<dyn Optimizer> = make_opt(ps, wd, use_muon);
    let mut opt_ctx: Vec<Box<dyn Optimizer>> = (0..num_ctx).map(|_| make_opt(ps, wd, use_muon)).collect();
    let mut opt_proj: Box<dyn Optimizer> = make_opt(hidden * DIM, wd, use_muon);
    let mut opt_head: Box<dyn Optimizer> = make_opt(VOCAB * hidden, wd, use_muon);
    let ad = hidden / 4;
    let mut opt_aq: Box<dyn Optimizer> = make_opt(if use_attention { ad } else { 1 }, wd, use_muon);
    let mut opt_ak: Box<dyn Optimizer> = make_opt(if use_attention { ad * hidden } else { 1 }, wd, use_muon);
    let mut opt_av: Box<dyn Optimizer> = make_opt(if use_attention { ad * hidden } else { 1 }, wd, use_muon);

    let (init_loss, init_bpb) = evaluate(&model, val, SEQ);
    println!("Initial val: loss={:.4} bpb={:.4}", init_loss, init_bpb);
    println!();
    println!("{:>6} | {:>10} | {:>10} | {:>10} | {:>8}", "step", "val_loss", "val_bpb", "best_bpb", "ms");
    println!("{}", "-".repeat(60));

    let t0 = Instant::now();
    let mut best_bpb = init_bpb;
    let mut results: Vec<(usize, f32, f32)> = Vec::new();
    let dl = train.len();

    for step in 1..=steps {
        let lr = cosine_lr(step, steps, base_lr, steps / 10);
        let off = (step * 97 + seed as usize) % (dl.saturating_sub(SEQ + 1));
        {
            model.train_step(&train[off..off + SEQ + 1], lr,
                opt_embed.as_mut(), &mut opt_ctx[..], opt_proj.as_mut(), opt_head.as_mut(),
                opt_aq.as_mut(), opt_ak.as_mut(), opt_av.as_mut());
        }

        if step % 500 == 0 || step == steps {
            let ms = t0.elapsed().as_millis();
            let (vl, vb) = evaluate(&model, val, SEQ);
            if vb < best_bpb && vb.is_finite() { best_bpb = vb; }
            println!("{:>6} | {:>10.4} | {:>10.4} | {:>10.4} | {:>6}ms", step, vl, vb, best_bpb, ms);
            results.push((step, vl, vb));
        }
    }

    let total = t0.elapsed();
    println!("\n=== Done ===");
    println!("Time: {:.1}s | BPB: {:.4} → {:.4} | Delta: {:.4}", total.as_secs_f64(), init_bpb, best_bpb, best_bpb - init_bpb);

    let _ = fs::create_dir_all(".trinity/results");
    let attn_tag = if use_attention { "_attn" } else { "" };
    let exp_name = format!("{}gram-{}-h{}{}", ngram_order, activation, hidden, attn_tag);

    let rj = serde_json::json!({
        "experiment": exp_name,
        "model": format!("{}-gram context + {} hidden{} + LM head", ngram_order, act_name, attn_str),
        "seed": seed, "steps": steps, "base_lr": base_lr, "wd": wd,
        "hidden_size": hidden, "activation": activation,
        "num_ctx": num_ctx, "ngram_order": ngram_order,
        "use_attention": use_attention,
        "train_tokens": train.len(), "val_tokens": val.len(),
        "initial_val_bpb": init_bpb, "final_val_bpb": best_bpb,
        "delta_bpb": best_bpb - init_bpb,
        "duration_seconds": total.as_secs_f64(),
        "results": results.iter().map(|(s, l, b)| serde_json::json!({"step":*s,"loss":*l,"bpb":*b})).collect::<Vec<_>>(),
    });
    let rp = format!(".trinity/results/{}gram_{}_seed{}.json",
        ngram_order, activation, seed);
    fs::File::create(&rp).unwrap().write_all(serde_json::to_string_pretty(&rj).unwrap().as_bytes()).unwrap();
    println!("Results: {}", rp);

    let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
    let ep = format!(".trinity/experience/trios_{}.trinity", chrono::Utc::now().format("%Y%m%d"));
    let _ = fs::create_dir_all(".trinity/experience");
    let _ = fs::OpenOptions::new().create(true).append(true).open(&ep).unwrap()
        .write_all(format!("[{}] TASK: {}-gram{} training | seed={} | steps={} | val_bpb={:.4}->{:.4} | {:.1}s\n",
            ts, ngram_order, attn_str, seed, steps, init_bpb, best_bpb, total.as_secs_f64()).as_bytes());
    println!("Experience: {}", ep);
}
