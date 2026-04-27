//! Hybrid N-gram + Causal Attention Training — L-f1 + L-f2 Lanes
//!
//! Architecture: n-gram base + HybridAttn (causal, RoPE, qk_gain=φ²) + ReLU²
//! - NGRAM=8, DIM=64, HIDDEN=828 (φ-scaled), VOCAB=128, NUM_CTX=6
//! - 1-layer causal attention (HybridAttn from crate::model_hybrid_attn)
//! - ReLU² activation with proper backward
//! - EMA val BPB (β=φ⁻¹), cosine LR, gradient accumulation
//!
//! Run: cargo run --release --bin hybrid_train -- --seed=43 --steps=54000

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_mut)]

use std::env;
use std::fs;
use std::time::Instant;

use trios_trainer::model_hybrid_attn::HybridAttn;
use trios_trainer::optimizer::MuonOptimizer;

const VOCAB: usize = 128;
const DIM: usize = 64;
const DEFAULT_HIDDEN: usize = 828;
const NUM_CTX: usize = 6;
const NGRAM: usize = NUM_CTX + 2;
const SEQ: usize = 128;
const LN_2: f32 = std::f32::consts::LN_2;
const PHI_INV: f32 = 0.618033988749895;
const GF16_FLOOR_FRAC: f32 = 0.7;
const GATE_FINAL_SEEDS: [u64; 3] = [42, 43, 44];
const CTX_WEIGHTS: [f32; NUM_CTX] = [0.70, 0.45, 0.30, 0.20, 0.13, 0.08];
const NCA_WEIGHT: f32 = 0.25;
const NCA_K: usize = 9;
const NCA_ENTROPY_MIN: f32 = 1.5;
const NCA_ENTROPY_MAX: f32 = 2.8;

fn load_data(path: &str) -> Vec<usize> {
    let raw = fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to load {}: {}. Using fallback.", path, e);
        b"The quick brown fox jumps over the lazy dog. "
            .repeat(100)
            .to_vec()
    });
    assert!(!raw.is_empty(), "loaded data is empty");
    raw.into_iter().map(|b| (b as usize) % VOCAB).collect()
}

fn layer_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std_inv = 1.0 / (var + eps).sqrt();
    x.iter().map(|v| (v - mean) * std_inv).collect()
}

fn layer_norm_backward(x: &[f32], y: &[f32], dy: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std_inv = 1.0 / (var + eps).sqrt();

    let sum_dy: f32 = dy.iter().sum();
    let sum_dy_y: f32 = dy.iter().zip(y.iter()).map(|(d, yi)| d * yi).sum();

    dy.iter()
        .zip(y.iter())
        .map(|(d, yi)| (d - sum_dy / n - yi * sum_dy_y / n) * std_inv)
        .collect()
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

fn cosine_lr(step: usize, max_steps: usize, base_lr: f32, warmup: usize) -> f32 {
    if step < warmup {
        return base_lr * step as f32 / warmup.max(1) as f32;
    }
    let p = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
    1e-5 + (base_lr - 1e-5) * 0.5 * (1.0 + (std::f32::consts::PI * p).cos())
}

struct AdamW {
    m: Vec<f32>,
    v: Vec<f32>,
    step: usize,
    beta1: f32,
    beta2: f32,
    wd: f32,
}

impl AdamW {
    fn new(size: usize, wd: f32) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            step: 0,
            beta1: 0.9,
            beta2: 0.999,
            wd,
        }
    }

    fn update(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        self.step += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);
        for i in 0..params.len() {
            params[i] -= self.wd * lr * params[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            params[i] -= lr * (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + 1e-8);
        }
    }
}

struct HybridModel {
    embed: Vec<f32>,
    ctx: Vec<Vec<f32>>,
    proj: Vec<f32>,
    attn: HybridAttn,
    attn_down: Vec<f32>,
    attn_up: Vec<f32>,
    lm_head: Vec<f32>,
    hidden: usize,
}

impl HybridModel {
    fn new(hidden: usize, seed: u64) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let lim = (6.0f32 / (3 * DIM) as f32).sqrt();
        let lim_h = (6.0f32 / (DIM + hidden) as f32).sqrt();
        let lim_o = (6.0f32 / (hidden + VOCAB) as f32).sqrt();

        let attn = HybridAttn::new().expect("attn construct with defaults");
        let d = attn.config().d_model;
        let lim_down = (2.0f32 / (hidden + d) as f32).sqrt();
        let lim_up = (2.0f32 / (d + hidden) as f32).sqrt();

        let mut m = Self {
            embed: (0..VOCAB * DIM).map(|_| rng() * lim).collect(),
            ctx: (0..NUM_CTX)
                .map(|_| (0..VOCAB * DIM).map(|_| rng() * lim).collect())
                .collect(),
            proj: (0..hidden * DIM).map(|_| rng() * lim_h).collect(),
            attn,
            attn_down: (0..d * hidden).map(|_| rng() * lim_down).collect(),
            attn_up: (0..hidden * d).map(|_| rng() * lim_up).collect(),
            lm_head: (0..VOCAB * hidden).map(|_| rng() * lim_o).collect(),
            hidden,
        };

        let attn_cfg = m.attn.config();
        let d = attn_cfg.d_model;
        let attn_lim = (2.0f32 / d as f32).sqrt();
        for w in m.attn.wq_mut() { *w = rng() * attn_lim; }
        for w in m.attn.wk_mut() { *w = rng() * attn_lim; }
        for w in m.attn.wv_mut() { *w = rng() * attn_lim; }
        for w in m.attn.wo_mut() { *w = rng() * attn_lim; }

        m
    }

    fn embed_tokens(&self, tokens: &[usize], pos: usize) -> Vec<f32> {
        let t_last = tokens[pos + NGRAM - 1].min(VOCAB - 1);
        let mut combined = self.embed[t_last * DIM..(t_last + 1) * DIM].to_vec();
        for (ci, cw) in CTX_WEIGHTS.iter().enumerate() {
            let ctx_idx = NGRAM - 2 - ci;
            let t = tokens[pos + ctx_idx].min(VOCAB - 1);
            let cv = &self.ctx[ci][t * DIM..(t + 1) * DIM];
            for j in 0..DIM {
                combined[j] += cv[j] * cw;
            }
        }
        combined
    }

    fn forward_position(
        &self,
        tokens: &[usize],
        pos: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let h = self.hidden;
        let d = self.attn.config().d_model;
        let combined = self.embed_tokens(tokens, pos);
        let ln = layer_norm(&combined, 1e-5);

        let mut hidden_raw = vec![0.0f32; h];
        for hi in 0..h {
            for j in 0..DIM {
                hidden_raw[hi] += self.proj[hi * DIM + j] * ln[j];
            }
        }
        let mut hidden = vec![0.0f32; h];
        for hi in 0..h {
            hidden[hi] = if hidden_raw[hi] > 0.0 {
                hidden_raw[hi] * hidden_raw[hi]
            } else {
                0.0
            };
        }

        let mut attn_in = vec![0.0f32; d];
        for di in 0..d {
            for hi in 0..h {
                attn_in[di] += self.attn_down[di * h + hi] * hidden[hi];
            }
        }

        let mut attn_out_saved = vec![0.0f32; d];
        if let Ok(attn_out) = self.attn.forward(&attn_in, 1) {
            attn_out_saved = attn_out.clone();
            let mut attn_up_out = vec![0.0f32; h];
            for hi in 0..h {
                for di in 0..d {
                    attn_up_out[hi] += self.attn_up[hi * d + di] * attn_out[di];
                }
            }
            for hi in 0..h {
                hidden[hi] += attn_up_out[hi] * 0.1;
            }
        }

        let mut logits = vec![0.0f32; VOCAB];
        for vi in 0..VOCAB {
            for hi in 0..h {
                logits[vi] += self.lm_head[vi * h + hi] * hidden[hi];
            }
        }
        (combined, ln, hidden, logits, attn_out_saved)
    }

    fn loss_on_seq(&self, tokens: &[usize]) -> f32 {
        if tokens.len() < NGRAM + 1 {
            return 0.0;
        }
        let count = tokens.len() - NGRAM;
        let mut total = 0.0f32;
        for i in 0..count {
            let target = tokens[i + NGRAM].min(VOCAB - 1);
            let (_, _, _, mut logits, _) = self.forward_position(tokens, i);
            softmax(&mut logits);
            total -= logits[target].max(1e-10).ln();
        }
        total / count as f32
    }
}

fn compute_grads_for_positions(
    model: &HybridModel,
    tokens: &[usize],
    positions: &[usize],
    g_embed: &mut [f32],
    g_ctx: &mut [Vec<f32>],
    g_proj: &mut [f32],
    g_head: &mut [f32],
    g_attn_down: &mut [f32],
    g_attn_up: &mut [f32],
) {
    let h = model.hidden;
    let d = model.attn.config().d_model;
    for &pos in positions {
        let (combined, ln, hidden, mut logits, attn_out_saved) = model.forward_position(tokens, pos);
        softmax(&mut logits);
        let target = tokens[pos + NGRAM].min(VOCAB - 1);

        let mut d_hidden = vec![0.0f32; h];
        for vi in 0..VOCAB {
            let grad = logits[vi] - if vi == target { 1.0 } else { 0.0 };
            for hi in 0..h {
                g_head[vi * h + hi] += grad * hidden[hi];
                d_hidden[hi] += grad * model.lm_head[vi * h + hi];
            }
        }

        let nca_loss = nca_entropy_loss(&logits);
        if nca_loss > 0.0 {
            let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f32 = probs.iter().sum();
            for p in probs.iter_mut() { *p /= sum; }
            let entropy: f32 = -probs.iter().map(|&p: &f32| p.max(1e-10).ln() * p).sum::<f32>();
            let nca_grad = if entropy < NCA_ENTROPY_MIN {
                -2.0 * NCA_WEIGHT * (NCA_ENTROPY_MIN - entropy)
            } else {
                2.0 * NCA_WEIGHT * (entropy - NCA_ENTROPY_MAX)
            };
            for vi in 0..VOCAB {
                let d_ent = probs[vi] * (probs[vi].max(1e-10).ln() + entropy);
                for hi in 0..h {
                    g_head[vi * h + hi] += nca_grad * d_ent * hidden[hi] * 0.01;
                }
            }
        }

        let mut d_attn_up_out = vec![0.0f32; h];
        for hi in 0..h {
            d_attn_up_out[hi] = d_hidden[hi] * 0.1;
        }

        let mut d_attn_out = vec![0.0f32; d];
        for hi in 0..h {
            for di in 0..d {
                g_attn_up[hi * d + di] += d_attn_up_out[hi] * attn_out_saved[di];
                d_attn_out[di] += d_attn_up_out[hi] * model.attn_up[hi * d + di];
            }
        }

        let d_attn_in = d_attn_out;
        for di in 0..d {
            for hi in 0..h {
                g_attn_down[di * h + hi] += d_attn_in[di] * hidden[hi];
            }
        }

        let mut d_raw = vec![0.0f32; h];
        for hi in 0..h {
            if hidden[hi] > 0.0 {
                let raw_val = hidden[hi].sqrt();
                d_raw[hi] = d_hidden[hi] * 2.0 * raw_val;
            }
        }

        let mut d_ln = vec![0.0f32; DIM];
        for hi in 0..h {
            for j in 0..DIM {
                g_proj[hi * DIM + j] += d_raw[hi] * ln[j];
                d_ln[j] += model.proj[hi * DIM + j] * d_raw[hi];
            }
        }

        let d_combined = layer_norm_backward(&combined, &ln, &d_ln, 1e-5);

        let t_last = tokens[pos + NGRAM - 1].min(VOCAB - 1);
        for j in 0..DIM {
            g_embed[t_last * DIM + j] += d_combined[j];
        }
        for (ci, cw) in CTX_WEIGHTS.iter().enumerate() {
            let ctx_idx = NGRAM - 2 - ci;
            let t = tokens[pos + ctx_idx].min(VOCAB - 1);
            for j in 0..DIM {
                g_ctx[ci][t * DIM + j] += cw * d_combined[j];
            }
        }
    }
}

fn evaluate(model: &HybridModel, tokens: &[usize]) -> f32 {
    let chunk_size = SEQ + 1;
    let num_chunks = 40usize;
    let max_start = tokens.len().saturating_sub(chunk_size);
    if max_start == 0 {
        return f32::MAX;
    }
    let step = if max_start >= num_chunks * chunk_size {
        max_start / num_chunks
    } else {
        chunk_size
    };
    let mut total = 0.0f32;
    let mut n = 0usize;
    for c in (0..max_start).step_by(step).take(num_chunks) {
        let end = (c + chunk_size).min(tokens.len());
        if end - c < NGRAM + 2 {
            continue;
        }
        let loss = model.loss_on_seq(&tokens[c..end]);
        if loss.is_finite() {
            total += loss / LN_2;
            n += 1;
        }
    }
    if n == 0 {
        f32::MAX
    } else {
        total / n as f32
    }
}

fn nca_entropy_loss(logits: &[f32]) -> f32 {
    let n = logits.len() as f32;
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = logits.iter().map(|&x| (x - max_val).exp()).collect::<Vec<_>>();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let entropy: f32 = -probs.iter().map(|&p: &f32| p.max(1e-10).ln() * p).sum::<f32>();
    if entropy < NCA_ENTROPY_MIN {
        NCA_WEIGHT * (NCA_ENTROPY_MIN - entropy).powi(2)
    } else if entropy > NCA_ENTROPY_MAX {
        NCA_WEIGHT * (entropy - NCA_ENTROPY_MAX).powi(2)
    } else {
        0.0
    }
}

fn find_arg<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> T {
    args.iter()
        .find(|a| a.starts_with(key))
        .and_then(|a| a[key.len()..].parse().ok())
        .unwrap_or(default)
}

fn gf16_floor(weights: &mut [f32]) {
    let scale = 16.0_f32;
    for w in weights.iter_mut() {
        *w = (*w * scale).round() / scale;
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let seed: u64 = find_arg(&args, "--seed=", 0u64);
    let steps: usize = find_arg(&args, "--steps=", 81000usize);
    let base_lr: f32 = find_arg(&args, "--lr=", 0.003f32);
    let hidden: usize = find_arg(&args, "--hidden=", DEFAULT_HIDDEN);
    let eval_every: usize = find_arg(&args, "--eval-every=", 1000usize);
    let accum: usize = find_arg(&args, "--accum=", 4usize);

    let gf16_floor_step = (GF16_FLOOR_FRAC * steps as f32).floor() as usize;

    let seeds: Vec<u64> = if seed > 0 {
        vec![seed]
    } else {
        GATE_FINAL_SEEDS.to_vec()
    };

    for &seed in &seeds {
    eprintln!(
        "=== Hybrid Train (ngram+attn) seed={} ===",
        seed
    );
    eprintln!(
        "steps={} lr={} hidden={} eval_every={} accum={}",
        steps, base_lr, hidden, eval_every, accum
    );
    eprintln!(
        "DIM={} NUM_CTX={} NGRAM={} SEQ={} VOCAB={}",
        DIM, NUM_CTX, NGRAM, SEQ, VOCAB
    );
    eprintln!("ctx_weights={:?}", CTX_WEIGHTS);

    let train_data = load_data("data/tiny_shakespeare.txt");
    let val_data = load_data("data/tiny_shakespeare_val.txt");
    eprintln!("train={} val={}", train_data.len(), val_data.len());

    let mut model = HybridModel::new(hidden, seed);

    let d = model.attn.config().d_model;
    let attn_params = d * model.hidden * 2;
    let total_params = VOCAB * DIM + NUM_CTX * VOCAB * DIM + hidden * DIM + VOCAB * hidden + attn_params;
    eprintln!("params={} ({:.1}K) attn_d={}", total_params, total_params as f64 / 1000.0, d);

        let train_data = load_data("data/tiny_shakespeare.txt");
        let val_data = load_data("data/tiny_shakespeare_val.txt");
        eprintln!("train={} val={}", train_data.len(), val_data.len());

        let mut model = HybridModel::new(hidden, seed);

        let wd = 0.04f32;
        let mut opt_embed = AdamW::new(VOCAB * DIM, wd);
        let mut opt_ctx: Vec<AdamW> = (0..NUM_CTX)
            .map(|_| AdamW::new(VOCAB * DIM, wd))
            .collect();
        let mut opt_proj = MuonOptimizer::with_matrix_shape(
            hidden * DIM, hidden, DIM,
            base_lr as f64, 0.95, wd as f64,
        );
        let mut opt_attn_down = AdamW::new(d * hidden, wd);
        let mut opt_attn_up = AdamW::new(hidden * d, wd);
        let mut opt_head = AdamW::new(VOCAB * hidden, wd);

        let init_bpb = evaluate(&model, &val_data);
        eprintln!("Initial val_bpb={:.4}", init_bpb);

        let mut best_ema_bpb = init_bpb;
        let mut ema_bpb = init_bpb;
        let t0 = Instant::now();
        let warmup = steps / 10;

        let mut rng_s = seed.wrapping_add(7919);

        for step in 1..=steps {
            let lr = cosine_lr(step, steps, base_lr, warmup);

            let mut g_embed = vec![0.0f32; VOCAB * DIM];
            let mut g_ctx: Vec<Vec<f32>> = (0..NUM_CTX)
                .map(|_| vec![0.0f32; VOCAB * DIM])
                .collect();
        let mut g_proj = vec![0.0f32; hidden * DIM];
        let mut g_head = vec![0.0f32; VOCAB * hidden];
        let mut g_attn_down = vec![0.0f32; d * hidden];
        let mut g_attn_up = vec![0.0f32; hidden * d];

            for _micro in 0..accum {
                rng_s = rng_s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let dl = train_data.len();
                let max_start = dl.saturating_sub(SEQ + 1);
                if max_start == 0 {
                    continue;
                }
                let chunk_start = (rng_s as usize) % max_start;
                let chunk = &train_data[chunk_start..chunk_start + SEQ + 1];

                let count = chunk.len().saturating_sub(NGRAM);
                if count == 0 {
                    continue;
                }
                let num_sample = 8.min(count);
                let mut positions: Vec<usize> = Vec::with_capacity(num_sample);
                for _ in 0..num_sample {
                    rng_s = rng_s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let p = (rng_s as usize) % count;
                    positions.push(p);
                }

            compute_grads_for_positions(
                &model,
                chunk,
                &positions,
                &mut g_embed,
                &mut g_ctx,
                &mut g_proj,
                &mut g_head,
                &mut g_attn_down,
                &mut g_attn_up,
            );
            }

            let total_positions = (accum * 8) as f32;
            for x in g_embed.iter_mut() {
                *x /= total_positions;
            }
            for gc in g_ctx.iter_mut() {
                for x in gc.iter_mut() {
                    *x /= total_positions;
                }
            }
            for x in g_proj.iter_mut() {
                *x /= total_positions;
            }
        for x in g_head.iter_mut() {
            *x /= total_positions;
        }
        for x in g_attn_down.iter_mut() {
            *x /= total_positions;
        }
        for x in g_attn_up.iter_mut() {
            *x /= total_positions;
        }

            opt_embed.update(&mut model.embed, &g_embed, lr);
            for (ci, oc) in opt_ctx.iter_mut().enumerate() {
                oc.update(&mut model.ctx[ci], &g_ctx[ci], lr);
            }
        opt_proj.step(&mut model.proj, &g_proj);
        opt_attn_down.update(&mut model.attn_down, &g_attn_down, lr);
        opt_attn_up.update(&mut model.attn_up, &g_attn_up, lr);
        opt_head.update(&mut model.lm_head, &g_head, lr);

            if step >= gf16_floor_step && step % eval_every == 0 {
                gf16_floor(&mut model.embed);
                gf16_floor(&mut model.proj);
                gf16_floor(&mut model.lm_head);
                for c in &mut model.ctx {
                    gf16_floor(c);
                }
            }

            if step % eval_every == 0 || step == steps {
                let val_bpb = evaluate(&model, &val_data);
                ema_bpb = PHI_INV * ema_bpb + (1.0 - PHI_INV) * val_bpb;
                if ema_bpb < best_ema_bpb && ema_bpb.is_finite() {
                    best_ema_bpb = ema_bpb;
                }
                let t = t0.elapsed().as_secs_f64();
                println!(
                    "seed={} step={} val_bpb={:.4} ema_bpb={:.4} best={:.4} t={:.1}s",
                    seed, step, val_bpb, ema_bpb, best_ema_bpb, t
                );
            }
        }

        println!("seed={} BPB={:.4}", seed, best_ema_bpb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gf16_floor_rounds_to_16th() {
        let mut w = vec![0.123456, -0.987654, 0.0, 1.5];
        gf16_floor(&mut w);
        assert!((w[0] - 0.125).abs() < 1e-6, "0.123456 -> 0.125");
        assert!((w[1] - (-1.0)).abs() < 1e-6, "-0.987654 -> -1.0");
        assert!((w[2] - 0.0).abs() < 1e-6, "0.0 stays");
        assert!((w[3] - 1.5).abs() < 1e-6, "1.5 stays");
    }

    #[test]
    fn gf16_floor_step_at_70pct() {
        let steps = 81000;
        let floor_step = (GF16_FLOOR_FRAC * steps as f32).floor() as usize;
        assert_eq!(floor_step, 56700, "GF16 floor activates at 56700 for 81K steps");
    }

    #[test]
    fn gate_final_seeds_are_42_43_44() {
        assert_eq!(&GATE_FINAL_SEEDS, &[42u64, 43, 44]);
    }

    #[test]
    fn phi_hidden_is_828() {
        assert_eq!(DEFAULT_HIDDEN, 828, "φ-scaled hidden = round(φ*512) = 828");
    }

    #[test]
    fn ema_beta_is_phi_inv() {
        assert!((PHI_INV - 0.618033988749895).abs() < 1e-12);
    }

    #[test]
    fn falsify_seed_outside_gate_final_set() {
        let allowed: Vec<u64> = GATE_FINAL_SEEDS.to_vec();
        assert!(!allowed.contains(&41), "seed 41 frozen out");
        assert!(!allowed.contains(&45), "seed 45 frozen out");
    }
}
