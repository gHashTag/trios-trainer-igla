#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::fs;
use std::time::Instant;

const VOCAB: usize = 128;
const LN_2: f32 = std::f32::consts::LN_2;
const SEQ: usize = 128;
const EVAL_INTERVAL: usize = 500;
const EVAL_CHUNKS: usize = 40;

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

fn layer_norm_backward(x: &[f32], y: &[f32], dy: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = (var + eps).sqrt();
    let dy_mean = dy.iter().sum::<f32>() / n;
    let dyy_mean = dy.iter().zip(y.iter()).map(|(d, yi)| d * yi).sum::<f32>() / n;
    dy.iter()
        .zip(y.iter())
        .map(|(d, yi)| (d - dy_mean - yi * dyy_mean) / std)
        .collect()
}

struct Model {
    dim: usize,
    hidden: usize,
    ctx_len: usize,
    embed: Vec<f32>,
    pos_embed: Vec<f32>,
    proj1: Vec<f32>,
    proj2: Vec<f32>,
    lm_head: Vec<f32>,
}

struct FwdState {
    cat: Vec<f32>,
    cat_ln: Vec<f32>,
    h1_raw: Vec<f32>,
    h1: Vec<f32>,
    h1_ln: Vec<f32>,
    h2_raw: Vec<f32>,
    h2: Vec<f32>,
    h2_ln: Vec<f32>,
    logits: Vec<f32>,
}

struct Grads {
    g_embed: Vec<f32>,
    g_pos: Vec<f32>,
    g_proj1: Vec<f32>,
    g_proj2: Vec<f32>,
    g_head: Vec<f32>,
}

struct AdamW {
    m: Vec<f32>,
    v: Vec<f32>,
    step: usize,
}

impl AdamW {
    fn new(size: usize) -> Self {
        Self { m: vec![0.0; size], v: vec![0.0; size], step: 0 }
    }
    fn update(&mut self, params: &mut [f32], grads: &[f32], lr: f32, wd: f32) {
        self.step += 1;
        let b1: f32 = 0.9;
        let b2: f32 = 0.999;
        let bc1 = 1.0 - b1.powi(self.step as i32);
        let bc2 = 1.0 - b2.powi(self.step as i32);
        for i in 0..params.len() {
            params[i] -= wd * lr * params[i];
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grads[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * grads[i] * grads[i];
            params[i] -= lr * (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + 1e-8);
        }
    }
}

impl Model {
    fn new(dim: usize, hidden: usize, ctx_len: usize, seed: u64) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let cat_dim = ctx_len * dim;
        let lim_e = (6.0f32 / (VOCAB + dim) as f32).sqrt();
        let lim_p1 = (6.0f32 / (cat_dim + hidden) as f32).sqrt();
        let lim_p2 = (6.0f32 / (hidden + hidden) as f32).sqrt();
        let lim_o = (6.0f32 / (hidden + VOCAB) as f32).sqrt();

        Self {
            dim,
            hidden,
            ctx_len,
            embed: (0..VOCAB * dim).map(|_| rng() * lim_e).collect(),
            pos_embed: (0..ctx_len * dim).map(|_| rng() * lim_e * 0.5).collect(),
            proj1: (0..hidden * cat_dim).map(|_| rng() * lim_p1).collect(),
            proj2: (0..hidden * hidden).map(|_| rng() * lim_p2).collect(),
            lm_head: (0..VOCAB * hidden).map(|_| rng() * lim_o).collect(),
        }
    }

    fn forward_one(&self, context: &[usize]) -> FwdState {
        let d = self.dim;
        let h = self.hidden;
        let cl = self.ctx_len;
        let cat_dim = cl * d;

        let mut cat = vec![0.0f32; cat_dim];
        for (ci, &tok) in context.iter().enumerate() {
            let t = tok.min(VOCAB - 1);
            for j in 0..d {
                cat[ci * d + j] = self.embed[t * d + j] + self.pos_embed[ci * d + j];
            }
        }

        let cat_ln = layer_norm(&cat, 1e-5);

        let mut h1_raw = vec![0.0f32; h];
        for hi in 0..h {
            let mut val = 0.0f32;
            for j in 0..cat_dim {
                val += self.proj1[hi * cat_dim + j] * cat_ln[j];
            }
            h1_raw[hi] = val;
        }
        let h1: Vec<f32> = h1_raw.iter().map(|&v| v * (v > 0.0) as u32 as f32).collect();
        let h1_ln = layer_norm(&h1, 1e-5);

        let mut h2_raw = vec![0.0f32; h];
        for hi in 0..h {
            let mut val = 0.0f32;
            for j in 0..h {
                val += self.proj2[hi * h + j] * h1_ln[j];
            }
            h2_raw[hi] = val;
        }
        let h2: Vec<f32> = h2_raw.iter().map(|&v| v * (v > 0.0) as u32 as f32).collect();
        let h2_ln = layer_norm(&h2, 1e-5);

        let mut logits = vec![0.0f32; VOCAB];
        for v in 0..VOCAB {
            for j in 0..h {
                logits[v] += self.lm_head[v * h + j] * h2_ln[j];
            }
        }

        FwdState { cat, cat_ln, h1_raw, h1, h1_ln, h2_raw, h2, h2_ln, logits }
    }

    fn backward_one(&self, st: &FwdState, context: &[usize], target: usize) -> (Grads, f32) {
        let d = self.dim;
        let h = self.hidden;
        let cl = self.ctx_len;
        let cat_dim = cl * d;

        let mut probs = st.logits.clone();
        softmax(&mut probs);
        let loss = -probs[target].max(1e-10).ln();

        let mut d_logits = probs;
        d_logits[target] -= 1.0;

        let mut g_embed = vec![0.0f32; VOCAB * d];
        let mut g_pos = vec![0.0f32; cl * d];
        let mut g_proj1 = vec![0.0f32; h * cat_dim];
        let mut g_proj2 = vec![0.0f32; h * h];
        let mut g_head = vec![0.0f32; VOCAB * h];

        let mut d_h2_ln = vec![0.0f32; h];
        for v in 0..VOCAB {
            let dl = d_logits[v];
            for j in 0..h {
                g_head[v * h + j] += dl * st.h2_ln[j];
                d_h2_ln[j] += dl * self.lm_head[v * h + j];
            }
        }

        let d_h2 = layer_norm_backward(&st.h2, &st.h2_ln, &d_h2_ln, 1e-5);
        let mut d_h2_raw = vec![0.0f32; h];
        for i in 0..h {
            d_h2_raw[i] = if st.h2_raw[i] > 0.0 { d_h2[i] } else { 0.0 };
        }

        let mut d_h1_ln = vec![0.0f32; h];
        for hi in 0..h {
            if d_h2_raw[hi] == 0.0 { continue; }
            for j in 0..h {
                g_proj2[hi * h + j] += d_h2_raw[hi] * st.h1_ln[j];
                d_h1_ln[j] += d_h2_raw[hi] * self.proj2[hi * h + j];
            }
        }

        let d_h1 = layer_norm_backward(&st.h1, &st.h1_ln, &d_h1_ln, 1e-5);
        let mut d_h1_raw = vec![0.0f32; h];
        for i in 0..h {
            d_h1_raw[i] = if st.h1_raw[i] > 0.0 { d_h1[i] } else { 0.0 };
        }

        let mut d_cat_ln = vec![0.0f32; cat_dim];
        for hi in 0..h {
            if d_h1_raw[hi] == 0.0 { continue; }
            for j in 0..cat_dim {
                g_proj1[hi * cat_dim + j] += d_h1_raw[hi] * st.cat_ln[j];
                d_cat_ln[j] += d_h1_raw[hi] * self.proj1[hi * cat_dim + j];
            }
        }

        let d_cat = layer_norm_backward(&st.cat, &st.cat_ln, &d_cat_ln, 1e-5);

        for (ci, &tok) in context.iter().enumerate() {
            let t = tok.min(VOCAB - 1);
            for j in 0..d {
                g_embed[t * d + j] += d_cat[ci * d + j];
                g_pos[ci * d + j] += d_cat[ci * d + j];
            }
        }

        (Grads { g_embed, g_pos, g_proj1, g_proj2, g_head }, loss)
    }

    fn loss_on_seq(&self, tokens: &[usize]) -> f32 {
        let cl = self.ctx_len;
        let count = tokens.len().saturating_sub(cl + 1);
        if count == 0 { return 0.0; }
        let mut total = 0.0f32;
        for i in 0..count {
            let context = &tokens[i..i + cl];
            let target = tokens[i + cl].min(VOCAB - 1);
            let st = self.forward_one(context);
            let mut probs = st.logits;
            softmax(&mut probs);
            total -= probs[target].max(1e-10).ln();
        }
        total / count as f32
    }
}

fn cosine_lr(step: usize, max_steps: usize, base_lr: f32, warmup: usize) -> f32 {
    if step < warmup {
        return base_lr * step as f32 / warmup.max(1) as f32;
    }
    let p = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
    1e-5 + (base_lr - 1e-5) * 0.5 * (1.0 + (std::f32::consts::PI * p).cos())
}

fn load_data(path: &str) -> Vec<usize> {
    let raw = fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to load {}: {}. Using fallback.", path, e);
        b"The quick brown fox jumps over the lazy dog. ".repeat(100).to_vec()
    });
    assert!(!raw.is_empty(), "loaded data is empty");
    raw.into_iter().map(|b| (b as usize) % VOCAB).collect()
}

fn evaluate(model: &Model, tokens: &[usize]) -> f32 {
    let cl = model.ctx_len;
    let dl = tokens.len();
    let num_possible = dl.saturating_sub(SEQ + 1);
    if num_possible == 0 { return f32::MAX; }
    let nc = EVAL_CHUNKS.min(num_possible);
    let stride = num_possible / nc;
    if stride == 0 { return f32::MAX; }
    let mut total = 0.0f32;
    let mut n = 0usize;
    for c in 0..nc {
        let start = c * stride;
        let end = (start + SEQ + 1).min(dl);
        if end <= start + cl + 1 { continue; }
        let chunk = &tokens[start..end];
        total += model.loss_on_seq(chunk) / LN_2;
        n += 1;
    }
    if n == 0 { f32::MAX } else { total / n as f32 }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.iter().find(|a| a.starts_with("--seed="))
        .and_then(|a| a[7..].parse().ok()).unwrap_or(43);
    let steps: usize = args.iter().find(|a| a.starts_with("--steps="))
        .and_then(|a| a[8..].parse().ok()).unwrap_or(15000);
    let lr: f32 = args.iter().find(|a| a.starts_with("--lr="))
        .and_then(|a| a[5..].parse().ok()).unwrap_or(0.003);
    let dim: usize = args.iter().find(|a| a.starts_with("--dim="))
        .and_then(|a| a[6..].parse().ok()).unwrap_or(32);
    let hidden: usize = args.iter().find(|a| a.starts_with("--hidden="))
        .and_then(|a| a[9..].parse().ok()).unwrap_or(384);
    let ctx_len: usize = args.iter().find(|a| a.starts_with("--ctx="))
        .and_then(|a| a[6..].parse().ok()).unwrap_or(12);

    let train_data = load_data("data/tiny_shakespeare.txt");
    let val_data = load_data("data/tiny_shakespeare_val.txt");
    let train_end = (train_data.len() as f64 * 0.9) as usize;
    let train = &train_data[..train_end];
    let val = if val_data.len() > 100 { &val_data } else { &train_data[train_end..] };

    let mut model = Model::new(dim, hidden, ctx_len, seed);
    let cat_dim = ctx_len * dim;
    let h = hidden;
    let wd = 0.04f32;

    let mut opt_embed = AdamW::new(VOCAB * dim);
    let mut opt_pos = AdamW::new(ctx_len * dim);
    let mut opt_p1 = AdamW::new(h * cat_dim);
    let mut opt_p2 = AdamW::new(h * h);
    let mut opt_head = AdamW::new(VOCAB * h);

    let mut acc_embed = vec![0.0f32; VOCAB * dim];
    let mut acc_pos = vec![0.0f32; ctx_len * dim];
    let mut acc_p1 = vec![0.0f32; h * cat_dim];
    let mut acc_p2 = vec![0.0f32; h * h];
    let mut acc_head = vec![0.0f32; VOCAB * h];
    let accum = 4;

    let mut best_bpb = f32::MAX;
    let warmup = steps / 10;
    let start = Instant::now();
    let dl = train.len();

    let total_params = VOCAB * dim + ctx_len * dim + h * cat_dim + h * h + VOCAB * h;
    eprintln!(
        "concat: dim={} hidden={} ctx={} cat_dim={} lr={} seed={} steps={} params≈{}",
        dim, h, ctx_len, cat_dim, lr, seed, steps, total_params
    );

    for step in 1..=steps {
        let off = (step * 97 + seed as usize) % dl.saturating_sub(SEQ + 1);
        let seq = &train[off..off + SEQ + 1];
        let count = seq.len().saturating_sub(ctx_len + 1);
        if count == 0 { continue; }

        let step_count = 8.min(count);
        let stride = count / step_count;

        for si in 0..step_count {
            let i = si * stride;
            let context = &seq[i..i + ctx_len];
            let target = seq[i + ctx_len].min(VOCAB - 1);
            let st = model.forward_one(context);
            let (grads, _loss) = model.backward_one(&st, context, target);

            for j in 0..VOCAB * dim { acc_embed[j] += grads.g_embed[j]; }
            for j in 0..ctx_len * dim { acc_pos[j] += grads.g_pos[j]; }
            for j in 0..h * cat_dim { acc_p1[j] += grads.g_proj1[j]; }
            for j in 0..h * h { acc_p2[j] += grads.g_proj2[j]; }
            for j in 0..VOCAB * h { acc_head[j] += grads.g_head[j]; }
        }

        if step % accum == 0 || step == steps {
            let num_acc = if step % accum == 0 { accum * step_count } else { (step % accum) * step_count };
            let inv = 1.0 / num_acc as f32;
            for x in acc_embed.iter_mut() { *x *= inv; }
            for x in acc_pos.iter_mut() { *x *= inv; }
            for x in acc_p1.iter_mut() { *x *= inv; }
            for x in acc_p2.iter_mut() { *x *= inv; }
            for x in acc_head.iter_mut() { *x *= inv; }

            let cur_lr = cosine_lr(step, steps, lr, warmup);
            opt_embed.update(&mut model.embed, &acc_embed, cur_lr, wd);
            opt_pos.update(&mut model.pos_embed, &acc_pos, cur_lr, wd);
            opt_p1.update(&mut model.proj1, &acc_p1, cur_lr, wd);
            opt_p2.update(&mut model.proj2, &acc_p2, cur_lr, wd);
            opt_head.update(&mut model.lm_head, &acc_head, cur_lr, wd);

            for x in acc_embed.iter_mut() { *x = 0.0; }
            for x in acc_pos.iter_mut() { *x = 0.0; }
            for x in acc_p1.iter_mut() { *x = 0.0; }
            for x in acc_p2.iter_mut() { *x = 0.0; }
            for x in acc_head.iter_mut() { *x = 0.0; }
        }

        if step % EVAL_INTERVAL == 0 || step == steps {
            let elapsed = start.elapsed().as_secs_f64();
            let val_bpb = evaluate(&model, val);
            if val_bpb < best_bpb && val_bpb.is_finite() {
                best_bpb = val_bpb;
            }
            eprintln!(
                "step={:5} val_bpb={:.4} best={:.4} t={}s",
                step, val_bpb, best_bpb, elapsed as u64
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!("done: best_bpb={:.4} time={:.1}s", best_bpb, elapsed);
    println!("BPB={:.4}", best_bpb);
    Ok(())
}
