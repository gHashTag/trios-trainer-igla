#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::fs;
use std::time::Instant;

const VOCAB: usize = 128;
const DIM: usize = 64;
const SEQ: usize = 64;
const LN_2: f32 = std::f32::consts::LN_2;
const ACCUM: usize = 4;
const EVAL_INTERVAL: usize = 500;
const EVAL_SAMPLES: usize = 50;

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
        let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            step: 0,
            beta1: 1.0 / phi as f32,
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

struct ForwardState {
    combined: Vec<f32>,
    normed: Vec<f32>,
    hidden_raw: Vec<f32>,
    hidden: Vec<f32>,
    hidden_norm: Vec<f32>,
    projected: Vec<f32>,
    logits: Vec<f32>,
}

struct Model {
    embed: Vec<f32>,
    ctx: Vec<Vec<f32>>,
    ctx_weights: Vec<f32>,
    proj_up: Vec<f32>,
    proj_down: Vec<f32>,
    hidden_dim: usize,
    num_ctx: usize,
    ngram: usize,
}

impl Model {
    fn new(hidden_dim: usize, num_ctx: usize, seed: u64) -> Self {
        let ngram = num_ctx + 2;
        let mut s = seed;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let lim = (6.0f32 / (3 * DIM) as f32).sqrt();
        let lim_h = (6.0f32 / (DIM + hidden_dim) as f32).sqrt();
        let ctx_weights: Vec<f32> = (0..num_ctx)
            .map(|i| 0.7f32 * 0.45f32.powi(i as i32))
            .collect();
        Self {
            embed: (0..VOCAB * DIM).map(|_| rng() * lim).collect(),
            ctx: (0..num_ctx)
                .map(|_| (0..VOCAB * DIM).map(|_| rng() * lim).collect())
                .collect(),
            ctx_weights,
            proj_up: (0..hidden_dim * DIM).map(|_| rng() * lim_h).collect(),
            proj_down: (0..DIM * hidden_dim).map(|_| rng() * lim_h).collect(),
            hidden_dim,
            num_ctx,
            ngram,
        }
    }

    fn forward_one(&self, context: &[usize]) -> ForwardState {
        let h = self.hidden_dim;
        let ng = self.ngram;
        let t0 = context[ng - 1].min(VOCAB - 1);
        let mut combined = self.embed[t0 * DIM..(t0 + 1) * DIM].to_vec();
        for (ci, cw) in self.ctx_weights.iter().enumerate() {
            let idx = ng - 2 - ci;
            let t = context[idx].min(VOCAB - 1);
            let cv = &self.ctx[ci][t * DIM..(t + 1) * DIM];
            for j in 0..DIM {
                combined[j] += cv[j] * cw;
            }
        }
        let normed = layer_norm(&combined, 1e-5);
        let mut hidden_raw = vec![0.0f32; h];
        for hi in 0..h {
            for j in 0..DIM {
                hidden_raw[hi] += self.proj_up[hi * DIM + j] * normed[j];
            }
        }
        let hidden: Vec<f32> = hidden_raw.iter().map(|&v| v.max(0.0)).collect();
        let hidden_norm = layer_norm(&hidden, 1e-5);
        let mut projected = vec![0.0f32; DIM];
        for j in 0..DIM {
            for i in 0..h {
                projected[j] += self.proj_down[j * h + i] * hidden_norm[i];
            }
            projected[j] += normed[j];
        }
        let mut logits = vec![0.0f32; VOCAB];
        for v in 0..VOCAB {
            for j in 0..DIM {
                logits[v] += projected[j] * self.embed[v * DIM + j];
            }
        }
        ForwardState {
            combined,
            normed,
            hidden_raw,
            hidden,
            hidden_norm,
            projected,
            logits,
        }
    }
}

struct Grads {
    g_embed: Vec<f32>,
    g_ctx: Vec<Vec<f32>>,
    g_proj_up: Vec<f32>,
    g_proj_down: Vec<f32>,
}

fn compute_grads(model: &Model, tokens: &[usize]) -> (Grads, f32) {
    let h = model.hidden_dim;
    let ng = model.ngram;
    let nc = model.num_ctx;
    let count = tokens.len().saturating_sub(ng);
    if count == 0 {
        return (
            Grads {
                g_embed: vec![0.0; VOCAB * DIM],
                g_ctx: (0..nc).map(|_| vec![0.0; VOCAB * DIM]).collect(),
                g_proj_up: vec![0.0; h * DIM],
                g_proj_down: vec![0.0; DIM * h],
            },
            0.0,
        );
    }

    let mut g_embed = vec![0.0f32; VOCAB * DIM];
    let mut g_ctx: Vec<Vec<f32>> = (0..nc).map(|_| vec![0.0f32; VOCAB * DIM]).collect();
    let mut g_proj_up = vec![0.0f32; h * DIM];
    let mut g_proj_down = vec![0.0f32; DIM * h];
    let mut total_loss = 0.0f32;

    for i in 0..count {
        let context = &tokens[i..i + ng];
        let target = tokens[i + ng].min(VOCAB - 1);
        let st = model.forward_one(context);
        let mut probs = st.logits.clone();
        softmax(&mut probs);
        total_loss -= probs[target].max(1e-10).ln();

        let mut d_logits = probs;
        d_logits[target] -= 1.0;

        let mut d_proj = vec![0.0f32; DIM];
        for v in 0..VOCAB {
            let dl = d_logits[v];
            for j in 0..DIM {
                d_proj[j] += dl * model.embed[v * DIM + j];
                g_embed[v * DIM + j] += dl * st.projected[j];
            }
        }

        let mut d_hn = vec![0.0f32; h];
        let mut d_normed_res = vec![0.0f32; DIM];
        for j in 0..DIM {
            for i in 0..h {
                g_proj_down[j * h + i] += d_proj[j] * st.hidden_norm[i];
                d_hn[i] += d_proj[j] * model.proj_down[j * h + i];
            }
            d_normed_res[j] = d_proj[j];
        }

        let d_hid = layer_norm_backward(&st.hidden, &st.hidden_norm, &d_hn, 1e-5);

        let mut d_hr = vec![0.0f32; h];
        for i in 0..h {
            d_hr[i] = if st.hidden_raw[i] > 0.0 {
                d_hid[i]
            } else {
                0.0
            };
        }

        let mut d_normed_proj = vec![0.0f32; DIM];
        for hi in 0..h {
            if d_hr[hi] == 0.0 {
                continue;
            }
            for j in 0..DIM {
                g_proj_up[hi * DIM + j] += d_hr[hi] * st.normed[j];
                d_normed_proj[j] += d_hr[hi] * model.proj_up[hi * DIM + j];
            }
        }

        let mut d_normed = vec![0.0f32; DIM];
        for j in 0..DIM {
            d_normed[j] = d_normed_proj[j] + d_normed_res[j];
        }

        let d_comb = layer_norm_backward(&st.combined, &st.normed, &d_normed, 1e-5);

        let t0 = context[ng - 1].min(VOCAB - 1);
        for j in 0..DIM {
            g_embed[t0 * DIM + j] += d_comb[j];
        }
        for (ci, cw) in model.ctx_weights.iter().enumerate() {
            let idx = ng - 2 - ci;
            let t = context[idx].min(VOCAB - 1);
            for j in 0..DIM {
                g_ctx[ci][t * DIM + j] += cw * d_comb[j];
            }
        }
    }

    let n = count as f32;
    for x in g_embed.iter_mut() {
        *x /= n;
    }
    for gc in g_ctx.iter_mut() {
        for x in gc.iter_mut() {
            *x /= n;
        }
    }
    for x in g_proj_up.iter_mut() {
        *x /= n;
    }
    for x in g_proj_down.iter_mut() {
        *x /= n;
    }

    (
        Grads {
            g_embed,
            g_ctx,
            g_proj_up,
            g_proj_down,
        },
        total_loss / n,
    )
}

fn evaluate(model: &Model, tokens: &[usize]) -> f32 {
    let ng = model.ngram;
    let dl = tokens.len();
    let num_possible = dl.saturating_sub(SEQ + 1);
    if num_possible == 0 {
        return f32::MAX;
    }
    let num_chunks = EVAL_SAMPLES.min(num_possible);
    let stride = num_possible / num_chunks;
    if stride == 0 {
        return f32::MAX;
    }
    let mut total = 0.0f32;
    let mut n = 0usize;
    for c in 0..num_chunks {
        let start = c * stride;
        let end = start + SEQ + 1;
        if end > dl {
            continue;
        }
        let chunk = &tokens[start..end];
        let cnt = chunk.len().saturating_sub(ng);
        if cnt == 0 {
            continue;
        }
        let mut loss = 0.0f32;
        for i in 0..cnt {
            let context = &chunk[i..i + ng];
            let target = chunk[i + ng].min(VOCAB - 1);
            let st = model.forward_one(context);
            let mut probs = st.logits.clone();
            softmax(&mut probs);
            loss -= probs[target].max(1e-10).ln();
        }
        total += loss / cnt as f32 / LN_2;
        n += 1;
    }
    if n == 0 {
        f32::MAX
    } else {
        total / n as f32
    }
}

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

fn find_arg<T: std::str::FromStr>(args: &[String], prefix: &str, default: T) -> T {
    args.iter()
        .find(|a| a.starts_with(prefix))
        .and_then(|a| a[prefix.len()..].parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = find_arg(&args, "--seed=", 43);
    let steps: usize = find_arg(&args, "--steps=", 27000);
    let lr: f32 = find_arg(&args, "--lr=", 0.003);
    let hidden_dim: usize = find_arg(&args, "--hidden=", 512);
    let num_ctx: usize = find_arg(&args, "--ctx=", 8);
    let ngram = num_ctx + 2;

    let train_data = load_data("data/tiny_shakespeare.txt");
    let val_data = load_data("data/tiny_shakespeare_val.txt");
    let train_end = (train_data.len() as f64 * 0.9) as usize;
    let train = &train_data[..train_end];
    let val = if val_data.len() > 100 {
        &val_data
    } else {
        &train_data[train_end..]
    };

    let mut model = Model::new(hidden_dim, num_ctx, seed);
    let h = hidden_dim;
    let wd = 0.04f32;
    let mut opt_embed = AdamW::new(VOCAB * DIM, wd);
    let mut opt_ctx: Vec<AdamW> = (0..num_ctx).map(|_| AdamW::new(VOCAB * DIM, wd)).collect();
    let mut opt_proj_up = AdamW::new(h * DIM, wd);
    let mut opt_proj_down = AdamW::new(DIM * h, wd);

    let mut acc_embed = vec![0.0f32; VOCAB * DIM];
    let mut acc_ctx: Vec<Vec<f32>> = (0..num_ctx).map(|_| vec![0.0f32; VOCAB * DIM]).collect();
    let mut acc_proj_up = vec![0.0f32; h * DIM];
    let mut acc_proj_down = vec![0.0f32; DIM * h];

    let mut best_bpb = f32::MAX;
    let warmup = steps / 10;
    let start = Instant::now();
    let dl = train.len();

    eprintln!(
        "train_v2: dim={} hidden={} ctx={} ngram={} lr={} seed={} steps={}",
        DIM, h, num_ctx, ngram, lr, seed, steps
    );

    for step in 1..=steps {
        let off = (step * 97 + seed as usize) % dl.saturating_sub(SEQ + 1);
        let seq = &train[off..off + SEQ + 1];

        let (grads, _loss) = compute_grads(&model, seq);

        for i in 0..VOCAB * DIM {
            acc_embed[i] += grads.g_embed[i];
        }
        for (ci, ac) in acc_ctx.iter_mut().enumerate() {
            for i in 0..VOCAB * DIM {
                ac[i] += grads.g_ctx[ci][i];
            }
        }
        for i in 0..h * DIM {
            acc_proj_up[i] += grads.g_proj_up[i];
        }
        for i in 0..DIM * h {
            acc_proj_down[i] += grads.g_proj_down[i];
        }

        if step % ACCUM == 0 || step == steps {
            let num_acc = if step % ACCUM == 0 {
                ACCUM
            } else {
                step % ACCUM
            };
            let inv = 1.0 / num_acc as f32;
            for x in acc_embed.iter_mut() {
                *x *= inv;
            }
            for ac in acc_ctx.iter_mut() {
                for x in ac.iter_mut() {
                    *x *= inv;
                }
            }
            for x in acc_proj_up.iter_mut() {
                *x *= inv;
            }
            for x in acc_proj_down.iter_mut() {
                *x *= inv;
            }

            let cur_lr = cosine_lr(step, steps, lr, warmup);
            opt_embed.update(&mut model.embed, &acc_embed, cur_lr);
            for (ci, oc) in opt_ctx.iter_mut().enumerate() {
                oc.update(&mut model.ctx[ci], &acc_ctx[ci], cur_lr);
            }
            opt_proj_up.update(&mut model.proj_up, &acc_proj_up, cur_lr);
            opt_proj_down.update(&mut model.proj_down, &acc_proj_down, cur_lr);

            for x in acc_embed.iter_mut() {
                *x = 0.0;
            }
            for ac in acc_ctx.iter_mut() {
                for x in ac.iter_mut() {
                    *x = 0.0;
                }
            }
            for x in acc_proj_up.iter_mut() {
                *x = 0.0;
            }
            for x in acc_proj_down.iter_mut() {
                *x = 0.0;
            }
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
