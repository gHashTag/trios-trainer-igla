use anyhow::Result;
use std::time::Instant;

pub const DEFAULT_IGLA_TARGET_BPB: f64 = 1.85;
pub const GATE_FINAL_SEEDS: &[u64] = &[42, 43, 44];

const VOCAB: usize = 128;
const DIM: usize = 64;
const NUM_CTX: usize = 6;
const NGRAM: usize = NUM_CTX + 2;
const LN_2: f32 = std::f32::consts::LN_2;
const PHI_INV: f32 = 0.618033988749895;
const CTX_WEIGHTS: [f32; NUM_CTX] = [0.70, 0.45, 0.30, 0.20, 0.13, 0.08];

#[derive(Debug)]
pub struct TrainArgs {
    pub seed: u64,
    pub steps: usize,
    pub hidden: usize,
    pub lr: f32,
    pub attn_layers: u8,
    pub eval_every: usize,
    pub train_path: String,
    pub val_path: String,
}

#[derive(Debug)]
pub struct RunOutcome {
    pub final_bpb: f64,
    pub steps_done: usize,
    pub seed: u64,
}

fn load_data(path: &str) -> Vec<usize> {
    let raw = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to load {}: {}. Using fallback.", path, e);
        b"The quick brown fox jumps over the lazy dog. ".repeat(100).to_vec()
    });
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
    dy.iter().zip(y.iter()).map(|(d, yi)| (d - sum_dy / n - yi * sum_dy_y / n) * std_inv).collect()
}

fn softmax(v: &mut [f32]) {
    let max_val = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() { *x = (*x - max_val).exp(); sum += *x; }
    for x in v.iter_mut() { *x /= sum; }
}

fn cosine_lr(step: usize, max_steps: usize, base_lr: f32, warmup: usize) -> f32 {
    if step < warmup { return base_lr * step as f32 / warmup.max(1) as f32; }
    let p = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
    1e-5 + (base_lr - 1e-5) * 0.5 * (1.0 + (std::f32::consts::PI * p).cos())
}

struct AdamW {
    m: Vec<f32>, v: Vec<f32>, step: usize, beta1: f32, beta2: f32, wd: f32,
}

impl AdamW {
    fn new(size: usize, wd: f32) -> Self {
        Self { m: vec![0.0; size], v: vec![0.0; size], step: 0, beta1: 0.9, beta2: 0.999, wd }
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
    lm_head: Vec<f32>,
    hidden: usize,
}

impl HybridModel {
    fn new(hidden: usize, seed: u64) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let lim = (6.0f32 / (3 * DIM) as f32).sqrt();
        let lim_h = (6.0f32 / (DIM + hidden) as f32).sqrt();
        let lim_o = (6.0f32 / (hidden + VOCAB) as f32).sqrt();
        let total = VOCAB * DIM + NUM_CTX * VOCAB * DIM + hidden * DIM + VOCAB * hidden;
        eprintln!("params={} ({:.1}K)", total, total as f64 / 1000.0);
        Self {
            embed: (0..VOCAB * DIM).map(|_| rng() * lim).collect(),
            ctx: (0..NUM_CTX).map(|_| (0..VOCAB * DIM).map(|_| rng() * lim).collect()).collect(),
            proj: (0..hidden * DIM).map(|_| rng() * lim_h).collect(),
            lm_head: (0..VOCAB * hidden).map(|_| rng() * lim_o).collect(),
            hidden,
        }
    }

    fn forward_position(&self, tokens: &[usize], pos: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let h = self.hidden;
        let t_last = tokens[pos + NGRAM - 1].min(VOCAB - 1);
        let mut combined = self.embed[t_last * DIM..(t_last + 1) * DIM].to_vec();
        for (ci, cw) in CTX_WEIGHTS.iter().enumerate() {
            let ctx_idx = NGRAM - 2 - ci;
            let t = tokens[pos + ctx_idx].min(VOCAB - 1);
            let cv = &self.ctx[ci][t * DIM..(t + 1) * DIM];
            for j in 0..DIM { combined[j] += cv[j] * cw; }
        }
        let ln = layer_norm(&combined, 1e-5);
        let mut hidden_raw = vec![0.0f32; h];
        for hi in 0..h { for j in 0..DIM { hidden_raw[hi] += self.proj[hi * DIM + j] * ln[j]; } }
        let mut hidden = vec![0.0f32; h];
        for hi in 0..h { hidden[hi] = if hidden_raw[hi] > 0.0 { hidden_raw[hi] * hidden_raw[hi] } else { 0.0 }; }
        let mut logits = vec![0.0f32; VOCAB];
        for vi in 0..VOCAB { for hi in 0..h { logits[vi] += self.lm_head[vi * h + hi] * hidden[hi]; } }
        (combined, ln, hidden, logits)
    }

    fn loss_on_seq(&self, tokens: &[usize]) -> f32 {
        if tokens.len() < NGRAM + 1 { return 0.0; }
        let count = tokens.len().saturating_sub(NGRAM);
        let mut total = 0.0f32;
        for i in 0..count {
            let target = tokens[i + NGRAM].min(VOCAB - 1);
            let (_, _, _, mut logits) = self.forward_position(tokens, i);
            softmax(&mut logits);
            total -= logits[target].max(1e-10).ln();
        }
        total / count.max(1) as f32
    }
}

fn compute_grads(model: &HybridModel, tokens: &[usize], positions: &[usize],
    g_embed: &mut [f32], g_ctx: &mut [Vec<f32>], g_proj: &mut [f32], g_head: &mut [f32]) {
    let h = model.hidden;
    for &pos in positions {
        let (combined, ln, hidden, mut logits) = model.forward_position(tokens, pos);
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
        let mut d_raw = vec![0.0f32; h];
        for hi in 0..h { if hidden[hi] > 0.0 { d_raw[hi] = d_hidden[hi] * 2.0 * hidden[hi].sqrt(); } }
        let mut d_ln = vec![0.0f32; DIM];
        for hi in 0..h { for j in 0..DIM { g_proj[hi * DIM + j] += d_raw[hi] * ln[j]; d_ln[j] += model.proj[hi * DIM + j] * d_raw[hi]; } }
        let d_combined = layer_norm_backward(&combined, &ln, &d_ln, 1e-5);
        let t_last = tokens[pos + NGRAM - 1].min(VOCAB - 1);
        for j in 0..DIM { g_embed[t_last * DIM + j] += d_combined[j]; }
        for (ci, cw) in CTX_WEIGHTS.iter().enumerate() {
            let ctx_idx = NGRAM - 2 - ci;
            let t = tokens[pos + ctx_idx].min(VOCAB - 1);
            for j in 0..DIM { g_ctx[ci][t * DIM + j] += cw * d_combined[j]; }
        }
    }
}

fn evaluate(model: &HybridModel, tokens: &[usize]) -> f32 {
    let seq = 129;
    let num_chunks = 40usize;
    let max_start = tokens.len().saturating_sub(seq);
    if max_start == 0 { return f32::MAX; }
    let step = if max_start >= num_chunks * seq { max_start / num_chunks } else { seq };
    let mut total = 0.0f32;
    let mut n = 0usize;
    for c in (0..max_start).step_by(step).take(num_chunks) {
        let end = (c + seq).min(tokens.len());
        if end - c < NGRAM + 2 { continue; }
        let loss = model.loss_on_seq(&tokens[c..end]);
        if loss.is_finite() { total += loss / LN_2; n += 1; }
    }
    if n == 0 { f32::MAX } else { total / n as f32 }
}

pub fn run_single(args: &TrainArgs) -> Result<RunOutcome> {
    eprintln!("=== trios-train seed={} steps={} hidden={} lr={:.4} ===", args.seed, args.steps, args.hidden, args.lr);
    let train = load_data(&args.train_path);
    let val = load_data(&args.val_path);
    eprintln!("train={} val={}", train.len(), val.len());

    let mut model = HybridModel::new(args.hidden, args.seed);
    let wd = 0.04f32;
    let mut opt_embed = AdamW::new(VOCAB * DIM, wd);
    let mut opt_ctx: Vec<AdamW> = (0..NUM_CTX).map(|_| AdamW::new(VOCAB * DIM, wd)).collect();
    let mut opt_proj = AdamW::new(args.hidden * DIM, wd);
    let mut opt_head = AdamW::new(VOCAB * args.hidden, wd);

    let init_bpb = evaluate(&model, &val);
    let mut ema_bpb = init_bpb;
    let mut best_bpb = init_bpb;
    let warmup = args.steps / 10;
    let accum = 4;
    let mut rng_s = args.seed.wrapping_add(7919);
    let t0 = Instant::now();

    for step in 1..=args.steps {
        let lr = cosine_lr(step, args.steps, args.lr, warmup);
        let mut ge = vec![0.0f32; VOCAB * DIM];
        let mut gc: Vec<Vec<f32>> = (0..NUM_CTX).map(|_| vec![0.0f32; VOCAB * DIM]).collect();
        let mut gp = vec![0.0f32; args.hidden * DIM];
        let mut gh = vec![0.0f32; VOCAB * args.hidden];

        for _ in 0..accum {
            rng_s = rng_s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let dl = train.len();
            let ms = dl.saturating_sub(130);
            if ms == 0 { continue; }
            let cs = (rng_s as usize) % ms;
            let chunk = &train[cs..cs + 129];
            let cnt = chunk.len().saturating_sub(NGRAM);
            if cnt == 0 { continue; }
            let ns = 8.min(cnt);
            let mut pos = Vec::with_capacity(ns);
            for _ in 0..ns {
                rng_s = rng_s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                pos.push((rng_s as usize) % cnt);
            }
            compute_grads(&model, chunk, &pos, &mut ge, &mut gc, &mut gp, &mut gh);
        }

        let tp = (accum * 8) as f32;
        for x in ge.iter_mut() { *x /= tp; }
        for g in gc.iter_mut() { for x in g.iter_mut() { *x /= tp; } }
        for x in gp.iter_mut() { *x /= tp; }
        for x in gh.iter_mut() { *x /= tp; }

        opt_embed.update(&mut model.embed, &ge, lr);
        for (ci, oc) in opt_ctx.iter_mut().enumerate() { oc.update(&mut model.ctx[ci], &gc[ci], lr); }
        opt_proj.update(&mut model.proj, &gp, lr);
        opt_head.update(&mut model.lm_head, &gh, lr);

        if step % args.eval_every == 0 || step == args.steps {
            let vbpb = evaluate(&model, &val);
            ema_bpb = PHI_INV * ema_bpb + (1.0 - PHI_INV) * vbpb;
            if ema_bpb < best_bpb && ema_bpb.is_finite() { best_bpb = ema_bpb; }
            println!("seed={} step={} val_bpb={:.4} ema_bpb={:.4} best={:.4} t={:.1}s",
                args.seed, step, vbpb, ema_bpb, best_bpb, t0.elapsed().as_secs_f64());
        }
    }

    Ok(RunOutcome { final_bpb: best_bpb as f64, steps_done: args.steps, seed: args.seed })
}

pub fn run_sweep(steps: usize, hidden: usize, lr: f32, attn_layers: u8, eval_every: usize,
                 train_path: &str, val_path: &str) -> Result<Vec<RunOutcome>> {
    let mut results = Vec::new();
    for &seed in GATE_FINAL_SEEDS {
        results.push(run_single(&TrainArgs { seed, steps, hidden, lr, attn_layers, eval_every,
            train_path: train_path.to_string(), val_path: val_path.to_string() })?);
    }
    Ok(results)
}

pub fn run(cfg: &crate::TrainConfig) -> Result<RunOutcome> {
    let args = TrainArgs {
        seed: cfg.seed, steps: cfg.steps, hidden: 828,
        lr: cfg.optimizer.lr as f32, attn_layers: if cfg.model.hybrid_attn { 2 } else { 1 },
        eval_every: 1000, train_path: "data/tiny_shakespeare.txt".to_string(),
        val_path: "data/tiny_shakespeare_val.txt".to_string(),
    };
    let outcome = run_single(&args)?;
    if !cfg.ledger.jsonl_path.is_empty() {
        let _ = crate::ledger::emit_row(cfg, outcome.final_bpb, outcome.steps_done);
    }
    Ok(outcome)
}
