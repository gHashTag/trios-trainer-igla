//! ARCH-EXPLORER: IGLA Architecture Exploration Agent
//!
//! Runs 5 architectural trials for finding BPB < 2.52:
//!   Trial X1: 6-gram h=384 + weight tying (embed=output weights)
//!   Trial X2: 6-gram h=384 + cosine lr schedule (vs constant)
//!   Trial X3: 6-gram h=320 (smaller, faster)
//!   Trial X4: 6-gram h=384 + gradient clipping 0.5
//!   Trial X5: 6-gram h=384 + warmup 500 steps then lr=0.004
//!
//! Usage:
//!   cargo run --release --bin arch_explorer -- --trial X1
//!   cargo run --release --bin arch_explorer -- --all

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use std::fs;
use std::io::Write;
use std::time::Instant;
use std::env;

const VOCAB: usize = 128;
const DIM: usize = 64;
const SEQ: usize = 64;
const LN_2: f32 = std::f32::consts::LN_2;
const MACHINE_ID: &str = "mac-arch-explorer";

// Trial configurations
#[derive(Debug, Clone)]
struct TrialConfig {
    name: String,
    hidden: usize,
    weight_tying: bool,
    cosine_lr: bool,
    gradient_clip: Option<f32>,
    warmup_steps: usize,
    base_lr: f32,
}

fn gelu(x: f32) -> f32 {
    let x3 = x * x * x;
    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
    0.5 * x * (1.0 + tanh_arg.tanh())
}

fn activate(x: f32, name: &str) -> f32 {
    match name {
        "gelu" => gelu(x),
        _ => x.max(0.0),
    }
}

fn activate_grad(x: f32, name: &str) -> f32 {
    match name {
        "gelu" => gelu(x),
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

struct AdamW {
    m: Vec<f32>, v: Vec<f32>, step: usize,
    beta1: f32, beta2: f32, eps: f32, wd: f32,
}

impl AdamW {
    fn new(size: usize, wd: f32) -> Self {
        let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
        Self { m: vec![0.0; size], v: vec![0.0; size], step: 0,
            beta1: 1.0 / phi as f32, beta2: 0.999, eps: 1e-8, wd }
    }
    fn update(&mut self, params: &mut [f32], grads: &[f32], lr: f32, clip: Option<f32>) {
        self.step += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);
        for i in 0..params.len() {
            let mut grad = grads[i];
            if let Some(c) = clip {
                grad = grad.max(-c).min(c);
            }
            params[i] -= self.wd * lr * params[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad * grad;
            params[i] -= lr * (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + self.eps);
        }
    }
}

struct NgramModel {
    embed: Vec<f32>,
    ctx: Vec<Vec<f32>>,
    ctx_weights: Vec<f32>,
    proj: Vec<f32>,
    lm_head: Vec<f32>,
    vocab: usize,
    dim: usize,
    hidden: usize,
    activation: String,
    weight_tying: bool,
}

impl NgramModel {
    fn new(vocab: usize, dim: usize, hidden: usize, activation: String, seed: u64, num_ctx: usize, weight_tying: bool) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let lim = (6.0f32 / (3 * dim) as f32).sqrt();
        let lim_h = (6.0f32 / (dim + hidden) as f32).sqrt();

        let ctx = (0..num_ctx).map(|_| {
            (0..vocab * dim).map(|_| rng() * lim).collect()
        }).collect();

        let base_weights: Vec<f32> = vec![0.7, 0.3, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06];
        let ctx_weights: Vec<f32> = base_weights.iter().take(num_ctx).cloned().collect();

        let embed: Vec<f32> = (0..vocab * dim).map(|_| rng() * lim).collect();

        // lm_head (weight tying disabled for now - needs h == dim for proper tying)
        let lim_o = (6.0f32 / (hidden + dim) as f32).sqrt();
        let lm_head: Vec<f32> = (0..vocab * hidden).map(|_| rng() * lim_o).collect();

        Self {
            embed,
            ctx,
            ctx_weights,
            proj: (0..hidden * dim).map(|_| rng() * lim_h).collect(),
            lm_head,
            vocab, dim, hidden, activation, weight_tying,
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

    fn predict(&self, hidden: &[f32]) -> Vec<f32> {
        let v = self.vocab;
        let h = self.hidden;
        let mut logits = vec![0.0f32; v];

        for (vi, logit) in logits.iter_mut().enumerate() {
            let w = &self.lm_head[vi * h..(vi + 1) * h];
            for (hi, hn) in hidden.iter().enumerate() { *logit += w[hi] * hn; }
        }
        logits
    }

    fn loss_on_seq(&self, tokens: &[usize]) -> f32 {
        let num_ctx = self.ctx.len();
        let ngram = num_ctx + 2;
        if tokens.len() < ngram + 1 { return 0.0; }
        let count = tokens.len() - ngram;
        let mut total = 0.0f32;

        for i in 0..count {
            let context = &tokens[i..i + ngram];
            let hidden = self.compute_hidden(context);
            let target = tokens[i + ngram].min(self.vocab - 1);
            let mut logits = self.predict(&hidden);
            softmax(&mut logits);
            total -= logits[target].max(1e-10).ln();
        }
        total / count as f32
    }

    #[allow(clippy::needless_range_loop)]
    fn train_step(&mut self, tokens: &[usize], lr: f32,
        opt_embed: &mut AdamW, opt_ctx: &mut [AdamW], opt_proj: &mut AdamW, opt_head: &mut AdamW,
        clip: Option<f32>) {
        let num_ctx = self.ctx.len();
        let ngram = num_ctx + 2;
        if tokens.len() < ngram + 1 { return; }
        let v = self.vocab;
        let d = self.dim;
        let h = self.hidden;
        let count = tokens.len() - ngram;

        let mut g_embed = vec![0.0f32; v * d];
        let mut g_ctx: Vec<Vec<f32>> = (0..num_ctx).map(|_| vec![0.0f32; v * d]).collect();
        let mut g_proj = vec![0.0f32; h * d];
        let mut g_head = vec![0.0f32; v * h];

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
            let hidden = &all_hidden[i];
            let mut d_hidden = vec![0.0f32; h];  // Always h since it comes from activation

            // Compute logits and gradients (no weight tying for now)
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
                    d_hidden[hi] += grad * self.lm_head[vi * h + hi];
                }
            }

            let act_grads: Vec<f32> = all_pre_act[i].iter().map(|&pv| activate_grad(pv, &self.activation)).collect();

            // Backprop through proj and layer norm
            for hi in 0..h {
                for di in 0..d {
                    g_proj[hi * d + di] += d_hidden[hi] * act_grads[hi] * all_ln[i][di];
                }
            }

            // Backprop to embed and ctx
            let t0 = all_contexts[i][ngram - 1].min(v - 1);
            for di in 0..d {
                let mut grad_sum = 0.0f32;
                for hi in 0..h {
                    grad_sum += self.proj[hi * d + di] * act_grads[hi] * d_hidden[hi];
                }
                g_embed[t0 * d + di] += grad_sum;
                for (ci, cw) in self.ctx_weights.iter().enumerate() {
                    let ctx_idx = ngram - 2 - ci;
                    let t = all_contexts[i][ctx_idx].min(v - 1);
                    g_ctx[ci][t * d + di] += cw * grad_sum;
                }
            }
        }

        let n = count as f32;
        for x in g_embed.iter_mut() { *x /= n; }
        for gc in g_ctx.iter_mut() { for x in gc.iter_mut() { *x /= n; } }
        for x in g_proj.iter_mut() { *x /= n; }
        for x in g_head.iter_mut() { *x /= n; }

        opt_embed.update(&mut self.embed, &g_embed, lr, clip);
        for (ci, oc) in opt_ctx.iter_mut().enumerate() {
            oc.update(&mut self.ctx[ci], &g_ctx[ci], lr, clip);
        }
        opt_proj.update(&mut self.proj, &g_proj, lr, clip);
        if !self.weight_tying {
            opt_head.update(&mut self.lm_head, &g_head, lr, clip);
        }
    }
}

fn evaluate(model: &NgramModel, tokens: &[usize], seq_len: usize) -> (f32, f32) {
    let mut total = 0.0f32;
    let mut n = 0usize;
    for c in (0..tokens.len()).step_by(seq_len + 1) {
        let end = (c + seq_len + 1).min(tokens.len());
        if end - c < model.ctx.len() + 3 { continue; }
        let loss = model.loss_on_seq(&tokens[c..end]);
        if loss.is_finite() { total += loss / LN_2; n += 1; }
    }
    if n == 0 { return (f32::MAX, f32::MAX); }
    let bpb = total / n as f32;
    (bpb * LN_2, bpb)
}

fn get_lr(step: usize, max_steps: usize, base_lr: f32, warmup: usize, cosine: bool) -> f32 {
    if cosine {
        if step < warmup {
            base_lr * step as f32 / warmup as f32
        } else {
            let p = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
            1e-5 + (base_lr - 1e-5) * 0.5 * (1.0 + (std::f32::consts::PI * p).cos())
        }
    } else {
        base_lr
    }
}

fn write_experience(trial_name: &str, config: &TrialConfig, best_bpb: f32, steps: usize, duration_sec: f64, outcome: &str) {
    let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
    let ep = format!(".trinity/experience/trios_{}.trinity", chrono::Utc::now().format("%Y%m%d"));
    let _ = fs::create_dir_all(".trinity/experience");

    let mut entry = format!(
        "[{}] ARCH-EXPLORER | trial={} | outcome={} | h={} | lr={:.6}",
        ts, trial_name, outcome, config.hidden, config.base_lr
    );

    if config.weight_tying { entry.push_str(" | weight_tying=true"); }
    if config.cosine_lr { entry.push_str(" | cosine_lr=true"); }
    if let Some(c) = config.gradient_clip { entry.push_str(&format!(" | grad_clip={:.2}", c)); }
    if config.warmup_steps > 0 { entry.push_str(&format!(" | warmup={}", config.warmup_steps)); }

    entry.push_str(&format!(" | bpb={:.4} | steps={} | {:.1}s\n", best_bpb, steps, duration_sec));

    let _ = fs::OpenOptions::new().create(true).append(true)
        .open(&ep).unwrap().write_all(entry.as_bytes());
}

fn run_trial(config: TrialConfig, seed: u64, max_steps: usize, prune_step: usize, prune_threshold: f32) -> (f32, usize, String, f64) {
    let ngram_order = 6;  // Fixed at 6-gram for all trials
    let num_ctx = 4;

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║  ARCH-EXPLORER TRIAL: {:<45} ║", config.name);
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║  {}-gram | h={} | lr={:.6} | seed={} {:<18}║", ngram_order, config.hidden, config.base_lr, seed,
        if config.weight_tying { "| tying" } else { "" });
    println!("║  cosine={} | clip={:<5} | warmup={:<4}",
        config.cosine_lr,
        config.gradient_clip.map(|c| format!("{:.2}", c)).unwrap_or_else(|| "none".to_string()),
        config.warmup_steps);
    println!("║                                                      ║");

    let tokens = load_data("data/tinyshakespeare.txt");
    let train_end = (tokens.len() as f64 * 0.9) as usize;
    let train = &tokens[..train_end];
    let val = &tokens[train_end..];

    let mut model = NgramModel::new(VOCAB, DIM, config.hidden, "relu".to_string(), seed, num_ctx, config.weight_tying);
    let ps = VOCAB * DIM;
    let mut opt_embed = AdamW::new(ps, 0.01);
    let mut opt_ctx: Vec<AdamW> = (0..num_ctx).map(|_| AdamW::new(ps, 0.01)).collect();

    let proj_size = if config.weight_tying { config.hidden * DIM } else { DIM * config.hidden };
    let mut opt_proj = AdamW::new(proj_size, 0.01);

    let head_size = if config.weight_tying { VOCAB * DIM } else { VOCAB * config.hidden };
    let mut opt_head = AdamW::new(head_size, 0.01);

    let (init_loss, init_bpb) = evaluate(&model, val, SEQ);
    println!("\nInitial val: loss={:.4} bpb={:.4}", init_loss, init_bpb);
    println!("\n{:>6} | {:>10} | {:>10} | {:>10} | {:>8}", "step", "val_loss", "val_bpb", "best_bpb", "lr");
    println!("{}", "-".repeat(60));

    let t0 = Instant::now();
    let mut best_bpb = init_bpb;
    let mut best_step = 0;
    let mut pruned = false;
    let mut results: Vec<(usize, f32, f32)> = Vec::new();
    let dl = train.len();

    for step in 1..=max_steps {
        let lr = get_lr(step, max_steps, config.base_lr, config.warmup_steps, config.cosine_lr);
        let off = (step * 97 + seed as usize) % (dl.saturating_sub(SEQ + 1));
        model.train_step(&train[off..off + SEQ + 1], lr,
            &mut opt_embed, &mut opt_ctx, &mut opt_proj, &mut opt_head,
            config.gradient_clip);

        if step % 500 == 0 || step == max_steps || step == prune_step {
            let _ms = t0.elapsed().as_millis();
            let (vl, vb) = evaluate(&model, val, SEQ);
            if vb < best_bpb && vb.is_finite() {
                best_bpb = vb;
                best_step = step;
            }
            println!("{:>6} | {:>10.4} | {:>10.4} | {:>10.4} | {:.6}", step, vl, vb, best_bpb, lr);
            results.push((step, vl, vb));

            // Pruning check
            if step == prune_step && vb > prune_threshold {
                println!("\n✂️  PRUNED at step {}: BPB={:.4} > {:.4}", step, vb, prune_threshold);
                pruned = true;
                break;
            }
        }
    }

    let total = t0.elapsed();
    let outcome = if pruned {
        format!("PRUNED at step {}", prune_step)
    } else {
        format!("COMPLETED {} steps", max_steps)
    };

    println!("\n=== TRIAL {} DONE ===", config.name);
    println!("Outcome: {}", outcome);
    println!("BPB: {:.4} → {:.4} | Delta: {:.4}", init_bpb, best_bpb, best_bpb - init_bpb);
    println!("Time: {:.1}s", total.as_secs_f64());

    write_experience(&config.name, &config, best_bpb, best_step, total.as_secs_f64(), &outcome);

    (best_bpb, best_step, outcome, total.as_secs_f64())
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Trial configurations
    let trials = [
        TrialConfig {
            name: "X1".to_string(),
            hidden: 384,
            weight_tying: false,     // DISABLED: needs h == dim for proper tying
            cosine_lr: false,
            gradient_clip: None,
            warmup_steps: 0,
            base_lr: 0.004,
        },
        TrialConfig {
            name: "X2".to_string(),
            hidden: 384,
            weight_tying: false,
            cosine_lr: true,         // Cosine LR schedule
            gradient_clip: None,
            warmup_steps: 0,
            base_lr: 0.004,
        },
        TrialConfig {
            name: "X3".to_string(),
            hidden: 320,             // Smaller hidden
            weight_tying: false,
            cosine_lr: false,
            gradient_clip: None,
            warmup_steps: 0,
            base_lr: 0.004,
        },
        TrialConfig {
            name: "X4".to_string(),
            hidden: 384,
            weight_tying: false,
            cosine_lr: false,
            gradient_clip: Some(0.5),  // Gradient clipping
            warmup_steps: 0,
            base_lr: 0.004,
        },
        TrialConfig {
            name: "X5".to_string(),
            hidden: 384,
            weight_tying: false,
            cosine_lr: false,
            gradient_clip: None,
            warmup_steps: 500,       // Warmup 500 steps
            base_lr: 0.004,
        },
    ];

    let max_steps = 5000;
    let prune_step = 3000;
    let prune_threshold = trios_trainer::invariants::ASHA_PRUNE_THRESHOLD as f32;
    let target_bpb = 2.52;

    // Determine which trial(s) to run
    let trial_idx = if args.len() > 1 && args[1] == "--all" {
        None  // Run all
    } else if let Some(idx) = args.iter().find(|a| a.starts_with("--trial=")).map(|a| {
        a[8..].parse::<usize>().unwrap_or(0)
    }) {
        if idx > 0 && idx <= trials.len() {
            Some(idx - 1)
        } else {
            println!("Invalid trial index. Running all trials.");
            None
        }
    } else {
        None  // Default: run all
    };

    let trials_to_run: Vec<_> = if let Some(idx) = trial_idx {
        vec![&trials[idx]]
    } else {
        trials.iter().collect()
    };

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║  🏎️  ARCH-EXPLORER (АГЕНТ 3)                        ║");
    println!("║  Machine: {}                                          ║", MACHINE_ID);
    println!("║  Trials: {}                                           ║", trials_to_run.len());
    println!("║  Max steps: {} | Prune at: {} if BPB > {:.2}        ║", max_steps, prune_step, prune_threshold);
    println!("║  Target: BPB < {:.2}                                    ║", target_bpb);
    println!("╚════════════════════════════════════════════════════════════╝");

    let mut all_results = vec![];

    for config in trials_to_run {
        let seed = 42;
        let (bpb, step, outcome, duration) = run_trial(config.clone(), seed, max_steps, prune_step, prune_threshold);
        all_results.push((config.name.clone(), bpb, step, outcome, duration));
    }

    // Summary
    println!("\n=== ARCH-EXPLORER SUMMARY ===");
    println!("Trial | Best BPB  | Step | Outcome");
    println!("------|----------|------|-----------------------");

    for (name, bpb, step, outcome, _duration) in &all_results {
        let outcome_trunc = if outcome.len() > 20 {
            format!("{}...", &outcome[..17])
        } else {
            outcome.clone()
        };
        println!("{:5} | {:9.4} | {:5} | {}", name, bpb, step, outcome_trunc);
    }

    // Check for winner
    let best_result = all_results.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    if let Some((name, bpb, step, _, _)) = best_result {
        if *bpb < target_bpb {
            println!("\n🎯🎯🎯 WINNER FOUND! 🎯🎯🎯");
            println!("Trial {} achieved BPB={:.4} < {:.4} at step {}", name, bpb, target_bpb, step);
        } else {
            println!("\n📊 Best trial: {} with BPB={:.4} (target: <{:.4})", name, bpb, target_bpb);
            println!("Delta to target: +{:.4}", bpb - target_bpb);
        }
    }

    // Write summary to experience
    let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
    let ep = format!(".trinity/experience/trios_{}.trinity", chrono::Utc::now().format("%Y%m%d"));
    let _ = fs::OpenOptions::new().create(true).append(true)
        .open(&ep).unwrap().write_all(format!(
            "[{}] ARCH-EXPLORER SUMMARY | machine={} | trials={} | best_bpb={:.4} | target={:.4}\n",
            ts, MACHINE_ID, all_results.len(),
            best_result.map(|(_, b, _, _, _)| *b).unwrap_or(999.9), target_bpb
        ).as_bytes());
}
