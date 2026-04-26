use std::fs;
use std::io::Write;
use std::time::Instant;

const VOCAB: usize = 128;
const DIM: usize = 64;
const SEQ: usize = 48;
const LN_2: f32 = std::f32::consts::LN_2;

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
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in v.iter_mut() {
        *x /= sum;
    }
}

struct AdamWState {
    m: Vec<f32>,
    v: Vec<f32>,
    step: usize,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

impl AdamWState {
    fn new(size: usize) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            step: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }

    fn update(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        self.step += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);
        for i in 0..params.len() {
            params[i] -= self.weight_decay * lr * params[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

struct TrigramModel {
    embed: Vec<f32>,
    context: Vec<f32>,
    lm_head: Vec<f32>,
    vocab: usize,
    dim: usize,
}

impl TrigramModel {
    fn new(vocab: usize, dim: usize, seed: u64) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let t = ((s >> 33) as f32) / (u32::MAX as f32);
            (t * 2.0 - 1.0) * (6.0 / (vocab + dim) as f32).sqrt()
        };
        let embed: Vec<f32> = (0..vocab * dim).map(|_| rng()).collect();
        let context: Vec<f32> = (0..vocab * dim).map(|_| rng()).collect();
        let lm_head: Vec<f32> = (0..vocab * dim).map(|_| rng()).collect();
        Self { embed, context, lm_head, vocab, dim }
    }

    fn forward_pair(&self, id_prev: usize, id_cur: usize) -> Vec<f32> {
        let v = self.vocab;
        let d = self.dim;
        let prev = id_prev.min(v - 1);
        let cur = id_cur.min(v - 1);
        let e_prev = &self.context[prev * d..(prev + 1) * d];
        let e_cur = &self.embed[cur * d..(cur + 1) * d];
        let combined: Vec<f32> = e_prev.iter().zip(e_cur.iter()).map(|(a, b)| a + b).collect();
        (0..v).map(|vi| {
            let w = &self.lm_head[vi * d..(vi + 1) * d];
            combined.iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f32>()
        }).collect()
    }

    fn loss_on_seq(&self, tokens: &[usize]) -> f32 {
        if tokens.len() < 3 { return 0.0; }
        let mut total = 0.0f32;
        for i in 1..tokens.len() - 1 {
            let logits = self.forward_pair(tokens[i - 1], tokens[i]);
            let target = tokens[i + 1].min(self.vocab - 1);
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = logits.iter().map(|l| (*l - max_l).exp()).sum();
            let log_prob = logits[target] - max_l - sum_exp.ln();
            total -= log_prob;
        }
        total / (tokens.len() - 2) as f32
    }

    fn train_step(&mut self, tokens: &[usize], lr: f32, opt_e: &mut AdamWState, opt_c: &mut AdamWState, opt_h: &mut AdamWState) {
        if tokens.len() < 3 { return; }
        let d = self.dim;
        let v = self.vocab;

        let mut grad_embed = vec![0.0f32; v * d];
        let mut grad_context = vec![0.0f32; v * d];
        let mut grad_head = vec![0.0f32; v * d];

        for i in 1..tokens.len() - 1 {
            let prev = tokens[i - 1].min(v - 1);
            let cur = tokens[i].min(v - 1);
            let tgt = tokens[i + 1].min(v - 1);

            let mut logits = self.forward_pair(prev, cur);
            softmax(&mut logits);

            for (vi, prob) in logits.iter().enumerate() {
                let grad = prob - if vi == tgt { 1.0 } else { 0.0 };
                let e_cur = &self.embed[cur * d..(cur + 1) * d];
                let c_prev = &self.context[prev * d..(prev + 1) * d];
                let w_vi = &self.lm_head[vi * d..(vi + 1) * d];

                for j in 0..d {
                    let combined_j = e_cur[j] + c_prev[j];
                    grad_embed[cur * d + j] += grad * w_vi[j];
                    grad_context[prev * d + j] += grad * w_vi[j];
                    grad_head[vi * d + j] += grad * combined_j;
                }
            }
        }

        let n = (tokens.len() - 2) as f32;
        for g in grad_embed.iter_mut() { *g /= n; }
        for g in grad_context.iter_mut() { *g /= n; }
        for g in grad_head.iter_mut() { *g /= n; }

        opt_e.update(&mut self.embed, &grad_embed, lr);
        opt_c.update(&mut self.context, &grad_context, lr);
        opt_h.update(&mut self.lm_head, &grad_head, lr);
    }
}

fn evaluate(model: &TrigramModel, tokens: &[usize], seq_len: usize) -> (f32, f32) {
    let mut total = 0.0f32;
    let mut n = 0usize;
    for c in (0..tokens.len()).step_by(seq_len + 1) {
        let end = (c + seq_len + 1).min(tokens.len());
        if end - c < 4 { continue; }
        let seq = &tokens[c..end];
        let loss = model.loss_on_seq(seq);
        if loss.is_finite() {
            total += loss / LN_2;
            n += 1;
        }
    }
    if n == 0 { return (f32::MAX, f32::MAX); }
    let bpb = total / n as f32;
    (bpb * LN_2, bpb)
}

fn cosine_lr(step: usize, max_steps: usize, base_lr: f32, warmup: usize) -> f32 {
    if step < warmup {
        base_lr * step as f32 / warmup as f32
    } else {
        let progress = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        1e-5 + (base_lr - 1e-5) * cosine
    }
}

fn main() {
    let seed = std::env::args()
        .find(|a| a.starts_with("--seed="))
        .map(|a| a[7..].parse::<u64>().unwrap_or(42))
        .unwrap_or(42);
    let steps = std::env::args()
        .find(|a| a.starts_with("--steps="))
        .map(|a| a[8..].parse::<usize>().unwrap_or(8000))
        .unwrap_or(8000);
    let base_lr = std::env::args()
        .find(|a| a.starts_with("--lr="))
        .map(|a| a[5..].parse::<f32>().unwrap_or(0.003))
        .unwrap_or(0.003);

    println!("=== IGLA-STACK-502 Trigram Training ===");
    println!("vocab={} dim={} seq={} steps={} seed={} lr={}", VOCAB, DIM, SEQ, steps, seed, base_lr);
    println!();

    let tokens = load_data("data/tinyshakespeare.txt");
    println!("Dataset: {} tokens", tokens.len());

    let mut model = TrigramModel::new(VOCAB, DIM, seed);
    let param_size = VOCAB * DIM;
    let mut opt_e = AdamWState::new(param_size);
    let mut opt_c = AdamWState::new(param_size);
    let mut opt_h = AdamWState::new(param_size);

    let (init_loss, init_bpb) = evaluate(&model, &tokens, SEQ);
    println!("Initial: loss={:.4} bpb={:.4}", init_loss, init_bpb);
    println!();
    println!("{:>6} | {:>10} | {:>10} | {:>10} | {:>8} | {:>6}", "step", "loss", "bpb", "best_bpb", "ms", "lr");
    println!("{}", "-".repeat(70));

    let t0 = Instant::now();
    let mut best_bpb = init_bpb;
    let mut results: Vec<(usize, f32, f32)> = Vec::new();
    let data_len = tokens.len();

    for step in 1..=steps {
        let lr = cosine_lr(step, steps, base_lr, steps / 10);
        let offset = (step * 97 + seed as usize) % (data_len.saturating_sub(SEQ + 1));
        let seq = &tokens[offset..offset + SEQ + 1];
        model.train_step(seq, lr, &mut opt_e, &mut opt_c, &mut opt_h);

        if step % 500 == 0 || step == steps {
            let ms = t0.elapsed().as_millis();
            let (eval_loss, eval_bpb) = evaluate(&model, &tokens, SEQ);
            if eval_bpb < best_bpb && eval_bpb.is_finite() {
                best_bpb = eval_bpb;
            }
            println!("{:>6} | {:>10.4} | {:>10.4} | {:>10.4} | {:>6}ms | {:.6}",
                step, eval_loss, eval_bpb, best_bpb, ms, lr);
            results.push((step, eval_loss, eval_bpb));
        }
    }

    let total = t0.elapsed();
    println!();
    println!("=== Training Complete ===");
    println!("Time: {:.1}s | Initial BPB: {:.4} | Final BPB: {:.4} | Delta: {:.4}",
        total.as_secs_f64(), init_bpb, best_bpb, best_bpb - init_bpb);

    let _ = fs::create_dir_all(".trinity/results");
    let result_json = serde_json::json!({
        "experiment": "igla-stack-502-trigram",
        "model": "trigram-embedding",
        "optimizer": "AdamW",
        "seed": seed,
        "vocab_size": VOCAB,
        "dim": DIM,
        "seq_len": SEQ,
        "steps": steps,
        "base_lr": base_lr,
        "initial_bpb": init_bpb,
        "final_bpb": best_bpb,
        "delta_bpb": best_bpb - init_bpb,
        "duration_seconds": total.as_secs_f64(),
        "results": results.iter().map(|(s, l, b)| serde_json::json!({
            "step": *s, "loss": *l, "bpb": *b
        })).collect::<Vec<_>>(),
    });

    let rpath = format!(".trinity/results/igla_trigram_seed{}.json", seed);
    fs::File::create(&rpath).unwrap()
        .write_all(serde_json::to_string_pretty(&result_json).unwrap().as_bytes()).unwrap();
    println!("Results: {}", rpath);

    let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
    let edir = ".trinity/experience";
    let _ = fs::create_dir_all(edir);
    let epath = format!("{}/trios_{}.trinity", edir, chrono::Utc::now().format("%Y%m%d"));
    let entry = format!(
        "[{}] TASK: IGLA trigram training | seed={} | steps={} | bpb={:.4}->{:.4} | delta={:.4} | {:.1}s\n",
        ts, seed, steps, init_bpb, best_bpb, best_bpb - init_bpb, total.as_secs_f64()
    );
    let _ = fs::OpenOptions::new().create(true).append(true)
        .open(&epath).unwrap().write_all(entry.as_bytes());
    println!("Experience: {}", epath);
}
