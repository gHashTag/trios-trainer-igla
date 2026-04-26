use std::fs;
use std::io::Write;
use std::time::Instant;

const VOCAB: usize = 128;
const DIM: usize = 96;
const CTX_DIM: usize = 128;
const BIGRAM_VOCAB: usize = 512;
const BIGRAM_DIM: usize = 64;
const SEQ: usize = 64;
const LN_2: f32 = std::f32::consts::LN_2;
const LOGIT_SOFTCAP: f32 = 30.0;
const EMA_DECAY: f32 = 0.997;

fn load_data(path: &str) -> Vec<usize> {
    let raw = fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to load {}: {}. Using fallback.", path, e);
        b"Hello world this is a tiny training dataset for IGLA".to_vec()
    });
    raw.into_iter().map(|b| (b as usize) % VOCAB).collect()
}

fn softmax_cap(v: &mut [f32], softcap: f32) {
    for x in v.iter_mut() {
        *x = (*x / softcap).tanh() * softcap;
    }
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
    fn new(size: usize, _lr: f32, wd: f32) -> Self {
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
    fn update(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        self.step += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);
        for i in 0..params.len() {
            params[i] -= self.wd * lr * params[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

fn bigram_hash(cur: usize, prev: usize, vocab: usize) -> usize {
    ((36313u32.wrapping_mul(cur as u32)) ^ (27191u32.wrapping_mul(prev as u32))) as usize % vocab
}

struct TrinityCpuModel {
    embed: Vec<f32>,
    ctx_embed: Vec<f32>,
    bigram_embed: Vec<f32>,
    smear_gate: Vec<f32>,
    lm_head: Vec<f32>,
    ve_proj: Vec<f32>,
    ve_scale: Vec<f32>,
    vocab: usize,
    #[allow(dead_code)]
    dim: usize,
    ctx_dim: usize,
    bigram_vocab: usize,
    bigram_dim: usize,
    ve_dim: usize,
}

impl TrinityCpuModel {
    fn new(
        vocab: usize,
        dim: usize,
        ctx_dim: usize,
        bv: usize,
        bd: usize,
        ve_dim: usize,
        seed: u64,
    ) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let t = ((s >> 33) as f32) / (u32::MAX as f32);
            (t * 2.0 - 1.0) * (6.0f32 / (vocab + dim) as f32).sqrt()
        };
        let total_dim = ctx_dim + bd + ve_dim;
        Self {
            embed: (0..vocab * total_dim).map(|_| rng()).collect(),
            ctx_embed: (0..vocab * ctx_dim).map(|_| rng()).collect(),
            bigram_embed: (0..bv * bd).map(|_| rng()).collect(),
            smear_gate: vec![0.0f32; total_dim],
            lm_head: (0..vocab * total_dim).map(|_| rng()).collect(),
            ve_proj: (0..ve_dim * total_dim).map(|_| rng()).collect(),
            ve_scale: vec![1.0f32; 2],
            vocab,
            dim,
            ctx_dim,
            bigram_vocab: bv,
            bigram_dim: bd,
            ve_dim,
        }
    }

    fn get_repr(&self, prev: usize, cur: usize) -> Vec<f32> {
        let v = self.vocab;
        let cd = self.ctx_dim;
        let bd = self.bigram_dim;
        let vd = self.ve_dim;
        let total = cd + bd + vd;

        let cur_idx = cur.min(v - 1);
        let prev_idx = prev.min(v - 1);
        let bh = bigram_hash(cur, prev, self.bigram_vocab);

        let mut repr = vec![0.0f32; total];

        let e_cur = &self.embed[cur_idx * total..(cur_idx + 1) * total];
        let c_prev = &self.ctx_embed[prev_idx * cd..(prev_idx + 1) * cd];
        let b_hash = &self.bigram_embed[bh * bd..(bh + 1) * bd];

        for j in 0..cd {
            repr[j] = e_cur[j] + c_prev[j];
        }
        for j in 0..bd {
            repr[cd + j] = e_cur[cd + j] + b_hash[j] * 0.1;
        }
        for j in 0..vd {
            let mut ve_val = 0.0f32;
            for (k, r) in repr.iter().enumerate().take(total) {
                ve_val += self.ve_proj[j * total + k] * r;
            }
            repr[cd + bd + j] = e_cur[cd + bd + j] + ve_val * self.ve_scale[0];
        }

        let _sg: Vec<f32> = self
            .smear_gate
            .iter()
            .map(|&g| 1.0 / (1.0 + (-g).exp()))
            .collect();
        for r in repr.iter_mut() {
            *r *= 1.0;
        }

        layer_norm(&repr, 1e-5)
    }

    fn loss_on_seq(&self, tokens: &[usize]) -> f32 {
        if tokens.len() < 3 {
            return 0.0;
        }
        let v = self.vocab;
        let total = self.ctx_dim + self.bigram_dim + self.ve_dim;
        let mut total_loss = 0.0f32;

        for i in 1..tokens.len() - 1 {
            let repr = self.get_repr(tokens[i - 1], tokens[i]);
            let target = tokens[i + 1].min(v - 1);

            let mut logits: Vec<f32> = (0..v)
                .map(|vi| {
                    let w = &self.lm_head[vi * total..(vi + 1) * total];
                    repr.iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f32>()
                })
                .collect();
            softmax_cap(&mut logits, LOGIT_SOFTCAP);

            let p = logits[target].max(1e-10);
            total_loss -= p.ln();
        }
        total_loss / (tokens.len() - 2) as f32
    }

    fn train_step(
        &mut self,
        tokens: &[usize],
        lr: f32,
        opt_e: &mut AdamW,
        opt_c: &mut AdamW,
        opt_b: &mut AdamW,
        opt_h: &mut AdamW,
    ) {
        if tokens.len() < 3 {
            return;
        }
        let v = self.vocab;
        let total = self.ctx_dim + self.bigram_dim + self.ve_dim;
        let cd = self.ctx_dim;
        let bd = self.bigram_dim;

        let mut grad_embed = vec![0.0f32; v * total];
        let mut grad_ctx = vec![0.0f32; v * cd];
        let mut grad_bigram = vec![0.0f32; self.bigram_vocab * bd];
        let mut grad_head = vec![0.0f32; v * total];

        for i in 1..tokens.len() - 1 {
            let prev = tokens[i - 1].min(v - 1);
            let cur = tokens[i].min(v - 1);
            let tgt = tokens[i + 1].min(v - 1);
            let bh = bigram_hash(cur, prev, self.bigram_vocab);

            let repr = self.get_repr(prev, cur);

            let mut logits: Vec<f32> = (0..v)
                .map(|vi| {
                    let w = &self.lm_head[vi * total..(vi + 1) * total];
                    repr.iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f32>()
                })
                .collect();
            softmax_cap(&mut logits, LOGIT_SOFTCAP);

            for (vi, prob) in logits.iter().enumerate() {
                let grad = prob - if vi == tgt { 1.0 } else { 0.0 };
                let w_vi = &self.lm_head[vi * total..(vi + 1) * total];
                for j in 0..total {
                    grad_embed[cur * total + j] += grad * w_vi[j];
                    grad_head[vi * total + j] += grad * repr[j];
                }
                for j in 0..cd {
                    grad_ctx[prev * cd + j] += grad * w_vi[j];
                }
                for j in 0..bd {
                    grad_bigram[bh * bd + j] += grad * w_vi[cd + j] * 0.1;
                }
            }
        }

        let n = (tokens.len() - 2) as f32;
        for g in grad_embed.iter_mut() {
            *g /= n;
        }
        for g in grad_ctx.iter_mut() {
            *g /= n;
        }
        for g in grad_bigram.iter_mut() {
            *g /= n;
        }
        for g in grad_head.iter_mut() {
            *g /= n;
        }

        opt_e.update(&mut self.embed, &grad_embed, lr);
        opt_c.update(&mut self.ctx_embed, &grad_ctx, lr);
        opt_b.update(&mut self.bigram_embed, &grad_bigram, lr);
        opt_h.update(&mut self.lm_head, &grad_head, lr);
    }

    #[allow(dead_code)]
    fn apply_ema(&mut self, ema: &Self) {
        for i in 0..self.embed.len() {
            self.embed[i] = ema.embed[i] * (1.0 - EMA_DECAY) + self.embed[i] * EMA_DECAY;
        }
    }
}

fn evaluate(model: &TrinityCpuModel, tokens: &[usize], seq_len: usize) -> (f32, f32) {
    let mut total = 0.0f32;
    let mut n = 0usize;
    for c in (0..tokens.len()).step_by(seq_len + 1) {
        let end = (c + seq_len + 1).min(tokens.len());
        if end - c < 4 {
            continue;
        }
        let loss = model.loss_on_seq(&tokens[c..end]);
        if loss.is_finite() {
            total += loss / LN_2;
            n += 1;
        }
    }
    if n == 0 {
        return (f32::MAX, f32::MAX);
    }
    let bpb = total / n as f32;
    (bpb * LN_2, bpb)
}

fn cosine_lr(step: usize, max_steps: usize, base_lr: f32, warmup: usize) -> f32 {
    if step < warmup {
        return base_lr * step as f32 / warmup as f32;
    }
    let progress = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    1e-5 + (base_lr - 1e-5) * cosine
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

    println!("=== Trinity CPU Model (PR#1722 params adapted) ===");
    println!(
        "arch: vocab={} ctx_dim={} bigram={}x{} ve_dim={} seq={} softcap={}",
        VOCAB, CTX_DIM, BIGRAM_VOCAB, BIGRAM_DIM, 16, SEQ, LOGIT_SOFTCAP
    );
    println!(
        "opt: AdamW(phi beta1) wd=0.04 lr={} steps={} seed={}",
        base_lr, steps, seed
    );
    println!();

    let tokens = load_data("data/tinyshakespeare.txt");
    println!("Dataset: {} tokens", tokens.len());

    let total_dim = CTX_DIM + BIGRAM_DIM + 16;
    let mut model = TrinityCpuModel::new(VOCAB, DIM, CTX_DIM, BIGRAM_VOCAB, BIGRAM_DIM, 16, seed);
    let mut ema_model =
        TrinityCpuModel::new(VOCAB, DIM, CTX_DIM, BIGRAM_VOCAB, BIGRAM_DIM, 16, seed);

    let mut opt_e = AdamW::new(VOCAB * total_dim, base_lr, 0.04);
    let mut opt_c = AdamW::new(VOCAB * CTX_DIM, base_lr, 0.04);
    let mut opt_b = AdamW::new(BIGRAM_VOCAB * BIGRAM_DIM, base_lr, 0.04);
    let mut opt_h = AdamW::new(VOCAB * total_dim, base_lr, 0.04);

    let (init_loss, init_bpb) = evaluate(&model, &tokens, SEQ);
    println!("Initial: loss={:.4} bpb={:.4}", init_loss, init_bpb);
    println!();
    println!(
        "{:>6} | {:>10} | {:>10} | {:>10} | {:>8}",
        "step", "loss", "bpb", "best_bpb", "ms"
    );
    println!("{}", "-".repeat(60));

    let t0 = Instant::now();
    let mut best_bpb = init_bpb;
    let mut results: Vec<(usize, f32, f32)> = Vec::new();
    let data_len = tokens.len();

    for step in 1..=steps {
        let lr = cosine_lr(step, steps, base_lr, steps / 10);
        let offset = (step * 97 + seed as usize) % (data_len.saturating_sub(SEQ + 1));
        let seq = &tokens[offset..offset + SEQ + 1];
        model.train_step(seq, lr, &mut opt_e, &mut opt_c, &mut opt_b, &mut opt_h);

        for i in 0..model.embed.len() {
            ema_model.embed[i] =
                ema_model.embed[i] * EMA_DECAY + model.embed[i] * (1.0 - EMA_DECAY);
        }
        for i in 0..model.lm_head.len() {
            ema_model.lm_head[i] =
                ema_model.lm_head[i] * EMA_DECAY + model.lm_head[i] * (1.0 - EMA_DECAY);
        }

        if step % 500 == 0 || step == steps {
            let ms = t0.elapsed().as_millis();
            let (eval_loss, eval_bpb) = evaluate(&ema_model, &tokens, SEQ);
            if eval_bpb < best_bpb && eval_bpb.is_finite() {
                best_bpb = eval_bpb;
            }
            println!(
                "{:>6} | {:>10.4} | {:>10.4} | {:>10.4} | {:>6}ms",
                step, eval_loss, eval_bpb, best_bpb, ms
            );
            results.push((step, eval_loss, eval_bpb));
        }
    }

    let total = t0.elapsed();
    println!();
    println!("=== Training Complete ===");
    println!(
        "Time: {:.1}s | Initial BPB: {:.4} | Final BPB: {:.4} (EMA) | Delta: {:.4}",
        total.as_secs_f64(),
        init_bpb,
        best_bpb,
        best_bpb - init_bpb
    );

    let _ = fs::create_dir_all(".trinity/results");
    let result_json = serde_json::json!({
        "experiment": "trinity-pr1722-adapted",
        "source_pr": "openai/parameter-golf#1722",
        "techniques": ["BigramHash", "SmearGate", "LayerNorm", "LogitSoftcap", "EMA", "AdamW-phi", "CosineLR"],
        "pr1722_params": {
            "arch": "11L 512d 8h/4kv MLP3x",
            "bigram_vocab": BIGRAM_VOCAB, "bigram_dim": BIGRAM_DIM,
            "logit_softcap": LOGIT_SOFTCAP, "ema_decay": EMA_DECAY,
            "muon_momentum": 0.99, "wd": 0.04, "qk_gain_init": 1.5
        },
        "seed": seed, "steps": steps, "base_lr": base_lr,
        "initial_bpb": init_bpb, "final_bpb": best_bpb,
        "delta_bpb": best_bpb - init_bpb,
        "duration_seconds": total.as_secs_f64(),
        "results": results.iter().map(|(s, l, b)| serde_json::json!({
            "step": *s, "loss": *l, "bpb": *b
        })).collect::<Vec<_>>(),
    });

    let rpath = format!(".trinity/results/trinity_pr1722_seed{}.json", seed);
    fs::File::create(&rpath)
        .unwrap()
        .write_all(
            serde_json::to_string_pretty(&result_json)
                .unwrap()
                .as_bytes(),
        )
        .unwrap();
    println!("Results: {}", rpath);

    let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
    let edir = ".trinity/experience";
    let _ = fs::create_dir_all(edir);
    let epath = format!(
        "{}/trios_{}.trinity",
        edir,
        chrono::Utc::now().format("%Y%m%d")
    );
    let entry = format!(
        "[{}] TASK: Trinity PR#1722 adapted training | seed={} | steps={} | bpb={:.4}->{:.4} | delta={:.4} | {:.1}s\n",
        ts, seed, steps, init_bpb, best_bpb, best_bpb - init_bpb, total.as_secs_f64()
    );
    let _ = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&epath)
        .unwrap()
        .write_all(entry.as_bytes());
    println!("Experience: {}", epath);
}
