use std::fs;
use std::io::Write;
use std::time::Instant;

use trios_trainer::neon_writer as nw;

const VOCAB: usize = 128;
const DIM: usize = 64;
const SEQ: usize = 32;
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

struct EmbeddingModel {
    embed: Vec<f32>,
    lm_head: Vec<f32>,
    vocab: usize,
    dim: usize,
}

impl EmbeddingModel {
    fn new(vocab: usize, dim: usize, seed: u64) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let t = ((s >> 33) as f32) / (u32::MAX as f32);
            t * 0.04 - 0.02
        };
        let embed: Vec<f32> = (0..vocab * dim).map(|_| rng()).collect();
        let lm_head: Vec<f32> = (0..vocab * dim).map(|_| rng()).collect();
        Self {
            embed,
            lm_head,
            vocab,
            dim,
        }
    }

    fn forward(&self, input_id: usize) -> Vec<f32> {
        let id = input_id.min(self.vocab - 1);
        let d = self.dim;
        let x = &self.embed[id * d..(id + 1) * d];
        (0..self.vocab)
            .map(|v| {
                let w = &self.lm_head[v * d..(v + 1) * d];
                x.iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f32>()
            })
            .collect()
    }

    fn loss_on_seq(&self, tokens: &[usize]) -> f32 {
        if tokens.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0f32;
        for i in 0..tokens.len() - 1 {
            let logits = self.forward(tokens[i]);
            let target = tokens[i + 1].min(self.vocab - 1);
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = logits.iter().map(|l| (*l - max_l).exp()).sum();
            let log_prob = logits[target] - max_l - sum_exp.ln();
            total -= log_prob;
        }
        total / (tokens.len() - 1) as f32
    }

    fn train_step(&mut self, tokens: &[usize], lr: f32) {
        if tokens.len() < 2 {
            return;
        }
        let d = self.dim;
        let v = self.vocab;

        for i in 0..tokens.len() - 1 {
            let inp = tokens[i].min(v - 1);
            let tgt = tokens[i + 1].min(v - 1);

            let mut logits = self.forward(inp);
            softmax(&mut logits);

            for (vi, logit) in logits.iter().enumerate().take(v) {
                let grad = *logit - if vi == tgt { 1.0 } else { 0.0 };
                let emb_off = inp * d;
                let head_off = vi * d;

                for j in 0..d {
                    let eg = grad * self.lm_head[head_off + j];
                    let wg = grad * self.embed[emb_off + j];
                    self.embed[emb_off + j] -= lr * eg;
                    self.lm_head[head_off + j] -= lr * wg;
                }
            }
        }
    }
}

fn evaluate(model: &EmbeddingModel, tokens: &[usize], seq_len: usize) -> (f32, f32) {
    let mut total = 0.0f32;
    let mut n = 0usize;
    for c in (0..tokens.len()).step_by(seq_len + 1) {
        let end = (c + seq_len + 1).min(tokens.len());
        if end - c < 3 {
            continue;
        }
        let seq = &tokens[c..end];
        let loss = model.loss_on_seq(seq);
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

fn main() {
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
        .map(|a| a[8..].parse::<usize>().unwrap_or(5000))
        .unwrap_or_else(|| {
            std::env::var("STEPS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5000)
        });
    let lr = std::env::args()
        .find(|a| a.starts_with("--lr="))
        .map(|a| a[5..].parse::<f32>().unwrap_or(0.1))
        .unwrap_or(0.1);

    let checkpoint_interval = nw::checkpoint_interval();

    // canon_name: prefer env var, fall back to deterministic name
    let canon_name = std::env::var("CANON_NAME")
        .unwrap_or_else(|_| format!("IGLA-TRAIN-igla_train-rng{}", seed));

    println!("=== IGLA-STACK-502 Embedding Training ===");
    println!(
        "vocab={} dim={} seq={} steps={} seed={} lr={} checkpoint_interval={}",
        VOCAB, DIM, SEQ, steps, seed, lr, checkpoint_interval
    );
    println!("canon_name={}", canon_name);
    println!();

    let tokens = load_data("data/tinyshakespeare.txt");
    println!("Dataset: {} tokens", tokens.len());

    let mut model = EmbeddingModel::new(VOCAB, DIM, seed);

    let (init_loss, init_bpb) = evaluate(&model, &tokens, SEQ);
    println!("Initial: loss={:.4} bpb={:.4}", init_loss, init_bpb);

    // Write step=0 ping so queue knows trainer started
    nw::bpb_sample(&canon_name, seed as i32, 0, init_bpb, None);
    eprintln!("[neon] wrote step=0 bpb={:.4}", init_bpb);

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
        let offset = (step * 97 + seed as usize) % (data_len.saturating_sub(SEQ + 1));
        let seq = &tokens[offset..offset + SEQ + 1];
        model.train_step(seq, lr);

        if step % checkpoint_interval == 0 || step == steps {
            let ms = t0.elapsed().as_millis();
            let (eval_loss, eval_bpb) = evaluate(&model, &tokens, SEQ);
            if eval_bpb < best_bpb && eval_bpb.is_finite() {
                best_bpb = eval_bpb;
            }
            println!(
                "{:>6} | {:>10.4} | {:>10.4} | {:>10.4} | {:>6}ms",
                step, eval_loss, eval_bpb, best_bpb, ms
            );
            results.push((step, eval_loss, eval_bpb));

            // P0 fix: wire bpb_sample to Neon every checkpoint_interval steps
            if eval_bpb.is_finite() {
                nw::bpb_sample(&canon_name, seed as i32, step as i32, eval_bpb, None);
                eprintln!("[neon] wrote step={} bpb={:.4}", step, eval_bpb);
            }
        }
    }

    let total = t0.elapsed();
    println!();
    println!("=== Training Complete ===");
    println!(
        "Time: {:.1}s | Initial BPB: {:.4} | Final BPB: {:.4} | Delta: {:.4}",
        total.as_secs_f64(),
        init_bpb,
        best_bpb,
        best_bpb - init_bpb
    );

    let _ = fs::create_dir_all(".trinity/results");
    let result_json = serde_json::json!({
        "experiment": "igla-stack-502-embed",
        "model": "bigram-embedding",
        "seed": seed,
        "vocab_size": VOCAB,
        "dim": DIM,
        "seq_len": SEQ,
        "steps": steps,
        "lr": lr,
        "initial_bpb": init_bpb,
        "final_bpb": best_bpb,
        "delta_bpb": best_bpb - init_bpb,
        "duration_seconds": total.as_secs_f64(),
        "results": results.iter().map(|(s, l, b)| serde_json::json!({
            "step": s, "loss": *l, "bpb": *b
        })).collect::<Vec<_>>(),
    });

    let rpath = format!(".trinity/results/igla_train_seed{}.json", seed);
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
    let entry =
        format!(
        "[{}] TASK: IGLA training | seed={} | steps={} | bpb={:.4}->{:.4} | delta={:.4} | {:.1}s\n",
        ts, seed, steps, init_bpb, best_bpb, best_bpb - init_bpb, total.as_secs_f64()
    );
    let _ = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&epath)
        .unwrap()
        .write_all(entry.as_bytes());
    println!("Experience: {}", epath);

    // L-R8: stdout must end with BPB=X.XXXX for ASHA worker parsing
    println!("BPB={:.4}", best_bpb);
}
