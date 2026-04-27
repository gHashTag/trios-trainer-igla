//! Attention Training Binary — T1-02 (roadmap: -0.30 BPB)
//!
//! Uses AttentionModel from attention.rs:
//! - Multi-head causal self-attention
//! - RoPE positional encoding
//! - QK-Norm + QK-Gain (phi^2 = 2.618)
//! - ReLU^2 activation in FFN
//! - Full analytical backward pass
//! - AdamW optimizer
//!
//! Architecture: Token Embed → N × (PreNorm → MHA → Res → PreNorm → FFN → Res) → LM Head
//!
//! Run: cargo run --release --bin attn_train -- --steps 27000 --seed 43

use std::env;
use std::fs;
use std::time::Instant;

use trios_trainer::attention::{AttentionConfig, AttentionModel};

const VOCAB: usize = 128;
const LN_2: f32 = std::f32::consts::LN_2;
const SEQ: usize = 64;

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

fn evaluate(model: &AttentionModel, data: &[usize]) -> f32 {
    let n_chunks = data.len() / (SEQ + 1);
    if n_chunks == 0 {
        return f32::MAX;
    }
    let sample = n_chunks.min(50);
    let mut total = 0.0f32;
    let mut count = 0usize;
    for i in 0..sample {
        let off = i * (data.len() / sample / (SEQ + 1)).max(1) * (SEQ + 1);
        if off + SEQ + 1 > data.len() {
            continue;
        }
        let chunk = &data[off..off + SEQ + 1];
        let (_, bpb) = model.loss_bpb(chunk);
        if bpb.is_finite() {
            total += bpb;
            count += 1;
        }
    }
    if count == 0 {
        f32::MAX
    } else {
        total / count as f32
    }
}

fn find_arg<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> T {
    for (i, a) in args.iter().enumerate() {
        if a.starts_with(key) {
            let val = if a.contains('=') {
                a[key.len()..].trim_start_matches('=').to_string()
            } else if i + 1 < args.len() {
                args[i + 1].clone()
            } else {
                continue;
            };
            return val.parse().unwrap_or(default);
        }
    }
    default
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let seed: u64 = find_arg(&args, "--seed=", 43u64);
    let steps: usize = find_arg(&args, "--steps=", 27000usize);
    let lr: f32 = find_arg(&args, "--lr=", 0.003f32);
    let d_model: usize = find_arg(&args, "--d-model=", 384usize);
    let n_heads: usize = find_arg(&args, "--heads=", 6usize);
    let n_layers: usize = find_arg(&args, "--layers=", 2usize);
    let qk_gain: f32 = find_arg(&args, "--qk-gain=", 2.618f32);
    let seq_len: usize = find_arg(&args, "--seq-len=", SEQ);
    let no_rope: bool = args.iter().any(|a| a == "--no-rope");

    println!("=== Attention Training (T1-02) ===");
    println!(
        "d_model={} n_heads={} n_layers={} lr={} seed={}",
        d_model, n_heads, n_layers, lr, seed
    );
    println!(
        "qk_gain={:.3} rope={} seq={} steps={}",
        qk_gain, !no_rope, seq_len, steps
    );
    println!(
        "params≈{}",
        d_model
            * (VOCAB
                + n_layers
                    * (3 * d_model * d_model + d_model * 4 * d_model + d_model * 4 * d_model)
                + VOCAB)
    );

    let train_data = load_data("data/tiny_shakespeare.txt");
    let val_data = load_data("data/tiny_shakespeare_val.txt");
    let train_end = (train_data.len() as f64 * 0.9) as usize;
    let train = &train_data[..train_end];
    let val = if val_data.len() > 100 {
        &val_data[..]
    } else {
        &train_data[train_end..]
    };

    let config = AttentionConfig {
        vocab_size: VOCAB,
        d_model,
        n_heads,
        n_layers,
        max_seq_len: seq_len,
        use_rope: !no_rope,
        qk_gain_init: qk_gain,
        lr,
        beta1: 0.618,
        beta2: 0.999,
        weight_decay: 0.01,
    };

    let mut model = AttentionModel::new(config);
    let start = Instant::now();
    let mut best_val_bpb = f32::MAX;

    for step in 1..=steps {
        let dl = train.len();
        let off = (step * 97 + seed as usize) % dl.saturating_sub(seq_len + 1);
        let chunk = &train[off..off + seq_len + 1];

        model.train_step(chunk);

        if step % 500 == 0 || step == steps {
            let elapsed = start.elapsed().as_secs_f64();
            let val_bpb = evaluate(&model, val);
            if val_bpb < best_val_bpb && val_bpb.is_finite() {
                best_val_bpb = val_bpb;
            }
            eprintln!(
                "step={:5} val_bpb={:.4} best={:.4} t={:.1}s",
                step, val_bpb, best_val_bpb, elapsed
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!("\n=== Training Complete ===");
    println!(
        "Steps={} Time={:.1}s best_val_bpb={:.4}",
        steps, elapsed, best_val_bpb
    );
    println!("vs champion 2.5193: {:+.4}", best_val_bpb - 2.5193);
    println!("BPB={:.4}", best_val_bpb);
}
