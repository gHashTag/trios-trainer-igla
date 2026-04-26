use std::fs;
use std::io::Write;
use std::time::Instant;
use trios_trainer::trinity_3k::{AdamWConfig, Trinity3kConfig, Trinity3kModel};

const LN_2: f32 = std::f32::consts::LN_2;

fn load_data(path: &str) -> Vec<usize> {
    let raw = fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to load {}: {}. Using fallback.", path, e);
        b"Hello world this is a tiny training dataset for testing".to_vec()
    });
    raw.into_iter().map(|b| (b as usize) % 128).collect()
}

fn cosine_lr(step: usize, max_steps: usize, base_lr: f32, warmup: usize) -> f32 {
    if step < warmup {
        return base_lr * step as f32 / warmup as f32;
    }
    let p = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
    1e-5 + (base_lr - 1e-5) * 0.5 * (1.0 + (std::f32::consts::PI * p).cos())
}

fn evaluate(model: &Trinity3kModel, val: &[usize], seq_len: usize) -> (f32, f32) {
    let mut total_bpb = 0.0f32;
    let mut n = 0usize;
    for start in (0..val.len()).step_by(seq_len + 1) {
        let end = (start + seq_len + 1).min(val.len());
        if end - start < 4 {
            continue;
        }
        let (_, bpb) = model.loss_bpb(&val[start..end]);
        if bpb.is_finite() {
            total_bpb += bpb;
            n += 1;
        }
    }
    if n == 0 {
        return (f32::MAX, f32::MAX);
    }
    let avg_bpb = total_bpb / n as f32;
    (avg_bpb * LN_2, avg_bpb)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args
        .iter()
        .find(|a| a.starts_with("--seed="))
        .map(|a| a[7..].parse::<u64>().unwrap_or(42))
        .unwrap_or(42);
    let steps: usize = args
        .iter()
        .find(|a| a.starts_with("--steps="))
        .map(|a| a[8..].parse::<usize>().unwrap_or(5000))
        .unwrap_or(5000);
    let base_lr: f32 = args
        .iter()
        .find(|a| a.starts_with("--lr="))
        .map(|a| a[5..].parse::<f32>().unwrap_or(3e-4))
        .unwrap_or(3e-4);
    let hidden_dim: usize = args
        .iter()
        .find(|a| a.starts_with("--hidden="))
        .map(|a| a[9..].parse::<usize>().unwrap_or(81))
        .unwrap_or(81);
    let n_heads: usize = args
        .iter()
        .find(|a| a.starts_with("--heads="))
        .map(|a| a[8..].parse::<usize>().unwrap_or(9))
        .unwrap_or(9);
    let n_layers: usize = args
        .iter()
        .find(|a| a.starts_with("--layers="))
        .map(|a| a[9..].parse::<usize>().unwrap_or(2))
        .unwrap_or(2);
    let seq_len: usize = args
        .iter()
        .find(|a| a.starts_with("--seq="))
        .map(|a| a[6..].parse::<usize>().unwrap_or(64))
        .unwrap_or(64);
    let wd: f32 = args
        .iter()
        .find(|a| a.starts_with("--wd="))
        .map(|a| a[5..].parse::<f32>().unwrap_or(0.01))
        .unwrap_or(0.01);

    let head_dim = hidden_dim / n_heads;
    let vocab_size = 128usize;

    let config = Trinity3kConfig {
        vocab_size,
        hidden_dim,
        n_heads,
        head_dim,
        n_layers,
        max_seq_len: seq_len,
    };

    println!("=== Trinity3k Transformer on TinyShakespeare ===");
    println!(
        "vocab={} hidden={} heads={} head_dim={} layers={} seq={} params={}",
        vocab_size,
        hidden_dim,
        n_heads,
        head_dim,
        n_layers,
        seq_len,
        config.total_params()
    );
    println!("steps={} seed={} lr={} wd={}", steps, seed, base_lr, wd);

    let mut model = Trinity3kModel::new(config)?;
    println!("Model created successfully");

    let tokens = load_data("data/tinyshakespeare.txt");
    println!("Dataset: {} tokens", tokens.len());

    let train_end = (tokens.len() as f64 * 0.9) as usize;
    let train = &tokens[..train_end];
    let val = &tokens[train_end..];
    println!("Split: {} train / {} val", train.len(), val.len());

    let (init_loss, init_bpb) = evaluate(&model, val, seq_len);
    println!("Initial val: loss={:.4} bpb={:.4}", init_loss, init_bpb);
    println!();
    println!(
        "{:>6} | {:>10} | {:>10} | {:>10} | {:>8}",
        "step", "val_loss", "val_bpb", "best_bpb", "ms"
    );
    println!("{}", "-".repeat(60));

    let t0 = Instant::now();
    let mut best_bpb = init_bpb;
    let mut results: Vec<(usize, f32, f32)> = Vec::new();
    let dl = train.len();
    let mut adamw_cfg = AdamWConfig {
        lr: base_lr,
        beta1: 0.618,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: wd,
    };

    for step in 1..=steps {
        let lr = cosine_lr(step, steps, base_lr, steps / 10);
        adamw_cfg.lr = lr;
        let off = (step * 97 + seed as usize) % (dl.saturating_sub(seq_len + 1));
        let seq_tokens = &train[off..off + seq_len + 1];
        model.train_step(seq_tokens, &adamw_cfg);

        if step % 500 == 0 || step == steps {
            let ms = t0.elapsed().as_millis();
            let (vl, vb) = evaluate(&model, val, seq_len);
            if vb < best_bpb && vb.is_finite() {
                best_bpb = vb;
            }
            println!(
                "{:>6} | {:>10.4} | {:>10.4} | {:>10.4} | {:>6}ms",
                step, vl, vb, best_bpb, ms
            );
            results.push((step, vl, vb));
        }
    }

    let total = t0.elapsed();
    println!("\n=== Done ===");
    println!(
        "Time: {:.1}s | BPB: {:.4} -> {:.4} | Delta: {:.4}",
        total.as_secs_f64(),
        init_bpb,
        best_bpb,
        best_bpb - init_bpb
    );

    let _ = fs::create_dir_all(".trinity/results");
    let exp_name = format!(
        "trinity3k-h{}-l{}-hd{}-s{}",
        hidden_dim, n_layers, n_heads, seq_len
    );

    let rj = serde_json::json!({
        "experiment": exp_name,
        "model": format!("Trinity3k transformer h={} layers={} heads={} head_dim={}", hidden_dim, n_layers, n_heads, head_dim),
        "seed": seed, "steps": steps, "base_lr": base_lr, "wd": wd,
        "hidden_dim": hidden_dim, "n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim,
        "seq_len": seq_len, "vocab_size": vocab_size,
        "train_tokens": train.len(), "val_tokens": val.len(),
        "initial_val_bpb": init_bpb, "final_val_bpb": best_bpb,
        "delta_bpb": best_bpb - init_bpb,
        "duration_seconds": total.as_secs_f64(),
        "results": results.iter().map(|(s, l, b)| serde_json::json!({"step":*s,"loss":*l,"bpb":*b})).collect::<Vec<_>>(),
    });
    let rp = format!(".trinity/results/{}_seed{}.json", exp_name, seed);
    fs::File::create(&rp)
        .unwrap()
        .write_all(serde_json::to_string_pretty(&rj).unwrap().as_bytes())
        .unwrap();
    println!("Results: {}", rp);

    let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
    let ep = format!(
        ".trinity/experience/trios_{}.trinity",
        chrono::Utc::now().format("%Y%m%d")
    );
    let _ = fs::create_dir_all(".trinity/experience");
    let _ = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&ep)
        .unwrap()
        .write_all(
            format!(
                "[{}] TASK: Trinity3k h={} l={} | seed={} | steps={} | val_bpb={:.4}->{:.4} | {:.1}s\n",
                ts,
                hidden_dim,
                n_layers,
                seed,
                steps,
                init_bpb,
                best_bpb,
                total.as_secs_f64()
            )
            .as_bytes(),
        );
    println!("Experience: {}", ep);

    Ok(())
}
