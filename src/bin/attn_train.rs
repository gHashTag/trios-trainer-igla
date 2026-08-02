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

/// Exit code for a corpus that could not be honestly loaded. Same value
/// `cpu_train` uses, so a sweep can tell a refused corpus from a crash.
const EXIT_BAD_CORPUS: i32 = 6;

/// Pinned split. Both files ship in `data/`; neither is derived from the other.
const DEFAULT_TRAIN_PATH: &str = "data/tiny_shakespeare.txt";
const DEFAULT_VAL_PATH: &str = "data/tiny_shakespeare_val.txt";

/// Opt-in placeholder, kept only so `TRIOS_ALLOW_SYNTHETIC_DATA=1` behaves as
/// documented; `.repeat(100)` reproduces the old fallback buffer byte for
/// byte. Nothing measured against it is a model result.
const SYNTHETIC_CORPUS_UNIT: &[u8] = b"The quick brown fox jumps over the lazy dog. ";

/// Corpus identity, carried beside the tokens.
///
/// A number that cannot name the bytes it was measured on is indistinguishable
/// from a fabricated one, so path, size, digest and the synthetic flag travel
/// with every corpus this binary loads.
#[derive(Debug)]
struct CorpusInfo {
    path: String,
    bytes: usize,
    sha256: String,
    synthetic: bool,
}

impl CorpusInfo {
    fn describe(&self) -> String {
        format!(
            "path={} bytes={} sha256={} data_synthetic={}",
            self.path, self.bytes, self.sha256, self.synthetic
        )
    }
}

/// Lowercase hex SHA-256. Same digest as `shasum -a 256`.
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut acc, b| {
            use std::fmt::Write as _;
            let _ = write!(acc, "{b:02x}");
            acc
        })
}

/// Value of `--flag=VALUE` or `--flag VALUE`, else `default`.
fn arg_path(flag: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    let prefix = format!("{flag}=");
    for (i, a) in args.iter().enumerate() {
        if let Some(v) = a.strip_prefix(prefix.as_str()) {
            return v.to_string();
        }
        if a == flag {
            // A flag given without a value must NOT fall back to the default:
            // that is how a typo turns into a silent run on other bytes.
            return args.get(i + 1).cloned().unwrap_or_default();
        }
    }
    default.to_string()
}

/// Read `path`, or refuse. Ported from `trios_trainer::train_loop::load_data`.
///
/// The old body substituted a pangram buffer whenever the read failed, so a
/// machine without the corpus still printed a BPB. A missing corpus is now a
/// hard, named refusal. `TRIOS_ALLOW_SYNTHETIC_DATA=1` opts back in, says so
/// on stderr on every run, and marks the corpus `data_synthetic=true` in every
/// line the run prints.
fn load_data(path: &str) -> Result<(Vec<usize>, CorpusInfo), String> {
    match fs::read(path) {
        Ok(raw) if raw.is_empty() => Err(format!("corpus '{path}' is empty (0 bytes)")),
        Ok(raw) => {
            let info = CorpusInfo {
                path: path.to_string(),
                bytes: raw.len(),
                sha256: sha256_hex(&raw),
                synthetic: false,
            };
            Ok((
                raw.into_iter().map(|b| (b as usize) % VOCAB).collect(),
                info,
            ))
        }
        Err(e) => {
            if std::env::var("TRIOS_ALLOW_SYNTHETIC_DATA").as_deref() != Ok("1") {
                return Err(format!(
                    "cannot read corpus '{path}': {e}. Refusing the synthetic fallback: \
                     it is the documented cause of the leak-tainted BPB rows \
                     (trios-trainer-igla#60). Provide the corpus, or set \
                     TRIOS_ALLOW_SYNTHETIC_DATA=1 to opt in - runs that do are stamped \
                     data_synthetic=true and their BPB is meaningless."
                ));
            }
            eprintln!(
                "[data] WARNING: TRIOS_ALLOW_SYNTHETIC_DATA=1 and '{path}' is unreadable \
                 ({e}). Using the synthetic corpus. Any BPB from this run is meaningless \
                 and every artifact it produces is stamped data_synthetic=true."
            );
            let raw = SYNTHETIC_CORPUS_UNIT.repeat(100);
            let info = CorpusInfo {
                path: path.to_string(),
                bytes: raw.len(),
                sha256: sha256_hex(&raw),
                synthetic: true,
            };
            Ok((
                raw.into_iter().map(|b| (b as usize) % VOCAB).collect(),
                info,
            ))
        }
    }
}

/// Load or stop. The refusal is printed and the process exits BEFORE any
/// training starts, so a refused run reports no number at all.
fn load_or_refuse(path: &str) -> (Vec<usize>, CorpusInfo) {
    match load_data(path) {
        Ok(loaded) => loaded,
        Err(e) => {
            eprintln!("CORPUS REFUSED: {e}");
            std::process::exit(EXIT_BAD_CORPUS);
        }
    }
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

    // Canon #93 forbids seeds {42, 43, 44, 45} (see src/seed_canon.rs); the
    // default is a permitted one so an argument-free run is not born invalid.
    let seed: u64 = find_arg(&args, "--seed=", 47u64);
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

    let (train_data, train_corpus) = load_or_refuse(&arg_path("--train-data", DEFAULT_TRAIN_PATH));
    let (val_data, val_corpus) = load_or_refuse(&arg_path("--val-data", DEFAULT_VAL_PATH));
    println!("Corpus train: {}", train_corpus.describe());
    println!("Corpus val:   {}", val_corpus.describe());
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The refusal is the point: a missing corpus must stop the run, not
    /// silently become 52 bytes of placeholder with a BPB printed beside it.
    #[test]
    fn missing_corpus_is_refused_and_a_real_one_names_itself() {
        std::env::remove_var("TRIOS_ALLOW_SYNTHETIC_DATA");
        let err = load_data("data/does_not_exist_r4_probe.txt")
            .expect_err("a missing corpus must refuse, not substitute a placeholder");
        assert!(
            err.contains("does_not_exist_r4_probe.txt"),
            "refusal must name the path it could not read: {err}"
        );

        let (tokens, corpus) =
            load_data("data/pangram_fixture_160b.bin").expect("shipped fixture must load");
        assert_eq!(tokens.len(), 160);
        assert_eq!(corpus.bytes, 160);
        assert_eq!(corpus.sha256.len(), 64, "sha256 must be 64 hex chars");
        assert!(!corpus.synthetic);
    }
}
