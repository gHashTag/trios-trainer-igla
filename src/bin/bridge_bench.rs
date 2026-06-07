//! bridge_bench — actual training-time comparison of three format gates
//! on a sandbox-scale bigram LM.
//!
//! Loop 155 builds the bridge from §9.4.1's encode-time grid to a
//! *training-time* number — the format-zoo arm at its smallest honest
//! scale. The model is a one-layer bigram: `logits = embed[prev] @
//! output_proj`. The embed table is round-tripped through the chosen
//! format (`f32` / `GF16` / `Posit16`) after every SGD step, the
//! shadow-weight pattern documented in F2 §9.4.
//!
//! Data: `data/tiny_shakespeare.txt` (1.1 M chars) for training,
//! `data/tiny_shakespeare_val.txt` (100 K chars) for held-out
//! evaluation. 50 SGD steps × 64 (prev, next) pairs per batch is the
//! shortest run that produces a stable val-BPB delta — long enough that
//! the format gate matters, short enough to run end-to-end in seconds.
//!
//! Output: per-seed JSON + summary at
//! `.trinity/results/bridge_bench_*.json`. F2 §9.4.3 quotes the
//! summary's headline `mean ± std` of (final val BPB) per format.
//!
//! Usage:
//!   cargo run --release --bin bridge_bench -- [--seeds=42,43,44]

use std::fs;
use std::io::Write;
use std::path::Path;

use trios_trainer::gf16::GF16;
use trios_trainer::phi_numbers::Posit16;

const VOCAB: usize = 128;
// Loop 160: model upgraded from bigram (single linear projection)
// to a 2-layer MLP with ReLU activation. The embed dimension HIDDEN
// is 128 (was 64); the MLP hidden width HIDDEN_MLP is 128. All three
// weight matrices (embed VOCAB×HIDDEN, W1 HIDDEN×HIDDEN_MLP, W2
// HIDDEN_MLP×VOCAB) are quantized under the same format gate after
// every SGD step (shadow-weight pattern: master in f32, all weights
// round-tripped through chosen format). This puts the format-zoo
// comparison one step closer to a real transformer block.
const HIDDEN: usize = 128;
const HIDDEN_MLP: usize = 128;
// Loop 156 hardening: STEPS 50 → 200 (4× longer training) to push the
// GF16-vs-f32 delta above the per-seed std band. The 77th-pass SEV-2 #1
// flagged that the 50-step delta (+0.0067 BPB) was 5× smaller than the
// per-seed std (0.033), so the rank ordering was preserved but the
// numerical delta was not statistically significant at N=3. The
// extended run (200 steps × 5 seeds = 5× more SGD updates × 1.67× more
// seeds) is the budget we use in §9.4.3 going forward.
const STEPS: usize = 200;
const BATCH: usize = 64;
// Loop 160 79th-pass SEV-3 closure: LR=0.5 was inherited unchanged
// from the bigram iteration. We did NOT re-tune for the MLP variant.
// The empirical justification is that all five seeds converge
// uniformly (val BPB 4.29-4.38, well below the random-byte ceiling of
// 7.0) with no sign of divergence or instability; a separate LR
// sweep is out of scope for the sandbox-scale §9.4.3 result.
const LR: f32 = 0.5;
const LN2: f32 = std::f32::consts::LN_2;

#[derive(Clone, Copy, Debug)]
enum Format {
    F32,
    Gf16,
    Posit16,
}

impl Format {
    fn slug(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Gf16 => "gf16",
            Self::Posit16 => "posit16",
        }
    }
    fn quantize(self, x: f32) -> f32 {
        match self {
            Self::F32 => x,
            Self::Gf16 => GF16::from_f32(x).to_f32(),
            Self::Posit16 => Posit16::from_f32(x).to_f32(),
        }
    }
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0x9E37_79B9_7F4A_7C15,
        }
    }
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }
    fn next_unit(&mut self) -> f32 {
        let u = self.next_u32() as f32 / (u32::MAX as f32 + 1.0);
        u * 2.0 - 1.0
    }
    fn pick(&mut self, max: usize) -> usize {
        (self.next_u32() as usize) % max
    }
}

fn xavier_init(seed: u64, n: usize, fan_in: usize) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    let scale = (1.0_f32 / fan_in as f32).sqrt();
    (0..n).map(|_| rng.next_unit() * scale).collect()
}

fn load_tokens(path: &str) -> Vec<u8> {
    fs::read(path).unwrap_or_default()
}

/// MLP forward pass.
/// Returns (hidden_pre_relu, hidden_post_relu, logits).
///   embed_row : HIDDEN
///   w1        : HIDDEN × HIDDEN_MLP
///   w2        : HIDDEN_MLP × VOCAB
fn forward_mlp(
    embed_row: &[f32],
    w1: &[f32],
    w2: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // h_pre[m] = Σ_h embed_row[h] * w1[h*HIDDEN_MLP + m]
    let mut h_pre = vec![0.0_f32; HIDDEN_MLP];
    for m in 0..HIDDEN_MLP {
        let mut s = 0.0_f32;
        for h in 0..HIDDEN {
            s += embed_row[h] * w1[h * HIDDEN_MLP + m];
        }
        h_pre[m] = s;
    }
    // ReLU
    let h_post: Vec<f32> = h_pre.iter().map(|x| x.max(0.0)).collect();
    // logits[v] = Σ_m h_post[m] * w2[m*VOCAB + v]
    let mut logits = vec![0.0_f32; VOCAB];
    for v in 0..VOCAB {
        let mut s = 0.0_f32;
        for m in 0..HIDDEN_MLP {
            s += h_post[m] * w2[m * VOCAB + v];
        }
        logits[v] = s;
    }
    (h_pre, h_post, logits)
}

fn softmax_inplace(logits: &mut [f32]) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

/// Run one (format, seed) training cell on the 2-layer MLP model.
/// Returns final held-out val BPB.
fn run_one(fmt: Format, seed: u64, train: &[u8], val: &[u8]) -> f64 {
    let mut embed = xavier_init(seed, VOCAB * HIDDEN, HIDDEN);
    let mut w1 = xavier_init(seed.wrapping_add(1), HIDDEN * HIDDEN_MLP, HIDDEN);
    let mut w2 = xavier_init(seed.wrapping_add(2), HIDDEN_MLP * VOCAB, HIDDEN_MLP);

    let mut step_rng = Lcg::new(seed.wrapping_add(3));
    let n_train_pairs = train.len().saturating_sub(1);
    if n_train_pairs == 0 {
        return f64::NAN;
    }

    for _step in 0..STEPS {
        let mut d_embed = vec![0.0_f32; embed.len()];
        let mut d_w1 = vec![0.0_f32; w1.len()];
        let mut d_w2 = vec![0.0_f32; w2.len()];

        for _ in 0..BATCH {
            let i = step_rng.pick(n_train_pairs);
            let prev = (train[i] as usize) % VOCAB;
            let next = (train[i + 1] as usize) % VOCAB;

            let embed_row: Vec<f32> =
                embed[prev * HIDDEN..(prev + 1) * HIDDEN].to_vec();
            let (h_pre, h_post, mut logits) = forward_mlp(&embed_row, &w1, &w2);
            softmax_inplace(&mut logits);

            // dL/dlogits = softmax - one_hot(next)
            let mut d_logits = logits.clone();
            d_logits[next] -= 1.0;
            let scale = 1.0_f32 / BATCH as f32;
            for v in d_logits.iter_mut() {
                *v *= scale;
            }

            // dL/dw2[m][v] = h_post[m] * d_logits[v]
            for m in 0..HIDDEN_MLP {
                let hpm = h_post[m];
                for v in 0..VOCAB {
                    d_w2[m * VOCAB + v] += hpm * d_logits[v];
                }
            }
            // dL/dh_post[m] = Σ_v d_logits[v] * w2[m][v]
            let mut d_h_post = vec![0.0_f32; HIDDEN_MLP];
            for m in 0..HIDDEN_MLP {
                let mut s = 0.0_f32;
                for v in 0..VOCAB {
                    s += d_logits[v] * w2[m * VOCAB + v];
                }
                d_h_post[m] = s;
            }
            // ReLU backward: zero out where h_pre <= 0
            let d_h_pre: Vec<f32> = (0..HIDDEN_MLP)
                .map(|m| if h_pre[m] > 0.0 { d_h_post[m] } else { 0.0 })
                .collect();
            // dL/dw1[h][m] = embed_row[h] * d_h_pre[m]
            for h in 0..HIDDEN {
                let eh = embed_row[h];
                for m in 0..HIDDEN_MLP {
                    d_w1[h * HIDDEN_MLP + m] += eh * d_h_pre[m];
                }
            }
            // dL/dembed_row[h] = Σ_m d_h_pre[m] * w1[h][m]
            for h in 0..HIDDEN {
                let mut s = 0.0_f32;
                for m in 0..HIDDEN_MLP {
                    s += d_h_pre[m] * w1[h * HIDDEN_MLP + m];
                }
                d_embed[prev * HIDDEN + h] += s;
            }
        }

        // SGD update + format-gate quantization on ALL three weight
        // matrices (the format-zoo shadow-weight pattern: master in
        // f32, every weight tensor round-tripped through the chosen
        // format after every SGD step).
        for i in 0..embed.len() {
            embed[i] -= LR * d_embed[i];
            embed[i] = fmt.quantize(embed[i]);
        }
        for i in 0..w1.len() {
            w1[i] -= LR * d_w1[i];
            w1[i] = fmt.quantize(w1[i]);
        }
        for i in 0..w2.len() {
            w2[i] -= LR * d_w2[i];
            w2[i] = fmt.quantize(w2[i]);
        }
    }

    // Held-out BPB on val (prev, next) pairs.
    let mut total_nll = 0.0_f64;
    let mut count = 0_usize;
    let n_val_pairs = val.len().saturating_sub(1);
    for i in 0..n_val_pairs {
        let prev = (val[i] as usize) % VOCAB;
        let next = (val[i + 1] as usize) % VOCAB;
        let embed_row: Vec<f32> =
            embed[prev * HIDDEN..(prev + 1) * HIDDEN].to_vec();
        let (_, _, mut logits) = forward_mlp(&embed_row, &w1, &w2);
        softmax_inplace(&mut logits);
        let p = logits[next].max(1e-30);
        total_nll -= (p.ln() as f64) / (LN2 as f64);
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        total_nll / count as f64
    }
}

fn parse_seeds() -> Vec<u64> {
    let mut out = Vec::new();
    for a in std::env::args().skip(1) {
        if let Some(v) = a.strip_prefix("--seeds=") {
            for tok in v.split(',') {
                if let Ok(s) = tok.parse::<u64>() {
                    out.push(s);
                }
            }
        }
    }
    if out.is_empty() {
        // Loop 156: default to 5 seeds (was 3 at Loop 155). The 77th-pass
        // SEV-2 #1 flagged the N=3 sample as borderline for statistical
        // significance; N=5 with 200 steps is the hardened budget.
        out = vec![42, 43, 44, 45, 46];
    }
    out
}

fn mean_std(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = if xs.len() > 1 {
        xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    (mean, var.sqrt())
}

fn main() {
    let seeds = parse_seeds();
    let train = load_tokens("data/tiny_shakespeare.txt");
    let val = load_tokens("data/tiny_shakespeare_val.txt");
    println!(
        "# bridge_bench (Loop 160: 2-layer MLP) — train_n={}, val_n={}, \
         vocab={VOCAB}, hidden={HIDDEN}, hidden_mlp={HIDDEN_MLP}, \
         steps={STEPS}, batch={BATCH}, lr={LR}, seeds={seeds:?}",
        train.len(),
        val.len()
    );
    if train.is_empty() || val.is_empty() {
        eprintln!("# FAIL  tiny_shakespeare data missing");
        std::process::exit(1);
    }

    let formats = [Format::F32, Format::Gf16, Format::Posit16];
    let result_dir = Path::new(".trinity/results");
    let _ = fs::create_dir_all(result_dir);

    let mut all_cells: Vec<serde_json::Value> = Vec::new();
    for &seed in &seeds {
        let mut per_seed = serde_json::Map::new();
        per_seed.insert("seed".to_string(), serde_json::json!(seed));
        let mut cells: Vec<serde_json::Value> = Vec::new();
        for &fmt in &formats {
            let bpb = run_one(fmt, seed, &train, &val);
            let cell = serde_json::json!({
                "format": fmt.slug(),
                "seed": seed,
                "val_bpb": bpb,
            });
            cells.push(cell.clone());
            all_cells.push(cell);
        }
        per_seed.insert("cells".to_string(), serde_json::json!(cells));
        let p = result_dir.join(format!("bridge_bench_seed{}.json", seed));
        fs::File::create(&p)
            .unwrap()
            .write_all(
                serde_json::to_string_pretty(&serde_json::Value::Object(per_seed))
                    .unwrap()
                    .as_bytes(),
            )
            .unwrap();
    }

    println!("\n## Final held-out val BPB (mean ± sample-std across seeds)");
    println!("    format    | mean    | std     | n_seeds");
    println!("    ----------+---------+---------+--------");
    let mut summary_rows = serde_json::Map::new();
    for &fmt in &formats {
        let bpbs: Vec<f64> = all_cells
            .iter()
            .filter(|c| c["format"].as_str() == Some(fmt.slug()))
            .filter_map(|c| c["val_bpb"].as_f64())
            .filter(|x| x.is_finite())
            .collect();
        let (m, s) = mean_std(&bpbs);
        println!(
            "    {:9} | {:.4}  | {:.4}  | {}",
            fmt.slug(),
            m,
            s,
            bpbs.len()
        );
        summary_rows.insert(
            fmt.slug().to_string(),
            serde_json::json!({
                "mean": m,
                "std": s,
                "n_seeds": bpbs.len(),
            }),
        );
    }

    let lo = seeds.iter().min().copied().unwrap_or(0);
    let hi = seeds.iter().max().copied().unwrap_or(0);
    let sum_path = result_dir.join(format!("bridge_bench_summary_seeds_{}-{}.json", lo, hi));
    let envelope = serde_json::json!({
        "tool": "bridge_bench",
        "mode": "summary",
        "seeds": seeds,
        "vocab": VOCAB,
        "hidden": HIDDEN,
        "steps": STEPS,
        "batch": BATCH,
        "lr": LR,
        "formats": formats.iter().map(|f| f.slug()).collect::<Vec<_>>(),
        "val_bpb_by_format": summary_rows,
        "git_anchor_hint": "f2-methodology branch HEAD at run time",
    });
    fs::File::create(&sum_path)
        .unwrap()
        .write_all(serde_json::to_string_pretty(&envelope).unwrap().as_bytes())
        .unwrap();
    println!("\n# Written summary: {}", sum_path.display());
}
