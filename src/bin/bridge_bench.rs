//! bridge_bench — actual training-time comparison of three format gates
//! on a sandbox-scale model.
//!
//! Loop 161 v4 architecture: **single-head self-attention block**.
//!
//!   input:    SEQ_LEN tokens (each 0..VOCAB) from a sliding window
//!   embed:    VOCAB × HIDDEN
//!   Q, K, V:  HIDDEN × HIDDEN_HEAD projections (HIDDEN_HEAD = HIDDEN
//!             — single head)
//!   attn:     softmax(Q @ K^T / sqrt(HIDDEN_HEAD)) → (SEQ_LEN×SEQ_LEN)
//!   context:  attn @ V → (SEQ_LEN × HIDDEN_HEAD)
//!   output:   context[last] @ W_O → logits (VOCAB)
//!
//! Five weight matrices are round-tripped through the chosen format
//! after every SGD step: embed, W_Q, W_K, W_V, W_O. This puts the
//! format-zoo comparison one architectural step closer to a real
//! transformer block (the next rung would be multi-head + layer
//! normalization + position embeddings, but the §9.4.3 framing
//! deliberately keeps the model minimal).
//!
//! History:
//!   Loop 155 — initial bigram model (1 weight matrix quantized).
//!   Loop 156 — STEPS 50 → 200, seeds 3 → 5 hardening.
//!   Loop 160 — bigram → 2-layer MLP (3 weight matrices quantized).
//!   Loop 161 — MLP → single-head attention (5 weight matrices).
//!
//! Usage:
//!   cargo run --release --bin bridge_bench -- [--seeds=42,43,44,45,46]

use std::fs;
use std::io::Write;
use std::path::Path;

use trios_trainer::gf16::GF16;
use trios_trainer::phi_numbers::Posit16;

const VOCAB: usize = 128;
// Loop 162: HIDDEN 64 → 128, STEPS 200 → 800. The 200-step / HIDDEN=64
// budget from Loop 161 placed the attention model on a high BPB
// plateau (4.83) where the GF16 penalty was rank-stable but not
// statistically significant at N=5 (t ≈ 1.7, below the 2.78 threshold
// at α=0.05, df=4). The Loop 162 hardening lifts the model into a
// converged regime where the format-quantization penalty is again
// observable above seed noise.
const HIDDEN: usize = 128;
const HIDDEN_HEAD: usize = HIDDEN; // single head
const SEQ_LEN: usize = 8;
const STEPS: usize = 800;
const BATCH: usize = 64;
// LR 0.5 inherited unchanged from Loop 156 / 160; all 5 seeds converge
// uniformly with no divergence — no LR re-tune for the attention upgrade.
const LR: f32 = 0.5;
const LN2: f32 = std::f32::consts::LN_2;

#[derive(Clone, Copy, Debug)]
enum Format {
    F32,
    Gf16,
    Posit16,
    /// BitNet b1.58 ternary {-1, 0, +1} with per-tensor abs-mean scale α.
    /// q = clip(round(w/α), -1, +1) * α, where α = mean(|w|). Per Ma et al.
    /// 2024 (arXiv:2402.17764) §2.1 — the "quantization function" used by
    /// the BitNet b1.58 paper as the reference 1.58-bit scheme.
    Bitnet158,
    /// INT4 round-to-nearest with per-tensor symmetric scale s = max(|w|)/7.
    /// q = clip(round(w/s), -8, +7) * s. The canonical RTN-INT4 baseline
    /// referenced by GPTQ (Frantar et al. 2210.17323 §3.1) before any
    /// Hessian-based error correction.
    Int4,
    /// bf16 = top 16 bits of f32 (sign + 8 exp + 7 mantissa). Per-element
    /// truncation, same as `convert_f32_to_bf16` in `format_ladder.rs`.
    Bf16,
}

impl Format {
    fn slug(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Gf16 => "gf16",
            Self::Posit16 => "posit16",
            Self::Bitnet158 => "bitnet158",
            Self::Int4 => "int4",
            Self::Bf16 => "bf16",
        }
    }

    /// In-place per-tensor quantization (the format gate as applied after
    /// each SGD step to every weight matrix in the shadow-weight pattern).
    /// Per-element formats (F32 / Gf16 / Posit16 / Bf16) ignore the tensor
    /// scope; per-tensor formats (Bitnet158 / Int4) compute their scale
    /// across the whole tensor before quantizing each element.
    fn quantize_tensor(self, w: &mut [f32]) {
        match self {
            Self::F32 => {}
            Self::Gf16 => {
                for x in w.iter_mut() {
                    *x = GF16::from_f32(*x).to_f32();
                }
            }
            Self::Posit16 => {
                for x in w.iter_mut() {
                    *x = Posit16::from_f32(*x).to_f32();
                }
            }
            Self::Bitnet158 => {
                // α = mean(|w|); if zero, leave the tensor untouched
                // (a fresh-init layer with no SGD update would otherwise
                // collapse to all-zeros).
                let mean_abs: f32 = w.iter().map(|x| x.abs()).sum::<f32>()
                    / (w.len().max(1) as f32);
                if mean_abs < 1e-30 {
                    return;
                }
                for x in w.iter_mut() {
                    let scaled = *x / mean_abs;
                    let q = scaled.round().clamp(-1.0, 1.0);
                    *x = q * mean_abs;
                }
            }
            Self::Int4 => {
                let max_abs: f32 = w
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0_f32, f32::max);
                let s = max_abs / 7.0;
                if s < 1e-30 {
                    return;
                }
                for x in w.iter_mut() {
                    let q = (*x / s).round().clamp(-8.0, 7.0);
                    *x = q * s;
                }
            }
            Self::Bf16 => {
                for x in w.iter_mut() {
                    let bits = x.to_bits() & 0xFFFF_0000;
                    *x = f32::from_bits(bits);
                }
            }
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

fn softmax_row(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in row.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in row.iter_mut() {
            *x /= sum;
        }
    }
}

/// Matrix multiply: out (M×N) = a (M×K) @ b (K×N), row-major flat.
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f32;
            for kk in 0..k {
                s += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = s;
        }
    }
}

/// Outer product accumulate: out (M×N) += a (M) outer b (N).
fn outer_accum(a: &[f32], b: &[f32], m: usize, n: usize, out: &mut [f32]) {
    for i in 0..m {
        let ai = a[i];
        for j in 0..n {
            out[i * n + j] += ai * b[j];
        }
    }
}

/// Forward attention block. Returns (logits, cache for backward).
/// Cache contains everything needed for backprop.
struct AttnCache {
    embed_rows: Vec<f32>,   // SEQ_LEN × HIDDEN
    q: Vec<f32>,            // SEQ_LEN × HIDDEN_HEAD
    k: Vec<f32>,            // SEQ_LEN × HIDDEN_HEAD
    v: Vec<f32>,            // SEQ_LEN × HIDDEN_HEAD
    attn_probs: Vec<f32>,   // SEQ_LEN × SEQ_LEN
    context: Vec<f32>,      // SEQ_LEN × HIDDEN_HEAD
}

#[allow(clippy::too_many_arguments)]
fn forward(
    embed: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_o: &[f32],
    tokens: &[usize],
) -> (Vec<f32>, AttnCache) {
    // Gather embeddings for the SEQ_LEN tokens.
    let mut embed_rows = vec![0.0_f32; SEQ_LEN * HIDDEN];
    for (i, &t) in tokens.iter().enumerate() {
        embed_rows[i * HIDDEN..(i + 1) * HIDDEN]
            .copy_from_slice(&embed[t * HIDDEN..(t + 1) * HIDDEN]);
    }
    // Q, K, V projections: SEQ_LEN × HIDDEN_HEAD.
    let mut q = vec![0.0_f32; SEQ_LEN * HIDDEN_HEAD];
    let mut k = vec![0.0_f32; SEQ_LEN * HIDDEN_HEAD];
    let mut v = vec![0.0_f32; SEQ_LEN * HIDDEN_HEAD];
    matmul(&embed_rows, w_q, SEQ_LEN, HIDDEN, HIDDEN_HEAD, &mut q);
    matmul(&embed_rows, w_k, SEQ_LEN, HIDDEN, HIDDEN_HEAD, &mut k);
    matmul(&embed_rows, w_v, SEQ_LEN, HIDDEN, HIDDEN_HEAD, &mut v);
    // Attention scores: Q @ K^T (SEQ_LEN × SEQ_LEN), scaled by 1/sqrt(d).
    let scale = 1.0_f32 / (HIDDEN_HEAD as f32).sqrt();
    let mut scores = vec![0.0_f32; SEQ_LEN * SEQ_LEN];
    for i in 0..SEQ_LEN {
        for j in 0..SEQ_LEN {
            let mut s = 0.0_f32;
            for d in 0..HIDDEN_HEAD {
                s += q[i * HIDDEN_HEAD + d] * k[j * HIDDEN_HEAD + d];
            }
            scores[i * SEQ_LEN + j] = s * scale;
        }
    }
    // Softmax per row → attn_probs.
    let mut attn_probs = scores.clone();
    for i in 0..SEQ_LEN {
        let row = &mut attn_probs[i * SEQ_LEN..(i + 1) * SEQ_LEN];
        softmax_row(row);
    }
    // Context = attn_probs @ V (SEQ_LEN × HIDDEN_HEAD).
    let mut context = vec![0.0_f32; SEQ_LEN * HIDDEN_HEAD];
    matmul(&attn_probs, &v, SEQ_LEN, SEQ_LEN, HIDDEN_HEAD, &mut context);
    // Logits = context[last] @ W_O (HIDDEN_HEAD × VOCAB) → VOCAB.
    let last_ctx = &context[(SEQ_LEN - 1) * HIDDEN_HEAD..SEQ_LEN * HIDDEN_HEAD];
    let mut logits = vec![0.0_f32; VOCAB];
    for vi in 0..VOCAB {
        let mut s = 0.0_f32;
        for d in 0..HIDDEN_HEAD {
            s += last_ctx[d] * w_o[d * VOCAB + vi];
        }
        logits[vi] = s;
    }
    let cache = AttnCache { embed_rows, q, k, v, attn_probs, context };
    (logits, cache)
}

#[allow(clippy::too_many_arguments)]
fn backward(
    cache: &AttnCache,
    tokens: &[usize],
    d_logits: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_o: &[f32],
    d_embed: &mut [f32],
    d_w_q: &mut [f32],
    d_w_k: &mut [f32],
    d_w_v: &mut [f32],
    d_w_o: &mut [f32],
) {
    // d_W_O[d, v] += last_ctx[d] * d_logits[v]
    let last_ctx = &cache.context[(SEQ_LEN - 1) * HIDDEN_HEAD..SEQ_LEN * HIDDEN_HEAD];
    for d in 0..HIDDEN_HEAD {
        let ld = last_ctx[d];
        for vi in 0..VOCAB {
            d_w_o[d * VOCAB + vi] += ld * d_logits[vi];
        }
    }
    // d_last_ctx[d] = Σ_v d_logits[v] * W_O[d, v]
    let mut d_context = vec![0.0_f32; SEQ_LEN * HIDDEN_HEAD];
    for d in 0..HIDDEN_HEAD {
        let mut s = 0.0_f32;
        for vi in 0..VOCAB {
            s += d_logits[vi] * w_o[d * VOCAB + vi];
        }
        d_context[(SEQ_LEN - 1) * HIDDEN_HEAD + d] = s;
    }
    // d_attn_probs = d_context @ V^T  (SEQ_LEN × SEQ_LEN)
    // d_V = attn_probs^T @ d_context  (SEQ_LEN × HIDDEN_HEAD)
    let mut d_attn_probs = vec![0.0_f32; SEQ_LEN * SEQ_LEN];
    for i in 0..SEQ_LEN {
        for j in 0..SEQ_LEN {
            let mut s = 0.0_f32;
            for d in 0..HIDDEN_HEAD {
                s += d_context[i * HIDDEN_HEAD + d] * cache.v[j * HIDDEN_HEAD + d];
            }
            d_attn_probs[i * SEQ_LEN + j] = s;
        }
    }
    let mut d_v = vec![0.0_f32; SEQ_LEN * HIDDEN_HEAD];
    for j in 0..SEQ_LEN {
        for d in 0..HIDDEN_HEAD {
            let mut s = 0.0_f32;
            for i in 0..SEQ_LEN {
                s += cache.attn_probs[i * SEQ_LEN + j] * d_context[i * HIDDEN_HEAD + d];
            }
            d_v[j * HIDDEN_HEAD + d] = s;
        }
    }
    // Softmax backward (per row): d_scores = p * (d_p - Σ_k p_k * d_p_k)
    let mut d_scores = vec![0.0_f32; SEQ_LEN * SEQ_LEN];
    for i in 0..SEQ_LEN {
        let p = &cache.attn_probs[i * SEQ_LEN..(i + 1) * SEQ_LEN];
        let dp = &d_attn_probs[i * SEQ_LEN..(i + 1) * SEQ_LEN];
        let mut dot = 0.0_f32;
        for jj in 0..SEQ_LEN {
            dot += p[jj] * dp[jj];
        }
        for jj in 0..SEQ_LEN {
            d_scores[i * SEQ_LEN + jj] = p[jj] * (dp[jj] - dot);
        }
    }
    // scores = Q @ K^T * scale → d_Q = d_scores @ K * scale,
    //                            d_K = d_scores^T @ Q * scale
    let scale = 1.0_f32 / (HIDDEN_HEAD as f32).sqrt();
    let mut d_q = vec![0.0_f32; SEQ_LEN * HIDDEN_HEAD];
    let mut d_k = vec![0.0_f32; SEQ_LEN * HIDDEN_HEAD];
    for i in 0..SEQ_LEN {
        for d in 0..HIDDEN_HEAD {
            let mut s = 0.0_f32;
            for j in 0..SEQ_LEN {
                s += d_scores[i * SEQ_LEN + j] * cache.k[j * HIDDEN_HEAD + d];
            }
            d_q[i * HIDDEN_HEAD + d] = s * scale;
        }
    }
    for j in 0..SEQ_LEN {
        for d in 0..HIDDEN_HEAD {
            let mut s = 0.0_f32;
            for i in 0..SEQ_LEN {
                s += d_scores[i * SEQ_LEN + j] * cache.q[i * HIDDEN_HEAD + d];
            }
            d_k[j * HIDDEN_HEAD + d] = s * scale;
        }
    }
    // d_W_Q[h, d] += Σ_i embed_rows[i, h] * d_q[i, d]
    // d_W_K, d_W_V analogous.
    for i in 0..SEQ_LEN {
        let er = &cache.embed_rows[i * HIDDEN..(i + 1) * HIDDEN];
        outer_accum(er, &d_q[i * HIDDEN_HEAD..(i + 1) * HIDDEN_HEAD], HIDDEN, HIDDEN_HEAD, d_w_q);
        outer_accum(er, &d_k[i * HIDDEN_HEAD..(i + 1) * HIDDEN_HEAD], HIDDEN, HIDDEN_HEAD, d_w_k);
        outer_accum(er, &d_v[i * HIDDEN_HEAD..(i + 1) * HIDDEN_HEAD], HIDDEN, HIDDEN_HEAD, d_w_v);
    }
    // d_embed_rows[i, h] = Σ_d (d_q[i,d]*W_Q[h,d] + d_k[i,d]*W_K[h,d] + d_v[i,d]*W_V[h,d])
    for i in 0..SEQ_LEN {
        let token = tokens[i];
        for h in 0..HIDDEN {
            let mut s = 0.0_f32;
            for d in 0..HIDDEN_HEAD {
                s += d_q[i * HIDDEN_HEAD + d] * w_q[h * HIDDEN_HEAD + d];
                s += d_k[i * HIDDEN_HEAD + d] * w_k[h * HIDDEN_HEAD + d];
                s += d_v[i * HIDDEN_HEAD + d] * w_v[h * HIDDEN_HEAD + d];
            }
            d_embed[token * HIDDEN + h] += s;
        }
    }
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

fn run_one(fmt: Format, seed: u64, train: &[u8], val: &[u8]) -> f64 {
    let mut embed = xavier_init(seed, VOCAB * HIDDEN, HIDDEN);
    let mut w_q = xavier_init(seed.wrapping_add(1), HIDDEN * HIDDEN_HEAD, HIDDEN);
    let mut w_k = xavier_init(seed.wrapping_add(2), HIDDEN * HIDDEN_HEAD, HIDDEN);
    let mut w_v = xavier_init(seed.wrapping_add(3), HIDDEN * HIDDEN_HEAD, HIDDEN);
    let mut w_o = xavier_init(seed.wrapping_add(4), HIDDEN_HEAD * VOCAB, HIDDEN_HEAD);

    let mut step_rng = Lcg::new(seed.wrapping_add(5));
    let n_train_windows = train.len().saturating_sub(SEQ_LEN);
    if n_train_windows == 0 {
        return f64::NAN;
    }

    for _step in 0..STEPS {
        let mut d_embed = vec![0.0_f32; embed.len()];
        let mut d_w_q = vec![0.0_f32; w_q.len()];
        let mut d_w_k = vec![0.0_f32; w_k.len()];
        let mut d_w_v = vec![0.0_f32; w_v.len()];
        let mut d_w_o = vec![0.0_f32; w_o.len()];

        for _ in 0..BATCH {
            let start = step_rng.pick(n_train_windows);
            let tokens: Vec<usize> = (0..SEQ_LEN)
                .map(|i| (train[start + i] as usize) % VOCAB)
                .collect();
            let next = (train[start + SEQ_LEN - 1] as usize) % VOCAB;
            // Predict the byte right AFTER the window's last position.
            // (Use start + SEQ_LEN if available; else wrap to next.)
            let next = if start + SEQ_LEN < train.len() {
                (train[start + SEQ_LEN] as usize) % VOCAB
            } else {
                next
            };

            let (mut logits, cache) = forward(&embed, &w_q, &w_k, &w_v, &w_o, &tokens);
            softmax_inplace(&mut logits);
            let mut d_logits = logits.clone();
            d_logits[next] -= 1.0;
            let scale = 1.0_f32 / BATCH as f32;
            for x in d_logits.iter_mut() {
                *x *= scale;
            }
            backward(
                &cache,
                &tokens,
                &d_logits,
                &w_q, &w_k, &w_v, &w_o,
                &mut d_embed,
                &mut d_w_q, &mut d_w_k, &mut d_w_v, &mut d_w_o,
            );
        }

        // SGD update on all 5 weight matrices, followed by per-tensor
        // format-gate quantization (the shadow-weight pattern: master
        // stays in f32 between forward passes; each tensor is round-
        // tripped through the chosen format after every step).
        for (w, dw) in [
            (&mut embed, &d_embed),
            (&mut w_q, &d_w_q),
            (&mut w_k, &d_w_k),
            (&mut w_v, &d_w_v),
            (&mut w_o, &d_w_o),
        ] {
            for i in 0..w.len() {
                w[i] -= LR * dw[i];
            }
            fmt.quantize_tensor(w);
        }
    }

    // Held-out BPB: average -log_2 P(next | last SEQ_LEN tokens) over val.
    let mut total_nll = 0.0_f64;
    let mut count = 0_usize;
    let n_val_windows = val.len().saturating_sub(SEQ_LEN);
    for start in 0..n_val_windows {
        let tokens: Vec<usize> = (0..SEQ_LEN)
            .map(|i| (val[start + i] as usize) % VOCAB)
            .collect();
        let next = (val[start + SEQ_LEN] as usize) % VOCAB;
        let (mut logits, _) = forward(&embed, &w_q, &w_k, &w_v, &w_o, &tokens);
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
        "# bridge_bench (Loop 161: single-head attention) — train_n={}, \
         val_n={}, vocab={VOCAB}, hidden={HIDDEN}, hidden_head={HIDDEN_HEAD}, \
         seq_len={SEQ_LEN}, steps={STEPS}, batch={BATCH}, lr={LR}, seeds={seeds:?}",
        train.len(),
        val.len()
    );
    if train.is_empty() || val.is_empty() {
        eprintln!("# FAIL  tiny_shakespeare data missing");
        std::process::exit(1);
    }

    let formats = [
        Format::F32,
        Format::Gf16,
        Format::Posit16,
        Format::Bitnet158,
        Format::Int4,
        Format::Bf16,
    ];
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
        "hidden_head": HIDDEN_HEAD,
        "seq_len": SEQ_LEN,
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
