//! Hybrid N-gram + Causal Attention Training — L-f1 + L-f2 Lanes
//!
//! Architecture: n-gram base + HybridAttn (causal, RoPE, qk_gain=φ²) + ReLU²
//! - NGRAM=8, DIM=64, HIDDEN=828 (φ-scaled), VOCAB=128, NUM_CTX=6
//! - 1-layer causal attention (HybridAttn from crate::model_hybrid_attn)
//! - ReLU² activation with proper backward
//! - EMA val BPB (β=φ⁻¹), cosine LR, gradient accumulation
//!
//! Run: cargo run --release --bin hybrid_train -- --seed=43 --steps=54000

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_mut)]

use std::env;
use std::fs;
use std::time::Instant;

use trios_trainer::model_hybrid_attn::HybridAttn;
use trios_trainer::neon_writer;
use trios_trainer::optimizer::MuonOptimizer;

const VOCAB: usize = 128;
const DIM: usize = 64;
const DEFAULT_HIDDEN: usize = 828;
const NUM_CTX: usize = 6;
const NGRAM: usize = NUM_CTX + 2;
const SEQ: usize = 128;
const LN_2: f32 = std::f32::consts::LN_2;
const PHI_INV: f32 = 0.618033988749895;
const GF16_FLOOR_FRAC: f32 = 0.7;
/// Seeds forbidden under Canon #93. Mirrors `seed_canon::parse_seed` and
/// `trios-train`, so no binary in this crate can start a run whose results
/// another binary would refuse to publish.
const FORBIDDEN_SEEDS: [u64; 4] = [42, 43, 44, 45];
/// Default sweep, taken from the crate constant rather than copied: this file
/// used to hard-code `[42, 43, 44]`, all three of them forbidden.
const GATE_FINAL_SEEDS: &[u64] = trios_trainer::train_loop::GATE_FINAL_SEEDS;
const DEFAULT_TRAIN_PATH: &str = "data/tiny_shakespeare.txt";
const DEFAULT_VAL_PATH: &str = "data/tiny_shakespeare_val.txt";
const CTX_WEIGHTS: [f32; NUM_CTX] = [0.70, 0.45, 0.30, 0.20, 0.13, 0.08];
/// Windows `evaluate` averages over. Named so the corpus precondition in
/// `main` can ask its question at the coverage this binary actually uses: a
/// chunk count computed from a different chunking is not a precondition.
const EVAL_NUM_CHUNKS: usize = 40;
const NCA_WEIGHT: f32 = 0.25;
const NCA_K: usize = 9;
const NCA_ENTROPY_MIN: f32 = 1.5;
const NCA_ENTROPY_MAX: f32 = 2.8;
/// Probability floor for the NCA entropy REGULARISER only - never for a loss
/// or an eval reading.
///
/// `-sum p*ln p` needs the `0*ln 0 = 0` limit, and this floor supplies it:
/// at `p == 0` the term is `0 * ln(1e-10) == 0`, the mathematically correct
/// value. It is named rather than written inline so that a search for a
/// probability clamp in this file finds nothing in a measurement path: the
/// identical expression in `loss_on_seq` was fabricating 33.21928 bpb, this
/// one shapes a gradient and is never reported as a number.
const NCA_ENTROPY_PROB_FLOOR: f32 = 1e-10;

/// Read a corpus from disk. A missing or empty file is an error.
///
/// The previous version substituted `"The quick brown fox ...".repeat(100)` on
/// a read failure: 45 distinct bytes that this architecture memorises to a BPB
/// sitting comfortably between `JEPA_PROXY_BPB_FLOOR` and `BPB_CHAMPION`, so
/// every guard downstream passed it and the ledger read it as a breakthrough.
fn load_data(path: &str) -> Result<Vec<usize>, String> {
    let raw = fs::read(path).map_err(|e| {
        format!(
            "cannot read corpus {path}: {e}. This trainer has no fallback corpus; \
             provide the file and re-run."
        )
    })?;
    if raw.is_empty() {
        return Err(format!("corpus {path} is empty"));
    }
    Ok(raw.into_iter().map(|b| (b as usize) % VOCAB).collect())
}

/// Refuse a seed forbidden under Canon #93.
///
/// A forbidden seed is a reason to refuse a run, not a constant to preserve:
/// results measured under 42/43/44/45 cannot be published by `trios-train` or
/// by the ledger, so producing them costs compute and buys nothing.
fn canon_check_seed(seed: u64) -> Result<(), String> {
    if FORBIDDEN_SEEDS.contains(&seed) {
        return Err(format!(
            "Canon #93 violation: seed {seed} is forbidden (allowed: 47, 89, 123, 144). \
             Pass --seed=<allowed>."
        ));
    }
    Ok(())
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

    dy.iter()
        .zip(y.iter())
        .map(|(d, yi)| (d - sum_dy / n - yi * sum_dy_y / n) * std_inv)
        .collect()
}

fn softmax(v: &mut [f32]) {
    let max_val = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    for x in v.iter_mut() {
        *x /= sum;
    }
}

fn cosine_lr(step: usize, max_steps: usize, base_lr: f32, warmup: usize) -> f32 {
    if step < warmup {
        return base_lr * step as f32 / warmup.max(1) as f32;
    }
    let p = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
    1e-5 + (base_lr - 1e-5) * 0.5 * (1.0 + (std::f32::consts::PI * p).cos())
}

struct AdamW {
    m: Vec<f32>,
    v: Vec<f32>,
    step: usize,
    beta1: f32,
    beta2: f32,
    wd: f32,
}

impl AdamW {
    fn new(size: usize, wd: f32) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            step: 0,
            beta1: 0.9,
            beta2: 0.999,
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
            params[i] -= lr * (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + 1e-8);
        }
    }
}

struct HybridModel {
    embed: Vec<f32>,
    ctx: Vec<Vec<f32>>,
    proj: Vec<f32>,
    attn: HybridAttn,
    attn_down: Vec<f32>,
    attn_up: Vec<f32>,
    lm_head: Vec<f32>,
    hidden: usize,
}

impl HybridModel {
    fn new(hidden: usize, seed: u64) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let lim = (6.0f32 / (3 * DIM) as f32).sqrt();
        let lim_h = (6.0f32 / (DIM + hidden) as f32).sqrt();
        let lim_o = (6.0f32 / (hidden + VOCAB) as f32).sqrt();

        let attn = HybridAttn::new().expect("attn construct with defaults");
        let d = attn.config().d_model;
        let lim_down = (2.0f32 / (hidden + d) as f32).sqrt();
        let lim_up = (2.0f32 / (d + hidden) as f32).sqrt();

        let mut m = Self {
            embed: (0..VOCAB * DIM).map(|_| rng() * lim).collect(),
            ctx: (0..NUM_CTX)
                .map(|_| (0..VOCAB * DIM).map(|_| rng() * lim).collect())
                .collect(),
            proj: (0..hidden * DIM).map(|_| rng() * lim_h).collect(),
            attn,
            attn_down: (0..d * hidden).map(|_| rng() * lim_down).collect(),
            attn_up: (0..hidden * d).map(|_| rng() * lim_up).collect(),
            lm_head: (0..VOCAB * hidden).map(|_| rng() * lim_o).collect(),
            hidden,
        };

        let attn_cfg = m.attn.config();
        let d = attn_cfg.d_model;
        let attn_lim = (2.0f32 / d as f32).sqrt();
        for w in m.attn.wq_mut() {
            *w = rng() * attn_lim;
        }
        for w in m.attn.wk_mut() {
            *w = rng() * attn_lim;
        }
        for w in m.attn.wv_mut() {
            *w = rng() * attn_lim;
        }
        for w in m.attn.wo_mut() {
            *w = rng() * attn_lim;
        }

        m
    }

    fn embed_tokens(&self, tokens: &[usize], pos: usize) -> Vec<f32> {
        let t_last = tokens[pos + NGRAM - 1].min(VOCAB - 1);
        let mut combined = self.embed[t_last * DIM..(t_last + 1) * DIM].to_vec();
        for (ci, cw) in CTX_WEIGHTS.iter().enumerate() {
            let ctx_idx = NGRAM - 2 - ci;
            let t = tokens[pos + ctx_idx].min(VOCAB - 1);
            let cv = &self.ctx[ci][t * DIM..(t + 1) * DIM];
            for j in 0..DIM {
                combined[j] += cv[j] * cw;
            }
        }
        combined
    }

    fn forward_position(
        &self,
        tokens: &[usize],
        pos: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let h = self.hidden;
        let d = self.attn.config().d_model;
        let combined = self.embed_tokens(tokens, pos);
        let ln = layer_norm(&combined, 1e-5);

        let mut hidden_raw = vec![0.0f32; h];
        for hi in 0..h {
            for j in 0..DIM {
                hidden_raw[hi] += self.proj[hi * DIM + j] * ln[j];
            }
        }
        let mut hidden = vec![0.0f32; h];
        for hi in 0..h {
            hidden[hi] = if hidden_raw[hi] > 0.0 {
                hidden_raw[hi] * hidden_raw[hi]
            } else {
                0.0
            };
        }

        let mut attn_in = vec![0.0f32; d];
        for di in 0..d {
            for hi in 0..h {
                attn_in[di] += self.attn_down[di * h + hi] * hidden[hi];
            }
        }

        let mut attn_out_saved = vec![0.0f32; d];
        if let Ok(attn_out) = self.attn.forward(&attn_in, 1) {
            attn_out_saved = attn_out.clone();
            let mut attn_up_out = vec![0.0f32; h];
            for hi in 0..h {
                for di in 0..d {
                    attn_up_out[hi] += self.attn_up[hi * d + di] * attn_out[di];
                }
            }
            for hi in 0..h {
                hidden[hi] += attn_up_out[hi] * 0.1;
            }
        }

        let mut logits = vec![0.0f32; VOCAB];
        for vi in 0..VOCAB {
            for hi in 0..h {
                logits[vi] += self.lm_head[vi * h + hi] * hidden[hi];
            }
        }
        (combined, ln, hidden, logits, attn_out_saved)
    }

    /// Mean cross-entropy in nats, or `None` when the sequence is too short to
    /// hold an n-gram pair or the forward pass produced a non-number.
    ///
    /// A short sequence used to return `0.0`, a loss no model achieves, which
    /// averaged into `evaluate` as a real reading. Clamping `logits[target]` to
    /// a 1e-10 floor was the same defect one step further on: `f32::max`
    /// ignores NaN, so a poisoned forward pass became a finite 23.03-nat
    /// measurement, and a merely UNDERFLOWED probability - a finite `0.0` out
    /// of the f32 softmax, which `is_nan` and `is_finite` both accept - became
    /// the identical 23.02585 nats / 33.21928 bpb. That constant is the
    /// crate's documented fake-measurement signature, and `final_val_bpb` is a
    /// SEALED field, so it shipped inside an authenticated declaration. A
    /// probability that is not finite and strictly positive is an absence, and
    /// a mean is not reported over absences.
    fn loss_on_seq(&self, tokens: &[usize]) -> Option<f32> {
        if tokens.len() < NGRAM + 1 {
            return None;
        }
        let count = tokens.len() - NGRAM;
        let mut total = 0.0f32;
        for i in 0..count {
            let target = tokens[i + NGRAM].min(VOCAB - 1);
            let (_, _, _, mut logits, _) = self.forward_position(tokens, i);
            softmax(&mut logits);
            let p = logits[target];
            if !p.is_finite() || p <= 0.0 {
                return None;
            }
            total -= p.ln();
        }
        Some(total / count as f32)
    }
}

fn compute_grads_for_positions(
    model: &HybridModel,
    tokens: &[usize],
    positions: &[usize],
    g_embed: &mut [f32],
    g_ctx: &mut [Vec<f32>],
    g_proj: &mut [f32],
    g_head: &mut [f32],
    g_attn_down: &mut [f32],
    g_attn_up: &mut [f32],
) {
    let h = model.hidden;
    let d = model.attn.config().d_model;
    for &pos in positions {
        let (combined, ln, hidden, mut logits, attn_out_saved) =
            model.forward_position(tokens, pos);
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

        let nca_loss = nca_entropy_loss(&logits);
        if nca_loss > 0.0 {
            let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f32 = probs.iter().sum();
            for p in probs.iter_mut() {
                *p /= sum;
            }
            let entropy: f32 = -probs
                .iter()
                .map(|&p: &f32| p.max(NCA_ENTROPY_PROB_FLOOR).ln() * p)
                .sum::<f32>();
            let nca_grad = if entropy < NCA_ENTROPY_MIN {
                -2.0 * NCA_WEIGHT * (NCA_ENTROPY_MIN - entropy)
            } else {
                2.0 * NCA_WEIGHT * (entropy - NCA_ENTROPY_MAX)
            };
            for vi in 0..VOCAB {
                let d_ent = probs[vi] * (probs[vi].max(NCA_ENTROPY_PROB_FLOOR).ln() + entropy);
                for hi in 0..h {
                    g_head[vi * h + hi] += nca_grad * d_ent * hidden[hi] * 0.01;
                }
            }
        }

        let mut d_attn_up_out = vec![0.0f32; h];
        for hi in 0..h {
            d_attn_up_out[hi] = d_hidden[hi] * 0.1;
        }

        let mut d_attn_out = vec![0.0f32; d];
        for hi in 0..h {
            for di in 0..d {
                g_attn_up[hi * d + di] += d_attn_up_out[hi] * attn_out_saved[di];
                d_attn_out[di] += d_attn_up_out[hi] * model.attn_up[hi * d + di];
            }
        }

        let d_attn_in = d_attn_out;
        for di in 0..d {
            for hi in 0..h {
                g_attn_down[di * h + hi] += d_attn_in[di] * hidden[hi];
            }
        }

        let mut d_raw = vec![0.0f32; h];
        for hi in 0..h {
            if hidden[hi] > 0.0 {
                let raw_val = hidden[hi].sqrt();
                d_raw[hi] = d_hidden[hi] * 2.0 * raw_val;
            }
        }

        let mut d_ln = vec![0.0f32; DIM];
        for hi in 0..h {
            for j in 0..DIM {
                g_proj[hi * DIM + j] += d_raw[hi] * ln[j];
                d_ln[j] += model.proj[hi * DIM + j] * d_raw[hi];
            }
        }

        let d_combined = layer_norm_backward(&combined, &ln, &d_ln, 1e-5);

        let t_last = tokens[pos + NGRAM - 1].min(VOCAB - 1);
        for j in 0..DIM {
            g_embed[t_last * DIM + j] += d_combined[j];
        }
        for (ci, cw) in CTX_WEIGHTS.iter().enumerate() {
            let ctx_idx = NGRAM - 2 - ci;
            let t = tokens[pos + ctx_idx].min(VOCAB - 1);
            for j in 0..DIM {
                g_ctx[ci][t * DIM + j] += cw * d_combined[j];
            }
        }
    }
}

/// Mean bits-per-byte over the held-out corpus, or `None` when nothing could
/// be measured.
///
/// The two "f32 maximum" (3.4e38) returns were sentinels that `is_finite()`
/// accepts, so a
/// failed evaluation was indistinguishable from a reading everywhere downstream
/// - including the printed headline and the ledger.
///
/// It then still dropped individual non-finite windows and published the mean
/// of the survivors, which is biased DOWNWARD: the windows a partial poison
/// kills are exactly the hard ones. `src/bin/trinity_pr1722.rs` takes the
/// correct line and this now matches it -- ONE unmeasurable window invalidates
/// the whole eval -- while the `dropped` counter keeps the skip from being
/// silent about HOW MUCH of the corpus failed.
fn evaluate(model: &HybridModel, tokens: &[usize]) -> Option<f32> {
    let chunk_size = SEQ + 1;
    let num_chunks = EVAL_NUM_CHUNKS;
    let max_start = tokens.len().saturating_sub(chunk_size);
    if max_start == 0 {
        return None;
    }
    let step = if max_start >= num_chunks * chunk_size {
        max_start / num_chunks
    } else {
        chunk_size
    };
    let mut total = 0.0f32;
    let mut n = 0usize;
    let mut dropped = 0usize;
    for c in (0..max_start).step_by(step).take(num_chunks) {
        let end = (c + chunk_size).min(tokens.len());
        if end - c < NGRAM + 2 {
            continue;
        }
        match model.loss_on_seq(&tokens[c..end]) {
            Some(loss) if loss.is_finite() => {
                total += loss / LN_2;
                n += 1;
            }
            _ => dropped += 1,
        }
    }
    if dropped > 0 {
        eprintln!(
            "EVAL ABORTED: {dropped} of {} windows produced no finite loss. A mean \
             over the {n} survivors is biased DOWNWARD and is not a held-out \
             measurement, so this eval reports nothing.",
            dropped + n
        );
        return None;
    }
    if n == 0 {
        None
    } else {
        Some(total / n as f32)
    }
}

/// Render an optional BPB without inventing a number for an absent one.
fn fmt_bpb(bpb: Option<f32>) -> String {
    match bpb {
        Some(v) => format!("{v:.4}"),
        None => "unmeasured".to_string(),
    }
}

fn nca_entropy_loss(logits: &[f32]) -> f32 {
    let n = logits.len() as f32;
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = logits
        .iter()
        .map(|&x| (x - max_val).exp())
        .collect::<Vec<_>>();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let entropy: f32 = -probs
        .iter()
        .map(|&p: &f32| p.max(NCA_ENTROPY_PROB_FLOOR).ln() * p)
        .sum::<f32>();
    if entropy < NCA_ENTROPY_MIN {
        NCA_WEIGHT * (NCA_ENTROPY_MIN - entropy).powi(2)
    } else if entropy > NCA_ENTROPY_MAX {
        NCA_WEIGHT * (entropy - NCA_ENTROPY_MAX).powi(2)
    } else {
        0.0
    }
}

/// Value of `--name=VALUE` or `--name VALUE`, whichever form was used.
///
/// The `=`-only parser silently ignored the space-separated form: `--steps 2`
/// left `steps` at its 81000 default and the run looked like it had honoured
/// the flag.
fn arg_value(args: &[String], name: &str) -> Option<String> {
    for (i, a) in args.iter().enumerate() {
        let Some(rest) = a.strip_prefix(name) else {
            continue;
        };
        if let Some(v) = rest.strip_prefix('=') {
            return Some(v.to_string());
        }
        if rest.is_empty() {
            return args.get(i + 1).cloned();
        }
    }
    None
}

fn find_arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    arg_value(args, name)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn gf16_floor(weights: &mut [f32]) {
    let scale = 16.0_f32;
    for w in weights.iter_mut() {
        *w = (*w * scale).round() / scale;
    }
}

/// Every argument this binary reads, in the spellings `arg_value` implements.
///
/// `arg_value` accepts BOTH `--name=VALUE` and `--name VALUE` for every one of
/// them, so each appears twice: `trios_trainer::reject_unknown_args` reads the
/// pair as "this flag takes a value in either spelling".
const KNOWN_ARGS: [&str; 16] = [
    "seed",
    "seed=",
    "steps",
    "steps=",
    "lr",
    "lr=",
    "hidden",
    "hidden=",
    "eval-every",
    "eval-every=",
    "accum",
    "accum=",
    "train-path",
    "train-path=",
    "val-path",
    "val-path=",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    // The corpus spellings here are `--train-path` / `--val-path`. Given
    // `--train` / `--val` this binary dropped both, trained the DEFAULT split
    // and published `bpb=5.0731` with exit 0 (measured at --steps=20). The
    // argument check therefore runs before anything is read, trained or
    // printed. See `trios_trainer::reject_unknown_args`.
    if let Err(reason) = trios_trainer::reject_unknown_args(&args, &KNOWN_ARGS) {
        eprintln!("{reason}");
        std::process::exit(i32::from(trios_trainer::EXIT_BAD_ARGS));
    }
    let seed: u64 = find_arg(&args, "--seed", 0u64);
    let steps: usize = find_arg(&args, "--steps", 81000usize);
    let base_lr: f32 = find_arg(&args, "--lr", 0.003f32);
    let hidden: usize = find_arg(&args, "--hidden", DEFAULT_HIDDEN);
    let eval_every: usize = find_arg(&args, "--eval-every", 1000usize);
    let accum: usize = find_arg(&args, "--accum", 4usize);
    let train_path =
        arg_value(&args, "--train-path").unwrap_or_else(|| DEFAULT_TRAIN_PATH.to_string());
    let val_path = arg_value(&args, "--val-path").unwrap_or_else(|| DEFAULT_VAL_PATH.to_string());

    let gf16_floor_step = (GF16_FLOOR_FRAC * steps as f32).floor() as usize;
    // Shared with `train_loop` so there is exactly one definition of the knob.
    // `gf16_floor` mutates the weights, so its cadence belongs to the recipe;
    // gating it on `eval_every` let an observation parameter change the model.
    let gf16_every = trios_trainer::train_loop::gf16_floor_every() as usize;

    let seeds: Vec<u64> = if seed > 0 {
        vec![seed]
    } else {
        GATE_FINAL_SEEDS.to_vec()
    };

    // Canon #93 at the effective post-config seed: whatever the sweep or the
    // flag resolved to is what the run would train under.
    for &s in &seeds {
        canon_check_seed(s)?;
    }

    for &seed in &seeds {
        eprintln!("=== Hybrid Train (ngram+attn) seed={} ===", seed);
        eprintln!(
            "steps={} lr={} hidden={} eval_every={} gf16_floor_every={} accum={}",
            steps, base_lr, hidden, eval_every, gf16_every, accum
        );
        eprintln!(
            "DIM={} NUM_CTX={} NGRAM={} SEQ={} VOCAB={}",
            DIM, NUM_CTX, NGRAM, SEQ, VOCAB
        );
        eprintln!("ctx_weights={:?}", CTX_WEIGHTS);

        let train_data = load_data(&train_path)?;
        let val_data = load_data(&val_path)?;
        eprintln!("train={} val={}", train_data.len(), val_data.len());

        // This binary writes ledger rows through `neon_writer::bpb_sample` and
        // carried NO corpus precondition: a run with `--train-path` equal to
        // `--val-path` completed and reported bpb=3.9591, and only an unset DSN
        // kept the row off the ledger. `reject_bpb`'s floor of 2.0 cannot catch
        // it either, because a 100% verbatim overlap lands near 2.48 on this
        // architecture - above the floor and BELOW `BPB_CHAMPION` = 2.5193,
        // i.e. it reads as a new champion. The guard is the shared one, asked
        // at this binary's own eval coverage.
        trios_trainer::train_loop::check_train_val_disjoint(
            &train_data,
            &val_data,
            trios_trainer::train_loop::eval_chunk_count(val_data.len(), EVAL_NUM_CHUNKS),
        )
        .map_err(|reason| {
            format!(
                "SPLIT REFUSED: {reason} Refusing to train: no BPB measured \
                 against this split is a model result, and this binary publishes \
                 its BPB to the ledger."
            )
        })?;

        let mut model = HybridModel::new(hidden, seed);

        let d = model.attn.config().d_model;
        let attn_params = d * model.hidden * 2;
        let total_params =
            VOCAB * DIM + NUM_CTX * VOCAB * DIM + hidden * DIM + VOCAB * hidden + attn_params;
        eprintln!(
            "params={} ({:.1}K) attn_d={}",
            total_params,
            total_params as f64 / 1000.0,
            d
        );

        let wd = 0.04f32;
        let mut opt_embed = AdamW::new(VOCAB * DIM, wd);
        let mut opt_ctx: Vec<AdamW> = (0..NUM_CTX).map(|_| AdamW::new(VOCAB * DIM, wd)).collect();
        let mut opt_proj = MuonOptimizer::with_matrix_shape(
            hidden * DIM,
            hidden,
            DIM,
            base_lr as f64,
            0.95,
            wd as f64,
        );
        let mut opt_attn_down = AdamW::new(d * hidden, wd);
        let mut opt_attn_up = AdamW::new(hidden * d, wd);
        let mut opt_head = AdamW::new(VOCAB * hidden, wd);

        let init_bpb = evaluate(&model, &val_data);
        eprintln!("Initial val_bpb={}", fmt_bpb(init_bpb));

        let mut best_ema_bpb: Option<f32> = init_bpb;
        let mut ema_bpb: Option<f32> = init_bpb;
        // The raw reading from the last evaluation this run TOOK. The headline
        // used to be `best_ema_bpb`: a running minimum of a phi-inverse-weighted
        // EMA seeded at init, which depends on `--eval-every` and is therefore
        // not comparable between runs with different eval cadence.
        let mut final_val_bpb: Option<f32> = None;
        let t0 = Instant::now();
        let warmup = steps / 10;

        let mut rng_s = seed.wrapping_add(7919);

        for step in 1..=steps {
            let lr = cosine_lr(step, steps, base_lr, warmup);

            let mut g_embed = vec![0.0f32; VOCAB * DIM];
            let mut g_ctx: Vec<Vec<f32>> =
                (0..NUM_CTX).map(|_| vec![0.0f32; VOCAB * DIM]).collect();
            let mut g_proj = vec![0.0f32; hidden * DIM];
            let mut g_head = vec![0.0f32; VOCAB * hidden];
            let mut g_attn_down = vec![0.0f32; d * hidden];
            let mut g_attn_up = vec![0.0f32; hidden * d];

            for _micro in 0..accum {
                rng_s = rng_s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let dl = train_data.len();
                let max_start = dl.saturating_sub(SEQ + 1);
                if max_start == 0 {
                    continue;
                }
                let chunk_start = (rng_s as usize) % max_start;
                let chunk = &train_data[chunk_start..chunk_start + SEQ + 1];

                let count = chunk.len().saturating_sub(NGRAM);
                if count == 0 {
                    continue;
                }
                let num_sample = 8.min(count);
                let mut positions: Vec<usize> = Vec::with_capacity(num_sample);
                for _ in 0..num_sample {
                    rng_s = rng_s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let p = (rng_s as usize) % count;
                    positions.push(p);
                }

                compute_grads_for_positions(
                    &model,
                    chunk,
                    &positions,
                    &mut g_embed,
                    &mut g_ctx,
                    &mut g_proj,
                    &mut g_head,
                    &mut g_attn_down,
                    &mut g_attn_up,
                );
            }

            let total_positions = (accum * 8) as f32;
            for x in g_embed.iter_mut() {
                *x /= total_positions;
            }
            for gc in g_ctx.iter_mut() {
                for x in gc.iter_mut() {
                    *x /= total_positions;
                }
            }
            for x in g_proj.iter_mut() {
                *x /= total_positions;
            }
            for x in g_head.iter_mut() {
                *x /= total_positions;
            }
            for x in g_attn_down.iter_mut() {
                *x /= total_positions;
            }
            for x in g_attn_up.iter_mut() {
                *x /= total_positions;
            }

            opt_embed.update(&mut model.embed, &g_embed, lr);
            for (ci, oc) in opt_ctx.iter_mut().enumerate() {
                oc.update(&mut model.ctx[ci], &g_ctx[ci], lr);
            }
            opt_proj.step(&mut model.proj, &g_proj);
            opt_attn_down.update(&mut model.attn_down, &g_attn_down, lr);
            opt_attn_up.update(&mut model.attn_up, &g_attn_up, lr);
            opt_head.update(&mut model.lm_head, &g_head, lr);

            if step >= gf16_floor_step && step % gf16_every == 0 {
                gf16_floor(&mut model.embed);
                gf16_floor(&mut model.proj);
                gf16_floor(&mut model.lm_head);
                for c in &mut model.ctx {
                    gf16_floor(c);
                }
            }

            if step % eval_every == 0 || step == steps {
                let t = t0.elapsed().as_secs_f64();
                let Some(val_bpb) = evaluate(&model, &val_data) else {
                    println!("seed={seed} step={step} val_bpb=unmeasured t={t:.1}s");
                    continue;
                };
                final_val_bpb = Some(val_bpb);
                let ema = match ema_bpb {
                    Some(prev) => PHI_INV * prev + (1.0 - PHI_INV) * val_bpb,
                    None => val_bpb,
                };
                ema_bpb = Some(ema);
                if ema.is_finite() && best_ema_bpb.is_none_or(|b| ema < b) {
                    best_ema_bpb = Some(ema);
                }
                println!(
                    "seed={} step={} val_bpb={:.4} ema_bpb={:.4} best={} t={:.1}s",
                    seed,
                    step,
                    val_bpb,
                    ema,
                    fmt_bpb(best_ema_bpb),
                    t
                );
                // R5-honest ledger write to ssot.bpb_samples (ARCH writer hook).
                // No-op if TRIOS_CANON_NAME unset; safe to call every eval.
                if let Ok(canon) = std::env::var("TRIOS_CANON_NAME") {
                    if !canon.is_empty() {
                        neon_writer::bpb_sample(
                            &canon,
                            seed as i32,
                            step as i32,
                            val_bpb,
                            Some(ema),
                        );
                    }
                }
            }
        }

        // The headline is the measured final val_bpb, or an admission that no
        // measurement was taken - matching `trios-train`.
        match final_val_bpb {
            Some(v) => println!("seed={seed} bpb={v:.4}"),
            None => println!("seed={seed} bpb=unmeasured"),
        }
    }

    // A DSN was configured means this run was supposed to be recorded. If every
    // write was dropped, exiting 0 would let a supervisor file it as a success.
    let code = neon_writer::ledger_exit_code();
    if code != 0 {
        std::process::exit(code);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gf16_floor_rounds_to_16th() {
        let mut w = vec![0.123456, -0.987654, 0.0, 1.5];
        gf16_floor(&mut w);
        assert!((w[0] - 0.125).abs() < 1e-6, "0.123456 -> 0.125");
        assert!((w[1] - (-1.0)).abs() < 1e-6, "-0.987654 -> -1.0");
        assert!((w[2] - 0.0).abs() < 1e-6, "0.0 stays");
        assert!((w[3] - 1.5).abs() < 1e-6, "1.5 stays");
    }

    #[test]
    fn gf16_floor_step_at_70pct() {
        let steps = 81000;
        let floor_step = (GF16_FLOOR_FRAC * steps as f32).floor() as usize;
        assert_eq!(
            floor_step, 56700,
            "GF16 floor activates at 56700 for 81K steps"
        );
    }

    /// 42/43/44 used to BE the sweep, and this test asserted they stayed that
    /// way. A forbidden seed is a reason to refuse a run, not a constant to
    /// preserve, so the assertion is now refusal.
    #[test]
    fn canon_93_refuses_the_forbidden_seeds() {
        for seed in [42u64, 43, 44, 45] {
            let err = canon_check_seed(seed)
                .expect_err("a seed forbidden under Canon #93 must not start a run");
            assert!(err.contains("forbidden"), "{err}");
            assert!(err.contains(&seed.to_string()), "{err}");
        }
    }

    #[test]
    fn canon_93_admits_the_allowed_seeds() {
        for seed in [47u64, 89, 123, 144] {
            assert!(canon_check_seed(seed).is_ok(), "seed {seed} is allowed");
        }
    }

    /// The default sweep must be startable: every seed in it has to survive the
    /// same guard `main` applies.
    #[test]
    fn gate_final_seeds_are_canon_93_clean() {
        assert!(!GATE_FINAL_SEEDS.is_empty());
        for &seed in GATE_FINAL_SEEDS {
            assert!(
                canon_check_seed(seed).is_ok(),
                "default sweep contains a seed the binary itself refuses: {seed}"
            );
        }
    }

    #[test]
    fn phi_hidden_is_828() {
        assert_eq!(
            DEFAULT_HIDDEN, 828,
            "phi-scaled hidden = round(phi*512) = 828"
        );
    }

    #[test]
    fn ema_beta_is_phi_inv() {
        assert!((PHI_INV - 0.618033988749895).abs() < 1e-12);
    }

    /// `--steps 2` used to leave `steps` at its 81000 default because only the
    /// `--steps=2` form was recognised.
    #[test]
    fn arg_value_reads_both_forms() {
        let sp: Vec<String> = ["prog", "--steps", "2", "--train-path", "data/x.txt"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(arg_value(&sp, "--steps").as_deref(), Some("2"));
        assert_eq!(
            arg_value(&sp, "--train-path").as_deref(),
            Some("data/x.txt")
        );
        assert_eq!(find_arg(&sp, "--steps", 81000usize), 2usize);

        let eq: Vec<String> = ["prog", "--steps=2"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(find_arg(&eq, "--steps", 81000usize), 2usize);

        let none: Vec<String> = vec!["prog".to_string()];
        assert_eq!(arg_value(&none, "--steps"), None);
        assert_eq!(find_arg(&none, "--steps", 81000usize), 81000usize);
    }

    /// A missing corpus is an error, never a 45-byte pangram that memorises to
    /// a plausible-looking BPB.
    #[test]
    fn load_data_refuses_a_missing_corpus() {
        let err = load_data("data/definitely_absent_corpus_for_tests.txt")
            .expect_err("a missing corpus must be an error");
        assert!(err.contains("cannot read corpus"), "{err}");
        assert!(!err.contains("quick brown fox"), "{err}");
    }

    /// An evaluation that measured nothing is an absence, not 3.4e38 and
    /// not `0.0`.
    #[test]
    fn fmt_bpb_never_invents_a_number() {
        assert_eq!(fmt_bpb(None), "unmeasured");
        assert_eq!(fmt_bpb(Some(2.6141)), "2.6141");
    }

    /// An UNDERFLOWED target probability is an absence, not 33.21928 bpb.
    ///
    /// This binary publishes its BPB through `neon_writer::bpb_sample`, and
    /// `final_val_bpb` is a SEALED field. An f32 softmax returns a finite,
    /// exact `0.0` for a target the model finds impossible - no NaN, no
    /// infinity, so the previous `p.is_nan()` guard accepted it - and
    /// clamping `p` to a 1e-10 floor then contributed exactly 23.02585 nats,
    /// which is 33.21928 bpb: the constant this crate documents as the
    /// fake-measurement signature.
    #[test]
    fn an_underflowed_target_probability_is_an_absence_not_33_bpb() {
        let mut model = HybridModel::new(64, 47);
        // Exactly one scoring position, so one forward pass fixes the reading.
        let tokens: Vec<usize> = vec![1; NGRAM + 1];
        let target = tokens[NGRAM].min(VOCAB - 1);
        let h = model.hidden;
        let (_, _, hidden, _, _) = model.forward_position(&tokens, 0);
        let norm2: f32 = hidden.iter().map(|x| x * x).sum();
        assert!(norm2 > 0.0, "fixture needs a non-zero hidden state");

        // Put the target's logit 400 nats under the maximum. f32 `exp`
        // underflows to exactly 0.0 below about -104, so the target
        // probability is an honest zero and NOT a NaN.
        let big = if target == 0 { 1 } else { 0 };
        model.lm_head.iter_mut().for_each(|w| *w = 0.0);
        let scale = 400.0 / norm2;
        for hi in 0..h {
            model.lm_head[big * h + hi] = hidden[hi] * scale;
        }
        let (_, _, _, mut logits, _) = model.forward_position(&tokens, 0);
        softmax(&mut logits);
        let p = logits[target];
        assert_eq!(p, 0.0, "fixture must UNDERFLOW, not poison");
        assert!(!p.is_nan() && p.is_finite(), "and it must look measurable");

        assert_eq!(
            model.loss_on_seq(&tokens),
            None,
            "an underflowed probability is an absence, not a number"
        );

        // The reading the clamp used to manufacture, computed rather than
        // quoted, so this test fails loudly if the floor is reintroduced.
        let laundered_nats = -(1e-10f32).ln();
        let laundered_bpb = laundered_nats / LN_2;
        assert!((laundered_nats - 23.02585).abs() < 1e-3, "{laundered_nats}");
        assert!((laundered_bpb - 33.21928).abs() < 1e-3, "{laundered_bpb}");
    }

    #[test]
    fn evaluate_returns_absence_on_a_corpus_too_short_to_measure() {
        let model = HybridModel::new(64, 47);
        let tokens: Vec<usize> = vec![1; SEQ];
        assert_eq!(evaluate(&model, &tokens), None);
        assert_eq!(model.loss_on_seq(&tokens[..NGRAM]), None);
    }
}
