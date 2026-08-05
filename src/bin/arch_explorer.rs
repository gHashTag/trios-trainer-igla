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

use std::env;
use std::fs;
use std::io::Write;
use std::time::Instant;

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
        _ => {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Exit code for a corpus that could not be honestly loaded. Same value
/// `cpu_train` uses, so a sweep can tell a refused corpus from a crash.
const EXIT_BAD_CORPUS: i32 = 6;

/// Default corpus. The shipped file is `data/tiny_shakespeare.txt`; the old
/// default here spelled it without the underscore, and that file has never
/// existed in this repo, so every argument-free run took the fallback path and
/// reported a BPB measured on 52 bytes.
const DEFAULT_TRAIN_PATH: &str = "data/tiny_shakespeare.txt";

/// Opt-in placeholder, kept only so `TRIOS_ALLOW_SYNTHETIC_DATA=1` behaves as
/// documented. Nothing measured against it is a model result.
const SYNTHETIC_CORPUS: &[u8] = b"Hello world this is a tiny training dataset for IGLA";

/// Corpus identity, carried beside the tokens.
///
/// A trial line that cannot name the bytes it measured is indistinguishable
/// from a fabricated one, so path, size, digest and the synthetic flag travel
/// with every number this binary reports.
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
/// The old body substituted a 52-byte pangram whenever the read failed, so a
/// clean checkout explored architectures against it and wrote the resulting
/// BPB into `.trinity/experience`. A missing corpus is now a hard, named
/// refusal. `TRIOS_ALLOW_SYNTHETIC_DATA=1` opts back in, says so on stderr on
/// every run, and stamps `data_synthetic=true` into everything the run writes.
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
            let raw = SYNTHETIC_CORPUS.to_vec();
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
/// artifact is created, so a refused run leaves nothing behind.
fn load_or_refuse(path: &str) -> (Vec<usize>, CorpusInfo) {
    match load_data(path) {
        Ok(loaded) => loaded,
        Err(e) => {
            eprintln!("CORPUS REFUSED: {e}");
            std::process::exit(EXIT_BAD_CORPUS);
        }
    }
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
    fn new(size: usize, wd: f32) -> Self {
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
    fn new(
        vocab: usize,
        dim: usize,
        hidden: usize,
        activation: String,
        seed: u64,
        num_ctx: usize,
        weight_tying: bool,
    ) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let lim = (6.0f32 / (3 * dim) as f32).sqrt();
        let lim_h = (6.0f32 / (dim + hidden) as f32).sqrt();

        let ctx = (0..num_ctx)
            .map(|_| (0..vocab * dim).map(|_| rng() * lim).collect())
            .collect();

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
            vocab,
            dim,
            hidden,
            activation,
            weight_tying,
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
            if ctx_idx == 0 && ci > 0 {
                break;
            }
            let t = tokens_context[ctx_idx].min(v - 1);
            let cv = &self.ctx[ci][t * d..(t + 1) * d];
            for j in 0..d {
                combined[j] += cv[j] * cw;
            }
        }

        let ln = layer_norm(&combined, 1e-5);

        let mut hidden = vec![0.0f32; h];
        for hi in 0..h {
            let w = &self.proj[hi * d..(hi + 1) * d];
            for (j, l) in ln.iter().enumerate() {
                hidden[hi] += w[j] * l;
            }
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
            for (hi, hn) in hidden.iter().enumerate() {
                *logit += w[hi] * hn;
            }
        }
        logits
    }

    /// Mean cross-entropy in nats, or `None` when the sequence is too short to
    /// hold a single context/target pair or the forward pass produced a
    /// non-number.
    ///
    /// A short sequence used to return `0.0`, a loss no model achieves.
    /// `f32::max` also ignores NaN, so clamping `logits[target]` with
    /// `.max(1e-10)` turned a poisoned forward pass into a finite 23.03-nat
    /// measurement - 33.2 bpb,
    /// which is positive, finite and small enough that every downstream guard
    /// accepted it. NaN is now an absence. The 1e-10 clamp is kept for a
    /// genuinely underflowed probability - capping it is a documented floor on
    /// surprisal, and dropping those chunks instead would bias the reported
    /// BPB downward.
    fn loss_on_seq(&self, tokens: &[usize]) -> Option<f32> {
        let num_ctx = self.ctx.len();
        let ngram = num_ctx + 2;
        if tokens.len() < ngram + 1 {
            return None;
        }
        let count = tokens.len() - ngram;
        let mut total = 0.0f32;

        for i in 0..count {
            let context = &tokens[i..i + ngram];
            let hidden = self.compute_hidden(context);
            let target = tokens[i + ngram].min(self.vocab - 1);
            let mut logits = self.predict(&hidden);
            softmax(&mut logits);
            let p = logits[target];
            if p.is_nan() {
                return None;
            }
            total -= p.max(1e-10).ln();
        }
        Some(total / count as f32)
    }

    #[allow(clippy::needless_range_loop)]
    fn train_step(
        &mut self,
        tokens: &[usize],
        lr: f32,
        opt_embed: &mut AdamW,
        opt_ctx: &mut [AdamW],
        opt_proj: &mut AdamW,
        opt_head: &mut AdamW,
        clip: Option<f32>,
    ) {
        let num_ctx = self.ctx.len();
        let ngram = num_ctx + 2;
        if tokens.len() < ngram + 1 {
            return;
        }
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
                for j in 0..d {
                    combined[j] += cv[j] * cw;
                }
            }
            let ln = layer_norm(&combined, 1e-5);
            let mut pre_act = vec![0.0f32; h];
            let mut hidden = vec![0.0f32; h];
            for hi in 0..h {
                let w = &self.proj[hi * d..(hi + 1) * d];
                for (j, l) in ln.iter().enumerate() {
                    pre_act[hi] += w[j] * l;
                }
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
            let mut d_hidden = vec![0.0f32; h]; // Always h since it comes from activation

            // Compute logits and gradients (no weight tying for now)
            let mut logits = vec![0.0f32; v];
            for (vi, logit) in logits.iter_mut().enumerate() {
                let w = &self.lm_head[vi * h..(vi + 1) * h];
                for (hi, hn) in hidden.iter().enumerate() {
                    *logit += w[hi] * hn;
                }
            }
            softmax(&mut logits);

            for (vi, prob) in logits.iter().enumerate() {
                let grad = prob - if vi == target { 1.0 } else { 0.0 };
                for hi in 0..h {
                    g_head[vi * h + hi] += grad * hidden[hi];
                    d_hidden[hi] += grad * self.lm_head[vi * h + hi];
                }
            }

            let act_grads: Vec<f32> = all_pre_act[i]
                .iter()
                .map(|&pv| activate_grad(pv, &self.activation))
                .collect();

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
        for x in g_embed.iter_mut() {
            *x /= n;
        }
        for gc in g_ctx.iter_mut() {
            for x in gc.iter_mut() {
                *x /= n;
            }
        }
        for x in g_proj.iter_mut() {
            *x /= n;
        }
        for x in g_head.iter_mut() {
            *x /= n;
        }

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

/// Mean `(loss, bpb)` over evenly spaced chunks, or `None` when nothing could
/// be measured.
///
/// This used to return `f32::MAX` for both the loss and the BPB. That is a
/// finite value, so it
/// passed the caller's `is_finite()` guard, survived into `best_bpb`, and was
/// handed to `write_experience`, which appended
/// `bpb=340282346638528860000000000000000000000.0000` to a file agents read
/// back later as prior evidence. An absence is now an absence.
///
/// It then still dropped individual non-finite windows and published the mean
/// of the survivors, which is biased DOWNWARD: the windows a partial poison
/// kills are exactly the hard ones. `src/bin/trinity_pr1722.rs` takes the
/// correct line and this now matches it -- ONE unmeasurable window invalidates
/// the whole eval -- while the `dropped` counter keeps the skip from being
/// silent about HOW MUCH of the corpus failed.
fn evaluate(model: &NgramModel, tokens: &[usize], seq_len: usize) -> Option<(f32, f32)> {
    let mut total = 0.0f32;
    let mut n = 0usize;
    let mut dropped = 0usize;
    for c in (0..tokens.len()).step_by(seq_len + 1) {
        let end = (c + seq_len + 1).min(tokens.len());
        if end - c < model.ctx.len() + 3 {
            continue;
        }
        // An unmeasurable chunk is COUNTED, never averaged in as `0.0` and
        // never as the 23.03 nats a laundered NaN used to produce.
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
        return None;
    }
    let bpb = total / n as f32;
    if bpb.is_finite() {
        Some((bpb * LN_2, bpb))
    } else {
        None
    }
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

/// Render an optional BPB without inventing a number for an absent one.
fn fmt_bpb(bpb: Option<f32>) -> String {
    match bpb {
        Some(v) => format!("{v:.4}"),
        None => "unmeasured".to_string(),
    }
}

/// The experience row for one trial, or `None` when the trial measured no BPB.
///
/// `.trinity/experience/` is read back later as prior evidence, so a row is a
/// claim. `best_bpb` used to be an `f32` that could be the `f32::MAX` sentinel,
/// and the row it produced read
/// `bpb=340282346638528860000000000000000000000.0000`. Kept separate from the
/// write so the refusal can be tested without touching the filesystem.
fn experience_row(
    ts: &str,
    trial_name: &str,
    config: &TrialConfig,
    best_bpb: Option<f32>,
    steps: usize,
    duration_sec: f64,
    outcome: &str,
) -> Option<String> {
    let best_bpb = best_bpb?;
    let mut entry = format!(
        "[{}] ARCH-EXPLORER | trial={} | outcome={} | h={} | lr={:.6}",
        ts, trial_name, outcome, config.hidden, config.base_lr
    );

    if config.weight_tying {
        entry.push_str(" | weight_tying=true");
    }
    if config.cosine_lr {
        entry.push_str(" | cosine_lr=true");
    }
    if let Some(c) = config.gradient_clip {
        entry.push_str(&format!(" | grad_clip={:.2}", c));
    }
    if config.warmup_steps > 0 {
        entry.push_str(&format!(" | warmup={}", config.warmup_steps));
    }

    entry.push_str(&format!(
        " | bpb={:.4} | steps={} | {:.1}s\n",
        best_bpb, steps, duration_sec
    ));
    Some(entry)
}

/// Append one trial row to the experience log, or refuse.
///
/// An unmeasured trial writes nothing at all: the absence of a row is honest,
/// a fabricated number is not.
fn write_experience(
    trial_name: &str,
    config: &TrialConfig,
    best_bpb: Option<f32>,
    steps: usize,
    duration_sec: f64,
    outcome: &str,
) {
    let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let Some(entry) = experience_row(
        &ts,
        trial_name,
        config,
        best_bpb,
        steps,
        duration_sec,
        outcome,
    ) else {
        eprintln!(
            "[arch-explorer] trial {trial_name} measured no BPB ({outcome}); \
             writing no experience row rather than a sentinel"
        );
        return;
    };

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
        .write_all(entry.as_bytes());
}

fn run_trial(
    config: TrialConfig,
    seed: u64,
    max_steps: usize,
    prune_step: usize,
    prune_threshold: f32,
) -> (Option<f32>, usize, String, f64) {
    let ngram_order = 6; // Fixed at 6-gram for all trials
    let num_ctx = 4;

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║  ARCH-EXPLORER TRIAL: {:<45} ║", config.name);
    println!("╠════════════════════════════════════════════════════════════╣");
    println!(
        "║  {}-gram | h={} | lr={:.6} | seed={} {:<18}║",
        ngram_order,
        config.hidden,
        config.base_lr,
        seed,
        if config.weight_tying { "| tying" } else { "" }
    );
    println!(
        "║  cosine={} | clip={:<5} | warmup={:<4}",
        config.cosine_lr,
        config
            .gradient_clip
            .map(|c| format!("{:.2}", c))
            .unwrap_or_else(|| "none".to_string()),
        config.warmup_steps
    );
    println!("║                                                      ║");

    let (tokens, corpus) = load_or_refuse(&arg_path("--train-data", DEFAULT_TRAIN_PATH));
    println!("Corpus: {}", corpus.describe());
    let train_end = (tokens.len() as f64 * 0.9) as usize;
    let train = &tokens[..train_end];
    let val = &tokens[train_end..];

    let mut model = NgramModel::new(
        VOCAB,
        DIM,
        config.hidden,
        "relu".to_string(),
        seed,
        num_ctx,
        config.weight_tying,
    );
    let ps = VOCAB * DIM;
    let mut opt_embed = AdamW::new(ps, 0.01);
    let mut opt_ctx: Vec<AdamW> = (0..num_ctx).map(|_| AdamW::new(ps, 0.01)).collect();

    let proj_size = if config.weight_tying {
        config.hidden * DIM
    } else {
        DIM * config.hidden
    };
    let mut opt_proj = AdamW::new(proj_size, 0.01);

    let head_size = if config.weight_tying {
        VOCAB * DIM
    } else {
        VOCAB * config.hidden
    };
    let mut opt_head = AdamW::new(head_size, 0.01);

    // A baseline nobody measured is not a baseline; it used to bind
    // `f32::MAX` and seed `best_bpb` with it.
    let init = evaluate(&model, val, SEQ);
    match init {
        Some((init_loss, init_bpb)) => {
            println!("\nInitial val: loss={:.4} bpb={:.4}", init_loss, init_bpb)
        }
        None => println!(
            "\nInitial val: unmeasured ({} val tokens produced zero finite windows)",
            val.len()
        ),
    }
    let init_bpb = init.map(|(_, b)| b);
    println!(
        "\n{:>6} | {:>10} | {:>10} | {:>10} | {:>8}",
        "step", "val_loss", "val_bpb", "best_bpb", "lr"
    );
    println!("{}", "-".repeat(60));

    let t0 = Instant::now();
    let mut best_bpb: Option<f32> = init_bpb;
    let mut best_step = 0;
    let mut pruned = false;
    let mut results: Vec<(usize, f32, f32)> = Vec::new();
    let dl = train.len();

    for step in 1..=max_steps {
        let lr = get_lr(
            step,
            max_steps,
            config.base_lr,
            config.warmup_steps,
            config.cosine_lr,
        );
        let off = (step * 97 + seed as usize) % (dl.saturating_sub(SEQ + 1));
        model.train_step(
            &train[off..off + SEQ + 1],
            lr,
            &mut opt_embed,
            &mut opt_ctx,
            &mut opt_proj,
            &mut opt_head,
            config.gradient_clip,
        );

        if step % 500 == 0 || step == max_steps || step == prune_step {
            let _ms = t0.elapsed().as_millis();
            // An eval that measured nothing is reported as nothing: no row in
            // `results`, no candidate for `best_bpb`, no invented number.
            let Some((vl, vb)) = evaluate(&model, val, SEQ) else {
                println!(
                    "{:>6} | {:>10} | {:>10} | {:>10} | {:.6}",
                    step,
                    "unmeasured",
                    "unmeasured",
                    fmt_bpb(best_bpb),
                    lr
                );
                continue;
            };
            if vb.is_finite() && best_bpb.is_none_or(|best| vb < best) {
                best_bpb = Some(vb);
                best_step = step;
            }
            println!(
                "{:>6} | {:>10.4} | {:>10.4} | {:>10} | {:.6}",
                step,
                vl,
                vb,
                fmt_bpb(best_bpb),
                lr
            );
            results.push((step, vl, vb));

            // Pruning check
            if step == prune_step && vb > prune_threshold {
                println!(
                    "\n✂️  PRUNED at step {}: BPB={:.4} > {:.4}",
                    step, vb, prune_threshold
                );
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
    let delta = match (init_bpb, best_bpb) {
        (Some(i), Some(b)) => format!("{:.4}", b - i),
        _ => "unmeasured".to_string(),
    };
    println!(
        "BPB: {} -> {} | Delta: {}",
        fmt_bpb(init_bpb),
        fmt_bpb(best_bpb),
        delta
    );
    println!("Time: {:.1}s", total.as_secs_f64());

    write_experience(
        &config.name,
        &config,
        best_bpb,
        best_step,
        total.as_secs_f64(),
        &outcome,
    );

    (best_bpb, best_step, outcome, total.as_secs_f64())
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Trial configurations
    let trials = [
        TrialConfig {
            name: "X1".to_string(),
            hidden: 384,
            weight_tying: false, // DISABLED: needs h == dim for proper tying
            cosine_lr: false,
            gradient_clip: None,
            warmup_steps: 0,
            base_lr: 0.004,
        },
        TrialConfig {
            name: "X2".to_string(),
            hidden: 384,
            weight_tying: false,
            cosine_lr: true, // Cosine LR schedule
            gradient_clip: None,
            warmup_steps: 0,
            base_lr: 0.004,
        },
        TrialConfig {
            name: "X3".to_string(),
            hidden: 320, // Smaller hidden
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
            gradient_clip: Some(0.5), // Gradient clipping
            warmup_steps: 0,
            base_lr: 0.004,
        },
        TrialConfig {
            name: "X5".to_string(),
            hidden: 384,
            weight_tying: false,
            cosine_lr: false,
            gradient_clip: None,
            warmup_steps: 500, // Warmup 500 steps
            base_lr: 0.004,
        },
    ];

    let max_steps = 5000;
    let prune_step = 3000;
    let prune_threshold = 2.60;
    let target_bpb = 2.52;

    // Determine which trial(s) to run
    let trial_idx = if args.len() > 1 && args[1] == "--all" {
        None // Run all
    } else if let Some(idx) = args
        .iter()
        .find(|a| a.starts_with("--trial="))
        .map(|a| a[8..].parse::<usize>().unwrap_or(0))
    {
        if idx > 0 && idx <= trials.len() {
            Some(idx - 1)
        } else {
            println!("Invalid trial index. Running all trials.");
            None
        }
    } else {
        None // Default: run all
    };

    let trials_to_run: Vec<_> = if let Some(idx) = trial_idx {
        vec![&trials[idx]]
    } else {
        trials.iter().collect()
    };

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║  🏎️  ARCH-EXPLORER (АГЕНТ 3)                        ║");
    println!(
        "║  Machine: {}                                          ║",
        MACHINE_ID
    );
    println!(
        "║  Trials: {}                                           ║",
        trials_to_run.len()
    );
    println!(
        "║  Max steps: {} | Prune at: {} if BPB > {:.2}        ║",
        max_steps, prune_step, prune_threshold
    );
    println!(
        "║  Target: BPB < {:.2}                                    ║",
        target_bpb
    );
    println!("╚════════════════════════════════════════════════════════════╝");

    let mut all_results = vec![];

    for config in trials_to_run {
        // Canon #93 forbids seeds {42, 43, 44, 45} (see src/seed_canon.rs).
        let seed = 47;
        let (bpb, step, outcome, duration) =
            run_trial(config.clone(), seed, max_steps, prune_step, prune_threshold);
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
        println!(
            "{:5} | {:>9} | {:5} | {}",
            name,
            fmt_bpb(*bpb),
            step,
            outcome_trunc
        );
    }

    // Check for winner. Only a measured trial can win: `Option<f32>` orders
    // `None` below every `Some`, so a minimum over the raw options would crown
    // the trial that measured nothing.
    let best_result = all_results
        .iter()
        .filter_map(|(name, bpb, step, _, _)| bpb.map(|b| (name, b, step)))
        .min_by(|a, b| a.1.total_cmp(&b.1));
    match best_result {
        Some((name, bpb, step)) if bpb < target_bpb => {
            println!("\nWINNER FOUND");
            println!(
                "Trial {} achieved BPB={:.4} < {:.4} at step {}",
                name, bpb, target_bpb, step
            );
        }
        Some((name, bpb, _)) => {
            println!(
                "\nBest trial: {} with BPB={:.4} (target: <{:.4})",
                name, bpb, target_bpb
            );
            println!("Delta to target: +{:.4}", bpb - target_bpb);
        }
        None => println!("\nNo trial produced a BPB measurement."),
    }

    // Write summary to experience
    let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
    let ep = format!(
        ".trinity/experience/trios_{}.trinity",
        chrono::Utc::now().format("%Y%m%d")
    );
    let _ = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&ep)
        .unwrap()
        .write_all(
            format!(
            "[{}] ARCH-EXPLORER SUMMARY | machine={} | trials={} | best_bpb={} | target={:.4}\n",
            ts, MACHINE_ID, all_results.len(),
            // `999.9` was a stand-in for "no trial measured anything" that
            // reads back from the experience file as a BPB.
            fmt_bpb(best_result.map(|(_, b, _)| b)), target_bpb
        )
            .as_bytes(),
        );
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

    fn small_model() -> NgramModel {
        NgramModel::new(8, 4, 8, "relu".to_string(), 47, 1, false)
    }

    fn small_config() -> TrialConfig {
        TrialConfig {
            name: "T".to_string(),
            hidden: 8,
            weight_tying: false,
            cosine_lr: false,
            gradient_clip: None,
            warmup_steps: 0,
            base_lr: 0.004,
        }
    }

    /// The exact laundering this guard removes: `f32::max` returns the
    /// non-NaN operand, so `NaN.max(1e-10)` is `1e-10`, whose negative log is
    /// 23.026 nats and whose BPB is 33.2 - positive, finite, and small enough
    /// that nothing downstream rejected it.
    #[test]
    fn f32_max_launders_nan_into_a_publishable_bpb() {
        let laundered = f32::NAN.max(1e-10);
        assert_eq!(laundered, 1e-10, "f32::max ignores NaN");
        let bpb = -laundered.ln() / LN_2;
        assert!(
            (bpb - 33.2).abs() < 0.05,
            "the laundered reading is 33.2 bpb, got {bpb}"
        );
        assert!(bpb > 0.0 && bpb < 64.0, "and it passes every downstream guard");
    }

    /// A poisoned forward pass is an absence, not 33.2 bpb.
    #[test]
    fn nan_forward_pass_yields_no_measurement() {
        let tokens: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 0, 1, 2];

        let healthy = small_model();
        assert!(
            healthy.loss_on_seq(&tokens).is_some_and(f32::is_finite),
            "the fixture must be measurable before the NaN is introduced"
        );

        let mut model = small_model();
        model.lm_head[0] = f32::NAN;

        assert_eq!(model.loss_on_seq(&tokens), None, "NaN is not a loss");
        assert_eq!(
            evaluate(&model, &tokens, 8),
            None,
            "a poisoned model measures nothing"
        );
        if let Some((_, v)) = evaluate(&model, &tokens, 8) {
            assert!(
                (v - 33.2).abs() > 1.0,
                "33.2 bpb is the laundered NaN, not a measurement: {v}"
            );
        }
    }

    /// A sequence too short to hold a context/target pair is an absence, not
    /// the perfect `0.0` loss it used to report, and `evaluate` no longer
    /// answers `f32::MAX` - a finite value every downstream guard accepted.
    #[test]
    fn short_sequence_yields_no_measurement() {
        let model = small_model();
        assert_eq!(model.loss_on_seq(&[1, 2, 3]), None);
        assert_eq!(model.loss_on_seq(&[]), None);
        assert_eq!(evaluate(&model, &[1, 2, 3], 8), None);
        assert_eq!(evaluate(&model, &[], 8), None);
    }

    /// An unmeasured trial contributes no row to `.trinity/experience/`. The
    /// old code formatted `f32::MAX` into it as
    /// `bpb=340282346638528860000000000000000000000.0000`.
    #[test]
    fn experience_row_refuses_an_unmeasured_trial() {
        let cfg = small_config();
        assert_eq!(
            experience_row("TS", "T", &cfg, None, 500, 1.0, "COMPLETED 500 steps"),
            None,
            "no measurement means no row"
        );

        let row = experience_row("TS", "T", &cfg, Some(2.6141), 500, 1.0, "COMPLETED")
            .expect("a measured trial still writes its row");
        assert!(row.contains("bpb=2.6141"), "{row}");
        assert!(!row.contains("34028234"), "no sentinel digits: {row}");

        // And the sentinel is exactly what used to be formatted here.
        assert!(format!("{:.4}", f32::MAX).starts_with("34028234"));
        assert_eq!(fmt_bpb(None), "unmeasured");
    }
}
