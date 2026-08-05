#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::fs;
use std::time::Instant;

const VOCAB: usize = 128;
const DIM: usize = 64;
const SEQ: usize = 64;
const LN_2: f32 = std::f32::consts::LN_2;
const ACCUM: usize = 4;
const EVAL_INTERVAL: usize = 500;
const EVAL_SAMPLES: usize = 50;

fn layer_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = (var + eps).sqrt();
    x.iter().map(|v| (v - mean) / std).collect()
}

fn layer_norm_backward(x: &[f32], y: &[f32], dy: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = (var + eps).sqrt();
    let dy_mean = dy.iter().sum::<f32>() / n;
    let dyy_mean = dy.iter().zip(y.iter()).map(|(d, yi)| d * yi).sum::<f32>() / n;
    dy.iter()
        .zip(y.iter())
        .map(|(d, yi)| (d - dy_mean - yi * dyy_mean) / std)
        .collect()
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
        let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            step: 0,
            beta1: 1.0 / phi as f32,
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

struct ForwardState {
    combined: Vec<f32>,
    normed: Vec<f32>,
    hidden_raw: Vec<f32>,
    hidden: Vec<f32>,
    hidden_norm: Vec<f32>,
    projected: Vec<f32>,
    logits: Vec<f32>,
}

struct Model {
    embed: Vec<f32>,
    ctx: Vec<Vec<f32>>,
    ctx_weights: Vec<f32>,
    proj_up: Vec<f32>,
    proj_down: Vec<f32>,
    hidden_dim: usize,
    num_ctx: usize,
    ngram: usize,
}

impl Model {
    fn new(hidden_dim: usize, num_ctx: usize, seed: u64) -> Self {
        let ngram = num_ctx + 2;
        let mut s = seed;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let lim = (6.0f32 / (3 * DIM) as f32).sqrt();
        let lim_h = (6.0f32 / (DIM + hidden_dim) as f32).sqrt();
        let ctx_weights: Vec<f32> = (0..num_ctx)
            .map(|i| 0.7f32 * 0.45f32.powi(i as i32))
            .collect();
        Self {
            embed: (0..VOCAB * DIM).map(|_| rng() * lim).collect(),
            ctx: (0..num_ctx)
                .map(|_| (0..VOCAB * DIM).map(|_| rng() * lim).collect())
                .collect(),
            ctx_weights,
            proj_up: (0..hidden_dim * DIM).map(|_| rng() * lim_h).collect(),
            proj_down: (0..DIM * hidden_dim).map(|_| rng() * lim_h).collect(),
            hidden_dim,
            num_ctx,
            ngram,
        }
    }

    fn forward_one(&self, context: &[usize]) -> ForwardState {
        let h = self.hidden_dim;
        let ng = self.ngram;
        let t0 = context[ng - 1].min(VOCAB - 1);
        let mut combined = self.embed[t0 * DIM..(t0 + 1) * DIM].to_vec();
        for (ci, cw) in self.ctx_weights.iter().enumerate() {
            let idx = ng - 2 - ci;
            let t = context[idx].min(VOCAB - 1);
            let cv = &self.ctx[ci][t * DIM..(t + 1) * DIM];
            for j in 0..DIM {
                combined[j] += cv[j] * cw;
            }
        }
        let normed = layer_norm(&combined, 1e-5);
        let mut hidden_raw = vec![0.0f32; h];
        for hi in 0..h {
            for j in 0..DIM {
                hidden_raw[hi] += self.proj_up[hi * DIM + j] * normed[j];
            }
        }
        let hidden: Vec<f32> = hidden_raw.iter().map(|&v| v.max(0.0)).collect();
        let hidden_norm = layer_norm(&hidden, 1e-5);
        let mut projected = vec![0.0f32; DIM];
        for j in 0..DIM {
            for i in 0..h {
                projected[j] += self.proj_down[j * h + i] * hidden_norm[i];
            }
            projected[j] += normed[j];
        }
        let mut logits = vec![0.0f32; VOCAB];
        for v in 0..VOCAB {
            for j in 0..DIM {
                logits[v] += projected[j] * self.embed[v * DIM + j];
            }
        }
        ForwardState {
            combined,
            normed,
            hidden_raw,
            hidden,
            hidden_norm,
            projected,
            logits,
        }
    }
}

struct Grads {
    g_embed: Vec<f32>,
    g_ctx: Vec<Vec<f32>>,
    g_proj_up: Vec<f32>,
    g_proj_down: Vec<f32>,
}

fn compute_grads(model: &Model, tokens: &[usize]) -> (Grads, f32) {
    let h = model.hidden_dim;
    let ng = model.ngram;
    let nc = model.num_ctx;
    let count = tokens.len().saturating_sub(ng);
    if count == 0 {
        return (
            Grads {
                g_embed: vec![0.0; VOCAB * DIM],
                g_ctx: (0..nc).map(|_| vec![0.0; VOCAB * DIM]).collect(),
                g_proj_up: vec![0.0; h * DIM],
                g_proj_down: vec![0.0; DIM * h],
            },
            0.0,
        );
    }

    let mut g_embed = vec![0.0f32; VOCAB * DIM];
    let mut g_ctx: Vec<Vec<f32>> = (0..nc).map(|_| vec![0.0f32; VOCAB * DIM]).collect();
    let mut g_proj_up = vec![0.0f32; h * DIM];
    let mut g_proj_down = vec![0.0f32; DIM * h];
    let mut total_loss = 0.0f32;

    for i in 0..count {
        let context = &tokens[i..i + ng];
        let target = tokens[i + ng].min(VOCAB - 1);
        let st = model.forward_one(context);
        let mut probs = st.logits.clone();
        softmax(&mut probs);
        total_loss -= probs[target].max(1e-10).ln();

        let mut d_logits = probs;
        d_logits[target] -= 1.0;

        let mut d_proj = vec![0.0f32; DIM];
        for v in 0..VOCAB {
            let dl = d_logits[v];
            for j in 0..DIM {
                d_proj[j] += dl * model.embed[v * DIM + j];
                g_embed[v * DIM + j] += dl * st.projected[j];
            }
        }

        let mut d_hn = vec![0.0f32; h];
        let mut d_normed_res = vec![0.0f32; DIM];
        for j in 0..DIM {
            for i in 0..h {
                g_proj_down[j * h + i] += d_proj[j] * st.hidden_norm[i];
                d_hn[i] += d_proj[j] * model.proj_down[j * h + i];
            }
            d_normed_res[j] = d_proj[j];
        }

        let d_hid = layer_norm_backward(&st.hidden, &st.hidden_norm, &d_hn, 1e-5);

        let mut d_hr = vec![0.0f32; h];
        for i in 0..h {
            d_hr[i] = if st.hidden_raw[i] > 0.0 {
                d_hid[i]
            } else {
                0.0
            };
        }

        let mut d_normed_proj = vec![0.0f32; DIM];
        for hi in 0..h {
            if d_hr[hi] == 0.0 {
                continue;
            }
            for j in 0..DIM {
                g_proj_up[hi * DIM + j] += d_hr[hi] * st.normed[j];
                d_normed_proj[j] += d_hr[hi] * model.proj_up[hi * DIM + j];
            }
        }

        let mut d_normed = vec![0.0f32; DIM];
        for j in 0..DIM {
            d_normed[j] = d_normed_proj[j] + d_normed_res[j];
        }

        let d_comb = layer_norm_backward(&st.combined, &st.normed, &d_normed, 1e-5);

        let t0 = context[ng - 1].min(VOCAB - 1);
        for j in 0..DIM {
            g_embed[t0 * DIM + j] += d_comb[j];
        }
        for (ci, cw) in model.ctx_weights.iter().enumerate() {
            let idx = ng - 2 - ci;
            let t = context[idx].min(VOCAB - 1);
            for j in 0..DIM {
                g_ctx[ci][t * DIM + j] += cw * d_comb[j];
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
    for x in g_proj_up.iter_mut() {
        *x /= n;
    }
    for x in g_proj_down.iter_mut() {
        *x /= n;
    }

    (
        Grads {
            g_embed,
            g_ctx,
            g_proj_up,
            g_proj_down,
        },
        total_loss / n,
    )
}

/// One eval pass: the mean, and the sample the mean was actually taken over.
///
/// Without the counts, an eval that measured 3 of 50 windows and one that
/// measured all 50 print the same shape of number. They are not the same
/// measurement; see `require_complete_sample`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct EvalSample {
    mean: f32,
    /// Windows the eval loop set out to measure.
    planned: usize,
    /// Windows that produced a finite loss.
    realised: usize,
}

impl EvalSample {
    fn dropped(&self) -> usize {
        self.planned.saturating_sub(self.realised)
    }
}

/// Refuse to publish a mean taken over fewer windows than were planned.
///
/// The `n == 0` guard below only fires under TOTAL poisoning. A NaN confined
/// to one embedding row, or an overflow that only occurs on certain contexts,
/// drops exactly the affected windows and leaves a plausible mean over the
/// easy remainder - biased DOWNWARD, the direction that manufactures a
/// champion. `src/train_loop.rs` takes the strict line and aborts on the first
/// non-finite window; this binary keeps the per-window filter but will not
/// publish what it produced unless the operator asks in writing.
fn require_complete_sample(
    label: &str,
    sample: Option<EvalSample>,
    eval_source: &str,
    eval_tokens: usize,
    allow_dropped: bool,
) -> EvalSample {
    let s = match sample {
        Some(s) => s,
        None => {
            eprintln!(
                "NO MEASUREMENT ({label}): zero finite eval windows on \
                 {eval_source} ({eval_tokens} eval tokens). Refusing to print a \
                 BPB nobody measured."
            );
            println!("BPB=unmeasured");
            std::process::exit(EXIT_NO_MEASUREMENT);
        }
    };
    if s.dropped() > 0 && !allow_dropped {
        eprintln!(
            "EVAL SAMPLE SHRANK ({}): {} of {} planned windows on {} went \
             non-finite and were dropped. A mean over the survivors is biased \
             DOWNWARD and is not a held-out measurement. Set {}=1 to publish it \
             anyway.",
            label,
            s.dropped(),
            s.planned,
            eval_source,
            ALLOW_DROPPED_EVAL_WINDOWS_VAR
        );
        println!("BPB=unmeasured");
        std::process::exit(EXIT_NO_MEASUREMENT);
    }
    s
}

/// Mean BPB over the eval stream, or `None` when nothing could be measured.
///
/// This used to return `f32::MAX` on every barren path. `f32::MAX` is finite,
/// so it passed the caller's `is_finite()` guard, survived into `best_bpb` and
/// was printed on stdout as `BPB=340282346638528859811704183484516925440.0000`
/// beside `exit 0` - exactly the shape of number the out-of-repo stdout parser
/// turned into ledger rows that later had to be retracted. An absence is now an
/// absence, and the caller has to decide what to do with it.
fn evaluate(model: &Model, tokens: &[usize]) -> Option<EvalSample> {
    let ng = model.ngram;
    let dl = tokens.len();
    let num_possible = dl.saturating_sub(SEQ + 1);
    if num_possible == 0 {
        return None;
    }
    let num_chunks = EVAL_SAMPLES.min(num_possible);
    let stride = num_possible / num_chunks;
    if stride == 0 {
        return None;
    }
    let mut total = 0.0f32;
    // `planned` counts windows this loop intends to measure (the two checks
    // below are shape checks, not measurement failures); `n` counts the ones
    // that produced a finite loss.
    let mut planned = 0usize;
    let mut n = 0usize;
    for c in 0..num_chunks {
        let start = c * stride;
        let end = start + SEQ + 1;
        if end > dl {
            continue;
        }
        let chunk = &tokens[start..end];
        let cnt = chunk.len().saturating_sub(ng);
        if cnt == 0 {
            continue;
        }
        planned += 1;
        let mut loss = 0.0f32;
        let mut chunk_ok = true;
        for i in 0..cnt {
            let context = &chunk[i..i + ng];
            let target = chunk[i + ng].min(VOCAB - 1);
            let st = model.forward_one(context);
            let mut probs = st.logits.clone();
            softmax(&mut probs);
            // `probs[target].max(1e-10)` used to launder a poisoned forward
            // pass into a finite reading: `f32::max` returns the OTHER operand
            // when one side is NaN, so a NaN probability silently became 1e-10
            // and contributed a plausible ~23 nats. A probability that is not
            // finite and positive is not a measurement, so the whole chunk is
            // dropped instead.
            let p = probs[target];
            if !p.is_finite() || p <= 0.0 {
                chunk_ok = false;
                break;
            }
            loss -= p.ln();
        }
        if !chunk_ok || !loss.is_finite() {
            continue;
        }
        total += loss / cnt as f32 / LN_2;
        n += 1;
    }
    if n == 0 || !total.is_finite() {
        None
    } else {
        Some(EvalSample {
            mean: total / n as f32,
            planned,
            realised: n,
        })
    }
}

/// Exit code for a corpus that could not be honestly loaded. Same value
/// `cpu_train` uses, so a sweep can tell a refused corpus from a crash.
const EXIT_BAD_CORPUS: i32 = 6;

/// Exit code for a run that measured nothing. Same value `cpu_train`,
/// `trinity_pr1722` and `igla_trigram` use, so a sweep can tell an unmeasured
/// run from a crash and from a refused corpus.
const EXIT_NO_MEASUREMENT: i32 = 7;

/// Exit code for an argument this binary does not understand. Same value
/// `cpu_train` uses.
const EXIT_BAD_ARGS: i32 = 4;

/// Opt-in that permits publishing a mean taken over fewer windows than were
/// planned. Off by default; see `require_complete_sample`.
const ALLOW_DROPPED_EVAL_WINDOWS_VAR: &str = "TRIOS_ALLOW_DROPPED_EVAL_WINDOWS";

/// Usage text. Without it `--help` starts a full 27000-step training run.
const USAGE: &str = "\
train_v2 - n-gram MLP char model, BPB measured on a held-out corpus.

Usage: train_v2 [--seed=N] [--steps=N] [--lr=F] [--hidden=N] [--ctx=N]
                [--train-data=PATH] [--val-data=PATH]

Prints `BPB=<value>` on stdout and exits 0 when a BPB was measured over the
FULL planned eval sample; prints `BPB=unmeasured` and exits 7 when no eval
window was finite or when the sample shrank. The realised window count is
printed beside the BPB as EVAL_WINDOWS_PLANNED / _REALISED / _DROPPED.
Env: TRIOS_ALLOW_DROPPED_EVAL_WINDOWS=1 (publish a reduced eval sample).
Exit codes: 4 = bad argument, 6 = corpus refused, 7 = nothing measured.";

/// Every argument this binary reads, as `--name=value` prefixes. `--train-data`
/// and `--val-data` also accept a space-separated value (see `arg_path`).
const KNOWN_VALUE_ARGS: [&str; 7] = [
    "seed",
    "steps",
    "lr",
    "hidden",
    "ctx",
    "train-data",
    "val-data",
];
/// The two flags `arg_path` will consume a following argv entry for.
const SPACE_VALUE_ARGS: [&str; 2] = ["--train-data", "--val-data"];

/// Name the first argument that means nothing here, or `None` if all are known.
///
/// Silently ignoring an unrecognised argument lets a caller believe it asked
/// for one run while the binary performs another.
fn first_unknown_arg(args: &[String]) -> Option<String> {
    let mut i = 1usize;
    while i < args.len() {
        let a = &args[i];
        if a == "--help" || a == "-h" {
            i += 1;
            continue;
        }
        if SPACE_VALUE_ARGS.contains(&a.as_str()) {
            // `arg_path` reads the next entry as this flag's value.
            i += 2;
            continue;
        }
        if KNOWN_VALUE_ARGS
            .iter()
            .any(|name| a.starts_with(&format!("--{name}=")))
        {
            i += 1;
            continue;
        }
        return Some(a.clone());
    }
    None
}

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

fn find_arg<T: std::str::FromStr>(args: &[String], prefix: &str, default: T) -> T {
    args.iter()
        .find(|a| a.starts_with(prefix))
        .and_then(|a| a[prefix.len()..].parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", USAGE);
        return Ok(());
    }
    if let Some(bad) = first_unknown_arg(&args) {
        eprintln!("UNKNOWN ARGUMENT: {bad}");
        eprintln!();
        eprintln!("{}", USAGE);
        std::process::exit(EXIT_BAD_ARGS);
    }
    // Whether a reduced eval sample may be published, decided once so every
    // eval in the run is graded the same way.
    let allow_dropped = std::env::var(ALLOW_DROPPED_EVAL_WINDOWS_VAR)
        .map(|v| v == "1")
        .unwrap_or(false);
    if allow_dropped {
        eprintln!(
            "{}=1: a reduced eval sample will be published. The mean is taken \
             over the windows that survived and is biased DOWNWARD.",
            ALLOW_DROPPED_EVAL_WINDOWS_VAR
        );
    }
    // Canon #93 forbids seeds {42, 43, 44, 45} (see src/seed_canon.rs); the
    // default is a permitted one so an argument-free run is not born invalid.
    let seed: u64 = find_arg(&args, "--seed=", 47);
    let steps: usize = find_arg(&args, "--steps=", 27000);
    let lr: f32 = find_arg(&args, "--lr=", 0.003);
    let hidden_dim: usize = find_arg(&args, "--hidden=", 512);
    let num_ctx: usize = find_arg(&args, "--ctx=", 8);
    let ngram = num_ctx + 2;

    let (train_data, train_corpus) = load_or_refuse(&arg_path("--train-data", DEFAULT_TRAIN_PATH));
    let (val_data, val_corpus) = load_or_refuse(&arg_path("--val-data", DEFAULT_VAL_PATH));
    println!("Corpus train: {}", train_corpus.describe());
    println!("Corpus val:   {}", val_corpus.describe());
    let train_end = (train_data.len() as f64 * 0.9) as usize;
    let train = &train_data[..train_end];
    let val = if val_data.len() > 100 {
        &val_data
    } else {
        &train_data[train_end..]
    };
    // Named here, where the choice is made, so a refusal can say which bytes
    // it failed to measure.
    let eval_source = if val_data.len() > 100 {
        val_corpus.path.clone()
    } else {
        format!("the held-out tail of {}", train_corpus.path)
    };

    let mut model = Model::new(hidden_dim, num_ctx, seed);
    let h = hidden_dim;
    let wd = 0.04f32;
    let mut opt_embed = AdamW::new(VOCAB * DIM, wd);
    let mut opt_ctx: Vec<AdamW> = (0..num_ctx).map(|_| AdamW::new(VOCAB * DIM, wd)).collect();
    let mut opt_proj_up = AdamW::new(h * DIM, wd);
    let mut opt_proj_down = AdamW::new(DIM * h, wd);

    let mut acc_embed = vec![0.0f32; VOCAB * DIM];
    let mut acc_ctx: Vec<Vec<f32>> = (0..num_ctx).map(|_| vec![0.0f32; VOCAB * DIM]).collect();
    let mut acc_proj_up = vec![0.0f32; h * DIM];
    let mut acc_proj_down = vec![0.0f32; DIM * h];

    // `None` until an eval actually measures something. The old `f32::MAX`
    // seed was finite, so `val_bpb < best_bpb && val_bpb.is_finite()` never
    // rejected it and never replaced it either. The whole sample is kept, not
    // just its mean, so the BPB that is finally printed can name the window
    // count it was averaged over.
    let mut best: Option<EvalSample> = None;
    let mut dropped_total = 0usize;
    let warmup = steps / 10;
    let start = Instant::now();
    let dl = train.len();

    eprintln!(
        "train_v2: dim={} hidden={} ctx={} ngram={} lr={} seed={} steps={}",
        DIM, h, num_ctx, ngram, lr, seed, steps
    );

    for step in 1..=steps {
        let off = (step * 97 + seed as usize) % dl.saturating_sub(SEQ + 1);
        let seq = &train[off..off + SEQ + 1];

        let (grads, _loss) = compute_grads(&model, seq);

        for i in 0..VOCAB * DIM {
            acc_embed[i] += grads.g_embed[i];
        }
        for (ci, ac) in acc_ctx.iter_mut().enumerate() {
            for i in 0..VOCAB * DIM {
                ac[i] += grads.g_ctx[ci][i];
            }
        }
        for i in 0..h * DIM {
            acc_proj_up[i] += grads.g_proj_up[i];
        }
        for i in 0..DIM * h {
            acc_proj_down[i] += grads.g_proj_down[i];
        }

        if step % ACCUM == 0 || step == steps {
            let num_acc = if step % ACCUM == 0 {
                ACCUM
            } else {
                step % ACCUM
            };
            let inv = 1.0 / num_acc as f32;
            for x in acc_embed.iter_mut() {
                *x *= inv;
            }
            for ac in acc_ctx.iter_mut() {
                for x in ac.iter_mut() {
                    *x *= inv;
                }
            }
            for x in acc_proj_up.iter_mut() {
                *x *= inv;
            }
            for x in acc_proj_down.iter_mut() {
                *x *= inv;
            }

            let cur_lr = cosine_lr(step, steps, lr, warmup);
            opt_embed.update(&mut model.embed, &acc_embed, cur_lr);
            for (ci, oc) in opt_ctx.iter_mut().enumerate() {
                oc.update(&mut model.ctx[ci], &acc_ctx[ci], cur_lr);
            }
            opt_proj_up.update(&mut model.proj_up, &acc_proj_up, cur_lr);
            opt_proj_down.update(&mut model.proj_down, &acc_proj_down, cur_lr);

            for x in acc_embed.iter_mut() {
                *x = 0.0;
            }
            for ac in acc_ctx.iter_mut() {
                for x in ac.iter_mut() {
                    *x = 0.0;
                }
            }
            for x in acc_proj_up.iter_mut() {
                *x = 0.0;
            }
            for x in acc_proj_down.iter_mut() {
                *x = 0.0;
            }
        }

        if step % EVAL_INTERVAL == 0 || step == steps {
            let elapsed = start.elapsed().as_secs_f64();
            // An intermediate eval is not a lesser measurement: it feeds the
            // BPB that is finally printed, so it is graded like any other.
            let sample = require_complete_sample(
                &format!("eval at step {step}"),
                evaluate(&model, val),
                &eval_source,
                val.len(),
                allow_dropped,
            );
            dropped_total += sample.dropped();
            if best.is_none_or(|b| sample.mean < b.mean) {
                best = Some(sample);
            }
            let best_text = match best {
                Some(b) => format!("{:.4}", b.mean),
                None => "unmeasured".to_string(),
            };
            eprintln!(
                "step={:5} val_bpb={:.4} best={} windows={}/{} t={}s",
                step, sample.mean, best_text, sample.realised, sample.planned, elapsed as u64
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    match best {
        Some(b) => {
            eprintln!("done: best_bpb={:.4} time={:.1}s", b.mean, elapsed);
            // This binary has no results JSON; stdout IS its results record,
            // so the realised window count is printed beside the BPB rather
            // than written to a file. A reader that cannot see the sample
            // cannot tell a full eval from a shrunken one.
            println!("EVAL_WINDOWS_PLANNED={}", b.planned);
            println!("EVAL_WINDOWS_REALISED={}", b.realised);
            println!("EVAL_WINDOWS_DROPPED={}", b.dropped());
            println!("EVAL_WINDOWS_DROPPED_TOTAL={}", dropped_total);
            println!("BPB={:.4}", b.mean);
            Ok(())
        }
        // A run that measured nothing has no BPB. Printing a sentinel under
        // the `BPB=` key beside `exit 0` is what put unmeasured numbers into
        // the ledger; the run now says so and fails.
        None => {
            eprintln!(
                "NO MEASUREMENT: every eval over {} steps produced zero finite \
                 windows on {} ({} eval tokens). Refusing to print a BPB nobody \
                 measured.",
                steps,
                eval_source,
                val.len()
            );
            eprintln!("done: best_bpb=unmeasured time={:.1}s", elapsed);
            println!("BPB=unmeasured");
            std::process::exit(EXIT_NO_MEASUREMENT);
        }
    }
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
