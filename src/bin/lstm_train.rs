#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::fs;
use std::time::Instant;

const VOCAB: usize = 128;
const DIM: usize = 64;
const SEQ: usize = 32;
const LN_2: f32 = std::f32::consts::LN_2;
const EVAL_INTERVAL: usize = 500;
const EVAL_SEQS: usize = 20;

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
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

struct LstmTrain {
    h: usize,
    d: usize,
    g4: usize,
    embed: Vec<f32>,
    w: Vec<f32>,
    b: Vec<f32>,
    head: Vec<f32>,
    h_state: Vec<f32>,
    c_state: Vec<f32>,
    s_h: Vec<Vec<f32>>,
    s_c_prev: Vec<Vec<f32>>,
    s_gates: Vec<Vec<f32>>,
    s_tc: Vec<Vec<f32>>,
    s_xh: Vec<Vec<f32>>,
    s_logits: Vec<Vec<f32>>,
    s_tok: Vec<usize>,
}

struct Grads {
    g_embed: Vec<f32>,
    g_w: Vec<f32>,
    g_b: Vec<f32>,
    g_head: Vec<f32>,
}

struct AdamW {
    m: Vec<f32>,
    v: Vec<f32>,
    step: usize,
}

impl AdamW {
    fn new(size: usize) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            step: 0,
        }
    }
    fn update(&mut self, p: &mut [f32], g: &[f32], lr: f32, wd: f32) {
        self.step += 1;
        let bc1 = 1.0 - 0.9f32.powi(self.step as i32);
        let bc2 = 1.0 - 0.999f32.powi(self.step as i32);
        for i in 0..p.len() {
            p[i] -= wd * lr * p[i];
            self.m[i] = 0.9 * self.m[i] + 0.1 * g[i];
            self.v[i] = 0.999 * self.v[i] + 0.001 * g[i] * g[i];
            p[i] -= lr * (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + 1e-8);
        }
    }
}

impl LstmTrain {
    fn new(hidden: usize, seed: u64) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let d = DIM + hidden;
        let g4 = 4 * hidden;
        let lim_e = (6.0 / (VOCAB + DIM) as f32).sqrt();
        let lim_w = (6.0 / (d + g4) as f32).sqrt();
        let lim_o = (6.0 / (hidden + VOCAB) as f32).sqrt();
        let mut b = vec![0.0f32; g4];
        for i in 0..hidden {
            b[hidden + i] = 2.0;
        }
        let cap = SEQ + 1;
        Self {
            h: hidden,
            d,
            g4,
            embed: (0..VOCAB * DIM).map(|_| rng() * lim_e).collect(),
            w: (0..g4 * d).map(|_| rng() * lim_w).collect(),
            b,
            head: (0..VOCAB * hidden).map(|_| rng() * lim_o).collect(),
            h_state: vec![0.0; hidden],
            c_state: vec![0.0; hidden],
            s_h: (0..cap).map(|_| vec![0.0; hidden]).collect(),
            s_c_prev: (0..cap).map(|_| vec![0.0; hidden]).collect(),
            s_gates: (0..cap).map(|_| vec![0.0; g4]).collect(),
            s_tc: (0..cap).map(|_| vec![0.0; hidden]).collect(),
            s_xh: (0..cap).map(|_| vec![0.0; d]).collect(),
            s_logits: (0..cap).map(|_| vec![0.0; VOCAB]).collect(),
            s_tok: vec![0; cap],
        }
    }

    fn forward_seq(&mut self, tokens: &[usize]) -> f32 {
        let h = self.h;
        let d = self.d;
        let g4 = self.g4;
        for v in 0..h {
            self.h_state[v] = 0.0;
            self.c_state[v] = 0.0;
        }

        let mut total_loss = 0.0f32;
        for t in 0..tokens.len() {
            let tok = tokens[t].min(VOCAB - 1);
            let ex = &self.embed[tok * DIM..(tok + 1) * DIM];
            let mut xh = vec![0.0f32; d];
            for j in 0..DIM {
                xh[j] = ex[j];
            }
            for j in 0..h {
                xh[DIM + j] = self.h_state[j];
            }

            let mut gates = vec![0.0f32; g4];
            for i in 0..g4 {
                let mut val = self.b[i];
                for j in 0..d {
                    val += self.w[i * d + j] * xh[j];
                }
                let pre = val;
                if i < h {
                    gates[i] = sigmoid(pre);
                } else if i < 2 * h {
                    gates[i] = sigmoid(pre);
                } else if i < 3 * h {
                    gates[i] = sigmoid(pre);
                } else {
                    gates[i] = pre.tanh();
                }
            }

            let c_prev = self.c_state.clone();
            let mut tc = vec![0.0f32; h];
            for i in 0..h {
                self.c_state[i] = gates[h + i] * c_prev[i] + gates[i] * gates[3 * h + i];
                tc[i] = self.c_state[i].tanh();
                self.h_state[i] = gates[2 * h + i] * tc[i];
            }

            let mut logits = vec![0.0f32; VOCAB];
            for v in 0..VOCAB {
                for j in 0..h {
                    logits[v] += self.head[v * h + j] * self.h_state[j];
                }
            }

            self.s_h[t] = self.h_state.clone();
            self.s_c_prev[t] = c_prev;
            self.s_gates[t] = gates;
            self.s_tc[t] = tc;
            self.s_xh[t] = xh;
            self.s_logits[t] = logits.clone();
            self.s_tok[t] = tok;

            if t + 1 < tokens.len() {
                let target = tokens[t + 1].min(VOCAB - 1);
                let mut probs = logits;
                softmax(&mut probs);
                total_loss -= probs[target].max(1e-10).ln();
            }
        }
        total_loss
    }

    fn backward_seq(&self, tokens: &[usize]) -> Grads {
        let h = self.h;
        let d = self.d;
        let g4 = self.g4;
        let seq_len = tokens.len();
        let n = (seq_len - 1).max(1) as f32;

        let mut g_embed = vec![0.0f32; VOCAB * DIM];
        let mut g_w = vec![0.0f32; g4 * d];
        let mut g_b = vec![0.0f32; g4];
        let mut g_head = vec![0.0f32; VOCAB * h];

        let mut dh = vec![0.0f32; h];
        let mut dc = vec![0.0f32; h];

        for t in (0..seq_len - 1).rev() {
            let target = tokens[t + 1].min(VOCAB - 1);
            let mut probs = self.s_logits[t].clone();
            softmax(&mut probs);
            let mut d_logits = probs;
            d_logits[target] -= 1.0;

            for v in 0..VOCAB {
                let dl = d_logits[v];
                for j in 0..h {
                    g_head[v * h + j] += dl * self.s_h[t][j];
                    dh[j] += dl * self.head[v * h + j];
                }
            }

            for i in 0..h {
                let o = self.s_gates[t][2 * h + i];
                let tc = self.s_tc[t][i];
                dc[i] += dh[i] * o * (1.0 - tc * tc);
            }

            let mut dg = vec![0.0f32; g4];
            for i in 0..h {
                let ig = self.s_gates[t][i];
                let fg = self.s_gates[t][h + i];
                let og = self.s_gates[t][2 * h + i];
                let gg = self.s_gates[t][3 * h + i];
                dg[i] = dc[i] * gg * ig * (1.0 - ig);
                dg[h + i] = dc[i] * self.s_c_prev[t][i] * fg * (1.0 - fg);
                dg[2 * h + i] = dh[i] * self.s_tc[t][i] * og * (1.0 - og);
                dg[3 * h + i] = dc[i] * ig * (1.0 - gg * gg);
                dc[i] *= fg;
            }

            let mut dxh = vec![0.0f32; d];
            for i in 0..g4 {
                let dgi = dg[i];
                if dgi.abs() < 1e-12 {
                    continue;
                }
                for j in 0..d {
                    g_w[i * d + j] += dgi * self.s_xh[t][j];
                    dxh[j] += dgi * self.w[i * d + j];
                }
                g_b[i] += dgi;
            }

            for j in 0..h {
                dh[j] = dxh[DIM + j];
            }
            let tok = self.s_tok[t];
            for j in 0..DIM {
                g_embed[tok * DIM + j] += dxh[j];
            }
        }

        let inv = 1.0 / n;
        for x in g_embed.iter_mut() {
            *x *= inv;
        }
        for x in g_w.iter_mut() {
            *x *= inv;
        }
        for x in g_b.iter_mut() {
            *x *= inv;
        }
        for x in g_head.iter_mut() {
            *x *= inv;
        }

        let max_norm = 1.0f32;
        let mut gn2 = 0.0f32;
        for x in g_w.iter() {
            gn2 += x * x;
        }
        for x in g_embed.iter() {
            gn2 += x * x;
        }
        let gn = gn2.sqrt();
        if gn > max_norm {
            let scale = max_norm / gn;
            for x in g_w.iter_mut() {
                *x *= scale;
            }
            for x in g_embed.iter_mut() {
                *x *= scale;
            }
            for x in g_b.iter_mut() {
                *x *= scale;
            }
            for x in g_head.iter_mut() {
                *x *= scale;
            }
        }

        Grads {
            g_embed,
            g_w,
            g_b,
            g_head,
        }
    }

    /// Mean BPB over the eval stream, or `None` when nothing could be measured.
    ///
    /// This used to return `f32::MAX` on every barren path. `f32::MAX` is
    /// finite, so it passed the caller's `is_finite()` guard, survived into
    /// `best_bpb` and was printed on stdout as
    /// `BPB=340282346638528859811704183484516925440.0000` beside `exit 0` -
    /// exactly the shape of number the out-of-repo stdout parser turned into
    /// ledger rows that later had to be retracted. An absence is now an
    /// absence, and the caller has to decide what to do with it.
    fn eval_bpb(&mut self, tokens: &[usize]) -> Option<EvalSample> {
        let dl = tokens.len();
        if dl < 2 {
            return None;
        }
        let num_possible = dl.saturating_sub(SEQ + 1);
        if num_possible == 0 {
            return None;
        }
        let nc = EVAL_SEQS.min(num_possible);
        let stride = num_possible / nc;
        if stride == 0 {
            return None;
        }
        let h = self.h;
        let d = self.d;
        let g4 = self.g4;

        let mut total = 0.0f32;
        // `planned` counts windows this loop intends to measure (the check
        // below is a shape check, not a measurement failure); `n` counts the
        // ones that produced a finite loss.
        let mut planned = 0usize;
        let mut n = 0usize;
        for c in 0..nc {
            let start = c * stride;
            let end = (start + SEQ + 1).min(dl);
            if end <= start + 2 {
                continue;
            }
            planned += 1;
            let chunk = &tokens[start..end];
            let mut h_st = vec![0.0f32; h];
            let mut c_st = vec![0.0f32; h];
            let mut loss = 0.0f32;
            let mut chunk_ok = true;
            for t in 0..chunk.len() - 1 {
                let tok = chunk[t].min(VOCAB - 1);
                let ex = &self.embed[tok * DIM..(tok + 1) * DIM];
                let mut xh = vec![0.0f32; d];
                for j in 0..DIM {
                    xh[j] = ex[j];
                }
                for j in 0..h {
                    xh[DIM + j] = h_st[j];
                }
                let mut gates = vec![0.0f32; g4];
                for i in 0..g4 {
                    let mut val = self.b[i];
                    for j in 0..d {
                        val += self.w[i * d + j] * xh[j];
                    }
                    if i < 3 * h {
                        gates[i] = sigmoid(val);
                    } else {
                        gates[i] = val.tanh();
                    }
                }
                let c_old = c_st.clone();
                for i in 0..h {
                    c_st[i] = gates[h + i] * c_old[i] + gates[i] * gates[3 * h + i];
                    h_st[i] = gates[2 * h + i] * c_st[i].tanh();
                }
                let mut logits = vec![0.0f32; VOCAB];
                for v in 0..VOCAB {
                    for j in 0..h {
                        logits[v] += self.head[v * h + j] * h_st[j];
                    }
                }
                softmax(&mut logits);
                let target = chunk[t + 1].min(VOCAB - 1);
                // `logits[target].max(1e-10)` used to launder a poisoned
                // forward pass into a finite reading: `f32::max` returns the
                // OTHER operand when one side is NaN, so a NaN probability
                // silently became 1e-10 and contributed a plausible ~23 nats.
                // A probability that is not finite and positive is not a
                // measurement, so the whole chunk is dropped instead.
                let p = logits[target];
                if !p.is_finite() || p <= 0.0 {
                    chunk_ok = false;
                    break;
                }
                loss -= p.ln();
            }
            if !chunk_ok || !loss.is_finite() {
                continue;
            }
            total += loss / (chunk.len() - 1) as f32 / LN_2;
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
}

/// One eval pass: the mean, and the sample the mean was actually taken over.
///
/// Without the counts, an eval that measured 3 of 20 windows and one that
/// measured all 20 print the same shape of number. They are not the same
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
/// The `n == 0` guard in `eval_bpb` only fires under TOTAL poisoning. A NaN
/// confined to one embedding row, or an overflow that only occurs on certain
/// contexts, drops exactly the affected windows and leaves a plausible mean
/// over the easy remainder - biased DOWNWARD, the direction that manufactures
/// a champion. `src/train_loop.rs` takes the strict line and aborts on the
/// first non-finite window; this binary keeps the per-window filter but will
/// not publish what it produced unless the operator asks in writing.
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

fn cosine_lr(step: usize, max_steps: usize, base_lr: f32, warmup: usize) -> f32 {
    if step < warmup {
        return base_lr * step as f32 / warmup.max(1) as f32;
    }
    let p = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
    1e-5 + (base_lr - 1e-5) * 0.5 * (1.0 + (std::f32::consts::PI * p).cos())
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

/// Usage text. Without it `--help` starts a full 10000-step training run.
const USAGE: &str = "\
lstm_train - single-layer LSTM char model, BPB measured on a held-out corpus.

Usage: lstm_train [--seed=N] [--steps=N] [--lr=F] [--hidden=N]
                  [--train-data=PATH] [--val-data=PATH]

Prints `BPB=<value>` on stdout and exits 0 when a BPB was measured over the
FULL planned eval sample; prints `BPB=unmeasured` and exits 7 when no eval
window was finite or when the sample shrank. The realised window count is
printed beside the BPB as EVAL_WINDOWS_PLANNED / _REALISED / _DROPPED.
Env: TRIOS_ALLOW_DROPPED_EVAL_WINDOWS=1 (publish a reduced eval sample).
Exit codes: 4 = bad argument, 6 = corpus refused, 7 = nothing measured.";

/// Every argument this binary reads, as `--name=value` prefixes. `--train-data`
/// and `--val-data` also accept a space-separated value (see `arg_path`).
const KNOWN_VALUE_ARGS: [&str; 6] = ["seed", "steps", "lr", "hidden", "train-data", "val-data"];
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
    let seed: u64 = args
        .iter()
        .find(|a| a.starts_with("--seed="))
        .and_then(|a| a[7..].parse().ok())
        .unwrap_or(47);
    let steps: usize = args
        .iter()
        .find(|a| a.starts_with("--steps="))
        .and_then(|a| a[8..].parse().ok())
        .unwrap_or(10000);
    let lr: f32 = args
        .iter()
        .find(|a| a.starts_with("--lr="))
        .and_then(|a| a[5..].parse().ok())
        .unwrap_or(0.003);
    let hidden: usize = args
        .iter()
        .find(|a| a.starts_with("--hidden="))
        .and_then(|a| a[9..].parse().ok())
        .unwrap_or(256);

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

    let mut m = LstmTrain::new(hidden, seed);
    let d = DIM + hidden;
    let g4 = 4 * hidden;
    let wd = 0.0f32;

    let mut opt_e = AdamW::new(VOCAB * DIM);
    let mut opt_w = AdamW::new(g4 * d);
    let mut opt_b = AdamW::new(g4);
    let mut opt_h = AdamW::new(VOCAB * hidden);

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
    let seq = SEQ;

    eprintln!(
        "lstm_v2: hidden={} seq={} lr={} seed={} steps={} params≈{}",
        hidden,
        seq,
        lr,
        seed,
        steps,
        VOCAB * DIM + g4 * d + g4 + VOCAB * hidden
    );

    for step in 1..=steps {
        let off = (step * 97 + seed as usize) % dl.saturating_sub(seq + 1);
        let chunk = &train[off..off + seq + 1];
        let _loss = m.forward_seq(chunk);
        let grads = m.backward_seq(chunk);
        let cur_lr = cosine_lr(step, steps, lr, warmup);
        opt_e.update(&mut m.embed, &grads.g_embed, cur_lr, wd);
        opt_w.update(&mut m.w, &grads.g_w, cur_lr, wd);
        opt_b.update(&mut m.b, &grads.g_b, cur_lr, 0.0);
        opt_h.update(&mut m.head, &grads.g_head, cur_lr, wd);

        if step % EVAL_INTERVAL == 0 || step == steps {
            let elapsed = start.elapsed().as_secs_f64();
            // An intermediate eval is not a lesser measurement: it feeds the
            // BPB that is finally printed, so it is graded like any other.
            let sample = require_complete_sample(
                &format!("eval at step {step}"),
                m.eval_bpb(val),
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
