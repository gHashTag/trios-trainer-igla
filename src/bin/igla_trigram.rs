use std::fs;
use std::io::Write;
use std::time::Instant;

const VOCAB: usize = 128;
const DIM: usize = 64;
const SEQ: usize = 48;
const LN_2: f32 = std::f32::consts::LN_2;

/// Exit code for a corpus that could not be honestly loaded. Same value
/// `cpu_train` uses, so a sweep can tell a refused corpus from a crash.
const EXIT_BAD_CORPUS: i32 = 6;

/// Exit code for a run that reached the end without a measurable eval window.
/// Same value `cpu_train` uses.
const EXIT_NO_MEASUREMENT: i32 = 7;

/// Comparison window, in tokens, for the train/val overlap guard. Mirrors
/// `train_loop::OVERLAP_WINDOW`.
const OVERLAP_WINDOW: usize = 256;

/// Fail the run above this fraction of val windows found verbatim in train.
/// Mirrors `train_loop::MAX_VAL_OVERLAP_FRACTION`.
const MAX_VAL_OVERLAP_FRACTION: f64 = 0.01;

/// A val stream shorter than this cannot support a BPB anyone should quote.
/// Mirrors `train_loop::MIN_VAL_TOKENS`.
const MIN_VAL_TOKENS: usize = 8192;

/// `evaluate` must average over at least this many chunks for the mean to mean
/// anything. Mirrors `train_loop::MIN_EVAL_CHUNKS`.
const MIN_EVAL_CHUNKS: usize = 8;

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
/// A results row that cannot name the bytes it measured is indistinguishable
/// from a fabricated one, so path, size, digest and the synthetic flag travel
/// with every number this binary reports.
#[derive(Debug, Clone)]
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
/// The old body substituted a 52-byte pangram whenever the read failed, which
/// is how this binary printed `Dataset: 52 tokens` and `Final BPB: 0.2876` on
/// a clean checkout with no arguments. A missing corpus is now a hard, named
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
/// result file is created, so a refused run leaves no artifact behind.
fn load_or_refuse(path: &str) -> (Vec<usize>, CorpusInfo) {
    match load_data(path) {
        Ok(loaded) => loaded,
        Err(e) => {
            eprintln!("CORPUS REFUSED: {e}");
            std::process::exit(EXIT_BAD_CORPUS);
        }
    }
}

/// Number of windows `evaluate` will actually average over, for `len` tokens
/// at `seq_len`. Must track the loop in `evaluate` exactly: a precondition
/// computed from a different chunking than the one that runs is not a
/// precondition.
fn eval_chunk_count(len: usize, seq_len: usize) -> usize {
    (0..len)
        .step_by(seq_len + 1)
        .filter(|&c| len.min(c + seq_len + 1) - c >= 4)
        .count()
}

/// Local mirror of `trios_trainer::train_loop::assert_train_val_disjoint`.
///
/// The library function is `pub(crate)` and `src/bin/*.rs` compile as separate
/// crates, so it cannot be linked from here; `src/bin/cpu_train.rs` mirrors the
/// same checks inline for the same reason. Same constants, same thresholds,
/// same full-coverage comparison - it returns the refusal instead of panicking
/// so the caller can exit before any result file exists.
///
/// Three preconditions, each already violated in a live run:
/// 1. Size: a val too short, or one yielding too few chunks, is not a held-out
///    measurement.
/// 2. Disjointness at full coverage on BOTH sides: every val window is looked
///    up in the set of train windows, so the detection probability is 1.0.
/// 3. Non-degeneracy: a periodic or heavily duplicated eval stream drives BPB
///    toward zero honestly, which is the signature that got 179 ledger rows
///    misfiled as leaks (#62).
fn check_train_val_disjoint(train: &[usize], val: &[usize], seq_len: usize) -> Result<(), String> {
    use std::collections::HashSet;

    if val.len() < MIN_VAL_TOKENS {
        return Err(format!(
            "VAL STREAM TOO SHORT: {} tokens, minimum {}. A BPB averaged over a \
             handful of windows is not a held-out measurement and must not be \
             reported as one.",
            val.len(),
            MIN_VAL_TOKENS
        ));
    }
    let chunks = eval_chunk_count(val.len(), seq_len);
    if chunks < MIN_EVAL_CHUNKS {
        return Err(format!(
            "VAL STREAM YIELDS ONLY {} EVAL CHUNK(S) at seq={}, minimum {}. \
             `evaluate` would average over too few windows for the mean to be \
             informative.",
            chunks, seq_len, MIN_EVAL_CHUNKS
        ));
    }
    if train.len() < OVERLAP_WINDOW {
        return Ok(()); // no train window to compare against
    }

    // Tokens are `% VOCAB` (0..=127), so the windows compare as `u8` slices:
    // exact, and 8x cheaper to hash than `usize` windows.
    let train_b: Vec<u8> = train.iter().map(|&t| t as u8).collect();
    let val_b: Vec<u8> = val.iter().map(|&t| t as u8).collect();
    let train_windows: HashSet<&[u8]> = train_b.windows(OVERLAP_WINDOW).collect();
    let val_total = val_b.len() - OVERLAP_WINDOW + 1;
    let hits = val_b
        .windows(OVERLAP_WINDOW)
        .filter(|w| train_windows.contains(*w))
        .count();
    let fraction = hits as f64 / val_total as f64;
    if fraction > MAX_VAL_OVERLAP_FRACTION {
        return Err(format!(
            "TRAIN/VAL OVERLAP DETECTED: {:.2}% of val windows ({} of {}, window \
             {} tokens) appear verbatim in train; threshold is {:.2}%. This is the \
             2026-04-30 ledger leak bug (trios-trainer-igla#60). Rebuild the split \
             byte-disjoint: head -c $((SIZE-100000)) for train, tail -c 100000 for val.",
            fraction * 100.0,
            hits,
            val_total,
            OVERLAP_WINDOW,
            MAX_VAL_OVERLAP_FRACTION * 100.0
        ));
    }

    let distinct: HashSet<&[usize]> = val.windows(8).collect();
    let total = val.len().saturating_sub(7).max(1);
    let ratio = distinct.len() as f64 / total as f64;
    if ratio < 0.05 {
        return Err(format!(
            "DEGENERATE EVAL CORPUS: only {:.3}% of val 8-grams are distinct \
             ({} of {}). BPB measured against this is not a model result.",
            ratio * 100.0,
            distinct.len(),
            total
        ));
    }
    Ok(())
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

struct AdamWState {
    m: Vec<f32>,
    v: Vec<f32>,
    step: usize,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

impl AdamWState {
    fn new(size: usize) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            step: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }

    fn update(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        self.step += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);
        for i in 0..params.len() {
            params[i] -= self.weight_decay * lr * params[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

struct TrigramModel {
    embed: Vec<f32>,
    context: Vec<f32>,
    lm_head: Vec<f32>,
    vocab: usize,
    dim: usize,
}

impl TrigramModel {
    fn new(vocab: usize, dim: usize, seed: u64) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let t = ((s >> 33) as f32) / (u32::MAX as f32);
            (t * 2.0 - 1.0) * (6.0 / (vocab + dim) as f32).sqrt()
        };
        let embed: Vec<f32> = (0..vocab * dim).map(|_| rng()).collect();
        let context: Vec<f32> = (0..vocab * dim).map(|_| rng()).collect();
        let lm_head: Vec<f32> = (0..vocab * dim).map(|_| rng()).collect();
        Self {
            embed,
            context,
            lm_head,
            vocab,
            dim,
        }
    }

    fn forward_pair(&self, id_prev: usize, id_cur: usize) -> Vec<f32> {
        let v = self.vocab;
        let d = self.dim;
        let prev = id_prev.min(v - 1);
        let cur = id_cur.min(v - 1);
        let e_prev = &self.context[prev * d..(prev + 1) * d];
        let e_cur = &self.embed[cur * d..(cur + 1) * d];
        let combined: Vec<f32> = e_prev
            .iter()
            .zip(e_cur.iter())
            .map(|(a, b)| a + b)
            .collect();
        (0..v)
            .map(|vi| {
                let w = &self.lm_head[vi * d..(vi + 1) * d];
                combined
                    .iter()
                    .zip(w.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
            })
            .collect()
    }

    /// Mean NLL over the sequence, or `None` when there was nothing to measure.
    ///
    /// A sequence shorter than a trigram used to return `0.0` -- a PERFECT
    /// loss, and a finite one, so `evaluate`'s `is_finite()` guard accepted it
    /// and averaged it in. "Too short to measure" and "predicted with
    /// certainty" are not the same statement, and the second one is what a
    /// near-zero BPB looks like. A non-finite total is an absence for the same
    /// reason: a NaN that reaches the caller as a number is a laundered
    /// failure, not a reading.
    fn loss_on_seq(&self, tokens: &[usize]) -> Option<f32> {
        if tokens.len() < 3 {
            return None;
        }
        let mut total = 0.0f32;
        for i in 1..tokens.len() - 1 {
            let logits = self.forward_pair(tokens[i - 1], tokens[i]);
            let target = tokens[i + 1].min(self.vocab - 1);
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = logits.iter().map(|l| (*l - max_l).exp()).sum();
            let log_prob = logits[target] - max_l - sum_exp.ln();
            total -= log_prob;
        }
        let mean = total / (tokens.len() - 2) as f32;
        mean.is_finite().then_some(mean)
    }

    fn train_step(
        &mut self,
        tokens: &[usize],
        lr: f32,
        opt_e: &mut AdamWState,
        opt_c: &mut AdamWState,
        opt_h: &mut AdamWState,
    ) {
        if tokens.len() < 3 {
            return;
        }
        let d = self.dim;
        let v = self.vocab;

        let mut grad_embed = vec![0.0f32; v * d];
        let mut grad_context = vec![0.0f32; v * d];
        let mut grad_head = vec![0.0f32; v * d];

        for i in 1..tokens.len() - 1 {
            let prev = tokens[i - 1].min(v - 1);
            let cur = tokens[i].min(v - 1);
            let tgt = tokens[i + 1].min(v - 1);

            let mut logits = self.forward_pair(prev, cur);
            softmax(&mut logits);

            for (vi, prob) in logits.iter().enumerate() {
                let grad = prob - if vi == tgt { 1.0 } else { 0.0 };
                let e_cur = &self.embed[cur * d..(cur + 1) * d];
                let c_prev = &self.context[prev * d..(prev + 1) * d];
                let w_vi = &self.lm_head[vi * d..(vi + 1) * d];

                for j in 0..d {
                    let combined_j = e_cur[j] + c_prev[j];
                    grad_embed[cur * d + j] += grad * w_vi[j];
                    grad_context[prev * d + j] += grad * w_vi[j];
                    grad_head[vi * d + j] += grad * combined_j;
                }
            }
        }

        let n = (tokens.len() - 2) as f32;
        for g in grad_embed.iter_mut() {
            *g /= n;
        }
        for g in grad_context.iter_mut() {
            *g /= n;
        }
        for g in grad_head.iter_mut() {
            *g /= n;
        }

        opt_e.update(&mut self.embed, &grad_embed, lr);
        opt_c.update(&mut self.context, &grad_context, lr);
        opt_h.update(&mut self.lm_head, &grad_head, lr);
    }
}

/// Mean loss and BPB over the eval windows, or `None` when not one window was
/// measurable.
///
/// The old body returned `(f32::MAX, f32::MAX)` for "zero finite windows".
/// That sentinel passes `is_finite()`, so it was pushed into `results` and
/// serialized into the published artifact as though someone had measured it.
/// `Option` makes an absent measurement unrepresentable as a number;
/// `src/bin/cpu_train.rs::eval_bpb` already reports it this way.
///
/// It then still dropped individual unmeasurable windows and published the mean
/// of the survivors, which is biased DOWNWARD: the windows a partial poison
/// kills are exactly the hard ones. `src/bin/trinity_pr1722.rs` takes the
/// correct line and this now matches it -- ONE unmeasurable window invalidates
/// the whole eval -- while the `dropped` counter keeps the skip from being
/// silent about HOW MUCH of the corpus failed.
fn evaluate(model: &TrigramModel, tokens: &[usize], seq_len: usize) -> Option<(f32, f32)> {
    let mut total = 0.0f32;
    let mut n = 0usize;
    let mut dropped = 0usize;
    for c in (0..tokens.len()).step_by(seq_len + 1) {
        let end = (c + seq_len + 1).min(tokens.len());
        if end - c < 4 {
            continue;
        }
        let seq = &tokens[c..end];
        match model.loss_on_seq(seq) {
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
    Some((bpb * LN_2, bpb))
}

fn cosine_lr(step: usize, max_steps: usize, base_lr: f32, warmup: usize) -> f32 {
    if step < warmup {
        base_lr * step as f32 / warmup as f32
    } else {
        let progress = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        1e-5 + (base_lr - 1e-5) * cosine
    }
}

fn main() {
    // Canon #93 forbids seeds {42, 43, 44, 45} (see src/seed_canon.rs); the
    // default is a permitted one so an argument-free run is not born invalid.
    let seed = std::env::args()
        .find(|a| a.starts_with("--seed="))
        .map(|a| a[7..].parse::<u64>().unwrap_or(47))
        .unwrap_or(47);
    let steps = std::env::args()
        .find(|a| a.starts_with("--steps="))
        .map(|a| a[8..].parse::<usize>().unwrap_or(8000))
        .unwrap_or(8000);
    let base_lr = std::env::args()
        .find(|a| a.starts_with("--lr="))
        .map(|a| a[5..].parse::<f32>().unwrap_or(0.003))
        .unwrap_or(0.003);

    println!("=== IGLA-STACK-502 Trigram Training ===");
    println!(
        "vocab={} dim={} seq={} steps={} seed={} lr={}",
        VOCAB, DIM, SEQ, steps, seed, base_lr
    );
    println!();

    let train_path = arg_path("--train-data", DEFAULT_TRAIN_PATH);
    let val_path = arg_path("--val-data", "");
    let held_out = !val_path.is_empty();

    let (tokens, corpus) = load_or_refuse(&train_path);
    println!("Train corpus: {}", corpus.describe());
    println!("Dataset: {} tokens", tokens.len());

    // The eval stream. Without `--val-data` this binary evaluated on `tokens` -
    // the very array it trains on - and published the reading as `final_bpb`
    // with a complete, correct corpus provenance block beside it, which made
    // the record look more trustworthy than the number was. Single-corpus mode
    // survives, but everything it reports now says `train_set` and carries
    // `held_out: false`.
    let (eval_tokens, eval_corpus) = if held_out {
        if val_path == train_path {
            eprintln!(
                "CORPUS REFUSED: --train-data and --val-data are the same path ({}). \
                 A val stream that is the train stream measures memorisation, not \
                 generalisation.",
                train_path
            );
            std::process::exit(EXIT_BAD_CORPUS);
        }
        let (val_tokens, val_corpus) = load_or_refuse(&val_path);
        println!("Val corpus:   {}", val_corpus.describe());
        if let Err(e) = check_train_val_disjoint(&tokens, &val_tokens, SEQ) {
            eprintln!("SPLIT REFUSED: {e}");
            std::process::exit(EXIT_BAD_CORPUS);
        }
        println!(
            "Split: byte-disjoint, {} val tokens, {} eval chunks (minimum {})",
            val_tokens.len(),
            eval_chunk_count(val_tokens.len(), SEQ),
            MIN_EVAL_CHUNKS
        );
        (val_tokens, val_corpus)
    } else {
        eprintln!(
            "[eval] WARNING: no --val-data given. Every BPB below is measured on \
             the TRAINING corpus. It is a memorisation reading, it is published \
             as `train_set_bpb` with held_out=false, and it is not comparable \
             with any held-out BPB."
        );
        (tokens.clone(), corpus.clone())
    };
    let label = if held_out { "val" } else { "train_set" };

    let mut model = TrigramModel::new(VOCAB, DIM, seed);
    let param_size = VOCAB * DIM;
    let mut opt_e = AdamWState::new(param_size);
    let mut opt_c = AdamWState::new(param_size);
    let mut opt_h = AdamWState::new(param_size);

    let (init_loss, init_bpb) = match evaluate(&model, &eval_tokens, SEQ) {
        Some(r) => r,
        None => {
            eprintln!(
                "NO MEASUREMENT: the initial eval produced zero finite windows on \
                 {}. Refusing to report an initial BPB nobody measured.",
                eval_corpus.path
            );
            std::process::exit(EXIT_NO_MEASUREMENT);
        }
    };
    println!(
        "Initial: loss={:.4} {}_bpb={:.4}",
        init_loss, label, init_bpb
    );
    println!();
    println!(
        "{:>6} | {:>10} | {:>10} | {:>10} | {:>8} | {:>6}",
        "step", "loss", "bpb", "best_bpb", "ms", "lr"
    );
    println!("{}", "-".repeat(70));

    let t0 = Instant::now();
    // `best_bpb` is the running MINIMUM over every eval; `final_bpb` is the
    // single reading taken at `step == steps`. They are different numbers. This
    // file used to keep only the minimum, initialise it to `init_bpb`, and
    // publish it under the name `final_bpb` - so a run whose every eval was
    // non-finite published its ~7.0 initial reading as its final measurement.
    let mut best_bpb = init_bpb;
    let mut final_bpb: Option<f32> = None;
    let mut results: Vec<(usize, f32, f32)> = Vec::new();
    let data_len = tokens.len();

    for step in 1..=steps {
        let lr = cosine_lr(step, steps, base_lr, steps / 10);
        let offset = (step * 97 + seed as usize) % (data_len.saturating_sub(SEQ + 1));
        let seq = &tokens[offset..offset + SEQ + 1];
        model.train_step(seq, lr, &mut opt_e, &mut opt_c, &mut opt_h);

        if step % 500 == 0 || step == steps {
            let ms = t0.elapsed().as_millis();
            match evaluate(&model, &eval_tokens, SEQ) {
                Some((eval_loss, eval_bpb)) => {
                    if eval_bpb < best_bpb && eval_bpb.is_finite() {
                        best_bpb = eval_bpb;
                    }
                    if step == steps {
                        final_bpb = Some(eval_bpb);
                    }
                    println!(
                        "{:>6} | {:>10.4} | {:>10.4} | {:>10.4} | {:>6}ms | {:.6}",
                        step, eval_loss, eval_bpb, best_bpb, ms, lr
                    );
                    results.push((step, eval_loss, eval_bpb));
                }
                // No finite window: no row. Once serialized, a sentinel is
                // indistinguishable from a measurement.
                None => println!(
                    "{:>6} | {:>10} | {:>10} | {:>10.4} | {:>6}ms | {:.6}",
                    step, "unmeasured", "unmeasured", best_bpb, ms, lr
                ),
            }
        }
    }

    let total = t0.elapsed();
    println!();
    println!("=== Training Complete ===");
    match final_bpb {
        Some(f) => println!(
            "Time: {:.1}s | Initial {} BPB: {:.4} | Best {} BPB: {:.4} | \
             Final {} BPB: {:.4} | Delta(best): {:.4} | Delta(final): {:.4}",
            total.as_secs_f64(),
            label,
            init_bpb,
            label,
            best_bpb,
            label,
            f,
            best_bpb - init_bpb,
            f - init_bpb
        ),
        None => println!(
            "Time: {:.1}s | Initial {} BPB: {:.4} | Best {} BPB: {:.4} | \
             Final {} BPB: unmeasured | Delta(best): {:.4}",
            total.as_secs_f64(),
            label,
            init_bpb,
            label,
            best_bpb,
            label,
            best_bpb - init_bpb
        ),
    }
    if held_out {
        println!("held_out=true: measured on {}", eval_corpus.path);
    } else {
        println!(
            "held_out=false: every BPB above is a train_set reading, measured on \
             the corpus this model was trained on ({}). Not a held-out \
             measurement and not citable as one.",
            eval_corpus.path
        );
    }

    let _ = fs::create_dir_all(".trinity/results");
    let mut result_json = serde_json::json!({
        "experiment": "igla-stack-502-trigram",
        "model": "trigram-embedding",
        "optimizer": "AdamW",
        "corpus_path": corpus.path,
        "corpus_bytes": corpus.bytes,
        "corpus_sha256": corpus.sha256,
        "data_synthetic": corpus.synthetic,
        "eval_corpus_path": eval_corpus.path,
        "eval_corpus_bytes": eval_corpus.bytes,
        "eval_corpus_sha256": eval_corpus.sha256,
        "held_out": held_out,
        "seed": seed,
        "vocab_size": VOCAB,
        "dim": DIM,
        "seq_len": SEQ,
        "steps": steps,
        "base_lr": base_lr,
        "duration_seconds": total.as_secs_f64(),
        "results": results.iter().map(|(s, l, b)| serde_json::json!({
            "step": *s, "loss": *l, "bpb": *b
        })).collect::<Vec<_>>(),
    });

    // The BPB keys are named for what they are. A train-set reading never
    // occupies a key called `final_bpb`, and `final_bpb` is the reading at
    // `step == steps` or nothing at all - never the best, never the initial.
    {
        let obj = result_json
            .as_object_mut()
            .expect("the json! literal above is an object");
        let final_value = match final_bpb {
            Some(f) => serde_json::json!(f),
            None => serde_json::Value::Null,
        };
        let delta_final = match final_bpb {
            Some(f) => serde_json::json!(f - init_bpb),
            None => serde_json::Value::Null,
        };
        if held_out {
            obj.insert("initial_bpb".to_string(), serde_json::json!(init_bpb));
            obj.insert("best_bpb".to_string(), serde_json::json!(best_bpb));
            obj.insert("final_bpb".to_string(), final_value);
            obj.insert(
                "delta_best_bpb".to_string(),
                serde_json::json!(best_bpb - init_bpb),
            );
            obj.insert("delta_final_bpb".to_string(), delta_final);
        } else {
            obj.insert(
                "train_set_initial_bpb".to_string(),
                serde_json::json!(init_bpb),
            );
            obj.insert(
                "train_set_best_bpb".to_string(),
                serde_json::json!(best_bpb),
            );
            obj.insert("train_set_bpb".to_string(), final_value);
            obj.insert(
                "train_set_delta_best_bpb".to_string(),
                serde_json::json!(best_bpb - init_bpb),
            );
            obj.insert("train_set_delta_bpb".to_string(), delta_final);
        }
    }

    let rpath = format!(".trinity/results/igla_trigram_seed{}.json", seed);
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
    let final_text = match final_bpb {
        Some(f) => format!("{f:.4}"),
        None => "unmeasured".to_string(),
    };
    let entry = format!(
        "[{}] TASK: IGLA trigram training | seed={} | steps={} | held_out={} | \
         eval_corpus={} | init_{}_bpb={:.4} best={:.4} final={} | {:.1}s\n",
        ts,
        seed,
        steps,
        held_out,
        eval_corpus.path,
        label,
        init_bpb,
        best_bpb,
        final_text,
        total.as_secs_f64()
    );
    let _ = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&epath)
        .unwrap()
        .write_all(entry.as_bytes());
    println!("Experience: {}", epath);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The refusal is the point: a missing corpus must stop the run, not
    /// silently become 52 bytes of pangram with a BPB printed beside it.
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
