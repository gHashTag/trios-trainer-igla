use std::fs;
use std::io::Write;
use std::time::Instant;

const VOCAB: usize = 128;
const DIM: usize = 96;
const CTX_DIM: usize = 128;
const BIGRAM_VOCAB: usize = 512;
const BIGRAM_DIM: usize = 64;
const SEQ: usize = 64;
const LN_2: f32 = std::f32::consts::LN_2;
const LOGIT_SOFTCAP: f32 = 30.0;
const EMA_DECAY: f32 = 0.997;

/// Exit code for a corpus that could not be honestly loaded. Same value
/// `cpu_train` uses, so a sweep can tell a refused corpus from a crash.
const EXIT_BAD_CORPUS: i32 = 6;

/// Exit code for a run that reached the end without a measurable eval window.
/// Same value `cpu_train` uses.
const EXIT_NO_MEASUREMENT: i32 = 7;

/// `evaluate` must average over at least this many chunks for the mean to mean
/// anything. Re-exported from the library rather than copied: the overlap
/// window, the overlap threshold and the minimum val size used to be four local
/// constants beside a local copy of the guard, and a constant that is copied is
/// a constant that will drift.
use trios_trainer::train_loop::MIN_EVAL_CHUNKS;

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
/// The old body substituted a 52-byte pangram whenever the read failed, so a
/// clean checkout trained on it and printed a BPB for it. A missing corpus is
/// now a hard, named refusal. `TRIOS_ALLOW_SYNTHETIC_DATA=1` opts back in,
/// says so on stderr on every run, and stamps `data_synthetic=true` into
/// everything the run writes.
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

/// Refuse a split that cannot carry a held-out measurement.
///
/// The body used to be a LOCAL COPY of
/// `trios_trainer::train_loop::assert_train_val_disjoint`, written because that
/// function was `pub(crate)` and `src/bin/*.rs` compile as separate crates. It
/// is now a thin adapter: same constants, same thresholds, same full-coverage
/// comparison, one implementation. The `step_by(256)` scan that detected an
/// overlap with probability 1/256 lived in exactly such a copy.
///
/// `seq_len` is this binary's own eval chunking, passed through so the size
/// precondition is asked at the coverage the run will actually use.
///
/// Returns the refusal instead of panicking so the caller can exit before any
/// result file exists.
fn check_train_val_disjoint(train: &[usize], val: &[usize], seq_len: usize) -> Result<(), String> {
    trios_trainer::train_loop::check_train_val_disjoint(
        train,
        val,
        eval_chunk_count(val.len(), seq_len),
    )
}

fn softmax_cap(v: &mut [f32], softcap: f32) {
    for x in v.iter_mut() {
        *x = (*x / softcap).tanh() * softcap;
    }
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
    fn new(size: usize, _lr: f32, wd: f32) -> Self {
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
    fn update(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        self.step += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);
        for i in 0..params.len() {
            params[i] -= self.wd * lr * params[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

fn bigram_hash(cur: usize, prev: usize, vocab: usize) -> usize {
    ((36313u32.wrapping_mul(cur as u32)) ^ (27191u32.wrapping_mul(prev as u32))) as usize % vocab
}

struct TrinityCpuModel {
    embed: Vec<f32>,
    ctx_embed: Vec<f32>,
    bigram_embed: Vec<f32>,
    smear_gate: Vec<f32>,
    lm_head: Vec<f32>,
    ve_proj: Vec<f32>,
    ve_scale: Vec<f32>,
    vocab: usize,
    #[allow(dead_code)]
    dim: usize,
    ctx_dim: usize,
    bigram_vocab: usize,
    bigram_dim: usize,
    ve_dim: usize,
}

impl TrinityCpuModel {
    fn new(
        vocab: usize,
        dim: usize,
        ctx_dim: usize,
        bv: usize,
        bd: usize,
        ve_dim: usize,
        seed: u64,
    ) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let t = ((s >> 33) as f32) / (u32::MAX as f32);
            (t * 2.0 - 1.0) * (6.0f32 / (vocab + dim) as f32).sqrt()
        };
        let total_dim = ctx_dim + bd + ve_dim;
        Self {
            embed: (0..vocab * total_dim).map(|_| rng()).collect(),
            ctx_embed: (0..vocab * ctx_dim).map(|_| rng()).collect(),
            bigram_embed: (0..bv * bd).map(|_| rng()).collect(),
            smear_gate: vec![0.0f32; total_dim],
            lm_head: (0..vocab * total_dim).map(|_| rng()).collect(),
            ve_proj: (0..ve_dim * total_dim).map(|_| rng()).collect(),
            ve_scale: vec![1.0f32; 2],
            vocab,
            dim,
            ctx_dim,
            bigram_vocab: bv,
            bigram_dim: bd,
            ve_dim,
        }
    }

    fn get_repr(&self, prev: usize, cur: usize) -> Vec<f32> {
        let v = self.vocab;
        let cd = self.ctx_dim;
        let bd = self.bigram_dim;
        let vd = self.ve_dim;
        let total = cd + bd + vd;

        let cur_idx = cur.min(v - 1);
        let prev_idx = prev.min(v - 1);
        let bh = bigram_hash(cur, prev, self.bigram_vocab);

        let mut repr = vec![0.0f32; total];

        let e_cur = &self.embed[cur_idx * total..(cur_idx + 1) * total];
        let c_prev = &self.ctx_embed[prev_idx * cd..(prev_idx + 1) * cd];
        let b_hash = &self.bigram_embed[bh * bd..(bh + 1) * bd];

        for j in 0..cd {
            repr[j] = e_cur[j] + c_prev[j];
        }
        for j in 0..bd {
            repr[cd + j] = e_cur[cd + j] + b_hash[j] * 0.1;
        }
        for j in 0..vd {
            let mut ve_val = 0.0f32;
            for (k, r) in repr.iter().enumerate().take(total) {
                ve_val += self.ve_proj[j * total + k] * r;
            }
            repr[cd + bd + j] = e_cur[cd + bd + j] + ve_val * self.ve_scale[0];
        }

        let _sg: Vec<f32> = self
            .smear_gate
            .iter()
            .map(|&g| 1.0 / (1.0 + (-g).exp()))
            .collect();
        for r in repr.iter_mut() {
            *r *= 1.0;
        }

        layer_norm(&repr, 1e-5)
    }

    /// Mean cross-entropy in nats over the windows of `tokens`, or `None` when
    /// no measurement exists.
    ///
    /// `None`, not a number, for a sequence too short to score: the old body
    /// returned 0.0, which is a perfect prediction -- the best value this
    /// function can produce -- for the case where it predicted nothing at all.
    ///
    /// `None`, not a clamp, for a probability that is not finite and strictly
    /// positive. `f32::max` returns the OTHER operand when one side is NaN, so
    /// clamping `logits[target]` to a 1e-10 floor turned a poisoned forward
    /// pass into exactly `-ln(1e-10)` = 23.0259 nats = 33.219 bpb, a finite
    /// reading indistinguishable from a measured one - and it did the same to
    /// a merely UNDERFLOWED probability, a finite `0.0` out of the f32 softmax
    /// that `is_nan` and `is_finite` both accept.
    fn loss_on_seq(&self, tokens: &[usize]) -> Option<f32> {
        if tokens.len() < 3 {
            return None;
        }
        let v = self.vocab;
        let total = self.ctx_dim + self.bigram_dim + self.ve_dim;
        let mut total_loss = 0.0f32;

        for i in 1..tokens.len() - 1 {
            let repr = self.get_repr(tokens[i - 1], tokens[i]);
            let target = tokens[i + 1].min(v - 1);

            let mut logits: Vec<f32> = (0..v)
                .map(|vi| {
                    let w = &self.lm_head[vi * total..(vi + 1) * total];
                    repr.iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f32>()
                })
                .collect();
            softmax_cap(&mut logits, LOGIT_SOFTCAP);

            let p = logits[target];
            if !p.is_finite() || p <= 0.0 {
                return None;
            }
            total_loss -= p.ln();
        }
        Some(total_loss / (tokens.len() - 2) as f32)
    }

    fn train_step(
        &mut self,
        tokens: &[usize],
        lr: f32,
        opt_e: &mut AdamW,
        opt_c: &mut AdamW,
        opt_b: &mut AdamW,
        opt_h: &mut AdamW,
    ) {
        if tokens.len() < 3 {
            return;
        }
        let v = self.vocab;
        let total = self.ctx_dim + self.bigram_dim + self.ve_dim;
        let cd = self.ctx_dim;
        let bd = self.bigram_dim;

        let mut grad_embed = vec![0.0f32; v * total];
        let mut grad_ctx = vec![0.0f32; v * cd];
        let mut grad_bigram = vec![0.0f32; self.bigram_vocab * bd];
        let mut grad_head = vec![0.0f32; v * total];

        for i in 1..tokens.len() - 1 {
            let prev = tokens[i - 1].min(v - 1);
            let cur = tokens[i].min(v - 1);
            let tgt = tokens[i + 1].min(v - 1);
            let bh = bigram_hash(cur, prev, self.bigram_vocab);

            let repr = self.get_repr(prev, cur);

            let mut logits: Vec<f32> = (0..v)
                .map(|vi| {
                    let w = &self.lm_head[vi * total..(vi + 1) * total];
                    repr.iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f32>()
                })
                .collect();
            softmax_cap(&mut logits, LOGIT_SOFTCAP);

            for (vi, prob) in logits.iter().enumerate() {
                let grad = prob - if vi == tgt { 1.0 } else { 0.0 };
                let w_vi = &self.lm_head[vi * total..(vi + 1) * total];
                for j in 0..total {
                    grad_embed[cur * total + j] += grad * w_vi[j];
                    grad_head[vi * total + j] += grad * repr[j];
                }
                for j in 0..cd {
                    grad_ctx[prev * cd + j] += grad * w_vi[j];
                }
                for j in 0..bd {
                    grad_bigram[bh * bd + j] += grad * w_vi[cd + j] * 0.1;
                }
            }
        }

        let n = (tokens.len() - 2) as f32;
        for g in grad_embed.iter_mut() {
            *g /= n;
        }
        for g in grad_ctx.iter_mut() {
            *g /= n;
        }
        for g in grad_bigram.iter_mut() {
            *g /= n;
        }
        for g in grad_head.iter_mut() {
            *g /= n;
        }

        opt_e.update(&mut self.embed, &grad_embed, lr);
        opt_c.update(&mut self.ctx_embed, &grad_ctx, lr);
        opt_b.update(&mut self.bigram_embed, &grad_bigram, lr);
        opt_h.update(&mut self.lm_head, &grad_head, lr);
    }

    #[allow(dead_code)]
    fn apply_ema(&mut self, ema: &Self) {
        for i in 0..self.embed.len() {
            self.embed[i] = ema.embed[i] * (1.0 - EMA_DECAY) + self.embed[i] * EMA_DECAY;
        }
    }
}

/// Mean loss and BPB over the eval windows, or `None` when any window was
/// unmeasurable.
///
/// The old body returned `(f32::MAX, f32::MAX)` for "zero finite windows".
/// That sentinel passes `is_finite()`, so it was pushed into `results` and
/// serialized into the published artifact as though someone had measured it.
/// `Option` makes an absent measurement unrepresentable as a number;
/// `src/bin/cpu_train.rs::eval_bpb` already reports it this way.
///
/// The loop no longer averages over "the windows that happened to come out
/// finite". It took an `f32` from `loss_on_seq` and kept it only
/// `if loss.is_finite()`, which did two wrong things at once: it averaged in
/// the laundered 23.0259-nat reading that a NaN forward pass produced, and it
/// dropped genuinely non-finite windows from `n` so the survivors reported a
/// confident mean over a corpus that had partly failed to evaluate. One
/// unmeasurable window now invalidates the whole eval, exactly as in
/// `src/train_loop.rs`.
fn evaluate(model: &TrinityCpuModel, tokens: &[usize], seq_len: usize) -> Option<(f32, f32)> {
    let mut total = 0.0f32;
    let mut n = 0usize;
    for c in (0..tokens.len()).step_by(seq_len + 1) {
        let end = (c + seq_len + 1).min(tokens.len());
        if end - c < 4 {
            continue;
        }
        let loss = model.loss_on_seq(&tokens[c..end])?;
        if !loss.is_finite() {
            return None;
        }
        total += loss / LN_2;
        n += 1;
    }
    if n == 0 {
        return None;
    }
    let bpb = total / n as f32;
    Some((bpb * LN_2, bpb))
}

fn cosine_lr(step: usize, max_steps: usize, base_lr: f32, warmup: usize) -> f32 {
    if step < warmup {
        return base_lr * step as f32 / warmup as f32;
    }
    let progress = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    1e-5 + (base_lr - 1e-5) * cosine
}

/// Every argument this binary reads, in the spellings its own parser reads.
///
/// `--seed`, `--steps` and `--lr` are matched with `starts_with("--name=")`, so
/// only the `=` form exists for them; `--train-data` / `--val-data` go through
/// `arg_path`, which accepts both spellings and therefore appear twice. See
/// `trios_trainer::reject_unknown_args`.
const KNOWN_ARGS: [&str; 7] = [
    "seed=",
    "steps=",
    "lr=",
    "train-data",
    "train-data=",
    "val-data",
    "val-data=",
];

fn main() {
    // First, before a corpus is opened: this binary feeds `--train-data` and
    // `--val-data` into the train/val disjointness guard, so an argument name
    // it did not recognise made the guard evaluate a different pair than the
    // operator asked for, while the run still published a BPB with exit 0.
    let args: Vec<String> = std::env::args().collect();
    if let Err(reason) = trios_trainer::reject_unknown_args(&args, &KNOWN_ARGS) {
        eprintln!("{reason}");
        std::process::exit(i32::from(trios_trainer::EXIT_BAD_ARGS));
    }
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

    println!("=== Trinity CPU Model (PR#1722 params adapted) ===");
    println!(
        "arch: vocab={} ctx_dim={} bigram={}x{} ve_dim={} seq={} softcap={}",
        VOCAB, CTX_DIM, BIGRAM_VOCAB, BIGRAM_DIM, 16, SEQ, LOGIT_SOFTCAP
    );
    println!(
        "opt: AdamW(phi beta1) wd=0.04 lr={} steps={} seed={}",
        base_lr, steps, seed
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

    let total_dim = CTX_DIM + BIGRAM_DIM + 16;
    let mut model = TrinityCpuModel::new(VOCAB, DIM, CTX_DIM, BIGRAM_VOCAB, BIGRAM_DIM, 16, seed);
    let mut ema_model =
        TrinityCpuModel::new(VOCAB, DIM, CTX_DIM, BIGRAM_VOCAB, BIGRAM_DIM, 16, seed);

    let mut opt_e = AdamW::new(VOCAB * total_dim, base_lr, 0.04);
    let mut opt_c = AdamW::new(VOCAB * CTX_DIM, base_lr, 0.04);
    let mut opt_b = AdamW::new(BIGRAM_VOCAB * BIGRAM_DIM, base_lr, 0.04);
    let mut opt_h = AdamW::new(VOCAB * total_dim, base_lr, 0.04);

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
        "{:>6} | {:>10} | {:>10} | {:>10} | {:>8}",
        "step", "loss", "bpb", "best_bpb", "ms"
    );
    println!("{}", "-".repeat(60));

    let t0 = Instant::now();
    // `best_bpb` is the running MINIMUM over every eval; `final_bpb` is the
    // single reading taken at `step == steps`. They are different numbers. This
    // file used to keep only the minimum, initialise it to `init_bpb`, and
    // print and publish it as `Final BPB ... (EMA)` - neither final (a running
    // minimum) nor an EMA (the EMA is which weights were evaluated, not how the
    // number was reduced). A run whose every eval was non-finite published its
    // ~7.0 initial reading as its final measurement.
    let mut best_bpb = init_bpb;
    let mut final_bpb: Option<f32> = None;
    let mut results: Vec<(usize, f32, f32)> = Vec::new();
    let data_len = tokens.len();

    for step in 1..=steps {
        let lr = cosine_lr(step, steps, base_lr, steps / 10);
        let offset = (step * 97 + seed as usize) % (data_len.saturating_sub(SEQ + 1));
        let seq = &tokens[offset..offset + SEQ + 1];
        model.train_step(seq, lr, &mut opt_e, &mut opt_c, &mut opt_b, &mut opt_h);

        for i in 0..model.embed.len() {
            ema_model.embed[i] =
                ema_model.embed[i] * EMA_DECAY + model.embed[i] * (1.0 - EMA_DECAY);
        }
        for i in 0..model.lm_head.len() {
            ema_model.lm_head[i] =
                ema_model.lm_head[i] * EMA_DECAY + model.lm_head[i] * (1.0 - EMA_DECAY);
        }

        if step % 500 == 0 || step == steps {
            let ms = t0.elapsed().as_millis();
            match evaluate(&ema_model, &eval_tokens, SEQ) {
                Some((eval_loss, eval_bpb)) => {
                    if eval_bpb < best_bpb && eval_bpb.is_finite() {
                        best_bpb = eval_bpb;
                    }
                    if step == steps {
                        final_bpb = Some(eval_bpb);
                    }
                    println!(
                        "{:>6} | {:>10.4} | {:>10.4} | {:>10.4} | {:>6}ms",
                        step, eval_loss, eval_bpb, best_bpb, ms
                    );
                    results.push((step, eval_loss, eval_bpb));
                }
                // No finite window: no row. Once serialized, a sentinel is
                // indistinguishable from a measurement.
                None => println!(
                    "{:>6} | {:>10} | {:>10} | {:>10.4} | {:>6}ms",
                    step, "unmeasured", "unmeasured", best_bpb, ms
                ),
            }
        }
    }

    let total = t0.elapsed();
    println!();
    println!("=== Training Complete ===");
    println!("Eval weights: EMA (decay {})", EMA_DECAY);
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
        "experiment": "trinity-pr1722-adapted",
        "source_pr": "openai/parameter-golf#1722",
        "techniques": ["BigramHash", "SmearGate", "LayerNorm", "LogitSoftcap", "EMA", "AdamW-phi", "CosineLR"],
        "pr1722_params": {
            "arch": "11L 512d 8h/4kv MLP3x",
            "bigram_vocab": BIGRAM_VOCAB, "bigram_dim": BIGRAM_DIM,
            "logit_softcap": LOGIT_SOFTCAP, "ema_decay": EMA_DECAY,
            "muon_momentum": 0.99, "wd": 0.04, "qk_gain_init": 1.5
        },
        "corpus_path": corpus.path,
        "corpus_bytes": corpus.bytes,
        "corpus_sha256": corpus.sha256,
        "data_synthetic": corpus.synthetic,
        "eval_corpus_path": eval_corpus.path,
        "eval_corpus_bytes": eval_corpus.bytes,
        "eval_corpus_sha256": eval_corpus.sha256,
        "held_out": held_out,
        "eval_weights": "ema",
        "seed": seed, "steps": steps, "base_lr": base_lr,
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

    let rpath = format!(".trinity/results/trinity_pr1722_seed{}.json", seed);
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
        "[{}] TASK: Trinity PR#1722 adapted training | seed={} | steps={} | \
         held_out={} | eval_corpus={} | eval_weights=ema | init_{}_bpb={:.4} \
         best={:.4} final={} | {:.1}s\n",
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
