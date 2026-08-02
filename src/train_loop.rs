use anyhow::{Context, Result};
use std::io::Read;
use std::time::Instant;

use crate::arch_config::{parse_gf16_enabled, parse_hidden_dim, parse_num_attn_layers};
use crate::fake_quant::{self, FormatKind};
use crate::model_hybrid_attn::{AttentionCache, HybridAttn};
use crate::objective::{nca_entropy_loss, NcaObjective};

pub const DEFAULT_IGLA_TARGET_BPB: f64 = 1.85;
/// Canon #93 sweep seeds - Lucas/Fibonacci aligned.
/// Forbidden under Canon #93: `{42, 43, 44, 45}`.
/// Allowed canon (Wave-29):  `{47, 89, 123, 144}`.
/// Sweep uses the first three; `144` is reserved for the bridge canon.
/// Wave-29 PR-A.1 replaces the legacy `{43, 44, 45}` (entirely
/// forbidden) with the Canon #93 triple.
/// Anchor: phi^2+phi^-2=3 - DOI 10.5281/zenodo.19227877
pub const GATE_FINAL_SEEDS: &[u64] = &[47, 89, 123];

const VOCAB: usize = 128;
const DIM: usize = 64;
const NUM_CTX: usize = 6;
const NGRAM: usize = NUM_CTX + 2;
const SEQ: usize = 128;
const LN_2: f32 = std::f32::consts::LN_2;
const PHI_INV: f32 = 0.618033988749895;
const CTX_WEIGHTS: [f32; NUM_CTX] = [0.70, 0.45, 0.30, 0.20, 0.13, 0.08];
const ATTN_SEQ: usize = 8;

fn attn_scale() -> f32 {
    std::env::var("TRIOS_ATTN_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1)
}

fn gf16_enabled() -> bool {
    std::env::var("TRIOS_GF16_DISABLE")
        .map(|v| v != "1")
        .unwrap_or(true)
}

/// Default cadence for `gf16_floor()`: rewrite the weights on every step past
/// the 70% mark.
pub const GF16_FLOOR_EVERY_DEFAULT: u64 = 1;

/// How often `gf16_floor()` rewrites embed/proj/lm_head/ctx once the run is
/// past `gf16_floor_step`.
///
/// This used to be `step % args.eval_every == 0`, which made an OBSERVATION
/// parameter change the artifact: `gf16_floor` mutates the weights in place,
/// so two runs identical except for `--eval-every` produced different weights,
/// different checkpoint hashes and different BPB (2.6141 vs 2.6169 on seed 47).
/// Nothing in the artifact said so, and no reviewer expects the eval cadence to
/// be part of the training recipe.
///
/// The default is 1 - "floor every step past the mark" - which is what the code
/// was pretending to mean. It is deliberately NOT `eval_every`: defaulting to
/// the old expression would preserve the defect under a better name, and runs
/// at different cadences were never comparable anyway, so there is no baseline
/// worth preserving. Anyone needing bit-compatibility with a pre-fix run sets
/// `TRIOS_GF16_FLOOR_EVERY` to that run's `eval_every`, and the value is now
/// recorded in the checkpoint sidecar.
///
/// A missing, unparseable or zero value resolves to the default; the resolved
/// number is printed in the run banner so a typo is visible at step 0.
pub fn gf16_floor_every() -> u64 {
    std::env::var("TRIOS_GF16_FLOOR_EVERY")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(GF16_FLOOR_EVERY_DEFAULT)
}

/// Wave 31 PR-B: resolve GF16 from `GF16_ENABLED` env knob (default false).
/// If `GF16_ENABLED=true` but feature `gf16` is not compiled in, returns Err.
/// Falls back to `gf16_enabled()` for legacy `TRIOS_GF16_DISABLE` path.
/// Anchor: phi^2+phi^-2=3 - DOI 10.5281/zenodo.19227877
fn resolve_gf16_knob() -> Result<bool> {
    let knob = parse_gf16_enabled().map_err(|e| anyhow::anyhow!("GF16_ENABLED: {e}"))?;
    if knob {
        #[cfg(not(feature = "gf16"))]
        {
            return Err(anyhow::anyhow!(
                "GF16_ENABLED=true but feature 'gf16' not compiled in; \
                 rebuild with: cargo build --features gf16"
            ));
        }
        #[cfg(feature = "gf16")]
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn attn_seq_override() -> usize {
    std::env::var("TRIOS_ATTN_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(ATTN_SEQ)
}

/// EPIC-446: checkpointing is ON by default. Off-by-default would reproduce the
/// exact defect being fixed for anyone who forgets the flag - the whole failure
/// was 1,851 experiments and no artifact. `TRIOS_CHECKPOINT_DISABLE=1` is the
/// escape hatch for CI and smoke runs.
fn checkpoint_enabled() -> bool {
    std::env::var("TRIOS_CHECKPOINT_DISABLE").as_deref() != Ok("1")
}

/// Extra checkpoint cadence: `TRIOS_CHECKPOINT_EVERY=N` (default 0 = final
/// step only). Deliberately a NEW variable, not `neon_writer::checkpoint_interval()`
/// (`TRIOS_CHECKPOINT_INTERVAL`, default 200): that one is BPB *sampling*
/// cadence read by `igla_train.rs` and `ngram_train_gf16.rs`, and silently
/// reusing it would turn an 81k-step run into 405 files for callers who set it
/// for an unrelated reason.
fn checkpoint_every_hit(step: usize) -> bool {
    let n: usize = std::env::var("TRIOS_CHECKPOINT_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    n > 0 && step % n == 0
}

/// Resolve the run identity ONCE per run. The same string names the checkpoint
/// directory and the ledger row, so the artifact and the BPB cannot drift
/// apart. R5: a missing canon_name must never cost a training run silently.
fn resolve_canon_name(seed: u64) -> String {
    std::env::var("TRIOS_CANON_NAME")
        .ok()
        .or_else(|| std::env::var("CANON_NAME").ok())
        .unwrap_or_else(|| format!("trios-train-rng{seed}"))
}

#[derive(Debug)]
pub struct TrainArgs {
    pub seed: u64,
    pub steps: usize,
    pub hidden: usize,
    pub lr: f32,
    pub attn_layers: u8,
    pub eval_every: usize,
    pub train_path: String,
    pub val_path: String,
}

/// What a run actually measured, with each number named for what it is.
///
/// The single `final_bpb` this replaces was `best_bpb`: the running MINIMUM of
/// an exponential moving average seeded with `init_bpb` (~7.0) and carrying
/// weight `PHI_INV = 0.618` on the stale value. It was neither final nor a
/// measurement. Two runs with byte-identical weights (sha `9edbf02cc9b49593...`)
/// printed 3.5506 and 4.4940 through it while the raw reading was 2.8534 in
/// both, because the EMA had not caught up and had caught up differently at
/// different eval cadences.
///
/// The EMA is kept - it is a useful early-stopping signal - but it is no longer
/// an output.
#[derive(Debug)]
pub struct RunOutcome {
    /// Raw `val_bpb` measured at the last eval, i.e. the one where
    /// `step == args.steps`. `None` when no such eval was performed (a
    /// zero-step run); callers must say so rather than substitute a number.
    pub final_val_bpb: Option<f64>,
    /// Minimum over the RAW readings, not over the EMA.
    pub best_val_bpb: Option<f64>,
    /// Last EMA value. An early-stopping signal, reported for completeness.
    pub ema_bpb: Option<f64>,
    /// Compatibility mirror of `final_val_bpb`, `NaN` when that is `None`.
    ///
    /// Kept only because `src/bin/format_champion_sweep.rs`,
    /// `src/bin/tri.rs` and `tests/champion_reproduction.rs` read this field
    /// and are outside this change's ownership. It now carries the MEASURED
    /// number instead of the EMA minimum, so those consumers are already
    /// fixed; the field itself should be deleted in the commit that updates
    /// them to `final_val_bpb`.
    pub final_bpb: f64,
    pub steps_done: usize,
    pub seed: u64,
}

/// Load a byte corpus as VOCAB-folded tokens.
///
/// Returns `(tokens, synthetic)`. The synthetic fallback is OPT-IN: silently
/// substituting `b"The quick brown fox..."` for a missing corpus is the
/// documented cause of the leak-tainted BPB rows (trios-trainer-igla#60), and
/// an `eprintln!` warning demonstrably did not prevent them.
///
/// The opt-in is a practical requirement, not a softening: `data/` tracks only
/// `fineweb_train.bin` / `fineweb_val.bin`, so a pure hard error would break
/// every CI job and fresh clone. What makes it non-regressive is that the
/// `synthetic` bit is threaded into `CheckpointMeta`, becomes byte 125 of the
/// hashed checkpoint header, and lands in the sidecar and the ledger row - a
/// synthetic run is now permanently and cryptographically self-identifying.
fn load_data(path: &str) -> Result<(Vec<usize>, bool)> {
    if std::path::Path::new(path).exists() {
        let raw = std::fs::read(path)
            .with_context(|| format!("failed to read corpus '{path}'"))?;
        assert_alphabet_fold_injective(path, &raw)?;
        return Ok((raw.into_iter().map(|b| (b as usize) % VOCAB).collect(), false));
    }
    if std::env::var("TRIOS_ALLOW_SYNTHETIC_DATA").as_deref() != Ok("1") {
        anyhow::bail!(
            "corpus '{path}' not found. Refusing the synthetic fallback: it is the \
             documented cause of the leak-tainted BPB rows (trios-trainer-igla#60). \
             Provide the corpus, or set TRIOS_ALLOW_SYNTHETIC_DATA=1 to opt in - \
             runs that do are stamped data_synthetic=true in the checkpoint header, \
             the sidecar and the ledger row, and their BPB is meaningless."
        );
    }
    eprintln!(
        "[data] WARNING: TRIOS_ALLOW_SYNTHETIC_DATA=1 and '{path}' is missing. \
         Using the synthetic corpus. Any BPB from this run is meaningless and \
         every artifact it produces is stamped data_synthetic=true."
    );
    let fallback = b"The quick brown fox jumps over the lazy dog. ".repeat(2500);
    Ok((
        fallback.into_iter().map(|b| (b as usize) % VOCAB).collect(),
        true,
    ))
}

/// Refuse a corpus that the `% VOCAB` fold would not encode injectively.
///
/// `VOCAB` is 128 and `load_data` folds every byte with `(b as usize) % VOCAB`,
/// which is the identity on ASCII and a two-to-one collapse on everything else.
/// For UTF-8 Cyrillic the collapse is total and silent: 0xD0 (the lead byte of
/// most Russian letters) folds onto `'P'`, 0xD1 onto `'Q'`, and 0xA0 - the
/// continuation byte of Cyrillic capital ER, the first letter of *Rossiya* and
/// *Rosstandart* - folds onto a space. A Cyrillic corpus therefore trained with
/// no warning at all and printed a figure labelled bits-per-BYTE that had in
/// fact been measured over a 128-symbol alphabet the corpus does not use. The
/// number is not merely optimistic, it names a unit it does not have.
///
/// Refusal, not a warning and not an env opt-in. An UNRECORDED opt-in is the
/// exact defect class this repository keeps finding (the synthetic-corpus
/// fallback above is opt-in *because* the opt-in is stamped into the header,
/// the sidecar and the ledger row; there is no such slot for this one). And
/// widening `VOCAB` to 256 is not the fix either: it would invalidate the
/// `7.00 = log2(128)` initialisation and the entire calibration curve measured
/// against it. This trainer is an ASCII-only fixture; it now says so.
pub(crate) fn assert_alphabet_fold_injective(path: &str, raw: &[u8]) -> Result<()> {
    let offenders = raw.iter().filter(|&&b| b as usize >= VOCAB).count();
    if offenders == 0 {
        return Ok(());
    }
    let (at, first) = raw
        .iter()
        .enumerate()
        .find(|(_, &b)| b as usize >= VOCAB)
        .map(|(i, &b)| (i, b))
        .unwrap_or((0, 0));
    let folded = (first as usize) % VOCAB;
    anyhow::bail!(
        "ALPHABET FOLD REFUSED: {offenders} of {} bytes in '{path}' are >= {VOCAB} and \
         would collide mod {VOCAB} (e.g. 0x{first:02X} at offset {at} -> 0x{folded:02X}, \
         and 0xA0 -> ' '); this trainer is an ASCII-only fixture and its BPB is not \
         bits-per-byte on this corpus. Supply an ASCII corpus, or measure this one \
         with a trainer whose alphabet covers it.",
        raw.len()
    );
}

/// Comparison window, in tokens, for the train/val overlap guard.
///
/// 256 verbatim bytes of natural text is decisive evidence of a copy, and a
/// shorter window than the previous 1024 is strictly MORE sensitive: it also
/// catches partial copies whose shared span is under 1 KB.
pub(crate) const OVERLAP_WINDOW: usize = 256;

/// Fail the run above this fraction of val windows found verbatim in train.
///
/// Not zero. The comparison is exact, so on a small val stream a single line of
/// shared boilerplate would otherwise be fatal. A real leak is not marginal:
/// the 2026-04-30 Dockerfile split put 90-100% of val inside train, and the
/// 99%-copy case this threshold exists for lands two orders of magnitude above
/// it.
pub(crate) const MAX_VAL_OVERLAP_FRACTION: f64 = 0.01;

/// A val stream shorter than this cannot support a BPB anyone should quote.
/// `data/pangram_fixture_160b.bin` is 160 bytes and yields exactly ONE 129-token
/// window, which `evaluate` used to report, unqualified, as `val_bpb`.
pub(crate) const MIN_VAL_TOKENS: usize = 8192;

/// `evaluate` must average over at least this many chunks for the mean to mean
/// anything.
pub(crate) const MIN_EVAL_CHUNKS: usize = 8;

/// Assert the val stream is large enough to measure and byte-disjoint from
/// train. Called by `run_single` and `run_single_muon` right after load, before
/// any gradient step.
///
/// Three preconditions, each of which has already been violated in a live run:
///
/// 1. **Size.** A val shorter than `MIN_VAL_TOKENS`, or one that yields fewer
///    than `MIN_EVAL_CHUNKS` chunks in `evaluate`, is refused. A single 129-token
///    window is not a held-out measurement.
/// 2. **Disjointness, at full coverage on BOTH sides.** The previous version
///    hashed every train window but still probed only `val[..1024]`, so a val
///    that was 99.0% a verbatim copy of train passed silently: the leak simply
///    had to start after the first window. Every val window is now looked up in
///    the train set and the overlap FRACTION is reported, so the panic says how
///    bad the leak is rather than merely that one exists.
/// 3. **Non-degeneracy.** A periodic or heavily duplicated eval stream drives
///    BPB toward zero honestly - the model really does predict it - which is the
///    signature that got 179 ledger rows misfiled as leaks (#62).
///
/// Tokens are `% VOCAB` (0..=127), so the windows are compared as `u8` slices:
/// exact, and 8x cheaper to hash than the `usize` windows this replaces.
pub(crate) fn assert_train_val_disjoint(train: &[usize], val: &[usize]) {
    use std::collections::HashSet;

    assert!(
        val.len() >= MIN_VAL_TOKENS,
        "VAL STREAM TOO SHORT: {} tokens, minimum {}. A BPB averaged over a \
         handful of windows is not a held-out measurement and must not be \
         reported as one (data/pangram_fixture_160b.bin is 160 bytes and \
         yielded exactly one 129-token window).",
        val.len(),
        MIN_VAL_TOKENS
    );
    let chunks = eval_chunk_count(val.len());
    assert!(
        chunks >= MIN_EVAL_CHUNKS,
        "VAL STREAM YIELDS ONLY {} EVAL CHUNK(S), minimum {}. `evaluate` would \
         average over too few windows for the mean to be informative.",
        chunks,
        MIN_EVAL_CHUNKS
    );

    if train.len() < OVERLAP_WINDOW {
        return; // no train window to compare against
    }

    // Full coverage on both sides. Every train window is hashed once
    // (O(n) memory, O(n) time) and EVERY val window is looked up in that set,
    // so the detection probability is 1.0 - no sampling, nothing to report as
    // a false-negative rate.
    let train_b: Vec<u8> = train.iter().map(|&t| t as u8).collect();
    let val_b: Vec<u8> = val.iter().map(|&t| t as u8).collect();
    let train_windows: HashSet<&[u8]> = train_b.windows(OVERLAP_WINDOW).collect();
    let val_total = val_b.len() - OVERLAP_WINDOW + 1;
    let hits = val_b
        .windows(OVERLAP_WINDOW)
        .filter(|w| train_windows.contains(*w))
        .count();
    let fraction = hits as f64 / val_total as f64;
    assert!(
        fraction <= MAX_VAL_OVERLAP_FRACTION,
        "TRAIN/VAL OVERLAP DETECTED: {:.2}% of val windows ({} of {}, window \
         {} tokens) appear verbatim in train; threshold is {:.2}%. This is the \
         2026-04-30 ledger leak bug (trios-trainer-igla#60). Rebuild the split \
         byte-disjoint: head -c $((SIZE-100000)) for train, tail -c 100000 for val.",
        fraction * 100.0,
        hits,
        val_total,
        OVERLAP_WINDOW,
        MAX_VAL_OVERLAP_FRACTION * 100.0
    );

    let distinct: HashSet<&[usize]> = val.windows(8).collect();
    let total = val.len().saturating_sub(7).max(1);
    let ratio = distinct.len() as f64 / total as f64;
    assert!(
        ratio >= 0.05,
        "DEGENERATE EVAL CORPUS: only {:.3}% of val 8-grams are distinct \
         ({} of {}). BPB measured against this is not a model result.",
        ratio * 100.0,
        distinct.len(),
        total
    );
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

fn gf16_floor(weights: &mut [f32]) {
    let scale = 16.0_f32;
    for w in weights.iter_mut() {
        *w = (*w * scale).round() / scale;
    }
}

/// Resolve the QAT format from `TRIOS_FORMAT_TYPE` (or alias `TRIOS_FAKE_QUANT_FORMAT`).
/// Returns `None` if the env var is unset or maps to F32 (no quantization).
/// Closes scarab->trios-train gap from #509: previously only `cpu_train` honoured the
/// env var, so `TRIOS_FORMAT_TYPE=fp16` produced identical BPB to F32 in production.
fn resolve_fake_quant_format() -> Option<FormatKind> {
    let raw = std::env::var("TRIOS_FORMAT_TYPE")
        .ok()
        .or_else(|| std::env::var("TRIOS_FAKE_QUANT_FORMAT").ok())?;
    let fmt = FormatKind::from_env(&raw)?;
    if fmt == FormatKind::F32 {
        return None;
    }
    Some(fmt)
}

/// Apply Phase-1 fake-quantization to every weight tensor in the hybrid model.
/// Skips identity formats (F32 / `is_unsupported_in_f32()`) so the call is a no-op
/// when QAT is disabled.
fn fake_quantize_model(model: &mut HybridModel, fmt: FormatKind) {
    if fmt == FormatKind::F32 || fmt.is_unsupported_in_f32() {
        return;
    }
    fake_quant::fake_quantize_weights(&mut model.embed, fmt);
    fake_quant::fake_quantize_weights(&mut model.proj, fmt);
    fake_quant::fake_quantize_weights(&mut model.lm_head, fmt);
    fake_quant::fake_quantize_weights(&mut model.attn_down, fmt);
    fake_quant::fake_quantize_weights(&mut model.attn_up, fmt);
    for c in model.ctx.iter_mut() {
        fake_quant::fake_quantize_weights(c, fmt);
    }
    fake_quant::fake_quantize_weights(model.attn.wq_mut(), fmt);
    fake_quant::fake_quantize_weights(model.attn.wk_mut(), fmt);
    fake_quant::fake_quantize_weights(model.attn.wv_mut(), fmt);
    fake_quant::fake_quantize_weights(model.attn.wo_mut(), fmt);
    fake_quant::fake_quantize_weights(model.attn.wq2_mut(), fmt);
    fake_quant::fake_quantize_weights(model.attn.wk2_mut(), fmt);
    fake_quant::fake_quantize_weights(model.attn.wv2_mut(), fmt);
    fake_quant::fake_quantize_weights(model.attn.wo2_mut(), fmt);
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

struct ForwardCache {
    combined: Vec<f32>,
    ln: Vec<f32>,
    hidden_pre_attn: Vec<f32>,
    attn_input: Vec<f32>,
    attn_out: Vec<f32>,
    hidden: Vec<f32>,
    logits: Vec<f32>,
    attn_v2_cache: Option<crate::model_hybrid_attn::ForwardCache>,
    attn_seq: usize,
    combined_seq: Vec<f32>,
    ln_seq: Vec<f32>,
}

impl HybridModel {
    fn new(hidden: usize, seed: u64, attn_layers: u8) -> Self {
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

        let attn = if attn_layers == 1 {
            let mut cfg = crate::model_hybrid_attn::HybridAttnConfig::default();
            cfg.num_attn_layers = 1;
            HybridAttn::with_config(cfg).expect("1-layer attn config valid")
        } else {
            HybridAttn::new().expect("2-layer attn defaults valid")
        };
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

        let d = m.attn.config().d_model;
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

        let attn_params = d * hidden * 2 + m.attn.total_weights();
        let total =
            VOCAB * DIM + NUM_CTX * VOCAB * DIM + hidden * DIM + VOCAB * hidden + attn_params;
        eprintln!(
            "params={} ({:.1}K) attn_d={} attn_layers={}",
            total,
            total as f64 / 1000.0,
            d,
            attn_layers
        );
        m
    }

    fn forward_cached(&self, tokens: &[usize], pos: usize) -> ForwardCache {
        let h = self.hidden;
        let d = self.attn.config().d_model;

        let attn_seq = attn_seq_override().min(pos + 1);
        let seq_start = pos + 1 - attn_seq;

        let mut attn_input = vec![0.0f32; attn_seq * d];
        let mut combined_seq = vec![0.0f32; attn_seq * DIM];
        let mut ln_seq = vec![0.0f32; attn_seq * DIM];

        for (si, p) in (seq_start..=pos).enumerate() {
            let t_last = tokens[p + NGRAM - 1].min(VOCAB - 1);
            let mut combined = self.embed[t_last * DIM..(t_last + 1) * DIM].to_vec();
            for (ci, cw) in CTX_WEIGHTS.iter().enumerate() {
                let ctx_idx = NGRAM - 2 - ci;
                let t = tokens[p + ctx_idx].min(VOCAB - 1);
                let cv = &self.ctx[ci][t * DIM..(t + 1) * DIM];
                for j in 0..DIM {
                    combined[j] += cv[j] * cw;
                }
            }
            let ln = layer_norm(&combined, 1e-5);
            combined_seq[si * DIM..(si + 1) * DIM].copy_from_slice(&combined);
            ln_seq[si * DIM..(si + 1) * DIM].copy_from_slice(&ln);
            attn_input[si * DIM..(si + 1) * DIM].copy_from_slice(&ln);
        }

        let d_model = self.attn.config().d_model;
        let num_heads = self.attn.config().num_heads;
        let (attn_output, attn_v2_cache) = self
            .attn
            .forward_cached(&attn_input, attn_seq)
            .unwrap_or_else(|_| {
                (
                    vec![0.0f32; attn_seq * d_model],
                    crate::model_hybrid_attn::ForwardCache::new(attn_seq, d_model, num_heads),
                )
            });

        let attn_out = attn_output[(attn_seq - 1) * d..attn_seq * d].to_vec();

        let mut attn_up_out = vec![0.0f32; h];
        for hi in 0..h {
            for di in 0..d {
                attn_up_out[hi] += self.attn_up[hi * d + di] * attn_out[di];
            }
        }

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
        let ln = layer_norm(&combined, 1e-5);

        let mut hidden_raw = vec![0.0f32; h];
        for hi in 0..h {
            for j in 0..DIM {
                hidden_raw[hi] += self.proj[hi * DIM + j] * ln[j];
            }
        }
        let mut hidden_pre_attn = vec![0.0f32; h];
        for hi in 0..h {
            hidden_pre_attn[hi] = if hidden_raw[hi] > 0.0 {
                hidden_raw[hi] * hidden_raw[hi]
            } else {
                0.0
            };
        }
        let mut hidden = hidden_pre_attn.clone();
        for hi in 0..h {
            hidden[hi] += attn_up_out[hi] * attn_scale();
        }

        let mut logits = vec![0.0f32; VOCAB];
        for vi in 0..VOCAB {
            for hi in 0..h {
                logits[vi] += self.lm_head[vi * h + hi] * hidden[hi];
            }
        }

        ForwardCache {
            combined,
            ln,
            hidden_pre_attn,
            attn_input,
            attn_out,
            hidden,
            logits,
            attn_v2_cache: Some(attn_v2_cache),
            attn_seq,
            combined_seq,
            ln_seq,
        }
    }

    /// Mean cross-entropy over the sequence, in nats/token.
    ///
    /// `None` means "could not measure". It must never be representable as a
    /// number, because the old code returned 0.0 here and 0.0 is also what a
    /// perfect prediction looks like - a failure indistinguishable from the
    /// best possible result (trios-trainer-igla#62).
    fn loss_on_seq(&self, tokens: &[usize]) -> Option<f32> {
        let count = tokens.len().saturating_sub(NGRAM);
        if tokens.len() < NGRAM + 1 || count == 0 {
            return None;
        }
        let mut total = 0.0f32;
        for i in 0..count {
            let target = tokens[i + NGRAM].min(VOCAB - 1);
            let fc = self.forward_cached(tokens, i);
            let mut logits = fc.logits;
            softmax(&mut logits);
            // `f32::max` IGNORES NaN, so `NaN.max(1e-10)` is 1e-10: a poisoned
            // forward pass would be laundered into a finite 23.03-nat reading
            // that sails through every `is_finite` check downstream.
            let p = logits[target];
            if !p.is_finite() {
                return None;
            }
            total -= p.max(1e-10).ln();
        }
        Some(total / count as f32)
    }
}

// ===================================================================
// EPIC-446 - checkpoint codec
//
// `HybridModel` is private to this module, so the codec lives here rather
// than in `checkpoint.rs`: that keeps the diff to zero visibility changes.
// The byte layout is documented in `crate::checkpoint`.
// ===================================================================

/// Run-identity and semantics scalars carried in the checkpoint header.
/// Everything here either changes the function the loaded weights compute,
/// or records the provenance of the run that produced them.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CheckpointMeta {
    pub seed: u64,
    pub step: u64,
    /// `TrainArgs.lr` - the training lr, distinct from `HybridAttnConfig::lr`.
    pub train_lr: f32,
    /// `attn_scale()`, default 0.1 (`TRIOS_ATTN_SCALE`).
    pub attn_scale: f32,
    /// `attn_seq_override()`, default `ATTN_SEQ` = 8 (`TRIOS_ATTN_SEQ`).
    pub attn_seq: u32,
    /// `gf16_enabled()` (`TRIOS_GF16_DISABLE`).
    pub gf16_enabled: bool,
    /// True when `load_data` used the opt-in synthetic corpus.
    pub data_synthetic: bool,
    /// "adamw" | "muon" | "muon-cwd". Derived from which entry point ran,
    /// NOT parsed from `canon_name` - the `ALGO_WHITELIST` guard in
    /// `neon_writer` gates the parsed string and can therefore record a lie.
    pub optimizer: String,
    /// Lowercase `FormatKind::name()`, or "f32" when QAT is off.
    pub fake_quant_format: String,
}
// NOTE: corpus identity is deliberately NOT here. Everything in this struct is
// carried in the fixed-width header, and the header has 24 free bytes -- not
// the 64 two SHA-256 digests need. Corpus provenance therefore lives in the
// sidecar (`CorpusProvenance`), written atomically beside the `.bin`. Moving
// the digests into the artifact itself is a format-version bump and worth
// doing, but it is a deliberate change, not a late one.

/// Copy `s` into a fixed-width NUL-padded ASCII field. Errors rather than
/// truncating: a silently shortened optimizer label is a falsified header.
fn ascii_field<const N: usize>(s: &str, what: &str) -> Result<[u8; N]> {
    anyhow::ensure!(
        s.is_ascii() && s.bytes().all(|b| (0x20..=0x7e).contains(&b)),
        "{what} {s:?} is not printable ASCII"
    );
    anyhow::ensure!(s.len() <= N, "{what} {s:?} exceeds {N} bytes");
    let mut out = [0u8; N];
    out[..s.len()].copy_from_slice(s.as_bytes());
    Ok(out)
}

/// Decode a fixed-width NUL-padded ASCII field. Rejects non-ASCII bytes and
/// any non-NUL byte appearing after the first NUL (i.e. NUL-*terminated*
/// rather than NUL-*padded* data, which would round-trip differently).
fn read_ascii_field(raw: &[u8], what: &str) -> Result<String> {
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    anyhow::ensure!(
        raw[end..].iter().all(|&b| b == 0),
        "{what} field is not NUL-padded"
    );
    anyhow::ensure!(
        raw[..end].iter().all(|&b| (0x20..=0x7e).contains(&b)),
        "{what} field is not printable ASCII"
    );
    Ok(String::from_utf8_lossy(&raw[..end]).into_owned())
}

fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn rd_u64(b: &[u8], off: usize) -> u64 {
    let mut x = [0u8; 8];
    x.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(x)
}

impl HybridModel {
    /// The 19 tensors in canonical order (see `crate::checkpoint` module docs).
    /// `attn_down` is index 8 and `attn_up` is index 9; they have identical
    /// lengths, so this ordering is the only thing preventing a swap.
    fn checkpoint_tensors(&self) -> Vec<&[f32]> {
        let mut t: Vec<&[f32]> = Vec::with_capacity(crate::checkpoint::CHECKPOINT_TENSOR_COUNT);
        t.push(&self.embed);
        for c in &self.ctx {
            t.push(c);
        }
        t.push(&self.proj);
        t.push(&self.attn_down);
        t.push(&self.attn_up);
        t.push(&self.lm_head);
        t.push(&self.attn.wq);
        t.push(&self.attn.wk);
        t.push(&self.attn.wv);
        t.push(&self.attn.wo);
        t.push(&self.attn.wq2);
        t.push(&self.attn.wk2);
        t.push(&self.attn.wv2);
        t.push(&self.attn.wo2);
        t
    }

    /// Element counts the header scalars imply, in canonical order. The
    /// on-disk directory must match this exactly.
    fn expected_tensor_counts(hidden: usize, d_model: usize) -> [u64; 19] {
        let vd = (VOCAB * DIM) as u64;
        let dd = (d_model * d_model) as u64;
        [
            vd,
            vd,
            vd,
            vd,
            vd,
            vd,
            vd,
            (hidden * DIM) as u64,
            (d_model * hidden) as u64,
            (hidden * d_model) as u64,
            (VOCAB * hidden) as u64,
            dd,
            dd,
            dd,
            dd,
            dd,
            dd,
            dd,
            dd,
        ]
    }

    /// Serialize every learned tensor plus the scalars a loader needs to
    /// rebuild the exact shapes. Format v1; see the `checkpoint` module docs
    /// for the byte layout. Length is exactly `304 + 4 * total f32 count`.
    fn to_checkpoint_bytes(&self, meta: &CheckpointMeta) -> Result<Vec<u8>> {
        use crate::checkpoint::{
            CHECKPOINT_FORMAT_VERSION, CHECKPOINT_HEADER_LEN, CHECKPOINT_MAGIC,
            CHECKPOINT_PAYLOAD_OFFSET, CHECKPOINT_TENSOR_COUNT,
        };

        let cfg = *self.attn.config();
        let tensors = self.checkpoint_tensors();
        anyhow::ensure!(
            tensors.len() == CHECKPOINT_TENSOR_COUNT,
            "expected {CHECKPOINT_TENSOR_COUNT} tensors, model produced {}",
            tensors.len()
        );
        let counts = Self::expected_tensor_counts(self.hidden, cfg.d_model);
        for (i, t) in tensors.iter().enumerate() {
            anyhow::ensure!(
                t.len() as u64 == counts[i],
                "tensor {i} has {} elements, header implies {}",
                t.len(),
                counts[i]
            );
        }

        let total: u64 = counts.iter().sum();
        let mut out = vec![0u8; CHECKPOINT_PAYLOAD_OFFSET + 4 * total as usize];

        out[0..8].copy_from_slice(CHECKPOINT_MAGIC);
        out[8..12].copy_from_slice(&CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        out[12..16].copy_from_slice(&(CHECKPOINT_HEADER_LEN as u32).to_le_bytes());
        out[16..20].copy_from_slice(&(VOCAB as u32).to_le_bytes());
        out[20..24].copy_from_slice(&(DIM as u32).to_le_bytes());
        out[24..28].copy_from_slice(&(NUM_CTX as u32).to_le_bytes());
        out[28..32].copy_from_slice(&(self.hidden as u32).to_le_bytes());
        out[32..36].copy_from_slice(&(cfg.d_model as u32).to_le_bytes());
        out[36..40].copy_from_slice(&(cfg.num_heads as u32).to_le_bytes());
        out[40..44].copy_from_slice(&(cfg.seq_len as u32).to_le_bytes());
        out[44..48].copy_from_slice(&(cfg.num_attn_layers as u32).to_le_bytes());
        out[48..52].copy_from_slice(&(NGRAM as u32).to_le_bytes());
        out[52..56].copy_from_slice(&(CHECKPOINT_TENSOR_COUNT as u32).to_le_bytes());
        out[56..64].copy_from_slice(&cfg.qk_gain.to_le_bytes());
        out[64..72].copy_from_slice(&cfg.lr.to_le_bytes());
        out[72..76].copy_from_slice(&meta.train_lr.to_le_bytes());
        out[76..80].copy_from_slice(&meta.attn_scale.to_le_bytes());
        out[80..84].copy_from_slice(&meta.attn_seq.to_le_bytes());
        for (i, w) in CTX_WEIGHTS.iter().enumerate() {
            out[84 + 4 * i..88 + 4 * i].copy_from_slice(&w.to_le_bytes());
        }
        out[108..116].copy_from_slice(&meta.seed.to_le_bytes());
        out[116..124].copy_from_slice(&meta.step.to_le_bytes());
        out[124] = u8::from(meta.gf16_enabled);
        out[125] = u8::from(meta.data_synthetic);
        // out[126..128] stays zero: reserved, rejected on load if nonzero.
        out[128..136].copy_from_slice(&ascii_field::<8>(&meta.optimizer, "optimizer")?);
        out[136..152].copy_from_slice(&ascii_field::<16>(
            &meta.fake_quant_format,
            "fake_quant_format",
        )?);

        for (i, c) in counts.iter().enumerate() {
            let off = CHECKPOINT_HEADER_LEN + 8 * i;
            out[off..off + 8].copy_from_slice(&c.to_le_bytes());
        }

        let mut off = CHECKPOINT_PAYLOAD_OFFSET;
        for t in &tensors {
            for v in t.iter() {
                out[off..off + 4].copy_from_slice(&v.to_le_bytes());
                off += 4;
            }
        }
        debug_assert_eq!(off, out.len());
        Ok(out)
    }

    /// Inverse of `to_checkpoint_bytes`. Runs every check in the load
    /// validation list and returns Err rather than reshaping silently: a load
    /// that produces a differently shaped model is worse than no load.
    fn from_checkpoint_bytes(bytes: &[u8]) -> Result<(Self, CheckpointMeta)> {
        use crate::checkpoint::{
            CHECKPOINT_FORMAT_VERSION, CHECKPOINT_HEADER_LEN, CHECKPOINT_MAGIC,
            CHECKPOINT_PAYLOAD_OFFSET, CHECKPOINT_TENSOR_COUNT,
        };
        use crate::model_hybrid_attn::HybridAttnConfig;

        anyhow::ensure!(
            bytes.len() >= CHECKPOINT_PAYLOAD_OFFSET,
            "checkpoint truncated: {} bytes, need at least {CHECKPOINT_PAYLOAD_OFFSET}",
            bytes.len()
        );
        anyhow::ensure!(&bytes[0..8] == CHECKPOINT_MAGIC, "bad magic (not TRIOSCKP)");
        let version = rd_u32(bytes, 8);
        anyhow::ensure!(
            version == CHECKPOINT_FORMAT_VERSION,
            "unsupported checkpoint format_version {version} (this build reads {CHECKPOINT_FORMAT_VERSION})"
        );
        let header_len = rd_u32(bytes, 12) as usize;
        anyhow::ensure!(
            header_len == CHECKPOINT_HEADER_LEN,
            "header_len {header_len} != {CHECKPOINT_HEADER_LEN}"
        );
        anyhow::ensure!(
            bytes[126] == 0 && bytes[127] == 0,
            "reserved bytes 126..128 are nonzero"
        );
        anyhow::ensure!(
            bytes[124] <= 1 && bytes[125] <= 1,
            "boolean header bytes 124/125 must be 0 or 1"
        );

        let vocab = rd_u32(bytes, 16) as usize;
        let dim = rd_u32(bytes, 20) as usize;
        let num_ctx = rd_u32(bytes, 24) as usize;
        let hidden = rd_u32(bytes, 28) as usize;
        let d_model = rd_u32(bytes, 32) as usize;
        let num_heads = rd_u32(bytes, 36) as usize;
        let cfg_seq_len = rd_u32(bytes, 40) as usize;
        let num_attn_layers = rd_u32(bytes, 44);
        let ngram = rd_u32(bytes, 48) as usize;
        let tensor_count = rd_u32(bytes, 52) as usize;

        anyhow::ensure!(
            tensor_count == CHECKPOINT_TENSOR_COUNT,
            "tensor_count {tensor_count} != {CHECKPOINT_TENSOR_COUNT}"
        );
        anyhow::ensure!(
            vocab == VOCAB && dim == DIM && num_ctx == NUM_CTX && ngram == NGRAM,
            "shape constants differ from this build: \
             vocab={vocab}/{VOCAB} dim={dim}/{DIM} num_ctx={num_ctx}/{NUM_CTX} ngram={ngram}/{NGRAM}"
        );
        // `forward_cached` allocates `attn_input` with stride `d_model` but
        // fills it with stride `DIM`; divergence corrupts silently.
        anyhow::ensure!(
            d_model == dim,
            "d_model {d_model} != dim {dim}; forward_cached assumes they are equal"
        );
        anyhow::ensure!(hidden > 0, "hidden is 0");
        anyhow::ensure!(
            num_attn_layers == 1 || num_attn_layers == 2,
            "num_attn_layers {num_attn_layers} not in {{1, 2}}"
        );

        let expected = Self::expected_tensor_counts(hidden, d_model);
        let mut counts = [0u64; CHECKPOINT_TENSOR_COUNT];
        for (i, slot) in counts.iter_mut().enumerate() {
            *slot = rd_u64(bytes, CHECKPOINT_HEADER_LEN + 8 * i);
            anyhow::ensure!(
                *slot == expected[i],
                "tensor directory entry {i} is {} but the header implies {}",
                *slot,
                expected[i]
            );
        }

        let total: u64 = counts.iter().sum();
        let want_len = CHECKPOINT_PAYLOAD_OFFSET as u64 + 4 * total;
        anyhow::ensure!(
            bytes.len() as u64 == want_len,
            "checkpoint length {} != {want_len} implied by the tensor directory",
            bytes.len()
        );

        for (i, w) in CTX_WEIGHTS.iter().enumerate() {
            let stored = rd_u32(bytes, 84 + 4 * i);
            anyhow::ensure!(
                stored == w.to_bits(),
                "ctx_weights[{i}] bits {stored:#010x} != this build's {:#010x}",
                w.to_bits()
            );
        }

        let optimizer = read_ascii_field(&bytes[128..136], "optimizer")?;
        let fake_quant_format = read_ascii_field(&bytes[136..152], "fake_quant_format")?;

        let cfg = HybridAttnConfig {
            d_model,
            num_heads,
            seq_len: cfg_seq_len,
            qk_gain: f64::from_bits(rd_u64(bytes, 56)),
            lr: f64::from_bits(rd_u64(bytes, 64)),
            num_attn_layers: num_attn_layers as u8,
        };
        // Re-checks qk_gain in {phi^2, phi^3} and lr in [0.002, 0.007].
        let mut attn = HybridAttn::with_config(cfg)
            .map_err(|e| anyhow::anyhow!("checkpoint header failed HybridAttnConfig::validate: {e:?}"))?;

        let mut off = CHECKPOINT_PAYLOAD_OFFSET;
        let mut take = |n: u64, off: &mut usize| -> Vec<f32> {
            let mut v = Vec::with_capacity(n as usize);
            for _ in 0..n {
                v.push(f32::from_bits(rd_u32(bytes, *off)));
                *off += 4;
            }
            v
        };

        let embed = take(counts[0], &mut off);
        let ctx: Vec<Vec<f32>> = (1..=NUM_CTX).map(|i| take(counts[i], &mut off)).collect();
        let proj = take(counts[7], &mut off);
        let attn_down = take(counts[8], &mut off);
        let attn_up = take(counts[9], &mut off);
        let lm_head = take(counts[10], &mut off);
        attn.wq = take(counts[11], &mut off);
        attn.wk = take(counts[12], &mut off);
        attn.wv = take(counts[13], &mut off);
        attn.wo = take(counts[14], &mut off);
        attn.wq2 = take(counts[15], &mut off);
        attn.wk2 = take(counts[16], &mut off);
        attn.wv2 = take(counts[17], &mut off);
        attn.wo2 = take(counts[18], &mut off);
        debug_assert_eq!(off, bytes.len());

        let meta = CheckpointMeta {
            seed: rd_u64(bytes, 108),
            step: rd_u64(bytes, 116),
            train_lr: f32::from_bits(rd_u32(bytes, 72)),
            attn_scale: f32::from_bits(rd_u32(bytes, 76)),
            attn_seq: rd_u32(bytes, 80),
            gf16_enabled: bytes[124] != 0,
            data_synthetic: bytes[125] != 0,
            optimizer,
            fake_quant_format,
        };
        Ok((
            Self {
                embed,
                ctx,
                proj,
                attn,
                attn_down,
                attn_up,
                lm_head,
                hidden,
            },
            meta,
        ))
    }
}

/// The run parameters that a reader needs in order to know what produced these
/// exact weights, separated from the ones that only decide when to look.
///
/// `gf16_floor_every` is in the artifact because `gf16_floor()` rewrites
/// embed/proj/lm_head/ctx in place; `eval_every` is in the artifact so a
/// reviewer can CONFIRM it is not part of the recipe, which is exactly the
/// claim that was false before `TRIOS_GF16_FLOOR_EVERY` existed.
struct RunKnobs {
    steps_total: u64,
    gf16_floor_every: u64,
    eval_every: u64,
}

/// Write the artifact, then record it - locally first, then in the ledger.
///
/// Order matters: the `.bin` always lands (a filesystem object must never
/// depend on a DSN), then the sidecar is written with `ledger = "pending"`,
/// then the ledger is attempted, then the sidecar is rewritten with the real
/// outcome. A crash between the artifact and the DB attempt therefore still
/// leaves an on-disk record naming the file and its hash; "pending" is itself
/// honest information, and reconciliation later is a plain
/// `INSERT ... ON CONFLICT DO NOTHING` replayed from the sidecars.
fn emit_checkpoint(
    model: &HybridModel,
    canon: &str,
    meta: &CheckpointMeta,
    corpus: &crate::checkpoint::CorpusProvenance,
    run: &RunKnobs,
    bpb: Option<f64>,
    best_val_bpb: Option<f64>,
    ema_bpb: Option<f64>,
) -> Result<()> {
    use crate::checkpoint::{CheckpointRecord, CHECKPOINT_FORMAT_VERSION, CHECKPOINT_RECORD_SCHEMA};

    let bytes = model.to_checkpoint_bytes(meta)?;
    let saved = crate::checkpoint::save(canon, meta.step as usize, &bytes)?;
    let cfg = *model.attn.config();
    let path_str = saved.path.to_string_lossy().into_owned();
    let (git_sha, git_dirty, git_provenance) = crate::checkpoint::resolve_git_provenance();
    // Schema 3. `lr`, `attn_scale` and `attn_seq` come from `meta` - the same
    // struct that was just hashed into the header - and NOT from re-reading
    // `TRIOS_ATTN_*` here, where an env change mid-run would make the sidecar
    // describe a configuration the weights were never trained under.
    let platform = crate::checkpoint::resolve_platform_provenance();
    let source_sha256 = crate::checkpoint::resolve_source_digest();
    let trainer_prov = crate::checkpoint::resolve_trainer_provenance();

    let record = |ledger: &str| CheckpointRecord {
        schema: CHECKPOINT_RECORD_SCHEMA.to_string(),
        canon_name: canon.to_string(),
        seed: meta.seed as i64,
        step: meta.step as i64,
        path: path_str.clone(),
        sha256: saved.sha256.clone(),
        bytes: saved.bytes,
        format_version: CHECKPOINT_FORMAT_VERSION,
        hidden: model.hidden as u32,
        d_model: cfg.d_model as u32,
        num_attn_layers: cfg.num_attn_layers as u32,
        optimizer: meta.optimizer.clone(),
        fake_quant_format: meta.fake_quant_format.clone(),
        data_synthetic: meta.data_synthetic,
        steps_total: run.steps_total,
        gf16_floor_every: run.gf16_floor_every,
        eval_every: run.eval_every,
        final_val_bpb: bpb,
        best_val_bpb,
        ema_bpb,
        git_sha: git_sha.clone(),
        git_provenance: git_provenance.to_string(),
        git_dirty,
        corpus: corpus.clone(),
        run_id: std::env::var("RAILWAY_DEPLOYMENT_ID").unwrap_or_default(),
        ledger: ledger.to_string(),
        ts: chrono::Utc::now().to_rfc3339(),
        // The f32 the optimizer actually stepped with, widened exactly: `--lr
        // 0.003` lands as 0.003000000026077032. The typed decimal is not what
        // ran, and this record describes what ran.
        lr: Some(meta.train_lr as f64),
        attn_scale: meta.attn_scale as f64,
        attn_seq: meta.attn_seq as u64,
        platform: platform.clone(),
        source_sha256: source_sha256.clone(),
        trainer: trainer_prov.clone(),
        vocab: VOCAB as u32,
    };

    crate::checkpoint::write_sidecar(&record("pending"))?;
    let outcome = crate::neon_writer::checkpoint_record(
        canon,
        meta.seed as i32,
        meta.step as i64,
        &path_str,
        &saved.sha256,
        saved.bytes as i64,
        &meta.optimizer,
        model.hidden as i32,
        CHECKPOINT_FORMAT_VERSION as i32,
        meta.data_synthetic,
        bpb,
    );
    crate::checkpoint::write_sidecar(&record(outcome.as_str()))?;

    eprintln!(
        "[ckpt] {} sha256={} bytes={} ledger={}",
        path_str,
        saved.sha256,
        saved.bytes,
        outcome.as_str()
    );
    Ok(())
}

fn compute_grads(
    model: &HybridModel,
    tokens: &[usize],
    positions: &[usize],
    g_embed: &mut [f32],
    g_ctx: &mut [Vec<f32>],
    g_proj: &mut [f32],
    g_head: &mut [f32],
    g_attn_down: &mut [f32],
    g_attn_up: &mut [f32],
    g_attn_weights: &mut [f32],
) {
    let h = model.hidden;
    let d = model.attn.config().d_model;
    let dd = d * d;

    for &pos in positions {
        let fc = model.forward_cached(tokens, pos);
        let ForwardCache {
            combined,
            ln,
            hidden_pre_attn,
            attn_input: _,
            attn_out,
            hidden,
            mut logits,
            attn_v2_cache,
            attn_seq,
            combined_seq,
            ln_seq,
        } = fc;

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

        let scale = attn_scale();
        let d_attn_up_out: Vec<f32> = d_hidden.iter().map(|&dh| dh * scale).collect();
        let mut d_attn_out_last = vec![0.0f32; d];
        for hi in 0..h {
            for di in 0..d {
                g_attn_up[hi * d + di] += d_attn_up_out[hi] * attn_out[di];
                d_attn_out_last[di] += d_attn_up_out[hi] * model.attn_up[hi * d + di];
            }
        }

        if let Some(cache) = attn_v2_cache {
            let seq = attn_seq;
            let mut d_output = vec![0.0f32; seq * d];
            d_output[(seq - 1) * d..seq * d].copy_from_slice(&d_attn_out_last);

            let mut grads = crate::model_hybrid_attn::AttentionGradients::new(d);
            let d_ai = model.attn.backward_v2(&d_output, &cache, &mut grads);

            for i in 0..dd {
                g_attn_weights[i] += grads.d_wq[i];
            }
            for i in 0..dd {
                g_attn_weights[dd + i] += grads.d_wk[i];
            }
            for i in 0..dd {
                g_attn_weights[2 * dd + i] += grads.d_wv[i];
            }
            for i in 0..dd {
                g_attn_weights[3 * dd + i] += grads.d_wo[i];
            }
            for i in 0..dd {
                g_attn_weights[4 * dd + i] += grads.d_wq2[i];
            }
            for i in 0..dd {
                g_attn_weights[5 * dd + i] += grads.d_wk2[i];
            }
            for i in 0..dd {
                g_attn_weights[6 * dd + i] += grads.d_wv2[i];
            }
            for i in 0..dd {
                g_attn_weights[7 * dd + i] += grads.d_wo2[i];
            }

            for si in 0..seq {
                let p = pos + 1 - attn_seq + si;
                let d_ln_si = &d_ai[si * DIM..(si + 1) * DIM];
                let c_si = &combined_seq[si * DIM..(si + 1) * DIM];
                let l_si = &ln_seq[si * DIM..(si + 1) * DIM];
                let d_combined_si = layer_norm_backward(c_si, l_si, d_ln_si, 1e-5);

                let t_last = tokens[p + NGRAM - 1].min(VOCAB - 1);
                for j in 0..DIM {
                    g_embed[t_last * DIM + j] += d_combined_si[j];
                }
                for (ci, cw) in CTX_WEIGHTS.iter().enumerate() {
                    let ctx_idx = NGRAM - 2 - ci;
                    let t = tokens[p + ctx_idx].min(VOCAB - 1);
                    for j in 0..DIM {
                        g_ctx[ci][t * DIM + j] += cw * d_combined_si[j];
                    }
                }
            }
        }

        let mut d_raw = vec![0.0f32; h];
        for hi in 0..h {
            if hidden_pre_attn[hi] > 0.0 {
                d_raw[hi] = d_hidden[hi] * 2.0 * hidden_pre_attn[hi].sqrt();
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

/// How many chunks `evaluate` would average over for a stream of `len` tokens.
///
/// Extracted so `assert_train_val_disjoint` and `evaluate` cannot disagree
/// about what "too few chunks to be a measurement" means. Mirrors the loop
/// bounds below exactly; a change to one must change the other.
pub(crate) fn eval_chunk_count(len: usize) -> usize {
    let seq = SEQ + 1;
    let num_chunks = 40usize;
    let max_start = len.saturating_sub(seq);
    if max_start == 0 {
        return 0;
    }
    let step = if max_start >= num_chunks * seq {
        max_start / num_chunks
    } else {
        seq
    };
    max_start.div_ceil(step.max(1)).min(num_chunks)
}

/// Mean bits-per-byte over evenly spaced chunks of `tokens`.
///
/// `None` means the evaluation could not be performed. A chunk that cannot be
/// measured invalidates the whole eval rather than being quietly dropped from
/// the average: silently skipping chunks is how an eval over a degenerate
/// corpus still produced a confident-looking number (#62).
fn evaluate(model: &HybridModel, tokens: &[usize]) -> Option<f32> {
    let seq = SEQ + 1;
    let num_chunks = 40usize;
    let max_start = tokens.len().saturating_sub(seq);
    if max_start == 0 {
        return None;
    }
    let step = if max_start >= num_chunks * seq {
        max_start / num_chunks
    } else {
        seq
    };
    let mut total = 0.0f32;
    let mut n = 0usize;
    for c in (0..max_start).step_by(step).take(num_chunks) {
        let end = (c + seq).min(tokens.len());
        let loss = model.loss_on_seq(&tokens[c..end])?;
        if !loss.is_finite() {
            return None;
        }
        total += loss / LN_2;
        n += 1;
    }
    if n == 0 {
        None
    } else {
        Some(total / n as f32)
    }
}

/// Reject a BPB reading that cannot be a real measurement of held-out text.
///
/// The crate already owns this law in `race::bpb` / `race::victory`, but
/// `train_loop` used to print and ship numbers its own tracker would refuse.
/// Measured calibration on the verified byte-disjoint tinyshakespeare split:
/// ~7.00 at init, ~3.33 at step 1000, 2.75-2.83 at step 12000. Even a 100%
/// verbatim train/val overlap only reaches 2.71 on this architecture, so a
/// near-zero reading is never a good model - it is a degenerate eval corpus.
fn guard_bpb(vbpb: f32, step: usize) -> Result<f32> {
    anyhow::ensure!(
        vbpb.is_finite(),
        "val_bpb is not finite at step {step}; refusing to emit"
    );
    anyhow::ensure!(
        (vbpb as f64) > crate::race::victory::JEPA_PROXY_BPB_FLOOR,
        "val_bpb={vbpb:.6} <= JEPA_PROXY_BPB_FLOOR at step {step}: the eval \
         corpus is degenerate or duplicated, not the model perfect. This is \
         the signature that mislabelled 179 ledger rows as data leaks (#62)."
    );
    Ok(vbpb)
}

pub fn run_single(args: &TrainArgs) -> Result<RunOutcome> {
    // Wave 31 PR-B: apply env-gated arch knobs (HIDDEN_DIM, NUM_ATTN_LAYERS).
    // Defaults preserve Wave-30 baseline (h=384, 1L).
    // Anchor: phi^2+phi^-2=3 - DOI 10.5281/zenodo.19227877
    let eff_hidden = if std::env::var("HIDDEN_DIM").is_ok() {
        let h = parse_hidden_dim().map_err(|e| anyhow::anyhow!("HIDDEN_DIM: {e}"))?;
        eprintln!("[arch-knob] HIDDEN_DIM={h} (override, Wave-31 PR-B)");
        h
    } else {
        args.hidden
    };
    let eff_attn_layers: u8 = if std::env::var("NUM_ATTN_LAYERS").is_ok() {
        let n = parse_num_attn_layers().map_err(|e| anyhow::anyhow!("NUM_ATTN_LAYERS: {e}"))?;
        eprintln!("[arch-knob] NUM_ATTN_LAYERS={n} (override, Wave-31 PR-B)");
        n as u8
    } else {
        args.attn_layers
    };
    // Wave 31 PR-B: validate GF16_ENABLED knob (default false, feature-gated).
    let _gf16_knob = resolve_gf16_knob()?;
    if _gf16_knob {
        eprintln!("[arch-knob] GF16_ENABLED=true (Wave-31 PR-B, feature=gf16)");
    }
    // `gf16_floor_every` is part of the RECIPE, not of the observation, so it
    // is banner-visible next to seed/steps and recorded in the sidecar.
    let gf16_every = gf16_floor_every() as usize;
    eprintln!(
        "=== trios-train seed={} steps={} hidden={} lr={:.4} attn_layers={} \
         eval_every={} gf16_floor_every={} ===",
        args.seed, args.steps, eff_hidden, args.lr, eff_attn_layers, args.eval_every, gf16_every
    );
    // EPIC-446: resolve run identity ONCE. The same string names the checkpoint
    // directory and the ledger row, so the artifact and the BPB cannot drift apart.
    let canon = resolve_canon_name(args.seed);

    let (train, train_synthetic) = load_data(&args.train_path)?;
    let (val, val_synthetic) = load_data(&args.val_path)?;
    let data_synthetic = train_synthetic || val_synthetic;
    eprintln!("train={} val={}", train.len(), val.len());
    assert_train_val_disjoint(&train, &val);
    // Hash the corpus once per run. This is what lets a third party confirm
    // the artifact and the number were produced from the same text.
    let corpus = crate::checkpoint::CorpusProvenance {
        train: crate::checkpoint::CorpusStream::describe(&args.train_path),
        val: crate::checkpoint::CorpusStream::describe(&args.val_path),
    };

    // #509 Phase-1b: wire QAT into the production `trios-train` path.
    // `scarab` spawns this binary and sets `TRIOS_FORMAT_TYPE`; previously
    // only `cpu_train` honoured it, so production traffic was F32 regardless.
    let fq_fmt = resolve_fake_quant_format();
    if let Some(fmt) = fq_fmt {
        eprintln!("QAT: FakeQuant enabled for format {:?}", fmt);
    }

    let mut model = HybridModel::new(eff_hidden, args.seed, eff_attn_layers);
    if let Some(fmt) = fq_fmt {
        fake_quantize_model(&mut model, fmt);
    }
    let d = model.attn.config().d_model;
    let dd = d * d;
    let attn_total = 8 * dd;
    let wd = 0.04f32;
    let mut opt_embed = AdamW::new(VOCAB * DIM, wd);
    let mut opt_ctx: Vec<AdamW> = (0..NUM_CTX).map(|_| AdamW::new(VOCAB * DIM, wd)).collect();
    let mut opt_proj = AdamW::new(eff_hidden * DIM, wd);
    let mut opt_attn_down = AdamW::new(d * eff_hidden, wd);
    let mut opt_attn_up = AdamW::new(eff_hidden * d, wd);
    let mut opt_head = AdamW::new(VOCAB * eff_hidden, wd);
    let mut opt_attn_w = AdamW::new(attn_total, wd);

    let init_bpb = evaluate(&model, &val)
        .ok_or_else(|| anyhow::anyhow!("initial eval produced no measurable chunk"))?;
    let init_bpb = guard_bpb(init_bpb, 0)?;
    eprintln!("Initial val_bpb={:.4}", init_bpb);
    let mut ema_bpb = init_bpb;
    // Raw readings, tracked separately from the EMA. `best_val_bpb` is the
    // minimum over the readings this run TOOK; `init_bpb` is excluded because
    // it describes the initialization, not the run.
    let mut best_val_bpb: Option<f32> = None;
    let mut final_val_bpb: Option<f32> = None;
    let warmup = args.steps / 10;
    let accum = 4;
    let mut rng_s = args.seed.wrapping_add(7919);
    let t0 = Instant::now();
    let gf16_floor_step = (0.7 * args.steps as f32).floor() as usize;
    let nca = NcaObjective::default();
    let mut last_nca_entropy = 0.0f64;

    for step in 1..=args.steps {
        let lr = cosine_lr(step, args.steps, args.lr, warmup);
        let mut ge = vec![0.0f32; VOCAB * DIM];
        let mut gc: Vec<Vec<f32>> = (0..NUM_CTX).map(|_| vec![0.0f32; VOCAB * DIM]).collect();
        let mut gp = vec![0.0f32; eff_hidden * DIM];
        let mut gh = vec![0.0f32; VOCAB * eff_hidden];
        let mut g_ad = vec![0.0f32; d * eff_hidden];
        let mut g_au = vec![0.0f32; eff_hidden * d];
        let mut g_aw = vec![0.0f32; 8 * dd];

        for _ in 0..accum {
            rng_s = rng_s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let dl = train.len();
            let ms = dl.saturating_sub(SEQ + 1);
            if ms == 0 {
                continue;
            }
            let cs = (rng_s as usize) % ms;
            let chunk = &train[cs..cs + SEQ + 1];
            let cnt = chunk.len().saturating_sub(NGRAM);
            if cnt == 0 {
                continue;
            }
            let ns = 8.min(cnt);
            let mut pos = Vec::with_capacity(ns);
            for _ in 0..ns {
                rng_s = rng_s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                pos.push((rng_s as usize) % cnt);
            }
            compute_grads(
                &model, chunk, &pos, &mut ge, &mut gc, &mut gp, &mut gh, &mut g_ad, &mut g_au,
                &mut g_aw,
            );
        }

        let tp = (accum * 8) as f32;
        for x in ge.iter_mut() {
            *x /= tp;
        }
        for g in gc.iter_mut() {
            for x in g.iter_mut() {
                *x /= tp;
            }
        }
        for x in gp.iter_mut() {
            *x /= tp;
        }
        for x in gh.iter_mut() {
            *x /= tp;
        }
        for x in g_ad.iter_mut() {
            *x /= tp;
        }
        for x in g_au.iter_mut() {
            *x /= tp;
        }
        for x in g_aw.iter_mut() {
            *x /= tp;
        }

        {
            rng_s = rng_s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let dl = train.len();
            let ms = dl.saturating_sub(SEQ + 1);
            if ms > 0 {
                let cs = (rng_s as usize) % ms;
                let chunk = &train[cs..cs + SEQ + 1];
                if chunk.len() > NGRAM {
                    let fc = model.forward_cached(chunk, 0);
                    let k = nca.k_states;
                    let h_min = fc.hidden.iter().cloned().fold(f32::MAX, f32::min);
                    let h_max = fc.hidden.iter().cloned().fold(f32::MIN, f32::max);
                    let range = (h_max - h_min).max(1e-6);
                    let nca_state: Vec<f32> = fc
                        .hidden
                        .iter()
                        .map(|&h| (((h - h_min) / range) * (k as f32 - 1.0)).round().max(0.0))
                        .collect();
                    let (nca_loss_val, nca_ent) = nca_entropy_loss(
                        &nca_state,
                        k,
                        nca.entropy_min,
                        nca.entropy_max,
                        nca.weight,
                    );
                    last_nca_entropy = nca_ent;
                    if nca_loss_val > 0.0 {
                        let scale = 1.0 + (nca_loss_val as f32).min(5.0);
                        for x in gp.iter_mut() {
                            *x *= scale;
                        }
                    }
                }
            }
        }

        opt_embed.update(&mut model.embed, &ge, lr);
        for (ci, oc) in opt_ctx.iter_mut().enumerate() {
            oc.update(&mut model.ctx[ci], &gc[ci], lr);
        }
        opt_proj.update(&mut model.proj, &gp, lr);
        opt_attn_down.update(&mut model.attn_down, &g_ad, lr);
        opt_attn_up.update(&mut model.attn_up, &g_au, lr);
        opt_head.update(&mut model.lm_head, &gh, lr);

        {
            let mut attn_flat = Vec::with_capacity(attn_total);
            attn_flat.extend_from_slice(&model.attn.wq);
            attn_flat.extend_from_slice(&model.attn.wk);
            attn_flat.extend_from_slice(&model.attn.wv);
            attn_flat.extend_from_slice(&model.attn.wo);
            attn_flat.extend_from_slice(&model.attn.wq2);
            attn_flat.extend_from_slice(&model.attn.wk2);
            attn_flat.extend_from_slice(&model.attn.wv2);
            attn_flat.extend_from_slice(&model.attn.wo2);
            opt_attn_w.update(&mut attn_flat, &g_aw, lr);
            model.attn.wq.copy_from_slice(&attn_flat[0..dd]);
            model.attn.wk.copy_from_slice(&attn_flat[dd..2 * dd]);
            model.attn.wv.copy_from_slice(&attn_flat[2 * dd..3 * dd]);
            model.attn.wo.copy_from_slice(&attn_flat[3 * dd..4 * dd]);
            model.attn.wq2.copy_from_slice(&attn_flat[4 * dd..5 * dd]);
            model.attn.wk2.copy_from_slice(&attn_flat[5 * dd..6 * dd]);
            model.attn.wv2.copy_from_slice(&attn_flat[6 * dd..7 * dd]);
            model.attn.wo2.copy_from_slice(&attn_flat[7 * dd..8 * dd]);
        }

        // #509 Phase-1b: STE fake-quantize after each optimizer step.
        if let Some(fmt) = fq_fmt {
            fake_quantize_model(&mut model, fmt);
        }

        // NOT `args.eval_every`: `gf16_floor` mutates the weights, so gating it
        // on the eval cadence made an observation parameter change the artifact.
        if gf16_enabled() && step >= gf16_floor_step && step % gf16_every == 0 {
            gf16_floor(&mut model.embed);
            gf16_floor(&mut model.proj);
            gf16_floor(&mut model.lm_head);
            for c in &mut model.ctx {
                gf16_floor(c);
            }
        }

        if step % args.eval_every == 0 || step == args.steps {
            let vbpb = evaluate(&model, &val)
                .ok_or_else(|| anyhow::anyhow!("eval produced no measurable chunk at step {step}"))?;
            let vbpb = guard_bpb(vbpb, step)?;
            ema_bpb = PHI_INV * ema_bpb + (1.0 - PHI_INV) * vbpb;
            best_val_bpb = Some(match best_val_bpb {
                Some(b) if b <= vbpb => b,
                _ => vbpb,
            });
            if step == args.steps {
                final_val_bpb = Some(vbpb);
            }
            println!(
                "seed={} step={} val_bpb={:.4} ema_bpb={:.4} best_val_bpb={:.4} nca_h={:.3} t={:.1}s",
                args.seed,
                step,
                vbpb,
                ema_bpb,
                best_val_bpb.unwrap_or(vbpb),
                last_nca_entropy,
                t0.elapsed().as_secs_f64()
            );
            // R5/L8: flush stdout immediately so seed-agent reads the JSONL
            // line. Without this the line stays in the BufWriter for the
            // child stdout pipe and the parent reader times out before EOF.
            // Refs: trios-railway#100, trios-trainer-igla#57.
            use std::io::Write as _;
            let _ = std::io::stdout().flush();

            // Bug A fix: write eval to Neon bpb_samples if TRIOS_CANON_NAME
            // is set (scarab sets this env var for the trainer subprocess).
            // EPIC-446: `canon` is resolved once at the top of the run so the
            // checkpoint directory and this row cannot name different runs.
            crate::neon_writer::bpb_sample(
                &canon,
                args.seed as i32,
                step as i32,
                vbpb,
                Some(ema_bpb as f32),
            );

            // EPIC-446: emit the artifact AFTER bpb_sample so the checkpoint
            // row carries the BPB these exact weights produced.
            if checkpoint_enabled() && (step == args.steps || checkpoint_every_hit(step)) {
                let meta = CheckpointMeta {
                    seed: args.seed,
                    step: step as u64,
                    train_lr: args.lr,
                    attn_scale: attn_scale(),
                    attn_seq: attn_seq_override() as u32,
                    gf16_enabled: gf16_enabled(),
                    data_synthetic,
                    optimizer: "adamw".into(),
                    fake_quant_format: fq_fmt
                        .map(|f| f.name().to_string())
                        .unwrap_or_else(|| "f32".into()),
                };
                let res = emit_checkpoint(
                    &model,
                    &canon,
                    &meta,
                    &corpus,
                    &RunKnobs {
                        steps_total: args.steps as u64,
                        gf16_floor_every: gf16_every as u64,
                        eval_every: args.eval_every as u64,
                    },
                    Some(vbpb as f64),
                    best_val_bpb.map(|v| v as f64),
                    Some(ema_bpb as f64),
                );
                // Asymmetric on purpose: a run that finishes with no artifact
                // is the failure this exists to prevent and must not exit 0,
                // but losing a whole run to a transient full disk mid-way
                // would be worse than a missing intermediate file.
                match res {
                    Ok(()) => {}
                    Err(e) if step == args.steps => {
                        return Err(e.context("final-step checkpoint failed"))
                    }
                    Err(e) => eprintln!(
                        "[ckpt] WARNING: periodic checkpoint at step {step} failed: {e:#}"
                    ),
                }
            }
        }
    }

    Ok(RunOutcome {
        final_val_bpb: final_val_bpb.map(|v| v as f64),
        best_val_bpb: best_val_bpb.map(|v| v as f64),
        ema_bpb: Some(ema_bpb as f64),
        // Compat mirror; NaN rather than a substituted number when the run
        // took no final measurement. See `RunOutcome::final_bpb`.
        final_bpb: final_val_bpb.map(|v| v as f64).unwrap_or(f64::NAN),
        steps_done: args.steps,
        seed: args.seed,
    })
}

pub fn run_single_muon(args: &TrainArgs, use_cwd: bool) -> Result<RunOutcome> {
    // Wave 31 PR-B: apply env-gated arch knobs.
    let eff_hidden = if std::env::var("HIDDEN_DIM").is_ok() {
        let h = parse_hidden_dim().map_err(|e| anyhow::anyhow!("HIDDEN_DIM: {e}"))?;
        eprintln!("[arch-knob] HIDDEN_DIM={h} (override, Wave-31 PR-B)");
        h
    } else {
        args.hidden
    };
    let eff_attn_layers: u8 = if std::env::var("NUM_ATTN_LAYERS").is_ok() {
        let n = parse_num_attn_layers().map_err(|e| anyhow::anyhow!("NUM_ATTN_LAYERS: {e}"))?;
        eprintln!("[arch-knob] NUM_ATTN_LAYERS={n} (override, Wave-31 PR-B)");
        n as u8
    } else {
        args.attn_layers
    };
    let _gf16_knob = resolve_gf16_knob()?;
    let label = if use_cwd { "MuonCwd" } else { "Muon" };
    let gf16_every = gf16_floor_every() as usize;
    eprintln!(
        "=== P1 {} seed={} steps={} hidden={} eval_every={} gf16_floor_every={} ===",
        label, args.seed, args.steps, eff_hidden, args.eval_every, gf16_every
    );
    // EPIC-446: see the note in `run_single` - one canon string for both the
    // checkpoint directory and the ledger row.
    let canon = resolve_canon_name(args.seed);

    let (train, train_synthetic) = load_data(&args.train_path)?;
    let (val, val_synthetic) = load_data(&args.val_path)?;
    let data_synthetic = train_synthetic || val_synthetic;
    assert_train_val_disjoint(&train, &val);
    // Hash the corpus once per run. This is what lets a third party confirm
    // the artifact and the number were produced from the same text.
    let corpus = crate::checkpoint::CorpusProvenance {
        train: crate::checkpoint::CorpusStream::describe(&args.train_path),
        val: crate::checkpoint::CorpusStream::describe(&args.val_path),
    };

    // #509 Phase-1b: same QAT wiring for the Muon path.
    let fq_fmt = resolve_fake_quant_format();
    if let Some(fmt) = fq_fmt {
        eprintln!("QAT: FakeQuant enabled for format {:?}", fmt);
    }

    let mut model = HybridModel::new(eff_hidden, args.seed, eff_attn_layers);
    if let Some(fmt) = fq_fmt {
        fake_quantize_model(&mut model, fmt);
    }
    let d = model.attn.config().d_model;
    let dd = d * d;
    let muon_lr = 0.0235f64;
    let muon_mom = 0.95f64;
    let muon_wd = 0.01f64;
    let adamw_wd = 0.04f32;
    let cwd_lambda = 0.01f64;

    let mut opt_embed = AdamW::new(VOCAB * DIM, adamw_wd);
    let mut opt_ctx: Vec<AdamW> = (0..NUM_CTX)
        .map(|_| AdamW::new(VOCAB * DIM, adamw_wd))
        .collect();
    let mut opt_proj_muon = crate::optimizer::MuonOptimizer::with_matrix_shape(
        eff_hidden * DIM,
        eff_hidden,
        DIM,
        muon_lr,
        muon_mom,
        muon_wd,
    );
    opt_proj_muon.ns_steps = if use_cwd { 3 } else { 1 };
    let mut opt_attn_down = AdamW::new(d * eff_hidden, adamw_wd);
    let mut opt_attn_up = AdamW::new(eff_hidden * d, adamw_wd);
    let mut opt_head = AdamW::new(VOCAB * eff_hidden, adamw_wd);
    let mut opt_attn_w_muon = AdamW::new(8 * dd, adamw_wd);
    let _cwd_lambda = cwd_lambda;

    let init_bpb = evaluate(&model, &val)
        .ok_or_else(|| anyhow::anyhow!("initial eval produced no measurable chunk"))?;
    let init_bpb = guard_bpb(init_bpb, 0)?;
    eprintln!("Initial val_bpb={:.4}", init_bpb);
    let mut ema_bpb = init_bpb;
    // See `run_single`: raw readings are tracked separately from the EMA.
    let mut best_val_bpb: Option<f32> = None;
    let mut final_val_bpb: Option<f32> = None;
    let warmup = args.steps / 10;
    let accum = 4;
    let mut rng_s = args.seed.wrapping_add(7919);
    let t0 = Instant::now();
    let gf16_floor_step = (0.7 * args.steps as f32).floor() as usize;
    let nca = NcaObjective::default();
    let mut last_nca_entropy = 0.0f64;

    for step in 1..=args.steps {
        let lr = cosine_lr(step, args.steps, args.lr, warmup);
        let mut ge = vec![0.0f32; VOCAB * DIM];
        let mut gc: Vec<Vec<f32>> = (0..NUM_CTX).map(|_| vec![0.0f32; VOCAB * DIM]).collect();
        let mut gp = vec![0.0f32; eff_hidden * DIM];
        let mut gh = vec![0.0f32; VOCAB * eff_hidden];
        let mut g_ad = vec![0.0f32; d * eff_hidden];
        let mut g_au = vec![0.0f32; eff_hidden * d];
        let mut g_aw = vec![0.0f32; 8 * dd];

        for _ in 0..accum {
            rng_s = rng_s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let dl = train.len();
            let ms = dl.saturating_sub(SEQ + 1);
            if ms == 0 {
                continue;
            }
            let cs = (rng_s as usize) % ms;
            let chunk = &train[cs..cs + SEQ + 1];
            let cnt = chunk.len().saturating_sub(NGRAM);
            if cnt == 0 {
                continue;
            }
            let ns = 8.min(cnt);
            let mut pos = Vec::with_capacity(ns);
            for _ in 0..ns {
                rng_s = rng_s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                pos.push((rng_s as usize) % cnt);
            }
            compute_grads(
                &model, chunk, &pos, &mut ge, &mut gc, &mut gp, &mut gh, &mut g_ad, &mut g_au,
                &mut g_aw,
            );
        }

        let tp = (accum * 8) as f32;
        for x in ge.iter_mut() {
            *x /= tp;
        }
        for g in gc.iter_mut() {
            for x in g.iter_mut() {
                *x /= tp;
            }
        }
        for x in gp.iter_mut() {
            *x /= tp;
        }
        for x in gh.iter_mut() {
            *x /= tp;
        }
        for x in g_ad.iter_mut() {
            *x /= tp;
        }
        for x in g_au.iter_mut() {
            *x /= tp;
        }
        for x in g_aw.iter_mut() {
            *x /= tp;
        }

        {
            rng_s = rng_s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let dl = train.len();
            let ms = dl.saturating_sub(SEQ + 1);
            if ms > 0 {
                let cs = (rng_s as usize) % ms;
                let chunk = &train[cs..cs + SEQ + 1];
                if chunk.len() > NGRAM {
                    let fc = model.forward_cached(chunk, 0);
                    let k = nca.k_states;
                    let h_min = fc.hidden.iter().cloned().fold(f32::MAX, f32::min);
                    let h_max = fc.hidden.iter().cloned().fold(f32::MIN, f32::max);
                    let range = (h_max - h_min).max(1e-6);
                    let nca_state: Vec<f32> = fc
                        .hidden
                        .iter()
                        .map(|&h| (((h - h_min) / range) * (k as f32 - 1.0)).round().max(0.0))
                        .collect();
                    let (nca_loss_val, nca_ent) = nca_entropy_loss(
                        &nca_state,
                        k,
                        nca.entropy_min,
                        nca.entropy_max,
                        nca.weight,
                    );
                    last_nca_entropy = nca_ent;
                    if nca_loss_val > 0.0 {
                        let scale = 1.0 + (nca_loss_val as f32).min(5.0);
                        for x in gp.iter_mut() {
                            *x *= scale;
                        }
                    }
                }
            }
        }

        opt_embed.update(&mut model.embed, &ge, lr);
        for (ci, oc) in opt_ctx.iter_mut().enumerate() {
            oc.update(&mut model.ctx[ci], &gc[ci], lr);
        }
        opt_proj_muon.step(&mut model.proj, &gp);
        opt_attn_down.update(&mut model.attn_down, &g_ad, lr);
        opt_attn_up.update(&mut model.attn_up, &g_au, lr);
        opt_head.update(&mut model.lm_head, &gh, lr);

        {
            let mut attn_flat = Vec::with_capacity(8 * dd);
            attn_flat.extend_from_slice(&model.attn.wq);
            attn_flat.extend_from_slice(&model.attn.wk);
            attn_flat.extend_from_slice(&model.attn.wv);
            attn_flat.extend_from_slice(&model.attn.wo);
            attn_flat.extend_from_slice(&model.attn.wq2);
            attn_flat.extend_from_slice(&model.attn.wk2);
            attn_flat.extend_from_slice(&model.attn.wv2);
            attn_flat.extend_from_slice(&model.attn.wo2);
            opt_attn_w_muon.update(&mut attn_flat, &g_aw, lr);
            model.attn.wq.copy_from_slice(&attn_flat[0..dd]);
            model.attn.wk.copy_from_slice(&attn_flat[dd..2 * dd]);
            model.attn.wv.copy_from_slice(&attn_flat[2 * dd..3 * dd]);
            model.attn.wo.copy_from_slice(&attn_flat[3 * dd..4 * dd]);
            model.attn.wq2.copy_from_slice(&attn_flat[4 * dd..5 * dd]);
            model.attn.wk2.copy_from_slice(&attn_flat[5 * dd..6 * dd]);
            model.attn.wv2.copy_from_slice(&attn_flat[6 * dd..7 * dd]);
            model.attn.wo2.copy_from_slice(&attn_flat[7 * dd..8 * dd]);
        }

        // #509 Phase-1b: STE fake-quantize after each optimizer step (Muon path).
        if let Some(fmt) = fq_fmt {
            fake_quantize_model(&mut model, fmt);
        }

        // See `run_single`: the floor cadence is a recipe knob, not `eval_every`.
        if gf16_enabled() && step >= gf16_floor_step && step % gf16_every == 0 {
            gf16_floor(&mut model.embed);
            gf16_floor(&mut model.proj);
            gf16_floor(&mut model.lm_head);
            for c in &mut model.ctx {
                gf16_floor(c);
            }
        }

        if step % args.eval_every == 0 || step == args.steps {
            let vbpb = evaluate(&model, &val)
                .ok_or_else(|| anyhow::anyhow!("eval produced no measurable chunk at step {step}"))?;
            let vbpb = guard_bpb(vbpb, step)?;
            ema_bpb = PHI_INV * ema_bpb + (1.0 - PHI_INV) * vbpb;
            best_val_bpb = Some(match best_val_bpb {
                Some(b) if b <= vbpb => b,
                _ => vbpb,
            });
            if step == args.steps {
                final_val_bpb = Some(vbpb);
            }
            println!(
                "{} seed={} step={} val_bpb={:.4} ema_bpb={:.4} best_val_bpb={:.4} nca_h={:.3} t={:.1}s",
                label,
                args.seed,
                step,
                vbpb,
                ema_bpb,
                best_val_bpb.unwrap_or(vbpb),
                last_nca_entropy,
                t0.elapsed().as_secs_f64()
            );
            // R5/L8: flush stdout immediately. See note in run_single().
            use std::io::Write as _;
            let _ = std::io::stdout().flush();

            // Bug A fix: write eval to Neon bpb_samples if TRIOS_CANON_NAME
            // is set (scarab sets this env var for the trainer subprocess).
            // EPIC-446: `canon` is resolved once at the top of the run, from
            // TRIOS_CANON_NAME -> CANON_NAME -> a deterministic seed fallback,
            // so direct trios-train invocations still produce telemetry and the
            // checkpoint directory cannot name a different run than this row.
            crate::neon_writer::bpb_sample(
                &canon,
                args.seed as i32,
                step as i32,
                vbpb,
                Some(ema_bpb as f32),
            );

            // EPIC-446: same emission point as `run_single`. The optimizer
            // label comes from which entry point ran, not from canon_name.
            if checkpoint_enabled() && (step == args.steps || checkpoint_every_hit(step)) {
                let meta = CheckpointMeta {
                    seed: args.seed,
                    step: step as u64,
                    train_lr: args.lr,
                    attn_scale: attn_scale(),
                    attn_seq: attn_seq_override() as u32,
                    gf16_enabled: gf16_enabled(),
                    data_synthetic,
                    optimizer: if use_cwd { "muon-cwd".into() } else { "muon".into() },
                    fake_quant_format: fq_fmt
                        .map(|f| f.name().to_string())
                        .unwrap_or_else(|| "f32".into()),
                };
                let res = emit_checkpoint(
                    &model,
                    &canon,
                    &meta,
                    &corpus,
                    &RunKnobs {
                        steps_total: args.steps as u64,
                        gf16_floor_every: gf16_every as u64,
                        eval_every: args.eval_every as u64,
                    },
                    Some(vbpb as f64),
                    best_val_bpb.map(|v| v as f64),
                    Some(ema_bpb as f64),
                );
                match res {
                    Ok(()) => {}
                    Err(e) if step == args.steps => {
                        return Err(e.context("final-step checkpoint failed"))
                    }
                    Err(e) => eprintln!(
                        "[ckpt] WARNING: periodic checkpoint at step {step} failed: {e:#}"
                    ),
                }
            }
        }
    }

    Ok(RunOutcome {
        final_val_bpb: final_val_bpb.map(|v| v as f64),
        best_val_bpb: best_val_bpb.map(|v| v as f64),
        ema_bpb: Some(ema_bpb as f64),
        // Compat mirror; NaN rather than a substituted number when the run
        // took no final measurement. See `RunOutcome::final_bpb`.
        final_bpb: final_val_bpb.map(|v| v as f64).unwrap_or(f64::NAN),
        steps_done: args.steps,
        seed: args.seed,
    })
}

pub fn run_sweep(
    steps: usize,
    hidden: usize,
    lr: f32,
    attn_layers: u8,
    eval_every: usize,
    train_path: &str,
    val_path: &str,
) -> Result<Vec<RunOutcome>> {
    let mut results = Vec::new();
    for &seed in GATE_FINAL_SEEDS {
        results.push(run_single(&TrainArgs {
            seed,
            steps,
            hidden,
            lr,
            attn_layers,
            eval_every,
            train_path: train_path.to_string(),
            val_path: val_path.to_string(),
        })?);
    }
    Ok(results)
}

pub fn run(cfg: &crate::TrainConfig) -> Result<RunOutcome> {
    let args = TrainArgs {
        seed: cfg.seed,
        steps: cfg.steps,
        hidden: 828,
        lr: cfg.optimizer.lr as f32,
        attn_layers: if cfg.model.hybrid_attn { 2 } else { 1 },
        eval_every: 1000,
        train_path: cfg.data.train_path.clone(),
        val_path: cfg.data.val_path.clone(),
    };
    let outcome = run_single(&args)?;
    // The ledger row carries the MEASURED final val_bpb. It used to carry
    // `best_bpb`, the running minimum of the EMA, which is not a measurement of
    // anything. A run with no final measurement emits no row rather than a
    // substituted one.
    if !cfg.ledger.jsonl_path.is_empty() {
        if let Some(bpb) = outcome.final_val_bpb {
            let _ = crate::ledger::emit_row(cfg, bpb, outcome.steps_done);
        } else {
            eprintln!(
                "[ledger] no final val_bpb was measured; refusing to emit a Gate-2 row"
            );
        }
    }
    Ok(outcome)
}

#[cfg(test)]
mod fake_quant_wiring_tests {
    //! Phase-1b regression tests for trios#509 - verify that the
    //! `trios-train` path now honours `TRIOS_FORMAT_TYPE` end-to-end.
    use super::*;
    use std::sync::Mutex;

    // Env-var tests must be serialised - `std::env` is process-global.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn resolves_fp16_from_trios_format_type() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_FAKE_QUANT_FORMAT");
        std::env::set_var("TRIOS_FORMAT_TYPE", "fp16");
        let fmt = resolve_fake_quant_format();
        std::env::remove_var("TRIOS_FORMAT_TYPE");
        assert_eq!(fmt, Some(FormatKind::Fp16));
    }

    #[test]
    fn resolves_gf16_alias() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_FORMAT_TYPE");
        std::env::set_var("TRIOS_FAKE_QUANT_FORMAT", "gf16");
        let fmt = resolve_fake_quant_format();
        std::env::remove_var("TRIOS_FAKE_QUANT_FORMAT");
        assert_eq!(fmt, Some(FormatKind::Gf16));
    }

    #[test]
    fn f32_resolves_to_none() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_FAKE_QUANT_FORMAT");
        std::env::set_var("TRIOS_FORMAT_TYPE", "f32");
        let fmt = resolve_fake_quant_format();
        std::env::remove_var("TRIOS_FORMAT_TYPE");
        assert_eq!(fmt, None);
    }

    #[test]
    fn unset_resolves_to_none() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_FORMAT_TYPE");
        std::env::remove_var("TRIOS_FAKE_QUANT_FORMAT");
        assert_eq!(resolve_fake_quant_format(), None);
    }

    #[test]
    fn unknown_format_resolves_to_none() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_FAKE_QUANT_FORMAT");
        std::env::set_var("TRIOS_FORMAT_TYPE", "imaginary_float");
        let fmt = resolve_fake_quant_format();
        std::env::remove_var("TRIOS_FORMAT_TYPE");
        assert_eq!(fmt, None);
    }

    #[test]
    fn fake_quantize_model_actually_changes_weights() {
        let _g = ENV_LOCK.lock().unwrap();
        let mut model = HybridModel::new(64, 1597, 2);
        let embed_before = model.embed.clone();
        let lm_head_before = model.lm_head.clone();
        fake_quantize_model(&mut model, FormatKind::Fp16);
        // At least one weight must change after fp16 round-trip.
        let embed_changed = model
            .embed
            .iter()
            .zip(embed_before.iter())
            .any(|(a, b)| a != b);
        let head_changed = model
            .lm_head
            .iter()
            .zip(lm_head_before.iter())
            .any(|(a, b)| a != b);
        assert!(
            embed_changed || head_changed,
            "fake_quantize_model(Fp16) must change at least one weight"
        );
    }

    #[test]
    fn fake_quantize_model_f32_is_noop() {
        let _g = ENV_LOCK.lock().unwrap();
        let mut model = HybridModel::new(64, 1597, 2);
        let embed_before = model.embed.clone();
        fake_quantize_model(&mut model, FormatKind::F32);
        assert_eq!(model.embed, embed_before);
    }
}

#[cfg(test)]
mod checkpoint_codec_tests {
    //! EPIC-446 evidence: a training run must produce a checkpoint file on
    //! disk whose content hash is recorded against that run's identity, and
    //! the checkpoint must load back bit-identically.
    //!
    //! These live here rather than in `tests/` because `HybridModel` is
    //! private to this module and `tests/` only sees `pub` items. Widening
    //! visibility to allow an integration test would be a strictly larger diff
    //! for the same evidence.
    use super::*;
    use crate::checkpoint::{
        self, sha256_hex, CHECKPOINT_HEADER_LEN, CHECKPOINT_PAYLOAD_OFFSET,
    };
    use std::sync::Mutex;

    // `TRIOS_CHECKPOINT_DIR` is process-global, so every test that touches it
    // must be serialised. `pub(super)` so `measurement_truth_tests` shares this
    // exact lock: two separate mutexes would not serialise anything.
    pub(super) static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn meta_for(step: u64) -> CheckpointMeta {
        CheckpointMeta {
            seed: 47,
            step,
            train_lr: 0.003,
            attn_scale: 0.1,
            attn_seq: 8,
            gf16_enabled: true,
            data_synthetic: false,
            optimizer: "adamw".to_string(),
            fake_quant_format: "f32".to_string(),
        }
    }

    /// A model with adversarial float bit patterns planted in it: a NaN with a
    /// non-canonical payload, a negative zero, and both infinities. These are
    /// exactly the values a `==`-based comparison would wave through.
    fn poisoned_model(hidden: usize) -> HybridModel {
        let mut m = HybridModel::new(hidden, 1597, 2);
        m.embed[0] = f32::from_bits(0x7fc0_1234); // quiet NaN, custom payload
        m.embed[1] = -0.0;
        m.embed[2] = 0.0;
        m.proj[0] = f32::INFINITY;
        m.proj[1] = f32::NEG_INFINITY;
        m.lm_head[0] = f32::from_bits(0x0000_0001); // smallest subnormal
        m.ctx[3][7] = f32::from_bits(0xffff_ffff); // negative NaN, all-ones
        m.attn.wq[0] = -0.0;
        m.attn.wo2[5] = f32::from_bits(0x8000_0000); // negative zero again
        m
    }

    fn assert_tensors_bit_identical(a: &HybridModel, b: &HybridModel) {
        let ta = a.checkpoint_tensors();
        let tb = b.checkpoint_tensors();
        assert_eq!(ta.len(), tb.len(), "tensor count differs");
        for (i, (x, y)) in ta.iter().zip(tb.iter()).enumerate() {
            assert_eq!(x.len(), y.len(), "tensor {i} length differs");
            for (j, (p, q)) in x.iter().zip(y.iter()).enumerate() {
                // `to_bits`, never `==`: f32 `==` is false for NaN and true
                // across +0.0/-0.0, so an `==`-based check would pass
                // vacuously on ordinary data and miss exactly the bit-mangling
                // class of bug this test exists to catch.
                assert_eq!(
                    p.to_bits(),
                    q.to_bits(),
                    "tensor {i} element {j}: {:#010x} != {:#010x}",
                    p.to_bits(),
                    q.to_bits()
                );
            }
        }
        assert_eq!(a.hidden, b.hidden);
        assert_eq!(a.attn.config().d_model, b.attn.config().d_model);
        assert_eq!(a.attn.config().num_heads, b.attn.config().num_heads);
        assert_eq!(a.attn.config().seq_len, b.attn.config().seq_len);
        assert_eq!(
            a.attn.config().num_attn_layers,
            b.attn.config().num_attn_layers
        );
        assert_eq!(
            a.attn.config().qk_gain.to_bits(),
            b.attn.config().qk_gain.to_bits()
        );
        assert_eq!(a.attn.config().lr.to_bits(), b.attn.config().lr.to_bits());
    }

    #[test]
    fn round_trip_through_disk_is_bit_identical_and_hash_matches() {
        let _g = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("TRIOS_CHECKPOINT_DIR", dir.path());

        let model = poisoned_model(32);
        let meta = meta_for(1234);
        let bytes = model.to_checkpoint_bytes(&meta).unwrap();

        let saved = checkpoint::save("round-trip-run", 1234, &bytes).unwrap();
        let loaded = checkpoint::load("round-trip-run", 1234).unwrap();

        // 1. Byte-for-byte equality through the filesystem.
        assert_eq!(loaded, bytes, "bytes changed on the way through disk");

        // 2. The digest is over the file as it exists on disk, and the
        //    reported size is the real size.
        assert_eq!(sha256_hex(&loaded), saved.sha256);
        let on_disk_len = std::fs::metadata(&saved.path).unwrap().len();
        assert_eq!(saved.bytes, on_disk_len);
        assert_eq!(saved.sha256.len(), 64);
        assert!(saved.sha256.bytes().all(|b| b.is_ascii_hexdigit()
            && !b.is_ascii_uppercase()));

        // 3. Every tensor survives bit-identically.
        let (restored, restored_meta) = HybridModel::from_checkpoint_bytes(&loaded).unwrap();
        assert_tensors_bit_identical(&model, &restored);

        // 4. The metadata survives too.
        assert_eq!(restored_meta, meta);

        std::env::remove_var("TRIOS_CHECKPOINT_DIR");
    }

    #[test]
    fn sha256_hex_matches_known_vectors() {
        // Independent of our code: these are the published SHA-256 digests of
        // the empty string and "abc", so a wrong hasher cannot pass.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn file_length_is_exactly_the_documented_formula() {
        let model = HybridModel::new(384, 47, 2);
        let bytes = model.to_checkpoint_bytes(&meta_for(81000)).unwrap();
        // VOCAB=128, DIM=64, NUM_CTX=6, d_model=64, hidden=384:
        //   7*8192 + 384*64 + 64*384 + 384*64 + 128*384 + 8*4096 = 212_992 f32
        let total: usize = 7 * 128 * 64 + 384 * 64 + 64 * 384 + 384 * 64 + 128 * 384 + 8 * 64 * 64;
        assert_eq!(total, 212_992);
        assert_eq!(bytes.len(), CHECKPOINT_PAYLOAD_OFFSET + 4 * total);
        assert_eq!(bytes.len(), 852_272);
        assert_eq!(CHECKPOINT_PAYLOAD_OFFSET, 304);
        assert_eq!(CHECKPOINT_HEADER_LEN, 152);
        assert_eq!(&bytes[0..8], b"TRIOSCKP");
    }

    // -- negative cases: every one must be Err, never a partial load ---------

    /// `unwrap_err()` would require `HybridModel: Debug`; this keeps the
    /// production type unchanged.
    fn expect_err(bytes: &[u8]) -> String {
        match HybridModel::from_checkpoint_bytes(bytes) {
            Ok(_) => panic!("expected a rejection, got a loaded model"),
            Err(e) => format!("{e:#}"),
        }
    }

    fn good_bytes() -> Vec<u8> {
        HybridModel::new(16, 47, 2)
            .to_checkpoint_bytes(&meta_for(7))
            .unwrap()
    }

    #[test]
    fn rejects_flipped_magic_byte() {
        let mut b = good_bytes();
        b[3] ^= 0x01;
        let err = expect_err(&b);
        assert!(err.contains("magic"), "{err}");
    }

    #[test]
    fn rejects_unknown_format_version() {
        let mut b = good_bytes();
        b[8..12].copy_from_slice(&2u32.to_le_bytes());
        let err = expect_err(&b);
        assert!(err.contains("format_version"), "{err}");
    }

    #[test]
    fn rejects_truncation_by_one_byte() {
        let mut b = good_bytes();
        b.pop();
        let err = expect_err(&b);
        assert!(err.contains("length"), "{err}");
    }

    #[test]
    fn rejects_altered_directory_count() {
        let mut b = good_bytes();
        // Entry 7 is `proj`; claim one element fewer.
        let off = CHECKPOINT_HEADER_LEN + 8 * 7;
        let old = u64::from_le_bytes(b[off..off + 8].try_into().unwrap());
        b[off..off + 8].copy_from_slice(&(old - 1).to_le_bytes());
        let err = expect_err(&b);
        assert!(err.contains("directory entry 7"), "{err}");
    }

    #[test]
    fn rejects_nonzero_reserved_byte() {
        let mut b = good_bytes();
        b[127] = 1;
        let err = expect_err(&b);
        assert!(err.contains("reserved"), "{err}");
    }

    #[test]
    fn rejects_header_len_mismatch() {
        let mut b = good_bytes();
        b[12..16].copy_from_slice(&160u32.to_le_bytes());
        let err = expect_err(&b);
        assert!(err.contains("header_len"), "{err}");
    }

    #[test]
    fn rejects_qk_gain_outside_phi() {
        let mut b = good_bytes();
        b[56..64].copy_from_slice(&1.0f64.to_le_bytes());
        let err = expect_err(&b);
        assert!(err.contains("validate"), "{err}");
    }

    #[test]
    fn rejects_num_attn_layers_out_of_range() {
        let mut b = good_bytes();
        b[44..48].copy_from_slice(&3u32.to_le_bytes());
        let err = expect_err(&b);
        assert!(err.contains("num_attn_layers"), "{err}");
    }

    #[test]
    fn rejects_nul_terminated_rather_than_nul_padded_optimizer() {
        let mut b = good_bytes();
        // "adamw\0\0\0" -> "adamw\0x\0": a non-NUL byte after the first NUL.
        b[128 + 6] = b'x';
        let err = expect_err(&b);
        assert!(err.contains("NUL-padded"), "{err}");
    }

    #[test]
    fn rejects_empty_and_short_buffers() {
        expect_err(&[]);
        expect_err(&[0u8; 303]);
        expect_err(b"TRIOSCKP");
    }

    #[test]
    fn sanitize_run_name_cannot_escape_the_checkpoint_dir() {
        // Separators become `_`; the result is a single, harmless component.
        assert_eq!(checkpoint::sanitize_run_name("../../etc"), ".._.._etc");
        assert_eq!(checkpoint::sanitize_run_name("a/b"), "a_b");
        assert_eq!(checkpoint::sanitize_run_name(".."), "_");
        assert_eq!(checkpoint::sanitize_run_name(""), "_");
        assert_eq!(
            checkpoint::sanitize_run_name("IGLA-gf16-adamw-h384"),
            "IGLA-gf16-adamw-h384"
        );
    }

    /// End-to-end: a real (tiny) `run_single` must leave an artifact on disk,
    /// a sidecar naming it, and a checkpoint that loads back.
    ///
    /// This is also where the layer-2 claim is asserted rather than stated:
    /// `HybridAttn::with_config` zero-fills all eight weight blocks and
    /// `HybridModel::new` randomizes only `wq/wk/wv/wo`, so with `wo2 = 0` the
    /// layer-2 gradients are identically zero at init and weight decay
    /// (`wd * lr * 0`) cannot break the symmetry. If that inspection is right,
    /// `wq2/wk2/wv2/wo2` are still exactly zero after training. This belongs
    /// in a test, NOT in the ledger: it is a code-inspection result.
    #[test]
    fn run_single_emits_a_loadable_artifact_and_freezes_layer_two() {
        let _g = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let train_path = data.join("train.txt");
        let val_path = data.join("val.txt");
        // Distinct corpora so `assert_train_val_disjoint` is a real check and
        // `data_synthetic` stays false. The streams must also be non-periodic:
        // the previous fixture repeated a 36-byte string, which yields 36
        // distinct 8-grams and is exactly the degenerate eval corpus the
        // entropy precondition now rejects (#62). Deterministic LCG bytes keep
        // the test hermetic while looking like real text to the guard.
        let gen = |seed: u64, len: usize| -> Vec<u8> {
            let mut s = seed;
            (0..len)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    b' ' + ((s >> 33) % 95) as u8
                })
                .collect()
        };
        // val must clear `MIN_VAL_TOKENS` and yield >= `MIN_EVAL_CHUNKS`: the
        // old 2880-byte fixture was itself too small to support a `val_bpb`,
        // which is the defect the guard now refuses.
        std::fs::write(&train_path, gen(0xA1CE, 40000)).unwrap();
        std::fs::write(&val_path, gen(0xB0BA, 12000)).unwrap();

        std::env::set_var("TRIOS_CHECKPOINT_DIR", dir.path().join("ckpt"));
        std::env::set_var("TRIOS_CANON_NAME", "IGLA-test/canon");
        // Never touch a real database from a unit test. The developer shell
        // commonly has DATABASE_URL pointed at an unrelated Postgres.
        for k in [
            "DATABASE_URL",
            "NEON_DATABASE_URL",
            "TRIOS_NEON_DSN",
            "TRIOS_DATABASE_URL",
            "TRIOS_CHECKPOINT_DISABLE",
            "TRIOS_CHECKPOINT_EVERY",
            "HIDDEN_DIM",
            "NUM_ATTN_LAYERS",
            "GF16_ENABLED",
        ] {
            std::env::remove_var(k);
        }

        let steps = 4usize;
        let outcome = run_single(&TrainArgs {
            seed: 47,
            steps,
            hidden: 16,
            lr: 0.003,
            attn_layers: 2,
            eval_every: steps,
            train_path: train_path.to_string_lossy().into_owned(),
            val_path: val_path.to_string_lossy().into_owned(),
        })
        .expect("run_single");
        assert_eq!(outcome.steps_done, steps);

        // `TRIOS_CANON_NAME` contains a `/`; the directory component must be
        // sanitized while the ledger identity keeps the original.
        let run_dir = dir.path().join("ckpt").join("IGLA-test_canon");
        let bin = run_dir.join(format!("{steps}.bin"));
        let json = run_dir.join(format!("{steps}.json"));
        assert!(bin.is_file(), "no checkpoint at {bin:?}");
        assert!(json.is_file(), "no sidecar at {json:?}");
        // No interrupted-write leftovers.
        for e in std::fs::read_dir(&run_dir).unwrap() {
            let name = e.unwrap().file_name().to_string_lossy().into_owned();
            assert!(!name.contains(".tmp."), "leftover temp file {name}");
        }

        let raw = std::fs::read(&bin).unwrap();
        let rec: checkpoint::CheckpointRecord =
            serde_json::from_slice(&std::fs::read(&json).unwrap()).unwrap();
        assert_eq!(rec.schema, "trios-checkpoint-record/4");
        // Schema 2: what changed the artifact is now IN the artifact.
        assert_eq!(rec.steps_total, steps as u64);
        assert_eq!(rec.gf16_floor_every, gf16_floor_every());
        assert_eq!(rec.eval_every, steps as u64);
        assert!(!rec.git_provenance.is_empty(), "provenance strength unrecorded");
        // Schema 3: the first-order inputs the record used to omit. `lr` must
        // be the value that ran, never a defaulted 0.0 - which is itself a
        // legal learning rate and so unusable as a sentinel.
        assert_eq!(rec.lr, Some(0.003f32 as f64), "lr not recorded as the f32 that ran");
        assert_eq!(rec.attn_scale, attn_scale() as f64);
        assert_eq!(rec.attn_seq, attn_seq_override() as u64);
        assert_eq!(rec.platform.os, std::env::consts::OS);
        assert_eq!(rec.platform.arch, std::env::consts::ARCH);
        assert_eq!(rec.platform.pointer_width, usize::BITS);
        assert!(!rec.platform.libc.is_empty(), "libc field left blank, not 'undetermined'");
        assert!(
            !rec.platform.toolchain_provenance.is_empty(),
            "toolchain strength unrecorded"
        );
        // The tests run from the crate root, so the tree IS reachable and the
        // sentinel must not appear.
        assert_eq!(rec.source_sha256.len(), 64, "source digest is {}", rec.source_sha256);
        assert!(rec.source_sha256.chars().all(|c| c.is_ascii_hexdigit()));
        // Schema 4: the record names the EXECUTOR and the ALPHABET. Without the
        // first, `ckpt_replay` had no hash to check the binary it was about to
        // run against, and a `/bin/sh` stub graded VERIFIED. Without the second,
        // "bits-per-byte" named a unit the fold does not deliver off ASCII.
        assert_eq!(
            rec.trainer.provenance,
            checkpoint::TRAINER_PROVENANCE_SELF_HASHED,
            "the trainer did not hash itself: {}",
            rec.trainer.provenance
        );
        assert_eq!(rec.trainer.sha256.len(), 64, "trainer digest is {}", rec.trainer.sha256);
        assert!(rec.trainer.sha256.chars().all(|c| c.is_ascii_hexdigit()));
        // Re-derivable by an outside party with `shasum -a 256`, which is the
        // whole point: the digest is over the bytes on disk, not an image.
        let exe = std::fs::read(&rec.trainer.path).expect("recorded trainer path is readable");
        assert_eq!(rec.trainer.sha256, sha256_hex(&exe), "trainer hash is not the file's");
        assert_eq!(rec.vocab, VOCAB as u32, "the record must state its own alphabet");
        // The measured reading, not the EMA, is what the sidecar carries.
        assert_eq!(rec.final_val_bpb, outcome.final_val_bpb);
        assert_eq!(rec.best_val_bpb, outcome.best_val_bpb);
        assert_eq!(rec.canon_name, "IGLA-test/canon", "ledger identity unsanitized");
        assert_eq!(rec.sha256, sha256_hex(&raw));
        assert_eq!(rec.bytes, raw.len() as u64);
        assert_eq!(rec.step, steps as i64);
        assert_eq!(rec.seed, 47);
        assert_eq!(rec.optimizer, "adamw");
        assert!(!rec.data_synthetic);
        // No DSN was reachable, so the sidecar must say so rather than claim
        // a write happened.
        assert_ne!(rec.ledger, "pending", "sidecar was never finalized");
        assert_ne!(rec.ledger, "written");

        let (model, meta) = HybridModel::from_checkpoint_bytes(&raw).unwrap();
        assert_eq!(meta.step, steps as u64);
        assert_eq!(meta.seed, 47);
        assert!(!meta.data_synthetic);
        assert_eq!(model.hidden, 16);
        assert_eq!(model.attn.config().num_attn_layers, 2);
        // Layer 1 actually trained.
        assert!(
            model.attn.wq.iter().any(|v| *v != 0.0),
            "layer 1 wq is all zero - the run did not train"
        );
        // Layer 2 is a permanent no-op in the default 2-layer config.
        for (name, w) in [
            ("wq2", &model.attn.wq2),
            ("wk2", &model.attn.wk2),
            ("wv2", &model.attn.wv2),
            ("wo2", &model.attn.wo2),
        ] {
            assert!(
                w.iter().all(|v| v.to_bits() == 0),
                "{name} is not all-zero after training; the layer-2 freeze \
                 inspection is wrong and half the attention payload is live"
            );
        }

        std::env::remove_var("TRIOS_CHECKPOINT_DIR");
        std::env::remove_var("TRIOS_CANON_NAME");
    }

    /// `load_data` must refuse to substitute the synthetic corpus unless the
    /// caller opted in. This is the regression that produced the
    /// leak-tainted BPB rows.
    #[test]
    fn load_data_refuses_synthetic_substitution_by_default() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_ALLOW_SYNTHETIC_DATA");
        let err = load_data("/nonexistent/corpus-that-does-not-exist.bin")
            .map(|_| ())
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("Refusing the synthetic fallback"), "{msg}");

        std::env::set_var("TRIOS_ALLOW_SYNTHETIC_DATA", "1");
        let (tokens, synthetic) =
            load_data("/nonexistent/corpus-that-does-not-exist.bin").unwrap();
        std::env::remove_var("TRIOS_ALLOW_SYNTHETIC_DATA");
        assert!(synthetic, "the opt-in path must report itself as synthetic");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn only_exact_step_bin_names_are_checkpoints() {
        use std::path::Path;
        assert!(checkpoint::is_checkpoint_file(Path::new("c/81000.bin")));
        assert!(!checkpoint::is_checkpoint_file(Path::new(
            "c/81000.bin.tmp.4242"
        )));
        assert!(!checkpoint::is_checkpoint_file(Path::new("c/81000.json")));
        assert!(!checkpoint::is_checkpoint_file(Path::new("c/final.bin")));
    }
}

#[cfg(test)]
mod measurement_truth_tests {
    //! The measurement path must report what it measured.
    //!
    //! Three defects, all previously live and all cited as evidence:
    //!
    //! * `assert_train_val_disjoint` hashed every TRAIN window but probed only
    //!   `val[..1024]`, so a val that was 99% a verbatim copy of train passed
    //!   silently as long as the copy did not start at offset 0.
    //! * A 160-byte val yields exactly one 129-token window, and that single
    //!   window was reported, unqualified, as `val_bpb`.
    //! * `DONE: bpb=` was `best_bpb`, the running minimum of an EMA seeded from
    //!   `init_bpb` (~7.0) with weight `PHI_INV` on the stale value - not the
    //!   measurement, and not equal to it.
    use super::checkpoint_codec_tests::ENV_LOCK;
    use super::*;

    /// Deterministic pseudo-text over an `alphabet`-symbol range. Hermetic, and
    /// non-periodic enough to satisfy the 8-gram entropy precondition.
    fn lcg_bytes(seed: u64, len: usize, alphabet: u64) -> Vec<u8> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                b'a' + ((s >> 33) % alphabet) as u8
            })
            .collect()
    }

    fn lcg_tokens(seed: u64, len: usize, alphabet: u64) -> Vec<usize> {
        lcg_bytes(seed, len, alphabet)
            .into_iter()
            .map(|b| (b as usize) % VOCAB)
            .collect()
    }

    /// Run the guard and return its panic message, or `None` if it accepted.
    fn guard_message(train: &[usize], val: &[usize]) -> Option<String> {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {})); // the panic is the expected result
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_train_val_disjoint(train, val)
        }));
        std::panic::set_hook(prev);
        match out {
            Ok(()) => None,
            Err(e) => Some(
                e.downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "<non-string panic>".to_string()),
            ),
        }
    }

    /// The exact scenario the guard's own comment named and could not see.
    ///
    /// The copied region starts at val offset 120, so `val[..1024]` is NOT a
    /// substring of train and the previous probe accepted this corpus. The
    /// panic must now also REPORT how bad the overlap is: "detected" and
    /// "99% of the eval set is training data" are different findings.
    #[test]
    fn a_val_that_is_99_percent_a_copy_of_train_is_rejected() {
        let train = lcg_tokens(0xA1CE, 40_000, 95);
        let mut val = lcg_tokens(0xB0BA, 120, 95);
        val.extend_from_slice(&train[20_001..20_001 + 11_880]);
        assert_eq!(val.len(), 12_000);

        // Precondition: the old probe window is clean, so this corpus is a
        // false negative for the pre-fix guard rather than a repeat of a case
        // it already caught.
        assert!(
            !train.windows(1024).any(|w| w == &val[..1024]),
            "fixture is invalid: the old val[..1024] probe would have caught this"
        );

        let msg = guard_message(&train, &val).expect("a 99% copy must be rejected");
        assert!(msg.contains("TRAIN/VAL OVERLAP DETECTED"), "{msg}");
        assert!(
            msg.contains("98.98% of val windows (11625 of 11745"),
            "overlap fraction not reported: {msg}"
        );
        assert!(
            msg.contains("threshold is 1.00%"),
            "threshold not stated in the panic: {msg}"
        );
    }

    /// A byte-disjoint corpus of the same shape must still be accepted, so the
    /// test above is evidence about overlap and not about the guard rejecting
    /// everything.
    #[test]
    fn a_disjoint_val_of_the_same_shape_is_accepted() {
        let train = lcg_tokens(0xA1CE, 40_000, 95);
        let val = lcg_tokens(0xB0BA, 12_000, 95);
        assert_eq!(guard_message(&train, &val), None);
    }

    /// `data/pangram_fixture_160b.bin` is 160 bytes. It used to produce a
    /// `val_bpb` from a single window.
    #[test]
    fn a_160_token_val_is_rejected() {
        let train = lcg_tokens(0xA1CE, 40_000, 95);
        let val = lcg_tokens(0xB0BA, 160, 95);
        let msg = guard_message(&train, &val).expect("a 160-token val must be rejected");
        assert!(msg.contains("VAL STREAM TOO SHORT"), "{msg}");
        assert!(msg.contains("160 tokens"), "{msg}");
        assert!(msg.contains("minimum 8192"), "{msg}");
    }

    /// The size floor is expressed in the same terms `evaluate` uses.
    #[test]
    fn a_160_token_val_yields_exactly_one_eval_chunk() {
        assert_eq!(eval_chunk_count(160), 1);
        assert!(eval_chunk_count(160) < MIN_EVAL_CHUNKS);
        assert_eq!(eval_chunk_count(SEQ), 0, "too short for a single window");
        assert!(
            eval_chunk_count(MIN_VAL_TOKENS) >= MIN_EVAL_CHUNKS,
            "the token floor must imply the chunk floor"
        );
    }

    /// The knob that decides whether the weights get floored is a recipe
    /// parameter with its own name and its own default.
    #[test]
    fn gf16_floor_cadence_is_its_own_knob() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_GF16_FLOOR_EVERY");
        assert_eq!(gf16_floor_every(), GF16_FLOOR_EVERY_DEFAULT);
        assert_eq!(gf16_floor_every(), 1, "default must floor every step");
        std::env::set_var("TRIOS_GF16_FLOOR_EVERY", "500");
        assert_eq!(gf16_floor_every(), 500);
        // Junk and zero resolve to the default rather than disabling the floor
        // silently or dividing by zero.
        for junk in ["0", "", "nonsense", "-3"] {
            std::env::set_var("TRIOS_GF16_FLOOR_EVERY", junk);
            assert_eq!(gf16_floor_every(), GF16_FLOOR_EVERY_DEFAULT, "junk={junk:?}");
        }
        std::env::remove_var("TRIOS_GF16_FLOOR_EVERY");
    }

    /// `final_val_bpb` is the reading; `ema_bpb` is the lagging signal. They
    /// diverge, so which one a run reports is not a cosmetic choice.
    ///
    /// The pre-fix `RunOutcome::final_bpb` was the running MINIMUM of the EMA.
    /// On a monotonically improving run that minimum IS the last EMA value, so
    /// asserting `final_val_bpb != ema_bpb` here is exactly the assertion that
    /// the number `DONE:` prints has changed.
    #[test]
    fn final_val_bpb_is_the_measurement_not_the_ema() {
        let _g = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let train_path = dir.path().join("train.txt");
        let val_path = dir.path().join("val.txt");
        // An 8-symbol alphabet is ~3 bits/byte, so the model leaves the ~7.0
        // init reading fast and the EMA has something to lag behind.
        std::fs::write(&train_path, lcg_bytes(0xA1CE, 40_000, 8)).unwrap();
        std::fs::write(&val_path, lcg_bytes(0xB0BA, 9_000, 8)).unwrap();

        for k in [
            "DATABASE_URL",
            "NEON_DATABASE_URL",
            "TRIOS_NEON_DSN",
            "TRIOS_DATABASE_URL",
            "TRIOS_CANON_NAME",
            "TRIOS_CHECKPOINT_EVERY",
            "TRIOS_GF16_FLOOR_EVERY",
            "HIDDEN_DIM",
            "NUM_ATTN_LAYERS",
            "GF16_ENABLED",
        ] {
            std::env::remove_var(k);
        }
        std::env::set_var("TRIOS_CHECKPOINT_DISABLE", "1");

        // Kept to the minimum that still shows the lag: `evaluate` always
        // averages 40 chunks, so every extra eval is ~4800 debug-mode forward
        // passes. Three evals (init + 2) are enough for the EMA to be visibly
        // behind the reading.
        let steps = 12usize;
        let outcome = run_single(&TrainArgs {
            seed: 47,
            steps,
            hidden: 16,
            lr: 0.02,
            attn_layers: 1,
            eval_every: 6,
            train_path: train_path.to_string_lossy().into_owned(),
            val_path: val_path.to_string_lossy().into_owned(),
        })
        .expect("run_single");
        std::env::remove_var("TRIOS_CHECKPOINT_DISABLE");

        let final_val = outcome.final_val_bpb.expect("the last step was an eval");
        let ema = outcome.ema_bpb.expect("the EMA is always defined");
        let best = outcome.best_val_bpb.expect("at least one reading was taken");

        assert!(
            (final_val - ema).abs() > 0.1,
            "final_val_bpb={final_val:.4} and ema_bpb={ema:.4} did not diverge; \
             this fixture no longer demonstrates the defect"
        );
        assert!(
            ema > final_val,
            "the EMA is seeded from init (~7.0) and must lag a improving run: \
             final_val_bpb={final_val:.4} ema_bpb={ema:.4}"
        );
        assert!(
            best <= final_val,
            "best_val_bpb={best:.4} must be a minimum over the raw readings, \
             not over the EMA"
        );
        // The compatibility mirror carries the measurement, not the EMA.
        assert_eq!(outcome.final_bpb, final_val);
    }
}

#[cfg(test)]
mod alphabet_fold_tests {
    //! The record must be able to describe its own alphabet.
    //!
    //! `VOCAB = 128` and `load_data` folds `(b as usize) % VOCAB`. That is the
    //! identity on ASCII and a silent two-to-one collapse on everything else.
    //! A UTF-8 Cyrillic corpus was accepted with no warning and produced a
    //! figure labelled bits-per-BYTE that had been measured over 128 symbols.
    use super::*;

    /// "Rossiya" in UTF-8 Cyrillic, plus the capital ER whose continuation byte
    /// is 0xA0 - the byte that folds onto a space.
    const CYRILLIC: &[u8] = "Rossiya \u{0420}\u{043e}\u{0441}\u{0441}\u{0438}\u{044f}\n".as_bytes();

    #[test]
    fn the_fold_that_destroys_cyrillic_is_refused() {
        let err = assert_alphabet_fold_injective("/tmp/cyr.txt", CYRILLIC)
            .expect_err("a Cyrillic corpus must not be folded mod 128 in silence");
        let msg = err.to_string();
        assert!(msg.starts_with("ALPHABET FOLD REFUSED:"), "message was {msg:?}");
        assert!(msg.contains("/tmp/cyr.txt"), "message must name the file: {msg:?}");
        assert!(
            msg.contains("bits-per-byte"),
            "message must say what the number is not: {msg:?}"
        );
    }

    #[test]
    fn the_two_bytes_that_collide_are_really_in_the_fixture() {
        // Not an assumption about UTF-8: the collision is measured here, so the
        // guard's justification is checked rather than asserted.
        assert!(CYRILLIC.contains(&0xD0), "fixture lost its Cyrillic lead byte");
        assert!(CYRILLIC.contains(&0xA0), "fixture lost the 0xA0 continuation byte");
        assert_eq!(0xD0usize % VOCAB, b'P' as usize);
        assert_eq!(0xA0usize % VOCAB, b' ' as usize);
        assert_eq!(0xD1usize % VOCAB, b'Q' as usize);
    }

    #[test]
    fn a_pure_ascii_corpus_is_accepted() {
        let ascii = b"To be, or not to be, that is the question:\n";
        assert!(ascii.iter().all(|&b| (b as usize) < VOCAB));
        assert_alphabet_fold_injective("/tmp/ascii.txt", ascii)
            .expect("ASCII folds injectively mod 128 and must be accepted");
    }

    /// The canonical recipe must be unaffected. If either shipped corpus ever
    /// grows a non-ASCII byte, this test - not a silent BPB - is what says so.
    #[test]
    fn the_shipped_tinyshakespeare_corpora_are_ascii() {
        for path in ["data/tiny_shakespeare.txt", "data/tiny_shakespeare_val.txt"] {
            let Ok(raw) = std::fs::read(path) else {
                continue; // not a fixture this checkout carries
            };
            assert!(
                !raw.is_empty(),
                "{path} is empty; the calibration curve was not measured on nothing"
            );
            assert_alphabet_fold_injective(path, &raw)
                .unwrap_or_else(|e| panic!("{path} is no longer ASCII: {e}"));
        }
    }

    /// End to end through `load_data`, so the guard is proven to be WIRED and
    /// not merely present.
    #[test]
    fn load_data_refuses_a_cyrillic_file() {
        let dir = std::env::temp_dir().join(format!(
            "trios-alphabet-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("cyr.txt");
        std::fs::write(&path, CYRILLIC).expect("write fixture");
        let err = load_data(&path.to_string_lossy())
            .expect_err("load_data must refuse a corpus its alphabet destroys");
        assert!(
            err.to_string().starts_with("ALPHABET FOLD REFUSED:"),
            "load_data returned {err:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
