use anyhow::{Context, Result};
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use crate::arch_config::{parse_gf16_enabled, parse_hidden_dim, parse_num_attn_layers};
use crate::fake_quant::{self, FormatKind};
use crate::model_hybrid_attn::{AttentionCache, HybridAttn};
use crate::objective::{nca_entropy_loss, NcaObjective};

/// GATE-2's target. NOTE, and left deliberately unchanged: 1.85 is BELOW
/// `invariants::PUBLISHED_BPB_FLOOR` (2.0), so no reading that would satisfy
/// this gate can survive `guard_bpb` - a run good enough to "pass" is refused
/// as unpublishable before it prints. That is not a bug in either number; it is
/// the honest state of the evidence. The measured calibration on this
/// architecture bottoms out around 2.6, and every historical sub-2.0 pass
/// (1.5492, 1.038) has been retracted. Moving this target is a claim about
/// results and needs new measurements, not an edit here.
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

/// Legacy GF16 reading: the inverted `TRIOS_GF16_DISABLE` variable, default ON.
///
/// Call it through [`resolve_gf16_knob`] and nowhere else. Reading it directly
/// is what made `GF16_ENABLED` inert: the gate and the four sidecar fields all
/// consulted this function, so the documented knob could not move either the
/// weights or the record of what produced them.
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

/// Wave 31 PR-B: resolve the GF16 knob the run will actually execute.
///
/// `GF16_ENABLED`, WHEN SET, is authoritative. When it is unset the legacy
/// `TRIOS_GF16_DISABLE` reading in [`gf16_enabled`] decides, and that reading
/// is ON by default - which is what every published artifact in this
/// repository was produced with.
///
/// The bug this shape removes: `parse_gf16_enabled` defaults to `"false"`, so
/// an explicit `GF16_ENABLED=false` was indistinguishable from "unset" and the
/// resolved value was discarded anyway; the live gate read `gf16_enabled()`
/// directly. `GF16_ENABLED=false` therefore changed nothing - same checkpoint
/// hash as unset - while the sidecar still recorded `gf16_enabled: true` and
/// `entrypoint` printed the knob as consumed. Reading the raw variable here is
/// what makes "set" and "unset" two different facts.
///
/// The default stays ON deliberately. Flipping it would silently invalidate
/// every number already published from this crate, including the aarch64
/// anchor in README.md; documenting it correctly costs nothing and breaks
/// nothing. `GF16_ENABLED=false` is what turns the floor off.
///
/// `GF16_ENABLED=true` without the `gf16` feature compiled in is still an
/// error, unchanged.
/// Anchor: phi^2+phi^-2=3 - DOI 10.5281/zenodo.19227877
fn resolve_gf16_knob() -> Result<bool> {
    if std::env::var("GF16_ENABLED").is_ok() {
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
        return Ok(false);
    }
    Ok(gf16_enabled())
}

fn attn_seq_override() -> usize {
    std::env::var("TRIOS_ATTN_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(ATTN_SEQ)
}

/// How many windows `evaluate` averages over, unless told otherwise.
///
/// 40 windows of `SEQ + 1` = 129 tokens is 5,160 bytes: 5.16% of the
/// 100,000-byte val corpus. That was a hardcoded literal in two places and
/// appeared in no field of the checkpoint record, so every published BPB was a
/// 5% sample presented as if it were the corpus.
pub const EVAL_CHUNKS_DEFAULT: usize = 40;

/// The declared eval coverage: how many windows to average, `0` = every window.
///
/// The default stays at `EVAL_CHUNKS_DEFAULT` ON PURPOSE. The defect is that
/// the coverage was never STATED, not that 40 is the wrong number, and moving
/// the default would silently make every existing BPB incomparable with every
/// new one - the same mistake as gating `gf16_floor` on `eval_every`. This is
/// an OBSERVATION parameter: it changes what is measured, never the weights,
/// and (unlike `eval_every`, see `gf16_floor_every`) nothing in the training
/// loop reads it.
///
/// `TRIOS_EVAL_CHUNKS=0` means full coverage: non-overlapping windows tiling
/// the whole val stream, which is the only setting with no sampling error at
/// all - and, since the finite-population correction landed in `evaluate`, the
/// only setting whose recorded `val_bpb_stderr` is exactly `0.0`. A missing or
/// unparseable value resolves to the default; `0` is a legal value and is NOT
/// treated as junk.
pub fn eval_chunks_target() -> usize {
    std::env::var("TRIOS_EVAL_CHUNKS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(EVAL_CHUNKS_DEFAULT)
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
///
/// This cadence is INDEPENDENT of `--eval-every`. Both call sites used to sit
/// inside the eval guard, so the requested cadence was intersected with the
/// observation cadence and the difference was dropped in silence: measured,
/// `TRIOS_CHECKPOINT_EVERY=50 --eval-every 100 --steps 200` wrote 100.bin and
/// 200.bin only. A checkpoint step that is not an eval step now takes its own
/// reading for the artifact record without touching the published trajectory.
///
/// Resolved ONCE per run, and a non-empty value that does not parse is a HARD
/// ERROR. It used to be re-read from the environment on every single step
/// through `.ok().and_then(|s| s.parse().ok()).unwrap_or(0)`, which turned
/// `TRIOS_CHECKPOINT_EVERY=1_000` (a Rust integer literal, and the shape a
/// reader of this codebase would naturally type) and `1000 ` with any trailing
/// junk into 0: no artifacts, no warning, and nothing in the banner or the
/// sidecar to say the request had been dropped. That is the same silent
/// substitution as "1,851 experiments, zero artifacts", one layer up.
///
/// Absent or empty stays 0 = final step only; surrounding whitespace is
/// trimmed, exactly as `gf16_floor_every` does. `0` remains a legal explicit
/// value meaning the default. Everything else - `1_000`, `1k`, `-3`, `1000x` -
/// stops the run naming the variable.
fn resolve_checkpoint_every() -> Result<usize> {
    let raw = match std::env::var("TRIOS_CHECKPOINT_EVERY") {
        Err(std::env::VarError::NotPresent) => return Ok(0),
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!(
                "TRIOS_CHECKPOINT_EVERY is set to a non-UTF-8 value. Unset it for \
                 final-step-only checkpointing, or set a plain decimal number of steps."
            )
        }
        Ok(raw) => raw,
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(0);
    }
    trimmed.parse::<usize>().map_err(|e| {
        anyhow::anyhow!(
            "TRIOS_CHECKPOINT_EVERY={raw:?} is not a step count ({e}). Use a plain \
             decimal number - no underscores, no suffixes, no sign - or unset the \
             variable for final-step-only checkpointing. Resolving this to 0 is what \
             used to happen, and it wrote no artifacts and said nothing."
        )
    })
}

/// Whether `step` is on the resolved cadence. `every` comes from
/// `resolve_checkpoint_every()` once per run; nothing here reads the
/// environment, so an export mid-run cannot change what gets saved.
fn checkpoint_every_hit(step: usize, every: usize) -> bool {
    every > 0 && step.is_multiple_of(every)
}

/// Emit an artifact of the INITIAL weights, before the first optimizer step:
/// `TRIOS_CHECKPOINT_INIT=1`.
///
/// The earliest artifact the trainer could produce was after one optimizer
/// step (both loops are `for step in 1..=args.steps`), so the question "do the
/// two architectures already disagree at initialisation, or only after
/// arithmetic?" could not be asked of any file this program wrote. Those are
/// different defects with different fixes - init/RNG versus floating-point
/// contraction and reduction order - and a 12000-step artifact cannot tell them
/// apart. See `docs/DIVERGENCE-LOCALIZATION.md`.
///
/// Off by default: `0.bin` is a new file in every existing run directory, and
/// this is an evidence knob, not a recipe knob. It changes no weight and no
/// number - it only saves bytes that already existed.
fn checkpoint_init_enabled() -> bool {
    std::env::var("TRIOS_CHECKPOINT_INIT").as_deref() == Ok("1")
}

/// Resolve the run identity ONCE per run. The same string names the checkpoint
/// directory and the ledger row, so the artifact and the BPB cannot drift
/// apart. R5: a missing canon_name must never cost a training run silently.
pub fn resolve_canon_name(seed: u64) -> String {
    std::env::var("TRIOS_CANON_NAME")
        .ok()
        .or_else(|| std::env::var("CANON_NAME").ok())
        .unwrap_or_else(|| format!("trios-train-rng{seed}"))
}

/// True while `run_sweep` is driving several seeds through `run_single` in one
/// process.
///
/// `resolve_canon_name` returns `TRIOS_CANON_NAME` verbatim and cannot vary
/// with the seed - that variable is exactly what the scarab and the Railway
/// workers set - so a sweep resolved three runs to one checkpoint directory and
/// one `{step}.bin`. Each save renamed over the previous one; three hashes and
/// three `DONE:` lines were printed and one artifact survived. The ledger was
/// worse: three rows claimed three different `sha256` values for one `path`.
///
/// The seed therefore scopes the DIRECTORY on the sweep path (see
/// `checkpoint::run_dir`), while `canon_name` stays the ledger identity it has
/// always been. The single-seed layout is untouched: it is where the README and
/// `ckpt_replay` look.
static SWEEP_SEED_SCOPE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Sets `SWEEP_SEED_SCOPE` for its lifetime, including across the `?` of a
/// failed seed - a sweep that dies half-way must not leave later single-seed
/// runs in this process writing into `seed{n}/` subdirectories.
struct SweepSeedScope;

impl SweepSeedScope {
    fn enter() -> Self {
        SWEEP_SEED_SCOPE.store(true, std::sync::atomic::Ordering::SeqCst);
        Self
    }
}

impl Drop for SweepSeedScope {
    fn drop(&mut self) {
        SWEEP_SEED_SCOPE.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

/// `Some(seed)` only inside `run_sweep`. See `SWEEP_SEED_SCOPE`.
fn checkpoint_seed_scope(seed: u64) -> Option<u64> {
    if SWEEP_SEED_SCOPE.load(std::sync::atomic::Ordering::SeqCst) {
        Some(seed)
    } else {
        None
    }
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

/// Exit code for a refused `--eval-every`, matching clap's usage-error code.
///
/// The refusal belongs on the `--eval-every` argument declaration, next to
/// `default_value_t = 1000`, as a `value_parser` range. It is here instead
/// because `src/bin/trios-train.rs` is outside this change's ownership, and a
/// guard at the library boundary catches every caller rather than one CLI -
/// `format_champion_sweep`, `tri` and `run_sweep` all build `TrainArgs`
/// themselves. When the `value_parser` lands, this stays: a library that
/// panics on a legal-looking argument is a library defect, not a CLI one.
pub const EVAL_EVERY_USAGE_EXIT: i32 = 2;

/// Refuse `eval_every == 0` with the message clap would have printed.
///
/// `0` is a natural operator guess: it means "full coverage" for the
/// neighbouring `TRIOS_EVAL_CHUNKS` (`eval_chunks_target`) and "final step
/// only" for `TRIOS_CHECKPOINT_EVERY` (`resolve_checkpoint_every`). On this
/// knob it meant `step % 0`, which panicked with "attempt to calculate the
/// remainder with a divisor of zero" at the first step - AFTER the initial
/// evaluation had run and BEFORE any artifact could be written, so the run cost
/// its startup, produced nothing, and exited 101 naming an arithmetic operation
/// instead of an argument.
///
/// Pure and `Result`-returning so it is testable without ending the process;
/// see `refuse_eval_every_or_exit` for the caller-facing side.
pub fn validate_eval_every(eval_every: usize) -> Result<()> {
    if eval_every > 0 {
        return Ok(());
    }
    anyhow::bail!(
        "invalid value '0' for '--eval-every <EVAL_EVERY>': a cadence of 0 is not \
         \"never\" - it is `step % 0`, which panics at the first training step, \
         after the initial evaluation and before any artifact is written. To \
         evaluate only at the end, pass --eval-every equal to --steps: the final \
         step is always evaluated. (0 does mean full coverage for --eval-chunks \
         and final-step-only for TRIOS_CHECKPOINT_EVERY; it does not mean either \
         here.)"
    )
}

/// `validate_eval_every`, terminating the process with clap's usage-error code
/// instead of returning.
///
/// `std::process::exit` from a library is deliberate and bounded: it is called
/// on the first line of the entry points, before any file, model or ledger
/// handle exists, so there is nothing to unwind and nothing half-written. The
/// alternative - an `anyhow::Error` - reaches `main` and exits 1, which is the
/// code this crate uses for "the run failed", not "the invocation was wrong".
fn refuse_eval_every_or_exit(eval_every: usize) {
    if let Err(e) = validate_eval_every(eval_every) {
        eprintln!("error: {e}");
        eprintln!();
        eprintln!("For more information, try '--help'.");
        std::process::exit(EVAL_EVERY_USAGE_EXIT);
    }
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
        let raw = std::fs::read(path).with_context(|| format!("failed to read corpus '{path}'"))?;
        assert_alphabet_fold_injective(path, &raw)?;
        return Ok((
            raw.into_iter().map(|b| (b as usize) % VOCAB).collect(),
            false,
        ));
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
pub const OVERLAP_WINDOW: usize = 256;

/// Fail the run above this fraction of val windows found verbatim in train.
///
/// Not zero. The comparison is exact, so on a small val stream a single line of
/// shared boilerplate would otherwise be fatal. A real leak is not marginal:
/// the 2026-04-30 Dockerfile split put 90-100% of val inside train, and the
/// 99%-copy case this threshold exists for lands two orders of magnitude above
/// it.
pub const MAX_VAL_OVERLAP_FRACTION: f64 = 0.01;

/// A val stream shorter than this cannot support a BPB anyone should quote.
/// `data/pangram_fixture_160b.bin` is 160 bytes and yields exactly ONE 129-token
/// window, which `evaluate` used to report, unqualified, as `val_bpb`.
pub const MIN_VAL_TOKENS: usize = 8192;

/// `evaluate` must average over at least this many chunks for the mean to mean
/// anything.
pub const MIN_EVAL_CHUNKS: usize = 8;

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
pub fn assert_train_val_disjoint(train: &[usize], val: &[usize]) {
    // The guard asks the question at the coverage the run will actually use:
    // a stream that yields enough windows at full coverage can still yield too
    // few at `--eval-chunks 4`.
    let chunks = eval_chunk_count(val.len(), eval_chunks_target());
    if let Err(reason) = check_train_val_disjoint(train, val, chunks) {
        panic!("{reason}");
    }
}

/// The body of `assert_train_val_disjoint`, as a value instead of a panic.
///
/// THE single implementation of this guard. It used to be copied into
/// `trinity_pr1722`, `ngram_train_gf16`, `igla_trigram` and `cpu_train`,
/// because it was `pub(crate)` and `src/bin/*.rs` compile as separate crates -
/// and a guard living in five copies is a guard that will drift. It already
/// had: the `step_by(256)` sampled scan that detected an overlap with
/// probability 1/256 lived in exactly such a copy, and `ngram_train_gf16`'s
/// copy still probed only `val[..1024]`, which cannot see a leak that starts
/// one token later.
///
/// `eval_chunks` is the number of windows the CALLER's own `evaluate` will
/// average over. It is a parameter rather than a constant because the binaries
/// chunk differently (`SEQ` is 128 here, 64 in `trinity_pr1722` and
/// `ngram_train_gf16`), and a precondition computed from a different chunking
/// than the one that runs is not a precondition.
///
/// Returns `Err(reason)` so a caller with results already on disk can exit with
/// its own code instead of unwinding.
pub fn check_train_val_disjoint(
    train: &[usize],
    val: &[usize],
    eval_chunks: usize,
) -> Result<(), String> {
    use std::collections::HashSet;

    if val.len() < MIN_VAL_TOKENS {
        return Err(format!(
            "VAL STREAM TOO SHORT: {} tokens, minimum {}. A BPB averaged over a \
             handful of windows is not a held-out measurement and must not be \
             reported as one (data/pangram_fixture_160b.bin is 160 bytes and \
             yielded exactly one 129-token window).",
            val.len(),
            MIN_VAL_TOKENS
        ));
    }
    if eval_chunks < MIN_EVAL_CHUNKS {
        return Err(format!(
            "VAL STREAM YIELDS ONLY {} EVAL CHUNK(S), minimum {}. `evaluate` would \
             average over too few windows for the mean to be informative.",
            eval_chunks, MIN_EVAL_CHUNKS
        ));
    }

    if train.len() < OVERLAP_WINDOW {
        return Ok(()); // no train window to compare against
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

/// First-moment decay of the AdamW that `trios-train` actually steps with.
///
/// Named because the sidecar records it. It is 0.9, NOT the `1/phi = 0.618` of
/// `optimizer::AdamWCpu`: the production trainer has always used its own
/// `AdamW` below and never constructs `AdamWCpu`, so no published `trios-train`
/// BPB was produced with the phi-branded constants. Reading the value from the
/// same constant the update rule uses is what stops the record and the code
/// drifting apart.
const ADAMW_BETA1: f32 = 0.9;
/// Second-moment decay. See `ADAMW_BETA1`.
const ADAMW_BETA2: f32 = 0.999;
/// Denominator epsilon. See `ADAMW_BETA1`.
const ADAMW_EPS: f32 = 1e-8;

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
            beta1: ADAMW_BETA1,
            beta2: ADAMW_BETA2,
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
            params[i] -= lr * (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + ADAMW_EPS);
        }
    }
}

/// The hyperparameters behind the string `optimizer: "adamw"`, for the sidecar.
///
/// `wd` is a per-run local (0.04 on both the AdamW and the Muon path) rather
/// than a constant, so it is passed in from the construction site instead of
/// being restated here, where it could disagree with the optimizers that were
/// actually built.
fn adamw_record_params(wd: f32, source: &str) -> crate::checkpoint::OptimizerParams {
    crate::checkpoint::OptimizerParams {
        beta1: ADAMW_BETA1 as f64,
        beta2: ADAMW_BETA2 as f64,
        eps: ADAMW_EPS as f64,
        weight_decay: wd as f64,
        source: source.to_string(),
    }
}

/// The AdamW instances `run_single` steps, in the order it steps them.
///
/// The order is the format's canonical order (see `crate::checkpoint`'s
/// resume-record section) and the names are the instance names, not tensor
/// names: they exist so that a record restored into a differently built loop
/// fails on the name rather than silently pouring `attn_up`'s moments into
/// `attn_down`, which has the identical element count.
fn adamw_instance_names(num_ctx: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(7 + num_ctx);
    names.push("embed".to_string());
    for i in 0..num_ctx {
        names.push(format!("ctx{i}"));
    }
    for n in ["proj", "attn_down", "attn_up", "head", "attn_w"] {
        names.push(n.to_string());
    }
    names
}

/// Snapshot every AdamW instance's whole mutable state.
///
/// `m`, `v` and `step` are the entirety of it: `beta1`, `beta2` and `wd` are
/// constants of the recipe and are checked, not restored (see
/// `checkpoint::verify_resume`).
#[allow(clippy::too_many_arguments)]
fn capture_adamw_state(
    opt_embed: &AdamW,
    opt_ctx: &[AdamW],
    opt_proj: &AdamW,
    opt_attn_down: &AdamW,
    opt_attn_up: &AdamW,
    opt_head: &AdamW,
    opt_attn_w: &AdamW,
) -> Vec<crate::checkpoint::ResumeOptimizerState> {
    let names = adamw_instance_names(opt_ctx.len());
    let mut refs: Vec<&AdamW> = Vec::with_capacity(names.len());
    refs.push(opt_embed);
    refs.extend(opt_ctx.iter());
    refs.push(opt_proj);
    refs.push(opt_attn_down);
    refs.push(opt_attn_up);
    refs.push(opt_head);
    refs.push(opt_attn_w);
    names
        .into_iter()
        .zip(refs)
        .map(|(name, o)| crate::checkpoint::ResumeOptimizerState {
            name,
            step: o.step as u64,
            m: o.m.clone(),
            v: o.v.clone(),
        })
        .collect()
}

/// Pour a validated record back into freshly constructed optimizers.
///
/// Refuses on instance count, instance name or element count. There is no
/// partial restore: a loop that got some moments and zeroed the rest would
/// run to completion and produce weights that are a segment of nothing.
#[allow(clippy::too_many_arguments)]
fn restore_adamw_state(
    rec: &crate::checkpoint::ResumeRecord,
    opt_embed: &mut AdamW,
    opt_ctx: &mut [AdamW],
    opt_proj: &mut AdamW,
    opt_attn_down: &mut AdamW,
    opt_attn_up: &mut AdamW,
    opt_head: &mut AdamW,
    opt_attn_w: &mut AdamW,
) -> Result<()> {
    use crate::checkpoint::{resume_refusal, RESUME_REASON_SHAPE};
    let names = adamw_instance_names(opt_ctx.len());
    let mut slots: Vec<&mut AdamW> = Vec::with_capacity(names.len());
    slots.push(opt_embed);
    slots.extend(opt_ctx.iter_mut());
    slots.push(opt_proj);
    slots.push(opt_attn_down);
    slots.push(opt_attn_up);
    slots.push(opt_head);
    slots.push(opt_attn_w);

    if rec.optimizers.len() != slots.len() {
        return Err(resume_refusal(
            RESUME_REASON_SHAPE,
            format!(
                "the record carries {} optimizer instances, this loop runs {}",
                rec.optimizers.len(),
                slots.len()
            ),
        ));
    }
    for (i, (slot, name)) in slots.iter_mut().zip(names.iter()).enumerate() {
        let src = &rec.optimizers[i];
        if &src.name != name {
            return Err(resume_refusal(
                RESUME_REASON_SHAPE,
                format!(
                    "optimizer instance {i} is {:?}, this loop runs {name:?}",
                    src.name
                ),
            ));
        }
        if src.m.len() != slot.m.len() {
            return Err(resume_refusal(
                RESUME_REASON_SHAPE,
                format!(
                    "optimizer instance {name:?} carries {} elements, this loop allocated {}",
                    src.m.len(),
                    slot.m.len()
                ),
            ));
        }
        slot.m.copy_from_slice(&src.m);
        slot.v.copy_from_slice(&src.v);
        slot.step = src.step as usize;
    }
    Ok(())
}

fn gf16_floor(weights: &mut [f32]) {
    let scale = 16.0_f32;
    for w in weights.iter_mut() {
        *w = (*w * scale).round() / scale;
    }
}

/// The override that lets an operator run a format the crate declares it
/// cannot faithfully simulate. Same variable `matrix_runner` reads, so one
/// export covers both paths and neither can be relaxed without the other.
pub const UNFAITHFUL_FORMAT_OVERRIDE_ENV: &str = "TRIOS_ALLOW_UNFAITHFUL_FORMAT";

/// Every spelling `FormatKind::from_env` accepts, for the refusal message.
///
/// Built from `FormatKind::all()` rather than typed out, so a format added to
/// the enum cannot go missing from the list of things the operator is told to
/// choose between. These are the CANONICAL names; `from_env` also accepts the
/// aliases documented on it (`fp32`/`binary32`/`float32` for `f32`, and so on).
fn accepted_format_spellings() -> String {
    let mut names: Vec<&'static str> = FormatKind::all().iter().map(|f| f.name()).collect();
    names.sort_unstable();
    names.dedup();
    names.join(", ")
}

/// The refusal `matrix_runner::resolve_format_faithful` raises, word for word.
///
/// DUPLICATED ON PURPOSE, and the duplication is the point of this comment:
/// `resolve_format_faithful` lives in a binary and cannot be imported, so the
/// two refusals are kept textually identical rather than paraphrased. This one
/// is `pub` so that the binary can adopt it and the copy can be deleted; until
/// then, an edit to either must be made to both.
pub fn unfaithful_format_refusal(format: &str, kind: FormatKind) -> String {
    format!(
        "NON-FAITHFUL FORMAT {format:?} ({kind:?}): FormatKind::is_faithful() is \
         false, i.e. the crate itself declares the f32 round trip through this \
         format is not really this format (identity passthrough, mantissa-mask \
         stand-in, or a deferred encoder). Publishing it alongside real kernels \
         makes an identity look like a result. Set \
         TRIOS_ALLOW_UNFAITHFUL_FORMAT=1 to run it anyway; the row is then \
         stamped format_faithful=false."
    )
}

/// Resolve the QAT format from `TRIOS_FORMAT_TYPE` (or alias `TRIOS_FAKE_QUANT_FORMAT`).
/// `Ok(None)` means no quantization: the variable is unset, or it names F32.
/// Closes scarab->trios-train gap from #509: previously only `cpu_train` honoured the
/// env var, so `TRIOS_FORMAT_TYPE=fp16` produced identical BPB to F32 in production.
///
/// Two things that used to be silent are now refusals.
///
/// AN UNRECOGNISED SPELLING. `FormatKind::from_env` answers `None` for anything
/// it does not know, and this function propagated that `None` as "QAT off":
/// `TRIOS_FORMAT_TYPE=int_8` printed nothing mentioning a format, trained f32,
/// and recorded `fake_quant_format: "f32"`. That is the same silent
/// substitution as "1,851 experiments, zero artifacts", one knob over from
/// `resolve_checkpoint_every` - which was hardened against exactly it - and it
/// is worse here, because the operator asked for a MEASUREMENT of int8 and got
/// a measurement of f32 wearing no label at all.
///
/// A FORMAT THE CRATE CANNOT SIMULATE. `fake_quantize_model` returns
/// immediately for a format in `is_unsupported_in_f32()`, so the weights are
/// untouched - measured: an `fp80` payload bit-identical to the `f32` control
/// after the 256-byte header, same `final_val_bpb` to all 16 digits - while the
/// label `fp80` went into the sidecar AND into the hashed header, which is what
/// made an identity run mint a distinct-looking artifact. `matrix_runner` has
/// refused this since 2026-08-03; the binary that mints checkpoints did not.
/// The predicate is `FormatKind::is_faithful()`, which is strictly wider than
/// `is_unsupported_in_f32()` (the latter implies the former is false - see
/// `fake_quant::unsupported_in_f32_implies_not_faithful`) and therefore also
/// catches `int32` and the `mxfp*` element-only stand-ins.
///
/// `TRIOS_ALLOW_UNFAITHFUL_FORMAT=1` lets the second refusal through, and the
/// artifact then carries its own retraction: `format_faithful: false` in the
/// sidecar, derived from the same label by `checkpoint::format_label_faithful`.
/// There is no override for the first: an unrecognised spelling names no
/// format, so there is nothing to stamp and nothing to retract.
pub fn resolve_fake_quant_format() -> Result<Option<FormatKind>> {
    let raw = match std::env::var("TRIOS_FORMAT_TYPE")
        .ok()
        .or_else(|| std::env::var("TRIOS_FAKE_QUANT_FORMAT").ok())
    {
        Some(raw) => raw,
        None => return Ok(None),
    };
    let fmt = match FormatKind::from_env(&raw) {
        Some(fmt) => fmt,
        None => anyhow::bail!(
            "TRIOS_FORMAT_TYPE={raw:?} is not a format this build knows. Resolving \
             it to \"no quantization\" is what used to happen: the run trained f32, \
             recorded fake_quant_format=\"f32\", and said nothing about the format \
             that was asked for. Accepted spellings (aliases also accepted, see \
             FormatKind::from_env): {}",
            accepted_format_spellings()
        ),
    };
    if fmt == FormatKind::F32 {
        return Ok(None);
    }
    if !fmt.is_faithful() {
        let allowed = std::env::var(UNFAITHFUL_FORMAT_OVERRIDE_ENV)
            .map(|v| v == "1")
            .unwrap_or(false);
        if !allowed {
            anyhow::bail!("[R5-honesty] {}", unfaithful_format_refusal(&raw, fmt));
        }
        eprintln!(
            "[trios-train] WARNING: format={raw} is NOT faithful and is running \
             only because TRIOS_ALLOW_UNFAITHFUL_FORMAT=1. The checkpoint sidecar \
             will carry format_faithful=false. Do not read it as a measurement of \
             {raw}."
        );
    }
    Ok(Some(fmt))
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
            // A clamp here is a fabricated measurement. `f32::max` IGNORES
            // NaN, so clamping to a 1e-10 floor turned a poisoned forward pass
            // into a finite 23.03-nat reading; the same floor also turned a
            // merely UNDERFLOWED probability - a finite 0.0 out of the f32
            // softmax, which every `is_nan`/`is_finite` guard accepts - into
            // that same 23.02585 nats / 33.21928 bpb. Both are absences.
            let p = logits[target];
            if !p.is_finite() || p <= 0.0 {
                return None;
            }
            total -= p.ln();
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
    /// `resolve_gf16_knob()`: `GF16_ENABLED` when set, else the legacy
    /// `TRIOS_GF16_DISABLE` reading. It must be the SAME value that gated
    /// `gf16_floor()` in the loop that produced these weights - this field
    /// used to be filled from `gf16_enabled()` independently, so a run started
    /// with `GF16_ENABLED=false` recorded `true`.
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
        // Once per checkpoint, so the cost is unmeasurable - and as a
        // `debug_assert_eq!` it never executed in ANY run that produced
        // evidence, because every artifact this repository has ever minted came
        // out of a `--release` build. A serialiser that wrote fewer bytes than
        // it allocated ships the tail of the buffer as weights.
        anyhow::ensure!(
            off == out.len(),
            "checkpoint serialiser wrote {off} bytes into a {} byte buffer",
            out.len()
        );
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
        let mut attn = HybridAttn::with_config(cfg).map_err(|e| {
            anyhow::anyhow!("checkpoint header failed HybridAttnConfig::validate: {e:?}")
        })?;

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
        // Same promotion, same reason as in `to_checkpoint_bytes`: once per
        // load, and a `debug_assert_eq!` here has never run in release. The
        // total length is already checked against the tensor directory above,
        // so this is the reader's own post-condition - that it consumed exactly
        // what it validated - and a failure means the two disagree.
        anyhow::ensure!(
            off == bytes.len(),
            "checkpoint reader consumed {off} of {} bytes",
            bytes.len()
        );

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
///
/// `seed_scope` is `Some(seed)` only on the sweep path, where it gives each
/// seed its own directory; the artifact, its sidecar and the ledger row all
/// carry the path that was actually written, so no two rows can claim
/// different hashes for one file.
#[allow(clippy::too_many_arguments)]
fn emit_checkpoint(
    model: &HybridModel,
    canon: &str,
    seed_scope: Option<u64>,
    meta: &CheckpointMeta,
    corpus: &crate::checkpoint::CorpusProvenance,
    run: &RunKnobs,
    // `eval` is the grid `bpb` was measured on, so the record states its own
    // sampling plan instead of leaving `num_chunks` a literal in the loop
    // above; `val_len` is what that coverage is a fraction OF.
    eval: &EvalStats,
    val_len: usize,
    opt_params: &crate::checkpoint::OptimizerParams,
    bpb: Option<f64>,
    min_observed_val_bpb: Option<f64>,
    ema_bpb: Option<f64>,
) -> Result<crate::checkpoint::SavedCheckpoint> {
    use crate::checkpoint::{
        CheckpointRecord, CHECKPOINT_FORMAT_VERSION, CHECKPOINT_RECORD_SCHEMA,
    };

    let bytes = model.to_checkpoint_bytes(meta)?;
    let saved = crate::checkpoint::save_scoped(canon, seed_scope, meta.step as usize, &bytes)?;
    let cfg = *model.attn.config();
    let path_str = saved.path.to_string_lossy().into_owned();
    // Schema 9. What the RECORD carries - the sidecar and the ledger row alike -
    // is the artifact relative to the digest scope, never an absolute path: an
    // absolute one published the builder's home directory in every locally
    // produced sidecar, and those sidecars are committed as evidence. The
    // ABSOLUTE `path_str` survives for exactly one purpose below, the `[ckpt]`
    // line on this operator's own stderr, which is a console message and not an
    // artifact anybody else reads.
    let record_path = crate::checkpoint::scope_relative_artifact_path(&saved.path);
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
        path: record_path.clone(),
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
        min_observed_val_bpb,
        ema_bpb,
        git_sha: git_sha.clone(),
        git_provenance: git_provenance.to_string(),
        git_dirty,
        corpus: corpus.clone(),
        // Schema 7. Absent when the variable is unset or blank, never `""`:
        // the same `env_nonempty` rule `neon_writer` applies to the nullable
        // ledger columns, finally applied to the evidence document those
        // columns are derived from.
        run_id: crate::checkpoint::resolve_run_id(),
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
        // Schema 5. The same bool that was just hashed into byte 124 of the
        // header, not a re-read of `TRIOS_GF16_DISABLE`: it gates the in-place
        // `gf16_floor()` rewrite, so it decides the weights.
        gf16_enabled: meta.gf16_enabled,
        // Schema 6. The plan `bpb` was measured on, taken from the `EvalStats`
        // that produced that number and NOT re-derived here from
        // `eval_chunks_target()`, where an env change mid-run would make the
        // sidecar describe a grid the reading never came from.
        eval_chunks: Some(eval.plan.chunks as u32),
        eval_tokens: Some(eval.tokens() as u64),
        eval_seq: Some(eval.plan.seq as u32),
        val_bpb_stderr: eval.stderr.map(|s| s as f64),
        optimizer_params: Some(opt_params.clone()),
        // Schema 9. Derived from the SAME string that was just hashed into
        // bytes 136..152 of the header, not from a second read of
        // `TRIOS_FORMAT_TYPE`: the sidecar has to qualify the label the
        // artifact actually carries, and an env change mid-run would otherwise
        // let the two disagree. `resolve_fake_quant_format` refuses an
        // unfaithful format outright, so this is `false` only when the operator
        // set `TRIOS_ALLOW_UNFAITHFUL_FORMAT=1` - in which case the artifact
        // states its own retraction instead of leaving the reader to notice
        // that an `fp80` payload is bit-identical to the `f32` control.
        format_faithful: crate::checkpoint::format_label_faithful(&meta.fake_quant_format),
    };

    crate::checkpoint::write_sidecar_scoped(&record("pending"), seed_scope)?;
    let outcome = crate::neon_writer::checkpoint_record(
        canon,
        meta.seed as i32,
        meta.step as i64,
        // The SAME string the sidecar carries. A ledger row and the sidecar
        // beside the file must not spell one artifact two ways.
        &record_path,
        &saved.sha256,
        saved.bytes as i64,
        &meta.optimizer,
        model.hidden as i32,
        CHECKPOINT_FORMAT_VERSION as i32,
        meta.data_synthetic,
        bpb,
    );
    crate::checkpoint::write_sidecar_scoped(&record(outcome.as_str()), seed_scope)?;

    eprintln!(
        "[ckpt] {} sha256={} bytes={} ledger={} eval_chunks={} eval_tokens={} coverage={:.4}",
        path_str,
        saved.sha256,
        saved.bytes,
        outcome.as_str(),
        eval.plan.chunks,
        eval.tokens(),
        eval.coverage(val_len)
    );
    // Returned, not dropped: the resume record beside this file has to carry
    // the digest of THESE bytes, and re-hashing the path afterwards would
    // reopen the gap this function exists to close.
    Ok(saved)
}

/// Assemble the resume record for the artifact just written.
///
/// Every field comes from the values this run is EXECUTING with - `args`, the
/// resolved knobs, the corpus hashes taken at startup, the digest
/// `emit_checkpoint` computed by re-reading the file - and none of it is
/// re-read from the environment here, where an export mid-run would let the
/// record describe a recipe the moments were never produced under.
#[allow(clippy::too_many_arguments)]
fn build_resume_record(
    args: &TrainArgs,
    step: u64,
    rng_s: u64,
    hidden: usize,
    d_model: usize,
    attn_layers: u8,
    gf16_enabled: bool,
    gf16_floor_every: usize,
    data_synthetic: bool,
    fake_quant_format: &str,
    corpus: &crate::checkpoint::CorpusProvenance,
    weight_sha256: &str,
    ema_bpb: Option<f64>,
    min_observed_val_bpb: Option<f64>,
    optimizers: Vec<crate::checkpoint::ResumeOptimizerState>,
) -> crate::checkpoint::ResumeRecord {
    crate::checkpoint::ResumeRecord {
        seed: args.seed,
        step,
        rng_s,
        steps_total: args.steps as u64,
        eval_every: args.eval_every as u64,
        gf16_floor_every: gf16_floor_every as u64,
        hidden: hidden as u32,
        d_model: d_model as u32,
        num_attn_layers: attn_layers as u32,
        vocab: VOCAB as u32,
        dim: DIM as u32,
        num_ctx: NUM_CTX as u32,
        base_lr: args.lr,
        weight_decay: TRAIN_LOOP_WEIGHT_DECAY as f32,
        gf16_enabled,
        data_synthetic,
        ema_bpb,
        min_observed_val_bpb,
        weight_sha256: weight_sha256.to_string(),
        train_sha256: corpus.train.sha256.clone(),
        val_sha256: corpus.val.sha256.clone(),
        fake_quant_format: fake_quant_format.to_string(),
        optimizer: "adamw".to_string(),
        optimizers,
    }
}

/// Write the resume record beside the artifact and say so on stderr.
///
/// The `[resume]` line mirrors the `[ckpt]` line above it, digest included, so
/// an operator watching a run can see that the audit surface exists rather
/// than discovering at audit time that it does not.
fn emit_resume_record(
    saved: &crate::checkpoint::SavedCheckpoint,
    rec: &crate::checkpoint::ResumeRecord,
) -> Result<()> {
    let out = crate::checkpoint::save_resume(&saved.path, rec)?;
    eprintln!(
        "[resume] {} sha256={} bytes={} step={} of {} instances={} pairs_with={}",
        out.path.to_string_lossy(),
        out.sha256,
        out.bytes,
        rec.step,
        rec.steps_total,
        rec.optimizers.len(),
        rec.weight_sha256
    );
    Ok(())
}

/// Save the weights as initialised, before the first optimizer step.
///
/// A thin front door onto `emit_checkpoint` - it adds no hashing, no sidecar
/// and no ledger logic of its own - whose whole job is to fix the three fields
/// that a step-0 record must not guess:
///
///   * `step: 0` in `meta`, checked here rather than trusted, so the file is
///     `0.bin` and the record says `"step": 0`;
///   * `min_observed_val_bpb: None` and `ema_bpb: None`, because neither
///     quantity exists before the run has taken a second reading. `init_bpb`
///     is not a run minimum and is not an EMA, and putting it in either field
///     would make the record claim a measurement that was never made;
///   * `final_val_bpb: Some(init_bpb)`, which IS measured - `init_stats` is the
///     evaluation of these exact bytes, taken by the caller a few lines above.
#[allow(clippy::too_many_arguments)]
fn emit_init_checkpoint(
    model: &HybridModel,
    args: &TrainArgs,
    canon: &str,
    seed_scope: Option<u64>,
    meta: &CheckpointMeta,
    corpus: &crate::checkpoint::CorpusProvenance,
    gf16_every: usize,
    init_stats: &EvalStats,
    val_len: usize,
    opt_params: &crate::checkpoint::OptimizerParams,
    init_bpb: f32,
) -> Result<crate::checkpoint::SavedCheckpoint> {
    anyhow::ensure!(
        meta.step == 0,
        "emit_init_checkpoint called with step={}; the initial-weights artifact \
         is step 0 by definition",
        meta.step
    );
    emit_checkpoint(
        model,
        canon,
        seed_scope,
        meta,
        corpus,
        &RunKnobs {
            steps_total: args.steps as u64,
            gf16_floor_every: gf16_every as u64,
            eval_every: args.eval_every as u64,
        },
        init_stats,
        val_len,
        opt_params,
        Some(init_bpb as f64),
        None,
        None,
    )
    .context("initial-weights checkpoint failed")
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

/// The window grid `evaluate` will walk: the sampling plan, as a value.
///
/// Extracted so the plan is a thing that can be recorded and printed instead of
/// two hardcoded literals buried in a loop bound. `assert_train_val_disjoint`,
/// `evaluate` and the checkpoint sidecar all read the same plan, so they cannot
/// disagree about what was measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EvalPlan {
    /// Window length in tokens, `SEQ + 1`.
    pub seq: usize,
    /// Stride between window starts. Equal to `seq` at full coverage.
    pub stride: usize,
    /// Number of windows the walk will visit.
    pub chunks: usize,
    /// The requested coverage this plan was resolved from; `0` = full.
    pub target: usize,
}

/// Resolve the window grid for a stream of `len` tokens at `target` coverage.
///
/// `None` means the stream is too short for even one window - not a
/// measurement, and never silently rounded up to one.
///
/// `target == 0` is full coverage: non-overlapping windows tiling the stream,
/// every byte read exactly once, no sampling error. Any other `target` keeps
/// the pre-existing grid EXACTLY (evenly spaced starts, `max_start / target`
/// apart, capped at `target` windows) so that a run at the default 40 is
/// bit-comparable with every run recorded before this parameter existed.
pub(crate) fn eval_plan(len: usize, target: usize) -> Option<EvalPlan> {
    let seq = SEQ + 1;
    let max_start = len.saturating_sub(seq);
    if max_start == 0 {
        return None;
    }
    let stride = if target == 0 {
        seq
    } else if max_start >= target * seq {
        max_start / target
    } else {
        seq
    };
    let stride = stride.max(1);
    let mut chunks = max_start.div_ceil(stride);
    if target > 0 {
        chunks = chunks.min(target);
    }
    Some(EvalPlan {
        seq,
        stride,
        chunks,
        target,
    })
}

/// How many chunks `evaluate` would average over for a stream of `len` tokens.
///
/// Mirrors `eval_plan` because it IS `eval_plan`; kept as a name because the
/// size guard in `assert_train_val_disjoint` asks exactly this question.
pub fn eval_chunk_count(len: usize, target: usize) -> usize {
    eval_plan(len, target).map(|p| p.chunks).unwrap_or(0)
}

/// One BPB reading and the spread of the per-window readings behind it.
///
/// A mean with no `n` and no dispersion is not a measurement result, which is
/// what the record used to carry: seventeen digits of `val_bpb`, compared
/// against a champion at the fourth decimal, over an unstated 5% sample.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EvalStats {
    /// Mean bits-per-byte over the windows in `plan`.
    pub mean: f32,
    /// Sample standard deviation (n-1) of the per-window readings. `None` for
    /// a single window, where the spread is undefined - NOT 0.0, which would
    /// read as a perfectly repeatable measurement.
    pub stdev: Option<f32>,
    /// `(stdev / sqrt(n)) * sqrt(1 - n/N)`: the standard error of THIS mean,
    /// finite-population corrected against the `N` windows full coverage of the
    /// same stream would have walked.
    ///
    /// The correction is not cosmetic. Without it the field reported
    /// `s / sqrt(n)`, which is the standard error of a mean drawn from an
    /// INFINITE population; the val stream is finite and the grid samples it
    /// WITHOUT replacement, so at full coverage (`n == N`, every byte read
    /// exactly once) there is no sampling error left and the field must be
    /// exactly `0.0`. Measured before this change, a `TRIOS_EVAL_CHUNKS=0` run
    /// over 775 of 775 windows still wrote `val_bpb_stderr = 0.0117720` - the
    /// spread BETWEEN windows, published as the uncertainty OF the mean, while
    /// `eval_chunks_target`'s own doc comment called that setting "the only
    /// setting with no sampling error at all". The doc and the artifact
    /// disagreed and the artifact was the wrong one. `stdev` is untouched: the
    /// between-window spread is a real quantity, it is simply not this one.
    ///
    /// Still WITHIN-GRID: it is the error of the mean over THIS fixed aliased
    /// grid, and says nothing about the much larger spread between grids. See
    /// `docs/EVAL-UNCERTAINTY.md`.
    pub stderr: Option<f32>,
    /// The grid actually walked. `plan.chunks` is `n`.
    pub plan: EvalPlan,
}

impl EvalStats {
    /// Tokens the eval actually looked at.
    pub fn tokens(&self) -> usize {
        self.plan.chunks * self.plan.seq
    }
    /// Fraction of `stream_len` the eval looked at, as a plain ratio.
    pub fn coverage(&self, stream_len: usize) -> f64 {
        if stream_len == 0 {
            return 0.0;
        }
        self.tokens() as f64 / stream_len as f64
    }
}

/// Mean bits-per-byte over the `target` window grid of `tokens`, with spread.
///
/// `None` means the evaluation could not be performed. A chunk that cannot be
/// measured invalidates the whole eval rather than being quietly dropped from
/// the average: silently skipping chunks is how an eval over a degenerate
/// corpus still produced a confident-looking number (#62).
fn evaluate(model: &HybridModel, tokens: &[usize], target: usize) -> Option<EvalStats> {
    let plan = eval_plan(tokens.len(), target)?;
    let max_start = tokens.len() - plan.seq;
    let mut readings: Vec<f32> = Vec::with_capacity(plan.chunks);
    for c in (0..max_start).step_by(plan.stride).take(plan.chunks) {
        let end = (c + plan.seq).min(tokens.len());
        let loss = model.loss_on_seq(&tokens[c..end])?;
        if !loss.is_finite() {
            return None;
        }
        readings.push(loss / LN_2);
    }
    let n = readings.len();
    if n == 0 {
        return None;
    }
    let mean = readings.iter().sum::<f32>() / n as f32;
    // Two-pass, in f64: the one-pass sum-of-squares form loses the whole
    // difference when the readings sit near 3.0 and differ in the third
    // decimal, which is precisely the regime being measured.
    let (stdev, stderr) = if n >= 2 {
        let var = readings
            .iter()
            .map(|&r| {
                let d = r as f64 - mean as f64;
                d * d
            })
            .sum::<f64>()
            / (n as f64 - 1.0);
        let s = var.sqrt();
        // Finite-population correction. `N` is the number of windows FULL
        // coverage of this same stream would walk - the plan resolved at
        // target 0 - so the ratio n/N is the fraction of the population this
        // grid actually read. `eval_plan` can never return more chunks than
        // that (any target > 0 has stride >= seq), so the factor is in [0, 1];
        // it is clamped anyway rather than trusting that argument to survive a
        // future change to the plan.
        let population = eval_chunk_count(tokens.len(), 0).max(n);
        let fpc = (1.0 - (n as f64) / (population as f64))
            .clamp(0.0, 1.0)
            .sqrt();
        (Some(s as f32), Some((s / (n as f64).sqrt() * fpc) as f32))
    } else {
        (None, None)
    };
    Some(EvalStats {
        mean,
        stdev,
        stderr,
        plan: EvalPlan { chunks: n, ..plan },
    })
}

/// The eval plan and its uncertainty, as stdout tokens.
///
/// A NEW line with a new leading token. The `DONE:` line printed by
/// `bin/trios-train` is deliberately untouched, format and all: parsers keyed
/// on `DONE: ... bpb=<4dp>` must keep working, so this appends rather than
/// widens.
fn print_eval_uncertainty(seed: u64, step: usize, stats: &EvalStats, val_len: usize) {
    let fmt = |v: Option<f32>| match v {
        Some(x) => format!("{x:.6}"),
        None => "unmeasured".to_string(),
    };
    println!(
        "EVAL-PLAN: seed={} step={} val_bpb_mean={:.6} val_bpb_stdev={} val_bpb_stderr={} \
         eval_chunks={} eval_chunks_target={} eval_seq={} eval_stride={} eval_tokens={} \
         val_tokens={} coverage={:.4}",
        seed,
        step,
        stats.mean,
        fmt(stats.stdev),
        fmt(stats.stderr),
        stats.plan.chunks,
        stats.plan.target,
        stats.plan.seq,
        stats.plan.stride,
        stats.tokens(),
        val_len,
        stats.coverage(val_len)
    );
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
}

/// Reject a BPB reading that cannot be a real measurement of held-out text.
///
/// The crate already owns this law in `race::bpb` / `race::victory`, but
/// `train_loop` used to print and ship numbers its own tracker would refuse.
/// Measured calibration on the verified byte-disjoint tinyshakespeare split:
/// ~7.00 at init, ~3.33 at step 1000, 2.75-2.83 at step 12000. Even a 100%
/// verbatim train/val overlap only reaches 2.71 on this architecture, so a
/// near-zero reading is never a good model - it is a degenerate eval corpus.
///
/// The bound below is `invariants::PUBLISHED_BPB_FLOOR` (2.0), derived from
/// exactly that calibration, and NOT `race::victory::JEPA_PROXY_BPB_FLOOR`
/// (0.1). Every value that reaches this function is printed, returned as the
/// run's result and written to `ssot.bpb_samples`; guarding it with the
/// proxy-artefact detector left the bound 26x below what the docstring three
/// lines up already implied, which is how the retracted 1.5492 passed.
fn guard_bpb(vbpb: f32, step: usize) -> Result<f32> {
    anyhow::ensure!(
        vbpb.is_finite(),
        "val_bpb is not finite at step {step}; refusing to emit"
    );
    let floor = crate::invariants::PUBLISHED_BPB_FLOOR;
    anyhow::ensure!(
        vbpb > floor,
        "val_bpb={vbpb:.6} <= PUBLISHED_BPB_FLOOR {floor} at step {step}: below \
         anything this architecture can honestly reach (100% verbatim train/val \
         overlap still only reaches ~2.71), so the eval corpus is degenerate or \
         duplicated, not the model perfect. This is the signature that \
         mislabelled 179 ledger rows as data leaks (#62)."
    );
    Ok(vbpb)
}

pub fn run_single(args: &TrainArgs) -> Result<RunOutcome> {
    run_single_resumed(args, None)
}

/// `run_single`, optionally warm-started from a `{step}.bin` + `{step}.resume`
/// pair written by an earlier run of the SAME recipe.
///
/// This is the window-audit entry point. `ckpt_replay` re-executes from step 0
/// because the `TRIOSCKP` container carries weights only, so verifying one
/// claim cost an auditor the vendor's entire training budget; with a resume
/// record an auditor re-executes one challenged window of N steps out of T.
/// See `docs/WINDOW-AUDIT.md` for what that does and does not establish - in
/// particular it is a SAME-MACHINE claim, and the cross-architecture boundary
/// documented in `docs/CROSS-ARCH-DIVERGENCE.md` is unchanged by it.
///
/// `resume_from` may name either half of the pair. Everything that decides the
/// weights is checked before the first gradient - weight digest, corpus
/// digests, shape, both cadences, seed, total steps, lr, weight decay, GF16
/// and the QAT format - and a mismatch is an `Err`, never a fresh optimizer.
pub fn run_single_resumed(args: &TrainArgs, resume_from: Option<&Path>) -> Result<RunOutcome> {
    // Before anything is opened, allocated or measured: `--eval-every 0` used
    // to reach `step % 0` and abort with exit 101 naming an arithmetic
    // operation. See `refuse_eval_every_or_exit`.
    refuse_eval_every_or_exit(args.eval_every);
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
    // Wave 31 PR-B: the ONE resolution of the GF16 knob. It gates the in-place
    // `gf16_floor()` rewrite below and is what the sidecar records, so a run
    // cannot execute one value and report the other.
    let gf16_on = resolve_gf16_knob()?;
    if std::env::var("GF16_ENABLED").is_ok() {
        eprintln!("[arch-knob] GF16_ENABLED={gf16_on} (override, Wave-31 PR-B)");
    }
    // `gf16_floor_every` is part of the RECIPE, not of the observation, so it
    // is banner-visible next to seed/steps and recorded in the sidecar.
    let gf16_every = gf16_floor_every() as usize;
    // Resolved ONCE, and banner-visible for the same reason `gf16_floor_every`
    // is: a typo in the cadence must be readable at step 0, not inferred from
    // an empty directory at step 12000.
    let ckpt_every = resolve_checkpoint_every()?;
    let ckpt_init = checkpoint_init_enabled();
    eprintln!(
        "=== trios-train seed={} steps={} hidden={} lr={:.4} attn_layers={} \
         eval_every={} gf16_enabled={} gf16_floor_every={} checkpoint_every={} \
         checkpoint_init={} ===",
        args.seed,
        args.steps,
        eff_hidden,
        args.lr,
        eff_attn_layers,
        args.eval_every,
        gf16_on,
        gf16_every,
        ckpt_every,
        ckpt_init
    );
    // #509 Phase-1b: wire QAT into the production `trios-train` path.
    // `scarab` spawns this binary and sets `TRIOS_FORMAT_TYPE`; previously
    // only `cpu_train` honoured it, so production traffic was F32 regardless.
    //
    // Resolved HERE, before the corpus is read and before the optimizer is
    // bound, for the reason `matrix_runner` states at its own guard: a run that
    // is going to be refused for its label should cost no compute at all. The
    // refusals it can raise are an unrecognised spelling and an unfaithful
    // format; see `resolve_fake_quant_format`.
    let fq_fmt = resolve_fake_quant_format()?;
    if let Some(fmt) = fq_fmt {
        eprintln!("QAT: FakeQuant enabled for format {:?}", fmt);
    }
    // The label the header, the sidecar and the resume record all carry,
    // resolved once from `fq_fmt` rather than spelled out at each of the three
    // sites: a resume record whose format string disagreed with the artifact
    // beside it would be refused for a mismatch that never happened.
    let fq_label = fq_fmt
        .map(|f| f.name().to_string())
        .unwrap_or_else(|| "f32".to_string());

    // EPIC-446: resolve run identity ONCE. The same string names the checkpoint
    // directory and the ledger row, so the artifact and the BPB cannot drift apart.
    let canon = resolve_canon_name(args.seed);
    // Bind the optimizer that ACTUALLY executes, before step 1 and before any
    // artifact or ledger row can be minted under it. `trios-train` never bound,
    // so `neon_writer`'s anti-mislabelling guard had nothing to compare against
    // and fell back to the algo parsed out of the canon suffix - a label, not a
    // fact. An `Err` here is a canon name that names a different optimizer than
    // the one about to run, which must stop the run rather than be discovered
    // as a silent dropped row at the first checkpoint.
    crate::neon_writer::bind_executed_optimizer(&canon, "adamw")
        .map_err(|e| anyhow::anyhow!("[R5-honesty] {e}"))?;
    // R5-4: `Some(seed)` only under `run_sweep`, where one canon name covers
    // several runs. See `SWEEP_SEED_SCOPE`.
    let seed_scope = checkpoint_seed_scope(args.seed);

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

    // Either a cold start from the seed, or a warm start from a validated
    // pair. The two are kept in one expression so no later code has to ask
    // which happened: after this, `model` holds the weights the next step acts
    // on, and `resume` is `Some` only if every binding check passed.
    let (mut model, resume) = match resume_from {
        None => {
            let mut m = HybridModel::new(eff_hidden, args.seed, eff_attn_layers);
            if let Some(fmt) = fq_fmt {
                fake_quantize_model(&mut m, fmt);
            }
            (m, None)
        }
        Some(path) => {
            let (weights_path, resume_path) = crate::checkpoint::resolve_resume_pair(path)?;
            let rec = crate::checkpoint::load_resume_file(&resume_path)?;
            let raw = std::fs::read(&weights_path)
                .with_context(|| format!("failed to read resume weights {weights_path:?}"))?;
            let weight_sha256 = crate::checkpoint::sha256_hex(&raw);
            crate::checkpoint::verify_resume(
                &rec,
                &crate::checkpoint::ResumeExpectation {
                    seed: args.seed,
                    steps_total: args.steps as u64,
                    eval_every: args.eval_every as u64,
                    gf16_floor_every: gf16_every as u64,
                    gf16_enabled: gf16_on,
                    hidden: eff_hidden as u32,
                    d_model: DIM as u32,
                    num_attn_layers: eff_attn_layers as u32,
                    vocab: VOCAB as u32,
                    dim: DIM as u32,
                    num_ctx: NUM_CTX as u32,
                    base_lr: args.lr,
                    weight_decay: TRAIN_LOOP_WEIGHT_DECAY as f32,
                    fake_quant_format: fq_label.clone(),
                    weight_sha256: weight_sha256.clone(),
                    train_sha256: corpus.train.sha256.clone(),
                    val_sha256: corpus.val.sha256.clone(),
                },
            )?;
            let (m, meta) = HybridModel::from_checkpoint_bytes(&raw)
                .with_context(|| format!("resume weights {weights_path:?}"))?;
            // The header is a SECOND statement of the same facts, written by
            // the run that produced the weights rather than by the record
            // beside them. Checking the two against each other costs nothing.
            if meta.step != rec.step
                || meta.seed != rec.seed
                || meta.gf16_enabled != rec.gf16_enabled
                || meta.optimizer != rec.optimizer
                || m.hidden != eff_hidden
                || m.attn.config().num_attn_layers != eff_attn_layers
            {
                return Err(crate::checkpoint::resume_refusal(
                    crate::checkpoint::RESUME_REASON_SHAPE,
                    format!(
                        "the checkpoint header says step={} seed={} gf16={} optimizer={:?} \
                         hidden={} layers={}; the resume record says step={} seed={} gf16={} \
                         optimizer={:?} and this run wants hidden={} layers={}",
                        meta.step,
                        meta.seed,
                        meta.gf16_enabled,
                        meta.optimizer,
                        m.hidden,
                        m.attn.config().num_attn_layers,
                        rec.step,
                        rec.seed,
                        rec.gf16_enabled,
                        rec.optimizer,
                        eff_hidden,
                        eff_attn_layers
                    ),
                ));
            }
            eprintln!(
                "[resume] warm start from {weights_path:?} sha256={weight_sha256} step={} of {} \
                 (record {resume_path:?})",
                rec.step, rec.steps_total
            );
            // NOT re-quantized: these weights came out of a loop that already
            // applied the STE at the end of the step that wrote them, so a
            // second pass would be a rounding this run performed and the
            // monolithic run did not.
            (m, Some(rec))
        }
    };
    let d = model.attn.config().d_model;
    let dd = d * d;
    let attn_total = 8 * dd;
    // Named so `unhonoured_fields` grades `optimizer.weight_decay` against the
    // value this loop actually applies, and cannot drift away from it.
    let wd = TRAIN_LOOP_WEIGHT_DECAY as f32;
    let mut opt_embed = AdamW::new(VOCAB * DIM, wd);
    let mut opt_ctx: Vec<AdamW> = (0..NUM_CTX).map(|_| AdamW::new(VOCAB * DIM, wd)).collect();
    let mut opt_proj = AdamW::new(eff_hidden * DIM, wd);
    let mut opt_attn_down = AdamW::new(d * eff_hidden, wd);
    let mut opt_attn_up = AdamW::new(eff_hidden * d, wd);
    let mut opt_head = AdamW::new(VOCAB * eff_hidden, wd);
    let mut opt_attn_w = AdamW::new(attn_total, wd);
    if let Some(rec) = &resume {
        restore_adamw_state(
            rec,
            &mut opt_embed,
            &mut opt_ctx,
            &mut opt_proj,
            &mut opt_attn_down,
            &mut opt_attn_up,
            &mut opt_head,
            &mut opt_attn_w,
        )?;
        eprintln!(
            "[resume] restored {} optimizer instances ({} moment elements)",
            rec.optimizers.len(),
            rec.optimizers.iter().map(|o| 2 * o.m.len()).sum::<usize>()
        );
    }
    let opt_params = adamw_record_params(wd, "train_loop::AdamW");

    // The declared sampling plan, resolved ONCE for the run: an eval whose
    // coverage changed between step 0 and the final step would be comparing
    // two different measurements.
    let eval_chunks = eval_chunks_target();
    let init_stats = evaluate(&model, &val, eval_chunks)
        .ok_or_else(|| anyhow::anyhow!("initial eval produced no measurable chunk"))?;
    let init_bpb = guard_bpb(init_stats.mean, 0)?;
    eprintln!(
        "Initial val_bpb={:.4} over {} windows of {} tokens ({:.2}% of {} val tokens)",
        init_bpb,
        init_stats.plan.chunks,
        init_stats.plan.seq,
        init_stats.coverage(val.len()) * 100.0,
        val.len()
    );
    print_eval_uncertainty(args.seed, 0, &init_stats, val.len());
    // On a warm start the EMA and the running minimum continue the trajectory
    // the record captured. They are the run's reported numbers, not its
    // weights: restoring them is what makes a segmented run REPORT what the
    // monolithic run reports, and `init_bpb` here is a reading of the RESUMED
    // weights, which is not where that trajectory was.
    let mut ema_bpb = resume
        .as_ref()
        .and_then(|r| r.ema_bpb)
        .map(|v| v as f32)
        .unwrap_or(init_bpb);
    // Raw readings, tracked separately from the EMA. `min_observed_val_bpb` is
    // the minimum over the readings this run TOOK - optimistic by construction,
    // see the field doc on `CheckpointRecord`; `init_bpb` is excluded because
    // it describes the initialization, not the run.
    let mut best_val_bpb: Option<f32> = resume
        .as_ref()
        .and_then(|r| r.min_observed_val_bpb)
        .map(|v| v as f32);
    let mut final_val_bpb: Option<f32> = None;
    // Proof that this run left something behind. A run that never enters the
    // step loop (`--steps 0`) skipped every emission point silently and still
    // exited 0 - the exact shape of "1,851 experiments, zero artifacts".
    let mut artifact_emitted = false;
    let warmup = args.steps / 10;
    let accum = 4;
    // The batch sampler. On a warm start it comes from the record: this single
    // u64 decides which windows of the corpus every remaining step trains on,
    // so a resumed run that re-derived it from the seed would train on the
    // FIRST window sequence again and diverge immediately while still printing
    // a healthy-looking BPB curve.
    let mut rng_s = match &resume {
        Some(r) => r.rng_s,
        None => args.seed.wrapping_add(7919),
    };
    // Steps already executed. The loop below runs `start_step + 1 ..= steps`,
    // and `cosine_lr` is a function of `(step, args.steps)`, so the resumed
    // segment sits on the same schedule the monolithic run would have applied
    // to the same step numbers.
    let start_step = resume.as_ref().map(|r| r.step as usize).unwrap_or(0);
    let t0 = Instant::now();
    let gf16_floor_step = (0.7 * args.steps as f32).floor() as usize;
    let nca = NcaObjective::default();
    let mut last_nca_entropy = 0.0f64;

    // The step-0 artifact: the weights as initialised, before a single gradient
    // has been applied. It goes through `emit_checkpoint` like every other
    // artifact, so the hashing, the sidecar and the ledger row are the same
    // code, not a second implementation.
    //
    // It deliberately does NOT set `artifact_emitted`: a run that saved nothing
    // but its own initialisation still produced no TRAINED artifact, and the
    // guard at the end of this function exists to catch exactly that.
    //
    // Never on a warm start: these weights are step `start_step`, not an
    // initialisation, and writing them as `0.bin` would mint an artifact whose
    // header states a step it was not taken at.
    if ckpt_init && checkpoint_enabled() && resume.is_some() {
        eprintln!(
            "[resume] TRIOS_CHECKPOINT_INIT is set but this is a warm start at step \
             {start_step}; no 0.bin will be written, because these weights are not an \
             initialisation."
        );
    }
    if ckpt_init && checkpoint_enabled() && resume.is_none() {
        let saved = emit_init_checkpoint(
            &model,
            args,
            &canon,
            seed_scope,
            &CheckpointMeta {
                seed: args.seed,
                step: 0,
                train_lr: args.lr,
                attn_scale: attn_scale(),
                attn_seq: attn_seq_override() as u32,
                gf16_enabled: gf16_on,
                data_synthetic,
                optimizer: "adamw".into(),
                fake_quant_format: fq_label.clone(),
            },
            &corpus,
            gf16_every,
            &init_stats,
            val.len(),
            &opt_params,
            init_bpb,
        )?;
        // The step-0 warm start: zeroed moments, zeroed instance counters and
        // the sampler state before the first draw. It is genuine state, not a
        // placeholder, so an auditor challenging the FIRST window starts from
        // the same kind of artifact as one challenging any later window.
        emit_resume_record(
            &saved,
            &build_resume_record(
                args,
                0,
                rng_s,
                eff_hidden,
                d,
                eff_attn_layers,
                gf16_on,
                gf16_every,
                data_synthetic,
                &fq_label,
                &corpus,
                &saved.sha256,
                None,
                None,
                capture_adamw_state(
                    &opt_embed,
                    &opt_ctx,
                    &opt_proj,
                    &opt_attn_down,
                    &opt_attn_up,
                    &opt_head,
                    &opt_attn_w,
                ),
            ),
        )?;
    }

    for step in start_step + 1..=args.steps {
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
        if gf16_on && step >= gf16_floor_step && step % gf16_every == 0 {
            gf16_floor(&mut model.embed);
            gf16_floor(&mut model.proj);
            gf16_floor(&mut model.lm_head);
            for c in &mut model.ctx {
                gf16_floor(c);
            }
        }

        // The two cadences are INDEPENDENT. `TRIOS_CHECKPOINT_EVERY` used to
        // live inside the `step % args.eval_every == 0` guard, so the requested
        // artifact cadence was silently intersected with the observation
        // cadence: `TRIOS_CHECKPOINT_EVERY=50 --eval-every 100 --steps 200`
        // wrote 100.bin and 200.bin and exited 0, half the requested artifacts
        // never written and no warning. That is the same
        // observation-parameter-decides-the-artifact coupling already removed
        // from `gf16_floor` above, on the one function whose absence caused
        // "1,851 experiments, zero artifacts".
        let is_eval_step = step % args.eval_every == 0 || step == args.steps;
        let is_ckpt_step =
            checkpoint_enabled() && (step == args.steps || checkpoint_every_hit(step, ckpt_every));
        if is_eval_step || is_ckpt_step {
            let stats = evaluate(&model, &val, eval_chunks).ok_or_else(|| {
                anyhow::anyhow!("eval produced no measurable chunk at step {step}")
            })?;
            let vbpb = guard_bpb(stats.mean, step)?;
            // A checkpoint-only step measures its own weights - the artifact
            // must carry the reading these exact bytes produce - but it does
            // NOT touch the published trajectory. Folding it into the EMA, the
            // running best or the ledger would make the artifact cadence
            // change the reported numbers, which is the coupling this block
            // exists to break: two runs of the same recipe must stay
            // comparable however often they were snapshotted.
            if is_eval_step {
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
                // Appended, not merged into the line above: the existing line has
                // downstream parsers and the uncertainty is new information.
                print_eval_uncertainty(args.seed, step, &stats, val.len());
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
            }

            // EPIC-446: emit the artifact AFTER bpb_sample so the checkpoint
            // row carries the BPB these exact weights produced.
            if is_ckpt_step {
                let meta = CheckpointMeta {
                    seed: args.seed,
                    step: step as u64,
                    train_lr: args.lr,
                    attn_scale: attn_scale(),
                    attn_seq: attn_seq_override() as u32,
                    gf16_enabled: gf16_on,
                    data_synthetic,
                    optimizer: "adamw".into(),
                    fake_quant_format: fq_label.clone(),
                };
                let res = emit_checkpoint(
                    &model,
                    &canon,
                    seed_scope,
                    &meta,
                    &corpus,
                    &RunKnobs {
                        steps_total: args.steps as u64,
                        gf16_floor_every: gf16_every as u64,
                        eval_every: args.eval_every as u64,
                    },
                    &stats,
                    val.len(),
                    &opt_params,
                    Some(vbpb as f64),
                    best_val_bpb.map(|v| v as f64),
                    // On a checkpoint-only step the EMA describes the last
                    // EVAL step, not these weights. `None` says so; a stale
                    // number would read as a measurement of this artifact.
                    if is_eval_step {
                        Some(ema_bpb as f64)
                    } else {
                        None
                    },
                );
                // Asymmetric on purpose: a run that finishes with no artifact
                // is the failure this exists to prevent and must not exit 0,
                // but losing a whole run to a transient full disk mid-way
                // would be worse than a missing intermediate file.
                match res {
                    Ok(saved) => {
                        artifact_emitted = true;
                        // The resume record beside the artifact, carrying the
                        // digest of the bytes just written. Without it this
                        // `.bin` can only be checked by re-executing from step
                        // 0, which is the cost that makes nobody check.
                        //
                        // `ema_bpb` and `best_val_bpb` are recorded LIVE here,
                        // unlike the artifact record above which reports
                        // `None` for the EMA on a checkpoint-only step: this is
                        // trajectory STATE to be restored, not a measurement of
                        // these weights.
                        let rec = build_resume_record(
                            args,
                            step as u64,
                            rng_s,
                            eff_hidden,
                            d,
                            eff_attn_layers,
                            gf16_on,
                            gf16_every,
                            data_synthetic,
                            &fq_label,
                            &corpus,
                            &saved.sha256,
                            Some(ema_bpb as f64),
                            best_val_bpb.map(|v| v as f64),
                            capture_adamw_state(
                                &opt_embed,
                                &opt_ctx,
                                &opt_proj,
                                &opt_attn_down,
                                &opt_attn_up,
                                &opt_head,
                                &opt_attn_w,
                            ),
                        );
                        // Same asymmetry as the artifact itself, for the same
                        // reason: the final step must not ship an artifact
                        // nobody can warm-start from without saying so.
                        match emit_resume_record(&saved, &rec) {
                            Ok(()) => {}
                            Err(e) if step == args.steps => {
                                return Err(e.context("final-step resume record failed"))
                            }
                            Err(e) => eprintln!(
                                "[resume] WARNING: resume record at step {step} failed: {e:#}"
                            ),
                        }
                    }
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

    // A run that was asked for artifacts and produced none is a failure, and
    // must not be reported as a success. The in-loop guard above only fires on
    // a checkpoint that was ATTEMPTED; `--steps 0` never enters the loop at
    // all, so it printed a DONE line, wrote nothing and exited 0.
    anyhow::ensure!(
        !checkpoint_enabled() || artifact_emitted,
        "run finished with no checkpoint artifact (steps={}, eval_every={}): \
         checkpointing is enabled, so this run produced nothing to reproduce it \
         from. Set TRIOS_CHECKPOINT_DISABLE=1 if that is genuinely intended.",
        args.steps,
        args.eval_every
    );

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
    // See `run_single`: the same refusal, on the same knob, before any work.
    refuse_eval_every_or_exit(args.eval_every);
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
    // See `run_single`: the ONE resolution, gating the floor and recorded in
    // every sidecar this path emits.
    let gf16_on = resolve_gf16_knob()?;
    if std::env::var("GF16_ENABLED").is_ok() {
        eprintln!("[arch-knob] GF16_ENABLED={gf16_on} (override, Wave-31 PR-B)");
    }
    let label = if use_cwd { "MuonCwd" } else { "Muon" };
    let gf16_every = gf16_floor_every() as usize;
    // See `run_single`: resolved once, printed once, unparseable is fatal.
    let ckpt_every = resolve_checkpoint_every()?;
    let ckpt_init = checkpoint_init_enabled();
    eprintln!(
        "=== P1 {} seed={} steps={} hidden={} eval_every={} gf16_enabled={} \
         gf16_floor_every={} checkpoint_every={} checkpoint_init={} ===",
        label,
        args.seed,
        args.steps,
        eff_hidden,
        args.eval_every,
        gf16_on,
        gf16_every,
        ckpt_every,
        ckpt_init
    );
    // #509 Phase-1b: same QAT wiring for the Muon path, resolved at the same
    // point and for the same reason - see `run_single`.
    let fq_fmt = resolve_fake_quant_format()?;
    if let Some(fmt) = fq_fmt {
        eprintln!("QAT: FakeQuant enabled for format {:?}", fmt);
    }

    // EPIC-446: see the note in `run_single` - one canon string for both the
    // checkpoint directory and the ledger row.
    let canon = resolve_canon_name(args.seed);
    // See `run_single`: the executed optimizer is bound before step 1. The name
    // is the one `run_with_optimizer` dispatched on, so a `muon-cwd` run cannot
    // be recorded as `muon`.
    crate::neon_writer::bind_executed_optimizer(&canon, if use_cwd { "muon-cwd" } else { "muon" })
        .map_err(|e| anyhow::anyhow!("[R5-honesty] {e}"))?;
    // R5-4: see the note in `run_single`.
    let seed_scope = checkpoint_seed_scope(args.seed);

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
    // The proj matrix is stepped by `MuonOptimizer`, everything else by the
    // same `AdamW` as `run_single`; `source` says so rather than letting four
    // AdamW numbers stand for a mixed recipe.
    let opt_params = adamw_record_params(
        adamw_wd,
        "train_loop::AdamW (proj: optimizer::MuonOptimizer)",
    );

    let eval_chunks = eval_chunks_target();
    let init_stats = evaluate(&model, &val, eval_chunks)
        .ok_or_else(|| anyhow::anyhow!("initial eval produced no measurable chunk"))?;
    let init_bpb = guard_bpb(init_stats.mean, 0)?;
    eprintln!(
        "Initial val_bpb={:.4} over {} windows of {} tokens ({:.2}% of {} val tokens)",
        init_bpb,
        init_stats.plan.chunks,
        init_stats.plan.seq,
        init_stats.coverage(val.len()) * 100.0,
        val.len()
    );
    print_eval_uncertainty(args.seed, 0, &init_stats, val.len());
    let mut ema_bpb = init_bpb;
    // See `run_single`: raw readings are tracked separately from the EMA.
    let mut best_val_bpb: Option<f32> = None;
    let mut final_val_bpb: Option<f32> = None;
    // Proof that this run left something behind. A run that never enters the
    // step loop (`--steps 0`) skipped every emission point silently and still
    // exited 0 - the exact shape of "1,851 experiments, zero artifacts".
    let mut artifact_emitted = false;
    let warmup = args.steps / 10;
    let accum = 4;
    let mut rng_s = args.seed.wrapping_add(7919);
    let t0 = Instant::now();
    let gf16_floor_step = (0.7 * args.steps as f32).floor() as usize;
    let nca = NcaObjective::default();
    let mut last_nca_entropy = 0.0f64;

    // See `run_single`: the step-0 artifact, on the same emission path, and
    // deliberately not counted as `artifact_emitted`.
    if ckpt_init && checkpoint_enabled() {
        emit_init_checkpoint(
            &model,
            args,
            &canon,
            seed_scope,
            &CheckpointMeta {
                seed: args.seed,
                step: 0,
                train_lr: args.lr,
                attn_scale: attn_scale(),
                attn_seq: attn_seq_override() as u32,
                gf16_enabled: gf16_on,
                data_synthetic,
                optimizer: if use_cwd {
                    "muon-cwd".into()
                } else {
                    "muon".into()
                },
                fake_quant_format: fq_fmt
                    .map(|f| f.name().to_string())
                    .unwrap_or_else(|| "f32".into()),
            },
            &corpus,
            gf16_every,
            &init_stats,
            val.len(),
            &opt_params,
            init_bpb,
        )?;
    }

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
        if gf16_on && step >= gf16_floor_step && step % gf16_every == 0 {
            gf16_floor(&mut model.embed);
            gf16_floor(&mut model.proj);
            gf16_floor(&mut model.lm_head);
            for c in &mut model.ctx {
                gf16_floor(c);
            }
        }

        // See `run_single`: the artifact cadence is independent of the eval
        // cadence, and a checkpoint-only step measures its own weights without
        // disturbing the published trajectory.
        let is_eval_step = step % args.eval_every == 0 || step == args.steps;
        let is_ckpt_step =
            checkpoint_enabled() && (step == args.steps || checkpoint_every_hit(step, ckpt_every));
        if is_eval_step || is_ckpt_step {
            let stats = evaluate(&model, &val, eval_chunks).ok_or_else(|| {
                anyhow::anyhow!("eval produced no measurable chunk at step {step}")
            })?;
            let vbpb = guard_bpb(stats.mean, step)?;
            if is_eval_step {
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
                // See `run_single`: appended, never merged into the line above.
                print_eval_uncertainty(args.seed, step, &stats, val.len());
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
            }

            // EPIC-446: same emission point as `run_single`. The optimizer
            // label comes from which entry point ran, not from canon_name.
            if is_ckpt_step {
                let meta = CheckpointMeta {
                    seed: args.seed,
                    step: step as u64,
                    train_lr: args.lr,
                    attn_scale: attn_scale(),
                    attn_seq: attn_seq_override() as u32,
                    gf16_enabled: gf16_on,
                    data_synthetic,
                    optimizer: if use_cwd {
                        "muon-cwd".into()
                    } else {
                        "muon".into()
                    },
                    fake_quant_format: fq_fmt
                        .map(|f| f.name().to_string())
                        .unwrap_or_else(|| "f32".into()),
                };
                let res = emit_checkpoint(
                    &model,
                    &canon,
                    seed_scope,
                    &meta,
                    &corpus,
                    &RunKnobs {
                        steps_total: args.steps as u64,
                        gf16_floor_every: gf16_every as u64,
                        eval_every: args.eval_every as u64,
                    },
                    &stats,
                    val.len(),
                    &opt_params,
                    Some(vbpb as f64),
                    best_val_bpb.map(|v| v as f64),
                    // On a checkpoint-only step the EMA describes the last
                    // EVAL step, not these weights. `None` says so; a stale
                    // number would read as a measurement of this artifact.
                    if is_eval_step {
                        Some(ema_bpb as f64)
                    } else {
                        None
                    },
                );
                // `Ok(_)`, not `Ok(())`: `emit_checkpoint` now hands back the
                // `SavedCheckpoint` so the AdamW path can write a resume
                // record beside the artifact. This path deliberately does NOT
                // write one - `MuonOptimizer` carries a momentum buffer and a
                // step counter that the format does not serialise - and
                // `--resume-from` is refused for it in
                // `run_with_optimizer_resumed` before any work starts.
                match res {
                    Ok(_) => artifact_emitted = true,
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

    // A run that was asked for artifacts and produced none is a failure, and
    // must not be reported as a success. The in-loop guard above only fires on
    // a checkpoint that was ATTEMPTED; `--steps 0` never enters the loop at
    // all, so it printed a DONE line, wrote nothing and exited 0.
    anyhow::ensure!(
        !checkpoint_enabled() || artifact_emitted,
        "run finished with no checkpoint artifact (steps={}, eval_every={}): \
         checkpointing is enabled, so this run produced nothing to reproduce it \
         from. Set TRIOS_CHECKPOINT_DISABLE=1 if that is genuinely intended.",
        args.steps,
        args.eval_every
    );

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

// -- CLI policy helpers -----------------------------------------------------
//
// These live in the library rather than in `src/bin/trios-train.rs` for one
// reason: `cargo test --lib` is the gate every change runs, and a refusal that
// exists only inside a binary crate is a refusal no lib test can prove. Each is
// a pure function of its arguments - none of them reads the environment - so
// the tests at the bottom of this file cannot race the process-wide env that
// other tests mutate.

/// Optimizers this crate can actually execute.
///
/// Deliberately the same list as `neon_writer::ALGO_WHITELIST` (asserted by
/// `supported_optimizers_match_the_write_side_whitelist`): an optimizer whose
/// results could never be published is not an optimizer worth starting a run
/// with.
pub const SUPPORTED_OPTIMIZERS: &[&str] = &["adamw", "muon", "muon-cwd"];

/// The single refusal point for an unsupported `--optimizer`.
///
/// The R5-honest dispatch used to live only in `trios-train`'s single-seed
/// arm, so `--sweep --optimizer soap` ran AdamW, printed three `DONE:` lines,
/// printed the headline `GATE-2:` verdict and exited 0 - on the branch that
/// produces the gate verdict, which is the branch whose output gets published.
/// Both arms now go through this function before any seed trains.
pub fn ensure_supported_optimizer(optimizer: &str) -> Result<()> {
    if SUPPORTED_OPTIMIZERS.contains(&optimizer) {
        return Ok(());
    }
    Err(anyhow::anyhow!(
        "[R5-honesty] unsupported optimizer={optimizer:?}: only {SUPPORTED_OPTIMIZERS:?} \
         are implemented. Refusing silent AdamW fallback. Fix env, redeploy, or \
         implement the optimizer first."
    ))
}

/// Run one seed under the named optimizer.
///
/// The one place that maps an optimizer name onto a training loop, so the
/// sweep arm and the single-seed arm cannot dispatch differently.
pub fn run_with_optimizer(optimizer: &str, args: &TrainArgs) -> Result<RunOutcome> {
    run_with_optimizer_resumed(optimizer, args, None)
}

/// `run_with_optimizer`, with the window-audit warm start.
///
/// The Muon arms REFUSE a warm start rather than accept it and start from
/// zeroed state. `MuonOptimizer` carries a momentum buffer and its own step
/// counter (`src/optimizer.rs`), and the `TRIOSRSM` format version 1
/// serialises neither; a run that quietly restarted them at zero would finish,
/// would print a BPB, and would not be a segment of the run it claims to
/// continue. The refusal is raised BEFORE `ensure_supported_optimizer` has
/// spawned any work, so a mistaken audit command costs nothing.
pub fn run_with_optimizer_resumed(
    optimizer: &str,
    args: &TrainArgs,
    resume_from: Option<&Path>,
) -> Result<RunOutcome> {
    ensure_supported_optimizer(optimizer)?;
    if resume_from.is_some() && optimizer != "adamw" {
        return Err(crate::checkpoint::resume_refusal(
            crate::checkpoint::RESUME_REASON_MUON,
            format!(
                "--resume-from was given with optimizer={optimizer:?}, but the Muon path \
                 carries state this format does not serialise: MuonOptimizer's momentum \
                 buffer and its own step counter. Version 1 of TRIOSRSM covers the AdamW \
                 path only. Re-run the window on --optimizer adamw, or extend the format \
                 first - resuming from zeroed momentum would produce weights that are a \
                 segment of no run at all."
            ),
        ));
    }
    match optimizer {
        "adamw" => run_single_resumed(args, resume_from),
        "muon" => run_single_muon(args, false),
        "muon-cwd" => run_single_muon(args, true),
        // Unreachable while these arms and `SUPPORTED_OPTIMIZERS` list the same
        // names. Kept as a hard error rather than a `_ => run_single(args)`
        // fallback: if the two lists ever drift, the run must stop, not quietly
        // become an AdamW run wearing another name.
        other => Err(anyhow::anyhow!(
            "[R5-honesty] optimizer={other:?} is listed in SUPPORTED_OPTIMIZERS but has \
             no dispatch arm; refusing to substitute another optimizer for it."
        )),
    }
}

/// Format the `DONE:` line in ONE place, so the sweep arm and the single-seed
/// arm cannot report different things about the same kind of run.
///
/// The sweep arm printed no `opt=` token at all, which is what let a run
/// launched as `--optimizer soap` and executed as AdamW leave a stdout trail
/// that named no optimizer. `bpb=unmeasured` is mirrored here too: `NaN` parses
/// as a number in every downstream reader, so a run that took no final
/// measurement has to say the word.
pub fn format_done_line(outcome: &RunOutcome, optimizer: &str) -> String {
    match outcome.final_val_bpb {
        Some(bpb) => format!(
            "DONE: seed={} bpb={:.4} steps={} opt={}",
            outcome.seed, bpb, outcome.steps_done, optimizer
        ),
        None => format!(
            "DONE: seed={} bpb=unmeasured steps={} opt={}",
            outcome.seed, outcome.steps_done, optimizer
        ),
    }
}

/// What `trios-train` decided to do about schema DDL this run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutomigrateDecision {
    /// `TRINITY_AUTOMIGRATE=0`: an operator disabled it explicitly.
    Disabled,
    /// No DSN in the environment; there is nothing to migrate.
    NoDsn,
    /// A DSN is reachable but nobody consented to running DDL against it.
    NoConsent,
    /// Explicit consent plus a DSN: run `Migrator::up`.
    Apply,
}

impl AutomigrateDecision {
    /// One line naming the decision, for stdout/stderr.
    pub fn reason(self) -> &'static str {
        match self {
            Self::Disabled => "TRINITY_AUTOMIGRATE=0 - schema DDL disabled by operator",
            Self::NoDsn => "no DSN in environment - nothing to migrate",
            Self::NoConsent => {
                "refused: a DSN is set but TRIOS_ALLOW_AUTOMIGRATE=1 is not. \
                 Having a DSN in the environment is not consent to alter that schema."
            }
            Self::Apply => "TRIOS_ALLOW_AUTOMIGRATE=1 - applying schema DDL to the configured DSN",
        }
    }
}

/// Decide whether this process may run `Migrator::up`.
///
/// Quoting `tests/ledger_seaorm.rs`, which gates the identical operation:
/// "Having a DSN in the environment is not consent". That test refuses to touch
/// an ambient database without an explicit opt-in, while the flagship binary
/// defaulted `TRINITY_AUTOMIGRATE` to "1" and applied DDL to whatever DSN
/// happened to be exported - no host check, no consent flag. A survey run on
/// 2026-08-03 did in fact create tables and insert rows into a database nobody
/// intended to touch. The asymmetry is closed here: writing rows may stay
/// default-on, applying DDL must not be.
///
/// `TRINITY_AUTOMIGRATE=0` is kept as a veto so existing deployments that set
/// it keep their meaning; it is no longer the only thing standing between an
/// ambient DSN and a schema change.
pub fn decide_automigrate(
    automigrate: Option<&str>,
    consent: Option<&str>,
    dsn: Option<&str>,
) -> AutomigrateDecision {
    if automigrate.map(str::trim) == Some("0") {
        return AutomigrateDecision::Disabled;
    }
    if dsn.map(str::trim).unwrap_or("").is_empty() {
        return AutomigrateDecision::NoDsn;
    }
    if consent.map(str::trim) != Some("1") {
        return AutomigrateDecision::NoConsent;
    }
    AutomigrateDecision::Apply
}

/// Run `GATE_FINAL_SEEDS` in sequence.
///
/// Every seed gets its own checkpoint directory for the duration of this call
/// (`{canon}/seed{n}/{step}.bin`). Without that, `TRIOS_CANON_NAME` - which the
/// scarab and the Railway workers set, and which does not vary with the seed -
/// collapsed all three onto one path: three hashes were printed, three `DONE:`
/// lines were printed, and one file survived. A GATE-2 verdict is a three-seed
/// claim and must stand on three seeds' evidence.
///
/// `optimizer` is a REQUIRED parameter and not an `Option`. This function had
/// no optimizer argument at all, so `--sweep --optimizer soap` ran AdamW three
/// times, printed three `DONE:` lines that named no optimizer, printed
/// `GATE-2:` and exited 0 - on the branch that produces the published gate
/// verdict. Every seed is dispatched through `run_with_optimizer`, and the name
/// is refused ONCE, before the first seed trains, rather than three times or
/// not at all.
pub fn run_sweep(
    steps: usize,
    hidden: usize,
    lr: f32,
    attn_layers: u8,
    eval_every: usize,
    train_path: &str,
    val_path: &str,
    optimizer: &str,
) -> Result<Vec<RunOutcome>> {
    // Before `SweepSeedScope::enter()`: a refused sweep must leave no process
    // state behind at all.
    ensure_supported_optimizer(optimizer)?;
    let _seed_scope = SweepSeedScope::enter();
    let mut results = Vec::new();
    for &seed in GATE_FINAL_SEEDS {
        results.push(run_with_optimizer(
            optimizer,
            &TrainArgs {
                seed,
                steps,
                hidden,
                lr,
                attn_layers,
                eval_every,
                train_path: train_path.to_string(),
                val_path: val_path.to_string(),
            },
        )?);
    }
    Ok(results)
}

// -- declaration truth: what a config says vs what this build can execute ----
//
// `run(cfg)` used to build its `TrainArgs` with `hidden: 828` and
// `eval_every: 1000` written into the literal, derive `attn_layers` from
// `hybrid_attn` alone, and drop every other declared field on the floor. So
// `--config configs/gate2-attempt.toml` - a file declaring d_model=384,
// n_layers=4, kind="muon+adamw" - minted a checkpoint sidecar recording
// hidden=828 and optimizer=adamw under the run name "gate2-attempt". The
// artifact and the declaration it is filed under described different runs.
//
// The rule now: honour the declaration or refuse it. A run that cannot execute
// what the config declares must not produce an artifact labelled with that
// declaration.

/// The model width config mode runs, and the only width it can run.
///
/// `TrainConfig` has no field for `TrainArgs.hidden`; `model.d_model` is the
/// declaration graded against it, because that is the width the sidecar records
/// and the number a reader of the TOML would expect the artifact to carry. It
/// is NOT the attention block's `d_model`, which is fixed separately by
/// `HybridAttnConfig` and is checked here as `model.n_heads` / `model.seq_len`.
pub const CONFIG_MODE_HIDDEN: usize = 828;

/// The eval cadence config mode used to substitute in silence.
pub const CONFIG_MODE_EVAL_EVERY: usize = 1000;

/// Where a config-mode run must declare its eval cadence, since the TOML schema
/// has no field for it.
pub const CONFIG_EVAL_EVERY_ENV: &str = "TRIOS_EVAL_EVERY";

/// The AdamW weight decay `run_single` applies to every tensor.
pub const TRAIN_LOOP_WEIGHT_DECAY: f64 = 0.04;

/// The LR schedule `run_single` applies. Fixed; there is no other one.
pub const TRAIN_LOOP_SCHEDULE: &str = "cosine";

/// The gradient-accumulation batch: `accum` (4) chunks x 8 sampled positions,
/// which is literally the divisor `tp` the loop normalises gradients by.
pub const TRAIN_LOOP_BATCH_SIZE: usize = 32;

/// One declared field this build cannot execute, and the value it would
/// otherwise have substituted behind the declaration's back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnhonouredField {
    /// The field as it is spelled in the TOML, e.g. `model.d_model`.
    pub field: String,
    /// What the config declares.
    pub declared: String,
    /// What this build would run instead.
    pub substitute: String,
}

impl std::fmt::Display for UnhonouredField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} declares {} but this build would run {}",
            self.field, self.declared, self.substitute
        )
    }
}

fn unhonoured(
    field: &str,
    declared: impl std::fmt::Display,
    substitute: impl std::fmt::Display,
) -> UnhonouredField {
    UnhonouredField {
        field: field.to_string(),
        declared: declared.to_string(),
        substitute: substitute.to_string(),
    }
}

/// Every declared field this build cannot execute, given `eval_every` as
/// declared out-of-band (see `CONFIG_EVAL_EVERY_ENV`).
///
/// Pure: reads no environment, so a lib test cannot race the process-wide env
/// other tests mutate. An empty result means the config is executable exactly
/// as written.
///
/// # Why the eval cadence is graded here at all
///
/// `TrainConfig` has no `eval_every` field, so config mode substituted 1000.
/// That is not a cosmetic default. The cadence used to gate `gf16_floor()`, the
/// in-place rewrite of embed/proj/lm_head/ctx - two seed-47 runs differing only
/// in `--eval-every` produced different weights and BPB 2.6141 vs 2.6169. That
/// coupling has since been broken (the floor now runs on `gf16_floor_every()`,
/// see the guard in `run_single`), but the cadence still decides which steps are
/// measured, hence the EMA, `min_observed_val_bpb`, every `bpb_samples` row and
/// the `eval_every` field of the sidecar. It is an observation parameter of a
/// published measurement, and a measurement whose observation parameters were
/// chosen for you by a default is not a declared measurement. So config mode
/// requires it to be said out loud.
///
/// # What is deliberately NOT graded
///
/// `data.corpus` is a label for `train_path`/`val_path`, which ARE honoured and
/// whose bytes are hashed into every artifact. `data.batch_tokens` has no
/// counterpart to compare against: the token count a step touches depends on
/// the sampled positions, so any number stated here would be invented. Both
/// remain declared-and-ignored, and that is a real remaining mislabel risk -
/// closing it means trimming the schema, which is a change to files this one
/// does not own.
pub fn unhonoured_fields(
    cfg: &crate::TrainConfig,
    eval_every: Option<usize>,
) -> Vec<UnhonouredField> {
    let mut out = Vec::new();
    let attn = crate::model_hybrid_attn::HybridAttnConfig::default();

    // -- [model] -------------------------------------------------------------
    if cfg.model.d_model != CONFIG_MODE_HIDDEN {
        out.push(unhonoured(
            "model.d_model",
            cfg.model.d_model,
            format!("hidden={CONFIG_MODE_HIDDEN} (config mode has no width knob)"),
        ));
    }
    let attn_layers = usize::from(if cfg.model.hybrid_attn { 2u8 } else { 1u8 });
    if cfg.model.n_layers != attn_layers {
        out.push(unhonoured(
            "model.n_layers",
            cfg.model.n_layers,
            format!(
                "attn_layers={attn_layers} (derived from model.hybrid_attn={})",
                cfg.model.hybrid_attn
            ),
        ));
    }
    if cfg.model.n_heads != attn.num_heads {
        out.push(unhonoured(
            "model.n_heads",
            cfg.model.n_heads,
            format!("{} (fixed by HybridAttnConfig)", attn.num_heads),
        ));
    }
    if cfg.model.vocab_size != VOCAB {
        out.push(unhonoured(
            "model.vocab_size",
            cfg.model.vocab_size,
            format!("{VOCAB} (byte-level, fixed by train_loop::VOCAB)"),
        ));
    }
    if cfg.model.seq_len != attn.seq_len {
        out.push(unhonoured(
            "model.seq_len",
            cfg.model.seq_len,
            format!(
                "{} attention positions (HybridAttnConfig; the n-gram context is \
                 NUM_CTX={NUM_CTX} slots)",
                attn.seq_len
            ),
        ));
    }

    // -- [optimizer] ---------------------------------------------------------
    if !SUPPORTED_OPTIMIZERS.contains(&cfg.optimizer.kind.as_str()) {
        out.push(unhonoured(
            "optimizer.kind",
            format!("{:?}", cfg.optimizer.kind),
            format!("adamw (only {SUPPORTED_OPTIMIZERS:?} are implemented)"),
        ));
    }
    // Compared in f32, the space `AdamW` actually holds them in. Widening
    // ADAMW_BETA1 to f64 instead makes the declaration `0.9` fail against
    // itself: `0.9f32 as f64` is 0.8999999761581421, which is 2.4e-8 away from
    // the 0.9 the TOML parsed. The same trap `neon_writer::bpb_refusal` records
    // for the publication floor.
    if cfg.optimizer.beta1 as f32 != ADAMW_BETA1 {
        out.push(unhonoured(
            "optimizer.beta1",
            cfg.optimizer.beta1,
            ADAMW_BETA1,
        ));
    }
    if cfg.optimizer.beta2 as f32 != ADAMW_BETA2 {
        out.push(unhonoured(
            "optimizer.beta2",
            cfg.optimizer.beta2,
            ADAMW_BETA2,
        ));
    }
    if (cfg.optimizer.weight_decay - TRAIN_LOOP_WEIGHT_DECAY).abs() > 1e-9 {
        out.push(unhonoured(
            "optimizer.weight_decay",
            cfg.optimizer.weight_decay,
            TRAIN_LOOP_WEIGHT_DECAY,
        ));
    }
    if cfg.optimizer.schedule != TRAIN_LOOP_SCHEDULE {
        out.push(unhonoured(
            "optimizer.schedule",
            format!("{:?}", cfg.optimizer.schedule),
            format!("{TRAIN_LOOP_SCHEDULE:?} (train_loop::cosine_lr, the only schedule)"),
        ));
    }
    // `run_single` sets `warmup = args.steps / 10` and nothing can move it.
    let warmup = cfg.steps / 10;
    if cfg.optimizer.warmup_steps != warmup {
        out.push(unhonoured(
            "optimizer.warmup_steps",
            cfg.optimizer.warmup_steps,
            format!("{warmup} (steps/10, fixed)"),
        ));
    }

    // -- [data] --------------------------------------------------------------
    if cfg.data.batch_size != TRAIN_LOOP_BATCH_SIZE {
        out.push(unhonoured(
            "data.batch_size",
            cfg.data.batch_size,
            format!("{TRAIN_LOOP_BATCH_SIZE} (accum=4 chunks x 8 sampled positions)"),
        ));
    }

    // -- [objective] ---------------------------------------------------------
    // The loss is plain cross-entropy with an NCA entropy term that scales the
    // proj gradient; there is no JEPA term at all, and the NCA weight comes
    // from `NcaObjective::default()`, never from the config.
    if (cfg.objective.w_ce - 1.0).abs() > 1e-9 {
        out.push(unhonoured("objective.w_ce", cfg.objective.w_ce, 1.0));
    }
    if cfg.objective.w_jepa.abs() > 1e-9 {
        out.push(unhonoured(
            "objective.w_jepa",
            cfg.objective.w_jepa,
            "0 (no JEPA term exists in this loss)",
        ));
    }
    let nca_weight = NcaObjective::default().weight;
    if (cfg.objective.w_nca - nca_weight).abs() > 1e-9 {
        out.push(unhonoured(
            "objective.w_nca",
            cfg.objective.w_nca,
            format!("{nca_weight} (NcaObjective::default)"),
        ));
    }

    // -- the eval cadence ----------------------------------------------------
    if eval_every.is_none() {
        out.push(unhonoured(
            "eval cadence",
            format!("nothing ({CONFIG_EVAL_EVERY_ENV} unset; TrainConfig has no field for it)"),
            format!("eval_every={CONFIG_MODE_EVAL_EVERY}"),
        ));
    }

    out
}

/// Parse the out-of-band eval cadence declaration.
///
/// Pure, for the same reason `unhonoured_fields` is. `None` means "not
/// declared", which `unhonoured_fields` turns into a refusal; a declared value
/// that cannot be a cadence is an error here rather than a silent fallback.
pub fn parse_config_eval_every(raw: Option<&str>) -> Result<Option<usize>> {
    let Some(raw) = raw else { return Ok(None) };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let parsed: usize = trimmed
        .parse()
        .with_context(|| format!("{CONFIG_EVAL_EVERY_ENV}={raw:?} is not a step count"))?;
    anyhow::ensure!(
        parsed > 0,
        "{CONFIG_EVAL_EVERY_ENV}={raw:?}: an eval cadence of 0 measures nothing"
    );
    Ok(Some(parsed))
}

/// The `TrainArgs` a config declares, or the refusal that says why it cannot be
/// built.
///
/// The error names every unhonoured field, what the config declared and what
/// the run would otherwise have substituted - all of them, not the first one,
/// so one run tells the whole truth about the gap.
pub fn config_train_args(cfg: &crate::TrainConfig, eval_every: Option<usize>) -> Result<TrainArgs> {
    let gaps = unhonoured_fields(cfg, eval_every);
    if !gaps.is_empty() {
        let listed = gaps
            .iter()
            .map(|g| format!("  - {g}"))
            .collect::<Vec<_>>()
            .join("\n");
        return Err(anyhow::anyhow!(
            "[declaration-truth] config {:?} declares {} parameter(s) this build cannot \
             execute:\n{listed}\nRefusing to run: an artifact filed under this \
             declaration would record what was substituted, not what was declared. \
             Declare what this crate executes, or implement the declaration first.",
            cfg.name,
            gaps.len()
        ));
    }
    Ok(TrainArgs {
        seed: cfg.seed,
        steps: cfg.steps,
        // Checked equal to `cfg.model.d_model` above.
        hidden: CONFIG_MODE_HIDDEN,
        lr: cfg.optimizer.lr as f32,
        attn_layers: if cfg.model.hybrid_attn { 2 } else { 1 },
        // Checked present above.
        eval_every: eval_every.unwrap_or(CONFIG_MODE_EVAL_EVERY),
        train_path: cfg.data.train_path.clone(),
        val_path: cfg.data.val_path.clone(),
    })
}

pub fn run(cfg: &crate::TrainConfig) -> Result<RunOutcome> {
    let eval_every = parse_config_eval_every(std::env::var(CONFIG_EVAL_EVERY_ENV).ok().as_deref())?;
    let args = config_train_args(cfg, eval_every)?;
    // The declared optimizer, dispatched through the one function that maps a
    // name onto a loop. `config_train_args` has already refused any name that
    // is not in `SUPPORTED_OPTIMIZERS`, so config mode can no longer run AdamW
    // under another optimizer's name.
    let outcome = run_with_optimizer(&cfg.optimizer.kind, &args)?;
    // The ledger row carries the MEASURED final val_bpb. It used to carry
    // `best_bpb`, the running minimum of the EMA, which is not a measurement of
    // anything. A run with no final measurement emits no row rather than a
    // substituted one.
    if !cfg.ledger.jsonl_path.is_empty() {
        if let Some(bpb) = outcome.final_val_bpb {
            // `let _ =` used to be here. `emit_row` refuses an embargoed SHA, a
            // step below the R8 floor of 4000 and a non-finite BPB, and every
            // one of those refusals was discarded without a character of
            // output: `tests/embargo_block.rs` proves the refusal happens, and
            // the production caller then made it indistinguishable from a
            // successful write. A run whose ledger row was rejected must say so.
            //
            // INCOMPLETE (blocked on file ownership): this should also call
            // `crate::neon_writer::note_dropped()` so `ledger_exit_code()`
            // counts the loss and the process exits non-zero. That function is
            // module-private (`src/neon_writer.rs:151`) and making it
            // `pub(crate)` is a one-line change in a file this change does not
            // own. Until then the failure is loud but not fatal.
            if let Err(e) = crate::ledger::emit_row(cfg, bpb, outcome.steps_done) {
                eprintln!(
                    "[ledger] ERROR: the Gate-2 row for step {} was REFUSED and not \
                     written to {}: {e:#}",
                    outcome.steps_done, cfg.ledger.jsonl_path
                );
            }
        } else {
            eprintln!("[ledger] no final val_bpb was measured; refusing to emit a Gate-2 row");
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
        assert_eq!(fmt.unwrap(), Some(FormatKind::Fp16));
    }

    #[test]
    fn resolves_gf16_alias() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_FORMAT_TYPE");
        std::env::set_var("TRIOS_FAKE_QUANT_FORMAT", "gf16");
        let fmt = resolve_fake_quant_format();
        std::env::remove_var("TRIOS_FAKE_QUANT_FORMAT");
        assert_eq!(fmt.unwrap(), Some(FormatKind::Gf16));
    }

    #[test]
    fn f32_resolves_to_none() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_FAKE_QUANT_FORMAT");
        std::env::set_var("TRIOS_FORMAT_TYPE", "f32");
        let fmt = resolve_fake_quant_format();
        std::env::remove_var("TRIOS_FORMAT_TYPE");
        assert_eq!(fmt.unwrap(), None);
    }

    #[test]
    fn unset_resolves_to_none() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_FORMAT_TYPE");
        std::env::remove_var("TRIOS_FAKE_QUANT_FORMAT");
        assert_eq!(resolve_fake_quant_format().unwrap(), None);
    }

    /// An unrecognised spelling used to resolve to "QAT off". It now stops the
    /// run, and the message has to name the variable and quote the rejected
    /// text - otherwise the operator is back to guessing which of the two
    /// aliases they typo'd.
    #[test]
    fn unknown_format_is_refused_not_silently_dropped() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_FAKE_QUANT_FORMAT");
        std::env::set_var("TRIOS_FORMAT_TYPE", "imaginary_float");
        let err = resolve_fake_quant_format().expect_err("must refuse");
        std::env::remove_var("TRIOS_FORMAT_TYPE");
        let msg = err.to_string();
        assert!(msg.contains("TRIOS_FORMAT_TYPE"), "{msg}");
        assert!(msg.contains("imaginary_float"), "{msg}");
        // The accepted spellings must actually be listed.
        assert!(msg.contains("fp16"), "{msg}");
    }

    /// The near-miss that motivated the refusal: `int_8` is one underscore away
    /// from a format this build implements, and it used to train f32 and record
    /// `fake_quant_format: "f32"` without printing the word "format" once.
    #[test]
    fn a_near_miss_spelling_is_refused_too() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_FAKE_QUANT_FORMAT");
        std::env::set_var("TRIOS_FORMAT_TYPE", "int_8");
        let err = resolve_fake_quant_format().expect_err("must refuse");
        std::env::remove_var("TRIOS_FORMAT_TYPE");
        assert!(err.to_string().contains("int_8"), "{err}");
    }

    /// `--eval-every 0` used to reach `step % 0`. The refusal must name the
    /// argument and point at the knob that does mean "only at the end".
    #[test]
    fn eval_every_zero_is_refused_naming_steps() {
        assert!(validate_eval_every(1).is_ok());
        assert!(validate_eval_every(1000).is_ok());
        let err = validate_eval_every(0).expect_err("0 must be refused");
        let msg = err.to_string();
        assert!(msg.contains("--eval-every"), "{msg}");
        assert!(msg.contains("--steps"), "{msg}");
        // clap's usage-error code, not this crate's "the run failed" code.
        assert_eq!(EVAL_EVERY_USAGE_EXIT, 2);
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
    use crate::checkpoint::{self, sha256_hex, CHECKPOINT_HEADER_LEN, CHECKPOINT_PAYLOAD_OFFSET};
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
        assert!(saved
            .sha256
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));

        // 3. Every tensor survives bit-identically.
        let (restored, restored_meta) = HybridModel::from_checkpoint_bytes(&loaded).unwrap();
        assert_tensors_bit_identical(&model, &restored);

        // 4. The metadata survives too.
        assert_eq!(restored_meta, meta);

        std::env::remove_var("TRIOS_CHECKPOINT_DIR");
    }

    /// The measured sweep defect: three seeds, one `TRIOS_CANON_NAME`, one
    /// path. `fs::rename` is atomic, which also means it replaces silently, so
    /// two artifacts were destroyed and all three were reported as landed.
    #[test]
    fn save_refuses_to_overwrite_a_different_checkpoint() {
        let _g = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("TRIOS_CHECKPOINT_DIR", dir.path());

        let first = poisoned_model(16)
            .to_checkpoint_bytes(&meta_for(60))
            .unwrap();
        let mut second_model = poisoned_model(16);
        second_model.embed[5] = 0.25; // a different model, same run, same step
        let second = second_model.to_checkpoint_bytes(&meta_for(60)).unwrap();
        assert_ne!(
            first, second,
            "the fixture must differ, or this proves nothing"
        );

        let landed = checkpoint::save("collide-run", 60, &first).unwrap();
        let err = checkpoint::save("collide-run", 60, &second)
            .map(|_| ())
            .expect_err("a differing overwrite must be refused, not renamed over");
        let msg = format!("{err:#}");
        assert!(msg.contains("refusing to overwrite"), "{msg}");
        // The refusal names both hashes and the path, so the operator can see
        // which artifact is on disk and which one was rejected.
        assert!(
            msg.contains(&landed.sha256),
            "existing hash not named: {msg}"
        );
        assert!(
            msg.contains(&sha256_hex(&second)),
            "incoming hash not named: {msg}"
        );
        assert!(msg.contains("60.bin"), "path not named: {msg}");

        // Nothing was written and nothing was deleted.
        assert_eq!(std::fs::read(&landed.path).unwrap(), first);
        for e in std::fs::read_dir(landed.path.parent().unwrap()).unwrap() {
            let name = e.unwrap().file_name().to_string_lossy().into_owned();
            assert!(!name.contains(".tmp."), "leftover temp file {name}");
        }

        // Re-saving IDENTICAL bytes is not an overwrite: the file that would
        // result is the file that is already there.
        let again = checkpoint::save("collide-run", 60, &first).expect("idempotent re-save");
        assert_eq!(again.sha256, landed.sha256);

        std::env::remove_var("TRIOS_CHECKPOINT_DIR");
    }

    /// Two seeds of one sweep must not collide, and the single-seed layout
    /// must not move: `ckpt_replay` and the README both point at the flat one.
    #[test]
    fn a_seed_scoped_save_lands_beside_its_siblings_not_on_top_of_them() {
        let _g = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("TRIOS_CHECKPOINT_DIR", dir.path());

        let a = poisoned_model(16)
            .to_checkpoint_bytes(&meta_for(60))
            .unwrap();
        let mut model_b = poisoned_model(16);
        model_b.embed[5] = 0.25;
        let b = model_b.to_checkpoint_bytes(&meta_for(60)).unwrap();

        let s47 = checkpoint::save_scoped("sweep-run", Some(47), 60, &a).unwrap();
        let s89 = checkpoint::save_scoped("sweep-run", Some(89), 60, &b).unwrap();
        assert_ne!(s47.path, s89.path, "two seeds resolved to one path");
        assert_ne!(s47.sha256, s89.sha256);
        assert_eq!(
            std::fs::read(&s47.path).unwrap(),
            a,
            "seed 47 was overwritten"
        );
        assert_eq!(std::fs::read(&s89.path).unwrap(), b);
        assert_eq!(
            s47.path,
            dir.path().join("sweep-run").join("seed47").join("60.bin")
        );

        // The unscoped path is still the flat one.
        assert_eq!(
            checkpoint::checkpoint_path("sweep-run", 60),
            dir.path().join("sweep-run").join("60.bin")
        );

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
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
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
        // Named by the constant, not by a copy of the string: this assertion
        // read "/7" while `CHECKPOINT_RECORD_SCHEMA` had moved to "/8", which
        // fails here and - because it fires while holding `ENV_LOCK` - poisons
        // the mutex and takes every other env-serialised test with it.
        assert_eq!(rec.schema, checkpoint::CHECKPOINT_RECORD_SCHEMA);
        // A single-seed run keeps the flat layout the README and `ckpt_replay`
        // point at: no `seed{n}/` component.
        assert!(
            !run_dir.join(checkpoint::seed_scope_component(47)).exists(),
            "a single-seed run must not move into a per-seed subdirectory"
        );
        // Schema 5: the flag that gates the in-place `gf16_floor()` rewrite is
        // byte 124 of the hashed header and was in no sidecar at all.
        // ... and the value recorded is the RESOLVED knob, not a second,
        // independent reading of the legacy env var.
        assert_eq!(rec.gf16_enabled, resolve_gf16_knob().expect("resolve gf16"));
        // Schema 2: what changed the artifact is now IN the artifact.
        assert_eq!(rec.steps_total, steps as u64);
        assert_eq!(rec.gf16_floor_every, gf16_floor_every());
        assert_eq!(rec.eval_every, steps as u64);
        assert!(
            !rec.git_provenance.is_empty(),
            "provenance strength unrecorded"
        );
        // Schema 3: the first-order inputs the record used to omit. `lr` must
        // be the value that ran, never a defaulted 0.0 - which is itself a
        // legal learning rate and so unusable as a sentinel.
        assert_eq!(
            rec.lr,
            Some(0.003f32 as f64),
            "lr not recorded as the f32 that ran"
        );
        assert_eq!(rec.attn_scale, attn_scale() as f64);
        assert_eq!(rec.attn_seq, attn_seq_override() as u64);
        assert_eq!(rec.platform.os, std::env::consts::OS);
        assert_eq!(rec.platform.arch, std::env::consts::ARCH);
        assert_eq!(rec.platform.pointer_width, usize::BITS);
        assert!(
            !rec.platform.libc.is_empty(),
            "libc field left blank, not 'undetermined'"
        );
        assert!(
            !rec.platform.toolchain_provenance.is_empty(),
            "toolchain strength unrecorded"
        );
        // The tests run from the crate root, so the tree IS reachable and the
        // sentinel must not appear.
        assert_eq!(
            rec.source_sha256.len(),
            64,
            "source digest is {}",
            rec.source_sha256
        );
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
        assert_eq!(
            rec.trainer.sha256.len(),
            64,
            "trainer digest is {}",
            rec.trainer.sha256
        );
        assert!(rec.trainer.sha256.chars().all(|c| c.is_ascii_hexdigit()));
        // Re-derivable by an outside party with `shasum -a 256`, which is the
        // whole point: the digest is over the bytes on disk, not an image.
        let exe = std::fs::read(&rec.trainer.path).expect("recorded trainer path is readable");
        assert_eq!(
            rec.trainer.sha256,
            sha256_hex(&exe),
            "trainer hash is not the file's"
        );
        assert_eq!(
            rec.vocab, VOCAB as u32,
            "the record must state its own alphabet"
        );
        // The measured reading, not the EMA, is what the sidecar carries.
        assert_eq!(rec.final_val_bpb, outcome.final_val_bpb);
        assert_eq!(rec.min_observed_val_bpb, outcome.best_val_bpb);
        // Schema 6: the record must state the plan the reading came from, not
        // just the reading. 40 windows of 129 tokens is a 5% sample of a
        // 100,000-byte val corpus, and until now no field said so.
        assert_eq!(rec.eval_seq, Some((SEQ + 1) as u32));
        let chunks = rec.eval_chunks.expect("schema 6 states its coverage");
        assert!(chunks > 0);
        assert_eq!(
            rec.eval_tokens,
            Some(chunks as u64 * (SEQ + 1) as u64),
            "eval_tokens must be chunks * seq, not a re-derivation"
        );
        let stderr = rec.val_bpb_stderr.expect("more than one window was read");
        assert!(
            stderr.is_finite() && stderr > 0.0,
            "a 40-window mean has a positive standard error, got {stderr}"
        );
        // `optimizer: "adamw"` was one string carrying four numbers.
        let params = rec
            .optimizer_params
            .as_ref()
            .expect("schema 6 names the hyperparameters");
        // Compared after narrowing back to `f32`. The constants ARE `f32` - the
        // update rule steps in `f32` - and the sidecar widens them, but this
        // crate's `serde_json` is built without the `float_roundtrip` feature,
        // so reading an `f64` back can land 1 ULP away (measured here:
        // beta2 parsed as 0.9990000128746032, exactly 0.999f32 is
        // 0.9990000128746033). Narrowing is the comparison that matches what
        // the optimizer actually held; the JSON TEXT carries the exact widened
        // decimal, which is what an outside reader parses.
        assert_eq!(params.beta1 as f32, ADAMW_BETA1);
        assert_eq!(params.beta2 as f32, ADAMW_BETA2);
        assert_eq!(params.eps as f32, ADAMW_EPS);
        assert_eq!(params.weight_decay as f32, 0.04_f32, "the wd that ran");
        assert_eq!(params.source, "train_loop::AdamW");
        // And they are NOT the phi-branded constants in `optimizer.rs`, which
        // this trainer never constructs. Recording 0.618 here would describe a
        // run that never happened.
        assert_ne!(params.beta1, 1.0 / ((1.0 + 5.0_f64.sqrt()) / 2.0));
        assert_eq!(
            rec.canon_name, "IGLA-test/canon",
            "ledger identity unsanitized"
        );
        // Schema 9. The record names the file that was actually written - the
        // same string the ledger row carries, so the two cannot describe
        // different files - but it names it RELATIVE to the digest scope, and
        // this checkpoint dir is a tempdir outside that scope. The assertion is
        // therefore the derivation the product performs, plus the property that
        // derivation exists for: no absolute path reaches the record.
        assert_eq!(
            rec.path,
            crate::checkpoint::scope_relative_artifact_path(&bin)
        );
        assert!(
            !rec.path.starts_with('/'),
            "the record published an absolute path: {}",
            rec.path
        );
        assert!(rec.path.ends_with(&format!("{steps}.bin")), "{}", rec.path);
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
        let (tokens, synthetic) = load_data("/nonexistent/corpus-that-does-not-exist.bin").unwrap();
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
    use crate::checkpoint;

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

    /// An UNDERFLOWED target probability is an absence, not 33.21928 bpb.
    ///
    /// The NaN door was closed last round; this is the door standing open
    /// beside it. An f32 softmax returns a finite, exact `0.0` for a target the
    /// model finds impossible - no NaN, no infinity, so `is_nan()` and
    /// `is_finite()` both accept it - and clamping `p` to a 1e-10 floor then
    /// contributed exactly 23.02585 nats, which is 33.21928 bpb: the constant
    /// this crate documents as the fake-measurement signature. `final_val_bpb`
    /// is a SEALED field, so the fabricated number shipped inside an
    /// authenticated declaration.
    #[test]
    fn an_underflowed_target_probability_is_an_absence_not_33_bpb() {
        let mut model = HybridModel::new(64, 47, 2);
        // Exactly one scoring position, so one forward pass fixes the reading.
        let tokens: Vec<usize> = vec![1; NGRAM + 1];
        let target = tokens[NGRAM].min(VOCAB - 1);
        let h = model.hidden;
        let hidden = model.forward_cached(&tokens, 0).hidden;
        let norm2: f32 = hidden.iter().map(|x| x * x).sum();
        assert!(norm2 > 0.0, "fixture needs a non-zero hidden state");

        // Put the target's logit 400 nats under the maximum. f32 `exp`
        // underflows to exactly 0.0 below about -104, so the target
        // probability is an honest zero and NOT a NaN: every pre-existing
        // guard in this function accepts it.
        let big = if target == 0 { 1 } else { 0 };
        model.lm_head.iter_mut().for_each(|w| *w = 0.0);
        let scale = 400.0 / norm2;
        for hi in 0..h {
            model.lm_head[big * h + hi] = hidden[hi] * scale;
        }
        let mut logits = model.forward_cached(&tokens, 0).logits;
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
        let d = EVAL_CHUNKS_DEFAULT;
        assert_eq!(eval_chunk_count(160, d), 1);
        assert!(eval_chunk_count(160, d) < MIN_EVAL_CHUNKS);
        assert_eq!(eval_chunk_count(SEQ, d), 0, "too short for a single window");
        assert!(
            eval_chunk_count(MIN_VAL_TOKENS, d) >= MIN_EVAL_CHUNKS,
            "the token floor must imply the chunk floor"
        );
    }

    /// The default plan must be the grid that was there before it had a name.
    ///
    /// Every BPB in the ledger was measured on 40 windows of `SEQ + 1` starting
    /// `max_start / 40` apart. Naming the parameter must not move that grid by
    /// one token, or the new numbers stop being comparable with the old ones -
    /// which is the whole reason the default is still 40.
    #[test]
    fn the_default_plan_is_the_grid_that_was_hardcoded() {
        let plan = eval_plan(100_000, EVAL_CHUNKS_DEFAULT).expect("100k tokens is measurable");
        assert_eq!(plan.seq, SEQ + 1);
        assert_eq!(plan.seq, 129);
        assert_eq!(plan.chunks, 40);
        assert_eq!(plan.stride, (100_000 - 129) / 40);
        // 40 * 129 = 5,160 of 100,000 tokens. The headline was a 5% sample.
        assert_eq!(plan.chunks * plan.seq, 5_160);
    }

    /// `0` means every window, and is a value rather than junk.
    #[test]
    fn zero_chunks_means_full_coverage() {
        let full = eval_plan(100_000, 0).expect("measurable");
        assert_eq!(full.stride, full.seq, "non-overlapping tiling");
        assert_eq!(full.chunks, (100_000usize - 129).div_ceil(129));
        assert!(
            full.chunks > EVAL_CHUNKS_DEFAULT * 19,
            "full coverage must be far more than the 40-window sample: {}",
            full.chunks
        );
        // A stream too short for one window is still not a measurement.
        assert!(eval_plan(SEQ, 0).is_none());
    }

    /// The coverage knob is an OBSERVATION parameter with a declared default.
    #[test]
    fn eval_chunks_is_its_own_knob_and_defaults_to_forty() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_EVAL_CHUNKS");
        assert_eq!(eval_chunks_target(), EVAL_CHUNKS_DEFAULT);
        assert_eq!(eval_chunks_target(), 40, "the published grid, unmoved");
        std::env::set_var("TRIOS_EVAL_CHUNKS", "128");
        assert_eq!(eval_chunks_target(), 128);
        // `0` is FULL COVERAGE, not junk: unlike `TRIOS_GF16_FLOOR_EVERY` it
        // must not be swallowed by the default.
        std::env::set_var("TRIOS_EVAL_CHUNKS", "0");
        assert_eq!(eval_chunks_target(), 0);
        for junk in ["", "nonsense", "-3", "4.5"] {
            std::env::set_var("TRIOS_EVAL_CHUNKS", junk);
            assert_eq!(eval_chunks_target(), EVAL_CHUNKS_DEFAULT, "junk={junk:?}");
        }
        std::env::remove_var("TRIOS_EVAL_CHUNKS");
    }

    /// A mean with no spread is not a measurement result.
    ///
    /// The estimator's own scatter across windows is the quantity the headline
    /// BPB was missing; a single window has no spread and must say so rather
    /// than report 0.0, which reads as perfect repeatability.
    #[test]
    fn evaluate_reports_the_spread_of_its_own_windows() {
        let _g = ENV_LOCK.lock().unwrap();
        let model = HybridModel::new(16, 47, 1);
        let val = lcg_tokens(0xB0BA, 20_000, 95);

        let stats = evaluate(&model, &val, EVAL_CHUNKS_DEFAULT).expect("measurable");
        assert_eq!(stats.plan.chunks, 40);
        assert_eq!(stats.tokens(), 40 * 129);
        let stdev = stats.stdev.expect("40 windows have a sample stdev");
        let stderr = stats.stderr.expect("and therefore a standard error");
        assert!(
            stdev > 0.0,
            "40 distinct windows cannot all read identically"
        );
        // `s / sqrt(n)` is the standard error of a mean drawn from an INFINITE
        // population. The val stream is finite and the grid reads it WITHOUT
        // replacement, so the finite-population correction applies: 40 of the
        // `N` windows that tile this 20,000-token stream.
        let population = eval_chunk_count(val.len(), 0);
        assert!(
            population > 40,
            "the sample must be a strict subset: N={population}"
        );
        let expected = stdev / 40f32.sqrt() * (1.0 - 40.0 / population as f32).sqrt();
        assert!(
            (stderr - expected).abs() < 1e-6,
            "stderr must be (s/sqrt(n))*sqrt(1-n/N) with N={population}: {stderr} vs {expected}"
        );
        assert!(
            stderr < stdev / 40f32.sqrt(),
            "the correction must shrink the band, not grow it"
        );

        // Full coverage reads every window exactly once. There is no sampling
        // left to be uncertain about, so the standard error is exactly zero -
        // the claim `eval_chunks_target`'s doc comment makes, now enforced.
        let full = evaluate(&model, &val, 0).expect("measurable");
        assert_eq!(
            full.plan.chunks, population,
            "target 0 walks the whole grid"
        );
        assert!(
            full.stdev
                .expect("the windows still differ from each other")
                > 0.0,
            "full coverage does not make the readings identical"
        );
        assert_eq!(
            full.stderr,
            Some(0.0),
            "at n == N the mean IS the population mean: no sampling error"
        );

        // One window: the mean exists, the spread does not.
        let one = evaluate(&model, &val[..129 + 1], EVAL_CHUNKS_DEFAULT).expect("one window");
        assert_eq!(one.plan.chunks, 1);
        assert_eq!(one.stdev, None, "a single draw has no sample stdev");
        assert_eq!(one.stderr, None, "and no standard error - not 0.0");
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
            assert_eq!(
                gf16_floor_every(),
                GF16_FLOOR_EVERY_DEFAULT,
                "junk={junk:?}"
            );
        }
        std::env::remove_var("TRIOS_GF16_FLOOR_EVERY");
    }

    /// The documented GF16 knob must be able to turn GF16 off.
    ///
    /// `GF16_ENABLED=false` was indistinguishable from `GF16_ENABLED` unset:
    /// `parse_gf16_enabled` defaults to `"false"`, so an explicit false fell
    /// through to the legacy `TRIOS_GF16_DISABLE` reading, which is ON by
    /// default. Measured: `GF16_ENABLED=false` produced the byte-identical
    /// artifact `a645d688...` that unset produced, while `TRIOS_GF16_DISABLE=1`
    /// produced `22684cc8...` at bpb 3.4327 against 3.5353 - a 0.1026 bpb
    /// effect, larger than the declared expanded uncertainty. The sidecar
    /// meanwhile recorded `gf16_enabled: true` on a run the operator had set to
    /// false. That is the repository fabricating a recipe field, and it is
    /// checkable by anyone who flips the knob and watches the hash not move.
    ///
    /// The default stays ON: it is what every published number was produced
    /// with. What changes is that "set" and "unset" are now two facts.
    #[test]
    fn gf16_enabled_false_is_distinguishable_from_unset() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("GF16_ENABLED");
        std::env::remove_var("TRIOS_GF16_DISABLE");
        let unset = resolve_gf16_knob().expect("unset must resolve");
        assert!(
            unset,
            "the executed default is GF16 ON; flipping it invalidates every \
             published artifact, including the aarch64 anchor in README.md"
        );

        std::env::set_var("GF16_ENABLED", "false");
        let explicit_off = resolve_gf16_knob().expect("GF16_ENABLED=false must resolve");
        std::env::remove_var("GF16_ENABLED");
        assert!(!explicit_off, "GF16_ENABLED=false must turn the floor off");
        assert_ne!(
            unset, explicit_off,
            "a documented knob whose explicit value changes nothing is a \
             fabricated recipe field"
        );

        // The legacy variable keeps working while GF16_ENABLED is unset.
        std::env::set_var("TRIOS_GF16_DISABLE", "1");
        let legacy_off = resolve_gf16_knob().expect("legacy path must resolve");
        std::env::remove_var("TRIOS_GF16_DISABLE");
        assert!(!legacy_off, "TRIOS_GF16_DISABLE=1 must still disable GF16");
    }

    /// A cadence that was asked for and cannot be honoured must stop the run,
    /// not resolve to "no artifacts at all".
    ///
    /// `TRIOS_CHECKPOINT_EVERY=1_000` is the shape a reader of this codebase
    /// types - it is a valid Rust integer literal - and `1000 ` is what a
    /// trailing space in a YAML or shell assignment produces. Both used to hit
    /// `.parse().ok().unwrap_or(0)`, which is the "final step only" default:
    /// the caller asked for periodic artifacts, received none, and neither the
    /// banner nor the sidecar nor stderr mentioned it.
    #[test]
    fn checkpoint_cadence_rejects_a_value_it_cannot_honour() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_CHECKPOINT_EVERY");
        assert_eq!(
            resolve_checkpoint_every().expect("absent is a legal state"),
            0,
            "absent must stay final-step-only"
        );

        // Junk is fatal and the message must name the variable, so the operator
        // can fix it without reading this file.
        for junk in ["1_000", "1k", "1000x", "-3", "nonsense", "1e3", "0x10"] {
            std::env::set_var("TRIOS_CHECKPOINT_EVERY", junk);
            let err = resolve_checkpoint_every()
                .expect_err(&format!("{junk:?} must not silently resolve to 0"))
                .to_string();
            assert!(
                err.contains("TRIOS_CHECKPOINT_EVERY"),
                "error must name the variable: {err}"
            );
        }

        // Legal values, including the whitespace a shell assignment leaves and
        // the explicit 0 that means the documented default.
        for (raw, want) in [
            ("50", 50usize),
            (" 50 ", 50),
            ("1000", 1000),
            ("0", 0),
            ("", 0),
        ] {
            std::env::set_var("TRIOS_CHECKPOINT_EVERY", raw);
            assert_eq!(
                resolve_checkpoint_every().expect("legal"),
                want,
                "raw={raw:?}"
            );
        }
        std::env::remove_var("TRIOS_CHECKPOINT_EVERY");

        // The hit test is now a pure function of the resolved number: nothing
        // re-reads the environment mid-run.
        assert!(!checkpoint_every_hit(50, 0), "0 means final step only");
        assert!(checkpoint_every_hit(50, 50));
        assert!(!checkpoint_every_hit(51, 50));
    }

    /// `TRIOS_CHECKPOINT_INIT=1` must leave the weights as initialised on disk,
    /// before the first optimizer step, and they must differ from the weights
    /// one step later.
    ///
    /// Without `0.bin` the earliest artifact any run could produce was AFTER a
    /// gradient had been applied, so "do the two architectures disagree at
    /// initialisation, or only after arithmetic?" - the question that decides
    /// whether the cross-architecture divergence is an RNG defect or a
    /// floating-point one - could not be asked of any file this program wrote.
    /// See `docs/DIVERGENCE-LOCALIZATION.md`.
    #[test]
    fn checkpoint_init_emits_the_weights_before_the_first_step() {
        let _g = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let train_path = dir.path().join("train.txt");
        let val_path = dir.path().join("val.txt");
        std::fs::write(&train_path, lcg_bytes(0xA1CE, 40_000, 8)).unwrap();
        std::fs::write(&val_path, lcg_bytes(0xB0BA, 9_000, 8)).unwrap();

        for k in [
            "DATABASE_URL",
            "NEON_DATABASE_URL",
            "TRIOS_NEON_DSN",
            "TRIOS_DATABASE_URL",
            "TRIOS_CHECKPOINT_DISABLE",
            "TRIOS_CHECKPOINT_EVERY",
            "TRIOS_GF16_FLOOR_EVERY",
            "HIDDEN_DIM",
            "NUM_ATTN_LAYERS",
            "GF16_ENABLED",
        ] {
            std::env::remove_var(k);
        }
        std::env::set_var("TRIOS_CHECKPOINT_DIR", dir.path().join("ckpt"));
        std::env::set_var("TRIOS_CANON_NAME", "IGLA-init");
        std::env::set_var("TRIOS_CHECKPOINT_INIT", "1");

        let outcome = run_single(&TrainArgs {
            seed: 47,
            steps: 1,
            hidden: 16,
            lr: 0.02,
            attn_layers: 1,
            eval_every: 1,
            train_path: train_path.to_string_lossy().into_owned(),
            val_path: val_path.to_string_lossy().into_owned(),
        });
        std::env::remove_var("TRIOS_CHECKPOINT_INIT");
        std::env::remove_var("TRIOS_CANON_NAME");
        let outcome = outcome.expect("run_single");
        assert_eq!(outcome.steps_done, 1);

        let run_dir = dir.path().join("ckpt").join("IGLA-init");
        let zero = run_dir.join("0.bin");
        let one = run_dir.join("1.bin");
        assert!(zero.is_file(), "no initial-weights artifact at {zero:?}");
        assert!(one.is_file(), "no step-1 artifact at {one:?}");

        // A step-0 artifact identical to the step-1 artifact would mean the
        // save happened after the optimizer, which is the one thing this file
        // exists to rule out.
        let a = std::fs::read(&zero).unwrap();
        let b = std::fs::read(&one).unwrap();
        assert_ne!(
            a, b,
            "0.bin equals 1.bin: the initial weights were saved after the first \
             optimizer step, not before it"
        );

        // The record must say step 0 and must not present a run minimum or an
        // EMA it never took.
        let rec: checkpoint::CheckpointRecord =
            serde_json::from_slice(&std::fs::read(run_dir.join("0.json")).unwrap()).unwrap();
        assert_eq!(rec.step, 0, "the step-0 sidecar must say step 0");
        assert_eq!(
            rec.min_observed_val_bpb, None,
            "no run minimum exists at step 0"
        );
        assert_eq!(rec.ema_bpb, None, "no EMA exists at step 0");
        assert!(
            rec.final_val_bpb.is_some_and(|v| v.is_finite() && v > 0.0),
            "the init reading was measured on these exact bytes and must be recorded"
        );
        assert_eq!(rec.steps_total, 1);
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
        let best = outcome
            .best_val_bpb
            .expect("at least one reading was taken");

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

    /// The requested artifact cadence must be delivered in full, not
    /// intersected with the observation cadence.
    ///
    /// Measured before the fix: `TRIOS_CHECKPOINT_EVERY=50 --eval-every 100
    /// --steps 200` wrote 100.bin and 200.bin, exit 0, no warning. Half the
    /// requested artifacts were never written, because both emission points
    /// sat inside `if step % args.eval_every == 0` - an observation parameter
    /// deciding what got saved, on the one function whose absence produced
    /// "1,851 experiments, zero artifacts".
    #[test]
    fn checkpoint_cadence_is_not_intersected_with_the_eval_cadence() {
        let _g = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let train_path = dir.path().join("train.txt");
        let val_path = dir.path().join("val.txt");
        std::fs::write(&train_path, lcg_bytes(0xA1CE, 40_000, 8)).unwrap();
        std::fs::write(&val_path, lcg_bytes(0xB0BA, 9_000, 8)).unwrap();

        for k in [
            "DATABASE_URL",
            "NEON_DATABASE_URL",
            "TRIOS_NEON_DSN",
            "TRIOS_DATABASE_URL",
            "TRIOS_CHECKPOINT_DISABLE",
            "TRIOS_GF16_FLOOR_EVERY",
            "HIDDEN_DIM",
            "NUM_ATTN_LAYERS",
            "GF16_ENABLED",
        ] {
            std::env::remove_var(k);
        }
        std::env::set_var("TRIOS_CHECKPOINT_DIR", dir.path().join("ckpt"));
        std::env::set_var("TRIOS_CANON_NAME", "IGLA-cadence");
        std::env::set_var("TRIOS_CHECKPOINT_EVERY", "50");

        let outcome = run_single(&TrainArgs {
            seed: 47,
            steps: 200,
            hidden: 16,
            lr: 0.02,
            attn_layers: 1,
            eval_every: 100,
            train_path: train_path.to_string_lossy().into_owned(),
            val_path: val_path.to_string_lossy().into_owned(),
        })
        .expect("run_single");
        std::env::remove_var("TRIOS_CHECKPOINT_EVERY");
        std::env::remove_var("TRIOS_CANON_NAME");

        let run_dir = dir.path().join("ckpt").join("IGLA-cadence");
        for step in [50, 100, 150, 200] {
            let bin = run_dir.join(format!("{step}.bin"));
            let json = run_dir.join(format!("{step}.json"));
            assert!(
                bin.is_file(),
                "checkpoint_every=50 was requested and {bin:?} is missing: the \
                 cadence was intersected with eval_every=100 again"
            );
            assert!(json.is_file(), "no sidecar at {json:?}");
        }

        // Every artifact carries a reading of ITS OWN weights - that is why the
        // checkpoint-only steps evaluate rather than skip.
        for step in [50, 100, 150, 200] {
            let rec: checkpoint::CheckpointRecord = serde_json::from_slice(
                &std::fs::read(run_dir.join(format!("{step}.json"))).unwrap(),
            )
            .unwrap();
            assert_eq!(rec.step, step as i64);
            assert!(
                rec.final_val_bpb.is_some(),
                "artifact at step {step} carries no reading of its own weights"
            );
        }

        // ...and the published trajectory is untouched by the artifact cadence.
        // A checkpoint-only step folds nothing into the EMA, so its record says
        // `null` rather than repeating the last eval step's lagging number.
        for step in [50, 150] {
            let rec: checkpoint::CheckpointRecord = serde_json::from_slice(
                &std::fs::read(run_dir.join(format!("{step}.json"))).unwrap(),
            )
            .unwrap();
            assert!(
                rec.ema_bpb.is_none(),
                "step {step} is not an eval step; a stale EMA there would read \
                 as a measurement of these weights"
            );
        }
        for step in [100, 200] {
            let rec: checkpoint::CheckpointRecord = serde_json::from_slice(
                &std::fs::read(run_dir.join(format!("{step}.json"))).unwrap(),
            )
            .unwrap();
            assert!(rec.ema_bpb.is_some(), "step {step} IS an eval step");
        }
        assert_eq!(
            outcome.final_val_bpb,
            serde_json::from_slice::<checkpoint::CheckpointRecord>(
                &std::fs::read(run_dir.join("200.json")).unwrap()
            )
            .unwrap()
            .final_val_bpb
        );
    }

    /// A run that emitted nothing must not exit 0.
    ///
    /// `--steps 0` never enters the step loop, so it reached no emission point,
    /// wrote no artifact, printed a `DONE:` line and returned success. The
    /// in-loop failure handling could not see it: it only fires on a checkpoint
    /// that was attempted.
    #[test]
    fn a_run_that_emits_no_artifact_is_an_error() {
        let _g = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let train_path = dir.path().join("train.txt");
        let val_path = dir.path().join("val.txt");
        std::fs::write(&train_path, lcg_bytes(0xA1CE, 40_000, 8)).unwrap();
        std::fs::write(&val_path, lcg_bytes(0xB0BA, 9_000, 8)).unwrap();

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
        std::env::set_var("TRIOS_CHECKPOINT_DIR", dir.path().join("ckpt"));
        std::env::set_var("TRIOS_CANON_NAME", "IGLA-no-artifact");

        let args = TrainArgs {
            seed: 47,
            steps: 0,
            hidden: 16,
            lr: 0.02,
            attn_layers: 1,
            eval_every: 100,
            train_path: train_path.to_string_lossy().into_owned(),
            val_path: val_path.to_string_lossy().into_owned(),
        };
        let err = run_single(&args).expect_err("a zero-artifact run must not succeed");
        assert!(
            err.to_string().contains("no checkpoint artifact"),
            "unexpected error: {err:#}"
        );

        // The escape hatch is explicit, never silent.
        std::env::set_var("TRIOS_CHECKPOINT_DISABLE", "1");
        run_single(&args).expect("checkpointing off is a legal, DECLARED choice");
        std::env::remove_var("TRIOS_CHECKPOINT_DISABLE");
        std::env::remove_var("TRIOS_CANON_NAME");
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
        assert!(
            msg.starts_with("ALPHABET FOLD REFUSED:"),
            "message was {msg:?}"
        );
        assert!(
            msg.contains("/tmp/cyr.txt"),
            "message must name the file: {msg:?}"
        );
        assert!(
            msg.contains("bits-per-byte"),
            "message must say what the number is not: {msg:?}"
        );
    }

    #[test]
    fn the_two_bytes_that_collide_are_really_in_the_fixture() {
        // Not an assumption about UTF-8: the collision is measured here, so the
        // guard's justification is checked rather than asserted.
        assert!(
            CYRILLIC.contains(&0xD0),
            "fixture lost its Cyrillic lead byte"
        );
        assert!(
            CYRILLIC.contains(&0xA0),
            "fixture lost the 0xA0 continuation byte"
        );
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
        let dir =
            std::env::temp_dir().join(format!("trios-alphabet-{}-{}", std::process::id(), line!()));
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

#[cfg(test)]
mod dispatch_and_consent_tests {
    //! Three ways `trios-train` used to exit 0 while its own guards stood by.
    //!
    //! 1. The optimizer dispatch lived only in the non-sweep arm, so
    //!    `--sweep --optimizer soap` ran AdamW three times, printed three
    //!    `DONE:` lines naming no optimizer, printed `GATE-2:` and exited 0 -
    //!    on the branch that produces the published gate verdict.
    //! 2. The `DONE:` line the sweep printed carried no `opt=` token at all, so
    //!    no stdout parser could see which optimizer had actually run.
    //! 3. `TRINITY_AUTOMIGRATE` defaulted to "1", so an ambient DSN was treated
    //!    as permission to run schema DDL against it.
    use super::*;

    /// The refusal must happen for the sweep too, and BEFORE the first seed
    /// trains: an unsupported optimizer is not a run to be salvaged three
    /// times over. The corpus paths below do not exist, which is the point -
    /// if this returns `Ok`, or fails with an I/O error instead, then the
    /// dispatch is being reached after the guard rather than before it.
    #[test]
    fn sweep_refuses_an_unsupported_optimizer_before_it_trains() {
        let err = run_sweep(
            1,
            8,
            0.003,
            1,
            1,
            "/nonexistent/train-that-must-never-be-opened.txt",
            "/nonexistent/val-that-must-never-be-opened.txt",
            "soap",
        )
        .expect_err("--sweep --optimizer soap must refuse, not run AdamW");
        let msg = err.to_string();
        assert!(
            msg.contains("unsupported optimizer"),
            "the refusal must name the defect, got {msg:?}"
        );
        assert!(
            msg.contains("soap"),
            "the refusal must name the optimizer asked for, got {msg:?}"
        );
        assert!(
            !msg.contains("nonexistent"),
            "the guard must fire before any corpus is opened, got {msg:?}"
        );
    }

    /// Every supported name reaches a dispatch arm; the two lists cannot drift
    /// apart without this failing.
    #[test]
    fn every_supported_optimizer_is_accepted_by_the_sweep_guard() {
        for name in SUPPORTED_OPTIMIZERS {
            ensure_supported_optimizer(name)
                .unwrap_or_else(|e| panic!("{name} is listed as supported but refused: {e}"));
        }
    }

    /// The sweep and the single-seed arm share one formatter, so both carry
    /// `opt=`.
    #[test]
    fn a_done_line_always_names_the_optimizer() {
        let measured = RunOutcome {
            final_val_bpb: Some(2.6141),
            best_val_bpb: Some(2.6141),
            ema_bpb: Some(2.7),
            final_bpb: 2.6141,
            steps_done: 12000,
            seed: 47,
        };
        let line = format_done_line(&measured, "muon-cwd");
        assert!(line.starts_with("DONE: "), "line was {line:?}");
        assert!(line.contains("opt=muon-cwd"), "line was {line:?}");
        assert!(line.contains("bpb=2.6141"), "line was {line:?}");
    }

    /// A run that took no final measurement says the word rather than printing
    /// a `NaN` that `f64::from_str` hands back to a parser as a number.
    #[test]
    fn an_unmeasured_done_line_says_unmeasured_and_still_names_the_optimizer() {
        let unmeasured = RunOutcome {
            final_val_bpb: None,
            best_val_bpb: None,
            ema_bpb: None,
            final_bpb: f64::NAN,
            steps_done: 0,
            seed: 89,
        };
        let line = format_done_line(&unmeasured, "adamw");
        assert!(line.contains("bpb=unmeasured"), "line was {line:?}");
        assert!(line.contains("opt=adamw"), "line was {line:?}");
        assert!(
            !line.to_ascii_lowercase().contains("nan"),
            "line was {line:?}"
        );
    }

    /// "Having a DSN in the environment is not consent" - tests/ledger_seaorm.rs.
    #[test]
    fn automigrate_is_refused_without_the_consent_flag() {
        let dsn = Some("postgres://u:p@127.0.0.1:5432/whatever");
        assert_eq!(
            decide_automigrate(None, None, dsn),
            AutomigrateDecision::NoConsent,
            "an ambient DSN alone must not authorise DDL"
        );
        assert_eq!(
            decide_automigrate(Some("1"), None, dsn),
            AutomigrateDecision::NoConsent,
            "TRINITY_AUTOMIGRATE=1 alone must not authorise DDL"
        );
        assert_eq!(
            decide_automigrate(None, Some("0"), dsn),
            AutomigrateDecision::NoConsent
        );
        assert_eq!(
            decide_automigrate(None, Some("true"), dsn),
            AutomigrateDecision::NoConsent,
            "only the exact string \"1\" is consent"
        );
        assert!(
            AutomigrateDecision::NoConsent
                .reason()
                .contains("not consent"),
            "the refusal must say why: {}",
            AutomigrateDecision::NoConsent.reason()
        );
    }

    /// The consent flag alone still needs a DSN, and the old veto still vetoes.
    #[test]
    fn automigrate_applies_only_on_consent_plus_a_dsn() {
        let dsn = Some("postgres://u:p@127.0.0.1:5432/whatever");
        assert_eq!(
            decide_automigrate(None, Some("1"), dsn),
            AutomigrateDecision::Apply
        );
        assert_eq!(
            decide_automigrate(Some("1"), Some("1"), dsn),
            AutomigrateDecision::Apply
        );
        assert_eq!(
            decide_automigrate(Some("0"), Some("1"), dsn),
            AutomigrateDecision::Disabled,
            "TRINITY_AUTOMIGRATE=0 stays a veto"
        );
        assert_eq!(
            decide_automigrate(None, Some("1"), None),
            AutomigrateDecision::NoDsn
        );
        assert_eq!(
            decide_automigrate(None, Some("1"), Some("   ")),
            AutomigrateDecision::NoDsn,
            "a blank DSN is no DSN"
        );
    }
}
