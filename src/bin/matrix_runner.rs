// matrix_runner -- Phase C orchestrator for the 312-cell FormatxAlgorithm matrix
// (gHashTag/trios#446, kernel stubs from PR #101 + #102).
//
// Purpose: run ONE (format, algo, seed, hidden) cell end-to-end by invoking the
// already-instrumented `cpu_train` binary with the right env vars, parse the
// resulting `.trinity/results/cpu_train_<format>_<algo>_seed<seed>.json`, and
// write a row into `ssot.bpb_samples` (Railway phd-postgres-ssot SSOT) using
// the `MATRIX_DATABASE_URL` env var. When NO DSN is configured the row is only
// echoed on stdout and that is a legitimate exit 0. When a DSN IS configured
// and no row lands, the process exits non-zero (see R5 GUARD 6).
//
// Constitutional notes:
//   * R1 Rust-only -- pipeline is pure Rust, no .py or .sh shims.
//   * R5 honest   -- REVISED 2026-08-03. "log and continue" was read as
//                   "print the failure, then report success", and the
//                   workflow's collect-ledger job gates on that success. An
//                   error is still logged rather than panicked, but the exit
//                   code now tells the truth about whether a row landed.
//   * R7 witness  -- emits run_id, sha, step, seed, bpb verbatim so the
//                   matrix-bot (L-C6) can reconstruct #446 body.
//
// HARDENING (PASS-21, 2026-05-14): three R5 guards close the leaderboard
// pollution bug-class documented in:
//   * trios#777 (silent format collapse -- 704 bit-identical clusters)
//   * trios#779 (sidecar metadata poisoning -- 15 fake algo suffixes)
//   * leaderboard-snapshot skill Q7 (algo whitelist) + Q3 (format drift)
//
//   1. ALGO_WHITELIST -- `--algo` must be one of {adamw, muon, muon-cwd}. Any
//      other value is rejected with ExitCode(3) and a clear error message. No
//      silent fallback. Matches the same whitelist enforced in trios-train.rs:298.
//
//   2. FORMAT_ALIAS_MAP -- `binary16 -> fp16`, `binary32 -> fp32`, `fp8e4m3 ->
//      fp8_e4m3`, `fp8e5m2 -> fp8_e5m2`. Normalisation happens BEFORE canon_name
//      assembly so the leaderboard never sees both spellings.
//
//   3. SEED_CANON -- `--seed` must be one of Fibonacci/Lucas {47, 89, 123, 144,
//      1597, 2584, 4181, 6765, 10946}. The legacy default `42` is FORBIDDEN
//      (see leaderboard-snapshot skill `SEED_FORBIDDEN` constant).
//
//   4. CANON_NAME -- assembled as `IGLA-MATRIX-{format}-h{hidden}-LR{lr}-rng{seed}-{algo}`
//      conforming to the IGLA canonical regex
//      `^IGLA-[A-Z][A-Z0-9-]*-[a-z0-9_]+-h\d+-LR[0-9.]+-rng\d+-[a-z0-9-]+$`.
//      Old format `cpu_train_{format}_{algo}` is dead.
//
// ----------------------------------------------------------------------------
// RETRACTION (2026-08-03) -- EVERY ROW THIS BINARY WROTE BEFORE THIS COMMIT
// CARRIES A LABEL THAT DOES NOT DESCRIBE THE RUN. Two independent defects:
//
//   1. LR MISLABEL. `run_cpu_train` took no `lr` and never passed `--lr` to the
//      child, so EVERY cell trained at `cpu_train`'s own default 0.003 no
//      matter what `--lr` said. The parsed `lr` reached `canon_name` and the
//      ledger row and nothing else. The nightly schedule passes LR=0.001, so
//      every nightly row ever written is stamped `LR001` and was trained at
//      0.003. Verified by execution: `--lr=0.001` and `--lr=0.9` through this
//      runner produced a bit-identical bpb=4.299640655517578.
//      This is the `--eval-every` defect inverted: there an OBSERVATION knob
//      silently changed the artefact; here a RECIPE knob silently changed
//      nothing while the record claimed it did.
//
//   2. FORMAT ROWS THAT ARE NOT DISTINCT MEASUREMENTS. `fp80` is returned
//      UNCHANGED by `fake_quantize_f32` (`FormatKind::is_unsupported_in_f32`,
//      src/fake_quant.rs), so every `fp80` row in the ledger is arithmetically
//      an `fp32` row -- including any `fp80` vs `fp32` "tie", which is an
//      identity, not a result. `posit16` is an IEEE 10-bit mantissa mask, not
//      a posit encoder, and `gf4`/`gf12`/`gf20`/`gf24` are absent from the GF
//      dispatch and fall through to that same mask.
//
//   3. `git_sha()` stamped a short sha with NO dirty check, so a row could
//      name a commit whose tree it was not built from (observed: sha 3c1f751
//      recorded from a tree with 874 dirty entries).
//
// Rows written from this commit onward: `--lr` is passed to the child AND the
// child's own reported `lr` is checked against the request before a row is
// built; a non-faithful format is refused outright unless
// TRIOS_ALLOW_UNFAITHFUL_FORMAT=1, which stamps `format_faithful=false` on the
// row; `format_lr_token` is injective; git provenance carries a dirty flag.
// Old rows cannot be repaired -- they must be read as "trained at 0.003" and,
// for fp80/posit16, as "not a distinct format measurement".
// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
// UNSCREENED ROWS (2026-08-06) -- EVERY ROW THIS BINARY HAS ALREADY PUBLISHED
// INTO `ssot.bpb_samples` WAS WRITTEN WITHOUT PASSING THE WRITE-SIDE BPB
// SCREEN. `write_row_async` opens its own `tokio_postgres` connection and
// INSERTs directly, so it never entered `src/neon_writer.rs` and therefore
// never reached `reject_bpb` -- the check that module documents as living "on
// the WRITE side so no caller can bypass it". Concretely, the rows already on
// the public leaderboard were never tested for `is_finite`, never tested
// against `BPB_SENTINEL_CEILING` (the f32::MAX sentinel that was once
// published and then divided by), and never tested against
// `invariants::PUBLISHED_BPB_FLOOR` (the bound that admits 2.61 and refuses
// the retracted 1.5492). The same path also read `MATRIX_DATABASE_URL` /
// `DATABASE_URL` and connected on their presence alone, ignoring the
// `TRIOS_LEDGER_WRITE=1` opt-in that `neon_writer` declares mandatory.
//
// From this commit onward both gates run in `resolve_write_gate`, BEFORE any
// socket is opened. That fixes what is written NEXT; it does not retroactively
// screen anything. Existing rows must be re-screened at read time before they
// are cited.
// ----------------------------------------------------------------------------
//
// Anchor: phi^2 + phi^-2 = 3.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, ExitCode};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::runtime::Runtime;
use tokio_postgres::{Client, NoTls};
use trios_trainer::fake_quant::FormatKind;

// ----------------------------------------------------------------------------
// R5 HARDENING CONSTANTS
// ----------------------------------------------------------------------------

/// Only optimizers actually implemented by `trios-train.rs` dispatch (PR #135).
/// Everything else (lion, soap, tiger, lamb, prodigy, ...) used to silently
/// collapse to AdamW and produce byte-identical BPB under fake labels -- see
/// trios#777 / trios-trainer-igla#140. NEVER add a value here without
/// implementing the optimizer in `train_loop::*` first.
/// HOTFIX 2026-05-14: `muon-cwd` removed -- cpu_train.rs `AlgoOpt::from_env`
/// does not implement it yet, so allowing it through caused 3 failing matrix
/// cells (fp16/fp32/gf16 x muon-cwd seed=1597) in run 25856938977 and would
/// have silently produced bit-identical bpb to `muon` if from_env had a
/// fallback. Follow-up: implement true CWD variant in cpu_train.rs, then
/// re-add. See trios-trainer-igla#146 (to be opened).
const ALGO_WHITELIST: &[&str] = &["adamw", "muon"];

/// Fibonacci + Lucas seed canon. Skill `leaderboard-snapshot` SEED_CANON.
/// The legacy default `42` is FORBIDDEN -- it appeared in 2554 rows before
/// the canon-name reform of 2026-05-12.
const SEED_CANON: &[i64] = &[47, 89, 123, 144, 1597, 2584, 4181, 6765, 10946];

/// IEEE / TRIOS canonical format spellings. Aliases like `binary16` map to the
/// canonical name to prevent format-drift duplicates in the leaderboard
/// (8483 rows under `binary16` are actually `fp16`; 245 `binary32` are `fp32`).
///
/// R5 GUARD 5 (2026-08-03): the old `other => other` arm passed ANY spelling
/// through untouched. `--format=surveyprobe2` therefore reached `canon_name`
/// and the `format` column verbatim while `cpu_train` silently fell back to
/// F32, so the axis the matrix exists to study was a claim, never a
/// measurement. An unrecognised spelling is now an error: a name is canonical
/// only if `FormatKind::from_env` can actually resolve it to a kernel.
fn normalize_format(raw: &str) -> Result<&str, String> {
    let mapped = match raw {
        "binary16" | "float16" | "f16" => "fp16",
        "binary32" | "float32" | "f32_alias" => "fp32",
        "binary64" | "float64" | "f64_alias" => "fp64",
        "fp8e4m3" => "fp8_e4m3",
        "fp8e5m2" => "fp8_e5m2",
        "fp6e3m2" => "fp6_e3m2",
        "fp6e2m3" => "fp6_e2m3",
        "fp4e2m1" => "fp4_e2m1",
        // `f32` is also a TRIOS-supported spelling alongside `fp32` - keep both.
        other => other,
    };
    if FormatKind::from_env(mapped).is_none() {
        return Err(format!(
            "unknown format {raw:?} (normalized to {mapped:?}): no FormatKind \
             kernel resolves it. Refusing to write an unvalidated string into \
             canon_name and the `format` column."
        ));
    }
    Ok(mapped)
}

/// The format the run actually EXECUTED, as reported by `cpu_train` in its
/// results file, must equal the format that was requested.
///
/// Mirrors the rationale of `neon_writer::bind_executed_optimizer` (src/
/// neon_writer.rs:555-575): a row recorded under the wrong format is worse
/// than a row not recorded at all, because it is indistinguishable from a real
/// measurement of that format. `bind_executed_optimizer` closes that hole for
/// the optimizer column on the WRITE side; this closes it for the format
/// column here.
///
/// Implemented locally on purpose: `src/neon_writer.rs` is owned by another
/// change in flight, and refactoring this binary's raw INSERT into
/// `neon_writer` is out of scope for this round. The exit code is made honest
/// where the INSERT actually lives.
fn check_executed_format(requested: &str, executed: &str) -> Result<(), String> {
    let want = FormatKind::from_env(requested).ok_or_else(|| {
        format!("requested format {requested:?} does not resolve to a FormatKind")
    })?;
    let got = FormatKind::from_env(executed).ok_or_else(|| {
        format!("cpu_train reported executed format {executed:?}, which does not resolve")
    })?;
    if want != got {
        return Err(format!(
            "FORMAT MISMATCH: requested {requested:?} ({want:?}) but cpu_train \
             executed {executed:?} ({got:?}). The `format` column would name a \
             kernel that never ran."
        ));
    }
    Ok(())
}

/// The LR the child ACTUALLY trained at, as it reports it in its own results
/// file, must equal the LR this row is going to be labelled with.
///
/// F1 FIX. Passing `--lr` to the child is necessary but not sufficient: the
/// bug being closed is precisely that a label was trusted without a
/// measurement behind it, and a silently-ignored flag would reproduce it
/// exactly. So the request is checked against the child's own report, in the
/// same shape as the `format_executed` refusal.
///
/// TOLERANCE, and why it is not the literal `1e-12` on the raw request:
/// `cpu_train` parses `--lr` into an **f32** and serialises that f32 widened
/// back to f64, so `--lr=0.001` comes back as `0.0010000000474974513` -- off
/// the f64 request by 4.7e-11. Comparing the raw request at 1e-12 would refuse
/// EVERY cell. The comparison is therefore against the request NARROWED TO f32
/// AND WIDENED BACK, which is the exact value the trainer is able to hold; at
/// 1e-12 that is effectively an equality test, and it is a STRONGER claim than
/// a loosened absolute tolerance would have been: it says the child trained at
/// exactly the f32 this label denotes, not merely near it.
fn check_executed_lr(requested: f64, executed: Option<f64>) -> Result<(), String> {
    let got = executed.ok_or_else(|| {
        "cpu_train did not report `lr` in its results file. Without it the LR in \
         canon_name is a request, not a measurement -- which is exactly the \
         defect this check exists to close. Rebuild cpu_train and re-run."
            .to_string()
    })?;
    let representable = (requested as f32) as f64;
    if (got - representable).abs() > 1e-12 {
        return Err(format!(
            "LR MISMATCH: requested {requested} (f32-representable as \
             {representable}) but cpu_train reports it trained at {got}. The \
             canon_name would name a learning rate that never ran."
        ));
    }
    Ok(())
}

/// Whether a row for this format is a distinct measurement at all.
///
/// F2 FIX. `FormatKind::is_faithful()` is the crate's OWN marker for "the
/// round trip through this format is not really this format", and until now it
/// was read by NOTHING outside its definition and two unit tests. Meanwhile
/// `fp80` is returned UNCHANGED by `fake_quantize_f32` (it is
/// `is_unsupported_in_f32`), so an `fp80` row is arithmetically an `fp32` row,
/// and an `fp80`-equals-`fp32` "tie" is an identity dressed as a result.
/// `posit16` is an IEEE 10-bit mantissa mask, not a posit encoder.
///
/// Enforcement lives HERE rather than in `cpu_train`: `cpu_train` measuring a
/// degenerate kernel is a legitimate experiment; this binary PUBLISHING that
/// measurement as a peer of the real ones is the defect.
///
/// Returns `Ok(true)` for a faithful format, `Ok(false)` for an unfaithful one
/// that the operator explicitly opted into via `TRIOS_ALLOW_UNFAITHFUL_FORMAT=1`
/// (the row is then stamped `format_faithful=false`), and `Err` otherwise.
fn resolve_format_faithful(format: &str, allow_unfaithful: bool) -> Result<bool, String> {
    let kind = FormatKind::from_env(format)
        .ok_or_else(|| format!("format {format:?} does not resolve to a FormatKind"))?;
    if kind.is_faithful() {
        return Ok(true);
    }
    if allow_unfaithful {
        return Ok(false);
    }
    Err(format!(
        "NON-FAITHFUL FORMAT {format:?} ({kind:?}): FormatKind::is_faithful() is \
         false, i.e. the crate itself declares the f32 round trip through this \
         format is not really this format (identity passthrough, mantissa-mask \
         stand-in, or a deferred encoder). Publishing it alongside real kernels \
         makes an identity look like a result. Set \
         TRIOS_ALLOW_UNFAITHFUL_FORMAT=1 to run it anyway; the row is then \
         stamped format_faithful=false."
    ))
}

/// Build the IGLA-canonical canon_name. Pattern:
///   IGLA-{LANE}-{format}-h{hidden}-LR{lr}-rng{seed}-{algo}
/// where `lr` is rendered with the leading-zero notation expected by the
/// leaderboard-snapshot regex (e.g. `0.001` -> `LR001`, `0.0001` -> `LR0001`).
/// LANE token defaults to `MATRIX`; see leaderboard-snapshot SKILL.md LANE registry
/// for the canonical list (MATRIX, SHORT-WAVE-MATRIX, COVERAGE-A, SCARAB-ADAMW, ...).
fn build_canon_name(format: &str, hidden: i32, lr: f64, seed: i64, algo: &str) -> String {
    build_canon_name_lane("MATRIX", format, hidden, lr, seed, algo)
}

fn build_canon_name_lane(
    lane: &str,
    format: &str,
    hidden: i32,
    lr: f64,
    seed: i64,
    algo: &str,
) -> String {
    let lr_token = format_lr_token(lr);
    let algo_dashed = algo.replace('_', "-");
    let lane_upper = lane.trim().to_ascii_uppercase();
    format!("IGLA-{lane_upper}-{format}-h{hidden}-LR{lr_token}-rng{seed}-{algo_dashed}")
}

/// Render a learning rate as the canon LR token. The regex accepts either
/// leading-zero compact form (`LR0001`) or decimal form (`LR0.0001`) -- we
/// emit the compact form because every champion canon since Wave-35 uses it.
///
/// F4 FIX (2026-08-03): the previous body was NOT INJECTIVE, and `canon_name`
/// is the ledger's identity key that `ON CONFLICT` de-duplicates on, so two
/// different recipes collided into one row:
///   * `format!("{:.6}", lr)` truncated everything below 1e-6, so `1e-7` and
///     `0` both rendered `LR0`;
///   * values >= 1 kept their bare digits, so `0.1` ("0.1" -> strip "0." ->
///     "1") and `1.0` ("1") both rendered `LR1`.
///
/// Both are now distinguishable:
///   * the source string is the SHORTEST ROUND-TRIP decimal (`{}` on f64,
///     which never uses exponent notation), so no magnitude is truncated;
///   * a value in (0,1) keeps the legacy compact form (`0.001` -> `001`) and
///     therefore NEVER contains a `.`, while every other value is emitted with
///     an explicit `.` (`1.0` -> `1.0`, `0` -> `0.0`). The two token shapes are
///     disjoint, so no cross-branch collision is possible, and within each
///     branch the round-trip property of the decimal makes distinct f64 values
///     produce distinct strings.
///
/// Every LR token this repo has actually published (`001`, `0001`, `003`,
/// `01`) is byte-identical to what the old body produced, so no historical
/// canon_name changes shape.
fn format_lr_token(lr: f64) -> String {
    // Shortest decimal that round-trips back to this f64. Rust's `Display` for
    // floats never emits exponent notation, so `1e-7` becomes "0.0000001"
    // rather than being flattened to zero by a fixed precision.
    let s = format!("{lr}");
    if let Some(rest) = s.strip_prefix("0.") {
        // 0 < lr < 1 -- legacy compact form. Contains no '.' by construction.
        rest.to_string()
    } else if s.contains('.') {
        // 1.5 -> "1.5". Dot is allowed by the leaderboard regex.
        s
    } else {
        // 1 -> "1.0", 0 -> "0.0". The appended ".0" is what keeps these from
        // colliding with the compact form of 0.1 and of a hypothetical 0.0.
        format!("{s}.0")
    }
}

/// Parsed `.trinity/results/cpu_train_<fmt>_<algo>_seed<seed>.json` payload.
/// We only keep the fields we need for the matrix row; extra fields are
/// ignored by `#[serde(default)]` on every one.
#[derive(Debug, Deserialize, Default)]
struct CpuTrainResult {
    #[serde(default)]
    #[allow(dead_code)]
    algo: String,
    #[serde(default)]
    #[allow(dead_code)]
    seed: i64,
    #[serde(default)]
    steps: i64,
    #[serde(default)]
    #[allow(dead_code)]
    dim: i64,
    #[serde(default)]
    initial_bpb: f64,
    /// The BPB measured at `step == steps`. Since 2026-08-03 `cpu_train` emits
    /// this and `best_bpb` as two distinct keys; before that, this field
    /// carried the running MINIMUM over all evals while this binary paired it
    /// with `step = steps`.
    #[serde(default)]
    final_bpb: f64,
    #[serde(default)]
    best_bpb: f64,
    #[serde(default)]
    delta_bpb: f64,
    /// Format `cpu_train` actually executed. Absent in results written before
    /// 2026-08-03; an empty string means "trainer too old to declare it", which
    /// is reported rather than treated as a match.
    #[serde(default)]
    format_executed: String,
    /// The learning rate `cpu_train` actually trained at, as it reports it.
    /// `Option` on purpose: `0.0` is a legal (if useless) LR, so a `#[serde
    /// (default)] f64` would make "trainer did not declare it" indistinguishable
    /// from "trainer trained at zero" -- the same sentinel-vs-measurement
    /// confusion that `loss_on_seq` was fixed for. `None` is refused, never
    /// treated as a match.
    #[serde(default)]
    lr: Option<f64>,
}

/// Single matrix cell descriptor, logged verbatim for R7 witness trail.
#[derive(Debug, Serialize)]
struct MatrixRow {
    canon_name: String,
    format: String,
    algo: String,
    hidden: i32,
    seed: i64,
    step: i32,
    /// The BPB measured at `step`, i.e. `final_bpb`. `best_bpb` is carried
    /// alongside so a consumer can see the two are different numbers rather
    /// than inferring one from the other.
    bpb: f64,
    best_bpb: f64,
    initial_bpb: f64,
    delta_bpb: f64,
    format_executed: String,
    /// F1: the LR the child REPORTED training at, not the one that was asked
    /// for. `check_executed_lr` has already proven the two agree; carrying the
    /// reported value means the row cites a measurement rather than a request.
    lr: f64,
    /// F2: `false` marks a row whose format is the crate's own
    /// `is_faithful() == false` -- i.e. not a distinct measurement. Emitted on
    /// EVERY row, so a consumer never has to infer it from the format name.
    format_faithful: bool,
    sha: String,
    /// F5: `None` means `git status` could not be run. A failed query is not a
    /// clean tree.
    git_dirty: Option<bool>,
    git_provenance: String,
    run_id: String,
    ts_unix: i64,
}

/// The value of `--flag=value` if it was passed at all.
fn arg_opt(flag: &str) -> Option<String> {
    let key = format!("--{flag}=");
    env::args().find_map(|a| a.strip_prefix(&key).map(|v| v.to_string()))
}

fn arg_or(flag: &str, default: &str) -> String {
    arg_opt(flag).unwrap_or_else(|| default.to_string())
}

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Exit code for an argument this binary understands but cannot use. Distinct
/// from the algo (3) and format (5) rejections so a supervisor can tell a
/// malformed invocation from a rejected experiment.
const EXIT_BAD_ARGS: u8 = 4;

/// Exit code for a reading the write-side screen refuses to publish.
///
/// Distinct from the DB-write failure (6) on purpose: "the database did not
/// accept this row" and "this number is not a measurement" are different
/// facts, and only the second one means the cell must never be retried into
/// the leaderboard.
const EXIT_BPB_REFUSED: u8 = 9;

/// Read a known flag and parse it, or reject the invocation.
///
/// The R5 guards below already refuse an unsupported `--algo` and an unknown
/// `--format`. A recognised NUMERIC flag with an unusable value was the
/// remaining silent path, and it is worse: the substituted default became the
/// row's published LABEL, so the matrix asserted a seed and a learning rate
/// that had never run.
fn parse_flag_or_reject<T: std::str::FromStr>(flag: &str, default: &str) -> Result<T, ExitCode> {
    let raw = arg_or(flag, default);
    trios_trainer::parse_flag_value::<T>(flag, &raw).map_err(|e| {
        eprintln!("[matrix_runner] R5-REJECT {e}");
        ExitCode::from(EXIT_BAD_ARGS)
    })
}

/// Git identity of the tree this cell was produced from.
///
/// F5 FIX. The old body ran `git rev-parse --short HEAD` and stamped the
/// result with NO dirty check, so a row could name a commit whose tree it was
/// not built from -- observed this round as sha `3c1f751` recorded from a tree
/// with 874 dirty entries. This is the identical defect that
/// `checkpoint::resolve_git_provenance` was written to close for checkpoint
/// sidecars, so it is reused rather than re-derived: one implementation, one
/// meaning of "dirty", across both artefact families.
///
/// Returns `(sha, dirty, provenance)`. `dirty == None` means `git status`
/// could not be run: a failed query is NOT a clean tree, and recording it as
/// one is the whole point of the fix. The sha is truncated to the same 7-char
/// short form the `sha` column has always carried so old rows stay readable.
fn git_identity() -> (String, Option<bool>, &'static str) {
    let (full_sha, dirty, provenance) = trios_trainer::checkpoint::resolve_git_provenance();
    let short = if full_sha.is_empty() {
        "unknown".to_string()
    } else {
        full_sha.chars().take(7).collect()
    };
    (short, dirty, provenance)
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// The exact argv handed to the child, isolated so a test can assert on it
/// without spawning a trainer. F1: `--lr` was missing here, which is why every
/// cell trained at `cpu_train`'s default 0.003 regardless of its label.
fn build_cpu_train_argv(
    algo: &str,
    seed: i64,
    dim: i32,
    steps: i32,
    vocab: i32,
    seq: i32,
    lr: f64,
) -> Vec<String> {
    vec![
        "run".to_string(),
        "--quiet".to_string(),
        "--release".to_string(),
        "--bin".to_string(),
        "cpu_train".to_string(),
        "--".to_string(),
        format!("--seed={seed}"),
        format!("--steps={steps}"),
        format!("--dim={dim}"),
        format!("--vocab={vocab}"),
        format!("--seq={seq}"),
        format!("--algo={algo}"),
        format!("--lr={lr}"),
    ]
}

#[allow(clippy::too_many_arguments)]
fn run_cpu_train(
    format_type: &str,
    algo: &str,
    seed: i64,
    dim: i32,
    steps: i32,
    vocab: i32,
    seq: i32,
    lr: f64,
) -> Result<CpuTrainResult, String> {
    // Prefer `cargo run --release --bin cpu_train` so the child resolves the
    // already-compiled artefact from `target/release/`. In CI this is
    // pre-built in an earlier job, so the call is a no-op rebuild.
    let mut cmd = Command::new("cargo");
    cmd.args(build_cpu_train_argv(algo, seed, dim, steps, vocab, seq, lr));
    cmd.env("TRIOS_FORMAT_TYPE", format_type);
    cmd.env("TRIOS_ALGO_TYPE", algo);

    eprintln!(
        "[matrix_runner] spawning cpu_train format={format_type} algo={algo} \
         seed={seed} dim={dim} steps={steps} lr={lr}"
    );
    let out = cmd
        .output()
        .map_err(|e| format!("spawn cpu_train failed: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cpu_train exited with {}: stderr tail=\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .rev()
                .take(20)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    // Read the path the CHILD says it wrote, rather than reconstructing it.
    //
    // This used to be `format!(".trinity/results/cpu_train_{format_type}_\
    // {algo}_seed{seed}.json")` -- a name keyed on 3 of the 7 parameters that
    // define a cell. Two cells differing only in dim/seq/steps/lr resolved to
    // the SAME path, so this reader could pick up the previous cell's file and
    // "verify" an lr that never ran; that is the mechanism behind the reported
    // matrix flake. A reconstructed path is a guess about someone else's
    // behaviour. `cpu_train` prints `Results: <path>`, and that line is now the
    // handshake: if it is absent, the child did not write a results file and
    // this cell has no measurement.
    let stdout = String::from_utf8_lossy(&out.stdout);
    let announced = stdout
        .lines()
        .filter_map(|l| l.strip_prefix("Results: "))
        .next_back()
        .map(|p| p.trim().to_string())
        .ok_or_else(|| {
            format!(
                "cpu_train printed no `Results: <path>` line, so it named no \
                 results file. Refusing to guess a path and read whatever cell \
                 happens to be sitting there (format={format_type} algo={algo} \
                 seed={seed} dim={dim} steps={steps} lr={lr})."
            )
        })?;
    let path: PathBuf = PathBuf::from(announced);
    let bytes = fs::read(&path).map_err(|e| format!("read result {path:?}: {e}"))?;
    let parsed: CpuTrainResult =
        serde_json::from_slice(&bytes).map_err(|e| format!("parse result {path:?}: {e}"))?;
    Ok(parsed)
}

/// The DSN this binary would use, read from the two names the
/// format-algo-matrix workflow documents.
///
/// MATRIX-first, then the neutral name -- the precedence this binary has
/// always used. It is only ever exercised on an environment
/// `enforce_dsn_conflict_gate()` (first statement of `main`) has already
/// declared unambiguous: if these two names disagreed the process exited 78
/// before any work ran, so the winner here cannot be a database nobody named.
fn configured_dsn() -> Option<String> {
    env::var("MATRIX_DATABASE_URL")
        .ok()
        .or_else(|| env::var("DATABASE_URL").ok())
        .filter(|d| !d.is_empty())
}

/// Everything that must be true before this binary opens a socket to the SSOT.
#[derive(Debug, Clone, PartialEq, Eq)]
enum WriteGate {
    /// The reading itself must never be published, for the named reason.
    Refused(String),
    /// No DSN configured: nothing was promised, so nothing is owed.
    NoDsn,
    /// A DSN is configured but `TRIOS_LEDGER_WRITE` is not `1`.
    NotOptedIn,
    /// Cleared to connect, with this DSN.
    Cleared(String),
}

/// Decide, without touching the network, whether this cell may be written.
///
/// ORDER IS LOAD-BEARING AND IS TESTED (`tests/ledger_write_guard.rs`).
///
///   1. The BPB screen runs FIRST, before the DSN is even looked at, so a
///      sentinel can never cost a connection and can never depend on whether
///      a database happened to be reachable. It is
///      `neon_writer::bpb_refusal_reason` -- the same implementation
///      `bpb_sample` uses -- and not a second copy of the rules, because the
///      defect being closed here is precisely that this binary had its own
///      write path and therefore no rules at all.
///   2. `TRIOS_LEDGER_WRITE=1` is then required. A DSN alone is not
///      permission: any of these names is routinely present in a developer or
///      CI shell for reasons that have nothing to do with this trainer, and
///      connecting on their presence is how a scouting run put rows on the
///      shared ledger before anyone noticed. `neon_writer::check_write_opt_in`
///      prints that module's own wording, once, naming the variable.
fn resolve_write_gate(bpb: f64, dsn: Option<String>) -> WriteGate {
    if let Some(reason) = trios_trainer::neon_writer::bpb_refusal_reason(bpb as f32) {
        return WriteGate::Refused(reason);
    }
    let Some(dsn) = dsn else {
        return WriteGate::NoDsn;
    };
    if !trios_trainer::neon_writer::check_write_opt_in() {
        return WriteGate::NotOptedIn;
    }
    WriteGate::Cleared(dsn)
}

/// `--write-preflight=<bpb>`: run the gate above on a supplied reading and
/// stop.
///
/// This exists so the ORDER of the gates is observable from outside the
/// binary. The refusal it must produce sits, in a real run, behind a full
/// training run whose BPB nobody can dictate -- there is no way to ask
/// `cpu_train` for `f32::MAX` -- so without this entry point the only evidence
/// that the screen precedes the socket would be a reading of the source.
///
/// It calls exactly the same [`resolve_write_gate`] the write path calls, and
/// when the gate CLEARS it opens the same connection with the same
/// `connect: ` error prefix, so a test can prove the socket is genuinely
/// downstream of both gates rather than absent from this mode. It never
/// executes DDL and never INSERTs: the connection is dropped as soon as it is
/// established.
fn run_write_preflight(raw: &str) -> ExitCode {
    let bpb: f64 = match trios_trainer::parse_flag_value::<f64>("write-preflight", raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[matrix_runner] R5-REJECT {e}");
            return ExitCode::from(EXIT_BAD_ARGS);
        }
    };
    eprintln!(
        "[matrix_runner] write-preflight: gating bpb={bpb} only; no training was \
         performed and no row will be written"
    );
    match resolve_write_gate(bpb, configured_dsn()) {
        WriteGate::Refused(reason) => refuse_bpb(&reason),
        WriteGate::NoDsn => {
            announce_no_dsn();
            ExitCode::SUCCESS
        }
        WriteGate::NotOptedIn => {
            announce_not_opted_in();
            ExitCode::SUCCESS
        }
        WriteGate::Cleared(dsn) => {
            eprintln!("[matrix_runner] write-preflight: gates cleared, probing the connection");
            match Runtime::new().map_err(|e| format!("build tokio runtime: {e}")) {
                Ok(rt) => match rt.block_on(probe_connect_async(&dsn)) {
                    Ok(()) => {
                        eprintln!(
                            "[matrix_runner] write-preflight: connection OK, nothing written"
                        );
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("[matrix_runner] write-preflight: {e}");
                        ExitCode::from(6)
                    }
                },
                Err(e) => {
                    eprintln!("[matrix_runner] write-preflight: {e}");
                    ExitCode::from(6)
                }
            }
        }
    }
}

/// Say that the reading is not publishable, and that no database was touched.
fn refuse_bpb(reason: &str) -> ExitCode {
    eprintln!("[matrix_runner] R5-REJECT BPB REFUSED by the write-side screen: {reason}");
    eprintln!(
        "[matrix_runner] No database was contacted and nothing was written. This \
         cell has no publishable measurement; do not retry it into the leaderboard."
    );
    ExitCode::from(EXIT_BPB_REFUSED)
}

fn announce_no_dsn() {
    eprintln!(
        "[matrix_runner] no DSN configured (MATRIX_DATABASE_URL and \
         DATABASE_URL both unset/empty); row only echoed on stdout. \
         Nothing was promised to the SSOT, so this is exit 0."
    );
}

fn announce_not_opted_in() {
    eprintln!(
        "[matrix_runner] a DSN is configured but the ledger write opt-in is not \
         set, so no connection was opened and the row is only echoed on stdout. \
         Nothing was promised to the SSOT, so this is exit 0."
    );
}

/// Open a connection and drop it. No DDL, no INSERT.
///
/// Shares `write_row_async`'s `connect: ` error prefix deliberately: the
/// preflight's whole job is to be the same socket, reached the same way.
async fn probe_connect_async(dsn: &str) -> Result<(), String> {
    let (_client, connection) = tokio_postgres::connect(dsn, NoTls)
        .await
        .map_err(|e| format!("connect: {e}"))?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("[matrix_runner] preflight conn task: {e}");
        }
    });
    Ok(())
}

/// Write a single bpb_samples row into the Railway SSOT.
///
/// SECURITY NOTE (unaddressed here on purpose): the connection below is
/// `NoTls`, so the DSN's password and every row travel the network in
/// plaintext. Adding TLS is a separate change with its own certificate story
/// and is deliberately NOT attempted in this pass; it is recorded so the next
/// reader does not mistake the omission for a decision that the endpoint is
/// safe. Until then, treat `MATRIX_DATABASE_URL` as a credential that has been
/// exposed on every run.
///
/// Uses plain `NoTls` because the Railway DSN in the session is an external
/// proxy `interchange.proxy.rlwy.net:30942` which speaks plaintext PG (the
/// workaround path for the stalled Neon access in trios-railway#62). If
/// callers later switch to a TLS-required endpoint they can point
/// `MATRIX_DATABASE_URL` at it; this binary is intentionally simple and does
/// NOT reuse `src/neon_writer.rs`'s rustls path.
/// SCHEMA LIMIT (2026-08-03, recorded so it is not mistaken for a decision):
/// `lr`, `format_faithful`, `git_dirty` and `git_provenance` are emitted on the
/// `MATRIX_ROW` stdout witness -- the record the matrix-bot reconstructs from --
/// but are NOT inserted below, because `ssot.bpb_samples` has no such columns
/// and adding them is a migration against a live SSOT that this pass cannot
/// exercise (it runs with no DSN by construction). Until that migration lands,
/// the DATABASE row still carries only the labels; the stdout witness is the
/// artefact that carries the proof.
async fn write_row_async(dsn: &str, row: &MatrixRow) -> Result<(), String> {
    let (client, connection) = tokio_postgres::connect(dsn, NoTls)
        .await
        .map_err(|e| format!("connect: {e}"))?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("[matrix_runner] conn task: {e}");
        }
    });

    ensure_schema(&client).await?;

    let stmt = "INSERT INTO ssot.bpb_samples \
                (canon_name, format, algo, hidden, seed, step, bpb, sha, run_id, ts) \
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, to_timestamp($10)) \
                ON CONFLICT DO NOTHING";
    client
        .execute(
            stmt,
            &[
                &row.canon_name,
                &row.format,
                &row.algo,
                &row.hidden,
                &row.seed,
                &row.step,
                &row.bpb,
                &row.sha,
                &row.run_id,
                &(row.ts_unix as f64),
            ],
        )
        .await
        .map_err(|e| format!("insert: {e}"))?;
    Ok(())
}

/// Ensure `ssot.bpb_samples` exists. DDL is idempotent and keeps the worker
/// green even if the bootstrap migration from `trios-railway#62` lands later.
async fn ensure_schema(client: &Client) -> Result<(), String> {
    let ddl = [
        "CREATE SCHEMA IF NOT EXISTS ssot",
        "CREATE TABLE IF NOT EXISTS ssot.bpb_samples (\
            id BIGSERIAL PRIMARY KEY, \
            canon_name TEXT NOT NULL, \
            format TEXT NOT NULL, \
            algo TEXT NOT NULL, \
            hidden INT NOT NULL, \
            seed BIGINT NOT NULL, \
            step INT NOT NULL, \
            bpb DOUBLE PRECISION NOT NULL, \
            sha TEXT, \
            run_id TEXT, \
            ts TIMESTAMPTZ NOT NULL DEFAULT now() \
         )",
        "CREATE INDEX IF NOT EXISTS bpb_samples_format_algo_idx \
            ON ssot.bpb_samples (format, algo)",
        "CREATE INDEX IF NOT EXISTS bpb_samples_canon_name_idx \
            ON ssot.bpb_samples (canon_name)",
        "CREATE INDEX IF NOT EXISTS bpb_samples_ts_desc_idx \
            ON ssot.bpb_samples (ts DESC)",
    ];
    for stmt in ddl.iter() {
        client
            .execute(*stmt, &[])
            .await
            .map_err(|e| format!("ddl {stmt}: {e}"))?;
    }
    Ok(())
}

fn main() -> ExitCode {
    // FIRST STATEMENT, deliberately. The DSN read that decides where this
    // cell's row lands is ~180 lines below, AFTER a full training run and after
    // the `MATRIX_ROW` witness has been printed. Gating there would still be
    // "before the connection", but the operator would have paid for the
    // training and read a witness line first, and the refusal's own sentence
    // ("no database was contacted and nothing was written") would sit under a
    // page of this binary's output. Gating here costs four environment reads
    // and makes that sentence checkable.
    //
    // `MATRIX_DATABASE_URL` is part of `DSN_ENV_VARS` as of 2026-08-05, so this
    // call sees the very pair -- MATRIX_DATABASE_URL vs DATABASE_URL -- that
    // the read below would otherwise resolve by silent precedence.
    trios_trainer::neon_writer::enforce_dsn_conflict_gate();

    let format_raw = arg_or("format", &env_or("TRIOS_FORMAT_TYPE", "fp32"));
    let algo_raw = arg_or("algo", &env_or("TRIOS_ALGO_TYPE", "adamw"));
    // Every one of these used to be `.parse().unwrap_or(<default>)`, so a known
    // flag carrying an unusable value ran at the default and this binary then
    // LABELLED the row with that default. `--seed=oops` published a seed-47
    // row for a run nobody asked for. An unparseable value is a refusal; see
    // `parse_flag_or_reject`.
    // Seed default switched from legacy `42` (FORBIDDEN) to Lucas `47`.
    let seed: i64 = match parse_flag_or_reject("seed", "47") {
        Ok(v) => v,
        Err(code) => return code,
    };
    let dim: i32 = match parse_flag_or_reject("hidden", "128") {
        Ok(v) => v,
        Err(code) => return code,
    };
    let steps: i32 = match parse_flag_or_reject("steps", "3000") {
        Ok(v) => v,
        Err(code) => return code,
    };
    let vocab: i32 = match parse_flag_or_reject("vocab", "128") {
        Ok(v) => v,
        Err(code) => return code,
    };
    let seq: i32 = match parse_flag_or_reject("seq", "32") {
        Ok(v) => v,
        Err(code) => return code,
    };
    // LR default matches Wave-35 champion canon.
    let lr: f64 = match parse_flag_or_reject("lr", "0.001") {
        Ok(v) => v,
        Err(code) => return code,
    };

    // -- R5 GUARD 1 -- ALGO WHITELIST ----------------------------------------
    let algo = algo_raw.trim();
    if !ALGO_WHITELIST.contains(&algo) {
        eprintln!(
            "[matrix_runner] R5-REJECT unsupported algo={algo:?}: \
             only {:?} are implemented. \
             Refusing silent AdamW fallback -- see trios#777 / leaderboard-snapshot Q7. \
             Fix --algo or TRIOS_ALGO_TYPE, then re-run.",
            ALGO_WHITELIST
        );
        return ExitCode::from(3);
    }

    // -- R5 GUARD 2 -- FORMAT ALIAS NORMALIZATION + VALIDATION --------------
    let format = match normalize_format(format_raw.trim()) {
        Ok(f) => f.to_string(),
        Err(e) => {
            eprintln!("[matrix_runner] R5-REJECT {e}");
            return ExitCode::from(5);
        }
    };
    if format != format_raw.trim() {
        eprintln!(
            "[matrix_runner] format alias normalized: {format_raw:?} -> {format:?} \
             (see leaderboard-snapshot Q3 / format-drift)"
        );
    }

    // -- R5 GUARD 3 -- SEED CANON -------------------------------------------
    if !SEED_CANON.contains(&seed) {
        eprintln!(
            "[matrix_runner] R5-REJECT seed={seed} is not in SEED_CANON {:?}. \
             Legacy seeds like 42/43/44/45 are FORBIDDEN.",
            SEED_CANON
        );
        return ExitCode::from(4);
    }

    // -- R5 GUARD 8 -- FORMAT MUST BE A DISTINCT MEASUREMENT ---------------
    // See `resolve_format_faithful`. Checked BEFORE spawning so a degenerate
    // cell costs no compute at all.
    let allow_unfaithful = env_or("TRIOS_ALLOW_UNFAITHFUL_FORMAT", "0") == "1";
    let format_faithful = match resolve_format_faithful(&format, allow_unfaithful) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[matrix_runner] R5-REJECT {e}");
            return ExitCode::from(7);
        }
    };
    if !format_faithful {
        eprintln!(
            "[matrix_runner] WARNING: format={format} is NOT faithful and is \
             running only because TRIOS_ALLOW_UNFAITHFUL_FORMAT=1. The row will \
             carry format_faithful=false. Do not read it as a measurement of \
             {format}."
        );
    }

    // -- R5 GUARD 4 -- CANON NAME (IGLA pattern) ---------------------------
    let lane = arg_or("lane", &env_or("TRIOS_LANE", "MATRIX"));
    let canon_name = build_canon_name_lane(&lane, &format, dim, lr, seed, algo);

    // `--dry-run-canon`: run every label guard above, print the canon_name this
    // cell WOULD be filed under, and exit without training. Exists so the
    // identity key -- the thing `ON CONFLICT` de-duplicates on, and the thing
    // F4 showed was not injective -- is observable from a test in one second
    // instead of one training run.
    if arg_or("dry-run-canon", "0") == "1" || env::args().any(|a| a == "--dry-run-canon") {
        println!("CANON_NAME {canon_name}");
        eprintln!("[matrix_runner] dry-run-canon: no training performed, no row written");
        return ExitCode::SUCCESS;
    }

    // `--write-preflight=<bpb>`: the write-path gate, on a supplied reading,
    // without training. See `run_write_preflight` for why this seam exists.
    if let Some(raw) = arg_opt("write-preflight") {
        return run_write_preflight(&raw);
    }

    eprintln!(
        "[matrix_runner] cell: format={format} algo={algo} seed={seed} \
         hidden={dim} lr={lr} steps={steps} vocab={vocab} seq={seq} \
         format_faithful={format_faithful}"
    );

    let result = match run_cpu_train(&format, algo, seed, dim, steps, vocab, seq, lr) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[matrix_runner] cpu_train ERROR: {e}");
            return ExitCode::from(2);
        }
    };

    // -- R5 GUARD 5 -- EXECUTED FORMAT == REQUESTED FORMAT ------------------
    // See `check_executed_format` for the bind_executed_optimizer rationale.
    if result.format_executed.is_empty() {
        eprintln!(
            "[matrix_runner] R5-REJECT cpu_train did not report `format_executed`. \
             Without it the `format` column is a request, not a measurement. \
             Rebuild cpu_train (>= 2026-08-03) and re-run."
        );
        return ExitCode::from(5);
    }
    if let Err(e) = check_executed_format(&format, &result.format_executed) {
        eprintln!("[matrix_runner] R5-REJECT {e}");
        return ExitCode::from(5);
    }

    // -- R5 GUARD 9 -- EXECUTED LR == REQUESTED LR --------------------------
    // F1: a cell that cannot PROVE it ran at the labelled lr must not become a
    // row. See `check_executed_lr`.
    if let Err(e) = check_executed_lr(lr, result.lr) {
        eprintln!("[matrix_runner] R5-REJECT {e}");
        return ExitCode::from(8);
    }
    let executed_lr = result.lr.unwrap_or(lr);

    let (sha, git_dirty, git_provenance) = git_identity();

    let row = MatrixRow {
        canon_name,
        format: format.clone(),
        algo: algo.to_string(),
        hidden: dim,
        seed,
        step: result.steps as i32,
        bpb: result.final_bpb,
        best_bpb: result.best_bpb,
        initial_bpb: result.initial_bpb,
        delta_bpb: result.delta_bpb,
        format_executed: result.format_executed.clone(),
        lr: executed_lr,
        format_faithful,
        sha,
        git_dirty,
        git_provenance: git_provenance.to_string(),
        run_id: env_or("GITHUB_RUN_ID", &format!("local-{}", now_unix())),
        ts_unix: now_unix(),
    };

    // R7 witness line: machine-parseable, one row per cell, grepable in CI.
    println!(
        "MATRIX_ROW {}",
        serde_json::to_string(&row).unwrap_or_else(|_| "{}".to_string())
    );

    // -- R5 GUARD 10 -- THE WRITE-SIDE SCREEN AND THE WRITE OPT-IN ---------
    //
    // Both gates run here, BEFORE any socket exists. `write_row_async` below
    // carries its own connection and never enters `neon_writer`, so until this
    // call every row this binary published skipped `reject_bpb` entirely and
    // treated the mere presence of a DSN as permission to write. See the
    // UNSCREENED ROWS note at the top of this file for what that means for the
    // rows already in `ssot.bpb_samples`.
    //
    // -- R5 GUARD 6 -- A CONFIGURED DSN THAT LANDS NOTHING IS A FAILURE -----
    //
    // The previous version printed "DB write skipped" for a hard connection
    // failure and returned ExitCode::SUCCESS regardless, while the workflow's
    // collect-ledger job gates on exactly that success. Three success reports
    // over three failures, on the pipeline that feeds the live SSOT.
    //
    // The distinction that matters is CONFIGURED vs NOT CONFIGURED:
    //   * no DSN in the environment -> nothing was promised, exit 0;
    //   * a DSN was configured and 0 of N writes landed -> exit non-zero.
    match resolve_write_gate(row.bpb, configured_dsn()) {
        WriteGate::Refused(reason) => return refuse_bpb(&reason),
        WriteGate::NoDsn => announce_no_dsn(),
        WriteGate::NotOptedIn => announce_not_opted_in(),
        WriteGate::Cleared(dsn) => {
            let attempted = 1usize;
            let mut landed = 0usize;
            let mut last_error = String::new();
            match Runtime::new() {
                Ok(rt) => match rt.block_on(write_row_async(&dsn, &row)) {
                    Ok(()) => {
                        landed += 1;
                        eprintln!("[matrix_runner] wrote row to ssot.bpb_samples");
                    }
                    Err(e) => {
                        last_error = e;
                    }
                },
                Err(e) => {
                    last_error = format!("build tokio runtime: {e}");
                }
            }
            if landed == 0 {
                eprintln!(
                    "[matrix_runner] DB WRITE FAILED: a DSN was configured, \
                     {attempted} write(s) attempted, {landed} landed. \
                     Last error: {last_error}"
                );
                eprintln!(
                    "[matrix_runner] This cell produced no row in ssot.bpb_samples. \
                     Exiting non-zero so downstream jobs do not treat it as landed."
                );
                return ExitCode::from(6);
            }
            eprintln!("[matrix_runner] DB write summary: attempted={attempted} landed={landed}");
        }
    }

    // -- R5 GUARD 7 -- THE SHARED neon_writer LEDGER ------------------------
    //
    // GUARD 6 above covers this binary's OWN raw INSERT. `src/neon_writer.rs`
    // keeps a second, process-wide tally for every write that goes through it,
    // and `neon_writer::ledger_exit_code()` is the one verdict the other
    // trainer binaries honour. Reporting it here means a future call routed
    // through that module cannot be dropped silently just because this main
    // ended in `ExitCode::SUCCESS`. With no DSN configured it returns 0, so
    // the ordinary exit status is unchanged.
    let landed = trios_trainer::neon_writer::landed_writes();
    let dropped = trios_trainer::neon_writer::dropped_writes();
    eprintln!(
        "[matrix_runner] ledger: attempted={} landed={landed} dropped={dropped}",
        landed + dropped
    );
    let ledger_code = trios_trainer::neon_writer::ledger_exit_code();
    if ledger_code != 0 {
        return ExitCode::from(ledger_code as u8);
    }

    ExitCode::SUCCESS
}

// ----------------------------------------------------------------------------
// TESTS -- R5 guards verified at compile-time
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn algo_whitelist_contains_only_implemented() {
        assert_eq!(ALGO_WHITELIST.len(), 2);
        assert!(ALGO_WHITELIST.contains(&"adamw"));
        assert!(ALGO_WHITELIST.contains(&"muon"));
        assert!(
            !ALGO_WHITELIST.contains(&"muon-cwd"),
            "muon-cwd must stay out until cpu_train.rs implements it (see follow-up issue)"
        );
        for &fake in &["lion", "soap", "tiger", "lamb", "prodigy", "adafactor"] {
            assert!(!ALGO_WHITELIST.contains(&fake), "fake algo leaked: {fake}");
        }
    }

    #[test]
    fn seed_canon_excludes_forbidden() {
        for &forbidden in &[42_i64, 43, 44, 45] {
            assert!(
                !SEED_CANON.contains(&forbidden),
                "FORBIDDEN seed {forbidden} leaked into canon"
            );
        }
        for &allowed in &[47_i64, 89, 123, 144, 1597, 2584, 4181] {
            assert!(
                SEED_CANON.contains(&allowed),
                "Lucas/Fibonacci seed {allowed} missing"
            );
        }
    }

    #[test]
    fn format_alias_normalization() {
        assert_eq!(normalize_format("binary16").unwrap(), "fp16");
        assert_eq!(normalize_format("binary32").unwrap(), "fp32");
        assert_eq!(normalize_format("fp8e4m3").unwrap(), "fp8_e4m3");
        assert_eq!(normalize_format("fp8e5m2").unwrap(), "fp8_e5m2");
        assert_eq!(normalize_format("float16").unwrap(), "fp16");
        // Canonical names pass through untouched.
        assert_eq!(normalize_format("fp16").unwrap(), "fp16");
        assert_eq!(normalize_format("gf16").unwrap(), "gf16");
        assert_eq!(normalize_format("bf16").unwrap(), "bf16");
        // The whole nightly format axis must survive validation.
        for f in [
            "fp32", "f32", "gf16", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "int8", "int4", "nf4",
            "posit16", "fp80",
        ] {
            assert!(normalize_format(f).is_ok(), "nightly format rejected: {f}");
        }
    }

    #[test]
    fn unknown_format_spelling_is_rejected() {
        // The exact string that used to reach the `format` column untouched
        // while cpu_train silently ran F32.
        for bogus in ["surveyprobe2", "fp13", "", "FP16 ", "not-a-format"] {
            assert!(
                normalize_format(bogus).is_err(),
                "unvalidated format leaked through normalize_format: {bogus:?}"
            );
        }
    }

    #[test]
    fn executed_format_must_match_requested() {
        assert!(check_executed_format("fp16", "fp16").is_ok());
        // Alias spellings that resolve to the same kernel are a match.
        assert!(check_executed_format("fp32", "f32").is_ok());
        // The silent-F32-fallback signature.
        assert!(check_executed_format("gf16", "f32").is_err());
        assert!(check_executed_format("fp8_e4m3", "fp32").is_err());
        // An unresolvable executed name is an error, not a pass.
        assert!(check_executed_format("fp16", "surveyprobe2").is_err());
        assert!(check_executed_format("fp16", "").is_err());
    }

    #[test]
    fn canon_name_matches_igla_pattern() {
        let canon = build_canon_name("fp16", 128, 0.001, 1597, "adamw");
        assert_eq!(canon, "IGLA-MATRIX-fp16-h128-LR001-rng1597-adamw");

        // Compound algo names with internal dashes pass through as-is.
        // (Historical: this used `muon-cwd` before the 2026-05-14 hotfix
        // removed it from the whitelist; we keep the dash-preservation
        // contract under a hypothetical name so future re-adds are safe.)
        let canon_dash = build_canon_name("gf16", 384, 0.0001, 2584, "muon-cwd");
        assert_eq!(canon_dash, "IGLA-MATRIX-gf16-h384-LR0001-rng2584-muon-cwd");
    }

    #[test]
    fn canon_name_honours_lane_registry() {
        // Default MATRIX lane via legacy entry-point.
        assert_eq!(
            build_canon_name("fp16", 128, 0.001, 1597, "adamw"),
            "IGLA-MATRIX-fp16-h128-LR001-rng1597-adamw"
        );
        // SHORT-WAVE-MATRIX lane (matches champion canon LANE).
        assert_eq!(
            build_canon_name_lane("SHORT-WAVE-MATRIX", "gf16", 128, 0.0001, 1597, "adamw"),
            "IGLA-SHORT-WAVE-MATRIX-gf16-h128-LR0001-rng1597-adamw"
        );
        // Lane is upper-cased even if caller passes lowercase.
        assert_eq!(
            build_canon_name_lane("matrix", "fp32", 64, 0.001, 47, "muon"),
            "IGLA-MATRIX-fp32-h64-LR001-rng47-muon"
        );
        // SCARAB-ADAMW lane (legacy wave registry entry).
        assert_eq!(
            build_canon_name_lane("SCARAB-ADAMW", "binary16", 384, 0.0001, 123, "adamw"),
            "IGLA-SCARAB-ADAMW-binary16-h384-LR0001-rng123-adamw"
        );
    }

    #[test]
    fn lr_token_compact_form() {
        assert_eq!(format_lr_token(0.001), "001");
        assert_eq!(format_lr_token(0.0001), "0001");
        assert_eq!(format_lr_token(0.003), "003");
        assert_eq!(format_lr_token(0.01), "01");
    }

    /// F4: `canon_name` is the ledger's identity key that `ON CONFLICT`
    /// de-duplicates on, so two LRs sharing a token merge two recipes into one
    /// row. The two collisions that existed are named explicitly.
    #[test]
    fn lr_token_is_injective() {
        // The full LR set the two workflows can produce: the dispatch input
        // default (0.001,0.0001), the nightly and PR-smoke constant (0.001),
        // cpu_train's own default (0.003), plus the pairs that used to collide.
        let lrs: &[f64] = &[
            0.001, 0.0001, 0.003, 0.01, 0.1, 0.5, 1.0, 1.5, 3.0, 0.9, 1e-5, 1e-6, 1e-7, 0.0,
        ];
        let mut seen: Vec<(f64, String)> = Vec::new();
        for &lr in lrs {
            let token = format_lr_token(lr);
            for (prev_lr, prev_token) in &seen {
                assert_ne!(
                    *prev_token, token,
                    "LR token collision: {prev_lr} and {lr} both render {token:?}; \
                     canon_name would de-duplicate two different recipes into one row"
                );
            }
            seen.push((lr, token));
        }
        // The two documented collisions, called out individually so a
        // regression names the defect rather than an index.
        assert_ne!(
            format_lr_token(0.1),
            format_lr_token(1.0),
            "0.1 and 1.0 both rendered LR1 before this fix"
        );
        assert_ne!(
            format_lr_token(1e-7),
            format_lr_token(0.0),
            "1e-7 and 0 both rendered LR0 before this fix ({:.6} truncation)",
            1e-7
        );
    }

    /// F1: the missing `--lr` is the whole defect. Assert on the argv itself so
    /// the regression is caught without spawning a trainer.
    #[test]
    fn cpu_train_argv_carries_lr() {
        let argv = build_cpu_train_argv("adamw", 1597, 128, 200, 128, 32, 0.001);
        assert!(
            argv.iter().any(|a| a == "--lr=0.001"),
            "argv does not pass --lr, so the child would train at its own \
             default 0.003 while the row is labelled 0.001: {argv:?}"
        );
        let argv_high = build_cpu_train_argv("adamw", 1597, 128, 200, 128, 32, 0.9);
        assert!(argv_high.iter().any(|a| a == "--lr=0.9"), "{argv_high:?}");
    }

    /// F1 round trip: a cell that cannot prove it ran at the labelled lr must
    /// not become a row.
    #[test]
    fn executed_lr_must_match_requested() {
        // The exact value cpu_train reports for `--lr=0.001` (f32 widened).
        assert!(check_executed_lr(0.001, Some(0.0010000000474974513)).is_ok());
        assert!(check_executed_lr(0.9, Some(0.8999999761581421)).is_ok());
        assert!(check_executed_lr(0.003, Some(0.003000000026077032)).is_ok());
        // The defect signature: label says 0.001, trainer ran its default.
        assert!(check_executed_lr(0.001, Some(0.003000000026077032)).is_err());
        assert!(check_executed_lr(0.9, Some(0.003000000026077032)).is_err());
        // A trainer too old to declare `lr` is refused, not assumed to match.
        assert!(check_executed_lr(0.001, None).is_err());
        // Zero is a legal report, distinguishable from "not declared".
        assert!(check_executed_lr(0.0, Some(0.0)).is_ok());
        assert!(check_executed_lr(0.001, Some(0.0)).is_err());
    }

    /// F2: `is_faithful()` had no call site outside its own definition and two
    /// unit tests. This is the call site.
    #[test]
    fn unfaithful_formats_are_refused_by_default() {
        // fp80 is returned UNCHANGED by fake_quantize_f32, so an fp80 row is an
        // fp32 row wearing a different name. posit16 is a 10-bit mantissa mask.
        for degenerate in ["fp80", "posit16"] {
            assert!(
                resolve_format_faithful(degenerate, false).is_err(),
                "{degenerate} must not be publishable as a peer of real kernels"
            );
            assert_eq!(
                resolve_format_faithful(degenerate, true),
                Ok(false),
                "opt-in must permit {degenerate} AND stamp format_faithful=false"
            );
        }
        // Real kernels are unaffected, with or without the opt-in.
        for real in [
            "fp32", "fp16", "bf16", "gf16", "fp8_e4m3", "fp8_e5m2", "int8", "int4",
        ] {
            assert_eq!(
                resolve_format_faithful(real, false),
                Ok(true),
                "real kernel {real} was refused"
            );
            assert_eq!(resolve_format_faithful(real, true), Ok(true));
        }
        // An unresolvable name is an error under both settings -- the opt-in
        // relaxes faithfulness, not validation.
        assert!(resolve_format_faithful("surveyprobe2", true).is_err());
    }

    /// F5: a sha with no dirty flag can name a commit the row was not built
    /// from. The flag must be carried, and an unavailable answer must not be
    /// rendered as "clean".
    #[test]
    fn git_identity_carries_dirty_flag() {
        let (sha, dirty, provenance) = git_identity();
        assert!(!sha.is_empty());
        assert!(sha.len() <= 7, "short sha shape changed: {sha:?}");
        // Whatever the tree state, the provenance string must be one of the
        // declared constants -- never an empty string or a guess.
        assert!(
            matches!(
                provenance,
                trios_trainer::checkpoint::GIT_PROVENANCE_VERIFIED
                    | trios_trainer::checkpoint::GIT_PROVENANCE_ASSERTED
                    | trios_trainer::checkpoint::GIT_PROVENANCE_NONE
            ),
            "unknown provenance {provenance:?}"
        );
        // `None` is a legal answer (git unavailable); `Some(false)` asserted on
        // a tree that was never queried is what the fix forbids.
        if provenance == trios_trainer::checkpoint::GIT_PROVENANCE_NONE {
            assert!(dirty.is_none());
        }
    }
}
