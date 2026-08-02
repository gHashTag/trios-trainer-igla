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
fn format_lr_token(lr: f64) -> String {
    // Map common LRs to their compact tokens.
    // Anything outside the table falls back to a dot-notation string.
    let s = format!("{:.6}", lr);
    // strip trailing zeros, then "0." prefix
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    if let Some(rest) = trimmed.strip_prefix("0.") {
        // 0.001 -> "001", 0.0001 -> "0001"
        rest.to_string()
    } else {
        // 1.5 or 0.5 etc. -- keep as-is, dot is allowed by the regex.
        trimmed.to_string()
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
    sha: String,
    run_id: String,
    ts_unix: i64,
}

fn arg_or(flag: &str, default: &str) -> String {
    let key = format!("--{flag}=");
    for a in env::args() {
        if let Some(v) = a.strip_prefix(&key) {
            return v.to_string();
        }
    }
    default.to_string()
}

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Short git sha if we're inside a git tree, else "unknown".
fn git_sha() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn run_cpu_train(
    format_type: &str,
    algo: &str,
    seed: i64,
    dim: i32,
    steps: i32,
    vocab: i32,
    seq: i32,
) -> Result<CpuTrainResult, String> {
    // Prefer `cargo run --release --bin cpu_train` so the child resolves the
    // already-compiled artefact from `target/release/`. In CI this is
    // pre-built in an earlier job, so the call is a no-op rebuild.
    let mut cmd = Command::new("cargo");
    cmd.args([
        "run",
        "--quiet",
        "--release",
        "--bin",
        "cpu_train",
        "--",
        &format!("--seed={seed}"),
        &format!("--steps={steps}"),
        &format!("--dim={dim}"),
        &format!("--vocab={vocab}"),
        &format!("--seq={seq}"),
        &format!("--algo={algo}"),
    ]);
    cmd.env("TRIOS_FORMAT_TYPE", format_type);
    cmd.env("TRIOS_ALGO_TYPE", algo);

    eprintln!(
        "[matrix_runner] spawning cpu_train format={format_type} algo={algo} \
         seed={seed} dim={dim} steps={steps}"
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

    let path: PathBuf = PathBuf::from(format!(
        ".trinity/results/cpu_train_{format_type}_{algo}_seed{seed}.json"
    ));
    let bytes = fs::read(&path).map_err(|e| format!("read result {path:?}: {e}"))?;
    let parsed: CpuTrainResult =
        serde_json::from_slice(&bytes).map_err(|e| format!("parse result {path:?}: {e}"))?;
    Ok(parsed)
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
    let format_raw = arg_or("format", &env_or("TRIOS_FORMAT_TYPE", "fp32"));
    let algo_raw = arg_or("algo", &env_or("TRIOS_ALGO_TYPE", "adamw"));
    // Seed default switched from legacy `42` (FORBIDDEN) to Lucas `47`.
    let seed: i64 = arg_or("seed", "47").parse().unwrap_or(47);
    let dim: i32 = arg_or("hidden", "128").parse().unwrap_or(128);
    let steps: i32 = arg_or("steps", "3000").parse().unwrap_or(3000);
    let vocab: i32 = arg_or("vocab", "128").parse().unwrap_or(128);
    let seq: i32 = arg_or("seq", "32").parse().unwrap_or(32);
    // LR default matches Wave-35 champion canon.
    let lr: f64 = arg_or("lr", "0.001").parse().unwrap_or(0.001);

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

    eprintln!(
        "[matrix_runner] cell: format={format} algo={algo} seed={seed} \
         hidden={dim} lr={lr} steps={steps} vocab={vocab} seq={seq}"
    );

    let result = match run_cpu_train(&format, algo, seed, dim, steps, vocab, seq) {
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

    // -- R5 GUARD 4 -- CANON NAME (IGLA pattern) ---------------------------
    let lane = arg_or("lane", &env_or("TRIOS_LANE", "MATRIX"));
    let canon_name = build_canon_name_lane(&lane, &format, dim, lr, seed, algo);

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
        sha: git_sha(),
        run_id: env_or("GITHUB_RUN_ID", &format!("local-{}", now_unix())),
        ts_unix: now_unix(),
    };

    // R7 witness line: machine-parseable, one row per cell, grepable in CI.
    println!(
        "MATRIX_ROW {}",
        serde_json::to_string(&row).unwrap_or_else(|_| "{}".to_string())
    );

    let dsn = env::var("MATRIX_DATABASE_URL")
        .ok()
        .or_else(|| env::var("DATABASE_URL").ok());

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
    match dsn {
        Some(dsn) if !dsn.is_empty() => {
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
            eprintln!(
                "[matrix_runner] DB write summary: attempted={attempted} landed={landed}"
            );
        }
        _ => {
            eprintln!(
                "[matrix_runner] no DSN configured (MATRIX_DATABASE_URL and \
                 DATABASE_URL both unset/empty); row only echoed on stdout. \
                 Nothing was promised to the SSOT, so this is exit 0."
            );
        }
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
}
