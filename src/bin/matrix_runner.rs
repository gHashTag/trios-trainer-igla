// matrix_runner — Phase C orchestrator for the 312-cell Format×Algorithm matrix
// (gHashTag/trios#446, kernel stubs from PR #101 + #102).
//
// Purpose: run ONE (format, algo, seed, hidden) cell end-to-end by invoking the
// already-instrumented `cpu_train` binary with the right env vars, parse the
// resulting `.trinity/results/cpu_train_<format>_<algo>_seed<seed>.json`, and
// write a row into `ssot.bpb_samples` (Railway phd-postgres-ssot SSOT) using
// the `MATRIX_DATABASE_URL` env var. When the DSN is unset the row is only
// echoed on stdout (R5: never panic, never block CI).
//
// Constitutional notes:
//   * R1 Rust-only — pipeline is pure Rust, no .py or .sh shims.
//   * R5 honest   — on any DB or training error: log and continue.
//   * R7 witness  — emits run_id, sha, step, seed, bpb verbatim so the
//                   matrix-bot (L-C6) can reconstruct #446 body.
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

/// Parsed `.trinity/results/cpu_train_<fmt>_<algo>_seed<seed>.json` payload.
/// We only keep the fields we need for the matrix row; extra fields are
/// ignored by `#[serde(default)]` on every one.
#[derive(Debug, Deserialize, Default)]
struct CpuTrainResult {
    #[serde(default)]
    algo: String,
    #[serde(default)]
    seed: i64,
    #[serde(default)]
    steps: i64,
    #[serde(default)]
    dim: i64,
    #[serde(default)]
    initial_bpb: f64,
    #[serde(default)]
    final_bpb: f64,
    #[serde(default)]
    delta_bpb: f64,
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
    bpb: f64,
    initial_bpb: f64,
    delta_bpb: f64,
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
    let format = arg_or("format", &env_or("TRIOS_FORMAT_TYPE", "f32"));
    let algo = arg_or("algo", &env_or("TRIOS_ALGO_TYPE", "adamw"));
    let seed: i64 = arg_or("seed", "42").parse().unwrap_or(42);
    let dim: i32 = arg_or("hidden", "96").parse().unwrap_or(96);
    let steps: i32 = arg_or("steps", "3000").parse().unwrap_or(3000);
    let vocab: i32 = arg_or("vocab", "128").parse().unwrap_or(128);
    let seq: i32 = arg_or("seq", "32").parse().unwrap_or(32);

    eprintln!(
        "[matrix_runner] cell: format={format} algo={algo} seed={seed} \
         hidden={dim} steps={steps} vocab={vocab} seq={seq}"
    );

    let result = match run_cpu_train(&format, &algo, seed, dim, steps, vocab, seq) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[matrix_runner] cpu_train ERROR: {e}");
            return ExitCode::from(2);
        }
    };

    let row = MatrixRow {
        canon_name: format!("cpu_train_{format}_{algo}"),
        format: format.clone(),
        algo: algo.clone(),
        hidden: dim,
        seed,
        step: result.steps as i32,
        bpb: result.final_bpb,
        initial_bpb: result.initial_bpb,
        delta_bpb: result.delta_bpb,
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

    match dsn {
        Some(dsn) if !dsn.is_empty() => {
            let rt = Runtime::new().expect("build tokio runtime");
            match rt.block_on(write_row_async(&dsn, &row)) {
                Ok(()) => eprintln!("[matrix_runner] wrote row to ssot.bpb_samples"),
                Err(e) => eprintln!("[matrix_runner] DB write skipped: {e}"),
            }
        }
        _ => {
            eprintln!(
                "[matrix_runner] MATRIX_DATABASE_URL unset; row only echoed on stdout \
                 (R5 non-blocking)."
            );
        }
    }

    ExitCode::SUCCESS
}
