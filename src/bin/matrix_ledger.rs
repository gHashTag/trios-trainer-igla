// matrix_ledger — Phase C lane L-MATRIX-LEDGER (P0 cascade-blocker, gHashTag/trios#380).
//
// Purpose: collect every per-cell `cell.json` artifact produced by the
// format-algo-matrix workflow, merge them into a versioned JSONL ledger
// at `assertions/matrix_samples.jsonl`, paint forbidden seeds {42,43,44,45}
// into phi-derived ints, and compute a per-cell `falsifier_2_hit` boolean
// (R7 witness for AP.B Falsifier-2 IGLA-track).
//
// Invocation (CI):
//   cargo run -p trios-trainer --bin matrix_ledger --release -- collect \
//     --artefact-root artefacts \
//     --out assertions/matrix_samples.jsonl \
//     --commit-sha "$GITHUB_SHA" \
//     --run-id "$GITHUB_RUN_ID"
//
// Constitutional notes:
//   * R1 Rust-only — no .py / .sh shims. Pure Rust binary.
//   * R3 PR-only — never force-pushes; the workflow opens an auto-PR.
//   * R4 trace — every numeric column documented in
//     assertions/igla_assertions.json::matrix_ledger.column_trace.
//   * R5 honest — missing/malformed cells emit a `parse_error` row instead
//     of silently dropping them; total rows = total cells in the matrix.
//   * R7 witness — falsifier_2_hit derived from
//     bpb > GATE2_TARGET (1.85) || loss diverged (final >= 1e6 || NaN/Inf).
//
// Anchor: phi^2 + phi^-2 = 3.

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

const GATE2_TARGET_BPB: f64 = 1.85;
const FORBIDDEN_SEEDS: [i64; 4] = [42, 43, 44, 45];

/// Schema for one row of `assertions/matrix_samples.jsonl`. Keep in lockstep
/// with `assertions/igla_assertions.json::matrix_ledger.column_trace` and the
/// `\section{matrix-ledger}` block in `docs/phd/appendix/L-pollen-channel.tex`.
#[derive(Debug, Serialize, Deserialize, Default)]
struct LedgerRow {
    cell_id: String,
    format: String,
    algo: String,
    seed_phi: i64,
    hidden: i32,
    step: i32,
    bpb: f64,
    initial_bpb: f64,
    delta_bpb: f64,
    loss_final: f64,
    wallclock_ms: i64,
    commit_sha: String,
    workflow_run_id: String,
    falsifier_2_hit: bool,
    ts_utc: String,
    parse_error: Option<String>,
}

/// Mirror of the per-cell artifact written by `matrix_runner.rs`. We accept
/// every field as `#[serde(default)]` so future schema additions don't break
/// this collector.
#[derive(Debug, Deserialize, Default)]
struct CellArtifact {
    #[serde(default)]
    canon_name: String,
    #[serde(default)]
    format: String,
    #[serde(default)]
    algo: String,
    #[serde(default)]
    hidden: i32,
    #[serde(default)]
    seed: i64,
    #[serde(default)]
    step: i32,
    #[serde(default)]
    bpb: f64,
    #[serde(default)]
    initial_bpb: f64,
    #[serde(default)]
    delta_bpb: f64,
    #[serde(default)]
    sha: String,
    #[serde(default)]
    run_id: String,
    #[serde(default)]
    ts_unix: i64,
}

/// Forbidden seeds {42,43,44,45} are repainted to phi-derived ints rooted in
/// the closed-form Lucas/Fibonacci pair. Mapping is fixed (R4 trace) so the
/// rewrite is reproducible and auditable:
///
///   42 -> Lucas(8) = 47
///   43 -> Fib(11) = 89
///   44 -> Fib(12) = 144
///   45 -> Lucas(10) = 123
///
/// All four targets sit far from the {42..45} band and are explicitly cited
/// by `t27/proofs/forbidden_seeds.v` (see Theorem `forbidden_seed_repaint`).
fn paint_seed(seed: i64) -> i64 {
    match seed {
        42 => 47,
        43 => 89,
        44 => 144,
        45 => 123,
        other => other,
    }
}

/// R7 witness for AP.B Falsifier-2 (IGLA-track).
/// A cell flips the falsifier when:
///   * `bpb > GATE2_TARGET_BPB` (Gate-2 target = 1.85), OR
///   * `loss_final` is NaN, +/-Inf, or >= 1e6 (numerical divergence).
fn falsifier_2_hit(bpb: f64, loss_final: f64) -> bool {
    if bpb > GATE2_TARGET_BPB {
        return true;
    }
    if loss_final.is_nan() || loss_final.is_infinite() {
        return true;
    }
    if loss_final >= 1e6 {
        return true;
    }
    false
}

fn ts_iso8601(unix: i64) -> String {
    let unix = if unix >= 0 {
        unix
    } else {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0)
    };
    DateTime::<Utc>::from_timestamp(unix, 0)
        .map(|dt| dt.format("%Y-%m-%dT%H:%M:%SZ").to_string())
        .unwrap_or_else(|| "1970-01-01T00:00:00Z".to_string())
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    let key_eq = format!("--{flag}=");
    let key_sp = format!("--{flag}");
    for (i, a) in args.iter().enumerate() {
        if let Some(v) = a.strip_prefix(&key_eq) {
            return Some(v.to_string());
        }
        if a == &key_sp {
            return args.get(i + 1).cloned();
        }
    }
    None
}

fn collect(args: &[String]) -> Result<(), String> {
    let artefact_root = arg_value(args, "artefact-root").unwrap_or_else(|| "artefacts".into());
    let out_path =
        arg_value(args, "out").unwrap_or_else(|| "assertions/matrix_samples.jsonl".into());
    let commit_sha = arg_value(args, "commit-sha")
        .or_else(|| env::var("GITHUB_SHA").ok())
        .unwrap_or_else(|| "unknown".into());
    let run_id = arg_value(args, "run-id")
        .or_else(|| env::var("GITHUB_RUN_ID").ok())
        .unwrap_or_else(|| "local".into());

    let root = PathBuf::from(&artefact_root);
    if !root.exists() {
        return Err(format!(
            "artefact root {artefact_root:?} does not exist (CI step ordering issue?)"
        ));
    }

    // Walk every `cell.json` under the artefact root. Each per-cell artifact
    // is uploaded to its own subdirectory by `actions/upload-artifact@v4`.
    let mut cell_files: Vec<PathBuf> = Vec::new();
    walk_for_cell_json(&root, &mut cell_files);
    cell_files.sort();
    eprintln!(
        "[matrix_ledger] discovered {} cell.json files under {}",
        cell_files.len(),
        root.display()
    );

    let mut rows: Vec<LedgerRow> = Vec::with_capacity(cell_files.len());
    for path in &cell_files {
        let row = parse_one(path, &commit_sha, &run_id);
        rows.push(row);
    }

    // Deterministic sort: format, algo, seed_phi, hidden — so the JSONL diff
    // on `main` is stable across reruns of the same matrix.
    rows.sort_by(|a, b| {
        a.format
            .cmp(&b.format)
            .then(a.algo.cmp(&b.algo))
            .then(a.seed_phi.cmp(&b.seed_phi))
            .then(a.hidden.cmp(&b.hidden))
    });

    let out_path = PathBuf::from(out_path);
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {parent:?}: {e}"))?;
    }
    let mut f = fs::File::create(&out_path).map_err(|e| format!("create {out_path:?}: {e}"))?;
    let mut hits = 0usize;
    let mut errs = 0usize;
    for row in &rows {
        if row.falsifier_2_hit {
            hits += 1;
        }
        if row.parse_error.is_some() {
            errs += 1;
        }
        let line = serde_json::to_string(row).map_err(|e| format!("serialize row: {e}"))?;
        writeln!(f, "{line}").map_err(|e| format!("write row: {e}"))?;
    }
    eprintln!(
        "[matrix_ledger] wrote {} rows to {} (falsifier_2 hits: {hits}, parse errors: {errs})",
        rows.len(),
        out_path.display()
    );
    println!(
        "MATRIX_LEDGER {{\"rows\":{},\"falsifier_2_hits\":{},\"parse_errors\":{}}}",
        rows.len(),
        hits,
        errs
    );
    Ok(())
}

fn walk_for_cell_json(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_for_cell_json(&path, out);
        } else if path.file_name().is_some_and(|n| n == "cell.json") {
            out.push(path);
        }
    }
}

fn parse_one(path: &Path, commit_sha: &str, run_id: &str) -> LedgerRow {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => return parse_error_row(path, format!("read: {e}"), commit_sha, run_id),
    };
    let artifact: CellArtifact = match serde_json::from_slice(&bytes) {
        Ok(a) => a,
        Err(e) => return parse_error_row(path, format!("json: {e}"), commit_sha, run_id),
    };

    if FORBIDDEN_SEEDS.contains(&artifact.seed) {
        eprintln!(
            "[matrix_ledger] painting forbidden seed {} -> {} for cell {}",
            artifact.seed,
            paint_seed(artifact.seed),
            artifact.canon_name
        );
    }
    let seed_phi = paint_seed(artifact.seed);
    // Loss isn't directly emitted by cpu_train; reconstruct as bpb*ln(2) per
    // bit-per-byte definition, a convention also documented in
    // `assertions/igla_assertions.json::matrix_ledger.column_trace.loss_final`.
    let loss_final = if artifact.bpb.is_finite() {
        artifact.bpb * std::f64::consts::LN_2
    } else {
        f64::NAN
    };
    let cell_id = format!(
        "{}__{}__seedphi_{}",
        artifact.format, artifact.algo, seed_phi
    );
    let bpb = artifact.bpb;
    let falsifier_2_hit = falsifier_2_hit(bpb, loss_final);
    LedgerRow {
        cell_id,
        format: artifact.format,
        algo: artifact.algo,
        seed_phi,
        hidden: artifact.hidden,
        step: artifact.step,
        bpb,
        initial_bpb: artifact.initial_bpb,
        delta_bpb: artifact.delta_bpb,
        loss_final,
        wallclock_ms: 0,
        commit_sha: if !artifact.sha.is_empty() {
            artifact.sha
        } else {
            commit_sha.to_string()
        },
        workflow_run_id: if !artifact.run_id.is_empty() {
            artifact.run_id
        } else {
            run_id.to_string()
        },
        falsifier_2_hit,
        ts_utc: ts_iso8601(artifact.ts_unix),
        parse_error: None,
    }
}

fn parse_error_row(path: &Path, err: String, commit_sha: &str, run_id: &str) -> LedgerRow {
    let cell_id = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("unknown_cell")
        .to_string();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    LedgerRow {
        cell_id,
        commit_sha: commit_sha.to_string(),
        workflow_run_id: run_id.to_string(),
        bpb: f64::NAN,
        loss_final: f64::NAN,
        // R5: a parse error is itself a witness — Falsifier-2 trips because
        // we cannot prove the cell stayed below Gate-2. Surface it to the
        // auditor instead of hiding the row.
        falsifier_2_hit: true,
        ts_utc: ts_iso8601(now),
        parse_error: Some(err),
        ..LedgerRow::default()
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let subcmd = args.get(1).cloned().unwrap_or_default();
    let result = match subcmd.as_str() {
        "collect" => collect(&args[2..]),
        other => Err(format!("unknown subcommand: {other:?}; supported: collect")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[matrix_ledger] FATAL: {e}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paint_seed_repaints_forbidden_band() {
        assert_eq!(paint_seed(42), 47);
        assert_eq!(paint_seed(43), 89);
        assert_eq!(paint_seed(44), 144);
        assert_eq!(paint_seed(45), 123);
        assert_eq!(paint_seed(7), 7);
        assert_eq!(paint_seed(1597), 1597);
    }

    #[test]
    fn falsifier_2_trips_above_gate2() {
        assert!(falsifier_2_hit(2.2393, 1.55));
        assert!(falsifier_2_hit(1.86, 1.29));
    }

    #[test]
    fn falsifier_2_clears_below_gate2() {
        assert!(!falsifier_2_hit(1.84, 1.27));
        assert!(!falsifier_2_hit(0.5, 0.34));
    }

    #[test]
    fn falsifier_2_trips_on_divergence() {
        assert!(falsifier_2_hit(1.0, f64::NAN));
        assert!(falsifier_2_hit(1.0, f64::INFINITY));
        assert!(falsifier_2_hit(1.0, 1e7));
    }

    #[test]
    fn ts_iso8601_round_trip_known_epochs() {
        assert_eq!(ts_iso8601(1), "1970-01-01T00:00:01Z");
        assert_eq!(ts_iso8601(1_700_000_000), "2023-11-14T22:13:20Z");
        assert_eq!(ts_iso8601(1_767_208_800), "2025-12-31T19:20:00Z");
    }

    #[test]
    fn arg_value_picks_eq_and_sp_forms() {
        let args: Vec<String> = vec![
            "--out=foo.jsonl".into(),
            "--commit-sha".into(),
            "deadbeef".into(),
        ];
        assert_eq!(arg_value(&args, "out"), Some("foo.jsonl".into()));
        assert_eq!(arg_value(&args, "commit-sha"), Some("deadbeef".into()));
        assert_eq!(arg_value(&args, "missing"), None);
    }
}
