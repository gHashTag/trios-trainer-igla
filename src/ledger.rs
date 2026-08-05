//! Triplet-validated emit to `assertions/seed_results.jsonl`.
//!
//! Enforces the standing rule:
//! `BPB=<v> @ step=<N> seed=<S> sha=<7c> jsonl_row=<L> gate_status=<g>`.
//!
//! Refuses to write if:
//! - SHA is in `.embargo`
//! - step < 4000 (R8 — Gate-2 candidate floor)
//! - target_bpb < 0 (config corruption)
//!
//! Publishing is a separate act from recording. `ledger.push = true` alone no
//! longer runs git: [`ALLOW_PUSH_ENV`] must also be set to `1`, or the push
//! fails loudly without spawning a single subprocess.

use anyhow::{bail, Context, Result};
use chrono::Utc;
use serde::Serialize;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::Command;

use crate::TrainConfig;

#[derive(Debug, Serialize)]
pub struct LedgerRow {
    pub agent: String,
    pub bpb: f64,
    pub step: usize,
    pub seed: u64,
    pub sha: String,
    pub jsonl_row: usize,
    pub gate_status: String,
    pub ts: String,
}

/// Emit a ledger row using the provided SHA (skips `git rev-parse`).
/// Useful for tests that need to verify embargo enforcement without git.
pub fn emit_row_with_sha(cfg: &TrainConfig, bpb: f64, step: usize, sha: &str) -> Result<LedgerRow> {
    if step < 4000 {
        bail!(
            "R8 violation: step {} < 4000. Gate-2 row requires ≥ 4000 steps.",
            step
        );
    }
    if !bpb.is_finite() || bpb <= 0.0 {
        bail!("non-finite BPB {bpb}");
    }
    if is_embargoed(&cfg.ledger.embargo_path, sha)? {
        bail!("embargo violation: SHA {sha} is in embargo list");
    }
    let gate_status = if bpb < cfg.target_bpb {
        "victory_candidate".into()
    } else if let Some(c) = cfg.champion_bpb {
        if bpb < c {
            "below_champion".into()
        } else {
            "below_target_evidence".into()
        }
    } else {
        "below_target_evidence".into()
    };
    let jsonl_row = next_row_index(&cfg.ledger.jsonl_path)?;
    let row = LedgerRow {
        agent: format!("trios-trainer-{}", cfg.name),
        bpb,
        step,
        seed: cfg.seed,
        sha: sha.to_string(),
        jsonl_row,
        gate_status,
        ts: Utc::now().to_rfc3339(),
    };
    append_row(&cfg.ledger.jsonl_path, &row)?;
    Ok(row)
}

pub fn emit_row(cfg: &TrainConfig, bpb: f64, step: usize) -> Result<LedgerRow> {
    if step < 4000 {
        bail!(
            "R8 violation: step {} < 4000. Gate-2 row requires ≥ 4000 steps.",
            step
        );
    }
    if !bpb.is_finite() || bpb <= 0.0 {
        bail!("non-finite BPB {bpb}");
    }

    let sha = head_sha7()?;
    if is_embargoed(&cfg.ledger.embargo_path, &sha)? {
        bail!("embargo violation: HEAD SHA {sha} is in embargo list");
    }

    let gate_status = if bpb < cfg.target_bpb {
        "victory_candidate".into()
    } else if let Some(c) = cfg.champion_bpb {
        if bpb < c {
            "below_champion".into()
        } else {
            "below_target_evidence".into()
        }
    } else {
        "below_target_evidence".into()
    };

    let jsonl_row = next_row_index(&cfg.ledger.jsonl_path)?;
    let row = LedgerRow {
        agent: format!("trios-trainer-{}", cfg.name),
        bpb,
        step,
        seed: cfg.seed,
        sha,
        jsonl_row,
        gate_status,
        ts: Utc::now().to_rfc3339(),
    };

    append_row(&cfg.ledger.jsonl_path, &row)?;

    if cfg.ledger.push {
        push_row(&cfg.ledger.jsonl_path, &row)?;
    }

    Ok(row)
}

fn head_sha7() -> Result<String> {
    let out = Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .output()
        .context("git rev-parse")?;
    Ok(String::from_utf8(out.stdout)?.trim().to_string())
}

pub fn is_embargoed<P: AsRef<Path>>(path: P, sha: &str) -> Result<bool> {
    let p = path.as_ref();
    if !p.exists() {
        return Ok(false);
    }
    let f = std::fs::File::open(p).with_context(|| format!("open {}", p.display()))?;
    for line in BufReader::new(f).lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with(sha) || sha.starts_with(trimmed) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Index the next appended row will occupy, counting data rows only.
///
/// This used to be `lines().count()`, which counted the `_schema` header the
/// ledger carries on its first line. Every emitted `jsonl_row` was therefore
/// one greater than the row's actual index among the data rows, so the triplet
/// `BPB=... jsonl_row=<L>` pointed at the wrong line of the file it cites.
/// Blank lines are skipped for the same reason: they are not rows.
fn next_row_index<P: AsRef<Path>>(path: P) -> Result<usize> {
    let p = path.as_ref();
    if !p.exists() {
        return Ok(0);
    }
    let f = std::fs::File::open(p)?;
    let mut rows = 0usize;
    for line in BufReader::new(f).lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(serde_json::Value::Object(obj)) =
            serde_json::from_str::<serde_json::Value>(trimmed)
        {
            if obj.contains_key("_schema") {
                continue;
            }
        }
        rows += 1;
    }
    Ok(rows)
}

fn append_row<P: AsRef<Path>>(path: P, row: &LedgerRow) -> Result<()> {
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    let json = serde_json::to_string(row)?;
    writeln!(f, "{json}")?;
    Ok(())
}

/// Opt-in for the git side effect. Must be exactly `1`.
///
/// Recording a measurement and publishing it are different acts. This library
/// used to do the second as a side effect of the first: any config with
/// `ledger.push = true` turned a training run into `git add` + `git commit` +
/// `git push origin HEAD`. The gate is a refusal the caller can see, never a
/// silent skip, so a run that was configured to publish and did not says so.
pub const ALLOW_PUSH_ENV: &str = "TRIOS_LEDGER_ALLOW_PUSH";

/// True only when [`ALLOW_PUSH_ENV`] is set to exactly `1`.
pub fn push_allowed() -> bool {
    std::env::var(ALLOW_PUSH_ENV).as_deref() == Ok("1")
}

/// Real git runner. `args[0]` names the subcommand for the error message.
fn run_git(args: &[&str]) -> Result<()> {
    let st = Command::new("git").args(args).status()?;
    if !st.success() {
        bail!("git {} failed", args[0]);
    }
    Ok(())
}

fn push_row<P: AsRef<Path>>(path: P, row: &LedgerRow) -> Result<()> {
    push_row_with(path, row, run_git)
}

/// Body of [`push_row`] with the git invocation injected, so a test can prove
/// that the refusal path issues no git command at all rather than assert it.
fn push_row_with<P, F>(path: P, row: &LedgerRow, mut git: F) -> Result<()>
where
    P: AsRef<Path>,
    F: FnMut(&[&str]) -> Result<()>,
{
    if !push_allowed() {
        bail!(
            "ledger push refused: {ALLOW_PUSH_ENV} is not set to 1. \
             No git subprocess was run: no add, no commit, no push."
        );
    }
    let p = path.as_ref();
    let path_str = p
        .to_str()
        .with_context(|| format!("ledger path is not UTF-8: {}", p.display()))?;
    // Single commit per row with the triplet in the message — easy to grep
    let msg = format!(
        "feat(igla-trainer): row {} BPB={:.4} @ {}K seed={} sha={} status={}",
        row.jsonl_row,
        row.bpb,
        row.step / 1000,
        row.seed,
        row.sha,
        row.gate_status
    );
    // Stage the file this row was appended to. The literal
    // `assertions/seed_results.jsonl` used to be hardcoded here, so a run
    // writing anywhere else staged a path it had not touched.
    git(&["add", path_str])?;
    git(&["commit", "-m", &msg])?;
    git(&["push", "origin", "HEAD"])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use std::sync::Mutex;

    /// Serialize mutation of the process-global `TRIOS_LEDGER_ALLOW_PUSH`.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn sample_row() -> LedgerRow {
        LedgerRow {
            agent: "trios-trainer-test".to_string(),
            bpb: 2.5,
            step: 5000,
            seed: 47,
            sha: "abc1234".to_string(),
            jsonl_row: 0,
            gate_status: "below_target_evidence".to_string(),
            ts: "2026-08-03T00:00:00Z".to_string(),
        }
    }

    /// Record every git invocation instead of running one.
    fn recorder(calls: &mut Vec<Vec<String>>) -> impl FnMut(&[&str]) -> Result<()> + '_ {
        move |args: &[&str]| {
            calls.push(args.iter().map(|s| s.to_string()).collect());
            Ok(())
        }
    }

    #[test]
    fn push_is_refused_without_the_env_var_and_spawns_no_git() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var(ALLOW_PUSH_ENV);
        let mut calls: Vec<Vec<String>> = Vec::new();
        let err = push_row_with("ledger.jsonl", &sample_row(), recorder(&mut calls))
            .expect_err("push must be refused when the opt-in is unset");
        assert!(
            err.to_string().contains(ALLOW_PUSH_ENV),
            "refusal must name the variable: {err}"
        );
        assert!(
            calls.is_empty(),
            "refusal ran git commands: {calls:?} (must be none: no add, no commit, no push)"
        );
    }

    #[test]
    fn push_is_refused_when_the_env_var_is_not_exactly_one() {
        let _g = ENV_LOCK.lock().unwrap();
        for val in ["", "0", "true", "yes", "1 "] {
            std::env::set_var(ALLOW_PUSH_ENV, val);
            let mut calls: Vec<Vec<String>> = Vec::new();
            let err = push_row_with("ledger.jsonl", &sample_row(), recorder(&mut calls))
                .expect_err("only the exact value 1 may arm the push");
            assert!(
                err.to_string().contains(ALLOW_PUSH_ENV),
                "value {val:?}: {err}"
            );
            assert!(calls.is_empty(), "value {val:?} ran git: {calls:?}");
        }
        std::env::remove_var(ALLOW_PUSH_ENV);
    }

    #[test]
    fn armed_push_stages_the_path_it_was_given() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::set_var(ALLOW_PUSH_ENV, "1");
        let mut calls: Vec<Vec<String>> = Vec::new();
        let res = push_row_with(
            "some/other/ledger.jsonl",
            &sample_row(),
            recorder(&mut calls),
        );
        std::env::remove_var(ALLOW_PUSH_ENV);
        res.expect("armed push must succeed with a recording runner");
        assert_eq!(calls.len(), 3, "expected add, commit, push: {calls:?}");
        assert_eq!(calls[0], vec!["add", "some/other/ledger.jsonl"]);
        assert_eq!(calls[1][0], "commit");
        assert_eq!(calls[2], vec!["push", "origin", "HEAD"]);
    }

    #[test]
    fn next_row_index_skips_the_schema_header() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("seed_results.jsonl");
        let mut f = std::fs::File::create(&path).expect("create");
        writeln!(
            f,
            r#"{{"_schema":"BPB=<v> @ step=<N> seed=<S>","version":1}}"#
        )
        .expect("write");
        drop(f);
        assert_eq!(
            next_row_index(&path).expect("index"),
            0,
            "first data row must be index 0, not 1"
        );
    }

    #[test]
    fn next_row_index_counts_data_rows_after_the_header() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("seed_results.jsonl");
        let mut f = std::fs::File::create(&path).expect("create");
        writeln!(f, r#"{{"_schema":"x"}}"#).expect("write");
        writeln!(f, r#"{{"bpb":2.61,"step":12000,"seed":47}}"#).expect("write");
        writeln!(f).expect("write");
        writeln!(f, r#"{{"bpb":2.60,"step":12000,"seed":89}}"#).expect("write");
        drop(f);
        assert_eq!(next_row_index(&path).expect("index"), 2);
    }

    #[test]
    fn next_row_index_of_a_missing_file_is_zero() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(
            next_row_index(dir.path().join("absent.jsonl")).expect("index"),
            0
        );
    }
}
