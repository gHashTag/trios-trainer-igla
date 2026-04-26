//! Triplet-validated emit to `assertions/seed_results.jsonl`.
//!
//! Enforces the standing rule:
//! `BPB=<v> @ step=<N> seed=<S> sha=<7c> jsonl_row=<L> gate_status=<g>`.
//!
//! Refuses to write if:
//! - SHA is in `.embargo`
//! - step < 4000 (R8 — Gate-2 candidate floor)
//! - target_bpb < 0 (config corruption)

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

fn next_row_index<P: AsRef<Path>>(path: P) -> Result<usize> {
    let p = path.as_ref();
    if !p.exists() {
        return Ok(0);
    }
    let f = std::fs::File::open(p)?;
    Ok(BufReader::new(f).lines().count())
}

fn append_row<P: AsRef<Path>>(path: P, row: &LedgerRow) -> Result<()> {
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    let json = serde_json::to_string(row)?;
    writeln!(f, "{json}")?;
    Ok(())
}

fn push_row<P: AsRef<Path>>(_path: P, row: &LedgerRow) -> Result<()> {
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
    let st = Command::new("git")
        .args(["add", "assertions/seed_results.jsonl"])
        .status()?;
    if !st.success() {
        bail!("git add failed");
    }
    let st = Command::new("git").args(["commit", "-m", &msg]).status()?;
    if !st.success() {
        bail!("git commit failed");
    }
    let st = Command::new("git")
        .args(["push", "origin", "HEAD"])
        .status()?;
    if !st.success() {
        bail!("git push failed");
    }
    Ok(())
}
