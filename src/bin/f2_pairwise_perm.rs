//! F2 pairwise paired-permutation test — Loop 110 (Issue #1021 protocol).
//!
//! Implements the exact Zmigrod-Vieira-Cotterell (2022, arXiv:2205.01416)
//! paired-permutation test for the champion-scale phi-ladder follow-up paper.
//! At N=5 seeds the exact test enumerates 2^5 = 32 sign-flip vectors per pair
//! and emits exact p-values. BH-correction (Benjamini-Hochberg per Liu, Leung
//! & Shao, arXiv:1712.03305) is applied across the 4 zoo comparisons within
//! each phi-config.
//!
//! Input: long-form CSV with columns:
//!     config, stratum, seed, val_bpb
//! Each (config, stratum) cell must have N=5 seeds (the protocol pre-registers
//! seeds [42, 43, 44, 45, 46]).
//!
//! Output: CSV with columns
//!     stratum, phi_config, zoo_config, n, diff_mean, ci_lo, ci_hi,
//!     p_raw, p_bh
//! Rows: 16 pairs per stratum × number of strata.
//!
//! Usage:
//!     f2_pairwise_perm --input <long_form.csv> --output <pairwise.csv> \
//!         --phi-configs GFTernary,GF8,GF16,GF32 \
//!         --zoo-configs BitNet-1.58,INT8,FP8,bf16
//!
//! Pre-registration: this binary is committed at Loop 110 as part of the
//! Issue #1021 protocol contribution to the F2 framework.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::process::ExitCode;

use trios_trainer::race::stats::exact_paired_sign_flip_perm as exact_paired_perm;

#[derive(Debug)]
struct InputRow {
    config: String,
    stratum: String,
    #[allow(dead_code)]
    seed: u64,
    val_bpb: f64,
}

// Loop 112 C: the exact_paired_perm primitive lives in
// `src/race/stats.rs::exact_paired_sign_flip_perm`. This binary
// imports it via the `exact_paired_perm` alias above so that any
// future drift is impossible — both binaries (this one and
// f2_iloco_score, post-refactor) consume the same source.

/// Student-t two-sided 95% CI on the seed-mean difference at N=5.
fn ci95_t(diffs: &[f64]) -> (f64, f64) {
    let n = diffs.len();
    if n < 2 {
        return (f64::NAN, f64::NAN);
    }
    let mean: f64 = diffs.iter().sum::<f64>() / n as f64;
    let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>()
        / (n as f64 - 1.0);
    let se = (var / n as f64).sqrt();
    // t_{0.975, 4} ≈ 2.776 — matches the companion F2 paper at N=5.
    let t = 2.776;
    (mean - t * se, mean + t * se)
}

/// Benjamini-Hochberg adjusted p-values for a slice. Returns a new vector of
/// same length with the BH-adjusted p-values in the original input order.
fn bh_adjust(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    let mut indexed: Vec<(usize, f64)> =
        p_values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut adjusted = vec![0.0_f64; m];
    let mut running_min = 1.0_f64;
    // Walk from largest p to smallest (i = m..=1).
    for rank in (1..=m).rev() {
        let (orig_idx, raw_p) = indexed[rank - 1];
        let bh = raw_p * m as f64 / rank as f64;
        if bh < running_min {
            running_min = bh;
        }
        adjusted[orig_idx] = running_min.min(1.0);
    }
    adjusted
}


fn parse_input(path: &str) -> Result<Vec<InputRow>, String> {
    let file = File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    let mut header: Option<Vec<String>> = None;
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if header.is_none() {
            header = Some(line.split(',').map(|s| s.to_string()).collect());
            continue;
        }
        let h = header.as_ref().unwrap();
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != h.len() {
            continue;
        }
        let get = |key: &str| -> Option<&str> {
            h.iter().position(|c| c == key).map(|i| fields[i])
        };
        let config = get("config").ok_or("missing config column")?.to_string();
        let stratum = get("stratum").ok_or("missing stratum column")?.to_string();
        let seed: u64 = get("seed").ok_or("missing seed column")?
            .parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
        let val_bpb: f64 = get("val_bpb").ok_or("missing val_bpb column")?
            .parse().map_err(|e: std::num::ParseFloatError| e.to_string())?;
        rows.push(InputRow { config, stratum, seed, val_bpb });
    }
    Ok(rows)
}


fn main() -> ExitCode {
    // Minimal arg parsing — extends to clap later.
    let args: Vec<String> = std::env::args().collect();
    let get_arg = |flag: &str| -> Option<String> {
        args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
    };
    let input = match get_arg("--input") {
        Some(s) => s,
        None => {
            eprintln!("usage: f2_pairwise_perm --input <csv> --output <csv> \
                --phi-configs <csv-list> --zoo-configs <csv-list>");
            return ExitCode::from(64);
        }
    };
    let output = match get_arg("--output") {
        Some(s) => s,
        None => {
            eprintln!("missing --output");
            return ExitCode::from(64);
        }
    };
    let phi_list: Vec<String> = get_arg("--phi-configs")
        .unwrap_or_else(|| "GFTernary,GF8,GF16,GF32".to_string())
        .split(',').map(|s| s.to_string()).collect();
    let zoo_list: Vec<String> = get_arg("--zoo-configs")
        .unwrap_or_else(|| "BitNet-1.58,INT8,FP8,bf16".to_string())
        .split(',').map(|s| s.to_string()).collect();

    let rows = match parse_input(&input) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("parse {input}: {e}");
            return ExitCode::from(1);
        }
    };

    // Group rows by (config, stratum) → seed-ordered bpb values.
    let mut grouped: BTreeMap<(String, String), Vec<f64>> = BTreeMap::new();
    for r in rows {
        grouped.entry((r.config.clone(), r.stratum.clone()))
            .or_default()
            .push(r.val_bpb);
    }
    let strata: Vec<String> = grouped.keys().map(|k| k.1.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter().collect();

    let mut out = File::create(&output)
        .map_err(|e| eprintln!("create {output}: {e}"))
        .map_err(|_| ExitCode::from(1));
    if out.is_err() {
        return ExitCode::from(1);
    }
    let out = out.as_mut().unwrap();
    // Loop 111: W3C-PROV preamble matching f2_provenance_check's schema
    // (34th adversarial pass discovery — §5.1 schema promises every CSV
    // carries a preamble; this binary previously emitted only narrative
    // comments).
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(-1);
    let git_sha = std::env::var("F2_GIT_SHA")
        .unwrap_or_else(|_| "unknown".to_string());
    let host = std::env::var("HOST")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unknown".to_string());
    writeln!(out, "# f2_pairwise_perm — Loop 110 (Issue #1021 protocol).")
        .ok();
    writeln!(out, "# Method: exact Zmigrod-Vieira-Cotterell paired-permutation")
        .ok();
    writeln!(out, "#   over 2^N sign-flip vectors; BH-correction within phi-config.")
        .ok();
    writeln!(out, "# prov:generatedAt = {} (unix seconds UTC)", now_secs).ok();
    writeln!(out,
        "# prov:wasGeneratedBy = f2_pairwise_perm --input {} --output {}",
        input, output).ok();
    writeln!(out, "# prov:agent_git_sha = {}", git_sha).ok();
    writeln!(out, "# prov:host = {}", host).ok();
    writeln!(out, "# prov:phi_configs = {}", phi_list.join("|")).ok();
    writeln!(out, "# prov:zoo_configs = {}", zoo_list.join("|")).ok();
    writeln!(out, "# prov:trainer_internals_schema = {}",
        trios_trainer::race::multi_seed::TRAINER_INTERNALS_SCHEMA).ok();
    writeln!(out, "# prov:cargo_pkg_version = {}",
        env!("CARGO_PKG_VERSION")).ok();
    writeln!(out, "stratum,phi_config,zoo_config,n,diff_mean,ci_lo,ci_hi,p_raw,p_bh")
        .ok();

    let mut any_missing = false;
    for stratum in &strata {
        // Collect per-pair results, then BH-correct within each phi-config.
        for phi in &phi_list {
            let phi_key = (phi.clone(), stratum.clone());
            let Some(phi_bpb) = grouped.get(&phi_key) else {
                eprintln!("# WARN: ({phi}, {stratum}) absent from input — skipping phi row");
                any_missing = true;
                continue;
            };
            let mut p_raw_vec: Vec<f64> = Vec::with_capacity(zoo_list.len());
            let mut row_payload: Vec<(String, usize, f64, f64, f64)> =
                Vec::with_capacity(zoo_list.len());
            for zoo in &zoo_list {
                let zoo_key = (zoo.clone(), stratum.clone());
                let Some(zoo_bpb) = grouped.get(&zoo_key) else {
                    eprintln!("# WARN: ({zoo}, {stratum}) absent from input — skipping zoo");
                    any_missing = true;
                    continue;
                };
                if phi_bpb.len() != zoo_bpb.len() {
                    eprintln!("# WARN: seed-count mismatch {phi}/{zoo} at {stratum} — skipping");
                    continue;
                }
                let diffs: Vec<f64> = phi_bpb.iter().zip(zoo_bpb.iter())
                    .map(|(p, z)| p - z).collect();
                let (mean, p_raw) = exact_paired_perm(&diffs);
                let (lo, hi) = ci95_t(&diffs);
                row_payload.push((zoo.clone(), diffs.len(), mean, lo, hi));
                p_raw_vec.push(p_raw);
            }
            let p_bh = bh_adjust(&p_raw_vec);
            for (i, (zoo, n, mean, lo, hi)) in row_payload.iter().enumerate() {
                let raw = p_raw_vec[i];
                let bh = p_bh[i];
                writeln!(out,
                    "{stratum},{phi},{zoo},{n},{mean:.6},{lo:.6},{hi:.6},{raw:.6},{bh:.6}")
                    .ok();
            }
        }
    }
    if any_missing {
        eprintln!("# Some (config, stratum) cells were missing; output is incomplete.");
        ExitCode::from(2)
    } else {
        ExitCode::SUCCESS
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_perm_known_result() {
        // diffs = [1.0, 2.0, 3.0, 4.0, 5.0] — all positive, mean = 3.0.
        // Only the all-positive assignment matches |mean| >= 3.0 exactly.
        // Other assignments produce smaller |mean|. Two extreme: all + and all -.
        // p_two_sided = 2/32 = 0.0625.
        let diffs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, p) = exact_paired_perm(&diffs);
        assert!((mean - 3.0).abs() < 1e-12);
        assert!((p - 2.0 / 32.0).abs() < 1e-12,
            "p = {p} (expected 0.0625)");
    }

    #[test]
    fn exact_perm_null_result() {
        // diffs = [+1, -1, +1, -1, 0] — mean = 0. Every permutation gives
        // mean with |mean| >= 0, so p = 1.0.
        let diffs = vec![1.0, -1.0, 1.0, -1.0, 0.0];
        let (mean, p) = exact_paired_perm(&diffs);
        assert!(mean.abs() < 1e-12);
        assert!((p - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bh_adjusts_monotonically() {
        // Sorted ascending: smallest stays smallest, largest stays largest
        // (after BH correction; the running-min step keeps monotonicity).
        let p = vec![0.01, 0.04, 0.06, 0.20];
        let adj = bh_adjust(&p);
        // BH at rank 1: 0.01 × 4/1 = 0.04
        // BH at rank 2: 0.04 × 4/2 = 0.08
        // BH at rank 3: 0.06 × 4/3 = 0.08
        // BH at rank 4: 0.20 × 4/4 = 0.20
        // Running-min from rank 4 down: 0.20, 0.08, 0.08, 0.04
        // So adjusted in original order: [0.04, 0.08, 0.08, 0.20]
        assert!((adj[0] - 0.04).abs() < 1e-12);
        assert!((adj[1] - 0.08).abs() < 1e-12);
        assert!((adj[2] - 0.08).abs() < 1e-12);
        assert!((adj[3] - 0.20).abs() < 1e-12);
    }

    /// Reference implementation copied verbatim from
    /// `src/bin/f2_iloco_score.rs:67-90`. Loop 111 33rd/34th-pass
    /// audit established that f2_pairwise_perm's primitive should
    /// agree with this reference for inputs that don't hit edge
    /// cases (N>=2, no float-tie-to-machine-epsilon). This module-local
    /// copy is kept in sync via the equivalence test below; if the
    /// upstream primitive changes, this test will diverge and surface
    /// the drift.
    fn iloco_permutation_test_paired_reference(a: &[f64], b: &[f64]) -> (f64, f64) {
        let n = a.len().min(b.len());
        if n < 2 {
            return (f64::NAN, f64::NAN);
        }
        let diffs: Vec<f64> = (0..n).map(|i| a[i] - b[i]).collect();
        let observed: f64 = diffs.iter().sum();
        let total: u64 = 1u64 << n;
        let mut ge_count: u64 = 0;
        for mask in 0..total {
            let mut s = 0.0_f64;
            for (i, d) in diffs.iter().enumerate() {
                let sign = if (mask >> i) & 1 == 1 { -1.0 } else { 1.0 };
                s += sign * d;
            }
            if s.abs() >= observed.abs() - 1e-15 {
                ge_count += 1;
            }
        }
        let p = ge_count as f64 / total as f64;
        (observed / n as f64, p.clamp(0.0, 1.0))
    }

    #[test]
    fn primitive_matches_f2_iloco_score_reference() {
        // Loop 111 C: verify f2_pairwise_perm's exact_paired_perm primitive
        // agrees with f2_iloco_score's permutation_test_paired reference
        // on representative inputs.
        let cases: &[(&[f64], &[f64])] = &[
            (&[1.20, 1.18, 1.22, 1.19, 1.21], &[1.40, 1.38, 1.42, 1.39, 1.41]),
            // Symmetric input — should give p = 1.0 in both.
            (&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 2.0, 3.0, 4.0, 5.0]),
            // All-positive diffs — p = 2/32 = 0.0625 in both.
            (&[5.0, 4.0, 3.0, 2.0, 1.0], &[0.0, 0.0, 0.0, 0.0, 0.0]),
            // Mixed-sign diffs.
            (&[1.1, 0.9, 1.2, 1.0, 1.05], &[1.0, 1.0, 1.0, 1.0, 1.0]),
            // Different N (>=2): N=3, N=4.
            (&[1.5, 2.5, 3.5], &[1.0, 2.0, 3.0]),
            (&[10.0, 20.0, 30.0, 40.0], &[12.0, 18.0, 32.0, 38.0]),
        ];
        for (i, (a, b)) in cases.iter().enumerate() {
            let diffs: Vec<f64> = (0..a.len()).map(|k| a[k] - b[k]).collect();
            let (m_ours, p_ours) = exact_paired_perm(&diffs);
            let (m_ref, p_ref) = iloco_permutation_test_paired_reference(a, b);
            assert!((m_ours - m_ref).abs() < 1e-12,
                "case {i}: mean diff {m_ours} vs ref {m_ref}");
            assert!((p_ours - p_ref).abs() < 1e-12,
                "case {i}: p {p_ours} vs ref {p_ref} (diffs={diffs:?})");
        }
    }

    #[test]
    fn ci95_known_input() {
        // diffs centered at 0.0 with known variance.
        let diffs = vec![-1.0, 0.0, 1.0, 0.0, 0.0];
        let (lo, hi) = ci95_t(&diffs);
        // mean = 0.0; var = 0.5; se = sqrt(0.1) ≈ 0.3162; t*se ≈ 0.878
        assert!(lo < 0.0);
        assert!(hi > 0.0);
        assert!((lo + hi).abs() < 1e-6, "CI should be symmetric: {lo}, {hi}");
    }
}
