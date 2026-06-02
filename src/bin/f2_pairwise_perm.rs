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

#[derive(Debug)]
struct InputRow {
    config: String,
    stratum: String,
    #[allow(dead_code)]
    seed: u64,
    val_bpb: f64,
}

/// Exact paired-permutation test enumerating 2^N sign-flip vectors.
///
/// Returns (mean_diff, p_two_sided).
/// At N=5 we enumerate all 32 ± assignments and count |mean| >= |observed|.
fn exact_paired_perm(diffs: &[f64]) -> (f64, f64) {
    let n = diffs.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let observed_mean: f64 = diffs.iter().sum::<f64>() / n as f64;
    let abs_observed = observed_mean.abs();

    let total = 1u64 << n;
    let mut at_least_as_extreme: u64 = 0;
    for mask in 0..total {
        let mut sum = 0.0;
        for (i, d) in diffs.iter().enumerate() {
            let flip = (mask >> i) & 1 == 1;
            sum += if flip { -d } else { *d };
        }
        let m = sum / n as f64;
        if m.abs() >= abs_observed {
            at_least_as_extreme += 1;
        }
    }
    let p = at_least_as_extreme as f64 / total as f64;
    (observed_mean, p)
}

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
    writeln!(out, "# f2_pairwise_perm — Loop 110 (Issue #1021 protocol).")
        .ok();
    writeln!(out, "# Method: exact Zmigrod-Vieira-Cotterell paired-permutation")
        .ok();
    writeln!(out, "#   over 2^N sign-flip vectors; BH-correction within phi-config.")
        .ok();
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
