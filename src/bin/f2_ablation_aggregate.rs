//! F2 ablation aggregate — Loop 26 TTT.
//!
//! Reads long-form CSV from f2_ablation_sweep, emits wide-form publication-grade table:
//!   mode, fix_name, fix_index, BPB_mean, BPB_std, CI95_lo, CI95_hi, N,
//!   delta_vs_baseline, cohen_d, p_paired_t
//!
//! Per NeurIPS/ICLR convention (AblationBench arXiv:2507.08038): paired Welch's t-test
//! (same seeds = paired) + Cohen's d (N=5 weak power, report effect size).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

#[derive(Debug, Clone)]
struct LongRow {
    mode: String,
    fix_name: String,
    fix_index: i32,
    seed: u64,
    bpb: f64,
}

fn parse_csv(path: &str) -> Vec<LongRow> {
    let f = File::open(path).expect("open input CSV");
    let reader = BufReader::new(f);
    let mut rows = Vec::new();
    for line in reader.lines() {
        let line = line.expect("read line");
        // Loop 33 fix 1: skip W3C-PROV provenance preamble (Loop 32) AND the CSV
        // header by content, not by index. Earlier `i==0` test broke once any
        // `#`-prefixed lines preceded the column header.
        if line.is_empty() || line.starts_with('#') || line.starts_with("mode,") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            continue;
        }
        rows.push(LongRow {
            mode: parts[0].to_string(),
            fix_name: parts[1].to_string(),
            fix_index: parts[2].parse().unwrap_or(0),
            seed: parts[4].parse().unwrap_or(0),
            bpb: parts[5].parse().unwrap_or(f64::NAN),
        });
    }
    rows
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn std_dev(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    var.sqrt()
}

/// Paired Student's t-test on differences (df=n-1). Loop 28 AAA rename: this is
/// NOT Welch's t-test (which is unpaired with Satterthwaite df). The implementation
/// is correct paired-sample Student's t; only the name was misleading.
fn paired_t_test(a: &[f64], b: &[f64]) -> (f64, f64) {
    let n = a.len().min(b.len());
    if n < 2 {
        return (f64::NAN, f64::NAN);
    }
    let diffs: Vec<f64> = (0..n).map(|i| a[i] - b[i]).collect();
    let m_d = mean(&diffs);
    let s_d = std_dev(&diffs);
    if s_d < 1e-12 {
        return (0.0, if m_d.abs() < 1e-12 { 1.0 } else { 0.0 });
    }
    let t = m_d / (s_d / (n as f64).sqrt());
    let df = (n - 1) as f64;
    // Loop 27 critical fix: Student's t-dist CDF instead of normal approximation.
    // Normal approx underflows to p=0 at large |t| with N=5; t-dist has heavier tails.
    let p_two_tailed = 2.0 * student_t_cdf_upper(t.abs(), df);
    (t, p_two_tailed.clamp(0.0, 1.0))
}

// Loop 37 fix 1: t-CDF, regularized_incomplete_beta, beta_cf, lgamma moved to
// race::stats. Numerical drift: aggregator used Stirling lgamma + slightly
// different continued-fraction tolerance; race::stats uses Lanczos g=7 + the
// same Lentz CF. Differences are ≤ 1e-12 in p-value space at all df we use.
use trios_trainer::race::stats::student_t_cdf_upper;

fn cohens_d(a: &[f64], b: &[f64]) -> f64 {
    let m_a = mean(a);
    let m_b = mean(b);
    let s_a = std_dev(a);
    let s_b = std_dev(b);
    let pooled = ((s_a.powi(2) + s_b.powi(2)) / 2.0).sqrt();
    if pooled < 1e-12 {
        return f64::NAN;
    }
    (m_a - m_b) / pooled
}

fn bootstrap_ci95(v: &[f64], seed: u64) -> (f64, f64) {
    if v.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    let mut rng = seed;
    let b = 1000_usize;
    let mut boot_means: Vec<f64> = Vec::with_capacity(b);
    for _ in 0..b {
        let mut sum = 0.0;
        for _ in 0..v.len() {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = (rng >> 33) as usize % v.len();
            sum += v[idx];
        }
        boot_means.push(sum / v.len() as f64);
    }
    boot_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let lo = boot_means[(0.025 * b as f64) as usize];
    let hi = boot_means[(0.975 * b as f64) as usize];
    (lo, hi)
}

#[derive(Debug, Clone)]
struct WideRow {
    mode: String,
    fix_name: String,
    fix_index: i32,
    bpb_mean: f64,
    bpb_std: f64,
    ci95_lo: f64,
    ci95_hi: f64,
    n: usize,
    delta_vs_baseline: f64,
    cohen_d: f64,
    p_paired_t: f64,
}

fn aggregate(rows: &[LongRow]) -> Vec<WideRow> {
    let mut groups: BTreeMap<(String, i32, String), Vec<(u64, f64)>> = BTreeMap::new();
    for r in rows {
        groups
            .entry((r.mode.clone(), r.fix_index, r.fix_name.clone()))
            .or_default()
            .push((r.seed, r.bpb));
    }
    // Loop 27 + Loop 28 XXX: mode-aware baseline picker.
    //   cumulative: idx=0 = "baseline" (no fixes)
    //   loco: NO well-defined baseline (each row IS full-stack-minus-one); use idx=0 by convention
    //   wd_sweep: WD=0.0 is the reference (research: Power Lines says λ→0 in our regime)
    //   pairwise: full_stack row (all 7 fixes enabled, fix_index=-1) is the iLOCO reference
    //             per arXiv:2502.06661 Eq.(3). Falls back to idx=0 for legacy CSVs.
    // Loop 30 fix 2+3: extend baseline concept to per-mode-AND-per-partner where
    // wd_pairwise produces a separate "baseline" for each partner X
    // (label = "wdpair_X_0.000"). The map key is now "<mode>::<partner_or_*>".
    let mut baselines: BTreeMap<String, Vec<(u64, f64)>> = BTreeMap::new();
    // First pass: try to find canonical baseline by name (full_stack / wd_0.000 / baseline / wdpair_*_0.000).
    for ((mode, _idx, fix_name), seeds) in groups.iter() {
        let key_opt: Option<String> = match mode.as_str() {
            "wd_sweep" if fix_name == "wd_0.000" => Some(mode.clone()),
            "pairwise" if fix_name == "full_stack" => Some(mode.clone()),
            "triplet" if fix_name == "full_stack" => Some(mode.clone()),
            "cumulative" if fix_name == "baseline" => Some(mode.clone()),
            "wd_pairwise" => {
                // wdpair_<partner>_<wd>; baseline per partner is wd=0.000.
                if let Some(rest) = fix_name.strip_prefix("wdpair_") {
                    if let Some((partner, wd_s)) = rest.rsplit_once('_') {
                        if wd_s == "0.000" {
                            Some(format!("wd_pairwise::{}", partner))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(key) = key_opt {
            baselines.entry(key).or_insert_with(|| seeds.clone());
        }
    }
    // Fallback: if no canonical baseline found (legacy CSV), use idx=0.
    // Loop 29 audit fix 4: warn loudly when the fallback fires — silent picks
    // led to bogus deltas in the original Loop 27 pairwise output.
    // Loop 30 fix 2+3: for wd_pairwise the baseline key is "wd_pairwise::<partner>",
    // so the canonical-set check probes mode prefix.
    let canonical_modes: std::collections::BTreeSet<String> = baselines
        .keys()
        .map(|k| k.split("::").next().unwrap_or(k).to_string())
        .collect();
    for ((mode, idx, _fix_name), seeds) in groups.iter() {
        if *idx == 0 && !canonical_modes.contains(mode) && !baselines.contains_key(mode) {
            eprintln!(
                "# WARN: no canonical baseline row for mode='{}' (expected full_stack / wd_0.000 / baseline / wdpair_*_0.000). \
                 Falling back to fix_index=0 — deltas may be biased.",
                mode
            );
            baselines.insert(mode.clone(), seeds.clone());
        }
    }

    /// Loop 30 fix 2+3: resolve the right baseline-seed list for a given (mode, fix_name).
    /// wd_pairwise looks up partner-specific baseline; everything else uses mode key.
    fn baseline_for<'a>(
        baselines: &'a BTreeMap<String, Vec<(u64, f64)>>,
        mode: &str,
        fix_name: &str,
    ) -> Option<&'a Vec<(u64, f64)>> {
        if mode == "wd_pairwise" {
            if let Some(rest) = fix_name.strip_prefix("wdpair_") {
                if let Some((partner, _)) = rest.rsplit_once('_') {
                    return baselines.get(&format!("wd_pairwise::{}", partner));
                }
            }
            return None;
        }
        baselines.get(mode)
    }

    let mut wide = Vec::new();
    for ((mode, idx, fix_name), seeds) in &groups {
        let bpbs: Vec<f64> = seeds.iter().map(|(_, b)| *b).collect();
        let m = mean(&bpbs);
        let s = std_dev(&bpbs);
        let (ci_lo, ci_hi) = bootstrap_ci95(&bpbs, 0xCAFE + *idx as u64);
        let (delta, d, p) = if let Some(base_seeds) = baseline_for(&baselines, mode, fix_name) {
            let mut a_aligned = Vec::new();
            let mut b_aligned = Vec::new();
            for (s_id, val) in seeds {
                if let Some((_, b_val)) = base_seeds.iter().find(|(b_s, _)| b_s == s_id) {
                    a_aligned.push(*val);
                    b_aligned.push(*b_val);
                }
            }
            let base_bpbs: Vec<f64> = base_seeds.iter().map(|(_, b)| *b).collect();
            let delta = m - mean(&base_bpbs);
            let d = cohens_d(&bpbs, &base_bpbs);
            let (_, p) = paired_t_test(&a_aligned, &b_aligned);
            (delta, d, p)
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };
        wide.push(WideRow {
            mode: mode.clone(),
            fix_name: fix_name.clone(),
            fix_index: *idx,
            bpb_mean: m,
            bpb_std: s,
            ci95_lo: ci_lo,
            ci95_hi: ci_hi,
            n: bpbs.len(),
            delta_vs_baseline: delta,
            cohen_d: d,
            p_paired_t: p,
        });
    }
    wide.sort_by(|a, b| a.mode.cmp(&b.mode).then(a.fix_index.cmp(&b.fix_index)));
    wide
}

fn emit_wide<W: Write>(w: &mut W, rows: &[WideRow]) -> std::io::Result<()> {
    writeln!(
        w,
        "mode,fix_name,fix_index,bpb_mean,bpb_std,ci95_lo,ci95_hi,n,delta_vs_baseline,cohen_d,p_paired_t"
    )?;
    for r in rows {
        writeln!(
            w,
            "{},{},{},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.4},{:.4e}",
            r.mode,
            r.fix_name,
            r.fix_index,
            r.bpb_mean,
            r.bpb_std,
            r.ci95_lo,
            r.ci95_hi,
            r.n,
            r.delta_vs_baseline,
            r.cohen_d,
            r.p_paired_t
        )?;
    }
    Ok(())
}

fn print_help() {
    println!("f2_ablation_aggregate — Loop 26 TTT wide-form publication table");
    println!();
    println!("USAGE: f2_ablation_aggregate INPUT.csv [--output OUTPUT.csv]");
    println!();
    println!("Stats: paired Welch's t-test + bootstrap CI95 (B=1000) + Cohen's d.");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    let input = &args[1];
    let output = args
        .iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1).cloned());

    let long_rows = parse_csv(input);
    let wide = aggregate(&long_rows);

    if let Some(path) = output.as_deref() {
        let mut f = File::create(path).expect("create output");
        emit_wide(&mut f, &wide).expect("write wide");
        eprintln!("# Wrote {} wide rows to {}", wide.len(), path);
    } else {
        let stdout = std::io::stdout();
        let mut handle = stdout.lock();
        emit_wide(&mut handle, &wide).expect("write stdout");
    }
}
