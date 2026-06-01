//! Side-by-side PSE comparator across strata — Loop 48 Option C.
//!
//! Reads one or more `f2_dual_mediation` output CSVs (each tagged with its
//! stratum: canonical, wd0, warmup0) and produces a long-form join keyed by
//! `(fix_x, pse_name)` × `stratum`. Direct visualization of how the
//! mediation decomposition shifts under iterated Pearl CDE — the empirical
//! payoff of the Loop 31 (wd_stratified) + Loop 41 (warmup_stratified) work.
//!
//! Each input is tagged by the user via `--canonical PATH` / `--wd0 PATH` /
//! `--warmup0 PATH` flags. At least one is required; missing strata emit
//! NaN in their column slots.
//!
//! Output schema:
//!   fix_x, pse_name,
//!   estimate_canonical, ci95_lo_canonical, ci95_hi_canonical,
//!   estimate_wd0,       ci95_lo_wd0,       ci95_hi_wd0,
//!   estimate_warmup0,   ci95_lo_warmup0,   ci95_hi_warmup0,
//!   stable_across_strata (bool — all present CIs overlap)
//!
//! Per Loop 47 audit: by carrying the stratum in the column suffix, we avoid
//! the silent-drop problem (canonical present but wd0 missing → NaN, not
//! omitted row).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

/// One PSE row from a `f2_dual_mediation` CSV.
#[derive(Debug, Clone)]
struct PseRow {
    fix_x: String,
    pse_name: &'static str,
    estimate: f64,
    ci95_lo: f64,
    ci95_hi: f64,
}

/// Read `f2_dual_mediation` CSV and produce one row per (fix_x, pse_name).
/// Skips `#`-preamble (W3C-PROV + Loop 47 stratum banner).
fn parse_dual_mediation_csv(path: &std::path::Path) -> std::io::Result<Vec<PseRow>> {
    let f = File::open(path)?;
    let r = BufReader::new(f);
    let mut header: Option<BTreeMap<String, usize>> = None;
    let mut out = Vec::new();
    for line in r.lines() {
        let line = line?;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if header.is_none() {
            let mut map = BTreeMap::new();
            for (i, name) in parts.iter().enumerate() {
                map.insert(name.to_string(), i);
            }
            header = Some(map);
            continue;
        }
        let h = header.as_ref().unwrap();
        let fx_idx = match h.get("fix_x") {
            Some(i) => *i,
            None => continue,
        };
        let fx = parts[fx_idx].to_string();
        // Each row in dual_mediation has 4 PSEs spread across 4 column triples.
        // Extract all 4 from this row.
        let specs: &[(&'static str, &str, &str, &str)] = &[
            ("NDE", "nde", "ci95_nde_lo", "ci95_nde_hi"),
            ("NIE_M1", "nie_m1", "ci95_nie_m1_lo", "ci95_nie_m1_hi"),
            ("NIE_M2", "nie_m2", "ci95_nie_m2_lo", "ci95_nie_m2_hi"),
            ("NIE_chain", "nie_chain", "ci95_nie_chain_lo", "ci95_nie_chain_hi"),
        ];
        for (pse_name, est_col, lo_col, hi_col) in specs {
            let (Some(ec), Some(lc), Some(hc)) = (h.get(*est_col), h.get(*lo_col), h.get(*hi_col)) else {
                continue;
            };
            let est: f64 = parts[*ec].parse().unwrap_or(f64::NAN);
            let lo: f64 = parts[*lc].parse().unwrap_or(f64::NAN);
            let hi: f64 = parts[*hc].parse().unwrap_or(f64::NAN);
            if est.is_finite() && lo.is_finite() && hi.is_finite() {
                out.push(PseRow {
                    fix_x: fx.clone(),
                    pse_name,
                    estimate: est,
                    ci95_lo: lo,
                    ci95_hi: hi,
                });
            }
        }
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stratum {
    Canonical,
    Wd0,
    Warmup0,
}

impl Stratum {
    fn suffix(&self) -> &'static str {
        match self {
            Stratum::Canonical => "canonical",
            Stratum::Wd0 => "wd0",
            Stratum::Warmup0 => "warmup0",
        }
    }
}

/// CIs overlap iff each lower bound is ≤ the other's upper bound.
fn cis_overlap(a: &PseRow, b: &PseRow) -> bool {
    a.ci95_lo <= b.ci95_hi && b.ci95_lo <= a.ci95_hi
}

#[derive(Debug, Clone)]
struct ComparedRow {
    fix_x: String,
    pse_name: &'static str,
    /// Estimates and CIs indexed by stratum index (0=canonical, 1=wd0, 2=warmup0).
    /// NaN slot means "no input for that stratum".
    estimates: [f64; 3],
    ci_los: [f64; 3],
    ci_his: [f64; 3],
    /// True iff every pair of present (non-NaN) CIs overlaps.
    stable: bool,
}

fn build_comparison(per_stratum: &[(Stratum, Vec<PseRow>)]) -> Vec<ComparedRow> {
    // Collect every (fix_x, pse_name) seen anywhere.
    let mut keys: BTreeMap<(String, &'static str), [Option<PseRow>; 3]> = BTreeMap::new();
    for (s, rows) in per_stratum {
        let slot = match s {
            Stratum::Canonical => 0,
            Stratum::Wd0 => 1,
            Stratum::Warmup0 => 2,
        };
        for r in rows {
            let key = (r.fix_x.clone(), r.pse_name);
            let entry = keys.entry(key).or_insert([None, None, None]);
            entry[slot] = Some(r.clone());
        }
    }
    let mut out = Vec::new();
    for ((fx, pse), slots) in keys {
        let mut estimates = [f64::NAN; 3];
        let mut ci_los = [f64::NAN; 3];
        let mut ci_his = [f64::NAN; 3];
        let mut present: Vec<&PseRow> = Vec::new();
        for (i, opt) in slots.iter().enumerate() {
            if let Some(r) = opt {
                estimates[i] = r.estimate;
                ci_los[i] = r.ci95_lo;
                ci_his[i] = r.ci95_hi;
                present.push(r);
            }
        }
        let stable = if present.len() < 2 {
            // Single stratum or none → not meaningful to call "stable".
            false
        } else {
            (0..present.len())
                .flat_map(|i| ((i + 1)..present.len()).map(move |j| (i, j)))
                .all(|(i, j)| cis_overlap(present[i], present[j]))
        };
        out.push(ComparedRow {
            fix_x: fx,
            pse_name: pse,
            estimates,
            ci_los,
            ci_his,
            stable,
        });
    }
    // Sort by |canonical estimate| desc (most interesting first); ties broken by fix_x.
    out.sort_by(|a, b| {
        let abs_a = if a.estimates[0].is_finite() { a.estimates[0].abs() } else { 0.0 };
        let abs_b = if b.estimates[0].is_finite() { b.estimates[0].abs() } else { 0.0 };
        abs_b
            .partial_cmp(&abs_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.fix_x.cmp(&b.fix_x))
            .then_with(|| a.pse_name.cmp(b.pse_name))
    });
    out
}

fn emit<W: Write>(w: &mut W, rows: &[ComparedRow]) -> std::io::Result<()> {
    writeln!(
        w,
        "# Cross-stratum PSE comparison (Loop 48 Option C)"
    )?;
    writeln!(
        w,
        "# `stable_across_strata` = true iff every pair of present 95% CIs overlap;"
    )?;
    writeln!(
        w,
        "# false flags PSEs whose magnitude shifts MEANINGFULLY between strata."
    )?;
    writeln!(
        w,
        "fix_x,pse_name,estimate_canonical,ci95_lo_canonical,ci95_hi_canonical,estimate_wd0,ci95_lo_wd0,ci95_hi_wd0,estimate_warmup0,ci95_lo_warmup0,ci95_hi_warmup0,stable_across_strata"
    )?;
    for r in rows {
        writeln!(
            w,
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            r.fix_x, r.pse_name,
            r.estimates[0], r.ci_los[0], r.ci_his[0],
            r.estimates[1], r.ci_los[1], r.ci_his[1],
            r.estimates[2], r.ci_los[2], r.ci_his[2],
            r.stable
        )?;
    }
    Ok(())
}

fn print_help() {
    println!("f2_stratum_compare — Loop 48: cross-stratum PSE comparison");
    println!();
    println!("USAGE: f2_stratum_compare [FLAGS]");
    println!();
    println!("FLAGS:");
    println!("  --help, -h          Print this help and exit");
    println!("  --canonical PATH    Dual-mediation CSV from canonical-stratum input");
    println!("  --wd0 PATH          Dual-mediation CSV from --mode wd_stratified input");
    println!("  --warmup0 PATH      Dual-mediation CSV from --mode warmup_stratified input");
    println!("  --out PATH          Write comparison CSV to file (default stdout)");
    println!();
    println!("At least one --canonical/--wd0/--warmup0 flag is required.");
    println!("Missing strata produce NaN in their columns; `stable_across_strata`");
    println!("becomes false when present CIs disagree.");
    println!();
    println!("Per docs/F2_BINARIES.md: canonical = marginal NDE/NIE; wd0/warmup0 =");
    println!("Pearl Controlled Direct/Indirect Effects at the disabled value.");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    let mut canonical: Option<PathBuf> = None;
    let mut wd0: Option<PathBuf> = None;
    let mut warmup0: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        let next = || {
            args.get(i + 1).cloned().unwrap_or_else(|| {
                eprintln!("# ERROR: {} requires a path", a);
                std::process::exit(2);
            })
        };
        match a.as_str() {
            "--canonical" => {
                canonical = Some(PathBuf::from(next()));
                i += 2;
            }
            "--wd0" => {
                wd0 = Some(PathBuf::from(next()));
                i += 2;
            }
            "--warmup0" => {
                warmup0 = Some(PathBuf::from(next()));
                i += 2;
            }
            "--out" => {
                out_path = Some(PathBuf::from(next()));
                i += 2;
            }
            s if s.starts_with("--") => {
                eprintln!("# ERROR: unknown flag {}", s);
                std::process::exit(2);
            }
            _ => {
                eprintln!("# ERROR: positional args not supported; use --canonical/--wd0/--warmup0.");
                std::process::exit(2);
            }
        }
    }
    let mut per_stratum: Vec<(Stratum, Vec<PseRow>)> = Vec::new();
    for (s, opt) in [
        (Stratum::Canonical, canonical),
        (Stratum::Wd0, wd0),
        (Stratum::Warmup0, warmup0),
    ] {
        if let Some(p) = opt {
            eprintln!("# Loading {} from {}", s.suffix(), p.display());
            let rows = parse_dual_mediation_csv(&p).expect("read CSV");
            eprintln!("# {} PSE rows from {}", rows.len(), s.suffix());
            per_stratum.push((s, rows));
        }
    }
    if per_stratum.is_empty() {
        eprintln!("# ERROR: no inputs given; need at least one of --canonical/--wd0/--warmup0");
        std::process::exit(2);
    }
    let compared = build_comparison(&per_stratum);
    let n_stable = compared.iter().filter(|r| r.stable).count();
    let n_total = compared.iter().filter(|r| {
        // Only count rows where ≥2 strata had data.
        r.estimates.iter().filter(|e| e.is_finite()).count() >= 2
    }).count();
    eprintln!(
        "# {} of {} multi-stratum PSEs are STABLE (CIs overlap across strata)",
        n_stable, n_total
    );
    if let Some(p) = out_path {
        let mut f = File::create(&p).expect("create out CSV");
        emit(&mut f, &compared).expect("write");
        eprintln!("# Wrote {} rows to {}", compared.len(), p.display());
    } else {
        let stdout = std::io::stdout();
        let mut h = stdout.lock();
        emit(&mut h, &compared).expect("write stdout");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_dual_csv(path: &std::path::Path, fix_x: &str, nde: f64, nde_lo: f64, nde_hi: f64) {
        let mut f = File::create(path).unwrap();
        writeln!(f, "# preamble").unwrap();
        writeln!(
            f,
            "rank,fix_x,n,delta_x,nde,se_nde,ci95_nde_lo,ci95_nde_hi,nie_m1,se_nie_m1,ci95_nie_m1_lo,ci95_nie_m1_hi,nie_m2,se_nie_m2,ci95_nie_m2_lo,ci95_nie_m2_hi,nie_chain,se_nie_chain,ci95_nie_chain_lo,ci95_nie_chain_hi"
        ).unwrap();
        writeln!(
            f,
            "1,{},5,1.0,{:.6},0.1,{:.6},{:.6},2.0,0.1,1.5,2.5,0.5,0.1,0.0,1.0,-0.5,0.1,-1.0,0.0",
            fix_x, nde, nde_lo, nde_hi
        ).unwrap();
    }

    #[test]
    fn cis_overlap_basic_cases() {
        let a = PseRow { fix_x: "x".into(), pse_name: "NDE", estimate: 0.0, ci95_lo: -1.0, ci95_hi: 1.0 };
        let b = PseRow { fix_x: "x".into(), pse_name: "NDE", estimate: 0.5, ci95_lo: 0.0, ci95_hi: 2.0 };
        let c = PseRow { fix_x: "x".into(), pse_name: "NDE", estimate: 5.0, ci95_lo: 4.0, ci95_hi: 6.0 };
        assert!(cis_overlap(&a, &b));
        assert!(!cis_overlap(&a, &c));
    }

    #[test]
    fn parse_dual_mediation_csv_extracts_four_pses_per_row() {
        let tmp = std::env::temp_dir().join("f2_stratum_compare_parse.csv");
        write_dual_csv(&tmp, "rms", -4.0, -4.5, -3.5);
        let rows = parse_dual_mediation_csv(&tmp).unwrap();
        // 1 input row × 4 PSEs = 4 output rows.
        assert_eq!(rows.len(), 4);
        let nde = rows.iter().find(|r| r.pse_name == "NDE").unwrap();
        assert!((nde.estimate - (-4.0)).abs() < 1e-9);
        assert!((nde.ci95_lo - (-4.5)).abs() < 1e-9);
        assert!((nde.ci95_hi - (-3.5)).abs() < 1e-9);
    }

    #[test]
    fn build_comparison_marks_stable_when_cis_overlap() {
        // Canonical and warmup0 both give NDE = -4 ± 0.5 → CIs overlap → stable.
        let canon = vec![PseRow {
            fix_x: "rms".into(),
            pse_name: "NDE",
            estimate: -4.0,
            ci95_lo: -4.5,
            ci95_hi: -3.5,
        }];
        let warmup = vec![PseRow {
            fix_x: "rms".into(),
            pse_name: "NDE",
            estimate: -4.1,
            ci95_lo: -4.6,
            ci95_hi: -3.6,
        }];
        let cmp = build_comparison(&[(Stratum::Canonical, canon), (Stratum::Warmup0, warmup)]);
        assert_eq!(cmp.len(), 1);
        assert!(cmp[0].stable);
        // wd0 slot stays NaN.
        assert!(cmp[0].estimates[1].is_nan());
    }

    #[test]
    fn build_comparison_flags_unstable_when_cis_disjoint() {
        let canon = vec![PseRow {
            fix_x: "rms".into(),
            pse_name: "NDE",
            estimate: -4.0,
            ci95_lo: -4.5,
            ci95_hi: -3.5,
        }];
        let wd0 = vec![PseRow {
            fix_x: "rms".into(),
            pse_name: "NDE",
            estimate: -1.0,
            ci95_lo: -1.5,
            ci95_hi: -0.5,
        }];
        let cmp = build_comparison(&[(Stratum::Canonical, canon), (Stratum::Wd0, wd0)]);
        assert!(!cmp[0].stable, "CIs [-4.5,-3.5] vs [-1.5,-0.5] should not overlap");
    }

    #[test]
    fn build_comparison_handles_missing_strata_as_nan() {
        let only_canon = vec![PseRow {
            fix_x: "rms".into(),
            pse_name: "NDE",
            estimate: -4.0,
            ci95_lo: -4.5,
            ci95_hi: -3.5,
        }];
        let cmp = build_comparison(&[(Stratum::Canonical, only_canon)]);
        assert_eq!(cmp.len(), 1);
        assert!(cmp[0].estimates[0].is_finite());
        assert!(cmp[0].estimates[1].is_nan());
        assert!(cmp[0].estimates[2].is_nan());
        // Single-stratum PSE is not "stable across strata" (no comparison possible).
        assert!(!cmp[0].stable);
    }
}
