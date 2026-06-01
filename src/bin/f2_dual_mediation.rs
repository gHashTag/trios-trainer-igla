//! F2 dual-mediator path-specific effect decomposer — Loop 34.
//!
//! Implements the natural counterfactual decomposition of a Total Effect into
//! four path-specific effects when two mediators M1, M2 are causally ordered
//! (X → M1 → M2 → Y plus the X→Y and X→M2→Y shortcuts), per Zhao & Luo (2020)
//! arXiv:2007.16031. Closed-form computation from interventional point estimates
//! (our LOCO, pairwise, triplet rows) under linearity + no-XM-interaction.
//!
//! ## Decomposition (additive)
//!
//!   TE(X) = NDE(X)               // X → Y directly, both mediators blocked
//!          + NIE_M1(X)           // X → M1 → Y only (M2 path blocked)
//!          + NIE_M2(X)           // X → M2 → Y only (M1 path blocked)
//!          + NIE_chain(X)        // X → M1 → M2 → Y (the chain)
//!
//! ## Mapping to our long-form CSV
//!
//! Let Δ_S = BPB(remove S) − BPB(full_stack). Our ablations give us Δ for any
//! S ⊆ {X, M1, M2}. Under linearity + sequential ignorability + no interaction:
//!
//!   NDE(X)       = Δ_{X, M1, M2}                       — direct effect when
//!                                                       both M1 and M2 paths blocked
//!   NIE_chain(X) = (Δ_X − Δ_{X,M1}) − (Δ_{X,M2} − Δ_{X,M1,M2})
//!                                                       — what removing M2 does
//!                                                       MORE when M1 is also gone
//!   NIE_M1(X)    = Δ_X − Δ_{X,M1} − NIE_chain(X)
//!   NIE_M2(X)    = Δ_X − Δ_{X,M2} − NIE_chain(X)
//!   sanity: TE   = NDE + NIE_M1 + NIE_M2 + NIE_chain (asymptotically; assumed
//!                  identity under the no-interaction model).
//!
//! Sign convention: a positive Δ_S means "removing S hurts BPB" (S is helpful).
//!
//! ## References
//! - Zhao & Luo 2020 (arXiv:2007.16031) — 4-way decomposition under sequential
//!   ignorability.
//! - arXiv:2505.04983 (May 2025) — small-N finite-sample variance of the
//!   two-mediator decomposition via delta-method (we use per-seed point estimates
//!   directly + percentile bootstrap; delta-method left for Loop 35).
//! - VanderWeele & Vansteelandt 2014 — canonical NDE/NIE definitions.
//!
//! ## Loop 33 finding context
//! Loop 33 established the WD → warmup → BPB chain (warmup mediates 75% of
//! RmsNorm's residual effect at WD=0). This binary quantifies how that 75%
//! actually splits among the four paths.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use trios_trainer::race::ablation::{
    all_mode_strings, is_canonical_fix, ModeKind, CANONICAL_FIX_NAMES,
};
use trios_trainer::race::stats::student_t_critical_two_sided;

#[derive(Debug, Clone)]
struct LongRow {
    mode: String,
    fix_name: String,
    seed: u64,
    bpb: f64,
}

fn parse_csv(path: &str) -> Vec<LongRow> {
    let f = File::open(path).expect("open input CSV");
    let r = BufReader::new(f);
    let mut rows = Vec::new();
    for line in r.lines() {
        let line = line.expect("read");
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
            seed: parts[4].parse().unwrap_or(0),
            bpb: parts[5].parse().unwrap_or(f64::NAN),
        });
    }
    rows
}

// Loop 38 fix 1+2 + Loop 39 fix 1: mean + sample_se migrated to race::stats.
use trios_trainer::race::stats::{mean, sample_se};

/// Canonical (sorted) pair label "pair_<a>_<b>" using min(a,b), max(a,b).
fn pair_label(a: &str, b: &str) -> String {
    if a <= b {
        format!("pair_{}_{}", a, b)
    } else {
        format!("pair_{}_{}", b, a)
    }
}

/// Triplet label "triplet_<a>_<b>_<c>" using lexicographic sort.
fn triplet_label(a: &str, b: &str, c: &str) -> String {
    let mut v = [a, b, c];
    v.sort();
    format!("triplet_{}_{}_{}", v[0], v[1], v[2])
}

/// Loop 34: enumerate all 6 permutations of (a, b, c) as triplet labels.
/// Loop 29's run_triplet emits labels in AblationFix::ALL index order, which is
/// NOT lexicographic. Probe all 6 permutations and return the first matching one
/// found in `by_key`. Returns the matching seed series if any.
fn lookup_triplet_any_perm<'a>(
    by_key: &'a BTreeMap<(String, String), Vec<(u64, f64)>>,
    a: &str,
    b: &str,
    c: &str,
) -> Option<&'a Vec<(u64, f64)>> {
    let perms = [
        (a, b, c), (a, c, b), (b, a, c),
        (b, c, a), (c, a, b), (c, b, a),
    ];
    // Loop 38 fix 4 → Loop 39 fix 2: stratum-aware lookup uses the registry
    // in race::ablation. Adding a new stratum (e.g. warmup0) requires no
    // changes here — the iteration order is centralized.
    for mode in all_mode_strings(ModeKind::Triplet) {
        for (x, y, z) in &perms {
            let key = (mode.clone(), format!("triplet_{}_{}_{}", x, y, z));
            if let Some(v) = by_key.get(&key) {
                return Some(v);
            }
        }
    }
    None
}

/// Loop 34: permutation-tolerant pair lookup. Loop 27 emits pair labels in
/// AblationFix::ALL order, not lexicographic; matches order is e.g.
/// "pair_warmup_wd" not "pair_wd_warmup".
fn lookup_pair_any_perm<'a>(
    by_key: &'a BTreeMap<(String, String), Vec<(u64, f64)>>,
    a: &str,
    b: &str,
) -> Option<&'a Vec<(u64, f64)>> {
    // Loop 38 fix 4 → Loop 39 fix 2: use the stratum registry.
    for mode in all_mode_strings(ModeKind::Pairwise) {
        for (x, y) in [(a, b), (b, a)] {
            let key = (mode.clone(), format!("pair_{}_{}", x, y));
            if let Some(v) = by_key.get(&key) {
                return Some(v);
            }
        }
    }
    None
}

/// Collect per-seed BPB samples aligned by seed id, then return per-seed Δ vs full.
/// Loop 35 fix 1: also returns the number of `full_seeds` so callers can detect
/// alignment loss (target series missing some seeds) and emit a diagnostic.
fn aligned_delta_with_count(
    target_seeds: &[(u64, f64)],
    full_seeds: &[(u64, f64)],
) -> (Vec<f64>, usize) {
    let mut out = Vec::new();
    for (sid, full_bpb) in full_seeds {
        if let Some((_, t_bpb)) = target_seeds.iter().find(|(s, _)| s == sid) {
            out.push(t_bpb - full_bpb);
        }
    }
    (out, full_seeds.len())
}

fn aligned_delta(
    target_seeds: &[(u64, f64)],
    full_seeds: &[(u64, f64)],
) -> Vec<f64> {
    aligned_delta_with_count(target_seeds, full_seeds).0
}

#[derive(Debug, Clone)]
struct DualMediationRow {
    fix_x: String,
    delta_x: f64,
    nde: f64,           // Δ_{X,M1,M2}
    nie_m1: f64,        // X → M1 → Y only
    nie_m2: f64,        // X → M2 → Y only
    nie_chain: f64,     // X → M1 → M2 → Y
    sum: f64,           // NDE + NIE_M1 + NIE_M2 + NIE_chain (should ≈ Δ_X)
    residual: f64,      // Δ_X − sum (sanity for no-interaction assumption)
    pct_m1: f64,        // NIE_M1 / Δ_X × 100
    pct_m2: f64,        // NIE_M2 / Δ_X × 100
    pct_chain: f64,     // NIE_chain / Δ_X × 100
    pct_nde: f64,       // NDE / Δ_X × 100
    // Loop 35 (Loop 34 Option A): closed-form SEs via per-seed PSE variance.
    // For LINEAR functionals (our case — NDE/NIE_* are linear combos of
    // 4 means), the multivariate delta-method (Miles & Shpitser 2017,
    // arXiv:1710.02011 §3) reduces to the sample variance of the per-seed
    // PSE values divided by N. Equivalent to influence-function SE for
    // M-estimators when nuisance is plug-in (Kawakami-Tian, arXiv:2505.04983).
    se_nde: f64,
    se_nie_m1: f64,
    se_nie_m2: f64,
    se_nie_chain: f64,
    // Loop 36 fix 1: 95% CIs computed from SE × t_{0.975, N-1}. At N=5 (df=4)
    // the multiplier is ~2.776, not 1.96 — naive Gaussian undercoverage is what
    // arXiv:2508.10083 (Owen Aug 2025) flagged as the dominant small-N CI failure.
    ci95_nde_lo: f64,
    ci95_nde_hi: f64,
    ci95_nie_m1_lo: f64,
    ci95_nie_m1_hi: f64,
    ci95_nie_m2_lo: f64,
    ci95_nie_m2_hi: f64,
    ci95_nie_chain_lo: f64,
    ci95_nie_chain_hi: f64,
    n_seeds: usize,
}

/// Loop 47 audit fix 2: detect which stratum the input rows belong to.
///
/// At default `--m1 wd --m2 warmup`, a CSV from `warmup_stratified` has the
/// mediator pinned at its disabled value — so what the binary reports as
/// "NDE" / "NIE_*" are **Controlled Direct Effects** (Pearl CDE), not the
/// marginal NDE/NIE the user might assume from the column names. Surfacing
/// the stratum in the output header forces consumers to read the CDE framing
/// rather than the canonical labels.
fn detect_input_stratum(rows: &[LongRow]) -> &'static str {
    let mut has_canonical = false;
    let mut has_wd0 = false;
    let mut has_warmup0 = false;
    for r in rows {
        if r.mode.starts_with("wd0_") {
            has_wd0 = true;
        } else if r.mode.starts_with("warmup0_") {
            has_warmup0 = true;
        } else {
            has_canonical = true;
        }
    }
    match (has_canonical, has_wd0, has_warmup0) {
        (true, false, false) => "canonical",
        (false, true, false) => "wd0",
        (false, false, true) => "warmup0",
        (false, false, false) => "empty",
        _ => "mixed",
    }
}

fn compute_dual_mediation(
    rows: &[LongRow],
    m1: &str,
    m2: &str,
) -> Vec<DualMediationRow> {
    if !is_canonical_fix(m1) || !is_canonical_fix(m2) {
        eprintln!(
            "# ERROR: mediator(s) must be canonical fix names. Got M1='{}', M2='{}'. Valid: {:?}",
            m1, m2, CANONICAL_FIX_NAMES
        );
        return Vec::new();
    }
    if m1 == m2 {
        eprintln!("# ERROR: M1 and M2 must differ. Got both = '{}'", m1);
        return Vec::new();
    }

    let mut by_key: BTreeMap<(String, String), Vec<(u64, f64)>> = BTreeMap::new();
    for r in rows {
        by_key
            .entry((r.mode.clone(), r.fix_name.clone()))
            .or_default()
            .push((r.seed, r.bpb));
    }

    // Loop 38 fix 4 → Loop 39 fix 2: full_stack baseline lookup walks the
    // pairwise modes (preferred) then triplet modes via the central registry.
    let full_raw = all_mode_strings(ModeKind::Pairwise)
        .into_iter()
        .chain(all_mode_strings(ModeKind::Triplet))
        .find_map(|mode| by_key.get(&(mode, "full_stack".to_string())).cloned())
        .unwrap_or_default();
    if full_raw.is_empty() {
        eprintln!("# ERROR (dual-mediation): no full_stack baseline found.");
        return Vec::new();
    }
    // Loop 36 fix 4: dedupe seeds. Concatenating two CSVs that both have seed=42
    // would silently double-count the baseline; dedupe keeps the first
    // occurrence and emits a warning so the user knows.
    let mut full: Vec<(u64, f64)> = Vec::with_capacity(full_raw.len());
    let mut seen: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    let mut dups = 0usize;
    for (sid, bpb) in &full_raw {
        if seen.insert(*sid) {
            full.push((*sid, *bpb));
        } else {
            dups += 1;
        }
    }
    if dups > 0 {
        eprintln!(
            "# WARN (dual-mediation): {} duplicate seed id(s) in full_stack baseline (e.g. from concatenated CSVs); using first occurrence per seed.",
            dups
        );
    }

    let mut out = Vec::new();
    for x in CANONICAL_FIX_NAMES.iter() {
        if *x == m1 || *x == m2 {
            continue;
        }
        // Loop 38 fix 4 → Loop 39 fix 2: LOCO lookup walks the stratum registry.
        let loco_x = match all_mode_strings(ModeKind::Loco)
            .into_iter()
            .find_map(|mode| by_key.get(&(mode, x.to_string())))
        {
            Some(v) => v,
            None => continue,
        };
        let pair_xm1 = match lookup_pair_any_perm(&by_key, x, m1) {
            Some(v) => v,
            None => continue,
        };
        let pair_xm2 = match lookup_pair_any_perm(&by_key, x, m2) {
            Some(v) => v,
            None => continue,
        };
        let trip_xm1m2 = match lookup_triplet_any_perm(&by_key, x, m1, m2) {
            Some(v) => v,
            None => continue,
        };

        // Per-seed Δ samples. Loop 35 fix 1: detect seed alignment loss.
        let (dx_seeds, expected_n) = aligned_delta_with_count(loco_x, &full);
        let d_xm1_seeds = aligned_delta(pair_xm1, &full);
        let d_xm2_seeds = aligned_delta(pair_xm2, &full);
        let d_xm1m2_seeds = aligned_delta(trip_xm1m2, &full);
        // Loop 35 fix 1: warn (don't silently NaN) when seed coverage is partial.
        // A seed missing in any of the four series produces an undercount → biased
        // mean. We skip the row rather than emit half-baked numbers.
        if dx_seeds.len() < expected_n
            || d_xm1_seeds.len() < expected_n
            || d_xm2_seeds.len() < expected_n
            || d_xm1m2_seeds.len() < expected_n
        {
            eprintln!(
                "# WARN (dual-mediation): seed coverage incomplete for X={} (loco={}/{}, pair_xm1={}/{}, pair_xm2={}/{}, triplet={}/{}). Skipping row.",
                x,
                dx_seeds.len(), expected_n,
                d_xm1_seeds.len(), expected_n,
                d_xm2_seeds.len(), expected_n,
                d_xm1m2_seeds.len(), expected_n,
            );
            continue;
        }

        let delta_x = mean(&dx_seeds);
        let d_xm1 = mean(&d_xm1_seeds);
        let d_xm2 = mean(&d_xm2_seeds);
        let d_xm1m2 = mean(&d_xm1m2_seeds);

        // Path-specific effects per Zhao-Luo identification under no-XM-interaction.
        let nde = d_xm1m2;
        let nie_chain = (delta_x - d_xm1) - (d_xm2 - d_xm1m2);
        let nie_m1 = (delta_x - d_xm1) - nie_chain;
        let nie_m2 = (delta_x - d_xm2) - nie_chain;

        // Loop 35 (Loop 34 Option A): per-seed PSEs for delta-method SEs.
        // The coefficient matrix A (rows = PSEs, cols = [Δ_X, d_xm1, d_xm2, d_xm1m2]):
        //   NDE       = [0, 0, 0, 1]
        //   NIE_chain = [1, -1, -1, 1]
        //   NIE_M1    = [0, 0, 1, -1]   (after algebra)
        //   NIE_M2    = [0, 1, 0, -1]   (after algebra)
        let n_seeds = dx_seeds.len();
        let mut per_seed_nde = Vec::with_capacity(n_seeds);
        let mut per_seed_chain = Vec::with_capacity(n_seeds);
        let mut per_seed_m1 = Vec::with_capacity(n_seeds);
        let mut per_seed_m2 = Vec::with_capacity(n_seeds);
        for i in 0..n_seeds {
            let dx = dx_seeds[i];
            let d1 = d_xm1_seeds[i];
            let d2 = d_xm2_seeds[i];
            let d12 = d_xm1m2_seeds[i];
            per_seed_nde.push(d12);
            per_seed_chain.push(dx - d1 - d2 + d12);
            per_seed_m1.push(d2 - d12);
            per_seed_m2.push(d1 - d12);
        }
        let se_nde = sample_se(&per_seed_nde);
        let se_nie_chain = sample_se(&per_seed_chain);
        let se_nie_m1 = sample_se(&per_seed_m1);
        let se_nie_m2 = sample_se(&per_seed_m2);
        // Loop 36 fix 1: 95% CI = estimate ± t_{0.975, df=N-1} × SE.
        let df = (n_seeds as f64) - 1.0;
        let t_crit = if df >= 1.0 {
            student_t_critical_two_sided(0.05, df)
        } else {
            f64::NAN
        };
        let ci = |est: f64, se: f64| (est - t_crit * se, est + t_crit * se);
        let (ci95_nde_lo, ci95_nde_hi) = ci(nde, se_nde);
        let (ci95_nie_m1_lo, ci95_nie_m1_hi) = ci(nie_m1, se_nie_m1);
        let (ci95_nie_m2_lo, ci95_nie_m2_hi) = ci(nie_m2, se_nie_m2);
        let (ci95_nie_chain_lo, ci95_nie_chain_hi) = ci(nie_chain, se_nie_chain);
        let sum = nde + nie_m1 + nie_m2 + nie_chain;
        let residual = delta_x - sum;

        let pct = |v: f64| if delta_x.abs() > 1e-6 { v / delta_x * 100.0 } else { f64::NAN };

        out.push(DualMediationRow {
            fix_x: x.to_string(),
            delta_x,
            nde,
            nie_m1,
            nie_m2,
            nie_chain,
            sum,
            residual,
            pct_m1: pct(nie_m1),
            pct_m2: pct(nie_m2),
            pct_chain: pct(nie_chain),
            pct_nde: pct(nde),
            se_nde,
            se_nie_m1,
            se_nie_m2,
            se_nie_chain,
            ci95_nde_lo,
            ci95_nde_hi,
            ci95_nie_m1_lo,
            ci95_nie_m1_hi,
            ci95_nie_m2_lo,
            ci95_nie_m2_hi,
            ci95_nie_chain_lo,
            ci95_nie_chain_hi,
            n_seeds,
        });
    }
    // Sort by |Δ_X| descending — biggest total effects first.
    out.sort_by(|a, b| {
        b.delta_x.abs().partial_cmp(&a.delta_x.abs()).unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

fn emit<W: Write>(
    w: &mut W,
    rows: &[DualMediationRow],
    m1: &str,
    m2: &str,
    stratum: &str,
) -> std::io::Result<()> {
    writeln!(w, "# Dual-mediator decomposition (Zhao-Luo 2020 arXiv:2007.16031)")?;
    writeln!(w, "# INPUT STRATUM = {} (Loop 47 audit fix 2)", stratum)?;
    if stratum != "canonical" {
        writeln!(
            w,
            "# NOTE: rows came from a stratified sweep. NDE/NIE_* below are Pearl"
        )?;
        writeln!(
            w,
            "# CONTROLLED DIRECT/INDIRECT EFFECTS at the disabled-value of the stratum"
        )?;
        writeln!(
            w,
            "# variable (e.g. warmup0 = warmup_steps held at 0); NOT marginal effects."
        )?;
    }
    writeln!(w, "# M1 = {}, M2 = {}", m1, m2)?;
    writeln!(
        w,
        "# NDE = Δ_{{X,M1,M2}};  NIE_M1 = (Δ_X − Δ_{{X,M1}}) − NIE_chain;"
    )?;
    writeln!(
        w,
        "# NIE_M2 = (Δ_X − Δ_{{X,M2}}) − NIE_chain;"
    )?;
    writeln!(
        w,
        "# NIE_chain = (Δ_X − Δ_{{X,M1}}) − (Δ_{{X,M2}} − Δ_{{X,M1,M2}})"
    )?;
    writeln!(
        w,
        "# Loop 35: SEs from per-seed PSE variance (multivariate delta-method,"
    )?;
    writeln!(w, "# Miles & Shpitser 2017 arXiv:1710.02011 §3)")?;
    writeln!(
        w,
        "# Loop 36: 95% CIs use t_{{0.975, df=N-1}} (arXiv:2508.10083 — BCa undercovers at N=5)"
    )?;
    writeln!(
        w,
        "rank,fix_x,n,delta_x,nde,se_nde,ci95_nde_lo,ci95_nde_hi,nie_m1,se_nie_m1,ci95_nie_m1_lo,ci95_nie_m1_hi,nie_m2,se_nie_m2,ci95_nie_m2_lo,ci95_nie_m2_hi,nie_chain,se_nie_chain,ci95_nie_chain_lo,ci95_nie_chain_hi,sum,residual,pct_nde,pct_m1,pct_m2,pct_chain"
    )?;
    for (rank, r) in rows.iter().enumerate() {
        writeln!(
            w,
            "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2},{:.2},{:.2},{:.2}",
            rank + 1, r.fix_x, r.n_seeds, r.delta_x,
            r.nde, r.se_nde, r.ci95_nde_lo, r.ci95_nde_hi,
            r.nie_m1, r.se_nie_m1, r.ci95_nie_m1_lo, r.ci95_nie_m1_hi,
            r.nie_m2, r.se_nie_m2, r.ci95_nie_m2_lo, r.ci95_nie_m2_hi,
            r.nie_chain, r.se_nie_chain, r.ci95_nie_chain_lo, r.ci95_nie_chain_hi,
            r.sum, r.residual,
            r.pct_nde, r.pct_m1, r.pct_m2, r.pct_chain
        )?;
    }
    Ok(())
}

fn print_help() {
    println!("f2_dual_mediation — Loop 34: Zhao-Luo two-mediator path-specific decomposition");
    println!();
    println!("USAGE: f2_dual_mediation [FLAGS] CSV...");
    println!();
    println!("FLAGS:");
    println!("  --help, -h          Print this help and exit");
    println!("  --m1 NAME           First mediator (default: wd)");
    println!("  --m2 NAME           Second mediator (default: warmup)");
    println!("  --out PATH          Write decomposition CSV to file (default stdout)");
    println!();
    println!("Required rows: loco + pairwise (pair_<X>_<M1>, pair_<X>_<M2>) + triplet");
    println!("(triplet_<X>_<M1>_<M2>) + full_stack baseline.");
    println!();
    println!("Refs: arXiv:2007.16031 (Zhao-Luo 2020), arXiv:2505.04983 (small-N 2025).");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    let mut inputs: Vec<String> = Vec::new();
    let mut out_path: Option<String> = None;
    let mut m1 = String::from("wd");
    let mut m2 = String::from("warmup");
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if a == "--out" {
            if i + 1 >= args.len() {
                eprintln!("# ERROR: --out requires a path");
                std::process::exit(2);
            }
            out_path = Some(args[i + 1].clone());
            i += 2;
        } else if a == "--m1" {
            if i + 1 >= args.len() {
                eprintln!("# ERROR: --m1 requires a fix name");
                std::process::exit(2);
            }
            m1 = args[i + 1].clone();
            i += 2;
        } else if a == "--m2" {
            if i + 1 >= args.len() {
                eprintln!("# ERROR: --m2 requires a fix name");
                std::process::exit(2);
            }
            m2 = args[i + 1].clone();
            i += 2;
        } else if a.starts_with("--") {
            eprintln!("# ERROR: unknown flag {}", a);
            std::process::exit(2);
        } else {
            inputs.push(a.clone());
            i += 1;
        }
    }
    if inputs.is_empty() {
        eprintln!("# ERROR: no input CSVs given. See --help.");
        std::process::exit(2);
    }
    let mut all_rows = Vec::new();
    for p in &inputs {
        eprintln!("# Loading {}", p);
        all_rows.extend(parse_csv(p));
    }
    // Loop 47 audit fix 2: surface input stratum so user reads "Pearl CDE at
    // mediator=0", not marginal NDE/NIE.
    let stratum = detect_input_stratum(&all_rows);
    eprintln!(
        "# Loaded {} rows; M1='{}', M2='{}'; INPUT STRATUM = {}",
        all_rows.len(), m1, m2, stratum
    );
    let rows = compute_dual_mediation(&all_rows, &m1, &m2);
    if rows.is_empty() {
        eprintln!("# ERROR: no decomposition rows produced (missing loco/pairwise/triplet rows?).");
        std::process::exit(1);
    }
    if let Some(path) = out_path.as_deref() {
        let mut f = File::create(path).expect("create out CSV");
        emit(&mut f, &rows, &m1, &m2, stratum).expect("write");
        eprintln!("# Wrote {} rows to {}", rows.len(), path);
    } else {
        let stdout = std::io::stdout();
        let mut h = stdout.lock();
        emit(&mut h, &rows, &m1, &m2, stratum).expect("write stdout");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_rows() -> Vec<LongRow> {
        // X=rms, M1=wd, M2=warmup. Single seed for clean arithmetic.
        // full = 4.0
        // Δ_X      = 1.0  (LOCO rms = 5.0)
        // Δ_{X,M1} = 0.5  (pair rms,wd = 4.5)  ⇒ removing wd recovered 0.5 of X effect
        // Δ_{X,M2} = 0.7  (pair rms,warmup = 4.7)
        // Δ_{X,M1,M2} = 0.3 (triplet)
        vec![
            LongRow { mode: "pairwise".into(), fix_name: "full_stack".into(), seed: 1, bpb: 4.0 },
            LongRow { mode: "loco".into(), fix_name: "rms".into(), seed: 1, bpb: 5.0 },
            LongRow { mode: "pairwise".into(), fix_name: "pair_rms_wd".into(), seed: 1, bpb: 4.5 },
            LongRow { mode: "pairwise".into(), fix_name: "pair_rms_warmup".into(), seed: 1, bpb: 4.7 },
            LongRow { mode: "triplet".into(), fix_name: "triplet_rms_warmup_wd".into(), seed: 1, bpb: 4.3 },
            // Throw in canonical full_stack from triplet too for resolver coverage.
            LongRow { mode: "triplet".into(), fix_name: "full_stack".into(), seed: 1, bpb: 4.0 },
        ]
    }

    #[test]
    fn dual_mediation_identifies_paths_for_rms() {
        let rows = compute_dual_mediation(&synth_rows(), "wd", "warmup");
        assert_eq!(rows.len(), 1, "only rms should appear (M1=wd, M2=warmup excluded)");
        let r = &rows[0];
        assert!((r.delta_x - 1.0).abs() < 1e-9);
        assert!((r.nde - 0.3).abs() < 1e-9, "NDE expected 0.3, got {}", r.nde);
        // NIE_chain = (1.0 - 0.5) - (0.7 - 0.3) = 0.5 - 0.4 = 0.1
        assert!((r.nie_chain - 0.1).abs() < 1e-9, "chain expected 0.1, got {}", r.nie_chain);
        // NIE_M1 = (1.0 - 0.5) - 0.1 = 0.4
        assert!((r.nie_m1 - 0.4).abs() < 1e-9, "M1 expected 0.4, got {}", r.nie_m1);
        // NIE_M2 = (1.0 - 0.7) - 0.1 = 0.2
        assert!((r.nie_m2 - 0.2).abs() < 1e-9, "M2 expected 0.2, got {}", r.nie_m2);
        // Sum = 0.3 + 0.4 + 0.2 + 0.1 = 1.0 = Δ_X (under no-interaction)
        assert!((r.sum - 1.0).abs() < 1e-9);
        assert!(r.residual.abs() < 1e-9);
    }

    #[test]
    fn dual_mediation_emits_finite_ses_at_n_geq_2() {
        // Loop 35 (Loop 34 Option A) test: per-seed PSE variance produces finite
        // SEs. Multi-seed synth (N=3) so sample variance is well-defined.
        let mut rows = synth_rows();
        // Add seed=2 and seed=3 with mild jitter so SEs are non-zero.
        for sid in [2u64, 3u64] {
            let jitter = 0.01 * (sid as f64 - 1.0);
            rows.push(LongRow { mode: "pairwise".into(), fix_name: "full_stack".into(), seed: sid, bpb: 4.0 + jitter });
            rows.push(LongRow { mode: "loco".into(), fix_name: "rms".into(), seed: sid, bpb: 5.0 + jitter });
            rows.push(LongRow { mode: "pairwise".into(), fix_name: "pair_rms_wd".into(), seed: sid, bpb: 4.5 + jitter });
            rows.push(LongRow { mode: "pairwise".into(), fix_name: "pair_rms_warmup".into(), seed: sid, bpb: 4.7 + jitter });
            rows.push(LongRow { mode: "triplet".into(), fix_name: "triplet_rms_warmup_wd".into(), seed: sid, bpb: 4.3 + jitter });
        }
        let out = compute_dual_mediation(&rows, "wd", "warmup");
        assert_eq!(out.len(), 1);
        let r = &out[0];
        assert_eq!(r.n_seeds, 3);
        // All four SEs must be finite (not NaN).
        for (name, se) in [
            ("se_nde", r.se_nde),
            ("se_nie_m1", r.se_nie_m1),
            ("se_nie_m2", r.se_nie_m2),
            ("se_nie_chain", r.se_nie_chain),
        ] {
            assert!(se.is_finite() && se >= 0.0, "{} = {} should be finite ≥ 0", name, se);
        }
        // Synth jitter is identical across all 4 series → per-seed PSEs are
        // constant → variance ≈ 0 → SE ≈ 0. Sanity.
        assert!(r.se_nde < 1e-6, "expected se_nde~0, got {}", r.se_nde);
    }

    #[test]
    fn dual_mediation_no_interaction_residual_lock() {
        // Loop 35 fix 6: lock the Loop 34 empirical finding that residual < 1e-6
        // for the suppression-pattern regime. Synthesize a wider scenario with
        // 3 distinct non-mediator fixes to give the lock real teeth.
        let mut rows = Vec::new();
        // Mediator-pair scenario: M1=wd, M2=warmup. Test against rms, gradclip, dropout.
        for sid in [1u64, 2u64, 3u64] {
            let j = 0.001 * (sid as f64);
            rows.push(LongRow { mode: "pairwise".into(), fix_name: "full_stack".into(), seed: sid, bpb: 5.13 + j });
            for (fix, loco_bpb) in [
                ("rms", 6.00),
                ("gradclip", 4.43),
                ("dropout", 4.17),
                ("wd", 0.07),
                ("warmup", 4.42),
            ] {
                rows.push(LongRow { mode: "loco".into(), fix_name: fix.into(), seed: sid, bpb: loco_bpb + j });
            }
            for (lbl, bpb) in [
                ("pair_rms_wd", 2.28),
                ("pair_gradclip_wd", 0.49),
                ("pair_dropout_wd", 0.63),
                ("pair_rms_warmup", 5.13),       // sum-of-effects approx
                ("pair_gradclip_warmup", 4.43),
                ("pair_warmup_dropout", 4.17),
            ] {
                rows.push(LongRow { mode: "pairwise".into(), fix_name: lbl.into(), seed: sid, bpb: bpb + j });
            }
            for (lbl, bpb) in [
                ("triplet_rms_warmup_wd", 0.26),
                ("triplet_gradclip_warmup_wd", 0.49),
                ("triplet_warmup_wd_dropout", 0.63),
            ] {
                rows.push(LongRow { mode: "triplet".into(), fix_name: lbl.into(), seed: sid, bpb: bpb + j });
            }
        }
        let out = compute_dual_mediation(&rows, "wd", "warmup");
        // 5 non-mediator fixes in canonical names; we provided triplets for 3.
        assert!(out.len() >= 3, "expected ≥3 rows, got {}", out.len());
        for r in &out {
            assert!(
                r.residual.abs() < 1e-6,
                "no-XM-interaction residual must hold (Loop 34 finding): fix_x={}, residual={}",
                r.fix_x, r.residual
            );
        }
    }

    #[test]
    fn detect_input_stratum_classifies_each_prefix() {
        // Loop 47 audit fix 2: stratum detection.
        let canonical = vec![
            LongRow { mode: "pairwise".into(), fix_name: "full_stack".into(), seed: 1, bpb: 4.0 },
            LongRow { mode: "loco".into(), fix_name: "rms".into(), seed: 1, bpb: 5.0 },
        ];
        assert_eq!(detect_input_stratum(&canonical), "canonical");
        let wd0 = vec![
            LongRow { mode: "wd0_pairwise".into(), fix_name: "full_stack".into(), seed: 1, bpb: 4.0 },
            LongRow { mode: "wd0_loco".into(), fix_name: "rms".into(), seed: 1, bpb: 5.0 },
        ];
        assert_eq!(detect_input_stratum(&wd0), "wd0");
        let warmup0 = vec![
            LongRow { mode: "warmup0_pairwise".into(), fix_name: "full_stack".into(), seed: 1, bpb: 4.0 },
        ];
        assert_eq!(detect_input_stratum(&warmup0), "warmup0");
        let mixed = vec![
            LongRow { mode: "pairwise".into(), fix_name: "full_stack".into(), seed: 1, bpb: 4.0 },
            LongRow { mode: "wd0_loco".into(), fix_name: "rms".into(), seed: 1, bpb: 5.0 },
        ];
        assert_eq!(detect_input_stratum(&mixed), "mixed");
        assert_eq!(detect_input_stratum(&[]), "empty");
    }

    #[test]
    fn dual_mediation_m1_m2_swap_is_symmetric() {
        // Loop 41 fix 4: swapping (M1, M2) on the CLI must swap NIE_M1 ↔ NIE_M2
        // (and their SEs and CIs) exactly; NDE and NIE_chain remain unchanged.
        // The Zhao-Luo decomposition is *labeling-symmetric* in M1, M2 under
        // the no-interaction assumption.
        let mut rows = synth_rows();
        for sid in [2u64, 3u64] {
            let jit = 0.01 * (sid as f64);
            rows.push(LongRow { mode: "pairwise".into(), fix_name: "full_stack".into(), seed: sid, bpb: 4.0 + jit });
            rows.push(LongRow { mode: "loco".into(), fix_name: "rms".into(), seed: sid, bpb: 5.0 + jit });
            rows.push(LongRow { mode: "pairwise".into(), fix_name: "pair_rms_wd".into(), seed: sid, bpb: 4.5 + jit });
            rows.push(LongRow { mode: "pairwise".into(), fix_name: "pair_rms_warmup".into(), seed: sid, bpb: 4.7 + jit });
            rows.push(LongRow { mode: "triplet".into(), fix_name: "triplet_rms_warmup_wd".into(), seed: sid, bpb: 4.3 + jit });
        }
        let a = compute_dual_mediation(&rows, "wd", "warmup");
        let b = compute_dual_mediation(&rows, "warmup", "wd");
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        let ra = &a[0];
        let rb = &b[0];
        // NDE and chain are M1↔M2 symmetric and must be byte-identical.
        assert!((ra.nde - rb.nde).abs() < 1e-9, "NDE drifted: {} vs {}", ra.nde, rb.nde);
        assert!((ra.nie_chain - rb.nie_chain).abs() < 1e-9);
        // SE for NDE and chain unchanged.
        assert!((ra.se_nde - rb.se_nde).abs() < 1e-9);
        assert!((ra.se_nie_chain - rb.se_nie_chain).abs() < 1e-9);
        // NIE_M1 in run a == NIE_M2 in run b (and vice versa).
        assert!((ra.nie_m1 - rb.nie_m2).abs() < 1e-9, "NIE_M1↔M2 swap broken: a.M1={}, b.M2={}", ra.nie_m1, rb.nie_m2);
        assert!((ra.nie_m2 - rb.nie_m1).abs() < 1e-9);
        // SEs swap too.
        assert!((ra.se_nie_m1 - rb.se_nie_m2).abs() < 1e-9);
        assert!((ra.se_nie_m2 - rb.se_nie_m1).abs() < 1e-9);
        // CIs swap.
        assert!((ra.ci95_nie_m1_lo - rb.ci95_nie_m2_lo).abs() < 1e-9);
        assert!((ra.ci95_nie_m1_hi - rb.ci95_nie_m2_hi).abs() < 1e-9);
    }

    #[test]
    fn dual_mediation_accepts_wd_stratified_mode_rows() {
        // Loop 38 fix 4: rows tagged wd0_pairwise / wd0_triplet / wd0_loco
        // (emitted by f2_ablation_sweep --mode wd_stratified) must be
        // discoverable by the lookup helpers without renaming the mode column.
        let rows = vec![
            LongRow { mode: "wd0_pairwise".into(), fix_name: "full_stack".into(), seed: 1, bpb: 4.0 },
            LongRow { mode: "wd0_loco".into(),     fix_name: "rms".into(),         seed: 1, bpb: 5.0 },
            LongRow { mode: "wd0_pairwise".into(), fix_name: "pair_rms_wd".into(),     seed: 1, bpb: 4.5 },
            LongRow { mode: "wd0_pairwise".into(), fix_name: "pair_rms_warmup".into(), seed: 1, bpb: 4.7 },
            LongRow { mode: "wd0_triplet".into(),  fix_name: "triplet_rms_warmup_wd".into(), seed: 1, bpb: 4.3 },
        ];
        let scores = compute_dual_mediation(&rows, "wd", "warmup");
        assert_eq!(
            scores.len(),
            1,
            "expected 1 row (rms) when reading wd_stratified-mode rows; got {}",
            scores.len()
        );
        let r = &scores[0];
        assert_eq!(r.fix_x, "rms");
        // Sanity: Δ_X = 5 − 4 = 1; the rest follows arithmetic in
        // dual_mediation_identifies_paths_for_rms.
        assert!((r.delta_x - 1.0).abs() < 1e-9);
    }

    #[test]
    fn dual_mediation_rejects_non_canonical_mediator() {
        let rows = compute_dual_mediation(&synth_rows(), "wd", "not_a_fix");
        assert!(rows.is_empty());
    }

    #[test]
    fn dual_mediation_rejects_same_m1_m2() {
        let rows = compute_dual_mediation(&synth_rows(), "wd", "wd");
        assert!(rows.is_empty());
    }

    #[test]
    fn pair_and_triplet_labels_are_canonical_sorted() {
        assert_eq!(pair_label("wd", "rms"), "pair_rms_wd");
        assert_eq!(pair_label("rms", "wd"), "pair_rms_wd");
        assert_eq!(triplet_label("wd", "rms", "warmup"), "triplet_rms_warmup_wd");
    }
}
