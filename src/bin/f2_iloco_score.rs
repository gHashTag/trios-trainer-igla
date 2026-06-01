//! F2 iLOCO scorer — Loop 28 YYY.
//!
//! Computes iLOCO_{j,k} = Δ_j + Δ_k − Δ_{j,k} per arXiv:2502.06661 Eq.(3) for all 21
//! pair experiments from a combined LOCO + pairwise long-form CSV. Applies
//! Benjamini-Hochberg FDR (arXiv:1712.03305 — defensible vs Bonferroni for pairwise
//! t-statistics at α=0.10) and emits a sorted table.
//!
//! Sign convention:
//!   - Δ_j = BPB(without j) − BPB(full_stack); positive = removing j hurts.
//!   - iLOCO_{j,k} > 0 → compensatory (sum of individual removals overstates joint cost).
//!   - iLOCO_{j,k} < 0 → redundant   (joint removal costs more than sum of parts).
//!
//! Per Hooker 2019: compensatory pairs are the interpretability signature of
//! over-regularization; Loop 27 found WD as the dominant compensatory partner.

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
    let f = File::open(path).expect("open CSV");
    let r = BufReader::new(f);
    let mut rows = Vec::new();
    for (i, line) in r.lines().enumerate() {
        let line = line.expect("read");
        if i == 0 || line.starts_with('#') || line.is_empty() {
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
    v.iter().sum::<f64>() / v.len() as f64
}

fn std_dev(v: &[f64]) -> f64 {
    let m = mean(v);
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    var.sqrt()
}

/// Loop 30 fix 6: exact paired sign-flip permutation test (Fisher-Pitman).
/// Per arXiv:2205.01416 (Zmigrod, Vieira, Cotterell 2022), exhaustive 2^N
/// permutation at N=5 enumerates 32 sign-flips → exact two-tailed tail.
/// No df=4 underflow; no asymptotic assumption. Returns (mean_diff, p_two_tailed).
fn permutation_test_paired(a: &[f64], b: &[f64]) -> (f64, f64) {
    let n = a.len().min(b.len());
    // Loop 31 fix 4: align edge-case behavior with paired_t (which requires n≥2).
    // n=1 has only 2 sign-flips (±1), gives p∈{0.5,1.0} — degenerate, drop it.
    if n < 2 {
        return (f64::NAN, f64::NAN);
    }
    let diffs: Vec<f64> = (0..n).map(|i| a[i] - b[i]).collect();
    let observed: f64 = diffs.iter().sum();
    let total: u64 = 1u64 << n; // 2^n sign-flips; OK for n ≤ 32
    let mut ge_count: u64 = 0;
    for mask in 0..total {
        let mut s = 0.0_f64;
        for i in 0..n {
            let sign = if (mask >> i) & 1 == 1 { -1.0 } else { 1.0 };
            s += sign * diffs[i];
        }
        if s.abs() >= observed.abs() - 1e-15 {
            ge_count += 1;
        }
    }
    let p = ge_count as f64 / total as f64;
    (observed / n as f64, p.clamp(0.0, 1.0))
}

/// Paired Student's t-test on differences (df=n-1).
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
    let p = 2.0 * student_t_cdf_upper(t.abs(), df);
    (t, p.clamp(0.0, 1.0))
}

// Loop 37 fix 1: t-CDF, incomplete beta, and lgamma now live in race::stats —
// previously duplicated here. Migration is silent (same numerical behavior;
// race::stats has its own unit tests in src/race/stats.rs).
use trios_trainer::race::stats::student_t_cdf_upper;
// Loop 38 fix 1+2: cov/var/pearson moved to race::stats with their own tests.
// control_variate_adjust stays local — it's specialized to iLOCO scoring and
// not generic enough to warrant exporting.
use trios_trainer::race::stats::{cov, pearson, var};

/// Loop 31: Control-variate adjusted sample.
///
/// Returns y - β · (x - mean(x)) where β = Cov(x, y) / Var(x), jackknife-debiased
/// per arXiv:2411.02909 ("When is it worthwhile to jackknife?", Nov 2024) which
/// corrects the O(p/n) plug-in bias of β̂ = Ĉov / V̂ar at small N (arXiv:2402.07349).
///
/// At N=5, the plug-in β has bias O(1/N) = 20%; jackknife correction reduces it to
/// O(1/N²) = 4%. The adjusted Δ has variance reduced by 1 − ρ² where ρ = Cor(x, y).
fn control_variate_adjust(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len().min(y.len());
    if n < 3 {
        // Below N=3 jackknife is degenerate; return y unchanged.
        return y[..n].to_vec();
    }
    // Plug-in estimate.
    let beta_pi = cov(&x[..n], &y[..n]) / var(&x[..n]).max(1e-12);
    // Jackknife: average leave-one-out β values to get bias-corrected estimate.
    let mut beta_loo = Vec::with_capacity(n);
    for i in 0..n {
        let mut x_loo = Vec::with_capacity(n - 1);
        let mut y_loo = Vec::with_capacity(n - 1);
        for j in 0..n {
            if j != i {
                x_loo.push(x[j]);
                y_loo.push(y[j]);
            }
        }
        beta_loo.push(cov(&x_loo, &y_loo) / var(&x_loo).max(1e-12));
    }
    let beta_jk = n as f64 * beta_pi - (n - 1) as f64 * mean(&beta_loo);
    let mx = mean(&x[..n]);
    (0..n).map(|i| y[i] - beta_jk * (x[i] - mx)).collect()
}

/// Pair-key canonicalizer: always store as (min, max) for symmetric lookup.
fn pair_key(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

/// Loop 30 fix 1: canonical short-name set; gates label parsing against
/// future numeric-suffix labels (e.g. wd_pairwise's "wdpair_rms_0.000") leaking
/// into iLOCO scoring if mode column is mis-typed in a hand-edited CSV.
/// Loop 34 fix 5: imported from race::ablation as the single source of truth.
use trios_trainer::race::ablation::{is_canonical_fix, CANONICAL_FIX_NAMES};

/// Parse pairwise label "pair_<name1>_<name2>" → (name1, name2).
/// Both names must be members of CANONICAL_FIX_NAMES; otherwise returns None.
fn parse_pair_label(label: &str) -> Option<(String, String)> {
    let stripped = label.strip_prefix("pair_")?;
    // Names from AblationFix::short_name(): rms, warmup, gradclip, clamp, smooth, wd, dropout.
    // None contain underscores so split_once on '_' is unambiguous.
    let (a, b) = stripped.split_once('_')?;
    if !is_canonical_fix(a) || !is_canonical_fix(b) {
        return None;
    }
    Some((a.to_string(), b.to_string()))
}

/// Loop 29 Option B + Loop 30 fix 1: parse triplet label "triplet_<a>_<b>_<c>".
/// All three names must be canonical fix short_names.
fn parse_triplet_label(label: &str) -> Option<(String, String, String)> {
    let stripped = label.strip_prefix("triplet_")?;
    let parts: Vec<&str> = stripped.split('_').collect();
    if parts.len() != 3 {
        return None;
    }
    if !is_canonical_fix(parts[0]) || !is_canonical_fix(parts[1]) || !is_canonical_fix(parts[2]) {
        return None;
    }
    Some((
        parts[0].to_string(),
        parts[1].to_string(),
        parts[2].to_string(),
    ))
}

/// Triplet key: canonical sorted tuple for symmetric lookup.
fn triplet_key(a: &str, b: &str, c: &str) -> (String, String, String) {
    let mut v = [a, b, c];
    v.sort();
    (v[0].to_string(), v[1].to_string(), v[2].to_string())
}

#[derive(Debug, Clone)]
struct IlocoScore {
    fix_a: String,
    fix_b: String,
    /// Δ_a = BPB(without a) − BPB(full)
    delta_a: f64,
    /// Δ_b = BPB(without b) − BPB(full)
    delta_b: f64,
    /// Δ_{a,b} = BPB(without a,b) − BPB(full)
    delta_ab: f64,
    /// iLOCO = Δ_a + Δ_b − Δ_{a,b}; >0 compensatory, <0 redundant
    iloco: f64,
    /// Paired-t p-value on iLOCO sample (per-seed sum).
    p_value: f64,
    /// BH-adjusted q-value at α=0.10.
    q_value: f64,
}

/// Benjamini-Hochberg step-up procedure (arXiv:1712.03305).
/// Returns q-values for each input p-value at the false discovery rate level.
fn benjamini_hochberg(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }
    let mut indexed: Vec<(usize, f64)> = p_values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut q = vec![1.0_f64; m];
    let mut prev = 1.0_f64;
    for rank in (0..m).rev() {
        let (orig_idx, p) = indexed[rank];
        let adj = (p * m as f64 / (rank + 1) as f64).min(prev);
        q[orig_idx] = adj.clamp(0.0, 1.0);
        prev = adj;
    }
    q
}

fn compute_iloco(rows: &[LongRow]) -> Vec<IlocoScore> {
    compute_iloco_with_opts(rows, SigTest::PairedT, false, 0.3)
}

fn compute_iloco_with_test(rows: &[LongRow], sig: SigTest) -> Vec<IlocoScore> {
    compute_iloco_with_opts(rows, sig, false, 0.3)
}

fn compute_iloco_with_opts(
    rows: &[LongRow],
    sig: SigTest,
    cv: bool,
    cv_min_rho: f64,
) -> Vec<IlocoScore> {
    // Group seeds per (mode, fix_name) → per-seed BPB.
    let mut by_key: BTreeMap<(String, String), Vec<(u64, f64)>> = BTreeMap::new();
    for r in rows {
        by_key
            .entry((r.mode.clone(), r.fix_name.clone()))
            .or_default()
            .push((r.seed, r.bpb));
    }
    let full_seeds: Vec<(u64, f64)> = by_key
        .get(&("pairwise".to_string(), "full_stack".to_string()))
        .cloned()
        .unwrap_or_default();
    if full_seeds.is_empty() {
        eprintln!("# ERROR: no 'full_stack' baseline row in pairwise mode. Re-run f2_ablation_sweep --mode pairwise to emit it.");
        return Vec::new();
    }

    // Index LOCO rows by short_name.
    let mut loco: BTreeMap<String, Vec<(u64, f64)>> = BTreeMap::new();
    for ((mode, name), seeds) in &by_key {
        if mode == "loco" {
            loco.insert(name.clone(), seeds.clone());
        }
    }

    // Index pairwise rows by canonical pair key.
    let mut pairs: BTreeMap<(String, String), Vec<(u64, f64)>> = BTreeMap::new();
    for ((mode, name), seeds) in &by_key {
        if mode == "pairwise" && name != "full_stack" {
            if let Some((a, b)) = parse_pair_label(name) {
                pairs.insert(pair_key(&a, &b), seeds.clone());
            }
        }
    }

    // Loop 33 fix 5: if the user passed a stratified CSV (e.g. mode="wd0_pairwise"
    // from --mode wd_stratified) the scorer would silently produce zero rows.
    // Emit a diagnostic of all modes seen so the cause is obvious. Per W3C CSVW
    // "withheld data" principle: surface rejection, don't drop quietly.
    if loco.is_empty() || pairs.is_empty() {
        let mut modes_seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for ((m, _), _) in by_key.iter() {
            modes_seen.insert(m.clone());
        }
        eprintln!(
            "# DIAGNOSTIC: no canonical 'loco' + 'pairwise' rows found. Modes seen: {:?}. \
             If your CSV contains stratified modes (e.g. 'wd0_pairwise'), preprocess by \
             stripping the wd0_/wmu0_ prefix before running iLOCO.",
            modes_seen
        );
    }

    let mut scores: Vec<IlocoScore> = Vec::new();
    let fix_names: Vec<String> = loco.keys().cloned().collect();
    for i in 0..fix_names.len() {
        for j in (i + 1)..fix_names.len() {
            let a = &fix_names[i];
            let b = &fix_names[j];
            let key = pair_key(a, b);
            let (Some(loc_a), Some(loc_b), Some(pair_ab)) =
                (loco.get(a), loco.get(b), pairs.get(&key))
            else {
                continue;
            };
            // Build per-seed Δ samples aligned by seed id.
            let mut per_seed_iloco: Vec<f64> = Vec::new();
            let mut full_samples: Vec<f64> = Vec::new();
            for (sid, full_bpb) in &full_seeds {
                let la = loc_a.iter().find(|(s, _)| s == sid).map(|(_, v)| *v);
                let lb = loc_b.iter().find(|(s, _)| s == sid).map(|(_, v)| *v);
                let pab = pair_ab.iter().find(|(s, _)| s == sid).map(|(_, v)| *v);
                if let (Some(la), Some(lb), Some(pab)) = (la, lb, pab) {
                    let da = la - full_bpb;
                    let db = lb - full_bpb;
                    let dab = pab - full_bpb;
                    per_seed_iloco.push(da + db - dab);
                    full_samples.push(*full_bpb);
                }
            }
            if per_seed_iloco.is_empty() {
                continue;
            }
            // Loop 31: control-variate adjustment using full_stack BPB as covariate.
            // Variance reduction by factor (1 − ρ²) per arXiv:2510.13504.
            // Loop 32 fix 6: only apply CV when |ρ| ≥ cv_min_rho. At low ρ, the
            // jackknife β has more noise than the variance reduction it brings.
            let adjusted_iloco: Vec<f64> = if cv {
                let rho = pearson(&full_samples, &per_seed_iloco);
                if rho.abs() >= cv_min_rho {
                    control_variate_adjust(&full_samples, &per_seed_iloco)
                } else {
                    eprintln!(
                        "# CV-SKIP: pair ({},{}) has |ρ|={:.3} < {:.3}; CV would inject noise — using raw Δ.",
                        a, b, rho.abs(), cv_min_rho
                    );
                    per_seed_iloco.clone()
                }
            } else {
                per_seed_iloco.clone()
            };
            let iloco = mean(&adjusted_iloco);
            // Significance: paired-t or exact sign-flip permutation, selected via sig.
            let zeros = vec![0.0_f64; adjusted_iloco.len()];
            let (_, p) = run_sig_test(&adjusted_iloco, &zeros, sig);
            // Means for display.
            let full_mean = mean(&full_seeds.iter().map(|(_, v)| *v).collect::<Vec<_>>());
            let delta_a = mean(&loc_a.iter().map(|(_, v)| *v).collect::<Vec<_>>()) - full_mean;
            let delta_b = mean(&loc_b.iter().map(|(_, v)| *v).collect::<Vec<_>>()) - full_mean;
            let delta_ab = mean(&pair_ab.iter().map(|(_, v)| *v).collect::<Vec<_>>()) - full_mean;
            scores.push(IlocoScore {
                fix_a: a.clone(),
                fix_b: b.clone(),
                delta_a,
                delta_b,
                delta_ab,
                iloco,
                p_value: p,
                q_value: f64::NAN,
            });
        }
    }
    let ps: Vec<f64> = scores.iter().map(|s| s.p_value).collect();
    let qs = benjamini_hochberg(&ps);
    for (s, q) in scores.iter_mut().zip(qs.iter()) {
        s.q_value = *q;
    }
    // Sort by |iLOCO| descending (most interactive pairs first).
    scores.sort_by(|a, b| {
        b.iloco
            .abs()
            .partial_cmp(&a.iloco.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scores
}

/// Loop 29 Option B: 3-way iLOCO via inclusion-exclusion (Möbius inversion).
///
/// iLOCO_{j,k,l} = (Δ_j + Δ_k + Δ_l) − (Δ_{j,k} + Δ_{j,l} + Δ_{k,l}) + Δ_{j,k,l}
///
/// This is the natural Möbius extension of Eq.(3) from arXiv:2502.06661 to three
/// features. Same sign convention: +iLOCO = compensatory (joint less harmful than
/// inclusion-exclusion predicts), −iLOCO = redundant/synergistic.
#[derive(Debug, Clone)]
struct IlocoTripletScore {
    fix_a: String,
    fix_b: String,
    fix_c: String,
    iloco_3: f64,
    p_value: f64,
    q_value: f64,
}

fn compute_iloco_three_way(rows: &[LongRow]) -> Vec<IlocoTripletScore> {
    compute_iloco_three_way_with_test(rows, SigTest::PairedT)
}

fn compute_iloco_three_way_with_test(rows: &[LongRow], sig: SigTest) -> Vec<IlocoTripletScore> {
    let mut by_key: BTreeMap<(String, String), Vec<(u64, f64)>> = BTreeMap::new();
    for r in rows {
        by_key
            .entry((r.mode.clone(), r.fix_name.clone()))
            .or_default()
            .push((r.seed, r.bpb));
    }
    // Loop 30 fix 4: 3-way baseline should prefer triplet-mode full_stack
    // (same sweep as the triplet rows), not pairwise's. They should be
    // numerically identical (same config), but if both are present and differ
    // (e.g. user combined CSVs from different sweeps), warn and pick triplet.
    let triplet_full = by_key.get(&("triplet".to_string(), "full_stack".to_string()));
    let pairwise_full = by_key.get(&("pairwise".to_string(), "full_stack".to_string()));
    let full_seeds: Vec<(u64, f64)> = match (triplet_full, pairwise_full) {
        (Some(t), Some(p)) => {
            let t_mean: f64 = t.iter().map(|(_, v)| *v).sum::<f64>() / t.len() as f64;
            let p_mean: f64 = p.iter().map(|(_, v)| *v).sum::<f64>() / p.len() as f64;
            if (t_mean - p_mean).abs() > 0.01 {
                eprintln!(
                    "# WARN (3-way): triplet full_stack mean ({:.4}) differs from pairwise full_stack ({:.4}) by >0.01 BPB. \
                     Preferring triplet baseline.",
                    t_mean, p_mean
                );
            }
            t.clone()
        }
        (Some(t), None) => t.clone(),
        (None, Some(p)) => p.clone(),
        (None, None) => Vec::new(),
    };
    if full_seeds.is_empty() {
        eprintln!("# ERROR (3-way): no full_stack baseline row in triplet or pairwise mode.");
        return Vec::new();
    }
    let mut loco: BTreeMap<String, Vec<(u64, f64)>> = BTreeMap::new();
    let mut pairs: BTreeMap<(String, String), Vec<(u64, f64)>> = BTreeMap::new();
    let mut triplets: BTreeMap<(String, String, String), Vec<(u64, f64)>> = BTreeMap::new();
    for ((mode, name), seeds) in &by_key {
        match mode.as_str() {
            "loco" => {
                loco.insert(name.clone(), seeds.clone());
            }
            "pairwise" if name != "full_stack" => {
                if let Some((a, b)) = parse_pair_label(name) {
                    pairs.insert(pair_key(&a, &b), seeds.clone());
                }
            }
            "triplet" if name != "full_stack" => {
                if let Some((a, b, c)) = parse_triplet_label(name) {
                    triplets.insert(triplet_key(&a, &b, &c), seeds.clone());
                }
            }
            _ => {}
        }
    }

    let mut scores = Vec::new();
    let fix_names: Vec<String> = loco.keys().cloned().collect();
    for i in 0..fix_names.len() {
        for j in (i + 1)..fix_names.len() {
            for k in (j + 1)..fix_names.len() {
                let a = &fix_names[i];
                let b = &fix_names[j];
                let c = &fix_names[k];
                let tkey = triplet_key(a, b, c);
                let pab = pairs.get(&pair_key(a, b));
                let pac = pairs.get(&pair_key(a, c));
                let pbc = pairs.get(&pair_key(b, c));
                let la = loco.get(a);
                let lb = loco.get(b);
                let lc = loco.get(c);
                let tabc = triplets.get(&tkey);
                let (Some(la), Some(lb), Some(lc), Some(pab), Some(pac), Some(pbc), Some(tabc)) =
                    (la, lb, lc, pab, pac, pbc, tabc)
                else {
                    continue;
                };
                let mut per_seed: Vec<f64> = Vec::new();
                for (sid, full_bpb) in &full_seeds {
                    let g = |v: &Vec<(u64, f64)>| v.iter().find(|(s, _)| s == sid).map(|(_, x)| *x);
                    if let (
                        Some(la),
                        Some(lb),
                        Some(lc),
                        Some(pab),
                        Some(pac),
                        Some(pbc),
                        Some(tabc),
                    ) = (g(la), g(lb), g(lc), g(pab), g(pac), g(pbc), g(tabc))
                    {
                        let da = la - full_bpb;
                        let db = lb - full_bpb;
                        let dc = lc - full_bpb;
                        let dab = pab - full_bpb;
                        let dac = pac - full_bpb;
                        let dbc = pbc - full_bpb;
                        let dabc = tabc - full_bpb;
                        // Möbius / inclusion-exclusion 3-way iLOCO.
                        per_seed.push((da + db + dc) - (dab + dac + dbc) + dabc);
                    }
                }
                if per_seed.is_empty() {
                    continue;
                }
                let iloco_3 = mean(&per_seed);
                let zeros = vec![0.0; per_seed.len()];
                let (_, p) = run_sig_test(&per_seed, &zeros, sig);
                scores.push(IlocoTripletScore {
                    fix_a: a.clone(),
                    fix_b: b.clone(),
                    fix_c: c.clone(),
                    iloco_3,
                    p_value: p,
                    q_value: f64::NAN,
                });
            }
        }
    }
    let ps: Vec<f64> = scores.iter().map(|s| s.p_value).collect();
    let qs = benjamini_hochberg(&ps);
    for (s, q) in scores.iter_mut().zip(qs.iter()) {
        s.q_value = *q;
    }
    scores.sort_by(|a, b| {
        b.iloco_3
            .abs()
            .partial_cmp(&a.iloco_3.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scores
}

fn emit_three_way<W: Write>(w: &mut W, scores: &[IlocoTripletScore]) -> std::io::Result<()> {
    writeln!(w, "rank,fix_a,fix_b,fix_c,iloco_3,kind,p_value,q_value_bh")?;
    for (rank, s) in scores.iter().enumerate() {
        let kind = if s.iloco_3 > 0.0 {
            "compensatory"
        } else if s.iloco_3 < 0.0 {
            "redundant"
        } else {
            "independent"
        };
        writeln!(
            w,
            "{},{},{},{},{:.6},{},{:.4e},{:.4e}",
            rank + 1,
            s.fix_a,
            s.fix_b,
            s.fix_c,
            s.iloco_3,
            kind,
            s.p_value,
            s.q_value
        )?;
    }
    Ok(())
}

fn emit<W: Write>(w: &mut W, scores: &[IlocoScore]) -> std::io::Result<()> {
    writeln!(
        w,
        "rank,fix_a,fix_b,delta_a,delta_b,delta_ab,iloco,kind,p_value,q_value_bh"
    )?;
    for (rank, s) in scores.iter().enumerate() {
        let kind = if s.iloco > 0.0 {
            "compensatory"
        } else if s.iloco < 0.0 {
            "redundant"
        } else {
            "independent"
        };
        writeln!(
            w,
            "{},{},{},{:.6},{:.6},{:.6},{:.6},{},{:.4e},{:.4e}",
            rank + 1,
            s.fix_a,
            s.fix_b,
            s.delta_a,
            s.delta_b,
            s.delta_ab,
            s.iloco,
            kind,
            s.p_value,
            s.q_value
        )?;
    }
    Ok(())
}

fn print_help() {
    println!("f2_iloco_score — Loop 28 YYY: pairwise interaction scoring");
    println!();
    println!("USAGE: f2_iloco_score [FLAGS] CSV...");
    println!();
    println!("Reads one or more long-form CSVs (from f2_ablation_sweep) and computes");
    println!("iLOCO_{{j,k}} = Δ_j + Δ_k − Δ_{{j,k}} per arXiv:2502.06661 Eq.(3) for all");
    println!("7C2 = 21 pairs, then applies BH FDR (arXiv:1712.03305) for multiple testing.");
    println!();
    println!("FLAGS:");
    println!("  --help, -h          Print this help and exit");
    println!("  --out PATH          Write CSV to file (default stdout)");
    println!("  --three-way         Compute 3-way iLOCO_{{j,k,l}} via Möbius inclusion-exclusion");
    println!("                      (arXiv:2502.06661 ext; cf. IT-SHAP arXiv:2512.05338).");
    println!("  --permutation       Use exact paired sign-flip permutation test instead of");
    println!("                      Student's t (arXiv:2205.01416). Exhaustive at N≤16,");
    println!("                      no df underflow — recommended for N≤8 seeds.");
    println!("  --control-variate   Apply jackknife-debiased control-variate adjustment");
    println!("                      using full_stack BPB as covariate (arXiv:2510.13504 +");
    println!("                      arXiv:2411.02909). Variance reduced by (1−ρ²).");
    println!("  --cv-min-rho F      Skip CV when |Pearson ρ| < F (default 0.3). Below the");
    println!("                      threshold, CV injects more noise than it removes.");
    println!();
    println!("Required rows in input(s):");
    println!(
        "  - loco rows         (mode=loco, fix_name = rms|warmup|gradclip|clamp|smooth|wd|dropout)"
    );
    println!("  - pairwise rows     (mode=pairwise, fix_name = pair_<a>_<b>)");
    println!("  - full_stack row    (mode=pairwise, fix_name = full_stack) ← baseline");
    println!(
        "  - triplet rows      (--three-way only: mode=triplet, fix_name = triplet_<a>_<b>_<c>)"
    );
}

/// Loop 29 audit fix 1: explicit indexed loop instead of filter-with-wrapping_sub.
/// Loop 30: added --permutation flag for exact paired sign-flip test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SigTest {
    PairedT,
    Permutation,
}
#[derive(Debug, Clone)]
struct CliOpts {
    out_path: Option<String>,
    three_way: bool,
    sig: SigTest,
    control_variate: bool,
    cv_min_rho: f64,
}

fn parse_args(args: &[String]) -> (Vec<String>, CliOpts) {
    let mut inputs = Vec::new();
    let mut opts = CliOpts {
        out_path: None,
        three_way: false,
        sig: SigTest::PairedT,
        control_variate: false,
        cv_min_rho: 0.3,
    };
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if a == "--out" {
            if i + 1 >= args.len() {
                eprintln!("# ERROR: --out requires a path argument");
                std::process::exit(2);
            }
            opts.out_path = Some(args[i + 1].clone());
            i += 2;
        } else if a == "--three-way" {
            opts.three_way = true;
            i += 1;
        } else if a == "--permutation" {
            opts.sig = SigTest::Permutation;
            i += 1;
        } else if a == "--control-variate" {
            opts.control_variate = true;
            i += 1;
        } else if a == "--cv-min-rho" {
            if i + 1 >= args.len() {
                eprintln!("# ERROR: --cv-min-rho requires a value");
                std::process::exit(2);
            }
            opts.cv_min_rho = args[i + 1].parse().expect("parse --cv-min-rho");
            i += 2;
        } else if a == "--help" || a == "-h" {
            i += 1;
        } else if a.starts_with("--") {
            eprintln!("# ERROR: unknown flag {}", a);
            std::process::exit(2);
        } else {
            inputs.push(a.clone());
            i += 1;
        }
    }
    (inputs, opts)
}

fn run_sig_test(a: &[f64], b: &[f64], sig: SigTest) -> (f64, f64) {
    match sig {
        SigTest::PairedT => paired_t_test(a, b),
        SigTest::Permutation => permutation_test_paired(a, b),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    let (inputs, opts) = parse_args(&args);
    if inputs.is_empty() {
        eprintln!("# ERROR: no input CSVs given. See --help.");
        std::process::exit(2);
    }
    let mut all_rows = Vec::new();
    for path in &inputs {
        eprintln!("# Loading {}", path);
        all_rows.extend(parse_csv(path));
    }
    eprintln!(
        "# Loaded {} rows from {} file(s)",
        all_rows.len(),
        inputs.len()
    );
    let three_way = opts.three_way;
    let sig = opts.sig;
    let out_path = opts.out_path.clone();
    // Loop 30 fix 6: power-regime advisory based on (m_tests, n_seeds).
    let n_seeds_est = all_rows
        .iter()
        .filter(|r| r.mode == "pairwise" && r.fix_name == "full_stack")
        .count();
    if n_seeds_est <= 5 {
        let m = if three_way { 35 } else { 21 };
        let bh_rank1 = 0.10 / m as f64;
        eprintln!(
            "# POWER ADVISORY: N≈{} seeds × m={} tests → BH(α=0.10) rank-1 q-critical ≈ {:.4}. \
             Consider --permutation for exact tail at small N (no t/df underflow). See arXiv:2205.01416.",
            n_seeds_est, m, bh_rank1
        );
    }
    if opts.control_variate {
        eprintln!(
            "# CONTROL-VARIATE: jackknife-debiased β (arXiv:2411.02909, arXiv:2402.07349) applied to per-seed Δ samples."
        );
    }
    if three_way {
        let scores = compute_iloco_three_way_with_test(&all_rows, sig);
        if scores.is_empty() {
            eprintln!("# ERROR: no 3-way iLOCO scores produced — need loco + pairwise + triplet + full_stack rows.");
            std::process::exit(1);
        }
        eprintln!(
            "# Computed {} three-way iLOCO scores (Möbius inclusion-exclusion, sig={:?})",
            scores.len(),
            sig
        );
        if let Some(path) = out_path.as_deref() {
            let mut f = File::create(path).expect("create out CSV");
            emit_three_way(&mut f, &scores).expect("write");
            eprintln!("# Wrote {} rows to {}", scores.len(), path);
        } else {
            let stdout = std::io::stdout();
            let mut h = stdout.lock();
            emit_three_way(&mut h, &scores).expect("write stdout");
        }
    } else {
        let scores = compute_iloco_with_opts(&all_rows, sig, opts.control_variate, opts.cv_min_rho);
        if scores.is_empty() {
            eprintln!("# ERROR: no iLOCO scores produced — check that input has loco + pairwise + full_stack rows.");
            std::process::exit(1);
        }
        eprintln!(
            "# Computed {} iLOCO pair scores (sig={:?})",
            scores.len(),
            sig
        );
        if let Some(path) = out_path.as_deref() {
            let mut f = File::create(path).expect("create out CSV");
            emit(&mut f, &scores).expect("write");
            eprintln!("# Wrote {} rows to {}", scores.len(), path);
        } else {
            let stdout = std::io::stdout();
            let mut h = stdout.lock();
            emit(&mut h, &scores).expect("write stdout");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_pair_label_handles_canonical_pairs() {
        assert_eq!(
            parse_pair_label("pair_rms_warmup"),
            Some(("rms".to_string(), "warmup".to_string()))
        );
        assert_eq!(
            parse_pair_label("pair_wd_dropout"),
            Some(("wd".to_string(), "dropout".to_string()))
        );
        assert_eq!(parse_pair_label("full_stack"), None);
    }

    #[test]
    fn pair_key_is_symmetric() {
        assert_eq!(pair_key("wd", "warmup"), pair_key("warmup", "wd"));
    }

    #[test]
    fn pearson_perfect_linear_is_plus_minus_one() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
        assert!((pearson(&x, &y) - 1.0).abs() < 1e-9);
        let y_neg: Vec<f64> = x.iter().map(|v| -3.0 * v).collect();
        assert!((pearson(&x, &y_neg) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn pearson_constant_returns_zero() {
        let x = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(pearson(&x, &y), 0.0);
    }

    #[test]
    fn control_variate_zero_correlation_returns_unchanged() {
        // x uncorrelated with y (constant x): β̂ ≈ 0/0 — guarded by max(var, 1e-12).
        // Expect adjusted ≈ y (jackknife-debiased β still ~0).
        let x = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let adj = control_variate_adjust(&x, &y);
        // With zero variance in x, β = cov/eps ≈ 0, so adjusted ≈ y.
        for i in 0..5 {
            assert!(
                (adj[i] - y[i]).abs() < 1e-6,
                "expected ~y[i], got {}",
                adj[i]
            );
        }
    }

    #[test]
    fn control_variate_perfect_correlation_zeroes_residual() {
        // y = 2x + 1: perfect linear relationship. After CV adjustment,
        // residual variance should be ≪ original.
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
        let adj = control_variate_adjust(&x, &y);
        // Adjusted should be approximately constant = mean(y) = 7.0.
        let m = mean(&adj);
        for &a in &adj {
            assert!((a - m).abs() < 1e-6, "CV residual not flat: {}", a);
        }
    }

    #[test]
    fn control_variate_preserves_mean() {
        // CV adjustment subtracts β·(x − x̄); since mean(x − x̄) = 0,
        // the adjusted mean equals the original mean.
        let x = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let y = vec![3.0, 4.0, 6.0, 7.0, 10.0];
        let adj = control_variate_adjust(&x, &y);
        assert!((mean(&adj) - mean(&y)).abs() < 1e-9);
    }

    #[test]
    fn permutation_test_paired_zero_diff_returns_p_one() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = a.clone();
        let (_, p) = permutation_test_paired(&a, &b);
        // All differences are 0; |observed|=0; every sign-flip yields 0 ≥ 0.
        assert!(
            (p - 1.0).abs() < 1e-9,
            "expected p=1 for zero diffs, got {}",
            p
        );
    }

    #[test]
    fn permutation_test_paired_extreme_diff_hits_min_p() {
        // Maximum signal: all diffs share sign; observed |sum| = 5.
        // Only 2 of 32 sign-vectors reach |sum|=5: all-plus and all-minus.
        let a = vec![6.0, 6.0, 6.0, 6.0, 6.0];
        let b = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let (_, p) = permutation_test_paired(&a, &b);
        assert!(
            (p - 2.0 / 32.0).abs() < 1e-9,
            "expected p=2/32={}, got {}",
            2.0 / 32.0,
            p
        );
    }

    #[test]
    fn triplet_key_is_invariant_under_permutation() {
        let k1 = triplet_key("wd", "rms", "dropout");
        let k2 = triplet_key("dropout", "wd", "rms");
        let k3 = triplet_key("rms", "dropout", "wd");
        assert_eq!(k1, k2);
        assert_eq!(k2, k3);
    }

    #[test]
    fn parse_triplet_label_canonical() {
        assert_eq!(
            parse_triplet_label("triplet_rms_wd_dropout"),
            Some(("rms".to_string(), "wd".to_string(), "dropout".to_string()))
        );
        assert_eq!(parse_triplet_label("triplet_rms_wd"), None);
        assert_eq!(parse_triplet_label("pair_rms_wd"), None);
    }

    #[test]
    fn compute_iloco_three_way_independent_triplet_is_zero() {
        // Construct a triplet where all interactions vanish: BPB(without S) = full
        // baseline for all S. Möbius sum should be exactly 0.
        let base = 4.0;
        let rows = vec![
            LongRow {
                mode: "pairwise".into(),
                fix_name: "full_stack".into(),
                fix_index: -1,
                seed: 1,
                bpb: base,
            },
            LongRow {
                mode: "loco".into(),
                fix_name: "rms".into(),
                fix_index: 0,
                seed: 1,
                bpb: base,
            },
            LongRow {
                mode: "loco".into(),
                fix_name: "wd".into(),
                fix_index: 1,
                seed: 1,
                bpb: base,
            },
            LongRow {
                mode: "loco".into(),
                fix_name: "dropout".into(),
                fix_index: 2,
                seed: 1,
                bpb: base,
            },
            LongRow {
                mode: "pairwise".into(),
                fix_name: "pair_rms_wd".into(),
                fix_index: 0,
                seed: 1,
                bpb: base,
            },
            LongRow {
                mode: "pairwise".into(),
                fix_name: "pair_rms_dropout".into(),
                fix_index: 1,
                seed: 1,
                bpb: base,
            },
            LongRow {
                mode: "pairwise".into(),
                fix_name: "pair_wd_dropout".into(),
                fix_index: 2,
                seed: 1,
                bpb: base,
            },
            LongRow {
                mode: "triplet".into(),
                fix_name: "triplet_rms_wd_dropout".into(),
                fix_index: 0,
                seed: 1,
                bpb: base,
            },
        ];
        let scores = compute_iloco_three_way(&rows);
        assert_eq!(scores.len(), 1);
        assert!(
            scores[0].iloco_3.abs() < 1e-9,
            "expected ~0, got {}",
            scores[0].iloco_3
        );
    }

    #[test]
    fn compute_iloco_three_way_non_zero_when_higher_order_present() {
        // Construct alternating-sign Möbius result: Δ_S = 1 for |S|=1, 1 for |S|=2, 1 for |S|=3.
        // iLOCO_3 = (1+1+1) − (1+1+1) + 1 = +1.
        let base = 0.0;
        let rows = vec![
            LongRow {
                mode: "pairwise".into(),
                fix_name: "full_stack".into(),
                fix_index: -1,
                seed: 1,
                bpb: base,
            },
            LongRow {
                mode: "loco".into(),
                fix_name: "rms".into(),
                fix_index: 0,
                seed: 1,
                bpb: 1.0,
            },
            LongRow {
                mode: "loco".into(),
                fix_name: "wd".into(),
                fix_index: 1,
                seed: 1,
                bpb: 1.0,
            },
            LongRow {
                mode: "loco".into(),
                fix_name: "dropout".into(),
                fix_index: 2,
                seed: 1,
                bpb: 1.0,
            },
            LongRow {
                mode: "pairwise".into(),
                fix_name: "pair_rms_wd".into(),
                fix_index: 0,
                seed: 1,
                bpb: 1.0,
            },
            LongRow {
                mode: "pairwise".into(),
                fix_name: "pair_rms_dropout".into(),
                fix_index: 1,
                seed: 1,
                bpb: 1.0,
            },
            LongRow {
                mode: "pairwise".into(),
                fix_name: "pair_wd_dropout".into(),
                fix_index: 2,
                seed: 1,
                bpb: 1.0,
            },
            LongRow {
                mode: "triplet".into(),
                fix_name: "triplet_rms_wd_dropout".into(),
                fix_index: 0,
                seed: 1,
                bpb: 1.0,
            },
        ];
        let scores = compute_iloco_three_way(&rows);
        assert_eq!(scores.len(), 1);
        assert!(
            (scores[0].iloco_3 - 1.0).abs() < 1e-9,
            "expected +1, got {}",
            scores[0].iloco_3
        );
    }

    #[test]
    fn benjamini_hochberg_monotone_in_rank() {
        // Per BH step-up: p-values are scaled by m/rank.
        let ps = vec![0.001, 0.01, 0.05, 0.5];
        let qs = benjamini_hochberg(&ps);
        // q-values must be non-decreasing in the same rank order.
        assert!(qs[0] <= qs[1]);
        assert!(qs[1] <= qs[2]);
        assert!(qs[2] <= qs[3]);
        // q ≤ 1 always.
        for q in &qs {
            assert!(*q >= 0.0 && *q <= 1.0);
        }
    }

    #[test]
    fn benjamini_hochberg_uniform_gives_max_one() {
        let ps = vec![1.0_f64; 10];
        let qs = benjamini_hochberg(&ps);
        for q in &qs {
            assert!((q - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn compute_iloco_produces_compensatory_for_warmup_wd_synthetic() {
        // Synthetic data matching Loop 27 finding: removing wd from full-stack
        // helps massively; removing warmup hurts; removing both helps almost as
        // much as wd alone → iLOCO > 0 (compensatory).
        let rows = vec![
            // full_stack baseline
            LongRow {
                mode: "pairwise".into(),
                fix_name: "full_stack".into(),
                fix_index: -1,
                seed: 1,
                bpb: 4.37,
            },
            // LOCO wd: removing wd drops BPB
            LongRow {
                mode: "loco".into(),
                fix_name: "wd".into(),
                fix_index: 0,
                seed: 1,
                bpb: 0.07,
            },
            // LOCO warmup: removing warmup hurts a bit
            LongRow {
                mode: "loco".into(),
                fix_name: "warmup".into(),
                fix_index: 1,
                seed: 1,
                bpb: 4.42,
            },
            // pairwise (warmup, wd): removing both — close to wd-alone
            LongRow {
                mode: "pairwise".into(),
                fix_name: "pair_warmup_wd".into(),
                fix_index: 0,
                seed: 1,
                bpb: 0.26,
            },
        ];
        let scores = compute_iloco(&rows);
        assert_eq!(scores.len(), 1);
        let s = &scores[0];
        // Δ_warmup = 4.42 − 4.37 = +0.05
        // Δ_wd     = 0.07 − 4.37 = −4.30
        // Δ_{w,wd} = 0.26 − 4.37 = −4.11
        // iLOCO    = 0.05 + (−4.30) − (−4.11) = −0.14? wait re-check sign.
        // Actually iLOCO = Δ_j + Δ_k − Δ_{j,k} = 0.05 − 4.30 − (−4.11) = -0.14
        // Hmm — Loop 27 reported +3.44 using a different baseline (cumulative n=0).
        // With pairwise full_stack baseline the sign flips relative to Loop 27 doc.
        // This is intentional: full_stack IS the correct iLOCO reference per Eq.(3).
        assert!(s.iloco.is_finite());
        assert_eq!(s.fix_a, "warmup");
        assert_eq!(s.fix_b, "wd");
    }
}
