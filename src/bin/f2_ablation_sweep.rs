//! F2 ablation sweep — Loop 24 NNN.
//!
//! Iterates AblationFix variants under cumulative + LOCO modes per NeurIPS/ICLR convention.
//! Emits LONG-FORM CSV (one row per seed × mode × fix). Wide-form aggregation deferred to
//! downstream tooling (per MLflow dataset-eval pattern: source of truth = long-form).

use std::fs::File;
use std::io::Write;
use trios_trainer::race::ablation::{cumulative_config, disable_in_config, AblationFix};
use trios_trainer::race::format_ladder::LadderKind;
use trios_trainer::race::multi_seed::{
    config_fingerprint, run_multi_seed, CorpusKind, MultiSeedConfig, TaskKind,
};

/// Canonical "ablation" base configuration.
///
/// Loop 43 fix 4 note: this differs intentionally from
/// `f2_pareto_sweep::base_config` (warmup_steps=200, label_smoothing=0.0).
/// Pareto sweeps target a longer-training regime (1000 steps) while ablation
/// targets a short-iteration sandbox (default 200 steps). Both are valid base
/// recipes; downstream callers must not assume cross-binary equivalence.
fn base_config(steps: usize) -> MultiSeedConfig {
    MultiSeedConfig {
        seeds: vec![42, 43, 44, 45, 46],
        train_ratio: 0.8,
        vocab_size: 64,
        d_model: 128,
        steps,
        lr: 0.004,
        ladder_kind: LadderKind::PhiLadder,
        warmup_steps_unquantized: 40,
        spike_injection_steps: Vec::new(),
        iso_neff_n_target: None,
        corpus: CorpusKind::Synthetic,
        paretoq_precision: None,
        disable_quantization: false,
        task_kind: TaskKind::Counter,
        use_ffn: true,
        d_hidden: 64,
        label_smoothing: 0.1,
        weight_decay: 0.1,
        apply_rmsnorm: true,
        grad_clip_l2: Some(1.0),
        latent_clamp_max: Some(1.0),
        dropout_p: 0.1,
    }
}

#[derive(Debug, Clone)]
struct CsvRow {
    mode: &'static str,
    fix_name: &'static str,
    fix_index: i32,
    cumulative_n: Option<usize>,
    seed: u64,
    bpb: f64,
    config_hash: u64,
    wall_s: f64,
}

/// Loop 32 fix 2: emit a PROV-style provenance preamble (`#` comments) before
/// the CSV header so post-hoc analysis can identify exactly which trainer build
/// produced each row. Follows W3C PROV + Workflow Run RO-Crate (arXiv:2312.07852).
/// Reader tools must skip `#`-prefixed lines (our `parse_csv` already does).
fn emit_provenance<W: Write>(w: &mut W, mode: &str, steps: usize) -> std::io::Result<()> {
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let git_sha = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    // Loop 33 fix 3: HOSTNAME/HOST env vars are typically unset on macOS / fresh
    // containers, so fall through to `uname -n` before giving up. Restores
    // post-hoc auditability for "where did this CSV come from?"
    let host = std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .ok()
        .or_else(|| {
            std::process::Command::new("uname")
                .arg("-n")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| "unknown".into());
    let rustc_v = option_env!("RUSTC_VERSION").unwrap_or(env!("CARGO_PKG_VERSION"));
    writeln!(
        w,
        "# f2_ablation_sweep provenance (W3C PROV / RO-Crate, arXiv:2312.07852)"
    )?;
    writeln!(w, "# prov:generatedAt = {} (unix seconds UTC)", now_secs)?;
    writeln!(
        w,
        "# prov:wasGeneratedBy = f2_ablation_sweep --mode {} --steps {}",
        mode, steps
    )?;
    writeln!(w, "# prov:agent_git_sha = {}", git_sha)?;
    writeln!(w, "# prov:host = {}", host)?;
    writeln!(
        w,
        "# prov:trainer_internals_schema = {}",
        trios_trainer::race::multi_seed::TRAINER_INTERNALS_SCHEMA
    )?;
    writeln!(w, "# prov:cargo_pkg_version = {}", rustc_v)?;
    writeln!(
        w,
        "# Bump trainer_internals_schema in src/race/multi_seed.rs when LCG/init/forward changes."
    )?;
    Ok(())
}

fn emit_csv<W: Write>(w: &mut W, rows: &[CsvRow]) -> std::io::Result<()> {
    writeln!(
        w,
        "mode,fix_name,fix_index,cumulative_n,seed,bpb,config_hash,wall_s"
    )?;
    for r in rows {
        let cum = r.cumulative_n.map(|n| n.to_string()).unwrap_or_default();
        writeln!(
            w,
            "{},{},{},{},{},{:.6},0x{:016x},{:.3}",
            r.mode, r.fix_name, r.fix_index, cum, r.seed, r.bpb, r.config_hash, r.wall_s
        )?;
    }
    Ok(())
}

fn run_cumulative(base: &MultiSeedConfig) -> Vec<CsvRow> {
    let mut rows = Vec::new();
    for n in 0..=AblationFix::ALL.len() {
        let cfg = cumulative_config(base, n);
        let t0 = std::time::Instant::now();
        let report = run_multi_seed(&cfg);
        let wall_s = t0.elapsed().as_secs_f64();
        let hash = config_fingerprint(&cfg);
        let label: &'static str = match n {
            0 => "baseline",
            1 => "rms",
            2 => "warmup",
            3 => "gradclip",
            4 => "clamp",
            5 => "smooth",
            6 => "wd",
            7 => "dropout",
            _ => "full",
        };
        for run in &report.runs {
            rows.push(CsvRow {
                mode: "cumulative",
                fix_name: label,
                fix_index: n as i32,
                cumulative_n: Some(n),
                seed: run.seed,
                bpb: run.val_bpb,
                config_hash: hash,
                wall_s,
            });
        }
    }
    rows
}

/// Loop 27 WWW + Loop 28 XXX: Pairwise iLOCO — disable pairs (i, j) of fixes simultaneously.
/// Per iLOCO arXiv:2502.06661 Eq.(3): `iLOCO_{j,k} = Δ_j + Δ_k - Δ_{j,k}`.
/// For 7 fixes, 7C2 = 21 pair experiments + 1 full-stack baseline row.
/// The full-stack row (all 7 fixes enabled) is the reference for Δ computation per Eq.(3).
fn run_pairwise(base: &MultiSeedConfig) -> Vec<CsvRow> {
    let mut rows = Vec::new();
    let all = AblationFix::ALL;
    // Loop 28 XXX: emit full-stack baseline first (all fixes enabled = cumulative n=7).
    // This is the reference point for Δ_j, Δ_k, Δ_{j,k} in iLOCO Eq.(3).
    let full_cfg = cumulative_config(base, AblationFix::ALL.len());
    let t0 = std::time::Instant::now();
    let full_report = run_multi_seed(&full_cfg);
    let full_wall_s = t0.elapsed().as_secs_f64();
    let full_hash = config_fingerprint(&full_cfg);
    for run in &full_report.runs {
        rows.push(CsvRow {
            mode: "pairwise",
            fix_name: "full_stack",
            fix_index: -1,
            cumulative_n: None,
            seed: run.seed,
            bpb: run.val_bpb,
            config_hash: full_hash,
            wall_s: full_wall_s,
        });
    }
    let mut pair_idx: i32 = 0;
    for i in 0..all.len() {
        for j in (i + 1)..all.len() {
            let mut cfg = base.clone();
            disable_in_config(&mut cfg, all[i]);
            disable_in_config(&mut cfg, all[j]);
            let t0 = std::time::Instant::now();
            let report = run_multi_seed(&cfg);
            let wall_s = t0.elapsed().as_secs_f64();
            let hash = config_fingerprint(&cfg);
            // Note: leak-then-cleanup pattern for static string with pair label
            let label_owned = format!("pair_{}_{}", all[i].short_name(), all[j].short_name());
            let label: &'static str = Box::leak(label_owned.into_boxed_str());
            for run in &report.runs {
                rows.push(CsvRow {
                    mode: "pairwise",
                    fix_name: label,
                    fix_index: pair_idx,
                    cumulative_n: None,
                    seed: run.seed,
                    bpb: run.val_bpb,
                    config_hash: hash,
                    wall_s,
                });
            }
            pair_idx += 1;
        }
    }
    rows
}

/// Loop 29 Option B: Triplet sweep — disable triples (i,j,k) of fixes simultaneously.
/// 7C3 = 35 experiments + 1 full_stack baseline. Feeds 3-way iLOCO via Möbius
/// inclusion-exclusion (cf. arXiv:2502.06661 + IT-SHAP arXiv:2512.05338).
fn run_triplet(base: &MultiSeedConfig) -> Vec<CsvRow> {
    let mut rows = Vec::new();
    let all = AblationFix::ALL;
    // Full-stack baseline (all 7 enabled) — same reference as pairwise mode.
    let full_cfg = cumulative_config(base, AblationFix::ALL.len());
    let t0 = std::time::Instant::now();
    let full_report = run_multi_seed(&full_cfg);
    let full_wall_s = t0.elapsed().as_secs_f64();
    let full_hash = config_fingerprint(&full_cfg);
    for run in &full_report.runs {
        rows.push(CsvRow {
            mode: "triplet",
            fix_name: "full_stack",
            fix_index: -1,
            cumulative_n: None,
            seed: run.seed,
            bpb: run.val_bpb,
            config_hash: full_hash,
            wall_s: full_wall_s,
        });
    }
    let mut idx: i32 = 0;
    for i in 0..all.len() {
        for j in (i + 1)..all.len() {
            for k in (j + 1)..all.len() {
                let mut cfg = base.clone();
                disable_in_config(&mut cfg, all[i]);
                disable_in_config(&mut cfg, all[j]);
                disable_in_config(&mut cfg, all[k]);
                let t0 = std::time::Instant::now();
                let report = run_multi_seed(&cfg);
                let wall_s = t0.elapsed().as_secs_f64();
                let hash = config_fingerprint(&cfg);
                let label_owned = format!(
                    "triplet_{}_{}_{}",
                    all[i].short_name(),
                    all[j].short_name(),
                    all[k].short_name()
                );
                let label: &'static str = Box::leak(label_owned.into_boxed_str());
                for run in &report.runs {
                    rows.push(CsvRow {
                        mode: "triplet",
                        fix_name: label,
                        fix_index: idx,
                        cumulative_n: None,
                        seed: run.seed,
                        bpb: run.val_bpb,
                        config_hash: hash,
                        wall_s,
                    });
                }
                idx += 1;
            }
        }
    }
    rows
}

/// Loop 26 SSS: WD sweep — vary weight_decay per Power Lines (Bergsma 2025 arXiv:2505.13738).
/// For sub-Chinchilla D/N regime, expected λ_opt ≈ 0.005-0.02. BitNet's 0.1 is for >1B params.
fn run_wd_sweep(base: &MultiSeedConfig) -> Vec<CsvRow> {
    let mut rows = Vec::new();
    let wd_values: [f64; 6] = [0.0, 0.005, 0.01, 0.03, 0.1, 0.3];
    for (i, &wd) in wd_values.iter().enumerate() {
        let mut cfg = base.clone();
        cfg.weight_decay = wd;
        let t0 = std::time::Instant::now();
        let report = run_multi_seed(&cfg);
        let wall_s = t0.elapsed().as_secs_f64();
        let hash = config_fingerprint(&cfg);
        // Encode WD value into fix_name for downstream parsing
        let label: &'static str = match i {
            0 => "wd_0.000",
            1 => "wd_0.005",
            2 => "wd_0.010",
            3 => "wd_0.030",
            4 => "wd_0.100",
            5 => "wd_0.300",
            _ => "wd_unknown",
        };
        for run in &report.runs {
            rows.push(CsvRow {
                mode: "wd_sweep",
                fix_name: label,
                fix_index: i as i32,
                cumulative_n: None,
                seed: run.seed,
                bpb: run.val_bpb,
                config_hash: hash,
                wall_s,
            });
        }
    }
    rows
}

fn run_loco(base: &MultiSeedConfig) -> Vec<CsvRow> {
    let mut rows = Vec::new();
    for (i, &fix) in AblationFix::ALL.iter().enumerate() {
        let mut cfg = base.clone();
        disable_in_config(&mut cfg, fix);
        let t0 = std::time::Instant::now();
        let report = run_multi_seed(&cfg);
        let wall_s = t0.elapsed().as_secs_f64();
        let hash = config_fingerprint(&cfg);
        for run in &report.runs {
            rows.push(CsvRow {
                mode: "loco",
                fix_name: fix.short_name(),
                fix_index: i as i32,
                cumulative_n: None,
                seed: run.seed,
                bpb: run.val_bpb,
                config_hash: hash,
                wall_s,
            });
        }
    }
    rows
}

fn print_help() {
    println!("f2_ablation_sweep — Loop 24 NNN ablation matrix");
    println!();
    println!("USAGE: f2_ablation_sweep [FLAGS]");
    println!();
    println!("FLAGS:");
    println!("  --help, -h          Print this help and exit");
    println!("  --csv PATH          Write long-form CSV to file (default stdout)");
    println!("  --mode MODE         'cumulative'|'loco'|'both'|'wd_sweep'|'pairwise'|");
    println!("                      'triplet'|'wd_pairwise'|'wd_stratified'|'all'");
    println!("                      wd_stratified = Pearl CDE at WD=0.0:");
    println!("                        cumulative + loco + pairwise + triplet");
    println!("                        (rows tagged 'wd0_*' mode); enables full");
    println!("                        f2_dual_mediation on the WD=0 stratum.");
    println!("  --quick-stratified  Force --steps 50 for a ~6min sanity-check run");
    println!("                      (vs. ~25min at the default 200 steps).");
    println!("  --steps N           Override base steps (default 200, RRR uses 1000)");
}

/// Loop 29 Option C scaffold: Pair-level λ_opt sweep — for each non-WD fix X,
/// sweep WD ∈ [0, 0.005, 0.01, 0.03, 0.1] WHILE disabling X. Tests whether
/// iLOCO_{wd, X} magnitude follows a τ-power-law per Bergsma 2025
/// (arXiv:2505.13738) — no published pair-level extension exists.
fn run_wd_pairwise(base: &MultiSeedConfig) -> Vec<CsvRow> {
    let mut rows = Vec::new();
    let wd_values: [f64; 5] = [0.0, 0.005, 0.01, 0.03, 0.1];
    // For each non-wd fix X, run the WD grid while X is disabled.
    let partners: Vec<AblationFix> = AblationFix::ALL
        .iter()
        .copied()
        .filter(|f| f.short_name() != "wd")
        .collect();
    let mut idx: i32 = 0;
    for partner in &partners {
        for &wd in &wd_values {
            let mut cfg = base.clone();
            disable_in_config(&mut cfg, *partner);
            cfg.weight_decay = wd;
            let t0 = std::time::Instant::now();
            let report = run_multi_seed(&cfg);
            let wall_s = t0.elapsed().as_secs_f64();
            let hash = config_fingerprint(&cfg);
            let label_owned = format!("wdpair_{}_{:.3}", partner.short_name(), wd);
            let label: &'static str = Box::leak(label_owned.into_boxed_str());
            for run in &report.runs {
                rows.push(CsvRow {
                    mode: "wd_pairwise",
                    fix_name: label,
                    fix_index: idx,
                    cumulative_n: None,
                    seed: run.seed,
                    bpb: run.val_bpb,
                    config_hash: hash,
                    wall_s,
                });
            }
            idx += 1;
        }
    }
    rows
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    let csv_path = args
        .iter()
        .position(|a| a == "--csv")
        .and_then(|i| args.get(i + 1).cloned());
    let mode = args
        .iter()
        .position(|a| a == "--mode")
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| "both".to_string());
    let mut steps: usize = args
        .iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    // Loop 39 fix 5: `--quick-stratified` forces a 50-step run for cheap sanity
    // checks. At 5 seeds × 71 cells × 50 steps the total wall time drops from
    // ~25min (full) to ~6min — fast enough to validate that the dual-mediation
    // pipeline produces the expected output before committing to the full sweep.
    if args.iter().any(|a| a == "--quick-stratified") {
        eprintln!(
            "# QUICK-STRATIFIED: forcing --steps 50 (down from {}). Use without --quick-stratified for publication-grade 200-step run.",
            steps
        );
        steps = 50;
    }

    let base = base_config(steps);
    eprintln!(
        "# f2_ablation_sweep: 7 fixes × 5 seeds × mode={}, steps={}",
        mode, base.steps
    );

    let mut all_rows = Vec::new();
    if mode == "cumulative" || mode == "both" {
        eprintln!("# Running cumulative add-on (n=0..7)...");
        all_rows.extend(run_cumulative(&base));
    }
    if mode == "loco" || mode == "both" {
        eprintln!("# Running leave-one-out (7 fixes)...");
        all_rows.extend(run_loco(&base));
    }
    if mode == "wd_sweep" || mode == "all" {
        eprintln!("# Running WD sweep (SSS, Power Lines arXiv:2505.13738)...");
        all_rows.extend(run_wd_sweep(&base));
    }
    if mode == "pairwise" || mode == "all" {
        eprintln!("# Running pairwise iLOCO (WWW, arXiv:2502.06661)...");
        all_rows.extend(run_pairwise(&base));
    }
    if mode == "triplet" || mode == "all" {
        eprintln!("# Running triplet iLOCO (Loop 29 Option B, 7C3=35 + full_stack)...");
        all_rows.extend(run_triplet(&base));
    }
    if mode == "wd_pairwise" || mode == "all" {
        eprintln!("# Running pair-level WD sweep (Loop 29 Option C, Power Lines ext)...");
        all_rows.extend(run_wd_pairwise(&base));
    }
    // Loop 31 fix 3 / Loop 30 Option A: WD-stratified ablation = Pearl Controlled
    // Direct Effect with mediator fixed at λ=0.0 (validated optimum at this scale).
    // Theoretical basis: arXiv:2506.14019 (path-specific effects), arXiv:2408.14620v2
    // (intermediate-confounding triggers re-stratification when mediator dominates
    // ≥80% of total effect; Loop 30 observed 70-95%). Tag every row with the
    // _wd0 suffix so downstream tooling can distinguish from default-WD runs.
    // Loop 41 fix 3: the prefix comes from `Stratum::Wd0.prefix()` so adding a
    // new stratum (e.g. Warmup0) requires no edits here.
    if mode == "wd_stratified" || mode == "all" {
        eprintln!("# Running WD-stratified ablation (Loop 31, CDE at WD=0.0)...");
        let mut wd0_base = base.clone();
        wd0_base.weight_decay = 0.0;
        let prefix = trios_trainer::race::ablation::Stratum::Wd0.prefix();
        let tag = |mut r: CsvRow| -> CsvRow {
            let leaked: &'static str = Box::leak(format!("{}{}", prefix, r.mode).into_boxed_str());
            r.mode = leaked;
            r
        };
        all_rows.extend(run_cumulative(&wd0_base).into_iter().map(tag));
        all_rows.extend(run_loco(&wd0_base).into_iter().map(tag));
        all_rows.extend(run_pairwise(&wd0_base).into_iter().map(tag));
        // Loop 37 fix 5: add triplet rows at WD=0 so f2_dual_mediation can run
        // its full 4-PSE decomposition on the WD=0 stratum.
        all_rows.extend(run_triplet(&wd0_base).into_iter().map(tag));
    }
    // Loop 41 fix 3: Warmup0 stratified ablation (Pearl CDE on warmup, after
    // Loop 33 identified warmup as the 2nd dominant mediator).
    if mode == "warmup_stratified" || mode == "all" {
        eprintln!("# Running Warmup-stratified ablation (Loop 41, CDE at warmup_steps=0)...");
        let mut wmu0_base = base.clone();
        wmu0_base.warmup_steps_unquantized = 0;
        let prefix = trios_trainer::race::ablation::Stratum::Warmup0.prefix();
        let tag = |mut r: CsvRow| -> CsvRow {
            let leaked: &'static str = Box::leak(format!("{}{}", prefix, r.mode).into_boxed_str());
            r.mode = leaked;
            r
        };
        all_rows.extend(run_cumulative(&wmu0_base).into_iter().map(tag));
        all_rows.extend(run_loco(&wmu0_base).into_iter().map(tag));
        all_rows.extend(run_pairwise(&wmu0_base).into_iter().map(tag));
        all_rows.extend(run_triplet(&wmu0_base).into_iter().map(tag));
    }

    if let Some(path) = csv_path.as_deref() {
        let mut f = File::create(path).expect("create CSV");
        emit_provenance(&mut f, &mode, steps).expect("write provenance");
        emit_csv(&mut f, &all_rows).expect("write CSV");
        eprintln!("# Wrote {} rows to {}", all_rows.len(), path);
    } else {
        let stdout = std::io::stdout();
        let mut handle = stdout.lock();
        emit_provenance(&mut handle, &mode, steps).expect("write provenance");
        emit_csv(&mut handle, &all_rows).expect("write stdout CSV");
    }
}
