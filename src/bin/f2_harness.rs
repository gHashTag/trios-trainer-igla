//! F2 head-to-head harness — Issue #1021
//!
//! Runs both arms of the F2 protocol at equal bit-budget:
//!   (a) Phi-ladder: GFTernary → GF8 → GF16 → GF32 (Lucas-exact accumulator)
//!   (b) Format zoo: BitNet-ternary + INT8 + FP8 + bf16
//!
//! Reports held-out BPB (N≥5 seeds, mean±MC-error) and lossy conversion counts.
//! Verdict rule: if zoo or posit matches phi-ladder on both metrics, moat
//! demotes to [Risk]. One run never promotes moat to [Verified].

use std::fs;
use trios_trainer::race::format_ladder::LadderKind;
use trios_trainer::race::multi_seed::CorpusKind;
use trios_trainer::race::multi_seed::{
    aggregate_verdict, bayesian_credible_diff, bayesian_credible_normalized,
    bootstrap_t_ci_diff, config_fingerprint, paired_bca_bootstrap, permutation_test,
    power_matrix, run_multi_seed, verdict_pareto_adjusted, verdict_pareto_welch, verdict_tost,
    verdict_welch, welch_mde, welch_power, welch_power_unequal, BayesianCrediblReport,
    F2AggregatedReport, MultiSeedConfig, MultiSeedReport, ParetoWelchReport, PermutationReport,
    F2_SCHEMA_VERSION,
};

#[derive(serde::Serialize)]
struct F2Output<'a> {
    schema_version: &'static str,
    phi: &'a MultiSeedReport,
    zoo: &'a MultiSeedReport,
    welch: trios_trainer::race::multi_seed::VerdictReport,
    tost: trios_trainer::race::multi_seed::TostReport,
    pareto_verdict: trios_trainer::race::multi_seed::ParetoVerdict,
    pareto_welch: ParetoWelchReport,
    bayesian: BayesianCrediblReport,
    permutation: PermutationReport,
    aggregated: F2AggregatedReport,
    bootstrap_ci_lo: f64,
    bootstrap_ci_hi: f64,
    bootstrap_bayes_disagree: bool,
    pooled_sigma: f64,
    sigma_phi: f64,
    sigma_zoo: f64,
    mde_at_power_08: f64,
    power_at_observed: f64,
    power_at_observed_unequal: f64,
}

fn parse_f64_flag(args: &[String], flag: &str) -> Option<f64> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn parse_output_path(args: &[String]) -> Option<String> {
    args.iter()
        .position(|a| a == "--output-path")
        .and_then(|i| args.get(i + 1).cloned())
}

fn print_arm(label: &str, cfg: &MultiSeedConfig, quiet: bool) -> MultiSeedReport {
    if !quiet {
        println!("\n## Arm: {} ({:?})", label, cfg.ladder_kind);
    }
    let t0 = std::time::Instant::now();
    let report = run_multi_seed(cfg);
    let wall = t0.elapsed().as_secs_f64();

    if quiet {
        return report;
    }

    println!(
        "  seeds={}  steps={}  d_model={}  vocab={}  warmup={}",
        report.runs.len(),
        cfg.steps,
        cfg.d_model,
        cfg.vocab_size,
        cfg.warmup_steps_unquantized,
    );
    println!(
        "  val_bpb = {:.4} ± {:.4} (std={:.4})",
        report.mean_val_bpb, report.mc_error, report.std_val_bpb
    );
    println!(
        "  lossy_conversions_total = {}  wall={:.1}s",
        report.total_lossy_conversions, wall
    );
    let total_spikes: u64 = report.runs.iter().map(|r| r.stability.loss_spike_count).sum();
    let total_grad_spikes: u64 = report.runs.iter().map(|r| r.stability.grad_norm_spike_count).sum();
    let total_nan: u64 = report.runs.iter().map(|r| r.stability.nan_step_count).sum();
    let worst_grad: f64 = report
        .runs
        .iter()
        .map(|r| r.stability.worst_grad_norm)
        .fold(0.0_f64, f64::max);
    println!(
        "  stability: loss_spikes={}  grad_spikes={}  nan_steps={}  worst_grad_norm={:.3}",
        total_spikes, total_grad_spikes, total_nan, worst_grad,
    );
    println!(
        "  pareto: N={}  bpw_stored={:.2}  bpw_effective={:.2}  N_eff={:.0}  opt_overhead={:.0} bits",
        report.pareto.n_params,
        report.pareto.bits_per_weight_stored,
        report.pareto.bits_per_weight_effective,
        report.pareto.n_eff,
        report.pareto.optimizer_overhead_bits,
    );
    println!("  per-seed:");
    for r in &report.runs {
        println!(
            "    seed={:<5}  train_bpb={:.4}  val_bpb={:.4}  lossy={}  spikes={}",
            r.seed, r.train_bpb, r.val_bpb, r.lossy_conversions, r.stability.loss_spike_count
        );
    }
    report
}

fn print_help() {
    println!("f2_harness — F2 head-to-head BPB comparison (Issue #1021)");
    println!();
    println!("USAGE: f2_harness [FLAGS]");
    println!();
    println!("FLAGS:");
    println!("  --help, -h            Print this help and exit");
    println!("  --json                Emit structured JSON to stdout (schema: {})", F2_SCHEMA_VERSION);
    println!("  --iso-neff            Scale phi to match zoo's N_eff (Kumar protocol)");
    println!("  --output-path PATH    Write JSON output to file (CI artifact)");
    println!("  --tost-delta VALUE    TOST equivalence bound in BPB units (default 0.01)");
    println!("  --corpus PATH         Use byte-level real corpus (e.g. tiny_shakespeare.txt)");
}

fn parse_str_flag(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    let json_mode = args.iter().any(|a| a == "--json");
    let output_path = parse_output_path(&args);
    let iso_neff_mode = args.iter().any(|a| a == "--iso-neff");
    let corpus_path = parse_str_flag(&args, "--corpus");
    let corpus = match corpus_path.as_deref() {
        Some(p) => CorpusKind::BytesFile(p.to_string()),
        None => CorpusKind::Synthetic,
    };

    if !json_mode {
        println!("# F2 head-to-head — Issue #1021 (sandbox prep, not Verdict)");
        println!("# Note: this binary runs the harness on a small synthetic corpus.");
        println!("# A Verdict run requires the full champion corpus + compute (see #1021 blockers).");
    }

    // Iso-N_eff target: scale phi to match zoo's N_eff (Kumar three-γ formula).
    // For zoo @ N=8192, P=8 vs phi P=1.58: N_phi ≈ 41,154 (~5.03× via real eff ratio).
    let phi_iso_target = if iso_neff_mode {
        Some(trios_trainer::race::multi_seed::iso_neff_target_n(8192, 8.0, 1.58))
    } else {
        None
    };

    let phi_arm = MultiSeedConfig {
        seeds: vec![42, 43, 44, 45, 46],
        train_ratio: 0.8,
        vocab_size: 64,
        d_model: 128,
        steps: 200,
        lr: 0.004,
        ladder_kind: LadderKind::PhiLadder,
        // 20% warmup — Continual-QAT schedule (arXiv:2502.11895).
        warmup_steps_unquantized: 40,
        spike_injection_steps: Vec::new(),
        iso_neff_n_target: phi_iso_target,
        corpus: corpus.clone(),
        paretoq_precision: None,
        disable_quantization: false,
        task_kind: trios_trainer::race::multi_seed::TaskKind::Counter,
        use_ffn: false,
        d_hidden: 64,
        label_smoothing: 0.0,
        weight_decay: 0.1,
        apply_rmsnorm: true,
        grad_clip_l2: Some(1.0),
        latent_clamp_max: Some(1.0),
        dropout_p: 0.1,
    };

    // Zoo stays at baseline N (no iso-N_eff override) — only phi scales to match its capacity.
    let zoo_arm = MultiSeedConfig {
        ladder_kind: LadderKind::FormatZoo,
        iso_neff_n_target: None,
        corpus: corpus.clone(),
        ..phi_arm.clone()
    };

    let phi_report = print_arm("Phi-ladder", &phi_arm, json_mode);
    let zoo_report = print_arm("Format-zoo", &zoo_arm, json_mode);

    if !json_mode {
        println!(
            "\n## Config fingerprints (FNV-1a 64)\n  PHI: 0x{:016x}\n  ZOO: 0x{:016x}",
            config_fingerprint(&phi_arm),
            config_fingerprint(&zoo_arm)
        );
    }

    let alpha = 0.01;
    let welch = verdict_welch(&phi_report, &zoo_report, alpha);
    // TOST equivalence bound: CLI override --tost-delta, else 0.01 BPB (Marchisio).
    let delta = parse_f64_flag(&args, "--tost-delta").unwrap_or(0.01);
    let tost = verdict_tost(&phi_report, &zoo_report, delta, alpha);
    let pareto_verdict = verdict_pareto_adjusted(&phi_report, &zoo_report);

    let phi_samples: Vec<f64> = phi_report.runs.iter().map(|r| r.val_bpb).collect();
    let zoo_samples: Vec<f64> = zoo_report.runs.iter().map(|r| r.val_bpb).collect();
    let (boot_lo, boot_hi) = bootstrap_t_ci_diff(&phi_samples, &zoo_samples, 0.95, 0xF2_BEEF);

    let pooled_sigma = ((phi_report.std_val_bpb.powi(2) + zoo_report.std_val_bpb.powi(2)) / 2.0).sqrt();
    let sigma_phi = phi_report.std_val_bpb;
    let sigma_zoo = zoo_report.std_val_bpb;
    let mde = welch_mde(pooled_sigma, alpha, phi_report.runs.len(), 0.8);
    let power_at_observed = welch_power(pooled_sigma, alpha, phi_report.runs.len(), welch.mean_diff.abs());
    // Loop 13: honest power with unequal variance (σ_phi ≠ σ_zoo).
    let power_at_observed_unequal = welch_power_unequal(
        sigma_phi,
        sigma_zoo,
        alpha,
        phi_report.runs.len(),
        zoo_report.runs.len(),
        welch.mean_diff.abs(),
    );

    let pareto_welch = verdict_pareto_welch(&phi_report, &zoo_report, alpha);
    let bayesian = bayesian_credible_diff(&phi_samples, &zoo_samples, 0.95, 0xBA1E5);
    // Loop 12 CC: aggregate_verdict tertiary uses Pareto-normalized Bayesian to match
    // primary (which is Pareto Welch on normalized samples). Direction-consistency fix.
    let bayesian_normalized =
        bayesian_credible_normalized(&phi_report, &zoo_report, 0.95, 0xBA1E5);
    let permutation = permutation_test(&phi_samples, &zoo_samples, alpha);
    let aggregated = aggregate_verdict(&welch, &permutation, &pareto_welch, &bayesian_normalized);
    let (bca_lo, bca_hi, bca_p) = paired_bca_bootstrap(&phi_samples, &zoo_samples, 0.95, 0xBCA51);

    // Bootstrap-t vs Bayesian disagreement detection (research recommendation).
    let bootstrap_bayes_disagree =
        (boot_lo - bayesian.credible_lo).abs() > 0.02 || (boot_hi - bayesian.credible_hi).abs() > 0.02;

    if json_mode || output_path.is_some() {
        let out = F2Output {
            schema_version: F2_SCHEMA_VERSION,
            phi: &phi_report,
            zoo: &zoo_report,
            welch: welch.clone(),
            tost: tost.clone(),
            pareto_verdict: pareto_verdict.clone(),
            pareto_welch: pareto_welch.clone(),
            bayesian: bayesian.clone(),
            permutation: permutation.clone(),
            aggregated: aggregated.clone(),
            bootstrap_ci_lo: boot_lo,
            bootstrap_ci_hi: boot_hi,
            bootstrap_bayes_disagree,
            pooled_sigma,
            sigma_phi,
            sigma_zoo,
            mde_at_power_08: mde,
            power_at_observed,
            power_at_observed_unequal,
        };
        let json = serde_json::to_string_pretty(&out).unwrap();
        if let Some(path) = &output_path {
            fs::write(path, &json).expect("write output failed");
            if !json_mode {
                println!("\n## Artifact written\n  {}", path);
            }
        }
        if json_mode {
            println!("{}", json);
            return;
        }
    }

    println!("\n## Welch superiority test (α = {:.3})", alpha);
    println!(
        "  μ_phi − μ_zoo = {:+.4}  t = {:.3}  df = {:.1}  p₂ = {:.3e}  Cohen's d = {:+.3}",
        welch.mean_diff, welch.t_statistic, welch.df, welch.p_value_two_sided, welch.cohens_d
    );
    println!("  SUPERIORITY VERDICT: {}", welch.verdict);

    println!("\n## TOST equivalence test (Δ = {:.3} BPB, α = {:.3})", delta, alpha);
    println!(
        "  p_L = {:.3e}  p_U = {:.3e}  p_TOST = {:.3e}",
        tost.p_lower, tost.p_upper, tost.p_tost
    );
    println!(
        "  EQUIVALENCE VERDICT: {}",
        if tost.equivalent { "EQUIVALENT (TIE → moat → [Risk])" } else { "NOT EQUIVALENT" }
    );

    println!("\n## Pareto comparison (Kumar et al. arXiv:2411.04330)");
    println!(
        "  PHI:  bpw_stored={:.2}  N_eff={:.0}  ({} params)",
        phi_report.pareto.bits_per_weight_stored,
        phi_report.pareto.n_eff,
        phi_report.pareto.n_params
    );
    println!(
        "  ZOO:  bpw_stored={:.2}  N_eff={:.0}  ({} params)",
        zoo_report.pareto.bits_per_weight_stored,
        zoo_report.pareto.n_eff,
        zoo_report.pareto.n_params
    );

    println!("\n## bpw-adjusted Pareto verdict (Frantar/Alistarh arXiv:2502.16440)");
    println!(
        "  eff_phi = {:.4}  eff_zoo = {:.4}",
        pareto_verdict.eff_phi, pareto_verdict.eff_zoo
    );
    println!(
        "  BPB_norm: phi = {:.4}  zoo = {:.4}  diff = {:+.4}",
        pareto_verdict.bpb_norm_phi, pareto_verdict.bpb_norm_zoo, pareto_verdict.diff_norm
    );
    println!("  PARETO VERDICT (scalar): {}", pareto_verdict.verdict);

    println!("\n## Variance-aware Pareto Welch (per-seed normalized BPB)");
    println!(
        "  mean_diff_norm = {:+.4}  t = {:.3}  df = {:.1}  p₂ = {:.3e}",
        pareto_welch.mean_diff_norm,
        pareto_welch.t_statistic,
        pareto_welch.df,
        pareto_welch.p_value_two_sided
    );
    println!("  PARETO WELCH VERDICT: {}", pareto_welch.verdict);

    println!("\n## Bayesian credible interval (NIG, Jeffreys prior, 95%)");
    println!(
        "  posterior_diff = {:+.4}  CrI = [{:+.4}, {:+.4}]  P(phi<zoo) = {:.3}",
        bayesian.mean_diff_posterior,
        bayesian.credible_lo,
        bayesian.credible_hi,
        bayesian.probability_phi_lower
    );

    println!("\n## Permutation test (Fisher-Pitman exact, nonparametric)");
    println!(
        "  observed |Δ| = {:.4}  n_permutations = {}  p = {:.4}",
        permutation.observed_diff_abs, permutation.n_permutations, permutation.p_value
    );
    println!("  PERMUTATION VERDICT: {}", permutation.verdict);

    println!("\n## Bootstrap-t 95% CI on mean_diff (B=1999, N≤10 recommended)");
    println!("  CI = [{:+.4}, {:+.4}]", boot_lo, boot_hi);

    println!("\n## Welch power analysis (small-N audit)");
    println!(
        "  σ_phi = {:.4}  σ_zoo = {:.4}  σ_pooled = {:.4}",
        sigma_phi, sigma_zoo, pooled_sigma
    );
    println!(
        "  MDE @ power=0.8 (equal σ) = {:.4}",
        mde
    );
    println!(
        "  power @ observed δ:  pooled = {:.3}   unequal σ = {:.3}",
        power_at_observed, power_at_observed_unequal
    );
    let matrix = power_matrix(pooled_sigma, alpha, &[0.01, 0.02, 0.05, 0.10], 0.8);
    println!("  required N (power=0.8, α=0.01):");
    for (delta, n) in matrix {
        println!("    δ={:.2} → N≥{}", delta, n);
    }

    println!("\n## Paired BCa bootstrap on per-seed Δᵢ = phi_i − zoo_i (arXiv:2511.19794)");
    println!(
        "  CI = [{:+.4}, {:+.4}]  p_value ≈ {:.3e}",
        bca_lo, bca_hi, bca_p
    );

    println!("\n## Aggregated F2 verdict (hierarchical decision tree)");
    println!("  primary (Pareto Welch):    {}", aggregated.primary_pareto_welch);
    println!("  secondary Welch:           {}", aggregated.secondary_welch);
    println!("  secondary Permutation:     {}", aggregated.secondary_permutation);
    println!("  tertiary P(phi<zoo):       {:.3}", aggregated.tertiary_bayes_phi_lower);
    println!("  secondary_agrees:          {}", aggregated.secondary_agrees_with_primary);
    println!("  bootstrap_bayes_disagree:  {}", bootstrap_bayes_disagree);
    println!("  ── AGGREGATED ──  {}", aggregated.verdict);
    println!("  rationale: {}", aggregated.rationale);

    println!("\n## Verdict gate notes");
    println!("  Per Issue #1021: any single sandbox run cannot promote moat to [Verified].");
    println!("  Demotion rule: TOST EQUIVALENT → moat → [Risk] (this is the correct test).");
    println!("  Welch superiority answers a different question (who wins on average).");
    println!("  Real Verdict requires: full BPB pipeline, champion corpus, N≥5 seeds at champion scale.");
}
