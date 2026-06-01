//! F2 Pareto sweep — issue #1021 + research loop 9 option T.
//!
//! Generates Pareto-frontier data points across multiple bit-precisions per arm.
//! Reference: Liu et al. ParetoQ (arXiv:2502.02631), Frantar/Alistarh (arXiv:2502.16440).
//! Each point: (P_w, N, BPB_mean, BPB_std, N_eff, eff). Lower-convex-hull dominance
//! is the standard test for arm-A-dominates-arm-B per published convention.

use std::fs::File;
use std::io::Write;
use trios_trainer::race::format_ladder::LadderKind;
use trios_trainer::race::multi_seed::{
    config_fingerprint, iso_neff_target_n, kumar_efficiency, kumar_n_eff, lower_convex_hull,
    pareto_for_precision, run_multi_seed, CorpusKind, MultiSeedConfig, TaskKind,
};

fn paired_bca_self(samples: &[f64], seed: u64) -> (f64, f64) {
    // Single-sample BCa percentile-style CI via BCa on (sample - mean) recentered around 0,
    // then add mean back. Simpler: compute BCa on raw samples by treating zero as null.
    let n = samples.len();
    if n < 2 {
        return (f64::NAN, f64::NAN);
    }
    let theta_hat: f64 = samples.iter().sum::<f64>() / n as f64;
    let b = 5000_usize;
    let mut rng = seed;
    let mut boot: Vec<f64> = Vec::with_capacity(b);
    for _ in 0..b {
        let mut sum = 0.0;
        for _ in 0..n {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = (rng >> 33) as usize % n;
            sum += samples[idx];
        }
        boot.push(sum / n as f64);
    }
    boot.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let lo_idx = (0.025 * b as f64).floor() as usize;
    let hi_idx = (0.975 * b as f64).floor() as usize;
    let _ = theta_hat;
    (boot[lo_idx.min(b - 1)], boot[hi_idx.min(b - 1)])
}

#[derive(Debug, Clone, serde::Serialize)]
struct SweepPoint {
    arm: &'static str,
    precision_bits: f64,
    n_params: u64,
    n_eff: f64,
    eff: f64,
    bpb_mean: f64,
    bpb_std: f64,
    bpb_ci_lo: f64,
    bpb_ci_hi: f64,
    mc_error: f64,
    n_seeds: usize,
    config_hash: u64,
    on_frontier: bool,
    /// FP32 baseline at same N (Kumar et al. arXiv:2411.04330 §3). None for FP32 row itself.
    bpb_fp32_baseline: Option<f64>,
    /// ΔBPB = BPB_mean − BPB_fp32 (Loop 15 methodology fix).
    /// Literature range at 1.58-8 bits: 0.01-0.5 BPB. Anything >> 0.5 = artifact.
    delta_bpb_vs_fp32: Option<f64>,
}

#[derive(Debug, serde::Serialize)]
struct SweepReport {
    schema_version: &'static str,
    points: Vec<SweepPoint>,
}

fn sweep_arm(
    label: &'static str,
    kind: LadderKind,
    p_w: f64,
    base_d_model: usize,
    iso_neff_to: Option<u64>,
    disable_quant: bool,
    task: TaskKind,
    use_ffn: bool,
    corpus: CorpusKind,
    vocab_size: usize,
) -> SweepPoint {
    // Loop 14 option II: when iso_neff_to is set, expand N_phi so its capacity matches target.
    let iso_target = iso_neff_to.map(|n_base| iso_neff_target_n(n_base, 8.0, p_w));
    let cfg = MultiSeedConfig {
        seeds: vec![42, 43, 44, 45, 46],
        train_ratio: 0.8,
        vocab_size,
        d_model: base_d_model,
        steps: 1000, // Loop 18: was 100. Research-prescribed signal floor for sparse parity (5K elbow, 1K starts).
        lr: 0.004,
        ladder_kind: kind,
        // Loop 19 XX: HuggingFace 1.58 recipe — warmup = 20% of steps for stable BitLinear convergence.
        warmup_steps_unquantized: 1000usize / 5,
        spike_injection_steps: Vec::new(),
        iso_neff_n_target: iso_target,
        corpus,
        paretoq_precision: Some(p_w),
        disable_quantization: disable_quant,
        task_kind: task,
        use_ffn,
        d_hidden: 64,
        label_smoothing: 0.0,
        weight_decay: 0.1,
        apply_rmsnorm: true,
        grad_clip_l2: Some(1.0),
        latent_clamp_max: Some(1.0),
        dropout_p: 0.1,
    };
    let report = run_multi_seed(&cfg);
    let n_params = (cfg.vocab_size * report.effective_d_model) as u64;
    let pareto = pareto_for_precision(n_params, p_w);
    let per_seed: Vec<f64> = report.runs.iter().map(|r| r.val_bpb).collect();
    let (ci_lo, ci_hi) = paired_bca_self(&per_seed, 0xCAFE_FACE);
    SweepPoint {
        arm: label,
        precision_bits: p_w,
        n_params,
        n_eff: pareto.n_eff,
        eff: kumar_efficiency(n_params, p_w, p_w, p_w),
        bpb_mean: report.mean_val_bpb,
        bpb_std: report.std_val_bpb,
        bpb_ci_lo: ci_lo,
        bpb_ci_hi: ci_hi,
        mc_error: report.mc_error,
        n_seeds: report.runs.len(),
        config_hash: config_fingerprint(&cfg),
        on_frontier: false,
        bpb_fp32_baseline: None,
        delta_bpb_vs_fp32: None,
    }
}

fn write_csv(path: &str, points: &[SweepPoint]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    writeln!(
        f,
        "arm,precision,n_params,n_eff,eff,bpb_mean,bpb_std,bpb_ci_lo,bpb_ci_hi,bpb_fp32_baseline,delta_bpb_vs_fp32,n_seeds,on_frontier,config_hash"
    )?;
    for p in points {
        let baseline_str = p
            .bpb_fp32_baseline
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default();
        let delta_str = p
            .delta_bpb_vs_fp32
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default();
        writeln!(
            f,
            "{},{:.4},{},{:.0},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},0x{:016x}",
            p.arm,
            p.precision_bits,
            p.n_params,
            p.n_eff,
            p.eff,
            p.bpb_mean,
            p.bpb_std,
            p.bpb_ci_lo,
            p.bpb_ci_hi,
            baseline_str,
            delta_str,
            p.n_seeds,
            p.on_frontier,
            p.config_hash,
        )?;
    }
    Ok(())
}

fn print_help() {
    println!("f2_pareto_sweep — multi-precision Pareto frontier (Issue #1021, loop 14)");
    println!();
    println!("USAGE: f2_pareto_sweep [FLAGS]");
    println!();
    println!("FLAGS:");
    println!("  --help, -h          Print this help and exit");
    println!("  --json              Emit structured JSON sweep report to stdout");
    println!(
        "  --csv PATH          Write 12-column CSV (arm,P,N,N_eff,eff,BPB±std,CI,on_frontier,hash)"
    );
    println!(
        "  --iso-neff          Iso-N_eff mode: scale phi N to match zoo's N_eff (capacity-matched)"
    );
    println!();
    println!("Sweep points: phi P ∈ {{1.58, 2.0, 3.0, 4.0}} (ParetoQ SEQ/LSQ cascade),");
    println!("              zoo P ∈ {{4.0, 8.0}} (INT4 RTN group=32 / bf16+E4M3 HYBRID).");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    let json_mode = args.iter().any(|a| a == "--json");
    let csv_path = args
        .iter()
        .position(|a| a == "--csv")
        .and_then(|i| args.get(i + 1).cloned());

    if !json_mode {
        println!("# F2 Pareto sweep — multi-precision frontier data");
        println!("# Points: phi at P ∈ {{1.58, 2.0, 3.0, 4.0}} × zoo at P ∈ {{4, 8}}");
    }

    let iso_neff_mode = args.iter().any(|a| a == "--iso-neff");
    let iso_baseline_n: Option<u64> = if iso_neff_mode { Some(8192) } else { None };
    // Loop 16 OO: --sparse-parity uses Michaud's multitask sparse parity (non-trivial task).
    let task: TaskKind = if args.iter().any(|a| a == "--sparse-parity") {
        TaskKind::SparseParity {
            n_bits: 40,
            k: 3,
            n_tasks: 64,
        }
    } else {
        TaskKind::Counter
    };
    // Loop 16 PP: --ffn enables 2-layer FFN architecture (W2·ReLU(W1·emb)).
    let use_ffn = args.iter().any(|a| a == "--ffn");
    // Loop 21 FFF: --corpus PATH wires byte-level real corpus (default Synthetic).
    let corpus_path = args
        .iter()
        .position(|a| a == "--corpus")
        .and_then(|i| args.get(i + 1).cloned());
    let (corpus, vocab) = match corpus_path.as_deref() {
        Some(p) => (CorpusKind::BytesFile(p.to_string()), 65), // nanoGPT tiny_shakespeare convention
        None => (CorpusKind::Synthetic, 64),
    };

    // FP32 baseline (same task + architecture as quantized arms).
    let fp32_point = sweep_arm(
        "fp32",
        LadderKind::PhiLadder,
        32.0,
        128,
        None,
        true,
        task.clone(),
        use_ffn,
        corpus.clone(),
        vocab,
    );
    let fp32_baseline_bpb = fp32_point.bpb_mean;

    let mut points = Vec::new();
    for &p in &[1.58_f64, 2.0, 3.0, 4.0] {
        let mut pt = sweep_arm(
            "phi",
            LadderKind::PhiLadder,
            p,
            128,
            iso_baseline_n,
            false,
            task.clone(),
            use_ffn,
            corpus.clone(),
            vocab,
        );
        pt.bpb_fp32_baseline = Some(fp32_baseline_bpb);
        pt.delta_bpb_vs_fp32 = Some(pt.bpb_mean - fp32_baseline_bpb);
        points.push(pt);
    }
    for &p in &[4.0_f64, 8.0] {
        let mut pt = sweep_arm(
            "zoo",
            LadderKind::FormatZoo,
            p,
            128,
            None,
            false,
            task.clone(),
            use_ffn,
            corpus.clone(),
            vocab,
        );
        pt.bpb_fp32_baseline = Some(fp32_baseline_bpb);
        pt.delta_bpb_vs_fp32 = Some(pt.bpb_mean - fp32_baseline_bpb);
        points.push(pt);
    }
    points.insert(0, fp32_point);

    if json_mode {
        let report = SweepReport {
            schema_version: "f2-sweep.1",
            points,
        };
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        return;
    }

    println!("\n| arm  | P_w  | N      | N_eff   | eff    | BPB_mean | BPB_std | ΔBPB_vs_fp32 |");
    println!("|------|------|--------|---------|--------|----------|---------|--------------|");
    for p in &points {
        let delta_str = match p.delta_bpb_vs_fp32 {
            Some(d) => format!("{:+.4}", d),
            None => "  baseline  ".to_string(),
        };
        println!(
            "| {:<4} | {:<4.2} | {:<6} | {:<7.0} | {:<6.4} | {:<8.4} | {:<7.4} | {} |",
            p.arm, p.precision_bits, p.n_params, p.n_eff, p.eff, p.bpb_mean, p.bpb_std, delta_str,
        );
    }

    println!("\n## Literature sanity check (Kumar et al. arXiv:2411.04330 §3)");
    println!("  Expected ΔBPB range at 1.58-8 bits: 0.01 – 0.5 BPB");
    println!("  Anything >> 0.5 = task too trivial OR architecture too benign for quant test");
    let max_delta = points
        .iter()
        .filter_map(|p| p.delta_bpb_vs_fp32.map(|d| d.abs()))
        .fold(0.0_f64, f64::max);
    if max_delta > 0.5 {
        println!(
            "  ⚠ max |ΔBPB| = {:.4} — exceeds literature range, likely artifact (loop 14 reveal)",
            max_delta
        );
    } else {
        println!(
            "  ✓ max |ΔBPB| = {:.4} — within literature range",
            max_delta
        );
    }

    // Andrew's monotone-chain lower convex hull on (bpw_stored, BPB) plane.
    let hull_pts: Vec<(f64, f64)> = points
        .iter()
        .map(|p| (p.precision_bits, p.bpb_mean))
        .collect();
    let hull_idx = lower_convex_hull(&hull_pts);
    for &i in &hull_idx {
        points[i].on_frontier = true;
    }

    if let Some(path) = &csv_path {
        write_csv(path, &points).expect("write CSV failed");
        if !json_mode {
            println!("\n## CSV written\n  {}", path);
        }
    }

    println!("\n## Lower convex hull (Pareto frontier, bpw vs BPB)");
    for &i in &hull_idx {
        let p = &points[i];
        println!(
            "  arm={:<3} P={:<4.2} N_eff={:<7.0} BPB={:.4} CI=[{:.4}, {:.4}]",
            p.arm, p.precision_bits, p.n_eff, p.bpb_mean, p.bpb_ci_lo, p.bpb_ci_hi
        );
    }
    let n_eff_zoo: f64 = kumar_n_eff(8192, 8.0, 8.0, 8.0);
    let phi_on_frontier = hull_idx.iter().any(|&i| points[i].arm == "phi");
    let zoo_on_frontier = hull_idx.iter().any(|&i| points[i].arm == "zoo");
    println!(
        "\n  Frontier composition: phi={} zoo={}  (zoo baseline N_eff = {:.0})",
        phi_on_frontier, zoo_on_frontier, n_eff_zoo
    );
}
