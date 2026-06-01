//! Integration test for F2 protocol harness — Issue #1021
//!
//! Sanity-checks the format_ladder + multi_seed integration end-to-end
//! at micro scale (small d_model, few steps). Does NOT validate verdict.

use trios_trainer::race::format_ladder::{ConversionCounter, LadderKind};
use trios_trainer::race::multi_seed::{run_multi_seed, verdict_welch, F2Verdict, MultiSeedConfig};

fn micro_config(kind: LadderKind) -> MultiSeedConfig {
    MultiSeedConfig {
        seeds: vec![1, 2, 3, 4, 5],
        train_ratio: 0.8,
        vocab_size: 32,
        d_model: 64,
        steps: 20,
        lr: 0.004,
        ladder_kind: kind,
        warmup_steps_unquantized: 0,
        spike_injection_steps: Vec::new(),
        iso_neff_n_target: None,
        corpus: trios_trainer::race::multi_seed::CorpusKind::Synthetic,
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
    }
}

#[test]
fn phi_ladder_arm_produces_finite_report() {
    let report = run_multi_seed(&micro_config(LadderKind::PhiLadder));
    assert_eq!(report.runs.len(), 5);
    assert!(report.mean_val_bpb.is_finite());
    assert!(report.mc_error >= 0.0);
    assert_eq!(report.ladder_kind, LadderKind::PhiLadder);
}

#[test]
fn format_zoo_arm_produces_finite_report() {
    let report = run_multi_seed(&micro_config(LadderKind::FormatZoo));
    assert_eq!(report.runs.len(), 5);
    assert!(report.mean_val_bpb.is_finite());
    assert_eq!(report.ladder_kind, LadderKind::FormatZoo);
}

#[test]
fn both_arms_finish_in_under_30_seconds() {
    let t0 = std::time::Instant::now();
    let _ = run_multi_seed(&micro_config(LadderKind::PhiLadder));
    let _ = run_multi_seed(&micro_config(LadderKind::FormatZoo));
    let elapsed = t0.elapsed().as_secs_f64();
    assert!(elapsed < 30.0, "F2 micro-harness too slow: {:.1}s", elapsed);
}

#[test]
fn conversion_counter_accumulates_across_arm() {
    let mut counter = ConversionCounter::new();
    for _ in 0..100 {
        let _ = counter.convert_f32_to_bf16(std::f32::consts::PI);
    }
    assert_eq!(counter.f32_to_bf16, 100);
    assert!(counter.lossy_total >= 100);
}

#[test]
fn arms_produce_divergent_lossy_counts() {
    // Both arms must record nonzero lossy conversions once dispatch is wired.
    let phi = run_multi_seed(&micro_config(LadderKind::PhiLadder));
    let zoo = run_multi_seed(&micro_config(LadderKind::FormatZoo));
    assert!(phi.total_lossy_conversions > 0, "phi-ladder must record lossy ops");
    assert!(zoo.total_lossy_conversions > 0, "format-zoo must record lossy ops");
}

#[test]
fn arms_produce_divergent_bpb() {
    // After wiring quantization, phi-ladder and zoo BPB must differ.
    // At micro-scale ternary saturates → phi BPB ≈ log2(vocab); zoo bf16 stays meaningful.
    let phi = run_multi_seed(&micro_config(LadderKind::PhiLadder));
    let zoo = run_multi_seed(&micro_config(LadderKind::FormatZoo));
    let gap = (phi.mean_val_bpb - zoo.mean_val_bpb).abs();
    let combined_mc = phi.mc_error + zoo.mc_error;
    assert!(
        gap > combined_mc,
        "arms must diverge beyond MC-error: gap={:.4} combined_mc={:.4}",
        gap, combined_mc
    );
}

#[test]
fn verdict_runs_and_returns_one_of_three_outcomes() {
    let phi = run_multi_seed(&micro_config(LadderKind::PhiLadder));
    let zoo = run_multi_seed(&micro_config(LadderKind::FormatZoo));
    let v = verdict_welch(&phi, &zoo, 0.01);
    assert!(matches!(
        v.verdict,
        F2Verdict::PhiWins | F2Verdict::Tie | F2Verdict::ZooWins
    ));
    assert!(v.p_value_two_sided >= 0.0 && v.p_value_two_sided <= 1.0);
}

#[test]
fn mc_error_is_std_over_sqrt_n() {
    let report = run_multi_seed(&micro_config(LadderKind::PhiLadder));
    let n = report.runs.len() as f64;
    let expected_mc = report.std_val_bpb / n.sqrt();
    assert!(
        (report.mc_error - expected_mc).abs() < 1e-9,
        "mc_error {} != std/sqrt(n) {}",
        report.mc_error,
        expected_mc
    );
}
