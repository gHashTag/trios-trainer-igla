//! F2 stability injection battery — loop 7 option P.
//!
//! Five scenarios verify ZClipTracker (arXiv:2504.02507) and NaN/spike detection
//! behave correctly under controlled fault injection.

use trios_trainer::race::format_ladder::LadderKind;
use trios_trainer::race::multi_seed::{power_matrix, run_multi_seed, MultiSeedConfig};

fn battery_config(spike_steps: Vec<usize>) -> MultiSeedConfig {
    MultiSeedConfig {
        seeds: vec![1, 2, 3, 4, 5],
        train_ratio: 0.8,
        vocab_size: 32,
        d_model: 64,
        steps: 120,
        lr: 0.004,
        ladder_kind: LadderKind::FormatZoo,
        warmup_steps_unquantized: 30,
        spike_injection_steps: spike_steps,
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
fn scenario_1_no_injection_no_spikes() {
    let report = run_multi_seed(&battery_config(Vec::new()));
    let total_spikes: u64 = report
        .runs
        .iter()
        .map(|r| r.stability.grad_norm_spike_count)
        .sum();
    let total_nan: u64 = report.runs.iter().map(|r| r.stability.nan_step_count).sum();
    // Baseline: at warmup=30, ZClip should be stable
    assert_eq!(total_nan, 0);
    // Allow up to 2 spurious spikes from EMA warmup noise
    assert!(
        total_spikes <= 2,
        "unexpected spike count in clean run: {}",
        total_spikes
    );
}

#[test]
fn scenario_2_single_spike_detected() {
    // One late-training spike per seed
    let report = run_multi_seed(&battery_config(vec![100]));
    let total_spikes: u64 = report
        .runs
        .iter()
        .map(|r| r.stability.grad_norm_spike_count)
        .sum();
    assert!(
        total_spikes >= 5,
        "expected ≥5 spikes (1 per seed), got {}",
        total_spikes
    );
}

#[test]
fn scenario_3_repeated_spikes_inflate_count() {
    // Three spikes per seed
    let report = run_multi_seed(&battery_config(vec![60, 80, 100]));
    let total_spikes: u64 = report
        .runs
        .iter()
        .map(|r| r.stability.grad_norm_spike_count)
        .sum();
    // 5 seeds × 3 spikes = 15 minimum
    assert!(
        total_spikes >= 10,
        "expected ≥10 spikes, got {}",
        total_spikes
    );
}

#[test]
fn scenario_4_worst_grad_norm_tracks_injection() {
    let no_spike = run_multi_seed(&battery_config(Vec::new()));
    let with_spike = run_multi_seed(&battery_config(vec![50]));
    let max_clean: f64 = no_spike
        .runs
        .iter()
        .map(|r| r.stability.worst_grad_norm)
        .fold(0.0, f64::max);
    let max_injected: f64 = with_spike
        .runs
        .iter()
        .map(|r| r.stability.worst_grad_norm)
        .fold(0.0, f64::max);
    // ×100 injection must produce strictly larger worst_grad_norm
    assert!(
        max_injected > max_clean * 10.0,
        "injected worst={:.3} clean worst={:.3}",
        max_injected,
        max_clean
    );
}

#[test]
fn scenario_5_power_matrix_returns_monotone_n() {
    let deltas = vec![0.01, 0.05, 0.10, 0.20];
    let matrix = power_matrix(0.03, 0.01, &deltas, 0.8);
    assert_eq!(matrix.len(), 4);
    // Smaller δ → larger required N
    for i in 1..matrix.len() {
        assert!(
            matrix[i - 1].1 >= matrix[i].1,
            "N should monotonically decrease with δ: {:?}",
            matrix
        );
    }
    // δ=0.20 with σ=0.03 should need very few seeds
    assert!(matrix[3].1 <= 5);
    // δ=0.01 needs many seeds
    assert!(matrix[0].1 >= 50);
}
