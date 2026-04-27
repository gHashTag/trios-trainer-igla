//! Champion reproduction guard (#320 L-T1 falsification).
//!
//! Loads champion.toml, validates config, then (behind --ignored) runs
//! the full 27K-step hybrid training and asserts final BPB ∈ [2.229, 2.249].
//!
//! Reference: gHashTag/trios@2446855 — BPB=2.2393 @ 27K steps, seed=43.

use trios_trainer::train_loop::{self, DEFAULT_IGLA_TARGET_BPB};
use trios_trainer::TrainConfig;

#[test]
fn champion_config_loads_and_validates() {
    let cfg = TrainConfig::from_toml("configs/champion.toml")
        .expect("champion.toml must load and validate");
    assert_eq!(cfg.name, "champion");
    assert_eq!(cfg.seed, 43);
    assert_eq!(cfg.steps, 27_000);
    assert!(
        (cfg.optimizer.lr - 0.004).abs() < 1e-9,
        "INV-8: lr must be 0.004"
    );
    assert!((cfg.target_bpb - 1.50).abs() < 1e-9);
    assert!(cfg.champion_bpb.is_some());
    let champ = cfg.champion_bpb.unwrap();
    assert!((champ - 2.2393).abs() < 1e-9, "champion_bpb must be 2.2393");
}

#[test]
fn champion_model_config_matches_spec() {
    let cfg = TrainConfig::from_toml("configs/champion.toml").unwrap();
    assert_eq!(cfg.model.d_model, 256);
    assert_eq!(cfg.model.n_layers, 2);
    assert_eq!(cfg.model.n_heads, 4);
    assert_eq!(cfg.model.vocab_size, 32_000);
    assert_eq!(cfg.model.seq_len, 1024);
    assert!(
        !cfg.model.hybrid_attn,
        "champion uses plain causal attention"
    );
}

#[test]
fn champion_optimizer_is_adamw_phi() {
    let cfg = TrainConfig::from_toml("configs/champion.toml").unwrap();
    assert_eq!(cfg.optimizer.kind, "adamw");
    assert!((cfg.optimizer.lr - 0.004).abs() < 1e-9);
    assert!((cfg.optimizer.beta1 - 0.9).abs() < 1e-9);
    assert!((cfg.optimizer.beta2 - 0.95).abs() < 1e-9);
    assert!((cfg.optimizer.weight_decay - 0.04).abs() < 1e-9);
    assert_eq!(cfg.optimizer.schedule, "phi");
    assert_eq!(cfg.optimizer.warmup_steps, 500);
}

#[test]
fn champion_objective_pure_ce() {
    let cfg = TrainConfig::from_toml("configs/champion.toml").unwrap();
    assert!((cfg.objective.w_ce - 1.0).abs() < 1e-9);
    assert!((cfg.objective.w_jepa - 0.0).abs() < 1e-9);
    assert!((cfg.objective.w_nca - 0.0).abs() < 1e-9);
}

#[test]
fn target_bpb_below_igla_gate() {
    let cfg = TrainConfig::from_toml("configs/champion.toml").unwrap();
    assert!(
        cfg.target_bpb < DEFAULT_IGLA_TARGET_BPB,
        "target_bpb ({}) must be below IGLA gate ({})",
        cfg.target_bpb,
        DEFAULT_IGLA_TARGET_BPB
    );
}

#[test]
fn champion_inv8_lr_in_phi_band() {
    let cfg = TrainConfig::from_toml("configs/champion.toml").unwrap();
    let lr = cfg.optimizer.lr;
    assert!(
        lr >= 0.001 && lr <= 0.01,
        "INV-8: lr={} not in φ-band [0.001, 0.01]",
        lr
    );
}

#[test]
#[ignore]
fn champion_bpb_reproduction_full_run() {
    let cfg = TrainConfig::from_toml("configs/champion.toml").expect("champion.toml must load");

    let outcome = train_loop::run(&cfg).expect("champion training must complete without error");

    assert!(
        outcome.final_bpb >= 2.229 && outcome.final_bpb <= 2.249,
        "champion BPB={:.4} outside tolerance [2.229, 2.249]. \
         Expected 2.2393 ± 0.01 per gHashTag/trios@2446855",
        outcome.final_bpb
    );

    assert_eq!(outcome.steps_done, cfg.steps, "must run full step count");
    assert!(outcome.final_bpb.is_finite(), "BPB must be finite");
}
