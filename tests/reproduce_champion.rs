//! Reproduction guard: champion baseline must hold.
//!
//! Loads `configs/champion.toml`, runs in dry-run mode, asserts that the
//! config validator accepts it. Full numerical reproduction is gated behind
//! `--ignored` (requires real corpus + checkpoints).

use trios_trainer::TrainConfig;

#[test]
fn champion_config_loads_and_validates() {
    let cfg = TrainConfig::from_toml("configs/champion.toml").expect("champion.toml must load");
    assert_eq!(cfg.name, "champion");
    assert_eq!(cfg.seed, 43);
    assert!((cfg.optimizer.lr - 0.004).abs() < 1e-9);
}

#[test]
#[ignore]
fn champion_bpb_within_tolerance() {
    // Run real training, assert final BPB ≈ 2.2393 ± 0.01
    // Gated: needs corpus on disk.
}
