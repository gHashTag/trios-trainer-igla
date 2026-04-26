//! L-T1 falsification witness: champion reproduction guard.
//!
//! The hypothesis (Gate G1) is **falsified iff** PR-1 + PR-2 cannot reproduce
//! champion within `BPB = 2.2393 ± 0.01`.
//!
//! This test has two halves:
//! 1. A *smoke* assertion that always runs in CI — config loads, the champion
//!    baseline constants the validator reads are within the expected band.
//!    Fails fast if anyone tampers with `configs/champion.toml`.
//! 2. A *full* `#[ignore]`d numerical reproduction that runs the real 27K-step
//!    training pipeline and asserts `final_bpb ∈ [2.229, 2.249]`. Gated behind
//!    `--ignored` because it requires the corpus on disk and ~2 h CPU.
//!
//! Mission: TRAINER-IGLA-SOT (`phi^2 + phi^-2 = 3`).
//! Refs: gHashTag/trios-trainer-igla#3 #4 (L-T1).

use trios_trainer::TrainConfig;

const CHAMPION_BPB: f64 = 2.2393;
const CHAMPION_TOL: f64 = 0.01;
const CHAMPION_LR: f64 = 0.004;
const CHAMPION_SEED: u64 = 43;

#[test]
fn champion_smoke_constants_intact() {
    let cfg = TrainConfig::from_toml("configs/champion.toml")
        .expect("configs/champion.toml must load and validate");
    assert_eq!(cfg.name, "champion", "champion run name immutable");
    assert_eq!(
        cfg.seed, CHAMPION_SEED,
        "champion seed=43 immutable (anchors single-seed baseline)"
    );
    assert!(
        (cfg.optimizer.lr - CHAMPION_LR).abs() < 1e-9,
        "champion lr={} != INV-8 anchor {}",
        cfg.optimizer.lr,
        CHAMPION_LR
    );
    let champ = cfg
        .champion_bpb
        .expect("champion.toml must declare champion_bpb baseline");
    assert!(
        (champ - CHAMPION_BPB).abs() < CHAMPION_TOL,
        "champion_bpb={} drifted from baseline {} ± {}",
        champ,
        CHAMPION_BPB,
        CHAMPION_TOL
    );
}

#[test]
#[ignore = "needs full corpus + ~2h CPU; run with `cargo test --release --ignored champion_bpb_within_tolerance`"]
fn champion_bpb_within_tolerance() {
    // Witness for falsification code REGRESSION:
    // run the full champion config and assert final BPB ≈ 2.2393 ± 0.01.
    let cfg = TrainConfig::from_toml("configs/champion.toml").expect("champion.toml loads");
    let outcome = trios_trainer::run(&cfg).expect("champion run completes");
    let lo = CHAMPION_BPB - CHAMPION_TOL;
    let hi = CHAMPION_BPB + CHAMPION_TOL;
    assert!(
        outcome.final_bpb >= lo && outcome.final_bpb <= hi,
        "REGRESSION: final_bpb={} not in [{}, {}] — champion path is broken",
        outcome.final_bpb,
        lo,
        hi
    );
    assert!(
        outcome.steps_done >= 4000,
        "R8: champion run must reach >=4000 steps, got {}",
        outcome.steps_done
    );
}
