//! # IGLA RACE — Coq-Proven Invariants (L-R14)

pub const PHI: f64 = 1.618033988749895;
pub const PHI_SQ: f64 = 2.618033988749895;
pub const PHI_CUBE: f64 = 4.23606797749979;
pub const PHI_INV6: f64 = 0.05572809000085359;
pub const INV3_ERROR_BOUND: f64 = PHI_INV6;
pub const TRINITY_IDENTITY: f64 = 3.0;
/// φ⁻³/2 ≈ 0.11803.  Used only as an algebraic trace in ema.rs.
/// For the real φ⁻³ ≈ 0.236068 use `PHI_INV3`.
pub const PHI_INV3_HALF: f64 = 0.11803398874989485;
/// φ⁻³ = 1/φ³ ≈ 0.2360679774997897 — the honest weight-decay anchor.
pub const PHI_INV3: f64 = 0.2360679774997897;
pub const LR_CHAMPION: f64 = 0.004;
pub const LR_SAFE_MIN: f64 = 0.002;
pub const LR_SAFE_MAX: f64 = 0.007;
pub const ASHA_PRUNE_THRESHOLD: f64 = 3.5;
/// RETRACTED as a measurement (2026-08-02).
///
/// `docs/audit/HONEST_FINDINGS.md:41` lists 2.5193 as a "stale placeholder".
/// No checkpoint artifact exists behind this number: it pre-dates
/// `checkpoint::save` doing any work (the function was a stub returning
/// `Ok(())`), so no weights were ever written for the run it claims to
/// describe, and it must not be cited as a result.
///
/// The value is retained only as an internal upper bound: it is asserted
/// against `ASHA_PRUNE_THRESHOLD` in the test below and printed by other
/// binaries as a comparison baseline. Changing it would break those call
/// sites without making any claim more honest, so it stays put and is
/// documented as retracted instead.
///
/// The only checkpoint-backed figure in this repo is raw val_bpb 2.6169 at
/// 12 000 steps (seed 47, h=384, 2 attention layers), sidecar
/// `checkpoints/igla-honest-provenance/12000.json`.
pub const BPB_CHAMPION: f64 = 2.5193;
/// Lowest bits-per-byte this architecture can honestly report.
///
/// Distinct from `race::victory::JEPA_PROXY_BPB_FLOOR` (0.1), which is a
/// DETECTOR for the constant-proxy artefact and is deliberately far below any
/// real reading. That floor was doing double duty as the publication gate, and
/// 0.1 is 26x below the band the measured calibration implies, so it admitted
/// every retracted number this audit exists to remove.
///
/// Derivation, from the calibration quoted in `train_loop::guard_bpb` and
/// measured on the verified byte-disjoint tinyshakespeare split (h=384, two
/// allocated attention blocks, one effective; 196,608 effective params of
/// 212,992 serialized, the layer-2 block being allocated and provably frozen
/// -- see the test `run_single_emits_a_loadable_artifact_and_freezes_layer_two`
/// in `src/train_loop.rs`; 128-symbol byte vocabulary):
///
///   * ~7.00 at init (= log2(128), the uniform-model ceiling),
///   * ~3.31 at step 1000,
///   * 2.59-2.64 at step 12000. An earlier revision of this comment said
///     "2.75-2.83", which the repository's own headline record falsifies: the
///     artifact-backed run reads `final_val_bpb` 2.6347548961639404 at step
///     12000 (README.md "why three figures", checkpoint sha256 `8a86fe69...`,
///     852 272 B), the archived sidecars read 2.616914749145508
///     (`checkpoints/igla-honest-provenance/12000.json`) and 2.614117383956909
///     (`checkpoints/igla-honest-20260802/12000.json`), and the four-cadence
///     table in README.md spans 2.5911 to 2.6348. The spread is the eval-cadence
///     effect, not noise: `train_loop` gates the in-place `gf16_floor` on
///     `step % eval_every == 0` after 70% of training, so runs differing only in
///     `--eval-every` end with different weights. Cite any of these only with
///     its cadence.
///   * ~2.71 even with a 100% verbatim train/val overlap - a total leak buys
///     only ~0.12 bpb on this architecture,
///   * `BPB_CHAMPION` = 2.5193, the crate's own best CLAIMED figure - retracted
///     as a measurement (see its doc comment) and named here only because the
///     floor must not refuse it.
///
/// A ~200K-parameter byte-level model therefore cannot honestly go under about
/// 2.5. 2.0 is a deliberately generous hard floor: it still admits 2.5193 with
/// room to spare, while refusing 1.5492 (the uncitable "honest Gate-2 pass" of
/// `LEAK_INVESTIGATION.md`, better than the champion at a commit where nothing
/// in the training loop sampled BPB at all) and 1.038. Anything under it is a
/// degenerate or duplicated eval corpus, an out-of-repo parser, or a sentinel -
/// never a measurement of held-out text.
///
/// What this floor does NOT do: it is a plausibility bound on the VALUE, so it
/// admits the 2.2393 / 2.2111 / 2.1919 family that `RETRACTION.md` grades
/// uncitable. Those are refused on PROVENANCE - no artifact, no corpus digest,
/// forbidden seed 43 - and no numeric threshold can express that.
///
/// Raise this only with new measured evidence. Never lower it to make a
/// fixture pass.
pub const PUBLISHED_BPB_FLOOR: f32 = 2.0;
pub const ASHA_RUNGS: [u64; 4] = [1_000, 3_000, 9_000, 27_000];
pub const MAX_ASHA_TRIALS: usize = 1_000;
pub const GF16_SAFE_D_MODEL: usize = 256;
pub const LUCAS: [u64; 7] = [2, 1, 3, 4, 7, 11, 18];
pub const NCA_ENTROPY_LO: f64 = PHI;
pub const NCA_ENTROPY_HI: f64 = PHI_SQ;
pub const NCA_ENTROPY_WIDTH: f64 = 1.0;
pub const NCA_GRID_SIZE: usize = 81;
pub const IGLA_TARGET_BPB: f64 = 1.5;
pub const BPB_VICTORY_TARGET: f64 = IGLA_TARGET_BPB;
pub const VICTORY_SEED_TARGET: u32 = 3;
pub const INV2_WARMUP_BLIND_STEPS: u64 = 4000;
pub const INV1_LR_SAFE_LO: f64 = LR_SAFE_MIN;
pub const INV1_LR_SAFE_HI: f64 = LR_SAFE_MAX;
pub const INV1_CHAMPION_LR: f64 = LR_CHAMPION;
pub const INV2_BPB_PRUNE_THRESHOLD: f64 = ASHA_PRUNE_THRESHOLD;
pub const INV3_D_MODEL_MIN: usize = GF16_SAFE_D_MODEL;
pub const INV4_NCA_GRID: usize = NCA_GRID_SIZE;
pub const INV4_NCA_K_STATES: usize = 9;
pub const INV4_ENTROPY_CERTIFIED_LO: f64 = NCA_ENTROPY_LO;
pub const INV4_ENTROPY_CERTIFIED_HI: f64 = NCA_ENTROPY_HI;
pub const INV4_ENTROPY_EMPIRICAL_LO: f64 = 1.5;
pub const INV4_ENTROPY_EMPIRICAL_HI: f64 = 2.8;

#[derive(Debug, Clone, PartialEq)]
pub enum GradientMode {
    RealMSE,
    ConstantProxy(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub enum InvError {
    Inv1BadGradient,
    Inv1LrOutOfBand(f64),
    Inv2ThresholdTooLow(f64),
    Inv3UnsafeDomain(usize),
    Inv4GridMismatch { grid: usize, k: usize },
    Inv5LucasClosureBroken,
}

impl std::fmt::Display for InvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InvError::Inv1BadGradient => write!(f, "INV-1 VIOLATED: gradient=ConstantProxy"),
            InvError::Inv1LrOutOfBand(lr) => write!(
                f,
                "INV-1 VIOLATED: lr={lr} outside φ-safe [{INV1_LR_SAFE_LO}, {INV1_LR_SAFE_HI}]"
            ),
            InvError::Inv2ThresholdTooLow(t) => write!(f, "INV-2 VIOLATED: threshold={t} < 3.5"),
            InvError::Inv3UnsafeDomain(d) => {
                write!(f, "INV-3 VIOLATED: GF16 with d_model={d} < 256")
            }
            InvError::Inv4GridMismatch { grid, k } => {
                write!(f, "INV-4 VIOLATED: NCA grid={grid} K={k}, expected 81/9")
            }
            InvError::Inv5LucasClosureBroken => {
                write!(f, "INV-5 VIOLATED: GF16 Lucas closure broken")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct InvTrialConfig {
    pub lr: f64,
    pub d_model: usize,
    pub bpb_prune_threshold: f64,
    pub warmup_blind_steps: u64,
    pub use_gf16: bool,
    pub nca_grid: usize,
    pub nca_k_states: usize,
    pub grad_mode: GradientMode,
    pub current_step: u64,
    pub last_bpb: f64,
}

pub fn validate_inv_config(cfg: &InvTrialConfig) -> Result<(), InvError> {
    match cfg.grad_mode {
        GradientMode::RealMSE => {}
        GradientMode::ConstantProxy(_) => return Err(InvError::Inv1BadGradient),
    }
    if !(INV1_LR_SAFE_LO..=INV1_LR_SAFE_HI).contains(&cfg.lr) {
        return Err(InvError::Inv1LrOutOfBand(cfg.lr));
    }
    if cfg.bpb_prune_threshold < INV2_BPB_PRUNE_THRESHOLD {
        return Err(InvError::Inv2ThresholdTooLow(cfg.bpb_prune_threshold));
    }
    if cfg.use_gf16 && cfg.d_model < INV3_D_MODEL_MIN {
        return Err(InvError::Inv3UnsafeDomain(cfg.d_model));
    }
    if cfg.nca_grid > 0 {
        if cfg.nca_grid != INV4_NCA_GRID || cfg.nca_k_states != INV4_NCA_K_STATES {
            return Err(InvError::Inv4GridMismatch {
                grid: cfg.nca_grid,
                k: cfg.nca_k_states,
            });
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct TrialConfig {
    pub lr: f64,
    pub d_model: usize,
    pub seed: u64,
    pub steps: u64,
    pub nca_weight: f32,
    pub jepa_weight: f32,
    pub ntp_weight: f32,
    pub use_gf16: bool,
}

pub fn validate_config(cfg: &TrialConfig) {
    assert!(
        cfg.lr >= LR_SAFE_MIN && cfg.lr <= LR_SAFE_MAX,
        "INV-1 VIOLATION: lr={} not in [{}, {}]",
        cfg.lr,
        LR_SAFE_MIN,
        LR_SAFE_MAX
    );
    assert!(cfg.lr > 0.0, "INV-1: lr must be positive, got {}", cfg.lr);
    if cfg.use_gf16 {
        assert!(
            cfg.d_model >= GF16_SAFE_D_MODEL,
            "INV-3 VIOLATION: GF16 requires d_model≥{}, got {}",
            GF16_SAFE_D_MODEL,
            cfg.d_model
        );
    }
    assert!(cfg.d_model > 0, "INV-3: d_model must be positive");
    assert!(
        cfg.nca_weight >= 0.0 && cfg.nca_weight <= 1.0,
        "INV-4 VIOLATION: nca_weight={} not in [0,1]",
        cfg.nca_weight
    );
    assert!(cfg.ntp_weight > 0.0, "INV-1: ntp_weight must be positive");
    assert!(
        cfg.steps > 0 && cfg.steps <= ASHA_RUNGS[3] * 2,
        "INV-2 VIOLATION: steps={} out of ASHA bounds [1, {}]",
        cfg.steps,
        ASHA_RUNGS[3] * 2
    );
}

pub fn validate_bpb(bpb: f64, trial_id: &str) {
    assert!(
        bpb > 0.0 && bpb < 20.0,
        "L-METRIC VIOLATION: BPB={:.4} out of range (0, 20) for trial {}",
        bpb,
        trial_id
    );
    assert!(
        bpb > 0.1,
        "L-METRIC VIOLATION: BPB={:.4} suspiciously low for trial {}",
        bpb,
        trial_id
    );
}

pub fn validate_nca_entropy(entropy: f64) {
    assert!(
        entropy >= NCA_ENTROPY_LO,
        "INV-4 VIOLATION: entropy={:.4} < φ={:.4}",
        entropy,
        NCA_ENTROPY_LO
    );
    assert!(
        entropy <= NCA_ENTROPY_HI,
        "INV-4 VIOLATION: entropy={:.4} > φ²={:.4}",
        entropy,
        NCA_ENTROPY_HI
    );
    assert!(
        (NCA_ENTROPY_HI - NCA_ENTROPY_LO - NCA_ENTROPY_WIDTH).abs() < 1e-10,
        "INV-4 INTERNAL: band width != 1 — Trinity constants corrupted!"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_trinity_identity() {
        let phi_inv = 1.0 / PHI;
        let result = PHI * PHI + phi_inv * phi_inv;
        assert!(
            (result - TRINITY_IDENTITY).abs() < 1e-10,
            "φ²+φ⁻²={result:.10} ≠ 3"
        );
    }
    #[test]
    fn test_phi_sq_eq_phi_plus_one() {
        assert!((PHI_SQ - (PHI + 1.0)).abs() < 1e-10);
    }
    #[test]
    fn test_entropy_band_width_exact() {
        assert!((NCA_ENTROPY_HI - NCA_ENTROPY_LO - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_champion_bpb_survives_asha() {
        assert!(BPB_CHAMPION < ASHA_PRUNE_THRESHOLD);
    }
    #[test]
    fn test_lr_champion_in_safe_range() {
        assert!(LR_CHAMPION >= LR_SAFE_MIN);
        assert!(LR_CHAMPION <= LR_SAFE_MAX);
    }
    #[test]
    fn test_gf16_precision_floor() {
        let phi_inv6 = (1.0_f64 / PHI).powi(6);
        assert!((phi_inv6 - PHI_INV6).abs() < 1e-8);
    }
    #[test]
    fn test_phi_inv3_half_is_half_of_phi_inv3() {
        assert!(
            (PHI_INV3_HALF - PHI_INV3 / 2.0).abs() < 1e-12,
            "PHI_INV3_HALF must be exactly PHI_INV3/2"
        );
    }
    #[test]
    fn test_phi_inv3_matches_anchor() {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let derived = 1.0 / (phi * phi * phi);
        assert!(
            (PHI_INV3 - derived).abs() < 1e-12,
            "PHI_INV3 drifted from 1/φ³"
        );
        assert!(
            (PHI_INV3 - 0.23607_f64).abs() < 0.001,
            "PHI_INV3 must be ≈0.23607 (was ~0.118 in old buggy comment)"
        );
    }
    #[test]
    fn test_validate_config_champion() {
        validate_config(&TrialConfig {
            lr: LR_CHAMPION,
            d_model: 384,
            seed: 43,
            steps: 27_000,
            nca_weight: 0.25,
            jepa_weight: 1.0,
            ntp_weight: 1.0,
            use_gf16: false,
        });
    }
    #[test]
    fn test_validate_config_gf16_guard() {
        let r = std::panic::catch_unwind(|| {
            validate_config(&TrialConfig {
                lr: 0.004,
                d_model: 128,
                seed: 42,
                steps: 3_000,
                nca_weight: 0.25,
                jepa_weight: 1.0,
                ntp_weight: 1.0,
                use_gf16: true,
            })
        });
        assert!(r.is_err());
    }
    #[test]
    fn test_validate_bpb_catches_jepa_proxy() {
        let r = std::panic::catch_unwind(|| validate_bpb(0.014, "J-002"));
        assert!(r.is_err());
    }
    /// The publication floor and the JEPA-proxy detector floor are two
    /// different numbers with two different jobs, and the ordering between
    /// them, `BPB_CHAMPION` and the retracted figures is the whole claim.
    #[test]
    fn published_floor_sits_between_the_detector_and_the_champion() {
        let published = PUBLISHED_BPB_FLOOR as f64;
        assert!(
            published > crate::race::victory::JEPA_PROXY_BPB_FLOOR,
            "the publication floor must be far above the proxy DETECTOR floor"
        );
        assert!(
            published < BPB_CHAMPION,
            "the floor must admit the crate's own best claimed figure"
        );
        // The two retracted numbers this floor CAN refuse, named so the build
        // refuses them.
        assert!(1.5492 < published, "LEAK_INVESTIGATION.md 'Gate-2 pass'");
        assert!(1.038 < published, "the second sub-champion outlier");
        // The honest calibration band must stay above the floor end to end:
        // init, step 1000, and the three artifact-backed step-12000 readings.
        for measured in [7.00, 3.31, 2.6348, 2.6169, 2.6141] {
            assert!(
                measured > published,
                "floor {published} would refuse the measured reading {measured}"
            );
        }
        // And the boundary, stated rather than implied: the 2.2x family
        // `RETRACTION.md` grades uncitable clears this floor comfortably. It is
        // refused on provenance, elsewhere, and pretending otherwise here would
        // be the same mistake this floor exists to correct.
        for retracted_by_provenance in [2.2393, 2.2111, 2.1919] {
            assert!(
                retracted_by_provenance > published,
                "documents that the floor does NOT catch {retracted_by_provenance}"
            );
        }
    }
    #[test]
    fn test_lucas_sequence() {
        assert_eq!(LUCAS[2], 3);
        assert_eq!(LUCAS[4], 7);
        assert_eq!(LUCAS[6], 18);
        for i in 2..LUCAS.len() {
            assert_eq!(LUCAS[i], LUCAS[i - 1] + LUCAS[i - 2]);
        }
    }
}
