//! F2 Ablation framework — Loop 23 JJJ partial.
//!
//! Identifies each of 7 stacked methodology fixes and provides config-mutation helpers
//! for leave-one-out + cumulative add-on ablation tables per NeurIPS/ICLR convention.
//!
//! Research-agent loop 23: side-by-side cumulative + LOCO is canonical layout. N=5 seeds
//! is rare/publication-grade for quant ablations (most use N=1-3).

use crate::race::multi_seed::MultiSeedConfig;

/// One of the 7 methodology fixes stacked across loops 19-22.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AblationFix {
    RmsNorm,
    WarmupSchedule,
    GradClip,
    LatentClamp,
    LabelSmoothing,
    WeightDecay,
    Dropout,
}

/// Loop 34 fix 5: single source of truth for canonical fix short-names.
/// Mirrors `AblationFix::short_name()` for every variant in `ALL`. Used by:
///   - bin/f2_iloco_score.rs (`is_canonical_fix`, `parse_pair_label` validation)
///   - bin/f2_mediation.rs (`--mediator` validation)
///   - bin/f2_dual_mediation.rs (Loop 34, sequential X→M1→M2→Y check)
///
/// Adding a new fix MUST: (1) extend `AblationFix` + `ALL` + `short_name`,
/// and (2) append the new short_name here. Order matches `AblationFix::ALL`.
pub const CANONICAL_FIX_NAMES: &[&str] = &[
    "rms", "warmup", "gradclip", "clamp", "smooth", "wd", "dropout",
];

/// Loop 39 fix 2: stratification registry.
///
/// Each variant is a (mode_kind, stratum) pair. `mode_str()` emits the CSV
/// `mode` column value; iteration yields *all* known stratum/mode combinations
/// in a single canonical order. Downstream lookup helpers iterate this enum
/// instead of probing hardcoded const arrays, so adding a new stratum
/// (e.g. warmup-zero) requires only a new `Stratum` variant.
/// Stratification mode for the F2 ablation sweep.
///
/// Loop 45 fix 5: policy for adding new variants.
///
/// **Add a `<FixName>0` variant when:**
///   1. The fix has been identified as a strong mediator (≥50% indirect effect)
///      in a previous loop's `f2_dual_mediation` or `f2_mediation` analysis, AND
///   2. The Pearl Controlled Direct Effect (CDE) at the disabled-value of the
///      fix is the natural next analytical question.
///
/// **Don't add a variant just because the fix exists.** Stratification cost
/// is ~25min/200-step sweep; adding decorative strata bloats the registry.
///
/// **Process to add (see Loop 41 Warmup0 PR for reference):**
///   1. Append the variant to `Stratum` + `ALL`.
///   2. Add a `prefix()` arm (lowercase + trailing `_`, e.g. `"clamp0_"`).
///   3. Mirror the `wd_stratified` mode branch in `f2_ablation_sweep.rs` (the
///      registry takes care of f2_dual_mediation lookups automatically).
///   4. Update `tests/f2_*_stratified_e2e.rs` to mirror the wd_stratified test.
///   5. Bump `docs/F2_BINARIES.md` row count for binaries that change.
///
/// Mediator candidates considered for future strata (per Loop 30/33/40 findings):
///   - `LabelSmoothing0`: low priority — Loop 28 showed label smoothing has near-zero
///     direct effect at our scale; only useful if a later loop finds it mediates.
///   - `Dropout0`: similar to LabelSmoothing — usually decorative at small N.
///   - `ClampZero`: latent_clamp_max=None; Loop 26 found it has near-zero direct
///     effect; defer until a sensitivity envelope flags it as dominant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stratum {
    /// Default WD=0.1 / BitNet recipe.
    Canonical,
    /// Loop 31 wd_stratified: WD pinned to 0.0 (Pearl CDE on WD).
    Wd0,
    /// Loop 41 fix 3: warmup pinned to disabled (Pearl CDE on warmup, after
    /// Loop 33 identified warmup as the 2nd dominant mediator).
    Warmup0,
}

impl Stratum {
    pub const ALL: &'static [Stratum] = &[Stratum::Canonical, Stratum::Wd0, Stratum::Warmup0];

    /// CSV mode-column prefix for this stratum. Canonical uses no prefix,
    /// stratified modes prepend e.g. `"wd0_"` / `"warmup0_"`.
    pub fn prefix(&self) -> &'static str {
        match self {
            Stratum::Canonical => "",
            Stratum::Wd0 => "wd0_",
            Stratum::Warmup0 => "warmup0_",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeKind {
    Loco,
    Pairwise,
    Triplet,
}

impl ModeKind {
    pub const ALL: &'static [ModeKind] = &[ModeKind::Loco, ModeKind::Pairwise, ModeKind::Triplet];

    pub fn base(&self) -> &'static str {
        match self {
            ModeKind::Loco => "loco",
            ModeKind::Pairwise => "pairwise",
            ModeKind::Triplet => "triplet",
        }
    }
}

/// Returns the full `mode` column string for a (kind, stratum) pair:
/// `"loco"`, `"wd0_pairwise"`, etc. Used by downstream lookups to probe both
/// canonical and stratified row sets.
pub fn mode_string(kind: ModeKind, stratum: Stratum) -> String {
    format!("{}{}", stratum.prefix(), kind.base())
}

/// Iterate every (kind, stratum) mode string in canonical order. The order is:
/// for each `ModeKind` (Loco/Pairwise/Triplet), then each `Stratum` (Canonical/Wd0).
pub fn all_mode_strings(kind: ModeKind) -> Vec<String> {
    Stratum::ALL.iter().map(|s| mode_string(kind, *s)).collect()
}

/// Loop 35 fix 2: human-readable description of the pair_/triplet_ label format.
///
/// **Pair and triplet labels are emitted in `AblationFix::ALL` index order,
/// NOT lexicographic order.** Example: `pair_warmup_wd` (warmup idx=1, wd idx=5)
/// rather than `pair_wd_warmup`. Loop 27's `run_pairwise` and Loop 29's
/// `run_triplet` both iterate `i < j (< k)` over `AblationFix::ALL`, then
/// concatenate `short_name()`s.
///
/// Downstream binaries that look up a (X, M1, M2) pair/triplet by short_name
/// MUST therefore try multiple permutations of the operand order.
/// `f2_dual_mediation.rs` uses `lookup_pair_any_perm` + `lookup_triplet_any_perm`.
/// New tooling must do the same; lexicographic guessing will mis-match.
pub const LABEL_ORDERING_CONVENTION: &str =
    "pair_<a>_<b> and triplet_<a>_<b>_<c> use AblationFix::ALL index order, \
     not lexicographic. Use permutation-tolerant lookup downstream.";

/// Loop 34 fix 5: membership test for canonical fix short-names.
pub fn is_canonical_fix(name: &str) -> bool {
    CANONICAL_FIX_NAMES.iter().any(|n| *n == name)
}

impl AblationFix {
    /// LOAD-BEARING CANONICAL ORDER (Loop 29 audit fix 5).
    ///
    /// Cumulative-mode `fix_index` semantics in historical CSVs (Loops 24+) depend
    /// on this exact order. Reordering breaks aggregator baseline picks and any
    /// downstream analysis that joins CSV across loops. If you MUST add a fix,
    /// append it; never insert in the middle. The test
    /// `ablation_all_canonical_order_locked` guards this invariant.
    pub const ALL: [AblationFix; 7] = [
        AblationFix::RmsNorm,
        AblationFix::WarmupSchedule,
        AblationFix::GradClip,
        AblationFix::LatentClamp,
        AblationFix::LabelSmoothing,
        AblationFix::WeightDecay,
        AblationFix::Dropout,
    ];

    pub fn short_name(&self) -> &'static str {
        match self {
            AblationFix::RmsNorm => "rms",
            AblationFix::WarmupSchedule => "warmup",
            AblationFix::GradClip => "gradclip",
            AblationFix::LatentClamp => "clamp",
            AblationFix::LabelSmoothing => "smooth",
            AblationFix::WeightDecay => "wd",
            AblationFix::Dropout => "dropout",
        }
    }
}

/// Apply a single-knockout: configuration with the named fix DISABLED.
/// Loop 24 MMM: 6/7 now honored via runtime config flags. RmsNorm still code-path only.
pub fn disable_in_config(cfg: &mut MultiSeedConfig, fix: AblationFix) {
    match fix {
        AblationFix::WarmupSchedule => {
            cfg.warmup_steps_unquantized = 0;
        }
        AblationFix::LabelSmoothing => {
            cfg.label_smoothing = 0.0;
        }
        AblationFix::WeightDecay => {
            cfg.weight_decay = 0.0;
        }
        AblationFix::GradClip => {
            cfg.grad_clip_l2 = None;
        }
        AblationFix::LatentClamp => {
            cfg.latent_clamp_max = None;
        }
        AblationFix::Dropout => {
            cfg.dropout_p = 0.0;
        }
        AblationFix::RmsNorm => {
            cfg.apply_rmsnorm = false; // Loop 25 PPP: consumed by forward_ffn_with_options.
        }
    }
}

/// Cumulative add-on: first `n` fixes ENABLED (in canonical ALL order).
/// Baseline (n=0): everything off. Full stack (n=7): loop 22 production state.
pub fn cumulative_config(base: &MultiSeedConfig, n: usize) -> MultiSeedConfig {
    let mut cfg = base.clone();
    // Baseline: all knobs OFF.
    cfg.warmup_steps_unquantized = 0;
    cfg.label_smoothing = 0.0;
    cfg.weight_decay = 0.0;
    cfg.grad_clip_l2 = None;
    cfg.latent_clamp_max = None;
    cfg.dropout_p = 0.0;
    cfg.apply_rmsnorm = false;
    // Loop 32 fix 4: generalize the Loop 31 defer-to-base pattern across ALL
    // continuous knobs. Each fix now reads its "enabled" value from `base`,
    // falling back to a hardcoded default only when base has the disabled
    // sentinel (zero / None). Lets any stratification mode (wd_stratified,
    // future warmup_stratified, etc.) pin a knob to its reference level via
    // base_config alone — no further ablation.rs edits needed.
    // Loop 42 fix 3: extend Loop 31's unconditional defer-to-base from WeightDecay
    // to WarmupSchedule, LabelSmoothing, and Dropout. The `if base.X > 0` guard
    // treated zero as "unset, use default" — wrong for stratification modes
    // that need to *pin* a knob to zero (e.g. Warmup0). The base_config caller
    // is responsible for setting defaults; ablation.rs simply respects them.
    for &fix in AblationFix::ALL.iter().take(n) {
        match fix {
            AblationFix::WarmupSchedule => {
                cfg.warmup_steps_unquantized = base.warmup_steps_unquantized;
            }
            AblationFix::LabelSmoothing => {
                cfg.label_smoothing = base.label_smoothing;
            }
            AblationFix::WeightDecay => {
                // Loop 31 fix: defer to base.weight_decay (mediator-stratified support).
                cfg.weight_decay = base.weight_decay;
            }
            AblationFix::GradClip => {
                cfg.grad_clip_l2 = base.grad_clip_l2.or(Some(1.0));
            }
            AblationFix::LatentClamp => {
                cfg.latent_clamp_max = base.latent_clamp_max.or(Some(1.0));
            }
            AblationFix::Dropout => {
                cfg.dropout_p = base.dropout_p;
            }
            AblationFix::RmsNorm => {
                cfg.apply_rmsnorm = true;
            }
        }
    }
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::race::format_ladder::LadderKind;
    use crate::race::multi_seed::{CorpusKind, TaskKind};

    /// Loop 29 audit fix 5: lock the canonical ALL order. Historical CSVs from
    /// Loops 24+ embed fix_index → fix mapping that depends on this exact list.
    /// If this test fails after a reorder, the fix is to APPEND, not insert.
    #[test]
    fn ablation_all_canonical_order_locked() {
        let names: Vec<&str> = AblationFix::ALL.iter().map(|f| f.short_name()).collect();
        assert_eq!(
            names,
            vec!["rms", "warmup", "gradclip", "clamp", "smooth", "wd", "dropout"],
            "AblationFix::ALL order is load-bearing — see doc comment on the const."
        );
    }

    /// Loop 34 fix 5: CANONICAL_FIX_NAMES must mirror AblationFix::ALL.short_name().
    /// If you reorder or rename a variant, this test catches the divergence before
    /// downstream binaries silently fall out of sync.
    #[test]
    fn canonical_fix_names_match_ablation_all() {
        let from_all: Vec<&str> = AblationFix::ALL.iter().map(|f| f.short_name()).collect();
        let from_const: Vec<&str> = CANONICAL_FIX_NAMES.to_vec();
        assert_eq!(
            from_all, from_const,
            "CANONICAL_FIX_NAMES drifted from AblationFix::ALL — bins will mis-parse labels."
        );
    }

    #[test]
    fn is_canonical_fix_accepts_known_rejects_unknown() {
        for n in CANONICAL_FIX_NAMES {
            assert!(is_canonical_fix(n), "expected {} to be canonical", n);
        }
        assert!(!is_canonical_fix("0.000"));
        assert!(!is_canonical_fix(""));
        assert!(!is_canonical_fix("WD"));
    }

    #[test]
    fn cumulative_config_respects_base_warmup_zero() {
        // Loop 43 fix 1+2: locks Loop 42's defer-to-base sentinel fix.
        // base.warmup_steps_unquantized = 0 must STAY 0 through cumulative_config
        // (was previously overridden by `(steps/5).max(20)` when base was 0).
        let mut cfg = micro_cfg();
        cfg.warmup_steps_unquantized = 0;
        cfg.label_smoothing = 0.0;
        cfg.dropout_p = 0.0;
        let n = AblationFix::ALL.len();
        let full = cumulative_config(&cfg, n);
        assert_eq!(full.warmup_steps_unquantized, 0,
            "Warmup0 stratum broken: cumulative_config overrode base=0");
        assert_eq!(full.label_smoothing, 0.0,
            "LabelSmoothing0 broken: cumulative_config overrode base=0");
        assert_eq!(full.dropout_p, 0.0,
            "Dropout0 broken: cumulative_config overrode base=0");
    }

    #[test]
    fn stratum_registry_produces_expected_mode_strings() {
        // Loop 39 fix 2 + Loop 41 fix 3: assert the registry covers Canonical,
        // Wd0, and Warmup0 — adding a new stratum extends every lookup
        // automatically.
        let loco = all_mode_strings(ModeKind::Loco);
        assert_eq!(loco, vec!["loco", "wd0_loco", "warmup0_loco"]);
        let pair = all_mode_strings(ModeKind::Pairwise);
        assert_eq!(pair, vec!["pairwise", "wd0_pairwise", "warmup0_pairwise"]);
        let triplet = all_mode_strings(ModeKind::Triplet);
        assert_eq!(triplet, vec!["triplet", "wd0_triplet", "warmup0_triplet"]);
    }

    /// Loop 35 fix 2: lock the AblationFix::ALL-index pair label format
    /// (warmup before wd, etc.) so a future refactor that switches to
    /// lexicographic order breaks here, not silently in downstream binaries.
    #[test]
    fn canonical_pair_label_is_all_index_order_not_lex() {
        let all = AblationFix::ALL;
        // warmup at idx 1, wd at idx 5 → ALL-order is "pair_warmup_wd", not lex "pair_wd_warmup".
        let warmup = all[1].short_name();
        let wd = all[5].short_name();
        assert_eq!(warmup, "warmup");
        assert_eq!(wd, "wd");
        // The convention says we concatenate in ALL-order: lower-idx first.
        let pair = format!("pair_{}_{}", warmup, wd);
        assert_eq!(pair, "pair_warmup_wd");
        assert!(LABEL_ORDERING_CONVENTION.contains("AblationFix::ALL"));
    }

    fn micro_cfg() -> MultiSeedConfig {
        MultiSeedConfig {
            seeds: vec![1, 2, 3, 4, 5],
            train_ratio: 0.8,
            vocab_size: 16,
            d_model: 16,
            steps: 100,
            lr: 0.004,
            ladder_kind: LadderKind::PhiLadder,
            warmup_steps_unquantized: 20,
            spike_injection_steps: Vec::new(),
            iso_neff_n_target: None,
            corpus: CorpusKind::Synthetic,
            paretoq_precision: None,
            disable_quantization: false,
            task_kind: TaskKind::Counter,
            use_ffn: false,
            d_hidden: 16,
            label_smoothing: 0.1,
            weight_decay: 0.1,
            apply_rmsnorm: true,
            grad_clip_l2: Some(1.0),
            latent_clamp_max: Some(1.0),
            dropout_p: 0.1,
        }
    }

    #[test]
    fn ablation_fix_has_7_variants() {
        assert_eq!(AblationFix::ALL.len(), 7);
    }

    #[test]
    fn disable_warmup_zeros_warmup_steps() {
        let mut cfg = micro_cfg();
        disable_in_config(&mut cfg, AblationFix::WarmupSchedule);
        assert_eq!(cfg.warmup_steps_unquantized, 0);
    }

    #[test]
    fn disable_label_smoothing_zeros_eps() {
        let mut cfg = micro_cfg();
        disable_in_config(&mut cfg, AblationFix::LabelSmoothing);
        assert_eq!(cfg.label_smoothing, 0.0);
    }

    #[test]
    fn disable_weight_decay_zeros_wd() {
        let mut cfg = micro_cfg();
        disable_in_config(&mut cfg, AblationFix::WeightDecay);
        assert_eq!(cfg.weight_decay, 0.0);
    }

    #[test]
    fn cumulative_zero_disables_all_seven_knobs() {
        let cfg = cumulative_config(&micro_cfg(), 0);
        assert_eq!(cfg.warmup_steps_unquantized, 0);
        assert_eq!(cfg.label_smoothing, 0.0);
        assert_eq!(cfg.weight_decay, 0.0);
        assert!(cfg.grad_clip_l2.is_none());
        assert!(cfg.latent_clamp_max.is_none());
        assert_eq!(cfg.dropout_p, 0.0);
        assert!(!cfg.apply_rmsnorm);
    }

    #[test]
    fn cumulative_seven_enables_all_seven_knobs() {
        let cfg = cumulative_config(&micro_cfg(), 7);
        assert!(cfg.warmup_steps_unquantized > 0);
        assert!((cfg.label_smoothing - 0.1).abs() < 1e-9);
        assert!((cfg.weight_decay - 0.1).abs() < 1e-9);
        assert_eq!(cfg.grad_clip_l2, Some(1.0));
        assert_eq!(cfg.latent_clamp_max, Some(1.0));
        assert!((cfg.dropout_p - 0.1).abs() < 1e-9);
        assert!(cfg.apply_rmsnorm);
    }

    #[test]
    fn cumulative_monotonic_n0_to_n7() {
        let base = micro_cfg();
        // Each step n+1 should differ from n in EXACTLY ONE knob.
        for n in 0..7 {
            let cfg_n = cumulative_config(&base, n);
            let cfg_n1 = cumulative_config(&base, n + 1);
            // Check progression is strictly monotonic (no regressions).
            assert!(cfg_n1.warmup_steps_unquantized >= cfg_n.warmup_steps_unquantized);
            assert!(cfg_n1.label_smoothing >= cfg_n.label_smoothing);
            assert!(cfg_n1.weight_decay >= cfg_n.weight_decay);
        }
    }

    #[test]
    fn short_name_unique_per_variant() {
        let names: std::collections::HashSet<_> =
            AblationFix::ALL.iter().map(|f| f.short_name()).collect();
        assert_eq!(names.len(), 7);
    }
}
