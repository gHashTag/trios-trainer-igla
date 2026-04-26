//! L1 — BPB-stream tracker.
//!
//! `BpbTracker` is the composition layer between L6 (`ema::EmaTracker`)
//! and L7 (`victory::SeedResult` / `JEPA_PROXY_BPB_FLOOR`).  Each
//! tracker is bound to a single seed and accumulates BPB observations
//! from one trial.  The tracker refuses, at record time, the same
//! pathologies the victory gate refuses at decision time:
//!
//! 1. pre-warmup observations (`step < INV2_WARMUP_BLIND_STEPS`)
//! 2. JEPA-MSE-proxy artefacts (`bpb ≤ JEPA_PROXY_BPB_FLOOR`)
//! 3. NaN / ±∞ readings
//!
//! The point is to **fail loud at the source** rather than discovering
//! a polluted seed three hours later when the victory gate refuses to
//! ratify it.  By the time you call `seed_result()` you are guaranteed
//! every contributing observation respected the L7 anchors.
//!
//! ## Coq anchor
//!
//! Composes INV-1 (BPB monotone-descent intent — `Admitted`) and INV-7
//! (`igla_found_criterion` — `Admitted`, `.v` file slated for L0).  No
//! new invariant introduced; no new Admitted line burned.
//!
//! ## Falsification witnesses (R8)
//!
//! Each `falsify_*` test demonstrates the tracker rejects a known-bad
//! input class, *and* leaves its internal state untouched after the
//! rejection — corruption-resistance is part of the contract.
//!
//! Refs: trios#143 lane L1 · INV-1 · INV-7 · L-R14 · R8.

use super::ema::{EmaError, EmaTracker};
use crate::invariants::INV2_WARMUP_BLIND_STEPS;
use super::victory::{SeedResult, JEPA_PROXY_BPB_FLOOR};
use crate::invariants::IGLA_TARGET_BPB;

// ----------------------------------------------------------------------
// Errors
// ----------------------------------------------------------------------

/// Reasons a BPB observation can be rejected at record time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpbError {
    /// Caller submitted an observation before warmup completed.
    BeforeWarmup,
    /// Reading is at or below `JEPA_PROXY_BPB_FLOOR` — TASK-5D guard.
    JepaProxyDetected,
    /// Reading is NaN or ±∞.
    NonFiniteBpb,
    /// Internal EMA refused the observation (NaN guard double-defence).
    EmaRejected,
}

impl core::fmt::Display for BpbError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BpbError::BeforeWarmup => write!(
                f,
                "BPB observation before INV2_WARMUP_BLIND_STEPS — reject"
            ),
            BpbError::JepaProxyDetected => write!(
                f,
                "BPB ≤ JEPA_PROXY_BPB_FLOOR (0.1) — TASK-5D proxy artefact, reject"
            ),
            BpbError::NonFiniteBpb => write!(f, "BPB is NaN or ±∞ — reject"),
            BpbError::EmaRejected => {
                write!(f, "EMA refused the observation (defence-in-depth)")
            }
        }
    }
}

impl std::error::Error for BpbError {}

impl From<EmaError> for BpbError {
    fn from(_: EmaError) -> Self {
        BpbError::EmaRejected
    }
}

// ----------------------------------------------------------------------
// Tracker
// ----------------------------------------------------------------------

/// Per-seed BPB tracker.  Bound to one seed for its entire lifetime so
/// the eventual `SeedResult` cannot accidentally mix evidence across
/// seeds.
#[derive(Debug, Clone, Copy)]
pub struct BpbTracker {
    seed: u64,
    ema: EmaTracker,
    /// Last accepted observation (raw, not smoothed).  `None` until
    /// `record()` accepts at least one BPB.
    last_bpb: Option<f64>,
    /// Step number of the last accepted observation.
    last_step: u64,
    /// Number of accepted observations.  Mirrors `ema.n_updates()` —
    /// kept locally so the tracker is fully introspectable without
    /// peeking inside the EMA.
    n_accepted: u64,
}

impl BpbTracker {
    /// Construct with an explicit smoothing α.  Returns
    /// `Err(BpbError::EmaRejected)` if α is out of `(0, 1]`.
    pub fn new(seed: u64, alpha: f64) -> Result<Self, BpbError> {
        let ema = EmaTracker::with_alpha(alpha)?;
        Ok(Self {
            seed,
            ema,
            last_bpb: None,
            last_step: 0,
            n_accepted: 0,
        })
    }

    /// Construct with the φ-band default smoothing (α = φ⁻³).
    pub fn phi_default(seed: u64) -> Self {
        Self {
            seed,
            ema: EmaTracker::phi_default(),
            last_bpb: None,
            last_step: 0,
            n_accepted: 0,
        }
    }

    /// Record a `(step, bpb)` reading.  Rejects each pathology with a
    /// typed `BpbError`; on rejection the internal state is unchanged.
    pub fn record(&mut self, step: u64, bpb: f64) -> Result<(), BpbError> {
        if !bpb.is_finite() {
            return Err(BpbError::NonFiniteBpb);
        }
        if step < INV2_WARMUP_BLIND_STEPS {
            return Err(BpbError::BeforeWarmup);
        }
        if bpb <= JEPA_PROXY_BPB_FLOOR {
            return Err(BpbError::JepaProxyDetected);
        }
        // Defence-in-depth: EMA also screens NaN, but we already did.
        // Convert any EmaError into BpbError::EmaRejected.
        self.ema.update(bpb)?;
        self.last_bpb = Some(bpb);
        self.last_step = step;
        self.n_accepted = self.n_accepted.saturating_add(1);
        Ok(())
    }

    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    #[inline]
    pub fn last_step(&self) -> u64 {
        self.last_step
    }

    #[inline]
    pub fn last_bpb(&self) -> Option<f64> {
        self.last_bpb
    }

    #[inline]
    pub fn n_accepted(&self) -> u64 {
        self.n_accepted
    }

    /// Smoothed (raw) EMA of accepted BPB observations.  `None` before
    /// any record has been accepted.
    pub fn smoothed(&self) -> Option<f64> {
        if self.n_accepted == 0 {
            None
        } else {
            Some(self.ema.raw())
        }
    }

    /// `true` once at least one observation past warmup has been
    /// accepted.
    #[inline]
    pub fn is_post_warmup(&self) -> bool {
        self.n_accepted > 0
    }

    /// Build a `SeedResult` for the L7 victory gate, using the latest
    /// raw BPB (not the smoothed value — the gate's contract is on the
    /// reported BPB at a given step).  Returns `None` if no observation
    /// has been accepted yet.
    pub fn seed_result(&self, sha: impl Into<String>) -> Option<SeedResult> {
        Some(SeedResult {
            seed: self.seed,
            step: self.last_step,
            bpb: self.last_bpb?,
            sha: sha.into(),
        })
    }

    /// Has this seed already crossed the IGLA target BPB on its latest
    /// accepted reading?  This is the hot-path predicate L11 uses to
    /// decide whether to keep training a seed.  Strict `<` per L7.
    pub fn has_crossed_target(&self) -> bool {
        match self.last_bpb {
            Some(b) => b < IGLA_TARGET_BPB,
            None => false,
        }
    }
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::race::victory::{check_victory, JEPA_PROXY_BPB_FLOOR};

    const POST_WARMUP: u64 = INV2_WARMUP_BLIND_STEPS + 1;

    // ----- admit (positive) cases ----------------------------------

    #[test]
    fn admit_phi_default_constructs() {
        let t = BpbTracker::phi_default(42);
        assert_eq!(t.seed(), 42);
        assert!(t.smoothed().is_none());
        assert!(!t.is_post_warmup());
        assert!(t.seed_result("deadbeef").is_none());
        assert!(!t.has_crossed_target());
    }

    #[test]
    fn admit_record_post_warmup_accepted() {
        let mut t = BpbTracker::phi_default(7);
        t.record(POST_WARMUP, 1.4).unwrap();
        assert_eq!(t.n_accepted(), 1);
        assert_eq!(t.last_step(), POST_WARMUP);
        assert_eq!(t.last_bpb(), Some(1.4));
        assert!(t.is_post_warmup());
        assert!(t.has_crossed_target()); // 1.4 < 1.5
    }

    #[test]
    fn admit_seed_result_round_trips_through_victory_gate() {
        // Three different seeds, all post-warmup, all below 1.5 →
        // `check_victory` must accept.
        let seeds = [11_u64, 22, 33];
        let bpbs = [1.49_f64, 1.45, 1.42];
        let mut results = Vec::new();
        for (s, b) in seeds.iter().zip(bpbs.iter()) {
            let mut t = BpbTracker::phi_default(*s);
            t.record(POST_WARMUP + 100, *b).unwrap();
            results.push(t.seed_result("abc1234").unwrap());
        }
        let report = check_victory(&results).unwrap();
        assert_eq!(report.winning_seeds.len(), 3);
    }

    #[test]
    fn admit_smoothed_tracks_ema_for_constant_stream() {
        let mut t = BpbTracker::new(0, 0.5).unwrap();
        for _ in 0..10 {
            t.record(POST_WARMUP, 1.7).unwrap();
        }
        // Constant input → EMA equals input.
        let s = t.smoothed().unwrap();
        assert!((s - 1.7).abs() < 1e-12);
    }

    #[test]
    fn admit_has_crossed_target_strict_inequality() {
        let mut t = BpbTracker::phi_default(0);
        // BPB exactly == target → does NOT cross.  L7 uses strict <.
        t.record(POST_WARMUP, IGLA_TARGET_BPB).unwrap();
        assert!(!t.has_crossed_target());
    }

    #[test]
    fn admit_state_independent_of_alpha_for_first_record() {
        // First update seeds the EMA exactly regardless of α.
        let mut a = BpbTracker::new(0, 0.1).unwrap();
        let mut b = BpbTracker::new(0, 0.9).unwrap();
        a.record(POST_WARMUP, 1.5).unwrap();
        b.record(POST_WARMUP, 1.5).unwrap();
        assert_eq!(a.smoothed(), b.smoothed());
    }

    // ----- falsification witnesses (R8) ----------------------------

    #[test]
    fn falsify_pre_warmup_step_rejected() {
        let mut t = BpbTracker::phi_default(0);
        let pre = INV2_WARMUP_BLIND_STEPS - 1;
        assert_eq!(t.record(pre, 1.4).unwrap_err(), BpbError::BeforeWarmup);
        // State untouched.
        assert_eq!(t.n_accepted(), 0);
        assert!(t.smoothed().is_none());
    }

    #[test]
    fn admit_warmup_boundary_step_accepted() {
        // step == INV2_WARMUP_BLIND_STEPS is the FIRST accepted step
        // (the guard rejects `step < warmup`, so == warmup passes).
        // This precisely mirrors L7 victory gate semantics — see
        // `victory::check_victory` `r.step < INV2_WARMUP_BLIND_STEPS`.
        let mut t = BpbTracker::phi_default(0);
        t.record(INV2_WARMUP_BLIND_STEPS, 1.4).unwrap();
        assert_eq!(t.n_accepted(), 1);
        assert_eq!(t.last_step(), INV2_WARMUP_BLIND_STEPS);
    }

    #[test]
    fn admit_warmup_plus_one_accepted() {
        // Boundary on the accept side.
        let mut t = BpbTracker::phi_default(0);
        t.record(INV2_WARMUP_BLIND_STEPS + 1, 1.4).unwrap();
        assert_eq!(t.n_accepted(), 1);
    }

    #[test]
    fn falsify_jepa_proxy_band_rejected() {
        let mut t = BpbTracker::phi_default(0);
        // 0.014 — the canonical TASK-5D bug value.
        assert_eq!(
            t.record(POST_WARMUP, 0.014).unwrap_err(),
            BpbError::JepaProxyDetected
        );
    }

    #[test]
    fn falsify_at_jepa_floor_rejected() {
        // BPB == floor → still proxy band (≤, not <).
        let mut t = BpbTracker::phi_default(0);
        assert_eq!(
            t.record(POST_WARMUP, JEPA_PROXY_BPB_FLOOR).unwrap_err(),
            BpbError::JepaProxyDetected
        );
    }

    #[test]
    fn falsify_nan_bpb_rejected() {
        let mut t = BpbTracker::phi_default(0);
        assert_eq!(
            t.record(POST_WARMUP, f64::NAN).unwrap_err(),
            BpbError::NonFiniteBpb
        );
        assert_eq!(t.n_accepted(), 0);
    }

    #[test]
    fn falsify_inf_bpb_rejected() {
        let mut t = BpbTracker::phi_default(0);
        assert_eq!(
            t.record(POST_WARMUP, f64::INFINITY).unwrap_err(),
            BpbError::NonFiniteBpb
        );
        assert_eq!(
            t.record(POST_WARMUP, f64::NEG_INFINITY).unwrap_err(),
            BpbError::NonFiniteBpb
        );
    }

    #[test]
    fn falsify_bad_alpha_rejected_at_construction() {
        // α out of (0, 1] propagates as BpbError::EmaRejected.
        assert_eq!(
            BpbTracker::new(0, 0.0).unwrap_err(),
            BpbError::EmaRejected
        );
        assert_eq!(
            BpbTracker::new(0, 2.0).unwrap_err(),
            BpbError::EmaRejected
        );
    }

    #[test]
    fn falsify_state_untouched_on_rejection() {
        // After three accepts and one rejection, the tracker must
        // expose three accepts — not four, not two.
        let mut t = BpbTracker::phi_default(0);
        t.record(POST_WARMUP, 1.4).unwrap();
        t.record(POST_WARMUP + 1, 1.35).unwrap();
        t.record(POST_WARMUP + 2, 1.30).unwrap();
        let _ = t.record(POST_WARMUP + 3, f64::NAN);
        assert_eq!(t.n_accepted(), 3);
        assert_eq!(t.last_step(), POST_WARMUP + 2);
        assert_eq!(t.last_bpb(), Some(1.30));
    }

    // ----- composition guards --------------------------------------

    #[test]
    fn composition_rejected_record_does_not_produce_seed_result_with_bad_bpb() {
        // If we successfully record one valid bpb then attempt a NaN,
        // `seed_result()` should reflect the LAST VALID record only.
        let mut t = BpbTracker::phi_default(99);
        t.record(POST_WARMUP, 1.49).unwrap();
        let _ = t.record(POST_WARMUP + 5, f64::NAN); // rejected
        let sr = t.seed_result("deadbeef").unwrap();
        assert_eq!(sr.seed, 99);
        assert_eq!(sr.step, POST_WARMUP);
        assert_eq!(sr.bpb, 1.49);
    }

    #[test]
    fn composition_three_seeds_below_target_reach_victory() {
        // End-to-end: build three trackers, feed valid streams, and
        // confirm `check_victory` admits the joint evidence.
        let mut trackers = [
            BpbTracker::phi_default(101),
            BpbTracker::phi_default(202),
            BpbTracker::phi_default(303),
        ];
        for (i, t) in trackers.iter_mut().enumerate() {
            t.record(POST_WARMUP, 1.5 - 0.05 * (i as f64 + 1.0)).unwrap();
        }
        let results: Vec<SeedResult> = trackers
            .iter()
            .map(|t| t.seed_result("sha1234").unwrap())
            .collect();
        let report = check_victory(&results).unwrap();
        assert_eq!(report.winning_seeds.len(), 3);
        // Min BPB should be 1.5 - 0.15 = 1.35.
        assert!((report.min_bpb - 1.35).abs() < 1e-12);
    }
}
