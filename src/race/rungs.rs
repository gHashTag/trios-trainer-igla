//! L10 — ASHA rung progression (INV-12: `asha_rungs_trinity`).
//!
//! ASHA evaluates each trial at a strictly-ordered sequence of rungs. To avoid
//! the J-001/J-002 champion-killer bug (`prune_threshold = 2.65`), the rungs
//! themselves must follow the Trinity base — they are 1000 × 3ᵏ, k ∈ {0,1,2,3}.
//!
//! ── Coq trail (L-R14) ────────────────────────────────────────────────────
//!   trinity-clara/proofs/igla/igla_asha_bound.v
//!     • Theorem  asha_rungs_trinity  (Qed) — INV-12 fully Proven
//!   trinity-clara/proofs/igla/lucas_closure_gf16.v
//!     • Theorem  lucas_recurrence_closed  (Qed) — L(1)=3 anchor for Trinity
//!   assertions/igla_assertions.json :: INV-12
//!     status        = "Proven", admitted_count = 0
//!     numeric_anchor.valid_rungs = [1000, 3000, 9000, 27000]
//!     runtime_check = { action: "abort" }
//!     runtime_target = "crates/trios-igla-race/src/invariants.rs::check_inv12_rung_valid"
//!
//! The runtime target is *re-homed* here (L10 lane) and re-exported from
//! `crate::invariants` only by reference, keeping L5 (invariants.rs) untouched.
//!
//! ── Algebraic anchor ─────────────────────────────────────────────────────
//!   TRINITY_BASE = 3    = φ² + φ⁻²   (Trinity Identity, Lucas L(1))
//!   RUNG_UNIT    = 1000                (BENCH-002 minimum-trial floor, #143)
//!   rung(k)      = RUNG_UNIT · TRINITY_BASE^k,  k ∈ {0, 1, 2, 3}
//!
//! ── Lane discipline (#143 R3, R6, R10) ───────────────────────────────────
//!   • Single new module + one-line lib.rs re-export. No edits to asha.rs
//!     (L2), invariants.rs (L5), sampler.rs (L8).
//!   • One atomic commit on main.

use std::fmt;

use crate::invariants::InvError;

// ─── Coq-anchored constants ──────────────────────────────────────────────

/// Trinity base: `3 = φ² + φ⁻²` (Lucas L(1) anchor).
///
/// Coq: `lucas_closure_gf16.v::lucas_recurrence_closed`. The literal `3` here
/// is the Trinity Identity itself, not a free numeric constant — see the
/// L-R14 traceability table for INV-12.
pub const TRINITY_BASE: u32 = 3;

/// First-rung step count, anchored in `assertions/igla_assertions.json::INV-12`.
/// Coq: `igla_asha_bound.v::asha_rungs_trinity`.
pub const RUNG_UNIT: u32 = 1_000;

/// Maximum rung exponent (inclusive). The progression terminates at
/// `RUNG_UNIT · TRINITY_BASE^MAX_RUNG_EXP` = 27000. This bound is fixed by the
/// JSON anchor `valid_rungs.len() - 1 = 3`.
pub const MAX_RUNG_EXP: u32 = 3;

/// Number of valid rungs (= MAX_RUNG_EXP + 1).
pub const RUNG_COUNT: usize = (MAX_RUNG_EXP as usize) + 1;

// Compile-time guards. If any of these fail the build, the JSON anchor and
// this module disagree — fix one before shipping.
const _: () = {
    assert!(TRINITY_BASE == 3, "Trinity base must be 3 (φ² + φ⁻²)");
    assert!(RUNG_UNIT == 1_000, "Rung unit pinned by INV-12 JSON anchor");
    assert!(
        MAX_RUNG_EXP == 3,
        "Max rung exponent pinned to 3 (rung 27000)"
    );
};

// ─── Rung type ───────────────────────────────────────────────────────────

/// A single ASHA rung. Construct only via [`Rung::new`] (validates INV-12)
/// or the pre-validated [`Rung::ALL`] constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rung {
    exp: u32,
}

impl Rung {
    /// Pre-validated rung constants, in ascending order.
    /// Coq: `igla_asha_bound.v::asha_rungs_trinity`.
    pub const ALL: [Rung; RUNG_COUNT] = [
        Rung { exp: 0 }, // 1_000  · 3⁰ = 1_000
        Rung { exp: 1 }, // 1_000  · 3¹ = 3_000
        Rung { exp: 2 }, // 1_000  · 3² = 9_000
        Rung { exp: 3 }, // 1_000  · 3³ = 27_000
    ];

    /// Construct from an exponent. INV-12 requires `exp ≤ MAX_RUNG_EXP`.
    pub fn from_exp(exp: u32) -> Result<Self, InvError> {
        if exp > MAX_RUNG_EXP {
            // Re-use existing InvError variant; INV-12 maps to "rung step
            // not in valid set", which is structurally identical to a grid
            // mismatch (out-of-bounds discrete enumerable).
            //
            // We deliberately do NOT add a new InvError variant here because
            // that would require touching invariants.rs (L5 lane). Instead,
            // the caller-facing API is `check_inv12_rung_valid` (below),
            // which returns a clear message via Display.
            Err(InvError::Inv4GridMismatch {
                grid: 1_000_000usize.saturating_add(exp as usize),
                k: 0,
            })
        } else {
            Ok(Self { exp })
        }
    }

    /// Try to build a Rung from a step count.
    /// Returns `Some(Rung)` iff `step` ∈ `[1000, 3000, 9000, 27000]`.
    pub fn from_step(step: u32) -> Option<Self> {
        Self::ALL.iter().copied().find(|r| r.step() == step)
    }

    /// Exponent k such that `step = RUNG_UNIT · 3^k`.
    #[inline]
    pub const fn exp(self) -> u32 {
        self.exp
    }

    /// Step count = `RUNG_UNIT · 3^exp`.
    #[inline]
    pub const fn step(self) -> u32 {
        // Manual integer power — `u32::pow` is const since 1.46 but using a
        // loop keeps the constructor `const`-portable across older compilers.
        let mut acc: u32 = RUNG_UNIT;
        let mut i = 0;
        while i < self.exp {
            acc *= TRINITY_BASE;
            i += 1;
        }
        acc
    }

    /// Next rung in the progression, or `None` if at the last rung.
    pub fn next(self) -> Option<Rung> {
        if self.exp >= MAX_RUNG_EXP {
            None
        } else {
            Some(Self { exp: self.exp + 1 })
        }
    }

    /// First (smallest) rung — useful as the ASHA seed rung.
    pub const fn first() -> Self {
        Self { exp: 0 }
    }

    /// Last (largest) rung.
    pub const fn last() -> Self {
        Self { exp: MAX_RUNG_EXP }
    }
}

impl fmt::Display for Rung {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rung[{}={}·3^{}]", self.step(), RUNG_UNIT, self.exp)
    }
}

// ─── Public iteration helpers ────────────────────────────────────────────

/// Returns all valid rungs as a `Vec<u32>` of step counts.
/// Coq: identical to `igla_assertions.json::INV-12.numeric_anchor.valid_rungs`.
pub fn all_rung_steps() -> Vec<u32> {
    Rung::ALL.iter().map(|r| r.step()).collect()
}

/// Iterator over `(Rung, step)` pairs in ascending order.
pub fn iter_rungs() -> impl Iterator<Item = (Rung, u32)> {
    Rung::ALL.iter().copied().map(|r| (r, r.step()))
}

// ─── Runtime guard (INV-12) ──────────────────────────────────────────────

/// Validate that an externally-provided step count is one of the INV-12 rungs.
///
/// Use at trial-config-load time and at every checkpoint commit. Per the JSON
/// `runtime_check.action = "abort"`, callers should treat `Err` as fatal.
///
/// Coq: `igla_asha_bound.v::asha_rungs_trinity` (Qed).
pub fn check_inv12_rung_valid(step: u32) -> Result<Rung, InvError> {
    // Encode the rejected step into Inv4GridMismatch so we don't add a
    // new InvError variant (avoids touching the L5 lane).
    Rung::from_step(step).ok_or(InvError::Inv4GridMismatch {
        grid: step as usize,
        k: 0,
    })
}

/// Convenience: validate a `usize` step (used by `asha.rs::record_checkpoint`).
pub fn check_inv12_rung_valid_usize(step: usize) -> Result<Rung, InvError> {
    let s32 = u32::try_from(step).map_err(|_| InvError::Inv4GridMismatch { grid: step, k: 0 })?;
    check_inv12_rung_valid(s32)
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Anchored against `assertions/igla_assertions.json::INV-12.numeric_anchor.valid_rungs`.
    /// Coq: `asha_rungs_trinity` (Qed).
    #[test]
    fn t01_valid_rungs_from_spec() {
        assert_eq!(all_rung_steps(), vec![1_000, 3_000, 9_000, 27_000]);
    }

    /// Each rung is exactly 3× the previous (Trinity progression).
    /// Coq: structural induction on `lucas_recurrence_closed`.
    #[test]
    fn t02_trinity_progression() {
        let steps = all_rung_steps();
        for w in steps.windows(2) {
            assert_eq!(w[1], w[0] * TRINITY_BASE);
        }
    }

    #[test]
    fn t03_rung_zero_is_unit() {
        assert_eq!(Rung::first().step(), RUNG_UNIT);
        assert_eq!(Rung::ALL[0].step(), 1_000);
    }

    #[test]
    fn t04_rung_last_is_27000() {
        assert_eq!(Rung::last().step(), 27_000);
        assert_eq!(Rung::ALL[RUNG_COUNT - 1].step(), 27_000);
    }

    /// Forbidden by INV-12: step counts that are multiples of RUNG_UNIT but
    /// NOT pure powers of 3.
    /// Coq: REJECT branch of `asha_rungs_trinity`.
    #[test]
    fn t05_arbitrary_multiples_of_unit_rejected() {
        for &bad in &[2_000_u32, 4_000, 5_000, 6_000, 7_000, 8_000, 10_000, 12_000] {
            assert!(
                check_inv12_rung_valid(bad).is_err(),
                "step={bad} must be rejected by INV-12"
            );
        }
    }

    /// Off-by-one near valid rungs must be rejected.
    #[test]
    fn t06_off_by_one_rejected() {
        for &bad in &[999_u32, 1_001, 2_999, 3_001, 8_999, 9_001, 26_999, 27_001] {
            assert!(
                check_inv12_rung_valid(bad).is_err(),
                "step={bad} must be rejected (off-by-one)"
            );
        }
    }

    #[test]
    fn t07_zero_step_rejected() {
        assert!(check_inv12_rung_valid(0).is_err());
    }

    /// `4000` is the warmup_blind_steps anchor (INV-2), NOT a valid rung.
    /// Catches a bug where a future agent confuses warmup with rung-1.
    #[test]
    fn t08_warmup_4000_is_not_a_rung() {
        assert!(check_inv12_rung_valid(4_000).is_err());
    }

    /// Falsification witness (R8). Mirrors the §0 R7 forbidden table:
    /// `prune_threshold = 2.65` was a champion-killer; analogously, any
    /// non-Trinity rung is a progression-killer. Document one canonical
    /// counter-example here.
    #[test]
    fn t09_falsification_witness_step_2000() {
        // 2000 is exactly 2·RUNG_UNIT — the simplest non-trinity multiple.
        // Asserted Err to lock the witness.
        let res = check_inv12_rung_valid(2_000);
        assert!(res.is_err(), "falsification witness: 2000 must Err");
    }

    #[test]
    fn t10_next_rung_monotone() {
        let mut r = Rung::first();
        let mut prev = r.step();
        while let Some(n) = r.next() {
            assert!(n.step() > prev, "next rung must strictly increase");
            prev = n.step();
            r = n;
        }
    }

    #[test]
    fn t11_next_rung_after_last_is_none() {
        assert!(Rung::last().next().is_none());
    }

    #[test]
    fn t12_iter_count_equals_four() {
        assert_eq!(iter_rungs().count(), RUNG_COUNT);
        assert_eq!(RUNG_COUNT, 4);
    }

    /// Round-trip: every valid step → `Rung` → step.
    #[test]
    fn t13_check_inv12_accepts_all_four_rungs() {
        for step in all_rung_steps() {
            let rung = check_inv12_rung_valid(step).expect("valid rung must pass");
            assert_eq!(rung.step(), step);
        }
    }

    /// `from_exp(MAX_RUNG_EXP + 1)` must fail — protects against silent
    /// rung-table extension without updating JSON + Coq theorem.
    #[test]
    fn t14_from_exp_above_max_rejected() {
        assert!(Rung::from_exp(MAX_RUNG_EXP + 1).is_err());
        assert!(Rung::from_exp(99).is_err());
    }

    /// Display formatter exposes the Trinity decomposition in logs.
    #[test]
    fn t15_display_shows_trinity_decomposition() {
        let s = format!("{}", Rung::ALL[2]);
        assert!(s.contains("9000"), "got {s}");
        assert!(s.contains("3^2"), "got {s}");
    }
}
