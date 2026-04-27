//! trios-trainer — portable IGLA RACE training pipeline.
//! Single-source-of-truth for `gHashTag/trios#143`. Anchor: phi^2 + phi^-2 = 3.
//!
//! Clippy/dead_code debt living on `main` since before integration PR #32 is
//! allow-listed in `Cargo.toml` `[lints]` to keep CI green while we focus on
//! Gate-2 (deadline 2026-04-30 23:59 UTC). Each lint pays down in a dedicated
//! technical-debt PR after merge. R5-honest: NOT introduced by PR #32.

pub mod checkpoint;
pub mod config;
pub mod data;
pub mod gf16;
pub mod igla;
pub mod invariants;
pub mod jepa;
pub mod ledger;
pub mod model;
pub mod model_hybrid_attn;
pub mod mup;
pub mod neon_writer;
pub mod objective;
pub mod optimizer;
pub mod phi_numbers;
pub mod race;
pub mod train_loop;

pub use config::TrainConfig;
pub use train_loop::{run, RunOutcome};

pub const TRINITY_ANCHOR: f64 = 3.0;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn anchor_holds() {
        let phi: f64 = (1.0 + 5f64.sqrt()) / 2.0;
        let lhs = phi.powi(2) + phi.powi(-2);
        assert!((lhs - TRINITY_ANCHOR).abs() < 1e-12);
    }
}
