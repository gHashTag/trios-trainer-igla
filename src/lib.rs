//! trios-trainer — portable IGLA RACE training pipeline.
//! Single-source-of-truth for `gHashTag/trios#143`. Anchor: phi^2 + phi^-2 = 3.

pub mod phi_numbers;
pub mod checkpoint;
pub mod config;
pub mod data;
pub mod gf16;
pub mod invariants;
pub mod jepa;
pub mod ledger;
pub mod model;
pub mod model_hybrid_attn;
pub mod objective;
pub mod optimizer;
pub mod mup;
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
