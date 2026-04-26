//! trios-trainer — portable IGLA RACE training pipeline.
//! Single-source-of-truth for `gHashTag/trios#143`. Anchor: phi^2 + phi^-2 = 3.

pub mod attention;
pub mod backward;
pub mod bench;
pub mod checkpoint;
pub mod config;
pub mod data;
pub mod forward;
pub mod gf16;
pub mod invariants;
pub mod jepa;
pub mod ledger;
pub mod model;
pub mod model_hybrid_attn;
pub mod objective;
pub mod optimizer;
pub mod ortho_init_baseline;
pub mod phi_numbers;
pub mod phi_ortho_init;
pub mod pipeline;
#[cfg(feature = "race")]
pub mod race;
pub mod real_igla_model;
pub mod real_igla_trainer;
pub mod residual_mix;
pub mod sliding_eval;
pub mod swa_phi;
pub mod train_loop;
pub mod train_model;
pub mod transformer;
pub mod transformer_trainer;
pub mod trinity_3k;

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
