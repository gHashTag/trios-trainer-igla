//! trios-trainer — portable IGLA RACE training pipeline.
//! Single-source-of-truth for `gHashTag/trios#143`. Anchor: phi^2 + phi^-2 = 3.

<<<<<<< HEAD
pub mod backward;
pub mod champion;
=======
pub mod config;
pub mod model;
pub mod optimizer;
pub mod objective;
pub mod gf16;
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
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
pub mod train_loop;
pub mod invariants;
pub mod model_hybrid_attn;

pub use backward::{
    clip_gradients, gelu_backward, layer_norm_backward, linear_backward,
    softmax_cross_entropy_backward, LinearGradients,
};
pub use config::TrainConfig;
pub use data::{tokenize_batch, BPETokenizer};
pub use forward::{gelu, layer_norm, matmul, softmax, vec_add, vec_scale};
pub use model::NgramModel;
pub use model_hybrid_attn::HybridAttn;
pub use optimizer::{phi_lr_schedule, AdamWCpu, MuonOptimizer, OptimizerKind};
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
