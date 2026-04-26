//! trios-trainer — portable IGLA RACE training pipeline.
//!
//! Single-source-of-truth for the training stack referenced by
//! `gHashTag/trios#143`. Replaces the scattered `trios-train-cpu`,
//! `trios-training`, `trios-training-ffi`, and `scripts/*.py` paths.
//!
//! # Layout
//!
//! - [`config`]      TOML config (champion, gate2, needle-rush variants)
//! - [`model`]       Transformer + HybridAttn (ex `trios-train-cpu/transformer.rs` + `hybrid_attn.rs`)
//! - [`optimizer`]   AdamW + Muon + φ-LR schedule (ex `trios-train-cpu/optimizer.rs`)
//! - [`jepa`]        T-JEPA loss + EMA target (ex `trios-train-cpu/jepa/`)
//! - [`objective`]   Combined loss (ex `trios-train-cpu/objective.rs`)
//! - [`data`]        BPE tokenizer + dataloaders (ex `trios-train-cpu/tokenizer.rs`)
//! - [`gf16`]        GoldenFloat16 (ex `trios-train-cpu/gf16.rs`)
//! - [`checkpoint`]  Save/load + resume
//! - [`ledger`]      Triplet-validated emit to `assertions/seed_results.jsonl`
//!
//! Invariants (ASHA, victory gate, embargo list) are imported from
//! `trios-igla-race` — this crate **never** re-implements them.

pub mod config;
pub mod model;
pub mod model_hybrid_attn;
pub mod optimizer;
pub mod jepa;
pub mod objective;
pub mod data;
pub mod gf16;
pub mod forward;
pub mod backward;
pub mod checkpoint;
pub mod ledger;
pub mod train_loop;
pub mod invariants;

pub use config::TrainConfig;
pub use train_loop::{run, RunOutcome};
pub use model::NgramModel;
pub use model_hybrid_attn::HybridAttn;
pub use data::{BPETokenizer, tokenize_batch};
pub use optimizer::{AdamWCpu, MuonOptimizer, OptimizerKind, phi_lr_schedule};
pub use forward::{gelu, matmul, layer_norm, softmax, vec_add, vec_scale};
pub use backward::{
    clip_gradients, gelu_backward,
    layer_norm_backward, linear_backward, LinearGradients,
    softmax_cross_entropy_backward
};

/// Compile-time anchor: φ² + φ⁻² = 3 (Trinity Identity).
/// Cross-checked against `trios_igla_race::invariants` at runtime.
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
