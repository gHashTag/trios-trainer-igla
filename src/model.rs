<<<<<<< HEAD
#![allow(
    clippy::needless_range_loop,
    dead_code,
    unused_imports,
    clippy::excessive_precision
)]
//! Model module with NgramModel stub (L-T1)
//!
//! This file contains the champion model implementation.
//! During L-T1, we provide a stub to allow compilation.

use crate::model_hybrid_attn::HybridAttn;

/// N-gram model (placeholder for L-T1)
///
/// This is the minimal placeholder that allows the build to succeed.
/// The actual champion implementation will be merged from
/// `crates/trios-train-cpu/src/transformer.rs` and `hybrid_attn.rs`.
pub struct NgramModel {
    _private: (),
}

impl NgramModel {
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Placeholder forward pass
    pub fn forward(&self, _tokens: &[usize]) -> Vec<f32> {
        vec![] // Placeholder - returns empty
    }

    /// Placeholder parameter count
    pub fn param_count(&self) -> usize {
        0 // Placeholder
    }
}

impl Default for NgramModel {
    fn default() -> Self {
        Self::new()
    }
=======
use anyhow::Result;
use crate::config::ModelConfig;

pub struct Model {
    pub d_model: usize,
    pub n_layers: usize,
    pub hybrid: bool,
}

pub fn build(cfg: &ModelConfig) -> Result<Model> {
    Ok(Model { d_model: cfg.d_model, n_layers: cfg.n_layers, hybrid: cfg.hybrid_attn })
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
}
