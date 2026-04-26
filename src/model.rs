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
}
