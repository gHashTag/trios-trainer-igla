//! Model module — facade. Concrete impls are migrated from
//! `trios-train-cpu/src/transformer.rs` + `hybrid_attn.rs` in the follow-up PR.

use anyhow::Result;
use crate::config::ModelConfig;

pub struct Model {
    pub d_model: usize,
    pub n_layers: usize,
    pub hybrid: bool,
}

pub fn build(cfg: &ModelConfig) -> Result<Model> {
    Ok(Model {
        d_model: cfg.d_model,
        n_layers: cfg.n_layers,
        hybrid: cfg.hybrid_attn,
    })
}
