//! Optimizer facade. Concrete impls migrated from
//! `trios-train-cpu/src/optimizer.rs` (AdamW, Muon, φ-LR schedule).

use crate::config::OptimizerConfig;
use anyhow::{bail, Result};

pub struct Optimizer {
    pub kind: String,
    pub lr: f64,
}

pub fn build(cfg: &OptimizerConfig) -> Result<Optimizer> {
    match cfg.kind.as_str() {
        "adamw" | "muon" | "muon+adamw" => {}
        other => bail!("unknown optimizer kind: {other}"),
    }
    Ok(Optimizer {
        kind: cfg.kind.clone(),
        lr: cfg.lr,
    })
}
