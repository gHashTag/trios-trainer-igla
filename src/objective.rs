//! Combined loss: w_ce·CE + w_jepa·JEPA + w_nca·NCA.
//! Migrated from `trios-train-cpu/src/objective.rs`.

use anyhow::Result;
use crate::config::ObjectiveConfig;

pub struct Objective {
    pub w_ce: f64,
    pub w_jepa: f64,
    pub w_nca: f64,
}

pub fn build(cfg: &ObjectiveConfig) -> Result<Objective> {
    Ok(Objective { w_ce: cfg.w_ce, w_jepa: cfg.w_jepa, w_nca: cfg.w_nca })
}
