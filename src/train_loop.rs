//! Training loop wired to ASHA + victory gate from `trios-igla-race`
//! (when the `trios-integration` feature is enabled) or to a local stub
//! constant (default build).

use anyhow::Result;
use crate::{TrainConfig, ledger};

/// Target BPB for IGLA RACE Gate-2.
///
/// When built with `--features trios-integration` this constant is verified
/// against the canonical `trios_igla_race::IGLA_TARGET_BPB` at runtime.
pub const DEFAULT_IGLA_TARGET_BPB: f64 = 1.85;

#[derive(Debug)]
pub struct RunOutcome {
    pub final_bpb: f64,
    pub steps_done: usize,
    pub jsonl_row: usize,
    pub gate_status: String,
}

/// Top-level entry. Loads model, optimizer, data, runs steps, emits ledger row
/// at the end, guarded by triplet + embargo.
pub fn run(cfg: &TrainConfig) -> Result<RunOutcome> {
    // -------- 0) sanity-check against runtime invariants
    #[cfg(feature = "trios-integration")]
    {
        use trios_igla_race::IGLA_TARGET_BPB;
        if (cfg.target_bpb - IGLA_TARGET_BPB).abs() > 1e-9 {
            tracing::warn!(
                "config.target_bpb {} != trios_igla_race::IGLA_TARGET_BPB {} — using config value",
                cfg.target_bpb, IGLA_TARGET_BPB
            );
        }
    }
    #[cfg(not(feature = "trios-integration"))]
    {
        if (cfg.target_bpb - DEFAULT_IGLA_TARGET_BPB).abs() > 1e-9 {
            tracing::warn!(
                "config.target_bpb {} != DEFAULT_IGLA_TARGET_BPB {} (build without `trios-integration`)",
                cfg.target_bpb, DEFAULT_IGLA_TARGET_BPB
            );
        }
    }

    // -------- 1) build model — L-T1 champion reproduction
    let _model = crate::champion::build(&cfg.model, cfg.seed)?;
    let _opt   = crate::optimizer::build(&cfg.optimizer)?;
    let _data  = crate::data::build(&cfg.data)?;
    let _obj   = crate::objective::build(&cfg.objective)?;

    // -------- 2) train loop (skeleton; full migration in follow-up PR)
    let mut bpb = f64::INFINITY;
    let mut step = 0usize;
    while step < cfg.steps {
        step += 1;
        if step % 1000 == 0 {
            bpb = 2.5; // placeholder, real impl reads from forward+backward
            tracing::info!(step, bpb, "checkpoint");
        }
    }

    // -------- 3) emit triplet-validated row
    let row = ledger::emit_row(cfg, bpb, step)?;

    Ok(RunOutcome {
        final_bpb: bpb,
        steps_done: step,
        jsonl_row: row.jsonl_row,
        gate_status: row.gate_status,
    })
}
