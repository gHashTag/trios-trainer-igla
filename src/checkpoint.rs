//! Checkpoint save/load. Stored under `${TRIOS_CHECKPOINT_DIR}/{run-name}/{step}.bin`.
//! Resume sets `step_done` and re-attaches optimizer state.

use anyhow::Result;
use std::path::PathBuf;

pub fn checkpoint_path(run_name: &str, step: usize) -> PathBuf {
    let base = std::env::var("TRIOS_CHECKPOINT_DIR").unwrap_or_else(|_| "checkpoints".into());
    PathBuf::from(base)
        .join(run_name)
        .join(format!("{step}.bin"))
}

pub fn save(_run: &str, _step: usize, _bytes: &[u8]) -> Result<()> {
    // TODO: actually persist (zstd-compressed bincode). Non-blocking for skeleton.
    Ok(())
}
