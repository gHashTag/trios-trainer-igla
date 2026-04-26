//! Data + tokenizer façade. Migrated from
//! `trios-train-cpu/src/{tokenizer.rs, data.rs}` and `trios-data` crate.

use crate::config::DataConfig;
use anyhow::Result;

pub struct DataPipeline {
    pub corpus: String,
    pub batch_size: usize,
    pub batch_tokens: usize,
}

pub fn build(cfg: &DataConfig) -> Result<DataPipeline> {
    Ok(DataPipeline {
        corpus: cfg.corpus.clone(),
        batch_size: cfg.batch_size,
        batch_tokens: cfg.batch_tokens,
    })
}
