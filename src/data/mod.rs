pub mod tokenizer;

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

pub use tokenizer::tokenize_batch;
pub use tokenizer::BPETokenizer;
