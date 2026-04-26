//! Data module
//!
//! - [`tokenizer`] — BPE tokenizer with 32k vocabulary

pub mod tokenizer;

pub use tokenizer::BPETokenizer;

pub fn tokenize_batch(_texts: &[&str]) -> Vec<Vec<u32>> {
    vec![]
}

pub fn build(_cfg: &crate::config::DataConfig) -> anyhow::Result<()> {
    Ok(())
}
