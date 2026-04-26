//! BPE tokenizer for IGLA-GF16
//!
//! Byte-Pair Encoding tokenizer with 32k vocabulary.
<<<<<<< HEAD
//! For CPU training, we use a simple implementation that loads vocabulary from file.
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

<<<<<<< HEAD
/// BPE tokenizer with 32k vocabulary
#[derive(Debug, Clone)]
pub struct BPETokenizer {
    /// Vocabulary mapping: token string -> ID
    vocab: HashMap<String, u32>,

    /// Reverse vocabulary: ID -> token string
    inverse_vocab: Vec<String>,

    /// Vocabulary size
=======
#[derive(Debug, Clone)]
pub struct BPETokenizer {
    vocab: HashMap<String, u32>,
    inverse_vocab: Vec<String>,
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    vocab_size: usize,
}

impl BPETokenizer {
<<<<<<< HEAD
    /// Create a new tokenizer from a vocabulary file
    ///
    /// # Arguments
    ///
    /// * `vocab_path` - Path to vocabulary file (one token per line)
    ///
    /// # Returns
    ///
    /// A new tokenizer instance
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    pub fn from_file<P: AsRef<Path>>(vocab_path: P) -> Result<Self, anyhow::Error> {
        let file = File::open(vocab_path)?;
        let reader = BufReader::new(file);

        let mut vocab = HashMap::new();
        let mut inverse_vocab = Vec::new();

        for (idx, line) in reader.lines().enumerate() {
            let token = line?;
            if idx < 32000 {
<<<<<<< HEAD
                // 32k vocab limit
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
                vocab.insert(token.clone(), idx as u32);
                inverse_vocab.push(token);
            }
        }

        let vocab_size = inverse_vocab.len();

        Ok(Self {
            vocab,
            inverse_vocab,
            vocab_size,
        })
    }

<<<<<<< HEAD
    /// Create a simple tokenizer with a predefined vocabulary
    ///
    /// For training, this creates a minimal tokenizer for testing.
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    pub fn new_dummy() -> Self {
        let mut vocab = HashMap::new();
        let mut inverse_vocab = Vec::new();

<<<<<<< HEAD
        // Create a 32k vocabulary (byte-level + common subwords)
        // Byte-level tokens (0-255)
        for i in 0..256 {
            let token = format!("<byte_{}>", i);
=======
        for i in 0..256 {
            let token = format!("<token_{}>", i);
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
            vocab.insert(token.clone(), i);
            inverse_vocab.push(token);
        }

<<<<<<< HEAD
        // Common subwords (256-32000)
        for i in 256..32000 {
            let token = format!("<subword_{}>", i);
            vocab.insert(token.clone(), i);
            inverse_vocab.push(token);
        }

        // Add special tokens
        vocab.insert("<pad>".to_string(), 32000);
        inverse_vocab.push("<pad>".to_string());
        vocab.insert("<unk>".to_string(), 32001);
        inverse_vocab.push("<unk>".to_string());
        vocab.insert("<eos>".to_string(), 32002);
=======
        vocab.insert("<pad>".to_string(), 256);
        inverse_vocab.push("<pad>".to_string());
        vocab.insert("<unk>".to_string(), 257);
        inverse_vocab.push("<unk>".to_string());
        vocab.insert("<eos>".to_string(), 258);
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
        inverse_vocab.push("<eos>".to_string());

        Self {
            vocab,
            inverse_vocab,
            vocab_size: 259,
        }
    }

<<<<<<< HEAD
    /// Create a tokenizer with 32k vocabulary (standard for language models)
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    pub fn new_32k() -> Self {
        let mut vocab = HashMap::new();
        let mut inverse_vocab = Vec::new();

<<<<<<< HEAD
        // Byte-level tokens (0-255)
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
        for i in 0..256 {
            let token = format!("<byte_{}>", i);
            vocab.insert(token.clone(), i);
            inverse_vocab.push(token);
        }

<<<<<<< HEAD
        // Common subwords (256-31999)
        // In production, these would be learned from data
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
        for i in 256..32000 {
            let token = format!("<subword_{}>", i);
            vocab.insert(token.clone(), i as u32);
            inverse_vocab.push(token);
        }

        Self {
            vocab,
            inverse_vocab,
            vocab_size: 32000,
        }
    }

<<<<<<< HEAD
    /// Encode text to token IDs
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to tokenize
    ///
    /// # Returns
    ///
    /// Vector of token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Simple character-level encoding for dummy tokenizer
        // In production, this would use BPE merge rules
=======
    pub fn encode(&self, text: &str) -> Vec<u32> {
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
        text.chars()
            .map(|c| self.vocab.get(&c.to_string()).copied().unwrap_or(257))
            .collect()
    }

<<<<<<< HEAD
    /// Decode token IDs to text
    ///
    /// # Arguments
    ///
    /// * `tokens` - Token IDs to decode
    ///
    /// # Returns
    ///
    /// Decoded text string
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .filter_map(|&id| self.inverse_vocab.get(id as usize).map(|s| s.as_str()))
            .collect()
    }

<<<<<<< HEAD
    /// Get vocabulary size
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

<<<<<<< HEAD
    /// Get token ID for a given string
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

<<<<<<< HEAD
    /// Get token string for a given ID
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.inverse_vocab.get(id as usize).map(|s| s.as_str())
    }
}

impl Default for BPETokenizer {
    fn default() -> Self {
        Self::new_dummy()
    }
}

<<<<<<< HEAD
/// Tokenize a batch of text sequences
///
/// # Arguments
///
/// * `tokenizer` - BPE tokenizer
/// * `texts` - Vector of text strings
/// * `max_len` - Maximum sequence length (padding/truncation)
///
/// # Returns
///
/// Vector of token ID vectors
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
pub fn tokenize_batch(tokenizer: &BPETokenizer, texts: &[&str], max_len: usize) -> Vec<Vec<u32>> {
    texts
        .iter()
        .map(|text| {
            let mut tokens = tokenizer.encode(text);
            if tokens.len() > max_len {
                tokens.truncate(max_len);
            } else {
<<<<<<< HEAD
                // Pad with pad token (256 for dummy tokenizer)
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
                tokens.resize(max_len, 256);
            }
            tokens
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_new_dummy() {
        let tokenizer = BPETokenizer::new_dummy();
        assert!(tokenizer.vocab_size() >= 256);

<<<<<<< HEAD
        // Check special tokens exist
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
        assert!(tokenizer.get_id("<pad>").is_some());
        assert!(tokenizer.get_id("<unk>").is_some());
        assert!(tokenizer.get_id("<eos>").is_some());
    }

    #[test]
    fn test_tokenizer_32k() {
        let tokenizer = BPETokenizer::new_32k();
        assert_eq!(tokenizer.vocab_size(), 32000);
    }

    #[test]
    fn test_tokenizer_encode() {
        let tokenizer = BPETokenizer::new_dummy();
        let text = "hello";
        let tokens = tokenizer.encode(text);

<<<<<<< HEAD
        // Should have at least one token per character
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_tokenizer_decode() {
        let tokenizer = BPETokenizer::new_dummy();
        let tokens = vec![0, 1, 2];
        let text = tokenizer.decode(&tokens);

<<<<<<< HEAD
        // Should produce some output
=======
>>>>>>> 20a55f6 (fix(L-T1): clean build — self-contained train_loop, remove broken stubs)
        assert!(!text.is_empty());
    }

    #[test]
    fn test_tokenize_batch() {
        let tokenizer = BPETokenizer::new_dummy();
        let texts = vec!["hello", "world"];
        let max_len = 10;

        let batch = tokenize_batch(&tokenizer, &texts, max_len);

        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].len(), max_len);
        assert_eq!(batch[1].len(), max_len);
    }

    #[test]
    fn test_tokenize_batch_truncation() {
        let tokenizer = BPETokenizer::new_dummy();
        let texts = vec!["hello world this is a very long text"];
        let max_len = 5;

        let batch = tokenize_batch(&tokenizer, &texts, max_len);

        assert_eq!(batch[0].len(), max_len);
    }
}
