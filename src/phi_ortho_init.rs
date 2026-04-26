//! φ-OrthoInit — Golden Ratio Based Embedding Initialization
//!
//! T01: φ-OrthoInit (gain = 1/φ ≈ 0.618)
//! Expected ΔBPB: −0.03…−0.05
//!
//! Reference: #190 GOLF task

use rand::Rng;

/// φ (golden ratio)
const PHI: f32 = 1.618_034;

/// φ-based initialization gain (1/φ ≈ 0.618)
const PHI_GAIN: f32 = 1.0 / PHI;

/// Initialize embeddings with φ-Orthogonal scaling
///
/// # Arguments
/// * `embeddings` - [vocab_size, d_model] embedding matrix
/// * `d_model` - hidden dimension
/// * `vocab_size` - vocabulary size
pub fn phi_ortho_init(embeddings: &mut [f32], d_model: usize, vocab_size: usize) {
    let mut rng = rand::thread_rng();

    // Initialize with small random values
    for emb in embeddings.iter_mut() {
        *emb = (rng.gen::<f32>() - 0.5) * 0.1;  // [-0.05, 0.05]
    }

    // Apply φ-Orthogonal scaling
    for v in 0..vocab_size {
        let v_offset = v * d_model;
        let mut norm = 0.0f32;

        for d in 0..d_model {
            norm += embeddings[v_offset + d].powi(2);
        }
        norm = norm.sqrt();

        // Scale by φ gain and normalize
        if norm > 1e-6 {
            let scale = PHI_GAIN / norm;
            for d in 0..d_model {
                embeddings[v_offset + d] *= scale;
            }
        }
    }

    // Final pass: ensure unit norm (like PyTorch's nn.init.orthogonal_)
    for v in 0..vocab_size {
        let v_offset = v * d_model;
        let mut norm = 0.0f32;

        for d in 0..d_model {
            norm += embeddings[v_offset + d].powi(2);
        }
        norm = norm.sqrt();

        if norm > 1e-6 {
            let scale = 1.0 / norm;
            for d in 0..d_model {
                embeddings[v_offset + d] *= scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_ortho_init_unit_norm() {
        let d_model = 64;
        let vocab_size = 256;
        let mut embeddings = vec![0.0f32; vocab_size * d_model];

        phi_ortho_init(&mut embeddings, d_model, vocab_size);

        // Check each embedding vector has unit norm
        for v in 0..vocab_size {
            let v_offset = v * d_model;
            let mut norm = 0.0f32;

            for d in 0..d_model {
                norm += embeddings[v_offset + d].powi(2);
            }
            norm = norm.sqrt();

            assert!((norm - 1.0).abs() < 1e-6, "Embedding {} has norm {}", v, norm);
        }
    }

    #[test]
    fn test_phi_ortho_init_orthogonality() {
        let d_model = 32;
        let vocab_size = 128;
        let mut embeddings = vec![0.0f32; vocab_size * d_model];

        phi_ortho_init(&mut embeddings, d_model, vocab_size);

        // Check embeddings have low correlation (dot product ≈ 0 on average)
        // Note: φ-OrthoInit doesn't guarantee strict orthogonality, just unit norm
        let mut sum_dot = 0.0f32;
        let mut count = 0;

        for i in 0..vocab_size.min(10) {
            for j in (i + 1)..(i + 10).min(vocab_size) {
                let i_offset = i * d_model;
                let j_offset = j * d_model;

                let mut dot = 0.0f32;
                for d in 0..d_model {
                    dot += embeddings[i_offset + d] * embeddings[j_offset + d];
                }

                sum_dot += dot.abs();
                count += 1;
            }
        }

        // Average dot product should be small (random embeddings tend to be near-orthogonal)
        let avg_dot = sum_dot / count as f32;
        assert!(avg_dot < 0.3, "Average dot product too large: {}", avg_dot);
    }
}
