//! JEPA loss computation
//!
//! Implements MSE loss with L2 normalization and variance-based anti-collapse.

/// JEPA loss configuration
#[derive(Debug, Clone, Copy)]
pub struct JepaLossConfig {
    pub use_l2_normalization: bool,
    pub stop_gradient: bool,
    pub anti_collapse_weight: f64,
}

impl Default for JepaLossConfig {
    fn default() -> Self {
        Self {
            use_l2_normalization: true,
            stop_gradient: true,
            anti_collapse_weight: 0.01,
        }
    }
}

/// JEPA loss result
#[derive(Debug, Clone)]
pub struct JepaLoss {
    pub total: f64,
    pub prediction: f64,
    pub variance: f64,
}

impl JepaLoss {
    /// Create a new loss result
    pub fn new(total: f64, prediction: f64, variance: f64) -> Self {
        Self { total, prediction, variance }
    }

    /// Check if variance indicates collapse (< 0.01)
    pub fn is_collapsed(&self) -> bool {
        self.variance < 0.01
    }
}

/// Compute JEPA loss with optional L2 normalization
///
/// # Arguments
/// * `predicted` - Predicted embeddings
/// * `target` - Target embeddings
/// * `config` - Loss configuration
///
/// # Returns
/// JepaLoss with total, prediction, and variance components
///
/// The total loss is: prediction_loss - variance * anti_collapse_weight
/// This encourages matching predictions while maintaining representation diversity.
pub fn compute_jepa_loss(
    predicted: &[f32],
    target: &[f32],
    config: JepaLossConfig,
) -> JepaLoss {
    assert_eq!(predicted.len(), target.len(), "predicted and target must have same length");

    let (pred_norm, tgt_norm) = if config.use_l2_normalization {
        (l2_normalize(predicted), l2_normalize(target))
    } else {
        (predicted.to_vec(), target.to_vec())
    };

    // Prediction loss (MSE)
    let prediction_loss = pred_norm
        .iter()
        .zip(tgt_norm.iter())
        .map(|(p, t)| (p - t).powi(2) as f64)
        .sum::<f64>() / pred_norm.len() as f64;

    // Variance computation (for anti-collapse)
    let mean = tgt_norm.iter().sum::<f32>() as f64 / tgt_norm.len() as f64;
    let variance = tgt_norm
        .iter()
        .map(|t| (*t as f64 - mean).powi(2))
        .sum::<f64>() / tgt_norm.len() as f64;

    // Total loss with anti-collapse
    let total = prediction_loss - variance * config.anti_collapse_weight;

    JepaLoss {
        total,
        prediction: prediction_loss,
        variance,
    }
}

/// L2 normalize a vector to unit length
///
/// # Arguments
/// * `v` - Vector to normalize
///
/// # Returns
/// Normalized vector (or original if norm is too small)
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

    if norm < 1e-8 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Compute MSE loss between two vectors
pub fn mse_loss(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2) as f64)
        .sum::<f64>() / a.len() as f64
}

/// Compute cosine similarity (higher = more similar)
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

    if norm_a < 1e-8 || norm_b < 1e-8 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0_f32, 4.0_f32];
        let normed = l2_normalize(&v);

        // Norm should be 1.0
        let norm: f32 = normed.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let v = vec![0.0_f32; 10];
        let normed = l2_normalize(&v);

        assert_eq!(normed, v);
    }

    #[test]
    fn test_mse_loss() {
        let a = vec![1.0_f32, 2.0_f32, 3.0_f32];
        let b = vec![1.0_f32, 2.0_f32, 3.0_f32];

        let loss = mse_loss(&a, &b);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_mse_loss_nonzero() {
        let a = vec![0.0_f32, 0.0_f32];
        let b = vec![1.0_f32, 1.0_f32];

        let loss = mse_loss(&a, &b);
        assert_eq!(loss, 1.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0_f32, 0.0_f32];
        let b = vec![1.0_f32, 0.0_f32];

        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0_f32, 0.0_f32];
        let b = vec![0.0_f32, 1.0_f32];

        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0_f32, 0.0_f32];
        let b = vec![-1.0_f32, 0.0_f32];

        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_jepa_loss_identical() {
        let v = vec![1.0_f32, 2.0_f32, 3.0_f32];
        let loss = compute_jepa_loss(&v, &v, JepaLossConfig::default());

        assert_eq!(loss.prediction, 0.0);
        // Variance > 0 for non-constant vectors
        assert!(loss.variance > 0.0);
    }

    #[test]
    fn test_jepa_loss_different() {
        // Use non-parallel vectors to get non-zero loss after normalization
        let a = vec![1.0_f32, 2.0_f32];
        let b = vec![3.0_f32, 1.0_f32];

        let loss = compute_jepa_loss(&a, &b, JepaLossConfig::default());

        assert!(loss.prediction > 0.0);
        assert!(!loss.is_collapsed());
    }

    #[test]
    fn test_jepa_loss_constant_vector() {
        let v = vec![1.0_f32; 100];

        let loss = compute_jepa_loss(&v, &v, JepaLossConfig::default());

        // Constant vector has zero variance
        assert!(loss.variance < 0.01);
        assert!(loss.is_collapsed());
    }

    #[test]
    fn test_jepa_loss_without_normalization() {
        let config = JepaLossConfig {
            use_l2_normalization: false,
            ..Default::default()
        };

        let a = vec![1.0_f32, 2.0_f32];
        let b = vec![2.0_f32, 4.0_f32];

        let loss = compute_jepa_loss(&a, &b, config);
        assert!(loss.prediction > 0.0);
    }

    #[test]
    fn test_jepa_loss_custom_anti_collapse() {
        let config = JepaLossConfig {
            anti_collapse_weight: 0.1,
            ..Default::default()
        };

        let a = vec![1.0_f32; 100];
        let loss = compute_jepa_loss(&a, &a, config);

        // With higher anti-collapse weight, total should be negative for collapsed
        assert!(loss.total < 0.0);
    }

    #[test]
    fn test_jepa_loss_dimensions_mismatch() {
        let a = vec![1.0_f32, 2.0_f32];
        let b = vec![1.0_f32, 2.0_f32, 3.0_f32];

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            compute_jepa_loss(&a, &b, JepaLossConfig::default());
        }));
        assert!(result.is_err());
    }

    #[test]
    fn test_jepa_loss_struct() {
        let loss = JepaLoss::new(1.5, 1.0, 0.5);

        assert_eq!(loss.total, 1.5);
        assert_eq!(loss.prediction, 1.0);
        assert_eq!(loss.variance, 0.5);
        assert!(!loss.is_collapsed());
    }

    #[test]
    fn test_jepa_loss_struct_collapsed() {
        let loss = JepaLoss::new(0.5, 0.5, 0.005);

        assert!(loss.is_collapsed());
    }

    #[test]
    fn test_l2_normalize_preserves_direction() {
        let a = vec![3.0_f32, 4.0_f32, 0.0_f32];
        let b = vec![6.0_f32, 8.0_f32, 0.0_f32];

        let a_norm = l2_normalize(&a);
        let b_norm = l2_normalize(&b);

        // After normalization, parallel vectors should be identical
        assert_eq!(a_norm, b_norm);
    }

    #[test]
    fn test_jepa_loss_large_vectors() {
        let mut a = vec![0.0_f32; 1000];
        let mut b = vec![0.0_f32; 1000];

        for i in 0..1000 {
            a[i] = (i as f32) / 1000.0;
            b[i] = a[i] + 0.1;
        }

        let loss = compute_jepa_loss(&a, &b, JepaLossConfig::default());
        assert!(loss.prediction > 0.0);
        assert!(loss.variance > 0.0);
    }
}
