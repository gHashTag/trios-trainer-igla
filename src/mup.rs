//! Maximal Update Parametrization (µP) transfer
//!
//! Reference: [Cerebras µP Guide](https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization)
//!
//! Key idea: optimal LR found at small width (8M proxy) transfers verbatim to
//! larger models (24M, 70M) with <5% degradation.
//!
//! Implementation:
//! - Input/output multiplier scaling
//! - Attention QK 1/√d_head scaling
//! - Per-parameter-group LR scaling (embedding_mult, output_mult, attn_mult)

use std::f64;

/// µP scaling factors for model width transfer
///
/// Computes the scaling multipliers for different parameter groups
/// to enable LR transfer from small (proxy) to large (target) models.
#[derive(Debug, Clone)]
pub struct MupScaler {
    /// Width dimension (hidden size or d_model)
    pub width: usize,

    /// Reference width for LR transfer (default: 8M proxy width)
    pub ref_width: usize,

    /// Input multiplier scale
    pub input_mult: f64,

    /// Output multiplier scale
    pub output_mult: f64,

    /// Attention QK multiplier scale (1/√d_head)
    pub attn_mult: f64,

    /// LR multiplier for embedding layer
    pub embedding_lr_mult: f64,

    /// LR multiplier for output layer
    pub output_lr_mult: f64,

    /// LR multiplier for attention layers
    pub attn_lr_mult: f64,
}

impl MupScaler {
    /// Create a new µP scaler for width transfer
    ///
    /// # Arguments
    ///
    /// * `width` - Current model width (d_model or hidden size)
    /// * `ref_width` - Reference width where LR* was found (default: proxy width)
    ///
    /// # Returns
    ///
    /// A new MupScaler with computed scaling factors
    pub fn new(width: usize, ref_width: usize) -> Self {
        let width_ratio = width as f64 / ref_width as f64;

        // µP scaling: multipliers scale with √(width / ref_width)
        let sqrt_ratio = width_ratio.sqrt();

        Self {
            width,
            ref_width,
            input_mult: sqrt_ratio,
            output_mult: sqrt_ratio,
            attn_mult: 1.0 / (width as f64).sqrt(), // 1/√d_head
            embedding_lr_mult: 1.0 / sqrt_ratio,    // LR scales inversely with √width
            output_lr_mult: 1.0 / sqrt_ratio,
            attn_lr_mult: 1.0 / sqrt_ratio,
        }
    }

    /// Create scaler for champion architecture (baseline, no µP scaling)
    pub fn champion() -> Self {
        Self {
            width: 256,
            ref_width: 256,
            input_mult: 1.0,
            output_mult: 1.0,
            attn_mult: 1.0 / (256.0_f64).sqrt(),
            embedding_lr_mult: 1.0,
            output_lr_mult: 1.0,
            attn_lr_mult: 1.0,
        }
    }

    /// Create scaler for 8M proxy (smallest for LR sweep)
    pub fn proxy_8m() -> Self {
        // 8M params ≈ d_model=128, n_layers=2, n_heads=4
        Self::new(128, 128)
    }

    /// Create scaler for 24M intermediate width
    pub fn proxy_24m() -> Self {
        // 24M params ≈ d_model=256, n_layers=3, n_heads=8
        Self::new(256, 128)
    }

    /// Create scaler for 70M Gate-2 target
    pub fn target_70m() -> Self {
        // 70M params ≈ d_model=384, n_layers=4, n_heads=12
        Self::new(384, 128)
    }

    /// Scale learning rate for a parameter group
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Base learning rate (from proxy sweep)
    /// * `group` - Parameter group type
    ///
    /// # Returns
    ///
    /// Scaled learning rate for this group
    pub fn scale_lr(&self, base_lr: f64, group: ParamGroup) -> f64 {
        match group {
            ParamGroup::Embedding => base_lr * self.embedding_lr_mult,
            ParamGroup::Output => base_lr * self.output_lr_mult,
            ParamGroup::Attention => base_lr * self.attn_lr_mult,
            ParamGroup::Hidden => base_lr,
        }
    }

    /// Validate that LR is in INV-8 band [1e-3, 1e-2]
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate to validate
    ///
    /// # Returns
    ///
    /// true if LR is in valid band, false otherwise
    pub fn validate_inv8(lr: f64) -> bool {
        lr >= 1e-3 && lr <= 1e-2
    }

    /// Compute width ratio for transfer analysis
    ///
    /// # Returns
    ///
    /// Ratio of current width to reference width
    pub fn width_ratio(&self) -> f64 {
        self.width as f64 / self.ref_width as f64
    }
}

/// Parameter group types for µP LR scaling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamGroup {
    /// Embedding layer (input tokens)
    Embedding,

    /// Output layer (logits)
    Output,

    /// Attention layers (Q, K, V projections)
    Attention,

    /// Hidden layers (MLP/FFN)
    Hidden,
}

/// µP LR sweep configuration for proxy experiments
///
/// Defines the LR search space and validation criteria.
#[derive(Debug, Clone)]
pub struct MupSweepConfig {
    /// Proxy model width (where LR sweep happens)
    pub proxy_width: usize,

    /// LR values to sweep (INV-8 band)
    pub lr_values: Vec<f64>,

    /// Target model width (for transfer validation)
    pub target_width: usize,

    /// Acceptable degradation threshold
    pub max_degradation_pct: f64,
}

impl Default for MupSweepConfig {
    fn default() -> Self {
        Self {
            proxy_width: 128, // 8M proxy
            lr_values: vec![1e-3, 2e-3, 4e-3, 8e-3, 1e-2],
            target_width: 384,        // 70M target
            max_degradation_pct: 5.0, // <5% degradation
        }
    }
}

impl MupSweepConfig {
    /// Create custom sweep config
    pub fn new(proxy_width: usize, target_width: usize) -> Self {
        Self {
            proxy_width,
            lr_values: vec![1e-3, 2e-3, 4e-3, 8e-3, 1e-2],
            target_width,
            max_degradation_pct: 5.0,
        }
    }

    /// Validate that all LR values are in INV-8 band
    ///
    /// # Returns
    ///
    /// true if all LR values are valid, false otherwise
    pub fn validate_lr_values(&self) -> bool {
        self.lr_values
            .iter()
            .all(|&lr| MupScaler::validate_inv8(lr))
    }

    /// Get the best LR from sweep results
    ///
    /// # Arguments
    ///
    /// * `results` - Slice of (lr, bpb) tuples from sweep runs
    ///
    /// # Returns
    ///
    /// The LR with minimum BPB
    pub fn pick_lr_star(&self, results: &[(f64, f64)]) -> Option<f64> {
        if results.is_empty() {
            return None;
        }
        results
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(lr, _)| *lr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mup_scaler_creation() {
        let scaler = MupScaler::new(256, 128);
        assert_eq!(scaler.width, 256);
        assert_eq!(scaler.ref_width, 128);
        assert!((scaler.input_mult - 2.0_f64.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_mup_scaler_champion() {
        let scaler = MupScaler::champion();
        assert_eq!(scaler.width, 256);
        assert_eq!(scaler.ref_width, 256);
        assert_eq!(scaler.input_mult, 1.0);
        assert_eq!(scaler.output_mult, 1.0);
    }

    #[test]
    fn test_mup_lr_scaling() {
        let scaler = MupScaler::new(256, 128);
        let base_lr = 0.004;

        let embedding_lr = scaler.scale_lr(base_lr, ParamGroup::Embedding);
        let hidden_lr = scaler.scale_lr(base_lr, ParamGroup::Hidden);

        assert!(embedding_lr < base_lr, "Embedding LR should scale down");
        assert_eq!(hidden_lr, base_lr, "Hidden LR should be unchanged");
    }

    #[test]
    fn test_inv8_validation() {
        assert!(MupScaler::validate_inv8(0.001));
        assert!(MupScaler::validate_inv8(0.004));
        assert!(MupScaler::validate_inv8(0.010));
        assert!(!MupScaler::validate_inv8(0.0005));
        assert!(!MupScaler::validate_inv8(0.020));
    }

    #[test]
    fn test_mup_sweep_config_default() {
        let config = MupSweepConfig::default();
        assert_eq!(config.proxy_width, 128);
        assert_eq!(config.target_width, 384);
        assert!(config.validate_lr_values());
    }

    #[test]
    fn test_mup_pick_lr_star() {
        let config = MupSweepConfig::default();
        let results = vec![(0.002, 2.5), (0.004, 2.3), (0.008, 2.7)];
        let lr_star = config.pick_lr_star(&results);
        assert_eq!(lr_star, Some(0.004));
    }

    #[test]
    fn test_proxy_scalers() {
        let proxy_8m = MupScaler::proxy_8m();
        assert_eq!(proxy_8m.width, 128);

        let proxy_24m = MupScaler::proxy_24m();
        assert_eq!(proxy_24m.width, 256);

        let target_70m = MupScaler::target_70m();
        assert_eq!(target_70m.width, 384);
    }

    #[test]
    fn test_width_ratio() {
        let scaler = MupScaler::new(256, 128);
        assert!((scaler.width_ratio() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_attn_mult_scaling() {
        let scaler = MupScaler::new(256, 128);
        // attn_mult = 1/√d_head = 1/√256 = 1/16
        let expected = 1.0 / (256.0_f64).sqrt();
        assert!((scaler.attn_mult - expected).abs() < 1e-6);
    }

    #[test]
    fn test_param_group_equality() {
        assert_eq!(ParamGroup::Embedding, ParamGroup::Embedding);
        assert_ne!(ParamGroup::Embedding, ParamGroup::Output);
    }
}
