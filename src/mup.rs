//! Maximal Update Parameterization (muP) for hyperparameter transfer.
//!
//! Reference: Yang et al. 2022 "Tensor Programs V: Tuning Large Neural Networks
//! via Zero-Shot Hyperparameter Transfer", Cerebras muP Practitioner Guide.
//!
//! Key idea: optimal LR found at a small proxy width transfers verbatim to
//! wider models WITHOUT re-sweeping. This enables cheap HP search at 8M params
//! that transfers to 24M, 70M, etc.
//!
//! Scaling rules:
//! - Embedding LR: scaled by `embedding_mult` (typically 1/width_ratio)
//! - Attention QK: 1/d_head scaling (preserved from base)
//! - Output LR: scaled by `output_mult` (typically width_ratio)
//! - Hidden layers: LR unchanged (muP core theorem)

/// muP configuration for parameter-group LR scaling.
#[derive(Debug, Clone)]
pub struct MuPConfig {
    /// Reference (proxy) width in d_model.
    pub proxy_d_model: usize,
    /// Target width in d_model.
    pub target_d_model: usize,
    /// Embedding LR multiplier (default: 1.0).
    pub embedding_mult: f64,
    /// Output head LR multiplier (default: 1.0).
    pub output_mult: f64,
    /// Attention QK LR multiplier (default: 1.0).
    pub attn_mult: f64,
    /// Base learning rate found at proxy width.
    pub base_lr: f64,
}

impl Default for MuPConfig {
    fn default() -> Self {
        Self {
            proxy_d_model: 64,
            target_d_model: 64,
            embedding_mult: 1.0,
            output_mult: 1.0,
            attn_mult: 1.0,
            base_lr: 0.004,
        }
    }
}

impl MuPConfig {
    pub fn new(proxy_d: usize, target_d: usize, base_lr: f64) -> Self {
        let ratio = target_d as f64 / proxy_d as f64;
        Self {
            proxy_d_model: proxy_d,
            target_d_model: target_d,
            embedding_mult: 1.0 / ratio,
            output_mult: ratio,
            attn_mult: 1.0,
            base_lr,
        }
    }

    /// Compute per-group learning rates according to muP scaling rules.
    pub fn scaled_lrs(&self) -> MuPLRScales {
        MuPLRScales {
            embedding_lr: self.base_lr * self.embedding_mult,
            hidden_lr: self.base_lr,
            attention_lr: self.base_lr * self.attn_mult,
            output_lr: self.base_lr * self.output_mult,
        }
    }

    /// Validate INV-8: all scaled LRs must be in [1e-3, 1e-2].
    /// Returns Ok(()) if valid, Err with details otherwise.
    pub fn validate_inv8(&self) -> Result<(), String> {
        let scales = self.scaled_lrs();
        let check = |name: &str, lr: f64| -> Result<(), String> {
            if !(1e-3..=1e-2).contains(&lr) {
                Err(format!(
                    "INV-8 violation: {}_lr={} not in [1e-3, 1e-2]",
                    name, lr
                ))
            } else {
                Ok(())
            }
        };
        check("embedding", scales.embedding_lr)?;
        check("hidden", scales.hidden_lr)?;
        check("attention", scales.attention_lr)?;
        check("output", scales.output_lr)?;
        Ok(())
    }

    /// Width ratio (target / proxy).
    pub fn width_ratio(&self) -> f64 {
        self.target_d_model as f64 / self.proxy_d_model as f64
    }

    /// Check if this config represents a transfer (proxy != target).
    pub fn is_transfer(&self) -> bool {
        self.proxy_d_model != self.target_d_model
    }

    /// Initialize weights with muP-correct scaling for a given layer.
    /// Returns the standard deviation for Xavier/He init.
    pub fn init_std(&self, fan_in: usize, fan_out: usize, layer_type: LayerType) -> f64 {
        let d = self.target_d_model as f64;
        match layer_type {
            LayerType::Embedding => (2.0 / (fan_in + fan_out) as f64).sqrt(),
            LayerType::Hidden => (2.0 / fan_in as f64).sqrt() / d.sqrt(),
            LayerType::Attention => (2.0 / (fan_in + fan_out) as f64).sqrt(),
            LayerType::Output => (2.0 / fan_in as f64).sqrt(),
        }
    }
}

/// Per-parameter-group LR scales from muP.
#[derive(Debug, Clone, Copy)]
pub struct MuPLRScales {
    pub embedding_lr: f64,
    pub hidden_lr: f64,
    pub attention_lr: f64,
    pub output_lr: f64,
}

/// Layer type for muP-correct initialization.
#[derive(Debug, Clone, Copy)]
pub enum LayerType {
    Embedding,
    Hidden,
    Attention,
    Output,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mup_default_no_scaling() {
        let mup = MuPConfig::default();
        let scales = mup.scaled_lrs();
        assert!((scales.embedding_lr - 0.004).abs() < 1e-9);
        assert!((scales.hidden_lr - 0.004).abs() < 1e-9);
        assert!((scales.output_lr - 0.004).abs() < 1e-9);
    }

    #[test]
    fn test_mup_transfer_8m_to_24m() {
        let mup = MuPConfig::new(64, 128, 0.004);
        assert!((mup.width_ratio() - 2.0).abs() < 1e-9);
        assert!(mup.is_transfer());
        let scales = mup.scaled_lrs();
        assert!((scales.embedding_lr - 0.002).abs() < 1e-9);
        assert!((scales.hidden_lr - 0.004).abs() < 1e-9);
        assert!((scales.output_lr - 0.008).abs() < 1e-9);
    }

    #[test]
    fn test_mup_transfer_8m_to_70m() {
        let mup = MuPConfig::new(64, 384, 0.004);
        assert!((mup.width_ratio() - 6.0).abs() < 1e-3);
        let scales = mup.scaled_lrs();
        assert!((scales.embedding_lr - 0.004 / 6.0).abs() < 1e-9);
        assert!((scales.output_lr - 0.004 * 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_mup_inv8_validation_passes() {
        let mup = MuPConfig::new(64, 64, 0.004);
        assert!(mup.validate_inv8().is_ok());
    }

    #[test]
    fn test_mup_inv8_violation_embedding() {
        let mup = MuPConfig::new(64, 512, 0.004);
        let scales = mup.scaled_lrs();
        assert!(scales.embedding_lr < 1e-3);
    }

    #[test]
    fn test_init_std_embedding() {
        let mup = MuPConfig::new(64, 128, 0.004);
        let std = mup.init_std(256, 128, LayerType::Embedding);
        assert!(std > 0.0 && std < 0.2);
    }

    #[test]
    fn test_init_std_hidden_scaled() {
        let mup = MuPConfig::new(64, 128, 0.004);
        let std = mup.init_std(128, 256, LayerType::Hidden);
        let expected = (2.0 / 128.0_f64).sqrt() / (128.0_f64).sqrt();
        assert!((std - expected).abs() < 1e-9);
    }
}
