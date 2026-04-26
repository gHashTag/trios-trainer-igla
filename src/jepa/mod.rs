//! T-JEPA (Ternary Joint Embedding Predictive Architecture)
//!
//! Implements masked prediction with EMA target encoder.
//! Based on LeJEPA/LeWorldModel principles.

pub mod ema;
pub mod loss;
pub mod masking;
pub mod predictor;

// Re-export common types
pub use ema::{compute_decay, ema_update, EmaConfig, EmaTarget};
pub use loss::{
    compute_jepa_loss, cosine_similarity, l2_normalize, mse_loss, JepaLoss, JepaLossConfig,
};
pub use masking::{get_masked, get_unmasked, mask_spans, MaskConfig, MaskResult};
pub use predictor::{PredictionOutput, Predictor, PredictorConfig};

/// JEPA training configuration
#[derive(Debug, Clone)]
pub struct JepaConfig {
    pub seed: u64,
    pub d_model: usize,
    pub mask_ratio: f64,
    pub min_span: usize,
    pub max_span: usize,
    pub num_spans: usize,
    pub ema_start: f64,
    pub ema_end: f64,
    pub ema_ramp_steps: usize,
    pub predictor_lr_mult: f64,
}

impl Default for JepaConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            d_model: 384,
            mask_ratio: 0.3,
            min_span: 3,
            max_span: 9,
            num_spans: 2,
            ema_start: 0.996,
            ema_end: 1.0,
            ema_ramp_steps: 30000,
            predictor_lr_mult: 0.1,
        }
    }
}

impl JepaConfig {
    /// Create a new config with custom d_model
    pub fn with_d_model(d_model: usize) -> Self {
        Self {
            d_model,
            ..Default::default()
        }
    }

    /// Get EMA config
    pub fn ema_config(&self) -> EmaConfig {
        EmaConfig {
            start: self.ema_start,
            end: self.ema_end,
            ramp_steps: self.ema_ramp_steps,
        }
    }

    /// Get mask config
    pub fn mask_config(&self) -> MaskConfig {
        MaskConfig {
            ratio: self.mask_ratio,
            min_span: self.min_span,
            max_span: self.max_span,
            num_spans: self.num_spans,
        }
    }
}

/// JEPA training result
#[derive(Debug, Clone)]
pub struct JepaResult {
    pub steps_completed: usize,
    pub final_loss: f64,
    pub final_variance: f64,
    pub loss_monotone: bool,
    pub ema_verified: bool,
    pub converged: bool,
}

impl JepaResult {
    /// Create a new result
    pub fn new(
        steps_completed: usize,
        final_loss: f64,
        final_variance: f64,
        loss_monotone: bool,
        ema_verified: bool,
    ) -> Self {
        let converged = final_loss < 1.0 && final_variance > 0.01;

        Self {
            steps_completed,
            final_loss,
            final_variance,
            loss_monotone,
            ema_verified,
            converged,
        }
    }

    /// Check if training was successful
    pub fn is_success(&self) -> bool {
        self.converged && self.ema_verified
    }
}

/// Architecture kind for IGLA Race
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchKind {
    Ngram,
    Jepa,
    Attention,
    Hybrid,
}

impl ArchKind {
    /// Get minimum rung for this architecture
    ///
    /// JEPA requires more steps for initial convergence
    pub fn min_rung(&self) -> i32 {
        match self {
            ArchKind::Jepa => 3000,
            _ => 1000,
        }
    }

    /// Get rung schedule for this architecture
    pub fn rung_schedule(&self) -> Vec<i32> {
        match self {
            ArchKind::Jepa => vec![3000, 9000, 27000],
            _ => vec![1000, 3000, 9000, 27000],
        }
    }

    /// Parse from string
    pub fn parse_arch(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ngram" => Some(ArchKind::Ngram),
            "jepa" => Some(ArchKind::Jepa),
            "attn" | "attention" => Some(ArchKind::Attention),
            "hybrid" => Some(ArchKind::Hybrid),
            _ => None,
        }
    }

    /// Convert to string
    pub fn as_str(&self) -> &'static str {
        match self {
            ArchKind::Ngram => "ngram",
            ArchKind::Jepa => "jepa",
            ArchKind::Attention => "attn",
            ArchKind::Hybrid => "hybrid",
        }
    }
}

impl std::fmt::Display for ArchKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jepa_config_default() {
        let config = JepaConfig::default();

        assert_eq!(config.seed, 42);
        assert_eq!(config.d_model, 384);
        assert_eq!(config.mask_ratio, 0.3);
        assert_eq!(config.ema_start, 0.996);
        assert_eq!(config.ema_end, 1.0);
    }

    #[test]
    fn test_jepa_config_with_d_model() {
        let config = JepaConfig::with_d_model(256);

        assert_eq!(config.d_model, 256);
        assert_eq!(config.mask_ratio, 0.3);
    }

    #[test]
    fn test_jepa_config_ema_config() {
        let config = JepaConfig::default();
        let ema_config = config.ema_config();

        assert_eq!(ema_config.start, 0.996);
        assert_eq!(ema_config.end, 1.0);
        assert_eq!(ema_config.ramp_steps, 30000);
    }

    #[test]
    fn test_jepa_config_mask_config() {
        let config = JepaConfig::default();
        let mask_config = config.mask_config();

        assert_eq!(mask_config.ratio, 0.3);
        assert_eq!(mask_config.min_span, 3);
        assert_eq!(mask_config.max_span, 9);
        assert_eq!(mask_config.num_spans, 2);
    }

    #[test]
    fn test_jepa_result_new() {
        let result = JepaResult::new(1000, 0.8, 0.05, true, true);

        assert_eq!(result.steps_completed, 1000);
        assert_eq!(result.final_loss, 0.8);
        assert_eq!(result.final_variance, 0.05);
        assert!(result.loss_monotone);
        assert!(result.ema_verified);
        assert!(result.converged);
    }

    #[test]
    fn test_jepa_result_not_converged() {
        let result = JepaResult::new(1000, 1.5, 0.005, true, true);

        assert!(!result.converged); // Variance too low (collapse)
        assert!(!result.is_success());
    }

    #[test]
    fn test_arch_kind_min_rung() {
        assert_eq!(ArchKind::Ngram.min_rung(), 1000);
        assert_eq!(ArchKind::Jepa.min_rung(), 3000);
        assert_eq!(ArchKind::Attention.min_rung(), 1000);
        assert_eq!(ArchKind::Hybrid.min_rung(), 1000);
    }

    #[test]
    fn test_arch_kind_rung_schedule() {
        let ngram_schedule = ArchKind::Ngram.rung_schedule();
        assert_eq!(ngram_schedule, vec![1000, 3000, 9000, 27000]);

        let jepa_schedule = ArchKind::Jepa.rung_schedule();
        assert_eq!(jepa_schedule, vec![3000, 9000, 27000]);
    }

    #[test]
    fn test_arch_kind_parse_arch() {
        assert_eq!(ArchKind::parse_arch("ngram"), Some(ArchKind::Ngram));
        assert_eq!(ArchKind::parse_arch("NGRAM"), Some(ArchKind::Ngram));
        assert_eq!(ArchKind::parse_arch("jepa"), Some(ArchKind::Jepa));
        assert_eq!(ArchKind::parse_arch("attn"), Some(ArchKind::Attention));
        assert_eq!(ArchKind::parse_arch("attention"), Some(ArchKind::Attention));
        assert_eq!(ArchKind::parse_arch("hybrid"), Some(ArchKind::Hybrid));
        assert_eq!(ArchKind::parse_arch("unknown"), None);
    }

    #[test]
    fn test_arch_kind_as_str() {
        assert_eq!(ArchKind::Ngram.as_str(), "ngram");
        assert_eq!(ArchKind::Jepa.as_str(), "jepa");
        assert_eq!(ArchKind::Attention.as_str(), "attn");
        assert_eq!(ArchKind::Hybrid.as_str(), "hybrid");
    }

    #[test]
    fn test_arch_kind_display() {
        assert_eq!(format!("{}", ArchKind::Jepa), "jepa");
        assert_eq!(format!("{}", ArchKind::Ngram), "ngram");
    }

    #[test]
    fn test_arch_kind_equality() {
        assert_eq!(ArchKind::Jepa, ArchKind::Jepa);
        assert_ne!(ArchKind::Jepa, ArchKind::Ngram);
    }
}
