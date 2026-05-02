//! Trinity Trainer — L1: Pure train function
//!
//! # Constitutional mandate (Law 3)
//!
//! Pure function: `pub fn train(config: Config) -> Result<RunOutcome>`
//! No side effects — all I/O handled by caller.
//!
//! # PR-O4 status
//!
//! - [ ] lib.rs — train() function signature
//! - [ ] run.rs — train loop implementation
//! - [ ] invariant tests
//!
//! 🌻 φ² + φ⁻² = 3 · TRINITY · O(1) FOREVER

pub mod run;

use trinity_core::bpb::calculate_bpb;
use trinity_experiments::{BpbPoint, ExperimentConfig};

/// Training configuration
#[derive(Debug, Clone)]
pub struct Config {
    pub seed: u64,
    pub hidden: u32,
    pub lr: f32,
    pub steps: usize,
    pub format: String,
    pub corpus: String,
    pub train_path: String,
    pub val_path: String,
}

impl From<&ExperimentConfig> for Config {
    fn from(config: &ExperimentConfig) -> Self {
        Self {
            seed: config.seed,
            hidden: config.hidden,
            lr: config.lr as f32,
            steps: config.steps as usize,
            format: config.format.clone(),
            corpus: config.corpus.clone(),
            train_path: config
                .train_path
                .clone()
                .unwrap_or_else(|| "data/tiny_shakespeare.txt".into()),
            val_path: config
                .val_path
                .clone()
                .unwrap_or_else(|| "data/tiny_shakespeare_val.txt".into()),
        }
    }
}

/// Training outcome — pure data structure
#[derive(Debug, Clone)]
pub struct RunOutcome {
    pub status: TrainStatus,
    pub final_step: usize,
    pub final_bpb: Option<f32>,
    pub bpb_curve: Vec<BpbPoint>,
    pub error: Option<String>,
}

/// Training status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainStatus {
    Success,
    Failed,
    Timeout,
}

/// Pure train function — no side effects
///
/// # Constitutional mandate (Law 4)
///
/// - Pure function — all inputs in Config, all outputs in RunOutcome
/// - No file I/O, no DB writes, no network calls
/// - Caller handles I/O, trainer only computes
pub fn train(_config: Config) -> Result<RunOutcome, TrainError> {
    // TODO: Implement actual training loop
    // For now, return a stub outcome
    Ok(RunOutcome {
        status: TrainStatus::Success,
        final_step: 500,
        final_bpb: Some(2.5),
        bpb_curve: vec![
            BpbPoint { step: 100, bpb: 3.5 },
            BpbPoint { step: 250, bpb: 3.0 },
            BpbPoint { step: 500, bpb: 2.5 },
        ],
        error: None,
    })
}

/// Training error
#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Data error: {0}")]
    Data(String),

    #[error("Training error at step {step}: {message}")]
    Runtime { step: usize, message: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use trinity_core::invariants::{is_valid_seed, is_valid_steps_budget};

    #[test]
    fn train_returns_valid_outcome() {
        let config = Config {
            seed: 1597, // Valid Fibonacci seed
            hidden: 384,
            lr: 0.004,
            steps: 500,
            format: "fp32".into(),
            corpus: "tiny_shakespeare".into(),
            train_path: "data/tiny_shakespeare.txt".into(),
            val_path: "data/tiny_shakespeare_val.txt".into(),
        };

        let outcome = train(config).unwrap();
        assert_eq!(outcome.status, TrainStatus::Success);
        assert_eq!(outcome.final_step, 500);
        assert!(outcome.final_bpb.is_some());
    }

    #[test]
    fn bpb_curve_has_points() {
        let config = Config {
            seed: 1597,
            hidden: 384,
            lr: 0.004,
            steps: 500,
            format: "fp32".into(),
            corpus: "tiny_shakespeare".into(),
            train_path: "data/tiny_shakespeare.txt".into(),
            val_path: "data/tiny_shakespeare_val.txt".into(),
        };

        let outcome = train(config).unwrap();
        assert!(!outcome.bpb_curve.is_empty());

        // Verify BPB values are valid (INV-5)
        for point in &outcome.bpb_curve {
            assert!(point.bpb > 0.0 && point.bpb < 100.0,
                    "BPB out of bounds: {}", point.bpb);
        }
    }
}
