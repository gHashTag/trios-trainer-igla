//! Checkpoint save/load. Stored under `${TRIOS_CHECKPOINT_DIR}/{run-name}/{step}.bin`.
//! Resume sets `step_done` and re-attaches optimizer state.
//!
//! P4 addition: `ema_average` for post-hoc EMA over last N checkpoints.
//! Reference: Sanyal et al. 2024 — free generalization gain at zero training cost.

use anyhow::Result;
use std::path::PathBuf;

pub fn checkpoint_path(run_name: &str, step: usize) -> PathBuf {
    let base = std::env::var("TRIOS_CHECKPOINT_DIR").unwrap_or_else(|_| "checkpoints".into());
    PathBuf::from(base)
        .join(run_name)
        .join(format!("{step}.bin"))
}

pub fn save(_run: &str, _step: usize, _bytes: &[u8]) -> Result<()> {
    // TODO: actually persist (zstd-compressed bincode). Non-blocking for skeleton.
    Ok(())
}

/// EMA checkpoint averaging result.
#[derive(Debug, Clone)]
pub struct EmaCheckpoint {
    pub weights: Vec<f32>,
    pub averaged_count: usize,
    pub effective_decay: f64,
}

/// Post-hoc EMA over the last `n` checkpoint weight vectors.
///
/// Each checkpoint is weighted by its step number (later = heavier),
/// with exponential decay: weight_i = decay^(n - 1 - i).
///
/// Reference: Sanyal et al. 2024 — EMA of last N checkpoints improves
/// generalization by 0.03+ BPB at zero training cost.
///
/// # Arguments
/// * `checkpoints` - Weight vectors from N consecutive checkpoints
/// * `decay` - EMA decay factor (0.999 typical). Higher = more weight on earlier.
/// * `steps` - Step numbers corresponding to each checkpoint
///
/// # Returns
/// EMA-averaged weight vector, or error if checkpoints is empty.
pub fn ema_average(
    checkpoints: &[Vec<f32>],
    decay: f64,
    steps: &[usize],
) -> Result<EmaCheckpoint> {
    anyhow::ensure!(!checkpoints.is_empty(), "need at least 1 checkpoint for EMA");
    anyhow::ensure!(
        checkpoints.len() == steps.len(),
        "checkpoints ({}) and steps ({}) must have same length",
        checkpoints.len(), steps.len()
    );

    let n = checkpoints.len();
    let dim = checkpoints[0].len();

    for (i, ckpt) in checkpoints.iter().enumerate() {
        anyhow::ensure!(
            ckpt.len() == dim,
            "checkpoint {} has dim {}, expected {}",
            i, ckpt.len(), dim
        );
    }

    let mut ema = vec![0.0f64; dim];
    let mut total_weight = 0.0f64;

    for (i, (ckpt, &step)) in checkpoints.iter().zip(steps.iter()).enumerate() {
        let step_weight = step as f64;
        let positional_weight = decay.powi((n - 1 - i) as i32);
        let w = step_weight * positional_weight;
        total_weight += w;
        for j in 0..dim {
            ema[j] += w * ckpt[j] as f64;
        }
    }

    if total_weight > 0.0 {
        for v in ema.iter_mut() {
            *v /= total_weight;
        }
    }

    let weights: Vec<f32> = ema.iter().map(|&v| v as f32).collect();

    Ok(EmaCheckpoint {
        weights,
        averaged_count: n,
        effective_decay: decay,
    })
}

/// Sweep EMA over different N values (P4 requirement).
///
/// Tries N in {3, 5, 10, 20} and returns all results.
/// Caller picks the best one (lowest validation loss).
pub fn ema_sweep(
    all_checkpoints: &[Vec<f32>],
    all_steps: &[usize],
    decay: f64,
) -> Vec<(usize, EmaCheckpoint)> {
    let n_values = [3usize, 5, 10, 20];
    let mut results = Vec::new();

    for &n in &n_values {
        if n > all_checkpoints.len() { continue; }
        let start = all_checkpoints.len() - n;
        let ckpts = &all_checkpoints[start..];
        let steps = &all_steps[start..];
        if let Ok(ema) = ema_average(ckpts, decay, steps) {
            results.push((n, ema));
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_average_single_checkpoint() {
        let ckpts = vec![vec![1.0f32, 2.0, 3.0]];
        let steps = vec![1000];
        let ema = ema_average(&ckpts, 0.999, &steps).unwrap();
        assert!((ema.weights[0] - 1.0).abs() < 1e-3);
        assert!((ema.weights[1] - 2.0).abs() < 1e-3);
        assert_eq!(ema.averaged_count, 1);
    }

    #[test]
    fn test_ema_average_two_checkpoints() {
        let ckpts = vec![vec![0.0f32], vec![2.0f32]];
        let steps = vec![1000, 2000];
        let ema = ema_average(&ckpts, 0.999, &steps).unwrap();
        assert!(ema.weights[0] > 0.0 && ema.weights[0] < 2.0);
        assert_eq!(ema.averaged_count, 2);
    }

    #[test]
    fn test_ema_average_later_heavier() {
        let ckpts = vec![vec![0.0f32], vec![10.0f32]];
        let steps = vec![100, 100];
        let ema = ema_average(&ckpts, 0.999, &steps).unwrap();
        assert!(ema.weights[0] > 5.0, "later checkpoint should weigh more");
    }

    #[test]
    fn test_ema_empty_fails() {
        let result = ema_average(&[], 0.999, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ema_dimension_mismatch_fails() {
        let ckpts = vec![vec![1.0f32], vec![1.0f32, 2.0f32]];
        let steps = vec![100, 200];
        let result = ema_average(&ckpts, 0.999, &steps);
        assert!(result.is_err());
    }

    #[test]
    fn test_ema_sweep() {
        let ckpts: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32]).collect();
        let steps: Vec<usize> = (0..20).map(|i| (i + 1) * 500).collect();
        let results = ema_sweep(&ckpts, &steps, 0.999);
        assert!(results.iter().any(|(n, _)| *n == 3));
        assert!(results.len() <= 4);
    }

    #[test]
    fn test_checkpoint_path() {
        let p = checkpoint_path("test-run", 5000);
        assert!(p.to_string_lossy().contains("test-run"));
        assert!(p.to_string_lossy().contains("5000"));
    }
}
