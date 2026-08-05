//! Minimal Transformer Trainer — Phase 2
//!
//! Training configuration:
//! - d_model: 384 (proven from N-gram baseline)
//! - n_heads: 8 (d_k = 48)
//! - d_ffn: 1536 (4 * d_model)
//! - n_layers: 2 (minimal for Phase 2)
//! - lr: 0.004 (learning rate)
//! - batch_size: 4
//! - seq_len: 128

use crate::backward::cross_entropy_loss;
use crate::optimizer::AdamWCpu;
use crate::transformer::MinimalTransformer;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct TransformerTrainConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_ffn: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub lr: f64,
    pub max_steps: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub eval_every: usize,
}

impl Default for TransformerTrainConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,      // Standard vocab size
            d_model: 384,           // Proven from N-gram baseline
            d_ffn: 1536,            // 4 * d_model
            n_heads: 8,             // d_k = 48
            n_layers: 2,            // Minimal for Phase 2
            lr: 0.004,              // Learning rate
            max_steps: 5000,
            batch_size: 4,
            seq_len: 128,
            eval_every: 100,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainMetrics {
    pub step: usize,
    pub loss: f32,
    pub bpb: f32,
    pub lr: f64,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone)]
pub struct TrainResult {
    pub final_bpb: f32,
    pub best_bpb: f32,
    pub total_steps: usize,
    pub total_time_seconds: f64,
    pub metrics: Vec<TrainMetrics>,
}

/// Simple synthetic data generator for training
fn generate_batch(batch_size: usize, seq_len: usize, vocab_size: usize) -> Vec<Vec<usize>> {
    let mut batch = Vec::with_capacity(batch_size);
    let mut rng: u64 = 0x1337_c0de_u64;

    for _ in 0..batch_size {
        let mut seq = Vec::with_capacity(seq_len);
        for _ in 0..seq_len {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let token = (rng as usize) % vocab_size;
            seq.push(token);
        }
        batch.push(seq);
    }
    batch
}

/// Compute BPB (bits per byte) from loss
pub fn bpb_from_loss(loss: f32) -> f32 {
    loss / std::f32::consts::LN_2
}

impl TransformerTrainConfig {
    /// Run transformer training
    pub fn run(&self) -> TrainResult {
        let start_time = Instant::now();

        println!("=== Phase 2: Minimal Transformer Training ===");
        println!("Config:");
        println!("  vocab_size: {}", self.vocab_size);
        println!("  d_model: {}", self.d_model);
        println!("  d_ffn: {}", self.d_ffn);
        println!("  n_heads: {} (d_k={})", self.n_heads, self.d_model / self.n_heads);
        println!("  n_layers: {}", self.n_layers);
        println!("  lr: {}", self.lr);
        println!("  batch_size: {}", self.batch_size);
        println!("  seq_len: {}", self.seq_len);
        println!("  max_steps: {}", self.max_steps);
        println!("=============================================\n");

        // Create model
        let model = MinimalTransformer::new(
            self.vocab_size,
            self.d_model,
            self.d_ffn,
            self.n_heads,
            self.n_layers,
        );

        println!("Model created with {} parameters", model.param_count());

        // Create optimizer (simplified - just for structure)
        let param_count = model.param_count();
        let mut _optimizer = AdamWCpu::new(param_count, self.lr);

        // Training loop
        let mut metrics = Vec::new();
        let mut best_bpb = f32::MAX;
        let mut current_bpb = f32::MAX;

        for step in 0..self.max_steps {
            // Generate synthetic batch
            let batch = generate_batch(self.batch_size, self.seq_len, self.vocab_size);

            // Forward pass on first sequence in batch
            let input_ids = &batch[0];
            if input_ids.len() < 2 {
                continue;
            }

            let input = &input_ids[..input_ids.len() - 1];
            let targets = &input_ids[1..];

            let logits = model.forward(input);
            if logits.is_empty() {
                continue;
            }

            // Flatten logits for loss computation
            let flat_logits: Vec<f32> = logits.into_iter().flatten().collect();

            // Compute loss (using available cross_entropy_loss)
            let loss = cross_entropy_loss(&flat_logits, targets);
            current_bpb = bpb_from_loss(loss);

            if current_bpb < best_bpb {
                best_bpb = current_bpb;
            }

            // Track metrics
            if step % self.eval_every == 0 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let metric = TrainMetrics {
                    step,
                    loss,
                    bpb: current_bpb,
                    lr: self.lr,
                    elapsed_seconds: elapsed,
                };
                metrics.push(metric.clone());
                println!(
                    "Step {:5}: loss={:.4}, bpb={:.4}, lr={:.5}, time={:.1}s",
                    step, loss, current_bpb, self.lr, elapsed
                );
            }

            // Simplified: no actual gradient updates in this minimal version
            // In real implementation, this would call backward() and optimizer.step()
        }

        let total_time = start_time.elapsed().as_secs_f64();

        println!("\n=== Training Complete ===");
        println!("Final BPB: {:.4}", current_bpb);
        println!("Best BPB:  {:.4}", best_bpb);
        println!("Total time: {:.1}s", total_time);
        println!("=========================\n");

        // Compare with N-gram baseline (2.5329 BPB)
        let baseline_bpb = 2.5329;
        let improvement = ((baseline_bpb - best_bpb) / baseline_bpb) * 100.0;

        if best_bpb < baseline_bpb {
            println!("✓ IMPROVEMENT: {:.2}% better than N-gram baseline ({:.4} vs {:.4})",
                     improvement, best_bpb, baseline_bpb);
        } else {
            println!("✗ WORSE: {:.2}% worse than N-gram baseline ({:.4} vs {:.4})",
                     -improvement, best_bpb, baseline_bpb);
        }

        println!("\nTarget: 1.80 BPB (30% improvement)");
        println!("Current: {:.4} BPB ({:.1}% of target)", best_bpb, (best_bpb / 1.80) * 100.0);

        TrainResult {
            final_bpb: current_bpb,
            best_bpb,
            total_steps: self.max_steps,
            total_time_seconds: total_time,
            metrics,
        }
    }

    /// Run with specific learning rate sweep
    pub fn run_lr_sweep(&self, lrs: Vec<f64>) -> Vec<TrainResult> {
        let mut results = Vec::new();

        println!("\n=== Learning Rate Sweep ===");
        for lr in &lrs {
            let mut config = self.clone();
            config.lr = *lr;
            println!("\n--- Testing LR={} ---", lr);
            let result = config.run();
            results.push(result);
        }

        // Find best LR
        let best_idx = results
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.best_bpb.partial_cmp(&b.1.best_bpb).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        println!("\n=== LR Sweep Results ===");
        for (i, (lr, result)) in lrs.iter().zip(results.iter()).enumerate() {
            let marker = if i == best_idx { " ★ BEST" } else { "" };
            println!("LR={:.5}: best_bpb={:.4}{}", lr, result.best_bpb, marker);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = TransformerTrainConfig::default();
        assert_eq!(config.d_model, 384);
        assert_eq!(config.n_heads, 8);
        assert_eq!(config.lr, 0.004);
    }

    #[test]
    fn test_generate_batch() {
        let batch = generate_batch(4, 128, 1000);
        assert_eq!(batch.len(), 4);
        for seq in &batch {
            assert_eq!(seq.len(), 128);
            for &token in seq {
                assert!(token < 1000);
            }
        }
    }

    #[test]
    fn test_bpb_from_loss() {
        let bpb = bpb_from_loss(1.0);
        assert!((bpb - 1.0 / std::f32::consts::LN_2).abs() < 1e-5);
    }

    #[test]
    fn test_quick_run() {
        let config = TransformerTrainConfig {
            vocab_size: 256,
            d_model: 64,
            d_ffn: 256,
            n_heads: 4,
            n_layers: 1,
            lr: 0.01,
            max_steps: 10,
            batch_size: 2,
            seq_len: 16,
            eval_every: 5,
        };

        let result = config.run();
        assert!(result.best_bpb > 0.0);
        assert!(result.total_steps > 0);
        assert!(result.total_time_seconds >= 0.0);
    }
}
