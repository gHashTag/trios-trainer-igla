//! CPU training benchmark for IGLA-GF16
//!
//! Runs a complete training benchmark and saves results to JSON.

use std::env;
use std::io::Write;
use trios_trainer::{
    estimate_model_size, train_cpu_trace, TrainConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("IGLA-GF16 CPU Training Benchmark");
    println!("===============================");
    println!();

    // Parse command line args
    let args: Vec<String> = env::args().collect();
    let max_steps = if args.len() > 1 {
        args[1].parse::<usize>()?
    } else {
        1000
    };

    // Create training configuration
    let config = TrainConfig {
        max_steps,
        batch_size: 4,
        seq_len: 128,
        learning_rate: 0.001,
        warmup_steps: 21,   // Fib #7
        grad_clip: 0.618,   // phi^-1
        log_every: 34,      // Fib #8
        checkpoint_path: "checkpoints/igla-gf16-cpu.bin".to_string(),
        ..Default::default()
    };

    // Print configuration
    println!("Configuration:");
    println!("  max_steps: {}", config.max_steps);
    println!("  batch_size: {}", config.batch_size);
    println!("  seq_len: {}", config.seq_len);
    println!("  learning_rate: {}", config.learning_rate);
    println!("  warmup_steps: {}", config.warmup_steps);
    println!("  grad_clip: {}", config.grad_clip);
    println!("  log_every: {}", config.log_every);
    println!();

    // Estimate model size
    let vocab_size = 32000;
    let dims = config.dims;
    let model_size_bytes = estimate_model_size(vocab_size, dims.d_model, 7, dims.d_ffn);
    let model_size_mb = model_size_bytes as f64 / (1024.0 * 1024.0);

    println!("Model Size Estimate:");
    println!("  vocab_size: {}", vocab_size);
    println!("  d_model: {}", dims.d_model);
    println!("  n_heads: {}", dims.n_heads);
    println!("  d_ffn: {}", dims.d_ffn);
    println!("  estimated_size: {:.2} MB", model_size_mb);
    println!();

    // Run training with trace capture
    println!("Training Loop:");
    println!("-------------");

    let mut stdout = std::io::stdout();

    let run = train_cpu_trace(&config, vocab_size, |trace| {
        // Print metrics to console
        writeln!(
            stdout,
            "step={:5} loss={:.4} bpb={:.4} {:.0}ms/step lr={:.6}",
            trace.step, trace.loss, trace.bpb, trace.ms_per_step, trace.lr
        )
        .ok();
        stdout.flush().ok();
    });

    println!();
    println!("Training completed!");
    println!();

    // Print final summary
    println!("Final Metrics:");
    println!("  Final BPB: {:.4}", run.metrics.final_bpb);
    println!("  Final Loss: {:.4}", run.metrics.final_loss);
    println!("  Total Time: {:.2}s", run.metrics.total_time_seconds);
    println!("  Avg ms/step: {:.2}", run.metrics.avg_ms_per_step);
    println!("  Checkpoint Size: {:.2} MB",
        run.metrics.checkpoint_size_bytes as f64 / (1024.0 * 1024.0));
    println!();

    // Save results to JSON
    let json_path = run.save_to_file()?;
    println!("Results saved to: {}", json_path.display());
    println!();

    // Check if we met the target
    println!("Target Check:");
    let bpb_target = 2.0;
    let time_target_minutes = 10.0;
    let size_target_mb = 16.0;
    let checkpoint_mb = run.metrics.checkpoint_size_bytes as f64 / (1024.0 * 1024.0);

    if run.metrics.final_bpb < bpb_target {
        println!("  ✅ BPB {:.4} < target {:.2}", run.metrics.final_bpb, bpb_target);
    } else {
        println!("  ❌ BPB {:.4} > target {:.2}", run.metrics.final_bpb, bpb_target);
    }

    if run.metrics.total_time_seconds / 60.0 < time_target_minutes {
        println!("  ✅ Time {:.1}min < target {:.0}min",
            run.metrics.total_time_seconds / 60.0, time_target_minutes);
    } else {
        println!("  ⚠️  Time {:.1}min > target {:.0}min (slower but acceptable)",
            run.metrics.total_time_seconds / 60.0, time_target_minutes);
    }

    if checkpoint_mb < size_target_mb {
        println!("  ✅ Size {:.2}MB < target {:.0}MB", checkpoint_mb, size_target_mb);
    } else {
        println!("  ❌ Size {:.2}MB > target {:.0}MB (needs optimization)",
            checkpoint_mb, size_target_mb);
    }

    Ok(())
}
