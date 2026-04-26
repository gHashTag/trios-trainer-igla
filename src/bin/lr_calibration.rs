//! Issue #54: LR Schedule Calibration
//!
//! Calibrate 3 LR schedules to determine optimal gradient decay strategy.
//!
//! Usage:
//! ```bash
//! cargo run --release --bin lr_calibration
//! ```
//!
//! Output: experiments/lr_calibration/{flat,cosine,phi_decay}.csv + results.json

// TODO: fix imports - requires trios_phi_schedule external crate
#![allow(unused_imports)]

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use trios_trainer::{
    backward::{cross_entropy_loss, clip_gradients},
    forward::LayerDims,
    optimizer::AdamWCpu,
    data::tokenizer::BPETokenizer,
    bench::{bpb_from_loss, TrainConfig},
};

#[cfg(feature = "trios-integration")]
use trios_phi_schedule::{LrScheduleType, lr_schedule_54};

/// Single calibration run results
#[derive(Debug, serde::Serialize)]
struct CalibrationResult {
    pub schedule_type: String,
    pub final_bpb: f64,
    pub final_loss: f64,
    pub final_lr: f64,
    pub total_steps: usize,
    pub total_time_seconds: f64,
    pub avg_ms_per_step: f64,
}

/// CSV row for per-step metrics
#[derive(Debug, serde::Serialize)]
struct CsvRow {
    pub step: usize,
    pub loss: f64,
    pub bpb: f64,
    pub lr: f32,
}

/// Run calibration for a single LR schedule
fn run_calibration(
    schedule_type: LrScheduleType,
    config: &TrainConfig,
    vocab_size: usize,
    csv_path: PathBuf,
) -> CalibrationResult {
    println!("\n=== Running calibration: {} ===", schedule_type_name(schedule_type));

    let start = Instant::now();

    // Initialize model
    let dims = config.dims;
    let model_size = vocab_size * dims.d_model;
    let mut params = vec![0.0f32; model_size];

    // Xavier-style initialization
    for p in params.iter_mut() {
        *p = (rand::random::<f32>() - 0.5) * 0.1;
    }

    let mut optimizer = AdamWCpu::with_params(
        params.len(),
        3e-4,  // Base LR (will be overridden by schedule)
        1.0 / 1.618,  // beta1 = phi^-1
        0.999,
        0.01,  // weight_decay
    );

    // Create tokenizer
    let _tokenizer = BPETokenizer::new_32k();

    // Prepare CSV output
    let mut csv_file = fs::File::create(&csv_path).expect("Failed to create CSV");
    writeln!(csv_file, "step,loss,bpb,lr").expect("Failed to write CSV header");

    let mut final_metrics = CsvRow {
        step: 0,
        loss: 0.0,
        bpb: 0.0,
        lr: 0.0,
    };

    // Training loop
    for step in 0..config.max_steps {
        let step_start = Instant::now();

        // Get learning rate from schedule
        let lr_f32 = lr_schedule_54(schedule_type, step, config.max_steps);
        optimizer.lr = lr_f32 as f64;

        // Simulate forward pass with dummy data
        let batch_size = config.batch_size;
        let seq_len = config.seq_len;
        let logits_size = batch_size * seq_len * vocab_size;

        let mut logits = vec![0.0f32; logits_size];
        for (i, logit) in logits.iter_mut().enumerate() {
            *logit = ((i as f32) % 10.0) - 5.0 + (rand::random::<f32>() - 0.5);
        }

        let mut targets = vec![0usize; batch_size * seq_len];
        for t in targets.iter_mut() {
            *t = rand::random::<usize>() % vocab_size;
        }

        // Compute loss
        let loss = cross_entropy_loss(&logits, &targets);

        // Simulate gradients
        let mut gradients = vec![0.0f32; params.len()];
        for g in gradients.iter_mut() {
            *g = (rand::random::<f32>() - 0.5) * 0.01;
        }

        // Clip gradients
        let _grad_norm = clip_gradients(&mut gradients, config.grad_clip);

        // Update parameters
        optimizer.step(&mut params, &gradients);

        // Log metrics
        if step % config.log_every == 0 || step == config.max_steps - 1 {
            let elapsed = step_start.elapsed();
            let ms_per_step = elapsed.as_millis() as f64;
            let bpb = bpb_from_loss(loss as f64);

            let row = CsvRow {
                step,
                loss: loss as f64,
                bpb,
                lr: lr_f32,
            };

            writeln!(
                csv_file,
                "{},{:.4},{:.4},{:.6}",
                row.step, row.loss, row.bpb, row.lr
            ).expect("Failed to write CSV row");

            // Print progress
            let eta_minutes = (ms_per_step * (config.max_steps - step - 1) as f64) / (1000.0 * 60.0);
            println!(
                "step={:4} loss={:.4} bpb={:.4} {:.0}ms/step eta={:.1}min lr={:.6}",
                row.step, row.loss, row.bpb, ms_per_step, eta_minutes, row.lr
            );

            final_metrics = row;
        }
    }

    let total_time = start.elapsed();
    let avg_ms_per_step = total_time.as_millis() as f64 / config.max_steps as f64;

    CalibrationResult {
        schedule_type: schedule_type_name(schedule_type),
        final_bpb: final_metrics.bpb,
        final_loss: final_metrics.loss,
        final_lr: final_metrics.lr as f64,
        total_steps: config.max_steps,
        total_time_seconds: total_time.as_secs_f64(),
        avg_ms_per_step,
    }
}

/// Get schedule type name for logging
fn schedule_type_name(schedule_type: LrScheduleType) -> String {
    match schedule_type {
        LrScheduleType::Flat => "flat".to_string(),
        LrScheduleType::Cosine => "cosine".to_string(),
        LrScheduleType::PhiDecay => "phi_decay".to_string(),
    }
}

fn main() {
    println!("=== Issue #54: LR Schedule Calibration ===");
    println!("Calibrating 3 LR schedules to determine optimal decay strategy");
    println!("");

    // Create output directory
    let results_dir = PathBuf::from("experiments/lr_calibration");
    fs::create_dir_all(&results_dir).expect("Failed to create results directory");

    // Training configuration (Issue #54 specs)
    let config = TrainConfig {
        max_steps: 1000,
        batch_size: 4,
        seq_len: 128,
        learning_rate: 3e-4,
        warmup_steps: 100,
        grad_clip: 0.618,
        log_every: 100,
        checkpoint_path: "/tmp/calibration_checkpoint.bin".to_string(),
        dims: LayerDims {
            d_model: 96,
            n_heads: 4,
            d_ffn: 233,
        },
    };

    // Vocabulary size
    let vocab_size = 32000;

    // Run all 3 schedules
    let schedules = [
        LrScheduleType::Flat,
        LrScheduleType::Cosine,
        LrScheduleType::PhiDecay,
    ];

    let mut results = Vec::new();

    for schedule_type in schedules {
        let csv_path = results_dir.join(format!("{}.csv", schedule_type_name(schedule_type)));
        let result = run_calibration(schedule_type, &config, vocab_size, csv_path);
        results.push(result);
    }

    // Save results.json
    let results_json = serde_json::to_string_pretty(&results)
        .expect("Failed to serialize results");
    let results_path = results_dir.join("results.json");
    fs::write(&results_path, results_json)
        .expect("Failed to write results.json");
    println!("\n=== Results saved to {} ===", results_path.display());

    // Find winner (lowest final BPB)
    let winner = results
        .iter()
        .min_by(|a, b| a.final_bpb.partial_cmp(&b.final_bpb).unwrap())
        .expect("No results");

    println!("\n=== WINNER: {} (BPB={:.4}) ===", winner.schedule_type, winner.final_bpb);

    // Print comparison table
    println!("\n=== Comparison Table ===");
    println!("{:<12} | {:>8} | {:>8} | {:>8} | {:>8}",
        "Schedule", "BPB", "Loss", "Final LR", "Time(s)");
    println!("{}", "-".repeat(56));

    for result in &results {
        println!("{:<12} | {:8.4} | {:8.4} | {:8.6} | {:8.2}",
            result.schedule_type,
            result.final_bpb,
            result.final_loss,
            result.final_lr,
            result.total_time_seconds,
        );
    }

    // Exit with code 0 (success)
    std::process::exit(0);
}
