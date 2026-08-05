//! Transformer Training Binary — Phase 2
//!
//! Run with: cargo run --bin transformer_train --release
//!
//! Target: Beat N-gram baseline (2.5329 BPB) with minimal transformer

use std::env;
use trios_trainer::transformer_trainer::{TrainResult, TransformerTrainConfig};

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse command line arguments
    let mut config = TransformerTrainConfig::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                if i + 1 < args.len() {
                    config.max_steps = args[i + 1].parse().unwrap_or(config.max_steps);
                    i += 2;
                }
            }
            "--lr" => {
                if i + 1 < args.len() {
                    config.lr = args[i + 1].parse().unwrap_or(config.lr);
                    i += 2;
                }
            }
            "--layers" => {
                if i + 1 < args.len() {
                    config.n_layers = args[i + 1].parse().unwrap_or(config.n_layers);
                    i += 2;
                }
            }
            "--sweep" => {
                // Learning rate sweep mode
                i += 1;
                run_lr_sweep(&config);
                return;
            }
            "--help" => {
                print_help();
                return;
            }
            _ => {
                i += 1;
            }
        }
    }

    // Run training
    let result = config.run();

    // Save metrics
    save_metrics(&result);
}

fn run_lr_sweep(base_config: &TransformerTrainConfig) {
    println!("\nRunning LR sweep with reduced steps for comparison...\n");

    let mut sweep_config = base_config.clone();
    sweep_config.max_steps = 1000; // Reduced for sweep

    let lrs = vec![
        0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010,
    ];

    let results = sweep_config.run_lr_sweep(lrs);

    // Find best result
    if let Some(best) = results.iter().min_by(|a, b| {
        a.best_bpb
            .partial_cmp(&b.best_bpb)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        println!("\n=== Best Configuration ===");
        println!("Best BPB: {:.4}", best.best_bpb);
        println!("Steps: {}", best.total_steps);
        println!("Time: {:.1}s", best.total_time_seconds);
    }
}

fn save_metrics(result: &TrainResult) {
    use std::fs::File;
    use std::io::Write;

    let filename = ".trinity/experiments/transformer_phase2_results.json";
    if let Ok(mut file) = File::create(filename) {
        use serde_json::json;
        use std::collections::HashMap;

        let metrics_data: Vec<HashMap<String, serde_json::Value>> = result
            .metrics
            .iter()
            .map(|m| {
                let mut map = HashMap::new();
                map.insert("step".to_string(), json!(m.step));
                map.insert("loss".to_string(), json!(m.loss));
                map.insert("bpb".to_string(), json!(m.bpb));
                map.insert("lr".to_string(), json!(m.lr));
                map.insert("time".to_string(), json!(m.elapsed_seconds));
                map
            })
            .collect();

        let output = json!({
            "final_bpb": result.final_bpb,
            "best_bpb": result.best_bpb,
            "total_steps": result.total_steps,
            "total_time_seconds": result.total_time_seconds,
            "metrics": metrics_data
        });

        let _ = file.write_all(serde_json::to_string_pretty(&output).unwrap().as_bytes());
        println!("\nMetrics saved to {}", filename);
    }
}

fn print_help() {
    println!("Transformer Training — Phase 2");
    println!("\nUsage: cargo run --bin transformer_train --release [OPTIONS]");
    println!("\nOptions:");
    println!("  --steps <N>     Maximum training steps (default: 5000)");
    println!("  --lr <FLOAT>    Learning rate (default: 0.004)");
    println!("  --layers <N>    Number of transformer layers (default: 2)");
    println!("  --sweep         Run learning rate sweep");
    println!("  --help          Show this help message");
    println!("\nExample:");
    println!("  cargo run --bin transformer_train --release --steps 10000 --lr 0.005");
}
