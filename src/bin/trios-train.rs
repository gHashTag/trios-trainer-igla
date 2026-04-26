//! `trios-train` - CLI entry point.
//!
//! ```bash
//! # Standalone mode (no config file needed)
//! trios-train --seed 43 --steps 54000
//!
//! # Config mode
//! trios-train --config configs/champion.toml
//!
//! # 3-seed sweep
//! trios-train --sweep --steps 54000
//! ```

use anyhow::Result;
use clap::Parser;
use trios_trainer::train_loop::{self, TrainArgs, GATE_FINAL_SEEDS};

#[derive(Parser, Debug)]
#[command(
    name = "trios-train",
    about = "IGLA RACE training pipeline (gHashTag/trios#143)"
)]
struct Cli {
    /// Path to TOML config (optional; standalone mode if omitted).
    #[arg(long, env = "TRIOS_CONFIG")]
    config: Option<std::path::PathBuf>,

    /// Override seed. Use 0 to run 3-seed sweep {42,43,44}.
    #[arg(long, env = "TRIOS_SEED", default_value_t = 43)]
    seed: u64,

    /// Number of training steps.
    #[arg(long, env = "TRIOS_STEPS", default_value_t = 54000)]
    steps: usize,

    /// Hidden dimension (phi-scaled: 828).
    #[arg(long, default_value_t = 828)]
    hidden: usize,

    /// Learning rate.
    #[arg(long, default_value_t = 0.003)]
    lr: f32,

    /// Number of attention layers.
    #[arg(long, default_value_t = 2)]
    attn_layers: u8,

    /// Evaluate every N steps.
    #[arg(long, default_value_t = 1000)]
    eval_every: usize,

    /// Path to training data.
    #[arg(long, default_value = "data/tiny_shakespeare.txt")]
    train_data: String,

    /// Path to validation data.
    #[arg(long, default_value = "data/tiny_shakespeare_val.txt")]
    val_data: String,

    /// Run 3-seed sweep {42, 43, 44} instead of single seed.
    #[arg(long)]
    sweep: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    if let Some(config_path) = &cli.config {
        let cfg = trios_trainer::TrainConfig::from_toml(config_path)?;
        tracing::info!(name = %cfg.name, seed = cfg.seed, steps = cfg.steps, "config mode");
        let outcome = train_loop::run(&cfg)?;
        tracing::info!(?outcome, "training complete");
        return Ok(());
    }

    if cli.sweep || cli.seed == 0 {
        tracing::info!("3-seed sweep: {:?}", GATE_FINAL_SEEDS);
        let results = train_loop::run_sweep(
            cli.steps,
            cli.hidden,
            cli.lr,
            cli.attn_layers,
            cli.eval_every,
            &cli.train_data,
            &cli.val_data,
        )?;
        for r in &results {
            println!(
                "DONE: seed={} bpb={:.4} steps={}",
                r.seed, r.final_bpb, r.steps_done
            );
        }
        let all_pass = results
            .iter()
            .all(|r| r.final_bpb < train_loop::DEFAULT_IGLA_TARGET_BPB);
        println!("GATE-2: {}", if all_pass { "PASS" } else { "NOT YET" });
    } else {
        let args = TrainArgs {
            seed: cli.seed,
            steps: cli.steps,
            hidden: cli.hidden,
            lr: cli.lr,
            attn_layers: cli.attn_layers,
            eval_every: cli.eval_every,
            train_path: cli.train_data.clone(),
            val_path: cli.val_data.clone(),
        };
        let outcome = train_loop::run_single(&args)?;
        println!(
            "DONE: seed={} bpb={:.4} steps={}",
            outcome.seed, outcome.final_bpb, outcome.steps_done
        );
    }

    Ok(())
}
