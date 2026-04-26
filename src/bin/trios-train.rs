//! `trios-train` — CLI entry point.
//!
//! ```bash
//! # Run champion reproduction on this machine
//! cargo run --release -p trios-trainer --bin trios-train -- \
//!     --config configs/champion.toml --seed 43
//!
//! # On Railway: image entrypoint = trios-train; configure via env
//! TRIOS_SEED=44 TRIOS_TARGET_BPB=1.50 trios-train --config /configs/gate2-attempt.toml
//! ```

use anyhow::Result;
use clap::Parser;
use trios_trainer::{run, TrainConfig};

#[derive(Parser, Debug)]
#[command(
    name = "trios-train",
    about = "IGLA RACE training pipeline (gHashTag/trios#143)"
)]
struct Cli {
    /// Path to TOML config (champion / gate2-attempt / needle-rush variant).
    #[arg(long, env = "TRIOS_CONFIG")]
    config: std::path::PathBuf,

    /// Override seed. Beats `TRIOS_SEED` env which beats config.
    #[arg(long, env = "TRIOS_SEED")]
    seed: Option<u64>,

    /// Dry-run: load config, validate invariants, exit before training loop.
    #[arg(long)]
    dry_run: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let mut cfg = TrainConfig::from_toml(&cli.config)?;
    if let Some(seed) = cli.seed {
        cfg.seed = seed;
    }

    tracing::info!(
        name = %cfg.name, seed = cfg.seed, steps = cfg.steps,
        target_bpb = cfg.target_bpb, "run config loaded"
    );

    if cli.dry_run {
        tracing::info!("dry-run requested → exit before train loop");
        return Ok(());
    }

    let outcome = run(&cfg)?;
    tracing::info!(?outcome, "training complete");
    Ok(())
}
