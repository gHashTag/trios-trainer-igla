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

    /// Override seed. Use 0 to run 3-seed sweep {43,44,45}.
    #[arg(long, env = "TRIOS_SEED", default_value_t = 43)]
    seed: u64,

    /// Number of training steps.
    #[arg(long, env = "TRIOS_STEPS", default_value_t = 54000)]
    steps: usize,

    /// Hidden dimension (phi-scaled: 828).
    #[arg(long, env = "TRIOS_HIDDEN", default_value_t = 828)]
    hidden: usize,

    /// Learning rate.
    #[arg(long, env = "TRIOS_LR", default_value_t = 0.003)]
    lr: f32,

    /// Number of attention layers.
    #[arg(long, env = "TRIOS_ATTN_LAYERS", default_value_t = 2)]
    attn_layers: u8,

    /// Evaluate every N steps.
    #[arg(long, env = "TRIOS_EVAL_EVERY", default_value_t = 1000)]
    eval_every: usize,

    /// Path to training data.
    #[arg(
        long,
        env = "TRIOS_TRAIN_PATH",
        default_value = "data/tiny_shakespeare.txt"
    )]
    train_data: String,

    /// Path to validation data.
    #[arg(
        long,
        env = "TRIOS_VAL_PATH",
        default_value = "data/tiny_shakespeare_val.txt"
    )]
    val_data: String,

    /// Run 3-seed sweep {43, 44, 45} instead of single seed.
    #[arg(long)]
    sweep: bool,

    /// Optimizer: adamw, muon, or muon-cwd (P1 lab).
    #[arg(long, env = "TRIOS_OPTIMIZER", default_value = "adamw")]
    optimizer: String,

    /// Context window (accepted for seed-agent compat; ignored — fixed by
    /// `train_loop::NUM_CTX`). seed-agent (gHashTag/trios-railway) passes
    /// `--ctx 12` because the legacy bisect found ctx=12 was the only working
    /// value. We accept the flag here so unknown-arg parse errors don't crash
    /// the trainer immediately. Refs: trios-railway#62, trios-trainer-igla#55.
    #[arg(long, env = "TRIOS_CTX")]
    #[allow(dead_code)]
    ctx: Option<usize>,

    /// Format type pass-through (accepted for seed-agent compat; honoured via
    /// `TRIOS_FORMAT_TYPE` env). gf16 is the default in production.
    #[arg(long, env = "TRIOS_FORMAT_TYPE")]
    #[allow(dead_code)]
    format: Option<String>,

    /// Neon database URL for bpb_samples writes (used by scarab worker).
    #[arg(long, env = "TRIOS_NEON_DSN")]
    #[allow(dead_code)]
    neon: Option<String>,
}

fn install_panic_hook() {
    // R5/L8: never let a panic vanish into the void. Print a one-line JSON
    // diagnostic to stderr so seed-agent (which streams stderr via
    // `Stdio::inherit()`) and Railway logs both capture it. Also write a
    // marker line to stdout so the parent reader sees a non-empty stream.
    std::panic::set_hook(Box::new(|info| {
        use std::io::Write as _;
        let loc = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_else(|| "unknown".to_string());
        let msg = info.to_string();
        eprintln!(
            r#"{{"event":"panic","loc":{:?},"msg":{:?},"step":-1}}"#,
            loc, msg
        );
        let _ = std::io::stderr().flush();
        // Stdout marker so seed-agent's parse_step_output()/parse_done_output()
        // log it as 'unrecognized' instead of producing zero JSONL silently.
        println!("PANIC: trios-train aborted at {loc} (msg: {msg})");
        let _ = std::io::stdout().flush();
    }));
}

fn main() -> Result<()> {
    install_panic_hook();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    eprintln!(
        "[trios-train] startup args={:?} cwd={:?}",
        std::env::args().collect::<Vec<_>>(),
        std::env::current_dir().ok()
    );
    use std::io::Write as _;
    let _ = std::io::stderr().flush();

    let cli = Cli::parse();

    // Set NEON_DATABASE_URL from --neon flag OR inherit from ENV (used by scarab worker)
    // scarab passes NEON_DATABASE_URL via ENV inheritance, so check that first
    if std::env::var("NEON_DATABASE_URL").is_err() {
        if let Some(neon_url) = &cli.neon {
            std::env::set_var("NEON_DATABASE_URL", neon_url);
        }
    }

    eprintln!(
        "[trios-train] parsed seed={} steps={} hidden={} lr={} ctx={:?} optimizer={} neon={:?}",
        cli.seed, cli.steps, cli.hidden, cli.lr, cli.ctx, cli.optimizer, cli.neon
    );
    let _ = std::io::stderr().flush();

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
        let outcome = match cli.optimizer.as_str() {
            "muon" => train_loop::run_single_muon(&args, false)?,
            "muon-cwd" => train_loop::run_single_muon(&args, true)?,
            _ => train_loop::run_single(&args)?,
        };
        println!(
            "DONE: seed={} bpb={:.4} steps={} opt={}",
            outcome.seed, outcome.final_bpb, outcome.steps_done, cli.optimizer
        );
        // R5/L8: flush so seed-agent reader sees DONE before EOF.
        use std::io::Write as _;
        let _ = std::io::stdout().flush();
    }

    Ok(())
}
