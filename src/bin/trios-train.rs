//! `trios-train` - CLI entry point.
//!
//! ```bash
//! # Standalone mode (no config file needed)
//! trios-train --seed 47 --steps 54000   # Canon #93 allowed seed
//!
//! # Config mode
//! trios-train --config configs/champion.toml
//!
//! # 3-seed sweep
//! trios-train --sweep --steps 54000
//! ```

use anyhow::Result;
use clap::Parser;
use migration::MigratorTrait;
use trios_trainer::neon_writer::strip_channel_binding;
use trios_trainer::seed_canon::parse_seed;
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

    /// Override seed. Use 0 to run 3-seed sweep.
    /// If the `SEED` env var is set, Canon #93 enforcement applies:
    /// seeds {42, 43, 44, 45} are forbidden; use {47, 89, 123, 144}.
    /// Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877
    #[arg(long, env = "TRIOS_SEED", default_value_t = 47)]
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

    /// Run 3-seed sweep {47, 89, 123} (Canon #93) instead of single seed.
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

    /// Format type pass-through.
    ///
    /// Honoured by `train_loop::resolve_fake_quant_format()` via the
    /// `TRIOS_FORMAT_TYPE` env var. Historically this flag was accepted but
    /// silently dropped because clap stored it in `cli.format` and `main()`
    /// never re-exported it; the result was a production-wide fp32-fallback
    /// (trios#509: 52 ≡ 2.942101 / 49 ≡ 2.998885 collapse, scarab triplets
    /// adamw-binary32 / adamw-GF16 / muon-GF16 producing identical BPB on the
    /// same seed). The fix below re-exports `cli.format` into the env so the
    /// `--format=gf16` CLI form behaves identically to `TRIOS_FORMAT_TYPE=gf16`.
    #[arg(long, env = "TRIOS_FORMAT_TYPE")]
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

/// Run SeaORM schema migrations at startup if TRINITY_AUTOMIGRATE != "0".
///
/// Gating: TRINITY_AUTOMIGRATE=0 disables for local CI; default is ON.
/// Logs: "[migrator] schema up-to-date (N migrations applied)"
fn run_automigrate() {
    let automigrate = std::env::var("TRINITY_AUTOMIGRATE").unwrap_or_else(|_| "1".to_string());
    if automigrate == "0" {
        eprintln!("[migrator] TRINITY_AUTOMIGRATE=0 — skipping");
        return;
    }

    let raw_dsn = std::env::var("DATABASE_URL")
        .or_else(|_| std::env::var("NEON_DATABASE_URL"))
        .or_else(|_| std::env::var("TRIOS_NEON_DSN"))
        .or_else(|_| std::env::var("TRIOS_DATABASE_URL"));

    let dsn = match raw_dsn {
        Ok(d) => strip_channel_binding(&d),
        Err(_) => {
            eprintln!("[migrator] DATABASE_URL unset — skipping automigrate");
            return;
        }
    };

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("migrator runtime");

    rt.block_on(async {
        match sea_orm::Database::connect(&dsn).await {
            Ok(db) => {
                match migration::Migrator::up(&db, None).await {
                    Ok(()) => eprintln!("[migrator] schema up-to-date"),
                    Err(e) => eprintln!("[migrator] migration failed (non-fatal): {e}"),
                }
                let _ = db.close().await;
            }
            Err(e) => eprintln!("[migrator] connect failed (non-fatal): {e}"),
        }
    });
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

    let mut cli = Cli::parse();

    // Canon #93 enforcement.
    //   * If `SEED` env var is set → validate via `parse_seed()` AND assign
    //     the validated value to `cli.seed`. SEED env therefore overrides
    //     `--seed` flag and `TRIOS_SEED`/clap default.
    //   * Else → directly validate `cli.seed` against the forbidden set
    //     (which is what `parse_seed()` does internally on its raw input).
    //     This catches the case where TRIOS_SEED or `--seed=43` slipped
    //     past clap.
    // Forbidden canon: {42, 43, 44, 45}; allowed canon: {47, 89, 123, 144}.
    // Wave-29 PR-A.1: previously `parse_seed()`'s return value was logged
    // and dropped — `cli.seed` could still be 43 if `--seed=43` was passed
    // alongside an unrelated `SEED` env. This patch eliminates the
    // validate-then-discard anti-pattern.
    // Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877
    if std::env::var("SEED").is_ok() {
        let canon_seed =
            parse_seed().map_err(|e| anyhow::anyhow!("Canon #93 violation (SEED env): {}", e))?;
        eprintln!(
            "[trios-train] Canon #93 OK: SEED={canon_seed} (overrides cli.seed={})",
            cli.seed
        );
        cli.seed = canon_seed;
    } else {
        const FORBIDDEN: &[u64] = &[42, 43, 44, 45];
        if FORBIDDEN.contains(&cli.seed) {
            return Err(anyhow::anyhow!(
                "Canon #93 violation: cli.seed={} is forbidden (allowed: 47, 89, 123, 144). \
                 Set SEED or TRIOS_SEED env var to an allowed value, or pass `--seed=<allowed>`.",
                cli.seed
            ));
        }
        eprintln!(
            "[trios-train] Canon #93 OK: seed={} (no SEED env, validated cli.seed)",
            cli.seed
        );
    }

    // Run SeaORM migrations at startup (gated by TRINITY_AUTOMIGRATE != "0").
    run_automigrate();

    // R5/L8 fix (trios#509 follow-up): re-export `--format` into the env so
    // `train_loop::resolve_fake_quant_format()` can see it. Without this line
    // the CLI flag was a no-op and every scarab-spawned trainer silently fell
    // back to F32 regardless of the strategy_queue config.
    if let Some(fmt) = &cli.format {
        if !fmt.is_empty() {
            std::env::set_var("TRIOS_FORMAT_TYPE", fmt);
        }
    }

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
