//! Trinity Runner — L2: Claim → Train → Write Loop
//!
//! # Constitutional mandate (Law 1)
//!
//! - No claim without verified source row
//! - Idempotent claim with `FOR UPDATE SKIP LOCKED`
//! - Retry logic with exponential backoff
//!
//! # PR-O5 status
//!
//! - [x] main.rs — claim loop
//! - [ ] e2e test with local Neon
//! - [ ] Docker integration
//!
//! 🌻 φ² + φ⁻² = 3 · TRINITY · O(1) FOREVER

use anyhow::Result;
use clap::Parser;
use std::time::Duration;
use tokio::time::sleep;
use trinity_experiments::{ExperimentRepo, PostgresExperimentRepo, ClaimResult};
use trinity_trainer::{Config, train};
use uuid::Uuid;

/// Trinity Runner — claim and train experiments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Neon database URL
    #[arg(long, value_name = "URL")]
    neon_url: String,

    /// Worker ID (generated if not provided)
    #[arg(long)]
    worker_id: Option<String>,

    /// Poll interval in seconds
    #[arg(long, default_value_t = 30)]
    poll_interval: u64,

    /// Run once and exit (don't loop)
    #[arg(long)]
    run_once: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let worker_id = match args.worker_id {
        Some(s) => Uuid::parse_str(&s)?,
        None => Uuid::new_v4(),
    };

    println!("[trinity-runner] starting: worker_id={}", worker_id);

    // Connect to Neon with retry
    let repo = connect_with_retry(&args.neon_url).await?;

    loop {
        // Claim next pending experiment
        match claim_experiment(&repo, worker_id).await? {
            ClaimResult::Claimed(claim) => {
                println!("[trinity-runner] claimed: id={} name={}", claim.id, claim.canon_name);

                // Run training
                let config = Config::from(&claim.config);
                match train(config) {
                    Ok(outcome) => {
                        println!("[trinity-runner] completed: step={} bpb={:?}",
                                outcome.final_step, outcome.final_bpb);

                        // Write result
                        repo.complete(
                            claim.id,
                            outcome.final_bpb.unwrap_or(0.0) as f64,
                            outcome.final_step as i32,
                            outcome.bpb_curve,
                        ).await?;
                    }
                    Err(e) => {
                        println!("[trinity-runner] failed: {}", e);
                        repo.fail(claim.id, e.to_string()).await?;
                    }
                }
            }
            ClaimResult::NoPending => {
                println!("[trinity-runner] no pending experiments");
            }
        }

        if args.run_once {
            println!("[trinity-runner] run_once=true, exiting");
            break;
        }

        sleep(Duration::from_secs(args.poll_interval)).await;
    }

    Ok(())
}

/// Connect to Neon with exponential backoff retry
async fn connect_with_retry(url: &str) -> Result<PostgresExperimentRepo> {
    let max_attempts = 10;
    let mut delay = Duration::from_secs(1);

    for attempt in 1..=max_attempts {
        match PostgresExperimentRepo::connect(url).await {
            Ok(repo) => {
                println!("[trinity-runner] connected to Neon (attempt {})", attempt);
                return Ok(repo);
            }
            Err(e) => {
                eprintln!("[trinity-runner] connect attempt {}/{} failed: {}", attempt, max_attempts, e);
                if attempt < max_attempts {
                    sleep(delay).await;
                    delay *= 2; // Exponential backoff
                }
            }
        }
    }

    anyhow::bail!("failed to connect after {} attempts", max_attempts)
}

/// Claim next pending experiment
async fn claim_experiment(repo: &PostgresExperimentRepo, worker_id: Uuid) -> Result<ClaimResult> {
    repo.claim_next(worker_id).await
}
