use clap::{Parser, Subcommand};
use std::sync::Arc;
use std::sync::RwLock;

#[derive(Parser)]
#[command(name = "tri", about = "IGLA Race CLI", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Race {
        #[command(subcommand)]
        race_cmd: RaceCommands,
    },
    Train,
}

#[derive(Subcommand)]
enum RaceCommands {
    Start,
    Status,
    Best,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Race { race_cmd } => match race_cmd {
            RaceCommands::Start => {
                let neon_url = std::env::var("DATABASE_URL").unwrap_or_default();
                if neon_url.is_empty() {
                    eprintln!("WARNING: DATABASE_URL not set — running in local-only mode (no Neon)");
                }

                let machine_id = hostname_or_default();
                let worker_id = 0u64;
                let best_bpb = Arc::new(RwLock::new(f64::MAX));

                let result = trios_trainer::race::asha::run_worker(
                    &neon_url,
                    &machine_id,
                    worker_id,
                    best_bpb.clone(),
                )
                .await;

                match result {
                    Ok(bpb) => {
                        println!("BPB={:.4}", bpb);
                    }
                    Err(e) => {
                        eprintln!("ASHA worker error: {e}");
                        std::process::exit(1);
                    }
                }
            }
            RaceCommands::Status => {
                let neon_url = std::env::var("DATABASE_URL").unwrap_or_default();
                if neon_url.is_empty() {
                    eprintln!("WARNING: DATABASE_URL not set — no leaderboard available");
                    eprintln!("No completed trials yet");
                    return Ok(());
                }
                let db = trios_trainer::race::neon::NeonDb::connect(&neon_url).await?;
                trios_trainer::race::status::show_status(&db).await?;
            }
            RaceCommands::Best => {
                let neon_url = std::env::var("DATABASE_URL").unwrap_or_default();
                if neon_url.is_empty() {
                    eprintln!("WARNING: DATABASE_URL not set — no best trial available");
                    eprintln!("No completed trials yet");
                    return Ok(());
                }
                let db = trios_trainer::race::neon::NeonDb::connect(&neon_url).await?;
                trios_trainer::race::status::show_best(&db).await?;
            }
        },
        Commands::Train => {
            let args = trios_trainer::train_loop::TrainArgs {
                seed: 42,
                steps: 5000,
                hidden: 256,
                lr: 0.003,
                attn_layers: 1,
                eval_every: 500,
                train_path: "data/tinyshakespeare.txt".to_string(),
                val_path: "data/tinyshakespeare.txt".to_string(),
            };

            let outcome = trios_trainer::train_loop::run_single(&args)?;
            println!("BPB={:.4}", outcome.final_bpb);
        }
    }

    Ok(())
}

fn hostname_or_default() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("MACHINE_ID"))
        .unwrap_or_else(|_| "local".to_string())
}
