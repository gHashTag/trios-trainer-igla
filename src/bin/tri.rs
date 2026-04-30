use clap::{Parser, Subcommand};
use std::process::Command as StdCommand;
use std::sync::Arc;
use std::sync::RwLock;

const RAILWAY_PROJECT: &str = "trios-trainer";
const RAILWAY_PROJECT_ID: &str = "abdf752c-20ac-4813-a586-04a031db96e8";
const GATE_SEEDS: &[u64] = &[43, 44, 45];

#[derive(Parser)]
#[command(
    name = "tri",
    about = "IGLA Race CLI — Railway deploy + local train",
    version
)]
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
    Train {
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 5000)]
        steps: usize,
        #[arg(long, default_value_t = 384)]
        hidden: usize,
        #[arg(long, default_value_t = 0.004)]
        lr: f64,
        #[arg(long, default_value_t = 2)]
        attn_layers: u8,
        #[arg(long, default_value_t = 500)]
        eval_every: usize,
        #[arg(long, default_value = "data/tinyshakespeare.txt")]
        train_data: String,
        #[arg(long, default_value = "data/tinyshakespeare.txt")]
        val_data: String,
        #[arg(long, default_value = "adamw")]
        optimizer: String,
    },
    Deploy {
        #[command(subcommand)]
        deploy_cmd: DeployCommands,
    },
}

#[derive(Subcommand)]
enum RaceCommands {
    Start,
    Status,
    Best,
}

#[derive(Subcommand)]
enum DeployCommands {
    #[command(about = "Deploy a single seed training container")]
    Seed {
        #[arg(long)]
        seed: u64,
        #[arg(long, default_value_t = 27000)]
        steps: usize,
        #[arg(long, default_value_t = 384)]
        hidden: usize,
        #[arg(long, default_value_t = 0.004)]
        lr: f64,
        #[arg(long, default_value_t = 2)]
        attn_layers: u8,
    },
    #[command(about = "Deploy all Gate-2 seeds (42, 43, 44)")]
    All {
        #[arg(long, default_value_t = 27000)]
        steps: usize,
        #[arg(long, default_value_t = 384)]
        hidden: usize,
        #[arg(long, default_value_t = 0.004)]
        lr: f64,
        #[arg(long, default_value_t = 2)]
        attn_layers: u8,
    },
    #[command(about = "List deployed training services")]
    Status,
    #[command(about = "Stream logs for a seed's training container")]
    Logs {
        #[arg(long)]
        seed: u64,
    },
    #[command(about = "Remove a seed's training container")]
    Remove {
        #[arg(long)]
        seed: u64,
    },
    #[command(about = "Initialize Railway project (create if needed)")]
    Init,
}

fn railway(args: &[&str]) -> anyhow::Result<()> {
    let status = StdCommand::new("railway")
        .args(args)
        .env("RAILWAY_NON_INTERACTIVE", "1")
        .status()?;
    if !status.success() {
        anyhow::bail!(
            "railway {} failed with exit code {:?}",
            args.join(" "),
            status.code()
        );
    }
    Ok(())
}

fn railway_output(args: &[&str]) -> anyhow::Result<String> {
    let output = StdCommand::new("railway")
        .args(args)
        .env("RAILWAY_NON_INTERACTIVE", "1")
        .output()?;
    if !output.status.success() {
        anyhow::bail!(
            "railway {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn railway_var_set(svc: &str, key: &str, value: &str) -> anyhow::Result<()> {
    railway(&[
        "variable",
        "set",
        "-s",
        svc,
        "-e",
        "production",
        &format!("{}={}", key, value),
    ])
}

fn create_service(svc: &str) -> anyhow::Result<()> {
    let output = StdCommand::new("railway")
        .args(["add", "--service", svc])
        .env("RAILWAY_NON_INTERACTIVE", "1")
        .output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("already exists") || stderr.contains("already") {
            return Ok(());
        }
        anyhow::bail!("railway add --service {} failed: {}", svc, stderr);
    }
    Ok(())
}

fn service_name(seed: u64) -> String {
    format!("trainer-seed-{}", seed)
}

fn deploy_seed(
    seed: u64,
    steps: usize,
    hidden: usize,
    lr: f64,
    attn_layers: u8,
) -> anyhow::Result<()> {
    let svc = service_name(seed);
    eprintln!(
        "Deploying {} (seed={}, steps={}, hidden={}, lr={:.4}, attn={}) ...",
        svc, seed, steps, hidden, lr, attn_layers
    );

    railway(&["link", "--project", RAILWAY_PROJECT_ID, "-e", "production"])?;

    let existing = railway_output(&["service", "list"]).unwrap_or_default();
    let svc_exists = existing.contains(&svc);

    if !svc_exists {
        eprintln!("Creating Railway service: {}", svc);
        create_service(&svc)?;
    } else {
        eprintln!("Service {} already exists, updating vars...", svc);
    }

    railway_var_set(&svc, "TRIOS_SEED", &seed.to_string())?;
    railway_var_set(&svc, "TRIOS_STEPS", &steps.to_string())?;
    railway_var_set(&svc, "TRIOS_HIDDEN", &hidden.to_string())?;
    railway_var_set(&svc, "TRIOS_LR", &format!("{:.6}", lr))?;
    railway_var_set(&svc, "TRIOS_ATTN_LAYERS", &attn_layers.to_string())?;
    railway_var_set(&svc, "TRIOS_OPTIMIZER", "adamw")?;
    railway_var_set(&svc, "TRIOS_EVAL_EVERY", "1000")?;

    railway(&[
        "link",
        "--project",
        RAILWAY_PROJECT_ID,
        "--service",
        &svc,
        "-e",
        "production",
    ])?;
    railway(&["up", "--detach"])?;

    eprintln!(
        "Deployed {} — stream logs: tri deploy logs --seed {}",
        svc, seed
    );
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Deploy { deploy_cmd } => match deploy_cmd {
            DeployCommands::Seed {
                seed,
                steps,
                hidden,
                lr,
                attn_layers,
            } => {
                deploy_seed(seed, steps, hidden, lr, attn_layers)?;
            }
            DeployCommands::All {
                steps,
                hidden,
                lr,
                attn_layers,
            } => {
                for &seed in GATE_SEEDS {
                    deploy_seed(seed, steps, hidden, lr, attn_layers)?;
                }
                eprintln!("All {} seeds deployed!", GATE_SEEDS.len());
            }
            DeployCommands::Status => {
                eprintln!("Railway services for {}:", RAILWAY_PROJECT);
                railway(&["status"])?;
            }
            DeployCommands::Logs { seed } => {
                let svc = service_name(seed);
                eprintln!("Streaming logs for {} ...", svc);
                railway(&["link", "--service", &svc, "--environment", "production"])?;
                railway(&["logs"])?;
            }
            DeployCommands::Remove { seed } => {
                let svc = service_name(seed);
                eprintln!("Removing {} ...", svc);
                eprintln!("Run manually: railway down -s {} -e production", svc);
            }
            DeployCommands::Init => {
                eprintln!("Initializing Railway project: {}", RAILWAY_PROJECT);
                railway(&["init", "--name", RAILWAY_PROJECT])?;
                eprintln!("Project '{}' ready", RAILWAY_PROJECT);
            }
        },
        Commands::Race { race_cmd } => match race_cmd {
            RaceCommands::Start => {
                let neon_url = std::env::var("DATABASE_URL").unwrap_or_default();
                if neon_url.is_empty() {
                    eprintln!("WARNING: DATABASE_URL not set — running in local-only mode");
                }
                let machine_id = hostname_or_default();
                let best_bpb = Arc::new(RwLock::new(f64::MAX));
                let result =
                    trios_trainer::race::asha::run_worker(&neon_url, &machine_id, 0, best_bpb)
                        .await;
                match result {
                    Ok(bpb) => println!("BPB={:.4}", bpb),
                    Err(e) => {
                        eprintln!("ASHA worker error: {e}");
                        std::process::exit(1);
                    }
                }
            }
            RaceCommands::Status => {
                let neon_url = std::env::var("DATABASE_URL").unwrap_or_default();
                if neon_url.is_empty() {
                    eprintln!("No DATABASE_URL set — no leaderboard");
                    return Ok(());
                }
                let db = trios_trainer::race::neon::NeonDb::connect(&neon_url).await?;
                trios_trainer::race::status::show_status(&db).await?;
            }
            RaceCommands::Best => {
                let neon_url = std::env::var("DATABASE_URL").unwrap_or_default();
                if neon_url.is_empty() {
                    eprintln!("No DATABASE_URL set");
                    return Ok(());
                }
                let db = trios_trainer::race::neon::NeonDb::connect(&neon_url).await?;
                trios_trainer::race::status::show_best(&db).await?;
            }
        },
        Commands::Train {
            seed,
            steps,
            hidden,
            lr,
            attn_layers,
            eval_every,
            train_data,
            val_data,
            optimizer,
        } => {
            let args = trios_trainer::train_loop::TrainArgs {
                seed,
                steps,
                hidden,
                lr: lr as f32,
                attn_layers,
                eval_every,
                train_path: train_data,
                val_path: val_data,
                precision: "fp32".to_string(),
                log_grad_norm: false,
            };
            let outcome = match optimizer.as_str() {
                "muon" => trios_trainer::train_loop::run_single_muon(&args, false)?,
                "muon-cwd" => trios_trainer::train_loop::run_single_muon(&args, true)?,
                _ => trios_trainer::train_loop::run_single(&args)?,
            };
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
