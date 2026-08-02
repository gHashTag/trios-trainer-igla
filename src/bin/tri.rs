use clap::{Parser, Subcommand};
use std::io::Write;
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
    Gardener {
        #[command(subcommand)]
        gardener_cmd: GardenerCommands,
    },
}

#[derive(Subcommand)]
enum RaceCommands {
    Start,
    Status,
    Best,
}

#[derive(Subcommand)]
enum GardenerCommands {
    #[command(about = "Garden status: processes, logs, disk, best BPB")]
    Status,
    #[command(about = "Harvest BPB from a Railway service")]
    Harvest {
        #[arg(help = "Railway service name")]
        service: String,
    },
    #[command(about = "Harvest all known Railway services")]
    HarvestAll,
    #[command(about = "Prune dead/stalled trios-train processes")]
    Prune,
    #[command(about = "Water (restart) crashed runs from log signatures")]
    Water,
    #[command(about = "Stream logs for a Railway service")]
    Logs {
        #[arg(help = "Railway service name")]
        service: String,
        #[arg(long, default_value_t = 50)]
        lines: usize,
    },
    #[command(about = "Generate HTML harvest report with charts")]
    Report,
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
        Commands::Gardener { gardener_cmd } => match gardener_cmd {
            GardenerCommands::Status => gardener_status()?,
            GardenerCommands::Harvest { service } => gardener_harvest(&service)?,
            GardenerCommands::HarvestAll => gardener_harvest_all()?,
            GardenerCommands::Prune => gardener_prune()?,
            GardenerCommands::Water => gardener_water()?,
            GardenerCommands::Logs { service, lines } => gardener_logs(&service, lines)?,
            GardenerCommands::Report => gardener_report()?,
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

fn gardener_status() -> anyhow::Result<()> {
    let log_dir = ".trinity/results";
    let harvest_log = ".trinity/gardener_harvest.log";

    println!("=== САД IGLA RACE ===");
    println!("Время: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"));

    let proc_out = std::process::Command::new("sh")
        .arg("-c")
        .arg("ps aux | grep trios-train | grep -v grep | wc -l")
        .output()?;
    let procs = String::from_utf8_lossy(&proc_out.stdout).trim().to_string();
    println!("Процессы trios-train (local): {}", procs);

    let disk_out = std::process::Command::new("sh")
        .arg("-c")
        .arg(&format!("du -sh {} 2>/dev/null || echo 'N/A'", log_dir))
        .output()?;
    println!("Логи: {}", String::from_utf8_lossy(&disk_out.stdout).trim());

    if std::path::Path::new(harvest_log).exists() {
        let best_out = std::process::Command::new("sh")
            .arg("-c")
            .arg(&format!(
                "grep -E 'best=([0-9.]+)' {} | sed 's/.*best=//' | sort -n | head -1",
                harvest_log
            ))
            .output()?;
        let best = String::from_utf8_lossy(&best_out.stdout).trim().to_string();
        println!("Лучший BPB (fleet): {}", if best.is_empty() { "N/A" } else { &best });

        let cnt_out = std::process::Command::new("sh")
            .arg("-c")
            .arg(&format!("wc -l < {}", harvest_log))
            .output()?;
        println!("Harvest entries: {}", String::from_utf8_lossy(&cnt_out.stdout).trim());
    }

    Ok(())
}

fn gardener_harvest(service: &str) -> anyhow::Result<()> {
    let logfile = ".trinity/gardener_harvest.log";
    let tag = format!("[{}]", service);
    let output = std::process::Command::new("railway")
        .args(["logs", "--service", service, "--lines", "50"])
        .env("RAILWAY_NON_INTERACTIVE", "1")
        .output()?;
    if !output.status.success() {
        anyhow::bail!("railway logs failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let latest = stdout
        .lines()
        .filter(|l| l.contains("step=") && l.contains("val_bpb="))
        .last();
    let line = match latest {
        Some(l) => format!("{} {}", tag, l),
        None => format!("{} NO_DATA {}", tag, chrono::Local::now().format("%Y-%m-%dT%H:%M:%S")),
    };
    println!("{}", line);
    std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(logfile)?
        .write_all(format!("{}\n", line).as_bytes())?;
    Ok(())
}

fn gardener_harvest_all() -> anyhow::Result<()> {
    // Known fleet services from the IGLA RACE project
    let services = vec![
        "scarab-fp16-seed77",
        "phi-ablation-control",
        "phi-ablation-treat",
        "phase1-gf16-h512-seed82",
        "phase1-bf16-seed76",
        "scarab-mxfp8-seed79",
        "scarab-gf16-seed75",
        "phase1-gf16-seed74",
        "short-wave-bf16-sgdm",
        "phase1-f32-seed76",
        "scarab-fp8-seed86",
        "scarab-bfloat16-seed87",
        "scarab-gf20-seed75",
        "scarab-gf20-seed78",
        "scarab-gf12-seed78",
        "scarab-lion-m1-rng144",
        "scarab-muon-rng144",
        "scarab-soap",
        "scarab-gf16-lion-seed81",
        "scarab-gf16-soap-seed81",
        "scarab-gf16-muon-seed80",
        "scarab-shampoo-rng89",
        "scarab-lamb",
        "scarab-signum",
        "scarab-signum-rng144",
        "scarab-prodigy-rng144",
        "scarab-int8-seed80",
        "scarab-posit8-seed79",
        "phase1-gf16-seed77",
        "phase1-gf16-seed76",
        "phase1-bf16-seed74",
        "phase1-f32-seed74",
        "phase1-rng123",
        "phase1-rng144",
        "phase1-rng89",
        "phase1-rng47",
    ];
    let mut ok = 0;
    let mut fail = 0;
    for svc in &services {
        match gardener_harvest(svc) {
            Ok(_) => ok += 1,
            Err(e) => {
                eprintln!("[harvest-all] {} FAILED: {}", svc, e);
                fail += 1;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(300));
    }
    println!("\n=== harvest-all complete: {} OK, {} FAILED ===", ok, fail);
    Ok(())
}

fn gardener_prune() -> anyhow::Result<()> {
    println!("=== PRUNE ===");
    let out = std::process::Command::new("sh")
        .arg("-c")
        .arg("ps aux | grep trios-train | grep -v grep | grep 'defunct' | awk '{print $2}'")
        .output()?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let pids: Vec<&str> = stdout
        .lines()
        .filter(|l| !l.is_empty())
        .collect();
    if pids.is_empty() {
        println!("No zombie/defunct processes found.");
    } else {
        println!("Zombie PIDs: {}", pids.join(", "));
        for pid in &pids {
            println!("Killing zombie PID {} ...", pid);
            let _ = std::process::Command::new("kill").arg("-9").arg(pid).status();
        }
    }
    Ok(())
}

fn gardener_water() -> anyhow::Result<()> {
    println!("=== WATER (restart crashed) ===");
    let out = std::process::Command::new("sh")
        .arg("-c")
        .arg("grep -l 'CUDA out of memory\\|Killed\\|Segmentation fault' .trinity/results/*.log 2>/dev/null")
        .output()?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let logs: Vec<&str> = stdout
        .lines()
        .filter(|l| !l.is_empty())
        .collect();
    if logs.is_empty() {
        println!("No crashed logs found.");
    } else {
        for log in &logs {
            println!("Crashed: {}", log);
        }
        println!("Tip: re-run with `tri train` or redeploy with `tri deploy`.");
    }
    Ok(())
}

fn gardener_logs(service: &str, lines: usize) -> anyhow::Result<()> {
    let status = std::process::Command::new("railway")
        .args(["logs", "--service", service, "--lines", &lines.to_string()])
        .env("RAILWAY_NON_INTERACTIVE", "1")
        .status()?;
    if !status.success() {
        anyhow::bail!("railway logs failed");
    }
    Ok(())
}

fn gardener_report() -> anyhow::Result<()> {
    let report = ".trinity/gardener_report.html";
    if !std::path::Path::new(report).exists() {
        anyhow::bail!("Report not found: {}. Run Python builder first.", report);
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open").arg(report).status()?;
    }
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open").arg(report).status()?;
    }
    println!("Opened {}", report);
    Ok(())
}

fn hostname_or_default() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("MACHINE_ID"))
        .unwrap_or_else(|_| "local".to_string())
}
