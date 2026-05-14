// SOVEREIGN SCARAB · pull-loop prototype
// Anchor: phi^2 + phi^-2 = 3 · TRINITY · DATABASE_URL (NEVER NEON_DATABASE_URL)
//
// Behaviour:
//   every POLL_SEC:
//     1. SELECT strategy FROM ssot.scarab_strategy WHERE service_id=$SERVICE_ID
//     2. write heartbeat (last_seen, current_gen, current_step, current_bpb)
//     3. if strategy.status='stop' → graceful shutdown
//     4. if strategy.generation > current_gen → kill running trainer, spawn new
//
// ENV vars:
//   DATABASE_URL    Postgres connection string (NOT NEON_DATABASE_URL)
//   SERVICE_ID      e.g. "local-A"
//   TRAINER_BIN     path to trios-train release binary
//   TRAIN_DATA      path to train.txt
//   VAL_DATA        path to val.txt
//   POLL_SEC        default 30
//   DRY_RUN         if "1" → skip spawning subprocess (heartbeat-only mode)

use anyhow::{Context, Result};
use std::env;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio_postgres::NoTls;

#[derive(Debug, Clone)]
struct Strategy {
    optimizer: String,
    format: String,
    hidden: i32,
    lr: f64,
    seed: i32,
    steps: i32,
    status: String,
    generation: i64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let database_url = env::var("DATABASE_URL").context("DATABASE_URL not set")?;
    let service_id = env::var("SERVICE_ID").context("SERVICE_ID not set")?;
    let trainer_bin = env::var("TRAINER_BIN").unwrap_or_else(|_|
        "/home/user/workspace/repos/gHashTag/trios-trainer-igla/target/release/trios-train".into());
    let train_data = env::var("TRAIN_DATA").unwrap_or_else(|_| "/tmp/honest_runs/train.txt".into());
    let val_data = env::var("VAL_DATA").unwrap_or_else(|_| "/tmp/honest_runs/val.txt".into());
    let poll_sec: u64 = env::var("POLL_SEC").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let dry_run = env::var("DRY_RUN").ok().as_deref() == Some("1");

    eprintln!("[scarab {service_id}] sovereign pull-loop starting · poll_sec={poll_sec} · dry_run={dry_run}");

    let (client, connection) = tokio_postgres::connect(&database_url, NoTls).await
        .context("postgres connect")?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("[scarab pg-conn] error: {e}");
        }
    });

    let started_at = chrono::Utc::now();
    let current_gen = Arc::new(Mutex::new(0_i64));
    let current_step = Arc::new(Mutex::new(0_i32));
    let current_bpb = Arc::new(Mutex::new(None::<f64>));
    let trainer_child: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));

    loop {
        // 1. fetch strategy
        let row_opt = client.query_opt(
            "SELECT optimizer, format, hidden, lr::float8 AS lr, seed, steps, status, generation \
             FROM ssot.scarab_strategy WHERE service_id=$1",
            &[&service_id],
        ).await?;

        let Some(row) = row_opt else {
            eprintln!("[scarab {service_id}] no strategy row — sleeping");
            tokio::time::sleep(Duration::from_secs(poll_sec)).await;
            continue;
        };

        let strategy = Strategy {
            optimizer:  row.get(0),
            format:     row.get(1),
            hidden:     row.get(2),
            lr:         row.get(3),
            seed:       row.get(4),
            steps:      row.get(5),
            status:     row.get(6),
            generation: row.get(7),
        };

        // 2. heartbeat
        let pid = trainer_child.lock().await.as_ref().map(|c| c.id() as i32).unwrap_or(0);
        let gen = *current_gen.lock().await;
        let step = *current_step.lock().await;
        let bpb = *current_bpb.lock().await;
        client.execute(
            "INSERT INTO ssot.scarab_heartbeat \
               (service_id, last_seen, current_gen, current_step, current_bpb, pid, started_at) \
             VALUES ($1, now(), $2, $3, $4, $5, $6) \
             ON CONFLICT (service_id) DO UPDATE SET \
               last_seen   = EXCLUDED.last_seen, \
               current_gen = EXCLUDED.current_gen, \
               current_step= EXCLUDED.current_step, \
               current_bpb = EXCLUDED.current_bpb, \
               pid         = EXCLUDED.pid, \
               started_at  = EXCLUDED.started_at",
            &[&service_id, &gen, &step, &bpb, &pid, &started_at],
        ).await?;

        // 3. stop?
        if strategy.status == "stop" {
            eprintln!("[scarab {service_id}] strategy.status=stop · graceful shutdown");
            if let Some(mut child) = trainer_child.lock().await.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
            return Ok(());
        }

        // 4. generation bump?
        if strategy.generation > gen {
            eprintln!(
                "[scarab {service_id}] gen {} → {} · respawn trainer · opt={} fmt={} h={} lr={} seed={} steps={}",
                gen, strategy.generation,
                strategy.optimizer, strategy.format, strategy.hidden, strategy.lr, strategy.seed, strategy.steps
            );
            if let Some(mut child) = trainer_child.lock().await.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
            if !dry_run {
                let mut cmd = Command::new(&trainer_bin);
                cmd.env("TRIOS_FORMAT_TYPE", &strategy.format)
                   .env("TRIOS_DISABLE_NEON", "1")
                   .env_remove("DATABASE_URL")
                   .env_remove("TRIOS_DATABASE_URL")
                   .args([
                       "--seed", &strategy.seed.to_string(),
                       "--steps", &strategy.steps.to_string(),
                       "--lr", &strategy.lr.to_string(),
                       "--hidden", &strategy.hidden.to_string(),
                       "--attn-layers", "1",
                       "--optimizer", &strategy.optimizer,
                       "--train-data", &train_data,
                       "--val-data", &val_data,
                       "--eval-every", "100",
                   ])
                   .stdout(Stdio::piped())
                   .stderr(Stdio::piped());
                let child = cmd.spawn().context("spawn trainer")?;
                eprintln!("[scarab {service_id}] spawned pid={}", child.id());
                *trainer_child.lock().await = Some(child);
            } else {
                eprintln!("[scarab {service_id}] DRY_RUN=1 → trainer NOT spawned");
            }
            *current_gen.lock().await = strategy.generation;
            *current_step.lock().await = 0;
            *current_bpb.lock().await = None;
        }

        // 5. is trainer still alive?
        {
            let mut guard = trainer_child.lock().await;
            if let Some(child) = guard.as_mut() {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        eprintln!("[scarab {service_id}] trainer exited status={status}");
                        *guard = None;
                    }
                    Ok(None) => { /* still running */ }
                    Err(e) => eprintln!("[scarab {service_id}] try_wait error: {e}"),
                }
            }
        }

        tokio::time::sleep(Duration::from_secs(poll_sec)).await;
    }
}
