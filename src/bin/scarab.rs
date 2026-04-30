//! Scarab Worker — stateless fungible pool.
//!
//! Rule 1: Never touch the container.
//! Rule 2: Change strategy only through Neon.
//! Rule 3: No account affinity. Any scarab takes any task.
//!
//! ENV:
//!   NEON_DATABASE_URL  — Neon Postgres connection string (required)
//!   SCARAB_ACCOUNT     — identity tag for logs only (optional)
//!                        NOT a routing key. Does not filter claim.
//!
//! Scaling: deploy another container. Done.
//! Strategy change: INSERT into strategy_queue. Container untouched.

use std::env;
use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;
use tokio::time::sleep;
use tokio_postgres::NoTls;

// ── StrategySpec: full spec lives in config_json JSONB ──────────────────────

#[derive(Debug, serde::Deserialize, Default)]
struct TrainerSpec {
    hidden: Option<u32>,
    lr: Option<f64>,
    steps: Option<u32>,
    ctx: Option<u32>,
    format: Option<String>,
    seed: Option<u64>,
    val_split_seed: Option<String>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct ConstraintsSpec {
    max_runtime_sec: Option<u64>,
    min_step_for_done: Option<u32>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct SubmissionSpec {
    track: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct StrategySpec {
    #[serde(default)]
    trainer: TrainerSpec,
    #[serde(default)]
    constraints: ConstraintsSpec,
    #[serde(default)]
    submission: SubmissionSpec,
}

struct Strategy {
    id: i64,
    canon_name: String,
    steps_budget: i32,
    spec: StrategySpec,
}

// ── claim_any_pending ────────────────────────────────────────────────────────

async fn claim_any_pending(
    client: &tokio_postgres::Client,
    worker_host: &str,
) -> anyhow::Result<Option<Strategy>> {
    let row = client
        .query_opt(
            r#"
            UPDATE strategy_queue
            SET status     = 'running',
                started_at = NOW(),
                worker_id  = $1
            WHERE id = (
                SELECT id FROM strategy_queue
                WHERE status = 'pending'
                ORDER BY priority DESC, id ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, canon_name, steps_budget, config_json
            "#,
            &[&worker_host],
        )
        .await?;

    let Some(row) = row else { return Ok(None) };

    let cfg_str: String = row.get(3);
    let raw: serde_json::Value = serde_json::from_str(&cfg_str)?;
    let spec: StrategySpec = if raw.get("trainer").is_some() {
        serde_json::from_value(raw)?
    } else {
        StrategySpec {
            trainer: serde_json::from_value(raw)?,
            ..Default::default()
        }
    };

    Ok(Some(Strategy {
        id: row.get(0),
        canon_name: row.get(1),
        steps_budget: row.get(2),
        spec,
    }))
}

// ── run_strategy ─────────────────────────────────────────────────────────────

async fn run_strategy(
    client: &tokio_postgres::Client,
    strat: Strategy,
    label: &str,
) -> anyhow::Result<()> {
    let t = &strat.spec.trainer;
    let hidden = t.hidden.unwrap_or(828).to_string();
    let lr = t.lr.unwrap_or(0.0004).to_string();
    let steps = t.steps.unwrap_or(strat.steps_budget as u32).to_string();
    let ctx = t.ctx.unwrap_or(12).to_string();
    let format = t.format.clone().unwrap_or_else(|| "fp32".into());
    let seed = t.seed.unwrap_or(1597).to_string();
    let neon = env::var("NEON_DATABASE_URL").unwrap_or_default();
    let max_secs = strat.spec.constraints.max_runtime_sec.unwrap_or(900);

    println!(
        "[{label}] START id={} name={} hidden={hidden} lr={lr} steps={steps} fmt={format} seed={seed}",
        strat.id, strat.canon_name
    );

    let mut cmd = Command::new("trios-igla");
    cmd.args([
        "train",
        "--hidden", &hidden,
        "--lr",     &lr,
        "--steps",  &steps,
        "--ctx",    &ctx,
        "--format", &format,
        "--seed",   &seed,
        "--exp-id", &strat.id.to_string(),
        "--neon-url", &neon,
    ])
    .stdout(Stdio::inherit())
    .stderr(Stdio::inherit());

    let (status_str, err_msg): (&str, Option<String>) =
        match tokio::time::timeout(Duration::from_secs(max_secs), cmd.status()).await {
            Ok(Ok(s)) if s.success() => ("done", None),
            Ok(Ok(s)) => ("failed", Some(format!("exit: {s}"))),
            Ok(Err(e)) => ("failed", Some(format!("spawn error: {e}"))),
            Err(_) => ("failed", Some(format!("timeout after {max_secs}s"))),
        };

    client
        .execute(
            "UPDATE strategy_queue \
             SET status = $1, finished_at = NOW(), error_msg = $2 \
             WHERE id = $3",
            &[&status_str, &err_msg, &strat.id],
        )
        .await?;

    println!("[{label}] DONE id={} status={status_str}", strat.id);
    Ok(())
}

// ── register + heartbeat ─────────────────────────────────────────────────────

/// Uses actual scarabs table columns:
///   railway_acc, railway_svc_name, host, last_heartbeat, registered_at
async fn register_scarab(client: &tokio_postgres::Client, label: &str, host: &str) -> String {
    client
        .query_one(
            "INSERT INTO scarabs \
             (railway_acc, railway_svc_name, host, last_heartbeat, registered_at) \
             VALUES ($1, $2, $3, NOW(), NOW()) RETURNING id::text",
            &[&label, &label, &host],
        )
        .await
        .map(|r| r.get::<_, String>(0))
        .unwrap_or_else(|e| {
            eprintln!("[scarab] register failed: {e}");
            "unknown".into()
        })
}

/// Uses actual column 'current_exp_id' (not 'current_strategy_id').
async fn heartbeat(client: &tokio_postgres::Client, scarab_id: &str, current_id: Option<i64>) {
    if scarab_id == "unknown" {
        return;
    }
    let _ = client
        .execute(
            "UPDATE scarabs \
             SET last_heartbeat = NOW(), current_exp_id = $1 \
             WHERE id = $2::uuid",
            &[&current_id, &scarab_id],
        )
        .await;
}

// ── LISTEN/NOTIFY ────────────────────────────────────────────────────────────────

async fn setup_notify_listener(db_url: &str) -> tokio::sync::mpsc::Receiver<()> {
    let (tx, rx) = tokio::sync::mpsc::channel::<()>(16);
    let db_url = db_url.to_owned();

    tokio::spawn(async move {
        loop {
            let Ok((client, conn)) = tokio_postgres::connect(&db_url, NoTls).await else {
                sleep(Duration::from_secs(5)).await;
                continue;
            };
            tokio::spawn(async move { let _ = conn.await; });

            if client.execute("LISTEN strategy_new", &[]).await.is_err() {
                sleep(Duration::from_secs(5)).await;
                continue;
            }

            loop {
                sleep(Duration::from_millis(500)).await;
                if tx.try_send(()).is_err() {
                    break;
                }
            }
        }
    });

    rx
}

// ── main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let db_url = env::var("NEON_DATABASE_URL").expect("NEON_DATABASE_URL not set");
    let label = env::var("SCARAB_ACCOUNT").unwrap_or_else(|_| "scarab".into());
    let host = env::var("HOSTNAME").unwrap_or_else(|_| "unknown".into());

    let (client, conn) = tokio_postgres::connect(&db_url, NoTls).await?;
    tokio::spawn(async move { let _ = conn.await; });

    let scarab_id = register_scarab(&client, &label, &host).await;
    println!("[scarab][{label}] ready | id={scarab_id} host={host}");
    println!("[scarab][{label}] fungible pool — no account filter");

    let mut notify_rx = setup_notify_listener(&db_url).await;

    loop {
        loop {
            match claim_any_pending(&client, &host).await {
                Ok(Some(strat)) => {
                    let sid = strat.id;
                    heartbeat(&client, &scarab_id, Some(sid)).await;
                    run_strategy(&client, strat, &label)
                        .await
                        .unwrap_or_else(|e| eprintln!("[{label}] run error: {e}"));
                    heartbeat(&client, &scarab_id, None).await;
                }
                Ok(None) => break,
                Err(e) => {
                    eprintln!("[{label}] claim error: {e}");
                    break;
                }
            }
        }

        heartbeat(&client, &scarab_id, None).await;
        tokio::select! {
            _ = notify_rx.recv() => {},
            _ = sleep(Duration::from_secs(30)) => {},
        }
    }
}
