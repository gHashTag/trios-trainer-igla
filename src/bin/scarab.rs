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
use std::sync::Arc;
use std::time::Duration;

use tokio::process::Command;
use tokio::time::sleep;
use tokio_postgres_rustls::MakeRustlsConnect;

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
    /// Override train data path (default: /work/data/tiny_shakespeare.txt)
    train_path: Option<String>,
    /// Override val data path (default: /work/data/tiny_shakespeare_val.txt)
    val_path: Option<String>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct ConstraintsSpec {
    /// Hard wall-clock timeout per experiment (seconds).
    max_runtime_sec: Option<u64>,
    min_step_for_done: Option<u32>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct SubmissionSpec {
    track: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

/// Full strategy specification stored in config_json.
/// No account binding. No worker affinity.
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

/// Claim the next pending strategy from the global pool.
///
/// Critically: NO `AND account = $1` filter.
/// Any free scarab takes any pending task.
/// `FOR UPDATE SKIP LOCKED` ensures two scarabs never race on the same row.
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

    // tokio-postgres here lacks the with-serde_json-1 feature, so JSONB
    // must be read as text and parsed. Safe because the column is JSONB
    // and Postgres emits valid JSON text. Same fix applied in PR #64.
    let cfg_str: String = row.get(3);
    let raw: serde_json::Value = serde_json::from_str(&cfg_str)?;
    // Support both formats:
    //   new:    { "trainer": {...}, "constraints": {...}, "submission": {...} }
    //   legacy: { "hidden": 828, "lr": 0.0004, ... }
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
    // Bug C fix: Read corpus paths from config_json.data.{train_path,val_path}
    // This respects explicit corpus tags in strategy_queue entries.
    // Falls back to tiny_shakespeare defaults for backward compatibility.
    let train_path = t
        .train_path
        .clone()
        .unwrap_or_else(|| "/work/data/tiny_shakespeare.txt".into());
    let val_path = t
        .val_path
        .clone()
        .unwrap_or_else(|| "/work/data/tiny_shakespeare_val.txt".into());
    let neon = env::var("NEON_DATABASE_URL").unwrap_or_default();
    let max_secs = strat.spec.constraints.max_runtime_sec.unwrap_or(900);

    println!(
        "[{label}] START id={} name={} hidden={hidden} lr={lr} steps={steps} fmt={format} seed={seed} train={train_path}",
        strat.id, strat.canon_name
    );

    let mut cmd = Command::new("trios-train");
    cmd.args([
        "--hidden",
        &hidden,
        "--lr",
        &lr,
        "--steps",
        &steps,
        "--ctx",
        &ctx,
        "--format",
        &format,
        "--seed",
        &seed,
        "--train-data",
        &train_path,
        "--val-data",
        &val_path,
    ])
    .env("TRIOS_EXPERIMENT_ID", strat.id.to_string())
    .env("TRIOS_CANON_NAME", &strat.canon_name)
    // Bug A fix: Explicitly forward Neon DSN to trainer subprocess.
    .env("NEON_DATABASE_URL", &neon)
    .stdout(Stdio::inherit())
    .stderr(Stdio::inherit());

    let (status_str, err_msg): (&str, Option<String>) =
        match tokio::time::timeout(Duration::from_secs(max_secs), cmd.status()).await {
            Ok(Ok(s)) if s.success() => ("done", None),
            Ok(Ok(s)) => ("failed", Some(format!("exit: {s}"))),
            Ok(Err(e)) => ("failed", Some(format!("spawn error: {e}"))),
            Err(_) => ("failed", Some(format!("timeout after {max_secs}s"))),
        };

    // Bug 3 fix: write final_bpb / final_step back to strategy_queue.
    // Query the latest bpb_samples row for this canon_name.
    let mut final_bpb: Option<f64> = None;
    let mut final_step: Option<i32> = None;
    if status_str == "done" {
        match client
            .query_opt(
                "SELECT step, bpb FROM bpb_samples \
                 WHERE canon_name = $1 \
                 ORDER BY ts DESC LIMIT 1",
                &[&strat.canon_name],
            )
            .await
        {
            Ok(Some(row)) => {
                final_step = Some(row.get::<_, i32>(0));
                final_bpb = Some(row.get::<_, f64>(1));
                println!(
                    "[{label}] final_bpb={:.4} final_step={} for id={}",
                    final_bpb.unwrap(),
                    final_step.unwrap(),
                    strat.id
                );
            }
            Ok(None) => {
                eprintln!(
                    "[{label}] WARNING: trainer exited clean but 0 bpb_samples rows for canon={}",
                    strat.canon_name
                );
            }
            Err(e) => {
                eprintln!("[{label}] bpb_samples query error: {e}");
            }
        }
    }

    client
        .execute(
            "UPDATE strategy_queue \
             SET status = $1, finished_at = NOW(), last_error = $2, \
                 final_bpb = $3, final_step = $4 \
             WHERE id = $5",
            &[&status_str, &err_msg, &final_bpb, &final_step, &strat.id],
        )
        .await?;

    println!("[{label}] DONE id={} status={status_str}", strat.id);
    Ok(())
}

// ── register + heartbeat ─────────────────────────────────────────────────────

async fn register_scarab(
    client: &tokio_postgres::Client,
    acc: &str,
    svc_id: &str,
    svc_name: &str,
    host: &str,
) -> String {
    client
        .query_one(
            "INSERT INTO scarabs (railway_acc, railway_svc_id, railway_svc_name, host, last_heartbeat, registered_at) \
             VALUES ($1, $2, $3, $4, NOW(), NOW()) RETURNING id::text",
            &[&acc, &svc_id, &svc_name, &host],
        )
        .await
        .map(|r| r.get::<_, String>(0))
        .unwrap_or_else(|e| {
            eprintln!("[scarab] register failed (check scarabs schema): {e}");
            "unknown".into()
        })
}

async fn heartbeat(client: &tokio_postgres::Client, scarab_id: &str, current_id: Option<i64>) {
    let _ = client
        .execute(
            "UPDATE scarabs \
             SET last_heartbeat = NOW(), current_exp_id = $1 \
             WHERE id = $2::uuid",
            &[&current_id, &scarab_id],
        )
        .await;
}

// ── LISTEN/NOTIFY helper ──────────────────────────────────────────────────────

/// Opens a dedicated connection for NOTIFY and forwards wakeups via mpsc.
/// Falls back to 30-second polling if the notify connection drops.
fn make_tls_config() -> rustls::ClientConfig {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth()
}

async fn connect_with_retry(db_url: &str) -> anyhow::Result<tokio_postgres::Client> {
    let tls_config = make_tls_config();
    let max_attempts: u32 = 30;
    for attempt in 1..=max_attempts {
        let tls = MakeRustlsConnect::new(tls_config.clone());
        match tokio_postgres::connect(db_url, tls).await {
            Ok((client, conn)) => {
                tokio::spawn(async move {
                    let _ = conn.await;
                });
                return Ok(client);
            }
            Err(e) => {
                eprintln!("[scarab] connect attempt {attempt}/{max_attempts} failed: {e}");
                if attempt >= max_attempts {
                    anyhow::bail!("connect failed after {max_attempts} attempts: {e:#}");
                }
                sleep(Duration::from_secs(2)).await;
            }
        }
    }
    unreachable!()
}

async fn setup_notify_listener(db_url: &str) -> tokio::sync::mpsc::Receiver<()> {
    let (tx, rx) = tokio::sync::mpsc::channel::<()>(16);
    let db_url = db_url.to_owned();
    let tls_config = make_tls_config();

    tokio::spawn(async move {
        loop {
            let tls = MakeRustlsConnect::new(tls_config.clone());
            let Ok((client, conn)) = tokio_postgres::connect(&db_url, tls).await else {
                sleep(Duration::from_secs(5)).await;
                continue;
            };
            tokio::spawn(async move {
                let _ = conn.await;
            });

            if client.execute("LISTEN strategy_new", &[]).await.is_err() {
                sleep(Duration::from_secs(5)).await;
                continue;
            }

            // Any incoming message triggers a claim attempt.
            loop {
                sleep(Duration::from_millis(500)).await;
                if tx.try_send(()).is_err() {
                    break; // channel full or closed — reconnect
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
    // RAILWAY_ACC identifies which account this scarab runs on (cosmetic, NOT a routing key).
    let acc = env::var("RAILWAY_ACC")
        .unwrap_or_else(|_| env::var("SCARAB_ACCOUNT").unwrap_or_else(|_| "scarab".into()));
    let svc_id = env::var("RAILWAY_SERVICE_ID").unwrap_or_else(|_| "unknown".into());
    let svc_name = env::var("RAILWAY_SERVICE_NAME").unwrap_or_else(|_| "scarab".into());
    let host = env::var("HOSTNAME").unwrap_or_else(|_| "unknown".into());

    let client = connect_with_retry(&db_url).await?;

    let scarab_id = register_scarab(&client, &acc, &svc_id, &svc_name, &host).await;
    println!("[scarab][{acc}] ready | id={scarab_id} host={host} svc={svc_name}");
    println!("[scarab][{acc}] fungible pool — no account filter");

    let mut notify_rx = setup_notify_listener(&db_url).await;

    loop {
        // Drain all pending strategies before sleeping.
        loop {
            match claim_any_pending(&client, &host).await {
                Ok(Some(strat)) => {
                    let sid = strat.id;
                    heartbeat(&client, &scarab_id, Some(sid)).await;
                    run_strategy(&client, strat, &acc)
                        .await
                        .unwrap_or_else(|e| eprintln!("[{acc}] run error: {e}"));
                    heartbeat(&client, &scarab_id, None).await;
                }
                Ok(None) => break, // queue empty
                Err(e) => {
                    eprintln!("[{acc}] claim error: {e}");
                    break;
                }
            }
        }

        // Sleep until next NOTIFY or 30-second fallback.
        heartbeat(&client, &scarab_id, None).await;
        tokio::select! {
            _ = notify_rx.recv() => {},
            _ = sleep(Duration::from_secs(30)) => {},
        }
    }
}
