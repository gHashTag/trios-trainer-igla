//! 🪲 Scarab Worker — Stateless Fungible Pool
//!
//! Правило №1: Контейнер не трогаем НИКОГДА.
//! Правило №2: Стратегию меняем ТОЛЬКО через Neon.
//! Правило №3: Нет account affinity. Любой скарабей берёт любую задачу.
//!
//! ENV:
//!   NEON_DATABASE_URL  — строка подключения к Neon Postgres (обязательно)
//!   SCARAB_ACCOUNT     — имя для логов (необязательно, по умолчанию "scarab")

use std::env;
use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;
use tokio::time::sleep;
use tokio_postgres::{AsyncMessage, NoTls};
use futures_util::StreamExt;

// ── StrategySpec: полная спецификация из JSONB ───────────────────────────

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

/// Полная спецификация стратегии (живёт целиком в config_json).
/// Нет никакой привязки к acc.
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

// ── claim_any_pending: БЕЗ acc filter ─────────────────────────────────────

async fn claim_any_pending(
    client: &tokio_postgres::Client,
) -> anyhow::Result<Option<Strategy>> {
    let row = client
        .query_opt(
            // ❤️ Ключевое место: НЕТ AND account = $1
            // Любой свободный скарабей берёт любую пендинг задачу.
            r#"
            UPDATE strategy_queue
            SET status = 'running',
                started_at = NOW(),
                worker_id = $1
            WHERE id = (
                SELECT id FROM strategy_queue
                WHERE status = 'pending'
                ORDER BY priority DESC, id ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, canon_name, steps_budget, config_json
            "#,
            &[&env::var("HOSTNAME").unwrap_or_else(|_| "scarab".into())],
        )
        .await?;

    match row {
        None => Ok(None),
        Some(row) => {
            let raw: serde_json::Value = row.get(3);
            // Поддерживаем оба формата:
            // 1) Новый: {"trainer":{...}, "constraints":{...}, "submission":{...}}
            // 2) Легаси: {"hidden":828, "lr":0.0004, ...}
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
    }
}

// ── run_strategy ───────────────────────────────────────────────────────────

async fn run_strategy(
    client: &tokio_postgres::Client,
    strat: Strategy,
    label: &str,
) -> anyhow::Result<()> {
    let t = &strat.spec.trainer;
    let hidden = t.hidden.unwrap_or(828).to_string();
    let lr     = t.lr.unwrap_or(0.0004).to_string();
    let steps  = t.steps.unwrap_or(strat.steps_budget as u32).to_string();
    let ctx    = t.ctx.unwrap_or(12).to_string();
    let format = t.format.clone().unwrap_or_else(|| "fp32".into());
    let seed   = t.seed.unwrap_or(1597).to_string();
    let neon   = env::var("NEON_DATABASE_URL").unwrap_or_default();

    // Timeout watchdog
    let max_secs = strat.spec.constraints.max_runtime_sec.unwrap_or(900);

    println!(
        "[{label}] START id={} name={} h={hidden} lr={lr} steps={steps} fmt={format} seed={seed}",
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
        match tokio::time::timeout(
            Duration::from_secs(max_secs),
            cmd.status(),
        ).await
    {
        Ok(Ok(s)) if s.success() => ("done", None),
        Ok(Ok(s)) => ("failed", Some(format!("exit: {s}"))),
        Ok(Err(e)) => ("failed", Some(format!("spawn: {e}"))),
        Err(_) => ("failed", Some(format!("timeout after {max_secs}s"))),
    };

    client
        .execute(
            "UPDATE strategy_queue \
             SET status=$1, finished_at=NOW(), error_msg=$2 WHERE id=$3",
            &[&status_str, &err_msg, &strat.id],
        )
        .await?;

    println!("[{label}] DONE id={} status={status_str}", strat.id);
    Ok(())
}

// ── register + heartbeat ─────────────────────────────────────────────

async fn register_scarab(
    client: &tokio_postgres::Client,
    label: &str,
    host: &str,
) -> anyhow::Result<String> {
    let r = client
        .query_one(
            "INSERT INTO scarabs (label, host, last_heartbeat, registered_at) \
             VALUES ($1,$2,NOW(),NOW()) RETURNING id::text",
            &[&label, &host],
        )
        .await
        .unwrap_or_else(|_| panic!("Cannot register scarab — run migrations/002 first"));
    Ok(r.get(0))
}

async fn heartbeat(
    client: &tokio_postgres::Client,
    scarab_id: &str,
    current_id: Option<i64>,
) {
    let _ = client
        .execute(
            "UPDATE scarabs SET last_heartbeat=NOW(), current_strategy_id=$1 WHERE id=$2::uuid",
            &[&current_id, &scarab_id],
        )
        .await;
}

// ── LISTEN channel helper ───────────────────────────────────────────
// tokio_postgres не поддерживает LISTEN напрямую —
// нужно получать AsyncMessage из connection stream.
// Но для простоты используем channel через mpsc.

async fn setup_listen(
    db_url: &str,
) -> anyhow::Result<tokio::sync::mpsc::Receiver<()>> {
    let (notify_tx, notify_rx) = tokio::sync::mpsc::channel::<()>(16);
    let db_url = db_url.to_owned();

    tokio::spawn(async move {
        loop {
            match tokio_postgres::connect(&db_url, NoTls).await {
                Err(e) => {
                    eprintln!("[notify] connect error: {e}");
                    sleep(Duration::from_secs(5)).await;
                }
                Ok((client, mut conn)) => {
                    if let Err(e) = client.execute("LISTEN strategy_new", &[]).await {
                        eprintln!("[notify] LISTEN error: {e}");
                        sleep(Duration::from_secs(5)).await;
                        continue;
                    }
                    loop {
                        match futures_util::poll!(std::pin::Pin::new(&mut conn)) {
                            std::task::Poll::Ready(Err(e)) => {
                                eprintln!("[notify] conn dropped: {e}");
                                break;
                            }
                            _ => {}
                        }
                        // Просто ждём сообщения через poll
                        sleep(Duration::from_millis(200)).await;
                        let _ = notify_tx.try_send(());
                    }
                    sleep(Duration::from_secs(2)).await;
                }
            }
        }
    });
    Ok(notify_rx)
}

// ── main loop ─────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let db_url  = env::var("NEON_DATABASE_URL").expect("NEON_DATABASE_URL not set");
    // SCARAB_ACCOUNT теперь только для логов — не влияет на то, какие задачи берёт
    let label   = env::var("SCARAB_ACCOUNT").unwrap_or_else(|_| "scarab".into());
    let host    = env::var("HOSTNAME").unwrap_or_else(|_| "unknown".into());

    let (client, conn) = tokio_postgres::connect(&db_url, NoTls).await?;
    tokio::spawn(async move { let _ = conn.await; });

    let scarab_id = register_scarab(&client, &label, &host).await
        .unwrap_or_else(|_| "unknown".into());

    println!("[✨ scarab][{label}] Ready. id={scarab_id} host={host}");
    println!("[✨ scarab][{label}] Fungible pool: NO account filter.");

    let mut notify_rx = setup_listen(&db_url).await?;

    loop {
        // Дрейним все pending задачи до пустой очереди
        loop {
            match claim_any_pending(&client).await {
                Ok(Some(strat)) => {
                    let sid = strat.id;
                    heartbeat(&client, &scarab_id, Some(sid)).await;
                    run_strategy(&client, strat, &label).await
                        .unwrap_or_else(|e| eprintln!("[{label}] error: {e}"));
                    heartbeat(&client, &scarab_id, None).await;
                }
                Ok(None) => break, // Очередь пуста
                Err(e) => {
                    eprintln!("[{label}] claim error: {e}");
                    break;
                }
            }
        }

        // Спим до следующего NOTIFY или 30 с fallback
        heartbeat(&client, &scarab_id, None).await;
        tokio::select! {
            _ = notify_rx.recv() => {
                println!("[{label}] NOTIFY strategy_new received");
            }
            _ = sleep(Duration::from_secs(30)) => {}
        }
    }
}
