//! Scarab Worker — один контейнер, стратегия из Neon.
//!
//! ENV:
//!   NEON_DATABASE_URL  — строка подключения к Neon Postgres
//!   SCARAB_ACCOUNT     — имя аккаунта воркера (acc0..acc5)
//!
//! Логика: бесконечный loop → claim_next → run_experiment → done.
//! Масштабирование: добавь ещё один сервис с другим SCARAB_ACCOUNT.
//! Изменение стратегии: INSERT в experiment_queue, контейнер не трогаем.

use std::env;
use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;
use tokio::time::sleep;
use tokio_postgres::NoTls;

#[derive(Debug, serde::Deserialize)]
struct ExpConfig {
    hidden: Option<u32>,
    lr: Option<f64>,
    steps: Option<u32>,
    ctx: Option<u32>,
    format: Option<String>,
    seed: Option<u64>,
}

struct Experiment {
    id: i64,
    canon_name: String,
    steps_budget: i32,
    config: ExpConfig,
}

async fn claim_next(
    client: &tokio_postgres::Client,
    account: &str,
) -> anyhow::Result<Option<Experiment>> {
    let row = client
        .query_opt(
            r#"
            UPDATE experiment_queue
            SET status = 'running', started_at = NOW()
            WHERE id = (
                SELECT id FROM experiment_queue
                WHERE status = 'pending'
                  AND account = $1
                ORDER BY priority DESC, id ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, canon_name, steps_budget, config_json
            "#,
            &[&account],
        )
        .await?;

    match row {
        None => Ok(None),
        Some(row) => {
            let config: ExpConfig =
                serde_json::from_value(row.get::<_, serde_json::Value>(3))?;
            Ok(Some(Experiment {
                id: row.get(0),
                canon_name: row.get(1),
                steps_budget: row.get(2),
                config,
            }))
        }
    }
}

async fn run_experiment(
    client: &tokio_postgres::Client,
    exp: Experiment,
    account: &str,
) -> anyhow::Result<()> {
    let c = &exp.config;
    let hidden = c.hidden.unwrap_or(828).to_string();
    let lr     = c.lr.unwrap_or(0.0004).to_string();
    let steps  = c.steps.unwrap_or(exp.steps_budget as u32).to_string();
    let ctx    = c.ctx.unwrap_or(12).to_string();
    let format = c.format.clone().unwrap_or_else(|| "fp32".into());
    let seed   = c.seed.unwrap_or(1597).to_string();
    let neon   = env::var("NEON_DATABASE_URL").unwrap_or_default();

    println!(
        "[scarab][{account}] START id={} name={} h={hidden} lr={lr} steps={steps} fmt={format}",
        exp.id, exp.canon_name
    );

    let exit = Command::new("trios-igla")
        .args([
            "train",
            "--hidden", &hidden,
            "--lr",     &lr,
            "--steps",  &steps,
            "--ctx",    &ctx,
            "--format", &format,
            "--seed",   &seed,
            "--exp-id", &exp.id.to_string(),
            "--neon-url", &neon,
        ])
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .await;

    let (status_str, err_msg): (&str, Option<String>) = match exit {
        Ok(s) if s.success() => ("done", None),
        Ok(s) => ("failed", Some(format!("exit: {s}"))),
        Err(e) => ("failed", Some(format!("spawn: {e}"))),
    };

    client
        .execute(
            "UPDATE experiment_queue \
             SET status=$1, finished_at=NOW(), error_msg=$2 WHERE id=$3",
            &[&status_str, &err_msg, &exp.id],
        )
        .await?;

    println!("[scarab][{account}] DONE id={} status={status_str}", exp.id);
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let db_url  = env::var("NEON_DATABASE_URL").expect("NEON_DATABASE_URL not set");
    let account = env::var("SCARAB_ACCOUNT").unwrap_or_else(|_| "acc0".into());
    let svc     = env::var("RAILWAY_SERVICE_NAME")
        .unwrap_or_else(|_| format!("scarab-{account}"));

    let (client, conn) = tokio_postgres::connect(&db_url, NoTls).await?;
    tokio::spawn(async move { let _ = conn.await; });

    // Регистрируем воркер
    let worker_id: String = client
        .query_one(
            "INSERT INTO workers (railway_acc, railway_svc_name, last_heartbeat, registered_at) \
             VALUES ($1,$2,NOW(),NOW()) RETURNING id::text",
            &[&account, &svc],
        )
        .await
        .map(|r| r.get(0))
        .unwrap_or_else(|_| "unknown".into());

    println!("[scarab][{account}] Ready. worker_id={worker_id}");

    loop {
        // Heartbeat каждые 30 с пока idle
        match claim_next(&client, &account).await {
            Ok(Some(exp)) => {
                let exp_id = exp.id;
                // Сообщаем DB что воркер занят
                let _ = client
                    .execute(
                        "UPDATE workers SET last_heartbeat=NOW(), current_exp_id=$1 \
                         WHERE id=$2::uuid",
                        &[&exp_id, &worker_id],
                    )
                    .await;

                run_experiment(&client, exp, &account).await
                    .unwrap_or_else(|e| eprintln!("[scarab] error: {e}"));

                // Освобождаем
                let _ = client
                    .execute(
                        "UPDATE workers SET current_exp_id=NULL, last_heartbeat=NOW() \
                         WHERE id=$1::uuid",
                        &[&worker_id],
                    )
                    .await;
            }
            Ok(None) => {
                let _ = client
                    .execute(
                        "UPDATE workers SET last_heartbeat=NOW() WHERE id=$1::uuid",
                        &[&worker_id],
                    )
                    .await;
                sleep(Duration::from_secs(10)).await;
            }
            Err(e) => {
                eprintln!("[scarab] claim error: {e}");
                sleep(Duration::from_secs(10)).await;
            }
        }
    }
}
