// SOVEREIGN SCARAB · pull-loop v4 — Eternal Agent on Railway
// ---------------------------------------------------------------------------
// Anchor: phi^2 + phi^-2 = 3 · TRINITY · DATABASE_URL only · ADR-CHAT-012
//
// v4 changes from v3:
//   • LISTEN/NOTIFY hybrid wakeup (channel `scarab_<service>` & `scarab_fleet`)
//   • Heartbeat writes applied_version (=local_gen) so Queen's drift view works
//   • RAILWAY_SERVICE_ID env preferred over SERVICE_ID
//   • Reports applied_fingerprint in scarab_result for forensic replay
//   • Library split (src/lib.rs) → unit-tested logic
//
// Lifecycle:
//   loop {
//     race [ LISTEN-NOTIFY future, sleep(poll_sec) ]
//     pull strategy from ssot.scarab_strategy WHERE service_id=$me
//     match status { killed→exit, draining→drain&exit, paused→pause loop,
//                    active→if gen>local: graceful restart trainer }
//     heartbeat (last_seen, applied_version, current step/bpb)
//     drain DONE→write scarab_result
//   }
// ---------------------------------------------------------------------------

use anyhow::{Context, Result};
use scarab_pull_loop::{canon_name, graceful_kill, Config, DoneEvent, Strategy};
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex};
use tokio_postgres::{AsyncMessage, NoTls};
use futures_util::stream::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = Config::from_env()?;
    eprintln!(
        "[scarab {sid}] v4 eternal · poll={p}s · grace={g}ms · dry={d} · replay={r} · listen={l}",
        sid = cfg.service_id, p = cfg.poll_sec, g = cfg.grace_ms,
        d = cfg.dry_run, r = cfg.auto_replay, l = cfg.listen_notify
    );

    // ------------ Postgres connection (worker-task channel for NOTIFY) ------
    let (client, mut connection) = tokio_postgres::connect(&cfg.database_url, NoTls)
        .await
        .context("postgres connect")?;

    // Channel between pg-connection-task and main loop: each NOTIFY pushes a generation.
    let (notify_tx, mut notify_rx) = mpsc::unbounded_channel::<i64>();

    // Drive the postgres connection. tokio_postgres needs `poll_message` to get NOTIFYs;
    // we use `futures_util::stream::poll_fn` via the convenience adapter on the connection.
    tokio::spawn(async move {
        let mut stream = futures_util::stream::poll_fn(move |cx| connection.poll_message(cx));
        while let Some(msg) = stream.next().await {
            match msg {
                Ok(AsyncMessage::Notification(n)) => {
                    if let Ok(g) = n.payload().parse::<i64>() {
                        let _ = notify_tx.send(g);
                    } else {
                        // scarab_fleet payload is JSON; ignore
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("[scarab pg-conn] {e}");
                    break;
                }
            }
        }
    });

    let client = Arc::new(client);

    // ------------ Subscribe to LISTEN channel for this scarab ---------------
    if cfg.listen_notify {
        let chan = format!("scarab_{}", cfg.service_id.replace('-', "_"));
        // tokio-postgres requires identifiers to be quoted; `LISTEN` doesn't accept params.
        // Channel name is derived from service_id which is operator-controlled — sanitize:
        // Only allow alnum + underscore (replace anything else with _).
        let sanitized: String = chan
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
            .collect();
        let stmt = format!("LISTEN {sanitized}");
        match client.batch_execute(&stmt).await {
            Ok(_) => eprintln!("[scarab {}] LISTENing on {sanitized}", cfg.service_id),
            Err(e) => eprintln!("[scarab {}] LISTEN failed: {e}", cfg.service_id),
        }
    }

    let started_at = chrono::Utc::now();
    let local_gen = Arc::new(Mutex::new(0_i64));
    let current_step = Arc::new(Mutex::new(0_i32));
    let current_bpb = Arc::new(Mutex::new(None::<f64>));
    let trainer_child: Arc<Mutex<Option<(Child, Instant, Strategy)>>> = Arc::new(Mutex::new(None));
    let (done_tx, mut done_rx) = mpsc::unbounded_channel::<DoneEvent>();

    loop {
        // 0. Wait for either NOTIFY or poll timer (whichever first)
        let sleep = tokio::time::sleep(Duration::from_secs(cfg.poll_sec));
        tokio::pin!(sleep);
        tokio::select! {
            _ = &mut sleep => {},
            maybe_gen = notify_rx.recv() => {
                if let Some(g) = maybe_gen {
                    eprintln!("[scarab {}] NOTIFY received gen={g}", cfg.service_id);
                }
            },
        };

        // 1. Fetch strategy
        let row_opt = client
            .query_opt(
                "SELECT optimizer, format, hidden, lr::float8 AS lr, seed, steps, status, generation \
                 FROM ssot.scarab_strategy WHERE service_id=$1",
                &[&cfg.service_id],
            )
            .await?;

        let Some(row) = row_opt else {
            eprintln!("[scarab {}] no strategy row · sleeping", cfg.service_id);
            continue;
        };

        let strategy = Strategy {
            optimizer: row.get(0),
            format: row.get(1),
            hidden: row.get(2),
            lr: row.get(3),
            seed: row.get(4),
            steps: row.get(5),
            status: row.get(6),
            generation: row.get(7),
        };

        // 2. Heartbeat (always — Queen needs to know we're alive even if paused)
        let pid = trainer_child
            .lock()
            .await
            .as_ref()
            .map(|c| c.0.id() as i32)
            .unwrap_or(0);
        let l_gen = *local_gen.lock().await;
        let step = *current_step.lock().await;
        let bpb = *current_bpb.lock().await;
        // v4: write applied_version = local_gen so ssot.fleet_status drift works.
        if let Err(e) = client
            .execute(
                "INSERT INTO ssot.scarab_heartbeat \
                   (service_id, last_seen, current_gen, current_step, current_bpb, pid, started_at, applied_version) \
                 VALUES ($1, now(), $2, $3, $4, $5, $6, $2) \
                 ON CONFLICT (service_id) DO UPDATE SET \
                   last_seen        = EXCLUDED.last_seen, \
                   current_gen      = EXCLUDED.current_gen, \
                   current_step     = EXCLUDED.current_step, \
                   current_bpb      = EXCLUDED.current_bpb, \
                   pid              = EXCLUDED.pid, \
                   started_at       = EXCLUDED.started_at, \
                   applied_version  = EXCLUDED.applied_version",
                &[&cfg.service_id, &l_gen, &step, &bpb, &pid, &started_at],
            )
            .await
        {
            eprintln!("[scarab {}] heartbeat err: {e}", cfg.service_id);
        }

        // 3. Status state-machine
        match strategy.status.as_str() {
            "killed" | "stop" => {
                eprintln!(
                    "[scarab {}] status={} · graceful shutdown",
                    cfg.service_id, strategy.status
                );
                if let Some((child, _, _)) = trainer_child.lock().await.take() {
                    let _ = graceful_kill(child, Duration::from_millis(cfg.grace_ms));
                }
                return Ok(());
            }
            "draining" => {
                eprintln!("[scarab {}] status=draining · wait current trainer, then exit", cfg.service_id);
                if let Some((mut child, _, _)) = trainer_child.lock().await.take() {
                    let _ = child.wait();
                }
                return Ok(());
            }
            "paused" => {
                if trainer_child.lock().await.is_some() {
                    eprintln!("[scarab {}] status=paused · stopping current trainer", cfg.service_id);
                    if let Some((child, _, _)) = trainer_child.lock().await.take() {
                        let _ = graceful_kill(child, Duration::from_millis(cfg.grace_ms));
                    }
                }
                continue;
            }
            "active" => {}
            other => {
                eprintln!("[scarab {}] unknown status={other} · treat as paused", cfg.service_id);
                continue;
            }
        }

        // 4. Generation drift → graceful restart
        if strategy.generation > l_gen {
            eprintln!(
                "[scarab {}] gen {} → {} · {} {} h={} seed={} lr={} steps={}",
                cfg.service_id,
                l_gen,
                strategy.generation,
                strategy.optimizer,
                strategy.format,
                strategy.hidden,
                strategy.seed,
                strategy.lr,
                strategy.steps
            );
            if let Some((child, _, _)) = trainer_child.lock().await.take() {
                let _ = graceful_kill(child, Duration::from_millis(cfg.grace_ms));
                eprintln!("[scarab {}] previous trainer stopped gracefully", cfg.service_id);
            }
            if !cfg.dry_run {
                // L-SS6 (#157, closes #83): build canon identity once and propagate
                // to trainer subprocess so logs + ssot.bpb_samples attribution carry
                // the canon name. Uses canon_name(&strategy, "RAILWAY") to match the
                // value already written on DONE in scarab_result (see lib.rs:134).
                let canon = canon_name(&strategy, "RAILWAY");
                let mut cmd = Command::new(&cfg.trainer_bin);
                cmd.env("TRIOS_FORMAT_TYPE", &strategy.format)
                    .env("TRIOS_DISABLE_NEON", "1")
                    .env("CANON_NAME", &canon)
                    .env("TRIOS_CANON_NAME", &canon)
                    .env("WORKER_SEED", strategy.seed.to_string())
                    .env_remove("DATABASE_URL")
                    .env_remove("TRIOS_DATABASE_URL")
                    .args([
                        "--seed", &strategy.seed.to_string(),
                        "--steps", &strategy.steps.to_string(),
                        "--lr", &strategy.lr.to_string(),
                        "--hidden", &strategy.hidden.to_string(),
                        "--attn-layers", "1",
                        "--optimizer", &strategy.optimizer,
                        "--train-data", &cfg.train_data,
                        "--val-data", &cfg.val_data,
                        "--eval-every", "500",
                    ])
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped());
                let mut child = cmd.spawn().context("spawn trainer")?;
                let child_pid = child.id();
                let stdout = child.stdout.take().context("stdout")?;
                eprintln!("[scarab {}] spawned pid={child_pid}", cfg.service_id);
                let t_start = Instant::now();

                let svc = cfg.service_id.clone();
                let bpb_arc = current_bpb.clone();
                let step_arc = current_step.clone();
                let strat_done = strategy.clone();
                let done_tx_clone = done_tx.clone();
                std::thread::spawn(move || {
                    let reader = BufReader::new(stdout);
                    let mut last_bpb: Option<f64> = None;
                    for line in reader.lines().flatten() {
                        if let Some(bpos) = line.find("bpb=") {
                            let rest = &line[bpos + 4..];
                            let bpb_str: String = rest
                                .chars()
                                .take_while(|c| c.is_ascii_digit() || *c == '.')
                                .collect();
                            if let Ok(v) = bpb_str.parse::<f64>() {
                                last_bpb = Some(v);
                                if let Ok(mut g) = bpb_arc.try_lock() {
                                    *g = Some(v);
                                }
                            }
                        }
                        if let Some(spos) = line.find("step=") {
                            let rest = &line[spos + 5..];
                            let s_str: String =
                                rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                            if let Ok(v) = s_str.parse::<i32>() {
                                if let Ok(mut g) = step_arc.try_lock() {
                                    *g = v;
                                }
                            }
                        }
                        if line.starts_with("DONE:") {
                            let wall_s = t_start.elapsed().as_secs() as i32;
                            eprintln!(
                                "[scarab {svc}] DONE · gen={} bpb={:?} wall={}s",
                                strat_done.generation, last_bpb, wall_s
                            );
                            let _ = done_tx_clone.send(DoneEvent {
                                strategy: strat_done.clone(),
                                final_bpb: last_bpb,
                                wall_s,
                            });
                        }
                    }
                });

                *trainer_child.lock().await = Some((child, t_start, strategy.clone()));
            }
            *local_gen.lock().await = strategy.generation;
            *current_step.lock().await = 0;
            *current_bpb.lock().await = None;
        }

        // 4b. Drain DONE events → scarab_result
        while let Ok(ev) = done_rx.try_recv() {
            let canon = canon_name(&ev.strategy, "RAILWAY");
            let fp = ev.strategy.fingerprint();
            match client
                .execute(
                    "INSERT INTO ssot.scarab_result \
                       (service_id, canon_name, optimizer, format, hidden, lr, seed, steps, final_bpb, wall_s, generation) \
                     VALUES ($1,$2,$3,$4,$5,$6::numeric,$7,$8,$9,$10,$11)",
                    &[
                        &cfg.service_id, &canon, &ev.strategy.optimizer, &ev.strategy.format,
                        &ev.strategy.hidden, &ev.strategy.lr.to_string(),
                        &ev.strategy.seed, &ev.strategy.steps,
                        &ev.final_bpb, &ev.wall_s, &ev.strategy.generation,
                    ],
                )
                .await
            {
                Ok(_) => eprintln!(
                    "[scarab {}] scarab_result written gen={} bpb={:?} fp={}",
                    cfg.service_id, ev.strategy.generation, ev.final_bpb, &fp[..16]
                ),
                Err(e) => eprintln!("[scarab {}] scarab_result INSERT err: {e}", cfg.service_id),
            }
        }

        // 5. trainer liveness
        let mut finished = false;
        {
            let mut guard = trainer_child.lock().await;
            if let Some((child, _, _)) = guard.as_mut() {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        eprintln!("[scarab {}] trainer exit status={status}", cfg.service_id);
                        *guard = None;
                        finished = true;
                    }
                    Ok(None) => {}
                    Err(e) => eprintln!("[scarab {}] try_wait err: {e}", cfg.service_id),
                }
            }
        }

        // 6. AUTO_REPLAY
        if cfg.auto_replay && finished && strategy.status == "active" {
            eprintln!("[scarab {}] AUTO_REPLAY · bumping for next gen", cfg.service_id);
            let _ = client
                .execute("SELECT ssot.bump_strategy($1)", &[&cfg.service_id])
                .await;
        }
    }
}
