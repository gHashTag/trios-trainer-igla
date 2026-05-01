// Real Neon writer for tjepa_train.
//
// Replaces the previous `eprintln!("NEON_SQL: ...")` log-only emits with actual
// INSERT/UPDATE statements over `tokio-postgres`. DSN is read from
// `TRIOS_NEON_DSN`; if unset the writer is a no-op so existing developer
// workflows that don't have Neon access keep working.
//
// Drift code: D2_NEON_NOT_WRITTEN (ref: trios-trainer-igla #36).
//
// Constitutional notes:
//   R5 — never panic the trainer on Neon errors; log a warn and continue.
//   R7 — emits forward step/seed/bpb verbatim (caller already builds the
//        triplet for ledger::emit_row).
//   R9 — writer never touches `ledger::emit_row`; embargo gate stays in trios.

#![allow(dead_code)]

use std::sync::OnceLock;
use std::time::Duration;

use tokio::runtime::Runtime;
use tokio_postgres::{Client, NoTls};

/// Lazy-initialised tokio runtime shared by all sync wrappers.
fn rt() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build tokio runtime")
    })
}

/// Connect to Neon and return a Client. Returns None if DSN is unset or
/// the connection fails.
fn connect() -> Option<Client> {
    let dsn = std::env::var("TRIOS_NEON_DSN")
        .or_else(|_| std::env::var("NEON_DATABASE_URL"))
        .or_else(|_| std::env::var("DATABASE_URL"))
        .ok()?;
    eprintln!("[neon_writer] connecting to Neon …");
    rt().block_on(async {
        match tokio_postgres::connect(&dsn, NoTls).await {
            Ok((client, conn)) => {
                // Drive the connection task in the background.
                tokio::spawn(async move {
                    if let Err(e) = conn.await {
                        eprintln!("[neon_writer] connection task error: {e}");
                    }
                });
                Some(client)
            }
            Err(e) => {
                eprintln!("[neon_writer] connect failed: {e}");
                None
            }
        }
    })
}

/// Run an SQL statement with up to `max_attempts` retries on transient errors.
/// Automatically reconnects if the cached connection is broken (Neon idle timeout).
fn execute(stmt: &str, params: &[&(dyn tokio_postgres::types::ToSql + Sync)]) {
    use std::sync::Mutex;
    static CLIENT: OnceLock<Mutex<Option<Client>>> = OnceLock::new();
    let cell = CLIENT.get_or_init(|| Mutex::new(connect()));

    // No DSN or initial connect failed.
    {
        let guard = cell.lock().unwrap();
        if guard.is_none() {
            eprintln!(
                "[neon_writer] DSN unset or unreachable; skipping: {}",
                short(stmt)
            );
            return;
        }
    }

    let max_attempts = 3u8;
    for attempt in 1..=max_attempts {
        // Reconnect if the cached client was reset (broken connection).
        {
            let mut guard = cell.lock().unwrap();
            if guard.is_none() {
                eprintln!("[neon_writer] reconnecting (attempt {attempt}) …");
                *guard = connect();
            }
        }

        let res = {
            let guard = cell.lock().unwrap();
            match guard.as_ref() {
                Some(c) => rt().block_on(c.execute(stmt, params)),
                None => break, // connect failed, give up
            }
        };

        match res {
            Ok(rows) => {
                eprintln!("[neon_writer] ok: {} ({} rows)", short(stmt), rows);
                return;
            }
            Err(e) => {
                eprintln!(
                    "[neon_writer] attempt {attempt}/{max_attempts} failed for {} : {e}",
                    short(stmt)
                );
                // Connection is likely broken — reset so next attempt reconnects.
                {
                    let mut guard = cell.lock().unwrap();
                    *guard = None;
                }
                std::thread::sleep(Duration::from_millis(500 * attempt as u64));
            }
        }
    }
    eprintln!(
        "[neon_writer] giving up after {max_attempts} attempts: {}",
        short(stmt)
    );
}

fn short(s: &str) -> &str {
    let limit = 80usize.min(s.len());
    &s[..limit]
}

/// Insert a fresh row into `igla_race_trials` (idempotent on `trial_id`).
pub fn trial_start(trial_id: &str, config_json: &str, agent_id: &str, branch: &str) {
    execute(
        "INSERT INTO igla_race_trials (trial_id, config, status, agent_id, branch) \
         VALUES ($1, $2::jsonb, 'running', $3, $4) \
         ON CONFLICT (trial_id) DO UPDATE SET status='running', agent_id=EXCLUDED.agent_id, branch=EXCLUDED.branch",
        &[&trial_id, &config_json, &agent_id, &branch],
    );
}

/// Upsert agent heartbeat and update the latest BPB on the trial row.
pub fn heartbeat(trial_id: &str, agent_id: &str, bpb: f32, step: usize) {
    execute(
        "INSERT INTO igla_agents_heartbeat (agent_id, machine_id, branch, task, status, last_heartbeat) \
         VALUES ($1, 'railway', 'main', $2, 'active', NOW()) \
         ON CONFLICT (agent_id) DO UPDATE SET status=EXCLUDED.status, last_heartbeat=EXCLUDED.last_heartbeat, task=EXCLUDED.task",
        &[&agent_id, &trial_id],
    );
    execute(
        "UPDATE igla_race_trials SET bpb_latest=$1, steps_done=$2 WHERE trial_id=$3",
        &[&(bpb as f64), &(step as i64), &trial_id],
    );
}

/// Mark a trial complete with the final BPB.
pub fn trial_complete(trial_id: &str, bpb: f32) {
    execute(
        "UPDATE igla_race_trials SET bpb_final=$1, status='complete' WHERE trial_id=$2",
        &[&(bpb as f64), &trial_id],
    );
}

/// Insert a single row into `public.bpb_samples` with checkpoint telemetry.
///
/// This is the canonical write path for IGLA RACE leader/follower telemetry.
/// Caller invokes this every `TRIOS_CHECKPOINT_INTERVAL` steps (default 200).
///
/// Schema (verified against NEON 2026-04-30):
///   id BIGSERIAL, canon_name TEXT, seed INT, step INT, bpb DOUBLE PRECISION, val_bpb_ema DOUBLE PRECISION, ts TIMESTAMPTZ
pub fn bpb_sample(canon_name: &str, seed: i32, step: i32, bpb: f32) {
    execute(
        "INSERT INTO public.bpb_samples (canon_name, seed, step, bpb, ts) \
         VALUES ($1, $2, $3, $4, NOW())",
        &[&canon_name, &seed, &step, &(bpb as f64)],
    );
}

/// Apply idempotent DDL: ensure `bpb_latest` column exists on `igla_race_trials`.
/// Safe to call repeatedly. No-op when the column already exists.
pub fn ensure_schema() {
    execute(
        "ALTER TABLE igla_race_trials ADD COLUMN IF NOT EXISTS bpb_latest DOUBLE PRECISION",
        &[],
    );
}

/// Default checkpoint interval honoured by trainers writing to bpb_samples.
/// Reads `TRIOS_CHECKPOINT_INTERVAL` env var; defaults to 200 (Fibonacci-friendly,
/// short enough to surface mid-training BPB before Gate-2 horizon at 4096 steps).
pub fn checkpoint_interval() -> usize {
    std::env::var("TRIOS_CHECKPOINT_INTERVAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// When DSN is unset, all writers must be silent no-ops (no panic).
    #[test]
    fn no_dsn_is_safe() {
        // Make sure DSN is unset for this test.
        std::env::remove_var("TRIOS_NEON_DSN");
        trial_start("00000000-0000-0000-0000-000000000000", "{}", "TEST", "main");
        heartbeat("00000000-0000-0000-0000-000000000000", "TEST", 2.5, 1);
        trial_complete("00000000-0000-0000-0000-000000000000", 2.5);
    }
}
