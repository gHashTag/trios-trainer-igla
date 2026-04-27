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

/// Lazy-initialised Neon client. `None` if `TRIOS_NEON_DSN` is unset or the
/// connection failed. Cached across calls.
fn client() -> Option<&'static Client> {
    static CLIENT: OnceLock<Option<Client>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            let dsn = std::env::var("TRIOS_NEON_DSN").ok()?;
            let connect = rt().block_on(async {
                let connector = tokio_postgres::connect(&dsn, NoTls).await;
                match connector {
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
            });
            connect
        })
        .as_ref()
}

/// Run an SQL statement with up to `max_attempts` retries on transient errors.
fn execute(stmt: &str, params: &[&(dyn tokio_postgres::types::ToSql + Sync)]) {
    let Some(c) = client() else {
        // No DSN or connect failed: log only, like the legacy eprintln path.
        eprintln!(
            "[neon_writer] DSN unset or unreachable; skipping: {}",
            short(stmt)
        );
        return;
    };
    let max_attempts = 3u8;
    for attempt in 1..=max_attempts {
        let res = rt().block_on(c.execute(stmt, params));
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
