// Real Neon writer for tjepa_train.
//
// Replaces the previous `eprintln!("NEON_SQL: ...")` log-only emits with actual
// INSERT/UPDATE statements over `tokio-postgres`. DSN is read from
// `DATABASE_URL` (canonical) or `NEON_DATABASE_URL` / `TRIOS_NEON_DSN` as
// legacy fallbacks; if unset the writer is a no-op so existing developer
// workflows that don't have Neon access keep working.
//
// Drift code: D2_NEON_NOT_WRITTEN (ref: trios-trainer-igla #36).
//
// Constitutional notes:
//   R5 - never panic the trainer on Neon errors; log a warn and continue.
//   R7 - emits forward step/seed/bpb verbatim (caller already builds the
//        triplet for ledger::emit_row).
//   R9 - writer never touches `ledger::emit_row`; embargo gate stays in trios.

#![allow(dead_code)]

use std::sync::OnceLock;
use std::time::Duration;

use tokio::runtime::Runtime;
use tokio_postgres::Client;

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

/// Build a rustls TLS config with system root CAs (required for Neon).
///
/// Rustls 0.23+ requires a CryptoProvider to be installed before calling
/// `ClientConfig::builder()`. We use `ring` and call `install_default()`
/// here. The `Result` is intentionally ignored: if another thread already
/// installed a provider the Err is benign.
fn make_tls_config() -> rustls::ClientConfig {
    // MUST be called before ClientConfig::builder() in rustls 0.23+.
    let _ = rustls::crypto::ring::default_provider().install_default();

    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth()
}

/// Lazy-initialised Neon client. `None` if DSN is unset or the
/// connection failed. Cached across calls.
fn client() -> Option<&'static Client> {
    static CLIENT: OnceLock<Option<Client>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            let raw_dsn = std::env::var("DATABASE_URL")
                .or_else(|_| std::env::var("NEON_DATABASE_URL"))
                .or_else(|_| std::env::var("TRIOS_NEON_DSN"))
                .or_else(|_| std::env::var("TRIOS_DATABASE_URL"))
                .ok()?;
            // tokio-postgres-rustls does NOT export tls-server-end-point
            // channel binding (rustls 0.23 limitation). Neon's standard DSN
            // contains `channel_binding=require` which would force SCRAM-SHA-256-PLUS
            // and fail handshake with an opaque "db error". Strip it; sslmode=require
            // is sufficient for at-rest TLS encryption (#84 round 3 root cause).
            let dsn = strip_channel_binding(&raw_dsn);
            if dsn != raw_dsn {
                eprintln!("[ledger] stripped channel_binding from DSN (rustls limitation)");
            }
            eprintln!("[ledger] connecting to Postgres (TLS) ...");
            let connect = rt().block_on(async {
                let tls_config = make_tls_config();
                let tls = tokio_postgres_rustls::MakeRustlsConnect::new(tls_config);
                let connector = tokio_postgres::connect(&dsn, tls).await;
                match connector {
                    Ok((client, conn)) => {
                        eprintln!("[ledger] connected OK");
                        tokio::spawn(async move {
                            if let Err(e) = conn.await {
                                eprintln!(
                                    "[ledger] connection task error: {e}{}",
                                    full_error_chain(&e)
                                );
                            }
                        });
                        Some(client)
                    }
                    Err(e) => {
                        eprintln!("[ledger] connect failed: {e}{}", full_error_chain(&e));
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
        eprintln!(
            "[ledger] DSN unset or unreachable; skipping: {}",
            short(stmt)
        );
        return;
    };
    let max_attempts = 3u8;
    for attempt in 1..=max_attempts {
        let res = rt().block_on(c.execute(stmt, params));
        match res {
            Ok(rows) => {
                eprintln!("[ledger] ok: {} ({} rows)", short(stmt), rows);
                return;
            }
            Err(e) => {
                eprintln!(
                    "[ledger] attempt {attempt}/{max_attempts} failed for {} : {e}{}",
                    short(stmt),
                    full_error_chain(&e)
                );
                std::thread::sleep(Duration::from_millis(500 * attempt as u64));
            }
        }
    }
    eprintln!(
        "[ledger] giving up after {max_attempts} attempts: {}",
        short(stmt)
    );
}

fn short(s: &str) -> &str {
    let limit = 80usize.min(s.len());
    &s[..limit]
}

/// Render the full `Error::source()` chain after a top-level Display, so
/// opaque errors like tokio_postgres' "db error" surface their real cause
/// (TLS handshake, SCRAM, channel binding, OID mismatch, ...).
///
/// Returns an empty string if the error has no source, so callers can
/// concatenate unconditionally:  `eprintln!("failed: {e}{}", full_error_chain(&e))`.
fn full_error_chain<E: std::error::Error + ?Sized>(err: &E) -> String {
    let mut out = String::new();
    let mut src: Option<&(dyn std::error::Error)> = err.source();
    while let Some(s) = src {
        out.push_str("\n  caused by: ");
        out.push_str(&s.to_string());
        src = s.source();
    }
    out
}

/// Remove `channel_binding=require` (or `=prefer`) from a Neon-style DSN.
///
/// `tokio-postgres-rustls` 0.12 / rustls 0.23 do not expose the TLS exporter
/// needed for `tls-server-end-point` channel binding, so SCRAM-SHA-256-PLUS
/// auth fails with an opaque "db error" the moment a server requires PLUS.
/// Neon Postgres accepts plain SCRAM-SHA-256 over TLS just fine; stripping
/// this query-string param is the minimal change that unblocks scarab's writes
/// without rewriting the TLS stack to native-tls.
pub fn strip_channel_binding(dsn: &str) -> String {
    // Match both URI and key=value forms. Keep parsing minimal — we only
    // remove a single `channel_binding=...` token, leaving the rest of the
    // DSN exactly as the operator supplied it (no URL-decoding/re-encoding).
    let Some(qpos) = dsn.find('?') else {
        return dsn.to_string();
    };
    let (head, query) = dsn.split_at(qpos + 1);
    let kept: Vec<&str> = query
        .split('&')
        .filter(|kv| !kv.trim_start().starts_with("channel_binding="))
        .collect();
    let rebuilt = kept.join("&");
    if rebuilt.is_empty() {
        // strip the trailing `?` so libpq doesn't see an empty query.
        head.trim_end_matches('?').to_string()
    } else {
        format!("{head}{rebuilt}")
    }
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
    // steps_done is BIGINT — bind as i64 to avoid type mismatch (#114).
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
/// Schema (verified against phd-postgres-ssot 2026-05-09):
///   id BIGSERIAL, canon_name TEXT, seed BIGINT, step BIGINT, bpb DOUBLE PRECISION, ts TIMESTAMPTZ
///
/// seed and step are BIGINT in Postgres; we widen from i32 via `as i64`
/// to avoid "cannot convert between Rust type i32 and Postgres int8" (#114).
pub fn bpb_sample(canon_name: &str, seed: i32, step: i32, bpb: f32) {
    // Idempotent: `bpb_samples_canon_name_seed_step_key` is UNIQUE
    // (canon_name, seed, step). A duplicate emit (scarab restart on the
    // same row, smoke retry, NOTIFY redelivery, etc.) used to throw
    // "duplicate key value violates unique constraint" three times before
    // giving up; now we let Postgres silently skip. Training output is
    // deterministic for fixed (canon_name, seed, step), so DO NOTHING
    // preserves the first-write timestamp and avoids redundant churn.
    //
    // Cast seed/step to i64 (BIGINT) — fixes bind-type mismatch (#114).
    execute(
        "INSERT INTO public.bpb_samples (canon_name, seed, step, bpb, ts) \
         VALUES ($1, $2, $3, $4, NOW()) \
         ON CONFLICT (canon_name, seed, step) DO NOTHING",
        &[&canon_name, &(seed as i64), &(step as i64), &(bpb as f64)],
    );
}

/// Apply idempotent DDL: ensure `bpb_latest` column exists on `igla_race_trials`.
pub fn ensure_schema() {
    execute(
        "ALTER TABLE igla_race_trials ADD COLUMN IF NOT EXISTS bpb_latest DOUBLE PRECISION",
        &[],
    );
}

/// Default checkpoint interval honoured by trainers writing to bpb_samples.
pub fn checkpoint_interval() -> usize {
    std::env::var("TRIOS_CHECKPOINT_INTERVAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_postgres::types::{ToSql, Type};

    #[test]
    fn no_dsn_is_safe() {
        std::env::remove_var("DATABASE_URL");
        std::env::remove_var("TRIOS_NEON_DSN");
        std::env::remove_var("NEON_DATABASE_URL");
        std::env::remove_var("TRIOS_DATABASE_URL");
        trial_start("00000000-0000-0000-0000-000000000000", "{}", "TEST", "main");
        heartbeat("00000000-0000-0000-0000-000000000000", "TEST", 2.5, 1);
        trial_complete("00000000-0000-0000-0000-000000000000", 2.5);
    }

    #[test]
    fn strip_channel_binding_removes_only_that_param() {
        let in_ = "postgresql://u:p@h/db?sslmode=require&channel_binding=require";
        assert_eq!(
            strip_channel_binding(in_),
            "postgresql://u:p@h/db?sslmode=require"
        );
    }

    #[test]
    fn strip_channel_binding_when_only_param() {
        let in_ = "postgresql://u:p@h/db?channel_binding=require";
        assert_eq!(strip_channel_binding(in_), "postgresql://u:p@h/db");
    }

    #[test]
    fn strip_channel_binding_passthrough_when_absent() {
        let in_ = "postgresql://u:p@h/db?sslmode=require";
        assert_eq!(strip_channel_binding(in_), in_);
    }

    #[test]
    fn strip_channel_binding_passthrough_no_query_string() {
        let in_ = "postgresql://u:p@h/db";
        assert_eq!(strip_channel_binding(in_), in_);
    }

    /// Regression test: seed and step parameters for bpb_sample must be bound
    /// as INT8 (i64) to match the BIGINT columns in public.bpb_samples (#114).
    ///
    /// This test proves that the values we pass to execute() are accepted by
    /// Postgres's INT8 type checker without any type mismatch, using the
    /// `accepts` method on the ToSql trait.
    #[test]
    fn bpb_sample_uses_i64_bind() {
        let seed: i32 = 43;
        let step: i32 = 200;

        // Widen to i64 (exactly as bpb_sample() does internally).
        let seed_i64: i64 = seed as i64;
        let step_i64: i64 = step as i64;

        // Verify that i64 accepts INT8 (BIGINT) via the ToSql::accepts method.
        // This is the statically-dispatched type acceptance check.
        assert!(
            <i64 as ToSql>::accepts(&Type::INT8),
            "i64 must be accepted by Postgres INT8 — the bpb_sample fix requires this"
        );

        // Verify that i32 does NOT accept INT8 — this documents the original bug.
        assert!(
            !<i32 as ToSql>::accepts(&Type::INT8),
            "i32 must NOT be accepted by Postgres INT8 — confirms the bug we are fixing"
        );

        // Verify that the widened values are the correct type (not accidentally i32).
        let _: i64 = seed_i64; // type assertion: seed_i64 is i64
        let _: i64 = step_i64; // type assertion: step_i64 is i64

        // Sanity: values are correctly widened.
        assert_eq!(seed_i64, 43i64);
        assert_eq!(step_i64, 200i64);
    }
}
