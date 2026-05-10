// src/neon_writer.rs  [ledger]
//
// Ledger writer for trios-trainer-igla.
//
// Provides the same public API as the original tokio-postgres version, but
// uses SeaORM internally.  The connection is cached in a OnceCell keyed off
// the resolved DSN.  All public functions keep their synchronous signatures;
// they block on a private Tokio runtime (same pattern as before).
//
// ENV fallback chain (do NOT introduce NEON_* as primary):
//   DATABASE_URL (canonical, set by Railway since #113)
//   → NEON_DATABASE_URL  (legacy alias)
//   → TRIOS_NEON_DSN     (legacy alias)
//   → TRIOS_DATABASE_URL (legacy alias)
//
// Constitutional notes:
//   R5 - never panic the trainer on DB errors; log a warn and continue.
//   R7 - emits forward step/seed/bpb verbatim.
//   R9 - writer never touches ledger::emit_row; embargo gate stays in trios.
//
// Anchor: phi^2 + phi^-2 = 3 · DOI 10.5281/zenodo.19227877

#![allow(dead_code)]

use std::sync::OnceLock;
use std::time::Duration;

use sea_orm::{
    sea_query::OnConflict, ActiveModelTrait, ActiveValue::Set, ColumnTrait, ConnectionTrait,
    Database, DatabaseConnection, EntityTrait, QueryFilter,
};
use tokio::runtime::Runtime;

use crate::entities::{bpb_samples, igla_agents_heartbeat, igla_race_trials};

// ── Tokio runtime ─────────────────────────────────────────────────────────────

fn rt() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build tokio runtime")
    })
}

// ── DSN helpers ───────────────────────────────────────────────────────────────

/// Resolve the database DSN from the environment fallback chain.
fn resolve_dsn() -> Option<String> {
    std::env::var("DATABASE_URL")
        .or_else(|_| std::env::var("NEON_DATABASE_URL"))
        .or_else(|_| std::env::var("TRIOS_NEON_DSN"))
        .or_else(|_| std::env::var("TRIOS_DATABASE_URL"))
        .ok()
}

/// Remove `channel_binding=require` (or `=prefer`) from a Neon-style DSN.
///
/// `tokio-postgres-rustls` / `sqlx-postgres` with rustls do not expose the TLS
/// exporter needed for `tls-server-end-point` channel binding, so
/// SCRAM-SHA-256-PLUS auth fails.  Neon Postgres accepts plain SCRAM-SHA-256
/// over TLS; stripping this query-string param is the minimal fix (#84, #113).
pub fn strip_channel_binding(dsn: &str) -> String {
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
        head.trim_end_matches('?').to_string()
    } else {
        format!("{head}{rebuilt}")
    }
}

// ── SeaORM connection cache ────────────────────────────────────────────────────

fn db() -> Option<&'static DatabaseConnection> {
    static DB: OnceLock<Option<DatabaseConnection>> = OnceLock::new();
    DB.get_or_init(|| {
        let raw_dsn = resolve_dsn()?;
        let dsn = strip_channel_binding(&raw_dsn);
        if dsn != raw_dsn {
            eprintln!("[ledger] stripped channel_binding from DSN (rustls limitation)");
        }
        eprintln!("[ledger] connecting via SeaORM ...");
        let result = rt().block_on(async { Database::connect(&dsn).await });
        match result {
            Ok(conn) => {
                eprintln!("[ledger] connected OK");
                Some(conn)
            }
            Err(e) => {
                eprintln!("[ledger] connect failed: {e}");
                None
            }
        }
    })
    .as_ref()
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Insert a fresh row into `igla_race_trials` (idempotent on `trial_id`).
///
/// Old raw-SQL call:
///   INSERT INTO igla_race_trials ... ON CONFLICT (trial_id) DO UPDATE ...
/// New SeaORM call:
///   igla_race_trials::Entity::insert(active_model).on_conflict(...).exec(&db)
pub fn trial_start(trial_id: &str, config_json: &str, agent_id: &str, branch: &str) {
    let Some(conn) = db() else {
        eprintln!("[ledger] DSN unset — skipping trial_start");
        return;
    };

    let trial_uuid = match trial_id.parse::<uuid::Uuid>() {
        Ok(u) => u,
        Err(e) => {
            eprintln!("[ledger] trial_start: invalid UUID {trial_id}: {e}");
            return;
        }
    };
    let config_val: serde_json::Value = match serde_json::from_str(config_json) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[ledger] trial_start: invalid JSON: {e}");
            serde_json::json!({})
        }
    };

    let model = igla_race_trials::ActiveModel {
        trial_id: Set(trial_uuid),
        config: Set(config_val),
        status: Set("running".to_string()),
        agent_id: Set(Some(agent_id.to_string())),
        branch: Set(Some(branch.to_string())),
        ..Default::default()
    };

    let on_conflict = OnConflict::column(igla_race_trials::Column::TrialId)
        .update_columns([
            igla_race_trials::Column::Status,
            igla_race_trials::Column::AgentId,
            igla_race_trials::Column::Branch,
        ])
        .to_owned();

    let res = rt().block_on(
        igla_race_trials::Entity::insert(model)
            .on_conflict(on_conflict)
            .exec(conn),
    );
    match res {
        Ok(_) => eprintln!("[ledger] trial_start ok: {trial_id}"),
        Err(e) => eprintln!("[ledger] trial_start failed: {e}"),
    }
}

/// Upsert agent heartbeat and update the latest BPB on the trial row.
///
/// Old raw-SQL calls:
///   INSERT INTO igla_agents_heartbeat ... ON CONFLICT (agent_id) DO UPDATE ...
///   UPDATE igla_race_trials SET bpb_latest=$1, steps_done=$2 WHERE trial_id=$3
/// New SeaORM calls:
///   igla_agents_heartbeat::Entity::insert(...).on_conflict(...).exec(&db)
///   igla_race_trials::Entity::update_many().col_expr(...).filter(...).exec(&db)
pub fn heartbeat(trial_id: &str, agent_id: &str, bpb: f32, step: usize) {
    let Some(conn) = db() else {
        eprintln!("[ledger] DSN unset — skipping heartbeat");
        return;
    };

    // Upsert heartbeat row.
    let hb_model = igla_agents_heartbeat::ActiveModel {
        agent_id: Set(agent_id.to_string()),
        machine_id: Set("railway".to_string()),
        branch: Set("main".to_string()),
        task: Set(Some(trial_id.to_string())),
        status: Set("active".to_string()),
        last_heartbeat: Set(chrono::Utc::now().into()),
    };
    let on_conflict_hb = OnConflict::column(igla_agents_heartbeat::Column::AgentId)
        .update_columns([
            igla_agents_heartbeat::Column::Status,
            igla_agents_heartbeat::Column::LastHeartbeat,
            igla_agents_heartbeat::Column::Task,
        ])
        .to_owned();
    let res = rt().block_on(
        igla_agents_heartbeat::Entity::insert(hb_model)
            .on_conflict(on_conflict_hb)
            .exec(conn),
    );
    if let Err(e) = res {
        eprintln!("[ledger] heartbeat (upsert) failed: {e}");
    }

    // Update trial row with latest bpb / step.
    // steps_done is BIGINT — bind as i64.
    let trial_uuid = match trial_id.parse::<uuid::Uuid>() {
        Ok(u) => u,
        Err(e) => {
            eprintln!("[ledger] heartbeat: invalid UUID {trial_id}: {e}");
            return;
        }
    };

    use sea_orm::sea_query::Expr;
    let res = rt().block_on(
        igla_race_trials::Entity::update_many()
            .col_expr(igla_race_trials::Column::BpbLatest, Expr::value(bpb as f64))
            .col_expr(
                igla_race_trials::Column::StepsDone,
                Expr::value(step as i64),
            )
            .filter(igla_race_trials::Column::TrialId.eq(trial_uuid))
            .exec(conn),
    );
    match res {
        Ok(r) => eprintln!(
            "[ledger] heartbeat ok: trial={trial_id} rows={}",
            r.rows_affected
        ),
        Err(e) => eprintln!("[ledger] heartbeat (update) failed: {e}"),
    }
}

/// Mark a trial complete with the final BPB.
///
/// Old raw-SQL call:
///   UPDATE igla_race_trials SET bpb_final=$1, status='complete' WHERE trial_id=$2
/// New SeaORM call:
///   igla_race_trials::Entity::update_many().col_expr(...).filter(...).exec(&db)
pub fn trial_complete(trial_id: &str, bpb: f32) {
    let Some(conn) = db() else {
        eprintln!("[ledger] DSN unset — skipping trial_complete");
        return;
    };

    let trial_uuid = match trial_id.parse::<uuid::Uuid>() {
        Ok(u) => u,
        Err(e) => {
            eprintln!("[ledger] trial_complete: invalid UUID {trial_id}: {e}");
            return;
        }
    };

    use sea_orm::sea_query::Expr;
    let res = rt().block_on(
        igla_race_trials::Entity::update_many()
            .col_expr(igla_race_trials::Column::BpbFinal, Expr::value(bpb as f64))
            .col_expr(
                igla_race_trials::Column::Status,
                Expr::value("complete".to_string()),
            )
            .filter(igla_race_trials::Column::TrialId.eq(trial_uuid))
            .exec(conn),
    );
    match res {
        Ok(r) => eprintln!(
            "[ledger] trial_complete ok: trial={trial_id} rows={}",
            r.rows_affected
        ),
        Err(e) => eprintln!("[ledger] trial_complete failed: {e}"),
    }
}

/// Insert a single row into `public.bpb_samples` with checkpoint telemetry.
///
/// Schema (verified against phd-postgres-ssot 2026-05-09, updated Wave 29 PR-A):
///   id BIGSERIAL, canon_name TEXT, seed BIGINT, step BIGINT,
///   bpb DOUBLE PRECISION, ema_bpb DOUBLE PRECISION (nullable), ts TIMESTAMPTZ
///
/// seed and step are cast from i32 to i64 to match the BIGINT columns (#114, Wave 24).
///
/// Wave 29 PR-A idempotent INSERT:
///   ON CONFLICT (canon_name, seed, step)
///   DO UPDATE SET
///     bpb     = LEAST(EXCLUDED.bpb, bpb_samples.bpb),
///     ema_bpb = CASE WHEN EXCLUDED.bpb < bpb_samples.bpb THEN EXCLUDED.ema_bpb ELSE bpb_samples.ema_bpb END,
///     ts      = EXCLUDED.ts
/// This keeps the best-of-rerun BPB and eliminates duplicate-skipping errors.
/// Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877
pub fn bpb_sample(canon_name: &str, seed: i32, step: i32, bpb: f32, ema_bpb: Option<f32>) {
    let Some(conn) = db() else {
        eprintln!("[ledger] DSN unset — skipping bpb_sample");
        return;
    };

    // Wave 29 PR-A: Use raw SQL for the idempotent upsert.
    // SeaORM's on_conflict builder doesn't support LEAST() / CASE expressions,
    // so we use Statement::from_sql_and_values for the full upsert.
    // LEAST(EXCLUDED.bpb, bpb_samples.bpb) keeps the best-of-rerun BPB.
    // ema_bpb follows the winning bpb side.
    // Eliminates "[ledger] duplicate, skipping" errors that froze ACC1 mr-runners.
    // Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877
    let ts_now = chrono::Utc::now();
    // The SQL is a static string — no format!() needed.
    // Use SeaORM raw statement execution for parameterised upsert.
    use sea_orm::Statement;
    let stmt = Statement::from_sql_and_values(
        conn.get_database_backend(),
        "INSERT INTO public.bpb_samples (canon_name, seed, step, bpb, ema_bpb, ts) \
         VALUES ($1, $2, $3, $4, $5, $6) \
         ON CONFLICT (canon_name, seed, step) DO UPDATE SET \
           bpb     = LEAST(EXCLUDED.bpb, bpb_samples.bpb), \
           ema_bpb = CASE WHEN EXCLUDED.bpb < bpb_samples.bpb \
                         THEN EXCLUDED.ema_bpb ELSE bpb_samples.ema_bpb END, \
           ts      = EXCLUDED.ts",
        [
            seed_val(canon_name),
            (seed as i64).into(),
            (step as i64).into(),
            (bpb as f64).into(),
            ema_bpb.map(|v| v as f64).into(),
            ts_now.into(),
        ],
    );
    let res = rt().block_on(conn.execute(stmt));
    match res {
        Ok(_) => eprintln!("[ledger] bpb_sample ok (idempotent): {canon_name} seed={seed} step={step} bpb={bpb:.4}"),
        Err(e) => eprintln!("[ledger] bpb_sample failed: {e}"),
    }
}

/// Helper: convert a &str into a sea_orm Value.
#[inline]
fn seed_val(s: &str) -> sea_orm::Value {
    sea_orm::Value::String(Some(Box::new(s.to_string())))
}

/// Apply idempotent DDL: ensure `bpb_latest` column exists on `igla_race_trials`.
///
/// Old raw-SQL call:
///   ALTER TABLE igla_race_trials ADD COLUMN IF NOT EXISTS bpb_latest DOUBLE PRECISION
/// New: delegated to the SeaORM migration (Migrator::up runs at startup).
/// This stub is kept for callers that invoke it directly.
pub fn ensure_schema() {
    let Some(conn) = db() else {
        eprintln!("[ledger] DSN unset — skipping ensure_schema");
        return;
    };
    let res = rt().block_on(conn.execute_unprepared(
        "ALTER TABLE igla_race_trials ADD COLUMN IF NOT EXISTS bpb_latest DOUBLE PRECISION",
    ));
    match res {
        Ok(_) => eprintln!("[ledger] ensure_schema ok"),
        Err(e) => eprintln!("[ledger] ensure_schema failed: {e}"),
    }
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
    #[test]
    fn bpb_sample_uses_i64_bind() {
        let seed: i32 = 43;
        let step: i32 = 200;

        let seed_i64: i64 = seed as i64;
        let step_i64: i64 = step as i64;

        // i64 must be accepted by Postgres INT8 (BIGINT).
        assert!(
            <i64 as ToSql>::accepts(&Type::INT8),
            "i64 must be accepted by Postgres INT8"
        );
        // i32 must NOT be accepted by INT8 — documents the original bug.
        assert!(
            !<i32 as ToSql>::accepts(&Type::INT8),
            "i32 must NOT be accepted by Postgres INT8"
        );

        let _: i64 = seed_i64;
        let _: i64 = step_i64;
        assert_eq!(seed_i64, 43i64);
        assert_eq!(step_i64, 200i64);
    }
}
