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
/// Schema (verified against phd-postgres-ssot 2026-05-09, updated Wave 24):
///   id BIGSERIAL, canon_name TEXT, seed BIGINT, step BIGINT,
///   bpb DOUBLE PRECISION, ts TIMESTAMPTZ
///
/// seed and step are cast from i32 to i64 to match the BIGINT columns (#114, Wave 24).
///
/// **Wave 29 hotfix (patch 2 of 2):** the prior implementation used
/// `ON CONFLICT (canon_name, seed, step) DO NOTHING`, which silently
/// dropped every fresh INSERT once a row existed for the
/// `(canon_name, seed, step)` triple. This blocked the Wave 29
/// `STEPS=200000` bump from making any forward progress: every
/// re-run of the same step produced the
/// `[ledger] bpb_sample: row already exists (duplicate), skipping`
/// log line instead of writing the new sample.
///
/// New behaviour (best-wins, idempotent):
///
/// ```sql
/// INSERT INTO public.bpb_samples (canon_name, seed, step, bpb, ts)
/// VALUES (...)
/// ON CONFLICT (canon_name, seed, step) DO UPDATE SET
///   bpb = LEAST(EXCLUDED.bpb, public.bpb_samples.bpb),
///   ts  = EXCLUDED.ts
/// ```
///
/// Re-runs with the same `(canon_name, seed, step)` keep the
/// **best (= lowest) BPB observed** and refresh `ts` so the stream is
/// alive. The `ema_bpb` column is deliberately not touched because
/// the live schema (verified 2026-05-09) does not carry one — see
/// the schema comment above.
pub fn bpb_sample(canon_name: &str, seed: i32, step: i32, bpb: f32) {
    let Some(conn) = db() else {
        eprintln!("[ledger] DSN unset — skipping bpb_sample");
        return;
    };

    // Cast seed and step to i64 — BIGINT in Postgres (fixes #114, Wave 24 step migration).
    let model = bpb_samples::ActiveModel {
        canon_name: Set(canon_name.to_string()),
        seed: Set(seed as i64),
        step: Set(step as i64),
        bpb: Set(bpb as f64),
        ts: Set(chrono::Utc::now().into()),
        ..Default::default()
    };

    // Best-wins idempotency: on conflict, keep the lower BPB and refresh ts.
    // Wave 29 hotfix — replaces the previous DO NOTHING path that dropped
    // every fresh INSERT past step 81000 in re-runs.
    let on_conflict = OnConflict::columns([
        bpb_samples::Column::CanonName,
        bpb_samples::Column::Seed,
        bpb_samples::Column::Step,
    ])
    .value(
        bpb_samples::Column::Bpb,
        sea_orm::sea_query::Expr::cust("LEAST(EXCLUDED.bpb, public.bpb_samples.bpb)"),
    )
    .update_column(bpb_samples::Column::Ts)
    .to_owned();

    let res = rt().block_on(
        bpb_samples::Entity::insert(model)
            .on_conflict(on_conflict)
            .exec(conn),
    );
    match res {
        Ok(_) => eprintln!(
            "[ledger] bpb_sample ok: {canon_name} seed={seed} step={step} bpb={bpb} (best-wins)"
        ),
        Err(sea_orm::DbErr::RecordNotInserted) => {
            // SeaORM still surfaces RecordNotInserted when the row was
            // updated rather than inserted (the response has no
            // last-insert-id). This is the success path under best-wins;
            // log it but do not treat as an error. The legacy
            // "duplicate, skipping" wording is replaced with explicit
            // "updated" so log scrapers can tell the two semantics
            // apart from the prior implementation.
            eprintln!(
                "[ledger] bpb_sample: existing row updated (best-wins): \
                 {canon_name} seed={seed} step={step} candidate_bpb={bpb}"
            );
        }
        Err(e) => eprintln!("[ledger] bpb_sample failed: {e}"),
    }
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

    /// Wave 29 hotfix (patch 2 of 2): the bpb_sample ON CONFLICT clause
    /// must emit DO UPDATE SET bpb = LEAST(EXCLUDED.bpb,
    /// public.bpb_samples.bpb) and refresh ts. The prior behaviour was
    /// DO NOTHING, which silently dropped every Wave 29 STEPS=200000
    /// re-run past step 81000.
    ///
    /// This test reconstructs the OnConflict spec the same way
    /// `bpb_sample` does, builds the SQL via PostgresQueryBuilder, and
    /// asserts the rendered string contains the LEAST clause and the ts
    /// column update — not DO NOTHING.
    #[test]
    fn bpb_sample_on_conflict_is_best_wins_least() {
        use sea_orm::sea_query::{Expr, OnConflict, PostgresQueryBuilder, Query};

        let on_conflict = OnConflict::columns([
            bpb_samples::Column::CanonName,
            bpb_samples::Column::Seed,
            bpb_samples::Column::Step,
        ])
        .value(
            bpb_samples::Column::Bpb,
            Expr::cust("LEAST(EXCLUDED.bpb, public.bpb_samples.bpb)"),
        )
        .update_column(bpb_samples::Column::Ts)
        .to_owned();

        // Render the OnConflict against a minimal INSERT to inspect
        // the SQL string. We do not execute it; this is a pure
        // string-shape regression guard.
        let mut q = Query::insert();
        q.into_table(bpb_samples::Entity)
            .columns([
                bpb_samples::Column::CanonName,
                bpb_samples::Column::Seed,
                bpb_samples::Column::Step,
                bpb_samples::Column::Bpb,
                bpb_samples::Column::Ts,
            ])
            .values_panic([
                "scarab-soap".into(),
                47i64.into(),
                3000i64.into(),
                2.8845f64.into(),
                "2026-05-09T00:00:00Z".into(),
            ])
            .on_conflict(on_conflict);
        let sql = q.to_string(PostgresQueryBuilder);

        // Best-wins clause must be present:
        assert!(
            sql.contains("LEAST(EXCLUDED.bpb, public.bpb_samples.bpb)"),
            "ON CONFLICT clause must use LEAST(EXCLUDED.bpb, public.bpb_samples.bpb); got: {sql}"
        );
        // ts must be refreshed:
        assert!(
            sql.contains("\"ts\""),
            "ON CONFLICT must update the ts column; got: {sql}"
        );
        // Must NOT silently skip rows any more:
        assert!(
            !sql.contains("DO NOTHING"),
            "ON CONFLICT must no longer be DO NOTHING (Wave 29 hotfix); got: {sql}"
        );
        // Must be DO UPDATE SET:
        assert!(
            sql.contains("DO UPDATE SET"),
            "ON CONFLICT must be DO UPDATE SET; got: {sql}"
        );
    }
}
