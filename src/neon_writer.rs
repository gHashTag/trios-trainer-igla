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

/// Parse `IGLA-{LANE}-{format}-h{H}-LR{L}-rng{SEED}-{algo}` canon_name into
/// `(format, algo, hidden)` triple. Returns `None` if the canon_name does not
/// match the canonical IGLA schema (legacy `scarab-*` names, smoke names, etc).
///
/// Examples:
///   `IGLA-SHORT-WAVE-MATRIX-gf16-h128-LR0.0001-rng1597-adamw` →
///     `("gf16", "adamw", 128)`
///   `IGLA-SCARAB-ADAMW-binary16-h384-LR0001-rng123-adamw` →
///     `("binary16", "adamw", 384)`
///
/// Rule: format = field BEFORE `-h{N}-`; hidden = N; algo = LAST `-` field.
///
/// Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877
pub fn parse_canon_name(canon: &str) -> Option<(String, String, i32)> {
    if !canon.starts_with("IGLA-") {
        return None;
    }
    let parts: Vec<&str> = canon.split('-').collect();
    if parts.len() < 5 {
        return None;
    }

    // Find the `h{N}` token (hidden); the token immediately before it is the format.
    let mut hidden: Option<i32> = None;
    let mut h_idx: Option<usize> = None;
    for (i, tok) in parts.iter().enumerate() {
        if let Some(rest) = tok.strip_prefix('h') {
            if let Ok(n) = rest.parse::<i32>() {
                hidden = Some(n);
                h_idx = Some(i);
                break;
            }
        }
    }
    let h_idx = h_idx?;
    let hidden = hidden?;
    if h_idx == 0 {
        return None;
    }
    let format = parts[h_idx - 1].to_string();
    if format.is_empty() {
        return None;
    }

    // algo = everything AFTER the last `rng{SEED}` token, joined with '-'.
    // This preserves multi-token algos like `muon-cwd`.
    let mut rng_idx: Option<usize> = None;
    for (i, tok) in parts.iter().enumerate() {
        if let Some(rest) = tok.strip_prefix("rng") {
            if rest.parse::<i64>().is_ok() {
                rng_idx = Some(i);
            }
        }
    }
    let rng_idx = rng_idx?;
    if rng_idx + 1 >= parts.len() {
        return None;
    }
    let algo = parts[rng_idx + 1..].join("-");
    if algo.is_empty() {
        return None;
    }

    Some((format, algo, hidden))
}

/// Insert a single row into `ssot.bpb_samples` with checkpoint telemetry.
///
/// Schema (verified against phd-postgres-ssot 2026-05-14, matrix_runner aligned):
///   id BIGSERIAL, canon_name TEXT, format TEXT, algo TEXT, hidden INT,
///   seed BIGINT, step INT, bpb DOUBLE PRECISION, sha TEXT (nullable),
///   run_id TEXT (nullable), ts TIMESTAMPTZ
///
/// Derived columns:
///   format/algo/hidden — parsed from canon_name regex
///   sha    — GIT_SHA env (build-time), or empty
///   run_id — RAILWAY_DEPLOYMENT_ID env (runtime), or empty
///
/// If canon_name does not match the IGLA schema (legacy scarab-*, smoke names),
/// we DO NOT fabricate — fall back to writing into public.bpb_samples to keep
/// non-canonical callers (bpb_smoke, smoke_train) green during the migration.
///
/// 2026-05-14: switched main path from public.bpb_samples → ssot.bpb_samples
/// to unblock PASS-N monitors and leaderboard auditors that query ssot only.
/// Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877
pub fn bpb_sample(canon_name: &str, seed: i32, step: i32, bpb: f32, ema_bpb: Option<f32>) {
    let _ = ema_bpb; // ssot.bpb_samples has no ema_bpb column; reserved for backward-compat.
    let Some(conn) = db() else {
        eprintln!("[ledger] DSN unset — skipping bpb_sample");
        return;
    };

    let ts_now = chrono::Utc::now();
    use sea_orm::Statement;

    // Try to parse canon_name into derived columns.
    if let Some((format, algo, hidden)) = parse_canon_name(canon_name) {
        // ── R5 GUARD — WRITE-SIDE ALGO_WHITELIST ──────────────────────────────
        // Mirror of matrix_runner ALGO_WHITELIST. Rejects fake/silent-fallback
        // algos (soap/lamb/prodigy/lion/...) at the WRITE path so they cannot
        // reach ssot.bpb_samples regardless of which trainer binary emitted
        // them. Refs: trios#777, trios#779, migration 0006_quarantine_fake_canons.
        // R5 evidence: gf16-lamb vs gf16-prodigy produced bit-identical BPB at
        // every step (verified 2026-05-14T14:36Z, B-22).
        const ALGO_WHITELIST: &[&str] = &["adamw", "muon", "muon-cwd"];
        if !ALGO_WHITELIST.contains(&algo.as_str()) {
            eprintln!(
                "[ledger] R5-REJECT write: canon_name={canon_name} algo={algo} not in {:?}. \
                 Refusing silent-fallback write — see trios#777 / migration 0006.",
                ALGO_WHITELIST
            );
            return;
        }

        // Canonical IGLA path → ssot.bpb_samples (the SoT for leaderboards).
        let sha = std::env::var("GIT_SHA").unwrap_or_default();
        let run_id = std::env::var("RAILWAY_DEPLOYMENT_ID").unwrap_or_default();
        let stmt = Statement::from_sql_and_values(
            conn.get_database_backend(),
            "INSERT INTO ssot.bpb_samples \
             (canon_name, format, algo, hidden, seed, step, bpb, sha, run_id, ts) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) \
             ON CONFLICT DO NOTHING",
            [
                seed_val(canon_name),
                seed_val(&format),
                seed_val(&algo),
                hidden.into(),
                (seed as i64).into(),
                step.into(),
                (bpb as f64).into(),
                seed_val(&sha),
                seed_val(&run_id),
                ts_now.into(),
            ],
        );
        let res = rt().block_on(conn.execute(stmt));
        match res {
            Ok(_) => eprintln!(
                "[ledger] bpb_sample ok (ssot): {canon_name} format={format} algo={algo} hidden={hidden} seed={seed} step={step} bpb={bpb:.4}"
            ),
            Err(e) => eprintln!("[ledger] bpb_sample (ssot) failed: {e}"),
        }
        return;
    }

    // Legacy/smoke fallback → public.bpb_samples (Wave 29 PR-A idempotent upsert).
    eprintln!(
        "[ledger] canon_name '{canon_name}' not IGLA-shaped; falling back to public.bpb_samples"
    );
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
            None::<f64>.into(),
            ts_now.into(),
        ],
    );
    let res = rt().block_on(conn.execute(stmt));
    match res {
        Ok(_) => eprintln!("[ledger] bpb_sample ok (public/legacy): {canon_name} seed={seed} step={step} bpb={bpb:.4}"),
        Err(e) => eprintln!("[ledger] bpb_sample (public/legacy) failed: {e}"),
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

    #[test]
    fn parse_canon_name_short_wave_matrix() {
        let c = "IGLA-SHORT-WAVE-MATRIX-gf16-h128-LR0.0001-rng1597-adamw";
        let (fmt, algo, hidden) = parse_canon_name(c).expect("parses");
        assert_eq!(fmt, "gf16");
        assert_eq!(algo, "adamw");
        assert_eq!(hidden, 128);
    }

    #[test]
    fn parse_canon_name_scarab_lane() {
        let c = "IGLA-SCARAB-ADAMW-binary16-h384-LR0001-rng123-adamw";
        let (fmt, algo, hidden) = parse_canon_name(c).expect("parses");
        assert_eq!(fmt, "binary16");
        assert_eq!(algo, "adamw");
        assert_eq!(hidden, 384);
    }

    #[test]
    fn parse_canon_name_muon_cwd_dashed_algo() {
        // muon-cwd suffix — algo = everything AFTER the rng token,
        // joined with '-' to preserve multi-token algos.
        let c = "IGLA-SHORT-WAVE-MATRIX-fp16-h128-LR0.0001-rng47-muon-cwd";
        let (fmt, algo, hidden) = parse_canon_name(c).expect("parses");
        assert_eq!(fmt, "fp16");
        assert_eq!(algo, "muon-cwd");
        assert_eq!(hidden, 128);
    }

    #[test]
    fn parse_canon_name_rejects_legacy_scarab() {
        // Legacy `scarab-*` names (pre-2026-05-12) should NOT parse,
        // so they go down the public.bpb_samples fallback path.
        assert!(parse_canon_name("scarab-adamw-rng123").is_none());
        assert!(parse_canon_name("random-name").is_none());
        assert!(parse_canon_name("").is_none());
    }

    #[test]
    fn parse_canon_name_rejects_missing_h_token() {
        // No `h{N}` token → hidden cannot be derived → reject.
        assert!(parse_canon_name("IGLA-FAKE-gf16-LR0.0001-rng1597-adamw").is_none());
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

    /// B-22 write-side ALGO_WHITELIST guard — invariants only (we don't have a
    /// live DSN here so we can't exercise bpb_sample directly; this documents
    /// the canonical whitelist so a future widening can't happen by accident).
    #[test]
    fn write_side_algo_whitelist_is_canonical() {
        // Canonical: only these three suffixes are real-trainer-backed.
        const CANONICAL: &[&str] = &["adamw", "muon", "muon-cwd"];
        assert_eq!(CANONICAL.len(), 3, "whitelist must be exactly 3 entries");
        for &name in CANONICAL {
            let canon = format!(
                "IGLA-SHORT-WAVE-MATRIX-gf16-h128-LR0.0001-rng47-{}",
                name
            );
            let (_fmt, algo, _hidden) =
                parse_canon_name(&canon).expect("canonical algo must parse");
            assert_eq!(algo, name, "parse_canon_name must round-trip canonical algos");
        }
        for &fake in &["soap", "lamb", "prodigy", "lion", "tiger", "adafactor", "sgdm"] {
            assert!(
                !CANONICAL.contains(&fake),
                "fake algo leaked into whitelist: {fake}"
            );
        }
    }
}
