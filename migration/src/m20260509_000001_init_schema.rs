// Migration: m20260509_000001_init_schema
//
// Mirrors the canonical /tmp/full_ddl.sql 1-to-1.
// All statements are idempotent (CREATE TABLE IF NOT EXISTS, etc.).
//
// Tables:
//   - public.bpb_samples       (legacy ledger, used by neon_writer)
//   - ssot.bpb_samples         (new SoT, per matrix_runner.rs)
//   - public.igla_race_trials  (trial lifecycle tracking)
//   - public.igla_agents_heartbeat (worker heartbeats)
//   - public.scarabs           (fungible worker pool, migration 002)
//   - public.strategy_queue    (job queue with NOTIFY trigger)
//
// Anchor: phi^2 + phi^-2 = 3 · DOI 10.5281/zenodo.19227877

use sea_orm_migration::prelude::*;

pub struct Migration;

impl MigrationName for Migration {
    fn name(&self) -> &str {
        "m20260509_000001_init_schema"
    }
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let db = manager.get_connection();

        // ── public.bpb_samples (legacy) ──────────────────────────────────
        db.execute_unprepared(
            "CREATE TABLE IF NOT EXISTS public.bpb_samples (
                id            BIGSERIAL PRIMARY KEY,
                canon_name    TEXT NOT NULL,
                seed          BIGINT NOT NULL,
                step          INT NOT NULL,
                bpb           DOUBLE PRECISION NOT NULL,
                ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (canon_name, seed, step)
            )",
        )
        .await?;
        db.execute_unprepared(
            "CREATE INDEX IF NOT EXISTS bpb_samples_canon_idx ON public.bpb_samples (canon_name)",
        )
        .await?;
        db.execute_unprepared(
            "CREATE INDEX IF NOT EXISTS bpb_samples_ts_idx ON public.bpb_samples (ts DESC)",
        )
        .await?;

        // ── ssot schema + ssot.bpb_samples ───────────────────────────────
        db.execute_unprepared("CREATE SCHEMA IF NOT EXISTS ssot")
            .await?;
        db.execute_unprepared(
            "CREATE TABLE IF NOT EXISTS ssot.bpb_samples (
                id            BIGSERIAL PRIMARY KEY,
                canon_name    TEXT NOT NULL,
                format        TEXT NOT NULL,
                algo          TEXT NOT NULL,
                hidden        INT NOT NULL,
                seed          BIGINT NOT NULL,
                step          INT NOT NULL,
                bpb           DOUBLE PRECISION NOT NULL,
                sha           TEXT,
                run_id        TEXT,
                ts            TIMESTAMPTZ NOT NULL DEFAULT now()
            )",
        )
        .await?;
        db.execute_unprepared(
            "CREATE INDEX IF NOT EXISTS ssot_bpb_format_algo_idx ON ssot.bpb_samples (format, algo)",
        )
        .await?;
        db.execute_unprepared(
            "CREATE INDEX IF NOT EXISTS ssot_bpb_canon_idx ON ssot.bpb_samples (canon_name)",
        )
        .await?;
        db.execute_unprepared(
            "CREATE INDEX IF NOT EXISTS ssot_bpb_ts_idx ON ssot.bpb_samples (ts DESC)",
        )
        .await?;

        // ── public.igla_race_trials ───────────────────────────────────────
        db.execute_unprepared(
            "CREATE TABLE IF NOT EXISTS public.igla_race_trials (
                trial_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                config        JSONB NOT NULL,
                status        VARCHAR(32) NOT NULL DEFAULT 'pending',
                agent_id      TEXT,
                branch        TEXT,
                final_bpb     DOUBLE PRECISION,
                final_step    INT,
                started_at    TIMESTAMPTZ,
                completed_at  TIMESTAMPTZ,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            )",
        )
        .await?;

        // Ensure bpb_latest column exists (added by ensure_schema() hotfix).
        db.execute_unprepared(
            "ALTER TABLE public.igla_race_trials ADD COLUMN IF NOT EXISTS bpb_latest DOUBLE PRECISION",
        )
        .await?;
        db.execute_unprepared(
            "ALTER TABLE public.igla_race_trials ADD COLUMN IF NOT EXISTS steps_done BIGINT",
        )
        .await?;
        db.execute_unprepared(
            "ALTER TABLE public.igla_race_trials ADD COLUMN IF NOT EXISTS bpb_final DOUBLE PRECISION",
        )
        .await?;

        // ── public.igla_agents_heartbeat ──────────────────────────────────
        db.execute_unprepared(
            "CREATE TABLE IF NOT EXISTS public.igla_agents_heartbeat (
                agent_id        TEXT PRIMARY KEY,
                machine_id      TEXT NOT NULL DEFAULT 'railway',
                branch          TEXT NOT NULL DEFAULT 'main',
                task            TEXT,
                status          VARCHAR(32) NOT NULL DEFAULT 'active',
                last_heartbeat  TIMESTAMPTZ NOT NULL DEFAULT now()
            )",
        )
        .await?;

        // ── public.scarabs ────────────────────────────────────────────────
        db.execute_unprepared(
            "CREATE TABLE IF NOT EXISTS public.scarabs (
                scarab_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                label                TEXT,
                host                 TEXT,
                railway_service_id   TEXT,
                railway_service_name TEXT,
                current_strategy_id  BIGINT,
                last_seen            TIMESTAMPTZ NOT NULL DEFAULT now(),
                created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
            )",
        )
        .await?;

        // ── public.strategy_queue ─────────────────────────────────────────
        db.execute_unprepared(
            "CREATE TABLE IF NOT EXISTS public.strategy_queue (
                id           BIGSERIAL PRIMARY KEY,
                canon_name   TEXT NOT NULL,
                config       JSONB NOT NULL,
                seed         BIGINT NOT NULL,
                priority     INT NOT NULL DEFAULT 0,
                status       VARCHAR(32) NOT NULL DEFAULT 'pending',
                worker_id    TEXT,
                error_msg    TEXT,
                started_at   TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
            )",
        )
        .await?;
        db.execute_unprepared(
            "CREATE INDEX IF NOT EXISTS idx_strategy_pending
                ON public.strategy_queue (priority DESC, id ASC)
                WHERE status = 'pending'",
        )
        .await?;

        // ── NOTIFY trigger for strategy_queue ─────────────────────────────
        db.execute_unprepared(
            "CREATE OR REPLACE FUNCTION public.notify_strategy_new() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('strategy_new', NEW.id::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql",
        )
        .await?;
        db.execute_unprepared("DROP TRIGGER IF EXISTS trg_strategy_new ON public.strategy_queue")
            .await?;
        db.execute_unprepared(
            "CREATE TRIGGER trg_strategy_new
                AFTER INSERT OR UPDATE OF status ON public.strategy_queue
                FOR EACH ROW WHEN (NEW.status = 'pending')
                EXECUTE FUNCTION public.notify_strategy_new()",
        )
        .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let db = manager.get_connection();

        db.execute_unprepared("DROP TRIGGER IF EXISTS trg_strategy_new ON public.strategy_queue")
            .await?;
        db.execute_unprepared("DROP FUNCTION IF EXISTS public.notify_strategy_new()")
            .await?;
        db.execute_unprepared("DROP TABLE IF EXISTS public.strategy_queue")
            .await?;
        db.execute_unprepared("DROP TABLE IF EXISTS public.scarabs")
            .await?;
        db.execute_unprepared("DROP TABLE IF EXISTS public.igla_agents_heartbeat")
            .await?;
        db.execute_unprepared("DROP TABLE IF EXISTS public.igla_race_trials")
            .await?;
        db.execute_unprepared("DROP TABLE IF EXISTS ssot.bpb_samples")
            .await?;
        db.execute_unprepared("DROP SCHEMA IF EXISTS ssot").await?;
        db.execute_unprepared("DROP TABLE IF EXISTS public.bpb_samples")
            .await?;

        Ok(())
    }
}
