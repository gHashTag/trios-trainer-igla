// migration/src/m20260510_000001_step_to_bigint.rs
//
// Wave 24 — ALTER step columns to BIGINT (closes Wave 23 drift).
//
// Why: public.bpb_samples.step was declared INT in the init migration, but
// long-running multi-run evaluations may push step beyond 2.1B. All other
// ledger columns (seed, steps_done) are already BIGINT. This migration
// lifts step to BIGINT for type uniformity and future-proofing.
//
// Tables altered:
//   - public.bpb_samples.step        INT  → BIGINT
//   - ssot.bpb_samples.step          INT  → BIGINT (guarded: IF EXISTS)
//   - public.igla_race_trials.final_step  INT → BIGINT
//   - public.igla_race_trials.steps_done  BIGINT already (no-op guard)
//
// All ALTER statements are idempotent via information_schema checks.
//
// down(): Reversing BIGINT → INT is lossy when values > INT_MAX exist.
// The conservative approach is to log a warning and return Ok(()) without
// making any changes. Callers must handle the rollback manually if needed.
//
// Anchor: phi^2 + phi^-2 = 3 · DOI 10.5281/zenodo.19227877

use sea_orm_migration::prelude::*;

pub struct Migration;

impl MigrationName for Migration {
    fn name(&self) -> &str {
        "m20260510_000001_step_to_bigint"
    }
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let db = manager.get_connection();

        // ── public.bpb_samples.step ──────────────────────────────────────
        db.execute_unprepared(
            r#"
DO $$
BEGIN
    IF (SELECT data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name   = 'bpb_samples'
          AND column_name  = 'step') = 'integer' THEN
        ALTER TABLE public.bpb_samples
            ALTER COLUMN step TYPE BIGINT USING step::BIGINT;
    END IF;
END $$;
"#,
        )
        .await?;

        // ── ssot.bpb_samples.step ────────────────────────────────────────
        // Guarded by IF EXISTS in case the ssot schema is absent on a fresh DB.
        db.execute_unprepared(
            r#"
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'ssot' AND table_name = 'bpb_samples'
    ) THEN
        IF (SELECT data_type
            FROM information_schema.columns
            WHERE table_schema = 'ssot'
              AND table_name   = 'bpb_samples'
              AND column_name  = 'step') = 'integer' THEN
            ALTER TABLE ssot.bpb_samples
                ALTER COLUMN step TYPE BIGINT USING step::BIGINT;
        END IF;
    END IF;
END $$;
"#,
        )
        .await?;

        // ── public.igla_race_trials.final_step ───────────────────────────
        db.execute_unprepared(
            r#"
DO $$
BEGIN
    IF (SELECT data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name   = 'igla_race_trials'
          AND column_name  = 'final_step') = 'integer' THEN
        ALTER TABLE public.igla_race_trials
            ALTER COLUMN final_step TYPE BIGINT USING final_step::BIGINT;
    END IF;
END $$;
"#,
        )
        .await?;

        // ── public.igla_race_trials.steps_done ──────────────────────────
        // The init migration already declared steps_done as BIGINT (line 109),
        // so this is a defensive no-op guard; it will never ALTER in practice.
        db.execute_unprepared(
            r#"
DO $$
BEGIN
    IF (SELECT data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name   = 'igla_race_trials'
          AND column_name  = 'steps_done') = 'integer' THEN
        ALTER TABLE public.igla_race_trials
            ALTER COLUMN steps_done TYPE BIGINT USING steps_done::BIGINT;
    END IF;
END $$;
"#,
        )
        .await?;

        // ── public.scarabs ───────────────────────────────────────────────
        // Audit result: scarabs has no step / final_step column (only
        // current_strategy_id which is already BIGINT). No ALTER needed.

        Ok(())
    }

    /// Reversing BIGINT → INT is a lossy operation: any value > 2,147,483,647
    /// would be silently truncated or cause a Postgres cast error. Rather than
    /// risk data loss, this down() is intentionally a no-op. Operators who need
    /// to roll back must do so manually after verifying no out-of-range values
    /// exist in any of the affected columns.
    async fn down(&self, _manager: &SchemaManager) -> Result<(), DbErr> {
        eprintln!(
            "[migration] m20260510_000001_step_to_bigint: down() is a deliberate no-op. \
             Reversing BIGINT → INT is lossy when values > INT_MAX exist. \
             Perform manual rollback if required."
        );
        Ok(())
    }
}
