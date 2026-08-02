// migration/src/m20260802_000003_checkpoints.rs
//
// EPIC-446: bind a training run to the checkpoint artifact it produced.
//
// Why a new table rather than columns on bpb_samples:
//   * bpb_sample fires every `eval_every` (default 1000 -> 81 rows for an 81k
//     run); checkpoints are a subset, so nearly every row would carry NULLs.
//   * public.bpb_samples upserts `bpb = LEAST(EXCLUDED.bpb, bpb_samples.bpb)`
//     - "keep the best", not last-write-wins. A rerun at the same
//     (canon, seed, step) would keep the OLD bpb while overwriting the
//     artifact hash, producing a row claiming a BPB the recorded checkpoint
//     did not produce.
//   * ssot.bpb_samples and public.bpb_samples are different tables reached by
//     different canon shapes: two column additions, two code paths.
//
// UNIQUE (canon_name, seed, step) mirrors the only unique tuple in the schema
// today (public.bpb_samples), so the evidence link is a plain cross-schema
// join that works for BOTH bpb paths:
//   SELECT b.bpb, c.path, c.sha256, c.bytes
//   FROM public.bpb_samples b
//   JOIN ssot.checkpoints c USING (canon_name, seed, step);
//
// Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877

use sea_orm_migration::prelude::*;

pub struct Migration;

impl MigrationName for Migration {
    fn name(&self) -> &str {
        "m20260802_000003_checkpoints"
    }
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        // The init migration already creates the schema; repeated here so this
        // migration is self-contained against a fresh database.
        manager
            .get_connection()
            .execute_unprepared("CREATE SCHEMA IF NOT EXISTS ssot")
            .await?;
        manager
            .get_connection()
            .execute_unprepared(
                "CREATE TABLE IF NOT EXISTS ssot.checkpoints ( \
                     id             BIGSERIAL PRIMARY KEY, \
                     canon_name     TEXT   NOT NULL, \
                     seed           BIGINT NOT NULL, \
                     step           BIGINT NOT NULL, \
                     path           TEXT   NOT NULL, \
                     sha256         TEXT   NOT NULL, \
                     bytes          BIGINT NOT NULL, \
                     algo           TEXT   NOT NULL, \
                     hidden         INT    NOT NULL, \
                     format_version INT    NOT NULL, \
                     data_synthetic BOOLEAN NOT NULL DEFAULT false, \
                     bpb            DOUBLE PRECISION, \
                     sha            TEXT, \
                     run_id         TEXT, \
                     ts             TIMESTAMPTZ NOT NULL DEFAULT now(), \
                     UNIQUE (canon_name, seed, step) \
                 )",
            )
            .await?;
        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        // Drops the table only, never the schema: ssot holds other tables.
        manager
            .get_connection()
            .execute_unprepared("DROP TABLE IF EXISTS ssot.checkpoints")
            .await?;
        Ok(())
    }
}
