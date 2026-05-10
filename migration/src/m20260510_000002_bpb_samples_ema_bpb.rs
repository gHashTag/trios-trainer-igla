// migration/src/m20260510_000002_bpb_samples_ema_bpb.rs
//
// Wave 29 PR-A: add ema_bpb DOUBLE PRECISION column to public.bpb_samples.
//
// Why: ON CONFLICT DO UPDATE for idempotent inserts needs ema_bpb to track
// the exponential moving average of BPB alongside the raw bpb value.
// The column is nullable so existing rows are not affected.
//
// Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877

use sea_orm_migration::prelude::*;

pub struct Migration;

impl MigrationName for Migration {
    fn name(&self) -> &str {
        "m20260510_000002_bpb_samples_ema_bpb"
    }
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        // Add ema_bpb column to public.bpb_samples if it doesn't exist.
        manager
            .get_connection()
            .execute_unprepared(
                "ALTER TABLE public.bpb_samples \
                 ADD COLUMN IF NOT EXISTS ema_bpb DOUBLE PRECISION",
            )
            .await?;
        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .get_connection()
            .execute_unprepared(
                "ALTER TABLE public.bpb_samples DROP COLUMN IF EXISTS ema_bpb",
            )
            .await?;
        Ok(())
    }
}
