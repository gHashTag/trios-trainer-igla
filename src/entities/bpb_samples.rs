// src/entities/bpb_samples.rs
//
// SeaORM entity for public.bpb_samples (legacy ledger table).
//
// Column types match the DDL in /tmp/full_ddl.sql:
//   id BIGSERIAL, canon_name TEXT, seed BIGINT, step BIGINT, bpb DOUBLE PRECISION,
//   ema_bpb DOUBLE PRECISION (nullable, added Wave 29 PR-A),
//   ts TIMESTAMPTZ
//
// Note: seed is BIGINT -> i64 in Rust (fixes the i32/INT8 mismatch from #114).
// Note: ema_bpb added Wave 29 PR-A for idempotent ON CONFLICT DO UPDATE.

use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "bpb_samples", schema_name = "public")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i64,
    pub canon_name: String,
    /// BIGINT — must be i64, not i32 (fixes #114).
    pub seed: i64,
    pub step: i64,
    pub bpb: f64,
    /// Exponential moving average of BPB. Nullable (NULL for rows written before Wave 29).
    pub ema_bpb: Option<f64>,
    pub ts: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
