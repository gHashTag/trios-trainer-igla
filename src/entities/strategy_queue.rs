// src/entities/strategy_queue.rs
//
// SeaORM entity for public.strategy_queue (job queue).

use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "strategy_queue", schema_name = "public")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i64,
    pub canon_name: String,
    pub config: Json,
    /// BIGINT -> i64.
    pub seed: i64,
    pub priority: i32,
    pub status: String,
    pub worker_id: Option<String>,
    pub error_msg: Option<String>,
    pub started_at: Option<DateTimeWithTimeZone>,
    pub completed_at: Option<DateTimeWithTimeZone>,
    pub created_at: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
