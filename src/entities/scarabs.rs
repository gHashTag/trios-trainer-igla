// src/entities/scarabs.rs
//
// SeaORM entity for public.scarabs (fungible worker pool).

use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "scarabs", schema_name = "public")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false, column_type = "Uuid")]
    pub scarab_id: Uuid,
    pub label: Option<String>,
    pub host: Option<String>,
    pub railway_service_id: Option<String>,
    pub railway_service_name: Option<String>,
    /// BIGINT -> i64.
    pub current_strategy_id: Option<i64>,
    pub last_seen: DateTimeWithTimeZone,
    pub created_at: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
