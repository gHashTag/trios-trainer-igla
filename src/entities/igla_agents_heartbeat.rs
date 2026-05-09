// src/entities/igla_agents_heartbeat.rs
//
// SeaORM entity for public.igla_agents_heartbeat.

use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "igla_agents_heartbeat", schema_name = "public")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub agent_id: String,
    pub machine_id: String,
    pub branch: String,
    pub task: Option<String>,
    pub status: String,
    pub last_heartbeat: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
