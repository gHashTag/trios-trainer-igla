// src/entities/igla_race_trials.rs
//
// SeaORM entity for public.igla_race_trials.

use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "igla_race_trials", schema_name = "public")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false, column_type = "Uuid")]
    pub trial_id: Uuid,
    pub config: Json,
    pub status: String,
    pub agent_id: Option<String>,
    pub branch: Option<String>,
    pub final_bpb: Option<f64>,
    pub final_step: Option<i64>,
    pub bpb_latest: Option<f64>,
    pub steps_done: Option<i64>,
    pub bpb_final: Option<f64>,
    pub started_at: Option<DateTimeWithTimeZone>,
    pub completed_at: Option<DateTimeWithTimeZone>,
    pub created_at: DateTimeWithTimeZone,
    pub updated_at: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
