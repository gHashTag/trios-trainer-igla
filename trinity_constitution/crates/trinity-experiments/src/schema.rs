//! Schema — Experiment domain model
//!
//! # Constitutional mandate (Law 2)
//!
//! Single source of truth for all experiments.
//! No derived tables — bpb_curve embedded as JSONB.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::error::Error;
use tokio_postgres::types::{to_sql_checked, FromSql, IsNull, ToSql, Type};
use uuid::Uuid;

/// Experiment status — one-way state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentStatus {
    Pending,
    Running,
    Done,
    Failed,
}

impl ExperimentStatus {
    /// Check if status is terminal (Done or Failed)
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done | Self::Failed)
    }
}

// Implement From/Into String for tokio-postgres compatibility
impl From<String> for ExperimentStatus {
    fn from(s: String) -> Self {
        match s.to_lowercase().as_str() {
            "pending" => ExperimentStatus::Pending,
            "running" => ExperimentStatus::Running,
            "done" => ExperimentStatus::Done,
            "failed" => ExperimentStatus::Failed,
            _ => panic!("Invalid status: {}", s),
        }
    }
}

impl From<ExperimentStatus> for String {
    fn from(status: ExperimentStatus) -> Self {
        match status {
            ExperimentStatus::Pending => "pending".into(),
            ExperimentStatus::Running => "running".into(),
            ExperimentStatus::Done => "done".into(),
            ExperimentStatus::Failed => "failed".into(),
        }
    }
}

impl<'a> FromSql<'a> for ExperimentStatus {
    fn from_sql(
        ty: &Type,
        raw: &'a [u8],
    ) -> Result<Self, Box<dyn Error + Sync + Send>> {
        let s = String::from_sql(ty, raw)?;
        Ok(ExperimentStatus::from(s))
    }

    fn accepts(ty: &Type) -> bool {
        <String as FromSql>::accepts(ty)
    }
}

impl ToSql for ExperimentStatus {
    fn to_sql(
        &self,
        ty: &Type,
        out: &mut bytes::BytesMut,
    ) -> Result<IsNull, Box<dyn Error + Sync + Send>>
    where
        Self: Sized,
    {
        let s: String = (*self).into();
        <String as ToSql>::to_sql(&s, ty, out)
    }

    fn accepts(ty: &Type) -> bool {
        <String as ToSql>::accepts(ty)
    }

    to_sql_checked!();
}

/// BPB curve point — embedded in experiment.bpb_curve JSONB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpbPoint {
    pub step: i32,
    pub bpb: f64,
}

/// Trainer configuration from config_json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub seed: u64,
    pub hidden: u32,
    pub lr: f64,
    pub steps: u32,
    pub format: String,
    pub corpus: String,
    pub train_path: Option<String>,
    pub val_path: Option<String>,
}

/// Experiment — single source of truth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    // Identity
    pub id: i64,
    pub canon_name: String,

    // PhD config (READ-ONLY after insert)
    pub phd_chapter: String,
    pub inv_id: String,
    pub config_json: serde_json::Value,
    pub required_image_tag: String,

    // Lifecycle (one-way state machine)
    pub status: ExperimentStatus,
    pub worker_id: Option<Uuid>,
    pub claimed_at: Option<DateTime<Utc>>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,

    // Result (filled exactly once on done/failed)
    pub final_bpb: Option<f64>,
    pub final_step: Option<i32>,
    pub bpb_curve: Vec<BpbPoint>,
    pub last_error: Option<String>,

    // Audit
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Experiment {
    /// Check if experiment has a valid result (R5-honest)
    pub fn has_valid_result(&self) -> bool {
        match self.status {
            ExperimentStatus::Done | ExperimentStatus::Failed => {
                self.final_bpb.is_some() || self.last_error.is_some()
            }
            _ => false,
        }
    }

    /// Parse config_json into ExperimentConfig
    pub fn parse_config(&self) -> Result<ExperimentConfig, serde_json::Error> {
        serde_json::from_value(self.config_json.clone())
    }
}
