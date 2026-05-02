//! Repository — ExperimentRepo trait and implementation
//!
//! # Constitutional mandate (Law 1)
//!
//! No claim without verified source row.
//! All writes are idempotent with `FOR UPDATE SKIP LOCKED`.

use super::schema::{Experiment, ExperimentConfig, ExperimentStatus, BpbPoint};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde_json::json;
use tokio_postgres::{Client, NoTls};
use uuid::Uuid;

/// Claim of a pending experiment
#[derive(Debug, Clone)]
pub struct PendingClaim {
    pub id: i64,
    pub canon_name: String,
    pub config: ExperimentConfig,
    pub steps_budget: i32,
}

/// Result of claiming an experiment
#[derive(Debug, Clone)]
pub enum ClaimResult {
    Claimed(PendingClaim),
    NoPending,
}

/// Repository trait for experiment operations
#[async_trait::async_trait]
pub trait ExperimentRepo: Send + Sync {
    /// Claim next pending experiment (idempotent with FOR UPDATE SKIP LOCKED)
    async fn claim_next(&self, worker_id: Uuid) -> Result<ClaimResult>;

    /// Start training — mark as running with started_at
    async fn start(&self, id: i64, worker_id: Uuid) -> Result<()>;

    /// Record BPB sample during training
    async fn record_bpb(&self, canon_name: &str, step: i32, bpb: f64) -> Result<()>;

    /// Complete experiment with final results
    async fn complete(
        &self,
        id: i64,
        final_bpb: f64,
        final_step: i32,
        bpb_curve: Vec<BpbPoint>,
    ) -> Result<()>;

    /// Fail experiment with error
    async fn fail(&self, id: i64, error: String) -> Result<()>;

    /// Get experiment by canon_name
    async fn get_by_canon_name(&self, canon_name: &str) -> Result<Option<Experiment>>;

    /// List pending experiments
    async fn list_pending(&self, limit: usize) -> Result<Vec<Experiment>>;
}

/// Postgres implementation of ExperimentRepo
pub struct PostgresExperimentRepo {
    client: Client,
}

impl PostgresExperimentRepo {
    /// Create new repo from existing client
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    /// Create new repo from connection string
    pub async fn connect(url: &str) -> Result<Self> {
        let (client, conn) = tokio_postgres::connect(url, NoTls).await?;
        tokio::spawn(async move {
            if let Err(e) = conn.await {
                eprintln!("[repo] connection error: {e}");
            }
        });
        Ok(Self::new(client))
    }
}

#[async_trait::async_trait]
impl ExperimentRepo for PostgresExperimentRepo {
    async fn claim_next(&self, worker_id: Uuid) -> Result<ClaimResult> {
        let row = self
            .client
            .query_opt(
                r#"
                UPDATE strategy_experiments
                SET status = 'running',
                    worker_id = $1,
                    claimed_at = NOW(),
                    updated_at = NOW()
                WHERE id = (
                    SELECT id FROM strategy_experiments
                    WHERE status = 'pending'
                    ORDER BY id ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING id, canon_name, config_json
                "#,
                &[&worker_id],
            )
            .await?;

        match row {
            Some(row) => {
                let id: i64 = row.get(0);
                let canon_name: String = row.get(1);
                let config_json: serde_json::Value = row.get(2);
                let config: ExperimentConfig = serde_json::from_value(config_json)?;

                Ok(ClaimResult::Claimed(PendingClaim {
                    id,
                    canon_name,
                    config,
                    steps_budget: 0, // Will be read from config
                }))
            }
            None => Ok(ClaimResult::NoPending),
        }
    }

    async fn start(&self, id: i64, worker_id: Uuid) -> Result<()> {
        self.client
            .execute(
                "UPDATE strategy_experiments SET started_at = NOW(), updated_at = NOW() WHERE id = $1 AND worker_id = $2",
                &[&id, &worker_id],
            )
            .await?;
        Ok(())
    }

    async fn record_bpb(&self, canon_name: &str, step: i32, bpb: f64) -> Result<()> {
        self.client
            .execute(
                r#"
                UPDATE strategy_experiments
                SET bpb_curve = COALESCE(bpb_curve, '[]'::jsonb) || jsonb_build_object('step', $1, 'bpb', $2),
                    updated_at = NOW()
                WHERE canon_name = $3 AND status = 'running'
                "#,
                &[&step, &bpb, &canon_name],
            )
            .await?;
        Ok(())
    }

    async fn complete(
        &self,
        id: i64,
        final_bpb: f64,
        final_step: i32,
        bpb_curve: Vec<BpbPoint>,
    ) -> Result<()> {
        let curve_json = serde_json::to_value(&bpb_curve)?;
        self.client
            .execute(
                r#"
                UPDATE strategy_experiments
                SET status = 'done',
                    finished_at = NOW(),
                    final_bpb = $1,
                    final_step = $2,
                    bpb_curve = $3,
                    updated_at = NOW()
                WHERE id = $4
                "#,
                &[&final_bpb, &final_step, &curve_json, &id],
            )
            .await?;
        Ok(())
    }

    async fn fail(&self, id: i64, error: String) -> Result<()> {
        self.client
            .execute(
                r#"
                UPDATE strategy_experiments
                SET status = 'failed',
                    finished_at = NOW(),
                    last_error = $1,
                    updated_at = NOW()
                WHERE id = $2
                "#,
                &[&error, &id],
            )
            .await?;
        Ok(())
    }

    async fn get_by_canon_name(&self, canon_name: &str) -> Result<Option<Experiment>> {
        let row = self
            .client
            .query_opt(
                "SELECT * FROM strategy_experiments WHERE canon_name = $1",
                &[&canon_name],
            )
            .await?;

        match row {
            Some(row) => Ok(Some(row_to_experiment(row)?)),
            None => Ok(None),
        }
    }

    async fn list_pending(&self, limit: usize) -> Result<Vec<Experiment>> {
        let rows = self
            .client
            .query(
                "SELECT * FROM strategy_experiments WHERE status = 'pending' ORDER BY id ASC LIMIT $1",
                &[&(limit as i64)],
            )
            .await?;

        rows.into_iter()
            .map(row_to_experiment)
            .collect()
    }
}

fn row_to_experiment(row: tokio_postgres::Row) -> Result<Experiment> {
    Ok(Experiment {
        id: row.get("id"),
        canon_name: row.get("canon_name"),
        phd_chapter: row.get("phd_chapter"),
        inv_id: row.get("inv_id"),
        config_json: row.get("config_json"),
        required_image_tag: row.get("required_image_tag"),
        status: row.get("status"),
        worker_id: row.get("worker_id"),
        claimed_at: row.get("claimed_at"),
        started_at: row.get("started_at"),
        finished_at: row.get("finished_at"),
        final_bpb: row.get("final_bpb"),
        final_step: row.get("final_step"),
        bpb_curve: row
            .get::<_, Option<serde_json::Value>>("bpb_curve")
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default(),
        last_error: row.get("last_error"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    })
}
