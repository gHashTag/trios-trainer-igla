//! Race ledger backend -- DELIBERATELY UNIMPLEMENTED, AND SAYS SO.
//!
//! `NeonDb` never opened a socket. `connect` slept 50ms, logged
//! "Connected to Neon (STUB)" and returned `Ok`; every writer returned `Ok(())`
//! after writing nothing; `query` returned an empty row set. That is the same
//! shape as the `checkpoint::save` stub that returned `Ok(())` for 1,851
//! experiments and produced zero artifacts -- and here it was worse, because
//! `tri race status` / `tri race best` turned the empty row set into positive
//! factual claims about a shared ledger ("No completed trials yet") that
//! nothing had been asked.
//!
//! So the type now refuses. `connect` returns `Err` for every connection
//! string, well-formed or not, and every method returns the same `Err`. A
//! caller cannot obtain a `NeonDb`, so it cannot receive an answer that was
//! never measured. Restoring the backend means implementing `connect` against
//! a real `tokio_postgres` client, not deleting the refusal.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialConfig {
    pub arch: String,
    #[serde(rename = "d_model")]
    pub hidden: usize,
    #[serde(rename = "n_gram")]
    pub context: usize,
    pub lr: f64,
    pub seed: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimizer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wd: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMeta {
    pub agent_id: String,
    pub branch: String,
    pub machine_id: String,
    pub worker_id: String,
}

impl Default for DashboardMeta {
    fn default() -> Self {
        Self {
            agent_id: "ALPHA".to_string(),
            branch: "main".to_string(),
            machine_id: "unknown".to_string(),
            worker_id: "w0".to_string(),
        }
    }
}

impl DashboardMeta {
    pub fn new(agent_id: &str, machine_id: &str, worker_id: &str) -> Self {
        Self {
            agent_id: agent_id.to_string(),
            branch: "main".to_string(),
            machine_id: machine_id.to_string(),
            worker_id: worker_id.to_string(),
        }
    }

    pub fn with_branch(mut self, branch: &str) -> Self {
        self.branch = branch.to_string();
        self
    }
}

#[derive(Debug, Clone)]
pub struct LessonEntry {
    pub lesson: String,
    pub lesson_type: String,
    pub pattern_count: i32,
}

/// The single refusal every race-backend entry point returns.
///
/// Printed by `tri race status` / `tri race best` / `tri race start` on stderr
/// before they exit non-zero.
pub const STUB_REFUSAL: &str =
    "race backend is a stub: no database is contacted, and no answer from it is a measurement";

fn refuse<T>() -> Result<T> {
    Err(anyhow::anyhow!("{STUB_REFUSAL}"))
}

/// A handle that cannot be obtained.
///
/// The private field keeps construction inside this module, and this module
/// never constructs one -- so every method below is unreachable by design and
/// exists only to keep the callers (`race::status`, `race::lessons`,
/// `race::asha`) compiling against the shape a real backend would have.
pub struct NeonDb {
    _uninhabitable: (),
}

impl NeonDb {
    /// Always `Err`, for every connection string.
    ///
    /// A well-formed DSN is not evidence that anything is reachable, and this
    /// module reaches nothing. See `STUB_REFUSAL`.
    pub async fn connect(conn_str: &str) -> Result<Self> {
        let _ = conn_str;
        refuse()
    }

    pub fn client(&self) -> &Self {
        self
    }

    pub async fn register_trial(
        &self,
        trial_id: &Uuid,
        machine_id: &str,
        worker_id: i32,
        config_json: &str,
    ) -> Result<()> {
        let _ = (trial_id, machine_id, worker_id, config_json);
        refuse()
    }

    pub async fn record_checkpoint(&self, trial_id: &Uuid, rung: i32, bpb: f64) -> Result<()> {
        let _ = (trial_id, rung, bpb);
        refuse()
    }

    pub async fn update_rung(&self, trial_id: &str, rung_steps: usize, bpb: f64) -> Result<()> {
        let _ = (trial_id, rung_steps, bpb);
        refuse()
    }

    pub async fn update_heartbeat(&self, trial_id: &str) -> Result<()> {
        let _ = trial_id;
        refuse()
    }

    pub async fn mark_pruned(&self, trial_id: &Uuid, at_step: i32, bpb: f64) -> Result<()> {
        let _ = (trial_id, at_step, bpb);
        refuse()
    }

    /// Record that the trainer died before producing a reading.
    ///
    /// Deliberately takes no BPB: a crash has no bits-per-byte. The previous
    /// caller passed the magic float `999.0` into `mark_pruned`, which is
    /// exactly what `neon_writer::reject_bpb` exists to catch on the other
    /// writer.
    pub async fn mark_crashed(&self, trial_id: &Uuid, at_step: i32, detail: &str) -> Result<()> {
        let _ = (trial_id, at_step, detail);
        refuse()
    }

    pub async fn mark_completed(&self, trial_id: &Uuid, bpb: f64, steps: i32) -> Result<()> {
        let _ = (trial_id, bpb, steps);
        refuse()
    }

    pub async fn mark_winner(&self, trial_id: &str, bpb: f64, steps: usize) -> Result<()> {
        let _ = (trial_id, bpb, steps);
        refuse()
    }

    pub async fn is_config_running(&self, machine_id: &str, config_json: &str) -> Result<bool> {
        let _ = (machine_id, config_json);
        refuse()
    }

    pub async fn get_median_bpb_at_rung(&self, rung_steps: usize) -> Result<Option<f64>> {
        let _ = rung_steps;
        refuse()
    }

    pub async fn store_lesson(
        &self,
        trial_id: &Uuid,
        outcome: &str,
        pruned_at_rung: i32,
        bpb_at_pruned: f64,
        lesson: &str,
        lesson_type: &str,
    ) -> Result<()> {
        let _ = (
            trial_id,
            outcome,
            pruned_at_rung,
            bpb_at_pruned,
            lesson,
            lesson_type,
        );
        refuse()
    }

    pub async fn get_top_lessons(&self, limit: i32) -> Result<Vec<LessonEntry>> {
        let _ = limit;
        refuse()
    }

    pub async fn query(
        &self,
        query: &str,
        _params: &[&(dyn tokio_postgres::types::ToSql + Sync)],
    ) -> Result<Vec<tokio_postgres::Row>> {
        let _ = query;
        refuse()
    }

    pub async fn query_one(
        &self,
        query: &str,
        _params: &[&(dyn tokio_postgres::types::ToSql + Sync)],
    ) -> Result<tokio_postgres::Row> {
        let _ = query;
        refuse()
    }
}

pub const SCHEMA_MIGRATION: &str = r#"
ALTER TABLE igla_race_trials ADD COLUMN IF NOT EXISTS branch TEXT DEFAULT 'main';
ALTER TABLE igla_race_trials ADD COLUMN IF NOT EXISTS agent_id TEXT;
ALTER TABLE igla_race_trials ADD COLUMN IF NOT EXISTS last_heartbeat TIMESTAMPTZ DEFAULT NOW();
"#;

pub mod queries {
    pub const LEADERBOARD: &str = r#"
SELECT
  agent_id,
  branch,
  config->>'arch' as arch,
  config->>'d_model' as d_model,
  rung_1000_bpb,
  rung_3000_bpb,
  final_bpb,
  status,
  last_heartbeat,
  EXTRACT(EPOCH FROM (NOW() - last_heartbeat)) as heartbeat_lag_sec
FROM igla_race_trials
ORDER BY COALESCE(final_bpb, rung_3000_bpb, rung_1000_bpb, 999) ASC
LIMIT 20;
"#;

    pub const ACTIVE_AGENTS: &str = r#"
SELECT agent_id, machine_id, branch, COUNT(*) as active_trials
FROM igla_race_trials
WHERE status='running'
  AND last_heartbeat > NOW() - INTERVAL '2 minutes'
GROUP BY agent_id, machine_id, branch;
"#;

    pub const BEST_BY_ARCH: &str = r#"
SELECT config->>'arch' as arch, MIN(final_bpb) as best_bpb, COUNT(*) as trials
FROM igla_race_trials
WHERE status IN ('completed', 'winner')
GROUP BY config->>'arch'
ORDER BY best_bpb ASC NULLS LAST;
"#;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trial_config_serialization() {
        let config = TrialConfig {
            arch: "ngram".to_string(),
            hidden: 384,
            context: 6,
            lr: 0.004,
            seed: 42,
            optimizer: Some("adamw".to_string()),
            wd: Some(0.01),
            activation: Some("relu".to_string()),
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("ngram"));
        assert!(json.contains("384"));
    }

    #[test]
    fn test_dashboard_meta_default() {
        let meta = DashboardMeta::default();
        assert_eq!(meta.agent_id, "ALPHA");
        assert_eq!(meta.branch, "main");
    }

    #[test]
    fn test_dashboard_meta_custom() {
        let meta = DashboardMeta::new("BETA", "mac-studio-2", "w1");
        assert_eq!(meta.agent_id, "BETA");
        assert_eq!(meta.machine_id, "mac-studio-2");
        assert_eq!(meta.worker_id, "w1");
        assert_eq!(meta.branch, "main");
    }

    #[test]
    fn test_dashboard_meta_with_branch() {
        let meta = DashboardMeta::new("GAMMA", "macbook-pro-1", "w0").with_branch("feat/jepa");
        assert_eq!(meta.branch, "feat/jepa");
    }

    #[test]
    fn test_schema_migration_contains_columns() {
        assert!(SCHEMA_MIGRATION.contains("branch"));
        assert!(SCHEMA_MIGRATION.contains("agent_id"));
        assert!(SCHEMA_MIGRATION.contains("last_heartbeat"));
    }

    #[test]
    fn test_queries_exist() {
        assert!(!queries::LEADERBOARD.is_empty());
        assert!(!queries::ACTIVE_AGENTS.is_empty());
        assert!(!queries::BEST_BY_ARCH.is_empty());
    }
}
