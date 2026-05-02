//! Trinity Experiments — L0: Single Source of Truth
//!
//! # Constitutional mandate (Law 2)
//!
//! `strategy_experiments` is the **only** table.
//! All other state is derived:
//! - bpb_curve → JSONB embedded in experiment row
//! - worker heartbeat → validated via worker_id timestamp
//! - historical ledger → git commits + GitHub issues
//!
//! # PR-O3 status
//!
//! - [ ] schema.rs — Experiment struct
//! - [ ] repo.rs — ExperimentRepo trait
//! - [ ] migration.rs — 0001_initial.sql (immutable)
//! - [ ] e2e test with Neon test DB
//!
//! 🌻 φ² + φ⁻² = 3 · TRINITY · O(1) FOREVER

pub mod schema;
pub mod repo;
pub mod migration;

pub use schema::{Experiment, ExperimentStatus, ExperimentConfig, BpbPoint};
pub use repo::{ExperimentRepo, PendingClaim, ClaimResult, PostgresExperimentRepo};
pub use migration::MIGRATION_0001;
