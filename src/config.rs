//! TrainConfig — single TOML schema for every training run.
//!
//! Loaded from `--config <path>` (CLI), then overridden by env vars with the
//! `TRIOS_` prefix (see [`TrainConfig::from_env_overrides`]). This is what
//! makes the same binary reproducible on any machine and on Railway.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Logical run name (e.g. "champion", "gate2-attempt-2", "needle-v1-mup").
    pub name: String,
    /// Number of optimizer steps. Must be ≥ 4000 for a Gate-2 candidate row.
    pub steps: usize,
    /// Random seed. Gate-2 victory needs ≥ 3 distinct seeds (43, 44, 45 by default).
    pub seed: u64,
    /// Target BPB for victory check.
    pub target_bpb: f64,
    /// Champion baseline BPB to guard against regression. If `None`, no guard.
    pub champion_bpb: Option<f64>,

    pub model: ModelConfig,
    pub optimizer: OptimizerConfig,
    pub data: DataConfig,
    pub objective: ObjectiveConfig,
    pub ledger: LedgerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    /// Use HybridAttn (L-h2) instead of plain causal attention.
    #[serde(default)]
    pub hybrid_attn: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// "adamw" | "muon" | "muon+adamw"
    pub kind: String,
    /// φ-anchored: lr = α_φ / φ³ ≈ 0.004 (INV-8). Validated at runtime.
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub weight_decay: f64,
    /// "phi" (φ-LR) | "cosine" | "flat"
    pub schedule: String,
    pub warmup_steps: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub corpus: String, // "fineweb" | "tinyshakespeare" | "wikitext-103"
    pub train_path: String, // Path to training data (supports absolute or relative to workdir)
    pub val_path: String,   // Path to validation data
    pub batch_size: usize,
    pub batch_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveConfig {
    /// CE weight (cross-entropy on next-token).
    pub w_ce: f64,
    /// JEPA weight (T-JEPA proxy loss).
    pub w_jepa: f64,
    /// NCA entropy regulariser (INV-4).
    pub w_nca: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerConfig {
    /// Path to assertions/seed_results.jsonl relative to repo root.
    pub jsonl_path: String,
    /// Push to git after each row?
    pub push: bool,
    /// Embargo list path. Hard-block if HEAD SHA matches.
    pub embargo_path: String,
}

impl TrainConfig {
    pub fn from_toml<P: AsRef<Path>>(path: P) -> Result<Self> {
        let txt = std::fs::read_to_string(&path)
            .with_context(|| format!("read config {}", path.as_ref().display()))?;
        let mut cfg: TrainConfig = toml::from_str(&txt)
            .with_context(|| format!("parse TOML {}", path.as_ref().display()))?;
        cfg.apply_env_overrides();
        cfg.validate()?;
        Ok(cfg)
    }

    /// Override with `TRIOS_*` env vars. Lets Railway/Docker re-parameterise
    /// without rebuilding the image.
    pub fn apply_env_overrides(&mut self) {
        if let Ok(s) = std::env::var("TRIOS_SEED") {
            if let Ok(v) = s.parse() {
                self.seed = v;
            }
        }
        if let Ok(s) = std::env::var("TRIOS_STEPS") {
            if let Ok(v) = s.parse() {
                self.steps = v;
            }
        }
        if let Ok(s) = std::env::var("TRIOS_TARGET_BPB") {
            if let Ok(v) = s.parse() {
                self.target_bpb = v;
            }
        }
        if let Ok(s) = std::env::var("TRIOS_LR") {
            if let Ok(v) = s.parse() {
                self.optimizer.lr = v;
            }
        }
        if let Ok(s) = std::env::var("TRIOS_LEDGER_PUSH") {
            self.ledger.push = matches!(s.as_str(), "1" | "true" | "yes");
        }
        if let Ok(s) = std::env::var("TRIOS_TRAIN_PATH") {
            self.data.train_path = s;
        }
        if let Ok(s) = std::env::var("TRIOS_VAL_PATH") {
            self.data.val_path = s;
        }
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.steps >= 1, "steps must be ≥ 1");
        anyhow::ensure!(self.target_bpb > 0.0, "target_bpb must be positive");
        // INV-8 (φ-band): lr ∈ [1e-3, 1e-2]
        anyhow::ensure!(
            (1e-3..=1e-2).contains(&self.optimizer.lr),
            "lr {} violates INV-8 φ-band [1e-3, 1e-2]",
            self.optimizer.lr
        );
        Ok(())
    }
}
