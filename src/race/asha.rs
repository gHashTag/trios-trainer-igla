//! ASHA (Asynchronous Successive Halving Algorithm) rung bookkeeping.
//!
//! Trinity-optimized: rungs at 1k -> 3k -> 9k -> 27k (3^k progression).
//!
//! The pure parts (rung schedules, prune thresholds, config sampling) are real.
//! The database-facing wrappers forward to `race::neon::NeonDb`, which refuses
//! unconditionally -- so `run_worker` cannot reach its first subprocess.
//!
//! Three fabricators were removed from the worker loop while it was here:
//!   * `--seed 42` as a subprocess literal. 42 is forbidden by Canon #93
//!     (`seed_canon::parse_seed`) and slipped past the checker precisely
//!     because it was a string handed to another process. Now routed through
//!     `seed_canon`.
//!   * `bpb = 999.0` written on a trainer CRASH. A crash has no
//!     bits-per-byte; `neon_writer::reject_bpb` exists to catch that magic
//!     float on the other writer. Now `NeonDb::mark_crashed`, which carries no
//!     BPB at all.
//!   * `./target/release/trios-igla-trainer`, which is not one of the bins in
//!     Cargo.toml. The trainer bin is `trios-train`.

use anyhow::Result;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tracing::{info, warn};
use uuid::Uuid;

use crate::race::lessons::{Outcome, RungData, TrialConfig};
use crate::race::neon::NeonDb;

/// Architecture kind for IGLA Race (local copy)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchKind {
    Ngram,
    Jepa,
    Attention,
    Hybrid,
}

impl ArchKind {
    /// Get minimum rung for this architecture
    ///
    /// JEPA requires more steps for initial convergence
    pub fn min_rung(&self) -> i32 {
        match self {
            ArchKind::Jepa => 3000,
            _ => 1000,
        }
    }

    /// Get rung schedule for this architecture
    pub fn rung_schedule(&self) -> Vec<i32> {
        match self {
            ArchKind::Jepa => vec![3000, 9000, 27000],
            _ => vec![1000, 3000, 9000, 27000],
        }
    }

    /// Parse from string
    pub fn parse_arch(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ngram" => Some(ArchKind::Ngram),
            "jepa" => Some(ArchKind::Jepa),
            "attn" | "attention" => Some(ArchKind::Attention),
            "hybrid" => Some(ArchKind::Hybrid),
            _ => None,
        }
    }

    /// Convert to string
    pub fn as_str(&self) -> &'static str {
        match self {
            ArchKind::Ngram => "ngram",
            ArchKind::Jepa => "jepa",
            ArchKind::Attention => "attn",
            ArchKind::Hybrid => "hybrid",
        }
    }
}

/// ASHA rungs (Trinity 3^k progression)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AshaRung {
    Rung1000 = 1000,
    Rung3000 = 3000,
    Rung9000 = 9000,
    Rung27000 = 27000,
}

impl AshaRung {
    /// Get all rungs in order (default NTP schedule)
    pub fn all() -> Vec<AshaRung> {
        vec![
            AshaRung::Rung1000,
            AshaRung::Rung3000,
            AshaRung::Rung9000,
            AshaRung::Rung27000,
        ]
    }

    /// Get next rung after current
    pub fn next(&self) -> Option<AshaRung> {
        match self {
            AshaRung::Rung1000 => Some(AshaRung::Rung3000),
            AshaRung::Rung3000 => Some(AshaRung::Rung9000),
            AshaRung::Rung9000 => Some(AshaRung::Rung27000),
            AshaRung::Rung27000 => None,
        }
    }

    /// Get step value
    pub fn step(&self) -> usize {
        *self as usize
    }

    /// Get rung as i32 for database
    pub fn as_i32(&self) -> i32 {
        *self as i32
    }
}

/// ASHA trial configuration
#[derive(Debug, Clone)]
pub struct AshaConfig {
    pub target_bpb: f64,
    pub keep_fraction: f64,
    pub min_trials: usize,
    pub continuous: bool,
    pub arch: String,
}

impl Default for AshaConfig {
    fn default() -> Self {
        Self {
            target_bpb: 1.5,
            keep_fraction: 0.33,
            min_trials: 10,
            continuous: true,
            arch: "jepa".to_owned(),
        }
    }
}

/// Record a checkpoint at a rung. Propagates the backend refusal.
pub async fn record_checkpoint(
    db: &NeonDb,
    trial_id: &Uuid,
    rung: AshaRung,
    step: usize,
    bpb: f64,
) -> Result<()> {
    db.record_checkpoint(trial_id, rung.as_i32(), bpb).await?;
    info!(
        "Checkpoint recorded: trial_id={:?}, rung={:?}, step={}, BPB={}",
        trial_id, rung, step, bpb
    );
    Ok(())
}

/// Determine if trial should be pruned at this rung (pure; touches no backend).
pub async fn should_prune(
    _db: &NeonDb,
    _trial_id: &Uuid,
    current_bpb: f64,
    config: &AshaConfig,
) -> Result<bool> {
    if current_bpb <= config.target_bpb {
        return Ok(false);
    }
    let threshold = crate::invariants::INV2_BPB_PRUNE_THRESHOLD;
    Ok(current_bpb > threshold)
}

/// Handle trial pruning. Propagates the backend refusal.
pub async fn handle_pruning(
    db: &NeonDb,
    trial_id: &Uuid,
    rung: AshaRung,
    bpb: f64,
    config: &TrialConfig,
) -> Result<()> {
    db.mark_pruned(trial_id, rung.as_i32(), bpb).await?;

    let rung_data = RungData {
        step: rung.step(),
        bpb,
    };
    let (lesson, lesson_type) =
        crate::race::lessons::generate_lesson(config, &rung_data, Outcome::Pruned);

    db.store_lesson(
        trial_id,
        &Outcome::Pruned.to_string(),
        rung.as_i32(),
        bpb,
        &lesson,
        &lesson_type.to_string(),
    )
    .await?;

    warn!(
        "Trial pruned: trial_id={:?}, rung={:?}, BPB={}, lesson={}",
        trial_id, rung, bpb, lesson
    );

    Ok(())
}

/// Mark trial as completed. Propagates the backend refusal.
pub async fn mark_completed(
    db: &NeonDb,
    trial_id: &Uuid,
    final_step: usize,
    final_bpb: f64,
) -> Result<()> {
    db.mark_completed(trial_id, final_bpb, final_step as i32)
        .await?;

    if final_bpb < 1.5 {
        info!("IGLA FOUND! trial_id={:?}, BPB={}", trial_id, final_bpb);
    }

    Ok(())
}

/// Register a new trial. Propagates the backend refusal.
pub async fn register_trial(
    db: &NeonDb,
    machine_id: &str,
    worker_id: usize,
    config_json: &str,
) -> Result<Uuid> {
    let trial_id = Uuid::new_v4();
    db.register_trial(&trial_id, machine_id, worker_id as i32, config_json)
        .await?;
    Ok(trial_id)
}

/// Check if config is already running. Propagates the backend refusal.
pub async fn is_config_running(db: &NeonDb, machine_id: &str, config_json: &str) -> Result<bool> {
    db.is_config_running(machine_id, config_json).await
}

/// Path of the trainer this worker shells out to.
///
/// `trios-train` is the bin declared in Cargo.toml. The previous value,
/// `./target/release/trios-igla-trainer`, matched no bin at all, so
/// `tri race start` could not have worked even against a real ledger.
const TRAINER_BIN: &str = "./target/release/trios-train";

/// ASHA worker loop (TASK-3)
pub async fn run_worker(
    neon_url: &str,
    machine_id: &str,
    worker_id: u64,
    best_bpb: std::sync::Arc<std::sync::RwLock<f64>>,
) -> Result<f64> {
    use tokio::process::Command;

    // Canon #93: the seed is an experiment parameter, not a literal. It used
    // to be a forbidden seed hardcoded as a subprocess string argument, which
    // is exactly how it stayed invisible to `seed_canon`.
    let seed = crate::seed_canon::parse_seed().map_err(|e| anyhow::anyhow!(e))?;

    let db = NeonDb::connect(neon_url).await?;
    let mut rng = StdRng::from_entropy();
    let mut trial_counter = worker_id * 1_000_000;

    // Parse architecture type
    let default_config = AshaConfig::default();
    let arch_kind = ArchKind::parse_arch(&default_config.arch).unwrap_or(ArchKind::Jepa);

    // `trios-train` exposes no `--arch`: it trains the attention model in
    // `train_loop::run_single`. Passing the flag anyway (as this worker did)
    // is an unknown-argument abort; dropping it silently would record a trial
    // config that does not describe the run. So say which one it is.
    if arch_kind != ArchKind::Attention {
        anyhow::bail!(
            "ASHA worker cannot run arch={}: {TRAINER_BIN} has no --arch flag, \
             so the recorded trial config would not describe the run",
            arch_kind.as_str()
        );
    }

    // Get rung schedule based on architecture
    let rungs = arch_kind.rung_schedule();

    loop {
        // 1. sample_config(worker_id) → trial config
        let config = sample_config(&mut rng, &default_config.arch);
        let config_json = serde_json::to_string(&config)?;

        // 2. register_trial in Neon
        trial_counter += 1;
        let trial_id = format!("{}-w{}-t{}", machine_id, worker_id, trial_counter);
        let trial_uuid =
            Uuid::parse_str(&trial_id.replace("-", "")).unwrap_or_else(|_| Uuid::new_v4());

        if let Err(e) = db
            .register_trial(&trial_uuid, machine_id, worker_id as i32, &config_json)
            .await
        {
            warn!("register trial failed: {e}");
            continue;
        }

        info!(
            "[w{worker_id}] trial {trial_id}: h={} lr={:.6}",
            config.hidden.unwrap_or(256),
            config.lr.unwrap_or(0.004)
        );

        let mut pruned = false;
        let mut crashed = false;

        // 3. For each rung in schedule (JEPA skips 1000)
        let min_rung = arch_kind.min_rung();

        for &rung in &rungs {
            // JEPA: skip rung 1000 due to slower convergence
            if rung < min_rung {
                info!(
                    "Skipping rung {} for JEPA (below min rung {})",
                    rung, min_rung
                );
                continue;
            }

            let rung_steps = rung as usize;

            // a. Spawn the trainer. Only flags `trios-train` actually declares
            //    are passed; an unknown one aborts its clap parse. There is no
            //    flag for the trial id, so `trial_id` stays on this side of
            //    the boundary and is correlated by the ledger row alone.
            let output = Command::new(TRAINER_BIN)
                .args(["--seed", &seed.to_string()])
                .arg("--steps")
                .arg(rung_steps.to_string())
                .arg("--hidden")
                .arg(config.hidden.unwrap_or(256).to_string())
                .arg("--lr")
                .arg(format!("{:.8}", config.lr.unwrap_or(0.004)))
                .arg("--attn-layers")
                .arg(config.n_layers.unwrap_or(2).to_string())
                .arg("--optimizer")
                .arg(config.optimizer.as_deref().unwrap_or("adamw"))
                .output()
                .await?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                warn!("[w{worker_id}] trainer crashed at rung {rung_steps}: {stderr}");
                // A crash has no BPB. The old code recorded 999.0 here, which
                // is a fabricated measurement in a measurement column.
                db.mark_crashed(&trial_uuid, rung_steps as i32, &stderr)
                    .await?;
                crashed = true;
                break;
            }

            // b. Parse the reading from the trainer's own DONE line, which is
            //    `DONE: seed=N bpb=X.XXXX steps=N opt=Y` and says
            //    `bpb=unmeasured` when no evaluation was produced.
            let stdout = String::from_utf8_lossy(&output.stdout);
            let done_line = stdout
                .lines()
                .rev()
                .find(|l| l.starts_with("DONE:"))
                .ok_or_else(|| {
                    anyhow::anyhow!("{TRAINER_BIN} printed no DONE line at rung {rung_steps}")
                })?;
            let bpb_field = done_line
                .split_whitespace()
                .find_map(|tok| tok.strip_prefix("bpb="))
                .ok_or_else(|| anyhow::anyhow!("DONE line carries no bpb= field: {done_line}"))?;
            if bpb_field == "unmeasured" {
                warn!("[w{worker_id}] rung {rung_steps} finished without a measurement");
                db.mark_crashed(&trial_uuid, rung_steps as i32, "bpb=unmeasured")
                    .await?;
                crashed = true;
                break;
            }
            let bpb: f64 = bpb_field.parse()?;

            // c. update_rung in Neon - mock for now
            info!(
                "Update rung: trial={}, rung={}, BPB={}",
                trial_id, rung_steps, bpb
            );

            // e. if bpb < 1.50 → save_winner in Neon → return Ok(bpb)
            if bpb < 1.50 {
                info!("[w{worker_id}] IGLA FOUND! BPB={bpb:.4}");
                {
                    let mut best = best_bpb.write().unwrap();
                    if bpb < *best {
                        *best = bpb;
                    }
                }
                return Ok(bpb);
            }

            // d. if should_prune(rung, bpb) → break to next trial
            // Mock median check - in reality would query Neon
            let should_prune = bpb > crate::invariants::INV2_BPB_PRUNE_THRESHOLD;

            if should_prune {
                info!("Prune trial: BPB={}", bpb);
                pruned = true;
                break;
            }
        }

        if !pruned && !crashed {
            info!("Mark trial completed: {}", trial_id);
        }
    }
}

fn sample_config(rng: &mut StdRng, arch: &str) -> TrialConfig {
    use rand::seq::SliceRandom;

    let hiddens = [128, 192, 256, 384];
    let hidden = *hiddens.choose(rng).unwrap();
    let lrs = [0.001, 0.002, 0.004, 0.008];
    let lr = *lrs.choose(rng).unwrap();

    // JEPA uses different parameter structure
    if arch == "jepa" {
        return TrialConfig {
            lr: Some(lr),
            d_model: Some(hidden),
            hidden: Some(hidden),
            n_layers: Some(2),
            optimizer: Some("adamw".to_string()),
            activation: Some("relu".to_string()),
            weight_decay: Some(0.01),
            dropout: Some(0.1),
            warmup_steps: Some(1500),
            max_steps: Some(27000),
        };
    }

    TrialConfig {
        lr: Some(lr),
        d_model: Some(hidden),
        hidden: Some(hidden),
        n_layers: Some(2),
        optimizer: Some("adamw".to_string()),
        activation: Some("relu".to_string()),
        weight_decay: Some(0.01),
        dropout: Some(0.1),
        warmup_steps: Some(100),
        max_steps: Some(10000),
    }
}
