//! L11 — Worker pool (composite invariant: INV-1 ∧ INV-2 ∧ INV-12).
//!
//! Spawns N parallel workers, each running ASHA-rung trials. Each trial is
//! gated by `invariants::validate_config` (L-R14 enforcement), sampled with
//! `sampler::sample_lr` (φ-band), and traversed rung-by-rung via the
//! [`rungs::iter_rungs`] schedule (L10). Telemetry is drained through a
//! single-writer mutex to `data/igla_trials.csv`.
//!
//! ── Coq trail (L-R14) ────────────────────────────────────────────────────
//! Composite of three invariants — each constant pulled symbolically from
//! its owning module, never re-declared here:
//!
//!   INV-1 (LR sampling) — `crate::sampler` + `crate::invariants::inv1_check_lr`
//!     Coq: `trinity-clara/proofs/igla/lr_convergence.v::lr_phi_band`
//!   INV-2 (ASHA prune)  — `crate::invariants::INV2_BPB_PRUNE_THRESHOLD`
//!     Coq: `trinity-clara/proofs/igla/igla_asha_bound.v::champion_survives_pruning`
//!   INV-12 (rung schedule) — `crate::rungs::Rung::ALL`
//!     Coq: `trinity-clara/proofs/igla/igla_asha_bound.v::asha_rungs_trinity`
//!
//! ── Lane discipline (#143 R3, R6, R10) ───────────────────────────────────
//!   • New module file. The only edit outside is a one-line
//!     `pub mod race;` in `lib.rs`.
//!   • Touches NO other lane file (asha.rs, invariants.rs, sampler.rs,
//!     rungs.rs are read-only here).
//!   • Pure-Rust core: no Neon/Postgres, no async runtime — unit tests run
//!     fully offline. The `bin/race` orchestrator (out-of-scope for L11)
//!     can wire this pool into Neon later.
//!   • One atomic commit on main.
//!
//! ── Forbidden values (R7) — enforced by `validate_config`, never appear here:
//!     prune_threshold = 2.65   (INV-2)
//!     warmup < 4000            (INV-2)
//!     d_model < 256 with GF16  (INV-3)
//!     lr ∉ [0.002, 0.007]      (INV-1)

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::invariants::{
    validate_inv_config, GradientMode, InvError, InvTrialConfig, INV1_CHAMPION_LR,
    INV2_BPB_PRUNE_THRESHOLD, INV2_WARMUP_BLIND_STEPS, INV3_D_MODEL_MIN,
    INV4_NCA_GRID, INV4_NCA_K_STATES,
};
use crate::race::rungs::{iter_rungs, Rung};
use crate::race::sampler::sample_lr;

// ─── Public types ────────────────────────────────────────────────────────

/// Status of a finished trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrialStatus {
    /// Trial reached the last rung without falling below the victory band.
    Completed,
    /// Trial pruned by INV-2 threshold (`bpb > 3.5`) at any rung.
    Pruned,
    /// Trial reached `bpb < 1.5` (victory band).
    Victory,
    /// Trial rejected at the gate by `validate_config`.
    GateRejected,
}

impl TrialStatus {
    /// CSV-safe label.
    pub fn as_str(self) -> &'static str {
        match self {
            TrialStatus::Completed => "completed",
            TrialStatus::Pruned => "pruned",
            TrialStatus::Victory => "victory",
            TrialStatus::GateRejected => "gate_rejected",
        }
    }
}

/// One row of telemetry, drained to CSV.
#[derive(Debug, Clone)]
pub struct TrialRecord {
    pub worker_id: u32,
    pub trial_id: u64,
    pub lr: f64,
    pub d_model: usize,
    pub final_rung_step: u32,
    pub final_bpb: f64,
    pub status: TrialStatus,
}

/// Pool configuration. Fields are kept primitive so callers can build it
/// from CLI flags / env without pulling new deps.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Number of OS threads to spawn (≥ 1).
    pub workers: u32,
    /// Trials each worker runs before joining.
    pub trials_per_worker: u32,
    /// Path of the CSV telemetry sink. Parent dir is created if missing.
    pub telemetry_path: PathBuf,
    /// Base seed; worker `w` uses `base_seed.wrapping_add(w as u64)`.
    pub base_seed: u64,
    /// `d_model` value used by every trial. Must satisfy INV-3 if `use_gf16`.
    pub d_model: usize,
    /// Whether trials run in GF16 domain (toggles INV-3 / INV-5 gates).
    pub use_gf16: bool,
}

impl PoolConfig {
    /// Conservative defaults suitable for local smoke tests.
    pub fn conservative(telemetry_path: impl Into<PathBuf>) -> Self {
        Self {
            workers: 4,
            trials_per_worker: 8,
            telemetry_path: telemetry_path.into(),
            base_seed: 42,
            d_model: INV3_D_MODEL_MIN, // 256
            use_gf16: false,
        }
    }
}

/// Aggregated outcome of a pool run.
#[derive(Debug, Clone)]
pub struct PoolReport {
    pub trials_total: u64,
    pub trials_completed: u64,
    pub trials_pruned: u64,
    pub trials_victorious: u64,
    pub gate_rejections: u64,
    /// Lowest `final_bpb` observed across non-rejected trials (`f64::INFINITY` if none).
    pub best_bpb: f64,
}

// ─── Telemetry sink ──────────────────────────────────────────────────────

/// Single-writer telemetry sink. Cheap to clone (`Arc`).
#[derive(Clone)]
pub struct TelemetrySink {
    inner: Arc<Mutex<BufWriter<File>>>,
}

impl TelemetrySink {
    /// Open / create `path` and write the CSV header. Parent dir is created
    /// if missing. Existing file is **truncated** — pool runs are immutable
    /// snapshots; downstream consumers should rotate by renaming on success.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        let mut w = BufWriter::new(f);
        writeln!(
            w,
            "worker_id,trial_id,lr,d_model,final_rung_step,final_bpb,status"
        )?;
        w.flush()?;
        Ok(Self {
            inner: Arc::new(Mutex::new(w)),
        })
    }

    /// Append one row. Lock is held only for the duration of the write —
    /// workers contend at most for a single `writeln!` call.
    pub fn record(&self, r: &TrialRecord) -> std::io::Result<()> {
        let mut guard = self
            .inner
            .lock()
            .expect("telemetry sink mutex poisoned");
        writeln!(
            guard,
            "{},{},{:.9},{},{},{:.9},{}",
            r.worker_id,
            r.trial_id,
            r.lr,
            r.d_model,
            r.final_rung_step,
            r.final_bpb,
            r.status.as_str()
        )
    }

    /// Flush the buffered writer. Call once after all workers join.
    pub fn flush(&self) -> std::io::Result<()> {
        self.inner
            .lock()
            .expect("telemetry sink mutex poisoned")
            .flush()
    }
}

// ─── BPB simulator (deterministic, gradient-free) ────────────────────────

/// Deterministic BPB stand-in used by L11 so the pool exercises the full
/// rung pipeline without depending on a training backend. The curve
/// **decays toward the JEPA gate target** as rungs increase, with a
/// penalty proportional to `|lr − champion_lr|`.
///
/// This is a *pipeline* simulator (it tests pool plumbing, not learning).
/// The real BPB comes from `crates/tjepa-train/`. Exposed publicly so
/// downstream callers (and tests) can reason about the curve.
///
/// Curve:
///   bpb(rung) = 3.5 − 1.4 · (1 − e^(−rung/9000))   →   asymptote 2.1
///   penalty   = 30 · |lr − champion|
///   noise     = 0.05 · seed_to_unit(seed, rung)    (deterministic)
///
/// Bounded below by 0 to keep the CSV schema sane.
pub fn simulate_bpb(lr: f64, rung_step: u32, seed: u64) -> f64 {
    let decay = 1.0 - (-(rung_step as f64) / 9_000.0).exp();
    let base = 3.5 - 1.4 * decay;
    let penalty = 30.0 * (lr - INV1_CHAMPION_LR).abs();
    let noise = 0.05 * deterministic_unit(seed, rung_step);
    (base + penalty + noise).max(0.0)
}

/// Deterministic [0, 1) value from `(seed, rung)`. Avoids pulling rand for
/// the simulator so tests are fully reproducible without an RNG argument.
fn deterministic_unit(seed: u64, rung: u32) -> f64 {
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(rung as u64);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    // Top 53 bits → f64 in [0, 1).
    ((x >> 11) as f64) / ((1u64 << 53) as f64)
}

// ─── Trial / worker / pool ───────────────────────────────────────────────

/// Build a TrialConfig with conservative champion-side defaults plus a
/// freshly sampled LR. All non-LR knobs live inside the φ-safe envelope
/// so `validate_config` can never reject for non-LR reasons unless the
/// caller intentionally overrides via `PoolConfig`.
fn champion_config(lr: f64, d_model: usize, use_gf16: bool) -> InvTrialConfig {
    InvTrialConfig {
        lr,
        d_model,
        bpb_prune_threshold: INV2_BPB_PRUNE_THRESHOLD,
        warmup_blind_steps: INV2_WARMUP_BLIND_STEPS,
        use_gf16,
        nca_grid: INV4_NCA_GRID,
        nca_k_states: INV4_NCA_K_STATES,
        grad_mode: GradientMode::RealMSE,
        current_step: 0,
        last_bpb: f64::MAX,
    }
}

/// Run a single trial: validate at the gate, march through rungs, write
/// final telemetry. Returns the resulting record.
pub fn run_trial(
    worker_id: u32,
    trial_id: u64,
    cfg: &InvTrialConfig,
    seed: u64,
    sink: &TelemetrySink,
) -> std::io::Result<TrialRecord> {
    // Gate first (L-R14): a rejected config produces a single GateRejected row
    // and short-circuits — no rung simulation is wasted on a bad config.
    if let Err(_e) = validate_inv_config(cfg) {
        let rec = TrialRecord {
            worker_id,
            trial_id,
            lr: cfg.lr,
            d_model: cfg.d_model,
            final_rung_step: 0,
            final_bpb: f64::NAN,
            status: TrialStatus::GateRejected,
        };
        sink.record(&rec)?;
        return Ok(rec);
    }

    let mut last_step: u32 = 0;
    let mut last_bpb = f64::INFINITY;
    let mut status = TrialStatus::Completed;

    for (_rung, step) in iter_rungs() {
        let bpb = simulate_bpb(cfg.lr, step, seed);
        last_step = step;
        last_bpb = bpb;

        // Victory: take the first rung that crosses the gate (no need to keep training).
        if bpb < crate::invariants::BPB_VICTORY_TARGET {
            status = TrialStatus::Victory;
            break;
        }
        // Prune: INV-2 threshold (3.5) at first rung only, mirroring ASHA's
        // early-stop semantics. Higher rungs continue to record BPB.
        if step == Rung::ALL[0].step() && bpb > INV2_BPB_PRUNE_THRESHOLD {
            status = TrialStatus::Pruned;
            break;
        }
    }

    let rec = TrialRecord {
        worker_id,
        trial_id,
        lr: cfg.lr,
        d_model: cfg.d_model,
        final_rung_step: last_step,
        final_bpb: last_bpb,
        status,
    };
    sink.record(&rec)?;
    Ok(rec)
}

/// Run a single worker — `trials_per_worker` independent trials.
fn run_worker(
    worker_id: u32,
    base_seed: u64,
    pool: PoolConfig,
    sink: TelemetrySink,
) -> std::io::Result<Vec<TrialRecord>> {
    // Per-worker RNG seeded deterministically from base_seed.
    let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(worker_id as u64));
    let mut out = Vec::with_capacity(pool.trials_per_worker as usize);

    for t in 0..pool.trials_per_worker {
        let lr = sample_lr(&mut rng);
        let cfg = champion_config(lr, pool.d_model, pool.use_gf16);
        // Trial-local seed mixes worker_id + index so simulator stays reproducible.
        let trial_seed = base_seed
            .wrapping_add((worker_id as u64) << 32)
            .wrapping_add(t as u64);
        let trial_id = ((worker_id as u64) << 32) | (t as u64);
        let rec = run_trial(worker_id, trial_id, &cfg, trial_seed, &sink)?;
        out.push(rec);
    }
    Ok(out)
}

/// Worker pool. Drains telemetry through a single `TelemetrySink`; threads
/// only contend on the sink's writeln-sized critical section.
pub struct WorkerPool {
    cfg: PoolConfig,
    sink: TelemetrySink,
}

impl WorkerPool {
    /// Build a pool and open its telemetry sink. Returns the file system
    /// error verbatim — the binary entry point should surface it to the user.
    pub fn new(cfg: PoolConfig) -> std::io::Result<Self> {
        let sink = TelemetrySink::open(&cfg.telemetry_path)?;
        Ok(Self { cfg, sink })
    }

    /// Pre-flight check: validate the static, non-LR parts of the config
    /// once, before paying for thread spawn. LR varies per trial so it is
    /// re-validated inside `run_trial`.
    pub fn preflight(&self) -> Result<(), InvError> {
        // Use the champion LR purely so the LR slot is in-band; the worker
        // path will sample fresh LRs on each trial.
        let probe = champion_config(INV1_CHAMPION_LR, self.cfg.d_model, self.cfg.use_gf16);
        validate_inv_config(&probe)
    }

    /// Run all workers to completion, joining each. Telemetry is flushed
    /// before this returns. Aggregated counts come back in the report.
    pub fn run(self) -> std::io::Result<PoolReport> {
        let workers = self.cfg.workers.max(1);
        let mut handles = Vec::with_capacity(workers as usize);
        for w in 0..workers {
            let cfg = self.cfg.clone();
            let sink = self.sink.clone();
            let base_seed = self.cfg.base_seed;
            handles.push(thread::spawn(move || run_worker(w, base_seed, cfg, sink)));
        }

        let mut report = PoolReport {
            trials_total: 0,
            trials_completed: 0,
            trials_pruned: 0,
            trials_victorious: 0,
            gate_rejections: 0,
            best_bpb: f64::INFINITY,
        };
        for h in handles {
            let recs = h
                .join()
                .map_err(|_| std::io::Error::other("worker panicked"))?;
            let recs = recs?;
            for r in &recs {
                report.trials_total += 1;
                match r.status {
                    TrialStatus::Completed => report.trials_completed += 1,
                    TrialStatus::Pruned => report.trials_pruned += 1,
                    TrialStatus::Victory => report.trials_victorious += 1,
                    TrialStatus::GateRejected => report.gate_rejections += 1,
                }
                if r.status != TrialStatus::GateRejected && r.final_bpb < report.best_bpb {
                    report.best_bpb = r.final_bpb;
                }
            }
        }
        self.sink.flush()?;
        Ok(report)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        // Mix in pid + nanos so parallel `cargo test` workers don't stomp.
        let pid = std::process::id();
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("trios_l11_{}_{}_{}.csv", name, pid, ns));
        p
    }

    fn read_lines(path: &Path) -> Vec<String> {
        let f = File::open(path).expect("telemetry file missing");
        BufReader::new(f)
            .lines()
            .map(|l| l.expect("non-utf8 line"))
            .collect()
    }

    /// Header is exactly the documented schema. Tests stay glued to the
    /// CSV consumed by `crates/trios-igla-status/`.
    #[test]
    fn test_csv_header_is_canonical() {
        let path = tmp_path("hdr");
        let _sink = TelemetrySink::open(&path).expect("open sink");
        let lines = read_lines(&path);
        assert_eq!(lines.len(), 1);
        assert_eq!(
            lines[0],
            "worker_id,trial_id,lr,d_model,final_rung_step,final_bpb,status"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Pool size respected: with `workers=N, trials_per_worker=1` we get
    /// exactly N data rows plus the header row.
    #[test]
    fn test_pool_size_respected() {
        let path = tmp_path("size");
        let cfg = PoolConfig {
            workers: 4,
            trials_per_worker: 1,
            telemetry_path: path.clone(),
            base_seed: 7,
            d_model: 256,
            use_gf16: false,
        };
        let pool = WorkerPool::new(cfg).expect("pool");
        let report = pool.run().expect("run");
        assert_eq!(report.trials_total, 4);
        let lines = read_lines(&path);
        assert_eq!(lines.len(), 1 + 4); // header + 4 trials
        let _ = std::fs::remove_file(&path);
    }

    /// `validate_config` must reject forbidden LRs (R7). The trial is
    /// recorded as `gate_rejected`, NOT silently dropped.
    #[test]
    fn test_gate_rejects_forbidden_lr() {
        let path = tmp_path("gate");
        let sink = TelemetrySink::open(&path).expect("open sink");
        let bad = champion_config(0.0001, 256, false); // far below INV-1 lo
        assert!(validate_config(&bad).is_err());
        let rec = run_trial(0, 0, &bad, 0, &sink).expect("trial");
        assert_eq!(rec.status, TrialStatus::GateRejected);
        sink.flush().unwrap();
        let lines = read_lines(&path);
        assert_eq!(lines.len(), 1 + 1);
        assert!(lines[1].ends_with(",gate_rejected"));
        let _ = std::fs::remove_file(&path);
    }

    /// All four ASHA rungs must be visited in strict ascending order.
    /// Anchors the L11 ↔ L10 contract.
    #[test]
    fn test_rung_schedule_monotone() {
        let mut prev = 0u32;
        let mut count = 0usize;
        for (_r, step) in iter_rungs() {
            assert!(step > prev, "rung step must be strictly ascending");
            prev = step;
            count += 1;
        }
        assert_eq!(count, 4);
        assert_eq!(prev, 27_000);
    }

    /// Champion LR yields BPB consistent with the simulator's asymptote.
    /// Fails fast if someone ships a curve that diverges at the highest
    /// rung — the pipeline test, not the learning test.
    #[test]
    fn test_simulate_bpb_decays_at_high_rung() {
        let high = simulate_bpb(INV1_CHAMPION_LR, 27_000, 1);
        let low = simulate_bpb(INV1_CHAMPION_LR, 1_000, 1);
        assert!(high < low, "bpb must decrease as rungs grow");
        assert!(high < 2.5 && high > 1.5, "champion asymptote ≈ 2.1 ± noise");
    }

    /// Best-BPB aggregator. With four workers and conservative trials,
    /// every recorded BPB participates in the min.
    #[test]
    fn test_pool_best_bpb_is_min_of_records() {
        let path = tmp_path("best");
        let cfg = PoolConfig {
            workers: 4,
            trials_per_worker: 3,
            telemetry_path: path.clone(),
            base_seed: 11,
            d_model: 256,
            use_gf16: false,
        };
        let pool = WorkerPool::new(cfg).expect("pool");
        let report = pool.run().expect("run");
        let lines = read_lines(&path);
        // Pull every non-header BPB and recompute the min.
        let mut min = f64::INFINITY;
        for l in lines.iter().skip(1) {
            let cols: Vec<&str> = l.split(',').collect();
            if cols.last().copied() == Some("gate_rejected") {
                continue;
            }
            let bpb: f64 = cols[5].parse().unwrap();
            if bpb < min {
                min = bpb;
            }
        }
        // Tolerance is one ULP of the CSV precision (`{:.9}`): ≤ 1e-9 is safe.
        assert!(
            (report.best_bpb - min).abs() < 1e-8,
            "report.best_bpb {} != min from CSV {}",
            report.best_bpb,
            min
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Concurrency: 8 workers × 4 trials each — every CSV line must be
    /// well-formed (correct column count). Catches any partial-write race
    /// bug introduced if someone replaces the mutex with a non-atomic sink.
    #[test]
    fn test_csv_concurrent_writes_well_formed() {
        let path = tmp_path("concurrent");
        let cfg = PoolConfig {
            workers: 8,
            trials_per_worker: 4,
            telemetry_path: path.clone(),
            base_seed: 99,
            d_model: 256,
            use_gf16: false,
        };
        let pool = WorkerPool::new(cfg).expect("pool");
        let report = pool.run().expect("run");
        assert_eq!(report.trials_total, 32);
        let lines = read_lines(&path);
        assert_eq!(lines.len(), 1 + 32);
        for l in lines.iter() {
            assert_eq!(
                l.matches(',').count(),
                6,
                "row must have 7 columns (6 commas): {l}"
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    /// Preflight catches a configured pool that violates a static
    /// invariant (e.g. GF16 with d_model below INV-3 minimum).
    #[test]
    fn test_preflight_rejects_unsafe_gf16_domain() {
        let path = tmp_path("preflight");
        let cfg = PoolConfig {
            workers: 1,
            trials_per_worker: 1,
            telemetry_path: path.clone(),
            base_seed: 0,
            d_model: 128, // < INV3_D_MODEL_MIN with use_gf16=true
            use_gf16: true,
        };
        let pool = WorkerPool::new(cfg).expect("pool");
        assert!(matches!(
            pool.preflight(),
            Err(InvError::Inv3UnsafeDomain(128))
        ));
        let _ = std::fs::remove_file(&path);
    }

    /// Determinism: identical base_seed → identical record sequence.
    /// Requires deterministic worker scheduling, so we serialise
    /// (`workers=1`) and only compare the sequence shape.
    #[test]
    fn test_determinism_under_serial_workers() {
        let p1 = tmp_path("det1");
        let p2 = tmp_path("det2");
        for path in [&p1, &p2] {
            let cfg = PoolConfig {
                workers: 1,
                trials_per_worker: 5,
                telemetry_path: path.clone(),
                base_seed: 12345,
                d_model: 256,
                use_gf16: false,
            };
            let pool = WorkerPool::new(cfg).expect("pool");
            pool.run().expect("run");
        }
        let a = read_lines(&p1);
        let b = read_lines(&p2);
        assert_eq!(a, b, "identical base_seed must yield identical CSV");
        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
    }

    /// Forbidden-value witness (R7). Verifies the module never names
    /// `2.65`, `< 4000`, `< 256`, or LRs outside `[0.002, 0.007]` as
    /// constants. The presence of the right anchors is checked by the
    /// upstream invariants module's tests.
    #[test]
    fn test_forbidden_threshold_unreachable() {
        // The module re-exports INV2_BPB_PRUNE_THRESHOLD = 3.5 — the only
        // threshold it knows. There is no path that constructs `2.65`.
        assert!((INV2_BPB_PRUNE_THRESHOLD - 3.5).abs() < 1e-12);
        // The two below are compile-time constants; clippy flags them as
        // tautological asserts. We keep them as `const { assert!(..) }`
        // so the build itself fails if a future edit weakens INV-2/INV-3.
        const _: () = assert!(INV2_WARMUP_BLIND_STEPS >= 4_000);
        const _: () = assert!(INV3_D_MODEL_MIN >= 256);
        // Champion LR sits inside the safe band by construction.
        assert!(
            (0.002..=0.007).contains(&INV1_CHAMPION_LR),
            "champion LR must live in the φ-safe band"
        );
    }
}
