// SOVEREIGN SCARAB · pull-loop v4 library
// ----------------------------------------------------------------------------
// Public types + helpers shared by `main.rs` and integration tests.
// Anchor: phi^2 + phi^-2 = 3 · TRINITY · DATABASE_URL only · ADR-CHAT-012
// ----------------------------------------------------------------------------

use anyhow::{Context, Result};
use std::env;
use std::process::Child;
use std::time::{Duration, Instant};

/// Strategy row as fetched from `ssot.scarab_strategy`.
#[derive(Debug, Clone, PartialEq)]
pub struct Strategy {
    pub optimizer: String,
    pub format: String,
    pub hidden: i32,
    pub lr: f64,
    pub seed: i32,
    pub steps: i32,
    pub status: String,
    pub generation: i64,
}

impl Strategy {
    /// Stable SHA-256 hex fingerprint over the immutable hyperparameters.
    /// Must match `ssot.scarab_fingerprint()` byte-for-byte.
    pub fn fingerprint(&self) -> String {
        use sha2::{Digest, Sha256};
        let raw = format!(
            "{}|{}|{}|{}|{}|{}",
            self.optimizer, self.format, self.hidden, self.lr, self.seed, self.steps
        );
        let mut h = Sha256::new();
        h.update(raw.as_bytes());
        format!("{:x}", h.finalize())
    }
}

/// DONE event emitted by the trainer-stdout reader thread.
#[derive(Debug)]
pub struct DoneEvent {
    pub strategy: Strategy,
    pub final_bpb: Option<f64>,
    pub wall_s: i32,
}

/// Runtime configuration loaded from environment.
/// Both `SERVICE_ID` and `RAILWAY_SERVICE_ID` are accepted (Railway-first).
#[derive(Debug, Clone)]
pub struct Config {
    pub database_url: String,
    pub service_id: String,
    pub trainer_bin: String,
    pub train_data: String,
    pub val_data: String,
    pub poll_sec: u64,
    pub grace_ms: u64,
    pub dry_run: bool,
    pub auto_replay: bool,
    pub listen_notify: bool,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let database_url = env::var("DATABASE_URL").context("DATABASE_URL not set")?;
        let service_id = env::var("RAILWAY_SERVICE_ID")
            .or_else(|_| env::var("SERVICE_ID"))
            .context("neither RAILWAY_SERVICE_ID nor SERVICE_ID set")?;
        let trainer_bin = env::var("TRAINER_BIN").unwrap_or_else(|_| {
            "/usr/local/bin/trios-train".into()
        });
        let train_data = env::var("TRAIN_DATA").unwrap_or_else(|_| "/data/train.txt".into());
        let val_data = env::var("VAL_DATA").unwrap_or_else(|_| "/data/val.txt".into());
        let poll_sec: u64 = env::var("POLL_SEC")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);
        let grace_ms: u64 = env::var("GRACE_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10_000);
        let dry_run = env::var("DRY_RUN").ok().as_deref() == Some("1");
        let auto_replay = env::var("AUTO_REPLAY").ok().as_deref() == Some("1");
        // LISTEN/NOTIFY default ON; set LISTEN_NOTIFY=0 to fall back to pure-poll.
        let listen_notify = env::var("LISTEN_NOTIFY").ok().as_deref() != Some("0");
        Ok(Self {
            database_url,
            service_id,
            trainer_bin,
            train_data,
            val_data,
            poll_sec,
            grace_ms,
            dry_run,
            auto_replay,
            listen_notify,
        })
    }
}

/// Graceful kill: SIGTERM, wait up to `grace`, then SIGKILL.
pub fn graceful_kill(mut child: Child, grace: Duration) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let pid = child.id() as i32;
        unsafe {
            libc::kill(pid, libc::SIGTERM);
        }
        let t0 = Instant::now();
        loop {
            match child.try_wait()? {
                Some(_) => return Ok(()),
                None => {
                    if t0.elapsed() >= grace {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Ok(());
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = child.kill();
        let _ = child.wait();
        Ok(())
    }
}

/// Canonical name as written to `ssot.scarab_result.canon_name`.
pub fn canon_name(s: &Strategy, lane: &str) -> String {
    format!(
        "IGLA-{lane}-{}-h{}-LR{}-rng{}-{}",
        s.format, s.hidden, s.lr, s.seed, s.optimizer
    )
}

// ----------------------------------------------------------------------------
// Tests (cargo test) — pure-Rust, no DB required.
// ----------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn s() -> Strategy {
        Strategy {
            optimizer: "muon".into(),
            format: "gf16".into(),
            hidden: 256,
            lr: 0.005,
            seed: 2584,
            steps: 50_000,
            status: "active".into(),
            generation: 7,
        }
    }

    #[test]
    fn fingerprint_is_stable() {
        let s = s();
        assert_eq!(s.fingerprint(), s.fingerprint());
        assert_eq!(s.fingerprint().len(), 64); // sha256 hex
    }

    #[test]
    fn fingerprint_changes_on_any_hyperparam() {
        let a = s();
        let mut b = a.clone();
        b.lr = 0.006;
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn canon_name_format() {
        let n = canon_name(&s(), "RAILWAY");
        assert_eq!(n, "IGLA-RAILWAY-gf16-h256-LR0.005-rng2584-muon");
    }

    #[test]
    fn fingerprint_matches_sql_format() {
        // Format used by ssot.scarab_fingerprint():
        //   optimizer|format|hidden::text|lr::text|seed::text|steps::text
        // Numeric formatting must match Postgres ::text. For lr=0.005 Postgres
        // outputs "0.005"; for seed=2584 → "2584". Rust's default Display for
        // f64 produces "0.005" and i32 produces "2584", so they agree.
        let raw = "muon|gf16|256|0.005|2584|50000";
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(raw.as_bytes());
        let want = format!("{:x}", h.finalize());
        assert_eq!(s().fingerprint(), want);
    }
}
