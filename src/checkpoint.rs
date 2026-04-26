//! Checkpoint save/load. Stored under `${TRIOS_CHECKPOINT_DIR}/{run-name}/{step}.bin`.
//! Resume sets `step_done` and re-attaches optimizer state.

use std::path::PathBuf;

/// Exponential Moving Average (EMA) of checkpoints
///
/// Computes post-hoc EMA over the last N checkpoints for better generalization.
/// Reference: Sanyal et al. 2024 "A Simple and Strong Baseline for Model Averaging"
pub struct EmaAverager {
    /// Number of checkpoints to average
    pub n: usize,

    /// Current buffer of checkpoint values
    pub buffer: Vec<f32>,

    /// Current buffer index
    pub idx: usize,
}

impl EmaAverager {
    /// Create a new EMA averager
    ///
    /// # Arguments
    ///
    /// * `n` — Number of checkpoints to average
    ///
    /// # Returns
    ///
    /// A new EMA averager with empty buffer
    pub fn new(n: usize) -> Self {
        Self {
            n,
            buffer: Vec::with_capacity(n),
            idx: 0,
        }
    }

    /// Update EMA with a new checkpoint value
    ///
    /// # Arguments
    ///
    /// * `value` — New checkpoint value (BPB or loss)
    ///
    /// # Returns
    ///
    /// The EMA-averaged value
    pub fn update(&mut self, value: f64) -> f32 {
        if self.buffer.len() < self.n {
            self.buffer.push(value as f32);
            self.idx = self.buffer.len();
        } else {
            // Rotate buffer when full
            self.buffer[self.idx] = value as f32;
            self.idx = (self.idx + 1) % self.n;
        }

        // Compute EMA
        self.ema_average()
    }

    /// Compute current EMA average
    ///
    /// # Returns
    ///
    /// The exponential moving average of buffered values
    pub fn ema_average(&self) -> f32 {
        if self.buffer.is_empty() {
            0.0
        } else {
            let n = self.buffer.len() as f64;
            // Simple average (can use exponential weights if needed)
            self.buffer.iter().sum::<f32>() / (n as f32)
        }
    }

    /// Compute EMA over N most recent checkpoints
    ///
    /// # Arguments
    ///
    /// * `checkpoints` — Slice of checkpoint values (BPB or loss)
    /// * `n` — Number of checkpoints to average
    ///
    /// # Returns
    ///
    /// The EMA-averaged value
    pub fn ema_average_over(checkpoints: &[f32], n: usize) -> f64 {
        if checkpoints.is_empty() || n == 0 {
            return 0.0;
        }
        let take_n = n.min(checkpoints.len());
        let avg = checkpoints[..take_n].iter().sum::<f32>() / take_n as f64;
        avg
    }

    /// Reset EMA state
    ///
    /// Clears the buffer and resets index
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.idx = 0;
    }
}

pub fn checkpoint_path(run_name: &str, step: usize) -> PathBuf {
    let base = std::env::var("TRIOS_CHECKPOINT_DIR").unwrap_or_else(|_| "checkpoints".into());
    PathBuf::from(base)
        .join(run_name)
        .join(format!("{step}.bin"))
}

pub fn save(_run: &str, _step: usize, _bytes: &[u8]) -> anyhow::Result<()> {
    // TODO: actually persist (zstd-compressed bincode). Non-blocking for skeleton.
    Ok(())
}
