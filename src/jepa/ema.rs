//! Exponential Moving Average (EMA) for target encoder
//!
//! Implements EMA decay schedule for JEPA target encoder updates.

/// EMA configuration
#[derive(Debug, Clone, Copy)]
pub struct EmaConfig {
    pub start: f64,  // Starting decay rate (0.996)
    pub end: f64,    // Target decay rate (1.0)
    pub ramp_steps: usize,  // Steps to reach end (30000)
}

impl Default for EmaConfig {
    fn default() -> Self {
        Self {
            start: 0.996,
            end: 1.0,
            ramp_steps: 30000,
        }
    }
}

/// EMA target encoder
///
/// Maintains a target encoder updated via EMA from the online encoder.
/// The decay rate linearly interpolates from start to end over ramp_steps.
pub struct EmaTarget {
    config: EmaConfig,
    step: usize,
}

impl EmaTarget {
    /// Create a new EMA target encoder
    pub fn new(config: EmaConfig) -> Self {
        Self { config, step: 0 }
    }

    /// Create with default configuration
    pub fn default_with_config() -> Self {
        Self::new(EmaConfig::default())
    }

    /// Get current decay rate based on current step
    pub fn decay(&self) -> f64 {
        if self.step >= self.config.ramp_steps {
            self.config.end
        } else {
            let progress = self.step as f64 / self.config.ramp_steps as f64;
            self.config.start + (self.config.end - self.config.start) * progress
        }
    }

    /// Update target parameters using EMA
    ///
    /// # Arguments
    /// * `target` - Target parameters to update (in-place)
    /// * `online` - Online encoder parameters
    pub fn update(&mut self, target: &mut [f32], online: &[f32]) {
        let decay = self.decay();
        ema_update(target, online, decay);
        self.step += 1;
    }

    /// Reset step counter (for new training run)
    pub fn reset(&mut self) {
        self.step = 0;
    }

    /// Get current step count
    pub fn step(&self) -> usize {
        self.step
    }
}

/// EMA update function (pure)
///
/// Implements: theta_target = decay * theta_target + (1 - decay) * theta_online
///
/// # Arguments
/// * `target` - Target parameters to update (in-place)
/// * `online` - Online encoder parameters
/// * `decay` - Decay rate (0.0 to 1.0, closer to 1.0 = slower update)
///
/// # Panics
/// Panics if target and online have different lengths
pub fn ema_update(target: &mut [f32], online: &[f32], decay: f64) {
    assert_eq!(
        target.len(),
        online.len(),
        "target and online must have same length"
    );

    let decay = decay as f32;
    let one_minus_decay = 1.0 - decay;

    for (t, o) in target.iter_mut().zip(online.iter()) {
        *t = decay * *t + one_minus_decay * *o;
    }
}

/// Compute EMA decay for a given step (pure function)
pub fn compute_decay(step: usize, ramp_steps: usize, start: f64, end: f64) -> f64 {
    if step >= ramp_steps {
        end
    } else {
        let progress = step as f64 / ramp_steps as f64;
        start + (end - start) * progress
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_decay_schedule() {
        let config = EmaConfig {
            start: 0.5,
            end: 1.0,
            ramp_steps: 100,
        };
        let mut ema = EmaTarget::new(config);

        assert_eq!(ema.decay(), 0.5);

        for _ in 0..50 {
            ema.step += 1;
        }
        // After 50 steps, should be at 0.75 (halfway)
        let decay_50 = ema.decay();
        assert!((decay_50 - 0.75).abs() < 0.01, "decay at step 50: {}", decay_50);

        for _ in 50..100 {
            ema.step += 1;
        }
        assert_eq!(ema.decay(), 1.0);
    }

    #[test]
    fn test_ema_decay_with_realistic_config() {
        let config = EmaConfig::default();
        let mut ema = EmaTarget::new(config);

        // At step 0: 0.996
        assert!((ema.decay() - 0.996).abs() < 1e-10);

        // At step 15000 (halfway): should be ~0.998
        ema.step = 15000;
        assert!((ema.decay() - 0.998).abs() < 0.001);

        // At step 30000+: 1.0
        ema.step = 30000;
        assert_eq!(ema.decay(), 1.0);

        ema.step = 50000;
        assert_eq!(ema.decay(), 1.0);
    }

    #[test]
    fn test_ema_update() {
        let mut target = vec![1.0_f32, 1.0_f32];
        let online = vec![2.0_f32, 0.0_f32];

        ema_update(&mut target, &online, 0.9);

        // target = 0.9 * 1.0 + 0.1 * online
        assert!((target[0] - 1.1).abs() < 1e-6);
        assert!((target[1] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_ema_update_converges() {
        let mut target = vec![10.0_f32];
        let online = vec![0.0_f32];

        // Use fixed decay for faster convergence in test
        for _ in 0..50 {
            ema_update(&mut target, &online, 0.9);
        }

        // Target should approach online (0.0)
        // 10.0 * 0.9^50 ≈ 0.005
        assert!(target[0].abs() < 0.1);
    }

    #[test]
    fn test_ema_reset() {
        let config = EmaConfig {
            start: 0.5,
            end: 1.0,
            ramp_steps: 100,
        };
        let mut ema = EmaTarget::new(config);

        ema.step = 50;
        assert_eq!(ema.step(), 50);

        ema.reset();
        assert_eq!(ema.step(), 0);
        assert_eq!(ema.decay(), 0.5);
    }

    #[test]
    fn test_compute_decay_pure() {
        let decay = compute_decay(0, 1000, 0.9, 1.0);
        assert_eq!(decay, 0.9);

        let decay = compute_decay(500, 1000, 0.9, 1.0);
        assert!((decay - 0.95).abs() < 0.001);

        let decay = compute_decay(1000, 1000, 0.9, 1.0);
        assert_eq!(decay, 1.0);

        let decay = compute_decay(2000, 1000, 0.9, 1.0);
        assert_eq!(decay, 1.0);
    }

    #[test]
    fn test_ema_update_different_lengths() {
        let mut target = vec![1.0_f32, 2.0_f32];
        let online = vec![3.0_f32];

        // Should panic due to length mismatch
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ema_update(&mut target, &online, 0.9);
        }));
        assert!(result.is_err());
    }

    #[test]
    fn test_ema_target_default() {
        let ema = EmaTarget::default_with_config();
        assert_eq!(ema.step(), 0);
        assert!((ema.decay() - 0.996).abs() < 1e-10);
    }

    #[test]
    fn test_ema_clips_to_end() {
        let config = EmaConfig {
            start: 0.8,
            end: 0.95,
            ramp_steps: 100,
        };
        let mut ema = EmaTarget::new(config);

        // Go past ramp_steps
        ema.step = 200;
        assert_eq!(ema.decay(), 0.95);
    }

    #[test]
    fn test_ema_update_preserves_length() {
        let mut target = vec![1.0_f32; 100];
        let online = vec![0.0_f32; 100];

        let original_len = target.len();
        ema_update(&mut target, &online, 0.9);

        assert_eq!(target.len(), original_len);
    }

    #[test]
    fn test_ema_high_decay() {
        let mut target = vec![5.0_f32];
        let online = vec![10.0_f32];

        // High decay = target changes slowly
        ema_update(&mut target, &online, 0.99);

        assert!(target[0] < 5.1); // Should barely move
        assert!(target[0] > 4.9);
    }

    #[test]
    fn test_ema_low_decay() {
        let mut target = vec![5.0_f32];
        let online = vec![10.0_f32];

        // Low decay = target changes quickly
        ema_update(&mut target, &online, 0.1);

        assert!(target[0] > 5.0);
        // With low decay, target should move significantly toward online
        assert!(target[0] > 9.0); // 0.1*5 + 0.9*10 = 9.5
    }
}
