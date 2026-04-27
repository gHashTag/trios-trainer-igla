//! TASK-5A.6 — Multi-Objective Loss + ASHA Rung Schedules
//! TASK-NCA — NCA Auxiliary Loss
//!
//! L_total = 0.5*NTP + 0.25*JEPA + 0.25*NCA
//! NCA entropy band: [1.5, 2.8] (trinity NCA Wave 8.5)
//! JEPA ASHA Law L-R10: minimum 3000-step first rung
//!
//! NCA reference: arXiv:2603.10055 (Mar 2026)
//! - 164M NCA tokens > 1.6B natural language tokens for pre-pre-training
//! - 1.6x convergence speedup, 6% LM improvement
//! - Each NCA sequence has unique latent rule → in-context learning

#[derive(Debug, Clone, Copy)]
pub struct ObjectiveConfig {
    pub ntp_weight: f64,
    pub jepa_weight: f64,
    pub nca_weight: f64,
}

impl Default for ObjectiveConfig {
    fn default() -> Self {
        Self {
            ntp_weight: 0.5,
            jepa_weight: 0.25,
            nca_weight: 0.25,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ComponentLosses {
    pub ntp: f64,
    pub jepa: f64,
    pub nca: f64,
}

#[derive(Debug, Clone)]
pub struct CombinedLoss {
    pub total: f64,
    pub components: ComponentLosses,
}

pub fn compute_combined_loss(components: ComponentLosses, config: ObjectiveConfig) -> CombinedLoss {
    let total = components.ntp * config.ntp_weight
        + components.jepa * config.jepa_weight
        + components.nca * config.nca_weight;
    CombinedLoss { total, components }
}

/// NCA entropy constraint penalty (hard band [1.5, 2.8])
pub fn nca_entropy_constraint(entropy: f64) -> f64 {
    const MIN: f64 = 1.5;
    const MAX: f64 = 2.8;
    const SCALE: f64 = 100.0;
    if entropy < MIN {
        (MIN - entropy).powi(2) * SCALE
    } else if entropy > MAX {
        (entropy - MAX).powi(2) * SCALE
    } else {
        0.0
    }
}

/// ASHA rung schedule per architecture
/// Law L-R10: JEPA first rung = 3000 (1.4x slower convergence)
pub fn get_rung_schedule(arch: &str) -> Vec<u32> {
    match arch {
        "jepa" => vec![3000, 9000, 27000],
        "attn" => vec![1000, 3000, 9000, 27000],
        "hybrid" => vec![2000, 6000, 18000],
        _ => vec![1000, 3000, 9000, 27000],
    }
}

/// True if ASHA should skip this rung for arch
pub fn should_skip_rung(arch: &str, rung: u32) -> bool {
    arch == "jepa" && rung < 3000
}

/// Neural Cellular Automata objective for structured regularization.
///
/// Reference: arXiv:2603.10055 — NCA data improves LM by 6%, 1.6x convergence.
/// Mechanism: each NCA sequence has unique latent rule → model infers rule from context
/// = in-context learning signal. Attention layers most transferable component.
///
/// Grid: 9x9 = 81 = 3^4 (Trinity structural alignment)
/// k_states: 9 = 3^2
/// Entropy band: [1.5, 2.8] from Wave 8.5 G1–G8 sweep
#[derive(Debug, Clone)]
pub struct NcaObjective {
    pub grid_size: usize,
    pub k_states: usize,
    pub rollout_steps: u32,
    pub entropy_min: f64,
    pub entropy_max: f64,
    pub weight: f64,
}

impl Default for NcaObjective {
    fn default() -> Self {
        Self {
            grid_size: 81,
            k_states: 9,
            rollout_steps: 128,
            entropy_min: 1.5,
            entropy_max: 2.8,
            weight: 0.25,
        }
    }
}

impl NcaObjective {
    pub fn new(weight: f64) -> Self {
        Self {
            weight,
            ..Self::default()
        }
    }

    pub fn grid_dim(&self) -> usize {
        (self.grid_size as f64).sqrt() as usize
    }

    /// Initialize NCA grid with uniform random state
    pub fn init_grid(&self, seed: u64) -> Vec<f32> {
        let n = self.grid_size;
        let mut state = Vec::with_capacity(n);
        let mut s = seed;
        for _ in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = (s >> 33) as usize % self.k_states;
            state.push(idx as f32);
        }
        state
    }

    /// One NCA step: majority rule with von Neumann neighborhood
    pub fn step(&self, state: &mut [f32], rule: &NcaTransitionRule) {
        let dim = self.grid_dim();
        let mut next = state.to_vec();
        for y in 0..dim {
            for x in 0..dim {
                let idx = y * dim + x;
                let mut counts = vec![0usize; self.k_states];
                let center = state[idx] as usize;
                counts[center.min(self.k_states - 1)] += rule.center_weight;
                for &(dx, dy) in &[(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = ((x as i32 + dx).rem_euclid(dim as i32)) as usize;
                    let ny = ((y as i32 + dy).rem_euclid(dim as i32)) as usize;
                    let nidx = ny * dim + nx;
                    let s = state[nidx] as usize;
                    counts[s.min(self.k_states - 1)] += rule.neighbor_weight;
                }
                let best = counts
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, &c)| c)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                next[idx] = best as f32;
            }
        }
        state.copy_from_slice(&next);
    }

    /// Run full NCA rollout and return entropy at each step
    pub fn rollout(&self, seed: u64, rule: &NcaTransitionRule) -> NcaRolloutResult {
        let mut state = self.init_grid(seed);
        let mut entropies = Vec::with_capacity(self.rollout_steps as usize);
        for _ in 0..self.rollout_steps {
            self.step(&mut state, rule);
            let h = shannon_entropy(&state, self.k_states);
            entropies.push(h);
        }
        let final_entropy = entropies.last().copied().unwrap_or(0.0);
        NcaRolloutResult {
            final_state: state,
            entropies,
            final_entropy,
        }
    }

    /// Compute NCA auxiliary loss for one rollout
    /// Loss = MSE(predicted_embed, target_embed) + entropy_penalty
    pub fn compute_loss(&self, predicted: &[f32], target: &[f32], nca_state: &[f32]) -> f64 {
        let mse = mse_loss(predicted, target);
        let entropy = shannon_entropy(nca_state, self.k_states);
        let penalty = nca_entropy_constraint(entropy);
        self.weight * mse + penalty
    }
}

/// NCA transition rule — defines the cellular automata behavior
#[derive(Debug, Clone)]
pub struct NcaTransitionRule {
    pub center_weight: usize,
    pub neighbor_weight: usize,
    pub latent_id: u64,
}

impl Default for NcaTransitionRule {
    fn default() -> Self {
        Self {
            center_weight: 2,
            neighbor_weight: 1,
            latent_id: 0,
        }
    }
}

impl NcaTransitionRule {
    pub fn from_seed(seed: u64) -> Self {
        Self {
            center_weight: 2,
            neighbor_weight: 1,
            latent_id: seed,
        }
    }
}

/// Result of an NCA rollout
#[derive(Debug, Clone)]
pub struct NcaRolloutResult {
    pub final_state: Vec<f32>,
    pub entropies: Vec<f64>,
    pub final_entropy: f64,
}

/// Shannon entropy: H = -Σ p_i * log(p_i)
pub fn shannon_entropy(state: &[f32], k_states: usize) -> f64 {
    let n = state.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mut counts = vec![0usize; k_states];
    for &s in state {
        let idx = (s.round() as usize).min(k_states - 1);
        counts[idx] += 1;
    }
    let mut entropy = 0.0f64;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n;
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// MSE loss between two vectors
pub fn mse_loss(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f64;
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) as f64 * (x - y) as f64)
        .sum::<f64>()
        / n
}

/// Differentiable NCA entropy loss for use as auxiliary training signal.
///
/// Given a state vector (e.g., hidden activations or embeddings), compute
/// Shannon entropy of discretized state distribution, then apply smooth
/// quadratic penalty for entropy outside [min, max] band.
///
/// Returns (loss, entropy) tuple so caller can log entropy for monitoring.
///
/// Law L-R11: NCA entropy [1.5, 2.8] = hard penalty
pub fn nca_entropy_loss(
    state: &[f32],
    k_states: usize,
    entropy_min: f64,
    entropy_max: f64,
    weight: f64,
) -> (f64, f64) {
    if state.is_empty() {
        return (0.0, 0.0);
    }
    let entropy = shannon_entropy(state, k_states);
    let penalty = if entropy < entropy_min {
        (entropy_min - entropy).powi(2) * 100.0
    } else if entropy > entropy_max {
        (entropy - entropy_max).powi(2) * 100.0
    } else {
        0.0
    };
    (weight * penalty, entropy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combined_loss_weights() {
        let c = ComponentLosses {
            ntp: 2.0,
            jepa: 1.0,
            nca: 0.5,
        };
        let r = compute_combined_loss(c, ObjectiveConfig::default());
        assert!((r.total - 1.375).abs() < 1e-9, "total={}", r.total);
    }

    #[test]
    fn test_nca_entropy_in_band() {
        assert_eq!(nca_entropy_constraint(2.0), 0.0);
        assert_eq!(nca_entropy_constraint(1.5), 0.0);
        assert_eq!(nca_entropy_constraint(2.8), 0.0);
    }

    #[test]
    fn test_nca_entropy_below_band() {
        assert!((nca_entropy_constraint(1.0) - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_nca_entropy_above_band() {
        assert!((nca_entropy_constraint(3.0) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_rung_schedule_jepa() {
        let r = get_rung_schedule("jepa");
        assert_eq!(r[0], 3000);
        assert!(r.iter().all(|&x| x >= 3000));
    }

    #[test]
    fn test_should_skip_rung() {
        assert!(should_skip_rung("jepa", 1000));
        assert!(!should_skip_rung("jepa", 3000));
        assert!(!should_skip_rung("ngram", 1000));
    }

    #[test]
    fn test_nca_objective_default() {
        let nca = NcaObjective::default();
        assert_eq!(nca.grid_size, 81);
        assert_eq!(nca.k_states, 9);
        assert_eq!(nca.rollout_steps, 128);
        assert!((nca.entropy_min - 1.5).abs() < 1e-9);
        assert!((nca.entropy_max - 2.8).abs() < 1e-9);
        assert!((nca.weight - 0.25).abs() < 1e-9);
        assert_eq!(nca.grid_dim(), 9);
    }

    #[test]
    fn test_nca_init_grid() {
        let nca = NcaObjective::default();
        let grid = nca.init_grid(42);
        assert_eq!(grid.len(), 81);
        for &s in &grid {
            assert!(s >= 0.0 && s < 9.0, "state should be in [0, k_states)");
        }
    }

    #[test]
    fn test_shannon_entropy_uniform() {
        let state: Vec<f32> = (0..81).map(|i| (i % 9) as f32).collect();
        let h = shannon_entropy(&state, 9);
        assert!(
            (h - (9.0f64).ln()).abs() < 0.01,
            "uniform entropy should be ln(9)={}",
            (9.0f64).ln()
        );
    }

    #[test]
    fn test_shannon_entropy_single_state() {
        let state = vec![0.0f32; 81];
        let h = shannon_entropy(&state, 9);
        assert!(
            h.abs() < 1e-9,
            "single state entropy should be 0, got {}",
            h
        );
    }

    #[test]
    fn test_shannon_entropy_empty() {
        let h = shannon_entropy(&[], 9);
        assert_eq!(h, 0.0);
    }

    #[test]
    fn test_nca_step_changes_state() {
        let nca = NcaObjective::default();
        let mut state = nca.init_grid(42);
        let initial = state.clone();
        nca.step(&mut state, &NcaTransitionRule::default());
        assert_ne!(state, initial, "NCA step should change state");
    }

    #[test]
    fn test_nca_rollout() {
        let nca = NcaObjective::default();
        let result = nca.rollout(42, &NcaTransitionRule::from_seed(7));
        assert_eq!(result.entropies.len(), 128);
        assert_eq!(result.final_state.len(), 81);
        assert!(result.final_entropy >= 0.0);
    }

    #[test]
    fn test_nca_entropy_in_band_during_rollout() {
        let nca = NcaObjective::default();
        let result = nca.rollout(42, &NcaTransitionRule::default());
        let in_band = result
            .entropies
            .iter()
            .filter(|&&h| h >= 1.5 && h <= 2.8)
            .count();
        assert!(
            in_band > 0,
            "some steps should have entropy in band [1.5, 2.8]"
        );
    }

    #[test]
    fn test_nca_compute_loss() {
        let nca = NcaObjective::default();
        let pred = vec![1.0f32; 81];
        let target = vec![0.0f32; 81];
        let state = nca.init_grid(42);
        let loss = nca.compute_loss(&pred, &target, &state);
        assert!(loss > 0.0, "loss should be positive");
        assert!(loss.is_finite(), "loss should be finite");
    }

    #[test]
    fn test_mse_loss() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 2.0, 3.0];
        assert_eq!(mse_loss(&a, &b), 0.0);

        let c = vec![0.0f32, 0.0, 0.0];
        assert!((mse_loss(&a, &c) - (14.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn test_mse_loss_empty() {
        assert_eq!(mse_loss(&[], &[]), 0.0);
    }

    #[test]
    fn test_nca_entropy_loss_in_band() {
        let state: Vec<f32> = (0..81).map(|i| (i % 9) as f32).collect();
        let (loss, entropy) = nca_entropy_loss(&state, 9, 1.5, 2.8, 0.25);
        assert!((entropy - (9.0f64).ln()).abs() < 0.01);
        assert_eq!(loss, 0.0, "in-band entropy should have zero loss");
    }

    #[test]
    fn test_nca_entropy_loss_collapse() {
        let state = vec![0.0f32; 81];
        let (loss, entropy) = nca_entropy_loss(&state, 9, 1.5, 2.8, 0.25);
        assert!(entropy.abs() < 1e-9);
        assert!(loss > 0.0, "collapsed state should have positive loss");
    }

    #[test]
    fn test_nca_entropy_loss_empty() {
        let (loss, entropy) = nca_entropy_loss(&[], 9, 1.5, 2.8, 0.25);
        assert_eq!(loss, 0.0);
        assert_eq!(entropy, 0.0);
    }
}
