#![allow(
    clippy::type_complexity,
    clippy::redundant_closure,
    clippy::manual_range_contains,
    dead_code
)]
//! Multi-Objective Loss + ASHA Rung Schedules + NCA Auxiliary Loss.
//!
//! Migrated from `trios-train-cpu/src/objective.rs`.
//! L_total = 0.5*NTP + 0.25*JEPA + 0.25*NCA
//! NCA entropy band: [1.5, 2.8] (trinity NCA Wave 8.5)
//! JEPA ASHA Law L-R10: minimum 3000-step first rung

#[derive(Debug, Clone, Copy)]
pub struct ObjectiveWeights {
    pub ntp: f64,
    pub jepa: f64,
    pub nca: f64,
}

impl Default for ObjectiveWeights {
    fn default() -> Self {
        Self {
            ntp: 0.5,
            jepa: 0.25,
            nca: 0.25,
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

pub fn compute_combined_loss(
    components: ComponentLosses,
    weights: ObjectiveWeights,
) -> CombinedLoss {
    let total = components.ntp * weights.ntp
        + components.jepa * weights.jepa
        + components.nca * weights.nca;
    CombinedLoss { total, components }
}

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

pub fn get_rung_schedule(arch: &str) -> Vec<u32> {
    match arch {
        "jepa" => vec![3000, 9000, 27000],
        "attn" => vec![1000, 3000, 9000, 27000],
        "hybrid" => vec![2000, 6000, 18000],
        _ => vec![1000, 3000, 9000, 27000],
    }
}

pub fn should_skip_rung(arch: &str, rung: u32) -> bool {
    arch == "jepa" && rung < 3000
}

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

    pub fn rollout(&self, seed: u64, rule: &NcaTransitionRule) -> NcaRolloutResult {
        let mut state = self.init_grid(seed);
        let mut entropies = Vec::with_capacity(self.rollout_steps as usize);
        for _ in 0..self.rollout_steps {
            self.step(&mut state, rule);
            entropies.push(shannon_entropy(&state, self.k_states));
        }
        let final_entropy = entropies.last().copied().unwrap_or(0.0);
        NcaRolloutResult {
            final_state: state,
            entropies,
            final_entropy,
        }
    }

    pub fn compute_loss(&self, predicted: &[f32], target: &[f32], nca_state: &[f32]) -> f64 {
        let mse = mse_loss(predicted, target);
        let entropy = shannon_entropy(nca_state, self.k_states);
        let penalty = nca_entropy_constraint(entropy);
        self.weight * mse + penalty
    }
}

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

#[derive(Debug, Clone)]
pub struct NcaRolloutResult {
    pub final_state: Vec<f32>,
    pub entropies: Vec<f64>,
    pub final_entropy: f64,
}

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

pub fn cross_entropy_loss(logits: &[f32], targets: &[usize]) -> f64 {
    if logits.is_empty() || targets.is_empty() {
        return 0.0;
    }
    let vocab_size = logits.len() / targets.len();
    let mut total_loss = 0.0f32;
    for (batch, &target) in targets.iter().enumerate() {
        let offset = batch * vocab_size;
        let max_logit = logits[offset..offset + vocab_size]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum_exp = 0.0f32;
        for v in 0..vocab_size {
            sum_exp += (logits[offset + v] - max_logit).exp();
        }
        let log_prob = logits[offset + target] - max_logit - sum_exp.ln();
        total_loss -= log_prob;
    }
    (total_loss / targets.len() as f32) as f64
}

pub fn combined_loss(logits: &[f32], targets: &[usize]) -> f64 {
    cross_entropy_loss(logits, targets)
}

pub fn build(cfg: &crate::config::ObjectiveConfig) -> anyhow::Result<Objective> {
    Ok(Objective {
        w_ce: cfg.w_ce,
        w_jepa: cfg.w_jepa,
        w_nca: cfg.w_nca,
    })
}

pub fn build_fn(_cfg: &str) -> Box<dyn Fn(&[f32], &[usize]) -> f64> {
    Box::new(|logits, targets| cross_entropy_loss(logits, targets))
}

impl Objective {
    pub fn from_config(cfg: &crate::config::ObjectiveConfig) -> Self {
        Self {
            w_ce: cfg.w_ce,
            w_jepa: cfg.w_jepa,
            w_nca: cfg.w_nca,
        }
    }
}

pub struct Objective {
    pub w_ce: f64,
    pub w_jepa: f64,
    pub w_nca: f64,
}

pub fn build_from_config(cfg: &crate::config::ObjectiveConfig) -> Objective {
    Objective {
        w_ce: cfg.w_ce,
        w_jepa: cfg.w_jepa,
        w_nca: cfg.w_nca,
    }
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
        let r = compute_combined_loss(c, ObjectiveWeights::default());
        assert!((r.total - 1.375).abs() < 1e-9);
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
        assert_eq!(nca.grid_dim(), 9);
    }

    #[test]
    fn test_nca_init_grid() {
        let nca = NcaObjective::default();
        let grid = nca.init_grid(42);
        assert_eq!(grid.len(), 81);
        for &s in &grid {
            assert!(s >= 0.0 && s < 9.0);
        }
    }

    #[test]
    fn test_shannon_entropy_uniform() {
        let state: Vec<f32> = (0..81).map(|i| (i % 9) as f32).collect();
        let h = shannon_entropy(&state, 9);
        assert!((h - (9.0f64).ln()).abs() < 0.01);
    }

    #[test]
    fn test_shannon_entropy_single_state() {
        let h = shannon_entropy(&vec![0.0f32; 81], 9);
        assert!(h.abs() < 1e-9);
    }

    #[test]
    fn test_shannon_entropy_empty() {
        assert_eq!(shannon_entropy(&[], 9), 0.0);
    }

    #[test]
    fn test_nca_step_changes_state() {
        let nca = NcaObjective::default();
        let mut state = nca.init_grid(42);
        let initial = state.clone();
        nca.step(&mut state, &NcaTransitionRule::default());
        assert_ne!(state, initial);
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
    fn test_nca_compute_loss() {
        let nca = NcaObjective::default();
        let pred = vec![1.0f32; 81];
        let target = vec![0.0f32; 81];
        let state = nca.init_grid(42);
        let loss = nca.compute_loss(&pred, &target, &state);
        assert!(loss > 0.0 && loss.is_finite());
    }

    #[test]
    fn test_mse_loss() {
        assert_eq!(mse_loss(&[1.0_f32, 2.0, 3.0], &[1.0_f32, 2.0, 3.0]), 0.0);
        assert!((mse_loss(&[1.0_f32, 2.0, 3.0], &[0.0; 3]) - 14.0 / 3.0).abs() < 1e-9);
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
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_nca_entropy_loss_collapse() {
        let (loss, entropy) = nca_entropy_loss(&vec![0.0f32; 81], 9, 1.5, 2.8, 0.25);
        assert!(entropy.abs() < 1e-9);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_nca_entropy_loss_empty() {
        let (loss, entropy) = nca_entropy_loss(&[], 9, 1.5, 2.8, 0.25);
        assert_eq!(loss, 0.0);
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let logits = vec![10.0f32, 0.0, 0.0];
        let targets = vec![0usize];
        let loss = cross_entropy_loss(&logits, &targets);
        assert!(loss > 0.0 && loss < 0.1);
    }
}
