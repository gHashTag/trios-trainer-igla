//! Optimizer for IGLA-GF16
//!
//! AdamW + Muon + φ-LR schedule. Migrated from `trios-train-cpu/src/optimizer.rs`.

use anyhow::{Result, bail};
use crate::config::OptimizerConfig;

#[derive(Debug, Clone)]
pub struct AdamWCpu {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub weight_decay: f64,
    pub eps: f64,
    step: usize,
    m: Vec<f64>,
    v: Vec<f64>,
}

impl AdamWCpu {
    pub fn new(param_count: usize, lr: f64) -> Self {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        Self {
            lr,
            beta1: 1.0 / phi,
            beta2: 0.999,
            weight_decay: 1.0 / (phi * phi * phi),
            eps: 1e-8,
            step: 0,
            m: vec![0.0; param_count],
            v: vec![0.0; param_count],
        }
    }

    pub fn with_phi_defaults(param_count: usize) -> Self {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        Self::new(param_count, 1.0 / (phi * phi * phi))
    }

    pub fn with_params(param_count: usize, lr: f64, beta1: f64, beta2: f64, weight_decay: f64) -> Self {
        Self { lr, beta1, beta2, weight_decay, eps: 1e-8, step: 0, m: vec![0.0; param_count], v: vec![0.0; param_count] }
    }

    pub fn step(&mut self, params: &mut [f32], gradients: &[f32]) {
        assert_eq!(params.len(), gradients.len(), "params and gradients must have same length");
        assert_eq!(params.len(), self.m.len(), "parameter count mismatch with optimizer state");

        self.step += 1;

        let bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);
        let step_size = self.lr * bias_correction2.sqrt() / bias_correction1;

        for i in 0..params.len() {
            params[i] -= self.weight_decay as f32 * params[i];

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * gradients[i] as f64;

            self.v[i] =
                self.beta2 * self.v[i] + (1.0 - self.beta2) * (gradients[i] * gradients[i]) as f64;

            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;

            params[i] -=
                step_size as f32 * (m_hat as f32 / ((v_hat.sqrt() as f32) + self.eps as f32));
        }
    }

    pub fn reset(&mut self) {
        self.step = 0;
        self.m.fill(0.0f64);
        self.v.fill(0.0f64);
    }

    pub fn step_count(&self) -> usize {
        self.step
    }
}

#[derive(Debug, Clone)]
pub struct SGDMomentum {
    pub lr: f64,
    pub momentum: f64,
    step: usize,
    velocity: Vec<f32>,
}

impl SGDMomentum {
    pub fn new(param_count: usize, lr: f64, momentum: f64) -> Self {
        Self {
            lr,
            momentum,
            step: 0,
            velocity: vec![0.0; param_count],
        }
    }

    pub fn step(&mut self, params: &mut [f32], gradients: &[f32]) {
        assert_eq!(params.len(), gradients.len());

        self.step += 1;

        for i in 0..params.len() {
            self.velocity[i] =
                self.momentum as f32 * self.velocity[i] - self.lr as f32 * gradients[i];

            params[i] += self.velocity[i];
        }
    }

    pub fn step_count(&self) -> usize {
        self.step
    }
}

#[derive(Debug, Clone)]
pub struct MuonOptimizer {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub ns_steps: usize,
    pub nesterov: bool,
    pub ns_a: f32,
    pub ns_b: f32,
    pub ns_c: f32,
    step: usize,
    momentum_buffer: Vec<f32>,
    param_rows: usize,
    param_cols: usize,
}

impl MuonOptimizer {
    pub fn new(param_count: usize, lr: f64, momentum: f64, weight_decay: f64) -> Self {
        let cols = (param_count as f64).sqrt().round() as usize;
        let cols = cols.max(1);
        let rows = (param_count as f64 / cols as f64).ceil() as usize;
        Self {
            lr,
            momentum,
            weight_decay,
            ns_steps: 5,
            nesterov: true,
            ns_a: 3.4445,
            ns_b: -4.7750,
            ns_c: 2.0315,
            step: 0,
            momentum_buffer: vec![0.0; param_count],
            param_rows: rows,
            param_cols: cols,
        }
    }

    pub fn with_matrix_shape(
        param_count: usize,
        rows: usize,
        cols: usize,
        lr: f64,
        momentum: f64,
        weight_decay: f64,
    ) -> Self {
        assert!(rows * cols >= param_count);
        Self {
            lr,
            momentum,
            weight_decay,
            ns_steps: 5,
            nesterov: true,
            ns_a: 3.4445,
            ns_b: -4.7750,
            ns_c: 2.0315,
            step: 0,
            momentum_buffer: vec![0.0; param_count],
            param_rows: rows,
            param_cols: cols,
        }
    }

    pub fn with_ns_coefficients(mut self, a: f32, b: f32, c: f32) -> Self {
        self.ns_a = a;
        self.ns_b = b;
        self.ns_c = c;
        self
    }

    pub fn step(&mut self, params: &mut [f32], gradients: &[f32]) {
        assert_eq!(params.len(), gradients.len());
        assert_eq!(params.len(), self.momentum_buffer.len());
        self.step += 1;

        let lr = self.lr as f32;
        let mom = self.momentum as f32;
        let wd = self.weight_decay as f32;
        let n = params.len();

        for p in params.iter_mut() {
            *p *= 1.0 - lr * wd;
        }

        for i in 0..n {
            self.momentum_buffer[i] = mom * self.momentum_buffer[i] + (1.0 - mom) * gradients[i];
        }

        let update = self.orthogonalize_update();

        for i in 0..n {
            params[i] -= lr * update[i];
        }
    }

    fn orthogonalize_update(&self) -> Vec<f32> {
        let n = self.momentum_buffer.len();
        let rows = self.param_rows;
        let cols = self.param_cols;
        let matrix_size = rows * cols;

        if matrix_size == 0 || rows < 2 || cols < 2 {
            return self.momentum_buffer.clone();
        }

        let mut m = vec![0.0f32; matrix_size];
        let copy_len = n.min(matrix_size);
        m[..copy_len].copy_from_slice(&self.momentum_buffer[..copy_len]);

        let norm = frobenius_norm(&m);
        if norm < 1e-8 {
            return self.momentum_buffer.clone();
        }

        let scale = 1.0 / norm;
        for v in m.iter_mut() {
            *v *= scale;
        }

        for _ in 0..self.ns_steps {
            m = newton_schulz_5(&m, rows, cols, self.ns_a, self.ns_b, self.ns_c);
        }

        let out_norm = frobenius_norm(&m);
        if out_norm > 1e-8 {
            let rescale = norm / out_norm;
            for v in m.iter_mut() {
                *v *= rescale;
            }
        }

        let mut result = vec![0.0f32; n];
        let copy_len = n.min(matrix_size);
        result[..copy_len].copy_from_slice(&m[..copy_len]);
        result
    }

    pub fn step_count(&self) -> usize {
        self.step
    }

    pub fn reset(&mut self) {
        self.step = 0;
        self.momentum_buffer.fill(0.0);
    }
}

fn frobenius_norm(m: &[f32]) -> f32 {
    m.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8)
}

fn newton_schulz_5(m: &[f32], rows: usize, cols: usize, a: f32, b: f32, c: f32) -> Vec<f32> {
    let mut mt_m = vec![0.0f32; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            let mut s = 0.0f32;
            for k in 0..rows {
                s += m[k * cols + i] * m[k * cols + j];
            }
            mt_m[i * cols + j] = s;
        }
    }

    let mut m_mt_m = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let mut s = 0.0f32;
            for k in 0..cols {
                s += m[i * cols + k] * mt_m[k * cols + j];
            }
            m_mt_m[i * cols + j] = s;
        }
    }

    let mut mt_m2 = vec![0.0f32; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            let mut s = 0.0f32;
            for k in 0..cols {
                s += mt_m[i * cols + k] * mt_m[k * cols + j];
            }
            mt_m2[i * cols + j] = s;
        }
    }

    let mut m_mt_m2 = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let mut s = 0.0f32;
            for k in 0..cols {
                s += m[i * cols + k] * mt_m2[k * cols + j];
            }
            m_mt_m2[i * cols + j] = s;
        }
    }

    let mut result = vec![0.0f32; rows * cols];
    for i in 0..(rows * cols) {
        result[i] = a * m[i] + b * m_mt_m[i] + c * m_mt_m2[i];
    }
    result
}

#[allow(dead_code)]
fn newton_schulz_cubic(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut mt_m = vec![0.0f32; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            let mut s = 0.0f32;
            for k in 0..rows {
                s += m[k * cols + i] * m[k * cols + j];
            }
            mt_m[i * cols + j] = s;
        }
    }

    let mut m_mt_m = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let mut s = 0.0f32;
            for k in 0..cols {
                s += m[i * cols + k] * mt_m[k * cols + j];
            }
            m_mt_m[i * cols + j] = s;
        }
    }

    let mut result = vec![0.0f32; rows * cols];
    for i in 0..(rows * cols) {
        result[i] = 1.5 * m[i] - 0.5 * m_mt_m[i];
    }
    result
}

pub enum OptimizerKind {
    AdamW(AdamWCpu),
    Muon(MuonOptimizer),
}

impl OptimizerKind {
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        match self {
            OptimizerKind::AdamW(opt) => opt.step(params, grads),
            OptimizerKind::Muon(opt) => opt.step(params, grads),
        }
    }

    pub fn reset(&mut self) {
        match self {
            OptimizerKind::AdamW(opt) => opt.reset(),
            OptimizerKind::Muon(opt) => opt.reset(),
        }
    }
}

pub fn phi_lr_schedule(step: usize, base_lr: f64, warmup_steps: usize) -> f64 {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;

    if step < warmup_steps {
        base_lr * (step as f64 / warmup_steps as f64)
    } else {
        let decay_steps = (step - warmup_steps) as f64 / warmup_steps as f64;
        base_lr * phi.powf(-decay_steps)
    }
}

pub struct Optimizer {
    pub kind: String,
    pub lr: f64,
}

pub fn build(cfg: &OptimizerConfig) -> Result<Optimizer> {
    match cfg.kind.as_str() {
        "adamw" | "muon" | "muon+adamw" => {}
        other => bail!("unknown optimizer kind: {other}"),
    }
    Ok(Optimizer { kind: cfg.kind.clone(), lr: cfg.lr })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phi() -> f64 {
        (1.0 + 5.0_f64.sqrt()) / 2.0
    }

    #[test]
    fn test_adamw_phi_defaults() {
        let optimizer = AdamWCpu::with_phi_defaults(100);
        let expected_beta1 = 1.0 / phi();
        let expected_weight_decay = 1.0 / (phi() * phi() * phi());
        assert!((optimizer.beta1 - expected_beta1).abs() < 1e-6);
        assert!((optimizer.weight_decay - expected_weight_decay).abs() < 1e-6);
    }

    #[test]
    fn test_adamw_custom_params() {
        let optimizer = AdamWCpu::with_params(100, 0.001, 0.9, 0.999, 0.01);
        assert_eq!(optimizer.lr, 0.001);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);
        assert_eq!(optimizer.weight_decay, 0.01);
    }

    #[test]
    fn test_adamw_step() {
        let mut params = vec![1.0f32; 10];
        let gradients = vec![0.1f32; 10];
        let mut optimizer = AdamWCpu::with_phi_defaults(10);
        let initial_param = params[0];
        optimizer.step(&mut params, &gradients);
        assert!(params[0] < initial_param);
        assert_eq!(optimizer.step_count(), 1);
        optimizer.step(&mut params, &gradients);
        assert_eq!(optimizer.step_count(), 2);
    }

    #[test]
    fn test_adamw_reset() {
        let mut params = vec![1.0f32; 10];
        let gradients = vec![0.1f32; 10];
        let mut optimizer = AdamWCpu::with_phi_defaults(10);
        optimizer.step(&mut params, &gradients);
        assert!(optimizer.m.iter().any(|&m| m != 0.0));
        optimizer.reset();
        assert_eq!(optimizer.step_count(), 0);
        assert!(optimizer.m.iter().all(|&m| m == 0.0));
        assert!(optimizer.v.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_phi_lr_schedule_warmup() {
        let base_lr = 0.1;
        let warmup_steps = 10;
        let lr_0 = phi_lr_schedule(0, base_lr, warmup_steps);
        assert_eq!(lr_0, 0.0);
        let lr_5 = phi_lr_schedule(5, base_lr, warmup_steps);
        assert!((lr_5 - 0.05).abs() < 1e-6);
        let lr_10 = phi_lr_schedule(10, base_lr, warmup_steps);
        assert!((lr_10 - base_lr).abs() < 1e-6);
    }

    #[test]
    fn test_phi_lr_schedule_decay() {
        let base_lr = 0.1;
        let warmup_steps = 10;
        let lr_10 = phi_lr_schedule(10, base_lr, warmup_steps);
        let lr_20 = phi_lr_schedule(20, base_lr, warmup_steps);
        let lr_30 = phi_lr_schedule(30, base_lr, warmup_steps);
        assert!(lr_20 < lr_10, "LR should decay");
        assert!(lr_30 < lr_20, "LR should continue decaying");
    }

    #[test]
    fn test_phi_lr_schedule_phi_factor() {
        let base_lr = 1.0;
        let warmup_steps = 1;
        let lr_1 = phi_lr_schedule(1, base_lr, warmup_steps);
        let lr_2 = phi_lr_schedule(2, base_lr, warmup_steps);
        assert!((lr_2 - lr_1 / phi()).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_momentum() {
        let mut params = vec![1.0f32; 10];
        let gradients = vec![0.1f32; 10];
        let mut optimizer = SGDMomentum::new(10, 0.01, 0.9);
        let initial_param = params[0];
        optimizer.step(&mut params, &gradients);
        assert!(params[0] < initial_param);
        assert_eq!(optimizer.step_count(), 1);
    }

    #[test]
    fn test_phi_constants_precision() {
        let optimizer = AdamWCpu::with_phi_defaults(10);
        let expected_beta1 = 1.0 / phi();
        assert!((optimizer.beta1 - expected_beta1).abs() < 1e-6);
        let expected_wd = 1.0 / (phi() * phi() * phi());
        assert!((optimizer.weight_decay - expected_wd).abs() < 1e-6);
        assert!((expected_wd - 0.23607).abs() < 0.001);
    }

    #[test]
    fn test_muon_creation() {
        let opt = MuonOptimizer::new(100, 0.02, 0.95, 0.01);
        assert_eq!(opt.step_count(), 0);
    }

    #[test]
    fn test_muon_step_decreases_param() {
        let mut params = vec![1.0f32; 10];
        let gradients = vec![0.1f32; 10];
        let mut opt = MuonOptimizer::new(10, 0.02, 0.95, 0.01);
        let initial = params[0];
        opt.step(&mut params, &gradients);
        assert!(params[0] < initial, "Muon should decrease params");
        assert_eq!(opt.step_count(), 1);
    }

    #[test]
    fn test_muon_reset() {
        let mut params = vec![1.0f32; 10];
        let gradients = vec![0.1f32; 10];
        let mut opt = MuonOptimizer::new(10, 0.02, 0.95, 0.01);
        opt.step(&mut params, &gradients);
        assert!(opt.step_count() > 0);
        opt.reset();
        assert_eq!(opt.step_count(), 0);
        assert!(opt.momentum_buffer.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_muon_with_matrix_shape() {
        let mut params = vec![1.0f32; 12];
        let gradients = vec![0.1f32; 12];
        let mut opt = MuonOptimizer::with_matrix_shape(12, 3, 4, 0.02, 0.95, 0.01);
        let initial = params[0];
        opt.step(&mut params, &gradients);
        assert!(params[0] < initial);
        assert_eq!(opt.param_rows, 3);
        assert_eq!(opt.param_cols, 4);
    }

    #[test]
    fn test_muon_orthogonalization() {
        let mut params = vec![1.0f32; 16];
        let gradients: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let mut opt = MuonOptimizer::with_matrix_shape(16, 4, 4, 0.02, 0.95, 0.0);
        for _ in 0..3 {
            opt.step(&mut params, &gradients);
        }
        for &p in &params {
            assert!(p.is_finite(), "Muon params should be finite");
        }
    }

    #[test]
    fn test_newton_schulz_cubic_legacy() {
        let identity: Vec<f32> = (0..4).map(|i| if i % 5 == 0 { 1.0f32 } else { 0.0f32 }).collect();
        let result = newton_schulz_cubic(&identity, 2, 2);
        for i in 0..4 {
            assert!((result[i] - identity[i]).abs() < 0.01, "cubic NS should preserve identity");
        }
    }

    #[test]
    fn optimizer_kind_dispatch() {
        let n = 4;
        let mut params_a = vec![1.0f32; n];
        let mut params_m = vec![1.0f32; n];
        let grads = vec![0.1f32; n];
        let mut adamw = OptimizerKind::AdamW(AdamWCpu::with_params(n, 0.004, 0.9, 0.999, 0.01));
        let mut muon = OptimizerKind::Muon(MuonOptimizer::new(n, 0.004, 0.95, 0.01));
        adamw.step(&mut params_a, &grads);
        muon.step(&mut params_m, &grads);
        assert!(params_a[0] < 1.0);
        assert!(params_m[0] < 1.0);
    }

    #[test]
    fn test_newton_schulz_5_finite_output() {
        let identity: Vec<f32> = (0..4).map(|i| if i % 5 == 0 { 1.0f32 } else { 0.0f32 }).collect();
        let result = newton_schulz_5(&identity, 2, 2, 3.4445, -4.7750, 2.0315);
        for &r in &result {
            assert!(r.is_finite(), "NS5 output should be finite");
        }
    }

    #[test]
    fn test_newton_schulz_5_coefficients() {
        let opt = MuonOptimizer::new(16, 0.02, 0.95, 0.01);
        assert!((opt.ns_a - 3.4445).abs() < 1e-4);
        assert!((opt.ns_b - (-4.7750)).abs() < 1e-4);
        assert!((opt.ns_c - 2.0315).abs() < 1e-4);
    }

    #[test]
    fn test_muon_ns5_orthogonalization() {
        let mut params = vec![1.0f32; 16];
        let gradients: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let mut opt = MuonOptimizer::with_matrix_shape(16, 4, 4, 0.02, 0.95, 0.0);
        for _ in 0..3 {
            opt.step(&mut params, &gradients);
        }
        for &p in &params {
            assert!(p.is_finite(), "Muon params should be finite");
        }
    }

    #[test]
    fn test_muon_custom_ns_coefficients() {
        let opt = MuonOptimizer::new(16, 0.02, 0.95, 0.01)
            .with_ns_coefficients(1.5, -0.5, 0.0);
        assert!((opt.ns_a - 1.5).abs() < 1e-4);
        assert!((opt.ns_b - (-0.5)).abs() < 1e-4);
        assert!((opt.ns_c - 0.0).abs() < 1e-4);
    }
}
