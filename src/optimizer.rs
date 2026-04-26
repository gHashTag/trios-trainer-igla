//! AdamW + Muon optimizer with phi-based defaults.
//!
//! Migrated from `trios-train-cpu/src/optimizer.rs` (L-T1, L-T2).

use anyhow::{Result, bail};

const PHI: f64 = (1.0 + 5.0_f64.sqrt()) / 2.0;
const PHI_SQ: f64 = PHI * PHI;
const PHI_CUBE: f64 = PHI * PHI * PHI;
const LR_SAFE_MIN: f64 = 0.002;
const LR_SAFE_MAX: f64 = 0.007;

pub struct AdamW {
    m: Vec<f32>,
    v: Vec<f32>,
    step: usize,
    beta1: f32,
    beta2: f32,
    wd: f32,
}

impl AdamW {
    pub fn new(size: usize, beta1: f32, beta2: f32, wd: f32) -> Self {
        Self { m: vec![0.0; size], v: vec![0.0; size], step: 0, beta1, beta2, wd }
    }

    pub fn with_phi_defaults(size: usize) -> Self {
        let lr = (1.0 / PHI_CUBE) as f32;
        Self::new(size, 0.9, 0.999, lr.max(LR_SAFE_MIN as f32).min(LR_SAFE_MAX as f32))
    }

    pub fn update(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        self.step += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);
        for i in 0..params.len() {
            params[i] -= self.wd * lr * params[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            params[i] -= lr * (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + 1e-8);
        }
    }

    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.update(params, grads, 1e-3);
    }

    pub fn new_with_lr(size: usize, _lr: f64) -> Self {
        Self { m: vec![0.0; size], v: vec![0.0; size], step: 0, beta1: 0.9, beta2: 0.999, wd: 0.01 }
    }

    pub fn reset(&mut self) {
        self.m.fill(0.0);
        self.v.fill(0.0);
        self.step = 0;
    }

    pub fn size(&self) -> usize {
        self.m.len()
    }
}

pub struct MuonOptimizer {
    momentum: Vec<f32>,
    lr: f64,
    momentum_beta: f64,
    wd: f64,
    step: usize,
}

impl MuonOptimizer {
    pub fn new(size: usize, lr: f64, momentum_beta: f64, wd: f64) -> Self {
        Self { momentum: vec![0.0; size], lr, momentum_beta, wd, step: 0 }
    }

    pub fn update(&mut self, params: &mut [f32], grads: &[f32]) {
        self.step += 1;
        for i in 0..params.len() {
            self.momentum[i] = self.momentum_beta as f32 * self.momentum[i]
                + (1.0 - self.momentum_beta as f32) * grads[i];
            params[i] -= self.lr as f32 * self.momentum[i] + self.wd as f32 * params[i];
        }
    }
}

pub fn build_adamw_phi_defaults(param_count: usize) -> AdamW {
    AdamW::with_phi_defaults(param_count)
}

pub fn build_muon(param_count: usize) -> MuonOptimizer {
    MuonOptimizer::new(param_count, 0.004, 0.95, 0.01)
}
