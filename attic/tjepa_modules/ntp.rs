//! T-JEPA NTP Head Module

pub struct NtpHead {
    w: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
    d_model: usize,
    t: u32,
    lr: f32,
}

impl NtpHead {
    pub fn new(d_model: usize, seed: u64, lr: f32) -> Self {
        let n = d_model * 128; // VOCAB
        let mut s = seed;
        let lim = (1.0f32 / d_model as f32).sqrt();
        let w: Vec<f32> = (0..n).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0) * lim
        }).collect();
        assert!(w.len() == n, "weight size mismatch");
        assert!(lr > 0.0 && lr.is_finite(), "lr must be positive finite");
        Self { m: vec![0.0; n], v: vec![0.0; n], w, d_model, t: 0, lr }
    }

    pub fn forward(&mut self, hidden: &[f32]) -> Vec<f32> {
        assert!(!hidden.is_empty(), "hidden must not be empty");
        assert!(hidden.len() == self.d_model, "hidden dimension mismatch");
        let n = 128;
        let mut logits = vec![0.0f32; n];
        for vi in 0..n {
            let mut sum = 0.0;
            for hi in 0..self.d_model {
                sum += self.w[vi * self.d_model + hi] * hidden[hi];
            }
            logits[vi] = sum;
        }
        self.softmax(&mut logits);
        logits
    }

    pub fn backward(&mut self, hidden: &[f32], target: usize, loss: f32) -> Vec<f32> {
        assert!(!hidden.is_empty());
        assert!(hidden.len() == self.d_model);
        assert!(target < 128, "target must be < vocab size");
        assert!(loss >= 0.0 && loss.is_finite(), "loss must be non-negative finite");

        let mut d_hidden = vec![0.0; self.d_model];
        for hi in 0..self.d_model {
            d_hidden[hi] = self.w[target * self.d_model + hi];
        }
        self.t += 1;
        d_hidden
    }

    pub fn step(&mut self, grads: &[f32]) {
        assert!(grads.len() == self.w.len(), "grads size mismatch");
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let eps = 1e-8_f32;
        let bc1 = 1.0_f32 - beta1.powi(self.t as i32);
        let bc2 = 1.0_f32 - beta2.powi(self.t as i32);

        for i in 0..self.w.len() {
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * grads[i];
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * grads[i] * grads[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            self.w[i] -= self.lr * (m_hat / (v_hat.sqrt() + eps));
        }
    }

    fn softmax(&self, v: &mut [f32]) {
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in v.iter_mut() {
            *x = (*x - max).exp();
            sum += *x;
        }
        for x in v.iter_mut() {
            *x /= sum.max(1e-8);
        }
    }

    pub fn forward_backward(&mut self, hidden: &[f32], target: usize, _loss: f32) -> (f32, Vec<f32>) {
        let logits = self.forward(hidden);
        let mut d_hidden = vec![0.0; self.d_model];
        for hi in 0..self.d_model {
            d_hidden[hi] = self.w[target * self.d_model + hi];
        }
        self.t += 1;
        (logits[target], d_hidden)
    }

    pub fn loss(&self, hidden: &[f32], target: usize) -> f32 {
        let logits = self.forward(hidden);
        let neg_log_p = -logits[target].max(1e-8).ln();
        neg_log_p
    }
}
