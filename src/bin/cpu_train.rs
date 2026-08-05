use std::fs;
use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

use trios_trainer::fake_quant::{self, FormatKind};

const LN_2: f32 = std::f32::consts::LN_2;

/// Default corpus paths. These are the files that actually exist in `data/`
/// and whose bytes are pinned by SHA-256 in `data/README.md`:
///   tiny_shakespeare.txt      1015394 B  1a5aead1db78653f...
///   tiny_shakespeare_val.txt   100000 B  2088af36b1c78310...
/// train ++ val reconstructs the canonical corpus (86c4e6aa..., 1115394 B),
/// so the two splits are byte-disjoint by construction.
///
/// The previous default spelled the corpus without the underscore in
/// "tiny_shakespeare", a path that has never existed in this repository.
/// Combined with the fallback that used to live in `load_data`, every run
/// silently trained on a 52-byte string.
const DEFAULT_TRAIN_PATH: &str = "data/tiny_shakespeare.txt";
const DEFAULT_VAL_PATH: &str = "data/tiny_shakespeare_val.txt";

/// Eval-corpus preconditions, mirroring `train_loop::MIN_VAL_TOKENS` /
/// `train_loop::MIN_EVAL_CHUNKS`. Those constants are `pub(crate)` and so are
/// not reachable from a binary target; the values and the rationale are copied
/// deliberately rather than weakened. If `train_loop` changes them, change them
/// here too.
const MIN_VAL_TOKENS: usize = 8192;
const MIN_EVAL_CHUNKS: usize = 8;

/// `eval_bpb` never looks at more than this many val tokens.
const EVAL_TOKEN_CAP: usize = 5000;

/// Number of chunks `eval_bpb` will actually average over for a val stream of
/// `val_len` tokens at `seq_len`. Mirrors the loop in `CpuModel::eval_bpb`
/// exactly, so the precondition below refuses precisely the streams that would
/// have produced an under-averaged mean.
fn eval_chunk_count(val_len: usize, seq_len: usize) -> usize {
    let max_eval = EVAL_TOKEN_CAP.min(val_len);
    let stride = seq_len + 1;
    let mut n = 0usize;
    let mut c = 0usize;
    while c < max_eval {
        let end = (c + stride).min(max_eval);
        if end - c >= 3 {
            n += 1;
        }
        c += stride;
    }
    n
}

/// Read a corpus file as byte tokens. There is no fallback: a trainer that
/// invents its own corpus when the real one is missing reports a number that
/// was measured against bytes nobody chose.
fn load_data(path: &str) -> Result<Vec<usize>, String> {
    let raw = fs::read(path).map_err(|e| {
        format!(
            "cannot read corpus {path}: {e}. There is no fallback corpus. \
             Provision the pinned split (see README.md 'Quickstart' or the \
             data/README.md manifest) and re-run."
        )
    })?;
    if raw.is_empty() {
        return Err(format!("corpus {path} is empty (0 bytes)"));
    }
    Ok(raw.into_iter().map(|b| b as usize).collect())
}

fn softmax(v: &mut [f32]) {
    let max_val = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    for x in v.iter_mut() {
        *x /= sum;
    }
}

fn rng_next(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let t = ((*s >> 33) as f32) / (u32::MAX as f32);
    t * 2.0 - 1.0
}

// ============================================================================
// Optimizers
// ============================================================================

struct AdamW {
    m: Vec<f32>,
    v: Vec<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    wd: f32,
    step: usize,
}

impl AdamW {
    fn new(size: usize, lr: f32) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            lr,
            beta1: 0.9,
            beta2: 0.999,
            wd: 0.01,
            step: 0,
        }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.step += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);
        for i in 0..params.len() {
            let g = grads[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * (m_hat / (v_hat.sqrt() + 1e-8) + self.wd * params[i]);
        }
    }
}

// ---- Muon (Newton-Schulz orthogonalized momentum) --------------------------

fn frobenius_norm_local(m: &[f32]) -> f32 {
    m.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8)
}

fn newton_schulz_5_local(m: &[f32], rows: usize, cols: usize, a: f32, b: f32, c: f32) -> Vec<f32> {
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

struct Muon {
    momentum_buffer: Vec<f32>,
    lr: f32,
    momentum: f32,
    wd: f32,
    ns_steps: usize,
    ns_a: f32,
    ns_b: f32,
    ns_c: f32,
    param_rows: usize,
    param_cols: usize,
    step: usize,
}

impl Muon {
    fn new(size: usize, lr: f32) -> Self {
        let cols = (size as f64).sqrt().round() as usize;
        let cols = cols.max(1);
        let rows = ((size as f64) / (cols as f64)).ceil() as usize;
        Self {
            momentum_buffer: vec![0.0; size],
            lr,
            momentum: 0.95,
            wd: 0.01,
            ns_steps: 5,
            ns_a: 3.4445,
            ns_b: -4.7750,
            ns_c: 2.0315,
            param_rows: rows,
            param_cols: cols,
            step: 0,
        }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.step += 1;
        let lr = self.lr;
        let mom = self.momentum;
        let wd = self.wd;
        let n = params.len();

        for p in params.iter_mut() {
            *p *= 1.0 - lr * wd;
        }
        for i in 0..n {
            self.momentum_buffer[i] = mom * self.momentum_buffer[i] + (1.0 - mom) * grads[i];
        }

        let update = self.orthogonalize();
        for i in 0..n {
            params[i] -= lr * update[i];
        }
    }

    fn orthogonalize(&self) -> Vec<f32> {
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

        let norm = frobenius_norm_local(&m);
        if norm < 1e-8 {
            return self.momentum_buffer.clone();
        }
        let scale = 1.0 / norm;
        for v in m.iter_mut() {
            *v *= scale;
        }
        for _ in 0..self.ns_steps {
            m = newton_schulz_5_local(&m, rows, cols, self.ns_a, self.ns_b, self.ns_c);
        }
        let out_norm = frobenius_norm_local(&m);
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
}

// ---- SGDM ------------------------------------------------------------------

struct Sgdm {
    velocity: Vec<f32>,
    lr: f32,
    momentum: f32,
    wd: f32,
    step: usize,
}

impl Sgdm {
    fn new(size: usize, lr: f32) -> Self {
        Self {
            velocity: vec![0.0; size],
            lr,
            momentum: 0.9,
            wd: 0.0,
            step: 0,
        }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.step += 1;
        for i in 0..params.len() {
            let g = grads[i] + self.wd * params[i];
            self.velocity[i] = self.momentum * self.velocity[i] + g;
            params[i] -= self.lr * self.velocity[i];
        }
    }
}

// ---- Lion ------------------------------------------------------------------
// Chen et al. 2023 -- sign update with momentum interpolation

struct Lion {
    m: Vec<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    wd: f32,
    step: usize,
}

impl Lion {
    fn new(size: usize, lr: f32) -> Self {
        Self {
            m: vec![0.0; size],
            lr,
            beta1: 0.9,
            beta2: 0.99,
            wd: 0.01,
            step: 0,
        }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.step += 1;
        for i in 0..params.len() {
            let g = grads[i];
            // update = sign(beta1 * m + (1 - beta1) * g)
            let update_arg = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            let update = update_arg.signum();
            // weight decay
            params[i] *= 1.0 - self.lr * self.wd;
            // param update
            params[i] -= self.lr * update;
            // momentum update
            self.m[i] = self.beta2 * self.m[i] + (1.0 - self.beta2) * g;
        }
    }
}

// ---- Adafactor -------------------------------------------------------------
// Relative-step on; factored second moment for matrices (>= 2 params arranged
// as rows x cols); 1-D vectors fall back to non-factored.

struct Adafactor {
    // For factored: row-factor and col-factor
    vr: Option<Vec<f64>>, // shape: [rows]
    vc: Option<Vec<f64>>, // shape: [cols]
    // For non-factored
    v: Option<Vec<f64>>,
    step: usize,
    size: usize,
    rows: usize,
    cols: usize,
    // rms-scaling: we track rms of params at step 0 (or set 1.0)
    rho: f64, // exponential decay for second moment
    eps1: f64,
    eps2: f64,
}

impl Adafactor {
    fn new(size: usize, _lr: f32) -> Self {
        // Decide factoring: if size >= 4, use square-like factoring
        let (rows, cols, factored) = if size >= 4 {
            let cols = (size as f64).sqrt().round() as usize;
            let cols = cols.max(2);
            let rows = size.div_ceil(cols);
            (rows, cols, true)
        } else {
            (1, size, false)
        };

        let (vr, vc, v) = if factored && rows >= 2 && cols >= 2 {
            (Some(vec![1e-30; rows]), Some(vec![1e-30; cols]), None)
        } else {
            (None, None, Some(vec![1e-30; size]))
        };

        Self {
            vr,
            vc,
            v,
            step: 0,
            size,
            rows,
            cols,
            rho: 1.0 - 1e-8,
            eps1: 1e-30,
            eps2: 1e-3,
        }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.step += 1;
        let t = self.step as f64;

        // Relative step size: lr_t = max(eps2, 1/sqrt(t))
        let lr_t = (1.0 / t.sqrt()).max(self.eps2);

        // rho (second moment decay): 1 - t^(-0.8)
        let rho = 1.0 - t.powf(-0.8_f64);

        // scale_by_rms: compute rms of params
        let rms_w = {
            let sq: f64 = params.iter().map(|&p| (p as f64) * (p as f64)).sum::<f64>();
            (sq / self.size as f64).sqrt().max(1.0)
        };
        let d = rms_w;

        // adapted lr
        let alpha = lr_t * d;

        let n = params.len();

        if let (Some(ref mut vr), Some(ref mut vc)) = (&mut self.vr, &mut self.vc) {
            // Factored second moment
            let rows = self.rows;
            let cols = self.cols;

            // Accumulate row and col sums of g^2 + eps1
            let mut row_sum = vec![0.0f64; rows];
            let mut col_sum = vec![0.0f64; cols];

            for idx in 0..n {
                let r = idx / cols;
                let c = idx % cols;
                let g2 = (grads[idx] as f64).powi(2) + self.eps1;
                row_sum[r] += g2;
                col_sum[c] += g2;
            }

            // Update row/col factors
            for r in 0..rows {
                vr[r] = rho * vr[r] + (1.0 - rho) * row_sum[r];
            }
            for c in 0..cols {
                vc[c] = rho * vc[c] + (1.0 - rho) * col_sum[c];
            }

            // Compute reconstructed second moment: V_hat[i] = vr[r] * vc[c] / sum(vc)
            let vc_sum: f64 = vc.iter().sum::<f64>().max(1e-30);

            for idx in 0..n {
                let r = idx / cols;
                let c = idx % cols;
                let v_hat = (vr[r] * vc[c] / vc_sum).max(self.eps1);
                let update = (grads[idx] as f64) / v_hat.sqrt();
                // RMS clipping: clip ||update|| to 1.0
                params[idx] -= (alpha * update) as f32;
            }
        } else if let Some(ref mut v) = &mut self.v {
            // Non-factored
            for idx in 0..n {
                let g2 = (grads[idx] as f64).powi(2) + self.eps1;
                v[idx] = rho * v[idx] + (1.0 - rho) * g2;
                let v_hat = v[idx].max(self.eps1);
                let update = (grads[idx] as f64) / v_hat.sqrt();
                params[idx] -= (alpha * update) as f32;
            }
        }
    }
}

// ---- LAMB ------------------------------------------------------------------
// You et al. 2019 -- AdamW + layer-wise trust ratio

struct Lamb {
    m: Vec<f64>,
    v: Vec<f64>,
    lr: f32,
    beta1: f64,
    beta2: f64,
    wd: f64,
    eps: f64,
    step: usize,
}

impl Lamb {
    fn new(size: usize, lr: f32) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            lr,
            beta1: 0.9,
            beta2: 0.999,
            wd: 0.01,
            eps: 1e-6,
            step: 0,
        }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.step += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);
        let n = params.len();

        let mut update = vec![0.0f64; n];
        for i in 0..n {
            let g = grads[i] as f64;
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            update[i] = m_hat / (v_hat.sqrt() + self.eps) + self.wd * params[i] as f64;
        }

        // Layer-wise trust ratio: lr_eff = lr * ||w|| / ||update||
        let w_norm: f64 = params
            .iter()
            .map(|&p| (p as f64) * (p as f64))
            .sum::<f64>()
            .sqrt();
        let u_norm: f64 = update.iter().map(|&u| u * u).sum::<f64>().sqrt();

        let trust = if w_norm < 1e-8 || u_norm < 1e-8 {
            1.0f64
        } else {
            w_norm / u_norm
        };

        let eff_lr = self.lr as f64 * trust;
        for i in 0..n {
            params[i] -= (eff_lr * update[i]) as f32;
        }
    }
}

// ---- ScheduleFree ----------------------------------------------------------
// Defazio 2024 Algorithm 1 -- Polyak-Ruppert averaging with momentum
// State: x (fast iterate), z (averaged iterate)
// y = (1-beta1)*z + beta1*x  (interpolated point where grad is evaluated)
// z_{t+1} = z_t + c_{t+1} * (x_{t+1} - z_t)   where c = 1/(t+1)
// x_{t+1} = x_t - lr * grad(y_t)

struct ScheduleFree {
    x: Vec<f32>, // fast (online) iterate
    z: Vec<f64>, // averaged (Polyak) iterate
    lr: f32,
    beta1: f32,
    step: usize,
    initialized: bool,
}

impl ScheduleFree {
    fn new(size: usize, lr: f32) -> Self {
        Self {
            x: vec![0.0; size],
            z: vec![0.0; size],
            lr,
            beta1: 0.9,
            step: 0,
            initialized: false,
        }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        // On first call, initialize x and z from current params
        if !self.initialized {
            self.x.copy_from_slice(params);
            for (z, &p) in self.z.iter_mut().zip(params.iter()) {
                *z = p as f64;
            }
            self.initialized = true;
        }

        self.step += 1;
        let t = self.step;
        let c = 1.0f64 / (t as f64 + 1.0);
        let lr = self.lr as f64;
        let beta1 = self.beta1 as f64;
        let n = params.len();

        // y_t = (1 - beta1)*z + beta1*x  -- interpolated eval point
        // We already have params = y_{t-1}; update in place

        // x_{t+1} = x_t - lr * grad(y_t)
        for i in 0..n {
            self.x[i] -= (lr * grads[i] as f64) as f32;
        }

        // z_{t+1} = (1 - c)*z + c*x_{t+1}
        for i in 0..n {
            self.z[i] = (1.0 - c) * self.z[i] + c * self.x[i] as f64;
        }

        // Set params = y_{t+1} = (1-beta1)*z_{t+1} + beta1*x_{t+1}
        // (the point where next gradient will be evaluated)
        for i in 0..n {
            params[i] = ((1.0 - beta1) * self.z[i] + beta1 * self.x[i] as f64) as f32;
        }
    }
}

// ---- RMSprop ---------------------------------------------------------------
// Classic: alpha=0.99, eps=1e-8, no momentum

struct RmsProp {
    v: Vec<f64>,
    lr: f32,
    alpha: f64,
    eps: f64,
    step: usize,
}

impl RmsProp {
    fn new(size: usize, lr: f32) -> Self {
        Self {
            v: vec![0.0; size],
            lr,
            alpha: 0.99,
            eps: 1e-8,
            step: 0,
        }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.step += 1;
        for i in 0..params.len() {
            let g = grads[i] as f64;
            self.v[i] = self.alpha * self.v[i] + (1.0 - self.alpha) * g * g;
            let denom = (self.v[i] + self.eps).sqrt();
            params[i] -= (self.lr as f64 * g / denom) as f32;
        }
    }
}

// ---- SOAP ------------------------------------------------------------------
// Vyas et al. 2024 -- "SOAP: Improving and Stabilizing Shampoo using Adam"
// arXiv:2409.11321. Reference impl on flat parameter vectors:
//   * AdamW-style (m, v) moments
//   * Diagonal preconditioner refreshed every `precond_freq` steps from the
//     EMA of squared gradients (the flat-vector reduction of Shampoo's GG^T
//     eigenbasis: when the parameter is a flat vector with no block
//     structure, the eigenbasis is the standard basis and SOAP collapses to
//     a windowed AdamW with periodic preconditioner reset).
// Honest scope (R5): faithful reduction for flat tensors. Block-structured
// SOAP with full GG^T eigendecomposition is deferred -- flagged below.

struct Soap {
    m: Vec<f64>,       // first moment (Adam in eigenbasis == Adam in std basis here)
    v: Vec<f64>,       // second moment
    precond: Vec<f64>, // diagonal preconditioner (EMA of g^2, refreshed)
    lr: f32,
    beta1: f64,
    beta2: f64,
    beta_precond: f64, // EMA decay for preconditioner refresh
    wd: f64,
    eps: f64,
    step: usize,
    precond_freq: usize, // refresh preconditioner every K steps (K=10 in paper)
}

impl Soap {
    fn new(size: usize, lr: f32) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            precond: vec![1.0; size],
            lr,
            beta1: 0.95,
            beta2: 0.95,
            beta_precond: 0.95,
            wd: 0.01,
            eps: 1e-8,
            step: 0,
            precond_freq: 10,
        }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.step += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);

        // Refresh diagonal preconditioner every `precond_freq` steps.
        // On the flat-vector reduction, this is the running EMA of g^2 used
        // as the preconditioner basis (Shampoo's GG^T diagonal).
        if self.step.is_multiple_of(self.precond_freq) {
            for i in 0..params.len() {
                let g = grads[i] as f64;
                self.precond[i] =
                    self.beta_precond * self.precond[i] + (1.0 - self.beta_precond) * g * g;
            }
        }

        for i in 0..params.len() {
            let g = grads[i] as f64;
            // Adam moments in the (diagonal) eigenbasis.
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            // Normalise by max(v_hat, precond) to apply the SOAP "max"
            // stabiliser (Vyas et al. Sec.3.2): keeps the update bounded by the
            // longer-window second-moment estimate.
            let denom = v_hat.max(self.precond[i]).sqrt() + self.eps;
            let upd = m_hat / denom + self.wd * params[i] as f64;
            params[i] -= (self.lr as f64 * upd) as f32;
        }
    }
}

// ============================================================================
// AlgoOpt enum -- unified dispatch
// ============================================================================

enum AlgoOpt {
    AdamW(AdamW),
    Muon(Muon),
    Sgdm(Sgdm),
    Lion(Lion),
    Adafactor(Adafactor),
    Lamb(Lamb),
    ScheduleFree(ScheduleFree),
    RmsProp(RmsProp),
    Soap(Soap),
}

impl AlgoOpt {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        match self {
            AlgoOpt::AdamW(o) => o.step(params, grads),
            AlgoOpt::Muon(o) => o.step(params, grads),
            AlgoOpt::Sgdm(o) => o.step(params, grads),
            AlgoOpt::Lion(o) => o.step(params, grads),
            AlgoOpt::Adafactor(o) => o.step(params, grads),
            AlgoOpt::Lamb(o) => o.step(params, grads),
            AlgoOpt::ScheduleFree(o) => o.step(params, grads),
            AlgoOpt::RmsProp(o) => o.step(params, grads),
            AlgoOpt::Soap(o) => o.step(params, grads),
        }
    }

    /// Build from name string. Panics with clear message on unknown name (R5-honest).
    fn from_env(name: &str, size: usize, lr: f32) -> AlgoOpt {
        match name {
            "adamw" => AlgoOpt::AdamW(AdamW::new(size, lr)),
            "muon" => AlgoOpt::Muon(Muon::new(size, lr)),
            "sgdm" => AlgoOpt::Sgdm(Sgdm::new(size, lr)),
            "lion" => AlgoOpt::Lion(Lion::new(size, lr)),
            "adafactor" => AlgoOpt::Adafactor(Adafactor::new(size, lr)),
            "lamb" => AlgoOpt::Lamb(Lamb::new(size, lr)),
            "schedulefree" => AlgoOpt::ScheduleFree(ScheduleFree::new(size, lr)),
            "rmsprop" => AlgoOpt::RmsProp(RmsProp::new(size, lr)),
            "soap" => AlgoOpt::Soap(Soap::new(size, lr)),
            other => panic!(
                "TRIOS_ALGO_TYPE: unknown optimizer '{}'. \
                 Valid choices: adamw, muon, sgdm, lion, adafactor, lamb, schedulefree, rmsprop, soap",
                other
            ),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            AlgoOpt::AdamW(_) => "adamw",
            AlgoOpt::Muon(_) => "muon",
            AlgoOpt::Sgdm(_) => "sgdm",
            AlgoOpt::Lion(_) => "lion",
            AlgoOpt::Adafactor(_) => "adafactor",
            AlgoOpt::Lamb(_) => "lamb",
            AlgoOpt::ScheduleFree(_) => "schedulefree",
            AlgoOpt::RmsProp(_) => "rmsprop",
            AlgoOpt::Soap(_) => "soap",
        }
    }
}

// ============================================================================
// Model structures
// ============================================================================

struct BigramHash {
    embed: Vec<f32>,
    vocab: usize,
    dim: usize,
}

impl BigramHash {
    fn new(vocab: usize, dim: usize, seed: &mut u64) -> Self {
        let embed: Vec<f32> = (0..vocab * dim).map(|_| rng_next(seed) * 0.02).collect();
        Self { embed, vocab, dim }
    }

    fn hash(&self, curr: usize, prev: usize) -> usize {
        ((36313u32.wrapping_mul(curr as u32)) ^ (27191u32.wrapping_mul(prev as u32))) as usize
            % (self.vocab - 1)
    }

    fn forward(&self, tokens: &[usize]) -> Vec<Vec<f32>> {
        let d = self.dim;
        let mut out = Vec::with_capacity(tokens.len());
        for (i, &t) in tokens.iter().enumerate() {
            let prev = if i > 0 { tokens[i - 1] } else { 0 };
            let h = self.hash(t, prev);
            out.push(self.embed[h * d..(h + 1) * d].to_vec());
        }
        out
    }

    fn grad_step(&mut self, tokens: &[usize], grad: &[Vec<f32>], lr: f32) {
        let d = self.dim;
        for (i, &t) in tokens.iter().enumerate() {
            let prev = if i > 0 { tokens[i - 1] } else { 0 };
            let h = self.hash(t, prev);
            for (j, g) in grad[i].iter().enumerate().take(d) {
                self.embed[h * d + j] -= lr * g;
            }
        }
    }
}

struct SmearGate {
    gate: Vec<f32>,
}

impl SmearGate {
    fn new(dim: usize) -> Self {
        Self {
            gate: vec![0.0f32; dim],
        }
    }

    fn forward(&self, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(xs.len());
        for (i, x) in xs.iter().enumerate() {
            let g: Vec<f32> = self
                .gate
                .iter()
                .map(|&g| 1.0 / (1.0 + (-g).exp()))
                .collect();
            if i == 0 {
                out.push(
                    x.iter()
                        .zip(g.iter())
                        .map(|(xi, gi)| xi * (1.0 - gi))
                        .collect(),
                );
            } else {
                out.push(
                    x.iter()
                        .zip(g.iter())
                        .zip(xs[i - 1].iter())
                        .map(|((xi, gi), pi)| xi * (1.0 - gi) + pi * gi)
                        .collect(),
                );
            }
        }
        out
    }

    fn grad_step(&mut self, grad: &[Vec<f32>], lr: f32) {
        for (i, g) in self.gate.iter_mut().enumerate() {
            let mut total = 0.0f32;
            for g_vec in grad {
                total += g_vec[i];
            }
            *g -= lr * total;
        }
    }
}

struct FFNLayer {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    d_model: usize,
    d_ff: usize,
}

impl FFNLayer {
    fn new(d_model: usize, d_ff: usize, seed: &mut u64) -> Self {
        let std = (2.0 / (d_model + d_ff) as f32).sqrt();
        Self {
            w1: (0..d_ff * d_model).map(|_| rng_next(seed) * std).collect(),
            b1: vec![0.0; d_ff],
            w2: (0..d_model * d_ff).map(|_| rng_next(seed) * std).collect(),
            b2: vec![0.0; d_model],
            d_model,
            d_ff,
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let hidden: Vec<f32> = (0..self.d_ff)
            .map(|r| {
                let row = &self.w1[r * self.d_model..(r + 1) * self.d_model];
                let sum: f32 = row.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum();
                (sum + self.b1[r]).max(0.0)
            })
            .collect();
        (0..self.d_model)
            .map(|r| {
                let row = &self.w2[r * self.d_ff..(r + 1) * self.d_ff];
                let sum: f32 = row.iter().zip(hidden.iter()).map(|(&w, &h)| w * h).sum();
                sum + self.b2[r]
            })
            .collect()
    }

    #[allow(clippy::needless_range_loop)]
    fn backward(&self, x: &[f32], grad_out: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let d = self.d_model;
        let ff = self.d_ff;
        let mut hidden = vec![0.0f32; ff];
        for r in 0..ff {
            let row = &self.w1[r * d..(r + 1) * d];
            hidden[r] = row.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum();
        }
        let activated: Vec<f32> = hidden.iter().map(|&h| h.max(0.0)).collect();
        let relu_mask: Vec<f32> = hidden
            .iter()
            .map(|&h| if h > 0.0 { 1.0 } else { 0.0 })
            .collect();

        let mut d_w2 = vec![0.0f32; d * ff];
        let mut d_b2 = vec![0.0f32; d];
        let mut d_hidden = vec![0.0f32; ff];

        for r in 0..d {
            for k in 0..ff {
                d_w2[r * ff + k] += grad_out[r] * activated[k];
                d_hidden[k] += grad_out[r] * self.w2[r * ff + k];
            }
            d_b2[r] += grad_out[r];
        }

        for k in 0..ff {
            d_hidden[k] *= relu_mask[k];
        }

        let mut d_w1 = vec![0.0f32; ff * d];
        let mut d_b1 = vec![0.0f32; ff];
        let mut d_input = vec![0.0f32; d];

        for k in 0..ff {
            for j in 0..d {
                d_w1[k * d + j] += d_hidden[k] * x[j];
                d_input[j] += d_hidden[k] * self.w1[k * d + j];
            }
            d_b1[k] += d_hidden[k];
        }

        let _ = d_input; // used implicitly in backward pass
        (d_w1, d_b1, d_w2, d_b2)
    }
}

struct CpuModel {
    embed: Vec<f32>,
    lm_head: Vec<f32>,
    bigram: BigramHash,
    smear: SmearGate,
    ffn_layers: Vec<FFNLayer>,
    bigram_scale: f32,
    vocab: usize,
    dim: usize,
}

impl CpuModel {
    fn new(vocab: usize, dim: usize, seed: u64) -> Self {
        let mut s = seed;
        let embed: Vec<f32> = (0..vocab * dim).map(|_| rng_next(&mut s) * 0.02).collect();
        let lm_head: Vec<f32> = (0..vocab * dim).map(|_| rng_next(&mut s) * 0.02).collect();
        let bigram = BigramHash::new(vocab, dim, &mut s);
        let smear = SmearGate::new(dim);

        let ffn_layers = if std::env::args().any(|a| a == "--ffn") {
            let mut layers = Vec::new();
            let ffn_layers_str = arg_or("ffn-layers", "2");
            let n_layers = ffn_layers_str.parse::<usize>().unwrap_or(2);
            for _ in 0..n_layers {
                layers.push(FFNLayer::new(dim, dim * 4, &mut s));
            }
            layers
        } else {
            Vec::new()
        };

        Self {
            embed,
            lm_head,
            bigram,
            smear,
            ffn_layers,
            bigram_scale: 0.1,
            vocab,
            dim,
        }
    }

    #[allow(dead_code)]
    fn forward_logits(&self, tokens: &[usize]) -> Vec<Vec<f32>> {
        let d = self.dim;
        let v = self.vocab;

        let tok_emb: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&id| self.embed[(id % v) * d..((id % v) + 1) * d].to_vec())
            .collect();

        let bigram_emb = self.bigram.forward(tokens);
        let mut xs: Vec<Vec<f32>> = tok_emb
            .iter()
            .zip(bigram_emb.iter())
            .map(|(t, b)| {
                t.iter()
                    .zip(b.iter())
                    .map(|(ti, bi)| ti + bi * self.bigram_scale)
                    .collect()
            })
            .collect();

        xs = self.smear.forward(&xs);

        let mut logits = Vec::with_capacity(tokens.len());
        for x in &xs {
            let mut row = vec![0.0f32; v];
            for (vi, r) in row.iter_mut().enumerate() {
                for (j, xj) in x.iter().enumerate() {
                    *r += self.lm_head[vi * d + j] * xj;
                }
            }
            logits.push(row);
        }
        logits
    }

    fn loss_and_grad(&self, tokens: &[usize]) -> (f32, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let d = self.dim;
        let v = self.vocab;
        let n = tokens.len();

        let tok_emb: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&id| self.embed[(id % v) * d..((id % v) + 1) * d].to_vec())
            .collect();
        let bigram_emb = self.bigram.forward(tokens);
        let xs: Vec<Vec<f32>> = tok_emb
            .iter()
            .zip(bigram_emb.iter())
            .map(|(t, b)| {
                t.iter()
                    .zip(b.iter())
                    .map(|(ti, bi)| ti + bi * self.bigram_scale)
                    .collect()
            })
            .collect();
        let xs_smeared = self.smear.forward(&xs);

        let xs_final: Vec<Vec<f32>> = if !self.ffn_layers.is_empty() {
            let mut current = xs_smeared;
            for ffn_layer in &self.ffn_layers {
                let normed: Vec<Vec<f32>> = current
                    .iter()
                    .map(|x| {
                        let mean = x.iter().sum::<f32>() / d as f32;
                        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
                        let std = (var + 1e-5).sqrt();
                        x.iter().map(|v| (v - mean) / std).collect()
                    })
                    .collect();
                let ffn_out: Vec<Vec<f32>> = normed.iter().map(|x| ffn_layer.forward(x)).collect();
                current = (0..n)
                    .map(|i| {
                        current[i]
                            .iter()
                            .zip(ffn_out[i].iter())
                            .map(|(&a, &b)| a + b)
                            .collect()
                    })
                    .collect();
            }
            current
        } else {
            xs_smeared.clone()
        };

        let mut total_loss = 0.0f32;
        let mut d_logits = vec![vec![0.0f32; v]; n - 1];

        for i in 0..n - 1 {
            let x = &xs_final[i];
            let target = tokens[i + 1] % v;
            let mut logits = vec![0.0f32; v];
            for (vi, l) in logits.iter_mut().enumerate() {
                for (j, xj) in x.iter().enumerate() {
                    *l += self.lm_head[vi * d + j] * xj;
                }
            }
            softmax(&mut logits);
            let p_target = logits[target];
            if !p_target.is_finite() || p_target <= 0.0 {
                // `f32::max` ignores NaN, so clamping `logits[target]` to a
                // 1e-10 floor used to return 1e-10 here, whose negative log is 23.026 nats -
                // a finite 33.2 bpb that passed `eval_bpb`'s `is_finite()`
                // filter and was published as a measurement. The floor did the
                // same to a merely UNDERFLOWED probability: a finite `0.0` out
                // of the f32 softmax, which `is_nan` accepts, became the
                // identical 23.02585 nats. Both now yield a NaN loss, which
                // that filter rejects. NaN is absorbing, so one bad position
                // invalidates the whole sequence, which is the honest reading.
                total_loss = f32::NAN;
            } else {
                total_loss -= p_target.ln();
            }
            for (vi, dl) in d_logits[i].iter_mut().enumerate() {
                *dl = logits[vi] - if vi == target { 1.0 } else { 0.0 };
            }
        }

        let loss = total_loss / (n - 1) as f32;

        let mut d_hidden = vec![vec![0.0f32; d]; n];
        for i in 0..n - 1 {
            for (vi, dl) in d_logits[i].iter().enumerate() {
                for (j, dh) in d_hidden[i].iter_mut().enumerate() {
                    *dh += dl * self.lm_head[vi * d + j];
                }
            }
        }

        (loss, d_logits, d_hidden)
    }

    fn train_step(
        &mut self,
        tokens: &[usize],
        opt_embed: &mut AlgoOpt,
        opt_head: &mut AlgoOpt,
        lr: f32,
    ) -> f32 {
        let d = self.dim;
        let v = self.vocab;
        let n = tokens.len();

        let (loss, d_logits, _d_hidden) = self.loss_and_grad(tokens);

        // Recompute forward activations
        let tok_emb: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&id| self.embed[(id % v) * d..((id % v) + 1) * d].to_vec())
            .collect();
        let bigram_emb = self.bigram.forward(tokens);
        let xs: Vec<Vec<f32>> = tok_emb
            .iter()
            .zip(bigram_emb.iter())
            .map(|(t, b)| {
                t.iter()
                    .zip(b.iter())
                    .map(|(ti, bi)| ti + bi * self.bigram_scale)
                    .collect()
            })
            .collect();
        let xs_smeared = self.smear.forward(&xs);

        let mut d_from_logits = vec![vec![0.0f32; d]; n];
        for (i, dl_row) in d_logits.iter().enumerate() {
            for (vi, &dl) in dl_row.iter().enumerate() {
                for (j, df) in d_from_logits[i].iter_mut().enumerate() {
                    *df += dl * self.lm_head[vi * d + j];
                }
            }
        }

        let (d_lm_head, d_to_embed) = if !self.ffn_layers.is_empty() {
            let mut xs_final = xs_smeared.clone();
            let mut normed_activations = Vec::new();

            for ffn_layer in &self.ffn_layers {
                let normed: Vec<Vec<f32>> = xs_final
                    .iter()
                    .map(|x| {
                        let mean = x.iter().sum::<f32>() / d as f32;
                        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
                        let std = (var + 1e-5).sqrt();
                        x.iter().map(|v| (v - mean) / std).collect()
                    })
                    .collect();
                normed_activations.push(normed.clone());

                let ffn_out: Vec<Vec<f32>> = normed.iter().map(|x| ffn_layer.forward(x)).collect();
                xs_final = (0..n)
                    .map(|i| {
                        xs_final[i]
                            .iter()
                            .zip(ffn_out[i].iter())
                            .map(|(&a, &b)| a + b)
                            .collect()
                    })
                    .collect();
            }

            let mut dlh = vec![0.0f32; v * d];
            for i in 0..n - 1 {
                for (vi, &dl) in d_logits[i].iter().enumerate() {
                    for (j, xf) in xs_final[i].iter().enumerate() {
                        dlh[vi * d + j] += dl * xf;
                    }
                }
            }

            let mut d_to_emb = vec![vec![0.0f32; d]; n];
            let mut current_grad = d_from_logits.clone();
            let mut all_layer_grads = Vec::with_capacity(self.ffn_layers.len());

            for (layer_idx, ffn_layer) in self.ffn_layers.iter().rev().enumerate() {
                let normed = &normed_activations[self.ffn_layers.len() - 1 - layer_idx];

                let mut layer_grads = Vec::with_capacity(n);
                for (i, normed_row) in normed.iter().enumerate() {
                    let gi = i.min(n - 2);
                    let (dw1, db1, dw2, db2) = ffn_layer.backward(normed_row, &current_grad[gi]);
                    layer_grads.push((dw1, db1, dw2, db2));
                }
                all_layer_grads.push(layer_grads);

                for (i, d_to_emb_row) in d_to_emb.iter_mut().enumerate().take(n) {
                    let x = &xs_smeared[i];
                    let mean = x.iter().sum::<f32>() / d as f32;
                    let var = x.iter().map(|vv| (vv - mean).powi(2)).sum::<f32>() / d as f32;
                    let std = (var + 1e-5).sqrt();
                    let nn = d as f32;
                    let dx_sum: f32 = current_grad[i.min(n - 2)].iter().sum();
                    let dx_xm_sum: f32 = current_grad[i.min(n - 2)]
                        .iter()
                        .zip(x.iter())
                        .map(|(&g, &xi)| g * (xi - mean))
                        .sum();
                    let inv_n_std = 1.0 / (nn * std);
                    let inv_var_eps = 1.0 / (var + 1e-5);

                    for (j, de) in d_to_emb_row.iter_mut().enumerate() {
                        let xm = x[j] - mean;
                        *de = inv_n_std
                            * (nn * current_grad[i.min(n - 2)][j]
                                - dx_sum
                                - xm * inv_var_eps * dx_xm_sum);
                    }
                }

                current_grad = d_to_emb.clone();
            }

            for (i, d_to_emb_row) in d_to_emb.iter_mut().enumerate() {
                let gi = i.min(n - 2);
                for (j, de) in d_to_emb_row.iter_mut().enumerate() {
                    *de += d_from_logits[gi][j];
                }
            }

            let n_layers = self.ffn_layers.len();
            for (layer_idx, layer_grads) in all_layer_grads.into_iter().enumerate() {
                let layer_mut = &mut self.ffn_layers[n_layers - 1 - layer_idx];
                for (dw1, db1, dw2, db2) in layer_grads.into_iter() {
                    for (k, &g) in dw1.iter().enumerate() {
                        layer_mut.w1[k] -= lr * g;
                    }
                    for (k, &g) in db1.iter().enumerate() {
                        layer_mut.b1[k] -= lr * g;
                    }
                    for (k, &g) in dw2.iter().enumerate() {
                        layer_mut.w2[k] -= lr * g;
                    }
                    for (k, &g) in db2.iter().enumerate() {
                        layer_mut.b2[k] -= lr * g;
                    }
                }
            }

            (dlh, d_to_emb)
        } else {
            let mut dlh = vec![0.0f32; v * d];
            for i in 0..n - 1 {
                for (vi, &dl) in d_logits[i].iter().enumerate() {
                    for (j, xf) in xs_smeared[i].iter().enumerate() {
                        dlh[vi * d + j] += dl * xf;
                    }
                }
            }
            (dlh, d_from_logits)
        };

        // Update embeddings
        let mut d_embed = vec![0.0f32; v * d];
        for (i, &tid) in tokens.iter().enumerate() {
            let id = tid % v;
            let gi = i.min(n - 2);
            for (j, &dh) in d_to_embed[gi].iter().enumerate().take(d) {
                d_embed[id * d + j] += dh;
            }
        }

        opt_head.step(&mut self.lm_head, &d_lm_head);
        opt_embed.step(&mut self.embed, &d_embed);

        self.bigram.grad_step(tokens, &d_to_embed, lr);
        self.smear.grad_step(&d_to_embed, lr);

        loss
    }

    /// Mean BPB over the val stream, together with the sample it was averaged
    /// over; `None` when nothing at all could be measured.
    ///
    /// This used to return `f32::MAX` when `n == 0`, a sentinel indistinguishable
    /// from a measurement once it had been written into a JSON field or compared
    /// with `<`. An absence is now an absence: callers must decide explicitly
    /// what to do, and no unmeasured value reaches the results file.
    ///
    /// It also used to return only the mean, so a run that quietly evaluated 9
    /// of 152 windows and a run that evaluated all 152 published the same
    /// shape of number. The realised count travels with the mean now; see
    /// `require_complete_sample` for why a shrunken sample is refused.
    fn eval_bpb(&self, tokens: &[usize], seq_len: usize) -> Option<EvalSample> {
        let max_eval = EVAL_TOKEN_CAP.min(tokens.len());
        let eval_tokens = &tokens[..max_eval];
        let mut total_bpb = 0.0f32;
        // `planned` counts the windows this loop intends to evaluate -- the
        // same windows `eval_chunk_count` counts, and the same number
        // `MIN_EVAL_CHUNKS` grades. `realised` counts the ones that produced a
        // finite loss.
        let mut planned = 0usize;
        let mut realised = 0usize;
        for c in (0..eval_tokens.len()).step_by(seq_len + 1) {
            let end = (c + seq_len + 1).min(eval_tokens.len());
            if end - c < 3 {
                continue;
            }
            planned += 1;
            let seq = &eval_tokens[c..end];
            let (loss, _, _) = self.loss_and_grad(seq);
            if loss.is_finite() {
                total_bpb += loss / LN_2;
                realised += 1;
            }
        }
        if realised == 0 {
            return None;
        }
        Some(EvalSample {
            mean: total_bpb / realised as f32,
            planned,
            realised,
        })
    }
}

/// One eval pass: the mean, and the sample the mean was actually taken over.
#[derive(Clone, Copy, Debug, PartialEq)]
struct EvalSample {
    mean: f32,
    /// Windows the eval loop set out to measure.
    planned: usize,
    /// Windows that produced a finite loss.
    realised: usize,
}

impl EvalSample {
    fn dropped(&self) -> usize {
        self.planned.saturating_sub(self.realised)
    }
}

/// Opt-in that permits publishing a mean taken over fewer windows than were
/// planned. Off by default; see `require_complete_sample`.
const ALLOW_DROPPED_EVAL_WINDOWS_VAR: &str = "TRIOS_ALLOW_DROPPED_EVAL_WINDOWS";

/// Fault-injection hook, used by `tests/eval_window_drop_truth.rs`.
///
/// `TRIOS_TEST_POISON_EMBED_ROW=<token_id>` writes NaN into one embedding row
/// right after init, so exactly the eval windows containing that token go
/// non-finite. That PARTIAL poison is the case the in-file total-poison test
/// (`test_nan_forward_pass_yields_no_measurement`, which NaNs `lm_head[0]` and
/// so poisons every window) cannot reach, and there is no way to produce it
/// from outside the process without a hook. Any run that sets this announces
/// itself on stderr and stamps `fault_injected: true` into the results file:
/// nothing measured under this variable is a model result.
const POISON_EMBED_ROW_VAR: &str = "TRIOS_TEST_POISON_EMBED_ROW";

/// Grade one eval against the sample it planned.
///
/// `MIN_EVAL_CHUNKS` grades the PLAN -- a window count computed from the stream
/// length before a single forward pass -- and never the realisation. The
/// dropped-window filter in `eval_bpb` therefore used to shrink the sample in
/// silence: a NaN confined to one embedding row, or an overflow that only
/// occurs on certain contexts, drops exactly the affected windows and leaves a
/// plausible mean over the easy remainder. That bias is DOWNWARD -- the
/// direction that manufactures a champion. `src/train_loop.rs` takes the strict
/// line (the first non-finite window aborts the eval); these binaries keep the
/// per-window filter but refuse to publish what it produced unless the caller
/// asked for the reduced sample in writing.
fn require_complete_sample(
    label: &str,
    sample: Option<EvalSample>,
    val_path: &str,
    allow_dropped: bool,
) -> Result<EvalSample, u8> {
    let s = match sample {
        Some(s) => s,
        None => {
            eprintln!(
                "NO MEASUREMENT ({label}): zero finite eval windows on {val_path}. \
                 Refusing to report a BPB nobody measured."
            );
            return Err(EXIT_NO_MEASUREMENT);
        }
    };
    if s.dropped() > 0 && !allow_dropped {
        eprintln!(
            "EVAL SAMPLE SHRANK ({}): {} of {} planned windows on {} went \
             non-finite and were dropped. A mean over the survivors is biased \
             DOWNWARD and is not a held-out measurement. Set {}=1 to publish it \
             anyway; the results file then carries eval_windows_dropped.",
            label,
            s.dropped(),
            s.planned,
            val_path,
            ALLOW_DROPPED_EVAL_WINDOWS_VAR
        );
        return Err(EXIT_NO_MEASUREMENT);
    }
    Ok(s)
}

/// Exit codes. `0` only when a BPB was actually measured against a real corpus.
const EXIT_BAD_ARGS: u8 = 4;
const EXIT_BAD_FORMAT: u8 = 5;
const EXIT_BAD_CORPUS: u8 = 6;
const EXIT_NO_MEASUREMENT: u8 = 7;
const EXIT_IO: u8 = 8;

/// Usage text. Without it `--help` started a full 3000-step training run and
/// overwrote the git-tracked results file for `--format f32 --algo adamw
/// --seed 42`, because `arg_or` matches only `--name=value` and ignored every
/// argument it did not recognise. `src/bin/train_v2.rs` has had a `USAGE` for
/// exactly this reason; the binary CI actually spawns did not.
const USAGE: &str = "\
cpu_train - embed+bigram+smear+lm_head char model, BPB measured on a held-out corpus.

Usage: cpu_train [--seed=N] [--steps=N] [--lr=F] [--vocab=N] [--dim=N] [--seq=N]
                 [--algo=NAME] [--ffn] [--ffn-layers=N]
                 [--train-data=PATH] [--val-data=PATH]

Writes .trinity/results/cpu_train_<format>_<algo>_seed<seed>.json and exits 0
only when a BPB was measured over the FULL planned eval sample.
Env: TRIOS_FORMAT_TYPE, TRIOS_ALGO_TYPE, TRIOS_TRAIN_PATH, TRIOS_VAL_PATH,
     TRIOS_ALLOW_DROPPED_EVAL_WINDOWS=1 (publish a reduced eval sample).
Exit codes: 4 = bad argument, 5 = unknown format, 6 = corpus refused,
            7 = nothing measured / eval sample shrank, 8 = results I/O.";

/// Every argument this binary reads. `--ffn` is a bare flag; the rest are
/// `--name=value`.
const KNOWN_VALUE_ARGS: [&str; 10] = [
    "seed",
    "steps",
    "lr",
    "vocab",
    "dim",
    "seq",
    "algo",
    "ffn-layers",
    "train-data",
    "val-data",
];
const KNOWN_FLAG_ARGS: [&str; 3] = ["--ffn", "--help", "-h"];

/// Name the first argument that means nothing here, or `None` if all are known.
///
/// Silently ignoring an unrecognised argument is how `--help` became a
/// training run: the caller believes it asked for something, the binary does
/// something else, and only the overwritten results file records the
/// disagreement.
fn first_unknown_arg<I: IntoIterator<Item = String>>(args: I) -> Option<String> {
    args.into_iter().skip(1).find(|a| {
        if KNOWN_FLAG_ARGS.contains(&a.as_str()) {
            return false;
        }
        !KNOWN_VALUE_ARGS
            .iter()
            .any(|name| a.starts_with(&format!("--{name}=")))
    })
}

fn main() -> ExitCode {
    // Argument handling comes first, before any training and before any file
    // is opened: `--help` used to fall through to a 3000-step run whose
    // results file replaced the checked-in one.
    let argv: Vec<String> = std::env::args().collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", USAGE);
        return ExitCode::SUCCESS;
    }
    if let Some(bad) = first_unknown_arg(argv) {
        eprintln!("UNKNOWN ARGUMENT: {bad}");
        eprintln!(
            "cpu_train ignored unrecognised arguments, so a caller could ask for \
             one run and silently get another. Refusing instead."
        );
        eprintln!();
        eprintln!("{}", USAGE);
        return ExitCode::from(EXIT_BAD_ARGS);
    }

    let format_type = std::env::var("TRIOS_FORMAT_TYPE").ok();
    // Every one of these used to be `.parse().unwrap_or(<default>)`. A KNOWN
    // flag with an unusable value therefore trained at the default and exited
    // 0: `--seed=oops --lr=0,001` printed `seed=42 lr=0.003`. The seed and the
    // learning rate are the identity of a reproducibility claim, so an
    // unparseable value is a refusal. See `parse_flag_or_refuse`.
    let seed: u64 = match parse_flag_or_refuse("seed", "42") {
        Ok(v) => v,
        Err(code) => return code,
    };
    let steps: usize = match parse_flag_or_refuse("steps", "3000") {
        Ok(v) => v,
        Err(code) => return code,
    };
    let lr: f32 = match parse_flag_or_refuse("lr", "0.003") {
        Ok(v) => v,
        Err(code) => return code,
    };
    let vocab: usize = match parse_flag_or_refuse("vocab", "128") {
        Ok(v) => v,
        Err(code) => return code,
    };
    let dim: usize = match parse_flag_or_refuse("dim", "96") {
        Ok(v) => v,
        Err(code) => return code,
    };
    let seq: usize = match parse_flag_or_refuse("seq", "32") {
        Ok(v) => v,
        Err(code) => return code,
    };

    // Resolve algo name: CLI --algo=<name> takes precedence, then TRIOS_ALGO_TYPE, then "adamw"
    let algo_name_raw = arg_or("algo", "");
    let algo_name: String = if algo_name_raw.is_empty() {
        std::env::var("TRIOS_ALGO_TYPE").unwrap_or_else(|_| "adamw".to_string())
    } else {
        algo_name_raw
    };
    let algo_name: &str = &algo_name.clone();

    // R5-honest: announce algo at startup so CI logs can grep it
    println!("ALGO: {} enabled", algo_name);

    // Parse format type for QAT (FakeQuant + STE).
    //
    // An unrecognised TRIOS_FORMAT_TYPE used to fall back to F32 while the
    // caller (matrix_runner) still wrote the requested spelling into the
    // `format` column, so a whole matrix axis could be a claim rather than a
    // measurement. Unknown spellings are now a hard, named exit -- the same
    // contract `AlgoOpt::from_env` already enforces for optimizers.
    let default_format = "f32".to_string();
    let format_suffix = format_type.as_ref().unwrap_or(&default_format);
    let format_kind = match format_type.as_deref().map(str::trim) {
        None | Some("") => FormatKind::F32,
        Some(raw) => match FormatKind::from_env(raw) {
            Some(k) => k,
            None => {
                eprintln!(
                    "TRIOS_FORMAT_TYPE: unknown format '{}'. Refusing the silent \
                     F32 fallback that made the format column unverifiable.",
                    raw
                );
                eprintln!(
                    "Valid choices: {}",
                    FormatKind::all()
                        .iter()
                        .map(|f| f.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                return ExitCode::from(EXIT_BAD_FORMAT);
            }
        },
    };
    // The format that will actually be executed, as opposed to the string the
    // caller asked for. matrix_runner compares the two.
    let format_executed = format_kind.name();
    let use_fake_quant = format_kind != FormatKind::F32;

    if use_fake_quant {
        println!("QAT: FakeQuant enabled for format {:?}", format_kind);
    }
    println!(
        "FORMAT: requested={} executed={}",
        format_suffix, format_executed
    );

    // Corpus paths. Train and val are separate pinned files; the val split is
    // never carved out of the train stream here, so there is no positional
    // split to get wrong.
    let train_path = arg_or(
        "train-data",
        &std::env::var("TRIOS_TRAIN_PATH").unwrap_or_else(|_| DEFAULT_TRAIN_PATH.to_string()),
    );
    let val_path = arg_or(
        "val-data",
        &std::env::var("TRIOS_VAL_PATH").unwrap_or_else(|_| DEFAULT_VAL_PATH.to_string()),
    );
    if train_path == val_path {
        eprintln!(
            "CORPUS REFUSED: --train-data and --val-data are the same path ({}). \
             A val stream that is the train stream measures memorisation, not \
             generalisation.",
            train_path
        );
        return ExitCode::from(EXIT_BAD_CORPUS);
    }

    let raw_train = match load_data(&train_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("CORPUS REFUSED (train): {e}");
            return ExitCode::from(EXIT_BAD_CORPUS);
        }
    };
    let raw_val = match load_data(&val_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("CORPUS REFUSED (val): {e}");
            return ExitCode::from(EXIT_BAD_CORPUS);
        }
    };
    let train_tokens: Vec<usize> = raw_train.iter().map(|&t| t % vocab).collect();
    let val_tokens: Vec<usize> = raw_val.iter().map(|&t| t % vocab).collect();

    println!("=== trios CPU Training (Analytical Backprop) ===");
    println!(
        "vocab={} dim={} seq={} steps={} seed={} lr={}",
        vocab, dim, seq, steps, seed, lr
    );

    println!(
        "Dataset: {} train / {} val tokens (train={} val={})",
        train_tokens.len(),
        val_tokens.len(),
        train_path,
        val_path
    );

    // ---- Eval-corpus precondition -----------------------------------------
    // Mirrors train_loop::assert_train_val_disjoint's size checks. A val stream
    // too short to yield MIN_EVAL_CHUNKS windows cannot produce a mean that
    // means anything, and a truncated or degenerate corpus must fail loud
    // rather than be memorised into a near-zero BPB.
    if val_tokens.len() < MIN_VAL_TOKENS {
        eprintln!(
            "VAL STREAM TOO SHORT: {} tokens from {}, minimum {}. A BPB averaged \
             over a handful of windows is not a held-out measurement.",
            val_tokens.len(),
            val_path,
            MIN_VAL_TOKENS
        );
        return ExitCode::from(EXIT_BAD_CORPUS);
    }
    let chunks = eval_chunk_count(val_tokens.len(), seq);
    if chunks < MIN_EVAL_CHUNKS {
        eprintln!(
            "VAL STREAM YIELDS ONLY {} EVAL CHUNK(S) at seq={}, minimum {}. \
             eval_bpb would average over too few windows for the mean to be \
             informative.",
            chunks, seq, MIN_EVAL_CHUNKS
        );
        return ExitCode::from(EXIT_BAD_CORPUS);
    }
    if train_tokens.len() < seq + 2 {
        eprintln!(
            "TRAIN STREAM TOO SHORT: {} tokens from {}, need at least {} for one \
             batch at seq={}.",
            train_tokens.len(),
            train_path,
            seq + 2,
            seq
        );
        return ExitCode::from(EXIT_BAD_CORPUS);
    }
    println!("Eval chunks: {} (minimum {})", chunks, MIN_EVAL_CHUNKS);

    let train_tokens: &[usize] = &train_tokens;
    let val_tokens: &[usize] = &val_tokens;

    let mut model = CpuModel::new(vocab, dim, seed);

    // Whether a reduced eval sample may be published, decided once so every
    // eval in the run is graded the same way.
    let allow_dropped = std::env::var(ALLOW_DROPPED_EVAL_WINDOWS_VAR)
        .map(|v| v == "1")
        .unwrap_or(false);
    if allow_dropped {
        eprintln!(
            "{}=1: a reduced eval sample will be published. The mean is taken \
             over the windows that survived and is biased DOWNWARD.",
            ALLOW_DROPPED_EVAL_WINDOWS_VAR
        );
    }

    // Fault injection, off unless asked for by name. See POISON_EMBED_ROW_VAR.
    let fault_injected = match std::env::var(POISON_EMBED_ROW_VAR) {
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(row) if row < vocab => {
                for x in model.embed[row * dim..(row + 1) * dim].iter_mut() {
                    *x = f32::NAN;
                }
                eprintln!(
                    "FAULT INJECTED: {}={} put NaN into embedding row {}. This run \
                     is a fault-injection probe, not a model result.",
                    POISON_EMBED_ROW_VAR, row, row
                );
                true
            }
            _ => {
                eprintln!(
                    "{}: '{}' is not a token id below vocab={}.",
                    POISON_EMBED_ROW_VAR, raw, vocab
                );
                return ExitCode::from(EXIT_BAD_ARGS);
            }
        },
        Err(_) => false,
    };

    // Build both optimizers from algo_name (same algo for embed and head)
    let mut opt_embed = AlgoOpt::from_env(algo_name, vocab * dim, lr);
    let mut opt_head = AlgoOpt::from_env(algo_name, vocab * dim, lr);

    // Apply FakeQuant to initial weights (QAT)
    if use_fake_quant {
        fake_quant::fake_quantize_weights(&mut model.embed, format_kind);
        fake_quant::fake_quantize_weights(&mut model.lm_head, format_kind);
        for ffn in &mut model.ffn_layers {
            fake_quant::fake_quantize_weights(&mut ffn.w1, format_kind);
            fake_quant::fake_quantize_weights(&mut ffn.b1, format_kind);
            fake_quant::fake_quantize_weights(&mut ffn.w2, format_kind);
            fake_quant::fake_quantize_weights(&mut ffn.b2, format_kind);
        }
    }

    // Apply FakeQuant to initial weights (QAT)
    if use_fake_quant {
        fake_quant::fake_quantize_weights(&mut model.embed, format_kind);
        fake_quant::fake_quantize_weights(&mut model.lm_head, format_kind);
        for ffn in &mut model.ffn_layers {
            fake_quant::fake_quantize_weights(&mut ffn.w1, format_kind);
            fake_quant::fake_quantize_weights(&mut ffn.b1, format_kind);
            fake_quant::fake_quantize_weights(&mut ffn.w2, format_kind);
            fake_quant::fake_quantize_weights(&mut ffn.b2, format_kind);
        }
    }

    // Every eval in this run, including the initial one, must average over the
    // full planned window set; `dropped_total` accumulates what was lost when
    // the operator opted into a reduced sample.
    let mut dropped_total = 0usize;
    let init_sample = match require_complete_sample(
        "initial eval",
        model.eval_bpb(val_tokens, seq),
        &val_path,
        allow_dropped,
    ) {
        Ok(s) => s,
        Err(code) => return ExitCode::from(code),
    };
    dropped_total += init_sample.dropped();
    let init_bpb = init_sample.mean;
    println!(
        "Initial val BPB: {:.4} (eval windows {}/{})",
        init_bpb, init_sample.realised, init_sample.planned
    );
    println!();
    println!(
        "{:>6} | {:>10} | {:>10} | {:>10} | {:>8}",
        "step", "train_loss", "val_bpb", "best_bpb", "ms"
    );
    println!("{}", "-".repeat(60));

    let t0 = Instant::now();
    // `best_bpb` is the running MINIMUM over every eval; `final_bpb` is the
    // single value measured at `step == steps`. They are different numbers and
    // are now reported under different keys: the results file used to write the
    // minimum under the name `final_bpb`, and matrix_runner then paired it with
    // `step = steps`, attributing an early-training minimum to the last step.
    let mut best_bpb = init_bpb;
    let mut final_sample: Option<EvalSample> = None;
    let data_len = train_tokens.len();
    let mut rng_state = seed;

    for step in 1..=steps {
        let progress = step as f32 / steps as f32;
        let warmup = 0.05;
        let current_lr = if progress < warmup {
            lr * progress / warmup
        } else {
            let decay_progress = (progress - warmup) / (1.0 - warmup);
            lr * 0.5 * (1.0 + (std::f32::consts::PI * decay_progress).cos())
        };

        let offset = {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng_state as usize) % (data_len.saturating_sub(seq + 1))
        };
        let batch = &train_tokens[offset..offset + seq + 1];
        let train_loss = model.train_step(batch, &mut opt_embed, &mut opt_head, current_lr);

        // Apply FakeQuant after optimizer step (QAT)
        if use_fake_quant {
            fake_quant::fake_quantize_weights(&mut model.embed, format_kind);
            fake_quant::fake_quantize_weights(&mut model.lm_head, format_kind);
            for ffn in &mut model.ffn_layers {
                fake_quant::fake_quantize_weights(&mut ffn.w1, format_kind);
                fake_quant::fake_quantize_weights(&mut ffn.b1, format_kind);
                fake_quant::fake_quantize_weights(&mut ffn.w2, format_kind);
                fake_quant::fake_quantize_weights(&mut ffn.b2, format_kind);
            }
        }

        if step % 500 == 0 || step == steps {
            let ms = t0.elapsed().as_millis();
            // An intermediate eval is not a lesser measurement: it feeds
            // `best_bpb`, which is published. It is graded like the final one.
            let sample = match require_complete_sample(
                &format!("eval at step {step}"),
                model.eval_bpb(val_tokens, seq),
                &val_path,
                allow_dropped,
            ) {
                Ok(s) => s,
                Err(code) => return ExitCode::from(code),
            };
            dropped_total += sample.dropped();
            if sample.mean < best_bpb && sample.mean.is_finite() {
                best_bpb = sample.mean;
            }
            if step == steps {
                final_sample = Some(sample);
            }
            println!(
                "{:>6} | {:>10.4} | {:>10.4} | {:>10.4} | {:>6}ms",
                step, train_loss, sample.mean, best_bpb, ms
            );
        }
    }

    let total = t0.elapsed();

    // A run whose last step produced no finite eval window has no final BPB.
    // Emitting one anyway is exactly the class of defect this binary exists to
    // avoid, so the run fails instead.
    let final_sample = match final_sample {
        Some(s) => s,
        None => {
            eprintln!(
                "NO MEASUREMENT: the eval at the final step (step={}) produced zero \
                 finite windows. Refusing to write a results file with a final_bpb \
                 nobody measured.",
                steps
            );
            return ExitCode::from(EXIT_NO_MEASUREMENT);
        }
    };
    let final_bpb = final_sample.mean;

    println!();
    println!("=== Training Complete ===");
    println!(
        "Time: {:.1}s | Init BPB: {:.4} | Best BPB: {:.4} | Final BPB: {:.4} | \
         Delta(best): {:.4} | Delta(final): {:.4}",
        total.as_secs_f64(),
        init_bpb,
        best_bpb,
        final_bpb,
        init_bpb - best_bpb,
        init_bpb - final_bpb
    );

    let rpath = results_path(&format_suffix, algo_name, seed, dim, seq, steps, lr);
    if let Some(parent) = std::path::Path::new(&rpath).parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            eprintln!("create results dir {parent:?}: {e}");
            return ExitCode::from(EXIT_IO);
        }
    }
    let result_json = serde_json::json!({
        "experiment": "cpu-backprop-scalable",
        "model": "embed+bigram+smear+lm_head",
        "algo": algo_name,
        "seed": seed,
        "vocab_size": vocab,
        "dim": dim,
        "seq_len": seq,
        "steps": steps,
        "lr": lr,
        "train_path": train_path,
        "val_path": val_path,
        "train_tokens": train_tokens.len(),
        "val_tokens": val_tokens.len(),
        // `eval_chunks` is the PLAN, computed from the stream length before a
        // single forward pass. The two keys below are the REALISATION of the
        // final eval: how many windows it set out to measure and how many
        // produced a finite loss. A consumer comparing runs must check that
        // realised == planned before comparing the means.
        "eval_chunks": chunks,
        "eval_windows_planned": final_sample.planned,
        "eval_windows_realised": final_sample.realised,
        "eval_windows_dropped": final_sample.dropped(),
        // Dropped windows across every eval in the run (init + intermediate +
        // final), since `best_bpb` is drawn from the intermediate ones.
        "eval_windows_dropped_total": dropped_total,
        // True only when TRIOS_TEST_POISON_EMBED_ROW was set. Nothing measured
        // under fault injection is a model result.
        "fault_injected": fault_injected,
        "format_requested": format_suffix,
        "format_executed": format_executed,
        "initial_bpb": init_bpb,
        // `best_bpb`: minimum over all evals, at an unrecorded step.
        // `final_bpb`: the value measured at step == steps. Consumers that pair
        // a bpb with `steps` must read `final_bpb`.
        "best_bpb": best_bpb,
        "final_bpb": final_bpb,
        // `delta_bpb` is kept for schema compatibility and is paired with
        // `final_bpb`; `delta_best_bpb` is the improvement to the minimum.
        "delta_bpb": init_bpb - final_bpb,
        "delta_best_bpb": init_bpb - best_bpb,
        "duration_seconds": total.as_secs_f64(),
    });

    let serialized = match serde_json::to_string_pretty(&result_json) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("serialize results: {e}");
            return ExitCode::from(EXIT_IO);
        }
    };
    match fs::File::create(&rpath).and_then(|mut f| f.write_all(serialized.as_bytes())) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("write results {rpath}: {e}");
            return ExitCode::from(EXIT_IO);
        }
    }
    println!("Results: {}", rpath);
    ExitCode::SUCCESS
}

fn arg_or(name: &str, default: &str) -> String {
    let prefix = format!("--{}=", name);
    std::env::args()
        .find(|a| a.starts_with(&prefix))
        .map(|a| a[prefix.len()..].to_string())
        .unwrap_or_else(|| default.to_string())
}

/// Read a known flag and parse it, or exit with the same refusal shape as
/// `first_unknown_arg`.
///
/// `first_unknown_arg` already refuses a flag NAME this binary does not know.
/// A name it DOES know carrying a value it cannot use was the remaining silent
/// path, and it is the worse one: an unknown name is at least visible in the
/// argv, whereas `--seed=oops` produced a complete, plausible, git-tracked
/// results file for a run nobody asked for.
fn parse_flag_or_refuse<T: std::str::FromStr>(name: &str, default: &str) -> Result<T, ExitCode> {
    let raw = arg_or(name, default);
    trios_trainer::parse_flag_value::<T>(name, &raw).map_err(|e| {
        eprintln!("{e}");
        eprintln!();
        eprintln!("{}", USAGE);
        ExitCode::from(EXIT_BAD_ARGS)
    })
}

/// Directory the results JSON is written to. Defaults to `.trinity/results`,
/// which `.gitignore` excludes.
const RESULTS_DIR_VAR: &str = "TRIOS_RESULTS_DIR";
const DEFAULT_RESULTS_DIR: &str = ".trinity/results";

/// Render a float for use inside a filename: shortest form that round-trips,
/// with the decimal point kept (it is legal in every filesystem this runs on)
/// so `0.01` and `0.001` cannot collapse onto each other.
fn lr_file_token(lr: f32) -> String {
    format!("{lr}")
}

/// Path of the results file for ONE cell of the matrix.
///
/// This used to be `cpu_train_{format}_{algo}_seed{seed}.json`: a name keyed on
/// 3 of the 7 parameters that define the run. Two cells differing only in
/// `dim`, `seq`, `steps` or `lr` wrote to the SAME path and the last one won,
/// and `matrix_runner` then read that file back to verify the executed `lr` -
/// which is the mechanism behind the reported matrix flake. Eight of those
/// legacy names are also git-TRACKED (they predate the `.trinity/results/`
/// ignore rule), so a five-step probe run overwrote checked-in evidence.
///
/// The full identity is in the name now, which makes the legacy spellings
/// unreachable from this binary and makes cross-cell collision impossible.
/// `matrix_runner` no longer reconstructs this string at all: it reads the
/// `Results:` line this binary prints, so the reader uses the writer's own
/// statement of where it wrote.
fn results_path(
    format_suffix: &str,
    algo_name: &str,
    seed: u64,
    dim: usize,
    seq: usize,
    steps: usize,
    lr: f32,
) -> String {
    let dir = std::env::var(RESULTS_DIR_VAR)
        .ok()
        .filter(|d| !d.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_RESULTS_DIR.to_string());
    format!(
        "{dir}/cpu_train_{format_suffix}_{algo_name}_seed{seed}_dim{dim}_seq{seq}\
         _steps{steps}_lr{}.json",
        lr_file_token(lr)
    )
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {

    /// The probability floor that was doing the laundering, named so that a
    /// search for a clamp in a measurement path finds nothing outside this
    /// regression test. The value is the point of the test: `-ln(1e-10)` is
    /// 23.02585 nats and 33.21928 bpb, the crate's fake-measurement signature.
    const LAUNDER_FLOOR: f32 = 1e-10;
    use super::*;

    // test_algoopt_dispatch -- each variant constructible, step() doesn't panic on 10-param vec
    #[test]
    fn test_algoopt_dispatch() {
        let size = 10;
        let lr = 0.01f32;
        let names = [
            "adamw",
            "muon",
            "sgdm",
            "lion",
            "adafactor",
            "lamb",
            "schedulefree",
            "rmsprop",
            "soap",
        ];
        for &name in &names {
            let mut opt = AlgoOpt::from_env(name, size, lr);
            let mut params = vec![0.5f32; size];
            let grads = vec![0.1f32; size];
            opt.step(&mut params, &grads);
            // Should not panic and params should be finite
            for &p in &params {
                assert!(p.is_finite(), "algo={} produced non-finite param", name);
            }
        }
    }

    // test_sgdm_recovers_minimum -- SGDM on f(x)=x^2 converges to 0 in <1000 steps
    #[test]
    fn test_sgdm_recovers_minimum() {
        let size = 1;
        let mut opt = Sgdm::new(size, 0.01);
        let mut params = vec![5.0f32];
        for _ in 0..1000 {
            let grad = vec![2.0 * params[0]]; // grad of x^2 = 2x
            opt.step(&mut params, &grad);
            if params[0].abs() < 1e-3 {
                return; // converged
            }
        }
        assert!(
            params[0].abs() < 1e-2,
            "SGDM did not converge: x = {}",
            params[0]
        );
    }

    // test_lion_sign_update -- Lion with grad=+1 monotonically decreases param toward -inf
    #[test]
    fn test_lion_sign_update() {
        let size = 1;
        let mut opt = Lion::new(size, 0.1);
        // Disable weight decay for clean test
        opt.wd = 0.0;
        let mut params = vec![10.0f32];
        let grads = vec![1.0f32]; // positive gradient -> negative sign update -> param decreases
        let mut prev = params[0];
        for _ in 0..20 {
            opt.step(&mut params, &grads);
            assert!(
                params[0] <= prev,
                "Lion param should decrease: prev={} now={}",
                prev,
                params[0]
            );
            prev = params[0];
        }
    }

    // test_lamb_trust_ratio -- LAMB scales update by ||w||/||u||
    #[test]
    fn test_lamb_trust_ratio() {
        let size = 4;
        let lr = 0.1f32;
        let mut opt = Lamb::new(size, lr);
        // large param norm, small grad -> trust ratio > 1 -> big step
        let mut params_big = vec![100.0f32; size];
        let grads = vec![0.01f32; size];
        let before: Vec<f32> = params_big.clone();
        opt.step(&mut params_big, &grads);

        // Reset and test with small param norm
        let mut opt2 = Lamb::new(size, lr);
        let mut params_small = vec![0.01f32; size];
        let grads2 = vec![0.01f32; size];
        let before2: Vec<f32> = params_small.clone();
        opt2.step(&mut params_small, &grads2);

        let delta_big = (before[0] - params_big[0]).abs();
        let delta_small = (before2[0] - params_small[0]).abs();

        // With large params (||w|| >> ||u||), the step should be larger
        assert!(
            delta_big > delta_small,
            "LAMB trust ratio test: big={} small={}",
            delta_big,
            delta_small
        );
    }

    // test_unknown_algo_panics -- AlgoOpt::from_env("foobar", ...) panics
    #[test]
    #[should_panic(expected = "unknown optimizer")]
    fn test_unknown_algo_panics() {
        let _ = AlgoOpt::from_env("foobar", 10, 0.01);
    }

    // Additional: ScheduleFree steps without panic
    #[test]
    fn test_schedulefree_steps() {
        let mut opt = ScheduleFree::new(10, 0.01);
        let mut params = vec![1.0f32; 10];
        let grads = vec![0.1f32; 10];
        for _ in 0..5 {
            opt.step(&mut params, &grads);
        }
        for &p in &params {
            assert!(p.is_finite());
        }
    }

    // Additional: Adafactor on 1D (non-factored path)
    #[test]
    fn test_adafactor_1d() {
        let mut opt = Adafactor::new(3, 0.01);
        let mut params = vec![1.0f32; 3];
        let grads = vec![0.1f32; 3];
        opt.step(&mut params, &grads);
        for &p in &params {
            assert!(p.is_finite());
        }
    }

    // Additional: RMSprop decreases param with consistent positive grad
    #[test]
    fn test_rmsprop_decreases() {
        let mut opt = RmsProp::new(1, 0.01);
        let mut params = vec![5.0f32];
        let grads = vec![1.0f32];
        let before = params[0];
        opt.step(&mut params, &grads);
        assert!(params[0] < before, "RMSprop should decrease param");
    }

    // test_soap_recovers_minimum -- SOAP on f(x)=x^2 converges with positive grad
    #[test]
    fn test_soap_recovers_minimum() {
        let mut opt = Soap::new(1, 0.05);
        let mut params = vec![5.0f32];
        for _ in 0..2000 {
            let grad = vec![2.0 * params[0]]; // grad of x^2 = 2x
            opt.step(&mut params, &grad);
            if params[0].abs() < 1e-2 {
                return;
            }
        }
        assert!(
            params[0].abs() < 5e-1,
            "SOAP did not converge: x = {}",
            params[0]
        );
    }

    // test_soap_precond_refresh -- preconditioner refreshes every K steps
    #[test]
    fn test_soap_precond_refresh() {
        let mut opt = Soap::new(2, 0.01);
        // After precond_freq=10 steps, opt.precond should differ from its
        // initial all-ones state if non-zero gradients have been observed.
        let mut params = vec![1.0f32, -1.0f32];
        for _ in 0..15 {
            let grads = vec![0.5f32, -0.5f32];
            opt.step(&mut params, &grads);
        }
        assert!(
            opt.precond.iter().any(|&p| (p - 1.0).abs() > 1e-9),
            "SOAP preconditioner should refresh after >precond_freq steps"
        );
    }

    // test_load_data_refuses_missing -- no fallback corpus, ever
    #[test]
    fn test_load_data_refuses_missing() {
        let err = load_data("data/this_file_does_not_exist_zzz.txt")
            .expect_err("a missing corpus must be an error, not a fabricated one");
        assert!(
            err.contains("data/this_file_does_not_exist_zzz.txt"),
            "error must name the path: {err}"
        );
        assert!(
            err.contains("no fallback corpus"),
            "error must state that no fallback exists: {err}"
        );
    }

    // test_default_corpus_paths_use_the_file_that_exists
    #[test]
    fn test_default_corpus_paths_spelling() {
        // The historical default omitted the underscore in "tiny_shakespeare",
        // a path that has never existed. Both defaults carry the underscore.
        assert_eq!(DEFAULT_TRAIN_PATH, "data/tiny_shakespeare.txt");
        assert_eq!(DEFAULT_VAL_PATH, "data/tiny_shakespeare_val.txt");
        assert_ne!(DEFAULT_TRAIN_PATH, DEFAULT_VAL_PATH);
    }

    // test_eval_chunk_count_matches_precondition
    #[test]
    fn test_eval_chunk_count_matches_precondition() {
        // The 160-byte pangram fixture yields nowhere near MIN_EVAL_CHUNKS at
        // the default seq, which is the whole point of the precondition.
        assert!(eval_chunk_count(160, 32) < MIN_EVAL_CHUNKS);
        // A stream at the size floor clears both checks.
        assert!(eval_chunk_count(MIN_VAL_TOKENS, 32) >= MIN_EVAL_CHUNKS);
        assert!(eval_chunk_count(MIN_VAL_TOKENS, 8) >= MIN_EVAL_CHUNKS);
        // The cap is respected: more tokens than EVAL_TOKEN_CAP does not add
        // chunks.
        assert_eq!(
            eval_chunk_count(EVAL_TOKEN_CAP, 32),
            eval_chunk_count(EVAL_TOKEN_CAP * 10, 32)
        );
        // Empty stream measures nothing.
        assert_eq!(eval_chunk_count(0, 32), 0);
    }

    // test_eval_bpb_absent_is_none -- not f32::MAX
    #[test]
    fn test_eval_bpb_absent_is_none() {
        let model = CpuModel::new(32, 8, 47);
        // Fewer than 3 tokens: no window can be evaluated.
        assert!(model.eval_bpb(&[1, 2], 8).is_none());
        assert!(model.eval_bpb(&[], 8).is_none());
        // A real stream does produce a reading.
        let tokens: Vec<usize> = (0..1024).map(|i| i % 32).collect();
        let sample = model.eval_bpb(&tokens, 8).expect("measurable stream");
        let bpb = sample.mean;
        assert!(bpb.is_finite());
        // And it is nowhere near the sentinel the old code returned.
        assert!(bpb < 64.0, "bpb={bpb} looks like a sentinel, not a reading");
        // A healthy model drops nothing: the mean covers the whole plan.
        assert_eq!(sample.realised, sample.planned);
        assert_eq!(sample.dropped(), 0);
        assert_eq!(sample.planned, eval_chunk_count(tokens.len(), 8));
    }

    /// The exact laundering the guard in `loss_and_grad` removes: `f32::max`
    /// returns the non-NaN operand, so clamping `NaN` to a 1e-10 floor yields `1e-10`, whose
    /// negative log is 23.026 nats and whose BPB is 33.2 - positive, finite,
    /// and below `BPB_SENTINEL_CEILING`, so `eval_bpb`'s `is_finite()` filter
    /// passed it straight through to the results file.
    #[test]
    fn test_f32_max_launders_nan_into_a_publishable_bpb() {
        let laundered = f32::NAN.max(LAUNDER_FLOOR);
        assert_eq!(laundered, LAUNDER_FLOOR, "f32::max ignores NaN");
        let bpb = -laundered.ln() / LN_2;
        assert!(
            (bpb - 33.2).abs() < 0.05,
            "the laundered reading is 33.2 bpb, got {bpb}"
        );
        assert!(
            bpb > 0.0 && bpb < 64.0,
            "and it passes every downstream guard"
        );
    }

    /// A NaN forward pass produces no measurement, not 33.2 bpb.
    #[test]
    fn test_nan_forward_pass_yields_no_measurement() {
        let tokens: Vec<usize> = (0..1024).map(|i| i % 32).collect();

        let healthy = CpuModel::new(32, 8, 47);
        let (healthy_loss, _, _) = healthy.loss_and_grad(&tokens[..9]);
        assert!(
            healthy_loss.is_finite(),
            "the fixture must be measurable before the NaN is introduced"
        );
        assert!(healthy.eval_bpb(&tokens, 8).is_some());

        let mut model = CpuModel::new(32, 8, 47);
        model.lm_head[0] = f32::NAN;

        let (loss, _, _) = model.loss_and_grad(&tokens[..9]);
        assert!(
            loss.is_nan(),
            "a poisoned forward pass is not a loss: {loss}"
        );
        assert!(
            (loss - 23.026).abs() > 1.0 || loss.is_nan(),
            "23.026 nats is the laundered NaN, not a loss: {loss}"
        );

        let bpb = model.eval_bpb(&tokens, 8);
        assert_eq!(bpb, None, "a poisoned model publishes nothing");
        if let Some(s) = bpb {
            assert!(
                (s.mean - 33.2).abs() > 1.0,
                "33.2 bpb is the laundered NaN, not a measurement: {}",
                s.mean
            );
        }
    }

    /// The case the total poison above cannot reach: ONE embedding row is
    /// NaN, so only the windows containing that token go non-finite.
    ///
    /// `eval_bpb` still returns a mean here -- the survivors are real windows
    /// and their losses are finite -- and before the realised count travelled
    /// with it, that mean was published as though it covered the whole eval.
    /// It does not, and the windows it drops are not a random subset.
    #[test]
    fn test_partial_poison_shrinks_the_sample_and_says_so() {
        let vocab = 32usize;
        let dim = 8usize;
        // Token 31 appears in roughly one window in eight, so most windows
        // stay measurable and the mean over them stays plausible.
        let tokens: Vec<usize> = (0..1024)
            .map(|i| if i % 71 == 0 { 31 } else { i % 30 })
            .collect();

        let healthy = CpuModel::new(vocab, dim, 47);
        let clean = healthy.eval_bpb(&tokens, 8).expect("measurable stream");
        assert_eq!(clean.realised, clean.planned, "fixture must start complete");

        let mut model = CpuModel::new(vocab, dim, 47);
        for x in model.embed[31 * dim..32 * dim].iter_mut() {
            *x = f32::NAN;
        }
        let poisoned = model
            .eval_bpb(&tokens, 8)
            .expect("a partial poison still leaves measurable windows");
        assert!(
            poisoned.dropped() > 0,
            "the poisoned token must cost some windows"
        );
        assert!(
            poisoned.realised > 0,
            "and must not cost all of them, or this is the total-poison case"
        );
        assert_eq!(poisoned.planned, clean.planned, "the PLAN is unchanged");
        assert!(
            poisoned.mean.is_finite(),
            "the mean over the survivors is finite -- that is the whole problem"
        );

        // The plan-level precondition cannot see any of this: it is computed
        // from the stream length and is identical either way.
        assert_eq!(
            eval_chunk_count(tokens.len(), 8),
            poisoned.planned,
            "MIN_EVAL_CHUNKS grades this number, which the poison does not move"
        );

        // So the refusal has to live where the realisation is known.
        assert_eq!(
            require_complete_sample("test", Some(poisoned), "fixture", false),
            Err(EXIT_NO_MEASUREMENT),
            "a shrunken sample is refused by default"
        );
        assert_eq!(
            require_complete_sample("test", Some(poisoned), "fixture", true),
            Ok(poisoned),
            "and published only when the operator opts in"
        );
    }

    // test_unknown_format_has_no_silent_fallback -- FormatKind resolution
    #[test]
    fn test_unknown_format_has_no_silent_fallback() {
        assert!(FormatKind::from_env("bogusfmt").is_none());
        assert!(FormatKind::from_env("surveyprobe2").is_none());
        assert!(FormatKind::from_env("gf16").is_some());
        assert_eq!(FormatKind::from_env("f32"), Some(FormatKind::F32));
    }
}
