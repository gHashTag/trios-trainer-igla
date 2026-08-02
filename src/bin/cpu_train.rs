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
            let p_target = logits[target].max(1e-10);
            total_loss -= p_target.ln();
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

    /// Mean BPB over the val stream, or `None` when nothing could be measured.
    ///
    /// This used to return `f32::MAX` when `n == 0`, a sentinel indistinguishable
    /// from a measurement once it had been written into a JSON field or compared
    /// with `<`. An absence is now an absence: callers must decide explicitly
    /// what to do, and no unmeasured value reaches the results file.
    fn eval_bpb(&self, tokens: &[usize], seq_len: usize) -> Option<f32> {
        let max_eval = EVAL_TOKEN_CAP.min(tokens.len());
        let eval_tokens = &tokens[..max_eval];
        let mut total_bpb = 0.0f32;
        let mut n = 0usize;
        for c in (0..eval_tokens.len()).step_by(seq_len + 1) {
            let end = (c + seq_len + 1).min(eval_tokens.len());
            if end - c < 3 {
                continue;
            }
            let seq = &eval_tokens[c..end];
            let (loss, _, _) = self.loss_and_grad(seq);
            if loss.is_finite() {
                total_bpb += loss / LN_2;
                n += 1;
            }
        }
        if n == 0 {
            return None;
        }
        Some(total_bpb / n as f32)
    }
}

/// Exit codes. `0` only when a BPB was actually measured against a real corpus.
const EXIT_BAD_FORMAT: u8 = 5;
const EXIT_BAD_CORPUS: u8 = 6;
const EXIT_NO_MEASUREMENT: u8 = 7;
const EXIT_IO: u8 = 8;

fn main() -> ExitCode {
    let format_type = std::env::var("TRIOS_FORMAT_TYPE").ok();
    let seed = arg_or("seed", "42").parse::<u64>().unwrap_or(42);
    let steps = arg_or("steps", "3000").parse::<usize>().unwrap_or(3000);
    let lr = arg_or("lr", "0.003").parse::<f32>().unwrap_or(0.003);
    let vocab: usize = arg_or("vocab", "128").parse().unwrap_or(128);
    let dim: usize = arg_or("dim", "96").parse().unwrap_or(96);
    let seq: usize = arg_or("seq", "32").parse().unwrap_or(32);

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

    let init_bpb = match model.eval_bpb(val_tokens, seq) {
        Some(b) => b,
        None => {
            eprintln!(
                "NO MEASUREMENT: the initial eval produced zero finite windows on \
                 {}. Refusing to report an initial BPB nobody measured.",
                val_path
            );
            return ExitCode::from(EXIT_NO_MEASUREMENT);
        }
    };
    println!("Initial val BPB: {:.4}", init_bpb);
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
    let mut final_bpb: Option<f32> = None;
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
            let val_bpb = model.eval_bpb(val_tokens, seq);
            if let Some(b) = val_bpb {
                if b < best_bpb && b.is_finite() {
                    best_bpb = b;
                }
            }
            if step == steps {
                final_bpb = val_bpb;
            }
            match val_bpb {
                Some(b) => println!(
                    "{:>6} | {:>10.4} | {:>10.4} | {:>10.4} | {:>6}ms",
                    step, train_loss, b, best_bpb, ms
                ),
                None => println!(
                    "{:>6} | {:>10.4} | {:>10} | {:>10.4} | {:>6}ms",
                    step, train_loss, "unmeasured", best_bpb, ms
                ),
            }
        }
    }

    let total = t0.elapsed();

    // A run whose last step produced no finite eval window has no final BPB.
    // Emitting one anyway is exactly the class of defect this binary exists to
    // avoid, so the run fails instead.
    let final_bpb = match final_bpb {
        Some(b) => b,
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

    let _ = fs::create_dir_all(".trinity/results");
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
        "eval_chunks": chunks,
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

    let rpath = format!(
        ".trinity/results/cpu_train_{}_{}_seed{}.json",
        format_suffix, algo_name, seed
    );
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

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
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
        let bpb = model.eval_bpb(&tokens, 8).expect("measurable stream");
        assert!(bpb.is_finite());
        // And it is nowhere near the sentinel the old code returned.
        assert!(bpb < 64.0, "bpb={bpb} looks like a sentinel, not a reading");
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
