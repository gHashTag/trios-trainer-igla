//! Hybrid n-gram + attention model for standalone CPU training.
//!
//! Single-file implementation: HybridModel, AdamW, gradient computation,
//! evaluation, GF16 floor, cosine LR, data loading.
//! Migrated from trios-train-cpu/src/bin/hybrid_train.rs (L-f2).

use anyhow::Result;

pub const VOCAB: usize = 128;
pub const DIM: usize = 64;
pub const NUM_CTX: usize = 6;
pub const NGRAM: usize = NUM_CTX + 2;
pub const PHI_INV: f32 = 0.618033988749895;
pub const CTX_WEIGHTS: [f32; NUM_CTX] = [0.70, 0.45, 0.30, 0.20, 0.13, 0.08];
const LN_2: f32 = std::f32::consts::LN_2;

pub fn load_data(path: &str) -> Result<Vec<usize>> {
    let raw = std::fs::read(path)?;
    Ok(raw.into_iter().map(|b| (b as usize) % VOCAB).collect())
}

pub fn load_data_fallback(path: &str) -> Vec<usize> {
    load_data(path).unwrap_or_else(|_| {
        b"The quick brown fox jumps over the lazy dog. "
            .repeat(100)
            .into_iter()
            .map(|b| (b as usize) % VOCAB)
            .collect()
    })
}

fn ln_(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let m = x.iter().sum::<f32>() / n;
    let v = x.iter().map(|v| (v - m).powi(2)).sum::<f32>() / n;
    let si = 1.0 / (v + eps).sqrt();
    x.iter().map(|v| (v - m) * si).collect()
}

fn ln_bw(x: &[f32], y: &[f32], dy: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let m = x.iter().sum::<f32>() / n;
    let v = x.iter().map(|v| (v - m).powi(2)).sum::<f32>() / n;
    let si = 1.0 / (v + eps).sqrt();
    let sd: f32 = dy.iter().sum();
    let sdy: f32 = dy.iter().zip(y.iter()).map(|(d, yi)| d * yi).sum();
    dy.iter()
        .zip(y.iter())
        .map(|(d, yi)| (d - sd / n - yi * sdy / n) * si)
        .collect()
}

fn smax(v: &mut [f32]) {
    let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - mx).exp();
        s += *x;
    }
    for x in v.iter_mut() {
        *x /= s;
    }
}

pub fn cosine_lr(step: usize, max_steps: usize, base_lr: f32, warmup: usize) -> f32 {
    if step < warmup {
        return base_lr * step as f32 / warmup.max(1) as f32;
    }
    let p = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
    1e-5 + (base_lr - 1e-5) * 0.5 * (1.0 + (std::f32::consts::PI * p).cos())
}

pub struct AdamW {
    m: Vec<f32>,
    v: Vec<f32>,
    step: usize,
    beta1: f32,
    beta2: f32,
    wd: f32,
}

impl AdamW {
    pub fn new(size: usize, wd: f32) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            step: 0,
            beta1: 0.9,
            beta2: 0.999,
            wd,
        }
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
}

pub struct HybridModel {
    pub embed: Vec<f32>,
    pub ctx: Vec<Vec<f32>>,
    pub proj: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub hidden: usize,
    attn_layers: u8,
    seq_len: usize,
}

impl HybridModel {
    pub fn new(hidden: usize, seed: u64, attn_layers: u8, seq_len: usize) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let lim = (6.0f32 / (3 * DIM) as f32).sqrt();
        let lim_h = (6.0f32 / (DIM + hidden) as f32).sqrt();
        let lim_o = (6.0f32 / (hidden + VOCAB) as f32).sqrt();
        Self {
            embed: (0..VOCAB * DIM).map(|_| rng() * lim).collect(),
            ctx: (0..NUM_CTX)
                .map(|_| (0..VOCAB * DIM).map(|_| rng() * lim).collect())
                .collect(),
            proj: (0..hidden * DIM).map(|_| rng() * lim_h).collect(),
            lm_head: (0..VOCAB * hidden)
                .map(|_| rng() * lim_o)
                .collect(),
            hidden,
            attn_layers,
            seq_len,
        }
    }
    pub fn total_params(&self) -> usize {
        let mut n = self.embed.len() + self.proj.len() + self.lm_head.len();
        for c in &self.ctx {
            n += c.len();
        }
        n
    }
    fn fwd_pos(&self, tokens: &[usize], pos: usize) -> Vec<f32> {
        let h = self.hidden;
        let tl = tokens[pos + NGRAM - 1].min(VOCAB - 1);
        let mut c = self.embed[tl * DIM..(tl + 1) * DIM].to_vec();
        for (ci, cw) in CTX_WEIGHTS.iter().enumerate() {
            let ci2 = NGRAM - 2 - ci;
            let t = tokens[pos + ci2].min(VOCAB - 1);
            let cv = &self.ctx[ci][t * DIM..(t + 1) * DIM];
            for j in 0..DIM {
                c[j] += cv[j] * cw;
            }
        }
        let n = ln_(&c, 1e-5);
        let mut hr = vec![0.0f32; h];
        for hi in 0..h {
            for j in 0..DIM {
                hr[hi] += self.proj[hi * DIM + j] * n[j];
            }
        }
        let mut hd = vec![0.0f32; h];
        for hi in 0..h {
            hd[hi] = if hr[hi] > 0.0 { hr[hi] * hr[hi] } else { 0.0 };
        }
        hd
    }
    fn loss_on_seq(&self, tokens: &[usize]) -> f32 {
        if tokens.len() < NGRAM + 1 {
            return 0.0;
        }
        let h = self.hidden;
        let ct = tokens.len().saturating_sub(NGRAM);
        let mut t = 0.0f32;
        for i in 0..ct {
            let tgt = tokens[i + NGRAM].min(VOCAB - 1);
            let hd = self.fwd_pos(tokens, i);
            let mut lo = vec![0.0f32; VOCAB];
            for vi in 0..VOCAB {
                for hi in 0..h {
                    lo[vi] += self.lm_head[vi * h + hi] * hd[hi];
                }
            }
            smax(&mut lo);
            t -= lo[tgt].max(1e-10).ln();
        }
        t / ct.max(1) as f32
    }
}

pub fn compute_grads(
    model: &HybridModel,
    tokens: &[usize],
    positions: &[usize],
    ge: &mut [f32],
    gc: &mut [Vec<f32>],
    gp: &mut [f32],
    gh: &mut [f32],
) {
    let h = model.hidden;
    for &pos in positions {
        let tl = tokens[pos + NGRAM - 1].min(VOCAB - 1);
        let mut c = model.embed[tl * DIM..(tl + 1) * DIM].to_vec();
        for (ci, cw) in CTX_WEIGHTS.iter().enumerate() {
            let ci2 = NGRAM - 2 - ci;
            let t = tokens[pos + ci2].min(VOCAB - 1);
            let cv = &model.ctx[ci][t * DIM..(t + 1) * DIM];
            for j in 0..DIM {
                c[j] += cv[j] * cw;
            }
        }
        let n = ln_(&c, 1e-5);
        let mut hr = vec![0.0f32; h];
        for hi in 0..h {
            for j in 0..DIM {
                hr[hi] += model.proj[hi * DIM + j] * n[j];
            }
        }
        let mut hd = vec![0.0f32; h];
        for hi in 0..h {
            hd[hi] = if hr[hi] > 0.0 { hr[hi] * hr[hi] } else { 0.0 };
        }
        let mut lo = vec![0.0f32; VOCAB];
        for vi in 0..VOCAB {
            for hi in 0..h {
                lo[vi] += model.lm_head[vi * h + hi] * hd[hi];
            }
        }
        smax(&mut lo);
        let tgt = tokens[pos + NGRAM].min(VOCAB - 1);
        let mut dh = vec![0.0f32; h];
        for vi in 0..VOCAB {
            let g = lo[vi] - if vi == tgt { 1.0 } else { 0.0 };
            for hi in 0..h {
                gh[vi * h + hi] += g * hd[hi];
                dh[hi] += g * model.lm_head[vi * h + hi];
            }
        }
        let mut dr = vec![0.0f32; h];
        for hi in 0..h {
            if hd[hi] > 0.0 {
                dr[hi] = dh[hi] * 2.0 * hd[hi].sqrt();
            }
        }
        let mut dl = vec![0.0f32; DIM];
        for hi in 0..h {
            for j in 0..DIM {
                gp[hi * DIM + j] += dr[hi] * n[j];
                dl[j] += model.proj[hi * DIM + j] * dr[hi];
            }
        }
        let dc = ln_bw(&c, &n, &dl, 1e-5);
        for j in 0..DIM {
            ge[tl * DIM + j] += dc[j];
        }
        for (ci, cw) in CTX_WEIGHTS.iter().enumerate() {
            let ci2 = NGRAM - 2 - ci;
            let t = tokens[pos + ci2].min(VOCAB - 1);
            for j in 0..DIM {
                gc[ci][t * DIM + j] += cw * dc[j];
            }
        }
    }
}

pub fn evaluate(model: &HybridModel, tokens: &[usize], seq_len: usize) -> f32 {
    let cs = seq_len + 1;
    let nc = 40usize;
    let ms = tokens.len().saturating_sub(cs);
    if ms == 0 {
        return f32::MAX;
    }
    let st = if ms >= nc * cs { ms / nc } else { cs };
    let mut t = 0.0f32;
    let mut n = 0usize;
    for c in (0..ms).step_by(st).take(nc) {
        let e = (c + cs).min(tokens.len());
        if e - c < NGRAM + 2 {
            continue;
        }
        let l = model.loss_on_seq(&tokens[c..e]);
        if l.is_finite() {
            t += l / LN_2;
            n += 1;
        }
    }
    if n == 0 {
        f32::MAX
    } else {
        t / n as f32
    }
}

pub fn gf16_floor(p: &mut [f32]) {
    let f = 1.0 / ((1.0f32 + 5.0f32.sqrt()) / 2.0).powi(6);
    for v in p.iter_mut() {
        *v = v.signum() * v.abs().max(f);
    }
}

pub fn build(_cfg: &crate::config::ModelConfig) -> Result<HybridModel> {
    Ok(HybridModel::new(828, 42, 2, 256))
}
