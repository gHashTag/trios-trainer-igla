use anyhow::Result;
use std::io::Read;
use std::time::Instant;

use crate::arch_config::{parse_gf16_enabled, parse_hidden_dim, parse_num_attn_layers};
use crate::fake_quant::{self, FormatKind};
use crate::model_hybrid_attn::{AttentionCache, HybridAttn};
use crate::objective::{nca_entropy_loss, NcaObjective};

pub const DEFAULT_IGLA_TARGET_BPB: f64 = 1.85;
/// Canon #93 sweep seeds — Lucas/Fibonacci aligned.
/// Forbidden under Canon #93: `{42, 43, 44, 45}`.
/// Allowed canon (Wave-29):  `{47, 89, 123, 144}`.
/// Sweep uses the first three; `144` is reserved for the bridge canon.
/// Wave-29 PR-A.1 replaces the legacy `{43, 44, 45}` (entirely
/// forbidden) with the Canon #93 triple.
/// Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877
pub const GATE_FINAL_SEEDS: &[u64] = &[47, 89, 123];

const VOCAB: usize = 128;
const DIM: usize = 64;
const NUM_CTX: usize = 6;
const NGRAM: usize = NUM_CTX + 2;
const SEQ: usize = 128;
const LN_2: f32 = std::f32::consts::LN_2;
const PHI_INV: f32 = 0.618033988749895;
const CTX_WEIGHTS: [f32; NUM_CTX] = [0.70, 0.45, 0.30, 0.20, 0.13, 0.08];
const ATTN_SEQ: usize = 8;

fn attn_scale() -> f32 {
    std::env::var("TRIOS_ATTN_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1)
}

fn gf16_enabled() -> bool {
    std::env::var("TRIOS_GF16_DISABLE")
        .map(|v| v != "1")
        .unwrap_or(true)
}

/// Wave 31 PR-B: resolve GF16 from `GF16_ENABLED` env knob (default false).
/// If `GF16_ENABLED=true` but feature `gf16` is not compiled in, returns Err.
/// Falls back to `gf16_enabled()` for legacy `TRIOS_GF16_DISABLE` path.
/// Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877
fn resolve_gf16_knob() -> Result<bool> {
    let knob = parse_gf16_enabled().map_err(|e| anyhow::anyhow!("GF16_ENABLED: {e}"))?;
    if knob {
        #[cfg(not(feature = "gf16"))]
        {
            return Err(anyhow::anyhow!(
                "GF16_ENABLED=true but feature 'gf16' not compiled in; \
                 rebuild with: cargo build --features gf16"
            ));
        }
        #[cfg(feature = "gf16")]
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn attn_seq_override() -> usize {
    std::env::var("TRIOS_ATTN_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(ATTN_SEQ)
}

#[derive(Debug)]
pub struct TrainArgs {
    pub seed: u64,
    pub steps: usize,
    pub hidden: usize,
    pub lr: f32,
    pub attn_layers: u8,
    pub eval_every: usize,
    pub train_path: String,
    pub val_path: String,
}

#[derive(Debug)]
pub struct RunOutcome {
    pub final_bpb: f64,
    pub steps_done: usize,
    pub seed: u64,
}

fn load_data(path: &str) -> Vec<usize> {
    if std::path::Path::new(path).exists() {
        let raw = std::fs::read(path).unwrap_or_else(|e| {
            panic!("Failed to read {}: {}", path, e);
        });
        return raw.into_iter().map(|b| (b as usize) % VOCAB).collect();
    }

    eprintln!("Data file '{}' not found, using synthetic fallback", path);

    let fallback = b"The quick brown fox jumps over the lazy dog. ".repeat(2500);
    fallback.into_iter().map(|b| (b as usize) % VOCAB).collect()
}

/// Assert train/val streams are byte-disjoint at a sample level. Called by
/// `run_single` and `run_single_muon` right after load, before any gradient
/// step. Panics if the first 1024 bytes of val occur as a substring of train
/// (the classic "val is a prefix of train" leak fixed in
/// trios-trainer-igla#60 — Dockerfile was doing `head -c 100000 train > val`).
///
/// Keeps the check cheap: only compares two 1 KB hash windows. Does not
/// catch all possible overlaps, but blocks the specific regression that
/// produced the 216 leak-tainted BPB < 0.1 rows in the 2026-04-30 ledger.
pub(crate) fn assert_train_val_disjoint(train: &[usize], val: &[usize]) {
    let probe_len = 1024.min(val.len());
    if probe_len == 0 || train.is_empty() {
        return; // nothing to compare (synthetic fallback path)
    }
    let val_probe = &val[..probe_len];
    // Scan train in 1 KB windows looking for val prefix identity.
    let hit = train
        .windows(probe_len)
        .step_by(probe_len / 4) // stride = 256 tokens, ~75% overlap coverage
        .any(|w| w == val_probe);
    assert!(
        !hit,
        "TRAIN/VAL OVERLAP DETECTED: first {} val tokens appear in train. \
         This is the 2026-04-30 ledger leak bug (trios-trainer-igla#60). \
         Rebuild image with byte-disjoint split: head -c $((SIZE-100000)) for train, tail -c 100000 for val.",
        probe_len
    );
}

fn layer_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std_inv = 1.0 / (var + eps).sqrt();
    x.iter().map(|v| (v - mean) * std_inv).collect()
}

fn layer_norm_backward(x: &[f32], y: &[f32], dy: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std_inv = 1.0 / (var + eps).sqrt();
    let sum_dy: f32 = dy.iter().sum();
    let sum_dy_y: f32 = dy.iter().zip(y.iter()).map(|(d, yi)| d * yi).sum();
    dy.iter()
        .zip(y.iter())
        .map(|(d, yi)| (d - sum_dy / n - yi * sum_dy_y / n) * std_inv)
        .collect()
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

fn cosine_lr(step: usize, max_steps: usize, base_lr: f32, warmup: usize) -> f32 {
    if step < warmup {
        return base_lr * step as f32 / warmup.max(1) as f32;
    }
    let p = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
    1e-5 + (base_lr - 1e-5) * 0.5 * (1.0 + (std::f32::consts::PI * p).cos())
}

struct AdamW {
    m: Vec<f32>,
    v: Vec<f32>,
    step: usize,
    beta1: f32,
    beta2: f32,
    wd: f32,
}

impl AdamW {
    fn new(size: usize, wd: f32) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            step: 0,
            beta1: 0.9,
            beta2: 0.999,
            wd,
        }
    }
    fn update(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
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

fn gf16_floor(weights: &mut [f32]) {
    let scale = 16.0_f32;
    for w in weights.iter_mut() {
        *w = (*w * scale).round() / scale;
    }
}

/// Resolve the QAT format from `TRIOS_FORMAT_TYPE` (or alias `TRIOS_FAKE_QUANT_FORMAT`).
/// Returns `None` if the env var is unset or maps to F32 (no quantization).
/// Closes scarab→trios-train gap from #509: previously only `cpu_train` honoured the
/// env var, so `TRIOS_FORMAT_TYPE=fp16` produced identical BPB to F32 in production.
fn resolve_fake_quant_format() -> Option<FormatKind> {
    let raw = std::env::var("TRIOS_FORMAT_TYPE")
        .ok()
        .or_else(|| std::env::var("TRIOS_FAKE_QUANT_FORMAT").ok())?;
    let fmt = FormatKind::from_env(&raw)?;
    if fmt == FormatKind::F32 {
        return None;
    }
    Some(fmt)
}

/// Apply Phase-1 fake-quantization to every weight tensor in the hybrid model.
/// Skips identity formats (F32 / `is_unsupported_in_f32()`) so the call is a no-op
/// when QAT is disabled.
fn fake_quantize_model(model: &mut HybridModel, fmt: FormatKind) {
    if fmt == FormatKind::F32 || fmt.is_unsupported_in_f32() {
        return;
    }
    fake_quant::fake_quantize_weights(&mut model.embed, fmt);
    fake_quant::fake_quantize_weights(&mut model.proj, fmt);
    fake_quant::fake_quantize_weights(&mut model.lm_head, fmt);
    fake_quant::fake_quantize_weights(&mut model.attn_down, fmt);
    fake_quant::fake_quantize_weights(&mut model.attn_up, fmt);
    for c in model.ctx.iter_mut() {
        fake_quant::fake_quantize_weights(c, fmt);
    }
    fake_quant::fake_quantize_weights(model.attn.wq_mut(), fmt);
    fake_quant::fake_quantize_weights(model.attn.wk_mut(), fmt);
    fake_quant::fake_quantize_weights(model.attn.wv_mut(), fmt);
    fake_quant::fake_quantize_weights(model.attn.wo_mut(), fmt);
    fake_quant::fake_quantize_weights(model.attn.wq2_mut(), fmt);
    fake_quant::fake_quantize_weights(model.attn.wk2_mut(), fmt);
    fake_quant::fake_quantize_weights(model.attn.wv2_mut(), fmt);
    fake_quant::fake_quantize_weights(model.attn.wo2_mut(), fmt);
}

struct HybridModel {
    embed: Vec<f32>,
    ctx: Vec<Vec<f32>>,
    proj: Vec<f32>,
    attn: HybridAttn,
    attn_down: Vec<f32>,
    attn_up: Vec<f32>,
    lm_head: Vec<f32>,
    hidden: usize,
}

struct ForwardCache {
    combined: Vec<f32>,
    ln: Vec<f32>,
    hidden_pre_attn: Vec<f32>,
    attn_input: Vec<f32>,
    attn_out: Vec<f32>,
    hidden: Vec<f32>,
    logits: Vec<f32>,
    attn_cache: Option<AttentionCache>,
    attn_seq: usize,
    combined_seq: Vec<f32>,
    ln_seq: Vec<f32>,
}

impl HybridModel {
    fn new(hidden: usize, seed: u64, attn_layers: u8) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let lim = (6.0f32 / (3 * DIM) as f32).sqrt();
        let lim_h = (6.0f32 / (DIM + hidden) as f32).sqrt();
        let lim_o = (6.0f32 / (hidden + VOCAB) as f32).sqrt();

        let attn = if attn_layers == 1 {
            let mut cfg = crate::model_hybrid_attn::HybridAttnConfig::default();
            cfg.num_attn_layers = 1;
            HybridAttn::with_config(cfg).expect("1-layer attn config valid")
        } else {
            HybridAttn::new().expect("2-layer attn defaults valid")
        };
        let d = attn.config().d_model;
        let lim_down = (2.0f32 / (hidden + d) as f32).sqrt();
        let lim_up = (2.0f32 / (d + hidden) as f32).sqrt();

        let mut m = Self {
            embed: (0..VOCAB * DIM).map(|_| rng() * lim).collect(),
            ctx: (0..NUM_CTX)
                .map(|_| (0..VOCAB * DIM).map(|_| rng() * lim).collect())
                .collect(),
            proj: (0..hidden * DIM).map(|_| rng() * lim_h).collect(),
            attn,
            attn_down: (0..d * hidden).map(|_| rng() * lim_down).collect(),
            attn_up: (0..hidden * d).map(|_| rng() * lim_up).collect(),
            lm_head: (0..VOCAB * hidden).map(|_| rng() * lim_o).collect(),
            hidden,
        };

        let d = m.attn.config().d_model;
        let attn_lim = (2.0f32 / d as f32).sqrt();
        for w in m.attn.wq_mut() {
            *w = rng() * attn_lim;
        }
        for w in m.attn.wk_mut() {
            *w = rng() * attn_lim;
        }
        for w in m.attn.wv_mut() {
            *w = rng() * attn_lim;
        }
        for w in m.attn.wo_mut() {
            *w = rng() * attn_lim;
        }

        let attn_params = d * hidden * 2 + m.attn.total_weights();
        let total =
            VOCAB * DIM + NUM_CTX * VOCAB * DIM + hidden * DIM + VOCAB * hidden + attn_params;
        eprintln!(
            "params={} ({:.1}K) attn_d={} attn_layers={}",
            total,
            total as f64 / 1000.0,
            d,
            attn_layers
        );
        m
    }

    fn forward_cached(&self, tokens: &[usize], pos: usize) -> ForwardCache {
        let h = self.hidden;
        let d = self.attn.config().d_model;

        let attn_seq = attn_seq_override().min(pos + 1);
        let seq_start = pos + 1 - attn_seq;

        let mut attn_input = vec![0.0f32; attn_seq * d];
        let mut combined_seq = vec![0.0f32; attn_seq * DIM];
        let mut ln_seq = vec![0.0f32; attn_seq * DIM];

        for (si, p) in (seq_start..=pos).enumerate() {
            let t_last = tokens[p + NGRAM - 1].min(VOCAB - 1);
            let mut combined = self.embed[t_last * DIM..(t_last + 1) * DIM].to_vec();
            for (ci, cw) in CTX_WEIGHTS.iter().enumerate() {
                let ctx_idx = NGRAM - 2 - ci;
                let t = tokens[p + ctx_idx].min(VOCAB - 1);
                let cv = &self.ctx[ci][t * DIM..(t + 1) * DIM];
                for j in 0..DIM {
                    combined[j] += cv[j] * cw;
                }
            }
            let ln = layer_norm(&combined, 1e-5);
            combined_seq[si * DIM..(si + 1) * DIM].copy_from_slice(&combined);
            ln_seq[si * DIM..(si + 1) * DIM].copy_from_slice(&ln);
            attn_input[si * DIM..(si + 1) * DIM].copy_from_slice(&ln);
        }

        let (attn_output, attn_cache) = self
            .attn
            .forward_with_cache(&attn_input, attn_seq)
            .unwrap_or_else(|_| (vec![0.0f32; attn_seq * d], AttentionCache::default()));

        let attn_out = attn_output[(attn_seq - 1) * d..attn_seq * d].to_vec();

        let mut attn_up_out = vec![0.0f32; h];
        for hi in 0..h {
            for di in 0..d {
                attn_up_out[hi] += self.attn_up[hi * d + di] * attn_out[di];
            }
        }

        let t_last = tokens[pos + NGRAM - 1].min(VOCAB - 1);
        let mut combined = self.embed[t_last * DIM..(t_last + 1) * DIM].to_vec();
        for (ci, cw) in CTX_WEIGHTS.iter().enumerate() {
            let ctx_idx = NGRAM - 2 - ci;
            let t = tokens[pos + ctx_idx].min(VOCAB - 1);
            let cv = &self.ctx[ci][t * DIM..(t + 1) * DIM];
            for j in 0..DIM {
                combined[j] += cv[j] * cw;
            }
        }
        let ln = layer_norm(&combined, 1e-5);

        let mut hidden_raw = vec![0.0f32; h];
        for hi in 0..h {
            for j in 0..DIM {
                hidden_raw[hi] += self.proj[hi * DIM + j] * ln[j];
            }
        }
        let mut hidden_pre_attn = vec![0.0f32; h];
        for hi in 0..h {
            hidden_pre_attn[hi] = if hidden_raw[hi] > 0.0 {
                hidden_raw[hi] * hidden_raw[hi]
            } else {
                0.0
            };
        }
        let mut hidden = hidden_pre_attn.clone();
        for hi in 0..h {
            hidden[hi] += attn_up_out[hi] * attn_scale();
        }

        let mut logits = vec![0.0f32; VOCAB];
        for vi in 0..VOCAB {
            for hi in 0..h {
                logits[vi] += self.lm_head[vi * h + hi] * hidden[hi];
            }
        }

        ForwardCache {
            combined,
            ln,
            hidden_pre_attn,
            attn_input,
            attn_out,
            hidden,
            logits,
            attn_cache: Some(attn_cache),
            attn_seq,
            combined_seq,
            ln_seq,
        }
    }

    fn loss_on_seq(&self, tokens: &[usize]) -> f32 {
        if tokens.len() < NGRAM + 1 {
            return 0.0;
        }
        let count = tokens.len().saturating_sub(NGRAM);
        let mut total = 0.0f32;
        for i in 0..count {
            let target = tokens[i + NGRAM].min(VOCAB - 1);
            let fc = self.forward_cached(tokens, i);
            let mut logits = fc.logits;
            softmax(&mut logits);
            total -= logits[target].max(1e-10).ln();
        }
        total / count.max(1) as f32
    }
}

fn compute_grads(
    model: &HybridModel,
    tokens: &[usize],
    positions: &[usize],
    g_embed: &mut [f32],
    g_ctx: &mut [Vec<f32>],
    g_proj: &mut [f32],
    g_head: &mut [f32],
    g_attn_down: &mut [f32],
    g_attn_up: &mut [f32],
    g_attn_weights: &mut [f32],
) {
    let h = model.hidden;
    let d = model.attn.config().d_model;
    let dd = d * d;

    for &pos in positions {
        let fc = model.forward_cached(tokens, pos);
        let ForwardCache {
            combined,
            ln,
            hidden_pre_attn,
            attn_input: _,
            attn_out,
            hidden,
            mut logits,
            attn_cache,
            attn_seq,
            combined_seq,
            ln_seq,
        } = fc;

        softmax(&mut logits);
        let target = tokens[pos + NGRAM].min(VOCAB - 1);
        let mut d_hidden = vec![0.0f32; h];
        for vi in 0..VOCAB {
            let grad = logits[vi] - if vi == target { 1.0 } else { 0.0 };
            for hi in 0..h {
                g_head[vi * h + hi] += grad * hidden[hi];
                d_hidden[hi] += grad * model.lm_head[vi * h + hi];
            }
        }

        let scale = attn_scale();
        let d_attn_up_out: Vec<f32> = d_hidden.iter().map(|&dh| dh * scale).collect();
        let mut d_attn_out_last = vec![0.0f32; d];
        for hi in 0..h {
            for di in 0..d {
                g_attn_up[hi * d + di] += d_attn_up_out[hi] * attn_out[di];
                d_attn_out_last[di] += d_attn_up_out[hi] * model.attn_up[hi * d + di];
            }
        }

        if let Some(cache) = attn_cache {
            let seq = attn_seq;
            let mut d_output = vec![0.0f32; seq * d];
            d_output[(seq - 1) * d..seq * d].copy_from_slice(&d_attn_out_last);

            let grads = model.attn.backward(&cache, &d_output);

            for i in 0..dd {
                g_attn_weights[i] += grads.gwq[i];
            }
            for i in 0..dd {
                g_attn_weights[dd + i] += grads.gwk[i];
            }
            for i in 0..dd {
                g_attn_weights[2 * dd + i] += grads.gwv[i];
            }
            for i in 0..dd {
                g_attn_weights[3 * dd + i] += grads.gwo[i];
            }
            for i in 0..dd {
                g_attn_weights[4 * dd + i] += grads.gwq2[i];
            }
            for i in 0..dd {
                g_attn_weights[5 * dd + i] += grads.gwk2[i];
            }
            for i in 0..dd {
                g_attn_weights[6 * dd + i] += grads.gwv2[i];
            }
            for i in 0..dd {
                g_attn_weights[7 * dd + i] += grads.gwo2[i];
            }

            let d_ai = grads.d_input;
            for si in 0..seq {
                let p = pos + 1 - attn_seq + si;
                let d_ln_si = &d_ai[si * DIM..(si + 1) * DIM];
                let c_si = &combined_seq[si * DIM..(si + 1) * DIM];
                let l_si = &ln_seq[si * DIM..(si + 1) * DIM];
                let d_combined_si = layer_norm_backward(c_si, l_si, d_ln_si, 1e-5);

                let t_last = tokens[p + NGRAM - 1].min(VOCAB - 1);
                for j in 0..DIM {
                    g_embed[t_last * DIM + j] += d_combined_si[j];
                }
                for (ci, cw) in CTX_WEIGHTS.iter().enumerate() {
                    let ctx_idx = NGRAM - 2 - ci;
                    let t = tokens[p + ctx_idx].min(VOCAB - 1);
                    for j in 0..DIM {
                        g_ctx[ci][t * DIM + j] += cw * d_combined_si[j];
                    }
                }
            }
        }

        let mut d_raw = vec![0.0f32; h];
        for hi in 0..h {
            if hidden_pre_attn[hi] > 0.0 {
                d_raw[hi] = d_hidden[hi] * 2.0 * hidden_pre_attn[hi].sqrt();
            }
        }
        let mut d_ln = vec![0.0f32; DIM];
        for hi in 0..h {
            for j in 0..DIM {
                g_proj[hi * DIM + j] += d_raw[hi] * ln[j];
                d_ln[j] += model.proj[hi * DIM + j] * d_raw[hi];
            }
        }
        let d_combined = layer_norm_backward(&combined, &ln, &d_ln, 1e-5);
        let t_last = tokens[pos + NGRAM - 1].min(VOCAB - 1);
        for j in 0..DIM {
            g_embed[t_last * DIM + j] += d_combined[j];
        }
        for (ci, cw) in CTX_WEIGHTS.iter().enumerate() {
            let ctx_idx = NGRAM - 2 - ci;
            let t = tokens[pos + ctx_idx].min(VOCAB - 1);
            for j in 0..DIM {
                g_ctx[ci][t * DIM + j] += cw * d_combined[j];
            }
        }
    }
}

fn evaluate(model: &HybridModel, tokens: &[usize]) -> f32 {
    let seq = SEQ + 1;
    let num_chunks = 40usize;
    let max_start = tokens.len().saturating_sub(seq);
    if max_start == 0 {
        return f32::MAX;
    }
    let step = if max_start >= num_chunks * seq {
        max_start / num_chunks
    } else {
        seq
    };
    let mut total = 0.0f32;
    let mut n = 0usize;
    for c in (0..max_start).step_by(step).take(num_chunks) {
        let end = (c + seq).min(tokens.len());
        if end - c < NGRAM + 2 {
            continue;
        }
        let loss = model.loss_on_seq(&tokens[c..end]);
        if loss.is_finite() {
            total += loss / LN_2;
            n += 1;
        }
    }
    if n == 0 {
        f32::MAX
    } else {
        total / n as f32
    }
}

pub fn run_single(args: &TrainArgs) -> Result<RunOutcome> {
    // Wave 31 PR-B: apply env-gated arch knobs (HIDDEN_DIM, NUM_ATTN_LAYERS).
    // Defaults preserve Wave-30 baseline (h=384, 1L).
    // Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877
    let eff_hidden = if std::env::var("HIDDEN_DIM").is_ok() {
        let h = parse_hidden_dim().map_err(|e| anyhow::anyhow!("HIDDEN_DIM: {e}"))?;
        eprintln!("[arch-knob] HIDDEN_DIM={h} (override, Wave-31 PR-B)");
        h
    } else {
        args.hidden
    };
    let eff_attn_layers: u8 = if std::env::var("NUM_ATTN_LAYERS").is_ok() {
        let n = parse_num_attn_layers().map_err(|e| anyhow::anyhow!("NUM_ATTN_LAYERS: {e}"))?;
        eprintln!("[arch-knob] NUM_ATTN_LAYERS={n} (override, Wave-31 PR-B)");
        n as u8
    } else {
        args.attn_layers
    };
    // Wave 31 PR-B: validate GF16_ENABLED knob (default false, feature-gated).
    let _gf16_knob = resolve_gf16_knob()?;
    if _gf16_knob {
        eprintln!("[arch-knob] GF16_ENABLED=true (Wave-31 PR-B, feature=gf16)");
    }
    eprintln!(
        "=== trios-train seed={} steps={} hidden={} lr={:.4} attn_layers={} ===",
        args.seed, args.steps, eff_hidden, args.lr, eff_attn_layers
    );
    let train = load_data(&args.train_path);
    let val = load_data(&args.val_path);
    eprintln!("train={} val={}", train.len(), val.len());
    assert_train_val_disjoint(&train, &val);

    // #509 Phase-1b: wire QAT into the production `trios-train` path.
    // `scarab` spawns this binary and sets `TRIOS_FORMAT_TYPE`; previously
    // only `cpu_train` honoured it, so production traffic was F32 regardless.
    let fq_fmt = resolve_fake_quant_format();
    if let Some(fmt) = fq_fmt {
        eprintln!("QAT: FakeQuant enabled for format {:?}", fmt);
    }

    let mut model = HybridModel::new(eff_hidden, args.seed, eff_attn_layers);
    if let Some(fmt) = fq_fmt {
        fake_quantize_model(&mut model, fmt);
    }
    let d = model.attn.config().d_model;
    let dd = d * d;
    let attn_total = 8 * dd;
    let wd = 0.04f32;
    let mut opt_embed = AdamW::new(VOCAB * DIM, wd);
    let mut opt_ctx: Vec<AdamW> = (0..NUM_CTX).map(|_| AdamW::new(VOCAB * DIM, wd)).collect();
    let mut opt_proj = AdamW::new(eff_hidden * DIM, wd);
    let mut opt_attn_down = AdamW::new(d * eff_hidden, wd);
    let mut opt_attn_up = AdamW::new(eff_hidden * d, wd);
    let mut opt_head = AdamW::new(VOCAB * eff_hidden, wd);
    let mut opt_attn_w = AdamW::new(attn_total, wd);

    let init_bpb = evaluate(&model, &val);
    eprintln!("Initial val_bpb={:.4}", init_bpb);
    let mut ema_bpb = init_bpb;
    let mut best_bpb = init_bpb;
    let warmup = args.steps / 10;
    let accum = 4;
    let mut rng_s = args.seed.wrapping_add(7919);
    let t0 = Instant::now();
    let gf16_floor_step = (0.7 * args.steps as f32).floor() as usize;
    let nca = NcaObjective::default();
    let mut last_nca_entropy = 0.0f64;

    for step in 1..=args.steps {
        let lr = cosine_lr(step, args.steps, args.lr, warmup);
        let mut ge = vec![0.0f32; VOCAB * DIM];
        let mut gc: Vec<Vec<f32>> = (0..NUM_CTX).map(|_| vec![0.0f32; VOCAB * DIM]).collect();
        let mut gp = vec![0.0f32; eff_hidden * DIM];
        let mut gh = vec![0.0f32; VOCAB * eff_hidden];
        let mut g_ad = vec![0.0f32; d * eff_hidden];
        let mut g_au = vec![0.0f32; eff_hidden * d];
        let mut g_aw = vec![0.0f32; 8 * dd];

        for _ in 0..accum {
            rng_s = rng_s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let dl = train.len();
            let ms = dl.saturating_sub(SEQ + 1);
            if ms == 0 {
                continue;
            }
            let cs = (rng_s as usize) % ms;
            let chunk = &train[cs..cs + SEQ + 1];
            let cnt = chunk.len().saturating_sub(NGRAM);
            if cnt == 0 {
                continue;
            }
            let ns = 8.min(cnt);
            let mut pos = Vec::with_capacity(ns);
            for _ in 0..ns {
                rng_s = rng_s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                pos.push((rng_s as usize) % cnt);
            }
            compute_grads(
                &model, chunk, &pos, &mut ge, &mut gc, &mut gp, &mut gh, &mut g_ad, &mut g_au,
                &mut g_aw,
            );
        }

        let tp = (accum * 8) as f32;
        for x in ge.iter_mut() {
            *x /= tp;
        }
        for g in gc.iter_mut() {
            for x in g.iter_mut() {
                *x /= tp;
            }
        }
        for x in gp.iter_mut() {
            *x /= tp;
        }
        for x in gh.iter_mut() {
            *x /= tp;
        }
        for x in g_ad.iter_mut() {
            *x /= tp;
        }
        for x in g_au.iter_mut() {
            *x /= tp;
        }
        for x in g_aw.iter_mut() {
            *x /= tp;
        }

        {
            rng_s = rng_s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let dl = train.len();
            let ms = dl.saturating_sub(SEQ + 1);
            if ms > 0 {
                let cs = (rng_s as usize) % ms;
                let chunk = &train[cs..cs + SEQ + 1];
                if chunk.len() > NGRAM {
                    let fc = model.forward_cached(chunk, 0);
                    let k = nca.k_states;
                    let h_min = fc.hidden.iter().cloned().fold(f32::MAX, f32::min);
                    let h_max = fc.hidden.iter().cloned().fold(f32::MIN, f32::max);
                    let range = (h_max - h_min).max(1e-6);
                    let nca_state: Vec<f32> = fc
                        .hidden
                        .iter()
                        .map(|&h| (((h - h_min) / range) * (k as f32 - 1.0)).round().max(0.0))
                        .collect();
                    let (nca_loss_val, nca_ent) = nca_entropy_loss(
                        &nca_state,
                        k,
                        nca.entropy_min,
                        nca.entropy_max,
                        nca.weight,
                    );
                    last_nca_entropy = nca_ent;
                    if nca_loss_val > 0.0 {
                        let scale = 1.0 + (nca_loss_val as f32).min(5.0);
                        for x in gp.iter_mut() {
                            *x *= scale;
                        }
                    }
                }
            }
        }

        opt_embed.update(&mut model.embed, &ge, lr);
        for (ci, oc) in opt_ctx.iter_mut().enumerate() {
            oc.update(&mut model.ctx[ci], &gc[ci], lr);
        }
        opt_proj.update(&mut model.proj, &gp, lr);
        opt_attn_down.update(&mut model.attn_down, &g_ad, lr);
        opt_attn_up.update(&mut model.attn_up, &g_au, lr);
        opt_head.update(&mut model.lm_head, &gh, lr);

        {
            let mut attn_flat = Vec::with_capacity(attn_total);
            attn_flat.extend_from_slice(&model.attn.wq);
            attn_flat.extend_from_slice(&model.attn.wk);
            attn_flat.extend_from_slice(&model.attn.wv);
            attn_flat.extend_from_slice(&model.attn.wo);
            attn_flat.extend_from_slice(&model.attn.wq2);
            attn_flat.extend_from_slice(&model.attn.wk2);
            attn_flat.extend_from_slice(&model.attn.wv2);
            attn_flat.extend_from_slice(&model.attn.wo2);
            opt_attn_w.update(&mut attn_flat, &g_aw, lr);
            model.attn.wq.copy_from_slice(&attn_flat[0..dd]);
            model.attn.wk.copy_from_slice(&attn_flat[dd..2 * dd]);
            model.attn.wv.copy_from_slice(&attn_flat[2 * dd..3 * dd]);
            model.attn.wo.copy_from_slice(&attn_flat[3 * dd..4 * dd]);
            model.attn.wq2.copy_from_slice(&attn_flat[4 * dd..5 * dd]);
            model.attn.wk2.copy_from_slice(&attn_flat[5 * dd..6 * dd]);
            model.attn.wv2.copy_from_slice(&attn_flat[6 * dd..7 * dd]);
            model.attn.wo2.copy_from_slice(&attn_flat[7 * dd..8 * dd]);
        }

        // #509 Phase-1b: STE fake-quantize after each optimizer step.
        if let Some(fmt) = fq_fmt {
            fake_quantize_model(&mut model, fmt);
        }

        if gf16_enabled() && step >= gf16_floor_step && step % args.eval_every == 0 {
            gf16_floor(&mut model.embed);
            gf16_floor(&mut model.proj);
            gf16_floor(&mut model.lm_head);
            for c in &mut model.ctx {
                gf16_floor(c);
            }
        }

        if step % args.eval_every == 0 || step == args.steps {
            let vbpb = evaluate(&model, &val);
            ema_bpb = PHI_INV * ema_bpb + (1.0 - PHI_INV) * vbpb;
            if ema_bpb < best_bpb && ema_bpb.is_finite() {
                best_bpb = ema_bpb;
            }
            println!(
                "seed={} step={} val_bpb={:.4} ema_bpb={:.4} best={:.4} nca_h={:.3} t={:.1}s",
                args.seed,
                step,
                vbpb,
                ema_bpb,
                best_bpb,
                last_nca_entropy,
                t0.elapsed().as_secs_f64()
            );
            // R5/L8: flush stdout immediately so seed-agent reads the JSONL
            // line. Without this the line stays in the BufWriter for the
            // child stdout pipe and the parent reader times out before EOF.
            // Refs: trios-railway#100, trios-trainer-igla#57.
            use std::io::Write as _;
            let _ = std::io::stdout().flush();

            // Bug A fix: write eval to Neon bpb_samples if TRIOS_CANON_NAME
            // is set (scarab sets this env var for the trainer subprocess).
            // EPIC-446 follow-up: same triple-source canon resolution as
            // run_single (TRIOS_CANON_NAME → CANON_NAME → fallback by seed).
            let canon = std::env::var("TRIOS_CANON_NAME")
                .ok()
                .or_else(|| std::env::var("CANON_NAME").ok())
                .unwrap_or_else(|| format!("trios-train-rng{}", args.seed));
            crate::neon_writer::bpb_sample(
                &canon,
                args.seed as i32,
                step as i32,
                vbpb,
                Some(ema_bpb as f32),
            );
        }
    }

    Ok(RunOutcome {
        final_bpb: best_bpb as f64,
        steps_done: args.steps,
        seed: args.seed,
    })
}

pub fn run_single_muon(args: &TrainArgs, use_cwd: bool) -> Result<RunOutcome> {
    // Wave 31 PR-B: apply env-gated arch knobs.
    let eff_hidden = if std::env::var("HIDDEN_DIM").is_ok() {
        let h = parse_hidden_dim().map_err(|e| anyhow::anyhow!("HIDDEN_DIM: {e}"))?;
        eprintln!("[arch-knob] HIDDEN_DIM={h} (override, Wave-31 PR-B)");
        h
    } else {
        args.hidden
    };
    let eff_attn_layers: u8 = if std::env::var("NUM_ATTN_LAYERS").is_ok() {
        let n = parse_num_attn_layers().map_err(|e| anyhow::anyhow!("NUM_ATTN_LAYERS: {e}"))?;
        eprintln!("[arch-knob] NUM_ATTN_LAYERS={n} (override, Wave-31 PR-B)");
        n as u8
    } else {
        args.attn_layers
    };
    let _gf16_knob = resolve_gf16_knob()?;
    let label = if use_cwd { "MuonCwd" } else { "Muon" };
    eprintln!(
        "=== P1 {} seed={} steps={} hidden={} ===",
        label, args.seed, args.steps, eff_hidden
    );
    let train = load_data(&args.train_path);
    let val = load_data(&args.val_path);
    assert_train_val_disjoint(&train, &val);

    // #509 Phase-1b: same QAT wiring for the Muon path.
    let fq_fmt = resolve_fake_quant_format();
    if let Some(fmt) = fq_fmt {
        eprintln!("QAT: FakeQuant enabled for format {:?}", fmt);
    }

    let mut model = HybridModel::new(eff_hidden, args.seed, eff_attn_layers);
    if let Some(fmt) = fq_fmt {
        fake_quantize_model(&mut model, fmt);
    }
    let d = model.attn.config().d_model;
    let dd = d * d;
    let muon_lr = 0.0235f64;
    let muon_mom = 0.95f64;
    let muon_wd = 0.01f64;
    let adamw_wd = 0.04f32;
    let cwd_lambda = 0.01f64;

    let mut opt_embed = AdamW::new(VOCAB * DIM, adamw_wd);
    let mut opt_ctx: Vec<AdamW> = (0..NUM_CTX)
        .map(|_| AdamW::new(VOCAB * DIM, adamw_wd))
        .collect();
    let mut opt_proj_muon = crate::optimizer::MuonOptimizer::with_matrix_shape(
        eff_hidden * DIM,
        eff_hidden,
        DIM,
        muon_lr,
        muon_mom,
        muon_wd,
    );
    opt_proj_muon.ns_steps = if use_cwd { 3 } else { 1 };
    let mut opt_attn_down = AdamW::new(d * eff_hidden, adamw_wd);
    let mut opt_attn_up = AdamW::new(eff_hidden * d, adamw_wd);
    let mut opt_head = AdamW::new(VOCAB * eff_hidden, adamw_wd);
    let mut opt_attn_w_muon = AdamW::new(8 * dd, adamw_wd);
    let _cwd_lambda = cwd_lambda;

    let init_bpb = evaluate(&model, &val);
    eprintln!("Initial val_bpb={:.4}", init_bpb);
    let mut ema_bpb = init_bpb;
    let mut best_bpb = init_bpb;
    let warmup = args.steps / 10;
    let accum = 4;
    let mut rng_s = args.seed.wrapping_add(7919);
    let t0 = Instant::now();
    let gf16_floor_step = (0.7 * args.steps as f32).floor() as usize;
    let nca = NcaObjective::default();
    let mut last_nca_entropy = 0.0f64;

    for step in 1..=args.steps {
        let lr = cosine_lr(step, args.steps, args.lr, warmup);
        let mut ge = vec![0.0f32; VOCAB * DIM];
        let mut gc: Vec<Vec<f32>> = (0..NUM_CTX).map(|_| vec![0.0f32; VOCAB * DIM]).collect();
        let mut gp = vec![0.0f32; eff_hidden * DIM];
        let mut gh = vec![0.0f32; VOCAB * eff_hidden];
        let mut g_ad = vec![0.0f32; d * eff_hidden];
        let mut g_au = vec![0.0f32; eff_hidden * d];
        let mut g_aw = vec![0.0f32; 8 * dd];

        for _ in 0..accum {
            rng_s = rng_s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let dl = train.len();
            let ms = dl.saturating_sub(SEQ + 1);
            if ms == 0 {
                continue;
            }
            let cs = (rng_s as usize) % ms;
            let chunk = &train[cs..cs + SEQ + 1];
            let cnt = chunk.len().saturating_sub(NGRAM);
            if cnt == 0 {
                continue;
            }
            let ns = 8.min(cnt);
            let mut pos = Vec::with_capacity(ns);
            for _ in 0..ns {
                rng_s = rng_s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                pos.push((rng_s as usize) % cnt);
            }
            compute_grads(
                &model, chunk, &pos, &mut ge, &mut gc, &mut gp, &mut gh, &mut g_ad, &mut g_au,
                &mut g_aw,
            );
        }

        let tp = (accum * 8) as f32;
        for x in ge.iter_mut() {
            *x /= tp;
        }
        for g in gc.iter_mut() {
            for x in g.iter_mut() {
                *x /= tp;
            }
        }
        for x in gp.iter_mut() {
            *x /= tp;
        }
        for x in gh.iter_mut() {
            *x /= tp;
        }
        for x in g_ad.iter_mut() {
            *x /= tp;
        }
        for x in g_au.iter_mut() {
            *x /= tp;
        }
        for x in g_aw.iter_mut() {
            *x /= tp;
        }

        {
            rng_s = rng_s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let dl = train.len();
            let ms = dl.saturating_sub(SEQ + 1);
            if ms > 0 {
                let cs = (rng_s as usize) % ms;
                let chunk = &train[cs..cs + SEQ + 1];
                if chunk.len() > NGRAM {
                    let fc = model.forward_cached(chunk, 0);
                    let k = nca.k_states;
                    let h_min = fc.hidden.iter().cloned().fold(f32::MAX, f32::min);
                    let h_max = fc.hidden.iter().cloned().fold(f32::MIN, f32::max);
                    let range = (h_max - h_min).max(1e-6);
                    let nca_state: Vec<f32> = fc
                        .hidden
                        .iter()
                        .map(|&h| (((h - h_min) / range) * (k as f32 - 1.0)).round().max(0.0))
                        .collect();
                    let (nca_loss_val, nca_ent) = nca_entropy_loss(
                        &nca_state,
                        k,
                        nca.entropy_min,
                        nca.entropy_max,
                        nca.weight,
                    );
                    last_nca_entropy = nca_ent;
                    if nca_loss_val > 0.0 {
                        let scale = 1.0 + (nca_loss_val as f32).min(5.0);
                        for x in gp.iter_mut() {
                            *x *= scale;
                        }
                    }
                }
            }
        }

        opt_embed.update(&mut model.embed, &ge, lr);
        for (ci, oc) in opt_ctx.iter_mut().enumerate() {
            oc.update(&mut model.ctx[ci], &gc[ci], lr);
        }
        opt_proj_muon.step(&mut model.proj, &gp);
        opt_attn_down.update(&mut model.attn_down, &g_ad, lr);
        opt_attn_up.update(&mut model.attn_up, &g_au, lr);
        opt_head.update(&mut model.lm_head, &gh, lr);

        {
            let mut attn_flat = Vec::with_capacity(8 * dd);
            attn_flat.extend_from_slice(&model.attn.wq);
            attn_flat.extend_from_slice(&model.attn.wk);
            attn_flat.extend_from_slice(&model.attn.wv);
            attn_flat.extend_from_slice(&model.attn.wo);
            attn_flat.extend_from_slice(&model.attn.wq2);
            attn_flat.extend_from_slice(&model.attn.wk2);
            attn_flat.extend_from_slice(&model.attn.wv2);
            attn_flat.extend_from_slice(&model.attn.wo2);
            opt_attn_w_muon.update(&mut attn_flat, &g_aw, lr);
            model.attn.wq.copy_from_slice(&attn_flat[0..dd]);
            model.attn.wk.copy_from_slice(&attn_flat[dd..2 * dd]);
            model.attn.wv.copy_from_slice(&attn_flat[2 * dd..3 * dd]);
            model.attn.wo.copy_from_slice(&attn_flat[3 * dd..4 * dd]);
            model.attn.wq2.copy_from_slice(&attn_flat[4 * dd..5 * dd]);
            model.attn.wk2.copy_from_slice(&attn_flat[5 * dd..6 * dd]);
            model.attn.wv2.copy_from_slice(&attn_flat[6 * dd..7 * dd]);
            model.attn.wo2.copy_from_slice(&attn_flat[7 * dd..8 * dd]);
        }

        // #509 Phase-1b: STE fake-quantize after each optimizer step (Muon path).
        if let Some(fmt) = fq_fmt {
            fake_quantize_model(&mut model, fmt);
        }

        if gf16_enabled() && step >= gf16_floor_step && step % args.eval_every == 0 {
            gf16_floor(&mut model.embed);
            gf16_floor(&mut model.proj);
            gf16_floor(&mut model.lm_head);
            for c in &mut model.ctx {
                gf16_floor(c);
            }
        }

        if step % args.eval_every == 0 || step == args.steps {
            let vbpb = evaluate(&model, &val);
            ema_bpb = PHI_INV * ema_bpb + (1.0 - PHI_INV) * vbpb;
            if ema_bpb < best_bpb && ema_bpb.is_finite() {
                best_bpb = ema_bpb;
            }
            println!(
                "{} seed={} step={} val_bpb={:.4} ema_bpb={:.4} best={:.4} nca_h={:.3} t={:.1}s",
                label,
                args.seed,
                step,
                vbpb,
                ema_bpb,
                best_bpb,
                last_nca_entropy,
                t0.elapsed().as_secs_f64()
            );
            // R5/L8: flush stdout immediately. See note in run_single().
            use std::io::Write as _;
            let _ = std::io::stdout().flush();

            // Bug A fix: write eval to Neon bpb_samples if TRIOS_CANON_NAME
            // is set (scarab sets this env var for the trainer subprocess).
            // EPIC-446 follow-up: also accept the unprefixed CANON_NAME that
            // the railway_template_deploy MCP tool injects, plus a deterministic
            // fallback derived from seed so direct trios-train invocations
            // still produce telemetry. R5: a missing canon_name must never
            // cost us a 27 000-step training run silently again.
            let canon = std::env::var("TRIOS_CANON_NAME")
                .ok()
                .or_else(|| std::env::var("CANON_NAME").ok())
                .unwrap_or_else(|| format!("trios-train-rng{}", args.seed));
            crate::neon_writer::bpb_sample(
                &canon,
                args.seed as i32,
                step as i32,
                vbpb,
                Some(ema_bpb as f32),
            );
        }
    }

    Ok(RunOutcome {
        final_bpb: best_bpb as f64,
        steps_done: args.steps,
        seed: args.seed,
    })
}

pub fn run_sweep(
    steps: usize,
    hidden: usize,
    lr: f32,
    attn_layers: u8,
    eval_every: usize,
    train_path: &str,
    val_path: &str,
) -> Result<Vec<RunOutcome>> {
    let mut results = Vec::new();
    for &seed in GATE_FINAL_SEEDS {
        results.push(run_single(&TrainArgs {
            seed,
            steps,
            hidden,
            lr,
            attn_layers,
            eval_every,
            train_path: train_path.to_string(),
            val_path: val_path.to_string(),
        })?);
    }
    Ok(results)
}

pub fn run(cfg: &crate::TrainConfig) -> Result<RunOutcome> {
    let args = TrainArgs {
        seed: cfg.seed,
        steps: cfg.steps,
        hidden: 828,
        lr: cfg.optimizer.lr as f32,
        attn_layers: if cfg.model.hybrid_attn { 2 } else { 1 },
        eval_every: 1000,
        train_path: cfg.data.train_path.clone(),
        val_path: cfg.data.val_path.clone(),
    };
    let outcome = run_single(&args)?;
    if !cfg.ledger.jsonl_path.is_empty() {
        let _ = crate::ledger::emit_row(cfg, outcome.final_bpb, outcome.steps_done);
    }
    Ok(outcome)
}

#[cfg(test)]
mod fake_quant_wiring_tests {
    //! Phase-1b regression tests for trios#509 — verify that the
    //! `trios-train` path now honours `TRIOS_FORMAT_TYPE` end-to-end.
    use super::*;
    use std::sync::Mutex;

    // Env-var tests must be serialised — `std::env` is process-global.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn resolves_fp16_from_trios_format_type() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_FAKE_QUANT_FORMAT");
        std::env::set_var("TRIOS_FORMAT_TYPE", "fp16");
        let fmt = resolve_fake_quant_format();
        std::env::remove_var("TRIOS_FORMAT_TYPE");
        assert_eq!(fmt, Some(FormatKind::Fp16));
    }

    #[test]
    fn resolves_gf16_alias() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_FORMAT_TYPE");
        std::env::set_var("TRIOS_FAKE_QUANT_FORMAT", "gf16");
        let fmt = resolve_fake_quant_format();
        std::env::remove_var("TRIOS_FAKE_QUANT_FORMAT");
        assert_eq!(fmt, Some(FormatKind::Gf16));
    }

    #[test]
    fn f32_resolves_to_none() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_FAKE_QUANT_FORMAT");
        std::env::set_var("TRIOS_FORMAT_TYPE", "f32");
        let fmt = resolve_fake_quant_format();
        std::env::remove_var("TRIOS_FORMAT_TYPE");
        assert_eq!(fmt, None);
    }

    #[test]
    fn unset_resolves_to_none() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_FORMAT_TYPE");
        std::env::remove_var("TRIOS_FAKE_QUANT_FORMAT");
        assert_eq!(resolve_fake_quant_format(), None);
    }

    #[test]
    fn unknown_format_resolves_to_none() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TRIOS_FAKE_QUANT_FORMAT");
        std::env::set_var("TRIOS_FORMAT_TYPE", "imaginary_float");
        let fmt = resolve_fake_quant_format();
        std::env::remove_var("TRIOS_FORMAT_TYPE");
        assert_eq!(fmt, None);
    }

    #[test]
    fn fake_quantize_model_actually_changes_weights() {
        let _g = ENV_LOCK.lock().unwrap();
        let mut model = HybridModel::new(64, 1597, 2);
        let embed_before = model.embed.clone();
        let lm_head_before = model.lm_head.clone();
        fake_quantize_model(&mut model, FormatKind::Fp16);
        // At least one weight must change after fp16 round-trip.
        let embed_changed = model
            .embed
            .iter()
            .zip(embed_before.iter())
            .any(|(a, b)| a != b);
        let head_changed = model
            .lm_head
            .iter()
            .zip(lm_head_before.iter())
            .any(|(a, b)| a != b);
        assert!(
            embed_changed || head_changed,
            "fake_quantize_model(Fp16) must change at least one weight"
        );
    }

    #[test]
    fn fake_quantize_model_f32_is_noop() {
        let _g = ENV_LOCK.lock().unwrap();
        let mut model = HybridModel::new(64, 1597, 2);
        let embed_before = model.embed.clone();
        fake_quantize_model(&mut model, FormatKind::F32);
        assert_eq!(model.embed, embed_before);
    }
}
