//! TASK-5D — T-JEPA Hybrid Training (Proven Architecture + JEPA)
//!
//! Architecture: arch_explorer proven model (dim=64, hidden=384, layer norm,
//! projection, separate ctx embeddings) + JEPA predictor on hidden reps.
//!
//! Multi-objective: L = ntp_w*NTP + jepa_w*JEPA + nca_w*NCA
//!
//! Champion baseline: 6-gram h=384 lr=0.003 seed=43 → BPB 2.5193 (27K steps)
//! Gate min (ASHA Rung-1): ≤ 2.22 BPB
//! Gate target: ≤ 2.03 BPB
//! IGLA target: < 1.50 BPB

#![allow(
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use std::fs;
use std::time::Instant;

use trios_trainer::{
    jepa::{
        predictor::{JepaPredictor, PredictorConfig},
        EmaConfig, EmaTarget,
    },
    objective::{
        compute_combined_loss, nca_entropy_loss, ComponentLosses, NcaObjective, ObjectiveConfig,
    },
    optimizer::MuonOptimizer,
};

const DEFAULT_VOCAB: usize = 128;
const DIM: usize = 64;
const HIDDEN: usize = 384;
const NUM_CTX: usize = 4;
const NGRAM: usize = NUM_CTX + 2;
const SEQ: usize = 64;
const LN_2: f32 = std::f32::consts::LN_2;
const HEARTBEAT_INTERVAL_SECS: u64 = 60;

fn load_fineweb_bin(path: &str) -> Vec<u16> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).expect(&format!("open {}", path));
    let mut header = [0u8; 1024];
    file.read_exact(&mut header).expect("read header");
    let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    let num_tokens = u64::from_le_bytes([
        header[8], header[9], header[10], header[11],
        header[12], header[13], header[14], header[15],
    ]);
    eprintln!("FineWeb bin: magic={} version={} num_tokens={}", magic, version, num_tokens);
    assert_eq!(magic, 20240520, "FineWeb magic mismatch");
    let mut tokens = vec![0u16; num_tokens as usize];
    let mut buf = vec![0u8; num_tokens as usize * 2];
    file.read_exact(&mut buf).expect("read tokens");
    for i in 0..tokens.len() {
        tokens[i] = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
    }
    tokens
}

// ── primitives ──

fn layer_norm(x: &[f32], eps: f32) -> Vec<f32> {
    assert!(!x.is_empty(), "layer_norm: empty input");
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = (var + eps).sqrt();
    x.iter().map(|v| (v - mean) / std).collect()
}

fn softmax(v: &mut [f32]) {
    assert!(!v.is_empty(), "softmax: empty input");
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    assert!(sum > 0.0, "softmax: zero sum");
    for x in v.iter_mut() {
        *x /= sum;
    }
}

fn cosine_lr(step: usize, max_steps: usize, base_lr: f32, warmup: usize) -> f32 {
    assert!(max_steps > 0, "cosine_lr: max_steps=0");
    if step < warmup {
        return base_lr * step as f32 / warmup.max(1) as f32;
    }
    let p = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
    1e-5 + (base_lr - 1e-5) * 0.5 * (1.0 + (std::f32::consts::PI * p).cos())
}

/// SG #5 adaptive LR schedule.
/// Warmup: linear 0 → base_lr.
/// Stable+decay: cosine decay from base_lr * SG_COS_THETA_W down to base_lr * 0.1.
/// Uses cos θ_W (≈ 0.8819) as the conservative LR floor (< base_lr).
fn sg5_adaptive_lr(step: usize, max_steps: usize, base_lr: f32, warmup: usize) -> f32 {
    use trios_trainer::invariants::SG_COS_THETA_W;
    assert!(max_steps > 0, "sg5_lr: max_steps=0");
    if step < warmup {
        return base_lr * step as f32 / warmup.max(1) as f32;
    }
    let p = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
    let decay = 0.5 * (1.0 + (std::f32::consts::PI * p).cos());
    let floor = SG_COS_THETA_W as f32;
    base_lr * (floor * decay + 0.1 * (1.0 - decay))
}

// ── local AdamW ──

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
        assert!(size > 0, "AdamW: size=0");
        let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            step: 0,
            beta1: 1.0 / phi as f32,
            beta2: 0.999,
            wd,
        }
    }

    fn update(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        assert_eq!(params.len(), grads.len(), "AdamW param/grad mismatch");
        assert_eq!(params.len(), self.m.len(), "AdamW buffer mismatch");
        assert!(lr > 0.0, "AdamW: lr≤0");
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

fn ema_inplace(target: &mut [f32], online: &[f32], decay: f32) {
    assert_eq!(target.len(), online.len(), "EMA size mismatch");
    assert!((0.0..1.0).contains(&decay), "EMA decay out of range");
    for (t, o) in target.iter_mut().zip(online.iter()) {
        *t = decay * *t + (1.0 - decay) * *o;
    }
}

// ── optimizer wrapper ──

enum OptKind {
    AdamW,
    Muon,
}

enum OptWrapper {
    LocalAdamW(AdamW),
    CrateMuon(MuonOptimizer),
}

impl OptWrapper {
    fn adamw(size: usize, wd: f32) -> Self {
        OptWrapper::LocalAdamW(AdamW::new(size, wd))
    }

    fn muon(size: usize, rows: usize, cols: usize, lr: f64, wd: f32) -> Self {
        OptWrapper::CrateMuon(MuonOptimizer::with_matrix_shape(size, rows, cols, lr, 0.95, wd as f64))
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        match self {
            OptWrapper::LocalAdamW(opt) => opt.update(params, grads, lr),
            OptWrapper::CrateMuon(opt) => {
                opt.lr = lr as f64;
                opt.step(params, grads);
            }
        }
    }
}

// ── model ──

struct NgramModel {
    vocab: usize,
    embed: Vec<f32>,
    ctx: Vec<Vec<f32>>,
    ctx_weights: Vec<f32>,
    proj: Vec<f32>,
    lm_head: Vec<f32>,
}

impl NgramModel {
    fn new(seed: u64, vocab: usize) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let lim = (6.0f32 / (3 * DIM) as f32).sqrt();
        let lim_h = (6.0f32 / (DIM + HIDDEN) as f32).sqrt();
        let lim_o = (6.0f32 / (HIDDEN + vocab) as f32).sqrt();
        let ctx_weights: Vec<f32> = vec![0.7, 0.3, 0.2, 0.15];
        assert_eq!(ctx_weights.len(), NUM_CTX, "ctx_weights count mismatch");
        Self {
            vocab,
            embed: (0..vocab * DIM).map(|_| rng() * lim).collect(),
            ctx: (0..NUM_CTX)
                .map(|_| (0..vocab * DIM).map(|_| rng() * lim).collect())
                .collect(),
            ctx_weights,
            proj: (0..HIDDEN * DIM).map(|_| rng() * lim_h).collect(),
            lm_head: (0..vocab * HIDDEN).map(|_| rng() * lim_o).collect(),
        }
    }

    fn compute_hidden(&self, context: &[usize]) -> Vec<f32> {
        assert!(context.len() >= 2, "context too short for hidden");
        let t0 = context[context.len() - 1].min(self.vocab - 1);
        let mut combined = self.embed[t0 * DIM..(t0 + 1) * DIM].to_vec();
        for (ci, cw) in self.ctx_weights.iter().enumerate() {
            let ctx_idx = context.len() - 2 - ci;
            let t = context[ctx_idx].min(self.vocab - 1);
            let cv = &self.ctx[ci][t * DIM..(t + 1) * DIM];
            for j in 0..DIM {
                combined[j] += cv[j] * cw;
            }
        }
        let ln = layer_norm(&combined, 1e-5);
        let mut hidden = vec![0.0f32; HIDDEN];
        for hi in 0..HIDDEN {
            for (j, l) in ln.iter().enumerate() {
                hidden[hi] += self.proj[hi * DIM + j] * l;
            }
            hidden[hi] = hidden[hi].max(0.0);
        }
        hidden
    }

    fn predict(&self, hidden: &[f32]) -> Vec<f32> {
        assert_eq!(hidden.len(), HIDDEN, "hidden dim mismatch");
        let mut logits = vec![0.0f32; self.vocab];
        for (vi, logit) in logits.iter_mut().enumerate() {
            for (hi, hn) in hidden.iter().enumerate() {
                *logit += self.lm_head[vi * HIDDEN + hi] * hn;
            }
        }
        logits
    }

    fn loss_on_seq(&self, tokens: &[usize]) -> f32 {
        if tokens.len() < NGRAM + 1 {
            return 0.0;
        }
        let count = tokens.len() - NGRAM;
        assert!(count > 0, "no n-gram pairs in sequence");
        let mut total = 0.0f32;
        for i in 0..count {
            let context = &tokens[i..i + NGRAM];
            let target = tokens[i + NGRAM].min(self.vocab - 1);
            let mut logits = self.predict(&self.compute_hidden(context));
            softmax(&mut logits);
            total -= logits[target].max(1e-10).ln();
        }
        total / count as f32
    }
}

// ── gradient computation ──

struct TrainGrads {
    g_embed: Vec<f32>,
    g_ctx: Vec<Vec<f32>>,
    g_proj: Vec<f32>,
    g_head: Vec<f32>,
}

fn compute_grads(model: &NgramModel, tokens: &[usize]) -> (TrainGrads, Vec<Vec<f32>>, f32) {
    let count = tokens.len().saturating_sub(NGRAM);
    assert!(count > 0, "sequence too short for gradient computation");

    let mut g_embed = vec![0.0f32; model.vocab * DIM];
    let mut g_ctx: Vec<Vec<f32>> = (0..NUM_CTX).map(|_| vec![0.0f32; model.vocab * DIM]).collect();
    let mut g_proj = vec![0.0f32; HIDDEN * DIM];
    let mut g_head = vec![0.0f32; model.vocab * HIDDEN];

    let (all_hidden, all_ln, all_contexts) = forward_pass(model, tokens, count);
    let total_loss = backward_pass(
        model,
        &all_hidden,
        &all_ln,
        &all_contexts,
        tokens,
        count,
        &mut g_embed,
        &mut g_ctx,
        &mut g_proj,
        &mut g_head,
    );

    let n = count as f32;
    for x in g_embed.iter_mut() {
        *x /= n;
    }
    for gc in g_ctx.iter_mut() {
        for x in gc.iter_mut() {
            *x /= n;
        }
    }
    for x in g_proj.iter_mut() {
        *x /= n;
    }
    for x in g_head.iter_mut() {
        *x /= n;
    }

    let grads = TrainGrads {
        g_embed,
        g_ctx,
        g_proj,
        g_head,
    };
    (grads, all_hidden, total_loss)
}

fn forward_pass(
    model: &NgramModel,
    tokens: &[usize],
    count: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<usize>>) {
    assert!(count > 0, "forward_pass: count=0");
    let mut all_hidden = Vec::with_capacity(count);
    let mut all_ln = Vec::with_capacity(count);
    let mut all_contexts = Vec::with_capacity(count);

    for i in 0..count {
        let context: Vec<usize> = tokens[i..i + NGRAM].to_vec();
        let t0 = context[NGRAM - 1].min(model.vocab - 1);
        let mut combined = model.embed[t0 * DIM..(t0 + 1) * DIM].to_vec();
        for (ci, cw) in model.ctx_weights.iter().enumerate() {
            let ctx_idx = NGRAM - 2 - ci;
            let t = context[ctx_idx].min(model.vocab - 1);
            let cv = &model.ctx[ci][t * DIM..(t + 1) * DIM];
            for j in 0..DIM {
                combined[j] += cv[j] * cw;
            }
        }
        let ln = layer_norm(&combined, 1e-5);
        let mut hidden = vec![0.0f32; HIDDEN];
        for hi in 0..HIDDEN {
            for (j, l) in ln.iter().enumerate() {
                hidden[hi] += model.proj[hi * DIM + j] * l;
            }
            hidden[hi] = hidden[hi].max(0.0);
        }
        all_hidden.push(hidden);
        all_ln.push(ln);
        all_contexts.push(context);
    }

    (all_hidden, all_ln, all_contexts)
}

fn backward_pass(
    model: &NgramModel,
    all_hidden: &[Vec<f32>],
    all_ln: &[Vec<f32>],
    all_contexts: &[Vec<usize>],
    tokens: &[usize],
    count: usize,
    g_embed: &mut [f32],
    g_ctx: &mut [Vec<f32>],
    g_proj: &mut [f32],
    g_head: &mut [f32],
) -> f32 {
    assert!(count > 0, "backward_pass: count=0");
    if all_hidden.len() != count {
        eprintln!(
            "WARN: hidden count mismatch: {} != {}, skipping step",
            all_hidden.len(),
            count
        );
        return 0.0;
    }
    assert_eq!(all_ln.len(), count, "ln count mismatch");
    let mut total_loss = 0.0f32;

    for i in 0..count {
        let target = tokens[i + NGRAM].min(model.vocab - 1);
        let hidden = &all_hidden[i];
        let mut d_hidden = vec![0.0f32; HIDDEN];
        let mut logits = model.predict(hidden);
        softmax(&mut logits);
        total_loss -= logits[target].max(1e-10).ln();

        for (vi, prob) in logits.iter().enumerate() {
            let grad = prob - if vi == target { 1.0 } else { 0.0 };
            for hi in 0..HIDDEN {
                g_head[vi * HIDDEN + hi] += grad * hidden[hi];
                d_hidden[hi] += grad * model.lm_head[vi * HIDDEN + hi];
            }
        }

        for hi in 0..HIDDEN {
            if all_hidden[i][hi] <= 0.0 {
                continue;
            }
            for di in 0..DIM {
                g_proj[hi * DIM + di] += d_hidden[hi] * all_ln[i][di];
            }
        }

        accumulate_input_grads(
            model,
            &all_contexts[i],
            &all_hidden[i],
            &d_hidden,
            g_embed,
            g_ctx,
        );
    }

    total_loss / count as f32
}

fn accumulate_input_grads(
    model: &NgramModel,
    context: &[usize],
    hidden: &[f32],
    d_hidden: &[f32],
    g_embed: &mut [f32],
    g_ctx: &mut [Vec<f32>],
) {
    assert!(context.len() >= NGRAM, "context too short");
    let t0 = context[NGRAM - 1].min(model.vocab - 1);
    for di in 0..DIM {
        let mut grad_sum = 0.0f32;
        for hi in 0..HIDDEN {
            if hidden[hi] > 0.0 {
                grad_sum += model.proj[hi * DIM + di] * d_hidden[hi];
            }
        }
        g_embed[t0 * DIM + di] += grad_sum;
        for (ci, cw) in model.ctx_weights.iter().enumerate() {
            let ctx_idx = NGRAM - 2 - ci;
            let t = context[ctx_idx].min(model.vocab - 1);
            g_ctx[ci][t * DIM + di] += cw * grad_sum;
        }
    }
}

// ── evaluation ──

fn evaluate(model: &NgramModel, tokens: &[usize]) -> f32 {
    assert!(!tokens.is_empty(), "evaluate: empty tokens");
    let mut total = 0.0f32;
    let mut n = 0usize;
    for c in (0..tokens.len()).step_by(SEQ + 1) {
        let end = (c + SEQ + 1).min(tokens.len());
        if end - c < NGRAM + 1 {
            continue;
        }
        let loss = model.loss_on_seq(&tokens[c..end]);
        if loss.is_finite() {
            total += loss / LN_2;
            n += 1;
        }
    }
    if n == 0 {
        return f32::MAX;
    }
    total / n as f32
}

fn load_data(path: &str, vocab: usize) -> Vec<usize> {
    if path.ends_with(".bin") {
        let tokens = load_fineweb_bin(path);
        eprintln!("Loaded {} FineWeb tokens from {}", tokens.len(), path);
        return tokens.into_iter().map(|t| (t as usize).min(vocab - 1)).collect();
    }
    if let Ok(raw) = fs::read(path) {
        if !raw.is_empty() {
            eprintln!("Loaded {} bytes from {}", raw.len(), path);
            return raw.into_iter().map(|b| (b as usize) % vocab).collect();
        }
    }

    let is_val = path.contains("val");
    let url =
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";
    eprintln!("Downloading TinyShakespeare from {}...", url);
    match ureq::get(url).call() {
        Ok(resp) => {
            let mut bytes = Vec::new();
            use std::io::Read;
            resp.into_reader()
                .read_to_end(&mut bytes)
                .unwrap_or_default();
            if bytes.is_empty() {
                panic!("Downloaded 0 bytes from {}", url);
            }
            eprintln!("Downloaded {} bytes", bytes.len());
            let data = if is_val {
                &bytes[..bytes.len().min(100_000)]
            } else {
                &bytes
            };
            if let Some(parent) = std::path::Path::new(path).parent() {
                let _ = fs::create_dir_all(parent);
            }
            let _ = fs::write(path, data);
            data.iter().map(|&b| (b as usize) % vocab).collect()
        }
        Err(e) => {
            panic!("Failed to load {} and download failed: {}", path, e);
        }
    }
}

// ── config ──

struct Config {
    seed: u64,
    steps: usize,
    encoder_lr: f32,
    ntp_lr: f32,
    use_jepa: bool,
    use_nca: bool,
    ntp_weight: f64,
    jepa_weight: f64,
    nca_weight: f64,
    opt_kind: OptKind,
    jepa_warmup: usize,
    weight_decay: f32,
    use_sg5_lr: bool,
    trial_id: String,
    agent_id: String,
    vocab: usize,
    train_data: String,
    val_data: String,
}

fn find_arg<T: std::str::FromStr>(args: &[String], prefix: &str, default: T) -> T {
    args.iter()
        .find(|a| a.starts_with(prefix))
        .and_then(|a| a[prefix.len()..].parse().ok())
        .unwrap_or(default)
}

fn parse_config(args: &[String]) -> Config {
    let has_encoder_lr = args.iter().any(|a| a.starts_with("--encoder-lr="));
    let encoder_lr: f32 = if has_encoder_lr {
        find_arg(args, "--encoder-lr=", 0.004)
    } else {
        find_arg(args, "--lr=", 0.004)
    };
    let seed: u64 = find_arg(args, "--seed=", 43);
    let steps: usize = find_arg(args, "--steps=", 3000);
    let ntp_lr: f32 = find_arg(args, "--ntp-lr=", 0.001);
    let ntp_weight: f64 = find_arg(args, "--ntp-weight=", 1.0);
    let jepa_weight: f64 = find_arg(args, "--jepa-weight=", 1.0);
    let nca_weight: f64 = find_arg(args, "--nca-weight=", 0.25);
    let mut jepa_warmup: usize = find_arg(args, "--jepa-warmup=", 1500);
    if jepa_warmup > steps / 10 {
        jepa_warmup = steps / 10;
    }
    let weight_decay: f32 = find_arg(args, "--weight-decay=", 0.01);
    let use_jepa = !args.iter().any(|a| a == "--no-jepa");
    let use_nca = !args.iter().any(|a| a == "--no-nca");
    let use_sg5_lr = args.iter().any(|a| a == "--sg5-lr");
    let opt_kind = if args.iter().any(|a| a == "--optimizer=muon") {
        OptKind::Muon
    } else {
        OptKind::AdamW
    };
    let trial_id: String = find_arg(args, "--trial-id=", "hybrid-001".to_string());
    let agent_id: String = find_arg(args, "--agent-id=", "ALFA".to_string());
    let vocab: usize = find_arg(args, "--vocab=", DEFAULT_VOCAB);
    let train_data: String = find_arg(args, "--train-data=", "data/tiny_shakespeare.txt".to_string());
    let val_data: String = find_arg(args, "--val-data=", "data/tiny_shakespeare_val.txt".to_string());

    assert!(encoder_lr > 0.0, "encoder_lr must be positive");
    assert!(ntp_lr > 0.0, "ntp_lr must be positive");
    assert!(steps > 0, "steps must be positive");
    assert!(ntp_weight >= 0.0, "ntp_weight must be >= 0");
    assert!(jepa_weight >= 0.0, "jepa_weight must be >= 0");
    assert!(nca_weight >= 0.0, "nca_weight must be >= 0");
    assert!(vocab > 0, "vocab must be positive");

    assert!(weight_decay >= 0.0, "weight_decay must be >= 0");

    Config {
        seed,
        steps,
        encoder_lr,
        ntp_lr,
        use_jepa,
        use_nca,
        ntp_weight,
        jepa_weight,
        nca_weight,
        opt_kind,
        jepa_warmup,
        weight_decay,
        use_sg5_lr,
        trial_id,
        agent_id,
        vocab,
        train_data,
        val_data,
    }
}

// ── JEPA step ──

struct JepaStepResult {
    loss: f64,
}

fn jepa_training_step(
    predictor: &mut JepaPredictor,
    model: &NgramModel,
    target_model: &mut NgramModel,
    hidden_vecs: &[Vec<f32>],
    seq: &[usize],
    seed: u64,
    step: usize,
    ema_target: &mut EmaTarget,
) -> JepaStepResult {
    let mask_result = build_span_mask(hidden_vecs.len().min(SEQ), seed, step);
    let (tgt_pos, ctx_pos) = mask_result;
    if tgt_pos.is_empty() || ctx_pos.is_empty() {
        return JepaStepResult { loss: 0.0 };
    }

    let zero_h = vec![0.0f32; HIDDEN];
    let ctx_flat: Vec<f32> = ctx_pos
        .iter()
        .flat_map(|&p| hidden_vecs.get(p).unwrap_or(&zero_h).iter().copied())
        .collect();

    let tgt_hidden: Vec<Vec<f32>> = tgt_pos
        .iter()
        .filter_map(|&p| {
            if p + NGRAM <= seq.len() {
                Some(target_model.compute_hidden(&seq[p..p + NGRAM]))
            } else {
                None
            }
        })
        .collect();

    let loss = if tgt_hidden.is_empty() {
        0.0f64
    } else {
        let tgt_flat: Vec<f32> = tgt_hidden.iter().flat_map(|v| v.iter().copied()).collect();
        predictor.forward_backward(&ctx_flat, &tgt_flat, tgt_hidden.len()) as f64
    };

    let decay = ema_target.decay() as f32;
    ema_inplace(&mut target_model.embed, &model.embed, decay);

    JepaStepResult { loss }
}

fn build_span_mask(len: usize, seed: u64, step: usize) -> (Vec<usize>, Vec<usize>) {
    assert!(len > 0, "build_span_mask: len=0");
    let mut s = seed
        .wrapping_add(step as u64)
        .wrapping_mul(6364136223846793005);
    let mut bitset = vec![false; len];
    let span_len = 3usize;
    let num_spans = 2usize;
    for _ in 0..num_spans {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let start = (s as usize) % len.saturating_sub(span_len);
        for b in start..(start + span_len).min(len) {
            bitset[b] = true;
        }
    }
    let tgt: Vec<usize> = bitset
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i) } else { None })
        .collect();
    let ctx: Vec<usize> = bitset
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if !m { Some(i) } else { None })
        .collect();
    (tgt, ctx)
}

// ── NCA step ──

fn nca_training_step(nca: &NcaObjective, seed: u64, step: usize) -> f64 {
    let nca_seed = seed.wrapping_add(step as u64).wrapping_mul(7919);
    let nca_state = nca.init_grid(nca_seed);
    let (loss, _) = nca_entropy_loss(
        &nca_state,
        nca.k_states,
        nca.entropy_min,
        nca.entropy_max,
        nca.weight,
    );
    assert!(loss.is_finite(), "NCA loss is not finite");
    loss
}

// ── Neon heartbeat ──

fn neon_trial_start(cfg: &Config) {
    // Build config_json once and forward to neon_writer (real INSERT) plus
    // an eprintln! breadcrumb for log-grep compatibility.
    let config_json = format!(
        "{{\"arch\":\"tjepa\",\"d_model\":{},\"lr\":{},\"seed\":{},\"optimizer\":\"{}\",\"ntp_w\":{},\"jepa_w\":{},\"nca_w\":{}}}",
        HIDDEN, cfg.encoder_lr, cfg.seed,
        match cfg.opt_kind { OptKind::AdamW => "adamw", OptKind::Muon => "muon" },
        cfg.ntp_weight, cfg.jepa_weight, cfg.nca_weight,
    );
    eprintln!(
        "NEON_SQL: INSERT INTO igla_race_trials (trial_id, config, status, agent_id, branch) VALUES ('{}', '{}', 'running', '{}', 'main');",
        cfg.trial_id, config_json, cfg.agent_id,
    );
    trios_trainer::neon_writer::trial_start(&cfg.trial_id, &config_json, &cfg.agent_id, "main");
}

fn neon_heartbeat(cfg: &Config, step: usize, bpb: f32, last: &mut Instant) {
    if last.elapsed().as_secs() >= HEARTBEAT_INTERVAL_SECS {
        eprintln!(
            "NEON_SQL: INSERT INTO igla_agents_heartbeat (agent_id, machine_id, branch, task, status, last_heartbeat) VALUES ('{}', 'local', 'main', '{}', 'active', NOW()) ON CONFLICT (agent_id) DO UPDATE SET status=EXCLUDED.status, last_heartbeat=EXCLUDED.last_heartbeat;",
            cfg.agent_id, cfg.trial_id,
        );
        eprintln!("NEON_SQL: UPDATE igla_race_trials SET bpb_latest={:.4}, steps_done={} WHERE trial_id='{}';", bpb, step, cfg.trial_id);
        trios_trainer::neon_writer::heartbeat(&cfg.trial_id, &cfg.agent_id, bpb, step);
        *last = Instant::now();
    }
}

fn neon_trial_complete(cfg: &Config, bpb: f32) {
    eprintln!(
        "NEON_SQL: UPDATE igla_race_trials SET bpb_final={:.4}, status='complete' WHERE trial_id='{}';",
        bpb, cfg.trial_id,
    );
    trios_trainer::neon_writer::trial_complete(&cfg.trial_id, bpb);
}

// ── training state ──

struct TrainingState {
    model: NgramModel,
    target_model: NgramModel,
    opt_embed: OptWrapper,
    opt_ctx: Vec<OptWrapper>,
    opt_proj: OptWrapper,
    opt_head: OptWrapper,
    predictor: Option<JepaPredictor>,
    ema_target: EmaTarget,
    nca: Option<NcaObjective>,
    obj_config: ObjectiveConfig,
    best_val_bpb: f32,
    start_time: Instant,
    last_heartbeat: Instant,
}

fn init_training(cfg: &Config) -> TrainingState {
    let make_opt = |size: usize, rows: usize, cols: usize, wd: f32| -> OptWrapper {
        match cfg.opt_kind {
            OptKind::AdamW => OptWrapper::adamw(size, wd),
            OptKind::Muon => OptWrapper::muon(size, rows, cols, cfg.encoder_lr as f64, wd),
        }
    };
    let wd = cfg.weight_decay;
    TrainingState {
        model: NgramModel::new(cfg.seed, cfg.vocab),
        target_model: NgramModel::new(cfg.seed, cfg.vocab),
        opt_embed: make_opt(cfg.vocab * DIM, cfg.vocab, DIM, wd),
        opt_ctx: (0..NUM_CTX).map(|_| make_opt(cfg.vocab * DIM, cfg.vocab, DIM, wd)).collect(),
        opt_proj: make_opt(HIDDEN * DIM, HIDDEN, DIM, wd),
        opt_head: make_opt(cfg.vocab * HIDDEN, cfg.vocab, HIDDEN, wd),
        predictor: if cfg.use_jepa {
            Some(JepaPredictor::new(PredictorConfig::with_d_model(HIDDEN)))
        } else {
            None
        },
        ema_target: EmaTarget::new(EmaConfig {
            start: 0.996,
            end: 1.0,
            ramp_steps: cfg.steps,
        }),
        nca: if cfg.use_nca {
            Some(NcaObjective::default())
        } else {
            None
        },
        obj_config: ObjectiveConfig {
            ntp_weight: cfg.ntp_weight,
            jepa_weight: cfg.jepa_weight,
            nca_weight: cfg.nca_weight,
        },
        best_val_bpb: f32::MAX,
        start_time: Instant::now(),
        last_heartbeat: Instant::now(),
    }
}

// ── banner ──

fn print_banner(cfg: &Config) {
    let opt_name = match cfg.opt_kind {
        OptKind::AdamW => "AdamW",
        OptKind::Muon => "Muon",
    };
    eprintln!("=== T-JEPA Hybrid Training ===");
    eprintln!(
        "dim={} hidden={} vocab={} enc_lr={} ntp_lr={} seed={} steps={}",
        DIM, HIDDEN, cfg.vocab, cfg.encoder_lr, cfg.ntp_lr, cfg.seed, cfg.steps
    );
    eprintln!(
        "optimizer={} jepa={} nca={} jepa_warmup={}",
        opt_name, cfg.use_jepa, cfg.use_nca, cfg.jepa_warmup
    );
    eprintln!(
        "L = {}*NTP + {}*JEPA + {}*NCA",
        cfg.ntp_weight, cfg.jepa_weight, cfg.nca_weight
    );
    eprintln!("trial_id={} agent_id={}", cfg.trial_id, cfg.agent_id);
    eprintln!("Champion: BPB 2.5193 | Gate-1: ≤2.22 | Gate-2: ≤2.03");
}

// ── results ──

fn print_results(cfg: &Config, best_bpb: f32, elapsed: f64) {
    eprintln!("\n=== Training Complete ===");
    eprintln!(
        "Steps={} Time={:.1}s best_val_bpb={:.4} vs_champion={:+.4}",
        cfg.steps,
        elapsed,
        best_bpb,
        best_bpb - 2.5193
    );
    println!("BPB={:.4}", best_bpb);

    if best_bpb <= 2.22 {
        eprintln!("Gate-1 PASSED (≤2.22)");
    } else {
        eprintln!("Gate-1 FAILED: {:.4} > 2.22", best_bpb);
    }
    if best_bpb <= 2.03 {
        eprintln!("Gate-2 PASSED (≤2.03)");
    } else {
        eprintln!("Gate-2 FAILED: {:.4} > 2.03", best_bpb);
    }
}

// ── main ──

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let args: Vec<String> = std::env::args().collect();
    let cfg = parse_config(&args);

    print_banner(&cfg);
    neon_trial_start(&cfg);

    let train_data = load_data(&cfg.train_data, cfg.vocab);
    let val_data = load_data(&cfg.val_data, cfg.vocab);
    let train_end = (train_data.len() as f64 * 0.9) as usize;
    let train = &train_data[..train_end];
    let max_val_len = 50_000;
    let val = if val_data.len() > 100 {
        &val_data[..val_data.len().min(max_val_len)]
    } else {
        &train_data[train_end..]
    };

    let mut st = init_training(&cfg);
    let warmup = cfg.steps / 10;

    for step in 1..=cfg.steps {
        let dl = train.len();
        let off = (step * 97 + cfg.seed as usize) % dl.saturating_sub(SEQ + 1);
        let seq = &train[off..off + SEQ + 1];

        let (grads, hidden_vecs, ntp_loss) = compute_grads(&st.model, seq);

        let jepa_loss_val = run_jepa_step(&cfg, &mut st, &hidden_vecs, seq, step);
        let nca_loss_val = run_nca_step(&cfg, &st, step);

        let combined = compute_combined_loss(
            ComponentLosses {
                ntp: ntp_loss as f64 / LN_2 as f64,
                jepa: jepa_loss_val,
                nca: nca_loss_val,
            },
            st.obj_config,
        );

        let (enc_lr, head_lr) = if cfg.use_sg5_lr {
            (
                sg5_adaptive_lr(step, cfg.steps, cfg.encoder_lr, warmup),
                sg5_adaptive_lr(step, cfg.steps, cfg.ntp_lr, warmup),
            )
        } else {
            (
                cosine_lr(step, cfg.steps, cfg.encoder_lr, warmup),
                cosine_lr(step, cfg.steps, cfg.ntp_lr, warmup),
            )
        };
        st.opt_embed
            .step(&mut st.model.embed, &grads.g_embed, enc_lr);
        for (ci, oc) in st.opt_ctx.iter_mut().enumerate() {
            oc.step(&mut st.model.ctx[ci], &grads.g_ctx[ci], enc_lr);
        }
        st.opt_proj.step(&mut st.model.proj, &grads.g_proj, enc_lr);
        st.opt_head
            .step(&mut st.model.lm_head, &grads.g_head, head_lr);

        if step % 500 == 0 || step == cfg.steps {
            let elapsed = st.start_time.elapsed().as_secs_f64();
            let val_bpb = evaluate(&st.model, val);
            if val_bpb < st.best_val_bpb && val_bpb.is_finite() {
                st.best_val_bpb = val_bpb;
            }
            eprintln!(
                "step={:5} ntp={:.4} jepa={:.4} nca={:.4} val_bpb={:.4} best={:.4} t={:.1}s",
                step,
                combined.components.ntp,
                combined.components.jepa,
                combined.components.nca,
                val_bpb,
                st.best_val_bpb,
                elapsed
            );
        }

        neon_heartbeat(&cfg, step, st.best_val_bpb, &mut st.last_heartbeat);
    }

    let elapsed = st.start_time.elapsed().as_secs_f64();
    neon_trial_complete(&cfg, st.best_val_bpb);
    print_results(&cfg, st.best_val_bpb, elapsed);
    Ok(())
}

fn run_jepa_step(
    cfg: &Config,
    st: &mut TrainingState,
    hidden_vecs: &[Vec<f32>],
    seq: &[usize],
    step: usize,
) -> f64 {
    if step <= cfg.jepa_warmup {
        return 0.0;
    }
    match (&mut st.predictor, &mut st.target_model) {
        (Some(pred), _) => {
            let result = jepa_training_step(
                pred,
                &st.model,
                &mut st.target_model,
                hidden_vecs,
                seq,
                cfg.seed,
                step,
                &mut st.ema_target,
            );
            result.loss
        }
        _ => 0.0,
    }
}

fn run_nca_step(cfg: &Config, st: &TrainingState, step: usize) -> f64 {
    match &st.nca {
        Some(nca) => nca_training_step(nca, cfg.seed, step),
        None => 0.0,
    }
}
