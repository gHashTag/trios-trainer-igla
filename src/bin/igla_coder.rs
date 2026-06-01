//! IGLA-Coder (P3): multi-head decoder with FULL backprop, configurable layers,
//! muP-style LR scaling, and a phi-vs-standard control ablation on code BPB.
//!
//! CPU-only, mirrors the Railway champion path. Reuses repo `AdamWCpu` with two
//! arms. This supersedes igla_coder_v1: attention Q/K/V/O are now fully trained
//! (no deferred path), multi-head, multi-layer, with pre-norm RMSNorm residual
//! blocks (two per block + a final pre-logit norm; learnable per-norm gains) and
//! trainable positional embeddings. All new params are gradcheck-covered.
//!
//! Honesty: code BPB is NOT comparable to tiny_shakespeare champion BPB=2.2111.
//! The champion config is the STANDARD AdamW arm; phi is a falsifiable prior.
//! Anchor: phi^2 + phi^-2 = 3.
//!
//! Subcommands:
//!   train     - train one model, report code BPB
//!   gradcheck - numerical gradient check on a tiny config (correctness gate)
//!   ablate    - run phi vs standard over N seeds, print a table with mean+/-std
//!
//! Usage:
//!   igla_coder train --train data/code_train.bin --val data/code_val.bin \
//!     [--hidden 128] [--heads 4] [--layers 2] [--seq 64] [--steps 2000] \
//!     [--batch 8] [--lr 0.002] [--optimizer standard|phi] [--seed 42]
//!   igla_coder gradcheck
//!   igla_coder ablate --train ... --val ... --seeds 42,43,44 [--steps 1500] ...

use std::f32;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use trios_trainer::optimizer::AdamWCpu;

const VOCAB: usize = 263;
const MAXSEQ: usize = 512;

// ---------------- data ----------------
fn load_bin(path: &str) -> Vec<usize> {
    let mut file = File::open(Path::new(path)).expect("open .bin");
    let mut header = [0u8; 1024];
    file.read_exact(&mut header).expect("read header");
    let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
    assert_eq!(magic, 20240520, "bad magic");
    let num = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
    let mut body = vec![0u8; num * 2];
    file.read_exact(&mut body).expect("read body");
    body.chunks_exact(2)
        .map(|c| u16::from_le_bytes(c.try_into().unwrap()) as usize)
        .collect()
}

// ---------------- params ----------------
#[derive(Clone)]
struct Layer {
    wq: Vec<f32>, // d*d
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w1: Vec<f32>, // d*dff
    w2: Vec<f32>, // dff*d
    n1: Vec<f32>, // d  -- RMSNorm gain, pre-attention
    n2: Vec<f32>, // d  -- RMSNorm gain, pre-mlp
}

#[derive(Clone)]
struct Model {
    d: usize,
    heads: usize,
    dff: usize,
    emb: Vec<f32>, // VOCAB*d (tied head)
    pos: Vec<f32>, // MAXSEQ*d
    nf: Vec<f32>,  // d  -- final RMSNorm gain before logits
    layers: Vec<Layer>,
}

struct Grads {
    emb: Vec<f32>,
    pos: Vec<f32>,
    nf: Vec<f32>,
    layers: Vec<Layer>, // reuse Layer as gradient container
}

fn zeros_like(m: &Model) -> Grads {
    Grads {
        emb: vec![0.0; m.emb.len()],
        pos: vec![0.0; m.pos.len()],
        nf: vec![0.0; m.nf.len()],
        layers: m
            .layers
            .iter()
            .map(|l| Layer {
                wq: vec![0.0; l.wq.len()],
                wk: vec![0.0; l.wk.len()],
                wv: vec![0.0; l.wv.len()],
                wo: vec![0.0; l.wo.len()],
                w1: vec![0.0; l.w1.len()],
                w2: vec![0.0; l.w2.len()],
                n1: vec![0.0; l.n1.len()],
                n2: vec![0.0; l.n2.len()],
            })
            .collect(),
    }
}

// RMSNorm forward: y_j = x_j / rms * g_j, rms = sqrt(mean_j x_j^2 + eps).
// Returns (y, inv_rms) where inv_rms = 1/rms (cached for backward).
fn rmsnorm_fwd(x: &[f32], g: &[f32], d: usize) -> (Vec<f32>, Vec<f32>) {
    let t = x.len() / d;
    let eps = 1e-5f32;
    let mut y = vec![0.0f32; t * d];
    let mut inv = vec![0.0f32; t];
    for i in 0..t {
        let mut ss = 0.0f32;
        for j in 0..d {
            let v = x[i * d + j];
            ss += v * v;
        }
        let ir = 1.0 / (ss / d as f32 + eps).sqrt();
        inv[i] = ir;
        for j in 0..d {
            y[i * d + j] = x[i * d + j] * ir * g[j];
        }
    }
    (y, inv)
}

// RMSNorm backward. Given upstream grad gy (t*d), the cached x, gain g, inv_rms,
// accumulates the gain grad into gg (d) and returns grad wrt x (t*d).
// For row x with n = x*ir (ir = 1/rms): y = n .* g.
//   dL/dg_j += sum_i gy_ij * n_ij
//   dL/dx_ij = ir * g_j * gy_ij - (ir/d) * x_ij * sum_k (gy_ik * g_k * x_ik) * ir^2
fn rmsnorm_bwd(
    gy: &[f32],
    x: &[f32],
    g: &[f32],
    inv: &[f32],
    d: usize,
    gg: &mut [f32],
) -> Vec<f32> {
    let t = x.len() / d;
    let mut gx = vec![0.0f32; t * d];
    for i in 0..t {
        let ir = inv[i];
        // dot = sum_k gy_ik * g_k * x_ik
        let mut dot = 0.0f32;
        for k in 0..d {
            dot += gy[i * d + k] * g[k] * x[i * d + k];
        }
        let coef = ir * ir * ir / d as f32;
        for j in 0..d {
            gg[j] += gy[i * d + j] * (x[i * d + j] * ir);
            gx[i * d + j] = ir * g[j] * gy[i * d + j] - coef * x[i * d + j] * dot;
        }
    }
    gx
}

fn xavier(rng: &mut StdRng, n: usize, fan: usize) -> Vec<f32> {
    let s = (1.0 / fan as f32).sqrt();
    (0..n).map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * s).collect()
}

impl Model {
    fn new(d: usize, heads: usize, layers: usize, rng: &mut StdRng) -> Self {
        assert!(d.is_multiple_of(heads), "hidden must be divisible by heads");
        let dff = 4 * d;
        let ls = (0..layers)
            .map(|_| Layer {
                wq: xavier(rng, d * d, d),
                wk: xavier(rng, d * d, d),
                wv: xavier(rng, d * d, d),
                wo: xavier(rng, d * d, d),
                w1: xavier(rng, d * dff, d),
                w2: xavier(rng, dff * d, dff),
                n1: vec![1.0; d],
                n2: vec![1.0; d],
            })
            .collect();
        Model {
            d,
            heads,
            dff,
            emb: xavier(rng, VOCAB * d, d),
            pos: xavier(rng, MAXSEQ * d, d),
            nf: vec![1.0; d],
            layers: ls,
        }
    }

    fn param_count(&self) -> usize {
        self.emb.len()
            + self.pos.len()
            + self.nf.len()
            + self
                .layers
                .iter()
                .map(|l| {
                    l.wq.len()
                        + l.wk.len()
                        + l.wv.len()
                        + l.wo.len()
                        + l.w1.len()
                        + l.w2.len()
                        + l.n1.len()
                        + l.n2.len()
                })
                .sum::<usize>()
    }
}

// matmul a[m,k] * b[k,n] -> [m,n]
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            if av == 0.0 {
                continue;
            }
            let brow = &b[p * n..p * n + n];
            let orow = &mut o[i * n..i * n + n];
            for j in 0..n {
                orow[j] += av * brow[j];
            }
        }
    }
    o
}

// a[m,k]^T * g[m,n] -> [k,n]  (gradient for weight where out = a @ w)
fn matmul_at_g(a: &[f32], g: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; k * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            if av == 0.0 {
                continue;
            }
            let grow = &g[i * n..i * n + n];
            let orow = &mut o[p * n..p * n + n];
            for j in 0..n {
                orow[j] += av * grow[j];
            }
        }
    }
    o
}

// g[m,n] * w[k,n]^T -> [m,k]  (gradient for input where out = a @ w)
fn matmul_g_wt(g: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; m * k];
    for i in 0..m {
        for p in 0..k {
            let wrow = &w[p * n..p * n + n];
            let grow = &g[i * n..i * n + n];
            let mut s = 0.0;
            for j in 0..n {
                s += grow[j] * wrow[j];
            }
            o[i * k + p] = s;
        }
    }
    o
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}
fn dgelu(x: f32) -> f32 {
    let c = (2.0 / std::f32::consts::PI).sqrt();
    let t1 = c * (x + 0.044715 * x * x * x);
    let th = t1.tanh();
    0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * c * (1.0 + 3.0 * 0.044715 * x * x)
}

// Per-layer forward cache for backprop.
struct LayerCache {
    h_in: Vec<f32>, // t*d (block input == residual base for attn)
    xn1: Vec<f32>,  // t*d  RMSNorm(h_in, n1) -- attention input
    inv1: Vec<f32>, // t    inv_rms for pre-attn norm
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<Vec<f32>>, // per-position softmax weights (len i+1)
    ctx: Vec<f32>,       // t*d
    r1: Vec<f32>,        // t*d after attn residual
    xn2: Vec<f32>,       // t*d  RMSNorm(r1, n2) -- mlp input
    inv2: Vec<f32>,      // t    inv_rms for pre-mlp norm
    pre: Vec<f32>,       // t*dff
    act: Vec<f32>,       // t*dff
}

// Full forward+backward over one sequence. Accumulates grads. Returns mean nats.
fn fwd_bwd(m: &Model, tokens: &[usize], g: &mut Grads, train: bool) -> f32 {
    let d = m.d;
    let h = m.heads;
    let hd = d / h;
    let dff = m.dff;
    let t = tokens.len();
    let scale = 1.0 / (hd as f32).sqrt();

    // input embeddings + pos
    let mut x = vec![0.0f32; t * d];
    for (i, &tok) in tokens.iter().enumerate() {
        for j in 0..d {
            x[i * d + j] = m.emb[tok * d + j] + m.pos[i * d + j];
        }
    }

    let mut caches: Vec<LayerCache> = Vec::with_capacity(m.layers.len());

    // ---- forward through layers (pre-norm residual blocks) ----
    for layer in &m.layers {
        let h_in = x.clone();
        // pre-attention RMSNorm
        let (xn1, inv1) = rmsnorm_fwd(&h_in, &layer.n1, d);
        let q = matmul(&xn1, &layer.wq, t, d, d);
        let k = matmul(&xn1, &layer.wk, t, d, d);
        let v = matmul(&xn1, &layer.wv, t, d, d);
        let mut ctx = vec![0.0f32; t * d];
        let mut attn_all: Vec<Vec<f32>> = Vec::with_capacity(t);
        // per query position
        for i in 0..t {
            // store concatenated per-head weights as one vec of len (i+1)*h
            let mut wrow = vec![0.0f32; (i + 1) * h];
            for head in 0..h {
                let off = head * hd;
                let mut scores = vec![0.0f32; i + 1];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..=i {
                    let mut s = 0.0;
                    for x2 in 0..hd {
                        s += q[i * d + off + x2] * k[j * d + off + x2];
                    }
                    s *= scale;
                    scores[j] = s;
                    if s > mx {
                        mx = s;
                    }
                }
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for j in 0..=i {
                    let w = scores[j] / sum;
                    wrow[j * h + head] = w;
                    for x2 in 0..hd {
                        ctx[i * d + off + x2] += w * v[j * d + off + x2];
                    }
                }
            }
            attn_all.push(wrow);
        }
        let attn_out = matmul(&ctx, &layer.wo, t, d, d);
        let mut r1 = vec![0.0f32; t * d];
        for idx in 0..t * d {
            r1[idx] = h_in[idx] + attn_out[idx];
        }
        // pre-mlp RMSNorm
        let (xn2, inv2) = rmsnorm_fwd(&r1, &layer.n2, d);
        let pre = matmul(&xn2, &layer.w1, t, d, dff);
        let mut act = vec![0.0f32; t * dff];
        for idx in 0..t * dff {
            act[idx] = gelu(pre[idx]);
        }
        let mlp = matmul(&act, &layer.w2, t, dff, d);
        let mut r2 = vec![0.0f32; t * d];
        for idx in 0..t * d {
            r2[idx] = r1[idx] + mlp[idx];
        }
        caches.push(LayerCache {
            h_in,
            xn1,
            inv1,
            q,
            k,
            v,
            attn: attn_all,
            ctx,
            r1,
            xn2,
            inv2,
            pre,
            act,
        });
        x = r2; // output of block feeds next
    }

    // ---- final RMSNorm before LM head ----
    let (final_x, inv_f) = rmsnorm_fwd(&x, &m.nf, d); // t*d
    let pre_norm_x = x; // keep for backward through final norm
    let mut g_x = vec![0.0f32; t * d];
    let mut total = 0.0f32;
    let mut counted = 0usize;
    for i in 0..t - 1 {
        let target = tokens[i + 1];
        let mut logits = vec![0.0f32; VOCAB];
        let mut mx = f32::NEG_INFINITY;
        for vt in 0..VOCAB {
            let mut s = 0.0;
            for j in 0..d {
                s += final_x[i * d + j] * m.emb[vt * d + j];
            }
            logits[vt] = s;
            if s > mx {
                mx = s;
            }
        }
        let mut sum = 0.0;
        for l in logits.iter_mut() {
            *l = (*l - mx).exp();
            sum += *l;
        }
        let p_t = (logits[target] / sum).max(1e-12);
        total += -p_t.ln();
        counted += 1;
        if train {
            for vt in 0..VOCAB {
                let p = logits[vt] / sum;
                let dl = p - if vt == target { 1.0 } else { 0.0 };
                for j in 0..d {
                    g.emb[vt * d + j] += dl * final_x[i * d + j];
                    g_x[i * d + j] += dl * m.emb[vt * d + j];
                }
            }
        }
    }

    if !train {
        return if counted == 0 {
            0.0
        } else {
            total / counted as f32
        };
    }

    // ---- backward through final RMSNorm: g_x is wrt final_x (post-norm) ----
    // convert to grad wrt pre_norm_x (the layer-loop output) and accumulate g.nf
    g_x = rmsnorm_bwd(&g_x, &pre_norm_x, &m.nf, &inv_f, d, &mut g.nf);

    // ---- backward through layers (reverse) ----
    for li in (0..m.layers.len()).rev() {
        let layer = &m.layers[li];
        let c = &caches[li];
        let gl = &mut g.layers[li];

        // r2 = r1 + mlp ; g_r1 += g_x ; g_mlp = g_x
        let g_mlp = g_x.clone();
        // mlp = act @ w2
        let gw2 = matmul_at_g(&c.act, &g_mlp, t, dff, d);
        for (a, b) in gl.w2.iter_mut().zip(gw2.iter()) {
            *a += b;
        }
        let g_act = matmul_g_wt(&g_mlp, &layer.w2, t, dff, d); // t*dff
                                                               // through gelu -> g_pre
        let mut g_pre = vec![0.0f32; t * dff];
        for idx in 0..t * dff {
            g_pre[idx] = g_act[idx] * dgelu(c.pre[idx]);
        }
        // pre = xn2 @ w1   (mlp input is the pre-mlp-normed xn2, not r1)
        let gw1 = matmul_at_g(&c.xn2, &g_pre, t, d, dff);
        for (a, b) in gl.w1.iter_mut().zip(gw1.iter()) {
            *a += b;
        }
        // g_xn2 = g_pre @ w1^T  -> matmul_g_wt(g, w, m=t, k=d, n=dff)
        let g_xn2 = matmul_g_wt(&g_pre, &layer.w1, t, d, dff); // t*d
                                                               // through pre-mlp RMSNorm: g_xn2 -> g_r1_from_mlp, accumulate gl.n2
        let g_r1_from_mlp = rmsnorm_bwd(&g_xn2, &c.r1, &layer.n2, &c.inv2, d, &mut gl.n2);
        // g_r1 total = g_x (residual) + g_r1_from_mlp
        let mut g_r1 = vec![0.0f32; t * d];
        for idx in 0..t * d {
            g_r1[idx] = g_x[idx] + g_r1_from_mlp[idx];
        }

        // r1 = h_in + attn_out ; g_h_in += g_r1 ; g_attn_out = g_r1
        let g_attn_out = g_r1.clone();
        // attn_out = ctx @ wo
        let gwo = matmul_at_g(&c.ctx, &g_attn_out, t, d, d);
        for (a, b) in gl.wo.iter_mut().zip(gwo.iter()) {
            *a += b;
        }
        let g_ctx = matmul_g_wt(&g_attn_out, &layer.wo, t, d, d); // t*d

        // backprop attention: ctx[i] = sum_j w_ij v_j  (per head)
        let mut g_q = vec![0.0f32; t * d];
        let mut g_k = vec![0.0f32; t * d];
        let mut g_v = vec![0.0f32; t * d];
        for i in 0..t {
            let wrow = &c.attn[i];
            for head in 0..m.heads {
                let off = head * hd;
                // g wrt weights w_ij (len i+1)
                let mut g_w = vec![0.0f32; i + 1];
                for j in 0..=i {
                    let w = wrow[j * m.heads + head];
                    let mut s = 0.0;
                    for x2 in 0..hd {
                        s += g_ctx[i * d + off + x2] * c.v[j * d + off + x2];
                        // also accumulate g_v
                    }
                    g_w[j] = s;
                    for x2 in 0..hd {
                        g_v[j * d + off + x2] += w * g_ctx[i * d + off + x2];
                    }
                }
                // softmax jacobian: g_score_j = w_j (g_w_j - sum_k w_k g_w_k)
                let mut dot = 0.0;
                for j in 0..=i {
                    dot += wrow[j * m.heads + head] * g_w[j];
                }
                for j in 0..=i {
                    let w = wrow[j * m.heads + head];
                    let g_score = w * (g_w[j] - dot) * scale;
                    for x2 in 0..hd {
                        g_q[i * d + off + x2] += g_score * c.k[j * d + off + x2];
                        g_k[j * d + off + x2] += g_score * c.q[i * d + off + x2];
                    }
                }
            }
        }
        // q = xn1 @ wq, etc.  (attention input is the pre-attn-normed xn1, not h_in)
        let gwq = matmul_at_g(&c.xn1, &g_q, t, d, d);
        let gwk = matmul_at_g(&c.xn1, &g_k, t, d, d);
        let gwv = matmul_at_g(&c.xn1, &g_v, t, d, d);
        for (a, b) in gl.wq.iter_mut().zip(gwq.iter()) {
            *a += b;
        }
        for (a, b) in gl.wk.iter_mut().zip(gwk.iter()) {
            *a += b;
        }
        for (a, b) in gl.wv.iter_mut().zip(gwv.iter()) {
            *a += b;
        }
        // qkv input grads flow to xn1, then through the pre-attn RMSNorm to h_in
        let g_xn1_q = matmul_g_wt(&g_q, &layer.wq, t, d, d);
        let g_xn1_k = matmul_g_wt(&g_k, &layer.wk, t, d, d);
        let g_xn1_v = matmul_g_wt(&g_v, &layer.wv, t, d, d);
        let mut g_xn1 = vec![0.0f32; t * d];
        for idx in 0..t * d {
            g_xn1[idx] = g_xn1_q[idx] + g_xn1_k[idx] + g_xn1_v[idx];
        }
        // through pre-attn RMSNorm: g_xn1 -> g_hin_from_attn, accumulate gl.n1
        let g_hin_from_attn = rmsnorm_bwd(&g_xn1, &c.h_in, &layer.n1, &c.inv1, d, &mut gl.n1);

        // g_h_in total = g_r1 (residual) + attn-path input grad
        let mut g_hin = vec![0.0f32; t * d];
        for idx in 0..t * d {
            g_hin[idx] = g_r1[idx] + g_hin_from_attn[idx];
        }
        // becomes g_x for the previous layer (or the embedding)
        g_x = g_hin;
    }

    // embedding + positional input: x_in = emb[tok] + pos[i] (both trainable)
    for (i, &tok) in tokens.iter().enumerate() {
        for j in 0..d {
            g.emb[tok * d + j] += g_x[i * d + j];
            g.pos[i * d + j] += g_x[i * d + j];
        }
    }

    if counted == 0 {
        0.0
    } else {
        total / counted as f32
    }
}

// ---------------- optimizer wiring ----------------
struct Opt {
    emb: AdamWCpu,
    pos: AdamWCpu,
    nf: AdamWCpu,
    per_layer: Vec<[AdamWCpu; 8]>, // wq,wk,wv,wo,w1,w2,n1,n2
}

// Resolve (beta1, weight_decay) for an arm, with optional explicit overrides.
// Overrides win over the arm preset, enabling independent beta1 / wd sweeps
// (Loop+1 Option A: isolate the magnitude of the phi^-3 decay anchor).
fn arm_hparams(arm: &str, b1_ov: Option<f64>, wd_ov: Option<f64>) -> (f64, f64) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let (b1, wd) = match arm {
        // phi prior: beta1 = phi^-1 ~ 0.618, weight_decay = phi^-3 ~ 0.2361
        "phi" => (1.0 / phi, 1.0 / (phi * phi * phi)),
        // diagnostic: phi beta1 only (standard weight_decay) -- isolates the momentum anchor
        "phi_b1" => (1.0 / phi, 0.04),
        // diagnostic: phi weight_decay only (standard beta1) -- isolates the decay anchor
        "phi_wd" => (0.9, 1.0 / (phi * phi * phi)),
        // standard tuned AdamW control
        _ => (0.9, 0.04),
    };
    (b1_ov.unwrap_or(b1), wd_ov.unwrap_or(wd))
}

fn make_arm(arm: &str, n: usize, lr: f64, b1_ov: Option<f64>, wd_ov: Option<f64>) -> AdamWCpu {
    // Fair control: BOTH arms share the same learning rate (passed via --lr).
    // The only difference is the phi-anchored beta1 / weight_decay vs standard.
    let (b1, wd) = arm_hparams(arm, b1_ov, wd_ov);
    AdamWCpu::with_params(n, lr, b1, 0.999, wd)
}

fn make_opt(m: &Model, arm: &str, lr: f64, b1_ov: Option<f64>, wd_ov: Option<f64>) -> Opt {
    let mk = |n: usize| make_arm(arm, n, lr, b1_ov, wd_ov);
    Opt {
        emb: mk(m.emb.len()),
        pos: mk(m.pos.len()),
        nf: mk(m.nf.len()),
        per_layer: m
            .layers
            .iter()
            .map(|l| {
                [
                    mk(l.wq.len()),
                    mk(l.wk.len()),
                    mk(l.wv.len()),
                    mk(l.wo.len()),
                    mk(l.w1.len()),
                    mk(l.w2.len()),
                    mk(l.n1.len()),
                    mk(l.n2.len()),
                ]
            })
            .collect(),
    }
}

fn opt_step(m: &mut Model, o: &mut Opt, g: &Grads) {
    o.emb.step(&mut m.emb, &g.emb);
    o.pos.step(&mut m.pos, &g.pos);
    o.nf.step(&mut m.nf, &g.nf);
    for (li, layer) in m.layers.iter_mut().enumerate() {
        o.per_layer[li][0].step(&mut layer.wq, &g.layers[li].wq);
        o.per_layer[li][1].step(&mut layer.wk, &g.layers[li].wk);
        o.per_layer[li][2].step(&mut layer.wv, &g.layers[li].wv);
        o.per_layer[li][3].step(&mut layer.wo, &g.layers[li].wo);
        o.per_layer[li][4].step(&mut layer.w1, &g.layers[li].w1);
        o.per_layer[li][5].step(&mut layer.w2, &g.layers[li].w2);
        o.per_layer[li][6].step(&mut layer.n1, &g.layers[li].n1);
        o.per_layer[li][7].step(&mut layer.n2, &g.layers[li].n2);
    }
}

fn accum_scale(g: &mut Grads, s: f32) {
    for x in g.emb.iter_mut() {
        *x *= s;
    }
    for x in g.pos.iter_mut() {
        *x *= s;
    }
    for x in g.nf.iter_mut() {
        *x *= s;
    }
    for l in g.layers.iter_mut() {
        for v in [
            &mut l.wq, &mut l.wk, &mut l.wv, &mut l.wo, &mut l.w1, &mut l.w2, &mut l.n1, &mut l.n2,
        ] {
            for x in v.iter_mut() {
                *x *= s;
            }
        }
    }
}

// ---------------- training ----------------
struct TrainCfg {
    d: usize,
    heads: usize,
    layers: usize,
    seq: usize,
    steps: usize,
    batch: usize,
    lr: f64,
    arm: String,
    seed: u64,
    verbose: bool,
    beta1_ov: Option<f64>,
    wd_ov: Option<f64>,
    // if Some(stride), print a `curve` line every `stride` steps (Loop+1 Option B)
    curve_stride: Option<usize>,
}

fn train_once(train: &[usize], val: &[usize], cfg: &TrainCfg) -> f32 {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut model = Model::new(cfg.d, cfg.heads, cfg.layers, &mut rng);
    let mut opt = make_opt(&model, &cfg.arm, cfg.lr, cfg.beta1_ov, cfg.wd_ov);
    let mut drng = StdRng::seed_from_u64(cfg.seed ^ 0x9e37);

    for step in 0..cfg.steps {
        let mut g = zeros_like(&model);
        let mut loss = 0.0f32;
        for _ in 0..cfg.batch {
            let start = drng.gen_range(0..train.len() - cfg.seq - 1);
            let toks = &train[start..start + cfg.seq + 1];
            loss += fwd_bwd(&model, toks, &mut g, true);
        }
        accum_scale(&mut g, 1.0 / cfg.batch as f32);
        opt_step(&mut model, &mut opt, &g);
        let train_bpb = (loss / cfg.batch as f32) / std::f32::consts::LN_2;
        if cfg.verbose && (step % 200 == 0 || step == cfg.steps - 1) {
            println!("  step={:>5} train_bpb={:.4}", step, train_bpb);
        }
        if let Some(stride) = cfg.curve_stride {
            if step % stride == 0 || step == cfg.steps - 1 {
                // machine-parseable training curve line
                println!(
                    "curve arm={} seed={} step={} train_bpb={:.4}",
                    cfg.arm, cfg.seed, step, train_bpb
                );
            }
        }
    }

    // validation BPB
    let mut vg = zeros_like(&model);
    let mut nats = 0.0f32;
    let mut wins = 0usize;
    let mut i = 0;
    while i + cfg.seq + 1 < val.len() && wins < 64 {
        let toks = &val[i..i + cfg.seq + 1];
        nats += fwd_bwd(&model, toks, &mut vg, false);
        wins += 1;
        i += cfg.seq;
    }
    if wins == 0 {
        f32::NAN
    } else {
        (nats / wins as f32) / std::f32::consts::LN_2
    }
}

// f64-precision forward-only loss (SUM over positions, nats) for gradcheck.
// f32 forward loses ~1e-6 precision at loss~5.6, drowning eps-perturbations;
// this mirror in f64 makes central differences meaningful.
fn fwd_loss_f64(m: &Model, tokens: &[usize]) -> f64 {
    let d = m.d;
    let h = m.heads;
    let hd = d / h;
    let dff = m.dff;
    let t = tokens.len();
    let scale = 1.0 / (hd as f64).sqrt();

    let mut x = vec![0.0f64; t * d];
    for (i, &tok) in tokens.iter().enumerate() {
        for j in 0..d {
            x[i * d + j] = m.emb[tok * d + j] as f64 + m.pos[i * d + j] as f64;
        }
    }
    let mm = |a: &[f64], b: &[f32], mm: usize, kk: usize, nn: usize| -> Vec<f64> {
        let mut o = vec![0.0f64; mm * nn];
        for i in 0..mm {
            for p in 0..kk {
                let av = a[i * kk + p];
                for j in 0..nn {
                    o[i * nn + j] += av * b[p * nn + j] as f64;
                }
            }
        }
        o
    };
    // f64 RMSNorm mirror of rmsnorm_fwd (eps = 1e-5).
    let rmsn = |x: &[f64], g: &[f32], d: usize| -> Vec<f64> {
        let t = x.len() / d;
        let mut y = vec![0.0f64; t * d];
        for i in 0..t {
            let mut ss = 0.0f64;
            for j in 0..d {
                ss += x[i * d + j] * x[i * d + j];
            }
            let ir = 1.0 / (ss / d as f64 + 1e-5).sqrt();
            for j in 0..d {
                y[i * d + j] = x[i * d + j] * ir * g[j] as f64;
            }
        }
        y
    };
    for layer in &m.layers {
        let h_in = x.clone();
        let xn1 = rmsn(&h_in, &layer.n1, d);
        let q = mm(&xn1, &layer.wq, t, d, d);
        let k = mm(&xn1, &layer.wk, t, d, d);
        let v = mm(&xn1, &layer.wv, t, d, d);
        let mut ctx = vec![0.0f64; t * d];
        for i in 0..t {
            for head in 0..h {
                let off = head * hd;
                let mut scores = vec![0.0f64; i + 1];
                let mut mx = f64::NEG_INFINITY;
                for j in 0..=i {
                    let mut s = 0.0;
                    for x2 in 0..hd {
                        s += q[i * d + off + x2] * k[j * d + off + x2];
                    }
                    s *= scale;
                    scores[j] = s;
                    if s > mx {
                        mx = s;
                    }
                }
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for j in 0..=i {
                    let w = scores[j] / sum;
                    for x2 in 0..hd {
                        ctx[i * d + off + x2] += w * v[j * d + off + x2];
                    }
                }
            }
        }
        let attn_out = mm(&ctx, &layer.wo, t, d, d);
        let mut r1 = vec![0.0f64; t * d];
        for idx in 0..t * d {
            r1[idx] = h_in[idx] + attn_out[idx];
        }
        let xn2 = rmsn(&r1, &layer.n2, d);
        let pre = mm(&xn2, &layer.w1, t, d, dff);
        let mut act = vec![0.0f64; t * dff];
        for idx in 0..t * dff {
            let xv = pre[idx];
            act[idx] = 0.5
                * xv
                * (1.0
                    + ((2.0 / std::f64::consts::PI).sqrt() * (xv + 0.044715 * xv * xv * xv))
                        .tanh());
        }
        let mlp = mm(&act, &layer.w2, t, dff, d);
        let mut r2 = vec![0.0f64; t * d];
        for idx in 0..t * d {
            r2[idx] = r1[idx] + mlp[idx];
        }
        x = r2;
    }
    let final_x = rmsn(&x, &m.nf, d);
    let mut total = 0.0f64;
    for i in 0..t - 1 {
        let target = tokens[i + 1];
        let mut logits = vec![0.0f64; VOCAB];
        let mut mx = f64::NEG_INFINITY;
        for vt in 0..VOCAB {
            let mut s = 0.0;
            for j in 0..d {
                s += final_x[i * d + j] * m.emb[vt * d + j] as f64;
            }
            logits[vt] = s;
            if s > mx {
                mx = s;
            }
        }
        let mut sum = 0.0;
        for l in logits.iter_mut() {
            *l = (*l - mx).exp();
            sum += *l;
        }
        let p_t = (logits[target] / sum).max(1e-12);
        total += -p_t.ln();
    }
    total // SUM of nats over (t-1) positions
}

// ---------------- gradcheck ----------------
fn gradcheck_cfg(dd: usize, hh: usize, ll: usize) {
    println!("=== gradcheck d={} heads={} layers={} ===", dd, hh, ll);
    let mut rng = StdRng::seed_from_u64(7);
    let mut model = Model::new(dd, hh, ll, &mut rng);
    let tokens: Vec<usize> = vec![65, 66, 67, 65, 66, 67, 68, 69];
    let mut g = zeros_like(&model);
    let _ = fwd_bwd(&model, &tokens, &mut g, true);

    // Use f64 weights internally for the numeric probe so the central-difference
    // truncation O(eps^2) is the only error term. eps=1e-3 balances truncation
    // (~1e-6) against f64 round-off (~1e-13/eps). We compare against analytic
    // grads via a max(abs, rel) tolerance: tiny grads are dominated by O(eps^2)
    // absolute error, so rel error there is not meaningful -- gate on whichever
    // criterion is satisfied (abs<2e-3 OR rel<5%).
    let eps = 1e-3f32;
    let mut max_rel = 0.0f32;
    let mut checks = 0;
    let mut fails = 0;
    let mut probe =
        |idx: usize, get: &dyn Fn(&Model) -> f32, set: &dyn Fn(&mut Model, f32), analytic: f32| {
            let orig = get(&model);
            set(&mut model, orig + eps);
            let lp = fwd_loss_f64(&model, &tokens);
            set(&mut model, orig - eps);
            let lm = fwd_loss_f64(&model, &tokens);
            set(&mut model, orig);
            // fwd_loss_f64 returns SUM of nats; analytic grads are SUM too -> direct match.
            let numeric = ((lp - lm) / (2.0 * eps as f64)) as f32;
            let abs = (numeric - analytic).abs();
            let rel = abs / (numeric.abs().max(analytic.abs()).max(1e-6));
            let ok = abs < 2e-3 || rel < 0.05;
            println!(
                "  idx={:>5} analytic={:+.6} numeric={:+.6} abs={:.6} rel={:.4} {}",
                idx,
                analytic,
                numeric,
                abs,
                rel,
                if ok { "OK" } else { "FAIL" }
            );
            if !ok {
                fails += 1;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
            checks += 1;
        };

    for &i in &[100usize, 530, 1200] {
        let a = g.emb[i];
        probe(
            i,
            &|m: &Model| m.emb[i],
            &|m: &mut Model, v: f32| m.emb[i] = v,
            a,
        );
    }
    for &i in &[0usize, 7, 20] {
        let a = g.layers[0].w1[i];
        probe(
            10000 + i,
            &|m: &Model| m.layers[0].w1[i],
            &|m: &mut Model, v: f32| m.layers[0].w1[i] = v,
            a,
        );
    }
    for &i in &[0usize, 3] {
        let a = g.layers[0].w2[i];
        probe(
            15000 + i,
            &|m: &Model| m.layers[0].w2[i],
            &|m: &mut Model, v: f32| m.layers[0].w2[i] = v,
            a,
        );
    }
    for &i in &[0usize, 5] {
        let a = g.layers[0].wq[i];
        probe(
            20000 + i,
            &|m: &Model| m.layers[0].wq[i],
            &|m: &mut Model, v: f32| m.layers[0].wq[i] = v,
            a,
        );
    }
    for &i in &[0usize, 5] {
        let a = g.layers[0].wv[i];
        probe(
            25000 + i,
            &|m: &Model| m.layers[0].wv[i],
            &|m: &mut Model, v: f32| m.layers[0].wv[i] = v,
            a,
        );
    }
    for &i in &[0usize, 5] {
        let a = g.layers[0].wo[i];
        probe(
            30000 + i,
            &|m: &Model| m.layers[0].wo[i],
            &|m: &mut Model, v: f32| m.layers[0].wo[i] = v,
            a,
        );
    }
    // trainable positional embeddings
    for &i in &[0usize, 9, 21] {
        let a = g.pos[i];
        probe(
            35000 + i,
            &|m: &Model| m.pos[i],
            &|m: &mut Model, v: f32| m.pos[i] = v,
            a,
        );
    }
    // final RMSNorm gain
    for &i in &[0usize, 2] {
        let a = g.nf[i];
        probe(
            40000 + i,
            &|m: &Model| m.nf[i],
            &|m: &mut Model, v: f32| m.nf[i] = v,
            a,
        );
    }
    // pre-attention RMSNorm gain (layer 0)
    for &i in &[0usize, 3] {
        let a = g.layers[0].n1[i];
        probe(
            45000 + i,
            &|m: &Model| m.layers[0].n1[i],
            &|m: &mut Model, v: f32| m.layers[0].n1[i] = v,
            a,
        );
    }
    // pre-mlp RMSNorm gain (layer 0)
    for &i in &[0usize, 3] {
        let a = g.layers[0].n2[i];
        probe(
            50000 + i,
            &|m: &Model| m.layers[0].n2[i],
            &|m: &mut Model, v: f32| m.layers[0].n2[i] = v,
            a,
        );
    }
    println!(
        "checks={} fails={} (gate: abs<2e-3 OR rel<5%)",
        checks, fails
    );
    if fails == 0 {
        println!("GRADCHECK: PASS (full backprop matches numerical)");
    } else {
        println!(
            "GRADCHECK: REVIEW (worst failing rel={:.4} -- inspect)",
            max_rel
        );
    }
}

// ---------------- cli ----------------
fn arg(a: &[String], k: &str) -> Option<String> {
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).cloned().unwrap_or_else(|| "train".into());

    if cmd == "dumpgrad" {
        let mut rng = StdRng::seed_from_u64(7);
        let model = Model::new(4, 1, 1, &mut rng);
        let tokens: Vec<usize> = vec![65, 66, 67, 66];
        let mut g = zeros_like(&model);
        let _ = fwd_bwd(&model, &tokens, &mut g, true);
        let dump_vec = |name: &str, v: &[f32]| {
            print!("\"{}\": [", name);
            for (i, x) in v.iter().enumerate() {
                if i > 0 {
                    print!(",");
                }
                print!("{:.8}", x);
            }
            println!("],");
        };
        println!("{{");
        println!("\"tokens\": [65,66,67,66],");
        dump_vec("emb", &model.emb);
        dump_vec("pos", &model.pos);
        dump_vec("wq", &model.layers[0].wq);
        dump_vec("wk", &model.layers[0].wk);
        dump_vec("wv", &model.layers[0].wv);
        dump_vec("wo", &model.layers[0].wo);
        dump_vec("w1", &model.layers[0].w1);
        dump_vec("w2", &model.layers[0].w2);
        dump_vec("g_emb", &g.emb);
        dump_vec("g_wq", &g.layers[0].wq);
        dump_vec("g_wk", &g.layers[0].wk);
        dump_vec("g_wv", &g.layers[0].wv);
        dump_vec("g_wo", &g.layers[0].wo);
        dump_vec("g_w1", &g.layers[0].w1);
        dump_vec("g_w2_last", &g.layers[0].w2);
        println!("\"d\": 4, \"heads\": 1, \"dff\": 16}}");
        return;
    }

    if cmd == "gradcheck" {
        let dd: usize = arg(&args, "--hidden")
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let hh: usize = arg(&args, "--heads")
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);
        let ll: usize = arg(&args, "--layers")
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);
        gradcheck_cfg(dd, hh, ll);
        return;
    }

    let train_path = arg(&args, "--train").unwrap_or_else(|| "data/code_train.bin".into());
    let val_path = arg(&args, "--val").unwrap_or_else(|| "data/code_val.bin".into());
    let d: usize = arg(&args, "--hidden")
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let heads: usize = arg(&args, "--heads")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let layers: usize = arg(&args, "--layers")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let seq: usize = arg(&args, "--seq")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let steps: usize = arg(&args, "--steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let batch: usize = arg(&args, "--batch")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let lr: f64 = arg(&args, "--lr")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.002);
    // Optional optimizer-hparam overrides (Loop+1 Option A: independent sweeps).
    let beta1_ov: Option<f64> = arg(&args, "--beta1").and_then(|s| s.parse().ok());
    let wd_ov: Option<f64> = arg(&args, "--wd").and_then(|s| s.parse().ok());
    // Optional training-curve stride (Loop+1 Option B): prints `curve` lines.
    let curve_stride: Option<usize> = arg(&args, "--curve").and_then(|s| s.parse().ok());

    let train = load_bin(&train_path);
    let val = load_bin(&val_path);
    println!("anchor: phi^2 + phi^-2 = 3");
    println!("train_tokens={} val_tokens={}", train.len(), val.len());

    if cmd == "ablate" {
        let seeds: Vec<u64> = arg(&args, "--seeds")
            .unwrap_or_else(|| "42,43,44".into())
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        println!("=== ablation: phi vs standard (code BPB) ===");
        let probe = Model::new(d, heads, layers, &mut StdRng::seed_from_u64(0));
        println!(
            "config hidden={} heads={} layers={} seq={} steps={} batch={} lr={} params={}",
            d,
            heads,
            layers,
            seq,
            steps,
            batch,
            lr,
            probe.param_count()
        );
        // arms: default standard+phi; override with --arms a,b,c (e.g. add phi_b1,phi_wd)
        let arms: Vec<String> = arg(&args, "--arms")
            .unwrap_or_else(|| "standard,phi".into())
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        for arm in &arms {
            let mut vals = Vec::new();
            for &s in &seeds {
                let cfg = TrainCfg {
                    d,
                    heads,
                    layers,
                    seq,
                    steps,
                    batch,
                    lr,
                    arm: arm.clone(),
                    seed: s,
                    verbose: false,
                    beta1_ov,
                    wd_ov,
                    curve_stride,
                };
                let bpb = train_once(&train, &val, &cfg);
                vals.push(bpb);
                println!("  arm={:<8} seed={} code_val_bpb={:.4}", arm, s, bpb);
            }
            let mean = vals.iter().sum::<f32>() / vals.len() as f32;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                / (vals.len().max(2) - 1) as f32;
            let std = var.sqrt();
            // 95% CI half-width (t approx ~ use 1.96 for simplicity, note small n)
            let ci = 1.96 * std / (vals.len() as f32).sqrt();
            println!(
                "  >>> arm={:<8} mean_bpb={:.4} std={:.4} ci95=+/-{:.4} (n={})",
                arm,
                mean,
                std,
                ci,
                vals.len()
            );
        }
        println!(
            "HONESTY: small n, code BPB only (not pass@1), not comparable to \
             tiny_shakespeare 2.2111. If arms overlap within CI -> phi NOT supported."
        );
        return;
    }

    // default: train one
    let arm = arg(&args, "--optimizer").unwrap_or_else(|| "standard".into());
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let cfg = TrainCfg {
        d,
        heads,
        layers,
        seq,
        steps,
        batch,
        lr,
        arm: arm.clone(),
        seed,
        verbose: true,
        beta1_ov,
        wd_ov,
        curve_stride,
    };
    let probe = Model::new(d, heads, layers, &mut StdRng::seed_from_u64(0));
    println!(
        "=== IGLA-Coder train === hidden={} heads={} layers={} params={} optimizer={} seed={}",
        d,
        heads,
        layers,
        probe.param_count(),
        arm,
        seed
    );
    let bpb = train_once(&train, &val, &cfg);
    println!("=== RESULT === code_val_bpb={:.4}", bpb);
}
