use std::fs;
use std::io::Write;
use std::time::Instant;

use trios_trainer::model_hybrid_attn::{HybridAttn, HybridAttnConfig, AttentionCache};

const LN_2: f32 = std::f32::consts::LN_2;

fn load_data(path: &str) -> Vec<usize> {
    if path.ends_with(".bin") && std::path::Path::new(path).exists() {
        return load_fineweb_bin(path);
    }
    let raw = fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to load {}: {}. Using fallback.", path, e);
        b"Hello world this is a tiny training dataset for IGLA".to_vec()
    });
    raw.into_iter().map(|b| b as usize).collect()
}

fn load_fineweb_bin(path: &str) -> Vec<usize> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| {
        panic!("open {}: {}", path, e);
    });
    let mut header = [0u8; 1024];
    file.read_exact(&mut header).expect("read header");
    let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    let num_tokens = u64::from_le_bytes([
        header[8], header[9], header[10], header[11],
        header[12], header[13], header[14], header[15],
    ]);
    eprintln!("FineWeb bin: path={} magic={} version={} num_tokens={}", path, magic, version, num_tokens);
    assert_eq!(magic, 20240520, "FineWeb magic mismatch");
    assert_eq!(version, 1, "FineWeb version mismatch");
    let mut tokens = vec![0u16; num_tokens as usize];
    let mut buf = vec![0u8; num_tokens as usize * 2];
    file.read_exact(&mut buf).expect("read tokens");
    for i in 0..tokens.len() {
        tokens[i] = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
    }
    tokens.into_iter().map(|t| t as usize).collect()
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

    /// Returns (d_w1, d_b1, d_w2, d_b2, d_input)
    #[allow(clippy::needless_range_loop)]
    fn backward(&self, x: &[f32], grad_out: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
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

        (d_w1, d_b1, d_w2, d_b2, d_input)
    }
}

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

fn layer_norm_rows(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * cols, "layer_norm_rows shape");
    let eps = 1e-5_f32;
    let mut out = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let n = cols as f32;
        let mean = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let std_inv = 1.0 / (var + eps).sqrt();
        for c in 0..cols {
            out[r * cols + c] = (row[c] - mean) * std_inv;
        }
    }
    out
}

fn layer_norm_rows_backward(
    x: &[f32],
    y: &[f32],
    dy: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let eps = 1e-5_f32;
    let mut dx = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row_x = &x[r * cols..(r + 1) * cols];
        let row_y = &y[r * cols..(r + 1) * cols];
        let row_dy = &dy[r * cols..(r + 1) * cols];
        let n = cols as f32;
        let mean: f32 = row_x.iter().sum::<f32>() / n;
        let var: f32 = row_x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let std_inv = 1.0 / (var + eps).sqrt();
        let sum_dy: f32 = row_dy.iter().sum();
        let sum_dy_y: f32 = row_dy.iter().zip(row_y.iter()).map(|(d, yi)| d * yi).sum();
        for c in 0..cols {
            dx[r * cols + c] = (row_dy[c] - sum_dy / n - row_y[c] * sum_dy_y / n) * std_inv;
        }
    }
    dx
}

struct AttnBlock {
    attn: HybridAttn,
    ffn: FFNLayer,
    opt_wq: AdamW,
    opt_wk: AdamW,
    opt_wv: AdamW,
    opt_wo: AdamW,
    opt_wq2: AdamW,
    opt_wk2: AdamW,
    opt_wv2: AdamW,
    opt_wo2: AdamW,
    opt_w1: AdamW,
    opt_b1: AdamW,
    opt_w2: AdamW,
    opt_b2: AdamW,
}

impl AttnBlock {
    fn new(dim: usize, seed: &mut u64, lr: f32) -> Self {
        let mut cfg = HybridAttnConfig::default();
        cfg.d_model = dim;
        cfg.num_heads = 4;
        cfg.seq_len = 64;
        cfg.num_attn_layers = 2;
        let attn = HybridAttn::with_config(cfg).expect("valid attn config");
        let ffn = FFNLayer::new(dim, dim * 4, seed);
        let dd = dim * dim;
        Self {
            attn,
            ffn,
            opt_wq: AdamW::new(dd, lr),
            opt_wk: AdamW::new(dd, lr),
            opt_wv: AdamW::new(dd, lr),
            opt_wo: AdamW::new(dd, lr),
            opt_wq2: AdamW::new(dd, lr),
            opt_wk2: AdamW::new(dd, lr),
            opt_wv2: AdamW::new(dd, lr),
            opt_wo2: AdamW::new(dd, lr),
            opt_w1: AdamW::new(dim * 4 * dim, lr),
            opt_b1: AdamW::new(dim * 4, lr),
            opt_w2: AdamW::new(dim * dim * 4, lr),
            opt_b2: AdamW::new(dim, lr),
        }
    }
}

struct BlockCache {
    x_before_attn: Vec<f32>,
    x_after_attn: Vec<f32>,
    x_after_ln: Vec<f32>,
    ffn_out: Vec<f32>,
    attn_cache: AttentionCache,
}

struct CpuModel {
    embed: Vec<f32>,
    lm_head: Vec<f32>,
    bigram: BigramHash,
    smear: SmearGate,
    blocks: Vec<AttnBlock>,
    bigram_scale: f32,
    vocab: usize,
    dim: usize,
}

fn flatten_2d(xs: &[Vec<f32>]) -> Vec<f32> {
    let mut out = Vec::with_capacity(xs.len() * xs[0].len());
    for x in xs {
        out.extend_from_slice(x);
    }
    out
}

fn unflatten_2d(flat: &[f32], seq_len: usize, d: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        out.push(flat[i * d..(i + 1) * d].to_vec());
    }
    out
}

impl CpuModel {
    fn new(vocab: usize, dim: usize, seed: u64, num_blocks: usize, lr: f32) -> Self {
        let mut s = seed;
        let embed: Vec<f32> = (0..vocab * dim).map(|_| rng_next(&mut s) * 0.02).collect();
        let lm_head: Vec<f32> = (0..vocab * dim).map(|_| rng_next(&mut s) * 0.02).collect();
        let bigram = BigramHash::new(vocab, dim, &mut s);
        let smear = SmearGate::new(dim);

        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(AttnBlock::new(dim, &mut s, lr));
        }

        Self {
            embed,
            lm_head,
            bigram,
            smear,
            blocks,
            bigram_scale: 0.1,
            vocab,
            dim,
        }
    }

    fn forward_logits(&self, tokens: &[usize]) -> Vec<Vec<f32>> {
        let d = self.dim;
        let v = self.vocab;
        let n = tokens.len();

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
        let mut x_flat = flatten_2d(&xs);

        for block in &self.blocks {
            let (attn_out, _) = block.attn.forward_with_cache(&x_flat, n).expect("attn forward");
            let x_ln = layer_norm_rows(&attn_out, n, d);
            let mut ffn_out = vec![0.0f32; n * d];
            for i in 0..n {
                let row = &x_ln[i * d..(i + 1) * d];
                let ffn_row = block.ffn.forward(row);
                for j in 0..d {
                    ffn_out[i * d + j] = ffn_row[j];
                }
            }
            for i in 0..n * d {
                x_flat[i] = attn_out[i] + ffn_out[i];
            }
        }

        let xs_final = unflatten_2d(&x_flat, n, d);

        let mut logits = Vec::with_capacity(n);
        for x in &xs_final {
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
        let mut x_flat = flatten_2d(&xs_smeared);

        for block in &self.blocks {
            let (attn_out, _) = block.attn.forward_with_cache(&x_flat, n).expect("attn forward");
            let x_ln = layer_norm_rows(&attn_out, n, d);
            let mut ffn_out = vec![0.0f32; n * d];
            for i in 0..n {
                let row = &x_ln[i * d..(i + 1) * d];
                let ffn_row = block.ffn.forward(row);
                for j in 0..d {
                    ffn_out[i * d + j] = ffn_row[j];
                }
            }
            for i in 0..n * d {
                x_flat[i] = attn_out[i] + ffn_out[i];
            }
        }

        let xs_final = unflatten_2d(&x_flat, n, d);

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
        opt_embed: &mut AdamW,
        opt_head: &mut AdamW,
    ) -> f32 {
        let d = self.dim;
        let v = self.vocab;
        let n = tokens.len();

        // Forward through embed + bigram + smear
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
        let mut x_flat = flatten_2d(&xs_smeared);

        // Forward through blocks, caching activations
        let mut block_caches: Vec<BlockCache> = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (attn_out, attn_cache) = block.attn.forward_with_cache(&x_flat, n).expect("attn forward");
            let x_ln = layer_norm_rows(&attn_out, n, d);
            let mut ffn_out = vec![0.0f32; n * d];
            for i in 0..n {
                let row = &x_ln[i * d..(i + 1) * d];
                let ffn_row = block.ffn.forward(row);
                for j in 0..d {
                    ffn_out[i * d + j] = ffn_row[j];
                }
            }
            let x_next: Vec<f32> = (0..n * d).map(|i| attn_out[i] + ffn_out[i]).collect();

            block_caches.push(BlockCache {
                x_before_attn: x_flat,
                x_after_attn: attn_out,
                x_after_ln: x_ln,
                ffn_out: ffn_out.clone(),
                attn_cache,
            });

            x_flat = x_next;
        }

        let xs_final = unflatten_2d(&x_flat, n, d);

        // Compute logits, loss, and d_logits
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

        // Gradient w.r.t. LM head
        let mut d_lm_head = vec![0.0f32; v * d];
        for i in 0..n - 1 {
            for (vi, &dl) in d_logits[i].iter().enumerate() {
                for (j, xf) in xs_final[i].iter().enumerate() {
                    d_lm_head[vi * d + j] += dl * xf;
                }
            }
        }

        // Gradient from LM head back to hidden states
        let mut d_upstream = vec![0.0f32; n * d];
        for i in 0..n - 1 {
            for (vi, &dl) in d_logits[i].iter().enumerate() {
                for j in 0..d {
                    d_upstream[i * d + j] += dl * self.lm_head[vi * d + j];
                }
            }
        }

        // Backward through blocks in reverse
        for (block_idx, block) in self.blocks.iter_mut().enumerate().rev() {
            let cache = &block_caches[block_idx];

            // Residual: x_out = x_after_attn + ffn_out
            // d_x_after_attn = d_upstream, d_ffn = d_upstream
            let mut d_x_after_attn = d_upstream.clone();
            let d_ffn_flat = d_upstream.clone();

            // Backprop through FFN (per position)
            let mut d_x_ln = vec![0.0f32; n * d];
            for i in 0..n {
                let x_ln_row = &cache.x_after_ln[i * d..(i + 1) * d];
                let d_ffn_row = &d_ffn_flat[i * d..(i + 1) * d];
                let (dw1, db1, dw2, db2, d_input_row) = block.ffn.backward(x_ln_row, d_ffn_row);
                block.opt_w1.step(&mut block.ffn.w1, &dw1);
                block.opt_b1.step(&mut block.ffn.b1, &db1);
                block.opt_w2.step(&mut block.ffn.w2, &dw2);
                block.opt_b2.step(&mut block.ffn.b2, &db2);
                for j in 0..d {
                    d_x_ln[i * d + j] = d_input_row[j];
                }
            }

            // Backprop through pre-FFN LayerNorm
            let d_from_ln = layer_norm_rows_backward(&cache.x_after_attn, &cache.x_after_ln, &d_x_ln, n, d);
            for i in 0..n * d {
                d_x_after_attn[i] += d_from_ln[i];
            }

            // Backprop through attention
            let grads = block.attn.backward(&cache.attn_cache, &d_x_after_attn);
            block.opt_wq.step(block.attn.wq_mut(), &grads.gwq);
            block.opt_wk.step(block.attn.wk_mut(), &grads.gwk);
            block.opt_wv.step(block.attn.wv_mut(), &grads.gwv);
            block.opt_wo.step(block.attn.wo_mut(), &grads.gwo);
            if !grads.gwq2.is_empty() {
                block.opt_wq2.step(block.attn.wq2_mut(), &grads.gwq2);
                block.opt_wk2.step(block.attn.wk2_mut(), &grads.gwk2);
                block.opt_wv2.step(block.attn.wv2_mut(), &grads.gwv2);
                block.opt_wo2.step(block.attn.wo2_mut(), &grads.gwo2);
            }

            d_upstream = grads.d_input;
        }

        // d_upstream now flows to embed / bigram / smear
        let d_to_embed = unflatten_2d(&d_upstream, n, d);

        // Update LM head
        opt_head.step(&mut self.lm_head, &d_lm_head);

        // Update embeddings
        let mut d_embed = vec![0.0f32; v * d];
        for (i, &tid) in tokens.iter().enumerate() {
            let id = tid % v;
            let gi = i.min(n - 2);
            for (j, &dh) in d_to_embed[gi].iter().enumerate().take(d) {
                d_embed[id * d + j] += dh;
            }
        }
        opt_embed.step(&mut self.embed, &d_embed);

        // Update bigram and smear
        let lr_embed = opt_embed.lr; // use embed lr for bigram/smear
        self.bigram.grad_step(tokens, &d_to_embed, lr_embed);
        self.smear.grad_step(&d_to_embed, lr_embed);

        loss
    }

    fn eval_bpb(&self, tokens: &[usize], seq_len: usize) -> f32 {
        let max_eval = 5000.min(tokens.len());
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
            return f32::MAX;
        }
        total_bpb / n as f32
    }
}

fn main() {
    let format_type = std::env::var("TRIOS_FORMAT_TYPE").ok();
    let seed = arg_or("seed", "42").parse::<u64>().unwrap_or(42);
    let steps = arg_or("steps", "3000").parse::<usize>().unwrap_or(3000);
    let lr = arg_or("lr", "0.003").parse::<f32>().unwrap_or(0.003);
    let vocab: usize = arg_or("vocab", "128").parse().unwrap_or(128);
    let dim: usize = arg_or("dim", "96").parse().unwrap_or(96);
    let seq: usize = arg_or("seq", "32").parse().unwrap_or(32);
    let num_blocks: usize = arg_or("blocks", "1").parse().unwrap_or(1);

    let default_format = "f32".to_string();
    let format_suffix = format_type.as_ref().unwrap_or(&default_format);

    let train_path = arg_or("train", "data/tinyshakespeare.txt");
    let val_path = arg_or("val", "");
    let raw_tokens = load_data(&train_path);
    let tokens: Vec<usize> = raw_tokens.iter().map(|&t| t % vocab).collect();
    
    let val_tokens: Vec<usize> = if !val_path.is_empty() {
        load_data(&val_path).iter().map(|&t| t % vocab).collect()
    } else {
        let train_end = (tokens.len() as f64 * 0.9) as usize;
        tokens[train_end..].to_vec()
    };

    println!("=== trios CPU Training (Deep Multi-Block) ===");
    println!(
        "vocab={} dim={} seq={} steps={} seed={} lr={} blocks={}",
        vocab, dim, seq, steps, seed, lr, num_blocks
    );

    let train_tokens: Vec<usize>;
    let val_tokens_ref: &[usize];
    if !val_path.is_empty() {
        train_tokens = tokens;
        val_tokens_ref = &val_tokens;
    } else {
        let train_end = (tokens.len() as f64 * 0.9) as usize;
        train_tokens = tokens[..train_end].to_vec();
        val_tokens_ref = &tokens[train_end..];
    }
    println!(
        "Dataset: {} train / {} val tokens",
        train_tokens.len(),
        val_tokens_ref.len()
    );

    let mut model = CpuModel::new(vocab, dim, seed, num_blocks, lr);
    let mut opt_embed = AdamW::new(vocab * dim, lr);
    let mut opt_head = AdamW::new(vocab * dim, lr);

    let init_bpb = model.eval_bpb(&val_tokens, seq);
    println!("Initial val BPB: {:.4}", init_bpb);
    println!();
    println!(
        "{:>6} | {:>10} | {:>10} | {:>10} | {:>8}",
        "step", "train_loss", "val_bpb", "best_bpb", "ms"
    );
    println!("{}", "-".repeat(60));

    let t0 = Instant::now();
    let mut best_bpb = init_bpb;
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

        // Update all optimizers' learning rates
        opt_embed.lr = current_lr;
        opt_head.lr = current_lr;
        for block in &mut model.blocks {
            block.opt_wq.lr = current_lr;
            block.opt_wk.lr = current_lr;
            block.opt_wv.lr = current_lr;
            block.opt_wo.lr = current_lr;
            block.opt_wq2.lr = current_lr;
            block.opt_wk2.lr = current_lr;
            block.opt_wv2.lr = current_lr;
            block.opt_wo2.lr = current_lr;
            block.opt_w1.lr = current_lr;
            block.opt_b1.lr = current_lr;
            block.opt_w2.lr = current_lr;
            block.opt_b2.lr = current_lr;
        }

        let offset = {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng_state as usize) % (data_len.saturating_sub(seq + 1))
        };
        let batch = &train_tokens[offset..offset + seq + 1];
        let train_loss = model.train_step(batch, &mut opt_embed, &mut opt_head);

        if step % 500 == 0 || step == steps {
            let ms = t0.elapsed().as_millis();
            let val_bpb = model.eval_bpb(val_tokens_ref, seq);
            if val_bpb < best_bpb && val_bpb.is_finite() {
                best_bpb = val_bpb;
            }
            println!(
                "{:>6} | {:>10.4} | {:>10.4} | {:>10.4} | {:>6}ms",
                step, train_loss, val_bpb, best_bpb, ms
            );
        }
    }

    let total = t0.elapsed();
    println!();
    println!("=== Training Complete ===");
    println!(
        "Time: {:.1}s | Init BPB: {:.4} | Best BPB: {:.4} | Delta: {:.4}",
        total.as_secs_f64(),
        init_bpb,
        best_bpb,
        init_bpb - best_bpb
    );

    let _ = fs::create_dir_all(".trinity/results");
    let result_json = serde_json::json!({
        "experiment": "cpu-backprop-deep",
        "model": "embed+bigram+smear+blocks+lm_head",
        "seed": seed,
        "vocab_size": vocab,
        "dim": dim,
        "seq_len": seq,
        "steps": steps,
        "lr": lr,
        "num_blocks": num_blocks,
        "initial_bpb": init_bpb,
        "final_bpb": best_bpb,
        "delta_bpb": init_bpb - best_bpb,
        "duration_seconds": total.as_secs_f64(),
    });

    let rpath = format!(
        ".trinity/results/cpu_train_deep_{}_seed{}.json",
        format_suffix, seed
    );
    fs::File::create(&rpath)
        .unwrap()
        .write_all(
            serde_json::to_string_pretty(&result_json)
                .unwrap()
                .as_bytes(),
        )
        .unwrap();
    println!("Results: {}", rpath);
}

fn arg_or(name: &str, default: &str) -> String {
    let prefix = format!("--{}=", name);
    std::env::args()
        .find(|a| a.starts_with(&prefix))
        .map(|a| a[prefix.len()..].to_string())
        .unwrap_or_else(|| default.to_string())
}
