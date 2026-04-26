#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::fs;
use std::time::Instant;

const VOCAB: usize = 128;
const DIM: usize = 64;
const SEQ: usize = 32;
const LN_2: f32 = std::f32::consts::LN_2;
const EVAL_INTERVAL: usize = 500;
const EVAL_SEQS: usize = 20;

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn softmax(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in v.iter_mut() {
        *x /= sum;
    }
}

struct LstmTrain {
    h: usize,
    d: usize,
    g4: usize,
    embed: Vec<f32>,
    w: Vec<f32>,
    b: Vec<f32>,
    head: Vec<f32>,
    h_state: Vec<f32>,
    c_state: Vec<f32>,
    s_h: Vec<Vec<f32>>,
    s_c_prev: Vec<Vec<f32>>,
    s_gates: Vec<Vec<f32>>,
    s_tc: Vec<Vec<f32>>,
    s_xh: Vec<Vec<f32>>,
    s_logits: Vec<Vec<f32>>,
    s_tok: Vec<usize>,
}

struct Grads {
    g_embed: Vec<f32>,
    g_w: Vec<f32>,
    g_b: Vec<f32>,
    g_head: Vec<f32>,
}

struct AdamW {
    m: Vec<f32>,
    v: Vec<f32>,
    step: usize,
}

impl AdamW {
    fn new(size: usize) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            step: 0,
        }
    }
    fn update(&mut self, p: &mut [f32], g: &[f32], lr: f32, wd: f32) {
        self.step += 1;
        let bc1 = 1.0 - 0.9f32.powi(self.step as i32);
        let bc2 = 1.0 - 0.999f32.powi(self.step as i32);
        for i in 0..p.len() {
            p[i] -= wd * lr * p[i];
            self.m[i] = 0.9 * self.m[i] + 0.1 * g[i];
            self.v[i] = 0.999 * self.v[i] + 0.001 * g[i] * g[i];
            p[i] -= lr * (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + 1e-8);
        }
    }
}

impl LstmTrain {
    fn new(hidden: usize, seed: u64) -> Self {
        let mut s = seed;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let d = DIM + hidden;
        let g4 = 4 * hidden;
        let lim_e = (6.0 / (VOCAB + DIM) as f32).sqrt();
        let lim_w = (6.0 / (d + g4) as f32).sqrt();
        let lim_o = (6.0 / (hidden + VOCAB) as f32).sqrt();
        let mut b = vec![0.0f32; g4];
        for i in 0..hidden {
            b[hidden + i] = 2.0;
        }
        let cap = SEQ + 1;
        Self {
            h: hidden,
            d,
            g4,
            embed: (0..VOCAB * DIM).map(|_| rng() * lim_e).collect(),
            w: (0..g4 * d).map(|_| rng() * lim_w).collect(),
            b,
            head: (0..VOCAB * hidden).map(|_| rng() * lim_o).collect(),
            h_state: vec![0.0; hidden],
            c_state: vec![0.0; hidden],
            s_h: (0..cap).map(|_| vec![0.0; hidden]).collect(),
            s_c_prev: (0..cap).map(|_| vec![0.0; hidden]).collect(),
            s_gates: (0..cap).map(|_| vec![0.0; g4]).collect(),
            s_tc: (0..cap).map(|_| vec![0.0; hidden]).collect(),
            s_xh: (0..cap).map(|_| vec![0.0; d]).collect(),
            s_logits: (0..cap).map(|_| vec![0.0; VOCAB]).collect(),
            s_tok: vec![0; cap],
        }
    }

    fn forward_seq(&mut self, tokens: &[usize]) -> f32 {
        let h = self.h;
        let d = self.d;
        let g4 = self.g4;
        for v in 0..h {
            self.h_state[v] = 0.0;
            self.c_state[v] = 0.0;
        }

        let mut total_loss = 0.0f32;
        for t in 0..tokens.len() {
            let tok = tokens[t].min(VOCAB - 1);
            let ex = &self.embed[tok * DIM..(tok + 1) * DIM];
            let mut xh = vec![0.0f32; d];
            for j in 0..DIM {
                xh[j] = ex[j];
            }
            for j in 0..h {
                xh[DIM + j] = self.h_state[j];
            }

            let mut gates = vec![0.0f32; g4];
            for i in 0..g4 {
                let mut val = self.b[i];
                for j in 0..d {
                    val += self.w[i * d + j] * xh[j];
                }
                let pre = val;
                if i < h {
                    gates[i] = sigmoid(pre);
                } else if i < 2 * h {
                    gates[i] = sigmoid(pre);
                } else if i < 3 * h {
                    gates[i] = sigmoid(pre);
                } else {
                    gates[i] = pre.tanh();
                }
            }

            let c_prev = self.c_state.clone();
            let mut tc = vec![0.0f32; h];
            for i in 0..h {
                self.c_state[i] = gates[h + i] * c_prev[i] + gates[i] * gates[3 * h + i];
                tc[i] = self.c_state[i].tanh();
                self.h_state[i] = gates[2 * h + i] * tc[i];
            }

            let mut logits = vec![0.0f32; VOCAB];
            for v in 0..VOCAB {
                for j in 0..h {
                    logits[v] += self.head[v * h + j] * self.h_state[j];
                }
            }

            self.s_h[t] = self.h_state.clone();
            self.s_c_prev[t] = c_prev;
            self.s_gates[t] = gates;
            self.s_tc[t] = tc;
            self.s_xh[t] = xh;
            self.s_logits[t] = logits.clone();
            self.s_tok[t] = tok;

            if t + 1 < tokens.len() {
                let target = tokens[t + 1].min(VOCAB - 1);
                let mut probs = logits;
                softmax(&mut probs);
                total_loss -= probs[target].max(1e-10).ln();
            }
        }
        total_loss
    }

    fn backward_seq(&self, tokens: &[usize]) -> Grads {
        let h = self.h;
        let d = self.d;
        let g4 = self.g4;
        let seq_len = tokens.len();
        let n = (seq_len - 1).max(1) as f32;

        let mut g_embed = vec![0.0f32; VOCAB * DIM];
        let mut g_w = vec![0.0f32; g4 * d];
        let mut g_b = vec![0.0f32; g4];
        let mut g_head = vec![0.0f32; VOCAB * h];

        let mut dh = vec![0.0f32; h];
        let mut dc = vec![0.0f32; h];

        for t in (0..seq_len - 1).rev() {
            let target = tokens[t + 1].min(VOCAB - 1);
            let mut probs = self.s_logits[t].clone();
            softmax(&mut probs);
            let mut d_logits = probs;
            d_logits[target] -= 1.0;

            for v in 0..VOCAB {
                let dl = d_logits[v];
                for j in 0..h {
                    g_head[v * h + j] += dl * self.s_h[t][j];
                    dh[j] += dl * self.head[v * h + j];
                }
            }

            for i in 0..h {
                let o = self.s_gates[t][2 * h + i];
                let tc = self.s_tc[t][i];
                dc[i] += dh[i] * o * (1.0 - tc * tc);
            }

            let mut dg = vec![0.0f32; g4];
            for i in 0..h {
                let ig = self.s_gates[t][i];
                let fg = self.s_gates[t][h + i];
                let og = self.s_gates[t][2 * h + i];
                let gg = self.s_gates[t][3 * h + i];
                dg[i] = dc[i] * gg * ig * (1.0 - ig);
                dg[h + i] = dc[i] * self.s_c_prev[t][i] * fg * (1.0 - fg);
                dg[2 * h + i] = dh[i] * self.s_tc[t][i] * og * (1.0 - og);
                dg[3 * h + i] = dc[i] * ig * (1.0 - gg * gg);
                dc[i] *= fg;
            }

            let mut dxh = vec![0.0f32; d];
            for i in 0..g4 {
                let dgi = dg[i];
                if dgi.abs() < 1e-12 {
                    continue;
                }
                for j in 0..d {
                    g_w[i * d + j] += dgi * self.s_xh[t][j];
                    dxh[j] += dgi * self.w[i * d + j];
                }
                g_b[i] += dgi;
            }

            for j in 0..h {
                dh[j] = dxh[DIM + j];
            }
            let tok = self.s_tok[t];
            for j in 0..DIM {
                g_embed[tok * DIM + j] += dxh[j];
            }
        }

        let inv = 1.0 / n;
        for x in g_embed.iter_mut() {
            *x *= inv;
        }
        for x in g_w.iter_mut() {
            *x *= inv;
        }
        for x in g_b.iter_mut() {
            *x *= inv;
        }
        for x in g_head.iter_mut() {
            *x *= inv;
        }

        let max_norm = 1.0f32;
        let mut gn2 = 0.0f32;
        for x in g_w.iter() {
            gn2 += x * x;
        }
        for x in g_embed.iter() {
            gn2 += x * x;
        }
        let gn = gn2.sqrt();
        if gn > max_norm {
            let scale = max_norm / gn;
            for x in g_w.iter_mut() {
                *x *= scale;
            }
            for x in g_embed.iter_mut() {
                *x *= scale;
            }
            for x in g_b.iter_mut() {
                *x *= scale;
            }
            for x in g_head.iter_mut() {
                *x *= scale;
            }
        }

        Grads {
            g_embed,
            g_w,
            g_b,
            g_head,
        }
    }

    fn eval_bpb(&mut self, tokens: &[usize]) -> f32 {
        let dl = tokens.len();
        if dl < 2 {
            return f32::MAX;
        }
        let num_possible = dl.saturating_sub(SEQ + 1);
        if num_possible == 0 {
            return f32::MAX;
        }
        let nc = EVAL_SEQS.min(num_possible);
        let stride = num_possible / nc;
        if stride == 0 {
            return f32::MAX;
        }
        let h = self.h;
        let d = self.d;
        let g4 = self.g4;

        let mut total = 0.0f32;
        let mut n = 0usize;
        for c in 0..nc {
            let start = c * stride;
            let end = (start + SEQ + 1).min(dl);
            if end <= start + 2 {
                continue;
            }
            let chunk = &tokens[start..end];
            let mut h_st = vec![0.0f32; h];
            let mut c_st = vec![0.0f32; h];
            let mut loss = 0.0f32;
            for t in 0..chunk.len() - 1 {
                let tok = chunk[t].min(VOCAB - 1);
                let ex = &self.embed[tok * DIM..(tok + 1) * DIM];
                let mut xh = vec![0.0f32; d];
                for j in 0..DIM {
                    xh[j] = ex[j];
                }
                for j in 0..h {
                    xh[DIM + j] = h_st[j];
                }
                let mut gates = vec![0.0f32; g4];
                for i in 0..g4 {
                    let mut val = self.b[i];
                    for j in 0..d {
                        val += self.w[i * d + j] * xh[j];
                    }
                    if i < 3 * h {
                        gates[i] = sigmoid(val);
                    } else {
                        gates[i] = val.tanh();
                    }
                }
                let c_old = c_st.clone();
                for i in 0..h {
                    c_st[i] = gates[h + i] * c_old[i] + gates[i] * gates[3 * h + i];
                    h_st[i] = gates[2 * h + i] * c_st[i].tanh();
                }
                let mut logits = vec![0.0f32; VOCAB];
                for v in 0..VOCAB {
                    for j in 0..h {
                        logits[v] += self.head[v * h + j] * h_st[j];
                    }
                }
                softmax(&mut logits);
                let target = chunk[t + 1].min(VOCAB - 1);
                loss -= logits[target].max(1e-10).ln();
            }
            total += loss / (chunk.len() - 1) as f32 / LN_2;
            n += 1;
        }
        if n == 0 {
            f32::MAX
        } else {
            total / n as f32
        }
    }
}

fn cosine_lr(step: usize, max_steps: usize, base_lr: f32, warmup: usize) -> f32 {
    if step < warmup {
        return base_lr * step as f32 / warmup.max(1) as f32;
    }
    let p = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
    1e-5 + (base_lr - 1e-5) * 0.5 * (1.0 + (std::f32::consts::PI * p).cos())
}

fn load_data(path: &str) -> Vec<usize> {
    let raw = fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to load {}: {}. Using fallback.", path, e);
        b"The quick brown fox jumps over the lazy dog. "
            .repeat(100)
            .to_vec()
    });
    assert!(!raw.is_empty(), "loaded data is empty");
    raw.into_iter().map(|b| (b as usize) % VOCAB).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args
        .iter()
        .find(|a| a.starts_with("--seed="))
        .and_then(|a| a[7..].parse().ok())
        .unwrap_or(43);
    let steps: usize = args
        .iter()
        .find(|a| a.starts_with("--steps="))
        .and_then(|a| a[8..].parse().ok())
        .unwrap_or(10000);
    let lr: f32 = args
        .iter()
        .find(|a| a.starts_with("--lr="))
        .and_then(|a| a[5..].parse().ok())
        .unwrap_or(0.003);
    let hidden: usize = args
        .iter()
        .find(|a| a.starts_with("--hidden="))
        .and_then(|a| a[9..].parse().ok())
        .unwrap_or(256);

    let train_data = load_data("data/tiny_shakespeare.txt");
    let val_data = load_data("data/tiny_shakespeare_val.txt");
    let train_end = (train_data.len() as f64 * 0.9) as usize;
    let train = &train_data[..train_end];
    let val = if val_data.len() > 100 {
        &val_data
    } else {
        &train_data[train_end..]
    };

    let mut m = LstmTrain::new(hidden, seed);
    let d = DIM + hidden;
    let g4 = 4 * hidden;
    let wd = 0.0f32;

    let mut opt_e = AdamW::new(VOCAB * DIM);
    let mut opt_w = AdamW::new(g4 * d);
    let mut opt_b = AdamW::new(g4);
    let mut opt_h = AdamW::new(VOCAB * hidden);

    let mut best_bpb = f32::MAX;
    let warmup = steps / 10;
    let start = Instant::now();
    let dl = train.len();
    let seq = SEQ;

    eprintln!(
        "lstm_v2: hidden={} seq={} lr={} seed={} steps={} params≈{}",
        hidden,
        seq,
        lr,
        seed,
        steps,
        VOCAB * DIM + g4 * d + g4 + VOCAB * hidden
    );

    for step in 1..=steps {
        let off = (step * 97 + seed as usize) % dl.saturating_sub(seq + 1);
        let chunk = &train[off..off + seq + 1];
        let _loss = m.forward_seq(chunk);
        let grads = m.backward_seq(chunk);
        let cur_lr = cosine_lr(step, steps, lr, warmup);
        opt_e.update(&mut m.embed, &grads.g_embed, cur_lr, wd);
        opt_w.update(&mut m.w, &grads.g_w, cur_lr, wd);
        opt_b.update(&mut m.b, &grads.g_b, cur_lr, 0.0);
        opt_h.update(&mut m.head, &grads.g_head, cur_lr, wd);

        if step % EVAL_INTERVAL == 0 || step == steps {
            let elapsed = start.elapsed().as_secs_f64();
            let val_bpb = m.eval_bpb(val);
            if val_bpb < best_bpb && val_bpb.is_finite() {
                best_bpb = val_bpb;
            }
            eprintln!(
                "step={:5} val_bpb={:.4} best={:.4} t={}s",
                step, val_bpb, best_bpb, elapsed as u64
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!("done: best_bpb={:.4} time={:.1}s", best_bpb, elapsed);
    println!("BPB={:.4}", best_bpb);
    Ok(())
}
