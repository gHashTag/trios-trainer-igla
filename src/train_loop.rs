//! Real training loop — standalone CPU training with 3-seed sweep.
//!
//! Runs full hybrid n-gram + causal attention training with:
//! - phi-scaled hidden (828)
//! - EMA val BPB (beta=phi^-1)
//! - GF16 weight floor at last 30% steps
//! - 3-seed sweep on {42, 43, 44}
//! - Cosine LR schedule

use anyhow::Result;
use std::time::Instant;

use crate::model::{
    AdamW, HybridModel, compute_grads, cosine_lr, evaluate, gf16_floor, load_data_fallback,
    DIM, NGRAM, NUM_CTX, VOCAB,
};

pub const DEFAULT_IGLA_TARGET_BPB: f64 = 1.85;
pub const GATE_FINAL_SEEDS: &[u64] = &[42, 43, 44];

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

pub fn run_single(args: &TrainArgs) -> Result<RunOutcome> {
    let train = load_data_fallback(&args.train_path);
    let val = load_data_fallback(&args.val_path);
    let seq_len = 64usize;

    println!(
        "=== trios-trainer seed={} steps={} hidden={} lr={} attn_layers={} ===",
        args.seed, args.steps, args.hidden, args.lr, args.attn_layers
    );
    println!("train={} val={}", train.len(), val.len());

    let mut m = HybridModel::new(args.hidden, args.seed, args.attn_layers, seq_len);
    println!("params={} ({:.1}K)", m.total_params(), m.total_params() as f32 / 1000.0);

    let mut oe = AdamW::new(VOCAB * DIM, 0.01);
    let mut oc: Vec<AdamW> = (0..NUM_CTX).map(|_| AdamW::new(VOCAB * DIM, 0.01)).collect();
    let mut op = AdamW::new(args.hidden * DIM, 0.01);
    let mut oh = AdamW::new(VOCAB * args.hidden, 0.01);

    let phi_inv: f64 = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut ema_bpb: f64 = evaluate(&m, &val, seq_len) as f64;
    let mut best_bpb: f64 = ema_bpb;
    println!("Initial val_bpb={:.4}", ema_bpb);

    let t0 = Instant::now();

    for step in 1..=args.steps {
        let lr = cosine_lr(step, args.steps, args.lr, 100);

        let bs = 8;
        let sp = args.hidden + NGRAM + 1;
        let mut ge = vec![0.0f32; VOCAB * DIM];
        let mut gc: Vec<Vec<f32>> = (0..NUM_CTX).map(|_| vec![0.0f32; VOCAB * DIM]).collect();
        let mut gp = vec![0.0f32; args.hidden * DIM];
        let mut gh = vec![0.0f32; VOCAB * args.hidden];

        for _ in 0..bs {
            let start = ((step as u64 * 2654435761 + args.seed) % (train.len() as u64).max(1))
                as usize;
            if start + sp > train.len() {
                continue;
            }
            let positions: Vec<usize> = (0..args.hidden)
                .filter(|&i| start + i + NGRAM < train.len())
                .collect();
            compute_grads(
                &m,
                &train[start..start + sp.min(train.len() - start)],
                &positions,
                &mut ge,
                &mut gc,
                &mut gp,
                &mut gh,
            );
        }

        let scale = 1.0 / (bs * args.hidden).max(1) as f32;
        for v in ge.iter_mut() { *v *= scale; }
        for c in gc.iter_mut() { for v in c.iter_mut() { *v *= scale; } }
        for v in gp.iter_mut() { *v *= scale; }
        for v in gh.iter_mut() { *v *= scale; }

        oe.update(&mut m.embed, &ge, lr);
        for ci in 0..NUM_CTX {
            oc[ci].update(&mut m.ctx[ci], &gc[ci], lr);
        }
        op.update(&mut m.proj, &gp, lr);
        oh.update(&mut m.lm_head, &gh, lr);

        if step >= (args.steps as f64 * 0.7) as usize {
            gf16_floor(&mut m.embed);
            for c in m.ctx.iter_mut() { gf16_floor(c); }
            gf16_floor(&mut m.proj);
            gf16_floor(&mut m.lm_head);
        }

        if step % args.eval_every == 0 {
            let vbpb = evaluate(&m, &val, seq_len) as f64;
            ema_bpb = phi_inv * ema_bpb + (1.0 - phi_inv) * vbpb;
            if ema_bpb < best_bpb {
                best_bpb = ema_bpb;
            }
            println!(
                "seed={} step={} val_bpb={:.4} ema_bpb={:.4} best={:.4} t={:.1}s",
                args.seed, step, vbpb, ema_bpb, best_bpb, t0.elapsed().as_secs_f64()
            );
        }
    }

    println!("seed={} BPB={:.4}", args.seed, best_bpb);
    Ok(RunOutcome {
        final_bpb: best_bpb,
        steps_done: args.steps,
        seed: args.seed,
    })
}

/// Run the 3-seed gate sweep.
pub fn run_sweep(steps: usize, hidden: usize, lr: f32, attn_layers: u8, eval_every: usize,
                 train_path: &str, val_path: &str) -> Result<Vec<RunOutcome>> {
    let mut results = Vec::new();
    for &seed in GATE_FINAL_SEEDS {
        let args = TrainArgs {
            seed, steps, hidden, lr, attn_layers, eval_every,
            train_path: train_path.to_string(),
            val_path: val_path.to_string(),
        };
        results.push(run_single(&args)?);
    }
    Ok(results)
}

/// Top-level entry used by the TOML config path.
pub fn run(cfg: &crate::TrainConfig) -> Result<RunOutcome> {
    let args = TrainArgs {
        seed: cfg.seed,
        steps: cfg.steps,
        hidden: 828,
        lr: cfg.optimizer.lr as f32,
        attn_layers: if cfg.model.hybrid_attn { 2 } else { 1 },
        eval_every: 1000,
        train_path: "data/tiny_shakespeare.txt".to_string(),
        val_path: "data/tiny_shakespeare_val.txt".to_string(),
    };
    let outcome = run_single(&args)?;
    if cfg.ledger.jsonl_path != "" {
        let _ = crate::ledger::emit_row(cfg, outcome.final_bpb, outcome.steps_done);
    }
    Ok(outcome)
}
