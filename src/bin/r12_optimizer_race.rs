//! R12: Muon vs AdamW — optimizer comparison using real NTP BPB
//! Epic: #110 Parameter Golf
//! TASK-5D: replaced quadratic MSE proxy with actual cross-entropy → BPB

use trios_trainer::backward::cross_entropy_loss;
use trios_trainer::optimizer::{AdamWCpu, MuonOptimizer, OptimizerKind};
use trios_trainer::pipeline::{bpb_from_loss, forward_f32_embeddings, backward_f32_embeddings};

const SEED: u64 = 43;
const STEPS: usize = 6000;
const VOCAB_SIZE: usize = 128;
const D_MODEL: usize = 384;
const CONTEXT_LEN: usize = 64;
const N_PARAMS: usize = VOCAB_SIZE * D_MODEL;

struct Config {
    name: &'static str,
    optimizer: OptimizerKind,
}

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
}

fn run_trial(cfg: &mut Config, seed: u64) -> Vec<(usize, f64)> {
    let mut rng = seed;

    let scale = (2.0_f32 / N_PARAMS as f32).sqrt();
    let mut embeddings: Vec<f32> = (0..N_PARAMS).map(|_| lcg(&mut rng) * scale).collect();

    let mut checkpoints = Vec::new();

    for step in 1..=STEPS {
        let input: Vec<f32> = (0..CONTEXT_LEN)
            .map(|i| ((i.wrapping_add(step)) % VOCAB_SIZE) as f32)
            .collect();
        let targets: Vec<usize> = input
            .iter()
            .map(|&v| ((v as usize) + 1) % VOCAB_SIZE)
            .collect();

        let logits = forward_f32_embeddings(&embeddings, &input, VOCAB_SIZE, D_MODEL);
        let loss = cross_entropy_loss(&logits, &targets);
        let bpb = bpb_from_loss(loss as f64);

        if [1000, 2000, 3000, 4000, 5000, 6000].contains(&step) {
            checkpoints.push((step, bpb));
        }

        let mut grads = backward_f32_embeddings(
            &embeddings, &logits, &input, &targets, VOCAB_SIZE, D_MODEL,
        );

        let max_norm = 1.0f32;
        let l2_sq: f32 = grads.iter().map(|g| g * g).sum();
        let l2 = l2_sq.sqrt();
        if l2 > max_norm {
            let scale = max_norm / l2;
            grads.iter_mut().for_each(|g| *g *= scale);
        }

        cfg.optimizer.step(&mut embeddings, &grads);
    }

    checkpoints
}

fn main() {
    let mut configs: Vec<Config> = vec![
        Config {
            name: "A: AdamW  lr=0.004",
            optimizer: OptimizerKind::AdamW(AdamWCpu::with_params(
                N_PARAMS, 0.004, 0.9, 0.999, 0.01,
            )),
        },
        Config {
            name: "B: Muon   lr=0.004",
            optimizer: OptimizerKind::Muon(MuonOptimizer::new(N_PARAMS, 0.004, 0.95, 0.01)),
        },
        Config {
            name: "C: Muon   lr=0.001",
            optimizer: OptimizerKind::Muon(MuonOptimizer::new(N_PARAMS, 0.001, 0.95, 0.01)),
        },
    ];

    println!("## R12 RESULT: Muon vs AdamW @ lr=0.004, seed={} (real NTP BPB)", SEED);
    println!("| Config | BPB@1k | BPB@2k | BPB@3k | BPB@4k | BPB@5k | BPB@6k |");
    println!("|--------|--------|--------|--------|--------|--------|--------|");

    let mut results = Vec::new();
    for cfg in &mut configs {
        let t0 = std::time::Instant::now();
        let pts = run_trial(cfg, SEED);
        let wall = t0.elapsed().as_secs_f64();
        let vals: Vec<String> = pts.iter().map(|(_, bpb)| format!("{:.4}", bpb)).collect();
        println!(
            "| {} | {} | wall={:.1}s |",
            cfg.name,
            vals.join(" | "),
            wall
        );
        results.push((cfg.name, pts, wall));
    }

    let winner = results
        .iter()
        .min_by(|a, b| {
            a.1.last()
                .unwrap()
                .1
                .partial_cmp(&b.1.last().unwrap().1)
                .unwrap()
        })
        .unwrap();
    println!("\nWinner: {}", winner.0);
}
