//! R12: Muon vs AdamW — optimizer comparison at lr=0.004, seed=43
//! Epic: #110 Parameter Golf

use trios_trainer::optimizer::{AdamWCpu, MuonOptimizer, OptimizerKind};

const SEED: u64 = 43;
const STEPS: usize = 6000;
const N_PARAMS: usize = 384 * 384; // d_model=384, single weight matrix

struct Config {
    name: &'static str,
    optimizer: OptimizerKind,
}

/// Minimal LCG for reproducible pseudo-random init
fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
}

/// Dummy BPB proxy: measures how well optimizer minimizes quadratic loss
/// Real BPB from NTP pipeline would replace this in TASK-5D
fn run_trial(cfg: &mut Config, seed: u64) -> Vec<(usize, f64)> {
    let mut rng = seed;

    // Xavier-init params
    let scale = (2.0_f32 / N_PARAMS as f32).sqrt();
    let mut params: Vec<f32> = (0..N_PARAMS).map(|_| lcg(&mut rng) * scale).collect();

    // Fixed target (random but reproducible)
    let mut rng2 = seed.wrapping_add(999);
    let target: Vec<f32> = (0..N_PARAMS).map(|_| lcg(&mut rng2) * scale).collect();

    let mut checkpoints = Vec::new();

    for step in 1..=STEPS {
        // Gradient: d_loss/d_params = 2*(params - target) / N (MSE)
        let grads: Vec<f32> = params.iter().zip(target.iter())
            .map(|(p, t)| 2.0 * (p - t) / N_PARAMS as f32)
            .collect();

        let mse: f64 = params.iter().zip(target.iter())
            .map(|(p, t)| ((p - t) * (p - t)) as f64)
            .sum::<f64>() / N_PARAMS as f64;
        let bpb_proxy = mse / std::f64::consts::LN_2;

        if [1000, 2000, 3000, 4000, 5000, 6000].contains(&step) {
            checkpoints.push((step, bpb_proxy));
        }

        cfg.optimizer.step(&mut params, &grads);
    }

    checkpoints
}

fn main() {
    let mut configs: Vec<Config> = vec![
        Config {
            name: "A: AdamW  lr=0.004",
            optimizer: OptimizerKind::AdamW(
                AdamWCpu::with_params(N_PARAMS, 0.004, 0.9, 0.999, 0.01)
            ),
        },
        Config {
            name: "B: Muon   lr=0.004",
            optimizer: OptimizerKind::Muon(
                MuonOptimizer::new(N_PARAMS, 0.004, 0.95, 0.01)
            ),
        },
        Config {
            name: "C: Muon   lr=0.001",
            optimizer: OptimizerKind::Muon(
                MuonOptimizer::new(N_PARAMS, 0.001, 0.95, 0.01)
            ),
        },
    ];

    println!("## R12 RESULT: Muon vs AdamW @ lr=0.004, seed={}", SEED);
    println!("| Config | BPB@1k | BPB@2k | BPB@3k | BPB@4k | BPB@5k | BPB@6k |");
    println!("|--------|--------|--------|--------|--------|--------|--------|");

    let mut results = Vec::new();
    for cfg in &mut configs {
        let t0 = std::time::Instant::now();
        let pts = run_trial(cfg, SEED);
        let wall = t0.elapsed().as_secs_f64();
        let vals: Vec<String> = pts.iter()
            .map(|(_, bpb)| format!("{:.4}", bpb))
            .collect();
        println!("| {} | {} | wall={:.1}s |",
            cfg.name, vals.join(" | "), wall);
        results.push((cfg.name, pts, wall));
    }

    // Winner = lowest BPB@6000
    let winner = results.iter()
        .min_by(|a, b| a.1.last().unwrap().1
            .partial_cmp(&b.1.last().unwrap().1).unwrap())
        .unwrap();
    println!("\nWinner: {}", winner.0);
}
