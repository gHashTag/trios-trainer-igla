//! P1 — Optimizer Lab: AdamW vs Muon vs MuonCwd
//!
//! Per TRAINING_FLOW_V2.md P1 spec:
//!   - Architecture: n-gram + 2-layer HybridAttn, hidden=828
//!   - 12K steps, seed 43 only (lab phase, NOT a Gate-2 row)
//!   - Compare: AdamW (control), Muon (NS-3), Muon+CWD
//!   - Hypothesis: Muon reduces final BPB by >= 0.05 vs AdamW

use anyhow::Result;
use trios_trainer::optimizer::{AdamWCpu, MuonOptimizer, MuonCwd};
use trios_trainer::train_loop::TrainArgs;

fn main() -> Result<()> {
    let base = TrainArgs {
        seed: 43,
        steps: 12_000,
        hidden: 828,
        lr: 0.003,
        attn_layers: 2,
        eval_every: 1000,
        train_path: "data/tiny_shakespeare.txt".to_string(),
        val_path: "data/tiny_shakespeare_val.txt".to_string(),
    };

    eprintln!("=== P1 Optimizer Lab ===");
    eprintln!("AdamW control: lr=0.003, wd=0.04, beta1=0.9, beta2=0.999");
    eprintln!("Muon: lr=0.0235, momentum=0.95, wd=0.01, NS-3");
    eprintln!("MuonCwd: same as Muon + cwd_lambda=0.01");
    eprintln!("Steps: 12K, Seed: 43, Data: tiny_shakespeare");

    // Run 1: AdamW control (use existing trios-train)
    eprintln!("\n--- Run 1/3: AdamW control ---");
    let adamw_out = run_adamw(&base)?;
    eprintln!("AdamW result: bpb={:.4}", adamw_out);

    // Run 2: Muon
    eprintln!("\n--- Run 2/3: Muon ---");
    let muon_out = run_muon(&base)?;
    eprintln!("Muon result: bpb={:.4}", muon_out);

    // Run 3: MuonCwd
    eprintln!("\n--- Run 3/3: MuonCwd ---");
    let muoncwd_out = run_muoncwd(&base)?;
    eprintln!("MuonCwd result: bpb={:.4}", muoncwd_out);

    eprintln!("\n=== P1 Leaderboard ===");
    eprintln!("AdamW:   bpb={:.4}", adamw_out);
    eprintln!("Muon:    bpb={:.4}", muon_out);
    eprintln!("MuonCwd: bpb={:.4}", muoncwd_out);

    let best = adamw_out.min(muon_out).min(muoncwd_out);
    let winner = if best == adamw_out { "AdamW" } else if best == muon_out { "Muon" } else { "MuonCwd" };
    let margin = adamw_out - best;
    eprintln!("\nWinner: {} (margin vs AdamW: {:.4})", winner, margin);

    if margin >= 0.05 {
        eprintln!("P1 DECISION: Muon beats AdamW by >= 0.05 — proceed with {}", winner);
    } else {
        eprintln!("P1 DECISION: Muon does NOT beat AdamW by >= 0.05 — proceed with AdamW");
    }

    Ok(())
}

fn run_adamw(args: &TrainArgs) -> Result<f64> {
    let outcome = trios_trainer::train_loop::run_single(args)?;
    Ok(outcome.final_bpb)
}

fn run_muon(args: &TrainArgs) -> Result<f64> {
    let outcome = trios_trainer::train_loop::run_single_muon(args, false)?;
    Ok(outcome.final_bpb)
}

fn run_muoncwd(args: &TrainArgs) -> Result<f64> {
    let outcome = trios_trainer::train_loop::run_single_muon(args, true)?;
    Ok(outcome.final_bpb)
}
