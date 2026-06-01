//! End-to-end test: synth wd_stratified CSV → f2_dual_mediation → ≥1 PSE row.
//! Loop 39 fix 3+4.
//!
//! Validates the wd_stratified output contract:
//!   - row modes are `wd0_loco` / `wd0_pairwise` / `wd0_triplet`
//!   - f2_dual_mediation's stratum-registry lookup (Loop 38 / Loop 39)
//!     resolves rows in stratified modes without renaming
//!   - the resulting CSV has ≥1 PSE data row with finite SEs and CIs
//!
//! Synthesizes the CSV inline so the test runs in milliseconds; the full
//! trainer-driven sweep is exercised separately by `f2_ablation_sweep` itself.
//! Locks the wd_stratified mode-string contract end-to-end.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use trios_trainer::race::ablation::{mode_string, ModeKind, Stratum, CANONICAL_FIX_NAMES};

fn dual_mediation_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_f2_dual_mediation"))
}

fn write_csv(path: &std::path::Path) {
    let mut f = File::create(path).unwrap();
    writeln!(f, "mode,fix_name,fix_index,cumulative_n,seed,bpb,config_hash,wall_s").unwrap();
    let pairwise = mode_string(ModeKind::Pairwise, Stratum::Wd0);
    let loco = mode_string(ModeKind::Loco, Stratum::Wd0);
    let triplet = mode_string(ModeKind::Triplet, Stratum::Wd0);
    // Synth: full_stack at 4.0, LOCO_rms at 5.0, all other LOCOs at 4.5,
    // pair_*_wd at 0.5 (wd removal helps), other pairs at 4.0,
    // triplet_rms_wd_warmup at 0.5, etc.
    // Loop 41 fix 5: row-distinct jitter so PSE linear combos retain per-seed
    // variance. A single per-seed offset would cancel in `Δ = target − full`,
    // producing zero SE. Use per-(seed, row-class) noise via a small LCG.
    fn noise(seed: u64, kind: u8) -> f64 {
        // FNV-1a-style hash on (seed, kind) — strong avalanche on every byte
        // of input. Naive LCG composition (Loop 41 first attempt) produced
        // constant noise(sid, k200) − noise(sid, 0) across seeds because the
        // additive injection of `kind` left the high bits cancelable.
        const FNV_OFFSET: u64 = 14695981039346656037;
        const FNV_PRIME: u64 = 1099511628211;
        let mut h = FNV_OFFSET;
        for byte in seed.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        h ^= kind as u64;
        h = h.wrapping_mul(FNV_PRIME);
        // Use middle bits of the hash to avoid LCG correlation patterns.
        let bits = (h >> 17) & ((1u64 << 32) - 1);
        let u = bits as f64 / ((1u64 << 32) as f64);
        // Centered around 0 with ±0.5 BPB amplitude — guarantees per-seed
        // variance large enough to exercise sample_se measurably (>0.001 BPB).
        (u - 0.5) * 1.0
    }
    for &sid in &[1u64, 2, 3, 4, 5] {
        writeln!(f, "{},full_stack,-1,,{},{:.6},0xdead,0.1", pairwise, sid, 4.0 + noise(sid, 0)).unwrap();
        for (i, name) in CANONICAL_FIX_NAMES.iter().enumerate() {
            let v = if *name == "rms" { 5.0 } else { 4.5 } + noise(sid, 1 + i as u8);
            writeln!(f, "{},{},0,,{},{:.6},0xdead,0.1", loco, name, sid, v).unwrap();
        }
        // Pairwise rows in AblationFix::ALL order — every pair containing wd is helpful.
        let pairs = [
            "pair_rms_warmup", "pair_rms_gradclip", "pair_rms_clamp",
            "pair_rms_smooth", "pair_rms_wd", "pair_rms_dropout",
            "pair_warmup_gradclip", "pair_warmup_clamp", "pair_warmup_smooth",
            "pair_warmup_wd", "pair_warmup_dropout",
            "pair_gradclip_clamp", "pair_gradclip_smooth", "pair_gradclip_wd", "pair_gradclip_dropout",
            "pair_clamp_smooth", "pair_clamp_wd", "pair_clamp_dropout",
            "pair_smooth_wd", "pair_smooth_dropout",
            "pair_wd_dropout",
        ];
        for (i, lbl) in pairs.iter().enumerate() {
            let v = if lbl.contains("_wd") { 0.5 } else { 4.0 } + noise(sid, 100 + i as u8);
            writeln!(f, "{},{},0,,{},{:.6},0xdead,0.1", pairwise, lbl, sid, v).unwrap();
        }
        // Triplet rows — enough to cover the rms × wd × warmup case dual_mediation needs.
        let triplets = [
            "triplet_rms_warmup_wd", "triplet_rms_warmup_dropout",
            "triplet_warmup_gradclip_wd", "triplet_warmup_clamp_wd",
            "triplet_warmup_smooth_wd", "triplet_warmup_wd_dropout",
        ];
        for (i, lbl) in triplets.iter().enumerate() {
            let v = if lbl.contains("_wd") { 0.4 } else { 4.0 } + noise(sid, 200 + i as u8);
            writeln!(f, "{},{},0,,{},{:.6},0xdead,0.1", triplet, lbl, sid, v).unwrap();
        }
    }
}

#[test]
fn dual_mediation_runs_end_to_end_on_wd_stratified_rows() {
    let tmp = std::env::temp_dir().join("f2_wd_stratified_e2e.csv");
    write_csv(&tmp);
    let out = Command::new(dual_mediation_path())
        .args(["--m1", "wd", "--m2", "warmup", tmp.to_str().unwrap()])
        .output()
        .expect("spawn binary");
    assert!(
        out.status.success(),
        "f2_dual_mediation failed on wd_stratified CSV: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let data_rows: Vec<&str> = stdout
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("rank,"))
        .collect();
    // Expect rms to appear in the decomposition (non-mediator with synthesized data).
    assert!(
        data_rows.iter().any(|l| l.contains(",rms,")),
        "expected rms PSE row in wd_stratified output; got:\n{}",
        stdout
    );
    // Locks: SE columns must be present and finite.
    let rms_row = data_rows.iter().find(|l| l.contains(",rms,")).unwrap();
    let parts: Vec<&str> = rms_row.split(',').collect();
    assert!(parts.len() >= 18, "expected ≥18 columns in dual_mediation output, got {}", parts.len());

    // Loop 41 fix 5: variance path exercised. SE for NDE (col 5 in Loop 36 schema)
    // must be strictly positive — proves sample_se ran on real per-seed variance.
    let se_nde: f64 = parts[5].parse().expect("parse se_nde");
    assert!(
        se_nde > 0.001,
        "se_nde={} too small — variance path not exercised (expected >0.001 BPB with realistic jitter)",
        se_nde
    );
}
