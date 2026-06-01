//! End-to-end test: synth warmup_stratified CSV → f2_dual_mediation → ≥1 PSE row.
//! Loop 42 fix 1+6.
//!
//! Mirrors `tests/f2_wd_stratified_e2e.rs` but for `Stratum::Warmup0`. Validates
//! that:
//!   * the `warmup0_*` mode prefix produced by `f2_ablation_sweep --mode
//!     warmup_stratified` is parsed by f2_dual_mediation (Loop 38 stratum
//!     registry extended in Loop 41 to include Warmup0)
//!   * the resulting CSV has finite SE/CI columns (variance path exercised)
//!
//! Synthesizes the CSV inline; the full trainer-driven sweep is exercised
//! separately by `f2_ablation_sweep` itself.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use trios_trainer::race::ablation::{mode_string, ModeKind, Stratum, CANONICAL_FIX_NAMES};

fn dual_mediation_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_f2_dual_mediation"))
}

fn noise(seed: u64, kind: u8) -> f64 {
    // FNV-1a on (seed, kind) — same recipe as wd_stratified test; needed so per-row
    // jitter doesn't cancel in the PSE linear combos.
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    let mut h = FNV_OFFSET;
    for byte in seed.to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h ^= kind as u64;
    h = h.wrapping_mul(FNV_PRIME);
    let bits = (h >> 17) & ((1u64 << 32) - 1);
    let u = bits as f64 / ((1u64 << 32) as f64);
    (u - 0.5) * 1.0
}

fn write_csv(path: &std::path::Path) {
    let mut f = File::create(path).unwrap();
    writeln!(f, "mode,fix_name,fix_index,cumulative_n,seed,bpb,config_hash,wall_s").unwrap();
    let pairwise = mode_string(ModeKind::Pairwise, Stratum::Warmup0);
    let loco = mode_string(ModeKind::Loco, Stratum::Warmup0);
    let triplet = mode_string(ModeKind::Triplet, Stratum::Warmup0);
    for &sid in &[1u64, 2, 3, 4, 5] {
        writeln!(f, "{},full_stack,-1,,{},{:.6},0xdead,0.1", pairwise, sid, 4.0 + noise(sid, 0)).unwrap();
        for (i, name) in CANONICAL_FIX_NAMES.iter().enumerate() {
            let v = if *name == "rms" { 5.0 } else { 4.5 } + noise(sid, 1 + i as u8);
            writeln!(f, "{},{},0,,{},{:.6},0xdead,0.1", loco, name, sid, v).unwrap();
        }
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
fn dual_mediation_runs_end_to_end_on_warmup_stratified_rows() {
    let tmp = std::env::temp_dir().join("f2_warmup_stratified_e2e.csv");
    write_csv(&tmp);
    let out = Command::new(dual_mediation_path())
        .args(["--m1", "wd", "--m2", "warmup", tmp.to_str().unwrap()])
        .output()
        .expect("spawn binary");
    assert!(
        out.status.success(),
        "f2_dual_mediation failed on warmup_stratified CSV: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let data_rows: Vec<&str> = stdout
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("rank,"))
        .collect();
    assert!(
        data_rows.iter().any(|l| l.contains(",rms,")),
        "expected rms PSE row in warmup_stratified output; stdout:\n{}",
        stdout
    );
    // Variance path: se_nde > 0.001 (Loop 41 lock).
    let rms_row = data_rows.iter().find(|l| l.contains(",rms,")).unwrap();
    let parts: Vec<&str> = rms_row.split(',').collect();
    let se_nde: f64 = parts[5].parse().expect("parse se_nde");
    assert!(
        se_nde > 0.001,
        "se_nde={} too small — variance path not exercised on warmup_stratified",
        se_nde
    );
}
