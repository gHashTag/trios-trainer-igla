//! End-to-end pipeline smoke test (Loop 57).
//!
//! Exercises the three-stratum chain that backs the headline empirical
//! claim in §5 of `papers/f2_methodology.md`:
//!
//! ```text
//! synth canonical CSV ─► f2_dual_mediation ─► canonical_dual.csv
//! synth wd0 CSV       ─► f2_dual_mediation ─► wd0_dual.csv
//! synth warmup0 CSV   ─► f2_dual_mediation ─► warmup0_dual.csv
//!                                              │
//!                                              ▼
//!                                   f2_stratum_compare
//!                                              │
//!                                              ▼
//!                                   3stratum.csv with stable_across_strata flags
//! ```
//!
//! The existing per-stratum e2e tests
//! (`f2_warmup_stratified_e2e`, `f2_wd_stratified_e2e`) each cover one
//! stratum-string-prefix → `f2_dual_mediation` hop. This test stitches
//! all three together plus the `f2_stratum_compare` cross-stratum hop,
//! catching schema drift between the per-binary contracts that
//! per-stratum tests miss.
//!
//! Synthesizes CSVs inline (no trainer invocation); the trainer's own
//! invariants are covered by the lib tests.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use trios_trainer::race::ablation::{mode_string, ModeKind, Stratum, CANONICAL_FIX_NAMES};

fn dual_mediation_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_f2_dual_mediation"))
}

fn stratum_compare_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_f2_stratum_compare"))
}

fn noise(seed: u64, kind: u8) -> f64 {
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

fn write_synth_csv(path: &Path, stratum: Stratum) {
    let mut f = File::create(path).unwrap();
    writeln!(
        f,
        "mode,fix_name,fix_index,cumulative_n,seed,bpb,config_hash,wall_s"
    )
    .unwrap();
    let pairwise = mode_string(ModeKind::Pairwise, stratum);
    let loco = mode_string(ModeKind::Loco, stratum);
    let triplet = mode_string(ModeKind::Triplet, stratum);
    for &sid in &[1u64, 2, 3, 4, 5] {
        writeln!(
            f,
            "{},full_stack,-1,,{},{:.6},0xdead,0.1",
            pairwise,
            sid,
            4.0 + noise(sid, 0)
        )
        .unwrap();
        for (i, name) in CANONICAL_FIX_NAMES.iter().enumerate() {
            let v = if *name == "rms" { 5.0 } else { 4.5 } + noise(sid, 1 + i as u8);
            writeln!(f, "{},{},0,,{},{:.6},0xdead,0.1", loco, name, sid, v).unwrap();
        }
        let pairs = [
            "pair_rms_warmup",
            "pair_rms_gradclip",
            "pair_rms_clamp",
            "pair_rms_smooth",
            "pair_rms_wd",
            "pair_rms_dropout",
            "pair_warmup_gradclip",
            "pair_warmup_clamp",
            "pair_warmup_smooth",
            "pair_warmup_wd",
            "pair_warmup_dropout",
            "pair_gradclip_clamp",
            "pair_gradclip_smooth",
            "pair_gradclip_wd",
            "pair_gradclip_dropout",
            "pair_clamp_smooth",
            "pair_clamp_wd",
            "pair_clamp_dropout",
            "pair_smooth_wd",
            "pair_smooth_dropout",
            "pair_wd_dropout",
        ];
        for (i, lbl) in pairs.iter().enumerate() {
            let v = if lbl.contains("_wd") { 0.5 } else { 4.0 } + noise(sid, 100 + i as u8);
            writeln!(f, "{},{},0,,{},{:.6},0xdead,0.1", pairwise, lbl, sid, v).unwrap();
        }
        let triplets = [
            "triplet_rms_warmup_wd",
            "triplet_rms_warmup_dropout",
            "triplet_warmup_gradclip_wd",
            "triplet_warmup_clamp_wd",
            "triplet_warmup_smooth_wd",
            "triplet_warmup_wd_dropout",
        ];
        for (i, lbl) in triplets.iter().enumerate() {
            let v = if lbl.contains("_wd") { 0.4 } else { 4.0 } + noise(sid, 200 + i as u8);
            writeln!(f, "{},{},0,,{},{:.6},0xdead,0.1", triplet, lbl, sid, v).unwrap();
        }
    }
}

fn run_dual_mediation(input: &Path, out: &Path) {
    let result = Command::new(dual_mediation_path())
        .args([
            "--m1",
            "wd",
            "--m2",
            "warmup",
            input.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("spawn f2_dual_mediation");
    assert!(
        result.status.success(),
        "f2_dual_mediation failed on {}: stderr={}",
        input.display(),
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn three_stratum_pipeline_produces_stability_flags() {
    let tmp = std::env::temp_dir().join("f2_three_stratum_pipeline_e2e");
    std::fs::create_dir_all(&tmp).unwrap();

    let canonical_sweep = tmp.join("canonical_sweep.csv");
    let wd0_sweep = tmp.join("wd0_sweep.csv");
    let warmup0_sweep = tmp.join("warmup0_sweep.csv");
    write_synth_csv(&canonical_sweep, Stratum::Canonical);
    write_synth_csv(&wd0_sweep, Stratum::Wd0);
    write_synth_csv(&warmup0_sweep, Stratum::Warmup0);

    let canonical_dual = tmp.join("canonical_dual.csv");
    let wd0_dual = tmp.join("wd0_dual.csv");
    let warmup0_dual = tmp.join("warmup0_dual.csv");
    run_dual_mediation(&canonical_sweep, &canonical_dual);
    run_dual_mediation(&wd0_sweep, &wd0_dual);
    run_dual_mediation(&warmup0_sweep, &warmup0_dual);

    let three_stratum = tmp.join("three_stratum.csv");
    let compare = Command::new(stratum_compare_path())
        .args([
            "--canonical",
            canonical_dual.to_str().unwrap(),
            "--wd0",
            wd0_dual.to_str().unwrap(),
            "--warmup0",
            warmup0_dual.to_str().unwrap(),
            "--out",
            three_stratum.to_str().unwrap(),
        ])
        .output()
        .expect("spawn f2_stratum_compare");
    assert!(
        compare.status.success(),
        "f2_stratum_compare failed: stderr={}",
        String::from_utf8_lossy(&compare.stderr)
    );

    let body = std::fs::read_to_string(&three_stratum).expect("read three_stratum.csv");
    let data_rows: Vec<&str> = body
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("fix_x,"))
        .collect();

    assert!(
        !data_rows.is_empty(),
        "f2_stratum_compare emitted no data rows; body=\n{}",
        body
    );

    let stable_col_idx = body
        .lines()
        .find(|l| l.starts_with("fix_x,"))
        .expect("header row missing")
        .split(',')
        .position(|c| c == "stable_across_strata")
        .expect("stable_across_strata column missing");

    let mut saw_true = false;
    let mut saw_false = false;
    for row in &data_rows {
        let parts: Vec<&str> = row.split(',').collect();
        match parts.get(stable_col_idx).map(|s| s.trim()) {
            Some("true") => saw_true = true,
            Some("false") => saw_false = true,
            Some(other) => panic!("unexpected stable_across_strata value: {:?}", other),
            None => panic!("missing stable column in row: {}", row),
        }
    }
    assert!(
        saw_true || saw_false,
        "no parseable stable_across_strata values found"
    );

    assert!(
        body.starts_with("# f2_stratum_compare")
            || body
                .lines()
                .any(|l| l.starts_with("# `stable_across_strata`")),
        "expected stratum-compare header comment; body starts with: {}",
        body.lines().take(3).collect::<Vec<_>>().join(" | ")
    );
}
