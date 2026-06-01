//! Provenance-preamble robustness for f2_dual_mediation — Loop 38 fix 6.
//!
//! `f2_ablation_sweep` (Loop 32+) emits W3C-PROV preamble lines (`# prov:*`)
//! BEFORE the CSV header. The aggregator's `parse_csv` skips these via the
//! Loop 33 fix; `f2_dual_mediation::parse_csv` was patched in Loop 38 to do the
//! same. This integration test locks the round-trip: synth CSV with preamble +
//! correct rows → dual_mediation subprocess → ≥ 1 PSE row.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_f2_dual_mediation"))
}

#[test]
fn dual_mediation_reads_csv_with_provenance_preamble() {
    let tmp = std::env::temp_dir().join("f2_dual_preamble.csv");
    let mut f = File::create(&tmp).unwrap();
    // W3C-PROV preamble (matches what f2_ablation_sweep emits).
    writeln!(f, "# f2_ablation_sweep provenance (W3C PROV / RO-Crate)").unwrap();
    writeln!(f, "# prov:generatedAt = 1700000000 (unix seconds UTC)").unwrap();
    writeln!(
        f,
        "# prov:wasGeneratedBy = f2_ablation_sweep --mode all --steps 200"
    )
    .unwrap();
    writeln!(f, "# prov:agent_git_sha = e475665").unwrap();
    writeln!(f, "# prov:host = test_host").unwrap();
    writeln!(
        f,
        "# prov:trainer_internals_schema = {}",
        trios_trainer::race::multi_seed::TRAINER_INTERNALS_SCHEMA
    )
    .unwrap();
    writeln!(f, "# prov:cargo_pkg_version = 0.1.0").unwrap();
    // CSV header + minimal rows for rms as X under M1=wd, M2=warmup.
    writeln!(
        f,
        "mode,fix_name,fix_index,cumulative_n,seed,bpb,config_hash,wall_s"
    )
    .unwrap();
    writeln!(f, "pairwise,full_stack,-1,,42,4.0,0xdead,0.1").unwrap();
    writeln!(f, "loco,rms,0,,42,5.0,0xdead,0.1").unwrap();
    writeln!(f, "pairwise,pair_rms_wd,0,,42,4.5,0xdead,0.1").unwrap();
    writeln!(f, "pairwise,pair_rms_warmup,0,,42,4.7,0xdead,0.1").unwrap();
    writeln!(f, "triplet,triplet_rms_warmup_wd,0,,42,4.3,0xdead,0.1").unwrap();
    drop(f);

    let out = Command::new(binary_path())
        .args(["--m1", "wd", "--m2", "warmup", tmp.to_str().unwrap()])
        .output()
        .expect("spawn binary");
    assert!(
        out.status.success(),
        "f2_dual_mediation failed on preamble CSV: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    // Expect exactly 1 data row (rms is the only non-mediator with data).
    let data_rows: Vec<&str> = stdout
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("rank,"))
        .collect();
    assert_eq!(
        data_rows.len(),
        1,
        "expected 1 PSE row when reading preamble CSV; got {}: stdout={}",
        data_rows.len(),
        stdout
    );
    // First column should be rank=1 then "rms".
    let parts: Vec<&str> = data_rows[0].split(',').collect();
    assert_eq!(parts[1], "rms");
}

/// Loop 44 fix 4: provenance preamble round-trip with stratified modes.
/// Locks the contract that `f2_dual_mediation` ingests CSV produced by
/// `f2_ablation_sweep --mode wd_stratified` (preamble + wd0_* rows).
#[test]
fn dual_mediation_reads_csv_with_preamble_and_wd0_stratum() {
    let tmp = std::env::temp_dir().join("f2_dual_preamble_wd0.csv");
    let mut f = File::create(&tmp).unwrap();
    writeln!(f, "# f2_ablation_sweep provenance").unwrap();
    writeln!(f, "# prov:generatedAt = 1700000000").unwrap();
    writeln!(
        f,
        "# prov:wasGeneratedBy = f2_ablation_sweep --mode wd_stratified --steps 200"
    )
    .unwrap();
    writeln!(f, "# prov:agent_git_sha = abc123").unwrap();
    writeln!(f, "# prov:host = test_host").unwrap();
    writeln!(
        f,
        "# prov:trainer_internals_schema = {}",
        trios_trainer::race::multi_seed::TRAINER_INTERNALS_SCHEMA
    )
    .unwrap();
    writeln!(f, "# prov:cargo_pkg_version = 0.1.0").unwrap();
    writeln!(
        f,
        "mode,fix_name,fix_index,cumulative_n,seed,bpb,config_hash,wall_s"
    )
    .unwrap();
    // wd0_* stratified modes (Loop 38+ stratum-registry lookup).
    writeln!(f, "wd0_pairwise,full_stack,-1,,42,4.0,0xdead,0.1").unwrap();
    writeln!(f, "wd0_loco,rms,0,,42,5.0,0xdead,0.1").unwrap();
    writeln!(f, "wd0_pairwise,pair_rms_wd,0,,42,4.5,0xdead,0.1").unwrap();
    writeln!(f, "wd0_pairwise,pair_rms_warmup,0,,42,4.7,0xdead,0.1").unwrap();
    writeln!(f, "wd0_triplet,triplet_rms_warmup_wd,0,,42,4.3,0xdead,0.1").unwrap();
    drop(f);

    let out = Command::new(binary_path())
        .args(["--m1", "wd", "--m2", "warmup", tmp.to_str().unwrap()])
        .output()
        .expect("spawn binary");
    assert!(
        out.status.success(),
        "dual_mediation failed on stratified preamble CSV: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let data_rows: Vec<&str> = stdout
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("rank,"))
        .collect();
    assert_eq!(
        data_rows.len(),
        1,
        "expected 1 PSE row from stratified preamble CSV; got {}: stdout={}",
        data_rows.len(),
        stdout
    );
    let parts: Vec<&str> = data_rows[0].split(',').collect();
    assert_eq!(parts[1], "rms");
}
