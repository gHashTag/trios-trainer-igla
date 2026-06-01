//! End-to-end exit code tests for `f2_provenance_check` — Loop 35 fix 4.
//!
//! In-binary tests (under `src/bin/f2_provenance_check.rs`) cannot reliably
//! subprocess the binary because `cargo test --bin` does not guarantee a
//! release artifact in `target/`. As an integration test (under `tests/`),
//! Cargo sets `CARGO_BIN_EXE_f2_provenance_check` to a freshly-built path —
//! guaranteed to exist, making the exit-code assertion CI-stable.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

fn binary_path() -> PathBuf {
    // Cargo sets this for every integration test; reliable across local + CI.
    PathBuf::from(env!("CARGO_BIN_EXE_f2_provenance_check"))
}

fn write_csv(path: &std::path::Path, schema: &str, sha: &str, ts: u64) {
    let mut f = File::create(path).unwrap();
    writeln!(f, "# f2_ablation_sweep provenance (W3C PROV / RO-Crate)").unwrap();
    writeln!(f, "# prov:generatedAt = {} (unix seconds UTC)", ts).unwrap();
    writeln!(
        f,
        "# prov:wasGeneratedBy = f2_ablation_sweep --mode loco --steps 200"
    )
    .unwrap();
    writeln!(f, "# prov:agent_git_sha = {}", sha).unwrap();
    writeln!(f, "# prov:host = test_host").unwrap();
    writeln!(f, "# prov:trainer_internals_schema = {}", schema).unwrap();
    writeln!(
        f,
        "# prov:cargo_pkg_version = {}",
        env!("CARGO_PKG_VERSION")
    )
    .unwrap();
    writeln!(
        f,
        "mode,fix_name,fix_index,cumulative_n,seed,bpb,config_hash,wall_s"
    )
    .unwrap();
    writeln!(f, "loco,wd,0,,42,0.5,0xdead,0.1").unwrap();
}

#[test]
fn exit_code_2_on_schema_mismatch() {
    let tmp = std::env::temp_dir().join("f2_prov_exit_fail_int.csv");
    write_csv(&tmp, "trainer_internals_v0_obsolete", "abc123", 1700000000);
    let status = Command::new(binary_path())
        .arg(tmp.to_str().unwrap())
        .status()
        .expect("spawn binary");
    let code = status.code().unwrap_or(-1);
    assert_eq!(code, 2, "expected exit code 2 (FAIL), got {}", code);
}

#[test]
fn exit_code_0_on_current_schema() {
    let tmp = std::env::temp_dir().join("f2_prov_exit_pass_int.csv");
    let cur = trios_trainer::race::multi_seed::TRAINER_INTERNALS_SCHEMA;
    // Use HEAD's git sha and a "now" timestamp so all checks pass.
    let head = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(1700000000);
    write_csv(&tmp, cur, &head, now);
    let status = Command::new(binary_path())
        .arg(tmp.to_str().unwrap())
        .status()
        .expect("spawn binary");
    let code = status.code().unwrap_or(-1);
    // 0=PASS or 1=WARN (e.g. if git rev-parse failed). Anything else is a regression.
    assert!(code == 0 || code == 1, "expected exit 0 or 1, got {}", code);
}

#[test]
fn exit_code_3_on_missing_preamble() {
    let tmp = std::env::temp_dir().join("f2_prov_exit_malformed_int.csv");
    let mut f = File::create(&tmp).unwrap();
    writeln!(
        f,
        "mode,fix_name,fix_index,cumulative_n,seed,bpb,config_hash,wall_s"
    )
    .unwrap();
    writeln!(f, "loco,wd,0,,42,0.5,0xdead,0.1").unwrap();
    drop(f);
    let status = Command::new(binary_path())
        .arg(tmp.to_str().unwrap())
        .status()
        .expect("spawn binary");
    let code = status.code().unwrap_or(-1);
    assert_eq!(code, 3, "expected exit code 3 (malformed), got {}", code);
}
