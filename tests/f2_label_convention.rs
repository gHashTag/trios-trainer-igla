//! End-to-end label-convention test — Loop 36 fix 2+3.
//!
//! Synthesizes pair/triplet labels exactly as `f2_ablation_sweep`'s
//! `run_pairwise` and `run_triplet` do (iterating `i<j(<k)` over
//! `AblationFix::ALL`), then runs them through `f2_dual_mediation`'s
//! permutation-tolerant lookup helpers via `f2_provenance_check`/`f2_dual_mediation`
//! subprocess. Locks the AllAblation-index label format end-to-end so any
//! future refactor (e.g. switching to lexicographic ordering) is caught.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use trios_trainer::race::ablation::{AblationFix, LABEL_ORDERING_CONVENTION};

fn dual_mediation_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_f2_dual_mediation"))
}

/// Generate pair labels in the SAME order as `run_pairwise` (Loop 27):
///   for i in 0..7 { for j in (i+1)..7 { label = "pair_<i.short>_<j.short>" } }
fn canonical_pair_labels() -> Vec<String> {
    let all = AblationFix::ALL;
    let mut out = Vec::new();
    for i in 0..all.len() {
        for j in (i + 1)..all.len() {
            out.push(format!(
                "pair_{}_{}",
                all[i].short_name(),
                all[j].short_name()
            ));
        }
    }
    out
}

/// Generate triplet labels in `run_triplet` order (Loop 29):
///   for i in 0..7 { for j in i+1..7 { for k in j+1..7 { ... } } }
fn canonical_triplet_labels() -> Vec<String> {
    let all = AblationFix::ALL;
    let mut out = Vec::new();
    for i in 0..all.len() {
        for j in (i + 1)..all.len() {
            for k in (j + 1)..all.len() {
                out.push(format!(
                    "triplet_{}_{}_{}",
                    all[i].short_name(),
                    all[j].short_name(),
                    all[k].short_name()
                ));
            }
        }
    }
    out
}

/// Write a synthetic CSV that contains every pair label and a few triplets in
/// `AblationFix::ALL` order. Then run f2_dual_mediation; if labels were silently
/// switched to lexicographic order the binary would return 0 rows.
#[test]
fn dual_mediation_finds_rows_when_labels_use_all_order() {
    let tmp = std::env::temp_dir().join("f2_label_e2e.csv");
    let mut f = File::create(&tmp).unwrap();
    // Header + minimal provenance.
    writeln!(
        f,
        "mode,fix_name,fix_index,cumulative_n,seed,bpb,config_hash,wall_s"
    )
    .unwrap();
    // Single seed, simple BPB pattern.
    writeln!(f, "pairwise,full_stack,-1,,42,4.0,0xdead,0.1").unwrap();
    // LOCO rows for every fix.
    for name in trios_trainer::race::ablation::CANONICAL_FIX_NAMES {
        writeln!(f, "loco,{},0,,42,5.0,0xdead,0.1", name).unwrap();
    }
    // Every pair in AblationFix::ALL order.
    for lbl in canonical_pair_labels() {
        writeln!(f, "pairwise,{},0,,42,4.5,0xdead,0.1", lbl).unwrap();
    }
    // Every triplet in ALL order.
    for lbl in canonical_triplet_labels() {
        writeln!(f, "triplet,{},0,,42,4.3,0xdead,0.1", lbl).unwrap();
    }
    drop(f);

    let out = Command::new(dual_mediation_path())
        .args(["--m1", "wd", "--m2", "warmup", tmp.to_str().unwrap()])
        .output()
        .expect("spawn binary");
    assert!(
        out.status.success(),
        "f2_dual_mediation failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    // 5 non-mediator fixes (rms, gradclip, clamp, smooth, dropout) → 5 data rows.
    let rank_rows: Vec<_> = stdout
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("rank,"))
        .collect();
    assert_eq!(
        rank_rows.len(),
        5,
        "expected 5 PSE rows, got {}: stdout={}",
        rank_rows.len(),
        stdout
    );
}

/// Locks the human-readable invariant string — surfaces it in tooling so a
/// future PR removing the convention triggers test failure here as well.
#[test]
fn label_ordering_convention_is_documented() {
    assert!(
        LABEL_ORDERING_CONVENTION.contains("AblationFix::ALL"),
        "convention doc must mention AblationFix::ALL ordering"
    );
    assert!(
        LABEL_ORDERING_CONVENTION.contains("permutation-tolerant"),
        "convention doc must reference permutation-tolerant lookup"
    );
}
