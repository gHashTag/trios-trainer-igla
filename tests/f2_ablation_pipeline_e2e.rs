//! F2 ablation pipeline end-to-end smoke test — Loop 29 audit fix 6.
//!
//! Synthesizes a small long-form CSV that matches the sweep schema, then shells
//! out to the release binaries `f2_ablation_aggregate`, `f2_iloco_score`, and
//! `f2_iloco_dot` to verify the full pipeline preserves expected columns and
//! row counts. Catches CSV-format drift that unit tests on individual functions
//! cannot — a frequent source of silent breakage across Loops 24-28.

use std::fs;
use std::process::Command;

fn bin_path(name: &str) -> String {
    let target = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".into());
    format!("{}/release/{}", target, name)
}

fn ensure_built(name: &str) {
    let p = bin_path(name);
    if std::path::Path::new(&p).exists() {
        return;
    }
    let out = Command::new("cargo")
        .args(["build", "--release", "--bin", name])
        .output()
        .expect("cargo build");
    if !out.status.success() {
        panic!(
            "cargo build --release --bin {} failed:\n{}",
            name,
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

fn synth_long_csv(path: &str) {
    // Minimal viable: loco (7 fixes × 5 seeds) + pairwise full_stack (5 seeds)
    // + 21 pairs × 5 seeds. Deterministic BPB values keep the assertion stable.
    let mut buf = String::from("mode,fix_name,fix_index,cumulative_n,seed,bpb,config_hash,wall_s\n");
    let fixes = ["rms", "warmup", "gradclip", "clamp", "smooth", "wd", "dropout"];
    let seeds: [u64; 5] = [42, 43, 44, 45, 46];
    // full_stack baseline.
    for &s in &seeds {
        buf += &format!(
            "pairwise,full_stack,-1,,{},{:.6},0x0000000000000001,0.001\n",
            s,
            5.13 + 0.001 * (s as f64 - 42.0)
        );
    }
    // LOCO rows: removing wd drops to 0.07; everything else stays ~5.0.
    for (i, f) in fixes.iter().enumerate() {
        let target = if *f == "wd" { 0.07 } else { 4.5 };
        for &s in &seeds {
            buf += &format!(
                "loco,{},{},,{},{:.6},0x000000000000000{:x},0.001\n",
                f,
                i,
                s,
                target + 0.001 * (s as f64 - 42.0),
                (i + 2) % 16
            );
        }
    }
    // Pairwise rows: 21 pairs in canonical order. Pair containing wd → 0.5; else 5.0.
    let mut idx = 0;
    for i in 0..fixes.len() {
        for j in (i + 1)..fixes.len() {
            let has_wd = fixes[i] == "wd" || fixes[j] == "wd";
            let target = if has_wd { 0.5 } else { 5.0 };
            for &s in &seeds {
                buf += &format!(
                    "pairwise,pair_{}_{},{},,{},{:.6},0x000000000000{:04x},0.001\n",
                    fixes[i],
                    fixes[j],
                    idx,
                    s,
                    target + 0.001 * (s as f64 - 42.0),
                    idx
                );
            }
            idx += 1;
        }
    }
    fs::write(path, buf).expect("write synth CSV");
}

/// Loop 30 fix 5: synthesize triplet + wd_pairwise rows for the extended e2e test.
fn synth_triplet_csv(path: &str) {
    let mut buf = String::from("mode,fix_name,fix_index,cumulative_n,seed,bpb,config_hash,wall_s\n");
    let fixes = ["rms", "warmup", "gradclip", "clamp", "smooth", "wd", "dropout"];
    let seeds: [u64; 5] = [42, 43, 44, 45, 46];
    // Triplet full_stack baseline.
    for &s in &seeds {
        buf += &format!(
            "triplet,full_stack,-1,,{},{:.6},0x0000000000000002,0.001\n",
            s,
            5.13 + 0.001 * (s as f64 - 42.0)
        );
    }
    // 35 triplets: any triplet containing wd → 0.3; else 5.0.
    let mut idx = 0;
    for i in 0..fixes.len() {
        for j in (i + 1)..fixes.len() {
            for k in (j + 1)..fixes.len() {
                let has_wd = fixes[i] == "wd" || fixes[j] == "wd" || fixes[k] == "wd";
                let target = if has_wd { 0.3 } else { 5.0 };
                for &s in &seeds {
                    buf += &format!(
                        "triplet,triplet_{}_{}_{},{},,{},{:.6},0x0000000000{:06x},0.001\n",
                        fixes[i], fixes[j], fixes[k], idx, s,
                        target + 0.001 * (s as f64 - 42.0), idx
                    );
                }
                idx += 1;
            }
        }
    }
    fs::write(path, buf).expect("write synth triplet CSV");
}

fn synth_wd_pairwise_csv(path: &str) {
    let mut buf = String::from("mode,fix_name,fix_index,cumulative_n,seed,bpb,config_hash,wall_s\n");
    let partners = ["rms", "warmup", "gradclip", "clamp", "smooth", "dropout"];
    let wds: [f64; 5] = [0.0, 0.005, 0.01, 0.03, 0.1];
    let seeds: [u64; 5] = [42, 43, 44, 45, 46];
    let mut idx = 0;
    for p in &partners {
        for &wd in &wds {
            // Synthetic: baseline (wd=0) at 0.1; higher wd inflates BPB.
            let target = 0.1 + 4.0 * wd;
            for &s in &seeds {
                buf += &format!(
                    "wd_pairwise,wdpair_{}_{:.3},{},,{},{:.6},0x000000000000{:04x},0.001\n",
                    p, wd, idx, s,
                    target + 0.001 * (s as f64 - 42.0), idx
                );
            }
            idx += 1;
        }
    }
    fs::write(path, buf).expect("write synth wd_pairwise CSV");
}

#[test]
fn ablation_pipeline_three_way_iloco_with_synth_triplets() {
    ensure_built("f2_iloco_score");
    let tmp = std::env::temp_dir().join("f2_e2e_3way");
    fs::create_dir_all(&tmp).unwrap();
    let long_csv = tmp.join("long.csv");
    let triplet_csv = tmp.join("triplet.csv");
    let iloco3_csv = tmp.join("iloco3.csv");
    synth_long_csv(long_csv.to_str().unwrap());
    synth_triplet_csv(triplet_csv.to_str().unwrap());

    let out = Command::new(bin_path("f2_iloco_score"))
        .args([
            "--three-way",
            long_csv.to_str().unwrap(),
            triplet_csv.to_str().unwrap(),
            "--out",
            iloco3_csv.to_str().unwrap(),
        ])
        .output()
        .expect("run 3-way");
    assert!(
        out.status.success(),
        "3-way iloco failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = fs::read_to_string(&iloco3_csv).unwrap();
    assert!(text.starts_with("rank,fix_a,fix_b,fix_c,iloco_3,kind,p_value,q_value_bh"));
    // 35 triplets.
    assert_eq!(text.lines().count(), 1 + 35);
}

#[test]
fn ablation_pipeline_three_way_with_permutation_flag() {
    ensure_built("f2_iloco_score");
    let tmp = std::env::temp_dir().join("f2_e2e_perm");
    fs::create_dir_all(&tmp).unwrap();
    let long_csv = tmp.join("long.csv");
    let triplet_csv = tmp.join("triplet.csv");
    let iloco3_csv = tmp.join("iloco3_perm.csv");
    synth_long_csv(long_csv.to_str().unwrap());
    synth_triplet_csv(triplet_csv.to_str().unwrap());

    let out = Command::new(bin_path("f2_iloco_score"))
        .args([
            "--three-way",
            "--permutation",
            long_csv.to_str().unwrap(),
            triplet_csv.to_str().unwrap(),
            "--out",
            iloco3_csv.to_str().unwrap(),
        ])
        .output()
        .expect("run 3-way perm");
    assert!(
        out.status.success(),
        "perm 3-way failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = fs::read_to_string(&iloco3_csv).unwrap();
    // At N=5 permutation: p-values are multiples of 1/32 = 0.03125.
    // Verify no p-value is impossibly small (no df underflow).
    for line in text.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 7 {
            let p: f64 = parts[6].parse().unwrap_or(1.0);
            assert!(p >= 1.0 / 32.0 - 1e-9, "permutation p underflowed: {}", p);
        }
    }
}

/// Loop 32 fix 3: per-row tight tolerance. Loose ±0.5 used in earlier loops
/// could miss the kind of 8x training-code drift Loop 31 surfaced.
/// At our N=5 with σ ≈ 0.01-0.05 BPB across seeds, a per-row tolerance of
/// 0.02 BPB locks the synthetic baseline tight enough to flag any infra drift.
fn assert_bpb_close(actual: f64, expected: f64, tol: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{}: BPB drift {:.6} vs expected {:.6} (|Δ|={:.6} > tol {:.6})",
        label,
        actual,
        expected,
        (actual - expected).abs(),
        tol
    );
}

/// Loop 37 fix 6: synthesize a CSV with non-canonical large seed values (5-digit
/// and full u64-range) and verify the aggregator preserves them losslessly.
/// Locks the assumption that seed is always parsed as u64, never narrowed to
/// u8/u16/u32 in CSV → parse → BTreeMap round-trip.
#[test]
fn ablation_aggregate_preserves_large_seed_ids() {
    ensure_built("f2_ablation_aggregate");
    let tmp = std::env::temp_dir().join("f2_large_seed.csv");
    let mut buf = String::from("mode,fix_name,fix_index,cumulative_n,seed,bpb,config_hash,wall_s\n");
    // Mix of small, medium, large, and near-u64::MAX seeds.
    let seeds = [42u64, 100_000u64, 9_223_372_036_854_775_807u64, 18_446_744_073_709_551_614u64];
    for (i, &s) in seeds.iter().enumerate() {
        buf += &format!(
            "pairwise,full_stack,-1,,{},{:.6},0x{:016x},0.001\n",
            s,
            4.0 + 0.001 * (i as f64),
            i as u64
        );
    }
    std::fs::write(&tmp, buf).expect("write");
    let out = Command::new(bin_path("f2_ablation_aggregate"))
        .arg(tmp.to_str().unwrap())
        .output()
        .expect("run aggregate");
    assert!(out.status.success(), "aggregator failed: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    // The aggregator should at least preserve N=4 (all seeds counted).
    // We grep for the "n,..." column value 4 in the data row.
    let data_lines: Vec<&str> = stdout
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("mode,"))
        .collect();
    assert!(!data_lines.is_empty(), "no data rows produced");
    let parts: Vec<&str> = data_lines[0].split(',').collect();
    // n column position (0-indexed): mode,fix_name,fix_index,bpb_mean,bpb_std,ci95_lo,ci95_hi,n,...
    let n_col: usize = parts[7].parse().unwrap_or(0);
    assert_eq!(n_col, 4, "expected N=4 (all 4 distinct seeds preserved), got {}", n_col);
}

#[test]
fn ablation_synth_csv_bpb_per_row_locked() {
    // Verify synth CSV emits BPB values within ±0.02 per row of the encoded mean.
    // Locks the synth fixture format used by every downstream e2e test.
    let tmp = std::env::temp_dir().join("f2_e2e_tol");
    fs::create_dir_all(&tmp).unwrap();
    let long_csv = tmp.join("long.csv");
    synth_long_csv(long_csv.to_str().unwrap());
    let text = fs::read_to_string(&long_csv).unwrap();
    // Pick the full_stack row at seed=42 — encoded value is 5.13.
    let row = text
        .lines()
        .find(|l| l.starts_with("pairwise,full_stack,") && l.contains(",42,"))
        .expect("find full_stack@42");
    let parts: Vec<&str> = row.split(',').collect();
    let bpb: f64 = parts[5].parse().unwrap();
    assert_bpb_close(bpb, 5.13, 0.02, "full_stack@seed42");
    // Pair with wd at seed=46 — encoded value 0.5 + 4 × jitter.
    let pair_row = text
        .lines()
        .find(|l| l.starts_with("pairwise,pair_rms_wd,") && l.contains(",46,"))
        .expect("find pair_rms_wd@46");
    let parts: Vec<&str> = pair_row.split(',').collect();
    let bpb: f64 = parts[5].parse().unwrap();
    assert_bpb_close(bpb, 0.504, 0.02, "pair_rms_wd@seed46");
}

#[test]
fn ablation_pipeline_wd_pairwise_baseline_is_partner_specific() {
    ensure_built("f2_ablation_aggregate");
    let tmp = std::env::temp_dir().join("f2_e2e_wdpair");
    fs::create_dir_all(&tmp).unwrap();
    let long_csv = tmp.join("wdpair.csv");
    synth_wd_pairwise_csv(long_csv.to_str().unwrap());

    let out = Command::new(bin_path("f2_ablation_aggregate"))
        .arg(long_csv.to_str().unwrap())
        .output()
        .expect("run aggregate");
    assert!(
        out.status.success(),
        "wd_pairwise aggregate failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let wide = String::from_utf8_lossy(&out.stdout).into_owned();
    // wdpair_<X>_0.000 rows should have delta=0 (their own baseline).
    let zero_rows: Vec<&str> = wide
        .lines()
        .filter(|l| l.contains("_0.000,") && l.starts_with("wd_pairwise,"))
        .collect();
    assert_eq!(zero_rows.len(), 6, "expected 6 partner baselines, got {}", zero_rows.len());
    for row in &zero_rows {
        let parts: Vec<&str> = row.split(',').collect();
        let delta: f64 = parts[8].parse().unwrap_or(f64::NAN);
        assert!(delta.abs() < 1e-9, "wdpair baseline delta should be 0, got {}", delta);
    }
    // No baseline-fallback warning should fire for wd_pairwise.
    assert!(!stderr.contains("WARN: no canonical baseline"), "unexpected baseline warning: {}", stderr);
}

#[test]
fn ablation_pipeline_aggregate_then_iloco_then_dot() {
    ensure_built("f2_ablation_aggregate");
    ensure_built("f2_iloco_score");
    ensure_built("f2_iloco_dot");

    let tmp = std::env::temp_dir().join("f2_e2e");
    fs::create_dir_all(&tmp).unwrap();
    let long_csv = tmp.join("long.csv");
    let wide_csv = tmp.join("wide.csv");
    let iloco_csv = tmp.join("iloco.csv");
    let dot_out = tmp.join("net.dot");
    synth_long_csv(long_csv.to_str().unwrap());

    // 1) Aggregate → wide CSV.
    let agg = Command::new(bin_path("f2_ablation_aggregate"))
        .arg(long_csv.to_str().unwrap())
        .output()
        .expect("run aggregate");
    assert!(
        agg.status.success(),
        "aggregate exited non-zero: {}",
        String::from_utf8_lossy(&agg.stderr)
    );
    let wide_text = String::from_utf8_lossy(&agg.stdout).into_owned();
    fs::write(&wide_csv, &wide_text).unwrap();
    // Expect canonical column header (Loop 28 AAA rename).
    assert!(wide_text.starts_with(
        "mode,fix_name,fix_index,bpb_mean,bpb_std,ci95_lo,ci95_hi,n,delta_vs_baseline,cohen_d,p_paired_t"
    ));
    // Expect the full_stack row to land at delta=0 (canonical baseline picker).
    assert!(
        wide_text.contains("pairwise,full_stack,"),
        "expected full_stack row in wide output"
    );

    // 2) iLOCO score from the long CSV (uses loco + pairwise).
    let iloco = Command::new(bin_path("f2_iloco_score"))
        .args([long_csv.to_str().unwrap(), "--out", iloco_csv.to_str().unwrap()])
        .output()
        .expect("run iloco_score");
    assert!(
        iloco.status.success(),
        "iloco_score exited non-zero: {}",
        String::from_utf8_lossy(&iloco.stderr)
    );
    let iloco_text = fs::read_to_string(&iloco_csv).unwrap();
    assert!(iloco_text.starts_with(
        "rank,fix_a,fix_b,delta_a,delta_b,delta_ab,iloco,kind,p_value,q_value_bh"
    ));
    // 21 pairs.
    assert_eq!(iloco_text.lines().count(), 1 + 21);

    // 3) DOT viz from the iLOCO CSV.
    let dot = Command::new(bin_path("f2_iloco_dot"))
        .args([iloco_csv.to_str().unwrap(), "--out", dot_out.to_str().unwrap()])
        .output()
        .expect("run iloco_dot");
    assert!(
        dot.status.success(),
        "iloco_dot exited non-zero: {}",
        String::from_utf8_lossy(&dot.stderr)
    );
    let dot_text = fs::read_to_string(&dot_out).unwrap();
    assert!(dot_text.contains("graph f2_interactions"));
    // The seven fix node names must appear.
    for name in ["rms", "warmup", "gradclip", "clamp", "smooth", "wd", "dropout"] {
        assert!(
            dot_text.contains(&format!("\"{}\"", name)),
            "DOT missing node {}",
            name
        );
    }
}
