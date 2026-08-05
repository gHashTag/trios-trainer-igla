//! matrix_row_label_truth -- the label on a matrix row must describe the run.
//!
//! `matrix_runner` writes the rows that become `ssot.bpb_samples`. Two
//! independent mislabellings were found in it on 2026-08-03, both proven by
//! execution rather than by reading:
//!
//!   F1  `run_cpu_train` never passed `--lr` to the child, so EVERY cell
//!       trained at `cpu_train`'s own default 0.003 while the row was stamped
//!       with whatever `--lr` the operator asked for. The nightly schedule
//!       passes LR=0.001, so every nightly row ever published says `LR001` and
//!       was trained at 0.003. This is the `--eval-every` defect inverted:
//!       there an OBSERVATION knob silently changed the artefact, here a
//!       RECIPE knob silently changed nothing while the record claimed it did.
//!
//!   F4  `format_lr_token` was not injective: `0.1` and `1.0` both rendered
//!       `LR1`, and `1e-7` and `0` both rendered `LR0`. `canon_name` is the
//!       ledger's identity key that `ON CONFLICT` de-duplicates on, so two
//!       different recipes collided into a single row.
//!
//! Every assertion here is made against a SPAWNED BINARY, not against an
//! internal function, because both defects were invisible from the inside:
//! the runner's own parsed `lr` was correct all along, it simply never reached
//! the trainer.
//!
//! No test in this file writes to a database: `DATABASE_URL` and
//! `MATRIX_DATABASE_URL` are removed from every child environment, which is
//! also the configuration under which `matrix_runner` legitimately exits 0
//! without a row.
//!
//! Anchor: phi^2 + phi^-2 = 3.

use std::collections::BTreeMap;
use std::process::Command;

const MATRIX_RUNNER: &str = env!("CARGO_BIN_EXE_matrix_runner");
const CPU_TRAIN: &str = env!("CARGO_BIN_EXE_cpu_train");

/// Keep the training budget small enough that the whole file runs in one
/// sitting: these tests exist to prove a LABEL tracks a RUN, not to measure a
/// model.
const TEST_STEPS: &str = "120";
const TEST_DIM: &str = "64";
/// Lucas seed -- `matrix_runner` rejects anything outside SEED_CANON.
const TEST_SEED: &str = "1597";
/// Distinct SEED_CANON members so the two tests that spawn a full
/// `matrix_runner` cell never share a `.trinity/results/` path.
const SEED_ROW_LR: &str = "2584";
const SEED_ROW_STAMPS: &str = "4181";

fn base_command(bin: &str) -> Command {
    let mut cmd = Command::new(bin);
    // Nothing in this suite may reach a database.
    cmd.env_remove("DATABASE_URL");
    cmd.env_remove("MATRIX_DATABASE_URL");
    cmd.env_remove("TRIOS_ALLOW_UNFAITHFUL_FORMAT");
    cmd
}

/// The canon_name `matrix_runner` would file a cell under, obtained from the
/// binary's own `--dry-run-canon` path so the assertion is about the shipped
/// code and not about a copy of it living in this test.
fn canon_name_for_lr(lr: &str) -> String {
    let out = base_command(MATRIX_RUNNER)
        .args([
            "--format=fp32",
            "--algo=adamw",
            &format!("--seed={TEST_SEED}"),
            "--hidden=128",
            &format!("--lr={lr}"),
            "--dry-run-canon",
        ])
        .output()
        .expect("spawn matrix_runner");
    assert!(
        out.status.success(),
        "matrix_runner --dry-run-canon failed for lr={lr}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    stdout
        .lines()
        .find_map(|l| l.strip_prefix("CANON_NAME ").map(|s| s.trim().to_string()))
        .unwrap_or_else(|| panic!("no CANON_NAME line for lr={lr}; stdout=\n{stdout}"))
}

/// Final BPB reported by `cpu_train` for one (seed, steps, dim, lr) tuple.
fn final_bpb_at_lr(lr: &str) -> f64 {
    let out = base_command(CPU_TRAIN)
        .args([
            &format!("--seed={TEST_SEED}"),
            &format!("--steps={TEST_STEPS}"),
            &format!("--dim={TEST_DIM}"),
            &format!("--lr={lr}"),
        ])
        .output()
        .expect("spawn cpu_train");
    assert!(
        out.status.success(),
        "cpu_train failed at lr={lr}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    // "Time: 0.9s | Init BPB: 7.0001 | Best BPB: ... | Final BPB: 5.4359 | ..."
    let line = stdout
        .lines()
        .find(|l| l.contains("Final BPB:"))
        .unwrap_or_else(|| panic!("no Final BPB line at lr={lr}; stdout=\n{stdout}"));
    let after = line
        .split("Final BPB:")
        .nth(1)
        .expect("Final BPB separator");
    after
        .trim()
        .split_whitespace()
        .next()
        .expect("Final BPB value")
        .parse::<f64>()
        .unwrap_or_else(|e| panic!("unparseable Final BPB in {line:?}: {e}"))
}

/// F4: `canon_name` is the identity key rows de-duplicate on. Two different
/// learning rates that share a canon_name are two recipes merged into one row.
#[test]
fn lr_token_is_injective_through_canon_name() {
    // The LR values the two workflows can actually produce -- the
    // `workflow_dispatch` input default `0.001,0.0001`, the nightly and
    // PR-smoke constant `0.001`, `cpu_train`'s own default `0.003` -- plus the
    // pairs that used to collide.
    let lrs = [
        "0.001", "0.0001", "0.003", "0.01", "0.1", "0.5", "1.0", "1.5", "0.9", "0.00001",
        "0.000001", "0.0000001", "0",
    ];
    let mut seen: BTreeMap<String, &str> = BTreeMap::new();
    for lr in lrs {
        let canon = canon_name_for_lr(lr);
        if let Some(prev) = seen.insert(canon.clone(), lr) {
            panic!(
                "LR TOKEN COLLISION: lr={prev} and lr={lr} both file under \
                 canon_name {canon:?}. ON CONFLICT would de-duplicate two \
                 different recipes into one row."
            );
        }
    }
    assert_eq!(seen.len(), lrs.len());

    // The two documented collisions, named so a regression reports the defect
    // rather than an index.
    assert_ne!(
        canon_name_for_lr("0.1"),
        canon_name_for_lr("1.0"),
        "0.1 and 1.0 both rendered LR1 before this fix"
    );
    assert_ne!(
        canon_name_for_lr("0.0000001"),
        canon_name_for_lr("0"),
        "1e-7 and 0 both rendered LR0 before this fix ({{:.6}} truncation)"
    );

    // The LR tokens this repo has already published must keep their shape, or
    // the fix would orphan every historical row instead of correcting the new
    // ones.
    assert_eq!(
        canon_name_for_lr("0.001"),
        "IGLA-MATRIX-fp32-h128-LR001-rng1597-adamw"
    );
    assert_eq!(
        canon_name_for_lr("0.0001"),
        "IGLA-MATRIX-fp32-h128-LR0001-rng1597-adamw"
    );
}

/// F1, measured end to end: the same seed, steps and dim at two different
/// learning rates must produce two different numbers. Before this fix the
/// matrix ran every cell at 0.003 regardless of its label, so this comparison
/// was bit-identical no matter which LRs were named.
#[test]
fn lr_changes_the_measurement() {
    let low = final_bpb_at_lr("0.001");
    let high = final_bpb_at_lr("0.9");
    assert!(
        low.is_finite(),
        "cpu_train reported a non-finite bpb at lr=0.001: {low}"
    );
    assert!(
        (low - high).abs() > 1e-9,
        "bpb is IDENTICAL at lr=0.001 ({low}) and lr=0.9 ({high}): the \
         learning rate is not reaching the training loop, so every row's LR \
         label is a claim with no measurement behind it"
    );
    // The default the mislabelled rows were really trained at. If either
    // labelled run matches it bit-for-bit, `--lr` is being dropped again.
    let default_lr = final_bpb_at_lr("0.003");
    assert!(
        (low - default_lr).abs() > 1e-9,
        "lr=0.001 produced exactly the bpb of cpu_train's default 0.003 \
         ({default_lr}): the flag is being ignored"
    );
}

/// F1 at the runner level: the row `matrix_runner` emits must carry the LR the
/// child REPORTED, and a different label must produce a different bpb. This is
/// the assertion that actually fails on the pre-fix binary -- `cpu_train`
/// honoured `--lr` all along; the runner simply never sent it.
#[test]
fn matrix_row_bpb_tracks_the_labelled_lr() {
    let row_low = matrix_row_for_lr(SEED_ROW_LR, "0.001");
    let row_high = matrix_row_for_lr(SEED_ROW_LR, "0.01");

    let bpb_low = row_low["bpb"].as_f64().expect("bpb in row");
    let bpb_high = row_high["bpb"].as_f64().expect("bpb in row");
    assert!(
        (bpb_low - bpb_high).abs() > 1e-9,
        "MATRIX_ROW bpb is identical at lr=0.001 ({bpb_low}) and lr=0.01 \
         ({bpb_high}): the runner is not passing --lr to the child, so the LR \
         in canon_name is a label with no run behind it"
    );

    // The row must cite what the child reported, not what was requested.
    // cpu_train holds lr as f32, so the reported value is the f32 widened.
    let reported = row_low["lr"].as_f64().expect("lr in row");
    assert!(
        (reported - (0.001_f32 as f64)).abs() < 1e-12,
        "row lr={reported} is not the f32-representable form of the requested \
         0.001"
    );
    let reported_high = row_high["lr"].as_f64().expect("lr in row");
    assert!((reported_high - (0.01_f32 as f64)).abs() < 1e-12);

    // Labels that differ must file under canon_names that differ.
    assert_ne!(
        row_low["canon_name"].as_str(),
        row_high["canon_name"].as_str()
    );
}

/// `seed` is a parameter, not a constant, ON PURPOSE: `cpu_train` writes its
/// results to `.trinity/results/cpu_train_<fmt>_<algo>_seed<seed>.json`, and
/// `matrix_runner` reads that same path back after its child exits. Two tests
/// running the same (format, algo, seed) tuple in parallel -- which is
/// `cargo test`'s default -- would clobber each other's results file between
/// the child's write and the parent's read, and the row that came back would
/// be a real measurement of the WRONG cell. That is the same class of defect
/// this file exists to close, so each test claims its own seed from
/// SEED_CANON.
fn matrix_row_for_lr(seed: &str, lr: &str) -> serde_json::Value {
    let out = base_command(MATRIX_RUNNER)
        .args([
            "--format=fp32",
            "--algo=adamw",
            &format!("--seed={seed}"),
            &format!("--hidden={TEST_DIM}"),
            &format!("--lr={lr}"),
            &format!("--steps={TEST_STEPS}"),
            "--vocab=128",
            "--seq=32",
        ])
        .output()
        .expect("spawn matrix_runner");
    assert!(
        out.status.success(),
        "matrix_runner failed at lr={lr}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let line = stdout
        .lines()
        .find_map(|l| l.strip_prefix("MATRIX_ROW "))
        .unwrap_or_else(|| panic!("no MATRIX_ROW line at lr={lr}; stdout=\n{stdout}"));
    serde_json::from_str(line).unwrap_or_else(|e| panic!("unparseable MATRIX_ROW: {e}\n{line}"))
}

/// F2: `fp80` is returned UNCHANGED by `fake_quantize_f32`, so an `fp80` row
/// is arithmetically an `fp32` row and an `fp80`-vs-`fp32` tie is an identity
/// dressed as a result. `posit16` is an IEEE 10-bit mantissa mask, not a posit
/// encoder. The crate has always known this -- `FormatKind::is_faithful()`
/// returns false for both -- and until now NOTHING read that marker.
#[test]
fn non_faithful_formats_are_refused_before_any_training() {
    for degenerate in ["fp80", "posit16"] {
        let out = base_command(MATRIX_RUNNER)
            .args([
                &format!("--format={degenerate}"),
                "--algo=adamw",
                &format!("--seed={TEST_SEED}"),
                "--hidden=128",
                "--lr=0.001",
                "--dry-run-canon",
            ])
            .output()
            .expect("spawn matrix_runner");
        assert!(
            !out.status.success(),
            "{degenerate} was accepted as a publishable format"
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("NON-FAITHFUL FORMAT"),
            "{degenerate} was rejected for the wrong reason: {stderr}"
        );
    }
}

/// The opt-in exists so a degenerate format can still be STUDIED; what it may
/// not do is enter the ledger unmarked.
#[test]
fn unfaithful_opt_in_is_explicit_and_stamped() {
    let out = base_command(MATRIX_RUNNER)
        .env("TRIOS_ALLOW_UNFAITHFUL_FORMAT", "1")
        .args([
            "--format=fp80",
            "--algo=adamw",
            &format!("--seed={TEST_SEED}"),
            "--hidden=128",
            "--lr=0.001",
            "--dry-run-canon",
        ])
        .output()
        .expect("spawn matrix_runner");
    assert!(
        out.status.success(),
        "opt-in did not permit fp80: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("format_faithful=false"),
        "opt-in ran fp80 without warning that the row is not a measurement: {stderr}"
    );
}

/// Every row must declare whether its format is a real kernel, so a consumer
/// never has to infer it from the format name.
#[test]
fn every_row_declares_format_faithful_and_git_dirty() {
    let row = matrix_row_for_lr(SEED_ROW_STAMPS, "0.001");
    assert_eq!(
        row["format_faithful"].as_bool(),
        Some(true),
        "fp32 row must be stamped faithful"
    );
    assert!(
        row.get("git_dirty").is_some(),
        "row carries no git_dirty: a sha with no dirty flag can name a commit \
         the row was not built from (observed: sha 3c1f751 from a tree with \
         874 dirty entries)"
    );
    let provenance = row["git_provenance"]
        .as_str()
        .expect("git_provenance in row");
    assert!(
        matches!(
            provenance,
            "verified-local" | "asserted-by-environment" | "unavailable"
        ),
        "unknown git_provenance {provenance:?}"
    );
}
