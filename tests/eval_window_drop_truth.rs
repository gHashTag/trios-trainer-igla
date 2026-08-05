//! An eval that quietly shrinks its own sample is not a measurement.
//!
//! `src/train_loop.rs` is strict: the first non-finite window aborts the
//! evaluation and returns `None`. Three sibling trainers adopted the opposite
//! convention -- `if loss.is_finite() { total += ...; n += 1; }` -- and then
//! published `total / n` without ever saying what `n` was. The `n == 0` guard
//! they all carry fires only under TOTAL poisoning, which is exactly the case
//! `cpu_train`'s own `test_nan_forward_pass_yields_no_measurement` covers by
//! NaN-ing `lm_head[0]`: every window dies, so nothing is published.
//!
//! The missing case is the PARTIAL poison. One bad embedding row, or an
//! overflow that only occurs on certain contexts, kills the windows that touch
//! it and leaves the rest. The survivors are the easy remainder, so the mean
//! over them is biased DOWNWARD -- the direction that manufactures a champion
//! -- and it is finite, plausible, and indistinguishable from a full eval.
//!
//! `cpu_train`'s `MIN_EVAL_CHUNKS` precondition cannot catch it: that number is
//! computed from the stream length before a single forward pass, so it grades
//! the PLAN and never the realisation. Same family as
//! `assert_train_val_disjoint`'s old `step_by(256)` scan -- a check whose
//! coverage was never the thing it claimed to cover.
//!
//! These are subprocess tests because the exit status and the results file are
//! the observables under test. Each run gets its own working directory, so a
//! test can never overwrite the git-tracked
//! `.trinity/results/cpu_train_f32_adamw_seed42.json`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// `cpu_train`'s exit code for a run that published nothing.
const EXIT_NO_MEASUREMENT: i32 = 7;

/// A vocabulary small enough to keep the run fast, large enough that the
/// poisoned row is one token among many.
const VOCAB: usize = 64;
const DIM: usize = 16;
const SEQ: usize = 32;

/// The token whose embedding row the fault-injection hook fills with NaN.
///
/// Byte `RARE_BYTE` maps to it under `token = byte % VOCAB`, and no other byte
/// in the fixtures does, so exactly the windows containing `RARE_BYTE` die.
const POISON_TOKEN: usize = 26;
const RARE_BYTE: u8 = b'Z'; // 90 % 64 == 26

/// A scratch directory that is removed when the test finishes.
struct Scratch {
    path: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "trios_eval_window_drop_{}_{}",
            name,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(path.join(".trinity/results")).expect("create scratch dir");
        Self { path }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// Write the train fixture: cycles of `a..p`, and NOT one byte of `RARE_BYTE`.
///
/// Keeping the poisoned token out of the train stream keeps its NaN from
/// entering a forward or backward pass, so the poison stays confined to the
/// one embedding row for the whole run.
fn write_train_corpus(dir: &Path) -> PathBuf {
    let path = dir.join("train.txt");
    let mut bytes = Vec::with_capacity(20_000);
    for i in 0..20_000usize {
        bytes.push(b'a' + (i % 16) as u8);
    }
    fs::write(&path, &bytes).expect("write train fixture");
    path
}

/// Write the val fixture: the same cycles, with `RARE_BYTE` every `period`
/// bytes so it lands in some eval windows and not others.
///
/// `cpu_train` evaluates the first 5000 tokens in windows of `SEQ + 1 = 33`,
/// so a period of 1000 puts the poisoned token in about 5 of ~152 windows.
fn write_val_corpus(dir: &Path, period: usize) -> PathBuf {
    let path = dir.join("val.txt");
    let mut bytes = Vec::with_capacity(20_000);
    for i in 0..20_000usize {
        if i % period == 0 && i > 0 {
            bytes.push(RARE_BYTE);
        } else {
            bytes.push(b'a' + (i % 16) as u8);
        }
    }
    fs::write(&path, &bytes).expect("write val fixture");
    path
}

fn cpu_train(scratch: &Scratch, train: &Path, val: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_cpu_train"));
    // Its own directory: the results file must never be the repo's.
    cmd.current_dir(&scratch.path);
    cmd.env_remove("TRIOS_FORMAT_TYPE");
    cmd.env_remove("TRIOS_ALGO_TYPE");
    cmd.env_remove("TRIOS_TRAIN_PATH");
    cmd.env_remove("TRIOS_VAL_PATH");
    cmd.env_remove("TRIOS_ALLOW_DROPPED_EVAL_WINDOWS");
    cmd.env_remove("TRIOS_TEST_POISON_EMBED_ROW");
    cmd.args([
        "--seed=47",
        "--steps=1",
        &format!("--vocab={VOCAB}"),
        &format!("--dim={DIM}"),
        &format!("--seq={SEQ}"),
        &format!("--train-data={}", train.display()),
        &format!("--val-data={}", val.display()),
    ]);
    cmd
}

/// Every results file in this scratch directory.
///
/// `results_path` used to hardcode `cpu_train_f32_adamw_seed47.json`. That name
/// is keyed on 3 of the 7 parameters that define a cell, which is the defect
/// `cpu_train::results_path` was changed to close, so reconstructing it here
/// would only re-encode the guess in the test. Each scratch directory holds
/// exactly one run, so the file it contains is by construction the file that
/// run wrote -- and an empty or crowded directory is itself worth reporting.
fn results_files(scratch: &Scratch) -> Vec<PathBuf> {
    let dir = scratch.path.join(".trinity/results");
    // A refused run writes nothing and does not even create the directory;
    // that is an empty result, not a test error.
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    found.sort();
    found
}

fn results_path(scratch: &Scratch) -> PathBuf {
    let mut found = results_files(scratch);
    assert_eq!(
        found.len(),
        1,
        "one run must leave exactly one results file, found {found:?}"
    );
    found.remove(0)
}

/// The control: an unpoisoned run measures every window it planned and says so.
#[test]
fn a_clean_run_publishes_a_complete_sample() {
    let scratch = Scratch::new("clean");
    let train = write_train_corpus(&scratch.path);
    let val = write_val_corpus(&scratch.path, 1000);

    let out = cpu_train(&scratch, &train, &val)
        .output()
        .expect("spawn cpu_train");
    assert!(
        out.status.success(),
        "a clean run must succeed: status={:?} stderr=\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );

    let json: serde_json::Value =
        serde_json::from_slice(&fs::read(results_path(&scratch)).expect("read results"))
            .expect("parse results");
    let planned = json["eval_windows_planned"].as_u64().expect("planned key");
    let realised = json["eval_windows_realised"]
        .as_u64()
        .expect("realised key");
    assert!(planned >= 8, "fixture must plan a real sample: {planned}");
    assert_eq!(realised, planned, "a clean run drops nothing");
    assert_eq!(json["eval_windows_dropped"].as_u64(), Some(0));
    assert_eq!(json["eval_windows_dropped_total"].as_u64(), Some(0));
    assert_eq!(json["fault_injected"].as_bool(), Some(false));
    assert!(
        json["final_bpb"].as_f64().expect("final_bpb").is_finite(),
        "and it does publish a BPB"
    );
}

/// The case that was missing: ONE poisoned embedding row.
///
/// Most windows survive and their mean is finite and plausible. The run must
/// refuse it anyway, because the mean no longer covers the sample it claims.
#[test]
fn a_partial_poison_refuses_instead_of_publishing_the_survivors() {
    let scratch = Scratch::new("partial");
    let train = write_train_corpus(&scratch.path);
    let val = write_val_corpus(&scratch.path, 1000);

    let out = cpu_train(&scratch, &train, &val)
        .env("TRIOS_TEST_POISON_EMBED_ROW", POISON_TOKEN.to_string())
        .output()
        .expect("spawn cpu_train");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();

    assert_eq!(
        out.status.code(),
        Some(EXIT_NO_MEASUREMENT),
        "a shrunken eval sample must exit {EXIT_NO_MEASUREMENT}, not publish. \
         stdout=\n{stdout}\nstderr=\n{stderr}"
    );
    assert!(
        stderr.contains("EVAL SAMPLE SHRANK"),
        "the refusal must name what shrank: stderr=\n{stderr}"
    );
    // The distinguishing evidence: it refused because the sample shrank, not
    // because nothing was measurable. Some windows really did survive.
    assert!(
        !stderr.contains("zero finite eval windows"),
        "this is a PARTIAL poison; a total one would prove nothing new: \
         stderr=\n{stderr}"
    );
    assert!(
        !stdout.contains("Final BPB"),
        "no BPB may be printed for a sample that shrank: stdout=\n{stdout}"
    );
    assert_eq!(
        results_files(&scratch),
        Vec::<PathBuf>::new(),
        "and no results file may be written"
    );
}

/// The mirror: the reduced sample is publishable, but only on request, and
/// only with the loss stamped into the record.
#[test]
fn the_opt_in_publishes_the_reduced_sample_and_stamps_the_loss() {
    let scratch = Scratch::new("optin");
    let train = write_train_corpus(&scratch.path);
    let val = write_val_corpus(&scratch.path, 1000);

    let out = cpu_train(&scratch, &train, &val)
        .env("TRIOS_TEST_POISON_EMBED_ROW", POISON_TOKEN.to_string())
        .env("TRIOS_ALLOW_DROPPED_EVAL_WINDOWS", "1")
        .output()
        .expect("spawn cpu_train");
    assert!(
        out.status.success(),
        "the opt-in must let the run finish: status={:?} stderr=\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );

    let json: serde_json::Value =
        serde_json::from_slice(&fs::read(results_path(&scratch)).expect("read results"))
            .expect("parse results");
    let planned = json["eval_windows_planned"].as_u64().expect("planned key");
    let realised = json["eval_windows_realised"]
        .as_u64()
        .expect("realised key");
    let dropped = json["eval_windows_dropped"]
        .as_u64()
        .expect("eval_windows_dropped must appear in the results JSON");
    assert!(dropped > 0, "the poison must have cost windows");
    assert!(
        realised > 0,
        "but not all of them, or this is the total-poison case"
    );
    assert_eq!(planned, realised + dropped);
    assert!(
        json["eval_windows_dropped_total"].as_u64().unwrap_or(0) >= dropped,
        "the per-run total must include the final eval's losses"
    );
    assert_eq!(
        json["fault_injected"].as_bool(),
        Some(true),
        "a fault-injected run must say so in its own record"
    );
    // The number it published is exactly the plausible one the silent path used
    // to hand over without comment.
    assert!(json["final_bpb"].as_f64().expect("final_bpb").is_finite());
}

/// F9: `--help` must print and exit, not train for 3000 steps and overwrite a
/// git-tracked results file.
///
/// `arg_or` matches only `--name=value`, so every unrecognised argument --
/// `--help` included -- was ignored and the run proceeded with defaults
/// (`--format f32 --algo adamw --seed 42`), which is precisely the cell whose
/// results file is checked in. Observed before the fix: steps 50 -> 3000,
/// final_bpb 7.0004 -> 3.6976.
#[test]
fn help_prints_and_writes_nothing() {
    let scratch = Scratch::new("help");
    let out = Command::new(env!("CARGO_BIN_EXE_cpu_train"))
        .current_dir(&scratch.path)
        .arg("--help")
        .output()
        .expect("spawn cpu_train");
    assert_eq!(out.status.code(), Some(0), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Usage: cpu_train"), "stdout=\n{stdout}");
    assert!(
        !stdout.contains("Training Complete"),
        "--help must not train: stdout=\n{stdout}"
    );
    let written: Vec<_> = fs::read_dir(scratch.path.join(".trinity/results"))
        .expect("read results dir")
        .filter_map(|e| e.ok().map(|e| e.file_name()))
        .collect();
    assert!(
        written.is_empty(),
        "--help must write no results file, found {written:?}"
    );
}

/// An argument this binary does not understand is a disagreement about what
/// the run is, and it fails rather than being ignored.
#[test]
fn an_unknown_argument_is_named_and_refused() {
    for bin in [
        env!("CARGO_BIN_EXE_cpu_train"),
        env!("CARGO_BIN_EXE_train_v2"),
        env!("CARGO_BIN_EXE_lstm_train"),
    ] {
        let scratch = Scratch::new("unknown");
        let out = Command::new(bin)
            .current_dir(&scratch.path)
            .arg("--nonsense=1")
            .output()
            .unwrap_or_else(|e| panic!("spawn {bin}: {e}"));
        assert_ne!(
            out.status.code(),
            Some(0),
            "{bin} must refuse an unknown argument"
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("--nonsense=1"),
            "{bin} must name the argument it refused: stderr=\n{stderr}"
        );
    }
}
