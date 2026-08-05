//! A trainer must not start on an argument it cannot honour.
//!
//! Eight binaries discarded unrecognised argument NAMES in silence. Measured
//! before the guard existed, with a flag no parser reads:
//!
//! ```text
//! $ timeout 12 ./target/release/hybrid_train --this-flag-does-not-exist; echo $?
//! 124            # still training after 12 seconds
//! ```
//!
//! Seven of the eight behaved that way; `ngram_train_gf16` exited 1 only
//! because it separately demands a corpus. "Still training" IS the defect: the
//! operator asked for one run, the fleet started another, and the BPB that runs
//! out the far end is published with exit 0 on a worker where nobody reads
//! stderr. So a child still alive at the timeout FAILS here - it is never
//! skipped and never treated as inconclusive.
//!
//! These are subprocess tests on purpose: the exit status is the observable
//! under test, and a unit test on `reject_unknown_args` cannot see whether a
//! binary calls it before it starts training.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// The refusal exit code, spelled out rather than imported: this is a
/// subprocess test against the binaries, and a constant read from the same
/// crate the binaries read cannot detect the two disagreeing.
const EXIT_BAD_ARGS: i32 = 4;

/// A refusal is an argument check, not a training run: it must land in
/// milliseconds. The budget is generous only so a loaded CI box cannot turn a
/// pass into a flake.
const REFUSAL_TIMEOUT: Duration = Duration::from_secs(10);

/// The eight binaries that ignored unknown argument names, and the path to each.
fn guarded_binaries() -> Vec<(&'static str, &'static str)> {
    vec![
        ("hybrid_train", env!("CARGO_BIN_EXE_hybrid_train")),
        ("ngram_train", env!("CARGO_BIN_EXE_ngram_train")),
        ("ngram_train_gf16", env!("CARGO_BIN_EXE_ngram_train_gf16")),
        ("igla_trigram", env!("CARGO_BIN_EXE_igla_trigram")),
        ("tjepa_train", env!("CARGO_BIN_EXE_tjepa_train")),
        ("concat_train", env!("CARGO_BIN_EXE_concat_train")),
        ("trinity_pr1722", env!("CARGO_BIN_EXE_trinity_pr1722")),
        ("arch_explorer", env!("CARGO_BIN_EXE_arch_explorer")),
    ]
}

/// Environment that must not decide the result: a DSN in the developer's shell
/// would send these runs to a live ledger.
fn clean_command(exe: &str) -> Command {
    let mut cmd = Command::new(exe);
    // The corpus paths in these binaries are repo-relative defaults; `cargo
    // test` does not guarantee the cwd of the spawned child.
    cmd.current_dir(env!("CARGO_MANIFEST_DIR"));
    for var in [
        "DATABASE_URL",
        "MATRIX_DATABASE_URL",
        "NEON_DATABASE_URL",
        "TRIOS_NEON_DSN",
        "TRIOS_DATABASE_URL",
        "TRIOS_LEDGER_WRITE",
        "SEED",
        "STEPS",
        "TRIOS_SEED",
        "TRIOS_STEPS",
    ] {
        cmd.env_remove(var);
    }
    cmd
}

/// Run to completion inside `budget`, or kill and say so.
///
/// `Ok((code, stderr, stdout))`, or `Err(elapsed)` when the child had to be
/// killed. A killed child is a failure of the thing under test, never a reason
/// to skip: it means the binary accepted the argument and started training.
#[allow(clippy::type_complexity)]
fn run_with_timeout(
    mut cmd: Command,
    budget: Duration,
) -> Result<(Option<i32>, String, String), Duration> {
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the binary under test");
    let started = Instant::now();
    let status = loop {
        match child.try_wait().expect("poll the child") {
            Some(status) => break status,
            None => {
                if started.elapsed() > budget {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(started.elapsed());
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    };
    let mut stdout = String::new();
    let mut stderr = String::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_string(&mut stdout);
    }
    if let Some(mut err) = child.stderr.take() {
        let _ = err.read_to_string(&mut stderr);
    }
    Ok((status.code(), stderr, stdout))
}

/// Every guarded binary refuses a flag no parser reads, names it, and stops.
#[test]
fn every_guarded_binary_refuses_an_unknown_flag_by_name() {
    const BOGUS: &str = "--this-flag-does-not-exist";
    let mut failures: Vec<String> = Vec::new();

    for (name, exe) in guarded_binaries() {
        let mut cmd = clean_command(exe);
        cmd.arg(BOGUS);
        match run_with_timeout(cmd, REFUSAL_TIMEOUT) {
            Err(elapsed) => failures.push(format!(
                "{name}: still running after {:.1}s with {BOGUS}. The binary \
                 accepted an argument it cannot read and started a run nobody \
                 asked for - which is the defect, not a slow machine.",
                elapsed.as_secs_f64()
            )),
            Ok((code, stderr, stdout)) => {
                if code != Some(EXIT_BAD_ARGS) {
                    failures.push(format!(
                        "{name}: exit {code:?}, expected {EXIT_BAD_ARGS}. stderr: {}",
                        stderr.trim()
                    ));
                }
                // The refusal must name the offending flag: an operator with a
                // typo among twenty flags needs to be told which one.
                if !stderr.contains(BOGUS) && !stdout.contains(BOGUS) {
                    failures.push(format!(
                        "{name}: refusal does not name {BOGUS}. stderr: {}",
                        stderr.trim()
                    ));
                }
                // ...and say what would have worked.
                if !stderr.contains("Accepted arguments:") {
                    failures.push(format!(
                        "{name}: refusal does not list the accepted arguments. stderr: {}",
                        stderr.trim()
                    ));
                }
            }
        }
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The near-miss that started this: a real flag of a NEIGHBOURING binary.
///
/// `hybrid_train` reads `--train-path` / `--val-path`. Given `--train` /
/// `--val` - the spellings `ngram_train_gf16` accepts - it dropped both,
/// trained the DEFAULT split and printed `bpb=5.0731` with exit 0 at 20 steps.
/// The corpus pair is the identity of a BPB, so this is the case the guard
/// exists for, and it is checked separately from the obviously-bogus flag.
#[test]
fn hybrid_train_refuses_the_corpus_flags_it_does_not_read() {
    let mut cmd = clean_command(env!("CARGO_BIN_EXE_hybrid_train"));
    cmd.args([
        "--seed=47",
        "--steps=20",
        "--train=/tmp/does_not_exist_train.txt",
        "--val=/tmp/does_not_exist_val.txt",
    ]);
    let (code, stderr, _stdout) = run_with_timeout(cmd, REFUSAL_TIMEOUT)
        .expect("hybrid_train must refuse before it trains, not after");
    assert_eq!(
        code,
        Some(EXIT_BAD_ARGS),
        "a corpus flag this binary cannot read must stop the run: {}",
        stderr.trim()
    );
    assert!(stderr.contains("--train"), "{}", stderr.trim());
    assert!(
        stderr.contains("--train-path=VALUE"),
        "the refusal must offer the spelling that works: {}",
        stderr.trim()
    );
}

/// Positive control: the guard refuses typos, not work.
///
/// A guard that rejected everything would pass the test above and break the
/// fleet, so one binary is run for real with a correct flag set and must reach
/// a measurement. `concat_train` is the choice because it writes nothing to
/// disk - it prints `BPB=<value>` and exits - so the run leaves no artifact and
/// the tempdir cwd proves it.
#[test]
fn a_correct_flag_set_still_runs() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let train = repo.join("data/tiny_shakespeare.txt");
    let val = repo.join("data/tiny_shakespeare_val.txt");
    assert!(
        train.is_file() && val.is_file(),
        "shipped corpus pair missing"
    );

    let tmp = std::env::temp_dir().join(format!(
        "unknown_flag_refusal_{}_{}",
        std::process::id(),
        Instant::now().elapsed().as_nanos()
    ));
    std::fs::create_dir_all(&tmp).expect("tempdir");

    let mut cmd = clean_command(env!("CARGO_BIN_EXE_concat_train"));
    // The cwd is the tempdir, not the repo: anything this run wrote relative to
    // the cwd would land here, and nothing should.
    cmd.current_dir(&tmp);
    cmd.args([
        "--seed=47",
        "--steps=1",
        "--dim=8",
        "--hidden=16",
        "--ctx=4",
        format!("--train-data={}", train.display()).as_str(),
        format!("--val-data={}", val.display()).as_str(),
    ]);
    let outcome = run_with_timeout(cmd, Duration::from_secs(180));
    let cleanup = std::fs::read_dir(&tmp)
        .map(|d| d.count())
        .unwrap_or_default();
    let _ = std::fs::remove_dir_all(&tmp);

    let (code, stderr, stdout) =
        outcome.expect("a one-step concat_train run must finish inside 180s");
    assert_eq!(
        code,
        Some(0),
        "a correct flag set must still run. stderr: {}",
        stderr.trim()
    );
    assert!(
        stdout.contains("BPB="),
        "the run must reach a measurement: {}",
        stdout.trim()
    );
    assert!(
        !stdout.contains("BPB=unmeasured"),
        "the run measured nothing: {}",
        stdout.trim()
    );
    assert_eq!(cleanup, 0, "the run wrote {cleanup} file(s) into its cwd");
}
