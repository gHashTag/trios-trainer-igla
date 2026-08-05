//! What a trainer binary is allowed to say about its own run.
//!
//! Three separate silent substitutions are covered here, all of the same
//! family: the binary produced a plausible, complete-looking record for a run
//! that did not happen the way the record says.
//!
//! 1. A KNOWN flag with an unparseable value. `--seed=oops` used to be
//!    `arg_or("seed", "42").parse().unwrap_or(42)`: the run trained at seed 42,
//!    printed `seed=42`, and exited 0. The seed IS the identity of a
//!    reproducibility claim, so the value that could not be read must stop the
//!    run, not be replaced by one that can.
//!
//! 2. The results path. It was keyed on `(format, algo, seed)` -- 3 of the 7
//!    parameters that define a cell -- so two cells differing only in
//!    `dim`/`seq`/`steps`/`lr` wrote to the SAME file, and `matrix_runner` read
//!    that file back to verify the executed `lr`. Eight of those legacy names
//!    are git-TRACKED, so a probe run overwrote checked-in evidence.
//!
//! 3. The faithfulness marker. `FormatKind::is_faithful()` declared
//!    Mxfp4/Mxfp6/Mxfp8 faithful while `max_finite()` in the same file states
//!    the block scale is not modelled. An MX format is DEFINED by its shared
//!    per-32-element E8M0 scale; without it the kernel is the bare element
//!    type, so a row labelled `mxfp4` names a format that was not executed.
//!
//! Subprocess tests, because the exit status and the file on disk are the
//! observables. Each run gets its own working directory, so no test can reach
//! the git-tracked files under `.trinity/results/`.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// `cpu_train`'s exit code for an argument it understands but cannot use.
const EXIT_BAD_ARGS: i32 = 4;

struct Scratch {
    path: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "trios_trainer_report_truth_{}_{}",
            name,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create scratch dir");
        Self { path }
    }

    fn results_files(&self) -> Vec<PathBuf> {
        let Ok(entries) = fs::read_dir(self.path.join(".trinity/results")) else {
            return Vec::new();
        };
        let mut found: Vec<PathBuf> = entries
            .map(|e| e.expect("dir entry").path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        found.sort();
        found
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn corpus(scratch: &Scratch, name: &str) -> PathBuf {
    let path = scratch.path.join(name);
    let bytes: Vec<u8> = (0..20_000usize).map(|i| b'a' + (i % 16) as u8).collect();
    fs::write(&path, &bytes).expect("write corpus fixture");
    path
}

fn cpu_train(scratch: &Scratch) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_cpu_train"));
    cmd.current_dir(&scratch.path);
    cmd.env_remove("TRIOS_FORMAT_TYPE");
    cmd.env_remove("TRIOS_ALGO_TYPE");
    cmd.env_remove("TRIOS_TRAIN_PATH");
    cmd.env_remove("TRIOS_VAL_PATH");
    cmd.env_remove("TRIOS_RESULTS_DIR");
    cmd
}

/// The confirmed defect: `--seed=oops --lr=0,001` printed `seed=42 lr=0.003`
/// and exited 0.
#[test]
fn an_unparseable_flag_value_refuses_and_writes_nothing() {
    for (flag, bad) in [
        ("seed", "oops"),
        ("lr", "0,001"),
        ("steps", "5.5"),
        ("dim", "-16"),
        ("seq", "many"),
        ("vocab", ""),
    ] {
        let scratch = Scratch::new(&format!("badflag_{flag}"));
        let train = corpus(&scratch, "train.txt");
        let val = corpus(&scratch, "val.txt");

        let out = cpu_train(&scratch)
            .args([
                format!("--{flag}={bad}"),
                "--steps=1".to_string(),
                "--dim=16".to_string(),
                "--vocab=64".to_string(),
                "--seq=32".to_string(),
                "--algo=adamw".to_string(),
                format!("--train-data={}", train.display()),
                format!("--val-data={}", val.display()),
            ])
            .output()
            .expect("spawn cpu_train");
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();

        assert_eq!(
            out.status.code(),
            Some(EXIT_BAD_ARGS),
            "--{flag}={bad} must be refused, not defaulted. stdout=\n{stdout}\n\
             stderr=\n{stderr}"
        );
        assert!(
            stderr.contains("UNPARSEABLE ARGUMENT"),
            "the refusal must name itself: stderr=\n{stderr}"
        );
        assert!(
            stderr.contains(&format!("--{flag}")),
            "the refusal must name the flag: stderr=\n{stderr}"
        );
        assert!(
            stderr.contains(bad) || bad.is_empty(),
            "the refusal must quote the text it could not use: stderr=\n{stderr}"
        );
        assert!(
            !stdout.contains("Final BPB"),
            "a refused invocation trains nothing: stdout=\n{stdout}"
        );
        assert_eq!(
            scratch.results_files(),
            Vec::<PathBuf>::new(),
            "and it writes no results file"
        );
    }
}

/// Two cells that differ ONLY in a parameter the old filename ignored must not
/// resolve to the same file. The old name was keyed on (format, algo, seed),
/// so `--lr=0.003` and `--lr=0.03` overwrote each other and the second run's
/// reader saw the first run's numbers.
#[test]
fn two_cells_differing_only_in_lr_do_not_share_a_results_file() {
    let scratch = Scratch::new("lr_collision");
    let train = corpus(&scratch, "train.txt");
    let val = corpus(&scratch, "val.txt");

    for lr in ["0.003", "0.03"] {
        let out = cpu_train(&scratch)
            .args([
                "--seed=47".to_string(),
                "--steps=1".to_string(),
                "--dim=16".to_string(),
                "--vocab=64".to_string(),
                "--seq=32".to_string(),
                "--algo=adamw".to_string(),
                format!("--lr={lr}"),
                format!("--train-data={}", train.display()),
                format!("--val-data={}", val.display()),
            ])
            .output()
            .expect("spawn cpu_train");
        assert!(
            out.status.success(),
            "lr={lr} run must succeed: stderr=\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    let files = scratch.results_files();
    assert_eq!(
        files.len(),
        2,
        "two lrs are two cells and must leave two files, found {files:?}"
    );
    for (path, lr) in files.iter().zip(["0.003", "0.03"]) {
        let json: serde_json::Value =
            serde_json::from_slice(&fs::read(path).expect("read results")).expect("parse results");
        let recorded = json["lr"].as_f64().expect("lr key");
        let expected: f64 = lr.parse().expect("fixture lr parses");
        assert!(
            (recorded - expected).abs() < 1e-9,
            "the file named for lr={lr} must record lr={expected}, got {recorded} \
             ({path:?})"
        );
    }
}

/// The binary states where it wrote, and that statement is the handshake
/// `matrix_runner` relies on instead of reconstructing a path.
#[test]
fn the_announced_results_path_is_the_file_that_exists() {
    let scratch = Scratch::new("announce");
    let train = corpus(&scratch, "train.txt");
    let val = corpus(&scratch, "val.txt");

    let out = cpu_train(&scratch)
        .args([
            "--seed=47".to_string(),
            "--steps=1".to_string(),
            "--dim=16".to_string(),
            "--vocab=64".to_string(),
            "--seq=32".to_string(),
            "--algo=adamw".to_string(),
            "--lr=0.003".to_string(),
            format!("--train-data={}", train.display()),
            format!("--val-data={}", val.display()),
        ])
        .output()
        .expect("spawn cpu_train");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();

    let announced = stdout
        .lines()
        .filter_map(|l| l.strip_prefix("Results: "))
        .next_back()
        .expect("cpu_train must announce its results path")
        .trim()
        .to_string();
    let announced_abs = scratch.path.join(&announced);
    assert!(
        announced_abs.exists(),
        "the announced path {announced:?} must be the file that exists; \
         on disk: {:?}",
        scratch.results_files()
    );

    // The legacy three-parameter spelling must be unreachable: it is the name
    // eight git-tracked evidence files carry.
    assert!(
        !announced.ends_with("cpu_train_f32_adamw_seed47.json"),
        "the legacy (format, algo, seed) name must not come back: {announced}"
    );
    for key in ["dim16", "seq32", "steps1", "lr0.003"] {
        assert!(
            announced.contains(key),
            "the run's identity must be in the name; missing {key} in {announced}"
        );
    }
}

/// The caller can redirect the results file out of the repository entirely.
#[test]
fn trios_results_dir_redirects_the_results_file() {
    let scratch = Scratch::new("resultsdir");
    let train = corpus(&scratch, "train.txt");
    let val = corpus(&scratch, "val.txt");
    let elsewhere = scratch.path.join("elsewhere/nested");

    let out = cpu_train(&scratch)
        .env("TRIOS_RESULTS_DIR", &elsewhere)
        .args([
            "--seed=47".to_string(),
            "--steps=1".to_string(),
            "--dim=16".to_string(),
            "--vocab=64".to_string(),
            "--seq=32".to_string(),
            "--algo=adamw".to_string(),
            "--lr=0.003".to_string(),
            format!("--train-data={}", train.display()),
            format!("--val-data={}", val.display()),
        ])
        .output()
        .expect("spawn cpu_train");
    assert!(
        out.status.success(),
        "stderr=\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let written: Vec<PathBuf> = fs::read_dir(&elsewhere)
        .expect("the redirected directory must be created")
        .map(|e| e.expect("dir entry").path())
        .collect();
    assert_eq!(
        written.len(),
        1,
        "the results file must land in TRIOS_RESULTS_DIR, found {written:?}"
    );
    assert_eq!(
        scratch.results_files(),
        Vec::<PathBuf>::new(),
        "and nothing may be left in the default location"
    );
}

/// The marker built to stop mislabelling must not vouch for a format that was
/// not executed. `matrix_runner`'s own guard reads `is_faithful()`, so this is
/// the difference between publishing an mxfp4 row and refusing it.
#[test]
fn mxfp_is_not_declared_faithful() {
    use trios_trainer::fake_quant::FormatKind;

    for fmt in [FormatKind::Mxfp4, FormatKind::Mxfp6, FormatKind::Mxfp8] {
        assert!(
            !fmt.is_faithful(),
            "{fmt:?} applies only the element mantissa mask; the shared \
             per-32-element E8M0 block scale that DEFINES an MX format is not \
             modelled (see max_finite() in src/fake_quant.rs), so a row labelled \
             {fmt:?} names a format that was not executed"
        );
    }

    // The element types themselves ARE faithful mantissa masks -- the defect is
    // the MX label, not the arithmetic, and nothing about the kernels changed.
    for fmt in [
        FormatKind::Fp4E2M1,
        FormatKind::Fp6E3M2,
        FormatKind::Fp8E4M3,
    ] {
        assert!(fmt.is_faithful(), "{fmt:?} is a plain element type");
    }
}
