//! THE STAGE TRACE MUST NOT MOVE THE THING IT OBSERVES.
//!
//! `TRIOS_TRACE_STAGE=1` makes the trainer hash every intermediate tensor of
//! one forward/backward pair at step 1 and print the hashes to stderr. Its
//! whole purpose is to name the first tensor that differs across the ISA
//! boundary. If switching it on could itself move one byte of the arithmetic,
//! every stage hash it produces would describe an execution that is not the
//! evidenced one, and the "first divergence" it reported could be an artefact
//! of the instrument. That is the defect class this repository exists to
//! remove: a measurement indistinguishable from the thing it measures.
//!
//! Inertness is therefore MEASURED here, not asserted in a comment. The test
//! trains ten steps twice - once with the variable unset, once with it set to
//! 1 - into two separate directories, and requires:
//!
//!   1. the two `10.bin` files to be byte-identical to EACH OTHER. This is the
//!      claim in its host-independent form and it holds on any machine.
//!   2. on the two hosts where a constant has actually been measured, both
//!      files to equal that constant. `local_isa_probe.py` measured
//!      `efef1cba...` on aarch64 macOS and `5913542e...` on x86_64 macOS under
//!      Rosetta 2, WITHOUT the trace. Reproducing them here proves the traced
//!      run is the same run as the evidenced one, not merely self-consistent.
//!   3. the traced run to have actually emitted `TRACE ` lines, and the
//!      untraced run to have emitted none. Without (3) a trace that silently
//!      did nothing would pass (1) and (2) perfectly, and the test would be
//!      certifying the inertness of a no-op.
//!
//! WHICH ARM THIS FILE CHECKS. Only the host it runs on. The x86_64 constant
//! below is asserted when this test is compiled for x86_64 macOS; the CROSS-arm
//! comparison - both binaries built from one tree and run on one host - is
//! `scripts/stage_trace_isa.py`, which checks the same two constants in situ
//! before it reads any stage verdict.
//!
//! WHY THE CONSTANTS ARE KEYED ON (os, arch) AND NOT arch ALONE. A third
//! measurement exists for the same declared inputs: `a32e9b2a...` on x86_64
//! Linux with glibc 2.39 (CI run 31004703001). Two of these three share a
//! declared architecture and differ anyway. A constant selected by
//! `target_arch` alone would therefore be wrong on x86_64 Linux, and the
//! failure would look like a defect in the trace rather than a defect in this
//! file's indexing. On any (os, arch) with no measured constant this test
//! still enforces (1) and (3), which is the actual inertness claim; it just
//! cannot also confirm the run is the evidenced one.
//!
//! A FAILURE HERE IS THE GUARD WORKING. It means the trace is no longer
//! read-only. The fix is to make the trace inert again - never to relax the
//! constant, which is a measurement and not a target.

use std::path::Path;
use std::process::Command;

use sha2::{Digest, Sha256};

/// The exact flags of `evidence/xarch-local-isa/probe.json`. The constants
/// below are only constants OF THIS RECIPE; changing a flag here invalidates
/// them.
const FLAGS: &[&str] = &[
    "--seed",
    "47",
    "--steps",
    "10",
    "--hidden",
    "384",
    "--attn-layers",
    "2",
    "--eval-every",
    "1000",
    "--lr",
    "0.003",
    "--optimizer",
    "adamw",
    "--train-data",
    "data/tiny_shakespeare.txt",
    "--val-data",
    "data/tiny_shakespeare_val.txt",
];

/// aarch64 macOS, native. Measured by `scripts/local_isa_probe.py`, n=2,
/// within-arm control passing, with NO trace compiled into the run.
const PIN_MACOS_AARCH64: &str = "efef1cba128a8c96e23124d1f139f73c11f8e00261b6148fcfb8cc427aaa0cac";
/// x86_64 macOS, under Rosetta 2 binary translation - not native x86_64
/// silicon. Same probe, same session, same corpus.
const PIN_MACOS_X86_64: &str = "5913542eb613abc3780ac959a0262af059f7b11ea6ed61b23bd0d62b9c8897ab";

/// The measured step-10 checkpoint for THIS host, or `None` where none has been
/// measured.
fn host_pin() -> Option<(&'static str, &'static str)> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => Some(("macos/aarch64", PIN_MACOS_AARCH64)),
        ("macos", "x86_64") => Some(("macos/x86_64 (Rosetta 2)", PIN_MACOS_X86_64)),
        _ => None,
    }
}

fn sha256_file(path: &Path) -> String {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("checkpoint {} is unreadable: {e}", path.display()));
    let mut h = Sha256::new();
    h.update(&bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// One ten-step run. `trace` selects whether `TRIOS_TRACE_STAGE=1` is set at
/// all - UNSET, not set to "0", because the default path is what an ordinary
/// run takes and that is the arm under test.
///
/// The environment is cleared and rebuilt from an allowlist rather than having
/// a denylist removed from it. A denylist has to be kept in step with every
/// variable the trainer learns to read; `env_clear` cannot fall behind.
fn train(dir: &Path, canon: &str, trace: bool) -> (String, usize) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_trios-train"));
    cmd.current_dir(env!("CARGO_MANIFEST_DIR"));
    cmd.env_clear();
    if let Ok(path) = std::env::var("PATH") {
        cmd.env("PATH", path);
    }
    if let Ok(home) = std::env::var("HOME") {
        cmd.env("HOME", home);
    }
    cmd.env("TRINITY_AUTOMIGRATE", "0")
        .env("TRIOS_CHECKPOINT_DIR", dir)
        .env("TRIOS_CANON_NAME", canon);
    if trace {
        cmd.env("TRIOS_TRACE_STAGE", "1");
    }
    cmd.args(FLAGS);

    let out = cmd.output().expect("trios-train is executable");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "the {} run exited {:?}\n--- stderr tail ---\n{}",
        canon,
        out.status.code(),
        &stderr[stderr.len().saturating_sub(3000)..]
    );

    let trace_lines = stderr.lines().filter(|l| l.starts_with("TRACE ")).count();
    let ckpt = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(dir)
        .join(canon)
        .join("10.bin");
    assert!(
        ckpt.exists(),
        "the {canon} run wrote no step-10 checkpoint at {}",
        ckpt.display()
    );
    (sha256_file(&ckpt), trace_lines)
}

#[test]
fn stage_trace_does_not_move_the_checkpoint() {
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/tiny_shakespeare.txt");
    assert!(
        corpus.exists(),
        "{} is missing. The corpus is not in git; rebuild it as \
         .github/workflows/cross-arch-repro.yml does. This test is NOT skipped \
         when the corpus is absent: a guard that quietly passes when it cannot \
         run is the failure mode it exists to catch.",
        corpus.display()
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    let off_dir = tmp.path().join("trace-off");
    let on_dir = tmp.path().join("trace-on");

    let (off_sha, off_lines) = train(&off_dir, "stage-trace-off", false);
    let (on_sha, on_lines) = train(&on_dir, "stage-trace-on", true);

    // (3) first: without it the two comparisons below could be certifying a
    // trace that never ran.
    assert_eq!(
        off_lines, 0,
        "the run with TRIOS_TRACE_STAGE unset emitted {off_lines} TRACE lines. \
         The trace is not off by default."
    );
    assert!(
        on_lines > 0,
        "the run with TRIOS_TRACE_STAGE=1 emitted no TRACE lines. There is no \
         trace to be inert about, and this test would otherwise pass by \
         comparing two identical untraced runs."
    );

    // (1) the claim, in the form that holds on every host.
    assert_eq!(
        off_sha, on_sha,
        "TRIOS_TRACE_STAGE=1 CHANGED THE ARTIFACT.\n  untraced 10.bin {off_sha}\n  \
         traced   10.bin {on_sha}\nThe trace is not read-only, so every stage \
         hash it prints describes an execution that is not the one under study."
    );

    // (2) and, where a constant has been measured, that this is that run.
    match host_pin() {
        Some((label, pin)) => {
            assert_eq!(
                off_sha, pin,
                "on {label} the UNTRACED ten-step checkpoint is {off_sha}, not \
                 the measured {pin}. Something other than the trace moved this \
                 recipe; the trace's inertness is not what failed here."
            );
            assert_eq!(
                on_sha, pin,
                "on {label} the TRACED ten-step checkpoint is {on_sha}, not the \
                 measured {pin}."
            );
        }
        None => {
            // Stated, not silent. A reader of a green run on an unpinned host
            // must know which half of the guard ran.
            eprintln!(
                "stage_trace_inert: no measured step-10 constant for {}/{}; \
                 checked trace-off == trace-on only. The pinned-constant half \
                 runs on macos/aarch64 and macos/x86_64.",
                std::env::consts::OS,
                std::env::consts::ARCH
            );
        }
    }
}
