//! tests/ckpt_replay_integrity_scope.rs - `--integrity-only` must grade the file
//! it was pointed at, and nothing else.
//!
//! # The defect these tests exist to keep closed
//!
//! `locate_published_artifact` in `src/bin/ckpt_replay.rs` used to end with a
//! third resolution arm - `if let Some(p) = as_str(root, "path")` - that fell
//! back to the path the record states when no `.bin` sat beside the record.
//! `integrity_verdict` is that function's only caller, so in integrity mode the
//! fallback decided the verdict.
//!
//! That made the mode blind to the one kind of rot it is deployed to catch. A
//! published `.bin` DELETED from the evidence tree, under a record that stays
//! behind, still names an absolute `path` on the training machine - and on the
//! training machine, or any checkout of this repository, that path is a whole,
//! unaltered copy. The tool hashed the copy, printed `INTEGRITY OK`, exit 0,
//! about a file that no longer exists at the place under audit.
//!
//! `.github/workflows/ckpt-replay-audit.yml` keys pass/fail on the exit code and
//! the verdict name, so a deleted published artifact passed the audit.
//!
//! Case (b) below is that regression, and it FAILS against the unpatched
//! binary: it guards the removal of the recorded-`path` arm from
//! `locate_published_artifact`, and the `PUBLISHED ARTIFACT MISSING` verdict
//! that `integrity_verdict` prints in its place.
//!
//! # Why the record is copied out of the tree rather than mutated in it
//!
//! `evidence/heldout/r6-heldout-test/` is committed evidence. Every case here
//! works on a copy in a `tempfile::TempDir` (repo rule: no `/tmp` literals), so
//! a failing test can never leave the published tree altered - which is the
//! exact condition case (c) is written to detect.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The record under audit: a real, committed, schema/8 checkpoint record whose
/// sibling `.bin` hashes to the digest it publishes.
const RECORD_DIR: &str = "evidence/heldout/r6-heldout-test";
const RECORD_NAME: &str = "12000.json";
const ARTIFACT_NAME: &str = "12000.bin";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Absolute path of the artifact that is still whole inside the repository.
///
/// This is what the recorded `path` points at, and therefore what an integrity
/// check must NOT reach for when grading a copy somewhere else.
fn in_repo_artifact() -> PathBuf {
    repo_root().join(RECORD_DIR).join(ARTIFACT_NAME)
}

/// Copy the record and its sibling artifact into a scratch directory.
fn stage(dir: &Path) -> PathBuf {
    let src = repo_root().join(RECORD_DIR);
    let record = dir.join(RECORD_NAME);
    std::fs::copy(src.join(RECORD_NAME), &record).expect("copy record");
    std::fs::copy(src.join(ARTIFACT_NAME), dir.join(ARTIFACT_NAME)).expect("copy artifact");
    record
}

/// Run the shipped binary the way the audit workflow runs it, and return
/// `(exit code, stdout + stderr)`.
fn grade(record: &Path) -> (Option<i32>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_ckpt_replay"))
        .arg("--record")
        .arg(record)
        .arg("--integrity-only")
        .output()
        .expect("run ckpt_replay");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.code(), text)
}

/// (a) The control. Record plus its published sibling, copied out of the
/// evidence tree: the bytes still hash to the record, so the verdict is
/// `INTEGRITY OK` and the exit code is 0.
///
/// Without this case, a fix that simply made integrity mode fail everything
/// would satisfy (b) and (c).
#[test]
fn sibling_artifact_present_is_integrity_ok() {
    let dir = tempfile::tempdir().expect("tempdir");
    let record = stage(dir.path());

    let (code, text) = grade(&record);
    assert_eq!(code, Some(0), "expected exit 0, got {code:?}:\n{text}");
    assert!(
        text.contains("INTEGRITY OK"),
        "expected INTEGRITY OK:\n{text}"
    );
    // The rule that found the file is part of the verdict, not a footnote.
    assert!(
        text.contains("resolved: published beside the record"),
        "expected the resolution rule to be named:\n{text}"
    );
}

/// (b) THE REGRESSION. The published artifact is deleted and the record's own
/// `path` is rewritten to the copy that is still whole inside the repository -
/// which is what a real deletion looks like, since the committed record already
/// carries an absolute path into this checkout.
///
/// Against the unpatched binary this prints `INTEGRITY OK`, exit 0, having
/// hashed `evidence/heldout/r6-heldout-test/12000.bin` while claiming to grade a
/// directory where no `.bin` exists at all.
#[test]
fn deleted_published_artifact_is_not_graded_through_the_recorded_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let record = stage(dir.path());

    // The deletion under audit.
    std::fs::remove_file(dir.path().join(ARTIFACT_NAME)).expect("remove published artifact");

    // Point `path` at the intact in-repo copy, explicitly, so the test does not
    // depend on the absolute path that happens to be committed in the record.
    let decoy = in_repo_artifact();
    assert!(
        decoy.is_file(),
        "fixture broken: {decoy:?} must exist for this test to mean anything"
    );
    let text = std::fs::read_to_string(&record).expect("read record");
    let mut root: serde_json::Value = serde_json::from_str(&text).expect("record is JSON");
    root["path"] = serde_json::Value::String(decoy.display().to_string());
    std::fs::write(
        &record,
        serde_json::to_string_pretty(&root).expect("re-serialise"),
    )
    .expect("write record");

    let (code, out) = grade(&record);

    assert!(
        !out.contains("INTEGRITY OK"),
        "a deleted published artifact must NOT pass; the tool followed the \
         recorded `path` to {decoy:?}:\n{out}"
    );
    assert_ne!(code, Some(0), "expected a non-zero exit, got 0:\n{out}");
    assert_eq!(code, Some(1), "expected EXIT_MISMATCH (1):\n{out}");
    assert!(
        out.contains("PUBLISHED ARTIFACT MISSING"),
        "expected the verdict to name the defect:\n{out}"
    );
    assert!(
        out.contains("was NOT followed"),
        "the verdict must say the recorded path was not followed:\n{out}"
    );
}

/// (c) The sibling is present and one byte of it has been flipped. This is the
/// evidence-rot case the mode was built for, and it must stay distinguishable
/// from (b): a file that is THERE and WRONG is a different fact about the
/// evidence than a file that is GONE.
#[test]
fn altered_sibling_artifact_is_artifact_altered() {
    let dir = tempfile::tempdir().expect("tempdir");
    let record = stage(dir.path());

    let artifact = dir.path().join(ARTIFACT_NAME);
    let mut bytes = std::fs::read(&artifact).expect("read artifact");
    assert!(!bytes.is_empty(), "fixture broken: artifact is empty");
    // Flip one bit in the last byte: same length, different digest. The length
    // is preserved on purpose, so the verdict cannot come from a size check.
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&artifact, &bytes).expect("write artifact");

    let (code, out) = grade(&record);
    assert_eq!(code, Some(1), "expected exit 1, got {code:?}:\n{out}");
    assert!(
        out.contains("ARTIFACT ALTERED"),
        "expected ARTIFACT ALTERED:\n{out}"
    );
    assert!(
        !out.contains("INTEGRITY OK"),
        "an altered artifact must not pass:\n{out}"
    );
}
