//! The declaration half of a record must not be forgeable in silence.
//!
//! # What these tests are about
//!
//! On 2026-08-05 the headline cross-architecture record
//! (`evidence/xarch-run-30767491098/12000.json`, produced by GitHub Actions run
//! 30767491098 on x86_64 Linux) was copied to a scratch directory and its
//! SIDECAR rewritten to claim `aarch64` / `macos`, a rustc string that never
//! existed, and `final_val_bpb = 1.5492` - the exact number this project
//! publicly retracted. The container bytes were left untouched. Both of this
//! repository's verifiers passed the forgery:
//!
//! ```text
//! $ python3 interop/triosckp_reader.py <tmp>/12000.bin --sidecar <tmp>/12000.json
//! RESULT PASS  container ok, sha256 bb14ab18...
//! $ ./target/release/ckpt_replay --record <tmp>/12000.json --integrity-only
//! INTEGRITY OK: ".../12000.json" (schema trios-checkpoint-record/4)
//! ```
//!
//! Neither was wrong about the bytes. Both were silent about the half of the
//! record that a conformity scheme rests on, because the container digest
//! covers the WEIGHTS and nothing at all covers the platform block, the corpus
//! and trainer digests, `git_sha`, `steps_total`, `eval_every` or
//! `final_val_bpb`.
//!
//! # What is asserted here, and what is not
//!
//! The seal is NOT a signature and these tests do not pretend it is. Test (i)
//! asserts the thing that stays true and must keep staying true: sidecar
//! tampering does not move the container digest, so the old verifiers are not
//! broken and their PASS is not being redefined. Tests (ii) assert that each of
//! the four forged fields moves the SEAL. Tests (iii) assert that
//! `--expect-provenance-seal` turns a published digest into a check: exit 0 on
//! the genuine record, non-zero on every forgery.
//!
//! Test (iv) is the round-7 correction and is a different attack: it does not
//! touch the declaration at all, it swaps the CONTAINER the declaration points
//! at. That was free while `sha256` and `bytes` sat outside the sealed set, and
//! it is the sharper attack, because the forger keeps an honest declaration and
//! simply attaches it to weights it did not describe. Test (v) closes the
//! adjacent hole where the two shipped verifiers disagreed about the record's
//! own `bytes` field, with the Rust side reporting the pass.
//!
//! An adversary who controls the record AND the channel the seal is published
//! on defeats all of this. Closing that needs key material this project does
//! not have; see docs/PROVENANCE-BINDING.md.
//!
//! Nothing here writes inside the repository: every case stages copies in a
//! `tempfile::tempdir()` and the evidence tree is only ever read.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use trios_trainer::provenance_seal::provenance_seal;

/// The headline record: the x86_64 Linux side of the published cross-arch pair.
const RECORD_DIR: &str = "evidence/xarch-run-30767491098";
const RECORD_NAME: &str = "12000.json";
const ARTIFACT_NAME: &str = "12000.bin";

/// The digest the container has carried since it was published. Hard-coded
/// rather than recomputed, so a test that starts hashing the wrong file fails
/// instead of agreeing with itself.
const PUBLISHED_BIN_SHA256: &str =
    "bb14ab18f2c8e7a9a4c19f452471018f3a72cc18765175db44858c2c4e5c03f3";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn genuine_record() -> Value {
    let text = std::fs::read_to_string(repo_root().join(RECORD_DIR).join(RECORD_NAME))
        .expect("read the published record");
    serde_json::from_str(&text).expect("the published record is JSON")
}

/// Copy the record and its published sibling into a scratch directory.
fn stage(dir: &Path) -> PathBuf {
    let src = repo_root().join(RECORD_DIR);
    let record = dir.join(RECORD_NAME);
    std::fs::copy(src.join(RECORD_NAME), &record).expect("copy record");
    std::fs::copy(src.join(ARTIFACT_NAME), dir.join(ARTIFACT_NAME)).expect("copy artifact");
    record
}

/// Write a record to `path` as JSON.
fn write_record(path: &Path, record: &Value) {
    std::fs::write(
        path,
        serde_json::to_string_pretty(record).expect("serialize"),
    )
    .expect("write record");
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Run the shipped binary and return `(exit code, stdout + stderr)`.
fn run(args: &[&std::ffi::OsStr]) -> (Option<i32>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_ckpt_replay"))
        .args(args)
        .output()
        .expect("run ckpt_replay");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.code(), text)
}

/// The four forgeries, by name, each a mutation of the genuine record.
///
/// `final_val_bpb` is set to 1.5492 on purpose: that is the retracted figure,
/// so this fixture is the exact attack the round-6 scout ran and not a
/// hypothetical one.
fn forgeries() -> Vec<(&'static str, Value)> {
    let mut out = Vec::new();

    let mut arch = genuine_record();
    arch["platform"]["arch"] = Value::from("aarch64");
    out.push(("platform.arch x86_64 -> aarch64", arch));

    let mut os = genuine_record();
    os["platform"]["os"] = Value::from("macos");
    out.push(("platform.os linux -> macos", os));

    let mut toolchain = genuine_record();
    toolchain["platform"]["toolchain"] = Value::from("rustc 1.99.9 (deadbeef0 2026-08-01)");
    out.push((
        "platform.toolchain -> a rustc that never existed",
        toolchain,
    ));

    let mut bpb = genuine_record();
    bpb["final_val_bpb"] = Value::from(1.5492_f64);
    out.push(("final_val_bpb -> the retracted 1.5492", bpb));

    out
}

/// (i) The control, and the boundary of the claim. Rewriting the sidecar does
/// NOT move the container digest - which is why the old verifiers passed the
/// forgery, and why nothing here may be read as strengthening THEM.
#[test]
fn sidecar_tampering_does_not_move_the_container_digest() {
    let dir = tempfile::tempdir().expect("tempdir");
    let record = stage(dir.path());
    let artifact = dir.path().join(ARTIFACT_NAME);
    let before = sha256_hex(&std::fs::read(&artifact).expect("read artifact"));
    assert_eq!(
        before, PUBLISHED_BIN_SHA256,
        "fixture broken: the staged artifact is not the published one"
    );

    for (name, forged) in forgeries() {
        write_record(&record, &forged);
        let after = sha256_hex(&std::fs::read(&artifact).expect("read artifact"));
        assert_eq!(
            after, PUBLISHED_BIN_SHA256,
            "forging {name} must not touch the container bytes"
        );
    }
}

/// (ii) One assertion per forged field. Written as four tests, not one loop
/// over four fixtures, so a failure names the field that stopped mattering.
#[test]
fn forging_the_architecture_changes_the_seal() {
    let genuine = provenance_seal(&genuine_record()).expect("seal");
    let mut forged = genuine_record();
    forged["platform"]["arch"] = Value::from("aarch64");
    assert_ne!(
        genuine,
        provenance_seal(&forged).expect("seal"),
        "an x86_64 record relabelled aarch64 must not seal to the same digest"
    );
}

#[test]
fn forging_the_os_changes_the_seal() {
    let genuine = provenance_seal(&genuine_record()).expect("seal");
    let mut forged = genuine_record();
    forged["platform"]["os"] = Value::from("macos");
    assert_ne!(genuine, provenance_seal(&forged).expect("seal"));
}

#[test]
fn forging_the_toolchain_changes_the_seal() {
    let genuine = provenance_seal(&genuine_record()).expect("seal");
    let mut forged = genuine_record();
    forged["platform"]["toolchain"] = Value::from("rustc 1.99.9 (deadbeef0 2026-08-01)");
    assert_ne!(genuine, provenance_seal(&forged).expect("seal"));
}

#[test]
fn forging_the_metric_changes_the_seal() {
    let genuine = provenance_seal(&genuine_record()).expect("seal");
    let mut forged = genuine_record();
    forged["final_val_bpb"] = Value::from(1.5492_f64);
    assert_ne!(
        genuine,
        provenance_seal(&forged).expect("seal"),
        "the retracted 1.5492 must not seal to the genuine record's digest"
    );
}

/// Deletion is an edit too. A verifier that skipped absent keys would let a
/// forger remove `git_sha` for free.
#[test]
fn deleting_a_declared_field_changes_the_seal() {
    let genuine = provenance_seal(&genuine_record()).expect("seal");
    let mut stripped = genuine_record();
    stripped
        .as_object_mut()
        .expect("record is an object")
        .remove("git_sha");
    assert_ne!(genuine, provenance_seal(&stripped).expect("seal"));
}

/// (iii.a) The genuine record under its own seal: exit 0, and the verdict says
/// the declaration was CHECKED rather than quoted.
#[test]
fn expected_seal_passes_on_the_genuine_record() {
    let dir = tempfile::tempdir().expect("tempdir");
    let record = stage(dir.path());
    let seal = provenance_seal(&genuine_record()).expect("seal");

    let (code, text) = run(&[
        "--record".as_ref(),
        record.as_os_str(),
        "--integrity-only".as_ref(),
        "--expect-provenance-seal".as_ref(),
        seal.as_ref(),
    ]);
    assert_eq!(code, Some(0), "expected exit 0, got {code:?}:\n{text}");
    assert!(
        text.contains("INTEGRITY OK"),
        "expected INTEGRITY OK:\n{text}"
    );
    assert!(
        text.contains("DECLARATION SEAL MATCHED"),
        "a matched seal must be stated, not implied:\n{text}"
    );
    assert!(
        !text.contains("UNAUTHENTICATED"),
        "an authenticated declaration must not also be called unauthenticated:\n{text}"
    );
}

/// (iii.b) Every forgery, graded against the GENUINE seal: non-zero, and the
/// message names both digests so the auditor is not told to trust a verdict.
#[test]
fn expected_seal_fails_on_every_forgery() {
    let genuine_seal = provenance_seal(&genuine_record()).expect("seal");

    for (name, forged) in forgeries() {
        let dir = tempfile::tempdir().expect("tempdir");
        let record = stage(dir.path());
        write_record(&record, &forged);
        let forged_seal = provenance_seal(&forged).expect("seal");

        let (code, text) = run(&[
            "--record".as_ref(),
            record.as_os_str(),
            "--integrity-only".as_ref(),
            "--expect-provenance-seal".as_ref(),
            genuine_seal.as_ref(),
        ]);
        assert_ne!(code, Some(0), "forgery {name} exited 0:\n{text}");
        assert!(
            text.contains("SEAL MISMATCH"),
            "forgery {name} must be named SEAL MISMATCH:\n{text}"
        );
        assert!(
            text.contains(&genuine_seal) && text.contains(&forged_seal),
            "forgery {name} must name BOTH digests:\n{text}"
        );
        assert!(
            !text.contains("INTEGRITY OK"),
            "forgery {name} must not also be graded as integrity-ok:\n{text}"
        );
    }
}

/// (iii.c) THE REGRESSION. Without `--expect-provenance-seal`, the forgery
/// still passes the byte check - it is the same bytes - but `INTEGRITY OK` may
/// never again appear alone. That line is the whole fix: the old output gave a
/// reader no way to know the declaration had not been checked.
#[test]
fn integrity_ok_never_appears_without_the_unauthenticated_line() {
    let dir = tempfile::tempdir().expect("tempdir");
    let record = stage(dir.path());
    write_record(&record, &forgeries()[0].1);

    let (code, text) = run(&[
        "--record".as_ref(),
        record.as_os_str(),
        "--integrity-only".as_ref(),
    ]);
    assert_eq!(
        code,
        Some(0),
        "the bytes are untouched, so the byte verdict must still be a pass:\n{text}"
    );
    assert!(text.contains("INTEGRITY OK"), "{text}");
    assert!(
        text.contains("UNAUTHENTICATED DECLARATION"),
        "INTEGRITY OK must never be printed alone again:\n{text}"
    );
    for field in [
        "platform.arch",
        "platform.os",
        "platform.toolchain",
        "final_val_bpb",
        "git_sha",
        "steps_total",
        "eval_every",
        "source_sha256",
    ] {
        assert!(
            text.contains(field),
            "the unauthenticated line must NAME {field}:\n{text}"
        );
    }
}

/// (iv) THE CONTAINER SWAP. Every case above tampers with a field that was
/// already sealed. This one leaves the declaration alone and swaps the
/// ARTIFACT: it points the record at a different `.bin` by rewriting `sha256`
/// and `bytes`, which is what an adversary does when the honest declaration is
/// exactly the one they want.
///
/// Before `sha256` and `bytes` entered the sealed set this was free. The
/// x86_64 Linux declaration could be attached to any other container, both
/// shipped verifiers passed - the digest in the record really did describe the
/// bytes beside it - and the seal still equalled the digest published for the
/// Linux run, because the seal named no artifact. That forgery produces the
/// claim this project is audited on, cross-architecture bit identity, out of an
/// authenticated declaration.
///
/// The substitute container is synthesised here rather than taken from
/// `evidence/`, so the test states its own premise and does not depend on which
/// other artifacts happen to be published.
#[test]
fn swapping_the_container_changes_the_seal() {
    let dir = tempfile::tempdir().expect("tempdir");
    let record_path = stage(dir.path());
    let genuine = genuine_record();
    let genuine_seal = provenance_seal(&genuine).expect("seal");

    // A different container of the SAME length: length alone was never the
    // thing that made the swap detectable, and a fixture that changed it would
    // let this test pass for the wrong reason.
    let original = std::fs::read(dir.path().join(ARTIFACT_NAME)).expect("read artifact");
    assert_eq!(
        sha256_hex(&original),
        PUBLISHED_BIN_SHA256,
        "fixture broken: the staged artifact is not the published one"
    );
    let mut substitute = original.clone();
    let last = substitute.len() - 1;
    substitute[last] ^= 0xff;
    assert_eq!(substitute.len(), original.len());
    assert_ne!(sha256_hex(&substitute), PUBLISHED_BIN_SHA256);

    let other = dir.path().join("substitute.bin");
    std::fs::write(&other, &substitute).expect("write substitute");

    // The declaration is untouched. Only the two fields that name the artifact
    // are rewritten, and they are rewritten HONESTLY: they describe the file
    // they now point at.
    let mut swapped = genuine.clone();
    swapped["sha256"] = Value::from(sha256_hex(&substitute));
    swapped["bytes"] = Value::from(substitute.len() as u64);
    let swapped_seal = provenance_seal(&swapped).expect("seal");

    assert_ne!(
        genuine_seal, swapped_seal,
        "a record pointed at a different container must not seal to the digest \
         published for the original one"
    );

    // And the same thing end to end: the shipped binary, handed the swapped
    // record and the published seal, must refuse it.
    write_record(&record_path, &swapped);
    std::fs::copy(&other, dir.path().join(ARTIFACT_NAME)).expect("install substitute");
    let (code, text) = run(&[
        "--record".as_ref(),
        record_path.as_os_str(),
        "--integrity-only".as_ref(),
        "--expect-provenance-seal".as_ref(),
        genuine_seal.as_ref(),
    ]);
    assert_ne!(code, Some(0), "the container swap exited 0:\n{text}");
    assert!(
        text.contains("SEAL MISMATCH"),
        "the container swap must be named SEAL MISMATCH:\n{text}"
    );
    assert!(
        !text.contains("INTEGRITY OK"),
        "the container swap must not also be graded integrity-ok:\n{text}"
    );
}

/// (v) The record must not be able to contradict itself about the length of the
/// file it publishes. `interop/triosckp_reader.py` has always compared `bytes`
/// against the container length; `--integrity-only` did not, so a record edited
/// to say `"bytes": 1` printed INTEGRITY OK in Rust and SIDECAR_MISMATCH in
/// Python. The verifier that reported success was the one that was wrong.
#[test]
fn a_record_lying_about_its_length_is_artifact_altered() {
    let dir = tempfile::tempdir().expect("tempdir");
    let record_path = stage(dir.path());
    let mut lying = genuine_record();
    lying["bytes"] = Value::from(1_u64);
    write_record(&record_path, &lying);

    let (code, text) = run(&[
        "--record".as_ref(),
        record_path.as_os_str(),
        "--integrity-only".as_ref(),
    ]);
    assert_ne!(code, Some(0), "a false length exited 0:\n{text}");
    assert!(
        text.contains("ARTIFACT ALTERED"),
        "a false length must be named ARTIFACT ALTERED:\n{text}"
    );
    assert!(
        !text.contains("INTEGRITY OK"),
        "a false length must not be graded integrity-ok:\n{text}"
    );
}

/// The seal is stable. A digest that moved between runs of the same binary on
/// the same record would be unpublishable, and every claim above would be
/// unfalsifiable.
#[test]
fn the_seal_is_deterministic_and_matches_the_cli() {
    let record = genuine_record();
    let first = provenance_seal(&record).expect("seal");
    let second = provenance_seal(&record).expect("seal");
    assert_eq!(first, second, "the seal must not depend on hash-map order");

    let path = repo_root().join(RECORD_DIR).join(RECORD_NAME);
    let (code, text) = run(&[
        "--record".as_ref(),
        path.as_os_str(),
        "--provenance-seal".as_ref(),
    ]);
    assert_eq!(code, Some(0), "{text}");
    assert!(
        text.contains(&first),
        "the CLI must print the same seal the library computes:\n{text}"
    );
}
