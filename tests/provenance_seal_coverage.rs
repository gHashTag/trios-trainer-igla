//! A seal that does not move when the finding is inverted authenticates
//! nothing, and this file is the guard that says so.
//!
//! # The defect, measured 2026-08-07
//!
//! `evidence/SEALS.txt` published three lines for documents that are NOT
//! checkpoint records. They were being sealed with the checkpoint FIELD LIST -
//! 18 or 19 declared names - and they carry almost none of them:
//!
//! | document                                          | present of declared |
//! |---------------------------------------------------|---------------------|
//! | `evidence/canonical-digest/signed-zero-census.json`| 1 of 18 (`schema`)  |
//! | `evidence/stage-trace-isa/verdict.json`            | 2 of 19, no `schema`|
//! | `evidence/xarch-local-isa/probe.json`              | 3 of 19             |
//!
//! Every substantive field - `canonical_digests_equal`, `raw_digests_equal`,
//! `signed_zero_only`, `verdict`, `first_divergence`, `stages_differing`,
//! `translation` - lay outside the seal. Copies with those findings INVERTED
//! sealed to the digests already published, to the character, and the
//! workflow's own command form graded all three exit 7 SKIPPED. The header of
//! `evidence/SEALS.txt` promises that "a forgery becomes detectable by anyone
//! who read them"; for those three lines it was not.
//!
//! # The two guards
//!
//! (a) `inverting_a_published_finding_moves_the_seal` and
//!     `inverting_the_first_boolean_moves_the_seal`: for every published
//!     non-checkpoint document, a copy with one finding flipped must NOT seal
//!     to the honest document's digest, and must not seal to the published one
//!     either. The forgery is built by substituting BYTES in the file, never by
//!     re-serializing a parsed value: a re-serialized copy differs in whitespace
//!     and would move a whole-file digest for a reason that has nothing to do
//!     with the finding, which would make this guard pass while measuring
//!     formatting.
//!
//! (b) `a_degenerate_field_set_cannot_return_a_field_list_digest` and
//!     `the_floor_sits_between_the_two_measured_populations`: a document with
//!     fewer than `SEAL_MIN_PRESENT_FIELDS` present sealed names must take the
//!     whole-file path or return an error, and the floor must still lie between
//!     the two populations it was derived from.
//!
//! Nothing here writes inside the repository. Every forgery lives in memory.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use trios_trainer::provenance_seal::{
    is_checkpoint_record, provenance_seal, seal_document, sealed_field_split, SealKind,
    DEGENERATE_SEAL_REFUSAL, SEAL_MIN_PRESENT_FIELDS,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every line of `evidence/SEALS.txt`, keyed by record path.
fn published_seals() -> BTreeMap<String, String> {
    let path = repo_root().join("evidence").join("SEALS.txt");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("could not read {}: {err}", path.display()));
    let mut table = BTreeMap::new();
    for line in text.lines() {
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let (record, digest) = line.split_once('\t').unwrap_or_else(|| {
            panic!("evidence/SEALS.txt line is not <path><TAB><digest>: {line:?}")
        });
        table.insert(record.to_owned(), digest.trim().to_owned());
    }
    assert!(
        !table.is_empty(),
        "evidence/SEALS.txt publishes nothing, so every guard here would grade an empty list"
    );
    table
}

/// `(path, raw bytes)` for every published document that is NOT a checkpoint
/// record and is present in this checkout.
///
/// Presence is checked rather than assumed, and an EMPTY result is a FAILURE:
/// a guard that grades no documents passes, which is the shape of defect this
/// file exists to remove.
fn published_non_checkpoint_documents() -> Vec<(String, Vec<u8>)> {
    let mut found = Vec::new();
    let mut missing = Vec::new();
    for path in published_seals().keys() {
        let full = repo_root().join(path);
        let Ok(raw) = std::fs::read(&full) else {
            missing.push(path.clone());
            continue;
        };
        let record: serde_json::Value = serde_json::from_slice(&raw)
            .unwrap_or_else(|err| panic!("{path} is published but is not JSON: {err}"));
        if !is_checkpoint_record(&record) {
            found.push((path.clone(), raw));
        }
    }
    assert!(
        !found.is_empty(),
        "no published non-checkpoint document was found in this checkout \
         (missing: {missing:?}); this guard would grade nothing, which passes"
    );
    found
}

/// The central finding of each document this item was written about, by name.
///
/// A table of NEEDLES, not of sealed fields: it says what the document is FOR,
/// so a guard failure names the claim that stopped being protected instead of a
/// key. A listed path that exists must contain its needle - a stale fixture is
/// a failure here, never a silent skip.
const CENTRAL_FINDINGS: &[(&str, &str, &str)] = &[
    (
        "evidence/canonical-digest/signed-zero-census.json",
        "\"raw_digests_equal\": false",
        "\"raw_digests_equal\": true",
    ),
    (
        "evidence/canonical-digest/signed-zero-census.json",
        "\"canonical_digests_equal\": false",
        "\"canonical_digests_equal\": true",
    ),
    (
        "evidence/stage-trace-isa/verdict.json",
        "\"stages_differing\": 18",
        "\"stages_differing\": 0",
    ),
    (
        "evidence/stage-trace-isa/verdict.json",
        "\"verdict\": \"FIRST DIVERGENCE",
        "\"verdict\": \"NO DIVERGENCE",
    ),
    (
        "evidence/xarch-local-isa/probe.json",
        "\"verdict\": \"ISA_SUFFICIENT_TO_DIVERGE\"",
        "\"verdict\": \"ISA_IRRELEVANT_CHECKPOINTS_MATCH\"",
    ),
    (
        "evidence/xarch-local-isa/probe.json",
        "\"translation\": \"x86_64 arm executed under Rosetta 2\"",
        "\"translation\": \"x86_64 arm executed natively, no Rosetta\"",
    ),
];

/// (a) THE GUARD. Each named finding, inverted in the document's own bytes.
#[test]
fn inverting_a_published_finding_moves_the_seal() {
    let published = published_seals();
    let mut graded = 0usize;
    let mut failures = Vec::new();

    for (path, needle, replacement) in CENTRAL_FINDINGS {
        let full = repo_root().join(path);
        let Ok(raw) = std::fs::read_to_string(&full) else {
            // Not in this checkout. `evidence/stage-trace-isa/` is published
            // before it is tracked, so absence is a real state - but it is
            // stated, not skipped in silence.
            println!("NOT IN THIS CHECKOUT, not graded: {path}");
            continue;
        };
        assert!(
            raw.contains(needle),
            "the fixture for {path} is stale: it no longer contains {needle:?}, so \
             this guard would be inverting nothing. Fix the needle, do not delete it."
        );
        let forged = raw.replacen(needle, replacement, 1);
        assert_ne!(forged, raw, "the substitution produced identical bytes");

        let honest_seal = seal_document(raw.as_bytes())
            .unwrap_or_else(|e| panic!("{path} could not be sealed: {e}"))
            .seal;
        let forged_seal = seal_document(forged.as_bytes())
            .unwrap_or_else(|e| panic!("the {path} forgery could not be sealed: {e}"))
            .seal;
        graded += 1;

        if forged_seal == honest_seal {
            failures.push(format!(
                "  {path}\n    inverted {needle:?} -> {replacement:?}\n    and the seal did \
                 NOT move: {forged_seal}\n    the published digest authenticates nothing \
                 this document says"
            ));
            continue;
        }
        if let Some(table) = published.get(*path) {
            if &forged_seal == table {
                failures.push(format!(
                    "  {path}\n    inverted {needle:?} -> {replacement:?}\n    and the \
                     forgery seals to the PUBLISHED digest {table}"
                ));
                continue;
            }
            assert_eq!(
                &honest_seal, table,
                "the honest {path} does not seal to its own published line; the fixture \
                 and evidence/SEALS.txt disagree before any forgery was made"
            );
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {graded} inverted finding(s) did not move the seal:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        graded >= 2,
        "only {graded} finding(s) were graded; this guard is meant to cover every \
         published non-checkpoint document in the checkout"
    );
    println!("guard (a): {graded} inverted finding(s), every seal moved");
}

/// (a, generic) The same property without a table: flip the first boolean in
/// the document's bytes, whatever it is called.
///
/// This leg covers a document nobody thought to add to `CENTRAL_FINDINGS`,
/// which is the way the previous version of this defect arrived.
#[test]
fn inverting_the_first_boolean_moves_the_seal() {
    let published = published_seals();
    let mut failures = Vec::new();
    let mut graded = 0usize;

    for (path, raw) in published_non_checkpoint_documents() {
        let text = String::from_utf8(raw).unwrap_or_else(|e| panic!("{path} is not UTF-8: {e}"));
        let forged = if text.contains(": false") {
            text.replacen(": false", ": true", 1)
        } else if text.contains(": true") {
            text.replacen(": true", ": false", 1)
        } else {
            panic!(
                "{path} carries no top-level boolean, so this guard cannot invert anything \
                 in it. That is not a pass: give it an inversion in CENTRAL_FINDINGS or \
                 say here why the document has no falsifiable finding."
            );
        };
        let honest_seal = seal_document(text.as_bytes()).expect("seal honest").seal;
        let forged_seal = seal_document(forged.as_bytes()).expect("seal forgery").seal;
        graded += 1;

        if forged_seal == honest_seal {
            failures.push(format!(
                "  {path}: flipping its first boolean did not move the seal ({forged_seal})"
            ));
        }
        if let Some(table) = published.get(&path) {
            assert_eq!(
                &honest_seal, table,
                "the honest {path} does not seal to its published line"
            );
        }
    }

    assert!(
        failures.is_empty(),
        "{} document(s) seal identically with a boolean flipped:\n{}",
        failures.len(),
        failures.join("\n")
    );
    println!("guard (a, generic): {graded} document(s), every flipped boolean moved the seal");
}

/// (b) THE FLOOR. A degenerate field set must not silently buy a field-list
/// digest: the `Value`-only API refuses, and the bytes API takes the whole-file
/// path.
#[test]
fn a_degenerate_field_set_cannot_return_a_field_list_digest() {
    // A document tagged as a checkpoint record and carrying nothing else.
    let raw = br#"{"schema":"trios-checkpoint-record/8"}"#;
    let record: serde_json::Value = serde_json::from_slice(raw).unwrap();
    let (present, absent) = sealed_field_split(&record).expect("split");
    assert!(
        present.len() < SEAL_MIN_PRESENT_FIELDS,
        "fixture broken: it was supposed to be below the floor"
    );

    let err = provenance_seal(&record)
        .expect_err("a field list over 1 present and many absent names is not a seal");
    let text = format!("{err:#}");
    assert!(
        text.contains(DEGENERATE_SEAL_REFUSAL),
        "the refusal must be identifiable: {text}"
    );
    assert!(
        text.contains(&SEAL_MIN_PRESENT_FIELDS.to_string()),
        "the refusal must state the floor it applied: {text}"
    );
    assert!(
        text.contains(&absent.len().to_string()),
        "the refusal must state how many names were absent: {text}"
    );

    let sealed = seal_document(raw).expect("the bytes API must have an answer");
    assert_eq!(
        sealed.kind,
        SealKind::WholeFile,
        "below the floor the bytes API must fall back to the whole file"
    );

    // And the same for every published non-checkpoint document, against the
    // digest a reader gets from `shasum -a 256`.
    for (path, bytes) in published_non_checkpoint_documents() {
        let record: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        assert!(
            provenance_seal(&record).is_err(),
            "{path} is not a checkpoint record and must not receive a field-list seal"
        );
        let sealed = seal_document(&bytes).expect("seal");
        assert_eq!(sealed.kind, SealKind::WholeFile, "{path}");
        assert_eq!(
            sealed.seal,
            format!("sha256:{}", shasum_of(&repo_root().join(&path))),
            "{path}: the published seal must be what `shasum -a 256` prints"
        );
    }
}

/// The census the floor was derived from, re-measured. If a published checkpoint
/// record ever drops below the floor, this fails rather than letting the record
/// silently change seal rules.
#[test]
fn the_floor_sits_between_the_two_measured_populations() {
    let mut checkpoint_min = usize::MAX;
    let mut other_max = 0usize;
    let mut counted_checkpoints = 0usize;

    for path in published_seals().keys() {
        let Ok(raw) = std::fs::read(repo_root().join(path)) else {
            continue;
        };
        let record: serde_json::Value = serde_json::from_slice(&raw).expect("json");
        let (present, _) = sealed_field_split(&record).expect("split");
        if is_checkpoint_record(&record) {
            counted_checkpoints += 1;
            checkpoint_min = checkpoint_min.min(present.len());
        } else {
            other_max = other_max.max(present.len());
        }
    }

    assert!(
        counted_checkpoints > 0,
        "no checkpoint record was measured, so this census proves nothing"
    );
    assert!(
        other_max < SEAL_MIN_PRESENT_FIELDS,
        "a non-checkpoint document now presents {other_max} sealed name(s), at or above \
         the floor of {SEAL_MIN_PRESENT_FIELDS}; the two populations have met and the \
         constant no longer separates them"
    );
    assert!(
        checkpoint_min >= SEAL_MIN_PRESENT_FIELDS,
        "a published checkpoint record presents only {checkpoint_min} sealed name(s), \
         below the floor of {SEAL_MIN_PRESENT_FIELDS}; it would silently change seal \
         rules"
    );
    println!(
        "floor census: checkpoint records present >= {checkpoint_min}, non-checkpoint \
         documents present <= {other_max}, floor {SEAL_MIN_PRESENT_FIELDS}"
    );
}

/// The whole-file branch is exactly `shasum -a 256`, executed rather than
/// asserted from the implementation.
fn shasum_of(path: &std::path::Path) -> String {
    let out = Command::new("shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()
        .unwrap_or_else(|e| panic!("could not run shasum on {}: {e}", path.display()));
    assert!(
        out.status.success(),
        "shasum failed on {}: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .expect("shasum printed no digest")
        .to_owned()
}

/// The shipped binary must agree with the library on both rules, and must print
/// a coverage line that does not claim fields the document lacks.
#[test]
fn the_cli_prints_present_and_absent_apart() {
    let documents = published_non_checkpoint_documents();
    let (path, bytes) = &documents[0];
    let expected = seal_document(bytes).expect("seal").seal;

    let out = Command::new(env!("CARGO_BIN_EXE_ckpt_replay"))
        .current_dir(repo_root())
        .args(["--record", path, "--provenance-seal"])
        .output()
        .expect("run ckpt_replay");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert_eq!(out.status.code(), Some(0), "{text}");
    assert!(
        text.contains(&expected),
        "the CLI must print the same seal the library computes:\n{text}"
    );
    assert!(
        text.contains("whole-file"),
        "the CLI must say WHICH rule produced the digest:\n{text}"
    );
    assert!(
        !text.contains("covers 18 declared field(s)")
            && !text.contains("covers 19 declared field(s)"),
        "the CLI must not claim to cover declared fields this document does not carry:\n{text}"
    );

    // A checkpoint record: present and absent counted apart, never as one
    // number that reads as coverage.
    let record = "evidence/xarch-rustc/linux-rustc191/lin191/300.json";
    let out = Command::new(env!("CARGO_BIN_EXE_ckpt_replay"))
        .current_dir(repo_root())
        .args(["--record", record, "--provenance-seal"])
        .output()
        .expect("run ckpt_replay");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert_eq!(out.status.code(), Some(0), "{text}");
    let raw = std::fs::read(repo_root().join(record)).expect("read record");
    let value: serde_json::Value = serde_json::from_slice(&raw).expect("json");
    let (present, absent) = sealed_field_split(&value).expect("split");
    assert!(
        text.contains(&format!(
            "covers {} of {} declared field(s) ({} absent)",
            present.len(),
            present.len() + absent.len(),
            absent.len()
        )),
        "the CLI must state present-vs-declared, not one number:\n{text}"
    );
}
