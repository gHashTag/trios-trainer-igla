//! The four-arm 12000-step cross-ISA result, guarded against silent edits.
//!
//! `evidence/headline-isa-12000/` publishes four `trios-checkpoint-record/9`
//! sidecars: {default features, `--features det-math`} x {aarch64, x86_64},
//! each after 12000 gradient steps of one configuration. The result they carry
//! is the strongest measurement this project owns -- the previously published
//! cross-ISA MATCH reached TEN steps, this reaches twelve thousand:
//!
//!     det-math  aarch64  65f36543...  ==  det-math  x86_64  65f36543...
//!     default   aarch64  8a86fe69...  !=  default   x86_64  cb7b24ca...
//!
//! and the default aarch64 value is bit-equal to the published headline in
//! `evidence/r9-headline/12000.json`.
//!
//! # What this file is for
//!
//! Those four hashes are hex strings in JSON files. Nothing stops an editor,
//! a merge, or a well-meaning cleanup pass from changing one character of one
//! of them, and a hash that has been altered is indistinguishable by eye from
//! one that has not. This test is what makes such an edit fail loudly. It was
//! confirmed to fail, not assumed to: a single hex character was changed in
//! `dm-x86_64-12000.json` and `det_math_pair_is_byte_identical` reported both
//! hashes and the file names; `det-math=1` was changed to `det-math=0` in the
//! same file and `det_math_records_declare_the_feature_on` reported the
//! feature strings. An assertion never seen to fail is not evidence.
//!
//! # Why there is no skip-on-missing-file path
//!
//! These four files ARE the deliverable. Every other trace of this experiment
//! lives under `checkpoints/isa-probe/headline-12000/`, which `.gitignore:13`
//! excludes wholesale, and the shell script that produced it was never in the
//! repository at all. A test that skipped when the records were absent would
//! pass on precisely the clone where the evidence had gone missing -- a
//! failure that reports success, which is the defect class this repository
//! exists to remove. A missing record is a failure here.
//!
//! # Scope, which is smaller than "the checkpoint is portable across ISA"
//!
//! The x86_64 arm ran under Rosetta 2 binary translation on an aarch64 macOS
//! host. No x86_64 processor executed it. This test guards the published
//! record of that measurement; it does not extend the measurement's reach.
//! See `evidence/headline-isa-12000/PROVENANCE.txt` section 5.
//!
//! This test reads committed JSON. It does not re-run the trainer and cannot:
//! the x86_64 arm is an observation no host running `cargo test` can
//! regenerate. `scripts/headline_isa_probe.py` is the reproduction path.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

const DIR: &str = "evidence/headline-isa-12000";

const DM_ARM64: &str = "evidence/headline-isa-12000/dm-arm64-12000.json";
const DM_X86_64: &str = "evidence/headline-isa-12000/dm-x86_64-12000.json";
const DF_ARM64: &str = "evidence/headline-isa-12000/df-arm64-12000.json";
const DF_X86_64: &str = "evidence/headline-isa-12000/df-x86_64-12000.json";

/// The published headline the default aarch64 arm must equal. Owned by
/// another deliverable; this test reads it and does not write it.
const R9_HEADLINE: &str = "evidence/r9-headline/12000.json";

const EXPECTED_STEP: u64 = 12000;
const EXPECTED_SEED: u64 = 47;

const DET_MATH_ON: &str = "det-math=1";
const DET_MATH_OFF: &str = "det-math=0";

const ALL_FOUR: [&str; 4] = [DM_ARM64, DM_X86_64, DF_ARM64, DF_X86_64];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Read and parse one record. Absence is a failure, never a skip.
fn record(rel: &str) -> Value {
    let path = repo_root().join(rel);
    let raw = fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "{rel}: this record is the deliverable and it could not be read \
             ({err}).\n  looked at: {}\n  The four records under {DIR} are the \
             only committed trace of the 12000-step cross-ISA measurement; \
             every artifact it produced lives under checkpoints/, which \
             .gitignore excludes. If this file is gone the measurement is \
             gone, so this is a failure and not a skip.",
            path.display()
        )
    });
    serde_json::from_str(&raw).unwrap_or_else(|err| panic!("{rel}: is not parseable JSON: {err}"))
}

fn string_field(rel: &str, value: &Value, path: &[&str]) -> String {
    let mut node = value;
    for key in path {
        node = node.get(key).unwrap_or_else(|| {
            panic!(
                "{rel}: has no field `{}` (missing at `{key}`)",
                path.join(".")
            )
        });
    }
    node.as_str()
        .unwrap_or_else(|| {
            panic!(
                "{rel}: field `{}` is {node}, which is not a string",
                path.join(".")
            )
        })
        .to_string()
}

fn u64_field(rel: &str, value: &Value, key: &str) -> u64 {
    let node = value
        .get(key)
        .unwrap_or_else(|| panic!("{rel}: has no field `{key}`"));
    node.as_u64().unwrap_or_else(|| {
        panic!("{rel}: field `{key}` is {node}, which is not a non-negative integer")
    })
}

fn sha256_of(rel: &str) -> String {
    let value = record(rel);
    let got = string_field(rel, &value, &["sha256"]);
    assert_eq!(
        got.len(),
        64,
        "{rel}: `sha256` is {} hex characters, not 64: {got}",
        got.len()
    );
    assert!(
        got.chars().all(|c| c.is_ascii_hexdigit()),
        "{rel}: `sha256` is not hexadecimal: {got}"
    );
    got
}

fn features_of(rel: &str) -> String {
    let value = record(rel);
    string_field(rel, &value, &["platform", "features"])
}

/// (a) The det-math pair is the result: two ISAs, byte-identical weights.
#[test]
fn det_math_pair_is_byte_identical() {
    let arm = sha256_of(DM_ARM64);
    let x86 = sha256_of(DM_X86_64);
    assert_eq!(
        arm, x86,
        "the det-math pair is no longer byte-identical, which is the whole \
         result these records publish.\n  {DM_ARM64}\n    sha256 {arm}\n  \
         {DM_X86_64}\n    sha256 {x86}\n  Either a record was edited, or the \
         measurement changed. If the measurement changed, re-run \
         scripts/headline_isa_probe.py and republish; do not adjust this \
         assertion to match the files."
    );
}

/// (b) The default pair is the control: without the feature, the ISAs diverge.
/// If this ever passed as a MATCH the det-math result would mean nothing,
/// because the arms would be agreeing for some reason other than the feature.
#[test]
fn default_pair_differs_across_the_isa_boundary() {
    let arm = sha256_of(DF_ARM64);
    let x86 = sha256_of(DF_X86_64);
    assert_ne!(
        arm, x86,
        "the default (no det-math) pair now MATCHES across the ISA boundary. \
         That is the control for the det-math result, and if the arms agree \
         without the feature then the feature is not what makes them \
         agree.\n  {DF_ARM64}\n  {DF_X86_64}\n    both sha256 {arm}\n  Check \
         that these are two different runs and not one run copied twice."
    );
}

/// (c) The default aarch64 arm IS the published headline run, not a lookalike.
#[test]
fn default_aarch64_equals_the_published_headline() {
    let arm = sha256_of(DF_ARM64);
    let headline_path = repo_root().join(R9_HEADLINE);
    let raw = fs::read_to_string(&headline_path).unwrap_or_else(|err| {
        panic!(
            "{R9_HEADLINE}: could not be read ({err}).\n  looked at: {}\n  \
             The four records under {DIR} are anchored to the published \
             headline through this file; without it the claim `the default \
             aarch64 arm is the headline run` cannot be checked. This test \
             does not own {R9_HEADLINE} and must not create it -- if it is \
             absent, the headline record itself needs committing.",
            headline_path.display()
        )
    });
    let headline: Value = serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("{R9_HEADLINE}: is not parseable JSON: {err}"));
    let published = string_field(R9_HEADLINE, &headline, &["sha256"]);
    assert_eq!(
        arm, published,
        "the default aarch64 arm no longer equals the published \
         headline.\n  {DF_ARM64}\n    sha256 {arm}\n  {R9_HEADLINE}\n    \
         sha256 {published}\n  These two records are supposed to describe the \
         same 852272 bytes. If they disagree, one of them is describing a \
         different run than it claims to."
    );
}

/// (d) Each record declares which side of the experiment it is on. Without
/// this, four hashes with no feature flags attached would prove nothing about
/// det-math: any pair of them could be relabelled.
#[test]
fn det_math_records_declare_the_feature_on() {
    for rel in [DM_ARM64, DM_X86_64] {
        let features = features_of(rel);
        assert!(
            features.contains(DET_MATH_ON),
            "{rel}: is published as a det-math arm but its \
             `platform.features` does not contain `{DET_MATH_ON}`.\n  \
             expected to find: {DET_MATH_ON}\n  actual features: \
             {features}\n  A hash labelled det-math that was not built with \
             det-math would make the MATCH in \
             det_math_pair_is_byte_identical mean something other than what \
             this directory says it means."
        );
    }
}

#[test]
fn default_records_declare_the_feature_off() {
    for rel in [DF_ARM64, DF_X86_64] {
        let features = features_of(rel);
        assert!(
            features.contains(DET_MATH_OFF),
            "{rel}: is published as a default (no det-math) arm but its \
             `platform.features` does not contain `{DET_MATH_OFF}`.\n  \
             expected to find: {DET_MATH_OFF}\n  actual features: {features}"
        );
    }
}

/// (e) All four ran the same experiment: same step count, same seed.
#[test]
fn all_four_declare_the_headline_step_and_seed() {
    for rel in ALL_FOUR {
        let value = record(rel);
        let step = u64_field(rel, &value, "step");
        assert_eq!(
            step, EXPECTED_STEP,
            "{rel}: declares step {step}, not {EXPECTED_STEP}. These records \
             publish a 12000-step result; a record at a different step is not \
             part of it."
        );
        let seed = u64_field(rel, &value, "seed");
        assert_eq!(
            seed, EXPECTED_SEED,
            "{rel}: declares seed {seed}, not {EXPECTED_SEED}. The four arms \
             differ only in ISA and feature set; a different seed makes the \
             comparison between them meaningless."
        );
    }
}

/// The two ISA labels must actually differ, or `dm-arm64` and `dm-x86_64`
/// could be two copies of one run and the MATCH would be a tautology.
#[test]
fn each_pair_spans_two_distinct_architectures() {
    for (arm_rel, x86_rel) in [(DM_ARM64, DM_X86_64), (DF_ARM64, DF_X86_64)] {
        let arm = string_field(arm_rel, &record(arm_rel), &["platform", "arch"]);
        let x86 = string_field(x86_rel, &record(x86_rel), &["platform", "arch"]);
        assert_eq!(
            arm, "aarch64",
            "{arm_rel}: declares platform.arch {arm}, expected aarch64"
        );
        assert_eq!(
            x86, "x86_64",
            "{x86_rel}: declares platform.arch {x86}, expected x86_64"
        );
        assert_ne!(
            arm, x86,
            "{arm_rel} and {x86_rel} declare the SAME platform.arch ({arm}). \
             They are supposed to be the two sides of an ISA comparison; if \
             both name one architecture, one run was copied rather than run."
        );
    }
}

/// The records must keep naming the artifacts they hash, and the four
/// canon names must be four distinct runs rather than one run copied.
#[test]
fn the_four_records_name_four_distinct_runs() {
    let mut names: Vec<String> = Vec::new();
    for rel in ALL_FOUR {
        let value = record(rel);
        let canon = string_field(rel, &value, &["canon_name"]);
        let path = string_field(rel, &value, &["path"]);
        assert!(
            path.ends_with(&format!("{canon}/{EXPECTED_STEP}.bin")),
            "{rel}: `path` is {path}, which does not end with \
             {canon}/{EXPECTED_STEP}.bin. The record no longer names the \
             artifact it hashes."
        );
        assert!(
            !names.contains(&canon),
            "{rel}: canon_name {canon} already appeared in another of the \
             four records. The four arms must be four runs; a duplicate name \
             means one was copied.\n  seen so far: {names:?}"
        );
        names.push(canon);
    }
    assert_eq!(
        names.len(),
        4,
        "expected four distinct runs under {DIR}, found {}: {names:?}",
        names.len()
    );
}

/// The directory must not quietly lose a record. Reading each of the four by
/// name is what catches deletion; this catches the opposite mistake, a fifth
/// record appearing that no assertion above covers.
#[test]
fn the_directory_holds_exactly_the_four_published_records() {
    let dir = repo_root().join(DIR);
    let mut found: Vec<String> = fs::read_dir(&dir)
        .unwrap_or_else(|err| panic!("{DIR}: could not be listed ({err}) at {}", dir.display()))
        .map(|entry| {
            entry
                .unwrap_or_else(|err| panic!("{DIR}: unreadable entry: {err}"))
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|name| name.ends_with(".json"))
        .collect();
    found.sort();

    let mut expected: Vec<String> = ALL_FOUR
        .iter()
        .map(|rel| {
            Path::new(rel)
                .file_name()
                .expect("constant has a file name")
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    expected.sort();

    assert_eq!(
        found, expected,
        "the set of records under {DIR} is not the four this test guards.\n  \
         on disk:  {found:?}\n  guarded:  {expected:?}\n  A record here that \
         no assertion covers is evidence nobody is checking; add it to \
         tests/headline_isa_12000.rs or take it out of the directory."
    );
}
