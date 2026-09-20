//! The feature set written into every sidecar must be the feature set the
//! manifest declares -- recomputed, not restated.
//!
//! # The defect this exists to remove
//!
//! `checkpoint::declared_feature_states` is a hand-maintained list, and its own
//! doc comment promised that "a new feature is added to the digest by editing
//! this list rather than by remembering to". Nobody remembered. `Cargo.toml`
//! declared `det-math`; the list carried six entries and `det-math` was not
//! among them, so `PlatformProvenance::features` could not name it and nothing
//! in the repository noticed.
//!
//! The consequence was measured on real artifacts rather than argued: a
//! `--features det-math` run and a default run, both 10 steps, produced
//! DIFFERENT weights -- `7459c716...` against `efef1cba...` -- while their
//! sidecars carried a byte-identical `source_sha256` and a byte-identical
//! `platform.features` string, `ci-strict=0,gf16=0,gpu=0,race=0,smoke=0,
//! trios-integration=0`. Outside `evidence/det-math-isa/`, nothing tied the
//! det-math weights to a det-math build. That is this repository's stated
//! defect class in one sentence: a field that looks authoritative and is
//! silently incomplete.
//!
//! # Why this test is written the way it is
//!
//! Adding a seventh hand-written entry fixes today's instance and leaves the
//! failure mode exactly where it was. So this test does not contain a list of
//! feature names. It parses the `[features]` table out of `Cargo.toml` at test
//! time and compares that set against the names
//! `checkpoint::compiled_feature_set()` actually emits -- the emitted string
//! itself, the one that lands in the sidecar, not an intermediate. A test that
//! duplicated the list would be the very defect it is meant to catch.
//!
//! `default` is dropped from the manifest side because it names no code of its
//! own; that is the rule the doc comment on `declared_feature_states` already
//! stated, and it is applied here rather than restated there.
//!
//! # What it catches, and what it does not
//!
//! Catches: a feature declared in the manifest and missing from the digest
//! (observed: adding `bogus_feature = []` to `[features]` fails this test by
//! name); a feature removed from the digest and left in the manifest (observed:
//! deleting the `det-math` entry fails naming `det-math`); a feature digested
//! that the manifest does not declare; and, under
//! `cargo test --features det-math`, a `det-math` entry whose emitted value
//! does not track the compiled `cfg!`.
//!
//! Does NOT catch: a feature that is declared, digested, and gates no code --
//! the digest describes the BINARY, and whether a flag changes the WEIGHTS is a
//! separate measurement. `gf16` is the standing example: its build hashes
//! differently and, on the one comparison made, produced byte-identical
//! artifacts. Nor does it catch a feature name that is spelled the same in both
//! places and wrong in both.

use std::collections::BTreeSet;
use std::path::PathBuf;

/// The manifest key that names no code of its own, and so is not expected in
/// the digest. Kept here beside the `.remove` that uses it.
const FEATURE_WITHOUT_CODE: &str = "default";

fn manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")
}

/// Feature names declared in the `[features]` table of `Cargo.toml`.
///
/// Hand-parsed rather than routed through the `toml` crate because `toml` is a
/// dependency of the library and not a dev-dependency of the test target, and
/// `Cargo.toml` is owned by another change in flight. The parser is therefore
/// written to REFUSE what it does not understand instead of skipping it: a
/// silent skip inside a completeness check would reintroduce, in the guard, the
/// exact defect the guard exists to catch.
fn features_declared_in_manifest() -> BTreeSet<String> {
    let path = manifest_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    let mut names = BTreeSet::new();
    let mut inside = false;
    let mut seen_table = false;
    // Depth of an unclosed `[ ... ]` value spilling onto following lines, e.g.
    // a long feature list. While it is non-zero the line is a continuation and
    // carries no key.
    let mut bracket_depth = 0usize;

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim();

        if bracket_depth == 0 && line.starts_with('[') {
            // A table header. `[features]` opens the region; any other header
            // closes it. Checked before the comment strip so a commented-out
            // header cannot move the region.
            let header = line.trim_start_matches('[').trim_end_matches(']').trim();
            inside = header == "features";
            if inside {
                assert!(
                    !seen_table,
                    "{}:{line_number}: a second [features] table -- this parser \
                     assumes one, and would silently read only the last",
                    path.display()
                );
                seen_table = true;
            }
            continue;
        }

        if !inside {
            continue;
        }

        // Comments and blank lines carry no key. Trailing comments are not
        // stripped from value text because the bracket count below only needs
        // to be balanced across the whole line, and `#` inside a feature array
        // does not occur in this manifest -- if it ever does, the assertion
        // below is what reports it rather than a wrong answer.
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if bracket_depth == 0 {
            let (key, rest) = line.split_once('=').unwrap_or_else(|| {
                panic!(
                    "{}:{line_number}: line inside [features] has no `=` and is \
                     not a comment: {line:?}. This parser refuses lines it does \
                     not understand rather than skipping them.",
                    path.display()
                )
            });
            let key = key.trim().trim_matches('"').to_string();
            assert!(
                !key.is_empty(),
                "{}:{line_number}: empty feature name in {line:?}",
                path.display()
            );
            let inserted = names.insert(key.clone());
            assert!(
                inserted,
                "{}:{line_number}: feature {key:?} declared twice",
                path.display()
            );
            bracket_depth = bracket_depth
                .saturating_add(rest.matches('[').count())
                .saturating_sub(rest.matches(']').count());
        } else {
            bracket_depth = bracket_depth
                .saturating_add(line.matches('[').count())
                .saturating_sub(line.matches(']').count());
        }
    }

    assert!(
        seen_table,
        "{}: no [features] table found -- the test would otherwise pass \
         vacuously on an empty set",
        path.display()
    );
    assert_eq!(
        bracket_depth,
        0,
        "{}: unbalanced `[` in the [features] table; the parse is not trustworthy",
        path.display()
    );
    assert!(
        names.remove(FEATURE_WITHOUT_CODE),
        "{}: [features] does not declare `{FEATURE_WITHOUT_CODE}`. Either the \
         manifest changed or the parser stopped finding the table; both make \
         this test's answer wrong rather than merely different.",
        path.display()
    );

    names
}

/// The `name=0|1` pairs the checkpoint writer actually emits, parsed back into
/// (name, enabled). Taken from the emitted string so the test reads what lands
/// in the sidecar rather than a list it keeps of its own.
fn features_emitted_by_writer() -> Vec<(String, bool)> {
    let emitted = trios_trainer::checkpoint::compiled_feature_set();
    assert!(
        !emitted.is_empty(),
        "compiled_feature_set() returned an empty string; a completeness check \
         against nothing would pass"
    );
    emitted
        .split(',')
        .map(|pair| {
            let (name, state) = pair.split_once('=').unwrap_or_else(|| {
                panic!("compiled_feature_set() emitted {pair:?}, which is not `name=0|1`")
            });
            let on = match state {
                "0" => false,
                "1" => true,
                other => panic!("compiled_feature_set() emitted state {other:?} for {name:?}"),
            };
            (name.to_string(), on)
        })
        .collect()
}

#[test]
fn every_declared_feature_is_digested_and_no_others() {
    let declared = features_declared_in_manifest();
    let emitted = features_emitted_by_writer();

    let emitted_names: BTreeSet<String> = emitted.iter().map(|(n, _)| n.clone()).collect();
    assert_eq!(
        emitted_names.len(),
        emitted.len(),
        "compiled_feature_set() emitted a duplicate name: {emitted:?}"
    );

    let missing: Vec<&String> = declared.difference(&emitted_names).collect();
    assert!(
        missing.is_empty(),
        "declared in Cargo.toml [features] but NOT in the checkpoint digest: \
         {missing:?}. Add each to `declared_feature_states` in \
         src/checkpoint.rs (sorted, and widen the array length) -- until then a \
         build with that feature on writes a `platform.features` string \
         indistinguishable from a default build's.\n  declared: {declared:?}\n  \
         emitted:  {emitted_names:?}"
    );

    let extra: Vec<&String> = emitted_names.difference(&declared).collect();
    assert!(
        extra.is_empty(),
        "in the checkpoint digest but NOT declared in Cargo.toml [features]: \
         {extra:?}. Either the feature was removed from the manifest and left \
         in `declared_feature_states`, or it is misspelled in one of the two \
         places.\n  declared: {declared:?}\n  emitted:  {emitted_names:?}"
    );
}

/// Set equality alone would still pass if every emitted VALUE were hardcoded
/// `false`. This asserts the one entry whose state a caller can flip from the
/// command line, so the file is a live check under both
/// `cargo test --test declared_features_cover_the_manifest` and the same
/// command with `--features det-math`.
#[test]
fn det_math_entry_tracks_the_compiled_cfg() {
    let emitted = features_emitted_by_writer();
    let (_, on) = emitted
        .iter()
        .find(|(name, _)| name == "det-math")
        .unwrap_or_else(|| {
            panic!(
                "compiled_feature_set() emits no `det-math` entry: {:?}",
                emitted.iter().map(|(n, _)| n).collect::<Vec<_>>()
            )
        });
    assert_eq!(
        *on,
        cfg!(feature = "det-math"),
        "the emitted `det-math` state disagrees with the compiled cfg -- a \
         det-math build and a default build would write the same \
         `platform.features` string, which is the condition this whole file \
         exists to make impossible"
    );
}
