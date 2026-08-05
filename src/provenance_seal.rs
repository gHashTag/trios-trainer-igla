//! Declaration seal: a digest over the part of a checkpoint record that no
//! container byte can confirm.
//!
//! # The defect this closes
//!
//! Every artifact in `evidence/` is a pair: a `.bin` container and a `.json`
//! sidecar. The container is covered by `sha256` and re-hashing it is what
//! `--integrity-only` and `interop/triosckp_reader.py` both do. The sidecar's
//! provenance half -- the platform triple, the toolchain string, the corpus
//! digests, the trainer digest, `git_sha`, `steps_total`, `eval_every` and
//! `final_val_bpb` -- is covered by NOTHING. It is free text written by the
//! party being audited, and the two verifiers this repository ships read it,
//! quote it, and pass the record.
//!
//! That was demonstrated, not supposed. The headline cross-architecture record
//! (`evidence/xarch-run-30767491098/12000.json`, an x86_64 Linux run) was
//! copied to a scratch directory and its sidecar rewritten to claim
//! `aarch64` / `macos`, a rustc string that never existed, and
//! `final_val_bpb = 1.5492` -- the exact number this project publicly
//! retracted. Both verifiers passed it: the Python reader printed
//! `RESULT PASS` and exit 0, and `ckpt_replay --record ... --integrity-only`
//! printed `INTEGRITY OK` and exit 0. Neither is wrong about the bytes; the
//! bytes were untouched. Both were silent about the half of the record a
//! conformity scheme actually rests on. The transcript is quoted in
//! `docs/PROVENANCE-BINDING.md`.
//!
//! # What this IS, stated before what it does
//!
//! A SEAL, not a signature. It is a digest of a declaration, and it binds
//! nothing on its own: an adversary who controls both the record and the
//! channel the seal is published on simply recomputes the seal over the
//! forgery. What a published seal buys is that the forgery becomes DETECTABLE
//! by anyone holding the published digest, instead of undetectable by anyone
//! at all. Closing the remaining gap needs key material -- a signature over
//! the seal by a party that is not the vendor -- which this project does not
//! have and does not pretend to.
//!
//! # Normative specification
//!
//! Two independent implementations must agree on the seal of every published
//! record, or the specification is wrong. The second implementation is
//! `provenance_seal()` in `interop/triosckp_reader.py`.
//!
//! ## 1. The sealed field set
//!
//! Exactly these roots, and nothing else:
//!
//! * `platform.os`, `platform.arch`, `platform.toolchain`, `platform.libc`
//! * every OTHER key present under `platform`, whatever it is named
//! * `corpus`, `source_sha256`, `trainer`, `git_sha`, `git_dirty`,
//!   `steps_total`, `eval_every`, `final_val_bpb`, `schema`
//! * `sha256`, `bytes`, `seed`, `step`, `gf16_floor_every`
//!
//! The last line is the round-7 correction and the reason every digest in
//! `docs/PROVENANCE-BINDING.md` section 4 was reminted. Leaving `sha256` and
//! `bytes` out was defended as "sealed by the container digest already", which
//! is circular: `sha256` IS the container digest, and a digest cannot seal
//! itself. Because the seal named no artifact, one authenticated declaration
//! fitted EVERY container. The x86_64 Linux record could be attached to the
//! aarch64 macOS `.bin` -- both are 852272 bytes -- and both shipped verifiers
//! passed, with the seal still matching the digest published for the Linux run.
//! That forgery manufactures the one claim this project rests on,
//! cross-architecture bit identity, out of a record nobody had to break.
//! `seed` and `step` join for the same reason at one remove (they name WHICH
//! artifact the recipe was supposed to produce), and `gf16_floor_every` because
//! it is a recipe parameter that demonstrably changes the weights and lives in
//! no container header.
//!
//! This set is a SUBSET of what `interop/triosckp_reader.py` classifies as
//! ECHOED, chosen because it is the environment declaration plus the recipe
//! parameters a conformity claim is quoted on. Sidecar fields that remain
//! outside the seal are named, one by one, in `docs/PROVENANCE-BINDING.md`;
//! an unsealed field is a forgeable field and the doc says so rather than
//! letting the seal be read as covering the whole record.
//!
//! `schema` is included even though the Python reader excludes it from ECHOED
//! (it checks the tag against field presence): a seal that did not cover the
//! schema tag would let a record be re-labelled without changing its digest.
//!
//! ## 2. Rendering
//!
//! The sealed fields are rendered as a listing of `dotted.key=value` lines,
//! sorted by KEY in byte order, each line terminated by `\n` (including the
//! last). Nested objects and arrays are flattened, so `corpus` contributes
//! `corpus.train.sha256=...` and never a serialized JSON blob -- there is no
//! canonical JSON here to disagree about.
//!
//! Values:
//!
//! | JSON shape        | rendering                                        |
//! |-------------------|--------------------------------------------------|
//! | string            | verbatim, with `\` `LF` `CR` escaped (clause 3)   |
//! | boolean           | `true` / `false`                                 |
//! | integer           | decimal, no sign for positives                   |
//! | non-integer number| Rust `{}` on `f64` (shortest round-trip)         |
//! | null              | `<null>`                                         |
//! | empty object      | `<empty-object>`                                 |
//! | empty array       | `<empty-array>`                                  |
//! | non-empty array   | flattened as `key.0`, `key.1`, ...               |
//! | ABSENT            | `<absent>`                                       |
//!
//! `<absent>` is what makes DELETION detectable: a record with `git_sha`
//! removed renders `git_sha=<absent>` and seals differently from one that
//! carries it. A verifier that simply skipped missing keys would give a
//! forger a free edit.
//!
//! The Python side reproduces Rust's `f64` Display rather than the other way
//! round: `repr()` gives the same shortest round-trip digits but writes
//! integral values as `2.0` where Rust writes `2`, and uses exponent notation
//! where Rust never does. `rust_f64_display()` in the reader normalises both.
//! One shape is beyond both: a JSON integer too large for `i64`/`u64`, which
//! `serde_json` widens to `f64` while Python keeps exact. No record on disk
//! contains one, and the interop check is what would catch it.
//!
//! ## 3. Escaping
//!
//! In string values only: `\` becomes `\\`, LF becomes `\n`, CR becomes `\r`.
//! Without this a toolchain string containing a newline could carry
//! `\nplatform.arch=x86_64` and forge a line of the listing, which is a
//! collision an attacker chooses rather than one they have to find.
//!
//! ## 4. The seal
//!
//! `sha256:` followed by the lowercase hex SHA-256 of the listing bytes
//! (UTF-8; ASCII for every record published here).
//!
//! # Not `canonical_digest.rs`
//!
//! Deliberately a separate object with a separate module. `canonical_digest`
//! digests WEIGHTS under a signed-zero normalisation rule; this digests a
//! DECLARATION. Folding one into the other would produce a number that means
//! neither thing.

use anyhow::{bail, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Prefix every seal carries, so a bare hex string can never be mistaken for
/// one and a grep can find it.
pub const SEAL_PREFIX: &str = "sha256:";

/// Rendering of a field the record does not carry. See clause 2.
pub const ABSENT: &str = "<absent>";

/// `platform` sub-keys the seal always states, present or not. Any other key
/// found under `platform` is sealed too; these four are listed because their
/// ABSENCE must also change the seal.
pub const SEALED_PLATFORM_KEYS: &[&str] = &["arch", "libc", "os", "toolchain"];

/// Top-level roots the seal always states, present or not. Sorted, and kept
/// sorted: this list is read as a specification, not just iterated.
///
/// `sha256` and `bytes` are in here, and the earlier claim that they were
/// "sealed by the container digest already" was circular -- `sha256` IS the
/// container digest, and no digest seals itself. A seal that named no artifact
/// bound its declaration to NO PARTICULAR `.bin`, so a single authenticated
/// declaration fitted every container of the same length: the x86_64 Linux
/// record could be handed the aarch64 macOS weights (both 852272 bytes) and
/// still match the seal published for the Linux run, forging exactly the
/// cross-architecture claim this project is audited on. `bytes` is sealed
/// beside `sha256` because a length is the one property a verifier can check
/// without hashing, and the two verifiers must not be able to disagree about
/// it.
///
/// `seed` and `step` name WHICH artifact the recipe was meant to produce;
/// `gf16_floor_every` is a recipe parameter that mutates the weights and
/// appears in no container header, so nothing else covered it.
///
/// A name listed here that the record does not carry renders `<absent>`, so
/// adding a name is also what stops the field being introduced later in
/// silence.
pub const SEALED_TOP_LEVEL: &[&str] = &[
    "bytes",
    "corpus",
    "eval_every",
    "final_val_bpb",
    "gf16_floor_every",
    "git_dirty",
    "git_sha",
    "schema",
    "seed",
    "sha256",
    "source_sha256",
    "step",
    "steps_total",
    "trainer",
];

/// Escape a string value so it cannot forge a line of the listing (clause 3).
fn escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            other => out.push(other),
        }
    }
    out
}

/// Render one scalar JSON value (clause 2). Containers never reach here.
fn render_scalar(value: &Value) -> String {
    match value {
        Value::Null => "<null>".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::String(s) => escape(s),
        Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                u.to_string()
            } else if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(f) = n.as_f64() {
                format!("{f}")
            } else {
                // `serde_json` without `arbitrary_precision` has no fourth
                // shape; if one ever appears, say so instead of inventing a
                // rendering that the Python side cannot reproduce.
                format!("<unrenderable-number:{n}>")
            }
        }
        Value::Array(_) | Value::Object(_) => unreachable!("containers are flattened"),
    }
}

/// Append `key=value` lines for `value` under `prefix`, flattening containers.
fn flatten(prefix: &str, value: &Value, out: &mut Vec<(String, String)>) {
    match value {
        Value::Object(map) => {
            if map.is_empty() {
                out.push((prefix.to_string(), "<empty-object>".to_string()));
                return;
            }
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for key in keys {
                flatten(&format!("{prefix}.{key}"), &map[key], out);
            }
        }
        Value::Array(items) => {
            if items.is_empty() {
                out.push((prefix.to_string(), "<empty-array>".to_string()));
                return;
            }
            for (index, item) in items.iter().enumerate() {
                flatten(&format!("{prefix}.{index}"), item, out);
            }
        }
        scalar => out.push((prefix.to_string(), render_scalar(scalar))),
    }
}

/// The exact bytes the seal is taken over, for a caller that wants to show its
/// work. Returned as text so an auditor can diff two records field by field
/// instead of being handed two digests and told they differ.
pub fn provenance_listing(record: &Value) -> Result<String> {
    let map = match record {
        Value::Object(map) => map,
        other => bail!(
            "a checkpoint record must be a JSON object; this one is a {}",
            match other {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(_) => "number",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => unreachable!(),
            }
        ),
    };

    let mut fields: Vec<(String, String)> = Vec::new();

    for key in SEALED_TOP_LEVEL {
        // `SEALED_TOP_LEVEL` may name a key twice without changing the seal:
        // the de-duplication below is by key, so a table edit cannot silently
        // double a line.
        match map.get(*key) {
            Some(value) => flatten(key, value, &mut fields),
            None => fields.push(((*key).to_string(), ABSENT.to_string())),
        }
    }

    match map.get("platform") {
        Some(Value::Object(platform)) => {
            for key in SEALED_PLATFORM_KEYS {
                match platform.get(*key) {
                    Some(value) => flatten(&format!("platform.{key}"), value, &mut fields),
                    None => fields.push((format!("platform.{key}"), ABSENT.to_string())),
                }
            }
            let mut siblings: Vec<&String> = platform
                .keys()
                .filter(|k| !SEALED_PLATFORM_KEYS.contains(&k.as_str()))
                .collect();
            siblings.sort();
            for key in siblings {
                flatten(&format!("platform.{key}"), &platform[key], &mut fields);
            }
        }
        Some(other) => {
            // `platform` present but not an object. The four required keys are
            // absent from it whatever it is, and the thing itself is sealed
            // under its own name so the anomaly cannot be edited away.
            for key in SEALED_PLATFORM_KEYS {
                fields.push((format!("platform.{key}"), ABSENT.to_string()));
            }
            flatten("platform", other, &mut fields);
        }
        None => {
            for key in SEALED_PLATFORM_KEYS {
                fields.push((format!("platform.{key}"), ABSENT.to_string()));
            }
        }
    }

    fields.sort_by(|a, b| a.0.cmp(&b.0));
    fields.dedup_by(|a, b| a.0 == b.0);

    let mut listing = String::new();
    for (key, value) in fields {
        listing.push_str(&key);
        listing.push('=');
        listing.push_str(&value);
        listing.push('\n');
    }
    Ok(listing)
}

/// The seal of one checkpoint record: `sha256:<64 hex>` over the listing.
///
/// See the module documentation for the normative definition. This function is
/// the reference implementation of it; `interop/triosckp_reader.py` is the
/// second, and the two agreeing on every record in `evidence/` is the interop
/// check that keeps the specification honest.
pub fn provenance_seal(record: &Value) -> Result<String> {
    let listing = provenance_listing(record)?;
    let digest = Sha256::digest(listing.as_bytes());
    let mut hex = String::with_capacity(SEAL_PREFIX.len() + 64);
    hex.push_str(SEAL_PREFIX);
    for byte in digest.iter() {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    Ok(hex)
}

/// The sealed field NAMES for a given record, in listing order.
///
/// Printed beside every verdict, because "this declaration is unauthenticated"
/// is only actionable if the reader is told which fields the sentence covers.
pub fn sealed_field_names(record: &Value) -> Result<Vec<String>> {
    Ok(provenance_listing(record)?
        .lines()
        .filter_map(|line| line.split_once('=').map(|(k, _)| k.to_string()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn listing_is_sorted_and_newline_terminated() {
        let listing = provenance_listing(&json!({"schema": "s", "git_sha": "abc"})).unwrap();
        assert!(listing.ends_with('\n'));
        let keys: Vec<&str> = listing
            .lines()
            .map(|l| l.split_once('=').unwrap().0)
            .collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "listing must be sorted by key");
    }

    #[test]
    fn absent_field_is_stated_not_skipped() {
        let listing = provenance_listing(&json!({})).unwrap();
        assert!(
            listing.contains("git_sha=<absent>"),
            "a deleted field must still occupy a line:\n{listing}"
        );
        assert!(listing.contains("platform.arch=<absent>"));
    }

    #[test]
    fn deleting_a_field_changes_the_seal() {
        let full = json!({"git_sha": "abc", "schema": "trios-checkpoint-record/4"});
        let stripped = json!({"schema": "trios-checkpoint-record/4"});
        assert_ne!(
            provenance_seal(&full).unwrap(),
            provenance_seal(&stripped).unwrap()
        );
    }

    #[test]
    fn nested_objects_are_flattened_not_serialized() {
        let listing =
            provenance_listing(&json!({"corpus": {"train": {"sha256": "aa", "bytes": 7}}}))
                .unwrap();
        assert!(listing.contains("corpus.train.sha256=aa\n"), "{listing}");
        assert!(listing.contains("corpus.train.bytes=7\n"), "{listing}");
    }

    #[test]
    fn platform_siblings_are_sealed_too() {
        let listing = provenance_listing(&json!({
            "platform": {"os": "linux", "arch": "x86_64", "features": "gf16"}
        }))
        .unwrap();
        assert!(listing.contains("platform.features=gf16\n"), "{listing}");
    }

    #[test]
    fn a_newline_in_a_value_cannot_forge_a_line() {
        let injected = json!({"platform": {"toolchain": "rustc 1.0\nplatform.arch=aarch64"}});
        let honest = json!({"platform": {"toolchain": "rustc 1.0", "arch": "aarch64"}});
        let listing = provenance_listing(&injected).unwrap();
        assert!(
            listing.contains("platform.toolchain=rustc 1.0\\nplatform.arch=aarch64\n"),
            "{listing}"
        );
        assert_ne!(
            provenance_seal(&injected).unwrap(),
            provenance_seal(&honest).unwrap()
        );
    }

    #[test]
    fn a_non_object_record_is_a_refusal_not_a_seal() {
        assert!(provenance_seal(&json!("not a record")).is_err());
    }

    #[test]
    fn seal_is_prefixed_and_64_hex() {
        let seal = provenance_seal(&json!({})).unwrap();
        let hex = seal.strip_prefix(SEAL_PREFIX).expect("prefix");
        assert_eq!(hex.len(), 64);
        assert!(hex
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
    }
}
