//! Recompute, do not quote: the population argument that justifies redefining
//! `path` inside `trios-checkpoint-record/9`.
//!
//! `src/checkpoint.rs` folds a BREAKING redefinition of `path` -- absolute
//! location -> scope-relative, or `outside-scope:<file name>` -- into an
//! existing version number instead of minting `/10`. That is only defensible if
//! no reader holds a `/9` document the widening silently reinterprets. Until
//! this file existed, the whole defence was a hand-copied census frozen in a doc
//! comment on 2026-08-05: `git log -S` returning no commits, and 0 records
//! tagged `/9` across 154 files. Committing the constant falsified both halves
//! the same day (see the note above `CHECKPOINT_RECORD_SCHEMA`). A number that
//! looks authoritative, sits three lines above the constant it describes, and is
//! wrong by 20 is this repository's own named defect class: a hand-copied list.
//!
//! So this file asserts the STABLE property rather than the dated reading:
//! every `/9`-or-later record that exists anywhere in the tree carries a
//! scope-relative `path`. That claim is what makes the redefinition safe, it is
//! recomputed on every `cargo test`, and it fails the moment an absolute `path`
//! reaches a `/9` record -- which is the only way a reader could come to hold a
//! document this widening reinterprets.
//!
//! Two trees, and they are NOT interchangeable:
//!
//! * `evidence/` is git-tracked, so a fresh clone has it. The pass rests here.
//! * `checkpoints/` is gitignored working-tree output. It is walked as a BONUS
//!   when present -- more records to falsify against -- and its absence is
//!   never a failure. A guard whose population lives only in a gitignored
//!   directory is a guard that evaporates on clone.
//!
//! And a guard over an empty population is a vacuous pass, indistinguishable
//! from a measurement, so `the_tracked_schema_nine_population_is_not_empty`
//! fails if `evidence/` stops carrying enough git-TRACKED `/9` records to test.
//!
//! TRACKED is the operative word, and it partitions rather than rejects. An
//! UNTRACKED `/9` record under `evidence/` is not a violation of anything: it is
//! ordinary working-tree output from a pass that has not committed yet, and it
//! neither satisfies nor breaks the scope-relative `path` convention. What it
//! cannot do is prop up the floor, because the floor exists to guarantee the
//! convention is asserted over a population a FRESH CLONE still has. So the
//! floor applies to the tracked subset only, and the untracked records are
//! printed as a NOTE. They are still checked for the convention itself by
//! `every_emitted_schema_nine_record_carries_a_scope_relative_path`, which walks
//! the whole tree on disk and does not care who tracks what.
//!
//! Population recomputed by hand on 2026-08-06 and re-derived by this file on
//! every run (`git ls-files --others --exclude-standard -- evidence` against a
//! walk of the tree). Four `/9` records exist under `evidence/`:
//!
//! * TRACKED, 3, all introduced by commit `e3edf9f`, each carrying
//!   `"format_faithful": true` -- `evidence/window-audit/monolith-100.json`,
//!   `evidence/window-audit/monolith-200.json`,
//!   `evidence/window-audit/segment-200.json`.
//! * UNTRACKED, 1 -- `evidence/r9-headline/12000.json`, produced by a different
//!   pass. Counted in the census, excluded from the floor.
//!
//! Those two figures are what this file asserts and prints; they are written
//! down here so a reader can check the code against a stated reading, NOT so the
//! reading can be quoted instead of running the test.
//!
//! Run with output to see the recomputed generation census:
//!
//! ```text
//! cargo test --test schema_nine_path_is_scope_relative -- --nocapture
//! ```
//!
//! The census is printed rather than asserted because its TOTAL is a moving
//! denominator: `checkpoints/` grows whenever anyone trains. Only the numbers
//! this file asserts are load-bearing.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use trios_trainer::checkpoint::PATH_OUTSIDE_SCOPE_PREFIX;

/// The schema family whose generation number carries the `path` convention.
const RECORD_SCHEMA_PREFIX: &str = "trios-checkpoint-record/";

/// First generation in which `path` is scope-relative. `/1`-`/8` published an
/// absolute path and are excluded from the path assertion for that reason --
/// they are not violations, they are the old contract.
const FIRST_SCOPE_RELATIVE_GENERATION: u32 = 9;

/// The floor on git-TRACKED `/9` records under `evidence/`.
///
/// Measured 2026-08-06 over `evidence/`: FOUR files carry a `/9` record, of
/// which THREE are tracked -- `window-audit/monolith-100.json`,
/// `window-audit/monolith-200.json`, `window-audit/segment-200.json`, all
/// introduced by commit `e3edf9f`, each carrying `"format_faithful": true`. The
/// fourth, `r9-headline/12000.json`, is untracked working-tree output and is
/// excluded from this floor by construction, not by exception.
///
/// An earlier brief counted `evidence/xarch-aarch64-reference/PROVENANCE.txt`
/// into the tracked population. It is not a record: it is prose which MENTIONS
/// the string `trios-checkpoint-record/9` while describing an artifact whose
/// sidecar is `/4`, and it has no `path` field to check. Re-measured 2026-08-06:
/// `grep -rl trios-checkpoint-record/9 evidence/` returns SIX files, of which
/// only FOUR parse as JSON objects tagged `/9`; the two extras are that file and
/// `evidence/r9-headline/PROVENANCE.txt`. `grep` cannot tell a record from a
/// sentence about records, which is how the count inflates. This file therefore
/// parses instead of grepping, and the discrepancy is written down rather than
/// silently adjusted: an unexplained threshold is the same defect as an
/// unexplained census.
///
/// Deliberately a FLOOR and not an equality: another pass may legitimately
/// commit a fourth record, and an equality would turn correct work into a red
/// test. Raise it when the tracked population grows for good; never lower it to
/// make a red test green -- a shrinking tracked `/9` population under
/// `evidence/` means published evidence was deleted, which is the thing worth
/// failing over.
const MIN_TRACKED_SCHEMA_NINE_RECORDS: usize = 3;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// One `*.json` file, read.
struct SidecarFile {
    /// Repo-relative, for messages a reader can act on.
    rel: String,
    /// `schema` as a string, `None` when the file has no string `schema` key.
    schema: Option<String>,
    /// Top-level `path`, `None` when absent.
    record_path: Option<String>,
    /// `trainer.path`, `None` when absent. Same convention, same prefix.
    trainer_path: Option<String>,
}

impl SidecarFile {
    /// `N` from `trios-checkpoint-record/N`, `None` for any other schema.
    fn generation(&self) -> Option<u32> {
        self.schema
            .as_deref()?
            .strip_prefix(RECORD_SCHEMA_PREFIX)?
            .parse()
            .ok()
    }
}

/// What a walk of one tree found. `files` is empty when the tree is absent;
/// `present` says which of the two it was, so a missing tree can never be
/// mistaken for a tree with nothing in it.
struct Tree {
    name: &'static str,
    present: bool,
    files: Vec<SidecarFile>,
    /// Files that are not readable as a JSON object. Reported, never ignored:
    /// a corrupted record would otherwise leave the population silently.
    unreadable: Vec<String>,
}

fn collect_json_paths(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => panic!("cannot read directory {}: {}", dir.display(), e),
    };
    for entry in entries {
        let entry =
            entry.unwrap_or_else(|e| panic!("cannot read entry in {}: {}", dir.display(), e));
        let path = entry.path();
        // Symlinks are not followed: a link could point outside the tree and
        // make the census describe something other than the named directory.
        let meta = match std::fs::symlink_metadata(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.file_type().is_symlink() {
            continue;
        }
        if meta.is_dir() {
            collect_json_paths(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("json") {
            out.push(path);
        }
    }
}

fn walk(name: &'static str) -> Tree {
    let root = repo_root();
    let dir = root.join(name);
    if !dir.is_dir() {
        return Tree {
            name,
            present: false,
            files: Vec::new(),
            unreadable: Vec::new(),
        };
    }
    let mut paths = Vec::new();
    collect_json_paths(&dir, &mut paths);
    paths.sort();

    let mut files = Vec::new();
    let mut unreadable = Vec::new();
    for path in paths {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let raw = match std::fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(e) => {
                unreadable.push(format!("{rel}: cannot read: {e}"));
                continue;
            }
        };
        let value: serde_json::Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => {
                unreadable.push(format!("{rel}: not JSON: {e}"));
                continue;
            }
        };
        let object = match value.as_object() {
            Some(o) => o,
            None => {
                unreadable.push(format!("{rel}: JSON is not an object"));
                continue;
            }
        };
        files.push(SidecarFile {
            rel,
            schema: object
                .get("schema")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            record_path: object
                .get("path")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            trainer_path: object
                .get("trainer")
                .and_then(|v| v.get("path"))
                .and_then(|v| v.as_str())
                .map(str::to_string),
        });
    }
    Tree {
        name,
        present: true,
        files,
        unreadable,
    }
}

/// `Ok(())` when `value` obeys the `/9` convention, `Err(why)` otherwise.
///
/// The convention, from `src/checkpoint.rs`: scope-relative, or the literal
/// `outside-scope:` prefix followed by a bare FILE NAME. A name is not a
/// location, so the prefixed form is checked to carry no separator -- otherwise
/// `outside-scope:/Users/...` would sail through as "documented".
fn violates_scope_relative_convention(value: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err("empty, which reads as a measurement that came out blank".to_string());
    }
    if let Some(name) = value.strip_prefix(PATH_OUTSIDE_SCOPE_PREFIX) {
        if name.is_empty() {
            return Err(format!(
                "{PATH_OUTSIDE_SCOPE_PREFIX} with no file name after it"
            ));
        }
        if name.contains('/') || name.contains('\\') {
            return Err(format!(
                "{PATH_OUTSIDE_SCOPE_PREFIX} must be followed by a bare file name, not a location"
            ));
        }
        return Ok(());
    }
    if value.starts_with('/') {
        return Err("absolute POSIX path".to_string());
    }
    if value.starts_with('\\') {
        return Err("absolute Windows path".to_string());
    }
    let bytes = value.as_bytes();
    if bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/')
    {
        return Err("absolute Windows path with a drive letter".to_string());
    }
    if value.split(['/', '\\']).any(|c| c == "..") {
        return Err("climbs out of the digest scope with `..`".to_string());
    }
    Ok(())
}

fn trees() -> Vec<Tree> {
    vec![walk("evidence"), walk("checkpoints")]
}

/// Files under `evidence/` that git does not track, best effort.
///
/// `evidence/` is a tracked tree; a `/9` record sitting there UNTRACKED would
/// inflate the population floor while being absent from a fresh clone, so the
/// caller uses this to partition the population before applying the floor.
///
/// `None` means the question could not be answered -- no `.git`, or git refused
/// -- and it is NOT the same as "nothing is untracked". The caller therefore
/// FAILS on `None` instead of falling back to the on-disk count: that fallback
/// would let an untracked file satisfy a floor whose entire purpose is
/// clone-durability, i.e. pass for a reason other than the one the test names.
fn untracked_under_evidence() -> Option<Vec<String>> {
    if !repo_root().join(".git").exists() {
        return None;
    }
    let out = Command::new("git")
        .args([
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "evidence",
        ])
        .current_dir(repo_root())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::to_string)
            .collect(),
    )
}

/// THE claim the doc comment rests on, recomputed.
#[test]
fn every_emitted_schema_nine_record_carries_a_scope_relative_path() {
    let mut checked = 0usize;
    let mut violations = Vec::new();

    for tree in trees() {
        for file in &tree.files {
            let generation = match file.generation() {
                Some(g) if g >= FIRST_SCOPE_RELATIVE_GENERATION => g,
                _ => continue,
            };
            checked += 1;
            let record_path = match &file.record_path {
                Some(p) => p,
                None => {
                    violations.push(format!(
                        "{} (/{generation}): no `path` field at all",
                        file.rel
                    ));
                    continue;
                }
            };
            if let Err(why) = violates_scope_relative_convention(record_path) {
                violations.push(format!(
                    "{} (/{generation}): path = {record_path:?} -- {why}",
                    file.rel
                ));
            }
            // `trainer.path` took the same widening one generation earlier and
            // shares the prefix, so a /9 record breaking it breaks the same
            // convention this note defends.
            if let Some(trainer_path) = &file.trainer_path {
                if let Err(why) = violates_scope_relative_convention(trainer_path) {
                    violations.push(format!(
                        "{} (/{generation}): trainer.path = {trainer_path:?} -- {why}",
                        file.rel
                    ));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "{} of {checked} record(s) at generation /{FIRST_SCOPE_RELATIVE_GENERATION} or later \
         carry a `path` that is NOT scope-relative. The redefinition of `path` inside /9 \
         (src/checkpoint.rs, above CHECKPOINT_RECORD_SCHEMA) is justified ONLY by there being \
         no such record; each line below is a reader holding a document the widening silently \
         reinterprets:\n  {}",
        violations.len(),
        violations.join("\n  ")
    );
}

/// A guard over an empty population passes for the wrong reason.
///
/// So this asserts one thing and says exactly that thing: the number of `/9`
/// records under `evidence/` that git TRACKS is at least
/// `MIN_TRACKED_SCHEMA_NINE_RECORDS`. Untracked `/9` records are reported and
/// excluded -- excluding them is the point, failing on them is not.
#[test]
fn the_tracked_schema_nine_population_is_not_empty() {
    let evidence = walk("evidence");
    assert!(
        evidence.present,
        "evidence/ is missing from {}. It is a TRACKED tree; a clone has it.",
        repo_root().display()
    );

    let nine: Vec<&str> = evidence
        .files
        .iter()
        .filter(|f| f.generation() == Some(FIRST_SCOPE_RELATIVE_GENERATION))
        .map(|f| f.rel.as_str())
        .collect();

    // No answer is not the answer "nothing is untracked". Without it the floor
    // below cannot be about tracked records at all, and applying it to the
    // on-disk count would report success for a claim never measured.
    let untracked = untracked_under_evidence().unwrap_or_else(|| {
        panic!(
            "the tracked-ness of the /9 population under evidence/ could not be determined: \
             `git ls-files --others --exclude-standard -- evidence` is unavailable or failed in \
             {}. This test asserts a floor on the TRACKED population; applying that floor to the \
             {} record(s) merely present on disk would let uncommitted files satisfy a guard \
             whose whole purpose is that a fresh clone still has them -- a pass for a reason \
             other than the one this test names. It fails instead. Run it from a git checkout. \
             On disk, tracked-ness unknown: {:?}",
            repo_root().display(),
            nine.len(),
            nine
        )
    });

    let is_tracked = |rel: &str| !untracked.iter().any(|u| u.as_str() == rel);
    let tracked: Vec<&str> = nine.iter().copied().filter(|rel| is_tracked(rel)).collect();
    let stray: Vec<&str> = nine
        .iter()
        .copied()
        .filter(|rel| !is_tracked(rel))
        .collect();

    if !stray.is_empty() {
        println!(
            "NOTE: {} /9 record(s) under evidence/ are UNTRACKED and are excluded from the floor \
             below, because a fresh clone does not have them. This is a note, not a finding: an \
             untracked record neither satisfies nor violates the scope-relative `path` \
             convention, and it is still checked for that convention by \
             `every_emitted_schema_nine_record_carries_a_scope_relative_path`, which walks the \
             tree on disk. Excluded: {:?}",
            stray.len(),
            stray
        );
    }

    assert!(
        tracked.len() >= MIN_TRACKED_SCHEMA_NINE_RECORDS,
        "the git-TRACKED /9 population under evidence/ is {} record(s), below the floor of {}. \
         The floor is what keeps this file from asserting the scope-relative `path` convention \
         over a population a fresh clone does not have -- a vacuous pass indistinguishable from \
         a measurement -- so it fails instead. Tracked and counted: {:?}. Untracked, excluded, \
         and NOT the cause of this failure: {:?}",
        tracked.len(),
        MIN_TRACKED_SCHEMA_NINE_RECORDS,
        tracked,
        stray
    );

    println!(
        "/9 records under evidence/: {} tracked (floor {}), {} untracked and excluded",
        tracked.len(),
        MIN_TRACKED_SCHEMA_NINE_RECORDS,
        stray.len()
    );
    for rel in &tracked {
        println!("  tracked    {rel}");
    }
    for rel in &stray {
        println!("  untracked  {rel}");
    }
}

/// The census the doc comment used to freeze, regenerated on demand.
///
/// Asserts almost nothing on purpose. The TOTAL is a moving denominator --
/// `checkpoints/` is gitignored and grows whenever anyone trains -- so it must
/// never be quoted in outreach. The one figure here that cannot move is the
/// count of records graded INCOMPARABLE by `ckpt_replay` (`/1` plus `/3`): no
/// new legacy record can ever be produced, so that number only falls if
/// evidence is deleted.
#[test]
fn print_record_generation_census() {
    let mut totals: BTreeMap<String, usize> = BTreeMap::new();
    let mut grand = 0usize;

    println!("record-generation census, recomputed by this test run (not transcribed):");
    for tree in trees() {
        if !tree.present {
            println!(
                "  {}/ ABSENT -- not a failure; see the module note",
                tree.name
            );
            continue;
        }
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut untagged = 0usize;
        for file in &tree.files {
            match &file.schema {
                Some(s) => {
                    *counts.entry(s.clone()).or_default() += 1;
                    *totals.entry(s.clone()).or_default() += 1;
                    grand += 1;
                }
                None => untagged += 1,
            }
        }
        println!(
            "  {}/  {} *.json, {} schema-tagged, {} untagged, {} unreadable",
            tree.name,
            tree.files.len() + tree.unreadable.len(),
            tree.files.len() - untagged,
            untagged,
            tree.unreadable.len()
        );
        for (schema, n) in &counts {
            println!("      {n:>4}  {schema}");
        }
        for note in &tree.unreadable {
            println!("      UNREADABLE  {note}");
        }
    }

    println!("  BOTH TREES  {grand} schema-tagged files");
    for (schema, n) in &totals {
        println!("      {n:>4}  {schema}");
    }

    let legacy = |n: u32| {
        totals
            .get(&format!("{RECORD_SCHEMA_PREFIX}{n}"))
            .copied()
            .unwrap_or(0)
    };
    let incomparable = legacy(1) + legacy(3);
    println!(
        "  CANNOT GROW: {incomparable} legacy records (/1 + /3) grade INCOMPARABLE; no new \
         legacy record can ever be produced. Measured 2026-08-06: all of them live under \
         checkpoints/, so this line reads 0 in a fresh clone. Cannot grow is not durable."
    );
    println!(
        "  MOVING: the {grand}-file total. checkpoints/ is gitignored and grows with every \
         training run -- never quote this denominator outside the repository."
    );
}
