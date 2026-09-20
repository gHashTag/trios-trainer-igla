//! The second implementation, executed.
//!
//! `src/provenance_seal.rs` states the normative rule: two independent
//! implementations must agree on every published record. The second
//! implementation is `interop/triosckp_reader.py`, and until this file existed
//! NOTHING ran it - no workflow, no Makefile target, no test. A rule that no job
//! enforces is a rule that fails silently, and it did. On 2026-08-06
//! `--verify-all evidence` reported `17 pair(s), 16 passed, 1 failed`, and the
//! single failure was `evidence/r9-headline/12000.json`, the record `README.md`
//! cites as the headline. The disagreement was weeks old and invisible because
//! nothing executed the check.
//!
//! Two tests:
//!
//! 1. `interop_reader_agrees_on_every_published_record` asserts THREE-WAY
//!    equality per tracked record: the seal this crate computes, the seal
//!    `interop/triosckp_reader.py` computes, and the digest
//!    `evidence/SEALS.txt` publishes. Until 2026-08-07 it asserted only the
//!    exit code of `--verify-all`, and the comment here argued that the exit
//!    code was the whole assertion. It was not. The reader computed a seal for
//!    every record and compared it to nothing, so the sweep printed
//!    `17 pair(s), 17 passed, 0 failed` and exited 0 on a tree where
//!    `evidence/window-audit/monolith-200.json` and `segment-200.json` sealed
//!    `85fecb38...` in Rust and `52223ed7...` in Python. The cause was
//!    serde_json parsing the 17-digit literal `3.7924702167510986` one ULP low
//!    without the `float_roundtrip` feature; the records never changed. A test
//!    named "agrees" that cannot observe a disagreement is the defect class
//!    this repository exists to remove.
//!
//! 2. `reader_still_rejects_a_dropped_mandatory_field` proves the fix that made
//!    (1) pass did not do so by weakening the reader. Two fields
//!    (`platform.rustflags_source`, `platform.rustflags_sha256`) were
//!    reclassified as optional BY CONSTRUCTION - they are `Option` in the writer
//!    and `None` on any tree without `.cargo/config.toml`, which is every fresh
//!    clone. That exemption is a hole unless the guard around it still bites, so
//!    this test drops mandatory fields from a COPY of a published record and
//!    asserts the reader still refuses it.
//!
//! NO SKIPS. If `python3` is not on PATH the tests FAIL and say so. A skip that
//! reports success is the defect class this repository exists to remove: it is
//! indistinguishable, in a green CI summary, from a check that ran and passed.
//!
//! The fixtures are written under `CARGO_TARGET_TMPDIR` and never under
//! `evidence/`. A test that mutates the published evidence tree to prove a point
//! about the published evidence tree is a test that can leave the repository
//! carrying its own fixture.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// The repository root, resolved at compile time. The reader takes paths
/// relative to it (`evidence`, `interop/triosckp_reader.py`), so every command
/// below runs with this as its working directory.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn reader_script() -> PathBuf {
    repo_root().join("interop").join("triosckp_reader.py")
}

/// Run the reader with `args` and return its output.
///
/// A missing interpreter is a FAILURE, never a skip, and the panic says which
/// binary was not found rather than leaving the reader of a red log to guess.
fn run_reader(args: &[&str]) -> Output {
    let script = reader_script();
    assert!(
        script.is_file(),
        "the second implementation is missing from the tree: {}",
        script.display()
    );

    let mut command = Command::new("python3");
    command.current_dir(repo_root());
    command.arg(&script);
    for arg in args {
        command.arg(arg);
    }

    match command.output() {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => panic!(
            "python3 is not on PATH, so the second implementation could not be \
             executed and NOTHING was checked. This test fails rather than \
             skips on purpose: a skipped interop gate is indistinguishable from \
             a passing one in a CI summary, and the disagreement this gate was \
             written to catch survived for weeks behind exactly that kind of \
             silence. Install python3 (standard library only is required) or \
             fix PATH."
        ),
        Err(err) => panic!("could not execute python3 {}: {err}", script.display()),
    }
}

fn stdout_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// The tracked sidecars under `evidence/`, straight out of git.
///
/// `git ls-files` and not a directory walk: the published table binds the
/// records this repository SHIPS, and a scratch `.json` a previous run left in
/// `evidence/` is not one of them. A git failure panics rather than returning
/// an empty list -- an empty list would make this gate pass by grading nothing.
fn tracked_evidence_sidecars() -> Vec<String> {
    let output = Command::new("git")
        .current_dir(repo_root())
        .args(["ls-files", "evidence"])
        .output()
        .unwrap_or_else(|err| panic!("could not run `git ls-files evidence`: {err}"));
    assert!(
        output.status.success(),
        "`git ls-files evidence` failed (exit {:?}); this gate refuses to grade \
         an empty list, because grading nothing passes.\n{}",
        output.status.code(),
        stderr_of(&output)
    );
    let paths: Vec<String> = stdout_of(&output)
        .lines()
        .filter(|line| line.ends_with(".json"))
        .map(str::to_owned)
        .collect();
    assert!(
        !paths.is_empty(),
        "git tracks no *.json under evidence/, so this gate would grade nothing"
    );
    paths
}

/// The digests published in `evidence/SEALS.txt`, keyed by record path.
fn published_seals() -> std::collections::BTreeMap<String, String> {
    let path = repo_root().join("evidence").join("SEALS.txt");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("could not read {}: {err}", path.display()));
    let mut table = std::collections::BTreeMap::new();
    for (number, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let (record, digest) = line.split_once('\t').unwrap_or_else(|| {
            panic!(
                "evidence/SEALS.txt:{} is neither a comment nor \
                 <path><TAB>sha256:<hex>: {line:?}",
                number + 1
            )
        });
        let digest = digest.trim();
        assert!(
            digest.len() == "sha256:".len() + 64
                && digest.starts_with("sha256:")
                && digest["sha256:".len()..]
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "evidence/SEALS.txt:{} publishes a malformed digest: {digest:?}",
            number + 1
        );
        assert!(
            table.insert(record.to_owned(), digest.to_owned()).is_none(),
            "evidence/SEALS.txt publishes {record} twice; the reviewer loop's \
             awk prints both values and compares neither"
        );
    }
    table
}

/// The one `provenance-seal <digest>` the Python reader prints for one record.
///
/// Parsed by its label rather than by `tail -1` on every `sha256:` in the
/// output: the reader also prints the CONTAINER digest, and a gate that picked
/// the wrong one of the two would compare a hash of the bytes against a hash of
/// the declaration and fail for a reason that is not the one it names.
fn python_seal_of(sidecar: &str) -> String {
    let container = format!("{}.bin", sidecar.trim_end_matches(".json"));
    let output = run_reader(&[&container, "--sidecar", sidecar]);
    let stdout = stdout_of(&output);
    let mut seals = stdout.lines().filter_map(|line| {
        line.split_once("provenance-seal ")
            .map(|(_, rest)| rest.split_whitespace().next().unwrap_or("").to_owned())
    });
    let seal = seals.next().unwrap_or_else(|| {
        panic!(
            "the Python reader printed no `provenance-seal` line for {sidecar}, \
             so this gate has nothing from the second implementation to \
             compare.\n----- stdout -----\n{stdout}\n----- stderr -----\n{}",
            stderr_of(&output)
        )
    });
    assert!(
        seals.next().is_none(),
        "the Python reader printed more than one `provenance-seal` line for \
         {sidecar}; this gate would be comparing an arbitrary one of them"
    );
    assert!(
        seal.starts_with("sha256:"),
        "the Python reader's seal for {sidecar} is not a sha256 digest: {seal:?}"
    );
    seal
}

/// THE GATE. For every tracked sidecar under `evidence/`, THREE digests must be
/// equal: the one this crate computes, the one the Python reader computes, and
/// the one `evidence/SEALS.txt` publishes.
///
/// Until 2026-08-07 this test asserted only `output.status.success()` on
/// `--verify-all`, which is why its name was a claim and not a measurement. The
/// reader computed a seal for every record and compared it to nothing, so
/// `--verify-all evidence` printed `17 pair(s), 17 passed, 0 failed` and exited
/// 0 on a tree where `evidence/window-audit/monolith-200.json` and
/// `segment-200.json` sealed `85fecb38...` in Rust and `52223ed7...` in Python
/// -- with `evidence/SEALS.txt` publishing the Rust value, so the shipped second
/// implementation called two published records forged. The exit code could not
/// see it. Three-way equality, per record, by name, is the assertion the test's
/// own name always claimed.
///
/// The `--verify-all` run is kept as a second, weaker leg: it covers the
/// container bytes and the field-by-field comparison this loop does not repeat.
#[test]
fn interop_reader_agrees_on_every_published_record() {
    let published = published_seals();
    let sidecars = tracked_evidence_sidecars();
    let mut disagreements = Vec::new();

    for sidecar in &sidecars {
        // Sealed from the FILE'S BYTES, through the same entry point the binary
        // uses. Clause 0 of src/provenance_seal.rs routes a non-checkpoint
        // document to a whole-file digest, and that decision needs the bytes:
        // this loop used to parse the record and call the field-list seal
        // directly, which is how two of the tracked documents here were graded
        // under a rule written for a different kind of file.
        let raw = std::fs::read(repo_root().join(sidecar))
            .unwrap_or_else(|err| panic!("could not read {sidecar}: {err}"));
        let rust = trios_trainer::provenance_seal::seal_document(&raw)
            .unwrap_or_else(|err| panic!("this crate could not seal {sidecar}: {err}"))
            .seal;
        let python = python_seal_of(sidecar);
        let table = published
            .get(sidecar)
            .cloned()
            .unwrap_or_else(|| "<UNPUBLISHED: no line in evidence/SEALS.txt>".to_owned());

        if rust != python || rust != table {
            disagreements.push(format!(
                "  {sidecar}\n    rust      (src/provenance_seal.rs): {rust}\n    \
                 python    (interop/triosckp_reader.py): {python}\n    \
                 published (evidence/SEALS.txt):         {table}"
            ));
        }
    }

    assert!(
        disagreements.is_empty(),
        "the two implementations and the published table do not agree on {} of \
         {} tracked record(s). src/provenance_seal.rs is normative: a \
         rust/python disagreement means the SPECIFICATION is wrong, and a \
         disagreement with the table means the record or the table moved.\n{}",
        disagreements.len(),
        sidecars.len(),
        disagreements.join("\n")
    );
    println!(
        "interop gate: rust == python == evidence/SEALS.txt on all {} tracked \
         record(s)",
        sidecars.len()
    );

    // Second leg: the sweep, which grades the container bytes and the
    // field-by-field sidecar comparison that the seal says nothing about.
    let output = run_reader(&["--verify-all", "evidence"]);
    let stdout = stdout_of(&output);
    assert!(
        output.status.success(),
        "the second implementation disagrees with the published evidence \
         (exit {:?}). src/provenance_seal.rs makes this a specification-level \
         failure, not a reader bug.\n\
         ----- reader stdout -----\n{stdout}\n\
         ----- reader stderr -----\n{}",
        output.status.code(),
        stderr_of(&output)
    );

    // A reader that found nothing would also exit 0. The summary line is the
    // only thing that distinguishes "every pair agreed" from "there were no
    // pairs", and this gate is worthless without that distinction.
    let summary = stdout
        .lines()
        .find(|line| line.starts_with("summary:"))
        .unwrap_or_else(|| {
            panic!("the reader printed no summary line, so nothing proves it graded any record:\n{stdout}")
        });
    assert!(
        !summary.contains("0 pair(s)"),
        "the reader graded zero pairs, which exits 0 and proves nothing: {summary}"
    );
    assert!(
        summary.contains("0 failed"),
        "exit 0 with a non-zero failure count is a contradiction in the reader: {summary}"
    );
    // The sweep must be comparing seals, not merely printing them. `0 matched`
    // with every pair passing is what this file's own history looked like.
    //
    // Read as a NUMBER, not as a substring. The substring form of this
    // assertion (`!summary.contains("0 matched the published table")`) was
    // statistically blind in the other direction: it fired on `20 matched`,
    // because "20 matched..." contains "0 matched...". A guard that fails on
    // its tenth success is not a guard.
    let matched: usize = summary
        .split("seals: ")
        .nth(1)
        .and_then(|rest| rest.split(" matched the published table").next())
        .and_then(|n| n.trim().parse().ok())
        .unwrap_or_else(|| {
            panic!(
                "the sweep's summary does not state how many seals matched the \
                 published table, so its PASS lines say nothing about the \
                 declaration half: {summary}"
            )
        });
    assert!(
        matched >= sidecars.len(),
        "the sweep matched {matched} seal(s) against the published table but git tracks \
         {} sidecar(s) under evidence/; some published declaration was not compared to \
         anything: {summary}",
        sidecars.len()
    );
    // A published line with no `.bin` pair is the case the three degenerate
    // seals lived in: unreached by the container walk, and therefore unchecked
    // until 2026-08-07. The sweep now seals those from their own bytes, and a
    // failure among them must not be reported only in the exit code.
    assert!(
        summary.contains(" 0 failed), "),
        "the sweep reports a failure among the published lines that have no .bin pair: \
         {summary}"
    );
    println!("interop gate: {summary}");
}

/// A record this repository publishes, tracked in git, carrying the full
/// schema/9 field set INCLUDING the optional-by-construction pair. Copying a
/// record that already has those two fields is deliberate: it lets the same
/// fixture prove the exemption fires AND that it is all-or-nothing.
const FIXTURE_STEM: &str = "window-audit/monolith-100";

/// Copy the published pair into `dir` and return `(bin, sidecar_json_value)`.
fn load_fixture(dir: &Path) -> (PathBuf, serde_json::Value) {
    let bin_src = repo_root()
        .join("evidence")
        .join(format!("{FIXTURE_STEM}.bin"));
    let json_src = repo_root()
        .join("evidence")
        .join(format!("{FIXTURE_STEM}.json"));
    assert!(
        bin_src.is_file() && json_src.is_file(),
        "the fixture base is missing from the published evidence: {} / {}",
        bin_src.display(),
        json_src.display()
    );

    let bin_dst = dir.join("fixture.bin");
    std::fs::copy(&bin_src, &bin_dst).expect("copy the fixture container");
    let raw = std::fs::read_to_string(&json_src).expect("read the fixture sidecar");
    let record: serde_json::Value = serde_json::from_str(&raw).expect("fixture sidecar is JSON");
    (bin_dst, record)
}

/// Write `record` beside `bin` under a distinct name and run the reader on the
/// pair. Returns `(exit_ok, combined_output)`.
///
/// BOTH streams, and that is not tidiness: the reader prints `RESULT PASS` on
/// stdout and `RESULT FAIL` on stderr, so a test that read only stdout would
/// find no verdict on exactly the runs it is asserting about. Measured, after a
/// first version of this test asserted on stdout alone and reported "exited
/// non-zero without printing a verdict" for every rejection.
fn read_variant(dir: &Path, bin: &Path, name: &str, record: &serde_json::Value) -> (bool, String) {
    let sidecar = dir.join(format!("{name}.json"));
    std::fs::write(
        &sidecar,
        serde_json::to_string_pretty(record).expect("render the fixture"),
    )
    .expect("write the fixture sidecar");

    let output = run_reader(&[
        bin.to_str().expect("fixture path is UTF-8"),
        "--sidecar",
        sidecar.to_str().expect("fixture path is UTF-8"),
    ]);
    let combined = format!(
        "----- reader stdout -----\n{}\n----- reader stderr -----\n{}",
        stdout_of(&output),
        stderr_of(&output)
    );
    (output.status.success(), combined)
}

/// Drop a field named either `top_level` or `platform.<key>`.
fn drop_field(record: &mut serde_json::Value, name: &str) {
    let object = record.as_object_mut().expect("a record is a JSON object");
    match name.split_once('.') {
        None => {
            assert!(
                object.remove(name).is_some(),
                "the fixture does not carry {name}, so dropping it proves nothing"
            );
        }
        Some((head, tail)) => {
            let nested = object
                .get_mut(head)
                .and_then(serde_json::Value::as_object_mut)
                .unwrap_or_else(|| panic!("the fixture has no {head} object"));
            assert!(
                nested.remove(tail).is_some(),
                "the fixture does not carry {name}, so dropping it proves nothing"
            );
        }
    }
}

/// BREAK THE GUARD AND WATCH IT FAIL.
///
/// Five variants of one published record. The control has to pass or the four
/// rejections would only prove the fixture machinery is broken; the last one is
/// the exemption itself, and it is asserted to PASS so that a later change that
/// quietly re-mandates the pair is caught here rather than by the next clean
/// clone.
#[test]
fn reader_still_rejects_a_dropped_mandatory_field() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("interop_reader_gate");
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    let (bin, pristine) = load_fixture(&dir);

    // Control. An unmodified copy of a published record must be accepted, or
    // every rejection below is evidence about the copy and not about the check.
    let (ok, stdout) = read_variant(&dir, &bin, "control", &pristine);
    assert!(
        ok,
        "the unmodified fixture was rejected, so nothing below is evidence \
         about the guard:\n{stdout}"
    );

    // Mandatory fields, one per schema rung that the /9 record reaches. Each is
    // dropped from a fresh copy so the variants cannot mask one another.
    for field in [
        "platform.arch",
        "platform.os",
        "platform.toolchain",
        "platform.remap_applied",
        "platform.source_digest_scope",
        "git_untracked",
        "format_faithful",
        "source_sha256",
    ] {
        let mut broken = pristine.clone();
        drop_field(&mut broken, field);
        let (ok, stdout) = read_variant(&dir, &bin, "dropped", &broken);
        assert!(
            !ok,
            "the reader ACCEPTED a record with the mandatory field {field} \
             dropped. The optional-by-construction exemption has been widened \
             into a hole.\n{stdout}"
        );
        assert!(
            stdout.contains("RESULT FAIL"),
            "the reader exited non-zero for {field} without printing a verdict, \
             which is an error rather than a rejection:\n{stdout}"
        );
    }

    // HALF the optional pair. src/checkpoint.rs states rustflags_sha256 is
    // "`None` exactly when `rustflags_source` is `None`", so exactly one of the
    // two present is a state the writer cannot produce: a dropped field wearing
    // the exemption's clothes. It must be refused.
    for half in ["platform.rustflags_sha256", "platform.rustflags_source"] {
        let mut broken = pristine.clone();
        drop_field(&mut broken, half);
        let (ok, stdout) = read_variant(&dir, &bin, "half-pair", &broken);
        assert!(
            !ok,
            "the reader ACCEPTED a record carrying only half of the \
             optional-by-construction pair ({half} dropped). All-or-nothing is \
             what keeps the exemption from being a free deletion.\n{stdout}"
        );
    }

    // THE EXEMPTION. Both gone is the state of every record minted from a clean
    // clone, and it must be accepted and REPORTED as absent by construction -
    // not merely tolerated in silence, which would be indistinguishable from the
    // reader having stopped looking.
    let mut clean_clone = pristine.clone();
    drop_field(&mut clean_clone, "platform.rustflags_sha256");
    drop_field(&mut clean_clone, "platform.rustflags_source");
    let (ok, stdout) = read_variant(&dir, &bin, "clean-clone", &clean_clone);
    assert!(
        ok,
        "the reader rejected a record minted from a clean clone. \
         `.cargo/config.toml` is gitignored, so this is the state of EVERY \
         fresh checkout and no re-mint could ever satisfy the check.\n{stdout}"
    );
    assert!(
        stdout.contains("ABSENT BY CONSTRUCTION"),
        "the reader passed the record without naming the class, so \
         'unmeasurable in this scope' prints the same as 'not dropped at \
         all':\n{stdout}"
    );
}
