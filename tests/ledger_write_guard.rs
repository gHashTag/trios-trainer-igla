//! Two green verdicts for things that were never checked.
//!
//! Both defects in this file have the same shape: a process reported success
//! for an operation it had not performed, on the two artefacts this crate
//! publishes -- the leaderboard rows and the honey ledger.
//!
//! 1. `matrix_runner` BYPASSED THE WRITE-SIDE BPB SCREEN ENTIRELY.
//!    `src/neon_writer.rs` documents its screen as living "on the WRITE side
//!    so no caller can bypass it", but `write_row_async` opens its own
//!    `tokio_postgres::connect(dsn, NoTls)` and INSERTs straight into
//!    `ssot.bpb_samples`. That path never enters `neon_writer`, so it never
//!    reached `reject_bpb`: no `is_finite` test, no `BPB_SENTINEL_CEILING`
//!    (the `f32::MAX` sentinel that was once published and then divided by),
//!    no `PUBLISHED_BPB_FLOOR`. The same path read `MATRIX_DATABASE_URL` /
//!    `DATABASE_URL` and connected on their presence alone, ignoring the
//!    `TRIOS_LEDGER_WRITE=1` opt-in `neon_writer` declares mandatory.
//!    `matrix_runner` is the binary the nightly workflow runs, so those are
//!    the rows on the public leaderboard.
//!
//! 2. `honey_audit` PRINTED "ledger integrity OK" AFTER CHECKING SCHEMA ONLY,
//!    and an EMPTY ledger passed with `total 0` and a green tick.
//!
//! WHAT IS ASSERTED, AND WHY IT IS ASSERTED FROM OUTSIDE. Ordering is the
//! whole claim -- "the reading is screened BEFORE a socket is opened" -- and
//! ordering is not observable from a unit test of either half. So every case
//! below spawns the shipped binary and reads what it printed. The negative
//! assertion (`connect:` MUST NOT appear) is the one that carries the
//! ordering; `preflight_reaches_the_socket_once_both_gates_clear` is its
//! vacuity guard, proving this mode does reach the socket when it is allowed
//! to, so the absence of `connect:` above means a refusal and not a missing
//! code path.
//!
//! WHY `--write-preflight`. In a real cell the BPB comes out of a training
//! run, and there is no way to ask `cpu_train` for `f32::MAX`, so the sentinel
//! case cannot be provoked end-to-end. `--write-preflight=<bpb>` runs the very
//! same `resolve_write_gate` the write path runs, on a supplied reading, with
//! no training. It is a seam for the ORDER, not a second copy of the rules.
//!
//! OFFLINE BY CONSTRUCTION. Every DSN below points at `127.0.0.1:1`, a port
//! nothing listens on, so the only reachable outcome of an actual connection
//! attempt is an immediate local refusal. No case in this file can write a row
//! anywhere, and no case needs a network.
//!
//! Anchor: phi^2 + phi^-2 = 3 - DOI 10.5281/zenodo.19227877

use std::process::{Command, Output};

use trios_trainer::invariants::PUBLISHED_BPB_FLOOR;
use trios_trainer::neon_writer::{bpb_refusal_reason, BPB_SENTINEL_CEILING};

const MATRIX_RUNNER: &str = env!("CARGO_BIN_EXE_matrix_runner");
const HONEY_AUDIT: &str = env!("CARGO_BIN_EXE_honey_audit");

/// A DSN that resolves instantly and connects to nothing: port 1 on loopback.
/// A test that had to wait for a network timeout would be a test nobody runs.
const UNROUTABLE_DSN: &str = "postgres://u:p@127.0.0.1:1/db";

/// Exit code `matrix_runner` uses for a reading that must not be published.
/// Spelled out rather than imported: this is an assertion about the shipped
/// binary, and a constant read through the library would still pass if the
/// binary and the library disagreed.
const EXIT_BPB_REFUSED: i32 = 9;

/// Exit code `honey_audit` uses for a ledger with zero deposits.
const EXIT_EMPTY_LEDGER: i32 = 46;

/// The substring the `connect:` error carries. Its ABSENCE is the evidence
/// that no socket was opened.
const CONNECT_MARKER: &str = "connect:";

fn run(bin: &str, args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut cmd = Command::new(bin);
    // Start from a known environment: an ambient DATABASE_URL on the
    // developer's machine is exactly the condition under test, and a case that
    // inherited one would be measuring the shell.
    cmd.env_remove("DATABASE_URL");
    cmd.env_remove("MATRIX_DATABASE_URL");
    cmd.env_remove("NEON_DATABASE_URL");
    cmd.env_remove("TRIOS_NEON_DSN");
    cmd.env_remove("TRIOS_DATABASE_URL");
    cmd.env_remove("TRIOS_LEDGER_WRITE");
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.args(args);
    cmd.output().unwrap_or_else(|e| panic!("spawn {bin}: {e}"))
}

fn describe(out: &Output) -> String {
    format!(
        "status={:?}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    )
}

fn combined(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// Run the write gate on a supplied reading, with no training.
///
/// The label guards that run before the write gate are satisfied explicitly --
/// `1597` is a SEED_CANON member, `fp32` is a faithful format, `adamw` is in
/// ALGO_WHITELIST -- because any of them firing would make a case below pass
/// for the wrong reason.
fn preflight(bpb: &str, env: &[(&str, &str)]) -> Output {
    let flag = format!("--write-preflight={bpb}");
    run(
        MATRIX_RUNNER,
        &[
            "--format=fp32",
            "--algo=adamw",
            "--seed=1597",
            "--hidden=64",
            &flag,
        ],
        env,
    )
}

// ---------------------------------------------------------------------------
// 1. matrix_runner: the BPB screen runs before the socket
// ---------------------------------------------------------------------------

/// The reported defect, at the value that caused it: `f32::MAX` was
/// `evaluate`'s "could not measure" sentinel, it passes `is_finite()`, and a
/// row reading 340282346638528859811704183484516925440.0 was published and
/// then divided by.
#[test]
fn sentinel_bpb_is_refused_before_any_connection() {
    let out = preflight(
        "340282346638528859811704183484516925440",
        &[
            ("MATRIX_DATABASE_URL", UNROUTABLE_DSN),
            // Opted IN, so the refusal below cannot be the opt-in gate
            // answering for the screen.
            ("TRIOS_LEDGER_WRITE", "1"),
        ],
    );
    let text = combined(&out);
    assert_eq!(
        out.status.code(),
        Some(EXIT_BPB_REFUSED),
        "a sentinel BPB must exit {EXIT_BPB_REFUSED}.\n{}",
        describe(&out)
    );
    assert!(
        text.contains("BPB REFUSED") && text.contains("BPB_SENTINEL_CEILING"),
        "the refusal must name the screen that refused.\n{}",
        describe(&out)
    );
    assert!(
        !text.contains(CONNECT_MARKER),
        "the screen must run BEFORE the socket: a refused reading may not cost \
         a connection attempt.\n{}",
        describe(&out)
    );
}

/// NaN is the other shape of "no measurement". `f32::max` ignores NaN, which
/// is how a poisoned forward pass once became a finite-looking reading, so the
/// write side must refuse it in its own right.
#[test]
fn non_finite_bpb_is_refused_before_any_connection() {
    for spelling in ["NaN", "inf", "-inf"] {
        let out = preflight(
            spelling,
            &[
                ("MATRIX_DATABASE_URL", UNROUTABLE_DSN),
                ("TRIOS_LEDGER_WRITE", "1"),
            ],
        );
        let text = combined(&out);
        assert_eq!(
            out.status.code(),
            Some(EXIT_BPB_REFUSED),
            "bpb={spelling} must be refused.\n{}",
            describe(&out)
        );
        assert!(
            text.contains("is not finite"),
            "bpb={spelling}: the refusal must name non-finiteness.\n{}",
            describe(&out)
        );
        assert!(
            !text.contains(CONNECT_MARKER),
            "bpb={spelling}: no socket may be opened for a value that is not a \
             reading.\n{}",
            describe(&out)
        );
    }
}

/// The retracted 1.5492 -- "honest Gate-2 pass" in a PR's own investigation
/// notes -- is below `PUBLISHED_BPB_FLOOR` and better than this crate's own
/// champion. It is the value the floor exists for.
#[test]
fn below_the_publication_floor_is_refused_before_any_connection() {
    let out = preflight(
        "1.5492",
        &[
            ("MATRIX_DATABASE_URL", UNROUTABLE_DSN),
            ("TRIOS_LEDGER_WRITE", "1"),
        ],
    );
    let text = combined(&out);
    assert_eq!(
        out.status.code(),
        Some(EXIT_BPB_REFUSED),
        "1.5492 is below the publication floor.\n{}",
        describe(&out)
    );
    assert!(
        text.contains("PUBLISHED_BPB_FLOOR"),
        "the refusal must name the floor.\n{}",
        describe(&out)
    );
    assert!(
        !text.contains(CONNECT_MARKER),
        "no socket for an unpublishable reading.\n{}",
        describe(&out)
    );
}

// ---------------------------------------------------------------------------
// 2. matrix_runner: a DSN alone is not permission to write
// ---------------------------------------------------------------------------

/// A real reading, a real DSN, and no `TRIOS_LEDGER_WRITE`. The binary used to
/// connect and INSERT on the DSN alone.
#[test]
fn a_dsn_without_the_opt_in_writes_nothing_and_says_why() {
    let out = preflight("2.61", &[("MATRIX_DATABASE_URL", UNROUTABLE_DSN)]);
    let text = combined(&out);
    assert!(
        !text.contains(CONNECT_MARKER),
        "a DSN without the opt-in must not open a socket.\n{}",
        describe(&out)
    );
    assert!(
        text.contains("TRIOS_LEDGER_WRITE"),
        "the skip must name the variable that would authorise the write.\n{}",
        describe(&out)
    );
    assert!(
        text.contains("will NOT write to the shared ledger"),
        "the skip must use neon_writer's own wording, so one grep finds every \
         binary that declined.\n{}",
        describe(&out)
    );
    assert!(
        out.status.success(),
        "a run that was never authorised to write promised nothing, so it is \
         exit 0 -- the same shape as having no DSN at all.\n{}",
        describe(&out)
    );
}

/// `DATABASE_URL` is the other name the write path reads, and it is the one
/// most likely to be present for unrelated reasons.
#[test]
fn the_neutral_dsn_name_is_gated_too() {
    let out = preflight("2.61", &[("DATABASE_URL", UNROUTABLE_DSN)]);
    let text = combined(&out);
    assert!(
        !text.contains(CONNECT_MARKER) && text.contains("TRIOS_LEDGER_WRITE"),
        "DATABASE_URL must be gated exactly like MATRIX_DATABASE_URL.\n{}",
        describe(&out)
    );
}

/// VACUITY GUARD. Everything above asserts that `connect:` is ABSENT, which
/// would also hold if this mode simply never opened a socket. It does open
/// one, once both gates clear -- and then fails against the dead port, which
/// is the proof that the socket sits downstream of the screen and the opt-in
/// rather than nowhere.
#[test]
fn preflight_reaches_the_socket_once_both_gates_clear() {
    let out = preflight(
        "2.61",
        &[
            ("MATRIX_DATABASE_URL", UNROUTABLE_DSN),
            ("TRIOS_LEDGER_WRITE", "1"),
        ],
    );
    let text = combined(&out);
    assert!(
        text.contains(CONNECT_MARKER),
        "with a publishable reading and the opt-in set, the connection must be \
         attempted; otherwise the `connect:`-absent assertions above prove \
         nothing.\n{}",
        describe(&out)
    );
    assert!(
        !out.status.success(),
        "a configured, opted-in write that landed nothing is a failure.\n{}",
        describe(&out)
    );
}

// ---------------------------------------------------------------------------
// 3. The exported screen itself
// ---------------------------------------------------------------------------

/// `bpb_refusal_reason` is the public door onto the private `reject_bpb`. It
/// is exercised here rather than only inside `neon_writer` because the point
/// of exporting it is that a writer in ANOTHER crate file can reach it.
#[test]
fn the_exported_screen_refuses_every_non_measurement() {
    for (value, needle) in [
        (f32::MAX, "SENTINEL"),
        (f32::INFINITY, "not finite"),
        (f32::NEG_INFINITY, "not finite"),
        (f32::NAN, "not finite"),
        (BPB_SENTINEL_CEILING, "SENTINEL"),
        (PUBLISHED_BPB_FLOOR, "FLOOR"),
        (1.5492, "FLOOR"),
        (0.0, "FLOOR"),
        (-1.0, "FLOOR"),
    ] {
        let reason = bpb_refusal_reason(value)
            .unwrap_or_else(|| panic!("bpb={value} must be refused, but passed the screen"));
        assert!(
            reason.contains(needle),
            "bpb={value} was refused for the wrong reason: {reason:?} does not \
             mention {needle:?}"
        );
    }
}

/// The screen must not be a wall. 2.61 is the measured raw val_bpb of the
/// verified byte-disjoint tinyshakespeare run at 12000 steps; if this crate
/// ever refuses its own honest calibration, the floor has been raised past the
/// evidence.
#[test]
fn the_exported_screen_admits_real_measurements() {
    for value in [2.61_f32, 2.5193, 2.7727, 3.31, 7.00, 4.2996] {
        assert!(
            bpb_refusal_reason(value).is_none(),
            "bpb={value} is a real measurement from this architecture and must \
             pass: {:?}",
            bpb_refusal_reason(value)
        );
    }
}

// ---------------------------------------------------------------------------
// 4. honey_audit: an empty ledger is not a pass
// ---------------------------------------------------------------------------

/// The binary lives here, not in `src/bin/honey_audit.rs`, because
/// `CARGO_BIN_EXE_*` is only defined when building an integration test. The
/// unit test `empty_blob_audits_clean` still holds and still asserts that
/// `audit_blob("")` is `Ok(empty)`: the PARSER is right to accept an empty
/// blob, and the VERDICT was wrong to call it a pass.
#[test]
fn honey_audit_empty_ledger_exits_non_zero() {
    let dir = std::env::temp_dir().join(format!("honey_audit_empty_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("hive_honey.jsonl");
    std::fs::write(&path, "").expect("write empty ledger");
    let path_arg = format!("--path={}", path.display());

    let out = run(HONEY_AUDIT, &[&path_arg], &[]);
    let text = combined(&out);
    assert_eq!(
        out.status.code(),
        Some(EXIT_EMPTY_LEDGER),
        "an emptied ledger must not read as a pass.\n{}",
        describe(&out)
    );
    assert!(
        text.contains("EMPTY LEDGER"),
        "the refusal must say the ledger was empty.\n{}",
        describe(&out)
    );
    assert!(
        !text.contains("integrity OK"),
        "nothing in this binary checks integrity, so nothing in it may claim \
         integrity is OK.\n{}",
        describe(&out)
    );
    assert!(
        !text.contains("schema OK"),
        "a passing-looking line must not precede the refusal: 0 deposits means \
         nothing was checked, not that a check passed.\n{}",
        describe(&out)
    );

    // A file of blank lines is the same fact wearing different bytes.
    std::fs::write(&path, "\n\n   \n").expect("write blank ledger");
    let out = run(HONEY_AUDIT, &[&path_arg], &[]);
    assert_eq!(
        out.status.code(),
        Some(EXIT_EMPTY_LEDGER),
        "blank lines are skipped, so a file of them holds zero deposits.\n{}",
        describe(&out)
    );

    // The declared escape hatch, and the only way back to exit 0.
    let out = run(HONEY_AUDIT, &[&path_arg, "--allow-empty"], &[]);
    assert!(
        out.status.success(),
        "--allow-empty exists for the fresh-checkout case.\n{}",
        describe(&out)
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// A populated ledger still passes -- and still does not claim integrity.
#[test]
fn honey_audit_reports_what_it_actually_measured() {
    let dir = std::env::temp_dir().join(format!("honey_audit_full_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("hive_honey.jsonl");
    std::fs::write(
        &path,
        "{\"ts\":\"2026-08-06T00:00:00Z\",\"lane\":\"L-ledger-write-truth\",\
         \"agent\":\"test\",\"sha\":\"deadbeef\"}\n",
    )
    .expect("write ledger");
    let path_arg = format!("--path={}", path.display());

    let out = run(HONEY_AUDIT, &[&path_arg], &[]);
    let text = combined(&out);
    assert!(
        out.status.success(),
        "a well-formed deposit must still pass.\n{}",
        describe(&out)
    );
    assert!(
        text.contains("schema OK") && text.contains("integrity NOT checked"),
        "the verdict must state what was measured and what was not.\n{}",
        describe(&out)
    );
    assert!(
        !text.contains("integrity OK"),
        "the sha is recorded and never compared; no line may say otherwise.\n{}",
        describe(&out)
    );

    std::fs::remove_dir_all(&dir).ok();
}
