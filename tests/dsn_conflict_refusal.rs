//! Two different database DSNs in scope must stop the process, not be ranked.
//!
//! WHAT THIS PINS, and why it needed a test at all.
//!
//! `neon_writer::resolve_dsn` read `DATABASE_URL -> NEON_DATABASE_URL ->
//! TRIOS_NEON_DSN -> TRIOS_DATABASE_URL`, so the project's OWN namespaced
//! variable was consulted LAST. On 2026-08-05 a run launched with
//! `TRIOS_DATABASE_URL` deliberately pointed at a dead host -- plus
//! `TRIOS_LEDGER_WRITE=1` -- on a machine whose shell exported an unrelated
//! `DATABASE_URL=postgresql://playra@localhost:5432/trios`, printed
//! `[ledger] connected OK` and inserted a real row into the operator's live
//! local database (`public.bpb_samples`, canon_name=trios-train-rng47, seed=47,
//! step=1). The redirection did nothing and said nothing. That is the defect
//! class this crate exists to catch: a process reporting success for an
//! operation the operator had explicitly sent elsewhere.
//!
//! The `TRIOS_LEDGER_WRITE` opt-in is NOT the missing guard here and is not
//! weakened by this file -- it was set, correctly, by the operator who wanted a
//! write. It answers "may this process write"; it cannot answer "to which
//! database".
//!
//! WHY A SUBPROCESS. The refusal is a `std::process::exit`, so it cannot be
//! provoked in-process without taking the test binary down with it. The pure
//! halves (`redact_dsn`, `dsn_conflict_message`) are unit-tested inside
//! `src/neon_writer.rs`; this file proves the whole process actually stops.
//!
//! WHY `env_clear`. Every child below starts from an EMPTY environment and is
//! given back only what the case is about. Inheriting this machine's ambient
//! `DATABASE_URL` is precisely the condition under test, so a test that
//! inherited it would be measuring the developer's shell.
//!
//! EVERY BINARY THAT CAN PICK A DATABASE (added 2026-08-05). `trios-train` was
//! the first binary put behind the gate and for a while it was the only one:
//! `tri`, `scarab` and `matrix_runner` each still built their OWN fallback
//! chain and connected on the first hit. `scarab` had the exact shape that was
//! fixed in `trios-train` -- `DATABASE_URL -> NEON_DATABASE_URL ->
//! TRIOS_DATABASE_URL` feeding `connect_with_retry` -- so a worker on a host
//! exporting a second DSN would claim strategies out of, and write rows into, a
//! database nobody had named, while printing `ready`. `matrix_runner`'s chain
//! (`MATRIX_DATABASE_URL -> DATABASE_URL`) reached a name the gate could not
//! even see, which is why `MATRIX_DATABASE_URL` is now a member of
//! `neon_writer::DSN_ENV_VARS`. The per-binary cases below assert the refusal
//! AND that nothing of the binary's own output precedes it: the refusal claims
//! "no database was contacted and nothing was written", and an empty stdout in
//! front of it is what makes that claim checkable from the outside.
//!
//! Anchor: phi^2 + phi^-2 = 3 - DOI 10.5281/zenodo.19227877

use std::process::{Command, Output};

/// sysexits.h `EX_CONFIG`, mirrored from `neon_writer::DSN_CONFLICT_EXIT_CODE`.
///
/// Spelled out rather than imported: these are subprocess assertions about the
/// shipped binary, and a constant read through the library would still pass if
/// the binary and the library ever disagreed.
const DSN_CONFLICT_EXIT_CODE: i32 = 78;

/// The line that opens the refusal. Asserted verbatim so a future change that
/// keeps the exit status but drops the explanation fails here.
const REFUSAL_MARKER: &str = "the environment names more than one database";

/// A one-step run from a completely empty environment.
///
/// `--steps 1` with a small model is the cheapest path that still reaches the
/// evaluation, which is the first thing in a real run that resolves a DSN. The
/// corpus paths are absolute so the child does not depend on its cwd, and
/// checkpoint writing is disabled so no case leaves an artifact behind.
fn trainer(args: &[(&str, &str)]) -> Output {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_trios-train"));
    cmd.env_clear();
    cmd.current_dir(root);
    // PATH and HOME are the only ambient facts restored, and neither can decide
    // which database is written: a child with no PATH at all is a different
    // machine from the one the operator ran on.
    if let Ok(path) = std::env::var("PATH") {
        cmd.env("PATH", path);
    }
    if let Ok(home) = std::env::var("HOME") {
        cmd.env("HOME", home);
    }
    cmd.env("TRIOS_CHECKPOINT_DISABLE", "1");
    for (k, v) in args {
        cmd.env(k, v);
    }
    cmd.arg("--steps")
        .arg("1")
        .arg("--hidden")
        .arg("64")
        // 8 is the floor `train_loop` enforces (it panics below it): the
        // cheapest evaluation that this trainer considers informative.
        .arg("--eval-chunks")
        .arg("8")
        .arg("--train-data")
        .arg(format!("{root}/data/tiny_shakespeare.txt"))
        .arg("--val-data")
        .arg(format!("{root}/data/tiny_shakespeare_val.txt"));
    cmd.output().expect("spawn trios-train")
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

/// The reported case, reproduced: the generic name and the project's own name
/// point at different databases. Neither may win.
#[test]
fn dsn_conflict_refuses_and_names_both_variables() {
    let out = trainer(&[
        ("DATABASE_URL", "postgresql://a:pw1@host-one:5432/db1"),
        ("TRIOS_DATABASE_URL", "postgresql://a:pw2@host-two:5432/db2"),
        // The write opt-in IS set, exactly as it was on 2026-08-05. The refusal
        // must not depend on withholding it: an operator who asked for a write
        // and named two databases is the whole failure mode.
        ("TRIOS_LEDGER_WRITE", "1"),
    ]);

    assert!(
        !out.status.success(),
        "a conflicted environment must not produce a successful run.\n{}",
        describe(&out)
    );
    assert_eq!(
        out.status.code(),
        Some(DSN_CONFLICT_EXIT_CODE),
        "the refusal must be distinguishable from a run that failed to record \
         itself (exit 1).\n{}",
        describe(&out)
    );

    let text = combined(&out);
    assert!(
        text.contains(REFUSAL_MARKER),
        "the refusal must say what is wrong.\n{}",
        describe(&out)
    );
    for needle in [
        "DATABASE_URL",
        "TRIOS_DATABASE_URL",
        "host-one:5432/db1",
        "host-two:5432/db2",
    ] {
        assert!(
            text.contains(needle),
            "the refusal must name {needle} so the operator knows which variable \
             to unset.\n{}",
            describe(&out)
        );
    }
    for secret in ["pw1", "pw2"] {
        assert!(
            !text.contains(secret),
            "a password reached the output.\n{}",
            describe(&out)
        );
    }
    // The failure being fixed is a row written to the database that was NOT
    // named. Nothing may reach either candidate, so neither of the two lines
    // `neon_writer::db()` prints around a connect may appear. Matched with the
    // `[ledger] ` prefix, verbatim as `db()` emits them: the refusal text
    // itself quotes the phrase "connected OK" when explaining what used to
    // happen, and a bare substring would match that quotation instead.
    for line in ["[ledger] connecting via SeaORM", "[ledger] connected OK"] {
        assert!(
            !text.contains(line),
            "no connection may be opened while the target database is \
             ambiguous, but {line:?} was printed.\n{}",
            describe(&out)
        );
    }
}

/// Every ordered pair of the four aliases, so the fix cannot be a special case
/// for the two variables that happened to collide on 2026-08-05.
#[test]
fn dsn_conflict_refuses_for_any_two_disagreeing_aliases() {
    const ALIASES: [&str; 4] = [
        "DATABASE_URL",
        "NEON_DATABASE_URL",
        "TRIOS_NEON_DSN",
        "TRIOS_DATABASE_URL",
    ];
    for (i, first) in ALIASES.iter().enumerate() {
        for second in ALIASES.iter().skip(i + 1) {
            let out = trainer(&[
                (first, "postgresql://a@host-one:5432/db1"),
                (second, "postgresql://a@host-two:5432/db2"),
                ("TRIOS_LEDGER_WRITE", "1"),
            ]);
            assert_eq!(
                out.status.code(),
                Some(DSN_CONFLICT_EXIT_CODE),
                "{first} vs {second} must refuse.\n{}",
                describe(&out)
            );
            let text = combined(&out);
            assert!(
                text.contains(first) && text.contains(second),
                "{first} vs {second}: both names must appear.\n{}",
                describe(&out)
            );
        }
    }
}

/// The other half of the rule, and the one that keeps the fix deployable: a
/// Railway service exports the same DSN under two names, and that is agreement,
/// not ambiguity. Refusing it would take the fleet down.
#[test]
fn dsn_conflict_absent_when_two_variables_agree() {
    let same = "postgresql://a:pw@no-such-host.invalid:5432/db1";
    let out = trainer(&[
        ("DATABASE_URL", same),
        ("TRIOS_DATABASE_URL", same),
        // Deliberately NOT opted in: this case is about the conflict gate, and
        // without the opt-in the run reaches the end without touching a
        // database, so `.invalid` never has to be resolved and the assertion
        // does not depend on how long a failed lookup takes.
    ]);

    let text = combined(&out);
    assert!(
        !text.contains(REFUSAL_MARKER),
        "identical values are not a conflict.\n{}",
        describe(&out)
    );
    assert_ne!(
        out.status.code(),
        Some(DSN_CONFLICT_EXIT_CODE),
        "identical values must not trigger the conflict exit.\n{}",
        describe(&out)
    );
    // Guard against a vacuous pass: the run really did happen, so the absence
    // of the refusal is a fact about the gate and not about a crash somewhere
    // earlier.
    assert!(
        out.status.success(),
        "an unambiguous, un-opted-in run must complete normally.\n{}",
        describe(&out)
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("DONE: seed=47"),
        "expected the trainer to finish a run.\n{}",
        describe(&out)
    );
}

/// One variable alone is the ordinary case and must be unaffected.
#[test]
fn dsn_conflict_absent_for_a_single_dsn() {
    let out = trainer(&[(
        "TRIOS_DATABASE_URL",
        "postgresql://a:pw@no-such-host.invalid:5432/db1",
    )]);
    let text = combined(&out);
    assert!(
        !text.contains(REFUSAL_MARKER),
        "one DSN cannot be ambiguous.\n{}",
        describe(&out)
    );
    assert!(
        out.status.success(),
        "a single un-opted-in DSN must still exit 0.\n{}",
        describe(&out)
    );
}

// ---------------------------------------------------------------------------
// The other three binaries that can pick a database.
// ---------------------------------------------------------------------------

/// The first line of the refusal, verbatim. Asserted as a PREFIX of stderr, so
/// a binary that printed a banner, parsed flags or opened a connection before
/// reaching the gate fails here even if it eventually refuses.
const REFUSAL_FIRST_LINE: &str = "[ledger] FATAL: the environment names more than one database.";

/// Run a shipped binary from an empty environment with only `envs` restored.
///
/// The same shape as [`trainer`], generalised: `PATH` and `HOME` are the only
/// ambient facts a child keeps, because neither can decide which database is
/// written, and a child with no `PATH` at all is a different machine from the
/// one the operator ran on.
fn gated_binary(exe: &str, args: &[&str], envs: &[(&str, &str)]) -> Output {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut cmd = Command::new(exe);
    cmd.env_clear();
    cmd.current_dir(root);
    if let Ok(path) = std::env::var("PATH") {
        cmd.env("PATH", path);
    }
    if let Ok(home) = std::env::var("HOME") {
        cmd.env("HOME", home);
    }
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.args(args);
    cmd.output().unwrap_or_else(|e| panic!("spawn {exe}: {e}"))
}

/// Everything a refusal from any binary must satisfy.
///
/// `first` and `second` are the two variable names the case set to different
/// values; both must be named so the operator knows what to unset.
fn assert_refused(out: &Output, first: &str, second: &str) {
    assert_eq!(
        out.status.code(),
        Some(DSN_CONFLICT_EXIT_CODE),
        "a conflicted environment must exit {DSN_CONFLICT_EXIT_CODE}, \
         distinguishably from any other failure.\n{}",
        describe(out)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.is_empty(),
        "nothing may be printed on stdout before the refusal: the refusal says \
         no database was contacted, and a witness line, a canon name or a \
         banner in front of it means the process had already done work.\n{}",
        describe(out)
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.starts_with(REFUSAL_FIRST_LINE),
        "the refusal must be the FIRST thing on stderr, not the last.\n{}",
        describe(out)
    );

    for needle in [first, second, "host-one:5432/db1", "host-two:5432/db2"] {
        assert!(
            stderr.contains(needle),
            "the refusal must name {needle}.\n{}",
            describe(out)
        );
    }
    for secret in ["pw1", "pw2"] {
        assert!(
            !stderr.contains(secret),
            "a password reached the output.\n{}",
            describe(out)
        );
    }
}

/// Two DSNs a scarab worker would have ranked silently. It must refuse before
/// `connect_with_retry`, the migrator, or the "ready" line.
///
/// `--help` is passed deliberately: scarab takes no arguments and ignores them,
/// so the flag proves the refusal does not depend on reaching argument
/// handling. The pair is the one scarab's own inline chain used to resolve by
/// precedence.
#[test]
fn dsn_conflict_refuses_in_scarab() {
    let out = gated_binary(
        env!("CARGO_BIN_EXE_scarab"),
        &["--help"],
        &[
            ("DATABASE_URL", "postgresql://a:pw1@host-one:5432/db1"),
            ("NEON_DATABASE_URL", "postgresql://a:pw2@host-two:5432/db2"),
        ],
    );
    assert_refused(&out, "DATABASE_URL", "NEON_DATABASE_URL");
    let text = combined(&out);
    // Prefix-matched, not word-matched: scarab's own lines all carry one of
    // these tags, and a bare "ready" would match the word "already" inside the
    // refusal's own explanation.
    for line in ["[scarab]", "[migrator]"] {
        assert!(
            !text.contains(line),
            "scarab printed {line:?} while the target database was ambiguous.\n{}",
            describe(&out)
        );
    }
}

/// The vacuity guard for the case above: exit 78 must come from the GATE, not
/// from scarab failing to start for some unrelated reason.
///
/// With no DSN at all scarab still cannot run -- it needs a database -- but it
/// must say so in its own words and with its own status.
#[test]
fn scarab_without_any_dsn_fails_differently() {
    let out = gated_binary(env!("CARGO_BIN_EXE_scarab"), &[], &[]);
    let text = combined(&out);
    assert_ne!(
        out.status.code(),
        Some(DSN_CONFLICT_EXIT_CODE),
        "an unset environment is not a conflict.\n{}",
        describe(&out)
    );
    assert!(
        !text.contains(REFUSAL_MARKER),
        "one absent DSN cannot be ambiguous.\n{}",
        describe(&out)
    );
    assert!(
        text.contains("no Postgres DSN found"),
        "the genuinely-unset case must keep its own message.\n{}",
        describe(&out)
    );
}

/// `matrix_runner` reached `MATRIX_DATABASE_URL`, which the gate could not see
/// until that name joined `neon_writer::DSN_ENV_VARS`.
///
/// Real training arguments are passed, not `--dry-run-canon`: the read this
/// protects sits AFTER a full training run and after the `MATRIX_ROW` witness
/// is printed, so an empty stdout is the evidence that the refusal came first.
/// A regression therefore fails on a trained-but-refused cell rather than
/// passing quietly.
#[test]
fn dsn_conflict_refuses_in_matrix_runner() {
    let out = gated_binary(
        env!("CARGO_BIN_EXE_matrix_runner"),
        &[
            "--format=fp32",
            "--algo=adamw",
            // A SEED_CANON member; anything else is rejected before the DSN is
            // read and would make this test pass for the wrong reason.
            "--seed=1597",
            "--hidden=64",
            "--steps=120",
            "--lr=0.001",
        ],
        &[
            (
                "MATRIX_DATABASE_URL",
                "postgresql://a:pw1@host-one:5432/db1",
            ),
            ("DATABASE_URL", "postgresql://a:pw2@host-two:5432/db2"),
        ],
    );
    assert_refused(&out, "MATRIX_DATABASE_URL", "DATABASE_URL");
    let text = combined(&out);
    for line in ["MATRIX_ROW", "CANON_NAME", "[matrix_runner]"] {
        assert!(
            !text.contains(line),
            "matrix_runner produced {line:?} for a cell whose destination was \
             ambiguous.\n{}",
            describe(&out)
        );
    }
}

/// The vacuity guard for `matrix_runner`: with one DSN the gate is silent and
/// the binary does its ordinary work.
#[test]
fn matrix_runner_with_a_single_dsn_is_unaffected() {
    let out = gated_binary(
        env!("CARGO_BIN_EXE_matrix_runner"),
        &[
            "--format=fp32",
            "--algo=adamw",
            "--seed=1597",
            "--hidden=128",
            "--lr=0.001",
            // No training, no row: this case is about the gate staying out of
            // the way, and the canon path is the cheapest way to prove the
            // binary got past it.
            "--dry-run-canon",
        ],
        &[(
            "MATRIX_DATABASE_URL",
            "postgresql://a:pw@no-such-host.invalid:5432/db1",
        )],
    );
    let text = combined(&out);
    assert!(
        !text.contains(REFUSAL_MARKER),
        "one DSN cannot be ambiguous.\n{}",
        describe(&out)
    );
    assert!(
        out.status.success() && text.contains("CANON_NAME"),
        "the binary must run normally on an unambiguous environment.\n{}",
        describe(&out)
    );
}

/// `tri`'s race arms read `DATABASE_URL` alone, so a differing
/// `TRIOS_DATABASE_URL` used to be ignored without a word.
///
/// `--help` is the argument precisely because it is the cheapest thing `tri`
/// can be asked to do: if even that is refused, the gate is genuinely ahead of
/// `clap` and of every subcommand, and the empty stdout below is the help text
/// that was NOT printed.
#[test]
fn dsn_conflict_refuses_in_tri() {
    let out = gated_binary(
        env!("CARGO_BIN_EXE_tri"),
        &["--help"],
        &[
            ("DATABASE_URL", "postgresql://a:pw1@host-one:5432/db1"),
            ("TRIOS_DATABASE_URL", "postgresql://a:pw2@host-two:5432/db2"),
        ],
    );
    assert_refused(&out, "DATABASE_URL", "TRIOS_DATABASE_URL");
    assert!(
        !combined(&out).contains("Usage:"),
        "clap ran before the gate.\n{}",
        describe(&out)
    );
}

/// The vacuity guard for `tri`: the gate must not have turned an ordinary
/// invocation into a refusal, and `tri`'s status semantics are unchanged -- a
/// present `DATABASE_URL` is still not evidence that anything is reachable, and
/// `--help` neither contacts a database nor claims one is there.
#[test]
fn tri_with_a_single_dsn_still_prints_help() {
    let out = gated_binary(
        env!("CARGO_BIN_EXE_tri"),
        &["--help"],
        &[(
            "DATABASE_URL",
            "postgresql://a:pw@no-such-host.invalid:5432/db1",
        )],
    );
    let text = combined(&out);
    assert!(
        !text.contains(REFUSAL_MARKER),
        "one DSN cannot be ambiguous.\n{}",
        describe(&out)
    );
    assert!(
        out.status.success() && text.contains("Usage:"),
        "an unambiguous environment must reach clap.\n{}",
        describe(&out)
    );
}
