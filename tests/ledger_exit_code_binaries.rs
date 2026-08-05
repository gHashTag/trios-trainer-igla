//! `trios-train` must tell the truth about its ledger in its EXIT STATUS.
//!
//! `neon_writer::ledger_exit_code()` exists so that "a DSN was configured, N
//! writes were attempted, 0 landed" is a non-zero exit. `smoke_train`,
//! `bpb_smoke`, `ngram_train_gf16` and `hybrid_train` all call it. `trios-train`
//! -- the binary CI runs, the binary README documents, and the binary that
//! produced the r6-headline run with a DSN set -- ended `Ok(())` regardless. A
//! run whose every `bpb_sample` was rejected on stderr still reported success.
//!
//! These are subprocess tests on purpose: the exit status is the observable
//! under test, and a unit test on `ledger_exit_code()` cannot see whether the
//! binary honours it. The two cases are the two halves of the contract:
//!
//!   (a) no DSN configured -> nothing was promised -> exit 0, unchanged;
//!   (b) DSN configured, every write dropped -> non-zero.
//!
//! `every_tracked_bin_source_is_a_declared_bin` at the bottom of this file is a
//! different subject and lives here only because this is the integration-test
//! file the change that added it was allowed to touch. See its own comment.

use std::process::{Command, Output};

/// The env vars `neon_writer::resolve_dsn` consults, in its own order
/// (src/neon_writer.rs). Every one is cleared before each case so a DSN in the
/// developer's or CI's environment cannot decide the result.
const DSN_VARS: [&str; 4] = [
    "DATABASE_URL",
    "NEON_DATABASE_URL",
    "TRIOS_NEON_DSN",
    "TRIOS_DATABASE_URL",
];

/// The write opt-in `neon_writer` also requires (`neon_writer::LEDGER_WRITE_OPT_IN`).
/// Spelled out rather than imported because these are subprocess tests against
/// the binary, not unit tests linked to the library.
const LEDGER_WRITE_OPT_IN: &str = "TRIOS_LEDGER_WRITE";

/// A one-step run of the smallest model that still reaches an evaluation.
///
/// `train_loop` evaluates when `step == args.steps`, and that evaluation calls
/// `neon_writer::bpb_sample`, so exactly one ledger write is attempted.
/// `TRIOS_CHECKPOINT_DISABLE=1` keeps the run from writing into `checkpoints/`.
fn trainer_command() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_trios-train"));
    // The corpus paths are repo-relative defaults; `cargo test` does not
    // guarantee the cwd of the spawned child.
    cmd.current_dir(env!("CARGO_MANIFEST_DIR"));
    for var in DSN_VARS {
        cmd.env_remove(var);
    }
    // A DSN alone is no longer permission to write: `src/neon_writer.rs`
    // requires `TRIOS_LEDGER_WRITE=1` as well, and without it reports itself as
    // having no DSN at all. Cleared here so the variable's presence in the
    // developer's shell cannot decide either case; the one case that needs it
    // sets it back explicitly.
    cmd.env_remove(LEDGER_WRITE_OPT_IN);
    // `SEED` triggers Canon #93 enforcement and would override `--seed`.
    cmd.env_remove("SEED");
    cmd.env_remove("TRIOS_SEED");
    cmd.env("TRIOS_CHECKPOINT_DISABLE", "1")
        // The migrator is a separate connection path; it is not what is under
        // test and its retry budget would dominate the runtime.
        .env("TRINITY_AUTOMIGRATE", "0")
        .env("TRIOS_CANON_NAME", "IGLA-TEST-fp32-h16-LR001-rng47-adamw")
        .args([
            "--seed",
            "47",
            "--steps",
            "1",
            "--hidden",
            "16",
            "--attn-layers",
            "1",
            "--eval-every",
            "1",
            // `evaluate` refuses fewer than 8 windows (src/train_loop.rs), so 8
            // is the smallest grid that produces a reading at all.
            "--eval-chunks",
            "8",
        ]);
    cmd
}

fn run(mut cmd: Command) -> Output {
    cmd.output().expect("spawn trios-train")
}

fn describe(out: &Output) -> String {
    format!(
        "status={:?}\n--- stdout ---\n{}\n--- stderr (tail) ---\n{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
            .lines()
            .rev()
            .take(20)
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

#[test]
fn trios_train_exits_zero_with_no_dsn_configured() {
    let out = run(trainer_command());
    assert!(
        out.status.success(),
        "with no DSN configured nothing was promised to any ledger, so the \
         exit status must be unchanged.\n{}",
        describe(&out)
    );
    // The run really did complete: without this a compile-time or arg-parse
    // failure that happened to exit 0 would pass the assertion above.
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("DONE: seed=47"),
        "expected the trainer to finish a run.\n{}",
        describe(&out)
    );
    // The no-DSN path has its OWN line, and it is the specific line this case
    // exists to pin. `trios-train.rs::exit_with_ledger_status` deliberately
    // stopped printing the tally here: `attempted=3 landed=0 dropped=3` on a
    // run that was never given a DSN reads as three writes lost in transport.
    // Asserting a substring both paths satisfy would have let that regress
    // back in unnoticed, so this asserts the line verbatim...
    assert!(
        stdout.contains("LEDGER: no DSN configured; no rows attempted"),
        "expected the no-DSN ledger line verbatim.\n{}",
        describe(&out)
    );
    // ...and that the tally line is absent, which is the half that actually
    // fails if the two paths are ever merged again.
    assert!(
        !stdout.contains("LEDGER: attempted="),
        "no DSN was configured, so no write was attempted and the tally line \
         must not be printed at all.\n{}",
        describe(&out)
    );
}

#[test]
fn trios_train_exits_non_zero_when_every_ledger_write_is_dropped() {
    let mut cmd = trainer_command();
    // Syntactically valid, deliberately unreachable. `.invalid` is the RFC 2606
    // TLD reserved to never resolve, so the failure is a name lookup: no packet
    // leaves the machine and the result does not depend on the network.
    //
    // A refused TCP port (127.0.0.1:1) is the more obvious choice and was tried
    // first; sqlx retries a refused connection until the pool's 30s acquire
    // timeout expires, twice, which put the case at ~66s. That timeout is set
    // inside `src/neon_writer.rs`, which this change does not own.
    cmd.env(
        "TRIOS_NEON_DSN",
        "postgres://nobody:nothing@no-such-host.invalid:5432/trios_ledger_exit_code_test",
    );
    // Required since the write opt-in landed in `src/neon_writer.rs`: without
    // it the trainer treats the DSN as absent, takes the no-DSN path and exits
    // 0, so this case would assert nothing about dropped writes. This is what
    // the case always meant by "DSN configured" -- it is the fixture reaching
    // the state under test, and every assertion below is unchanged.
    cmd.env(LEDGER_WRITE_OPT_IN, "1");
    let out = run(cmd);

    let stdout = String::from_utf8_lossy(&out.stdout);
    // Guard against a vacuous pass: the non-zero status must come from the
    // ledger, which requires that a write was actually attempted and lost.
    assert!(
        stdout.contains("LEDGER: attempted=1 landed=0 dropped=1"),
        "expected exactly one attempted ledger write, dropped.\n{}",
        describe(&out)
    );
    // The mirror of the assertion in the no-DSN case: a DSN WAS configured, so
    // the "nothing was promised" line must not appear.
    assert!(
        !stdout.contains("LEDGER: no DSN configured"),
        "a DSN was configured, so the no-DSN line must not be printed.\n{}",
        describe(&out)
    );
    assert!(
        stdout.contains("DONE: seed=47"),
        "the training run itself must still complete; the ledger verdict is \
         about recording, not about training.\n{}",
        describe(&out)
    );
    assert!(
        !out.status.success(),
        "a DSN was configured and every write was dropped: this run has no \
         ledger and must not report success.\n{}",
        describe(&out)
    );
}

/// The third state, which only exists since the write opt-in landed: a DSN IS
/// present but `TRIOS_LEDGER_WRITE=1` is not, so the run is not recorded and
/// must say so rather than exiting non-zero over writes it deliberately never
/// attempted.
///
/// Pinned because this is the exact drift that broke the case above: the two
/// tests were written when a DSN alone meant "recorded", and a change to the
/// meaning of "configured" silently turned the drop-path case into a run that
/// exercised the no-DSN path and asserted nothing. Now a future change to the
/// gate has to face a test that names it.
#[test]
fn a_dsn_without_the_write_opt_in_is_reported_as_no_dsn_and_exits_zero() {
    let mut cmd = trainer_command();
    cmd.env(
        "TRIOS_NEON_DSN",
        "postgres://nobody:nothing@no-such-host.invalid:5432/trios_ledger_exit_code_test",
    );
    // Deliberately NOT setting LEDGER_WRITE_OPT_IN: that is the subject.
    let out = run(cmd);

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("LEDGER: no DSN configured; no rows attempted"),
        "a DSN without the write opt-in must report itself exactly as an \
         unconfigured one.\n{}",
        describe(&out)
    );
    assert!(
        !stdout.contains("LEDGER: attempted="),
        "the opt-in was withheld, so no write may be attempted and no tally \
         line may be printed.\n{}",
        describe(&out)
    );
    assert!(
        out.status.success(),
        "declining to write is not a failure to write: this run promised no \
         ledger and must exit 0.\n{}",
        describe(&out)
    );
}

// ---------------------------------------------------------------------------
// Manifest audit: a source file under `src/bin/` that no `[[bin]]` declares is
// a file the compiler never reads.
//
// `autobins = false` in Cargo.toml means cargo does NOT pick up `src/bin/*.rs`
// automatically; every binary must be listed by hand. Four tracked files had
// drifted out of that list -- `concat_train.rs` and `ptq_eval.rs` were never
// added, `attn_train.rs` and `bench_cpu.rs` were commented out. The cost is not
// dead weight, it is FALSE ASSURANCE: `concat_train.rs` carried a hardening fix
// and a `#[test]` guarding it, neither of which had ever been compiled, so the
// test could not fail and the fix could not be trusted; `attn_train.rs` went on
// printing `f32::MAX` under a `BPB={:.4}` key next to "vs champion 2.5193".
//
// Note what does NOT work as a check: `grep 'path = "src/bin/attn_train.rs"'`
// over the manifest MATCHES the commented-out stanza. This test therefore
// strips whole-line comments before parsing, which is the difference between
// catching those two files and missing them.
// ---------------------------------------------------------------------------

/// Every `path = "..."` inside a `[[bin]]` table, with whole-line comments
/// removed first.
fn declared_bin_paths(manifest: &str) -> Vec<String> {
    let mut declared = Vec::new();
    let mut in_bin_table = false;
    for raw in manifest.lines() {
        let line = raw.trim();
        // A commented-out declaration declares nothing. This is the whole point.
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            in_bin_table = line.starts_with("[[bin]]");
            continue;
        }
        if !in_bin_table {
            continue;
        }
        if let Some(rest) = line.strip_prefix("path") {
            let rest = rest.trim_start();
            if let Some(rest) = rest.strip_prefix('=') {
                let value = rest.trim().trim_matches('"');
                declared.push(value.to_string());
            }
        }
    }
    declared
}

/// Git-tracked `*.rs` files sitting DIRECTLY in `src/bin/`.
///
/// Only the top level: cargo's own convention is that a subdirectory of
/// `src/bin/` is a module tree (`src/bin/foo/main.rs` plus its siblings), so a
/// file inside one is not expected to be its own binary target, and demanding a
/// `[[bin]]` for it would be wrong rather than strict.
///
/// KNOWN GAP, recorded rather than hidden. That exemption is right about
/// `[[bin]]` and wrong about compilation. As of 2026-08-03 the repo's only such
/// subdirectory is `src/bin/tjepa_modules/` (`mod.rs`, `encoder.rs`, `ntp.rs`),
/// and it is NOT a module tree of anything: `src/bin/tjepa_train.rs` declares no
/// `mod tjepa_modules`, and `grep -rn tjepa_modules src tests Cargo.toml` finds
/// no reference from outside the directory itself. Those three files are
/// therefore tracked, uncompiled and unreachable -- the same defect this audit
/// exists to catch, in the one shape it deliberately does not fail on.
/// Extending the audit to "every tracked file under src/bin/ is COMPILED by
/// something" is the correct next step; it is left undone here only because
/// those files were outside the ownership of the change that added this test,
/// and a test that fails on a file its author may not fix is a red suite, not a
/// finding. See this change's report.
fn tracked_bin_sources(root: &std::path::Path) -> Vec<String> {
    let out = Command::new("git")
        .current_dir(root)
        .args(["ls-files", "--", "src/bin"])
        .output()
        .expect("run `git ls-files`");
    assert!(
        out.status.success(),
        "`git ls-files` failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut files: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .filter(|p| p.ends_with(".rs"))
        // `src/bin/x.rs` has exactly three path components; anything deeper is
        // inside a module directory.
        .filter(|p| p.split('/').count() == 3)
        .map(str::to_string)
        .collect();
    files.sort();
    files
}

#[test]
fn every_tracked_bin_source_is_a_declared_bin() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("read Cargo.toml");

    let declared = declared_bin_paths(&manifest);
    let tracked = tracked_bin_sources(root);

    // Guard against a vacuous pass: if the listing or the parse silently
    // returned nothing, every file would trivially be "accounted for".
    assert!(
        tracked.len() > 10,
        "expected the repo's binaries to be found; got {tracked:?}"
    );
    assert!(
        declared.len() > 10,
        "expected [[bin]] stanzas to be parsed; got {declared:?}"
    );

    let undeclared: Vec<&String> = tracked.iter().filter(|p| !declared.contains(p)).collect();
    assert!(
        undeclared.is_empty(),
        "these files are git-tracked under src/bin/ but no [[bin]] declares them, \
         so nothing compiles them and no test in them can ever fail: {undeclared:#?}\n\
         Either add a [[bin]] stanza for each, or move the file to attic/. A \
         commented-out stanza does not count."
    );
}

#[test]
fn a_commented_out_stanza_does_not_count_as_a_declaration() {
    // The precise failure this audit exists to catch, pinned on a fixture so
    // the audit cannot be quietly relaxed into a substring match that a
    // commented-out `path = ...` would satisfy.
    let manifest = "\
[[bin]]\n\
name = \"live\"\n\
path = \"src/bin/live.rs\"\n\
\n\
# DISABLED: needs a module that lives in attic/.\n\
# [[bin]]\n\
# name = \"ghost\"\n\
# path = \"src/bin/ghost.rs\"\n";
    let declared = declared_bin_paths(manifest);
    assert_eq!(declared, vec!["src/bin/live.rs".to_string()]);
}

#[test]
fn a_path_outside_a_bin_table_is_not_a_bin_declaration() {
    // `[lib]` and `[[example]]` also carry `path = ...`; counting those would
    // let a file masquerade as a declared binary.
    let manifest = "\
[lib]\n\
name = \"trios_trainer\"\n\
path = \"src/lib.rs\"\n\
\n\
[[bin]]\n\
name = \"live\"\n\
path = \"src/bin/live.rs\"\n\
\n\
[[example]]\n\
name = \"demo\"\n\
path = \"src/bin/demo.rs\"\n";
    let declared = declared_bin_paths(manifest);
    assert_eq!(declared, vec!["src/bin/live.rs".to_string()]);
}
