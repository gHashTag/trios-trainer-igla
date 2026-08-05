//! tests/ckpt_replay_bpb_gate.rs - the auditor must grade the number it certifies.
//!
//! # The defect these tests exist to keep closed
//!
//! `ckpt_replay` ran the trainer with `cmd.status()`, which inherits the child's
//! stdout and discards it. `grep -n bpb src/bin/ckpt_replay.rs` returned two
//! hits, both inside a comment: no code path read any BPB. A sidecar whose
//! `final_val_bpb` had been replaced by hand with the retracted 1.5492 therefore
//! printed `VERIFIED`, exit 0, one screen below the replay's own honest
//! `DONE: seed=47 bpb=2.9744 steps=2000 opt=adamw`. `README.md` calls that
//! sidecar "the record the repository's own verifier grades" and prints
//! `final_val_bpb` as a row of the table, so the sentence was false in the only
//! way that matters.
//!
//! The end-to-end proof is a replay, which costs a training run. What is tested
//! here is the arithmetic that replay hangs on, at zero cost and with no
//! subprocess: the extraction of the trainer's own `bpb=` token, and the
//! four-decimal comparison against the recorded value.
//!
//! # Why the binary is included as a module
//!
//! `parse_done_bpb` and `bpb_agrees` live in a BINARY, and an integration test
//! cannot `use` a binary crate. The source file is compiled into this test
//! verbatim by `#[path]`, so these tests exercise the code that actually ships
//! rather than a copy of it that can drift away from it in silence.

#[path = "../src/bin/ckpt_replay.rs"]
mod ckpt_replay;

use ckpt_replay::{bpb_agrees, done_bpb_token, parse_done_bpb, render_bpb};

/// The exact line `trios-train` printed on 2026-08-02 for the r6 record whose
/// tampering opened this item. Copied from that run, not invented.
const REAL_DONE: &str = "DONE: seed=47 bpb=2.9744 steps=2000 opt=adamw";

/// The line the sweep path prints TODAY, first of the three.
///
/// Produced on 2026-08-03 (aarch64 macOS) by:
///
/// ```text
/// env -u DATABASE_URL -u NEON_DATABASE_URL -u TRIOS_DATABASE_URL \
///     TRINITY_AUTOMIGRATE=0 TRIOS_CHECKPOINT_DISABLE=1 \
///     ./target/release/trios-train --sweep --optimizer adamw --steps 1 \
///     --hidden 16 --attn-layers 1 --eval-every 1 --eval-chunks 8
/// DONE: seed=47 bpb=7.0517 steps=1 opt=adamw
/// DONE: seed=89 bpb=7.0051 steps=1 opt=adamw
/// DONE: seed=123 bpb=6.9896 steps=1 opt=adamw
/// ```
///
/// Copied from that run, not invented. It replaces
/// `DONE: seed=1597 bpb=2.5000 steps=1`, which was impossible for the path it
/// named twice over: 1597 is not in `GATE_FINAL_SEEDS {47, 89, 123}`, and a
/// one-step run on this architecture measures ~7.0, not 2.5. (2.5 with seed
/// 1597 is the documented `smoke_train` line, which trains nothing.)
const SWEEP_DONE: &str = "DONE: seed=47 bpb=7.0517 steps=1 opt=adamw";

/// The same line with its `opt=` token removed: the shape EVERY sweep log
/// written before 2026-08-03 has, because `run_sweep` took no optimizer
/// argument and the sweep arm printed no `opt=` field at all. Those logs still
/// exist and `parse_done_bpb` must still read them - which is the only reason
/// this form is kept after the sweep arm started naming its optimizer.
const LEGACY_SWEEP_DONE_NO_OPT: &str = "DONE: seed=47 bpb=7.0517 steps=1";

/// `final_val_bpb` of `checkpoints/r4-docs-repro/12000.json`, full f64 precision
/// as serde writes it.
const RECORDED_12000: f64 = 2.6347548961639404;

// ---- parse_done_bpb ---------------------------------------------------------

#[test]
fn parses_the_bpb_token_of_a_real_done_line() {
    assert_eq!(parse_done_bpb(REAL_DONE), Some(2.9744));
    assert_eq!(done_bpb_token(REAL_DONE), Some("2.9744"));
}

#[test]
fn parses_the_sweep_line() {
    assert_eq!(parse_done_bpb(SWEEP_DONE), Some(7.0517));
    assert_eq!(done_bpb_token(SWEEP_DONE), Some("7.0517"));
}

/// The archived sweep form, with no `opt=` field, still parses.
#[test]
fn parses_the_legacy_sweep_form_that_has_no_opt_field() {
    assert_eq!(parse_done_bpb(LEGACY_SWEEP_DONE_NO_OPT), Some(7.0517));
    assert_eq!(done_bpb_token(LEGACY_SWEEP_DONE_NO_OPT), Some("7.0517"));
}

/// A `DONE:` line with no `bpb=` token at all yields nothing. Silence is not a
/// measurement, and must never be read as one.
#[test]
fn a_done_line_without_a_bpb_token_yields_nothing() {
    assert_eq!(parse_done_bpb("DONE: seed=47 steps=2000 opt=adamw"), None);
    assert_eq!(done_bpb_token("DONE: seed=47 steps=2000 opt=adamw"), None);
}

/// The trainer prints this when a run took no final measurement. It is a token,
/// so `done_bpb_token` reports it; it is not a number, so `parse_done_bpb` does
/// not. That distinction is what separates "the replay said nothing" from "the
/// replay said something unusable" in the verdict text.
#[test]
fn unmeasured_is_a_token_but_not_a_value() {
    let line = "DONE: seed=47 bpb=unmeasured steps=2000 opt=adamw";
    assert_eq!(done_bpb_token(line), Some("unmeasured"));
    assert_eq!(parse_done_bpb(line), None);
}

/// `"NaN"` and `"inf"` both parse as f64. Accepting either would let a poisoned
/// forward pass be graded as a measurement - the same laundering this repository
/// has already had to remove from `loss_on_seq`.
#[test]
fn non_finite_tokens_are_rejected() {
    for token in ["NaN", "nan", "inf", "-inf", "infinity"] {
        let line = format!("DONE: seed=47 bpb={token} steps=2000 opt=adamw");
        assert_eq!(
            done_bpb_token(&line),
            Some(token),
            "the token itself should still be readable for the verdict text"
        );
        assert_eq!(
            parse_done_bpb(&line),
            None,
            "bpb={token} is not a measurement and must not be graded as one"
        );
    }
}

/// Only a line that BEGINS `DONE:` counts. A corpus, a path or a log message
/// that happens to contain the text must not be able to supply the number the
/// verdict is built on.
#[test]
fn only_a_line_that_begins_done_is_read() {
    assert_eq!(parse_done_bpb("[trainer] not DONE: bpb=1.0000"), None);
    assert_eq!(parse_done_bpb("seed=47 bpb=1.0000 steps=1"), None);
    // Leading whitespace is tolerated; the prefix is still the first thing on
    // the line.
    assert_eq!(parse_done_bpb("   DONE: bpb=1.0000"), Some(1.0));
}

// ---- the four-decimal comparison -------------------------------------------

/// The honest case: the record carries full f64 precision, the trainer prints
/// four decimals, and they agree at the precision the trainer chose.
#[test]
fn recorded_f64_agrees_with_the_four_decimal_token() {
    assert_eq!(render_bpb(RECORDED_12000), "2.6348");
    assert!(bpb_agrees(RECORDED_12000, "2.6348"));
}

/// The defect, reduced to one assertion: the retracted 1.5492 must not be
/// certifiable by a replay that measured 2.6348.
#[test]
fn the_retracted_number_does_not_agree() {
    assert!(!bpb_agrees(RECORDED_12000, "1.5492"));
}

/// The r6 record whose hand-edited copy printed VERIFIED, exit 0.
#[test]
fn the_r6_tampering_is_caught() {
    let honest = 2.9743857383728027_f64;
    let token = done_bpb_token(REAL_DONE).expect("the real line carries a token");
    assert!(bpb_agrees(honest, token));
    assert!(
        !bpb_agrees(1.5492, token),
        "a sidecar claiming 1.5492 must not agree with a replay that measured {token}"
    );
}

/// The stated limit, asserted rather than described: agreement is to four
/// decimal places and no further. A difference of 5e-5 is invisible here, and a
/// reader who needs more resolution than that needs a different instrument.
#[test]
fn agreement_is_four_decimals_and_no_further() {
    assert!(
        bpb_agrees(2.63475_f64 + 0.00003, "2.6348"),
        "a 3e-5 difference is below the resolution of the trainer's own printout"
    );
    assert!(
        !bpb_agrees(2.6348, "2.6349"),
        "one unit in the fourth decimal is the smallest difference this can see"
    );
}

/// A recorded value that is not finite cannot agree with anything. A record is
/// not allowed to certify `NaN` and have that count as a pass.
#[test]
fn a_non_finite_record_agrees_with_nothing() {
    assert!(!bpb_agrees(f64::NAN, "2.6348"));
    assert!(!bpb_agrees(f64::NAN, "NaN"));
    assert!(!bpb_agrees(f64::INFINITY, "inf"));
}

/// Round trip: whatever the trainer prints, re-rendering the value it parses to
/// reproduces the token. This is what makes the text comparison safe to state as
/// a numeric one.
#[test]
fn every_token_the_trainer_prints_round_trips() {
    for token in ["2.6348", "2.9744", "7.0000", "0.0000", "12.3456"] {
        let line = format!("DONE: seed=47 bpb={token} steps=1 opt=adamw");
        let value = parse_done_bpb(&line).expect("a finite token");
        assert_eq!(render_bpb(value), token);
        assert!(bpb_agrees(value, token));
    }
}
