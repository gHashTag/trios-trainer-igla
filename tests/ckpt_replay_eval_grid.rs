//! tests/ckpt_replay_eval_grid.rs - the auditor must replay on the grid the
//! record declares, and must tell an honest second laboratory from a forger.
//!
//! # The two defects these tests exist to keep closed
//!
//! 1. **The grid was never restored.** `ckpt_replay` spawns the trainer with
//!    `env_clear()` and re-exports a handful of `TRIOS_*` variables. Until this
//!    file existed, `TRIOS_EVAL_CHUNKS` was not among them, so the child always
//!    evaluated at `EVAL_CHUNKS_DEFAULT = 40` however the record described its
//!    own reading. Measured on 2026-08-03: a 200-step record written with
//!    `TRIOS_EVAL_CHUNKS=16` replayed BYTE-IDENTICALLY (`d2fceff3...` on both
//!    sides) and was graded `BPB MISMATCH`, exit 5, under a note asserting that
//!    "Byte-identical weights cannot measure a different metric on this host" -
//!    while the child's own `[ckpt] ... eval_chunks=40` line stood two screens
//!    above it. `docs/EVAL-UNCERTAINTY.md` publishes a reproduction recipe that
//!    sets exactly that variable, so the repository's documented uncertainty
//!    procedure produced records its own falsifier called liars.
//!
//! 2. **Every non-identity was fraud.** The one cross-architecture run this
//!    repository has performed (CI run 30767491098) was a faithful `x86_64`
//!    execution of the same source with the same locked dependency graph and a
//!    corpus verified against the manifest, and it earned `MISMATCH` - the same
//!    verdict a fabricated checkpoint earns. An instrument whose only outcomes
//!    are "bit-identical" and "indistinguishable from fraud" has no
//!    false-positive control, so `docs/REPRODUCIBILITY-GRADING.md`'s L2 rung is
//!    now implemented as its own exit code.
//!
//! Neither test spawns a trainer. What is asserted is the environment the child
//! WOULD be handed, and the verdict table that environment's result is graded
//! by; the end-to-end proof is a replay, and it costs a training run.
//!
//! # Why the binary is included as a module
//!
//! An integration test cannot `use` a binary crate, so the source file is
//! compiled in verbatim by `#[path]` - the same device
//! `tests/ckpt_replay_bpb_gate.rs` uses. These assertions therefore land on the
//! code that ships and not on a copy of it.

#[path = "../src/bin/ckpt_replay.rs"]
mod ckpt_replay;

use std::collections::HashMap;
use std::path::Path;

use serde_json::{json, Value};

use ckpt_replay::{
    bpb_agrees, bpb_within, grade_pair, recorded_eval_grid, replay_env, PairVerdict,
};

/// A schema-6 record, reduced to the fields `replay_env` reads. The values are
/// the ones the 200-step `grid-probe` run of 2026-08-03 actually wrote.
fn grid_probe_record(eval_chunks: Option<u64>) -> Value {
    let mut record = json!({
        "schema": "trios-checkpoint-record/6",
        "canon_name": "grid-probe",
        "seed": 47,
        "step": 200,
        "steps_total": 200,
        "hidden": 64,
        "num_attn_layers": 2,
        "optimizer": "adamw",
        "fake_quant_format": "f32",
        "data_synthetic": false,
        "eval_every": 100,
        "gf16_floor_every": 1,
        "attn_scale": 1.0,
        "attn_seq": 64,
        "eval_seq": 65,
    });
    if let Some(c) = eval_chunks {
        record["eval_chunks"] = json!(c);
    }
    record
}

fn env_map(record: &Value) -> HashMap<String, String> {
    replay_env(record, Path::new("/tmp/ckpt_replay-test/checkpoints"))
        .into_iter()
        .collect()
}

// ---- (a) the grid is restored ------------------------------------------------

/// The defect, reduced to one assertion: a record that declares a non-default
/// grid must hand that grid to the child.
#[test]
fn a_declared_grid_reaches_the_child() {
    let env = env_map(&grid_probe_record(Some(16)));
    assert_eq!(
        env.get("TRIOS_EVAL_CHUNKS").map(String::as_str),
        Some("16"),
        "the record declares eval_chunks=16; a replay at the default 40 measures a \
         different quantity and then calls the record a liar about it"
    );
}

/// Full coverage is a legal declared grid (`TRIOS_EVAL_CHUNKS=0` in
/// `train_loop::eval_chunks_target`), and it must not be confused with silence.
/// An achieved count is what the sidecar carries, so a record stating one is
/// replayed at it verbatim.
#[test]
fn every_declared_chunk_count_is_passed_through_verbatim() {
    for chunks in [1_u64, 10, 16, 40, 320] {
        let env = env_map(&grid_probe_record(Some(chunks)));
        assert_eq!(
            env.get("TRIOS_EVAL_CHUNKS").map(String::as_str),
            Some(chunks.to_string().as_str()),
            "eval_chunks={chunks} must be requested exactly as recorded"
        );
    }
}

/// The other half of the rule. Schemas 1 to 5 predate the knob, so their runs
/// provably used the hardcoded default; setting the variable from a guess would
/// be the same defect pointing the other way.
#[test]
fn a_record_that_declares_no_grid_leaves_the_variable_unset() {
    let record = grid_probe_record(None);
    assert_eq!(recorded_eval_grid(&record).0, None);
    let env = env_map(&record);
    assert!(
        !env.contains_key("TRIOS_EVAL_CHUNKS"),
        "an unset variable is what reproduces a pre-schema-6 run; \
         found {:?}",
        env.get("TRIOS_EVAL_CHUNKS")
    );
}

/// A `null` is not a declaration. `present()` is what enforces this everywhere
/// else in the binary, and the grid must obey the same rule.
#[test]
fn a_null_grid_is_silence_not_a_value() {
    let mut record = grid_probe_record(None);
    record["eval_chunks"] = Value::Null;
    assert_eq!(recorded_eval_grid(&record).0, None);
    assert!(!env_map(&record).contains_key("TRIOS_EVAL_CHUNKS"));
}

/// Factoring the environment out of `main` must not have dropped anything on the
/// way. These five are the variables the replay was already driven by, and
/// `TRINITY_AUTOMIGRATE=0` in particular is what keeps a replay from running a
/// migration against the live ledger.
#[test]
fn the_pre_existing_environment_survived_the_refactor() {
    let env = env_map(&grid_probe_record(Some(16)));
    assert_eq!(
        env.get("TRIOS_CHECKPOINT_DIR").map(String::as_str),
        Some("/tmp/ckpt_replay-test/checkpoints")
    );
    assert_eq!(env.get("TRIOS_CANON_NAME").map(String::as_str), Some("grid-probe"));
    assert_eq!(env.get("TRIOS_GF16_FLOOR_EVERY").map(String::as_str), Some("1"));
    assert_eq!(env.get("TRIOS_FORMAT_TYPE").map(String::as_str), Some("f32"));
    assert_eq!(env.get("TRINITY_AUTOMIGRATE").map(String::as_str), Some("0"));
    assert_eq!(env.get("TRIOS_ATTN_SEQ").map(String::as_str), Some("64"));
    assert!(
        !env.contains_key("TRIOS_ALLOW_SYNTHETIC_DATA"),
        "the record says data_synthetic=false; the replay must not be allowed to \
         substitute a synthetic corpus"
    );
    assert!(
        !env.contains_key("TRIOS_CHECKPOINT_EVERY"),
        "step == steps_total, so no intermediate cadence is requested"
    );
}

/// An intermediate checkpoint still asks for its own cadence, and a synthetic
/// corpus is still opted into rather than assumed.
#[test]
fn an_intermediate_synthetic_record_keeps_its_two_extra_variables() {
    let mut record = grid_probe_record(Some(40));
    record["step"] = json!(100);
    record["data_synthetic"] = json!(true);
    let env = env_map(&record);
    assert_eq!(env.get("TRIOS_CHECKPOINT_EVERY").map(String::as_str), Some("100"));
    assert_eq!(
        env.get("TRIOS_ALLOW_SYNTHETIC_DATA").map(String::as_str),
        Some("1")
    );
}

// ---- (c) the verdict table --------------------------------------------------

/// The four cases, and the four exit codes they map to. This is the whole of the
/// ladder: 0 and 1 were already here, 5 grades a number the bytes contradict,
/// and 6 is the rung that did not exist - an honest second laboratory.
#[test]
fn the_four_way_verdict_table_is_exactly_this() {
    assert_eq!(grade_pair(true, true), PairVerdict::Verified);
    assert_eq!(grade_pair(true, true).exit_code(), 0);

    assert_eq!(grade_pair(true, false), PairVerdict::BpbMismatch);
    assert_eq!(grade_pair(true, false).exit_code(), 5);

    assert_eq!(grade_pair(false, true), PairVerdict::SecondLaboratory);
    assert_eq!(
        grade_pair(false, true).exit_code(),
        6,
        "bytes differ and the metric agrees: L2 PASS / L3 FAIL, which must not \
         share an exit code with either a pass or a forgery"
    );

    assert_eq!(grade_pair(false, false), PairVerdict::Mismatch);
    assert_eq!(
        grade_pair(false, false).exit_code(),
        1,
        "a forger whose metric also disagrees still gets MISMATCH"
    );
}

/// Every exit code the table can produce is distinct. A verdict that shares a
/// code with another verdict is a verdict a caller cannot act on, and this whole
/// item exists because two different findings shared one.
#[test]
fn the_four_verdicts_have_four_distinct_exit_codes() {
    let codes: Vec<u8> = [
        PairVerdict::Verified,
        PairVerdict::BpbMismatch,
        PairVerdict::SecondLaboratory,
        PairVerdict::Mismatch,
    ]
    .into_iter()
    .map(PairVerdict::exit_code)
    .collect();
    let mut sorted = codes.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), codes.len(), "codes collide: {codes:?}");
}

// ---- the L2 tolerance -------------------------------------------------------

/// The four paired same-grid deltas published in
/// `docs/REPRODUCIBILITY-GRADING.md` for the 12 000-step aarch64 macOS /
/// x86_64 Linux pair: steps 1 000, 3 000, 8 000 and 12 000.
const PUBLISHED_PAIRS: &[(f64, f64)] = &[
    (3.3097, 3.3099),
    (3.0771, 3.0736),
    (2.6455, 2.6485),
    (2.6348, 2.6378),
];

/// The default tolerance must admit every reading the repository has actually
/// published for an honest cross-laboratory pair. If it did not, the constant
/// would be manufacturing failures out of the only evidence there is.
#[test]
fn the_default_tolerance_admits_every_published_pair() {
    for (mac, linux) in PUBLISHED_PAIRS {
        assert!(
            bpb_within(*mac, *linux, 0.005),
            "{mac} vs {linux} is an honest measured pair and must clear L2"
        );
    }
}

/// And it must not admit the retracted number. A tolerance loose enough to pass
/// 1.5492 against 2.6348 is not a tolerance, it is a laundering step.
#[test]
fn the_default_tolerance_does_not_admit_the_retracted_number() {
    assert!(
        !bpb_within(2.6347548961639404, 1.5492, 0.005),
        "1.5492 is 1.08 bpb away from the value it was attached to"
    );
    assert!(
        !bpb_within(2.6347548961639404, 1.5492, 1.0),
        "even a tenfold-loosened tolerance must not reach it"
    );
}

/// A tolerance is a number, and a non-number is not a permissive tolerance - it
/// is a broken one. `NaN` comparisons are false, and that must remain the
/// answer rather than becoming "everything passes".
#[test]
fn a_non_finite_reading_or_tolerance_agrees_with_nothing() {
    assert!(!bpb_within(f64::NAN, 2.6348, 0.005));
    assert!(!bpb_within(2.6348, f64::NAN, 0.005));
    assert!(!bpb_within(2.6348, 2.6348, f64::NAN));
    assert!(!bpb_within(2.6348, 2.6349, -0.01));
    assert!(
        bpb_within(2.6348, 2.6348, 0.0),
        "a zero tolerance is exact equality, which is a legal thing to demand"
    );
}

/// The two comparators answer different questions and must not be swapped. The
/// byte-identical path keeps its four-decimal text comparison; the tolerance
/// exists only for the rung where the bytes already failed.
#[test]
fn the_tolerance_never_loosens_the_byte_identical_path() {
    // 0.0030 apart: L2 passes, and the 4-dp text comparison still fails.
    assert!(bpb_within(2.6348, 2.6378, 0.005));
    assert!(
        !bpb_agrees(2.6348, "2.6378"),
        "a byte-identical replay is still graded at the precision the trainer prints"
    );
}
