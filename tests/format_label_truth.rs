//! A format LABEL must name arithmetic that ran.
//!
//! `TRIOS_FORMAT_TYPE=fp80` used to print "QAT: FakeQuant enabled for format
//! Fp80", write `fake_quant_format: "fp80"` into the sidecar AND into the
//! hashed `TRIOSCKP` header, and execute nothing: `fake_quantize_model` returns
//! immediately for a format in `is_unsupported_in_f32()`. Measured over three
//! 20-step seed-47 runs, the `fp80` payload after the 256-byte header is
//! bit-identical to the `f32` control and `final_val_bpb` agrees to all 16
//! digits - the only reason the `.bin` sha differed at all was the false label
//! sitting inside the hashed header, so the mislabel made an identity run look
//! like a distinct artifact.
//!
//! The crate already refused this everywhere except the binary that mints
//! checkpoints. `fake_quant::unsupported_in_f32_implies_not_faithful` says "The
//! arithmetic is defensible; the LABEL is not", and `matrix_runner` has
//! rejected such a row since 2026-08-03 unless `TRIOS_ALLOW_UNFAITHFUL_FORMAT=1`,
//! in which case it stamps `format_faithful=false`. This file holds the same
//! two claims for the checkpoint path:
//!
//!   1. an unfaithful format is REFUSED without the override;
//!   2. with the override the run proceeds and the sidecar carries its own
//!      retraction, `format_faithful: false`.
//!
//! Plus the second defect in the same function: an UNRECOGNISED spelling used
//! to resolve to "no quantization", so `TRIOS_FORMAT_TYPE=int_8` trained f32 and
//! recorded f32 without printing the word "format" once.
//!
//! Everything here runs IN PROCESS. Nothing spawns `trios-train`: the ambient
//! `DATABASE_URL` on a developer machine would make a spawned run write live
//! ledger rows, and none of the claims above need a training loop to state.
//! `TRIOS_FORMAT_TYPE`, its alias and the override are process-global, so every
//! case holds `ENV_LOCK` for its whole body.

use std::path::Path;
use std::sync::Mutex;

use trios_trainer::checkpoint::{
    self, CheckpointRecord, CorpusProvenance, OptimizerParams, PlatformProvenance,
    TrainerProvenance, CHECKPOINT_FORMAT_VERSION, CHECKPOINT_RECORD_SCHEMA,
    GIT_PROVENANCE_ASSERTED, TRAINER_PROVENANCE_SELF_HASHED,
};
use trios_trainer::fake_quant::FormatKind;
use trios_trainer::train_loop;

static ENV_LOCK: Mutex<()> = Mutex::new(());

const FORMAT_ENV: &str = "TRIOS_FORMAT_TYPE";
const FORMAT_ENV_ALIAS: &str = "TRIOS_FAKE_QUANT_FORMAT";
const OVERRIDE_ENV: &str = "TRIOS_ALLOW_UNFAITHFUL_FORMAT";

/// Resolve the format with a known environment: `TRIOS_FORMAT_TYPE` set to
/// `format`, the alias cleared, and the override either set to "1" or removed.
///
/// The variables are restored afterwards so a case cannot leak into the next
/// one through the process environment it shares with them.
fn resolve_with(format: &str, override_set: bool) -> anyhow::Result<Option<FormatKind>> {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let prev_fmt = std::env::var(FORMAT_ENV).ok();
    let prev_alias = std::env::var(FORMAT_ENV_ALIAS).ok();
    let prev_override = std::env::var(OVERRIDE_ENV).ok();

    std::env::set_var(FORMAT_ENV, format);
    std::env::remove_var(FORMAT_ENV_ALIAS);
    if override_set {
        std::env::set_var(OVERRIDE_ENV, "1");
    } else {
        std::env::remove_var(OVERRIDE_ENV);
    }

    let out = train_loop::resolve_fake_quant_format();

    restore(FORMAT_ENV, prev_fmt);
    restore(FORMAT_ENV_ALIAS, prev_alias);
    restore(OVERRIDE_ENV, prev_override);
    out
}

fn restore(key: &str, previous: Option<String>) {
    match previous {
        Some(v) => std::env::set_var(key, v),
        None => std::env::remove_var(key),
    }
}

/// (1) The refusal. Without the override, a format the crate declares it cannot
/// faithfully simulate must stop the run before a single weight is allocated.
#[test]
fn an_unfaithful_format_is_refused_without_the_override() {
    let err = resolve_with("fp80", false).expect_err("fp80 must be refused");
    let msg = err.to_string();
    // The wording is `matrix_runner::resolve_format_faithful`'s, word for word,
    // so an operator meets one sentence on both paths.
    assert!(msg.contains("NON-FAITHFUL FORMAT"), "{msg}");
    assert!(msg.contains("Fp80"), "{msg}");
    assert!(msg.contains("is_faithful()"), "{msg}");
    // The message has to name its own escape hatch, or the operator's only
    // route past it is to read the source.
    assert!(msg.contains(OVERRIDE_ENV), "{msg}");
    assert!(msg.contains("format_faithful=false"), "{msg}");
}

/// The refusal must cover the whole class, not the one format that was
/// measured. Every `is_unsupported_in_f32()` format is an identity passthrough
/// in `fake_quantize_f32`, so every one of them would mint an f32 artifact
/// under another name.
#[test]
fn every_format_unsupported_in_f32_is_refused() {
    let mut checked = Vec::new();
    for &fmt in FormatKind::all() {
        if !fmt.is_unsupported_in_f32() {
            continue;
        }
        let name = fmt.name();
        let err = match resolve_with(name, false) {
            Err(e) => e.to_string(),
            Ok(other) => panic!("{fmt:?} ({name}) resolved to {other:?} instead of being refused"),
        };
        // It must be refused FOR BEING UNFAITHFUL, not because `from_env`
        // failed to recognise its own canonical name - that would be a
        // vacuous pass.
        assert!(
            err.contains("NON-FAITHFUL FORMAT"),
            "{fmt:?} ({name}) was refused for the wrong reason: {err}"
        );
        checked.push(fmt);
    }
    assert!(
        checked.contains(&FormatKind::Fp80),
        "the measured case must be in the sweep; checked={checked:?}"
    );
}

/// (2) The override. The run proceeds - and the artifact has to state that its
/// own label is not a measurement.
#[test]
fn the_override_runs_and_the_sidecar_carries_format_faithful_false() {
    let fmt = resolve_with("fp80", true)
        .expect("the override must let it through")
        .expect("fp80 is not F32, so it resolves to a format");
    assert_eq!(fmt, FormatKind::Fp80);

    with_scoped_checkpoint_dir(|_| {
        let rec = record("IGLA-FORMAT-TRUTH-FP80", fmt.name());
        assert!(
            !rec.format_faithful,
            "an fp80 record must not claim a faithful format"
        );
        let path = checkpoint::write_sidecar(&rec).expect("sidecar write");
        let doc = read_json(&path);
        assert_eq!(doc["fake_quant_format"], serde_json::json!("fp80"));
        assert_eq!(doc["format_faithful"], serde_json::json!(false));
        // The retraction is worthless if the schema tag does not tell a reader
        // the key exists to be looked for. `/9` now also promises a
        // scope-relative `path` - a redefinition of a tag that was never
        // persisted, licensed and evidenced at `CHECKPOINT_RECORD_SCHEMA` - and
        // the test below holds that second half.
        assert_eq!(
            doc["schema"],
            serde_json::json!("trios-checkpoint-record/9")
        );
    });
}

/// The ordinary case must be unaffected: a kernel that really runs is not
/// refused, and its record says so.
#[test]
fn a_faithful_format_is_not_refused_and_stamps_true() {
    let fmt = resolve_with("fp16", false)
        .expect("fp16 is faithful and must not be refused")
        .expect("fp16 is not F32");
    assert_eq!(fmt, FormatKind::Fp16);

    with_scoped_checkpoint_dir(|_| {
        let rec = record("IGLA-FORMAT-TRUTH-FP16", fmt.name());
        assert!(rec.format_faithful);
        let path = checkpoint::write_sidecar(&rec).expect("sidecar write");
        assert_eq!(read_json(&path)["format_faithful"], serde_json::json!(true));
    });
}

/// The other half of what schema 9 now promises: a sidecar publishes no
/// absolute path, and above all not the builder's home directory.
///
/// Schema 7 gave `platform.source_digest_scope` this treatment and documented
/// at length why - "every locally produced sidecar published the builder's home
/// directory", and those sidecars are committed as evidence and uploaded as CI
/// artifacts - while `CheckpointRecord::path`, three fields above it, kept
/// writing the absolute path. Two tracked records under `evidence/` still carry
/// `/Users/<name>/trios-trainer-igla/...` there, in the same artifact set whose
/// provenance script proves that string no longer appears in the binary.
///
/// The checkpoint is SAVED here rather than invented, so the string under test
/// is the one the product derives from a real absolute path under a real
/// absolute `TRIOS_CHECKPOINT_DIR` - which is precisely the configuration that
/// leaked. The assertion is over the sidecar's whole TEXT, not over one field:
/// a document that stops leaking through `path` and starts leaking through the
/// next key has fixed nothing.
#[test]
fn a_sidecar_written_under_an_absolute_checkpoint_dir_publishes_no_home_directory() {
    with_scoped_checkpoint_dir(|dir| {
        assert!(dir.is_absolute(), "the leak needs an absolute dir to leak");
        let canon = "IGLA-FORMAT-TRUTH-PATH";
        let mut rec = record(canon, "f32");
        // A real save: the path in the record is then the path of a file that
        // exists, resolved by `checkpoint::save`, not a literal typed here.
        let payload = b"not a checkpoint, and nothing here reads it as one".to_vec();
        let saved = checkpoint::save(canon, rec.step as usize, &payload).expect("save");
        assert!(saved.path.is_absolute(), "{:?}", saved.path);
        assert!(
            saved.path.starts_with(dir),
            "the save must land under the scoped dir: {:?}",
            saved.path
        );
        rec.path = checkpoint::scope_relative_artifact_path(&saved.path);
        rec.sha256 = saved.sha256.clone();
        rec.bytes = saved.bytes;

        let side = checkpoint::write_sidecar(&rec).expect("sidecar write");
        let text = std::fs::read_to_string(&side).expect("sidecar readable");

        // The field itself: a name, not a location.
        let doc: serde_json::Value = serde_json::from_str(&text).expect("sidecar parses");
        let published = doc["path"].as_str().expect("path is a string");
        assert!(
            !published.starts_with('/'),
            "the record published an absolute path: {published}"
        );
        assert!(
            published.ends_with(&format!("{}.bin", rec.step)),
            "the record must still name the artifact: {published}"
        );

        // The document: no absolute location anywhere in it.
        let dir_str = dir.to_string_lossy().into_owned();
        assert!(
            !text.contains(&dir_str),
            "the sidecar published its absolute checkpoint dir {dir_str}"
        );
        assert!(!text.contains("/Users"), "the sidecar published /Users");
        let home = std::env::var("HOME").unwrap_or_default();
        if home.len() > 1 {
            assert!(
                !text.contains(&home),
                "the sidecar published $HOME ({home})"
            );
        }
    });
}

/// The second defect in the same function: an unrecognised spelling was
/// swallowed. `int_8` is one underscore from a format this build implements,
/// and it produced no output mentioning a format at all.
#[test]
fn an_unrecognised_spelling_is_refused_naming_the_variable() {
    let err = resolve_with("int_8", false).expect_err("int_8 must be refused");
    let msg = err.to_string();
    assert!(msg.contains(FORMAT_ENV), "{msg}");
    assert!(msg.contains("int_8"), "{msg}");
    // The accepted spellings have to be in the message; "not a format this
    // build knows" without a list is the same dead end as silence.
    assert!(msg.contains("fp16"), "{msg}");
    assert!(msg.contains("gf16"), "{msg}");
}

/// The override is for UNFAITHFUL formats, not for typos: there is nothing to
/// stamp `format_faithful=false` on when the string names no format.
#[test]
fn the_override_does_not_excuse_an_unrecognised_spelling() {
    let err =
        resolve_with("int_8", true).expect_err("int_8 must be refused even with the override");
    assert!(err.to_string().contains("int_8"), "{err}");
}

// ---------------------------------------------------------------------------
// Sidecar fixtures. Same shape as `tests/sidecar_overwrite_guard.rs`: every
// case writes under its OWN temporary `TRIOS_CHECKPOINT_DIR` and its own canon
// name, so nothing in `checkpoints/` is read or touched.
// ---------------------------------------------------------------------------

static CKPT_DIR_LOCK: Mutex<()> = Mutex::new(());

fn with_scoped_checkpoint_dir<T>(body: impl FnOnce(&Path) -> T) -> T {
    let guard = CKPT_DIR_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().expect("tempdir");
    let previous = std::env::var("TRIOS_CHECKPOINT_DIR").ok();
    std::env::set_var("TRIOS_CHECKPOINT_DIR", dir.path());
    std::env::remove_var("TRIOS_ALLOW_SIDECAR_OVERWRITE");
    let out = body(dir.path());
    restore("TRIOS_CHECKPOINT_DIR", previous);
    drop(guard);
    out
}

fn read_json(path: &Path) -> serde_json::Value {
    let bytes = std::fs::read(path).expect("sidecar readable");
    serde_json::from_slice(&bytes).expect("sidecar parses")
}

/// A complete record whose `format_faithful` is DERIVED from its own
/// `fake_quant_format`, by the same function `train_loop` uses. Hardcoding the
/// bool here would let the fixture assert a consistency the product does not
/// have.
///
/// `git_provenance` is ASSERTED so no case inspects a working tree.
fn record(canon: &str, fake_quant_format: &str) -> CheckpointRecord {
    CheckpointRecord {
        schema: CHECKPOINT_RECORD_SCHEMA.to_string(),
        canon_name: canon.to_string(),
        seed: 47,
        step: 20,
        // Schema 9: RELATIVE to the digest scope. The path case overwrites this
        // with the product's own derivation from a real save.
        path: format!("checkpoints/{canon}/20.bin"),
        sha256: "0e8bcbb2".to_string() + &"0".repeat(56),
        bytes: 852_272,
        format_version: CHECKPOINT_FORMAT_VERSION,
        hidden: 384,
        d_model: 384,
        num_attn_layers: 2,
        optimizer: "adamw".to_string(),
        fake_quant_format: fake_quant_format.to_string(),
        data_synthetic: false,
        steps_total: 20,
        gf16_floor_every: 1,
        eval_every: 10,
        final_val_bpb: Some(6.812_345_6),
        min_observed_val_bpb: Some(6.812_345_6),
        ema_bpb: Some(6.9),
        git_sha: "deadbeef".to_string(),
        git_provenance: GIT_PROVENANCE_ASSERTED.to_string(),
        git_dirty: None,
        corpus: CorpusProvenance::default(),
        run_id: None,
        ledger: "skipped-not-opted-in".to_string(),
        ts: "2026-08-05T00:00:00Z".to_string(),
        lr: Some(0.003_000_000_026_077_032),
        attn_scale: 0.1,
        attn_seq: 8,
        platform: PlatformProvenance::default(),
        source_sha256: "c".repeat(64),
        trainer: TrainerProvenance {
            path: "target/release/trios-train".to_string(),
            sha256: "b".repeat(64),
            provenance: TRAINER_PROVENANCE_SELF_HASHED.to_string(),
        },
        vocab: 128,
        gf16_enabled: true,
        eval_chunks: Some(40),
        eval_tokens: Some(5_160),
        eval_seq: Some(129),
        val_bpb_stderr: Some(0.055_186_1),
        optimizer_params: Some(OptimizerParams {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.04,
            source: "tests::format_label_truth".to_string(),
        }),
        format_faithful: checkpoint::format_label_faithful(fake_quant_format),
    }
}
