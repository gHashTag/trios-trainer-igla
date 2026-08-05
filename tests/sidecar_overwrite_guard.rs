//! The sidecar is the evidence document. It may GAIN information; it may not
//! restate any it already carries.
//!
//! `save_scoped` has refused to overwrite a `.bin` holding different bytes ever
//! since a sweep destroyed two of three artifacts - but it PERMITS re-saving
//! identical bytes, and that is exactly what a second run of the same recipe
//! does. Measured: same canon name, same step, identical checkpoint sha
//! `2af80d15`, and `final_val_bpb` 4.5670576 -> 4.4738498, `val_bpb_stderr`
//! 0.0551861 -> 0.0117720, `eval_chunks` 40 -> 775, exit 0, no warning. The
//! artifact guard was green the whole time because the artifact never changed;
//! what changed was the record of how it had been measured.
//!
//! Every case here writes under its OWN temporary `TRIOS_CHECKPOINT_DIR` and
//! its own canon name, so nothing in `checkpoints/` is read or touched. The
//! environment variable is process-wide, so `with_scoped_checkpoint_dir` holds
//! one mutex for the whole of each case: the tests are serialised against each
//! other rather than racing on the variable they all set.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use trios_trainer::checkpoint::{
    self, CheckpointRecord, CorpusProvenance, OptimizerParams, PlatformProvenance,
    TrainerProvenance, CHECKPOINT_FORMAT_VERSION, CHECKPOINT_RECORD_SCHEMA,
    GIT_PROVENANCE_ASSERTED, TRAINER_PROVENANCE_SELF_HASHED,
};

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Run `body` with `TRIOS_CHECKPOINT_DIR` pointed at a fresh temporary
/// directory, exclusively for the duration.
fn with_scoped_checkpoint_dir<T>(body: impl FnOnce(&Path) -> T) -> T {
    let guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().expect("tempdir");
    let previous = std::env::var("TRIOS_CHECKPOINT_DIR").ok();
    std::env::set_var("TRIOS_CHECKPOINT_DIR", dir.path());
    std::env::remove_var("TRIOS_ALLOW_SIDECAR_OVERWRITE");
    let out = body(dir.path());
    match previous {
        Some(v) => std::env::set_var("TRIOS_CHECKPOINT_DIR", v),
        None => std::env::remove_var("TRIOS_CHECKPOINT_DIR"),
    }
    drop(guard);
    out
}

/// A complete record, with `git_provenance` ASSERTED so that no case here
/// inspects a working tree: `git_untracked` is then `None` by construction and
/// two writes microseconds apart cannot disagree about a repository neither of
/// them looked at.
fn record(canon: &str, ledger: &str) -> CheckpointRecord {
    CheckpointRecord {
        schema: CHECKPOINT_RECORD_SCHEMA.to_string(),
        canon_name: canon.to_string(),
        seed: 47,
        step: 200,
        path: format!("checkpoints/{canon}/200.bin"),
        sha256: "2af80d15".to_string() + &"0".repeat(56),
        bytes: 852_272,
        format_version: CHECKPOINT_FORMAT_VERSION,
        hidden: 384,
        d_model: 384,
        num_attn_layers: 2,
        optimizer: "adamw".to_string(),
        fake_quant_format: "f32".to_string(),
        data_synthetic: false,
        steps_total: 200,
        gf16_floor_every: 1,
        eval_every: 100,
        final_val_bpb: Some(4.567_057_6),
        min_observed_val_bpb: Some(4.567_057_6),
        ema_bpb: Some(5.1),
        git_sha: "deadbeef".to_string(),
        git_provenance: GIT_PROVENANCE_ASSERTED.to_string(),
        git_dirty: None,
        corpus: CorpusProvenance::default(),
        run_id: None,
        ledger: ledger.to_string(),
        ts: "2026-08-03T00:00:00Z".to_string(),
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
            source: "tests::sidecar_overwrite_guard".to_string(),
        }),
    }
}

fn sidecar_of(rec: &CheckpointRecord) -> PathBuf {
    checkpoint::sidecar_path(
        &checkpoint::sanitize_run_name(&rec.canon_name),
        rec.step as usize,
    )
}

fn read_json(path: &Path) -> serde_json::Value {
    let bytes = std::fs::read(path).expect("sidecar readable");
    serde_json::from_slice(&bytes).expect("sidecar parses")
}

/// (a) Nothing on disk, nothing to protect: the first write lands.
#[test]
fn the_first_write_lands() {
    with_scoped_checkpoint_dir(|_| {
        let rec = record("IGLA-GUARD-FIRST", "written");
        let path = checkpoint::write_sidecar(&rec).expect("first write must succeed");
        assert_eq!(path, sidecar_of(&rec));
        let doc = read_json(&path);
        assert_eq!(doc["final_val_bpb"], serde_json::json!(4.567_057_6));
        assert_eq!(doc["eval_chunks"], serde_json::json!(40));
        assert_eq!(doc["ledger"], serde_json::json!("written"));
    });
}

/// (b) Re-writing the SAME document is not an overwrite: the file that would
/// result is the file that is already there. Same rule `save_scoped` applies to
/// identical bytes.
#[test]
fn an_identical_rewrite_is_allowed() {
    with_scoped_checkpoint_dir(|_| {
        let rec = record("IGLA-GUARD-IDENTICAL", "written");
        let path = checkpoint::write_sidecar(&rec).expect("first write");
        let before = std::fs::read(&path).expect("read back");
        checkpoint::write_sidecar(&rec).expect("an identical rewrite must be allowed");
        let after = std::fs::read(&path).expect("read back");
        assert_eq!(
            before, after,
            "an identical rewrite must leave the bytes alone"
        );
    });
}

/// (c) The documented two-phase write: `pending` first, then the real ledger
/// outcome plus the fields the first write could not yet state. It only ADDS,
/// so it is allowed - and this is the sequence `train_loop` performs on every
/// checkpoint, unchanged.
#[test]
fn a_pending_record_may_be_finalised_by_one_that_only_fills_it_in() {
    with_scoped_checkpoint_dir(|_| {
        let mut pending = record("IGLA-GUARD-PENDING", "pending");
        pending.final_val_bpb = None;
        pending.val_bpb_stderr = None;
        pending.eval_chunks = None;
        let path = checkpoint::write_sidecar(&pending).expect("pending write");
        let on_disk = read_json(&path);
        assert_eq!(on_disk["ledger"], serde_json::json!("pending"));
        assert_eq!(on_disk["final_val_bpb"], serde_json::Value::Null);

        // The finalising write: a terminal ledger, its own timestamp, and the
        // three measurements that were null.
        let mut final_rec = record("IGLA-GUARD-PENDING", "written");
        final_rec.ts = "2026-08-03T00:00:01Z".to_string();
        checkpoint::write_sidecar(&final_rec)
            .expect("finalising a pending record must be allowed");

        let on_disk = read_json(&path);
        assert_eq!(on_disk["ledger"], serde_json::json!("written"));
        assert_eq!(on_disk["ts"], serde_json::json!("2026-08-03T00:00:01Z"));
        assert_eq!(on_disk["final_val_bpb"], serde_json::json!(4.567_057_6));
        assert_eq!(on_disk["val_bpb_stderr"], serde_json::json!(0.055_186_1));
        assert_eq!(on_disk["eval_chunks"], serde_json::json!(40));
    });
}

/// (d) THE DEFECT. A second run of the same recipe re-measures and re-states.
/// The write is refused, it names the key, and the document that was already
/// there is byte-for-byte what it was.
#[test]
fn a_rewrite_that_changes_a_measurement_is_refused() {
    with_scoped_checkpoint_dir(|_| {
        let first = record("IGLA-GUARD-REMEASURED", "written");
        let path = checkpoint::write_sidecar(&first).expect("first write");
        let before = std::fs::read(&path).expect("read back");

        let mut second = record("IGLA-GUARD-REMEASURED", "written");
        second.final_val_bpb = Some(4.473_849_8);
        let err = checkpoint::write_sidecar(&second)
            .expect_err("a changed measurement must be refused");
        let msg = format!("{err:#}");

        assert!(
            msg.contains("final_val_bpb"),
            "the refusal must name the differing key: {msg}"
        );
        assert!(
            msg.contains("4.5670576") && msg.contains("4.4738498"),
            "the refusal must carry both values: {msg}"
        );
        assert!(
            msg.contains(&path.file_name().unwrap().to_string_lossy().to_string()),
            "the refusal must name the destination path: {msg}"
        );
        assert!(
            msg.contains("IGLA-GUARD-REMEASURED") && msg.contains("200"),
            "the refusal must name canon_name and step: {msg}"
        );
        assert!(
            msg.contains("TRIOS_ALLOW_SIDECAR_OVERWRITE"),
            "the refusal must name its one escape hatch: {msg}"
        );

        let after = std::fs::read(&path).expect("read back");
        assert_eq!(
            before, after,
            "a refused write must leave the record byte-identical"
        );

        // The measured shape of the defect: the whole reading moved together -
        // the BPB, its error bar and its coverage. All three must be named.
        let mut whole_reading = record("IGLA-GUARD-REMEASURED", "written");
        whole_reading.final_val_bpb = Some(4.473_849_8);
        whole_reading.val_bpb_stderr = Some(0.011_772_0);
        whole_reading.eval_chunks = Some(775);
        let msg = format!(
            "{:#}",
            checkpoint::write_sidecar(&whole_reading)
                .expect_err("a re-measured reading must be refused")
        );
        for key in ["final_val_bpb", "val_bpb_stderr", "eval_chunks"] {
            assert!(msg.contains(key), "the refusal must name {key}: {msg}");
        }
        assert_eq!(
            before,
            std::fs::read(&path).expect("read back"),
            "a refused write must leave the record byte-identical"
        );
    });
}

/// (e) A terminal ledger is a statement, so re-opening it as `pending` - which
/// is what the FIRST write of a second run does - is refused before that run
/// can restate anything else.
#[test]
fn a_second_run_is_refused_at_its_pending_write() {
    with_scoped_checkpoint_dir(|_| {
        let first = record("IGLA-GUARD-SECOND-RUN", "written");
        let path = checkpoint::write_sidecar(&first).expect("first write");
        let before = std::fs::read(&path).expect("read back");

        let mut second = record("IGLA-GUARD-SECOND-RUN", "pending");
        second.ts = "2026-08-03T01:00:00Z".to_string();
        let err = checkpoint::write_sidecar(&second)
            .expect_err("re-opening a terminal record must be refused");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("ledger"),
            "the refusal must name the ledger key: {msg}"
        );
        assert_eq!(
            before,
            std::fs::read(&path).expect("read back"),
            "a refused write must leave the record byte-identical"
        );
    });
}

/// (f) The escape hatch exists, is explicit, and is the only way through.
#[test]
fn the_escape_hatch_permits_the_write_and_says_so() {
    with_scoped_checkpoint_dir(|_| {
        let first = record("IGLA-GUARD-HATCH", "written");
        let path = checkpoint::write_sidecar(&first).expect("first write");

        let mut second = record("IGLA-GUARD-HATCH", "written");
        second.final_val_bpb = Some(4.473_849_8);
        checkpoint::write_sidecar(&second).expect_err("refused without the hatch");

        std::env::set_var("TRIOS_ALLOW_SIDECAR_OVERWRITE", "1");
        let result = checkpoint::write_sidecar(&second);
        std::env::remove_var("TRIOS_ALLOW_SIDECAR_OVERWRITE");
        result.expect("the escape hatch must permit the write");

        assert_eq!(
            read_json(&path)["final_val_bpb"],
            serde_json::json!(4.473_849_8),
            "the deliberate overwrite must actually land"
        );
    });
}
