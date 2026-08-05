//! The window audit: a segment must be byte-identical to the monolith.
//!
//! `ckpt_replay` is called a spot-check verifier but re-executes from step 0,
//! because the `TRIOSCKP` container serialises weights only - no AdamW
//! moments, no step counters, no batch-sampler state - and until now no binary
//! accepted a warm start. Verification therefore cost exactly what production
//! cost, so for a large model nobody re-runs anything and the scheme reduces
//! to a self-signed affidavit.
//!
//! The claim this file has to hold up is narrow and mechanical: run
//! `0 -> STEPS` in one process, then run `0 -> HALF` and resume `HALF ->
//! STEPS` in another, and the two final `.bin` files must have the same
//! SHA-256. If they do not, the segment is not a segment and the audit proves
//! nothing - so this test asserts byte equality and reports both digests when
//! it fails, rather than grading the two runs on a tolerance.
//!
//! Everything else here is a refusal. A resume that cannot be proved to
//! continue the run it claims to continue must stop the process with a named
//! reason: warm-starting from zeroed moments would run to completion, print a
//! plausible BPB, and be a segment of no run at all.
//!
//! `docs/WINDOW-AUDIT.md` states what this buys and - just as important - what
//! it does not: it is a SAME-MACHINE claim, and the cross-architecture
//! boundary in `docs/CROSS-ARCH-DIVERGENCE.md` is untouched by it.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use trios_trainer::checkpoint::{
    self, ResumeOptimizerState, ResumeRecord, RESUME_REASON_CADENCE, RESUME_REASON_CORPUS,
    RESUME_REASON_DIGEST, RESUME_REASON_MAGIC, RESUME_REASON_MISSING, RESUME_REASON_MUON,
    RESUME_REASON_RECIPE, RESUME_REASON_SHAPE, RESUME_REASON_TRAILING, RESUME_REASON_TRUNCATED,
    RESUME_REASON_VERSION, RESUME_REASON_WEIGHT_DIGEST, RESUME_REFUSAL_PREFIX,
};
use trios_trainer::train_loop::{self, TrainArgs};

/// The whole training case runs under this, because `TRIOS_CHECKPOINT_DIR`
/// and friends are process-wide.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// The train/val pair of the three-way tinyshakespeare split
/// (`data/README.md`): `train_core ++ test ++ val` is the canonical corpus,
/// and `tiny_shakespeare_test.txt` is read by neither run here. Byte
/// disjointness is not taken on trust - `assert_train_val_disjoint` checks
/// every window of both streams before step 1 of each run below.
const TRAIN_PATH: &str = "data/tiny_shakespeare_train_core.txt";
const VAL_PATH: &str = "data/tiny_shakespeare_val.txt";

/// A budget small enough for `cargo test` and large enough to exercise the
/// state that matters: several optimizer steps on both sides of the split, so
/// the bias-correction counters, the moments and the sampler have all moved.
const STEPS: usize = 8;
const HALF: usize = 4;
const HIDDEN: usize = 64;
const SEED: u64 = 47;
const LR: f32 = 0.003;

fn set(key: &str, value: &str) {
    std::env::set_var(key, value);
}

/// The environment both runs execute under, restored by the caller's guard.
///
/// `TRIOS_EVAL_CHUNKS=8` is the minimum `check_train_val_disjoint` accepts, so
/// the evaluation is as cheap as the guard permits; the DSN variables are
/// cleared because a test must not write rows into whatever database happens
/// to be in the developer's shell.
fn scoped_env(dir: &Path, canon: &str) {
    set("TRIOS_CHECKPOINT_DIR", &dir.to_string_lossy());
    set("TRIOS_CANON_NAME", canon);
    set("TRIOS_CHECKPOINT_EVERY", &HALF.to_string());
    set("TRIOS_EVAL_CHUNKS", "8");
    for k in [
        "DATABASE_URL",
        "NEON_DATABASE_URL",
        "TRIOS_NEON_DSN",
        "TRIOS_DATABASE_URL",
        "TRIOS_CHECKPOINT_INIT",
        "TRIOS_CHECKPOINT_DISABLE",
        "TRIOS_FORMAT_TYPE",
        "GF16_ENABLED",
        "TRIOS_GF16_DISABLE",
        "TRIOS_GF16_FLOOR_EVERY",
        "HIDDEN_DIM",
        "NUM_ATTN_LAYERS",
        "TRIOS_ATTN_SCALE",
        "TRIOS_ATTN_SEQ",
    ] {
        std::env::remove_var(k);
    }
}

fn args() -> TrainArgs {
    TrainArgs {
        seed: SEED,
        steps: STEPS,
        hidden: HIDDEN,
        lr: LR,
        attn_layers: 2,
        eval_every: HALF,
        train_path: TRAIN_PATH.to_string(),
        val_path: VAL_PATH.to_string(),
    }
}

fn artifact(dir: &Path, canon: &str, step: usize) -> PathBuf {
    dir.join(canon).join(format!("{step}.bin"))
}

fn digest_of(path: &Path) -> String {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    checkpoint::sha256_hex(&raw)
}

/// THE demonstration, in a tempdir with a tiny budget.
///
/// Run A is monolithic `0 -> STEPS`. Run B resumes from A's step-`HALF`
/// artifact and its record, and runs `HALF -> STEPS`. The two step-`STEPS`
/// artifacts must be byte-identical.
///
/// Both runs write into their OWN checkpoint directory. Sharing one would let
/// `save_scoped`'s overwrite guard turn a divergence into an error message
/// about two runs writing one path, which is a different fact than the one
/// under test.
#[test]
fn a_resumed_segment_reproduces_the_monolithic_run_byte_for_byte() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tempfile::tempdir().expect("tempdir");
    let mono_dir = tmp.path().join("monolith");
    let seg_dir = tmp.path().join("segment");

    scoped_env(&mono_dir, "window-audit-monolith");
    train_loop::run_single(&args()).expect("monolithic run");

    let mono_half = artifact(&mono_dir, "window-audit-monolith", HALF);
    let mono_final = artifact(&mono_dir, "window-audit-monolith", STEPS);
    let mono_resume = checkpoint::resume_sidecar_path(&mono_half);
    assert!(mono_half.is_file(), "no {HALF}.bin from the monolithic run");
    assert!(
        mono_final.is_file(),
        "no {STEPS}.bin from the monolithic run"
    );
    assert!(
        mono_resume.is_file(),
        "no resume record beside {mono_half:?}; the artifact is unauditable \
         without one"
    );

    // The record must be self-describing: an auditor reads it before deciding
    // which window to challenge.
    let rec = checkpoint::load_resume_file(&mono_resume).expect("record loads");
    assert_eq!(rec.step, HALF as u64);
    assert_eq!(rec.steps_total, STEPS as u64);
    assert_eq!(rec.seed, SEED);
    assert_eq!(rec.optimizer, "adamw");
    assert_eq!(
        rec.weight_sha256,
        digest_of(&mono_half),
        "the record must pair with the artifact beside it"
    );
    // MEASURED, not assumed: `run_single` builds embed, ctx[0..NUM_CTX],
    // proj, attn_down, attn_up, head, attn_w - that is 6 + NUM_CTX = 12
    // instances, not the 7 + NUM_CTX a reading of the construction site
    // suggests (`opt_ctx` is one binding holding NUM_CTX of them). A record
    // covering fewer would restore some moments and zero the rest.
    assert_eq!(
        rec.optimizers.len(),
        6 + 6,
        "the AdamW path runs 6 + NUM_CTX instances"
    );
    let names: Vec<&str> = rec.optimizers.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "embed",
            "ctx0",
            "ctx1",
            "ctx2",
            "ctx3",
            "ctx4",
            "ctx5",
            "proj",
            "attn_down",
            "attn_up",
            "head",
            "attn_w"
        ],
        "the canonical instance order is what stops attn_up's moments landing \
         in attn_down, which has the identical element count"
    );
    assert!(
        rec.optimizers.iter().any(|o| o.m.iter().any(|&x| x != 0.0)),
        "every moment in the record is zero after {HALF} steps, which means \
         the capture ran before the optimizer did"
    );

    scoped_env(&seg_dir, "window-audit-segment");
    train_loop::run_single_resumed(&args(), Some(&mono_half)).expect("resumed segment");

    let seg_final = artifact(&seg_dir, "window-audit-segment", STEPS);
    assert!(seg_final.is_file(), "no {STEPS}.bin from the resumed run");
    let mono_digest = digest_of(&mono_final);
    let seg_digest = digest_of(&seg_final);
    assert_eq!(
        mono_digest, seg_digest,
        "the resumed segment did not reproduce the monolithic run.\n  \
         monolith {STEPS}.bin sha256 = {mono_digest}\n  \
         segment  {STEPS}.bin sha256 = {seg_digest}\n\
         Some state that decides the weights is not in the resume record; \
         localise it before this format is used to grade anybody."
    );

    // The segment must not have produced a `{HALF}.bin` of its own: it started
    // there, it did not execute those steps, and minting an artifact for steps
    // it did not run would be the audit trail claiming work that never
    // happened.
    assert!(
        !artifact(&seg_dir, "window-audit-segment", HALF).exists(),
        "the resumed run wrote an artifact for a step it never executed"
    );

    // End-to-end refusal, on the same artifacts: a record whose weight digest
    // does not match the file it is loaded beside.
    let tampered_dir = tmp.path().join("tampered");
    let tampered_bin = tampered_dir.join("window-audit-tampered").join("4.bin");
    std::fs::create_dir_all(tampered_bin.parent().unwrap()).unwrap();
    std::fs::copy(&mono_half, &tampered_bin).unwrap();
    let mut broken = rec.clone();
    broken.weight_sha256 = "0".repeat(64);
    std::fs::write(
        checkpoint::resume_sidecar_path(&tampered_bin),
        checkpoint::resume_to_bytes(&broken).expect("encode"),
    )
    .unwrap();

    let out_dir = tmp.path().join("refused");
    scoped_env(&out_dir, "window-audit-refused");
    let err = train_loop::run_single_resumed(&args(), Some(&tampered_bin))
        .expect_err("a record that does not pair with these weights must stop the run");
    assert_refusal(&format!("{err:#}"), RESUME_REASON_WEIGHT_DIGEST);
    assert!(
        !out_dir.exists(),
        "the refused run still produced artifacts; a refusal must cost no \
         training and mint nothing"
    );
}

/// A refusal must name the format and the reason, so an audit log says which
/// KIND of mismatch stopped it.
fn assert_refusal(message: &str, reason: &str) {
    assert!(
        message.contains(RESUME_REFUSAL_PREFIX),
        "not a resume refusal: {message}"
    );
    assert!(
        message.contains(reason),
        "expected reason {reason:?} in: {message}"
    );
}

/// A record with two small instances, enough to exercise every field without
/// allocating a model.
fn sample_record() -> ResumeRecord {
    ResumeRecord {
        seed: SEED,
        step: 100,
        rng_s: 0x0123_4567_89ab_cdef,
        steps_total: 200,
        eval_every: 50,
        gf16_floor_every: 1,
        hidden: 64,
        d_model: 64,
        num_attn_layers: 2,
        vocab: 128,
        dim: 64,
        num_ctx: 6,
        base_lr: LR,
        weight_decay: 0.04,
        gf16_enabled: true,
        data_synthetic: false,
        ema_bpb: Some(3.25),
        min_observed_val_bpb: Some(3.1),
        weight_sha256: "a".repeat(64),
        train_sha256: "b".repeat(64),
        val_sha256: "c".repeat(64),
        fake_quant_format: "f32".to_string(),
        optimizer: "adamw".to_string(),
        optimizers: vec![
            ResumeOptimizerState {
                name: "embed".to_string(),
                step: 100,
                m: vec![0.5, -0.25, 1e-9],
                v: vec![1.0, 2.0, 3.0],
            },
            ResumeOptimizerState {
                name: "attn_w".to_string(),
                step: 0,
                m: vec![0.0, 0.0],
                v: vec![0.0, 0.0],
            },
        ],
    }
}

fn expectation_for(rec: &ResumeRecord) -> checkpoint::ResumeExpectation {
    checkpoint::ResumeExpectation {
        seed: rec.seed,
        steps_total: rec.steps_total,
        eval_every: rec.eval_every,
        gf16_floor_every: rec.gf16_floor_every,
        gf16_enabled: rec.gf16_enabled,
        hidden: rec.hidden,
        d_model: rec.d_model,
        num_attn_layers: rec.num_attn_layers,
        vocab: rec.vocab,
        dim: rec.dim,
        num_ctx: rec.num_ctx,
        base_lr: rec.base_lr,
        weight_decay: rec.weight_decay,
        fake_quant_format: rec.fake_quant_format.clone(),
        weight_sha256: rec.weight_sha256.clone(),
        train_sha256: rec.train_sha256.clone(),
        val_sha256: rec.val_sha256.clone(),
    }
}

#[test]
fn the_record_round_trips_exactly() {
    let rec = sample_record();
    let bytes = checkpoint::resume_to_bytes(&rec).expect("encode");
    let back = checkpoint::resume_from_bytes(&bytes).expect("decode");
    assert_eq!(rec, back, "the round trip is not the identity");
    // The moments are f32 bit patterns, not decimals: a denormal that came
    // back as zero would be a silently different optimizer.
    assert_eq!(back.optimizers[0].m[2].to_bits(), 1e-9f32.to_bits());
}

#[test]
fn a_truncated_record_is_refused() {
    let bytes = checkpoint::resume_to_bytes(&sample_record()).expect("encode");
    for cut in [0usize, 8, checkpoint::RESUME_HEADER_LEN, bytes.len() - 1] {
        let err = checkpoint::resume_from_bytes(&bytes[..cut])
            .expect_err("a truncated record must not load");
        let msg = format!("{err:#}");
        assert!(
            msg.contains(RESUME_REASON_TRUNCATED) || msg.contains(RESUME_REASON_MAGIC),
            "cut at {cut}: {msg}"
        );
    }
}

#[test]
fn a_truncated_record_on_disk_is_refused_by_the_loader() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("100.resume");
    let bytes = checkpoint::resume_to_bytes(&sample_record()).expect("encode");
    std::fs::write(&path, &bytes[..bytes.len() - 16]).expect("write");
    let err = checkpoint::load_resume_file(&path).expect_err("a short file must not load");
    assert_refusal(&format!("{err:#}"), RESUME_REASON_TRUNCATED);
}

#[test]
fn trailing_bytes_are_refused() {
    let mut bytes = checkpoint::resume_to_bytes(&sample_record()).expect("encode");
    bytes.push(0);
    let err = checkpoint::resume_from_bytes(&bytes).expect_err("a longer file must not load");
    assert_refusal(&format!("{err:#}"), RESUME_REASON_TRAILING);
}

#[test]
fn a_flipped_byte_is_refused_by_the_digest() {
    let mut bytes = checkpoint::resume_to_bytes(&sample_record()).expect("encode");
    // A moment, not a header field: the digest is what covers the payload,
    // and an edited moment is the one tamper the field checks cannot see.
    let payload = checkpoint::RESUME_HEADER_LEN + 2 * 32 + 4;
    bytes[payload] ^= 0x01;
    let err = checkpoint::resume_from_bytes(&bytes).expect_err("an edited record must not load");
    assert_refusal(&format!("{err:#}"), RESUME_REASON_DIGEST);
}

#[test]
fn a_foreign_container_is_refused() {
    let mut bytes = checkpoint::resume_to_bytes(&sample_record()).expect("encode");
    bytes[0..8].copy_from_slice(checkpoint::CHECKPOINT_MAGIC);
    let err = checkpoint::resume_from_bytes(&bytes)
        .expect_err("a TRIOSCKP file must not be read as a resume record");
    assert_refusal(&format!("{err:#}"), RESUME_REASON_MAGIC);
}

#[test]
fn an_unknown_version_is_refused() {
    let mut bytes = checkpoint::resume_to_bytes(&sample_record()).expect("encode");
    bytes[8..12].copy_from_slice(&(checkpoint::RESUME_FORMAT_VERSION + 1).to_le_bytes());
    let err = checkpoint::resume_from_bytes(&bytes).expect_err("a future version must not load");
    // The digest covers the version field too, so this reports whichever check
    // runs first; both are refusals and neither is a warm start.
    let msg = format!("{err:#}");
    assert!(
        msg.contains(RESUME_REASON_VERSION) || msg.contains(RESUME_REASON_DIGEST),
        "{msg}"
    );
}

#[test]
fn a_mismatched_weight_digest_is_refused() {
    let rec = sample_record();
    let mut want = expectation_for(&rec);
    want.weight_sha256 = "d".repeat(64);
    let err = checkpoint::verify_resume(&rec, &want).expect_err("wrong weights must be refused");
    assert_refusal(&format!("{err:#}"), RESUME_REASON_WEIGHT_DIGEST);
}

#[test]
fn a_wrong_corpus_hash_is_refused() {
    let rec = sample_record();
    let mut want = expectation_for(&rec);
    want.val_sha256 = "e".repeat(64);
    let err = checkpoint::verify_resume(&rec, &want).expect_err("a different val must be refused");
    assert_refusal(&format!("{err:#}"), RESUME_REASON_CORPUS);

    let mut want = expectation_for(&rec);
    want.train_sha256 = "f".repeat(64);
    let err =
        checkpoint::verify_resume(&rec, &want).expect_err("a different train must be refused");
    assert_refusal(&format!("{err:#}"), RESUME_REASON_CORPUS);
}

#[test]
fn a_differing_cadence_is_refused() {
    let rec = sample_record();
    let mut want = expectation_for(&rec);
    want.eval_every = rec.eval_every + 1;
    let err = checkpoint::verify_resume(&rec, &want).expect_err("a different eval cadence");
    assert_refusal(&format!("{err:#}"), RESUME_REASON_CADENCE);

    let mut want = expectation_for(&rec);
    want.gf16_floor_every = rec.gf16_floor_every + 1;
    let err = checkpoint::verify_resume(&rec, &want).expect_err("a different gf16 cadence");
    assert_refusal(&format!("{err:#}"), RESUME_REASON_CADENCE);
}

#[test]
fn a_differing_shape_is_refused() {
    let rec = sample_record();
    let mut want = expectation_for(&rec);
    want.hidden = rec.hidden * 2;
    let err = checkpoint::verify_resume(&rec, &want).expect_err("a different hidden");
    assert_refusal(&format!("{err:#}"), RESUME_REASON_SHAPE);
}

/// The schedule inputs. `cosine_lr(step, steps_total, base_lr)` means a
/// segment run to a different total applies different learning rates to the
/// same step numbers, which is a divergence with no error message.
#[test]
fn a_differing_recipe_is_refused() {
    let rec = sample_record();
    for mutate in [
        (|w: &mut checkpoint::ResumeExpectation| w.steps_total += 1) as fn(&mut _),
        |w: &mut checkpoint::ResumeExpectation| w.base_lr *= 2.0,
        |w: &mut checkpoint::ResumeExpectation| w.weight_decay = 0.0,
        |w: &mut checkpoint::ResumeExpectation| w.seed += 1,
        |w: &mut checkpoint::ResumeExpectation| w.gf16_enabled = !w.gf16_enabled,
        |w: &mut checkpoint::ResumeExpectation| w.fake_quant_format = "gf16".to_string(),
    ] {
        let mut want = expectation_for(&rec);
        mutate(&mut want);
        let err =
            checkpoint::verify_resume(&rec, &want).expect_err("a different recipe must be refused");
        assert_refusal(&format!("{err:#}"), RESUME_REASON_RECIPE);
    }
}

#[test]
fn a_finished_run_has_nothing_to_resume() {
    let mut rec = sample_record();
    rec.step = rec.steps_total;
    let err = checkpoint::verify_resume(&rec, &expectation_for(&rec))
        .expect_err("a finished run must not be resumable");
    assert_refusal(&format!("{err:#}"), checkpoint::RESUME_REASON_NOTHING_TO_DO);
}

/// A weights-only artifact - every `.bin` this repository published before
/// this format existed - cannot be warm-started, and must say so instead of
/// starting cold with zeroed moments.
#[test]
fn a_checkpoint_without_a_record_is_refused() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let bin = tmp.path().join("100.bin");
    std::fs::write(&bin, b"not really a checkpoint").expect("write");
    let err = checkpoint::resolve_resume_pair(&bin).expect_err("no record beside it");
    assert_refusal(&format!("{err:#}"), RESUME_REASON_MISSING);
}

/// Either half of the pair may be named on the command line.
#[test]
fn both_halves_of_the_pair_resolve_to_the_same_pair() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let bin = tmp.path().join("100.bin");
    let res = tmp.path().join("100.resume");
    std::fs::write(&bin, b"weights").expect("write");
    std::fs::write(&res, b"record").expect("write");
    assert_eq!(
        checkpoint::resolve_resume_pair(&bin).expect("from .bin"),
        checkpoint::resolve_resume_pair(&res).expect("from .resume")
    );
}

/// The Muon path is refused, not faked. `MuonOptimizer` carries a momentum
/// buffer and its own step counter that version 1 does not serialise, and the
/// refusal has to happen before any work: an audit command that costs a
/// training run before saying no is not usable.
#[test]
fn the_muon_path_refuses_a_warm_start() {
    for optimizer in ["muon", "muon-cwd"] {
        let err = train_loop::run_with_optimizer_resumed(
            optimizer,
            &args(),
            Some(Path::new("checkpoints/does-not-exist/100.bin")),
        )
        .expect_err("the Muon path must refuse a warm start");
        let msg = format!("{err:#}");
        assert_refusal(&msg, RESUME_REASON_MUON);
        assert!(
            msg.contains("does not serialise"),
            "the refusal must say WHAT is missing: {msg}"
        );
    }
}
