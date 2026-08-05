//! Champion reproduction guard - what it can and cannot guard.
//!
//! This file used to assert that `configs/champion.toml` carried
//! `d_model = 256`, `vocab_size = 32000`, `seq_len = 1024`, both AdamW betas,
//! `weight_decay`, `schedule`, `warmup_steps` and the whole `[objective]`
//! block, and that a full run reproduced `BPB = 2.2393 +/- 0.01` per
//! `gHashTag/trios@2446855`.
//!
//! Three problems, all fatal to those assertions:
//!
//! 1. **The trainer cannot execute nearly all of it.** `train_loop::run()`
//!    delegates to `config_train_args`, which builds `TrainArgs` from
//!    `seed`, `steps`, `optimizer.lr`, `model.hybrid_attn` (collapsed to a
//!    1-or-2 layer count), `data.train_path` and `data.val_path`, and still
//!    resolves `hidden: CONFIG_MODE_HIDDEN` (828) and
//!    `eval_every: CONFIG_MODE_EVAL_EVERY` (1000) - see `src/train_loop.rs`
//!    lines 2962 and 3240. What changed is the DISPOSITION of the other fields,
//!    not their fate: `unhonoured_fields` now grades each declaration against
//!    what this build would substitute and `config_train_args` REFUSES the run
//!    instead of substituting in silence. Asserting those values in a config
//!    file still proved only that a TOML file says what it says; asserting the
//!    refusal proves the declaration reaches a decision.
//! 2. **2.2393 is retracted.** `2446855` does not resolve to an object in this
//!    repository, no artifact exists behind the number (`checkpoint::save` was
//!    a stub returning `Ok(())` at every commit that could have produced it),
//!    and the seed it pinned - 43 - is forbidden by `src/seed_canon.rs` and by
//!    `TrainConfig::validate()`. See `RETRACTION.md`, section 2. The config now
//!    carries seed 47, which makes it loadable; it does not make 2.2393 real.
//! 3. **`+/- 0.01` was never a decidable tolerance.** The estimator's own
//!    between-grid sigma on `val_bpb` is `0.0358 bpb`, measured over seven
//!    window grids on one fixed set of weights
//!    ([docs/EVAL-UNCERTAINTY.md](../docs/EVAL-UNCERTAINTY.md), section 2). A
//!    `+/- 0.01` band is 3.5x TIGHTER than the noise of the instrument, so a
//!    conforming run and a non-conforming run are not distinguishable and the
//!    assertion could only ever have reported which grid it happened to draw.
//!
//! ## The tolerance a reproduction test may use
//!
//! Two bands, and which one applies is decided by the sampling plan, not by
//! preference (`docs/EVAL-UNCERTAINTY.md`, section 4a):
//!
//! * **UNPAIRED - `+/- 0.04 bpb` (one sigma).** The default. It applies the
//!   moment the two laboratories do not read the identical windows: a different
//!   `eval_chunks`, a different stride, a different prefix of the val corpus, or
//!   simply no record of which windows were read. It does NOT cover a different
//!   corpus, seed or step budget.
//! * **PAIRED - `+/- 0.003 bpb`, CONDITIONAL on an identical sampling plan.**
//!   Only when both arms evaluate the same windows of the same val corpus with
//!   the same `eval_chunks`, `eval_seq` and `eval_every`, so the sampling term
//!   is common-mode and cancels. The number comes from the cross-platform
//!   step-wise deltas (sample stdev `0.0031`, ~1/12 of sigma). Quoted without
//!   the pairing condition attached it is not a claim, it is a decoration.
//!
//! No test in this file asserts a BPB, so neither band is exercised here yet;
//! they are recorded so that the next reproduction test cannot be written
//! against a tolerance the instrument cannot resolve.
//!
//! What survives: the assertions on fields the trainer actually reads, the
//! `validate()` contract, and the pair of tests that pin the REFUSAL - one on
//! the library function that derives it, one on the exit status of the shipped
//! binary - so the gap cannot close or widen without someone noticing.
//!
//! ## Why these are no longer source-text assertions
//!
//! The two tests at the bottom of this file used to grep `src/train_loop.rs`
//! for the literals `hidden: 828` and `eval_every: 1000`. That assertion broke
//! the moment those literals became the named constants `CONFIG_MODE_HIDDEN`
//! and `CONFIG_MODE_EVAL_EVERY` - a pure rename, no behaviour moved - and its
//! panic message then claimed the gap had CLOSED, which was false. A test that
//! reports a rename as a behaviour change is worse than no test: it trains its
//! readers to disbelieve it. The subject here is what a declaration DOES, so it
//! is measured on `unhonoured_fields` and on the process exit status.

use std::process::{Command, Output};
use trios_trainer::train_loop::{
    self, CONFIG_EVAL_EVERY_ENV, CONFIG_MODE_EVAL_EVERY, CONFIG_MODE_HIDDEN,
    DEFAULT_IGLA_TARGET_BPB,
};
use trios_trainer::TrainConfig;

/// Between-grid sigma of `val_bpb`, MEASURED: seven window grids over one fixed
/// set of weights, `docs/EVAL-UNCERTAINTY.md` section 2. A correlated (nested)
/// sample, so this is a lower bound on the true dispersion.
const VAL_BPB_SIGMA: f64 = 0.0358;

/// The band a reproduction claim may use when the two runs did NOT read the
/// identical windows. One sigma, rounded up: `2.63 +/- 0.04 bpb`.
const REPRO_TOLERANCE_UNPAIRED: f64 = 0.04;

/// The band a reproduction claim may use ONLY when both runs share the sampling
/// plan (same corpus, `eval_chunks`, `eval_seq`, `eval_every`), so the sampling
/// term cancels. Derived from the cross-platform step-wise deltas (stdev
/// 0.0031). Never quote it without the pairing condition.
const REPRO_TOLERANCE_PAIRED: f64 = 0.003;

fn crate_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn champion_cfg() -> TrainConfig {
    let path = crate_root().join("configs/champion.toml");
    TrainConfig::from_toml(path.to_str().expect("config path is UTF-8"))
        .expect("champion.toml must load and validate")
}

/// The config must load and pass `validate()` (INV-8 phi-band on lr).
#[test]
fn champion_config_loads_and_validates() {
    let cfg = champion_cfg();
    assert_eq!(cfg.name, "champion");
    assert_eq!(cfg.seed, 47);
    assert_eq!(cfg.steps, 27_000);
    assert!(
        (cfg.optimizer.lr - 0.004).abs() < 1e-9,
        "INV-8: lr must be 0.004"
    );
    assert!((cfg.target_bpb - 1.50).abs() < 1e-9);
}

/// Only the fields that reach the trainer or the ledger are asserted here.
///
/// `hybrid_attn` is the single `[model]` field that reaches the trainer, and it
/// reaches it only as `if hybrid_attn { 2 } else { 1 }` attention layers.
#[test]
fn champion_consumed_fields_are_what_the_trainer_reads() {
    let cfg = champion_cfg();
    assert_eq!(cfg.seed, 47, "seed reaches TrainArgs.seed");
    assert_eq!(cfg.steps, 27_000, "steps reaches TrainArgs.steps");
    assert!(
        (cfg.optimizer.lr - 0.004).abs() < 1e-9,
        "optimizer.lr reaches TrainArgs.lr"
    );
    assert!(
        !cfg.model.hybrid_attn,
        "hybrid_attn=false is the only [model] field that reaches the trainer; \
         it selects attn_layers=1"
    );
    assert!(
        !cfg.data.train_path.is_empty() && !cfg.data.val_path.is_empty(),
        "data.train_path / data.val_path reach TrainArgs and must be set"
    );
}

/// The champion target must sit below the IGLA gate.
#[test]
fn target_bpb_below_igla_gate() {
    let cfg = champion_cfg();
    assert!(
        cfg.target_bpb < DEFAULT_IGLA_TARGET_BPB,
        "target_bpb ({}) must be below IGLA gate ({})",
        cfg.target_bpb,
        DEFAULT_IGLA_TARGET_BPB
    );
}

#[test]
fn champion_inv8_lr_in_phi_band() {
    let cfg = champion_cfg();
    let lr = cfg.optimizer.lr;
    assert!(
        (0.001..=0.01).contains(&lr),
        "INV-8: lr={lr} not in phi-band [0.001, 0.01]"
    );
}

/// Canon #93 on the shipped config, stated as the canon rather than as a
/// literal. `champion_config_loads_and_validates` pins the exact seed; this pins
/// the rule, so swapping 47 for another forbidden seed fails here too.
#[test]
fn champion_seed_obeys_canon_93() {
    let cfg = champion_cfg();
    const FORBIDDEN: &[u64] = &[42, 43, 44, 45];
    const ALLOWED: &[u64] = &[47, 89, 123, 144];
    assert!(
        !FORBIDDEN.contains(&cfg.seed),
        "seed {} is forbidden under Canon #93",
        cfg.seed
    );
    assert!(
        ALLOWED.contains(&cfg.seed),
        "seed {} is not in the Canon #93 allowed set {ALLOWED:?}",
        cfg.seed
    );
}

/// A guard on the CONSTANTS above, not a measurement.
///
/// It exists because `+/- 0.01` was in this file's header for months while the
/// instrument's own dispersion was 3.5x larger, and nothing failed. Any future
/// reproduction test must take its band from these constants, and this test
/// fails if someone tightens the unpaired band back below the measured sigma.
/// The measurement itself lives in `docs/EVAL-UNCERTAINTY.md`; only the
/// relations are checked here.
#[test]
fn reproduction_tolerances_are_not_tighter_than_the_instrument() {
    assert!(
        REPRO_TOLERANCE_UNPAIRED >= VAL_BPB_SIGMA,
        "the unpaired band ({REPRO_TOLERANCE_UNPAIRED}) must not be tighter than \
         the measured between-grid sigma ({VAL_BPB_SIGMA}): below it, a passing \
         run and a failing run are the same run read on two different grids"
    );
    assert!(
        REPRO_TOLERANCE_PAIRED < VAL_BPB_SIGMA,
        "the paired band ({REPRO_TOLERANCE_PAIRED}) is only defensible because \
         the sampling term cancels; if it stops being tighter than sigma it has \
         lost its meaning"
    );
    // The retired band, kept so the reason it was retired stays checkable.
    const RETIRED_BAND: f64 = 0.01;
    assert!(
        RETIRED_BAND < VAL_BPB_SIGMA,
        "the retired +/- 0.01 band must remain on record as tighter than sigma"
    );
}

// ---------------------------------------------------------------------------
// The gap tests - what `configs/champion.toml` declares that this build refuses
// ---------------------------------------------------------------------------

/// Every field `configs/champion.toml` declares that this build cannot execute,
/// spelled as `UnhonouredField::field` spells it.
///
/// Derived by running the shipped binary, not guessed:
/// `./target/release/trios-train --config configs/champion.toml` exits 1 and
/// prints `config "champion" declares 10 parameter(s) this build cannot
/// execute`, one bullet per entry below.
///
/// This list moving in either direction is a real event and must be reflected in
/// the header of `configs/champion.toml`:
///
/// * An entry **disappearing** means the build learned to honour that
///   declaration, or the config was edited down to what the build runs.
/// * An entry **appearing** means a declaration that used to be executable no
///   longer is.
const CHAMPION_UNHONOURED: &[&str] = &[
    "model.d_model",
    "model.n_layers",
    "model.vocab_size",
    "model.seq_len",
    "optimizer.beta2",
    "optimizer.schedule",
    "optimizer.warmup_steps",
    "data.batch_size",
    "objective.w_nca",
    // Not a TOML key: `TrainConfig` has no field for the eval cadence, so it is
    // declared out-of-band via `CONFIG_EVAL_EVERY_ENV` and graded under this
    // name. Left unset, as it is here, it is refused rather than defaulted.
    "eval cadence",
];

/// Fields `champion.toml` carries that `unhonoured_fields` deliberately does
/// NOT grade, per its own doc comment: `data.corpus` is a label for
/// `train_path`/`val_path` (which ARE honoured and hashed into every artifact),
/// and `data.batch_tokens` has no counterpart to compare against. They are the
/// remaining declared-and-ignored surface and are recorded here so that surface
/// stays exactly two fields wide.
const CHAMPION_UNGRADED: &[&str] = &["data.corpus", "data.batch_tokens"];

/// The library-side gap test: the exact set of refused fields.
///
/// `unhonoured_fields` is pure - it reads no environment - so this asserts a set
/// equality rather than a substring match, and `eval_every: None` reproduces the
/// shipped situation where the cadence is not declared anywhere.
#[test]
fn champion_toml_declares_exactly_these_unhonourable_fields() {
    let cfg = champion_cfg();
    let gaps = train_loop::unhonoured_fields(&cfg, None);

    let got: Vec<&str> = gaps.iter().map(|g| g.field.as_str()).collect();
    assert_eq!(
        got, CHAMPION_UNHONOURED,
        "the set of declarations configs/champion.toml makes that this build \
         cannot execute has changed. This is NOT a cosmetic event: the header of \
         configs/champion.toml states this list, and `--config \
         configs/champion.toml` refuses to run once per entry. Update both \
         together."
    );

    // The two substitutes that used to be silent literals inside `run()`.
    // Asserted through the constants, so renaming them cannot fake a change and
    // changing their VALUE cannot hide behind a rename.
    let d_model = gaps
        .iter()
        .find(|g| g.field == "model.d_model")
        .expect("model.d_model must be refused: champion.toml declares 256");
    assert_eq!(d_model.declared, cfg.model.d_model.to_string());
    assert!(
        d_model.substitute.contains(&CONFIG_MODE_HIDDEN.to_string()),
        "the refusal must name the width it would otherwise have substituted \
         (CONFIG_MODE_HIDDEN = {CONFIG_MODE_HIDDEN}); got {:?}",
        d_model.substitute
    );

    let cadence = gaps
        .iter()
        .find(|g| g.field == "eval cadence")
        .expect("an undeclared eval cadence must be refused, not defaulted");
    assert!(
        cadence.declared.contains(CONFIG_EVAL_EVERY_ENV),
        "the refusal must name the env var the cadence has to be declared in; \
         got {:?}",
        cadence.declared
    );
    assert!(
        cadence
            .substitute
            .contains(&CONFIG_MODE_EVAL_EVERY.to_string()),
        "the refusal must name the cadence it would otherwise have substituted \
         (CONFIG_MODE_EVAL_EVERY = {CONFIG_MODE_EVAL_EVERY}); got {:?}",
        cadence.substitute
    );

    // The ungraded surface stays exactly as wide as it is documented to be.
    for ungraded in CHAMPION_UNGRADED {
        assert!(
            !got.contains(ungraded),
            "{ungraded} is now graded by unhonoured_fields. That is an \
             improvement, not a failure - but the header of \
             configs/champion.toml lists it as parsed-and-ignored and must be \
             corrected, and CHAMPION_UNGRADED must shrink."
        );
    }
}

/// The same gap, observed where it actually costs something: the exit status.
///
/// A unit test on `unhonoured_fields` cannot see whether the binary acts on the
/// refusal. This one spawns `trios-train --config configs/champion.toml` and
/// requires a non-zero exit with every refused field named, because the defect
/// being guarded against is an artifact minted under a declaration it does not
/// match - and that artifact is minted by a PROCESS, not by a function.
#[test]
fn champion_toml_run_exits_non_zero_and_names_every_refused_field() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_trios-train"));
    cmd.current_dir(env!("CARGO_MANIFEST_DIR"));
    cmd.args(["--config", "configs/champion.toml"]);
    // A test whose subject is "what did this declaration ask for" cannot inherit
    // an ambient declaration. `TRIOS_EVAL_EVERY` in particular would satisfy one
    // of the ten refusals from outside the file.
    for var in [
        "SEED",
        "TRIOS_SEED",
        "TRIOS_STEPS",
        "TRIOS_LR",
        "TRIOS_TRAIN_PATH",
        "TRIOS_VAL_PATH",
        "TRIOS_EVAL_EVERY",
        "TRIOS_CANON_NAME",
        "CANON_NAME",
        "DATABASE_URL",
        "NEON_DATABASE_URL",
        "TRIOS_NEON_DSN",
        "TRIOS_DATABASE_URL",
    ] {
        cmd.env_remove(var);
    }
    cmd.env("TRIOS_CHECKPOINT_DISABLE", "1")
        .env("TRINITY_AUTOMIGRATE", "0");

    let out: Output = cmd.output().expect("spawn trios-train");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let describe = || {
        format!(
            "status={:?}\n--- stderr ---\n{}",
            out.status,
            stderr.lines().rev().take(30).collect::<Vec<_>>().join("\n")
        )
    };

    assert!(
        !out.status.success(),
        "`--config configs/champion.toml` must NOT exit 0. The config declares \
         {} parameter(s) this build cannot execute; a zero exit would mint an \
         artifact filed under a declaration the run did not honour.\n{}",
        CHAMPION_UNHONOURED.len(),
        describe()
    );
    assert!(
        stderr.contains("[declaration-truth]"),
        "the refusal must be attributable to the declaration-truth check rather \
         than to some unrelated startup failure.\n{}",
        describe()
    );
    for field in CHAMPION_UNHONOURED {
        assert!(
            stderr.contains(field),
            "the refusal must name `{field}` - all of them, not the first one, \
             so one run tells the whole truth about the gap.\n{}",
            describe()
        );
    }
}

// The former `champion_bpb_reproduction_full_run` test is deleted, not
// `#[ignore]`d. It asserted `outcome.final_bpb` inside [2.229, 2.249] "per
// gHashTag/trios@2446855" - a commit absent from this repository, a number with
// no artifact behind it, a band (+/- 0.01) 3.5x tighter than the estimator's own
// sigma of 0.0358, and a tolerance around a value this config could not have
// produced anyway - `--config configs/champion.toml` exits 1 rather than
// running, because the config declares `d_model = 256` and this build can only
// run `CONFIG_MODE_HIDDEN` (828), plus nine other declarations it cannot
// execute. Reinstating a reproduction test requires a
// checkpoint-backed reference with a corpus digest, a recorded eval cadence, a
// recorded sampling plan, and a band taken from `REPRO_TOLERANCE_UNPAIRED`
// unless that plan is identical on both arms; RETRACTION.md section 2 names the
// one figure in this tree that qualifies.
