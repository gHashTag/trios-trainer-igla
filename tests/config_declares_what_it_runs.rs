//! A declaration that is not executed must not produce a run.
//!
//! # The two defects these tests exist to keep closed
//!
//! **1. `--sweep` had no optimizer.** `train_loop::run_sweep` took no optimizer
//! argument, so the `--sweep` arm of `trios-train` dispatched nothing: measured
//! on 2026-08-03, `--sweep --optimizer soap --steps 1` printed three `DONE:`
//! lines from AdamW, printed the headline `GATE-2:` verdict and exited 0. The
//! single-seed arm refused the same name. `ensure_supported_optimizer`,
//! `run_with_optimizer` and `format_done_line` already existed in the library
//! with ZERO call sites, and `format_done_line`'s own doc comment names this
//! defect as the thing it fixes.
//!
//! **2. `--config` ignored the config.** `train_loop::run(cfg)` built its
//! `TrainArgs` with `hidden: 828` and `eval_every: 1000` written into the
//! literal and derived `attn_layers` from `hybrid_attn` alone, discarding
//! `d_model`, `n_layers`, `n_heads`, `seq_len`, `vocab_size` and
//! `optimizer.kind`. `--config configs/gate2-attempt.toml` - a file declaring
//! 384/4/muon+adamw - therefore minted a checkpoint sidecar recording
//! hidden=828 and optimizer=adamw under the run name "gate2-attempt".
//!
//! Both halves matter to the same claim: the product being sold here is a
//! declared-parameter envelope, and these were the two entry points that take a
//! declaration and ignore it.
//!
//! The subprocess cases spawn the built binary, as `tests/
//! ledger_exit_code_binaries.rs` does, because the observable under test is the
//! EXIT STATUS: a unit test on the refusal function cannot see whether the
//! binary honours it.

use std::process::{Command, Output};
use trios_trainer::config::{
    DataConfig, LedgerConfig, ModelConfig, ObjectiveConfig, OptimizerConfig,
};
use trios_trainer::train_loop::{self, SUPPORTED_OPTIMIZERS};
use trios_trainer::TrainConfig;

/// The env vars `neon_writer::resolve_dsn` consults. Cleared for every child so
/// a DSN in the developer's or CI's environment cannot decide the result.
const DSN_VARS: [&str; 4] = [
    "DATABASE_URL",
    "NEON_DATABASE_URL",
    "TRIOS_NEON_DSN",
    "TRIOS_DATABASE_URL",
];

/// Every env var that could reach `Cli` or `TrainConfig::apply_env_overrides`
/// and change what the child ran. A test whose subject is "what did the
/// declaration ask for" cannot inherit an ambient declaration.
const OVERRIDE_VARS: [&str; 9] = [
    "SEED",
    "TRIOS_SEED",
    "TRIOS_STEPS",
    "TRIOS_LR",
    "TRIOS_TRAIN_PATH",
    "TRIOS_VAL_PATH",
    "TRIOS_EVAL_EVERY",
    "TRIOS_CANON_NAME",
    "CANON_NAME",
];

fn trainer_command() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_trios-train"));
    // Corpus paths are repo-relative defaults; `cargo test` does not guarantee
    // the cwd of the spawned child.
    cmd.current_dir(env!("CARGO_MANIFEST_DIR"));
    for var in DSN_VARS.iter().chain(OVERRIDE_VARS.iter()) {
        cmd.env_remove(var);
    }
    cmd.env("TRIOS_CHECKPOINT_DISABLE", "1")
        // The migrator is a separate connection path, not what is under test,
        // and its retry budget would dominate the runtime.
        .env("TRINITY_AUTOMIGRATE", "0");
    cmd
}

/// The smallest run that still reaches an evaluation: 1 step, 16 wide, 1
/// attention layer, and the 8 windows `evaluate` refuses to go below.
fn tiny_run_args(cmd: &mut Command) {
    cmd.args([
        "--steps",
        "1",
        "--hidden",
        "16",
        "--attn-layers",
        "1",
        "--eval-every",
        "1",
        "--eval-chunks",
        "8",
    ]);
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
            .take(25)
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

// ---- (a) the sweep arm refuses an optimizer it cannot run -------------------

#[test]
fn sweep_with_an_unsupported_optimizer_exits_non_zero() {
    let mut cmd = trainer_command();
    cmd.args(["--sweep", "--optimizer", "soap"]);
    tiny_run_args(&mut cmd);
    let out = run(cmd);

    assert!(
        !out.status.success(),
        "`--sweep --optimizer soap` must not exit 0: nothing in this crate \
         implements soap, so a zero exit certifies a run that did not happen.\n{}",
        describe(&out)
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("soap"),
        "the refusal must name the optimizer that was refused.\n{}",
        describe(&out)
    );
    for supported in SUPPORTED_OPTIMIZERS {
        assert!(
            stderr.contains(supported),
            "the refusal must list {supported:?} as an optimizer that IS \
             implemented, so the operator can fix the call.\n{}",
            describe(&out)
        );
    }

    // The half that actually regressed: the sweep TRAINED and reported.
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("DONE:"),
        "a refused sweep must not train a single seed.\n{}",
        describe(&out)
    );
    assert!(
        !stdout.contains("GATE-2:"),
        "a refused sweep must not print a gate verdict.\n{}",
        describe(&out)
    );
}

/// The control: the same sweep with an optimizer that IS implemented runs, and
/// its `DONE:` lines now name it. Without this, the test above would pass on a
/// binary that refuses every sweep.
#[test]
fn sweep_with_a_supported_optimizer_runs_and_names_it() {
    let mut cmd = trainer_command();
    cmd.args(["--sweep", "--optimizer", "adamw"]);
    tiny_run_args(&mut cmd);
    let out = run(cmd);

    assert!(
        out.status.success(),
        "a sweep under a supported optimizer must still run.\n{}",
        describe(&out)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    for seed in train_loop::GATE_FINAL_SEEDS {
        assert!(
            stdout.contains(&format!("DONE: seed={seed} ")),
            "seed {seed} of the sweep must report.\n{}",
            describe(&out)
        );
    }
    assert_eq!(
        stdout.matches("opt=adamw").count(),
        train_loop::GATE_FINAL_SEEDS.len(),
        "every sweep DONE line must name the optimizer that produced it; the \
         sweep arm used to print no `opt=` token at all.\n{}",
        describe(&out)
    );
}

// ---- (b) config mode either honours the declaration or refuses --------------

/// `--config configs/gate2-attempt.toml` must not mint a run.
///
/// The accepted outcomes are the two the fix allows: a non-zero exit whose
/// message names the declaration, or a run that actually executes it. The
/// second is asserted rather than assumed - if the binary ever starts honouring
/// `d_model`, this test wants to see `hidden=384` in its banner, not the 828 it
/// used to substitute.
#[test]
fn config_mode_does_not_substitute_its_own_geometry() {
    let mut cmd = trainer_command();
    cmd.args(["--config", "configs/gate2-attempt.toml"]);
    let out = run(cmd);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);

    if out.status.success() {
        assert!(
            stdout.contains("hidden=384") || stderr.contains("hidden=384"),
            "the run exited 0, so it must have executed the declared d_model=384 \
             rather than substituting 828.\n{}",
            describe(&out)
        );
        return;
    }

    assert!(
        stderr.contains("d_model") && stderr.contains("828"),
        "the refusal must name the declared field and the value that would have \
         been substituted for it.\n{}",
        describe(&out)
    );
    assert!(
        !stdout.contains("DONE:"),
        "a refused config must not train.\n{}",
        describe(&out)
    );
}

/// The same contract at library level, where the message can be read whole.
#[test]
fn gate2_attempt_config_is_refused_field_by_field() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/configs/gate2-attempt.toml");
    let cfg = TrainConfig::from_toml(path).expect("gate2-attempt.toml must still load");
    assert_eq!(cfg.model.d_model, 384, "the declaration under test");
    assert_eq!(
        cfg.optimizer.kind, "muon+adamw",
        "the declaration under test"
    );

    // A cadence IS declared here, so the refusal below is about the model and
    // the optimizer and not about the missing cadence.
    let err = train_loop::config_train_args(&cfg, Some(1000))
        .expect_err("a config this build cannot execute must not yield TrainArgs");
    let msg = format!("{err:#}");

    for expected in [
        "d_model",
        "828",
        "n_layers",
        "muon+adamw",
        "vocab_size",
        "seq_len",
        "n_heads",
    ] {
        assert!(
            msg.contains(expected),
            "the refusal must name {expected:?}; it said:\n{msg}"
        );
    }
}

/// The cadence is refused on its own, because it is not in the TOML schema at
/// all and used to be defaulted to 1000 in silence. It is an observation
/// parameter of the published measurement (which steps are measured, hence the
/// EMA, `min_observed_val_bpb`, every `bpb_samples` row and the sidecar's
/// `eval_every`), and it gated the in-place `gf16_floor` weight rewrite until
/// that coupling was broken.
#[test]
fn an_undeclared_eval_cadence_is_itself_a_refusal() {
    let cfg = executable_config();
    train_loop::config_train_args(&cfg, Some(50)).expect("a declared cadence is executable");

    let err = train_loop::config_train_args(&cfg, None)
        .expect_err("an undeclared cadence must not be defaulted");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("eval")
            && msg.contains("1000")
            && msg.contains(train_loop::CONFIG_EVAL_EVERY_ENV),
        "the refusal must name the cadence, the value that would have been \
         substituted, and where to declare it; it said:\n{msg}"
    );
}

/// The refusal is not a blanket one: a config that declares exactly what this
/// crate executes is accepted, and every accepted field reaches `TrainArgs`.
#[test]
fn a_config_that_declares_what_this_crate_executes_is_honoured() {
    let cfg = executable_config();
    let args = train_loop::config_train_args(&cfg, Some(50))
        .expect("an executable declaration must produce TrainArgs");

    assert_eq!(args.hidden, cfg.model.d_model, "declared width is executed");
    assert_eq!(args.seed, cfg.seed);
    assert_eq!(args.steps, cfg.steps);
    assert_eq!(
        args.eval_every, 50,
        "the declared cadence, not the old 1000"
    );
    assert_eq!(args.attn_layers, 1, "hybrid_attn=false selects one layer");
    assert!((args.lr - cfg.optimizer.lr as f32).abs() < 1e-9);
    assert_eq!(args.train_path, cfg.data.train_path);
    assert_eq!(args.val_path, cfg.data.val_path);
}

/// Each field is refused on its own merit: change exactly one value away from
/// what the crate executes and exactly that field must be named.
#[test]
fn each_declared_field_is_graded_separately() {
    let mutate: Vec<(&str, fn(&mut TrainConfig))> = vec![
        ("model.d_model", |c| c.model.d_model = 384),
        ("model.n_layers", |c| c.model.n_layers = 4),
        ("model.n_heads", |c| c.model.n_heads = 6),
        ("model.vocab_size", |c| c.model.vocab_size = 32_000),
        ("model.seq_len", |c| c.model.seq_len = 1024),
        ("optimizer.kind", |c| c.optimizer.kind = "soap".into()),
        ("optimizer.beta2", |c| c.optimizer.beta2 = 0.95),
        ("optimizer.weight_decay", |c| c.optimizer.weight_decay = 0.1),
        ("optimizer.schedule", |c| {
            c.optimizer.schedule = "phi".into()
        }),
        ("optimizer.warmup_steps", |c| {
            c.optimizer.warmup_steps = 1_000
        }),
        ("data.batch_size", |c| c.data.batch_size = 8),
        ("objective.w_jepa", |c| c.objective.w_jepa = 0.5),
        ("objective.w_nca", |c| c.objective.w_nca = 0.1),
    ];
    for (field, apply) in mutate {
        let mut cfg = executable_config();
        apply(&mut cfg);
        let gaps = train_loop::unhonoured_fields(&cfg, Some(50));
        assert_eq!(
            gaps.len(),
            1,
            "changing {field} alone must produce exactly one refusal, got {gaps:?}"
        );
        assert_eq!(gaps[0].field, field, "the refusal must name {field}");
    }
}

// ---- (c) run_sweep cannot be called without an optimizer --------------------

/// The compile-level half of the guarantee is the signature: `run_sweep` takes
/// `optimizer: &str` and there is no default, so no caller can omit it - the
/// call in this test would not compile without it. What is asserted here is the
/// behaviour that makes the parameter worth having: an unsupported name is
/// refused BEFORE any seed trains.
///
/// `steps` is deliberately a number no test could afford to train. If the
/// refusal ever moves after the loop, this test stops being fast and starts
/// being a timeout, which is a louder failure than a wrong assertion.
#[test]
fn run_sweep_refuses_an_unsupported_optimizer_before_training() {
    let started = std::time::Instant::now();
    let err = train_loop::run_sweep(
        100_000_000,
        16,
        0.003,
        1,
        1,
        "data/tiny_shakespeare.txt",
        "data/tiny_shakespeare_val.txt",
        "soap",
    )
    .expect_err("an unsupported optimizer must not start a sweep");
    let msg = format!("{err:#}");
    assert!(msg.contains("soap"), "the refusal must name soap: {msg}");
    for supported in SUPPORTED_OPTIMIZERS {
        assert!(
            msg.contains(supported),
            "the refusal must list {supported:?}: {msg}"
        );
    }
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "the refusal must land before the first seed trains, not after"
    );
}

/// Every name in `SUPPORTED_OPTIMIZERS` is accepted by the one refusal point,
/// and a name outside it is not. This is what stops the whitelist and the
/// dispatch from drifting apart.
#[test]
fn the_refusal_point_agrees_with_the_supported_list() {
    for supported in SUPPORTED_OPTIMIZERS {
        train_loop::ensure_supported_optimizer(supported)
            .unwrap_or_else(|e| panic!("{supported:?} is listed as supported: {e}"));
    }
    for unsupported in ["soap", "lion", "tiger", "adamw ", "ADAMW", "muon+adamw", ""] {
        assert!(
            train_loop::ensure_supported_optimizer(unsupported).is_err(),
            "{unsupported:?} is not implemented and must be refused"
        );
    }
}

// ---- the fixture ------------------------------------------------------------

/// A config declaring exactly what `train_loop` executes today.
///
/// Every value here is the crate's own: width 828 (`CONFIG_MODE_HIDDEN`), one
/// attention layer, `HybridAttnConfig::default()` heads and seq_len, the
/// byte-level vocabulary, AdamW's betas and weight decay, the cosine schedule
/// with `steps/10` warmup, the `accum x positions` batch, and
/// `NcaObjective::default().weight`. It exists so the refusal above is provably
/// a comparison and not a blanket "config mode is dead".
fn executable_config() -> TrainConfig {
    TrainConfig {
        name: "declaration-truth-fixture".into(),
        steps: 4_000,
        seed: 47,
        target_bpb: 2.5,
        champion_bpb: None,
        model: ModelConfig {
            d_model: train_loop::CONFIG_MODE_HIDDEN,
            n_layers: 1,
            n_heads: 4,
            vocab_size: 128,
            seq_len: 8,
            hybrid_attn: false,
        },
        optimizer: OptimizerConfig {
            kind: "adamw".into(),
            lr: 0.004,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: train_loop::TRAIN_LOOP_WEIGHT_DECAY,
            schedule: train_loop::TRAIN_LOOP_SCHEDULE.into(),
            warmup_steps: 400,
        },
        data: DataConfig {
            corpus: "tinyshakespeare".into(),
            train_path: "data/tiny_shakespeare.txt".into(),
            val_path: "data/tiny_shakespeare_val.txt".into(),
            batch_size: train_loop::TRAIN_LOOP_BATCH_SIZE,
            batch_tokens: 0,
        },
        objective: ObjectiveConfig {
            w_ce: 1.0,
            w_jepa: 0.0,
            w_nca: 0.25,
        },
        ledger: LedgerConfig {
            jsonl_path: String::new(),
            push: false,
            embargo_path: ".embargo".into(),
        },
    }
}
