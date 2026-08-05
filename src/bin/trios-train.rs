//! `trios-train` - CLI entry point.
//!
//! ```bash
//! # Standalone mode (no config file needed)
//! trios-train --seed 47 --steps 54000   # Canon #93 allowed seed
//!
//! # Config mode
//! trios-train --config configs/champion.toml
//!
//! # 3-seed sweep
//! trios-train --sweep --steps 54000
//! ```

use anyhow::Result;
use clap::Parser;
use migration::MigratorTrait;
use trios_trainer::neon_writer::strip_channel_binding;
use trios_trainer::seed_canon::parse_seed;
use trios_trainer::train_loop::{self, TrainArgs, GATE_FINAL_SEEDS};

#[derive(Parser, Debug)]
#[command(
    name = "trios-train",
    about = "IGLA RACE training pipeline (gHashTag/trios#143)"
)]
struct Cli {
    /// Path to TOML config (optional; standalone mode if omitted).
    #[arg(long, env = "TRIOS_CONFIG")]
    config: Option<std::path::PathBuf>,

    /// Override seed. Use 0 to run 3-seed sweep.
    /// If the `SEED` env var is set, Canon #93 enforcement applies:
    /// seeds {42, 43, 44, 45} are forbidden; use {47, 89, 123, 144}.
    /// Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877
    #[arg(long, env = "TRIOS_SEED", default_value_t = 47)]
    seed: u64,

    /// Number of training steps.
    #[arg(long, env = "TRIOS_STEPS", default_value_t = 54000)]
    steps: usize,

    /// Hidden dimension (phi-scaled: 828).
    #[arg(long, env = "TRIOS_HIDDEN", default_value_t = 828)]
    hidden: usize,

    /// Learning rate.
    #[arg(long, env = "TRIOS_LR", default_value_t = 0.003)]
    lr: f32,

    /// Number of attention layers.
    #[arg(long, env = "TRIOS_ATTN_LAYERS", default_value_t = 2)]
    attn_layers: u8,

    /// Evaluate every N steps.
    #[arg(long, env = "TRIOS_EVAL_EVERY", default_value_t = 1000)]
    eval_every: usize,

    /// How many val windows each evaluation averages; 0 = full coverage.
    ///
    /// The same knob `TRIOS_EVAL_CHUNKS` reads (see
    /// `train_loop::eval_chunks_target`). The env var stays supported and the
    /// flag wins when both are given: this flag is re-exported into the env
    /// below, before any evaluation runs. It is deliberately NOT declared with
    /// clap's `env =` - `eval_chunks_target()` resolves an unparseable value to
    /// the default rather than aborting, and clap would turn that into a hard
    /// parse error.
    ///
    /// Existed only as an env var until now, so a second laboratory following
    /// the documented command line had no way to learn the published BPB was a
    /// 5% sample. Whichever route sets it, the achieved value is written to the
    /// checkpoint record's `eval_chunks` field from the `EvalStats` that
    /// produced the reading.
    #[arg(long)]
    eval_chunks: Option<usize>,

    /// Path to training data.
    #[arg(
        long,
        env = "TRIOS_TRAIN_PATH",
        default_value = "data/tiny_shakespeare.txt"
    )]
    train_data: String,

    /// Path to validation data.
    #[arg(
        long,
        env = "TRIOS_VAL_PATH",
        default_value = "data/tiny_shakespeare_val.txt"
    )]
    val_data: String,

    /// Run 3-seed sweep {47, 89, 123} (Canon #93) instead of single seed.
    #[arg(long)]
    sweep: bool,

    /// Optimizer: adamw, muon, or muon-cwd (P1 lab).
    #[arg(long, env = "TRIOS_OPTIMIZER", default_value = "adamw")]
    optimizer: String,

    /// Context window (accepted for seed-agent compat; ignored — fixed by
    /// `train_loop::NUM_CTX`). seed-agent (gHashTag/trios-railway) passes
    /// `--ctx 12` because the legacy bisect found ctx=12 was the only working
    /// value. We accept the flag here so unknown-arg parse errors don't crash
    /// the trainer immediately. Refs: trios-railway#62, trios-trainer-igla#55.
    #[arg(long, env = "TRIOS_CTX")]
    #[allow(dead_code)]
    ctx: Option<usize>,

    /// Format type pass-through.
    ///
    /// Honoured by `train_loop::resolve_fake_quant_format()` via the
    /// `TRIOS_FORMAT_TYPE` env var. Historically this flag was accepted but
    /// silently dropped because clap stored it in `cli.format` and `main()`
    /// never re-exported it; the result was a production-wide fp32-fallback
    /// (trios#509: 52 ≡ 2.942101 / 49 ≡ 2.998885 collapse, scarab triplets
    /// adamw-binary32 / adamw-GF16 / muon-GF16 producing identical BPB on the
    /// same seed). The fix below re-exports `cli.format` into the env so the
    /// `--format=gf16` CLI form behaves identically to `TRIOS_FORMAT_TYPE=gf16`.
    #[arg(long, env = "TRIOS_FORMAT_TYPE")]
    format: Option<String>,

    /// Neon database URL for bpb_samples writes (used by scarab worker).
    #[arg(long, env = "TRIOS_NEON_DSN")]
    #[allow(dead_code)]
    neon: Option<String>,
}

fn install_panic_hook() {
    // R5/L8: never let a panic vanish into the void. Print a one-line JSON
    // diagnostic to stderr so seed-agent (which streams stderr via
    // `Stdio::inherit()`) and Railway logs both capture it. Also write a
    // marker line to stdout so the parent reader sees a non-empty stream.
    std::panic::set_hook(Box::new(|info| {
        use std::io::Write as _;
        let loc = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_else(|| "unknown".to_string());
        let msg = info.to_string();
        eprintln!(
            r#"{{"event":"panic","loc":{:?},"msg":{:?},"step":-1}}"#,
            loc, msg
        );
        let _ = std::io::stderr().flush();
        // Stdout marker so seed-agent's parse_step_output()/parse_done_output()
        // log it as 'unrecognized' instead of producing zero JSONL silently.
        println!("PANIC: trios-train aborted at {loc} (msg: {msg})");
        let _ = std::io::stdout().flush();
    }));
}

/// Run SeaORM schema migrations at startup, and ONLY on explicit consent.
///
/// Gating: opt-IN, twice over. `TRINITY_AUTOMIGRATE` defaulted to "1"
/// (`.unwrap_or_else(|_| "1")`), so merely having a `DATABASE_URL` /
/// `NEON_DATABASE_URL` / `TRIOS_NEON_DSN` in the ambient environment was enough
/// for this trainer to run `Migrator::up` DDL against it - no host check, no
/// consent flag. A survey run on 2026-08-03 did in fact apply migrations and
/// insert rows into a database nobody intended to touch.
///
/// The repo's own `tests/ledger_seaorm.rs` gates the identical operation with
/// the words: "Having a DSN in the environment is not consent". Applying that
/// rule to the flagship binary is what `TRIOS_ALLOW_AUTOMIGRATE=1` is for.
/// Writing rows may stay default-on; applying DDL must not be.
///
/// The decision itself lives in `train_loop::decide_automigrate` so it is unit
/// testable without an ambient database. Exactly one `[migrator]` line names
/// the decision taken, in every branch, so a grep for `[migrator]` still tells
/// the whole story of what was and was not done.
fn run_automigrate() {
    let automigrate = std::env::var("TRINITY_AUTOMIGRATE").ok();
    let consent = std::env::var("TRIOS_ALLOW_AUTOMIGRATE").ok();
    let raw_dsn = std::env::var("DATABASE_URL")
        .or_else(|_| std::env::var("NEON_DATABASE_URL"))
        .or_else(|_| std::env::var("TRIOS_NEON_DSN"))
        .or_else(|_| std::env::var("TRIOS_DATABASE_URL"))
        .ok();

    let decision = train_loop::decide_automigrate(
        automigrate.as_deref(),
        consent.as_deref(),
        raw_dsn.as_deref(),
    );
    eprintln!("[migrator] {decision:?}: {}", decision.reason());
    if decision != train_loop::AutomigrateDecision::Apply {
        return;
    }

    // `Apply` is only returned with a non-empty DSN.
    let dsn = strip_channel_binding(raw_dsn.as_deref().unwrap_or_default());

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("migrator runtime");

    rt.block_on(async {
        match sea_orm::Database::connect(&dsn).await {
            Ok(db) => {
                match migration::Migrator::up(&db, None).await {
                    Ok(()) => eprintln!("[migrator] schema up-to-date"),
                    Err(e) => eprintln!("[migrator] migration failed (non-fatal): {e}"),
                }
                let _ = db.close().await;
            }
            Err(e) => eprintln!("[migrator] connect failed (non-fatal): {e}"),
        }
    });
}

/// Print the ledger tally and exit with the code the ledger demands.
///
/// `neon_writer::ledger_exit_code` exists to turn "a DSN was configured, N
/// writes were attempted and 0 landed" into a non-zero exit status. It was
/// already honoured by `smoke_train`, `bpb_smoke`, `ngram_train_gf16` and
/// `hybrid_train` -- but NOT by `trios-train`, the binary CI runs, the binary
/// README documents and the binary that produced the r6-headline run with a
/// DSN set. That run could have had every `bpb_sample` and `checkpoint_record`
/// rejected on stderr and still reported success to its supervisor.
///
/// With no DSN configured `ledger_exit_code()` returns 0 -- proved by
/// `ledger_exit_code_is_zero_without_a_dsn` in src/neon_writer.rs -- so the
/// ordinary exit status is unchanged. Every error path in `main` returns `Err`
/// before reaching here, so nothing is swallowed.
/// `rejected` is reported separately and counted into `attempted`. It used to
/// be counted nowhere: a run in which every BPB row was REFUSED as
/// unpublishable printed `attempted=0 landed=0 dropped=0`, telling its
/// supervisor that nothing had even been tried.
///
/// With no DSN the tally is NOT printed. It said
/// `LEDGER: attempted=3 landed=0 dropped=3 rejected=0` on a run where no DSN
/// had ever been configured: the counters are honest about the calls -- each
/// write function does note a drop when it finds no connection -- but the LINE
/// asserts that three writes were attempted and lost, and "dropped" means
/// "lost in transport" to everyone who reads it. A run that was never asked to
/// record itself must say so instead of reporting three failures it did not
/// have. The exit code on that path stays 0, as before.
fn exit_with_ledger_status() -> ! {
    let landed = trios_trainer::neon_writer::landed_writes();
    let dropped = trios_trainer::neon_writer::dropped_writes();
    let rejected = trios_trainer::neon_writer::rejected_writes();
    if trios_trainer::neon_writer::dsn_configured() {
        println!(
            "LEDGER: attempted={} landed={landed} dropped={dropped} rejected={rejected}",
            landed + dropped + rejected
        );
    } else {
        println!("LEDGER: no DSN configured; no rows attempted");
    }
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    std::process::exit(trios_trainer::neon_writer::ledger_exit_code());
}

fn main() -> Result<()> {
    install_panic_hook();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    eprintln!(
        "[trios-train] startup args={:?} cwd={:?}",
        std::env::args().collect::<Vec<_>>(),
        std::env::current_dir().ok()
    );
    use std::io::Write as _;
    let _ = std::io::stderr().flush();

    // trios#777 fix: promote un-prefixed env aliases set by wave-a-dispatch.yml
    // (`OPTIMIZER`, `FORMAT`, `HIDDEN`) into the canonical `TRIOS_*` names BEFORE
    // clap parses, so the CLI struct picks them up via #[arg(env = "TRIOS_*")].
    //
    // Without this promotion, 47 distinct canon_name configs (ranger/adafactor/
    // adamw/adopt/demo on binary16) all collapsed to bit-identical bpb=2.9504
    // because every trainer silently fell back to the clap defaults
    // (adamw / None→f32 / hidden=828). R5-evidence: ssot.bpb_samples shows 94
    // rows across 47 canons with bpb=2.9504425525665283 at step 80000.
    //
    // Resolution: TRIOS_* (canonical) > alias (un-prefixed) > clap default.
    // The alias only fires when the canonical env is unset, so existing
    // TRIOS_OPTIMIZER/TRIOS_FORMAT_TYPE/TRIOS_HIDDEN users are unaffected.
    //
    // Anchor: phi^2 + phi^-2 = 3 · DOI 10.5281/zenodo.19227877.
    for (canonical, alias) in [
        ("TRIOS_OPTIMIZER", "OPTIMIZER"),
        ("TRIOS_FORMAT_TYPE", "FORMAT"),
        ("TRIOS_HIDDEN", "HIDDEN"),
    ] {
        if std::env::var(canonical).is_err() {
            if let Ok(v) = std::env::var(alias) {
                if !v.is_empty() {
                    eprintln!("[trios-train][trios#777] promoting {alias}={v} -> {canonical}");
                    std::env::set_var(canonical, v);
                }
            }
        }
    }

    let mut cli = Cli::parse();

    // Canon #93 enforcement.
    //   * If `SEED` env var is set → validate via `parse_seed()` AND assign
    //     the validated value to `cli.seed`. SEED env therefore overrides
    //     `--seed` flag and `TRIOS_SEED`/clap default.
    //   * Else → directly validate `cli.seed` against the forbidden set
    //     (which is what `parse_seed()` does internally on its raw input).
    //     This catches the case where TRIOS_SEED or `--seed=43` slipped
    //     past clap.
    // Forbidden canon: {42, 43, 44, 45}; allowed canon: {47, 89, 123, 144}.
    // Wave-29 PR-A.1: previously `parse_seed()`'s return value was logged
    // and dropped — `cli.seed` could still be 43 if `--seed=43` was passed
    // alongside an unrelated `SEED` env. This patch eliminates the
    // validate-then-discard anti-pattern.
    // Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877
    if std::env::var("SEED").is_ok() {
        let canon_seed =
            parse_seed().map_err(|e| anyhow::anyhow!("Canon #93 violation (SEED env): {}", e))?;
        eprintln!(
            "[trios-train] Canon #93 OK: SEED={canon_seed} (overrides cli.seed={})",
            cli.seed
        );
        cli.seed = canon_seed;
    } else {
        const FORBIDDEN: &[u64] = &[42, 43, 44, 45];
        if FORBIDDEN.contains(&cli.seed) {
            return Err(anyhow::anyhow!(
                "Canon #93 violation: cli.seed={} is forbidden (allowed: 47, 89, 123, 144). \
                 Set SEED or TRIOS_SEED env var to an allowed value, or pass `--seed=<allowed>`.",
                cli.seed
            ));
        }
        eprintln!(
            "[trios-train] Canon #93 OK: seed={} (no SEED env, validated cli.seed)",
            cli.seed
        );
    }

    // Run SeaORM migrations at startup - only with TRIOS_ALLOW_AUTOMIGRATE=1.
    run_automigrate();

    // R5/L8 fix (trios#509 follow-up): re-export `--format` into the env so
    // `train_loop::resolve_fake_quant_format()` can see it. Without this line
    // the CLI flag was a no-op and every scarab-spawned trainer silently fell
    // back to F32 regardless of the strategy_queue config.
    if let Some(fmt) = &cli.format {
        if !fmt.is_empty() {
            std::env::set_var("TRIOS_FORMAT_TYPE", fmt);
        }
    }

    // `--eval-chunks` is the CLI face of `TRIOS_EVAL_CHUNKS`. Re-exported here,
    // before any run starts, so the flag and the env var reach exactly one
    // resolver (`train_loop::eval_chunks_target`) and the flag wins. The
    // achieved value is announced either way: it is a published measurement's
    // sampling grid, not a private detail.
    if let Some(chunks) = cli.eval_chunks {
        std::env::set_var("TRIOS_EVAL_CHUNKS", chunks.to_string());
    }
    eprintln!(
        "[trios-train] eval_chunks={} ({})",
        train_loop::eval_chunks_target(),
        match cli.eval_chunks {
            Some(_) => "--eval-chunks",
            None if std::env::var("TRIOS_EVAL_CHUNKS").is_ok() => "TRIOS_EVAL_CHUNKS",
            None => "default",
        }
    );

    // Set NEON_DATABASE_URL from --neon flag OR inherit from ENV (used by scarab worker)
    // scarab passes NEON_DATABASE_URL via ENV inheritance, so check that first
    if std::env::var("NEON_DATABASE_URL").is_err() {
        if let Some(neon_url) = &cli.neon {
            std::env::set_var("NEON_DATABASE_URL", neon_url);
        }
    }

    eprintln!(
        "[trios-train] parsed seed={} steps={} hidden={} lr={} ctx={:?} optimizer={} neon={:?}",
        cli.seed, cli.steps, cli.hidden, cli.lr, cli.ctx, cli.optimizer, cli.neon
    );
    let _ = std::io::stderr().flush();

    if let Some(config_path) = &cli.config {
        let cfg = trios_trainer::TrainConfig::from_toml(config_path)?;
        tracing::info!(name = %cfg.name, seed = cfg.seed, steps = cfg.steps, "config mode");
        let outcome = train_loop::run(&cfg)?;
        tracing::info!(?outcome, "training complete");
        // Config mode trains and writes to the same ledger, so it must not be
        // the one exit path that returns 0 regardless.
        exit_with_ledger_status();
    }

    if cli.sweep || cli.seed == 0 {
        tracing::info!("3-seed sweep: {:?}", GATE_FINAL_SEEDS);
        // `--optimizer` reaches the sweep. It did not: this branch had no
        // optimizer argument at all, so `--sweep --optimizer soap` ran AdamW
        // three times and exited 0 with a `GATE-2:` verdict attached.
        let results = train_loop::run_sweep(
            cli.steps,
            cli.hidden,
            cli.lr,
            cli.attn_layers,
            cli.eval_every,
            &cli.train_data,
            &cli.val_data,
            &cli.optimizer,
        )?;
        // One formatter for both branches (`train_loop::format_done_line`), so
        // the sweep arm and the single-seed arm cannot report different things
        // about the same kind of run. The sweep line used to carry no `opt=`
        // token, which is what let a run launched as `--optimizer soap` and
        // executed as AdamW leave a stdout trail naming no optimizer. It also
        // printed `r.final_bpb`, the compat mirror that is NaN when the run
        // took no final measurement - and `f64::from_str` accepts "NaN", so
        // every downstream parser read it back as a number.
        for r in &results {
            println!("{}", train_loop::format_done_line(r, &cli.optimizer));
        }
        // `.all()` alone is vacuously true on an empty result set: zero seeds,
        // zero measurements, `GATE-2: PASS`. A seed that took no final
        // measurement is not a passing seed either - `None` cannot be below a
        // target.
        let all_pass = !results.is_empty()
            && results.iter().all(|r| {
                r.final_val_bpb
                    .is_some_and(|bpb| bpb < train_loop::DEFAULT_IGLA_TARGET_BPB)
            });
        println!("GATE-2: {}", if all_pass { "PASS" } else { "NOT YET" });
    } else {
        let args = TrainArgs {
            seed: cli.seed,
            steps: cli.steps,
            hidden: cli.hidden,
            lr: cli.lr,
            attn_layers: cli.attn_layers,
            eval_every: cli.eval_every,
            train_path: cli.train_data.clone(),
            val_path: cli.val_data.clone(),
        };
        // R5-honest dispatch: every supported optimizer is named explicitly.
        // Any unsupported name is a hard error, NOT a silent AdamW fallback.
        // Pre-Wave-35 bug: 12 optimizer-named canons (lion/soap/tiger/...)
        // silently ran AdamW, producing byte-identical BPB across the fleet.
        // The match that used to live here was duplicated in the library as
        // `run_with_optimizer` and diverged from the sweep arm, which had no
        // dispatch at all; there is now exactly one.
        let outcome = train_loop::run_with_optimizer(&cli.optimizer, &args)?;
        // `bpb=` is the RAW val_bpb measured at the final step. It used to be
        // `best_bpb`: the running minimum of an EMA seeded at init (~7.0), so
        // two runs with byte-identical weights printed 3.5506 and 4.4940 while
        // the measurement was 2.8534 in both. A run that took no final
        // measurement says so instead of substituting a number.
        println!("{}", train_loop::format_done_line(&outcome, &cli.optimizer));
        // R5/L8: flush so seed-agent reader sees DONE before EOF.
        use std::io::Write as _;
        let _ = std::io::stdout().flush();
    }

    // A configured DSN is a statement that this run was supposed to be
    // recorded. If every write was dropped, exiting 0 would let a supervisor
    // file it as a success.
    exit_with_ledger_status();
}
