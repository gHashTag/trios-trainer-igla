//! Container entrypoint — Rust replacement for `scripts/entrypoint.sh`.
//!
//! LAWS.md L1 (Rust-only) forbids `.sh` files in the repository. This
//! binary reads the same `TRIOS_*` environment variables the script
//! used, logs the resolved configuration, and execs `trios-train` with
//! matching CLI flags. The behavior matches the legacy shim 1:1.
//!
//! Anchor: `phi^2 + phi^-2 = 3`.

use std::env;
use std::process::Command;
use trios_trainer::entrypoint_env::resolve_env_alias;

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn main() {
    // Wave 33 hotfix: accept both `TRIOS_<KEY>` and the un-prefixed alias.
    // See `trios_trainer::entrypoint_env` for the resolution order and the
    // root-cause analysis (Wave-29 STEPS=200000 silently dropped).
    let (seed, seed_src) = resolve_env_alias("TRIOS_SEED", "SEED", "43");
    let (steps, steps_src) = resolve_env_alias("TRIOS_STEPS", "STEPS", "81000");
    let (lr, lr_src) = resolve_env_alias("TRIOS_LR", "LR", "0.003");
    let (hidden, hidden_src) = resolve_env_alias("TRIOS_HIDDEN", "HIDDEN_DIM", "384");
    // Wave 34 hotfix: `OPTIMIZER` alias was forgotten in PR #130's STEPS/LR/HIDDEN/SEED
    // alias fan-out. Result: 38-service Wave-34 deploy that set `OPTIMIZER=lion` (etc.)
    // silently fell back to TRIOS_OPTIMIZER's default "adamw". Combined with the wildcard
    // dispatch arm in trios-train.rs (also fixed in this PR), 15 nominally-distinct
    // optimizers converged to bit-identical BPB=2.6814258098602295 on seed=123.
    let (optimizer, optimizer_src) = resolve_env_alias("TRIOS_OPTIMIZER", "OPTIMIZER", "adamw");

    // Wave 33 trace: emit a deterministic startup line with every resolved
    // knob plus its source (TRIOS_* / alias / default). Operators can
    // verify with a single `grep entrypoint-trace` that an env-var override
    // actually reached the trainer. Without this trace, Wave-29
    // STEPS=200000 silently degraded to default 81000 and 52 trainers
    // exited two cycles short; no log line told us why.
    println!(
        "[entrypoint-trace] seed=({}, src={}) steps=({}, src={}) lr=({}, src={}) hidden=({}, src={}) opt=({}, src={})",
        seed, seed_src.as_str(),
        steps, steps_src.as_str(),
        lr, lr_src.as_str(),
        hidden, hidden_src.as_str(),
        optimizer, optimizer_src.as_str(),
    );
    if let Ok(v) = env::var("NUM_ATTN_LAYERS") {
        println!("[entrypoint-trace] NUM_ATTN_LAYERS={v} (consumed inside train_loop::run_single)");
    }
    if let Ok(v) = env::var("GF16_ENABLED") {
        println!("[entrypoint-trace] GF16_ENABLED={v} (consumed inside train_loop::run_single)");
    }

    let train_data = env_or("TRIOS_TRAIN_DATA", "/work/data/tiny_shakespeare.txt");
    let val_data = env_or("TRIOS_VAL_DATA", "/work/data/tiny_shakespeare_val.txt");

    let trainer = env_or("TRIOS_TRAINER_BIN", "trios-train");
    if !matches!(
        trainer.as_str(),
        "trios-train" | "scarab" | "gf16_test" | "ngram_train_gf16"
    ) {
        eprintln!(
            "[entrypoint] TRIOS_TRAINER_BIN={trainer:?} is not in the allowed set \
             {{trios-train, gf16_test, ngram_train_gf16}}"
        );
        std::process::exit(2);
    }
    let trainer_path = format!("/usr/local/bin/{trainer}");

    println!(
        "[entrypoint] {trainer} seed={seed} steps={steps} lr={lr} hidden={hidden} opt={optimizer}"
    );
    println!("[entrypoint] train={train_data} val={val_data}");

    let mut cmd = Command::new(&trainer_path);
    cmd.arg(format!("--seed={seed}"))
        .arg(format!("--steps={steps}"))
        .arg(format!("--lr={lr}"))
        .arg(format!("--hidden={hidden}"))
        .arg(format!("--optimizer={optimizer}"))
        .arg(format!("--train-data={train_data}"))
        .arg(format!("--val-data={val_data}"));

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec();
        eprintln!("[entrypoint] exec failed: {err}");
        std::process::exit(1);
    }

    #[cfg(not(unix))]
    {
        match cmd.status() {
            Ok(status) => std::process::exit(status.code().unwrap_or(1)),
            Err(err) => {
                eprintln!("[entrypoint] spawn failed: {err}");
                std::process::exit(1);
            }
        }
    }
}
