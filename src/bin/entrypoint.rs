//! Container entrypoint — Rust replacement for `scripts/entrypoint.sh`.
//!
//! LAWS.md L1 (Rust-only) forbids `.sh` files in the repository. This
//! binary reads the same `TRIOS_*` environment variables the script
//! used, logs the resolved configuration, and execs `trios-train` with
//! matching CLI flags. The behavior matches the legacy shim 1:1.
//!
//! SCARAB MODE: When TRIOS_TRAINER_BIN=scarab, this entrypoint bypasses
//! all argument construction and directly execs `/usr/local/bin/scarab`.
//! Scarab manages its own lifecycle via Neon queue and does not accept
//! CLI args like --seed, --steps, etc.
//!
//! Anchor: `phi^2 + phi^-2 = 3`.

use std::env;
use std::process::Command;

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn main() {
    let trainer = env_or("TRIOS_TRAINER_BIN", "trios-train");

    // Scarab mode: bypass entrypoint logic, run scarab directly
    if trainer == "scarab" {
        println!("[entrypoint] SCARAB MODE: executing /usr/local/bin/scarab");
        println!(
            "[entrypoint] scarab reads NEON_DATABASE_URL from env and claims from strategy_queue"
        );

        let mut cmd = Command::new("/usr/local/bin/scarab");

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            let err = cmd.exec();
            eprintln!("[entrypoint] scarab exec failed: {err}");
            std::process::exit(1);
        }

        #[cfg(not(unix))]
        {
            let status = cmd
                .status()
                .unwrap_or_else(|err| panic!("[entrypoint] scarab spawn failed: {err}"));
            std::process::exit(status.code().unwrap_or(1));
        }
    }

    // Trainer mode: construct args and exec trios-train (or other whitelisted bin)
    let seed = env_or("TRIOS_SEED", "43");
    let steps = env_or("TRIOS_STEPS", "81000");
    let lr = env_or("TRIOS_LR", "0.003");
    let hidden = env_or("TRIOS_HIDDEN", "384");
    let optimizer = env_or("TRIOS_OPTIMIZER", "adamw");

    let train_data = env_or("TRIOS_TRAIN_DATA", "/work/data/tiny_shakespeare.txt");
    let val_data = env_or("TRIOS_VAL_DATA", "/work/data/tiny_shakespeare_val.txt");

    // Whitelist check for trainer binaries (scarab handled above)
    if !matches!(
        trainer.as_str(),
        "trios-train" | "gf16_test" | "ngram_train_gf16"
    ) {
        eprintln!(
            "[entrypoint] TRIOS_TRAINER_BIN={trainer:?} is not in the allowed set \
             {{trios-train, gf16_test, ngram_train_gf16, scarab}}"
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
        let status = cmd
            .status()
            .unwrap_or_else(|err| panic!("[entrypoint] spawn failed: {err}"));
        std::process::exit(status.code().unwrap_or(1));
    }
}
