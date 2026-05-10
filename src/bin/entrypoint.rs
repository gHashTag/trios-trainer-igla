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

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Canon #93 enforcement: read seed from env (`SEED`, falling back to
/// legacy `TRIOS_SEED`) and reject the forbidden canon set
/// `{42, 43, 44, 45}`. Returns the parsed seed string on success, an
/// `Err(message)` on rejection or parse failure.
///
/// Allowed canon set per Canon #93: `{47, 89, 123, 144}`. Seeds outside
/// both sets are accepted with a warning (operator override path).
///
/// Wave 29 hotfix: previously the entrypoint defaulted to `"43"`, which
/// was a forbidden canon and caused 3,453 rows in `public.bpb_samples`
/// to be written under seed=43, in turn blocking the unique constraint
/// `bpb_samples_canon_name_seed_step_key` on every fresh INSERT.
pub(crate) fn parse_seed() -> Result<String, String> {
    const FORBIDDEN: [u64; 4] = [42, 43, 44, 45];
    const ALLOWED: [u64; 4] = [47, 89, 123, 144];

    let raw = env::var("SEED")
        .or_else(|_| env::var("TRIOS_SEED"))
        .map_err(|_| {
            "SEED env var unset (and legacy TRIOS_SEED also unset). \
             Canon #93 requires one of {47, 89, 123, 144}."
                .to_string()
        })?;
    let seed: u64 = raw
        .parse()
        .map_err(|e| format!("SEED parse error: {e} (got {raw:?})"))?;
    if FORBIDDEN.contains(&seed) {
        return Err(format!(
            "seed {seed} is in forbidden canon set {{42, 43, 44, 45}}; \
             use one of {{47, 89, 123, 144}} (Canon #93)"
        ));
    }
    if !ALLOWED.contains(&seed) {
        eprintln!(
            "[entrypoint] WARN: seed {seed} outside Canon #93 allowed set \
             {{47, 89, 123, 144}} — proceeding but flagging"
        );
    }
    Ok(seed.to_string())
}

fn main() {
    let seed = match parse_seed() {
        Ok(s) => s,
        Err(msg) => {
            eprintln!("[entrypoint] {msg}");
            std::process::exit(2);
        }
    };
    let steps = env_or("TRIOS_STEPS", "81000");
    let lr = env_or("TRIOS_LR", "0.003");
    let hidden = env_or("TRIOS_HIDDEN", "384");
    let optimizer = env_or("TRIOS_OPTIMIZER", "adamw");

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

#[cfg(test)]
mod tests {
    //! Wave 29 hotfix tests — Canon #93 SEED validation.
    //!
    //! `parse_seed` reads from process env, so each test sets `SEED` (and
    //! clears `TRIOS_SEED`) before calling. Tests run serially via a
    //! mutex because env access is process-global.

    use super::parse_seed;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_env(seed: Option<&str>, f: impl FnOnce()) {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev_seed = std::env::var("SEED").ok();
        let prev_trios = std::env::var("TRIOS_SEED").ok();
        // SAFETY: these tests are single-threaded relative to each
        // other (the `_guard` mutex serialises them), and the binary
        // crate has no async tasks reading env in the test harness.
        unsafe {
            std::env::remove_var("TRIOS_SEED");
            match seed {
                Some(v) => std::env::set_var("SEED", v),
                None => std::env::remove_var("SEED"),
            }
        }
        f();
        // SAFETY: restore prior env so other tests are unaffected.
        unsafe {
            match prev_seed {
                Some(v) => std::env::set_var("SEED", v),
                None => std::env::remove_var("SEED"),
            }
            match prev_trios {
                Some(v) => std::env::set_var("TRIOS_SEED", v),
                None => std::env::remove_var("TRIOS_SEED"),
            }
        }
    }

    #[test]
    fn forbidden_seeds_rejected() {
        for s in ["42", "43", "44", "45"] {
            with_env(Some(s), || {
                let err = parse_seed().expect_err(&format!("seed={s} must be rejected"));
                assert!(
                    err.contains("forbidden canon set"),
                    "error message for seed={s} should cite forbidden canon, got {err:?}"
                );
            });
        }
    }

    #[test]
    fn allowed_canon_seeds_accepted() {
        for s in ["47", "89", "123", "144"] {
            with_env(Some(s), || {
                let v = parse_seed().expect("Canon #93 allowed seed must parse");
                assert_eq!(v, s);
            });
        }
    }

    #[test]
    fn missing_env_errors_cleanly() {
        with_env(None, || {
            let err = parse_seed().expect_err("missing env must be Err");
            assert!(
                err.contains("Canon #93"),
                "error message should cite Canon #93, got {err:?}"
            );
        });
    }

    #[test]
    fn unparseable_seed_rejected() {
        with_env(Some("not-a-number"), || {
            let err = parse_seed().expect_err("non-numeric must be Err");
            assert!(
                err.contains("parse error"),
                "error message should cite parse error, got {err:?}"
            );
        });
    }

    #[test]
    fn warning_seed_outside_canon_still_accepted() {
        // R5-honest: per Canon #93 contract, only forbidden seeds are hard-rejected.
        // Seeds outside both forbidden+allowed sets are accepted with a warning,
        // so legacy operator overrides still work.
        with_env(Some("100"), || {
            let v = parse_seed().expect("non-canon seed should parse with warning");
            assert_eq!(v, "100");
        });
    }
}
