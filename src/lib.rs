//! trios-trainer — portable IGLA RACE training pipeline.
//! Single-source-of-truth for `gHashTag/trios#143`. Anchor: phi^2 + phi^-2 = 3.
//!
//! Clippy/dead_code debt living on `main` since before integration PR #32 is
//! allow-listed in `Cargo.toml` `[lints]` to keep CI green while we focus on
//! Gate-2 (deadline 2026-04-30 23:59 UTC). Each lint pays down in a dedicated
//! technical-debt PR after merge. R5-honest: NOT introduced by PR #32.

pub mod arch_config;
pub mod checkpoint;
pub mod config;
pub mod data;
pub mod entities;
pub mod entrypoint_env;
pub mod fake_quant;
pub mod format_ladder;
pub mod gf16;
pub mod igla;
pub mod invariants;
pub mod jepa;
pub mod ledger;
pub mod model;
pub mod model_hybrid_attn;
pub mod multi_seed;
pub mod mup;
pub mod neon_writer;
pub mod objective;
pub mod optimizer;
pub mod phi_numbers;
pub mod race;
pub mod seed_canon;
pub mod train_loop;

pub use config::TrainConfig;
pub use train_loop::{run, RunOutcome};

pub const TRINITY_ANCHOR: f64 = 3.0;

/// Parse the value of a KNOWN command-line flag, or say why it could not be.
///
/// The binaries used to write `arg_or("seed", "42").parse().unwrap_or(42)`.
/// `--seed=oops` therefore trained at seed 42, printed `seed=42` beside a BPB,
/// and exited 0: the caller asked for one run and silently got another. The
/// seed, the step count and the learning rate ARE the identity of a
/// reproducibility claim, so a value that does not parse is a refusal and
/// never a default.
///
/// This is the value-substitution twin of `cpu_train`'s `first_unknown_arg`
/// guard, which already refuses an unrecognised flag NAME; a recognised name
/// with an unusable value was the remaining silent path. Dependency-free on
/// purpose: it is called from binaries that must not grow a CLI crate.
pub fn parse_flag_value<T: std::str::FromStr>(name: &str, raw: &str) -> Result<T, String> {
    raw.trim().parse::<T>().map_err(|_| {
        format!(
            "UNPARSEABLE ARGUMENT: --{name}={raw:?} is not a valid {}. \
             This binary used to substitute its own default here and exit 0, so \
             the run that happened was not the run that was asked for. Refusing \
             instead; fix --{name} and re-run.",
            std::any::type_name::<T>()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn anchor_holds() {
        let phi: f64 = (1.0 + 5f64.sqrt()) / 2.0;
        let lhs = phi.powi(2) + phi.powi(-2);
        assert!((lhs - TRINITY_ANCHOR).abs() < 1e-12);
    }

    /// The observed defect: `--seed=oops --lr=0,001` printed `seed=42
    /// lr=0.003` and exited 0. A refusal must name the flag AND the text it
    /// could not use, so the operator can see which of the two went wrong.
    #[test]
    fn parse_flag_value_refuses_and_names_the_offending_text() {
        let err = parse_flag_value::<u64>("seed", "oops")
            .expect_err("an unparseable seed must not become the default");
        assert!(err.contains("--seed"), "{err}");
        assert!(err.contains("oops"), "{err}");
        assert!(err.contains("u64"), "{err}");

        // The comma-decimal case that silently trained at 0.003.
        let err = parse_flag_value::<f32>("lr", "0,001").expect_err("0,001 is not an f32");
        assert!(err.contains("0,001"), "{err}");

        assert_eq!(parse_flag_value::<u64>("seed", "47"), Ok(47));
        assert_eq!(parse_flag_value::<usize>("steps", " 200 "), Ok(200));
        assert_eq!(parse_flag_value::<f64>("lr", "0.001"), Ok(0.001));
    }
}
