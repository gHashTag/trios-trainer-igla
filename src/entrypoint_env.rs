//! Wave 33 entrypoint env-var helpers.
//!
//! Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877
//!
//! # Why this module exists
//!
//! `entrypoint` is the container CMD that resolves `TRIOS_*` environment
//! variables and execs `trios-train` with matching `--<flag>=<value>` CLI
//! args. Until Wave 33 it only honoured the `TRIOS_<KEY>` form, so any
//! operator or cron script that set the un-prefixed alias (e.g. `STEPS`,
//! `HIDDEN_DIM`, `SEED`) had their override silently dropped.
//!
//! Concrete failure that motivated this module: Wave-29 cron set
//! `STEPS=200000` on 52 Railway services expecting trainers to run for
//! 200K steps. Instead they all exited DONE at the 81000 default because
//! `entrypoint` only read `TRIOS_STEPS`. The trace-line below makes that
//! kind of silent fallback visible.
//!
//! # Resolution order
//!
//! For each knob:
//! 1. `TRIOS_<KEY>` (canonical name, takes precedence)
//! 2. `<KEY>` (un-prefixed alias, accepted for operator/cron compatibility)
//! 3. caller-provided default
//!
//! `arch_config::parse_*` already reads the un-prefixed names *inside*
//! `train_loop::run_single` (so `HIDDEN_DIM` overrides the CLI value at
//! model-init time), but `STEPS` has no such second chance — the trainer
//! loop runs `for step in 1..=args.steps` where `args.steps` came from
//! `entrypoint`. Hence the alias must live here.

use std::env;

/// The source of a resolved env-var value, used by the startup trace line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveSrc {
    /// Value came from the canonical `TRIOS_<KEY>` env-var.
    TriosPrefixed,
    /// Value came from the un-prefixed alias (e.g. `STEPS`).
    Alias,
    /// Neither env-var was set; the caller-provided default was used.
    Default,
}

impl ResolveSrc {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResolveSrc::TriosPrefixed => "TRIOS_*",
            ResolveSrc::Alias => "alias",
            ResolveSrc::Default => "default",
        }
    }
}

/// Resolve an env-var value with a fallback alias, returning both the value
/// and which source provided it.
///
/// Reads the canonical `trios_key` first; falls back to `alias` if unset;
/// finally falls back to `default`.
///
/// # Examples
///
/// ```
/// use trios_trainer::entrypoint_env::{resolve_env_alias, ResolveSrc};
/// std::env::remove_var("TRIOS_FOO");
/// std::env::remove_var("FOO");
/// let (v, src) = resolve_env_alias("TRIOS_FOO", "FOO", "default_val");
/// assert_eq!(v, "default_val");
/// assert_eq!(src, ResolveSrc::Default);
/// ```
pub fn resolve_env_alias(trios_key: &str, alias: &str, default: &str) -> (String, ResolveSrc) {
    if let Ok(v) = env::var(trios_key) {
        (v, ResolveSrc::TriosPrefixed)
    } else if let Ok(v) = env::var(alias) {
        (v, ResolveSrc::Alias)
    } else {
        (default.to_string(), ResolveSrc::Default)
    }
}
