//! trios-trainer — portable IGLA RACE training pipeline.
//! Single-source-of-truth for `gHashTag/trios#143`. Anchor: phi^2 + phi^-2 = 3.
//!
//! Clippy/dead_code debt living on `main` since before integration PR #32 is
//! allow-listed in `Cargo.toml` `[lints]` to keep CI green while we focus on
//! Gate-2 (deadline 2026-04-30 23:59 UTC). Each lint pays down in a dedicated
//! technical-debt PR after merge. R5-honest: NOT introduced by PR #32.

pub mod arch_config;
pub mod canonical_digest;
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
pub mod provenance_seal;
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

/// Exit code for an argument this process cannot honour.
///
/// The value `cpu_train`, `lstm_train`, `train_v2` and `matrix_runner` already
/// use, so a sweep can tell "the CLI was refused" from "the trainer crashed".
/// It lives here rather than as a tenth private copy: a constant duplicated
/// once per binary is a constant that will drift, and the whole point of an
/// exit code is that a fleet can compare it across binaries.
pub const EXIT_BAD_ARGS: u8 = 4;

/// Refuse an argument whose NAME this binary does not read.
///
/// The NAME half of an argument is as much the identity of a run as the VALUE
/// half. `parse_flag_value` above exists because `--seed=oops` used to train at
/// 42 and print `seed=42`; this is the other half of the same defect, and it
/// was measured, not imagined:
///
/// ```text
/// $ hybrid_train --seed=47 --steps=20 --train=/tmp/nope_train.txt --val=/tmp/nope_val.txt
/// train=1015394 val=100000
/// seed=47 bpb=5.0731
/// $ echo $?
/// 0
/// ```
///
/// `hybrid_train` reads `--train-path` / `--val-path`. It dropped `--train` and
/// `--val` without a word, trained the DEFAULT split, and published a BPB with
/// exit 0 while the operator believed a corpus pair of their own had been
/// supplied. (The same defect was first reported at `bpb=4.0594`; the run above
/// is this crate's own reproduction of the mechanism at 20 steps.) Four of the
/// binaries that ignored unknown names feed a corpus pair into the train/val
/// disjointness guard, so a misspelled `--val-path` makes the leak guard
/// evaluate a different pair than the one asked for -- on a worker fleet where
/// nobody reads stderr.
///
/// `known` is the list of spellings the caller's own parser reads, WITHOUT the
/// leading dashes. The trailing `=` is significant, because the parsers differ:
///
/// | entry in `known` | accepted             | rejected                       |
/// |------------------|----------------------|--------------------------------|
/// | `"steps="`       | `--steps=200`        | `--steps 200` (parser reads only the `=` form, so the space form would silently train the default) |
/// | `"ctx3"`         | `--ctx3`             | `--ctx3=1` (parser compares the whole token) |
/// | both entries     | `--p=X` and `--p X`  | -                              |
///
/// A flag listed in BOTH forms takes a value in either spelling, so the token
/// after the bare form is consumed as that value. Anything else that does not
/// begin with `-` is an unexpected positional and is refused: none of the
/// trainers takes positional arguments, and a stray one is how a shell-quoting
/// mistake becomes a run nobody asked for. `--` ends the scan; everything after
/// it is passed through untouched. A leading program name (`argv[0]`, which
/// does not begin with `-`) is skipped, so `std::env::args()` can be handed in
/// whole.
///
/// Dependency-free on purpose, like `parse_flag_value`: it is called from
/// binaries that must not grow a CLI crate.
pub fn reject_unknown_args(argv: &[String], known: &[&str]) -> Result<(), String> {
    // Skip the program name, but only if it looks like one: a caller that
    // hands in a bare argument list must not have its first flag skipped.
    let mut i = usize::from(argv.first().is_some_and(|a| !a.starts_with('-')));
    while i < argv.len() {
        let token = &argv[i];
        if token == "--" {
            return Ok(());
        }
        if !token.starts_with('-') {
            return Err(format!(
                "UNEXPECTED ARGUMENT: {token:?} is not a flag and this binary takes \
                 no positional arguments. It would have been ignored in silence, so \
                 the run that happened would not be the run that was asked for. \
                 Refusing instead.\n{}",
                accepted_list(known)
            ));
        }
        let body = token.trim_start_matches('-');
        let (name, attached_value) = match body.split_once('=') {
            Some((n, _)) => (n, true),
            None => (body, false),
        };
        let equals_form = known.contains(&format!("{name}=").as_str());
        let bare_form = known.contains(&name);
        if !equals_form && !bare_form {
            return Err(format!(
                "UNKNOWN ARGUMENT: {token:?}. This binary does not read --{name}, and \
                 it used to discard such an argument in silence: the caller asks for \
                 one run and gets another, with a clean BPB and exit 0. Refusing \
                 instead.\n{}",
                accepted_list(known)
            ));
        }
        if attached_value && !equals_form {
            return Err(format!(
                "UNKNOWN ARGUMENT: {token:?}. --{name} is a switch here and takes no \
                 value; this binary compares the whole token, so the value would have \
                 been discarded together with the switch. Refusing instead.\n{}",
                accepted_list(known)
            ));
        }
        if !attached_value {
            if !bare_form {
                return Err(format!(
                    "UNKNOWN ARGUMENT: {token:?}. This binary reads --{name} only in \
                     the --{name}=VALUE form, so the space-separated spelling would \
                     have left --{name} at its default and exited 0. Refusing \
                     instead.\n{}",
                    accepted_list(known)
                ));
            }
            // Listed in both forms: it takes a value, and the bare spelling
            // consumes the next token as that value.
            if equals_form && argv.get(i + 1).is_some_and(|v| !v.starts_with('-')) {
                i += 1;
            }
        }
        i += 1;
    }
    Ok(())
}

/// The accepted spellings, rendered the way the parser reads them.
///
/// A refusal that does not say what WOULD have worked sends the operator back
/// to the source; these binaries have between three and twenty-one flags and no
/// `--help` worth the name.
fn accepted_list(known: &[&str]) -> String {
    let mut rendered: Vec<String> = Vec::new();
    for entry in known {
        let name = entry.strip_suffix('=').unwrap_or(entry);
        let takes_value = known.contains(&format!("{name}=").as_str());
        let bare = known.contains(&name);
        let spelling = match (takes_value, bare) {
            (true, true) => format!("--{name}=VALUE | --{name} VALUE"),
            (true, false) => format!("--{name}=VALUE"),
            _ => format!("--{name}"),
        };
        if !rendered.contains(&spelling) {
            rendered.push(spelling);
        }
    }
    format!("Accepted arguments: {}", rendered.join(", "))
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

    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    /// The measured incident: `hybrid_train`'s spellings are `--train-path` and
    /// `--val-path`, and `--train` / `--val` were dropped in silence.
    #[test]
    fn reject_unknown_args_refuses_and_names_the_offending_flag() {
        let known = [
            "seed=",
            "steps=",
            "train-path",
            "train-path=",
            "val-path",
            "val-path=",
        ];
        let err = reject_unknown_args(
            &argv(["hybrid_train", "--seed=47", "--train=/tmp/nope.txt"].as_slice()),
            &known,
        )
        .expect_err("a flag this binary does not read must not be discarded");
        assert!(err.contains("--train"), "{err}");
        assert!(
            err.contains("--train-path=VALUE"),
            "must say what would have worked: {err}"
        );
    }

    /// Both spellings the parsers actually implement, plus the shapes that used
    /// to slip through: a value on a switch, a switch spelling on an `=`-only
    /// flag, and a stray positional.
    #[test]
    fn reject_unknown_args_matches_the_parser_it_guards() {
        let known = ["steps=", "ctx3", "train-data", "train-data=", "help"];

        // Accepted: exactly what the parsers read.
        assert_eq!(
            reject_unknown_args(&argv(&["p", "--steps=200"]), &known),
            Ok(())
        );
        assert_eq!(reject_unknown_args(&argv(&["p", "--ctx3"]), &known), Ok(()));
        assert_eq!(
            reject_unknown_args(
                &argv(&["p", "--train-data", "data/x.txt", "--help"]),
                &known
            ),
            Ok(())
        );
        assert_eq!(
            reject_unknown_args(&argv(&["p", "--train-data=data/x.txt"]), &known),
            Ok(())
        );
        // No program name to skip, and `--` ends the scan.
        assert_eq!(reject_unknown_args(&argv(&["--ctx3"]), &known), Ok(()));
        assert_eq!(
            reject_unknown_args(&argv(&["p", "--", "--whatever"]), &known),
            Ok(())
        );

        // `--steps 200` would have left steps at its default and exited 0.
        let err = reject_unknown_args(&argv(&["p", "--steps", "200"]), &known).unwrap_err();
        assert!(err.contains("--steps=VALUE"), "{err}");
        // `--ctx3=1` is compared as a whole token, so the switch never fires.
        let err = reject_unknown_args(&argv(&["p", "--ctx3=1"]), &known).unwrap_err();
        assert!(err.contains("switch"), "{err}");
        // A stray positional is not a value of anything.
        let err = reject_unknown_args(&argv(&["p", "47"]), &known).unwrap_err();
        assert!(err.contains("positional"), "{err}");
        // A misspelling one character away from a real flag is still a refusal.
        let err = reject_unknown_args(&argv(&["p", "--train-path=x"]), &known).unwrap_err();
        assert!(err.contains("--train-path"), "{err}");
    }
}
