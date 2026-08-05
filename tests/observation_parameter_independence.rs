//! A DECLARED OBSERVATION PARAMETER MUST NOT CHANGE THE WEIGHTS.
//!
//! The record schema has been extended seven times. Every generation was
//! believed complete when it was written, and every generation was falsified by
//! exactly one more field that turned out to be part of the recipe rather than
//! part of the report. The flagship case was found by accident after 1,851
//! runs: `gf16_floor` -- which mutates `embed`, `proj`, `lm_head` and `ctx` in
//! place -- was gated on `step % args.eval_every == 0`, so `--eval-every`, a
//! knob whose entire documented purpose is deciding HOW OFTEN TO LOOK, silently
//! decided WHAT WAS THERE TO LOOK AT. Two seed-47 runs differing only in that
//! flag produced different weights and BPB 2.6141 vs 2.6169.
//!
//! Adding one more field to the schema does not close that class of defect; it
//! only closes the one instance. What closes the class is a PROCEDURE, and this
//! file is it: for each declared observation parameter, run the real release
//! binary twice with everything fixed except that one knob, and require the
//! emitted checkpoints to be byte-identical.
//!
//! Note what is NOT the invariant. Rust's type system already guarantees that
//! `evaluate(&model, ..)` cannot mutate the model -- the defect was never
//! inside the evaluator. The invariant that actually bites is one level up:
//!
//!     the weights at step k must not depend on ANY observation parameter.
//!
//! These are subprocess tests against the built binary on purpose. The subject
//! is the artifact a second laboratory would obtain by following the documented
//! command line, and a unit test on the library cannot see the file on disk.
//!
//! A FAILURE HERE IS THE PROCEDURE WORKING. It names a newly discovered
//! observation-coupled parameter, and the fix is to decouple the training loop
//! from it -- never to relax this test. See `docs/OBSERVATION-INDEPENDENCE.md`.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use sha2::{Digest, Sha256};

/// Every environment variable that can reach the trainer's recipe, cleared
/// before each run so the developer's or CI's shell cannot decide a result.
///
/// Deliberately over-broad: the DSN and ledger variables are not observation
/// parameters, but a run that reaches a real database is slower and can fail
/// for reasons that have nothing to do with the subject under test.
const CLEARED_ENV: &[&str] = &[
    // Ledger / database. `neon_writer::resolve_dsn` consults all four.
    "DATABASE_URL",
    "NEON_DATABASE_URL",
    "TRIOS_NEON_DSN",
    "TRIOS_DATABASE_URL",
    "TRIOS_LEDGER_WRITE",
    // Recipe knobs, all of which have a CLI flag or a fixed value below.
    "SEED",
    "TRIOS_SEED",
    "TRIOS_STEPS",
    "TRIOS_HIDDEN",
    "HIDDEN_DIM",
    "TRIOS_LR",
    "TRIOS_ATTN_LAYERS",
    "NUM_ATTN_LAYERS",
    "TRIOS_ATTN_SCALE",
    "TRIOS_ATTN_SEQ",
    "TRIOS_OPTIMIZER",
    "TRIOS_CTX",
    "TRIOS_CONFIG",
    "TRIOS_TRAIN_PATH",
    "TRIOS_VAL_PATH",
    "TRIOS_ALLOW_SYNTHETIC_DATA",
    "TRIOS_FORMAT_TYPE",
    "TRIOS_FAKE_QUANT_FORMAT",
    "GF16_ENABLED",
    "TRIOS_GF16_DISABLE",
    "TRIOS_GF16_FLOOR_EVERY",
    // Observation knobs. Each case sets back exactly the one it varies.
    "TRIOS_EVAL_EVERY",
    "TRIOS_EVAL_CHUNKS",
    "TRIOS_CHECKPOINT_EVERY",
    "TRIOS_CHECKPOINT_INIT",
    "TRIOS_CHECKPOINT_INTERVAL",
    "TRIOS_CHECKPOINT_DISABLE",
    "TRIOS_CHECKPOINT_DIR",
    "TRIOS_CANON_NAME",
    "CANON_NAME",
];

/// The recipe, held fixed across every case in this file.
///
/// Small on purpose -- `--hidden 64 --attn-layers 1` -- because the subject is
/// byte-identity, not model quality, and one case evaluates at full coverage
/// (775 windows) twice. Every value is passed explicitly rather than left to a
/// default so that a future change to a default cannot silently move the
/// comparison.
const RECIPE: &[&str] = &[
    "--steps",
    "200",
    "--hidden",
    "64",
    "--attn-layers",
    "1",
    "--lr",
    "0.003",
    "--optimizer",
    "adamw",
    "--train-data",
    "data/tiny_shakespeare.txt",
    "--val-data",
    "data/tiny_shakespeare_val.txt",
];

/// The eval cadence used by every case that is NOT varying it.
///
/// Carried as its own `Variant` field rather than as a base argument a case may
/// override: clap declares `--eval-every` with the default `ArgAction::Set` and
/// `trios-train` rejects a repeated occurrence outright ("the argument
/// '--eval-every <EVAL_EVERY>' cannot be used multiple times"), so "append and
/// let the last one win" is not available here.
const FIXED_EVAL_EVERY: &str = "100";

/// The seed every case that is NOT varying it runs at. Canon #93 allows
/// {47, 89, 123, 144}; the seed is a RECIPE parameter and is varied only by the
/// negative control at the bottom of this file.
const FIXED_SEED: &str = "47";

/// One side of a comparison: the knob's value, how it is delivered, and a
/// string the run must print to prove the value was actually honoured.
///
/// `proof` is not decoration. Without it a green test proves nothing: a knob
/// that the binary silently ignores -- exactly the failure mode that made
/// `TRIOS_CHECKPOINT_EVERY=1_000` resolve to 0 with no warning -- would produce
/// two identical runs and a passing assertion.
struct Variant {
    value: &'static str,
    /// The eval cadence this variant runs at. `FIXED_EVAL_EVERY` unless the
    /// case under test IS the eval cadence.
    eval_every: &'static str,
    /// The seed this variant runs at. `FIXED_SEED` everywhere except the
    /// negative control, which varies it on purpose.
    seed: &'static str,
    env: &'static [(&'static str, &'static str)],
    proof: &'static str,
}

/// One declared observation parameter and the two settings it is tested at.
struct Case {
    param: &'static str,
    a: Variant,
    b: Variant,
}

/// The step -> sha256 map a run left behind, read from the files themselves.
type Artifacts = BTreeMap<u64, String>;

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Every `*.bin` under `dir`, keyed by the step in its file name.
fn collect_bins(dir: &Path, out: &mut Artifacts) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_bins(&path, out);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("bin") {
            continue;
        }
        let step: u64 = match path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.parse().ok())
        {
            Some(s) => s,
            // A checkpoint whose name is not a step is not part of the
            // trajectory being compared; skip rather than guess.
            None => continue,
        };
        let bytes = std::fs::read(&path).expect("checkpoint file is readable");
        out.insert(step, sha256_hex(&bytes));
    }
}

/// Run one variant into its own directory and return what it wrote.
fn run_variant(case: &Case, v: &Variant, dir: &Path) -> Artifacts {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_trios-train"));
    // The corpus paths are repo-relative; `cargo test` does not guarantee the
    // cwd of the spawned child.
    cmd.current_dir(env!("CARGO_MANIFEST_DIR"));
    for var in CLEARED_ENV {
        cmd.env_remove(var);
    }
    cmd.env("TRINITY_AUTOMIGRATE", "0")
        .env("TRIOS_CHECKPOINT_DIR", dir)
        .env("TRIOS_CANON_NAME", "IGLA-OBSINDEP")
        .args(RECIPE)
        .args(["--seed", v.seed])
        .args(["--eval-every", v.eval_every]);
    for (k, val) in v.env {
        cmd.env(k, val);
    }

    let out = cmd.output().expect("spawn trios-train");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        out.status.success(),
        "{}={} did not complete: status={:?}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        case.param,
        v.value,
        out.status,
        stdout,
        stderr
    );
    assert!(
        stdout.contains(v.proof) || stderr.contains(v.proof),
        "{}={} was requested but the run never reported it ({:?} absent from its \
         output), so this comparison would have proved nothing.\n--- stdout ---\n{}",
        case.param,
        v.value,
        v.proof,
        stdout
    );

    let mut artifacts = Artifacts::new();
    collect_bins(dir, &mut artifacts);
    assert!(
        !artifacts.is_empty(),
        "{}={} wrote no checkpoint at all under {}\n--- stdout ---\n{}",
        case.param,
        v.value,
        dir.display(),
        stdout
    );
    artifacts
}

/// Run both variants and return their artifacts plus the steps both emitted.
///
/// The two runs need not stop at the same step: an artifact cadence changes
/// WHICH steps are snapshotted. The comparison is over the steps both actually
/// emitted, and the largest of those is the deepest point in training the pair
/// can be held to.
fn compare_variants(case: &Case) -> (Artifacts, Artifacts, Vec<u64>) {
    let dir_a = tempfile::tempdir().expect("tempdir");
    let dir_b = tempfile::tempdir().expect("tempdir");
    let a = run_variant(case, &case.a, dir_a.path());
    let b = run_variant(case, &case.b, dir_b.path());

    let common: Vec<u64> = a.keys().copied().filter(|k| b.contains_key(k)).collect();
    assert!(
        !common.is_empty(),
        "{} at {} emitted steps {:?} and at {} emitted steps {:?}: no common step, \
         so nothing was compared. Pin the cadences until the sets overlap rather \
         than dropping the assertion.",
        case.param,
        case.a.value,
        a.keys().collect::<Vec<_>>(),
        case.b.value,
        b.keys().collect::<Vec<_>>()
    );
    (a, b, common)
}

/// The whole procedure, for one parameter.
fn assert_observation_independent(case: Case) {
    let (a, b, common) = compare_variants(&case);
    for step in &common {
        assert_eq!(
            a[step],
            b[step],
            "OBSERVATION-COUPLED PARAMETER FOUND: {} changed the weights.\n  \
             {}={} -> step {} sha256={}\n  {}={} -> step {} sha256={}\n\
             The two runs are the same recipe and must be the same artifact. A \
             parameter that decides what is measured must not decide what is \
             measured ON. Fix the training loop (see the gf16_floor/--eval-every \
             precedent in src/train_loop.rs); do not relax this assertion. \
             Largest common step: {}.",
            case.param,
            case.param,
            case.a.value,
            step,
            a[step],
            case.param,
            case.b.value,
            step,
            b[step],
            common.iter().max().expect("non-empty")
        );
    }
}

/// `--eval-every`: how often to take a reading.
///
/// THE motivating case. Until `gf16_floor` was given its own cadence
/// (`TRIOS_GF16_FLOOR_EVERY`), this knob moved the weights.
#[test]
fn eval_every_does_not_move_the_weights() {
    assert_observation_independent(Case {
        param: "--eval-every",
        a: Variant {
            value: "50",
            eval_every: "50",
            seed: FIXED_SEED,
            env: &[],
            proof: "eval_every=50",
        },
        b: Variant {
            value: "100",
            eval_every: "100",
            seed: FIXED_SEED,
            env: &[],
            proof: "eval_every=100",
        },
    });
}

/// `TRIOS_EVAL_CHUNKS`: how much of the val stream one reading averages over.
///
/// `0` is full coverage (775 windows on a 100,000-token val stream) against the
/// default 40-window, 5.16% sample -- a nineteen-fold change in how hard the
/// run looks at itself, which must leave the artifact untouched.
#[test]
fn eval_chunks_does_not_move_the_weights() {
    assert_observation_independent(Case {
        param: "TRIOS_EVAL_CHUNKS",
        a: Variant {
            value: "40",
            eval_every: FIXED_EVAL_EVERY,
            seed: FIXED_SEED,
            env: &[("TRIOS_EVAL_CHUNKS", "40")],
            proof: "eval_chunks_target=40",
        },
        b: Variant {
            value: "0 (full coverage)",
            eval_every: FIXED_EVAL_EVERY,
            seed: FIXED_SEED,
            env: &[("TRIOS_EVAL_CHUNKS", "0")],
            proof: "eval_chunks_target=0",
        },
    });
}

/// `TRIOS_CANON_NAME`: what the run is CALLED.
///
/// Pure labelling -- it names the checkpoint directory and the ledger row -- so
/// it is the parameter with the least excuse to reach the weights, and
/// therefore the cheapest tripwire in the file.
#[test]
fn canon_name_does_not_move_the_weights() {
    assert_observation_independent(Case {
        param: "TRIOS_CANON_NAME",
        a: Variant {
            value: "IGLA-OBSINDEP-CANON-A",
            eval_every: FIXED_EVAL_EVERY,
            seed: FIXED_SEED,
            env: &[("TRIOS_CANON_NAME", "IGLA-OBSINDEP-CANON-A")],
            proof: "IGLA-OBSINDEP-CANON-A/200.bin",
        },
        b: Variant {
            value: "IGLA-OBSINDEP-CANON-B",
            eval_every: FIXED_EVAL_EVERY,
            seed: FIXED_SEED,
            env: &[("TRIOS_CANON_NAME", "IGLA-OBSINDEP-CANON-B")],
            proof: "IGLA-OBSINDEP-CANON-B/200.bin",
        },
    });
}

/// `TRIOS_CHECKPOINT_EVERY`: how often to SAVE.
///
/// The artifact cadence. It deliberately emits different step SETS -- 50 gives
/// {50,100,150,200}, 100 gives {100,200} -- which is why the comparison is over
/// the intersection. The steps both runs emit must agree byte for byte, or
/// snapshotting a run would have changed it.
#[test]
fn checkpoint_cadence_does_not_move_the_weights() {
    assert_observation_independent(Case {
        param: "TRIOS_CHECKPOINT_EVERY",
        a: Variant {
            value: "50",
            eval_every: FIXED_EVAL_EVERY,
            seed: FIXED_SEED,
            env: &[("TRIOS_CHECKPOINT_EVERY", "50")],
            proof: "checkpoint_every=50",
        },
        b: Variant {
            value: "100",
            eval_every: FIXED_EVAL_EVERY,
            seed: FIXED_SEED,
            env: &[("TRIOS_CHECKPOINT_EVERY", "100")],
            proof: "checkpoint_every=100",
        },
    });
}

/// The NEGATIVE control: a RECIPE parameter must move the weights.
///
/// Every assertion above is an equality, and an equality is only informative if
/// the comparison can come out unequal. If the checkpoint bytes did not depend
/// on the run at all -- a stubbed writer, a constant artifact, a `save` that
/// returns `Ok(())` without writing, which is the exact defect that produced
/// 1,851 experiments and zero artifacts -- every case in this file would pass
/// and prove nothing.
///
/// The seed is the cleanest recipe parameter to vary: 47 and 89 are both
/// allowed by Canon #93, and they must not produce the same weights.
#[test]
fn a_recipe_parameter_does_move_the_weights() {
    let case = Case {
        param: "--seed (negative control)",
        a: Variant {
            value: "47",
            eval_every: FIXED_EVAL_EVERY,
            seed: "47",
            env: &[],
            proof: "seed=47",
        },
        b: Variant {
            value: "89",
            eval_every: FIXED_EVAL_EVERY,
            seed: "89",
            env: &[],
            proof: "seed=89",
        },
    };
    let (a, b, common) = compare_variants(&case);
    for step in &common {
        assert_ne!(
            a[step], b[step],
            "seed 47 and seed 89 produced the SAME checkpoint at step {}: \
             sha256={}. The artifact does not depend on the run, so every \
             equality assertion in this file is vacuous.",
            step, a[step]
        );
    }
}

/// The control: same knobs twice must give the same bytes.
///
/// Without it, a green file above could mean "this trainer is deterministic in
/// nothing and every pair happened to match", or -- worse -- that the artifact
/// does not depend on the run at all. It also states the precondition every
/// other test in this file silently assumes.
#[test]
fn the_same_settings_twice_give_the_same_bytes() {
    assert_observation_independent(Case {
        param: "(control: nothing varied)",
        a: Variant {
            value: "run 1",
            eval_every: FIXED_EVAL_EVERY,
            seed: FIXED_SEED,
            env: &[],
            proof: "eval_every=100",
        },
        b: Variant {
            value: "run 2",
            eval_every: FIXED_EVAL_EVERY,
            seed: FIXED_SEED,
            env: &[],
            proof: "eval_every=100",
        },
    });
}
