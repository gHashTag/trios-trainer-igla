//! `ckpt_replay` - auditor-side spot-check verifier for checkpoint provenance.
//!
//! # What this binary is for
//!
//! "The development cycle is reproducible" is unfalsifiable if no one ever
//! re-runs it. This binary is the falsifier: it takes ONE checkpoint sidecar,
//! re-derives the artifact from the parameters that sidecar itself records, and
//! compares SHA-256 over the bytes on disk. It is the smallest instrument that
//! can turn a reproducibility claim into a measurement.
//!
//! It deliberately returns THREE kinds of answer, not one:
//!
//! ```text
//!   VERIFIED         exit 0   the record's own parameters reproduce its bytes
//!   MISMATCH         exit 1   they do not (both hashes are printed)
//!   ARTIFACT ALTERED exit 1   the .bin no longer hashes to its own record
//!   CORPUS MISMATCH  exit 1   the named corpus is not the corpus that was used
//!   TRAINER MISMATCH exit 1   the executable offered for the replay is not the
//!                             one the record names; NOTHING is executed
//!   INCOMPARABLE     exit 2   the record does not describe its own inputs, so
//!                             there is nothing to grade - not a pass, not a fail
//!   REFUSED          exit 3   the replay would cost more than the caller allowed
//!   ERROR            exit 4   the check itself could not be run
//! ```
//!
//! `TRAINER MISMATCH` closes the hole that made every other verdict decorative.
//! This binary used to hash the trainer it was about to execute and PRINT that
//! hash without comparing it to anything, because no schema had a field for it.
//! A twelve-line `/bin/sh` script that copies one pre-baked file into
//! `$TRIOS_CHECKPOINT_DIR` and performs no arithmetic was run against a genuine
//! `schema/3` record and graded `VERIFIED`, exit 0, "platform triple matches the
//! record". Only an honest vendor could produce a `MISMATCH`. Schema 4 records
//! `trainer.sha256`, and a binary that does not match it is refused BEFORE it
//! runs: an unnamed executable is not audited by watching what it does.
//!
//! `INCOMPARABLE` is the point. A `trios-checkpoint-record/1` sidecar carries no
//! `eval_every`, `steps_total` or `gf16_floor_every`; `gf16_floor()` rewrites
//! `embed`/`proj`/`lm_head`/`ctx` in place past the 70% mark, so a record that
//! cannot state that cadence cannot state its own recipe. Grading it PASS would
//! be a fabrication and grading it FAIL would be a slander. It gets neither.
//!
//! # Why a subprocess and not a library call
//!
//! The verifier re-executes the `trios-train` BINARY through
//! `std::process::Command`. It never calls `train_loop::` or `checkpoint::`.
//! That is what an external auditor can actually do - they have a binary, a
//! corpus and a JSON file - and it keeps the verdict independent of whatever
//! this crate's internals happen to be at the moment of the audit.
//!
//! The child environment is CLEARED (`env_clear`) down to `PATH`/`HOME`/`TMPDIR`
//! before the recorded parameters are set. `run_single()` reads at least
//! `HIDDEN_DIM`, `NUM_ATTN_LAYERS`, `GF16_ENABLED`, `TRIOS_GF16_DISABLE`,
//! `TRIOS_ATTN_SCALE`, `TRIOS_ATTN_SEQ`, `SEED` and `TRIOS_FORMAT_TYPE` from the
//! environment, and any of them silently changes the weights. An auditor's shell
//! must not be able to influence the verdict.
//!
//! # What the verdict is NOT
//!
//! A `VERIFIED` verdict is scoped to ONE platform triple. The same seed, the
//! same corpus and the same source produce DIFFERENT checkpoint hashes on macOS
//! and on Linux (see `docs/REPRODUCIBILITY-GRADING.md`). The sidecar schema does
//! not record a platform triple at all, so every success is printed with the
//! host it was obtained on and an explicit caveat that it does not generalise.
//!
//! ```bash
//! ckpt_replay --record checkpoints/my-run/300.json
//! ckpt_replay --record r/12000.json --max-steps 20000 --workdir /tmp/audit
//! ```

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use clap::Parser;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Prefix every accepted sidecar schema string starts with. The trailing
/// version is NOT compared: `/1`, `/2`, `/3` and anything later are all read by
/// FIELD PRESENCE, so a schema bump that only adds fields keeps working here.
const SCHEMA_PREFIX: &str = "trios-checkpoint-record/";

/// Everything the trainer needs in order to re-derive the artifact.
///
/// Order matters: the FIRST absent field is the one named in the
/// `INCOMPARABLE` line, and `eval_every` is checked before the other schema-2
/// additions because it is the sharpest of them - it is an observation
/// parameter that historically entered the recipe.
const REQUIRED_FIELDS: &[&str] = &[
    "seed",
    "step",
    // Without a recorded digest there is nothing to compare a replay against.
    // An empty string counts as absent, which is what `present()` enforces.
    "sha256",
    "hidden",
    "num_attn_layers",
    "optimizer",
    "fake_quant_format",
    "data_synthetic",
    "eval_every",
    "steps_total",
    "gf16_floor_every",
    "corpus.train.path",
    "corpus.train.bytes",
    "corpus.train.sha256",
    "corpus.val.path",
    "corpus.val.bytes",
    "corpus.val.sha256",
    // Schema 4. Deliberately LAST: a `schema/1` record is missing this one too,
    // and it must keep naming `eval_every` as its FIRST missing field so the
    // worked `INCOMPARABLE` examples in `docs/REPRODUCIBILITY-GRADING.md` stay
    // accurate. Without it there is no hash to check the executable against,
    // and an unnamed executable makes every other verdict decorative.
    "trainer.sha256",
];

/// Fields a schema might use to declare the platform the artifact was built on.
/// Schemas 1 and 2 have none of them; schema 3 carries `platform.*`. The
/// absence is itself reported, because a bit-for-bit verdict that does not name
/// a platform is not transferable to another one.
const PLATFORM_FIELDS: &[&str] = &[
    "platform.os",
    "platform.arch",
    "platform.libc",
    "platform.toolchain",
    "platform_triple",
    "target_triple",
    "host_triple",
    "os",
    "arch",
];

/// Recipe inputs that only schema 3 records. Used when present; their absence
/// is printed, never assumed away. They cannot make a `VERIFIED` verdict wrong
/// - bit-identity could not arise if the replay had used different values - but
/// they are the first suspects behind a `MISMATCH`.
const SCHEMA3_RECIPE_FIELDS: &[&str] = &["lr", "attn_scale", "attn_seq"];

const EXIT_MISMATCH: u8 = 1;
const EXIT_INCOMPARABLE: u8 = 2;
const EXIT_REFUSED: u8 = 3;
const EXIT_ERROR: u8 = 4;

#[derive(Parser, Debug)]
#[command(
    name = "ckpt_replay",
    about = "Re-derive a checkpoint from its own sidecar and grade the result"
)]
struct Args {
    /// Sidecar to grade (`{step}.json` written next to `{step}.bin`).
    #[arg(long)]
    record: PathBuf,

    /// Trainer binary to re-execute. Defaults to `trios-train` next to this
    /// binary, then to `target/release/trios-train`.
    #[arg(long)]
    trainer: Option<PathBuf>,

    /// Refuse to replay a run longer than this. A spot check that silently
    /// spends an hour is not a spot check.
    #[arg(long, default_value_t = 2000)]
    max_steps: u64,

    /// Scratch directory for the replay. Never written inside `checkpoints/`.
    /// Default: a fresh directory under the system temp dir.
    #[arg(long)]
    workdir: Option<PathBuf>,
}

/// Lowercase hex SHA-256, identical to `shasum -a 256`.
fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().fold(String::with_capacity(64), |mut acc, b| {
        use std::fmt::Write as _;
        let _ = write!(acc, "{b:02x}");
        acc
    })
}

fn sha256_file(path: &Path) -> Option<(String, u64)> {
    let raw = std::fs::read(path).ok()?;
    Some((sha256_hex(&raw), raw.len() as u64))
}

/// Follow a dotted path (`corpus.train.sha256`) through a JSON object.
fn dig<'a>(root: &'a Value, dotted: &str) -> Option<&'a Value> {
    let mut cur = root;
    for key in dotted.split('.') {
        cur = cur.get(key)?;
    }
    Some(cur)
}

/// A field counts as PRESENT only when it carries usable information: `null`
/// is absent, and an empty string is absent. A record that says `"sha256": ""`
/// tried and failed to describe its corpus; it must not grade as complete.
fn present(root: &Value, dotted: &str) -> bool {
    match dig(root, dotted) {
        None | Some(Value::Null) => false,
        Some(Value::String(s)) => !s.trim().is_empty(),
        Some(_) => true,
    }
}

fn as_u64(root: &Value, dotted: &str) -> Option<u64> {
    dig(root, dotted)?.as_u64()
}

fn as_str<'a>(root: &'a Value, dotted: &str) -> Option<&'a str> {
    dig(root, dotted)?.as_str()
}

fn as_f64(root: &Value, dotted: &str) -> Option<f64> {
    dig(root, dotted)?.as_f64()
}

/// Where the replayed trainer lives. Prefer an explicit `--trainer`, then a
/// sibling of this binary (the usual `target/release/` layout), then the
/// conventional relative path.
fn resolve_trainer(explicit: Option<PathBuf>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return if p.exists() { Some(p) } else { None };
    }
    if let Ok(me) = std::env::current_exe() {
        if let Some(dir) = me.parent() {
            let sibling = dir.join("trios-train");
            if sibling.exists() {
                return Some(sibling);
            }
        }
    }
    let conventional = PathBuf::from("target/release/trios-train");
    if conventional.exists() {
        return Some(conventional);
    }
    None
}

/// Ask the trainer to identify itself. Most binaries in this crate declare no
/// `--version`, so "unreported" is the expected answer and is printed as such
/// rather than being filled in with a guess.
fn trainer_version(trainer: &Path) -> String {
    match Command::new(trainer).arg("--version").output() {
        Ok(out) if out.status.success() => {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if s.is_empty() {
                "unreported".to_string()
            } else {
                s
            }
        }
        _ => "unreported (binary exposes no --version)".to_string(),
    }
}

/// Locate the `.bin` this record describes.
///
/// The recorded `path` is absolute and was written on the machine that trained,
/// so evidence directories that have been copied or mounted elsewhere will not
/// resolve. In that case the `{step}.bin` sitting next to the sidecar is used
/// instead; which file was graded is always printed.
fn locate_artifact(record_path: &Path, root: &Value, step: u64) -> Option<PathBuf> {
    if let Some(p) = as_str(root, "path") {
        let recorded = PathBuf::from(p);
        if recorded.is_file() {
            return Some(recorded);
        }
    }
    let sibling = record_path.with_file_name(format!("{step}.bin"));
    if sibling.is_file() {
        return Some(sibling);
    }
    None
}

fn make_workdir(explicit: Option<PathBuf>) -> std::io::Result<PathBuf> {
    let base = match explicit {
        Some(p) => p,
        None => {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            std::env::temp_dir().join(format!("ckpt_replay-{}-{}", std::process::id(), nanos))
        }
    };
    std::fs::create_dir_all(&base)?;
    Ok(base)
}

fn main() -> ExitCode {
    let args = Args::parse();

    let text = match std::fs::read_to_string(&args.record) {
        Ok(t) => t,
        Err(e) => {
            println!("ERROR: cannot read record {:?}: {e}", args.record);
            return ExitCode::from(EXIT_ERROR);
        }
    };
    let root: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            println!("ERROR: record {:?} is not JSON: {e}", args.record);
            return ExitCode::from(EXIT_ERROR);
        }
    };

    // (a) Schema. Accepted by prefix, read by field presence. An unknown
    // trailing version is fine; an unknown FORMAT is not.
    let schema = match as_str(&root, "schema") {
        Some(s) if s.starts_with(SCHEMA_PREFIX) => s.to_string(),
        Some(s) => {
            println!(
                "INCOMPARABLE: schema {s:?} is not a {SCHEMA_PREFIX}* record; \
                 this artifact cannot be graded from its own provenance record"
            );
            return ExitCode::from(EXIT_INCOMPARABLE);
        }
        None => {
            println!(
                "INCOMPARABLE: schema not recorded; \
                 this artifact cannot be graded from its own provenance record"
            );
            return ExitCode::from(EXIT_INCOMPARABLE);
        }
    };

    // (b) Does the record describe every input needed to re-derive the bytes?
    let missing: Vec<&str> = REQUIRED_FIELDS
        .iter()
        .copied()
        .filter(|f| !present(&root, f))
        .collect();
    if let Some(first) = missing.first() {
        println!(
            "INCOMPARABLE: {first} not recorded; \
             this artifact cannot be graded from its own provenance record"
        );
        println!("  record:  {:?} (schema {schema})", args.record);
        println!("  missing: {}", missing.join(", "));
        println!(
            "  note:    this is a verdict, not a failure. The run may have been \
             perfectly reproducible; its record simply does not say so."
        );
        return ExitCode::from(EXIT_INCOMPARABLE);
    }

    // Every field below is known present by the check above.
    let seed = as_u64(&root, "seed").unwrap_or_default();
    let step = as_u64(&root, "step").unwrap_or_default();
    let steps_total = as_u64(&root, "steps_total").unwrap_or_default();
    let hidden = as_u64(&root, "hidden").unwrap_or_default();
    let attn_layers = as_u64(&root, "num_attn_layers").unwrap_or_default();
    let eval_every = as_u64(&root, "eval_every").unwrap_or_default();
    let gf16_floor_every = as_u64(&root, "gf16_floor_every").unwrap_or_default();
    let optimizer = as_str(&root, "optimizer").unwrap_or("adamw").to_string();
    let fq_format = as_str(&root, "fake_quant_format").unwrap_or("f32").to_string();
    let data_synthetic = dig(&root, "data_synthetic")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let canon_name = as_str(&root, "canon_name").unwrap_or("ckpt-replay").to_string();
    let train_path = as_str(&root, "corpus.train.path").unwrap_or("").to_string();
    let val_path = as_str(&root, "corpus.val.path").unwrap_or("").to_string();

    // `lr`, `attn_scale` and `attn_seq` are inputs that schemas 1 and 2 do not
    // record even though they change the weights. They are read
    // opportunistically, so a schema-3 record is replayed with the values it
    // declares, and an older one is replayed at the trainer's defaults with
    // that stated in the verdict rather than papered over.
    let recorded_lr = as_f64(&root, "lr").or_else(|| as_f64(&root, "train_lr"));
    let recorded_attn_scale = as_f64(&root, "attn_scale").filter(|_| present(&root, "attn_scale"));
    let recorded_attn_seq = as_u64(&root, "attn_seq").filter(|_| present(&root, "attn_seq"));
    let unrecorded_recipe: Vec<&str> = SCHEMA3_RECIPE_FIELDS
        .iter()
        .copied()
        .filter(|f| !present(&root, f))
        .collect();

    // (c) Is the artifact still the artifact?
    let artifact = match locate_artifact(&args.record, &root, step) {
        Some(p) => p,
        None => {
            println!("ARTIFACT MISSING: no readable .bin for record {:?}", args.record);
            println!("  recorded path: {}", as_str(&root, "path").unwrap_or("(none)"));
            return ExitCode::from(EXIT_MISMATCH);
        }
    };
    let recorded_sha = as_str(&root, "sha256").unwrap_or("").to_string();
    let (artifact_sha, artifact_bytes) = match sha256_file(&artifact) {
        Some(v) => v,
        None => {
            println!("ARTIFACT MISSING: cannot read {artifact:?}");
            return ExitCode::from(EXIT_MISMATCH);
        }
    };
    if artifact_sha != recorded_sha {
        println!("ARTIFACT ALTERED: {artifact:?} no longer hashes to its record");
        println!("  recorded: {recorded_sha}");
        println!("  on disk:  {artifact_sha} ({artifact_bytes} bytes)");
        return ExitCode::from(EXIT_MISMATCH);
    }

    // (d) Is the corpus still the corpus? Hashed here, from the paths the
    // record names, with no help from this crate's own loader.
    for (label, path_field, sha_field, bytes_field) in [
        (
            "train",
            train_path.as_str(),
            "corpus.train.sha256",
            "corpus.train.bytes",
        ),
        (
            "val",
            val_path.as_str(),
            "corpus.val.sha256",
            "corpus.val.bytes",
        ),
    ] {
        let want_sha = as_str(&root, sha_field).unwrap_or("");
        let want_bytes = as_u64(&root, bytes_field).unwrap_or(0);
        match sha256_file(Path::new(path_field)) {
            None => {
                println!(
                    "CORPUS MISMATCH: {label} corpus {path_field:?} is not readable from {:?}",
                    std::env::current_dir().unwrap_or_default()
                );
                println!("  recorded: {want_sha} ({want_bytes} bytes)");
                return ExitCode::from(EXIT_MISMATCH);
            }
            Some((got_sha, got_bytes)) if got_sha != want_sha || got_bytes != want_bytes => {
                println!("CORPUS MISMATCH: {label} corpus {path_field:?}");
                println!("  recorded: {want_sha} ({want_bytes} bytes)");
                println!("  on disk:  {got_sha} ({got_bytes} bytes)");
                return ExitCode::from(EXIT_MISMATCH);
            }
            Some(_) => {}
        }
    }

    // (e) Cost. The replay must run the FULL configured length: a checkpoint at
    // step S of a T-step run depends on the schedule of that run, so `T` and
    // not `S` is what the caller is being asked to pay for.
    let replay_steps = step.max(steps_total);
    if replay_steps > args.max_steps {
        println!(
            "REFUSED: replaying {replay_steps} steps exceeds the --max-steps budget of {}; \
             re-run with --max-steps to authorise the cost",
            args.max_steps
        );
        return ExitCode::from(EXIT_REFUSED);
    }

    // (f) Replay.
    let trainer = match resolve_trainer(args.trainer.clone()) {
        Some(t) => t,
        None => {
            println!(
                "ERROR: no trainer binary found. Pass --trainer <path>, or build one with \
                 `cargo build --release --bin trios-train`."
            );
            return ExitCode::from(EXIT_ERROR);
        }
    };
    let trainer_sha = sha256_file(&trainer)
        .map(|(s, _)| s)
        .unwrap_or_else(|| "unreadable".to_string());
    let version = trainer_version(&trainer);

    // (f.1) Is the executor the executor? Checked BEFORE the subprocess is
    // spawned, and the check ends the run rather than annotating it: a binary
    // the record does not name must not be given the chance to write the file
    // it is about to be graded on. `trainer.sha256` is known present by the
    // REQUIRED_FIELDS check above, so an absent field already returned
    // INCOMPARABLE and never reaches here.
    let recorded_trainer_sha = as_str(&root, "trainer.sha256").unwrap_or("").to_string();
    if trainer_sha != recorded_trainer_sha {
        println!("TRAINER MISMATCH: {trainer:?} is not the executable this record names");
        println!("  recorded: {recorded_trainer_sha}");
        println!("  on disk:  {trainer_sha}");
        println!("  resolved: {}", trainer.display());
        println!(
            "  provenance: {}",
            as_str(&root, "trainer.provenance").unwrap_or("(not recorded)")
        );
        println!(
            "  recorded path: {}",
            as_str(&root, "trainer.path").unwrap_or("(none)")
        );
        println!(
            "  note:    nothing was executed. Pass --trainer <path> pointing at the \
             binary the record names, or rebuild it; a replay driven by an unnamed \
             executable grades the executable, not the claim."
        );
        return ExitCode::from(EXIT_MISMATCH);
    }

    let workdir = match make_workdir(args.workdir.clone()) {
        Ok(d) => d,
        Err(e) => {
            println!("ERROR: cannot create workdir: {e}");
            return ExitCode::from(EXIT_ERROR);
        }
    };
    let ckpt_dir = workdir.join("checkpoints");

    let mut cmd = Command::new(&trainer);
    // A scrubbed environment. `run_single()` reads arch and format knobs
    // straight from the environment; if the auditor's shell can reach them,
    // the auditor's shell can change the verdict.
    cmd.env_clear();
    for keep in ["PATH", "HOME", "TMPDIR"] {
        if let Ok(v) = std::env::var(keep) {
            cmd.env(keep, v);
        }
    }
    cmd.env("TRIOS_CHECKPOINT_DIR", &ckpt_dir)
        .env("TRIOS_CANON_NAME", &canon_name)
        .env("TRIOS_GF16_FLOOR_EVERY", gf16_floor_every.to_string())
        .env("TRIOS_FORMAT_TYPE", &fq_format)
        .env("TRINITY_AUTOMIGRATE", "0");
    if data_synthetic {
        cmd.env("TRIOS_ALLOW_SYNTHETIC_DATA", "1");
    }
    if step != steps_total {
        // The artifact is an intermediate checkpoint; ask for that cadence.
        cmd.env("TRIOS_CHECKPOINT_EVERY", step.to_string());
    }
    cmd.arg("--seed")
        .arg(seed.to_string())
        .arg("--steps")
        .arg(steps_total.to_string())
        .arg("--hidden")
        .arg(hidden.to_string())
        .arg("--attn-layers")
        .arg(attn_layers.to_string())
        .arg("--eval-every")
        .arg(eval_every.to_string())
        .arg("--optimizer")
        .arg(&optimizer)
        .arg("--train-data")
        .arg(&train_path)
        .arg("--val-data")
        .arg(&val_path);
    if let Some(lr) = recorded_lr {
        cmd.arg("--lr").arg(format!("{lr}"));
    }
    if let Some(scale) = recorded_attn_scale {
        cmd.env("TRIOS_ATTN_SCALE", format!("{scale}"));
    }
    if let Some(seq) = recorded_attn_seq {
        cmd.env("TRIOS_ATTN_SEQ", seq.to_string());
    }

    println!("replaying {replay_steps} steps with {trainer:?} ...");
    let status = match cmd.status() {
        Ok(s) => s,
        Err(e) => {
            println!("ERROR: could not execute {trainer:?}: {e}");
            return ExitCode::from(EXIT_ERROR);
        }
    };
    if !status.success() {
        println!("ERROR: trainer exited with {status}; nothing was graded");
        return ExitCode::from(EXIT_ERROR);
    }

    // `sanitize_run_name` maps anything outside [A-Za-z0-9._-] to '_'; mirrored
    // here rather than imported, because this binary is a black-box auditor.
    let dir_component: String = canon_name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let replayed = ckpt_dir.join(&dir_component).join(format!("{step}.bin"));
    let (replay_sha, replay_bytes) = match sha256_file(&replayed) {
        Some(v) => v,
        None => {
            println!("MISMATCH: the replay produced no artifact at {replayed:?}");
            println!(
                "  a checkpoint at step {step} is only written when \
                 step == steps_total or step % eval_every == 0"
            );
            return ExitCode::from(EXIT_MISMATCH);
        }
    };

    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let declared: Vec<String> = PLATFORM_FIELDS
        .iter()
        .filter(|f| present(&root, f))
        .map(|f| {
            let leaf = f.rsplit('.').next().unwrap_or(f);
            let val = as_str(&root, f)
                .map(|s| s.to_string())
                .or_else(|| dig(&root, f).map(|v| v.to_string()))
                .unwrap_or_default();
            format!("{leaf}={val}")
        })
        .collect();
    // Only os/arch are comparable here: `libc` and `toolchain` describe the
    // build of the trainer that WROTE the record, which this process cannot
    // observe from the outside.
    let declared_os = as_str(&root, "platform.os").or_else(|| as_str(&root, "os"));
    let declared_arch = as_str(&root, "platform.arch").or_else(|| as_str(&root, "arch"));
    let platform_conflict = declared_os.is_some_and(|d| d != os)
        || declared_arch.is_some_and(|d| d != arch);

    println!("--- ckpt_replay verdict ---");
    println!("record:        {:?} (schema {schema})", args.record);
    println!("artifact:      {artifact:?}");
    println!("recipe:        seed={seed} steps_total={steps_total} step={step} hidden={hidden} attn_layers={attn_layers}");
    println!("               optimizer={optimizer} fake_quant_format={fq_format} gf16_floor_every={gf16_floor_every} eval_every={eval_every} data_synthetic={data_synthetic}");
    match recorded_lr {
        Some(lr) => println!("lr:            {lr} (from the record)"),
        None => println!("lr:            NOT RECORDED by this schema; replay used the trainer default"),
    }
    if !unrecorded_recipe.is_empty() {
        println!(
            "unrecorded:    {} - replayed at the trainer's defaults",
            unrecorded_recipe.join(", ")
        );
    }
    if let Some(src) = as_str(&root, "source_sha256") {
        println!(
            "source_sha256: {src} (recorded only; this binary does NOT re-derive it - \
             the caveat applies to this field alone, trainer.sha256 below WAS re-hashed)"
        );
    }
    println!("trainer:       {trainer:?}");
    println!("               sha256={trainer_sha} version={version}");
    println!("               re-hashed here and it MATCHES the record's trainer.sha256");
    if let Some(vocab) = as_u64(&root, "vocab") {
        println!("vocab:         {vocab} (alphabet the corpus was folded onto)");
    } else {
        println!("vocab:         NOT RECORDED by this schema; the alphabet behind the BPB is unstated");
    }
    println!("workdir:       {workdir:?}");
    println!("recorded sha:  {recorded_sha} ({artifact_bytes} bytes)");
    println!("replay sha:    {replay_sha} ({replay_bytes} bytes)");
    println!("host:          {os}/{arch}");
    println!(
        "record says:   {}",
        if declared.is_empty() {
            "(no platform fields)".to_string()
        } else {
            declared.join(" ")
        }
    );

    if replay_sha == recorded_sha {
        println!("VERIFIED on {os}/{arch}");
        if declared.is_empty() {
            println!(
                "platform triple was NOT declared by the record; \
                 this verdict is valid only on this host"
            );
        } else if platform_conflict {
            // Bits matched anyway, which is a stronger result than the record
            // claimed; it is reported as the anomaly it is, not folded away.
            println!(
                "platform triple DIFFERS from the record ({}) yet the bytes matched on {os}/{arch}",
                declared.join(" ")
            );
        } else {
            println!("platform triple matches the record");
        }
        ExitCode::SUCCESS
    } else {
        println!("MISMATCH on {os}/{arch}");
        println!("  recorded: {recorded_sha}");
        println!("  replayed: {replay_sha}");
        if declared.is_empty() {
            println!(
                "platform triple was NOT declared by the record, so this MISMATCH does not \
                 distinguish a bad record from a different host"
            );
        } else if platform_conflict {
            println!(
                "the record was produced on {} and this host is {os}/{arch}; \
                 a cross-platform MISMATCH is expected, not evidence of a bad record",
                declared.join(" ")
            );
        }
        if !unrecorded_recipe.is_empty() {
            println!(
                "the record does not state {}; any of them could explain this MISMATCH",
                unrecorded_recipe.join(", ")
            );
        }
        ExitCode::from(EXIT_MISMATCH)
    }
}
