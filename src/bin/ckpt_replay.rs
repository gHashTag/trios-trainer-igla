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
//!   VERIFIED          exit 0   the record's own parameters reproduce its bytes,
//!                              and the metric it certifies, when it states one
//!   MISMATCH          exit 1   they do not (both hashes are printed, together
//!                              with every header field the record fails to state)
//!   ARTIFACT ALTERED  exit 1   the .bin no longer hashes to its own record, or
//!                              it is not the LENGTH the record declares, or
//!                              its bytes do not obey the TRIOSCKP format they
//!                              claim (bad magic, truncation, trailing garbage)
//!   PUBLISHED ARTIFACT
//!             MISSING exit 1   `--integrity-only` found no `.bin` beside the
//!                              record; the record's own `path` is NOT followed
//!   CORPUS MISMATCH   exit 1   the named corpus is not the corpus that was used
//!   TRAINER MISMATCH  exit 1   the executable offered for the replay is not the
//!                              one the record names; NOTHING is executed
//!   INCOMPARABLE      exit 2   the record does not describe its own inputs, or
//!                              it contradicts the artifact's own header, so
//!                              there is nothing to grade - not a pass, not a fail
//!   REFUSED           exit 3   the replay would cost more than the caller allowed
//!   ERROR             exit 4   the check itself could not be run
//!   BPB MISMATCH      exit 5   the WEIGHTS re-derived exactly and the NUMBER the
//!                              record certifies did not
//!   BPB NOT CONFIRMED exit 5   the record certifies a number and the replay
//!                              stated none, so the claim is unconfirmed
//!   GRID MISMATCH     exit 5   the replay did not run on the eval grid the
//!                              record declares, so its number is a reading of
//!                              a different quantity and is not graded
//!   L2 PASS / L3 FAIL exit 6   the bytes differ and the metric agrees within
//!                              the stated tolerance: an honest second
//!                              laboratory, not a forgery
//!   SKIPPED           exit 7   `--integrity-only` was asked to grade a JSON
//!                              that is not a checkpoint record at all
//! ```
//!
//! # `--integrity-only`: a weaker question, asked honestly
//!
//! Re-deriving a 12 000-step record costs hours, so no gate can afford to ask
//! the full question of every record in `evidence/` on every push. The flag
//! asks the one question that IS affordable: do the bytes on disk today still
//! hash to the digest published beside them? Nothing is re-derived and no
//! trainer is executed - `--integrity-only` returns before the corpus is even
//! looked at.
//!
//! That is deliberately a much weaker claim than `VERIFIED`, and the verdict
//! line says so in the verdict itself rather than in a footnote somewhere else:
//! an unaltered forgery passes this check, because a forgery hashes to its own
//! record too. What it catches is an evidence tree that has rotted, been
//! partially updated, or had one `.bin` swapped under a record that still names
//! the old digest - which is exactly what happens to a published corpus of
//! artifacts nobody re-hashes.
//!
//! The artifact is resolved DIFFERENTLY here, and the difference is load
//! bearing. The replay path prefers the `path` the record states, because that
//! is the file the record is about. For published evidence that preference is
//! wrong twice over: the recorded path was written on the training machine, and
//! when it is RELATIVE it resolves against the auditor's current directory and
//! can land on an unrelated local file. Measured on this repository:
//! `evidence/xarch-run-30767491098/12000.json` records
//! `path: checkpoints/r4-docs-repro/12000.bin`, and a local checkout has a file
//! at that path - the `aarch64` macOS artifact `8a86fe69...`, while the record
//! is the `x86_64` Linux one `bb14ab18...`. Grading the record against whatever
//! happens to sit at that relative path reports `ARTIFACT ALTERED` about a
//! published file that was never touched. So the copy PUBLISHED BESIDE THE
//! RECORD is the ONLY candidate (`<record stem>.bin`, then `{step}.bin`), the
//! recorded path is never followed here, and the file actually hashed is always
//! printed.
//!
//! There is no last resort on purpose. A `.bin` deleted from the evidence tree
//! under a record that stays behind is exactly the rot this mode exists to
//! catch, and a fallback to the recorded `path` grades that record against a
//! file that is still whole somewhere else - reporting `INTEGRITY OK`, exit 0,
//! about an artifact that is gone. Sibling-only resolution turns the deletion
//! into `PUBLISHED ARTIFACT MISSING`, exit 1, which is what an audit wiring
//! this binary to an exit code needs to see.
//!
//! # The eval grid is part of the recipe, and it is restored
//!
//! `final_val_bpb` is a reading taken on a GRID: `eval_chunks` windows of
//! `eval_seq` tokens each, selected by `train_loop::eval_plan`. The grid is an
//! observation parameter - it never touches the weights - but it decides the
//! number completely, and `TRIOS_EVAL_CHUNKS` moves it.
//!
//! This binary clears the child environment, and until now it never re-exported
//! that variable, so every replay evaluated at `EVAL_CHUNKS_DEFAULT = 40`
//! whatever the record declared. A record written with `TRIOS_EVAL_CHUNKS=16`
//! therefore replayed BYTE-IDENTICALLY and was graded `BPB MISMATCH`, exit 5,
//! under a note asserting that byte-identical weights cannot measure a different
//! metric on this host. They can, and that is precisely what had happened; the
//! child's own `[ckpt] ... eval_chunks=40` line was on the same screen.
//! `docs/EVAL-UNCERTAINTY.md` publishes a reproduction recipe that sets exactly
//! this variable, so the repository's own documented procedure produced records
//! its own falsifier called liars.
//!
//! The replay is now driven to the grid the record declares (schema 6:
//! `eval_chunks`, `eval_tokens`, `eval_seq`). The value exported is the ACHIEVED
//! chunk count the record carries, which reproduces the identical grid: for a
//! plan achieved at `C` chunks, `eval_plan(len, C)` returns that same plan,
//! including the `target = 0` (full coverage) case, because `max_start >= C*seq`
//! can only hold when `seq` divides `max_start`. A record from a schema that
//! predates the knob (1 to 5) leaves the variable UNSET, which is correct: those
//! runs provably used the hardcoded 40.
//!
//! The replay's own sidecar is then read back and its grid compared against the
//! record's. A disagreement is `GRID MISMATCH` - the two numbers are readings of
//! different quantities - and never a `BPB MISMATCH`, which is an accusation.
//!
//! # L2 and L3: an honest second laboratory is not a forger
//!
//! `docs/REPRODUCIBILITY-GRADING.md` defines a ladder: L2 is "the metric
//! reproduces within a stated tolerance", L3 is "the bytes reproduce on a
//! declared platform triple". This binary used to implement only L3, so its only
//! two outcomes were "bit-identical" and "indistinguishable from fraud". The one
//! cross-architecture run that has been performed (CI run 30767491098,
//! `x86_64` Linux against the `aarch64` macOS reference) was a faithful
//! execution of the same source with the same locked dependency graph and a
//! corpus verified against `data/MANIFEST.sha256`, and it earned the same verdict
//! a fabricated checkpoint earns. An instrument with no false-positive control
//! has no evidentiary value.
//!
//! So when the bytes differ and the METRIC agrees within
//! [`BPB_TOLERANCE_DEFAULT`] (overridable with `--bpb-tolerance`, and printed in
//! the verdict either way), the verdict is `L2 PASS ... / L3 FAIL ...` and exit
//! 6. A forger whose metric ALSO disagrees still gets `MISMATCH`, exit 1.
//!
//! # Why `BPB MISMATCH` is 5 and not 4
//!
//! The work item that asked for this verdict asked for "a NEW code 4, distinct
//! from 1 (weights) and 2 (INCOMPARABLE)". Exit 4 was already `ERROR` here, and
//! `.github/workflows/ckpt-replay-audit.yml` and `cross-arch-repro.yml` both
//! switch on it by name ("the check could not be RUN, so there is no verdict at
//! all to census"). Giving a *verdict* the code reserved for *no verdict* is the
//! same category error this binary exists to prevent, so the new verdict took
//! the next free number instead. Both workflows treat an unrecognised code as a
//! failure, so a `BPB MISMATCH` fails CI today without either file being edited.
//!
//! # The number is graded, not just the bytes
//!
//! `README.md` calls a checkpoint record "the record the repository's own
//! verifier grades" and prints `final_val_bpb` as a row of that table. Until
//! this was written, that sentence was false in the only way that matters: a
//! sidecar whose `final_val_bpb` had been replaced by hand with the retracted
//! 1.5492 still printed `VERIFIED`, exit 0, one screen below the replay's own
//! honest `DONE: seed=47 bpb=2.9744 ...`. `grep -n bpb src/bin/ckpt_replay.rs`
//! returned two hits, both in a comment: no code path read any BPB at all. The
//! trainer's stdout was inherited and discarded by `cmd.status()`, so the number
//! the auditor needed scrolled past the auditor's eyes ungraded.
//!
//! The child's stdout is now PIPED, echoed line by line as it arrives (a
//! 12 000-step replay must stay observable; `.output()` would hold it all back
//! until exit), and the LAST line beginning `DONE:` is kept. Its `bpb=` token is
//! compared against the record's `final_val_bpb`.
//!
//! Three limits, stated rather than hidden:
//!
//!   * **Precision.** `trios-train` renders that token with `{:.4}`, so the
//!     comparison is `format!("{:.4}", recorded)` against the captured token
//!     VERBATIM - agreement to FOUR DECIMAL PLACES and no further. Two runs
//!     differing by 5e-5 bpb are indistinguishable here. Every verdict line says
//!     "(4 dp)" so no reader can mistake it for bit-identity of the metric; the
//!     bit-identity claim is the sha256 line above it, which is unaffected.
//!   * **Intermediate checkpoints are NOT graded.** A sidecar at step S of a
//!     T-step run records the metric AT STEP S, while `DONE:` reports the metric
//!     at step T. Those are different quantities, and comparing them would
//!     manufacture a failure out of an honest record. When `step != steps_total`
//!     the metric is reported as not graded, with the cause named.
//!   * **`final_val_bpb` is NOT in `REQUIRED_FIELDS`.** Adding it there would
//!     retroactively re-grade every archived `schema/1` and `schema/2` record as
//!     ungradeable for a reason that has nothing to do with why they already
//!     are, and that is a separate decision. A record that does not state the
//!     number gets `WEIGHTS VERIFIED; final_val_bpb NOT GRADED (not recorded)`
//!     and exit 0: the bytes were still re-derived, and saying so is not a lie.
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
//! # Two evidence documents, not one
//!
//! A checkpoint ships as a PAIR: the unhashed sidecar, and the hashed 152-byte
//! TRIOSCKP header inside the `.bin` itself (specified in `src/checkpoint.rs`
//! and independently implemented in `interop/triosckp_reader.py`). This binary
//! used to open only the sidecar and read the `.bin` for nothing but its
//! SHA-256, which cost it two things:
//!
//!   * Thirteen scalars are written twice - `seed`, `step`, `hidden`,
//!     `d_model`, `num_attn_layers`, `format_version`, `vocab`,
//!     `data_synthetic`, `optimizer`, `fake_quant_format`, `lr`, `attn_scale`,
//!     `attn_seq` - and any disagreement between the two copies went entirely
//!     unreported. Two evidence documents contradicting each other is a
//!     stronger finding than either one alone; it is now `INCOMPARABLE`. The
//!     comparison is driven by field PRESENCE, so a schema that starts writing
//!     a fourteenth shared scalar is cross-checked from its first record.
//!   * `gf16_enabled` is byte 124 of the hashed header. It is the master switch
//!     on `gf16_floor()`, the function that rewrites `embed`/`proj`/`lm_head`/
//!     `ctx` in place, and the trainer's default is ENABLED. A run trained with
//!     `TRIOS_GF16_DISABLE=1` is perfectly reproducible - repeating it
//!     reproduces its bytes exactly - yet no sidecar schema up to and including
//!     `trios-checkpoint-record/4` could say so, and this binary clears the
//!     child environment, so the replay silently ran with the floor ON. The
//!     result was a `MISMATCH` with a matching trainer, a matching corpus and a
//!     matching platform: a false FAIL whose only available reading was that
//!     the vendor's record is a lie. Measured on two 200-step seed-47 runs
//!     differing in nothing but that variable: floor on -> `d2fceff3...`,
//!     val_bpb 3.7251; floor off -> `76d2cae9...`, val_bpb 3.6087. That is a
//!     0.116 bpb effect, larger than the eval-cadence effect. The artifact now
//!     gets read, and a `gf16_enabled` that is not 1 produces a cause-naming
//!     refusal instead of a false FAIL.
//!
//! `trios-checkpoint-record/5` began recording `gf16_enabled` in the sidecar
//! while this was being written. That changes the CAUSE but not the VERDICT: a
//! record that states the flag is cross-checked against byte 124 like any other
//! shared scalar, and the refusal now reports that the field is recorded but
//! that this binary does not yet drive the replay from it. Grading a run whose
//! replay is known to use the opposite setting would still be a fabrication.
//! Driving the child from the RECORD's copy of the field - never the header's -
//! is what would turn this refusal into a gradeable verdict.
//!
//! # Why the header is READ but never OBEYED
//!
//! Nothing decoded from the artifact is ever fed back into the replay. The
//! header is cross-checked against the record and reported; it is never a
//! source of replay parameters. Setting `TRIOS_GF16_DISABLE` from byte 124
//! would make a forged header dictate the terms of its own verification, and a
//! run that supplies the parameters it is graded against is not being graded.
//! An artifact that disagrees with its record is refused, not accommodated.
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

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use clap::Parser;
use serde_json::Value;
use sha2::{Digest, Sha256};
use trios_trainer::provenance_seal::{provenance_seal, sealed_field_names};

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
/// is printed, never assumed away. They cannot make a `VERIFIED` verdict
/// wrong - bit-identity could not arise if the replay had used different
/// values - but they are the first suspects behind a `MISMATCH`.
const SCHEMA3_RECIPE_FIELDS: &[&str] = &["lr", "attn_scale", "attn_seq"];

/// Every field of the TRIOSCKP header, paired with the sidecar paths that
/// state the SAME quantity. An empty path list means no sidecar schema carries
/// the field at all, so the record cannot state it however new it is.
///
/// This table drives the `MISMATCH` report: a failing verdict that names no
/// suspect is indistinguishable from an accusation. `header_len` and
/// `tensor_count` are omitted because format version 1 fixes them at 152 and
/// 19; they are validated, not reported as recipe.
const HEADER_FIELD_MAP: &[(&str, &[&str])] = &[
    ("format_version", &["format_version"]),
    ("vocab", &["vocab"]),
    ("dim", &[]),
    ("num_ctx", &[]),
    ("hidden", &["hidden"]),
    ("d_model", &["d_model"]),
    ("num_heads", &[]),
    ("attn_cfg_seq_len", &[]),
    ("num_attn_layers", &["num_attn_layers"]),
    ("ngram", &[]),
    ("qk_gain", &[]),
    ("attn_cfg_lr", &[]),
    ("train_lr", &["lr", "train_lr"]),
    ("attn_scale", &["attn_scale"]),
    ("attn_seq", &["attn_seq"]),
    ("ctx_weights", &[]),
    ("seed", &["seed"]),
    ("step", &["step"]),
    // Unstated by every schema up to /4, carried by /5. Listed with its path so
    // that a record which does state it stops being reported as silent about it.
    ("gf16_enabled", &["gf16_enabled"]),
    ("data_synthetic", &["data_synthetic"]),
    ("optimizer", &["optimizer"]),
    ("fake_quant_format", &["fake_quant_format"]),
];

/// Magic prefix of every checkpoint container.
const CKPT_MAGIC: &[u8; 8] = b"TRIOSCKP";

/// The only container version whose field offsets this binary knows.
const CKPT_FORMAT_VERSION: u32 = 1;

/// Absolute byte length of the fixed header (= offset of the tensor directory).
const CKPT_HEADER_LEN: usize = 152;

/// Number of tensors in the canonical order. Fixed for format version 1.
const CKPT_TENSOR_COUNT: usize = 19;

/// Absolute byte offset at which the payload starts.
const CKPT_PAYLOAD_OFFSET: usize = CKPT_HEADER_LEN + CKPT_TENSOR_COUNT * 8;

/// Number of context weights stored at offset 84.
const CKPT_CTX_WEIGHTS: usize = 6;

/// Size in bytes of one payload element (f32).
const CKPT_ELEMENT_SIZE: u128 = 4;

const EXIT_MISMATCH: u8 = 1;
const EXIT_INCOMPARABLE: u8 = 2;
const EXIT_REFUSED: u8 = 3;
const EXIT_ERROR: u8 = 4;
/// The weights re-derived and the metric did not, or the record certifies a
/// metric the replay never stated. Deliberately NOT 4: see "Why `BPB MISMATCH`
/// is 5 and not 4" at the top of this file.
const EXIT_BPB_MISMATCH: u8 = 5;
/// The bytes did NOT re-derive and the metric DID, within the stated tolerance.
/// A distinct code because it is a distinct finding: L2 PASS / L3 FAIL is what
/// an honest laboratory on another platform looks like, and collapsing it into
/// `MISMATCH` is what left this binary unable to tell one from a forgery.
const EXIT_L2_PASS_L3_FAIL: u8 = 6;
/// `--integrity-only` was pointed at a JSON that is not a checkpoint record.
/// A distinct code, not `INCOMPARABLE`: exit 2 means "this IS a record and it
/// cannot state its own recipe", which is a finding about a checkpoint. A
/// sidecar-shaped file that describes something else entirely (the ISA probe in
/// `evidence/xarch-local-isa/probe.json` is a `trios-local-isa-probe/1`
/// document with no artifact of its own) is not a finding at all, and a census
/// that cannot tell the two apart cannot report either honestly.
const EXIT_SKIPPED: u8 = 7;

/// Default `|recorded - replayed|` a metric may differ by and still count as
/// reproduced (the L2 rung of `docs/REPRODUCIBILITY-GRADING.md`).
///
/// Derived from the only paired cross-laboratory measurement this repository
/// has, both sides run on the SAME eval grid: the 12 000-step aarch64 macOS /
/// x86_64 Linux table in that document reports step-wise deltas of `+0.0002`
/// (step 1 000), `-0.0035` (3 000), `+0.0030` (8 000) and `+0.0030` (12 000).
/// The largest observed magnitude is 0.0035, so the default is the next round
/// number above it. It is deliberately NOT tighter than the evidence and NOT
/// generous enough to admit a second reading of the metric: the retracted 1.5492
/// sits 1.08 bpb from the value it was attached to.
const BPB_TOLERANCE_DEFAULT: f64 = 0.005;

/// Prefix of the trainer's final summary line, matched at the start of a
/// trimmed line so a corpus that happens to contain the word cannot forge one.
const DONE_PREFIX: &str = "DONE:";

/// Decimal places `trios-train` renders `bpb=` with (`{:.4}` in
/// `src/bin/trios-train.rs`). This is the RESOLUTION of the metric check and
/// therefore its limit; it is printed in every verdict line that mentions BPB.
const DONE_BPB_DECIMALS: usize = 4;

#[derive(Parser, Debug)]
#[command(
    name = "ckpt_replay",
    about = "Re-derive a checkpoint from its own sidecar and grade the result",
    long_about = "Re-derive a checkpoint from its own sidecar and grade the result.\n\n\
                  Both the BYTES and, when the record states one, the METRIC are graded:\n\
                  the replayed trainer's own `DONE: ... bpb=` token is compared against the\n\
                  record's `final_val_bpb` to 4 decimal places (the precision `trios-train`\n\
                  prints), and a disagreement is `BPB MISMATCH`.\n\n\
                  A BPB verdict is scoped to the EVAL GRID the record declares \
                  (eval_chunks / eval_seq, schema 6): the replay is driven to that \
                  grid, the grid the replay actually used is read back from its own \
                  sidecar, and a disagreement is GRID MISMATCH rather than a claim \
                  that the record lied about its number.\n\n\
                  Exit codes (also listed in interop/SPEC-SNAPSHOT.txt):\n  \
                    exit 0  VERIFIED - the bytes re-derived, and the metric was \
                  confirmed when the record states one\n  \
                    exit 1  MISMATCH / ARTIFACT ALTERED / ARTIFACT MISSING / \
                  PUBLISHED ARTIFACT MISSING / CORPUS MISMATCH / TRAINER MISMATCH\n  \
                    exit 2  INCOMPARABLE - the record cannot state its own recipe, \
                  or it contradicts its own artifact\n  \
                    exit 3  REFUSED - over the --max-steps budget; nothing was executed\n  \
                    exit 4  ERROR - the check could not be run, so there is no verdict\n  \
                    exit 5  BPB MISMATCH / BPB NOT CONFIRMED / GRID MISMATCH - the \
                  bytes matched and the number did not, or was not gradeable\n  \
                    exit 6  L2 PASS / L3 FAIL - the bytes differ and the metric agrees \
                  within --bpb-tolerance: an honest second laboratory, not a forgery\n  \
                    exit 7  SKIPPED - --integrity-only was pointed at a JSON that is \
                  not a checkpoint record"
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

    /// Metric tolerance for the L2 rung, in bpb. Used ONLY when the bytes
    /// differ; a byte-identical replay is still graded at the four decimals the
    /// trainer prints. The value in force is printed in the verdict.
    #[arg(long, default_value_t = BPB_TOLERANCE_DEFAULT)]
    bpb_tolerance: f64,

    /// Grade the artifact's INTEGRITY only: re-hash the published .bin and
    /// compare it with the digest the record itself carries. Nothing is
    /// re-derived, no trainer is executed, no corpus is read.
    ///
    /// INTEGRITY IS NOT REPRODUCTION. A pass here proves exactly one thing: the
    /// bytes on disk today are the bytes that were published under this record.
    /// It says NOTHING about whether the recipe the record states produces those
    /// bytes - that is the question this binary answers WITHOUT this flag, and
    /// answering it costs a training run (hours, for the 12 000-step records).
    /// An unaltered forgery passes this mode, because a forgery hashes to its
    /// own record too. Use it to keep a large published evidence tree honest at
    /// milliseconds per record; never read a pass as a reproducibility claim.
    ///
    /// ONLY the copy published BESIDE the record is graded (`<stem>.bin`, then
    /// `{step}.bin`). The recorded `path` is never followed here, because it was
    /// written on the training machine and, when relative, resolves against the
    /// auditor's own directory - and because following it would let a DELETED
    /// published artifact pass by being graded against a copy elsewhere. The
    /// file actually hashed is always printed.
    ///
    /// Exits: 0 the bytes match; 1 ARTIFACT ALTERED, ARTIFACT MISSING or
    /// PUBLISHED ARTIFACT MISSING; 2 the
    /// record carries no sha256 to compare against; 7 SKIPPED, the JSON is not
    /// a checkpoint record at all.
    #[arg(long)]
    integrity_only: bool,

    /// Print the record's PROVENANCE SEAL and exit 0. Nothing is verified.
    ///
    /// The seal is `sha256:` over the record's declaration half - the platform
    /// block, the corpus and trainer digests, `git_sha`, `git_dirty`,
    /// `steps_total`, `eval_every`, `final_val_bpb` and `schema` - which no
    /// container byte can confirm and which every other mode of this binary
    /// therefore quotes without checking. The exact rule, and the fields it
    /// does NOT cover, are in `src/provenance_seal.rs` and
    /// `docs/PROVENANCE-BINDING.md`.
    ///
    /// A seal is NOT a signature. Publishing it moves a forged declaration
    /// from undetectable to detectable-by-anyone-holding-the-published-digest;
    /// it does not stop an adversary who controls the publication channel too.
    #[arg(long)]
    provenance_seal: bool,

    /// Require the record's declaration to seal to this digest
    /// (`sha256:<64 hex>`, as printed by --provenance-seal). A disagreement is
    /// `SEAL MISMATCH`, exit 1, and nothing else is run.
    ///
    /// This is the ONLY way any mode of this binary states an opinion about
    /// the platform triple, the toolchain string or `final_val_bpb`. Without
    /// it, every verdict carries the UNAUTHENTICATED DECLARATION line instead.
    #[arg(long, value_name = "sha256:...")]
    expect_provenance_seal: Option<String>,
}

/// Say, next to every verdict, what the verdict did not cover.
///
/// The declaration half of a record is free text written by the party being
/// audited. `INTEGRITY OK` and `VERIFIED` are both statements about BYTES, and
/// printing either one alone is what let a sidecar rewritten to claim the
/// wrong architecture, a fabricated rustc and the retracted 1.5492 pass both
/// of this repository's verifiers (docs/PROVENANCE-BINDING.md). So the scope
/// is printed unconditionally, in the same breath as the verdict, and it names
/// the fields rather than gesturing at "provenance".
fn print_declaration_scope(root: &Value, authenticated_against: Option<&str>) {
    let seal = match provenance_seal(root) {
        Ok(s) => s,
        Err(e) => {
            println!("DECLARATION NOT SEALABLE: {e}");
            return;
        }
    };
    let names = sealed_field_names(root).unwrap_or_default();
    match authenticated_against {
        Some(expected) => println!(
            "DECLARATION SEAL MATCHED: the {} declared field(s) below seal to {expected}, the \
             digest supplied on the command line. That is as strong as the channel you got \
             that digest from - it is a seal, not a signature: {}",
            names.len(),
            names.join(", ")
        ),
        None => println!(
            "UNAUTHENTICATED DECLARATION: the verdict above grades BYTES. These {} field(s) \
             are written by the party being audited and NOTHING above checked them - pass \
             --expect-provenance-seal {seal} to make this line a check instead of a quote: {}",
            names.len(),
            names.join(", ")
        ),
    }
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

/// Render a TRI-STATE boolean: absent from the schema, present-but-`null`, or
/// measured. The three are different answers and are never collapsed - `null`
/// means the query was not run, which is not a "no", and that conflation is the
/// defect `git_dirty: Option<bool>` was introduced to fix in the first place.
fn tri_state(root: &Value, dotted: &str) -> String {
    match dig(root, dotted) {
        None => "NOT RECORDED by this schema".to_string(),
        Some(Value::Null) => "unknown (no tree was inspected)".to_string(),
        Some(Value::Bool(b)) => b.to_string(),
        Some(other) => format!("(not a boolean: {other})"),
    }
}

// ---- the metric the record certifies ----------------------------------------
//
// Four tiny pure functions, kept pure so `tests/ckpt_replay_bpb_gate.rs` can
// exercise them without spawning a trainer. They are the whole of the arithmetic
// behind a `BPB MISMATCH`; everything else in that verdict is printing.
//
// They are `pub(crate)` rather than private because an integration test cannot
// `use` a binary crate: that test compiles THIS FILE in with `#[path]`, so the
// assertions land on the code that ships and not on a copy of it.

/// The `bpb=` token of a `DONE:` line, EXACTLY as the trainer rendered it.
///
/// The token is returned as text and never re-rendered, because the comparison
/// this feeds is text against text: `trios-train` prints `{:.4}` and the record
/// carries full f64 precision, so the only honest meeting point is the string
/// the trainer itself chose to print.
///
/// `None` when the line is not a `DONE:` line or carries no `bpb=` token at all.
/// `bpb=unmeasured` - what the trainer prints when a run took no final
/// measurement - yields `Some("unmeasured")` here and `None` from
/// [`parse_done_bpb`]; that is the distinction between "said nothing" and "said
/// something unusable", and both callers need it.
pub(crate) fn done_bpb_token(line: &str) -> Option<&str> {
    if !line.trim_start().starts_with(DONE_PREFIX) {
        return None;
    }
    line.split_whitespace().find_map(|t| t.strip_prefix("bpb="))
}

/// The FINITE numeric value of a `DONE:` line's `bpb=` token.
///
/// `NaN` and `inf` parse as f64 and are rejected here on purpose: a non-finite
/// reading is the shape a poisoned forward pass takes, and treating one as a
/// measurement is exactly the laundering this repository has already had to
/// remove once. `unmeasured` and any other non-numeric token are `None` too.
pub(crate) fn parse_done_bpb(line: &str) -> Option<f64> {
    let value: f64 = done_bpb_token(line)?.parse().ok()?;
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

/// Render a recorded metric the way the trainer renders its own.
pub(crate) fn render_bpb(value: f64) -> String {
    format!("{value:.prec$}", prec = DONE_BPB_DECIMALS)
}

/// Does the recorded metric agree with what the replay printed?
///
/// Agreement is to [`DONE_BPB_DECIMALS`] places and NO FURTHER. A recorded
/// `2.6347548961639404` agrees with a replayed `2.6348` and disagrees with the
/// retracted `1.5492`; a difference of 5e-5 is invisible to this check and the
/// verdict line says so.
pub(crate) fn bpb_agrees(recorded: f64, replayed_token: &str) -> bool {
    recorded.is_finite() && render_bpb(recorded) == replayed_token
}

/// Does the recorded metric agree with the replayed one within `tolerance`?
///
/// This is the L2 question, and it is asked only when the BYTES already failed:
/// a byte-identical replay is graded by [`bpb_agrees`] at the four decimals the
/// trainer prints, and no tolerance is invented for it. A non-finite reading on
/// either side agrees with nothing, and a non-finite tolerance is refused rather
/// than allowed to admit everything.
pub(crate) fn bpb_within(recorded: f64, replayed: f64, tolerance: f64) -> bool {
    recorded.is_finite()
        && replayed.is_finite()
        && tolerance.is_finite()
        && tolerance >= 0.0
        && (recorded - replayed).abs() <= tolerance
}

/// The four-way grade a COMPARABLE pair of readings deserves: the bytes either
/// re-derived or did not, and the metric either agreed or did not.
///
/// Kept as an enum with an explicit exit code so the mapping is one table that a
/// test can read, rather than four `return` statements spread over a screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PairVerdict {
    /// Bytes and metric both re-derived. L3, scoped to this host.
    Verified,
    /// The bytes re-derived and the number the record certifies did not.
    BpbMismatch,
    /// The bytes did not re-derive and the metric did, within tolerance. This is
    /// what an honest laboratory on another platform looks like.
    SecondLaboratory,
    /// Neither re-derived. Nothing distinguishes this from a fabrication.
    Mismatch,
}

impl PairVerdict {
    pub(crate) fn exit_code(self) -> u8 {
        match self {
            PairVerdict::Verified => 0,
            PairVerdict::BpbMismatch => EXIT_BPB_MISMATCH,
            PairVerdict::SecondLaboratory => EXIT_L2_PASS_L3_FAIL,
            PairVerdict::Mismatch => EXIT_MISMATCH,
        }
    }
}

/// The whole of the four-way decision. `metric_agrees` is answered by
/// [`bpb_agrees`] when the bytes matched and by [`bpb_within`] when they did
/// not: those are different questions asked at different resolutions, and the
/// caller picks which one it asked.
pub(crate) fn grade_pair(bytes_match: bool, metric_agrees: bool) -> PairVerdict {
    match (bytes_match, metric_agrees) {
        (true, true) => PairVerdict::Verified,
        (true, false) => PairVerdict::BpbMismatch,
        (false, true) => PairVerdict::SecondLaboratory,
        (false, false) => PairVerdict::Mismatch,
    }
}

// ---- the artifact's own header ----------------------------------------------
//
// Decoded from the layout specified in the `src/checkpoint.rs` doc comment and
// implemented a second time in `interop/triosckp_reader.py`. Offsets are
// spelled out at every read rather than derived, so a divergence from the spec
// is a visible edit here and not an arithmetic accident.

/// Everything the 152-byte TRIOSCKP header states about the run.
#[derive(Debug, Clone)]
struct HeaderFacts {
    format_version: u32,
    vocab: u32,
    dim: u32,
    num_ctx: u32,
    hidden: u32,
    d_model: u32,
    num_heads: u32,
    attn_cfg_seq_len: u32,
    num_attn_layers: u32,
    ngram: u32,
    qk_gain: f64,
    attn_cfg_lr: f64,
    train_lr: f32,
    attn_scale: f32,
    attn_seq: u32,
    ctx_weights: [f32; CKPT_CTX_WEIGHTS],
    seed: u64,
    step: u64,
    /// Byte 124. The master switch on `gf16_floor()`, which rewrites
    /// `embed`/`proj`/`lm_head`/`ctx` IN PLACE. Unstated by every sidecar
    /// schema up to `/4`; `/5` records it, but the replay is not driven from it.
    gf16_enabled: u8,
    data_synthetic: u8,
    optimizer: String,
    fake_quant_format: String,
}

/// Why a container could not be decoded, and which verdict that deserves.
enum HeaderDefect {
    /// The bytes do not obey the format they claim. Graded `ARTIFACT ALTERED`:
    /// a file that is not a v1 TRIOSCKP container is not evidence about a v1
    /// TRIOSCKP run.
    Altered(String),
    /// The container is well formed but declares a version whose field offsets
    /// this binary does not know. Graded `INCOMPARABLE`: guessing at offsets to
    /// produce a verdict would be worse than abstaining.
    Unsupported(String),
}

fn le_u32(raw: &[u8], off: usize) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&raw[off..off + 4]);
    u32::from_le_bytes(b)
}

fn le_u64(raw: &[u8], off: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&raw[off..off + 8]);
    u64::from_le_bytes(b)
}

fn le_f32(raw: &[u8], off: usize) -> f32 {
    f32::from_bits(le_u32(raw, off))
}

fn le_f64(raw: &[u8], off: usize) -> f64 {
    f64::from_bits(le_u64(raw, off))
}

/// Render bytes as lowercase hex, for naming what was found where the magic
/// should have been without letting arbitrary bytes reach the terminal.
fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut acc, b| {
            use std::fmt::Write as _;
            let _ = write!(acc, "{b:02x}");
            acc
        })
}

/// Decode an ASCII, NUL-padded fixed-width field. The spec says "ASCII,
/// NUL-padded" and nothing more, so the strictest reading those words support
/// is enforced: bytes before the first NUL are the value, they must be
/// printable ASCII, and no byte after the first NUL may be anything but NUL.
fn ascii_field(raw: &[u8], off: usize, len: usize, name: &str) -> Result<String, HeaderDefect> {
    let field = &raw[off..off + len];
    let (value, padding) = match field.iter().position(|&b| b == 0) {
        Some(i) => field.split_at(i),
        None => (field, &[][..]),
    };
    if padding.iter().any(|&b| b != 0) {
        return Err(HeaderDefect::Altered(format!(
            "{name} padding after the first NUL at offset {off} is not all zero ({})",
            hex(padding)
        )));
    }
    if let Some(&bad) = value.iter().find(|&&b| !(0x20..=0x7e).contains(&b)) {
        return Err(HeaderDefect::Altered(format!(
            "{name} at offset {off} contains byte 0x{bad:02x}, outside printable ASCII"
        )));
    }
    Ok(String::from_utf8_lossy(value).into_owned())
}

/// Decode the header of a checkpoint image, rejecting anything that does not
/// obey format version 1. Every bound is checked before it is indexed, so a
/// truncated file becomes a verdict and never a panic.
fn parse_header(raw: &[u8]) -> Result<HeaderFacts, HeaderDefect> {
    if raw.len() < CKPT_MAGIC.len() {
        return Err(HeaderDefect::Altered(format!(
            "file is {} bytes, the magic alone is {}",
            raw.len(),
            CKPT_MAGIC.len()
        )));
    }
    if &raw[..CKPT_MAGIC.len()] != CKPT_MAGIC {
        return Err(HeaderDefect::Altered(format!(
            "expected magic {} ({}) at offset 0, found {}",
            String::from_utf8_lossy(CKPT_MAGIC),
            hex(CKPT_MAGIC),
            hex(&raw[..CKPT_MAGIC.len()])
        )));
    }
    if raw.len() < CKPT_HEADER_LEN {
        return Err(HeaderDefect::Altered(format!(
            "file is {} bytes, the fixed header alone is {CKPT_HEADER_LEN}",
            raw.len()
        )));
    }

    let format_version = le_u32(raw, 8);
    if format_version != CKPT_FORMAT_VERSION {
        return Err(HeaderDefect::Unsupported(format!(
            "the artifact declares format_version {format_version}; this auditor \
             implements version {CKPT_FORMAT_VERSION} and will not guess at the \
             field offsets of a format it has never seen"
        )));
    }
    let header_len = le_u32(raw, 12);
    if header_len as usize != CKPT_HEADER_LEN {
        return Err(HeaderDefect::Altered(format!(
            "format version 1 fixes header_len at {CKPT_HEADER_LEN}, the file declares {header_len}"
        )));
    }
    let tensor_count = le_u32(raw, 52);
    if tensor_count as usize != CKPT_TENSOR_COUNT {
        return Err(HeaderDefect::Altered(format!(
            "format version 1 fixes tensor_count at {CKPT_TENSOR_COUNT}, the file declares {tensor_count}"
        )));
    }
    if raw[126] != 0 || raw[127] != 0 {
        return Err(HeaderDefect::Altered(format!(
            "the 2 reserved bytes at offset 126 must be zero, found {}",
            hex(&raw[126..128])
        )));
    }

    let optimizer = ascii_field(raw, 128, 8, "optimizer")?;
    let fake_quant_format = ascii_field(raw, 136, 16, "fake_quant_format")?;

    if raw.len() < CKPT_PAYLOAD_OFFSET {
        return Err(HeaderDefect::Altered(format!(
            "file is {} bytes, the header plus the {CKPT_TENSOR_COUNT}-entry tensor \
             directory is {CKPT_PAYLOAD_OFFSET}",
            raw.len()
        )));
    }
    let mut elements: u128 = 0;
    for i in 0..CKPT_TENSOR_COUNT {
        elements += u128::from(le_u64(raw, CKPT_HEADER_LEN + i * 8));
    }
    // Spec: attn_down (8) and attn_up (9) always have identical element counts.
    // That makes them indistinguishable to the directory, but it also makes
    // inequality between them a detectable defect.
    let attn_down = le_u64(raw, CKPT_HEADER_LEN + 8 * 8);
    let attn_up = le_u64(raw, CKPT_HEADER_LEN + 9 * 8);
    if attn_down != attn_up {
        return Err(HeaderDefect::Altered(format!(
            "attn_down={attn_down} and attn_up={attn_up} must have identical element counts"
        )));
    }
    let expected = CKPT_PAYLOAD_OFFSET as u128 + CKPT_ELEMENT_SIZE * elements;
    if raw.len() as u128 != expected {
        return Err(HeaderDefect::Altered(format!(
            "the directory declares {elements} elements, so file_len must be {expected}; \
             the file is {} bytes",
            raw.len()
        )));
    }

    let mut ctx_weights = [0f32; CKPT_CTX_WEIGHTS];
    for (i, w) in ctx_weights.iter_mut().enumerate() {
        *w = le_f32(raw, 84 + i * 4);
    }

    Ok(HeaderFacts {
        format_version,
        vocab: le_u32(raw, 16),
        dim: le_u32(raw, 20),
        num_ctx: le_u32(raw, 24),
        hidden: le_u32(raw, 28),
        d_model: le_u32(raw, 32),
        num_heads: le_u32(raw, 36),
        attn_cfg_seq_len: le_u32(raw, 40),
        num_attn_layers: le_u32(raw, 44),
        ngram: le_u32(raw, 48),
        qk_gain: le_f64(raw, 56),
        attn_cfg_lr: le_f64(raw, 64),
        train_lr: le_f32(raw, 72),
        attn_scale: le_f32(raw, 76),
        attn_seq: le_u32(raw, 80),
        ctx_weights,
        seed: le_u64(raw, 108),
        step: le_u64(raw, 116),
        gf16_enabled: raw[124],
        data_synthetic: raw[125],
        optimizer,
        fake_quant_format,
    })
}

impl HeaderFacts {
    /// Printable value of one header field, by the name used in
    /// `HEADER_FIELD_MAP`. An unknown name is a table/struct drift and says so
    /// rather than silently printing nothing.
    fn value(&self, field: &str) -> String {
        match field {
            "format_version" => self.format_version.to_string(),
            "vocab" => self.vocab.to_string(),
            "dim" => self.dim.to_string(),
            "num_ctx" => self.num_ctx.to_string(),
            "hidden" => self.hidden.to_string(),
            "d_model" => self.d_model.to_string(),
            "num_heads" => self.num_heads.to_string(),
            "attn_cfg_seq_len" => self.attn_cfg_seq_len.to_string(),
            "num_attn_layers" => self.num_attn_layers.to_string(),
            "ngram" => self.ngram.to_string(),
            "qk_gain" => self.qk_gain.to_string(),
            "attn_cfg_lr" => self.attn_cfg_lr.to_string(),
            "train_lr" => (self.train_lr as f64).to_string(),
            "attn_scale" => (self.attn_scale as f64).to_string(),
            "attn_seq" => self.attn_seq.to_string(),
            "ctx_weights" => {
                let parts: Vec<String> = self
                    .ctx_weights
                    .iter()
                    .map(|w| (*w as f64).to_string())
                    .collect();
                format!("[{}]", parts.join(","))
            }
            "seed" => self.seed.to_string(),
            "step" => self.step.to_string(),
            "gf16_enabled" => self.gf16_enabled.to_string(),
            "data_synthetic" => self.data_synthetic.to_string(),
            "optimizer" => self.optimizer.clone(),
            "fake_quant_format" => self.fake_quant_format.clone(),
            other => format!("(unknown header field {other})"),
        }
    }
}

/// One scalar written into BOTH evidence documents, with two different values.
struct Disagreement {
    field: &'static str,
    header: String,
    record: String,
}

/// Compare every scalar the header and the sidecar both carry. Returns the
/// disagreements and how many fields were actually comparable, so a clean
/// verdict can state the size of the check instead of implying it was total.
///
/// A field the record omits is SKIPPED, not failed: an older schema simply
/// never made the claim, and its silence is reported by the `MISMATCH` path.
/// `lr` and `attn_scale` are stored as f32 in the header and as the f64
/// widening of that same f32 in the sidecar, so they are compared by rounding
/// the record back to f32 and comparing bit patterns - exact, and free of any
/// invented tolerance.
fn cross_check(header: &HeaderFacts, root: &Value) -> (Vec<Disagreement>, usize) {
    let mut out = Vec::new();
    let mut compared = 0usize;

    for (field, dotted, want) in [
        (
            "format_version",
            "format_version",
            u64::from(header.format_version),
        ),
        ("vocab", "vocab", u64::from(header.vocab)),
        ("hidden", "hidden", u64::from(header.hidden)),
        ("d_model", "d_model", u64::from(header.d_model)),
        (
            "num_attn_layers",
            "num_attn_layers",
            u64::from(header.num_attn_layers),
        ),
        ("attn_seq", "attn_seq", u64::from(header.attn_seq)),
        ("seed", "seed", header.seed),
        ("step", "step", header.step),
    ] {
        if !present(root, dotted) {
            continue;
        }
        compared += 1;
        match as_u64(root, dotted) {
            Some(got) if got == want => {}
            Some(got) => out.push(Disagreement {
                field,
                header: want.to_string(),
                record: got.to_string(),
            }),
            None => out.push(Disagreement {
                field,
                header: want.to_string(),
                record: dig(root, dotted)
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "(unreadable)".to_string()),
            }),
        }
    }

    for (field, dotted, want) in [
        ("optimizer", "optimizer", header.optimizer.as_str()),
        (
            "fake_quant_format",
            "fake_quant_format",
            header.fake_quant_format.as_str(),
        ),
    ] {
        if !present(root, dotted) {
            continue;
        }
        compared += 1;
        match as_str(root, dotted) {
            Some(got) if got == want => {}
            Some(got) => out.push(Disagreement {
                field,
                header: want.to_string(),
                record: got.to_string(),
            }),
            None => out.push(Disagreement {
                field,
                header: want.to_string(),
                record: dig(root, dotted)
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "(unreadable)".to_string()),
            }),
        }
    }

    // Boolean bytes. `gf16_enabled` is absent from every schema up to /4, so
    // for those records the loop skips it; listing it here is what made schema
    // /5 cross-checked from the first record it wrote, with no edit needed.
    for (field, dotted, byte) in [
        ("data_synthetic", "data_synthetic", header.data_synthetic),
        ("gf16_enabled", "gf16_enabled", header.gf16_enabled),
    ] {
        if !present(root, dotted) {
            continue;
        }
        compared += 1;
        let want = byte != 0;
        // A boolean byte that is neither 0 nor 1 is itself worth naming, so the
        // raw byte is carried into the message rather than collapsed to a bool.
        let shown = format!("{want} (byte {byte})");
        // A record may encode the flag as a JSON bool or as 0/1; both are read,
        // and anything else is a disagreement about the type as well as the value.
        let got = dig(root, dotted).and_then(|v| {
            v.as_bool().or_else(|| match v.as_u64() {
                Some(0) => Some(false),
                Some(1) => Some(true),
                _ => None,
            })
        });
        match got {
            Some(got) if got == want => {}
            Some(got) => out.push(Disagreement {
                field,
                header: shown,
                record: got.to_string(),
            }),
            None => out.push(Disagreement {
                field,
                header: shown,
                record: dig(root, dotted)
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "(unreadable)".to_string()),
            }),
        }
    }

    for (field, paths, want) in [
        ("lr", &["lr", "train_lr"][..], header.train_lr),
        ("attn_scale", &["attn_scale"][..], header.attn_scale),
    ] {
        let Some(dotted) = paths.iter().copied().find(|p| present(root, p)) else {
            continue;
        };
        compared += 1;
        match as_f64(root, dotted) {
            Some(got) if (got as f32).to_bits() == want.to_bits() => {}
            Some(got) => out.push(Disagreement {
                field,
                header: (want as f64).to_string(),
                record: got.to_string(),
            }),
            None => out.push(Disagreement {
                field,
                header: (want as f64).to_string(),
                record: dig(root, dotted)
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "(unreadable)".to_string()),
            }),
        }
    }

    (out, compared)
}

/// Header fields the record does not state, rendered `name=value`. These are
/// inputs the artifact declares and the sidecar is silent about, so every one
/// of them is a live suspect behind a `MISMATCH`.
fn unstated_header_fields(header: &HeaderFacts, root: &Value) -> Vec<String> {
    HEADER_FIELD_MAP
        .iter()
        .filter(|(_, paths)| !paths.iter().any(|p| present(root, p)))
        .map(|(name, _)| format!("{name}={}", header.value(name)))
        .collect()
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

// ---- integrity, which is not reproduction ------------------------------------

/// Locate the copy of the artifact that was PUBLISHED beside this record.
///
/// Only siblings are candidates, and that is the whole point. A record under
/// `evidence/` is graded where it was published, and the `path` it states is a
/// path on the TRAINING machine: absolute ones do not exist for anyone else,
/// and relative ones resolve against whatever directory the auditor happens to
/// be standing in. That is not a hypothetical - see the worked
/// `checkpoints/r4-docs-repro/12000.bin` collision in this file's header.
///
/// Following the recorded `path` would also make the mode unable to detect the
/// defect it exists to detect: a published `.bin` DELETED from the evidence tree
/// while its record stays behind. With a fallback, such a record is graded
/// against a byte-identical file somewhere else on the auditing machine and
/// passes, so the audit reports OK about a file that is gone. Sibling-only
/// resolution makes the deletion a verdict instead.
///
/// Returned with the RULE that found it, so the verdict can name the file it
/// hashed and how it chose it rather than implying there was only ever one
/// candidate.
fn locate_published_artifact(record_path: &Path, root: &Value) -> Option<(PathBuf, &'static str)> {
    // `isa-probe-arm64-0.json` names its artifact `isa-probe-arm64-0.bin`; the
    // step-named convention below does not reach it, because its step is 0 and
    // four such records share the directory.
    if let Some(stem) = record_path.file_stem() {
        let by_stem = record_path.with_file_name(format!("{}.bin", stem.to_string_lossy()));
        if by_stem.is_file() {
            return Some((by_stem, "published beside the record (<stem>.bin)"));
        }
    }
    if let Some(step) = as_u64(root, "step") {
        let by_step = record_path.with_file_name(format!("{step}.bin"));
        if by_step.is_file() {
            return Some((by_step, "published beside the record ({step}.bin)"));
        }
    }
    None
}

/// Answer the affordable question: do the published bytes still hash to the
/// digest published with them?
///
/// This function deliberately RETURNS before anything is re-derived. It never
/// reads the corpus, never resolves a trainer and never spawns a process, so a
/// caller cannot accidentally buy a training run by passing the wrong flag.
fn integrity_verdict(record_path: &Path, root: &Value, authenticated: Option<&str>) -> ExitCode {
    let schema = match as_str(root, "schema") {
        Some(s) if s.starts_with(SCHEMA_PREFIX) => s.to_string(),
        Some(s) => {
            println!("SKIPPED: {record_path:?} is a {s:?} document, not a {SCHEMA_PREFIX}* record");
            println!(
                "  note:  it publishes no artifact of its own, so there is no digest to \
                 re-check. A skip is counted, never silent: this is not a pass."
            );
            return ExitCode::from(EXIT_SKIPPED);
        }
        None => {
            println!("SKIPPED: {record_path:?} states no schema, so it is not a checkpoint record");
            println!("  note:  a skip is counted, never silent: this is not a pass.");
            return ExitCode::from(EXIT_SKIPPED);
        }
    };

    let recorded_sha = as_str(root, "sha256").unwrap_or("").trim().to_string();
    if recorded_sha.is_empty() {
        println!("INCOMPARABLE: {record_path:?} (schema {schema}) records no sha256");
        println!(
            "  note:  there is nothing to compare the bytes against. This is a verdict, \
             not a failure - and it is exit 2 rather than a skip, because the file IS a \
             checkpoint record and its silence is a fact about the evidence."
        );
        return ExitCode::from(EXIT_INCOMPARABLE);
    }

    let (artifact, rule) = match locate_published_artifact(record_path, root) {
        Some(found) => found,
        None => {
            println!(
                "PUBLISHED ARTIFACT MISSING: record {record_path:?} publishes no .bin beside it"
            );
            println!(
                "  the record's own `path` ({}) was NOT followed, because in integrity \
                 mode a path on the training machine is not the file under audit",
                as_str(root, "path").unwrap_or("(none)")
            );
            println!(
                "  looked for:    <stem>.bin and {{step}}.bin beside the record, and \
                 nothing else"
            );
            println!("  recorded sha:  {recorded_sha}");
            return ExitCode::from(EXIT_MISMATCH);
        }
    };
    let raw = match std::fs::read(&artifact) {
        Ok(raw) => raw,
        Err(e) => {
            println!("ARTIFACT MISSING: cannot read {artifact:?}: {e}");
            return ExitCode::from(EXIT_MISMATCH);
        }
    };
    let got = sha256_hex(&raw);
    if got != recorded_sha {
        println!("ARTIFACT ALTERED: {artifact:?} no longer hashes to its record");
        println!("  record:   {record_path:?} (schema {schema})");
        println!("  resolved: {rule}");
        println!("  recorded: {recorded_sha}");
        println!("  on disk:  {got} ({} bytes)", raw.len());
        return ExitCode::from(EXIT_MISMATCH);
    }

    // The length the record declares, compared against the length on disk.
    // `interop/triosckp_reader.py` has always done this (`SIDECAR_CHECKS` maps
    // `bytes` -> the container length, and a disagreement is
    // `SIDECAR_MISMATCH`); this binary did not, so a record edited to say
    // `"bytes": 1` printed INTEGRITY OK here and failed there. Two verifiers
    // disagreeing about the same record is the defect, not a difference of
    // opinion - and the weaker of the two was the one that reported success.
    // A record that states no length is not failed for it: absence is a fact
    // about the evidence, and the seal is what makes it undeletable.
    if let Some(declared_bytes) = as_u64(root, "bytes") {
        let actual_bytes = raw.len() as u64;
        if declared_bytes != actual_bytes {
            println!("ARTIFACT ALTERED: {artifact:?} is not the length its record declares");
            println!("  record:   {record_path:?} (schema {schema})");
            println!("  resolved: {rule}");
            println!("  declared: {declared_bytes} bytes");
            println!("  on disk:  {actual_bytes} bytes");
            println!(
                "  the recorded sha256 DID match these bytes, so the record contradicts \
                 itself: its digest describes this file and its length describes another. \
                 interop/triosckp_reader.py calls this SIDECAR_MISMATCH."
            );
            return ExitCode::from(EXIT_MISMATCH);
        }
    }

    println!("INTEGRITY OK: {record_path:?} (schema {schema})");
    println!("  artifact: {artifact:?} ({} bytes)", raw.len());
    println!("  resolved: {rule}");
    println!("  sha256:   {got} - re-hashed from disk, matches the record");
    println!(
        "  NOT A REPRODUCTION: nothing was re-derived and no trainer was executed. \
         This says the published bytes are unchanged, and says nothing about whether \
         the recipe in this record produces them; run ckpt_replay without \
         --integrity-only to ask that, at the cost of the run."
    );
    print_declaration_scope(root, authenticated);
    ExitCode::SUCCESS
}

// ---- the child's environment -------------------------------------------------

/// The eval grid the record declares, as `(chunks, seq)`. Both are schema-6
/// fields; either may be absent on its own, and absence is never a zero.
///
/// The chunk count carried by a record is the count the reading was ACHIEVED at,
/// not the target that was requested (`src/train_loop.rs` writes
/// `eval.plan.chunks`, taken from the `EvalStats` that produced the number). It
/// is therefore also the right value to REQUEST on replay: `eval_plan(len, C)`
/// returns the plan that achieved `C`, for every case including full coverage.
pub(crate) fn recorded_eval_grid(root: &Value) -> (Option<u64>, Option<u64>) {
    (
        as_u64(root, "eval_chunks").filter(|_| present(root, "eval_chunks")),
        as_u64(root, "eval_seq").filter(|_| present(root, "eval_seq")),
    )
}

/// Every environment variable the replay child is given BEYOND the three host
/// variables (`PATH`, `HOME`, `TMPDIR`) that survive `env_clear()`.
///
/// Factored out of `main` so that `tests/ckpt_replay_eval_grid.rs` can assert
/// what a given record puts in front of the trainer without paying for a
/// training run. Every value here comes from the RECORD; nothing comes from the
/// artifact header (see "Why the header is READ but never OBEYED") and nothing
/// from the auditor's own shell.
pub(crate) fn replay_env(root: &Value, ckpt_dir: &Path) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = Vec::new();
    let mut set = |k: &str, v: String| env.push((k.to_string(), v));

    let canon_name = as_str(root, "canon_name")
        .unwrap_or("ckpt-replay")
        .to_string();
    let gf16_floor_every = as_u64(root, "gf16_floor_every").unwrap_or_default();
    let fq_format = as_str(root, "fake_quant_format")
        .unwrap_or("f32")
        .to_string();

    set(
        "TRIOS_CHECKPOINT_DIR",
        ckpt_dir.to_string_lossy().into_owned(),
    );
    set("TRIOS_CANON_NAME", canon_name);
    set("TRIOS_GF16_FLOOR_EVERY", gf16_floor_every.to_string());
    set("TRIOS_FORMAT_TYPE", fq_format);
    set("TRINITY_AUTOMIGRATE", "0".to_string());

    if dig(root, "data_synthetic")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        set("TRIOS_ALLOW_SYNTHETIC_DATA", "1".to_string());
    }

    let step = as_u64(root, "step").unwrap_or_default();
    let steps_total = as_u64(root, "steps_total").unwrap_or_default();
    if step != steps_total {
        // The artifact is an intermediate checkpoint; ask for that cadence.
        set("TRIOS_CHECKPOINT_EVERY", step.to_string());
    }

    if let Some(scale) = as_f64(root, "attn_scale").filter(|_| present(root, "attn_scale")) {
        set("TRIOS_ATTN_SCALE", format!("{scale}"));
    }
    if let Some(seq) = as_u64(root, "attn_seq").filter(|_| present(root, "attn_seq")) {
        set("TRIOS_ATTN_SEQ", seq.to_string());
    }

    // The eval grid. Set ONLY when the record declares it: schemas 1 to 5
    // predate `TRIOS_EVAL_CHUNKS`, so their runs provably used the hardcoded
    // default, and leaving the variable unset is what reproduces them. Setting
    // it to a guess would be the same defect in the other direction.
    if let (Some(chunks), _) = recorded_eval_grid(root) {
        set("TRIOS_EVAL_CHUNKS", chunks.to_string());
    }

    env
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

    // (a.-1) The declaration seal. Answered before anything else, because it
    // is the only question here that costs nothing and the only one that is
    // about the half of the record no other mode checks.
    if args.provenance_seal {
        match provenance_seal(&root) {
            Ok(seal) => {
                println!("provenance-seal {seal}");
                println!(
                    "  covers {} declared field(s): {}",
                    sealed_field_names(&root).unwrap_or_default().len(),
                    sealed_field_names(&root).unwrap_or_default().join(", ")
                );
                println!(
                    "  a SEAL, not a signature: it is only worth the channel it is \
                     published on. See docs/PROVENANCE-BINDING.md."
                );
                return ExitCode::SUCCESS;
            }
            Err(e) => {
                println!("ERROR: cannot seal record {:?}: {e}", args.record);
                return ExitCode::from(EXIT_ERROR);
            }
        }
    }

    // (a.0) The declaration, checked against a digest the auditor brought with
    // them. A mismatch ends the run: every verdict below quotes the fields this
    // flag covers, so grading bytes under a declaration already known to be
    // wrong would print a true sentence beside a false one.
    let authenticated: Option<String> = match args.expect_provenance_seal.as_deref() {
        None => None,
        Some(expected) => {
            let expected = expected.trim();
            let got = match provenance_seal(&root) {
                Ok(seal) => seal,
                Err(e) => {
                    println!("ERROR: cannot seal record {:?}: {e}", args.record);
                    return ExitCode::from(EXIT_ERROR);
                }
            };
            if got != expected {
                println!(
                    "SEAL MISMATCH: record {:?} declares {got}, you expected {expected}",
                    args.record
                );
                println!(
                    "  the declaration - platform block, corpus and trainer digests, \
                     git_sha, git_dirty, seed, step, steps_total, eval_every, \
                     gf16_floor_every, final_val_bpb, schema, and the artifact this \
                     record names (sha256, bytes) - is not the one that seal was \
                     published for. Nothing was graded. Run --provenance-seal to see \
                     this record's own digest and the exact field list, and \
                     evidence/SEALS.txt for the published table."
                );
                return ExitCode::from(EXIT_MISMATCH);
            }
            Some(expected.to_string())
        }
    };

    // (a.0) The affordable question, asked and answered before anything
    // expensive is even resolved. Placed here rather than after the schema and
    // field-presence gates because those gates ask whether the record can state
    // its RECIPE, and integrity does not depend on the recipe: an archived
    // `schema/2` record that will never be gradeable as a reproduction still has
    // a digest, and that digest is still checkable.
    if args.integrity_only {
        return integrity_verdict(&args.record, &root, authenticated.as_deref());
    }

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
    let fq_format = as_str(&root, "fake_quant_format")
        .unwrap_or("f32")
        .to_string();
    let data_synthetic = dig(&root, "data_synthetic")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let canon_name = as_str(&root, "canon_name")
        .unwrap_or("ckpt-replay")
        .to_string();
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

    // The metric the record certifies. Read opportunistically and deliberately
    // NOT added to `REQUIRED_FIELDS`: a record that omits it is still gradeable
    // on its bytes. `states_bpb` is kept separate from the parsed value so a
    // record that carries the field as a string, a null-in-disguise or a
    // non-finite number is reported as stating something unusable rather than
    // as silent - the difference between an old schema and a forged one.
    let states_bpb = present(&root, "final_val_bpb");
    let recorded_bpb = as_f64(&root, "final_val_bpb").filter(|v| v.is_finite());

    // (c) Is the artifact still the artifact?
    let artifact = match locate_artifact(&args.record, &root, step) {
        Some(p) => p,
        None => {
            println!(
                "ARTIFACT MISSING: no readable .bin for record {:?}",
                args.record
            );
            println!(
                "  recorded path: {}",
                as_str(&root, "path").unwrap_or("(none)")
            );
            return ExitCode::from(EXIT_MISMATCH);
        }
    };
    let recorded_sha = as_str(&root, "sha256").unwrap_or("").to_string();
    // Read once and keep the bytes: the same image is hashed AND decoded, so
    // the header that is cross-checked below is provably the header inside the
    // file that matched the record, not a second read of a file that moved.
    let artifact_raw = match std::fs::read(&artifact) {
        Ok(raw) => raw,
        Err(e) => {
            println!("ARTIFACT MISSING: cannot read {artifact:?}: {e}");
            return ExitCode::from(EXIT_MISMATCH);
        }
    };
    let artifact_bytes = artifact_raw.len() as u64;
    let artifact_sha = sha256_hex(&artifact_raw);
    if artifact_sha != recorded_sha {
        println!("ARTIFACT ALTERED: {artifact:?} no longer hashes to its record");
        println!("  recorded: {recorded_sha}");
        println!("  on disk:  {artifact_sha} ({artifact_bytes} bytes)");
        return ExitCode::from(EXIT_MISMATCH);
    }

    // (c.1) The artifact's OWN header. A checkpoint is two evidence documents,
    // and a verdict reached from only one of them is half a verdict. Nothing
    // decoded here is ever fed back into the replay: see "Why the header is
    // READ but never OBEYED" at the top of this file.
    let header = match parse_header(&artifact_raw) {
        Ok(h) => h,
        Err(HeaderDefect::Altered(why)) => {
            println!("ARTIFACT ALTERED: {artifact:?} does not obey the TRIOSCKP format it claims");
            println!("  defect:   {why}");
            println!("  recorded: {recorded_sha} ({artifact_bytes} bytes)");
            println!(
                "  note:     these bytes DO hash to the record, so the defect was \
                 recorded, not introduced after the fact. Nothing was executed."
            );
            return ExitCode::from(EXIT_MISMATCH);
        }
        Err(HeaderDefect::Unsupported(why)) => {
            println!("INCOMPARABLE: format_version - {why}");
            println!("  record:   {:?} (schema {schema})", args.record);
            println!("  artifact: {artifact:?}");
            println!("  note:     this is a verdict, not a failure. Nothing was executed.");
            return ExitCode::from(EXIT_INCOMPARABLE);
        }
    };

    // (c.2) Do the two evidence documents agree? Thirteen scalars are written
    // into both, and until now no one compared them. A contradiction between
    // the record and the artifact it describes is a stronger finding than
    // either document alone, and it is not gradeable: there is no fact of the
    // matter about which copy states the run.
    let (disagreements, cross_checked) = cross_check(&header, &root);
    if let Some(first) = disagreements.first() {
        println!(
            "INCOMPARABLE: {} disagrees between the artifact header and the sidecar; \
             the two records of this run contradict each other",
            first.field
        );
        for d in &disagreements {
            println!(
                "  {}: header says {}, record says {}",
                d.field, d.header, d.record
            );
        }
        println!("  record:   {:?} (schema {schema})", args.record);
        println!("  artifact: {artifact:?} (sha256 {artifact_sha}, hashed header)");
        println!(
            "  note:     nothing was executed. The header is inside the hash and the \
             sidecar is not, but this binary does not adjudicate between them: a \
             replay can only be graded against evidence that agrees with itself."
        );
        return ExitCode::from(EXIT_INCOMPARABLE);
    }

    // (c.3) `gf16_enabled` - byte 124, inside the hash. It gates
    // `gf16_floor()`, which rewrites embed/proj/lm_head/ctx in place, and the
    // trainer's default is ENABLED while this binary clears the child
    // environment. Whether the record is silent about it (schemas up to /4) or
    // states it without this binary driving the replay from it (schema /5), the
    // replay provably runs at a setting the artifact says was not used, and
    // grading the resulting hash difference as MISMATCH would accuse a vendor
    // whose run is in fact perfectly reproducible.
    if header.gf16_enabled != 1 {
        let state = if header.gf16_enabled == 0 {
            "DISABLED".to_string()
        } else {
            format!("byte {}, which is neither 0 nor 1", header.gf16_enabled)
        };
        println!(
            "INCOMPARABLE: gf16_enabled = {} in the artifact and {}; \
             the replay cannot be driven to the setting the artifact declares",
            header.gf16_enabled,
            if present(&root, "gf16_enabled") {
                "in the record, but not in the replay this binary knows how to drive"
            } else {
                "in no field of this record's schema"
            }
        );
        // The record is silent about this field in every schema shipped so far.
        // If a later one starts carrying it, the cross-check above has already
        // confirmed the two agree, and the cause changes from "unrecorded" to
        // "recorded but not yet driven" - which is a different sentence, so it
        // is a different sentence here rather than a stale one.
        if present(&root, "gf16_enabled") {
            println!(
                "  cause:    the artifact declares the weight-mutating gf16 floor was {state} \
                 and the record agrees, but this binary does not drive the replay from that \
                 field: the child would run at the trainer's default, which is ENABLED."
            );
        } else {
            println!(
                "  cause:    the artifact declares the weight-mutating gf16 floor was {state}, \
                 the record is silent about it, and the trainer's default is ENABLED - so a \
                 replay driven by this record would train a different model."
            );
        }
        println!("  record:   {:?} (schema {schema})", args.record);
        println!("  artifact: {artifact:?} (byte 124 of the hashed TRIOSCKP header)");
        println!(
            "  note:     nothing was executed, and this binary deliberately does NOT \
             set TRIOS_GF16_DISABLE from the artifact: a header that supplies the \
             parameters it is verified against is not being verified. {}",
            if present(&root, "gf16_enabled") {
                "The record does state it, so this becomes gradeable as soon as the \
                 replay is driven from the RECORD's copy of the field."
            } else {
                "Record the setting in the sidecar and this becomes gradeable."
            }
        );
        return ExitCode::from(EXIT_INCOMPARABLE);
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
            "  recorded path: {} (relative to the record's source_digest_scope)",
            as_str(&root, "trainer.path").unwrap_or("(none)")
        );
        // The flags file is the usual reason an honest auditor lands here, so
        // the record's own statement about it is printed next to the remedy.
        //
        // Under a heading that says what the digest is worth, because on its
        // own it reads like a check and is not one: nothing here compares
        // `rustflags_sha256` against anything, and it could not - the file it
        // hashes is the gitignored `.cargo/config.toml` that
        // `scripts/repro_build.sh` writes, carrying the TRAINING host's
        // absolute paths, so a fresh clone cannot produce it and no third
        // party can ever match it.
        println!(
            "  build flags, QUOTED AND NOT CHECKED: the digest below hashes a gitignored, \
             host-absolute .cargo/config.toml that no fresh clone has, so it gates nothing \
             here and is printed only to tell an honest auditor which flags to rebuild with"
        );
        println!(
            "  recorded build flags: source={} remap_applied={} sha256={}",
            as_str(&root, "platform.rustflags_source").unwrap_or("(not recorded)"),
            match dig(&root, "platform.remap_applied") {
                Some(Value::Bool(b)) => b.to_string(),
                _ => "(not recorded)".to_string(),
            },
            as_str(&root, "platform.rustflags_sha256").unwrap_or("(not recorded)"),
        );
        // The other cause that actually fires, and the one no field named
        // before schema 8: the published `:gf16` image is a DIFFERENT BINARY
        // from the default build (8a357f46... vs 6a2874b9...), so an auditor
        // who pulled the tagged image lands here with every printed cause
        // above inapplicable. Compare this line against the feature set of the
        // binary being offered before suspecting substitution.
        println!(
            "  recorded features: {}",
            as_str(&root, "platform.features")
                .unwrap_or("(not recorded by this schema; /8 is the first to name it)"),
        );
        println!(
            "  note:    nothing was executed. Pass --trainer <path> pointing at the \
             binary the record names, or rebuild it; a replay driven by an unnamed \
             executable grades the executable, not the claim."
        );
        // Named explicitly because the obvious command PROVABLY does not
        // reproduce the pinned hash. `trainer.sha256` is decided by the
        // `--remap-path-prefix` flags that `scripts/repro_build.sh` writes into
        // `.cargo/config.toml`; that file is gitignored, so a fresh clone does
        // not have it, and `cargo build --release --locked` there bakes the
        // builder's absolute paths into .rodata and lands on a different hash.
        // Saying only "rebuild it" sent the auditor back through the exact door
        // they just came out of.
        println!(
            "  rebuild: bash scripts/repro_build.sh --bin trios-train   (NOT plain \
             `cargo build --release --locked`: the recorded hash is fixed by the \
             --remap-path-prefix flags that script writes into .cargo/config.toml, \
             which is gitignored and therefore absent from a fresh clone)"
        );
        println!(
            "  note:    a rebuilt binary can still differ across CPU architectures \
             and operating systems. Compare platform.os / platform.arch first; a \
             cross-platform hash mismatch is a declared boundary of the method, \
             not evidence of substitution."
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
    // Every remaining variable comes from the record, through one function so
    // that what a test asserts is what the child is handed. `TRIOS_EVAL_CHUNKS`
    // is in there: without it the child evaluated at the default 40 whatever the
    // record declared, and a record that declared anything else was graded a
    // liar for a grid this binary chose for it.
    for (key, value) in replay_env(&root, &ckpt_dir) {
        cmd.env(key, value);
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

    // The child's stdout is PIPED rather than inherited, because the number this
    // binary certifies is printed on it and `cmd.status()` threw it away. It is
    // echoed line by line AS IT ARRIVES and flushed after every line: a
    // 12 000-step replay takes minutes, and an auditor watching a silent
    // terminal cannot tell a long run from a hung one. `.output()` would have
    // captured the same text and shown the auditor none of it until exit.
    //
    // stderr is left INHERITED, so the trainer's own progress and warnings still
    // reach the terminal directly and are never interleaved into the text this
    // binary parses.
    cmd.stdout(Stdio::piped());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            println!("ERROR: could not execute {trainer:?}: {e}");
            return ExitCode::from(EXIT_ERROR);
        }
    };
    // The LAST `DONE:` line wins. The sweep path prints one per seed; a replay
    // driven by a record always names a single seed, so in practice there is
    // exactly one, and taking the last is the reading that stays correct if that
    // ever stops being true.
    let mut last_done: Option<String> = None;
    let mut stdout_defect: Option<String> = None;
    if let Some(pipe) = child.stdout.take() {
        use std::io::Write as _;
        for item in BufReader::new(pipe).lines() {
            match item {
                Ok(line) => {
                    println!("{line}");
                    let _ = std::io::stdout().flush();
                    if line.trim_start().starts_with(DONE_PREFIX) {
                        last_done = Some(line);
                    }
                }
                Err(e) => {
                    stdout_defect = Some(e.to_string());
                    break;
                }
            }
        }
    } else {
        stdout_defect = Some("the child exposed no stdout pipe".to_string());
    }
    if let Some(why) = &stdout_defect {
        // Not fatal on its own: the WEIGHTS verdict never needed stdout. It is
        // announced here and named again below if it is what stopped the metric
        // from being graded.
        println!("WARNING: the trainer's stdout ended early: {why}");
    }
    let status = match child.wait() {
        Ok(s) => s,
        Err(e) => {
            println!("ERROR: could not wait on {trainer:?}: {e}");
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

    // (f.2) Did the replay actually run on the grid the record declares?
    //
    // `TRIOS_EVAL_CHUNKS` was exported above, but exporting a variable is not
    // the same as observing its effect, and the whole defect being closed here
    // was a metric graded against a grid nobody checked. The replay writes its
    // own sidecar beside its own artifact; that sidecar states the grid the
    // reading was actually taken on, so it is read back and compared. A
    // disagreement makes the two numbers readings of DIFFERENT quantities, and
    // that is `GRID MISMATCH` - never an accusation that the record lied.
    let (rec_chunks, rec_seq) = recorded_eval_grid(&root);
    let replay_sidecar = replayed.with_extension("json");
    let replay_record: Option<Value> = std::fs::read_to_string(&replay_sidecar)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok());
    let (rep_chunks, rep_seq) = match &replay_record {
        Some(v) => recorded_eval_grid(v),
        None => (None, None),
    };
    let grid_disagreements: Vec<Disagreement> = [
        ("eval_chunks", rec_chunks, rep_chunks),
        ("eval_seq", rec_seq, rep_seq),
    ]
    .into_iter()
    .filter_map(
        |(field, recorded, replayed_value)| match (recorded, replayed_value) {
            (Some(a), Some(b)) if a != b => Some(Disagreement {
                field,
                // `header` and `record` are the two sides this struct prints; here
                // they are the replay's grid and the record's grid.
                header: b.to_string(),
                record: a.to_string(),
            }),
            _ => None,
        },
    )
    .collect();
    let grid_line = match (rec_chunks, rec_seq, rep_chunks, rep_seq) {
        (None, _, rep_c, _) => format!(
            "NOT RECORDED by this schema (schemas 1-5 predate TRIOS_EVAL_CHUNKS and \
             provably ran at the hardcoded default); the replay ran at {} chunks",
            rep_c
                .map(|c| c.to_string())
                .unwrap_or_else(|| "(unstated)".to_string())
        ),
        (Some(c), seq, Some(rc), rseq) => format!(
            "record eval_chunks={c}{} - the replay was driven to it with \
             TRIOS_EVAL_CHUNKS={c} and ran at eval_chunks={rc}{}",
            seq.map(|s| format!(" eval_seq={s}")).unwrap_or_default(),
            rseq.map(|s| format!(" eval_seq={s}")).unwrap_or_default()
        ),
        (Some(c), seq, None, _) => format!(
            "record eval_chunks={c}{} - TRIOS_EVAL_CHUNKS={c} was exported, but the \
             replay's own sidecar states no grid, so the request is unconfirmed",
            seq.map(|s| format!(" eval_seq={s}")).unwrap_or_default()
        ),
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
    let platform_conflict =
        declared_os.is_some_and(|d| d != os) || declared_arch.is_some_and(|d| d != arch);

    println!("--- ckpt_replay verdict ---");
    println!("record:        {:?} (schema {schema})", args.record);
    println!("artifact:      {artifact:?}");
    println!("recipe:        seed={seed} steps_total={steps_total} step={step} hidden={hidden} attn_layers={attn_layers}");
    println!("               optimizer={optimizer} fake_quant_format={fq_format} gf16_floor_every={gf16_floor_every} eval_every={eval_every} data_synthetic={data_synthetic}");
    match recorded_lr {
        Some(lr) => println!("lr:            {lr} (from the record)"),
        None => {
            println!("lr:            NOT RECORDED by this schema; replay used the trainer default")
        }
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
    // Both halves of the tree's state, printed together because either one
    // alone is misread. `git_dirty: false` says NO TRACKED FILE WAS MODIFIED;
    // it has never said the tree matched `git_sha`. An untracked `.rs` under
    // `src/` is compiled into the trainer and moves `source_sha256` while
    // `git_dirty` stays false, so the second field is what makes the first
    // safe to read.
    println!(
        "tree:          git_sha={} ({}) git_dirty={} git_untracked={}",
        as_str(&root, "git_sha").unwrap_or("(none)"),
        as_str(&root, "git_provenance").unwrap_or("(not recorded)"),
        tri_state(&root, "git_dirty"),
        tri_state(&root, "git_untracked"),
    );
    println!(
        "               git_dirty=false means no TRACKED modification, NOT that the \
         tree matched git_sha; git_untracked reports the files git was told to ignore \
         in that answer and which the source digest hashes anyway"
    );
    println!("trainer:       {trainer:?}");
    println!("               sha256={trainer_sha} version={version}");
    println!("               re-hashed here and it MATCHES the record's trainer.sha256");
    if let Some(vocab) = as_u64(&root, "vocab") {
        println!("vocab:         {vocab} (alphabet the corpus was folded onto)");
    } else {
        println!(
            "vocab:         NOT RECORDED by this schema; the alphabet behind the BPB is unstated"
        );
    }
    println!(
        "header:        TRIOSCKP v{} decoded from the artifact; {cross_checked} scalars \
         cross-checked against the record, all agree",
        header.format_version
    );
    println!(
        "               gf16_enabled={} (byte 124, inside the hash; {})",
        header.gf16_enabled,
        if present(&root, "gf16_enabled") {
            "the record states it too, and the two agree"
        } else {
            "no field of this record's schema states it"
        }
    );
    println!("eval grid:     {grid_line}");
    println!("workdir:       {workdir:?}");
    println!("recorded sha:  {recorded_sha} ({artifact_bytes} bytes)");
    println!("replay sha:    {replay_sha} ({replay_bytes} bytes)");
    println!(
        "recorded bpb:  {}",
        match (states_bpb, recorded_bpb) {
            (false, _) => "NOT RECORDED by this record; the metric is not graded below".to_string(),
            (true, Some(v)) => format!(
                "final_val_bpb {v} (renders as {} at {DONE_BPB_DECIMALS} dp)",
                render_bpb(v)
            ),
            (true, None) => format!(
                "final_val_bpb is present but not a finite number: {}",
                dig(&root, "final_val_bpb")
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "(unreadable)".to_string())
            ),
        }
    );
    println!(
        "replay DONE:   {}",
        last_done
            .as_deref()
            .unwrap_or("(the replay printed no DONE: line)")
    );
    println!("host:          {os}/{arch}");
    println!(
        "record says:   {}",
        if declared.is_empty() {
            "(no platform fields)".to_string()
        } else {
            declared.join(" ")
        }
    );

    // The platform caveat reads the same under every metric verdict, so it is
    // composed once here and printed under each of them rather than duplicated
    // into three drifting copies.
    let platform_note = if declared.is_empty() {
        "platform triple was NOT declared by the record; \
         this verdict is valid only on this host"
            .to_string()
    } else if platform_conflict {
        // Bits matched anyway, which is a stronger result than the record
        // claimed; it is reported as the anomaly it is, not folded away.
        format!(
            "platform triple DIFFERS from the record ({}) yet the bytes matched on {os}/{arch}",
            declared.join(" ")
        )
    } else {
        "platform triple matches the record".to_string()
    };

    if replay_sha == recorded_sha {
        // The bytes are the bytes. That used to be the ONLY verdict, and it is
        // why a sidecar carrying the retracted 1.5492 printed VERIFIED one
        // screen below the replay's own honest 2.9744. Grade the NUMBER before
        // any success is announced.
        //
        // A `DONE:` token is required to be a FINITE number before it is graded
        // against: `bpb=unmeasured` and `bpb=NaN` are the shapes "no
        // measurement" takes, and neither is a value to compare with.
        let replayed = last_done
            .as_deref()
            .and_then(|l| parse_done_bpb(l).and(done_bpb_token(l)));
        // An intermediate sidecar records the metric AT ITS OWN STEP while the
        // `DONE:` line reports the metric at `steps_total`. Comparing the two
        // would manufacture a failure out of an honest record, so it is not
        // done, and the abstention is printed with its cause.
        let final_step = step == steps_total;

        if let Some(first) = grid_disagreements.first() {
            // The bytes re-derived, so the recipe is right; what differs is the
            // GRID the two numbers were read on, and two readings of different
            // quantities cannot corroborate or contradict each other. Saying
            // "the recorded number is not the number these weights produced"
            // here would be false: these weights produce BOTH numbers, one per
            // grid.
            println!("GRID MISMATCH on {os}/{arch}");
            println!(
                "  the WEIGHTS re-derived exactly ({replay_sha}); the eval grid did not, \
                 so the metric is NOT graded"
            );
            for d in &grid_disagreements {
                println!(
                    "  {}: the record declares {}, the replay ran at {}",
                    d.field, d.record, d.header
                );
            }
            println!(
                "  note:     {} is an OBSERVATION parameter: it never touches the weights \
                 and it decides the number completely. A BPB verdict is scoped to the grid \
                 the record declares, and this replay was not on it.",
                first.field
            );
            println!("{platform_note}");
            ExitCode::from(EXIT_BPB_MISMATCH)
        } else if !states_bpb {
            println!("VERIFIED on {os}/{arch}");
            print_declaration_scope(&root, authenticated.as_deref());
            println!("{platform_note}");
            println!("WEIGHTS VERIFIED; final_val_bpb NOT GRADED (not recorded)");
            ExitCode::SUCCESS
        } else if recorded_bpb.is_some() && !final_step {
            println!("VERIFIED on {os}/{arch}");
            print_declaration_scope(&root, authenticated.as_deref());
            println!("{platform_note}");
            println!(
                "WEIGHTS VERIFIED; final_val_bpb NOT GRADED (the record states {} at step \
                 {step} of a {steps_total}-step run, and the replay's DONE: line states the \
                 metric at step {steps_total}: different quantities)",
                recorded_bpb.unwrap_or_default()
            );
            ExitCode::SUCCESS
        } else if let (Some(rec), Some(tok)) = (recorded_bpb, replayed) {
            if bpb_agrees(rec, tok) {
                println!("VERIFIED on {os}/{arch}");
                print_declaration_scope(&root, authenticated.as_deref());
                println!("{platform_note}");
                println!("final_val_bpb {tok} confirmed by replay ({DONE_BPB_DECIMALS} dp)");
                // Through the same table the other three cases go through, so
                // the mapping a test asserts is the mapping that ships.
                ExitCode::from(grade_pair(true, true).exit_code())
            } else {
                println!("BPB MISMATCH on {os}/{arch}");
                println!(
                    "  the WEIGHTS re-derived exactly ({replay_sha}) and the NUMBER \
                     this record certifies did not"
                );
                println!(
                    "  recorded: {rec} (final_val_bpb, renders as {} at {DONE_BPB_DECIMALS} dp)",
                    render_bpb(rec)
                );
                println!(
                    "  replayed: {tok} (the trainer's own DONE: line, {DONE_BPB_DECIMALS} dp)"
                );
                println!("  line:     {}", last_done.as_deref().unwrap_or("(none)"));
                println!(
                    "  note:     the comparison is to {DONE_BPB_DECIMALS} decimal places, the \
                     precision `trios-train` prints; a difference this large is not a \
                     rounding artefact. Byte-identical weights DO measure a different metric \
                     on a different eval grid, so that is checked first and reported as GRID \
                     MISMATCH; the grids agree here ({grid_line}), which is what makes this a \
                     statement about the number and not about the instrument."
                );
                println!("{platform_note}");
                ExitCode::from(grade_pair(true, false).exit_code())
            }
        } else {
            // Either the record states the metric as something that is not a
            // finite number, or the replay stated no metric at all. Neither is a
            // contradiction, and neither is a confirmation; a record whose
            // number nothing corroborates must not exit 0 under a README that
            // calls this binary the thing that grades it.
            println!("BPB NOT CONFIRMED on {os}/{arch}");
            println!("  the WEIGHTS re-derived exactly ({replay_sha}); the metric was NOT graded");
            match recorded_bpb {
                Some(v) => println!(
                    "  recorded: {v} (final_val_bpb, renders as {} at {DONE_BPB_DECIMALS} dp)",
                    render_bpb(v)
                ),
                None => println!(
                    "  recorded: {} - final_val_bpb is present, but not a finite number",
                    dig(&root, "final_val_bpb")
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "(unreadable)".to_string())
                ),
            }
            println!(
                "  replayed: {}",
                match (replayed, last_done.as_deref(), &stdout_defect) {
                    (Some(tok), _, _) =>
                        format!("{tok} - usable, but the record's own value is not"),
                    (None, Some(l), _) => format!("no finite bpb= token in {l:?}"),
                    (None, None, Some(why)) =>
                        format!("no DONE: line was captured - the trainer's stdout {why}"),
                    (None, None, None) => "the replay printed no DONE: line at all".to_string(),
                }
            );
            println!(
                "  note:     a record that certifies a number the replay does not state is \
                 unconfirmed, not confirmed. This is not a MISMATCH: nothing contradicted the \
                 record, and nothing corroborated it either."
            );
            println!("{platform_note}");
            ExitCode::from(EXIT_BPB_MISMATCH)
        }
    } else {
        // The bytes did not re-derive, which is an L3 failure and used to be the
        // only thing this binary could say. `docs/REPRODUCIBILITY-GRADING.md`
        // defines an L2 rung below it - "the metric reproduces within a stated
        // tolerance" - and without it the single honest cross-architecture run
        // this repository has ever performed earned the same verdict as a
        // fabricated checkpoint. The ladder is applied here, on the same page as
        // the failure, so the reader is never handed "fraud" as the only
        // available reading of a difference in floating-point evaluation order.
        //
        // L2 is asked ONLY when there is something to ask it about: two finite
        // numbers, taken at the same step, on the same eval grid. Any of those
        // missing and the rung is skipped with its cause named - a tolerance
        // applied to incomparable quantities is not a weaker verdict, it is a
        // fabricated one.
        let replayed_value = last_done.as_deref().and_then(parse_done_bpb);
        let ladder = match (
            recorded_bpb,
            replayed_value,
            step == steps_total,
            grid_disagreements.is_empty(),
        ) {
            (Some(rec), Some(rep), true, true) => Ok((rec, rep)),
            (None, _, _, _) => Err("the record states no finite final_val_bpb".to_string()),
            (_, None, _, _) => Err("the replay stated no finite bpb= token".to_string()),
            (_, _, false, _) => Err(format!(
                "the record is an intermediate checkpoint (step {step} of {steps_total}), \
                 so its metric and the replay's DONE: line are different quantities"
            )),
            (_, _, _, false) => {
                Err("the replay did not run on the eval grid the record declares".to_string())
            }
        };
        let metric_agrees = ladder
            .as_ref()
            .is_ok_and(|(rec, rep)| bpb_within(*rec, *rep, args.bpb_tolerance));
        let verdict = grade_pair(false, metric_agrees);

        // `(SecondLaboratory, Err)` cannot arise - `metric_agrees` is false
        // whenever the rung was not asked - and the arms are written so that the
        // numbers printed are the numbers graded, never a default standing in.
        match (verdict, &ladder) {
            (PairVerdict::SecondLaboratory, Ok((rec, rep))) => {
                println!(
                    "L2 PASS (metric agrees within {} bpb) / L3 FAIL (bytes differ on {os}/{arch})",
                    args.bpb_tolerance
                );
                println!(
                    "  recorded {rec} vs replayed {rep}, |delta| = {:.6} bpb, tolerance {} \
                     (--bpb-tolerance; default {BPB_TOLERANCE_DEFAULT} derived from the paired \
                     cross-laboratory deltas in docs/REPRODUCIBILITY-GRADING.md)",
                    (rec - rep).abs(),
                    args.bpb_tolerance
                );
                println!(
                    "  L2 without L3: the recipe reproduced the NUMBER and not the BYTES. \
                     That is the shape an honest laboratory on another platform takes, and \
                     it is why this is not exit 1 - but it is a WEAKER claim, not an \
                     acquittal. The cause is not diagnosed here; every suspect named below \
                     applies to this verdict too, and a record whose metric ALSO disagreed \
                     would be MISMATCH."
                );
            }
            (_, Ok((rec, rep))) => {
                println!("MISMATCH on {os}/{arch}");
                println!(
                    "  L2 was asked and FAILED too: recorded {rec} vs replayed {rep}, \
                     |delta| = {:.6} bpb, over the {} bpb tolerance. Neither the bytes \
                     nor the number reproduced.",
                    (rec - rep).abs(),
                    args.bpb_tolerance
                );
            }
            (_, Err(why)) => {
                println!("MISMATCH on {os}/{arch}");
                println!("  L2 NOT ASKED: {why}, so this MISMATCH grades the bytes only");
            }
        }
        println!("  recorded: {recorded_sha}");
        println!("  replayed: {replay_sha}");
        if !grid_disagreements.is_empty() {
            for d in &grid_disagreements {
                println!(
                    "  {}: the record declares {}, the replay ran at {}",
                    d.field, d.record, d.header
                );
            }
        }
        // The suspects below are the same under either verdict; only the name of
        // the byte-level finding changes, so it is named once and reused rather
        // than left saying "MISMATCH" under a line that says L3 FAIL.
        let byte_verdict = if verdict == PairVerdict::SecondLaboratory {
            "L3 FAIL"
        } else {
            "MISMATCH"
        };
        if declared.is_empty() {
            println!(
                "platform triple was NOT declared by the record, so this {byte_verdict} does \
                 not distinguish a bad record from a different host"
            );
        } else if platform_conflict {
            println!(
                "the record was produced on {} and this host is {os}/{arch}; \
                 a cross-platform {byte_verdict} is expected, not evidence of a bad record",
                declared.join(" ")
            );
        }
        if !unrecorded_recipe.is_empty() {
            println!(
                "the record does not state {}; any of them could explain this {byte_verdict}",
                unrecorded_recipe.join(", ")
            );
        }
        // A MISMATCH with no named suspect is an accusation, not a measurement.
        // Everything the artifact declares and the record does not is printed
        // here, unconditionally, so the auditor is never left with "the vendor
        // is lying" as the only reading available to them.
        let unstated = unstated_header_fields(&header, &root);
        if unstated.is_empty() {
            println!(
                "every field of the artifact header has a counterpart in the record, \
                 so this {byte_verdict} is not explained by anything the record left unsaid"
            );
        } else {
            println!(
                "the record does not state these fields of the artifact's own header; \
                 the replay ran at the trainer's defaults for all of them:"
            );
            println!("  {}", unstated.join(" "));
        }
        ExitCode::from(verdict.exit_code())
    }
}
