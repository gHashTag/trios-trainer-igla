//! Checkpoint save/load. Stored under `${TRIOS_CHECKPOINT_DIR}/{run-name}/{step}.bin`,
//! or `${TRIOS_CHECKPOINT_DIR}/{run-name}/seed{seed}/{step}.bin` on a multi-seed
//! sweep, where one `TRIOS_CANON_NAME` covers several runs (see `run_dir`).
//!
//! P4 addition: `ema_average` for post-hoc EMA over last N checkpoints.
//! Reference: Sanyal et al. 2024 - free generalization gain at zero training cost.
//!
//! # EPIC-446 - the on-disk artifact format (version 1)
//!
//! A checkpoint file is `${TRIOS_CHECKPOINT_DIR:-checkpoints}/{run}/{step}.bin`.
//! Every multi-byte integer and float is LITTLE-ENDIAN; floats are stored as
//! their IEEE-754 bit patterns via `to_le_bytes`, so NaN payloads and signed
//! zeros round-trip exactly. No padding, no alignment, no compression, no
//! trailer. Three sections:
//!
//! ```text
//!   header (152 bytes) | tensor directory (152 bytes) | payload
//! ```
//!
//! Header, at absolute byte offsets:
//!
//! ```text
//!    0   8  magic = b"TRIOSCKP"
//!    8   4  format_version: u32 = 1
//!   12   4  header_len: u32 = 152 (absolute offset of the tensor directory)
//!   16   4  vocab      20   4  dim         24   4  num_ctx
//!   28   4  hidden     32   4  d_model     36   4  num_heads
//!   40   4  attn_cfg_seq_len                44   4  num_attn_layers
//!   48   4  ngram      52   4  tensor_count = 19
//!   56   8  qk_gain: f64 bits
//!   64   8  attn_cfg_lr: f64 bits (HybridAttnConfig::lr, NOT the training lr)
//!   72   4  train_lr: f32 bits    76   4  attn_scale: f32 bits
//!   80   4  attn_seq: u32
//!   84  24  ctx_weights: [f32; 6] bits, index order 0..5
//!  108   8  seed: u64            116   8  step: u64
//!  124   1  gf16_enabled: u8     125   1  data_synthetic: u8
//!  126   2  reserved (must be zero, rejected if nonzero)
//!  128   8  optimizer: ASCII, NUL-padded
//!  136  16  fake_quant_format: ASCII, NUL-padded
//! ```
//!
//! The tensor directory is 19 consecutive `u64` element counts (f32 counts,
//! not byte counts) in the canonical order below. It is purely redundant with
//! the header scalars; it exists so that a load which would produce a
//! differently shaped model fails loudly instead of reshaping silently.
//!
//! Canonical tensor order (MUST NOT CHANGE within version 1):
//!
//! ```text
//!   0 embed  1..6 ctx[0..5]  7 proj  8 attn_down  9 attn_up  10 lm_head
//!  11 wq  12 wk  13 wv  14 wo  15 wq2  16 wk2  17 wv2  18 wo2
//! ```
//!
//! Note that `attn_down` (8) and `attn_up` (9) have identical element counts,
//! so the directory cannot detect a swap between them: canonical-order
//! discipline is the only protection there.
//!
//! `file_len = 304 + 4 * sum(directory counts)`.
//!
//! # Why a hand-rolled binary format
//!
//! `bincode`, `zstd`, `blake3` and `postcard` are all absent from `Cargo.lock`;
//! the historical `// TODO: zstd-compressed bincode` named two crates this
//! crate does not have. A JSON header is rejected for the *hashed* artifact
//! because `serde_json` float formatting and map ordering are reproducibility
//! hazards for `qk_gain = 2.618033988749895`; JSON is used only for the
//! unhashed sidecar. A fixed binary layout makes "two independent implementers
//! produce byte-compatible files" mechanically checkable.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// On-disk format version written by `save`. Bump only together with a
/// reader branch that keeps v1 files loadable.
pub const CHECKPOINT_FORMAT_VERSION: u32 = 1;

/// Magic prefix of every checkpoint file.
pub const CHECKPOINT_MAGIC: &[u8; 8] = b"TRIOSCKP";

/// Absolute byte length of the fixed header (= offset of the tensor directory).
pub const CHECKPOINT_HEADER_LEN: usize = 152;

/// Number of tensors in the canonical order. Fixed for format version 1.
pub const CHECKPOINT_TENSOR_COUNT: usize = 19;

/// Absolute byte offset at which the payload starts.
pub const CHECKPOINT_PAYLOAD_OFFSET: usize = CHECKPOINT_HEADER_LEN + CHECKPOINT_TENSOR_COUNT * 8;

/// Where the artifact landed and what it hashes to.
#[derive(Debug, Clone)]
pub struct SavedCheckpoint {
    pub path: PathBuf,
    /// Lowercase hex SHA-256 of the file on disk. Equal to `shasum -a 256`.
    pub sha256: String,
    pub bytes: u64,
}

/// Exponential Moving Average (EMA) of checkpoints
///
/// Computes post-hoc EMA over the last N checkpoints for better generalization.
/// Reference: Sanyal et al. 2024 "A Simple and Strong Baseline for Model Averaging"
pub struct EmaAverager {
    /// Number of checkpoints to average
    pub n: usize,

    /// Current buffer of checkpoint values
    pub buffer: Vec<f32>,

    /// Current buffer index
    pub idx: usize,
}

impl EmaAverager {
    /// Create a new EMA averager
    ///
    /// # Arguments
    ///
    /// * `n` - Number of checkpoints to average
    ///
    /// # Returns
    ///
    /// A new EMA averager with empty buffer
    pub fn new(n: usize) -> Self {
        Self {
            n,
            buffer: Vec::with_capacity(n),
            idx: 0,
        }
    }

    /// Update EMA with a new checkpoint value
    ///
    /// # Arguments
    ///
    /// * `value` - New checkpoint value (BPB or loss)
    ///
    /// # Returns
    ///
    /// The EMA-averaged value
    pub fn update(&mut self, value: f64) -> f32 {
        if self.buffer.len() < self.n {
            self.buffer.push(value as f32);
            self.idx = self.buffer.len();
        } else {
            // Rotate buffer when full
            self.buffer[self.idx] = value as f32;
            self.idx = (self.idx + 1) % self.n;
        }

        // Compute EMA
        self.ema_average()
    }

    /// Compute current EMA average
    ///
    /// # Returns
    ///
    /// The exponential moving average of buffered values
    pub fn ema_average(&self) -> f32 {
        if self.buffer.is_empty() {
            0.0
        } else {
            let n = self.buffer.len() as f64;
            // Simple average (can use exponential weights if needed)
            self.buffer.iter().sum::<f32>() / (n as f32)
        }
    }

    /// Compute EMA over N most recent checkpoints
    ///
    /// # Arguments
    ///
    /// * `checkpoints` - Slice of checkpoint values (BPB or loss)
    /// * `n` - Number of checkpoints to average
    ///
    /// # Returns
    ///
    /// The EMA-averaged value
    pub fn ema_average_over(checkpoints: &[f32], n: usize) -> f64 {
        if checkpoints.is_empty() || n == 0 {
            return 0.0;
        }
        let take_n = n.min(checkpoints.len());
        let avg = checkpoints[..take_n].iter().sum::<f32>() / take_n as f32;
        avg as f64
    }

    /// Reset EMA state
    ///
    /// Clears the buffer and resets index
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.idx = 0;
    }
}

/// Directory component that separates one seed of a multi-seed sweep from the
/// next. Generated from a `u64` in code, never from the environment, so unlike
/// the run name it needs no sanitisation.
pub fn seed_scope_component(seed: u64) -> String {
    format!("seed{seed}")
}

/// Directory the artifacts of one run land in, `{run-name}` or
/// `{run-name}/seed{seed}`.
///
/// `seed_scope` is `None` for a single-seed run: that layout is the one the
/// README and `ckpt_replay` point at, and it does not move.
///
/// It is `Some(seed)` on the sweep path. `TRIOS_CANON_NAME` is what the scarab
/// and the Railway workers set, and it does not vary with the seed, so three
/// sweep seeds resolved to ONE `{step}.bin`: each `save` atomically renamed
/// over the previous one, three different hashes were printed, three `DONE:`
/// lines were printed, and one file survived. A three-seed claim was standing
/// on one seed's evidence.
pub fn run_dir(run_name: &str, seed_scope: Option<u64>) -> PathBuf {
    let base = std::env::var("TRIOS_CHECKPOINT_DIR").unwrap_or_else(|_| "checkpoints".into());
    let dir = PathBuf::from(base).join(run_name);
    match seed_scope {
        Some(seed) => dir.join(seed_scope_component(seed)),
        None => dir,
    }
}

pub fn checkpoint_path(run_name: &str, step: usize) -> PathBuf {
    checkpoint_path_scoped(run_name, None, step)
}

/// `checkpoint_path` with the sweep's per-seed subdirectory. See `run_dir`.
pub fn checkpoint_path_scoped(run_name: &str, seed_scope: Option<u64>, step: usize) -> PathBuf {
    run_dir(run_name, seed_scope).join(format!("{step}.bin"))
}

/// Map a run name to a filesystem-safe directory component.
///
/// `canon_name` comes from `TRIOS_CANON_NAME`, which is env-controlled, so a
/// value containing `/` or `..` would otherwise escape the checkpoint dir via
/// `PathBuf::join`. Anything outside `[A-Za-z0-9._-]` becomes `_`; the results
/// "." and ".." become "_". The *unsanitized* name is what goes in the ledger.
pub fn sanitize_run_name(run: &str) -> String {
    let mut out: String = run
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if out.is_empty() || out == "." || out == ".." {
        out = "_".to_string();
    }
    out
}

/// Lowercase hex SHA-256. Same digest as `shasum -a 256`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().fold(String::with_capacity(64), |mut acc, b| {
        use std::fmt::Write as _;
        let _ = write!(acc, "{b:02x}");
        acc
    })
}

/// Write `bytes` to `checkpoint_path(sanitize_run_name(run), step)` atomically,
/// then hash the file as it exists on disk.
///
/// Sequence, in this exact order:
///   1. `fs::create_dir_all` the parent
///   2. write `{step}.bin.tmp.{pid}` in that same directory
///   3. `File::sync_all` before close
///   4. `fs::rename` tmp -> `{step}.bin` (atomic within one directory on POSIX)
///   5. best-effort parent-directory fsync (errors ignored - not portable, and
///      the rename already ordered the data)
///   6. re-read the FINAL path and SHA-256 it
///
/// Hashing the file rather than the buffer is what makes the digest
/// reproducible by an outside party with `shasum -a 256`. An interrupted run
/// leaves only a `.tmp.{pid}` file, which readers never accept: only a name
/// matching `{step}.bin` exactly is a checkpoint.
///
/// Step 0 of the sequence, before any of the above: if the final path already
/// holds DIFFERENT bytes, refuse. See `save_scoped`.
pub fn save(run: &str, step: usize, bytes: &[u8]) -> Result<SavedCheckpoint> {
    save_scoped(run, None, step, bytes)
}

/// `save` into the sweep's per-seed subdirectory. See `run_dir`.
///
/// # Refusal to overwrite
///
/// The `fs::rename` in step 4 is atomic, which also means it silently replaces
/// whatever was there. Three sweep seeds writing one path therefore destroyed
/// two artifacts and reported all three as landed. A save whose final path
/// already holds a DIFFERENT digest is now an error naming both hashes and the
/// path: an artifact that exists is evidence, and this function does not get to
/// delete evidence in order to report success.
///
/// Re-saving IDENTICAL bytes is allowed and is not an overwrite: the file that
/// would result is the file that is already there.
pub fn save_scoped(
    run: &str,
    seed_scope: Option<u64>,
    step: usize,
    bytes: &[u8],
) -> Result<SavedCheckpoint> {
    let final_path = checkpoint_path_scoped(&sanitize_run_name(run), seed_scope, step);
    if let Ok(existing) = std::fs::read(&final_path) {
        let existing_sha = sha256_hex(&existing);
        let incoming_sha = sha256_hex(bytes);
        anyhow::ensure!(
            existing_sha == incoming_sha,
            "refusing to overwrite checkpoint {final_path:?}: it already holds \
             sha256={existing_sha} ({} bytes) and this save carries \
             sha256={incoming_sha} ({} bytes). Two runs are writing one path - \
             on a sweep, give each seed its own directory (see \
             `checkpoint::run_dir`); otherwise set TRIOS_CANON_NAME or \
             TRIOS_CHECKPOINT_DIR per run. Nothing was written and nothing was \
             deleted.",
            existing.len(),
            bytes.len()
        );
    }
    let dir = final_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("checkpoint path {final_path:?} has no parent"))?
        .to_path_buf();
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create checkpoint dir {dir:?}"))?;

    let tmp_path = dir.join(format!("{step}.bin.tmp.{}", std::process::id()));
    {
        let mut f = std::fs::File::create(&tmp_path)
            .with_context(|| format!("failed to create {tmp_path:?}"))?;
        f.write_all(bytes)
            .with_context(|| format!("failed to write {tmp_path:?}"))?;
        f.sync_all()
            .with_context(|| format!("failed to fsync {tmp_path:?}"))?;
    }
    std::fs::rename(&tmp_path, &final_path)
        .with_context(|| format!("failed to rename {tmp_path:?} -> {final_path:?}"))?;
    // Best-effort: fsync the directory so the rename itself is durable. Not
    // portable (fails on some platforms), and the data write above is already
    // ordered before the rename, so a failure here is not an error.
    if let Ok(dir_handle) = std::fs::File::open(&dir) {
        let _ = dir_handle.sync_all();
    }

    let on_disk = std::fs::read(&final_path)
        .with_context(|| format!("failed to re-read checkpoint {final_path:?}"))?;
    Ok(SavedCheckpoint {
        sha256: sha256_hex(&on_disk),
        bytes: on_disk.len() as u64,
        path: final_path,
    })
}

/// Read a checkpoint file back. Errors if absent or unreadable. Performs NO
/// format validation - that is `HybridModel::from_checkpoint_bytes`.
pub fn load(run: &str, step: usize) -> Result<Vec<u8>> {
    let path = checkpoint_path(&sanitize_run_name(run), step);
    std::fs::read(&path).with_context(|| format!("failed to read checkpoint {path:?}"))
}

/// Sidecar path: `${TRIOS_CHECKPOINT_DIR:-checkpoints}/{run}/{step}.json`.
///
/// Like `checkpoint_path`, this does NOT sanitize `run_name`; callers pass the
/// already-sanitized component (`write_sidecar` does).
pub fn sidecar_path(run_name: &str, step: usize) -> PathBuf {
    sidecar_path_scoped(run_name, None, step)
}

/// `sidecar_path` with the sweep's per-seed subdirectory, so the record always
/// lands in the same directory as the `.bin` it describes. See `run_dir`.
pub fn sidecar_path_scoped(run_name: &str, seed_scope: Option<u64>, step: usize) -> PathBuf {
    run_dir(run_name, seed_scope).join(format!("{step}.json"))
}

/// One corpus stream, identified well enough for a third party to obtain the
/// same bytes and confirm it. `sha256` is over the raw file, so the check needs
/// no knowledge of this crate: `shasum -a 256 <path>`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CorpusStream {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

impl CorpusStream {
    /// Describe a corpus file. A stream that cannot be read is recorded with
    /// an empty hash rather than omitted: an absent field reads as "not
    /// applicable", an empty one reads as "we tried and could not".
    pub fn describe(path: &str) -> Self {
        match std::fs::read(path) {
            Ok(raw) => Self {
                path: path.to_string(),
                bytes: raw.len() as u64,
                sha256: sha256_hex(&raw),
            },
            Err(_) => Self {
                path: path.to_string(),
                bytes: 0,
                sha256: String::new(),
            },
        }
    }
}

/// Which text the run trained on and which text its BPB was measured against.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CorpusProvenance {
    pub train: CorpusStream,
    pub val: CorpusStream,
}

/// How a `git_sha` was obtained. The strength of the evidence is recorded, not
/// only its value.
pub const GIT_PROVENANCE_VERIFIED: &str = "verified-local";
/// `GIT_SHA` was exported by the environment and taken on trust. No tree was
/// inspected, so `git_dirty` is UNKNOWN on this path.
pub const GIT_PROVENANCE_ASSERTED: &str = "asserted-by-environment";
/// Neither `GIT_SHA` nor a working `git` was available.
pub const GIT_PROVENANCE_NONE: &str = "unavailable";

/// Resolve the commit that produced this artifact.
///
/// `GIT_SHA` wins when set (CI and the Docker image export it). Otherwise ask
/// git directly, because an empty field in an evidence record is worse than a
/// missing one: it looks authoritative and says nothing.
///
/// Returns `(sha, dirty, provenance)`. `dirty` is `None` when no tree was
/// inspected. The previous signature returned `dirty = false` unconditionally
/// on the `GIT_SHA` path, so any environment that exported the variable made
/// every artifact assert a clean tree without a tree ever being checked - an
/// unverified claim printed in the same field as a verified one.
pub fn resolve_git_provenance() -> (String, Option<bool>, &'static str) {
    if let Ok(sha) = std::env::var("GIT_SHA") {
        if !sha.is_empty() {
            return (sha, None, GIT_PROVENANCE_ASSERTED);
        }
    }
    let sha = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    // `None` when `git status` could not be run: a failed query is not a clean
    // tree, and recording it as one is the defect this replaces.
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty());
    if sha.is_empty() && dirty.is_none() {
        return (sha, None, GIT_PROVENANCE_NONE);
    }
    (sha, dirty, GIT_PROVENANCE_VERIFIED)
}

/// Whether the working tree carries files git is not tracking.
///
/// A SIBLING of `git_dirty`, deliberately NOT folded into it. `git_dirty` has
/// one fixed meaning on every record already on disk - "a TRACKED file was
/// modified" - and widening it here would silently change what those records
/// say without any of them being rewritten.
///
/// It exists because `git_dirty` answers a NARROWER question than the one a
/// reader asks of it. `resolve_git_provenance` runs
/// `git status --porcelain --untracked-files=no`, while
/// `collect_source_digest_inputs` walks `src/**/*.rs` off the FILESYSTEM: an
/// UNTRACKED `.rs` under `src/` is compiled into the binary and changes
/// `source_sha256`, and git is asked not to mention it. A record could
/// therefore read `git_sha: <commit>`, `git_provenance: verified-local`,
/// `git_dirty: false` for a tree that is not that commit. This is not a
/// hypothetical class: the repository this note was written in had 16 untracked
/// entries live at the time.
///
/// `--untracked-files=normal`, not `=all`: `normal` honours `.gitignore`, so
/// `checkpoints/` and `.cargo/config.toml` - gitignored by construction, and
/// the second one already a hashed digest input in its own right - do not make
/// every run report an untracked tree. Only `??` lines are counted, so the
/// answer stays a different fact from `git_dirty` rather than a superset of it.
///
/// `None` when the query could not be run, on exactly the rule `git_dirty`
/// follows: a failed query is not a clean tree.
pub fn resolve_git_untracked() -> Option<bool> {
    let out = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    Some(
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .any(|line| line.starts_with("??")),
    )
}

/// `RUSTC_VERSION` was exported into the compilation, so the string names the
/// compiler that actually built this binary.
pub const TOOLCHAIN_PROVENANCE_BUILD: &str = "compile-time-env";
/// `rustc --version` was run from `PATH` at save time. The string names a
/// compiler present on the RUNNING host, which is not proof that it is the one
/// that built this binary - a released image may carry a different rustc, or
/// none. Weaker evidence than `compile-time-env`, recorded as such.
pub const TOOLCHAIN_PROVENANCE_PATH: &str = "runtime-path-query";
/// Neither route yielded a toolchain string.
pub const TOOLCHAIN_PROVENANCE_NONE: &str = "unavailable";

/// Recorded where a value could not be established honestly on this platform.
/// Deliberately not an empty string and not a plausible-looking guess.
pub const PROVENANCE_UNDETERMINED: &str = "undetermined";

/// `libc_version` was parsed from `ldd --version` run at save time. Read the
/// LIMIT on `resolve_libc_version` before citing it.
pub const LIBC_PROVENANCE_LDD: &str = "ldd-version-runtime";
/// `libc_version` is the Darwin kernel release from `uname -r`, run at save
/// time. There is no glibc on this target; the Darwin release is what names the
/// libSystem/libm the run linked against.
pub const LIBC_PROVENANCE_DARWIN_RELEASE: &str = "uname-release-runtime";
/// No version query is implemented for this target (musl, msvc, ...). The
/// dimension is unanswered, and says so.
pub const LIBC_PROVENANCE_UNSUPPORTED: &str = "no-query-for-this-target";
/// The query for this target exists and was run, and produced nothing usable.
pub const LIBC_PROVENANCE_NONE: &str = "unavailable";

/// `(libc_version, libc_provenance)` for the host writing this record.
///
/// This is the dimension the flagship negative result turns on. The same seed,
/// the same corpus and a byte-identical compiler produced different checkpoint
/// hashes on x86_64 glibc and on aarch64 macOS, and libm differences are one of
/// the two candidate causes - yet the record named only the libc FAMILY
/// (`"gnu"`, `"undetermined"`), which cannot tell glibc 2.28 from glibc 2.39
/// and says nothing at all about the Apple arm of the experiment.
///
/// Never fabricated: a failed query yields `None` and a provenance value that
/// states which way it failed. `None` is not "no libc".
///
/// LIMIT. Both routes are RUN-TIME reads of the HOST, the same class of
/// evidence as `TOOLCHAIN_PROVENANCE_PATH`, and weaker than a compile-time
/// constant would be. `ldd` is a glibc-shipped script, so it reports the glibc
/// of the installation it came from rather than being read out of the loaded
/// image of this process; on an ordinary host those are the same libc, and on a
/// host where they are not, this field describes the installation. A statically
/// linked binary carried onto a different machine would be described by the
/// machine, not by itself.
fn resolve_libc_version() -> (Option<String>, &'static str) {
    // `cfg!` is compile-time in every branch, so the ROUTE is a property of the
    // binary even though the value it reads is not.
    if cfg!(target_env = "gnu") {
        return match first_line_of_command("ldd", &["--version"])
            .as_deref()
            .and_then(parse_libc_version_line)
        {
            Some(v) => (Some(v), LIBC_PROVENANCE_LDD),
            None => (None, LIBC_PROVENANCE_NONE),
        };
    }
    if cfg!(any(target_os = "macos", target_os = "ios")) {
        return match first_line_of_command("uname", &["-r"]).map(|l| l.trim().to_string()) {
            Some(v) if !v.is_empty() => (Some(v), LIBC_PROVENANCE_DARWIN_RELEASE),
            _ => (None, LIBC_PROVENANCE_NONE),
        };
    }
    (None, LIBC_PROVENANCE_UNSUPPORTED)
}

/// First stdout line of `program args`, or `None` if it could not be run.
fn first_line_of_command(program: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .map(|l| l.to_string())
}

/// The version token of an `ldd --version` banner, e.g.
/// `ldd (Ubuntu GLIBC 2.39-0ubuntu8.2) 2.39` -> `2.39`.
///
/// Pure and split out so the parse is testable without a glibc host. Returns
/// `None` rather than a guess when the last token does not begin with a digit:
/// an unrecognised banner is an unanswered question, and this module records
/// those as unanswered.
fn parse_libc_version_line(line: &str) -> Option<String> {
    // `split_whitespace` already ignores the surrounding whitespace, so no
    // `trim()` here.
    let last = line.split_whitespace().next_back()?;
    last.starts_with(|c: char| c.is_ascii_digit())
        .then(|| last.to_string())
}

/// The platform half of the provenance record, absent from schemas 1 and 2.
///
/// This is not bookkeeping. The cross-libc experiment showed the SAME seed,
/// the SAME corpus and the SAME source tree producing DIFFERENT checkpoint
/// hashes on macOS versus glibc, which makes the platform the single decisive
/// untracked variable in the whole method: two records could agree on every
/// field schema 2 carried and still describe runs that could never reproduce
/// each other.
///
/// Every field except `toolchain` and `source_digest_scope` is resolved at
/// COMPILE time, so it describes the binary rather than whatever host happens
/// to read the record later. Those two are run-time reads, and each says so in
/// its own documentation.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PlatformProvenance {
    /// `std::env::consts::OS`: "macos", "linux", ...
    pub os: String,
    /// `std::env::consts::ARCH`: "aarch64", "x86_64", ...
    pub arch: String,
    /// `usize::BITS`. Compile-time, so it is a property of the binary.
    pub pointer_width: u32,
    /// Compile-time `target_env`: "gnu", "musl" or "msvc".
    ///
    /// `"undetermined"` on Apple targets, where `target_env` is empty: the
    /// compiler is told no libc identity there, and probing the running host
    /// would describe the machine reading the record rather than the one that
    /// linked the binary. An undetermined libc is recorded as undetermined
    /// rather than guessed, even though this is the field the cross-libc
    /// experiment turned on.
    ///
    /// A FAMILY, never a version. Schema 8 adds `libc_version` next to it,
    /// because "gnu" cannot tell glibc 2.28 from glibc 2.39 and the two arms of
    /// the cross-architecture experiment recorded only "gnu" and
    /// "undetermined".
    pub libc: String,
    /// Best-effort rustc identity. Read `toolchain_provenance` before citing
    /// it; `"unknown"` when neither route produced a string.
    pub toolchain: String,
    /// `TOOLCHAIN_PROVENANCE_*`. A value and the strength of the evidence for
    /// it are different facts, and are recorded separately - the same rule
    /// `git_provenance` follows.
    pub toolchain_provenance: String,
    /// WHICH tree `source_sha256` was walked in, as a CLASSIFICATION:
    /// `SOURCE_DIGEST_SCOPE_REPO_ROOT`, `other:<sha256 of the path>`, or
    /// `SOURCE_DIGEST_SCOPE_NONE`. See `resolve_source_digest_scope`.
    ///
    /// `source_sha256` hashes RELATIVE paths, so it is undefined without the
    /// directory they resolved against. A run in a throwaway clone and a run in
    /// the repository both produce a well-formed 64-hex digest, and until this
    /// field the record could not tell them apart - which is exactly the case
    /// the clean-tree procedure in `docs/CLEAN-TREE-PROVENANCE.md` creates.
    ///
    /// Schema 6 recorded the ABSOLUTE PATH here, so every locally produced
    /// sidecar published the builder's home directory - in the same artifact
    /// set that reports 0 such paths surviving into the binary. Schema 7
    /// records the classification, which answers the same question and names no
    /// path.
    ///
    /// `#[serde(default)]`, so every earlier sidecar still deserializes. An
    /// ABSENT scope is not "the repository root": it means the record predates
    /// this field and is silent about where it was measured. Decide by field
    /// presence, the same rule the schema 3-7 blocks state.
    #[serde(default)]
    pub source_digest_scope: String,

    // ---- schema 7 additions -------------------------------------------
    /// Where the persisted build flags came from:
    /// `BUILD_FLAGS_SOURCE_CARGO_CONFIG`, or `None` when no flags file was
    /// found. See `resolve_build_flags_provenance` and read its LIMIT.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rustflags_source: Option<String>,
    /// SHA-256 over the bytes of that flags file. The CONTENT IS NEVER
    /// EMBEDDED: it holds the builder's expanded `$HOME`, `$CARGO_HOME` and
    /// `$RUSTUP_HOME`. `None` exactly when `rustflags_source` is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rustflags_sha256: Option<String>,
    /// True when that flags file asks for `--remap-path-prefix`.
    ///
    /// This is the input that DECIDES `trainer.sha256`, and until schema 7 no
    /// field named it: a `TRAINER MISMATCH` from an auditor who followed the
    /// documented `cargo build --release --locked` was indistinguishable from
    /// one caused by a substituted binary. Read as "the tree asked for
    /// remapping", not as "this binary was remapped" - see the LIMIT on
    /// `resolve_build_flags_provenance`.
    #[serde(default)]
    pub remap_applied: bool,

    // ---- schema 8 additions -------------------------------------------
    // Same rule as every additive block before it: `#[serde(default)]`, so a
    // schema 3-7 sidecar still deserializes, and a DEFAULTED value is not a
    // measurement. An absent `features` is not "no features"; an absent
    // `libc_version` is not "the libc had no version".
    /// The libc/system-library VERSION, where the family in `libc` is only a
    /// class. `None` when it could not be established - never a guess. Read
    /// `libc_provenance` before citing it, and read the LIMIT on
    /// `resolve_libc_version`.
    ///
    /// On Apple targets this is the DARWIN KERNEL RELEASE (`uname -r`, e.g.
    /// `25.5.0`), not a glibc version: there is no glibc there, and leaving the
    /// dimension blank was the defect. `libc_provenance` says which of the two
    /// this is, and `libc` stays `"undetermined"` because the compiler really
    /// is told no libc identity on that target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub libc_version: Option<String>,
    /// `LIBC_PROVENANCE_*`. A value and the strength of the evidence for it are
    /// different facts, and are recorded separately - the same rule
    /// `git_provenance` and `toolchain_provenance` follow.
    /// `LIBC_PROVENANCE_NONE` (the query ran and produced nothing) and
    /// `LIBC_PROVENANCE_UNSUPPORTED` (no query exists for this target) are
    /// different answers and are not merged.
    #[serde(default)]
    pub libc_provenance: String,
    /// The compiled feature set as `name=0|1` pairs, from
    /// `compiled_feature_set()`. Compile-time, so it describes the binary.
    ///
    /// It was already a `trios-source-tree/2` digest input and NO FIELD NAMED
    /// IT, so the digest could move for a reason the record did not state. An
    /// auditor who pulled the `:gf16` image published by `docker-publish.yml`
    /// met `TRAINER MISMATCH` with three printed causes, none of which was the
    /// real one - the two feature builds hash differently (`8a357f46...` vs
    /// `6a2874b9...`). See `compiled_feature_set` for what that measurement
    /// does and does not license.
    #[serde(default)]
    pub features: String,
}

/// Describe the platform that produced this artifact.
pub fn resolve_platform_provenance() -> PlatformProvenance {
    // `cfg!` is evaluated at compile time in every branch, so this reports the
    // target the binary was built for.
    let libc = if cfg!(target_env = "gnu") {
        "gnu"
    } else if cfg!(target_env = "musl") {
        "musl"
    } else if cfg!(target_env = "msvc") {
        "msvc"
    } else {
        PROVENANCE_UNDETERMINED
    };
    let (toolchain, toolchain_provenance) = resolve_toolchain();
    let (libc_version, libc_provenance) = resolve_libc_version();
    // Primes the shared digest cache if the trainer has not already asked for
    // `source_sha256`; the scope, the root and the flags then all describe the
    // same walk.
    let source_digest_scope = resolve_source_digest_scope();
    let (rustflags_source, rustflags_sha256, remap_applied) =
        resolve_build_flags_provenance(source_digest_root());
    PlatformProvenance {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        pointer_width: usize::BITS,
        libc: libc.to_string(),
        toolchain,
        toolchain_provenance: toolchain_provenance.to_string(),
        source_digest_scope,
        rustflags_source,
        rustflags_sha256,
        remap_applied,
        libc_version,
        libc_provenance: libc_provenance.to_string(),
        // The same string that enters `source_sha256` under
        // `SOURCE_DIGEST_FEATURES_KEY`, so the digest input is now also a
        // readable field instead of an unnamed reason for the digest to move.
        features: compiled_feature_set(),
    }
}

/// `(toolchain, provenance)`. This crate has no `build.rs`, so there is no
/// compiler-injected version constant; `CARGO_PKG_RUST_VERSION` is deliberately
/// NOT used, because it is the declared minimum supported version, not the
/// compiler that ran. Never returns a fabricated string.
fn resolve_toolchain() -> (String, &'static str) {
    if let Some(v) = option_env!("RUSTC_VERSION") {
        let v = v.trim();
        if !v.is_empty() {
            return (v.to_string(), TOOLCHAIN_PROVENANCE_BUILD);
        }
    }
    let probed = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());
    match probed {
        Some(v) => (v, TOOLCHAIN_PROVENANCE_PATH),
        None => ("unknown".to_string(), TOOLCHAIN_PROVENANCE_NONE),
    }
}

/// Written to `source_sha256` when no digest could be produced - typically
/// because the binary ran outside its source tree, so there was nothing to
/// hash. A sentinel, never an approximation.
pub const SOURCE_DIGEST_NOT_COMPUTED: &str = "not-computed";

/// Domain separator and version tag of the source-tree digest.
///
/// `/2` widened the input set (`Cargo.lock`, `rust-toolchain.toml`,
/// `migration/src/**/*.rs`, the compiled feature set) and added a presence flag
/// to the per-entry encoding. A `/1` and a `/2` digest over the same tree
/// therefore differ, which is the point: they are not the same claim, and the
/// domain tag is what stops the two being compared as if they were.
///
/// `/3` adds `.cargo/config.toml`, the file that decides `trainer.sha256`. It
/// is the same widening for the same reason, so the tag moves for the same
/// reason: a `/2` and a `/3` digest over one unchanged tree differ, and
/// comparing them would report a source change that did not happen.
pub const SOURCE_DIGEST_DOMAIN: &[u8] = b"trios-source-tree/3\n";

/// Key under which the compiled feature set enters the digest. Not a path; `<`
/// cannot appear in one this walk produces, so it cannot collide with a file.
pub const SOURCE_DIGEST_FEATURES_KEY: &str = "<features>";

/// The persisted build flags, relative to the walk scope.
///
/// `scripts/repro_build.sh` writes `[build] rustflags = [...]` here so that a
/// plain `cargo build` in this checkout compiles with the same
/// `--remap-path-prefix` set as the script. The file is gitignored on purpose:
/// it holds this host's expanded `$HOME`, `$CARGO_HOME` and `$RUSTUP_HOME`, and
/// committing it would hand the next laboratory a config describing a home that
/// does not exist there.
pub const BUILD_FLAGS_PATH: &str = ".cargo/config.toml";

/// `rustflags_source` when `.cargo/config.toml` was read at save time.
pub const BUILD_FLAGS_SOURCE_CARGO_CONFIG: &str = "cargo-config-toml";

/// The flag `scripts/repro_build.sh` persists, and the only one whose presence
/// this module reports. Matched as a substring of the file, which is enough to
/// say the file asks for remapping and deliberately not enough to say which
/// prefixes: those are absolute paths and never leave this process.
const BUILD_FLAGS_REMAP_MARKER: &str = "--remap-path-prefix";

/// `(rustflags_source, rustflags_sha256, remap_applied)` read from
/// `.cargo/config.toml` in the digest scope.
///
/// The flags file is the deciding input behind `trainer.sha256` and until now
/// no field named it, so a `TRAINER MISMATCH` could not be told apart from a
/// tampered binary. `None`/`None`/`false` means the file was absent, which is
/// the state of every fresh clone.
///
/// The CONTENT IS HASHED, NEVER COPIED. The file contains this host's expanded
/// `$HOME`, `$CARGO_HOME` and `$RUSTUP_HOME`; embedding it verbatim would put
/// back the 588 absolute paths `scripts/repro_build.sh` exists to remove, into
/// the very document that reports their removal. Two builders who ran the
/// script from the same layout agree on the hash; two who did not, do not.
///
/// LIMIT, stated because the fields would otherwise over-promise. This is a
/// RUN-TIME read of the working directory, exactly like `source_sha256`, and it
/// carries the same caveat: it describes the flags file that was underfoot when
/// the checkpoint was written, NOT the flags the running binary was compiled
/// with. This crate has no `build.rs`, so nothing stamps the real compilation
/// flags into the executable. `remap_applied: true` therefore reads "the tree
/// asked for remapping", not "this binary was remapped".
fn resolve_build_flags_provenance(root: Option<&Path>) -> (Option<String>, Option<String>, bool) {
    let path = match root {
        Some(dir) => dir.join(BUILD_FLAGS_PATH),
        None => PathBuf::from(BUILD_FLAGS_PATH),
    };
    match std::fs::read(&path) {
        Ok(raw) => {
            let remap = String::from_utf8_lossy(&raw).contains(BUILD_FLAGS_REMAP_MARKER);
            (
                Some(BUILD_FLAGS_SOURCE_CARGO_CONFIG.to_string()),
                Some(sha256_hex(&raw)),
                remap,
            )
        }
        Err(_) => (None, None, false),
    }
}

/// Recorded as the scope when the walk started at the git toplevel.
///
/// A CLASSIFICATION, not a path. The previous value was the builder's absolute
/// working directory, so every locally produced sidecar published
/// `/Users/<user>/...` - and those sidecars are uploaded as CI artifacts. The
/// fact the field exists to carry is WHICH TREE the relative digest ranged
/// over, and "the repository root" answers that without naming a home
/// directory.
pub const SOURCE_DIGEST_SCOPE_REPO_ROOT: &str = "repository-root";

/// Prefix of the scope recorded when the walk did NOT provably start at the git
/// toplevel: `other:<sha256 of the absolute path>`.
///
/// The hash is one-way, so the value still distinguishes two different
/// throwaway clones from each other and from the repository - which is the
/// whole job of the field - while publishing no path. "Not provably the
/// toplevel" includes the case where `git` could not be run at all: an
/// unanswered question is not a yes.
pub const SOURCE_DIGEST_SCOPE_OTHER_PREFIX: &str = "other:";

/// Classify `dir` as the repository root or as an opaque other tree.
fn classify_source_digest_scope(dir: &Path) -> String {
    let toplevel = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
        .filter(|p| !p.as_os_str().is_empty());
    if let Some(top) = toplevel {
        // Compared through `canonicalize` as well as literally, because on
        // macOS `/tmp` is a symlink and the two forms of the same directory
        // would otherwise classify differently.
        let same = top == dir
            || match (std::fs::canonicalize(&top), std::fs::canonicalize(dir)) {
                (Ok(a), Ok(b)) => a == b,
                _ => false,
            };
        if same {
            return SOURCE_DIGEST_SCOPE_REPO_ROOT.to_string();
        }
    }
    format!(
        "{SOURCE_DIGEST_SCOPE_OTHER_PREFIX}{}",
        sha256_hex(dir.to_string_lossy().replace('\\', "/").as_bytes())
    )
}

/// Every feature declared in `Cargo.toml` except `default`, which names no code
/// of its own. Kept in one place so a new feature is added to the digest by
/// editing this list rather than by remembering to.
fn declared_feature_states() -> [(&'static str, bool); 6] {
    // `cfg!` is evaluated at COMPILE time in every branch, so this describes
    // the binary that is running, not the manifest sitting on disk beside it.
    [
        ("ci-strict", cfg!(feature = "ci-strict")),
        ("gf16", cfg!(feature = "gf16")),
        ("gpu", cfg!(feature = "gpu")),
        ("race", cfg!(feature = "race")),
        ("smoke", cfg!(feature = "smoke")),
        ("trios-integration", cfg!(feature = "trios-integration")),
    ]
}

/// The compiled feature set as `name=0|1` pairs in declaration order, e.g.
/// `ci-strict=0,gf16=1,...`.
///
/// A digest input because it CAN change the binary, not because it is known to
/// change the result. The distinction matters, and the older wording here got
/// it wrong: it argued from `Cargo.toml:193` declaring `gf16` and `train_loop`
/// refusing to honour `GF16_ENABLED=true` without it that "two binaries built
/// from byte-identical sources can therefore behave differently", which reads
/// as a claim about outcomes and has since been FALSIFIED as one. The default
/// build and the `--features gf16` build hash differently (`8a357f46...` vs
/// `6a2874b9...`) and produced BYTE-IDENTICAL artifacts. So the feature set
/// decides the BINARY - which is exactly why it belongs in a digest whose job
/// is to describe what ran - and on the one comparison actually made it did not
/// decide the WEIGHTS. Neither half of that may be quoted without the other.
///
/// The practical consequence is a verdict, not a nicety: an auditor who pulled
/// the `:gf16` image published by `docker-publish.yml` hits `TRAINER MISMATCH`,
/// and until schema 8 no field named the feature set anywhere, so none of the
/// causes `ckpt_replay` printed was the real one. `PlatformProvenance::features`
/// now carries this string.
///
/// Every feature is listed with its state, not just the enabled ones, so adding
/// a feature changes the digest even when nobody turns it on.
pub fn compiled_feature_set() -> String {
    declared_feature_states()
        .iter()
        .map(|(name, on)| format!("{name}={}", u8::from(*on)))
        .collect::<Vec<_>>()
        .join(",")
}

/// One digest input: its key, and its bytes - or `None` when the input is
/// DECLARED but ABSENT. An absent input is hashed as absent rather than
/// skipped, so a tree missing `rust-toolchain.toml` cannot produce the same
/// digest as one that has it.
type SourceDigestInput = (String, Option<Vec<u8>>);

/// Digest the source tree that is running: every `src/**/*.rs`, every
/// `migration/src/**/*.rs`, `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`,
/// `.cargo/config.toml` and the compiled feature set, sorted by key, each
/// contributing `len(key) || key || present || len(bytes) || bytes` to one
/// SHA-256. The length prefixes make the encoding unambiguous, so no rename or
/// content shuffle can collide with another tree.
///
/// `Cargo.lock` is in because the lock pins the dependency graph the manifest
/// only ranges over; `rust-toolchain.toml` because its own header says a
/// project claiming reproducibility cannot leave its toolchain to chance, and
/// `cross-arch-repro.yml` names it as a held-fixed variable; `migration/src`
/// because it holds the schema of the evidence table itself.
///
/// `.cargo/config.toml` is in because that sentence about the toolchain is true
/// of it verbatim, and more sharply. `scripts/repro_build.sh` writes the
/// `--remap-path-prefix` flags there, and those flags DECIDE `trainer.sha256`:
/// `ckpt_replay` refuses to execute a binary whose hash differs from the
/// recorded one, so the file is a deciding input to the verdict. It is
/// gitignored by construction (it holds the builder's expanded `$HOME`), which
/// is exactly why hashing it matters: an auditor who clones at `git_sha` does
/// NOT receive it, builds with `cargo build --release --locked`, and gets a
/// different binary. Before this, that auditor saw `source_sha256` MATCH and
/// `trainer.sha256` mismatch, with nothing in the record explaining the gap.
/// Now the two digests differ too, which is the correct answer: the auditor's
/// tree really is not the builder's tree.
///
/// This covers TRACKED, DIRTY AND UNTRACKED files alike, which is the point:
/// `git_sha` with `git_dirty: true` names a commit the tree did not match, and
/// every archived checkpoint in this repo was produced by exactly such a tree.
///
/// Read the other side of that sentence carefully, because it is the trap this
/// walk sits on top of. `git_dirty: false` means NO TRACKED FILE WAS MODIFIED.
/// It has never meant "the tree matched the commit", and must not be read as
/// that here or anywhere else. This walk collects `src/**/*.rs` from the
/// FILESYSTEM, while `resolve_git_provenance` asks git with
/// `--untracked-files=no`: an untracked `.rs` under `src/` is compiled in and
/// changes this digest while `git_dirty` stays `false`. Schema 8 adds
/// `git_untracked` for exactly that gap; see `resolve_git_untracked`.
///
/// Computed once per process and cached, so all records from one run agree and
/// a long run does not re-walk the tree at every checkpoint.
///
/// LIMIT, stated because the field would otherwise over-promise. This is a
/// RUN-TIME read of the working directory. It is NOT the source the running
/// binary was compiled from, and widening the input set does not make it one:
/// appending a comment to a file here changes this digest while the checkpoint
/// stays byte-identical, and swapping the whole source tree under a
/// pre-compiled binary changes this digest without changing a single
/// instruction that executes. This crate has no `build.rs`, so nothing compiles
/// a digest of the sources INTO the binary; the binary -> source link stated as
/// unclosed on `TrainerProvenance` is still unclosed. What this field is: a
/// hash of the tree that was underfoot, strictly stronger than a commit hash
/// over a dirty tree, and now including the lock, the toolchain pin, the
/// migrations and the feature switches that the earlier version left out.
///
/// WHICH tree was underfoot is recorded separately, in
/// `PlatformProvenance::source_digest_scope`. The walk is relative to the
/// process working directory, so the digest alone does not say what it ranged
/// over: a run from `/tmp/cleantree` and a run from the repository produce
/// well-formed digests that are indistinguishable in shape. Read the two
/// fields together.
pub fn resolve_source_digest() -> String {
    source_digest_state().digest.clone()
}

/// WHICH tree `resolve_source_digest` walked, as a CLASSIFICATION:
/// `SOURCE_DIGEST_SCOPE_REPO_ROOT`, `other:<sha256 of the path>`, or
/// `SOURCE_DIGEST_SCOPE_NONE` when no digest was produced.
///
/// `source_sha256` is a walk of RELATIVE paths (`src`, `Cargo.toml`), so it is
/// undefined without the directory those paths were resolved against. Until
/// this field existed, a digest taken in a throwaway clone read exactly like a
/// digest taken in the repository, and the record could not tell an auditor
/// which tree the number described.
///
/// Schema 7 replaced the ABSOLUTE PATH this used to report. The path answered
/// the question at the cost of publishing the builder's home directory in every
/// sidecar - and those sidecars are uploaded as CI artifacts, while
/// `scripts/repro_build.sh` was busy proving the same home directory no longer
/// appears in the binary. The classification answers the same question: was
/// this the repository, or some other tree - and if some other tree, was it the
/// SAME other tree as the run next to it.
pub fn resolve_source_digest_scope() -> String {
    source_digest_state().scope.clone()
}

/// The absolute directory the digest walk started from, kept in-process only.
///
/// Never serialised - `resolve_source_digest_scope` publishes the
/// classification instead. This is what relative paths in the record (see
/// `resolve_trainer_provenance`) are made relative TO.
fn source_digest_root() -> Option<&'static Path> {
    source_digest_state().root.as_deref()
}

/// Recorded as the scope when `source_sha256` is `SOURCE_DIGEST_NOT_COMPUTED`,
/// or when the working directory could not be resolved. A sentinel, never a
/// path that was not actually walked.
pub const SOURCE_DIGEST_SCOPE_NONE: &str = "not-computed";

/// The digest and the directory it ranged over, resolved together exactly once.
///
/// One `OnceLock` rather than two, so the pair can never disagree: a scope
/// resolved by a separate cache could name a directory the digest was not taken
/// in if anything called `set_current_dir` between the two.
struct SourceDigestState {
    digest: String,
    scope: String,
    /// The directory the walk started from. In-process only; the record
    /// publishes `scope`, which names no path.
    root: Option<PathBuf>,
}

fn source_digest_state() -> &'static SourceDigestState {
    static CACHE: std::sync::OnceLock<SourceDigestState> = std::sync::OnceLock::new();
    CACHE.get_or_init(compute_source_digest)
}

fn compute_source_digest() -> SourceDigestState {
    // The working directory is read BEFORE the walk, so the recorded scope is
    // the directory the walk actually started from.
    let cwd = std::env::current_dir().ok();
    compute_source_digest_in(collect_source_digest_inputs().map(|inputs| (inputs, cwd)))
}

/// The pairing rule, split out so the sentinel branch is reachable in a test
/// without breaking the process working directory for every other test.
///
/// `None` means the tree could not be described. There is then no scope to
/// report either: naming a directory for a walk that failed would be exactly
/// the kind of plausible-looking value this module refuses elsewhere.
fn compute_source_digest_in(
    walked: Option<(Vec<SourceDigestInput>, Option<PathBuf>)>,
) -> SourceDigestState {
    match walked {
        Some((inputs, cwd)) => SourceDigestState {
            digest: digest_source_inputs(&inputs),
            scope: match cwd.as_deref() {
                Some(p) => classify_source_digest_scope(p),
                None => SOURCE_DIGEST_SCOPE_NONE.to_string(),
            },
            root: cwd,
        },
        None => SourceDigestState {
            digest: SOURCE_DIGEST_NOT_COMPUTED.to_string(),
            scope: SOURCE_DIGEST_SCOPE_NONE.to_string(),
            root: None,
        },
    }
}

/// `None` when the tree cannot be described honestly: no `src/**/*.rs`, no
/// `Cargo.toml`, or a file that vanished between the walk and the read.
fn collect_source_digest_inputs() -> Option<Vec<SourceDigestInput>> {
    let mut files: Vec<PathBuf> = Vec::new();
    if collect_rs_files(Path::new("src"), &mut files).is_err() || files.is_empty() {
        return None;
    }
    if !Path::new("Cargo.toml").is_file() {
        return None;
    }
    // `migration` is a separate crate in this repo and may legitimately be
    // absent from a stripped image. Absent is recorded as absent below, never
    // skipped.
    let mut migration: Vec<PathBuf> = Vec::new();
    let migration_walked = collect_rs_files(Path::new("migration/src"), &mut migration).is_ok();
    if migration_walked {
        files.extend(migration);
    }

    let mut inputs: Vec<SourceDigestInput> = Vec::new();
    for path in &files {
        let raw = match std::fs::read(path) {
            Ok(raw) => raw,
            // A file that vanished mid-walk means the digest would describe a
            // tree that never existed. Refuse rather than hash a subset.
            Err(_) => return None,
        };
        inputs.push((path.to_string_lossy().replace('\\', "/"), Some(raw)));
    }
    if !migration_walked {
        inputs.push(("migration/src".to_string(), None));
    }
    for name in [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        // The build flags that decide `trainer.sha256`. Gitignored, so an
        // auditor's clone lacks it; ABSENT and PRESENT must therefore not
        // digest alike, which the `Option` encoding in `digest_source_inputs`
        // guarantees.
        BUILD_FLAGS_PATH,
    ] {
        // `Cargo.toml` was proven present above; the others are declared
        // inputs that a stripped tree may lack, and their absence is a fact
        // about that tree, so it is hashed instead of ignored.
        inputs.push((name.to_string(), std::fs::read(name).ok()));
    }
    inputs.push((
        SOURCE_DIGEST_FEATURES_KEY.to_string(),
        Some(compiled_feature_set().into_bytes()),
    ));
    inputs.sort();
    Some(inputs)
}

fn digest_source_inputs(inputs: &[SourceDigestInput]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(SOURCE_DIGEST_DOMAIN);
    for (key, body) in inputs {
        hasher.update((key.len() as u64).to_le_bytes());
        hasher.update(key.as_bytes());
        match body {
            Some(raw) => {
                hasher.update([1u8]);
                hasher.update((raw.len() as u64).to_le_bytes());
                hasher.update(raw);
            }
            // Present-and-empty and absent must not encode identically: the
            // first is a file, the second is the lack of one.
            None => {
                hasher.update([0u8]);
                hasher.update(0u64.to_le_bytes());
            }
        }
    }
    format!("{:x}", hasher.finalize())
}

/// Collect `*.rs` under `dir`, recursively, in a deterministic order.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_rs_files(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// Recorded in `TrainerProvenance::provenance` when the running executable was
/// found and hashed from disk.
pub const TRAINER_PROVENANCE_SELF_HASHED: &str = "self-hashed";

/// The executable that produced the artifact, named by its own SHA-256.
///
/// This exists because `ckpt_replay` computed the hash of the binary it was
/// about to re-execute and printed it WITHOUT comparing it to anything: the
/// record had no field to compare against. A twelve-line `/bin/sh` script that
/// copies one pre-baked file into `$TRIOS_CHECKPOINT_DIR` and performs no
/// arithmetic was graded `VERIFIED` against a genuine record. With this field
/// the verifier can refuse to execute a binary the record does not name.
///
/// LIMIT, stated because the field would otherwise over-promise: this hashes
/// the RUNNING EXECUTABLE, so it ties an artifact to a BINARY, not to a source
/// tree. This crate has no `build.rs`, so nothing compiles a digest of the
/// sources INTO the binary; the binary -> source link is still unclosed, and
/// `source_sha256` (a run-time walk of the working directory) does not close it
/// either. What this field does close is executor substitution: two records
/// naming different `trainer.sha256` were not produced by the same program.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TrainerProvenance {
    /// `std::env::current_exe()`, RELATIVE to the digest scope
    /// (`target/release/trios-train`), or `outside-scope:<file name>` when the
    /// executable did not live under the walked tree. Informational: the path
    /// is not evidence, the hash is - which is why schema 7 could shorten it.
    ///
    /// Schema 6 recorded the absolute path, so every locally produced sidecar
    /// published the builder's home directory. Nothing consumes this field as a
    /// location: `ckpt_replay` resolves the trainer from `--trainer` and only
    /// PRINTS `trainer.path` inside its mismatch message.
    pub path: String,
    /// Lowercase hex SHA-256 over the executable's bytes re-read from disk,
    /// the same discipline `save` uses. Empty only when `provenance` says why.
    pub sha256: String,
    /// `"self-hashed"`, or `"unavailable: <reason>"`. Never an empty string,
    /// which would read as a measurement that came out blank.
    pub provenance: String,
}

/// Hash the executable that is running.
///
/// Resolves `std::env::current_exe()`, reads the file and SHA-256s the BYTES
/// RE-READ FROM DISK - not an in-memory image - so an auditor can reproduce the
/// digest with `shasum -a 256 <path>`. Every failure path records the reason in
/// `provenance` rather than leaving an empty field that looks authoritative.
pub fn resolve_trainer_provenance() -> TrainerProvenance {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            return TrainerProvenance {
                path: String::new(),
                sha256: String::new(),
                provenance: format!("unavailable: current_exe failed: {e}"),
            }
        }
    };
    let path = scope_relative_path(&exe, source_digest_root());
    match std::fs::read(&exe) {
        Ok(raw) => TrainerProvenance {
            path,
            sha256: sha256_hex(&raw),
            provenance: TRAINER_PROVENANCE_SELF_HASHED.to_string(),
        },
        Err(e) => TrainerProvenance {
            path,
            sha256: String::new(),
            provenance: format!("unavailable: cannot read executable: {e}"),
        },
    }
}

/// Recorded in place of a path - as `trainer.path` or as
/// `CheckpointRecord::path` - when the file is not under the digest scope,
/// followed by its file name. A name is not a location.
///
/// Named `TRAINER_...` until the checkpoint path was given the same treatment;
/// one prefix, because a reader who learns the convention on one field should
/// not have to learn a second spelling of it on the next.
pub const PATH_OUTSIDE_SCOPE_PREFIX: &str = "outside-scope:";

/// `path` expressed relative to `root`, or `outside-scope:<file name>`.
///
/// Split out and pure so the three branches are testable without moving the
/// process working directory. Both the literal and the canonicalised form of
/// `root` are tried, because `current_exe` returns a symlink-resolved path on
/// macOS while `current_dir` need not.
///
/// An ALREADY-RELATIVE `path` is returned as it stands. Relative paths in this
/// process are resolved against the working directory, which is exactly the
/// directory `root` names (see `compute_source_digest`), so such a path is
/// already scope-relative and re-deriving it would only risk changing it. The
/// one exception is a path that climbs OUT of the scope with `..`, which is not
/// under the scope and is classified like any other outsider.
fn scope_relative_path(path: &Path, root: Option<&Path>) -> String {
    let climbs_out = path
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir));
    if path.is_relative() && !climbs_out {
        return path.to_string_lossy().replace('\\', "/");
    }
    if let Some(root) = root {
        let candidates = [Some(root.to_path_buf()), std::fs::canonicalize(root).ok()];
        for base in candidates.into_iter().flatten() {
            if let Ok(rel) = path.strip_prefix(&base) {
                return rel.to_string_lossy().replace('\\', "/");
            }
        }
    }
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    format!("{PATH_OUTSIDE_SCOPE_PREFIX}{name}")
}

/// The string `CheckpointRecord::path` carries for an artifact written to
/// `path`: relative to the digest scope, or `outside-scope:<file name>`.
///
/// Public because the value belongs to the record, not to the caller: the one
/// site that mints checkpoints (`train_loop::emit_checkpoint`) asks this
/// function rather than deciding for itself, so the sidecar and the ledger row
/// cannot spell one file two ways.
pub fn scope_relative_artifact_path(path: &Path) -> String {
    scope_relative_path(path, source_digest_root())
}

/// The four numbers the string `optimizer: "adamw"` was standing in for.
///
/// A record that names an optimizer FAMILY does not name the optimizer: AdamW
/// with beta1 = 0.9 and AdamW with beta1 = 0.618 are different training
/// recipes, and until schema 6 both serialised as the same five characters.
///
/// LIMIT - read this before citing the values. They are read from the code path
/// that actually stepped the weights, which for `trios-train` is the private
/// `AdamW` in `train_loop`, NOT `optimizer::AdamWCpu`. The two disagree:
/// `AdamWCpu` carries the phi-branded constants (beta1 = 1/phi = 0.618,
/// weight_decay = 1/phi^3 = 0.236) and `trios-train` never constructs it, so no
/// published `trios-train` BPB was produced with those constants. `source`
/// names which one was read so the distinction survives in the record instead
/// of living in this comment.
///
/// `weight_decay` is the coefficient as the code holds it; the two paths also
/// APPLY it differently (`train_loop::AdamW` does `p -= wd * lr * p`, decoupled
/// and lr-scaled, while `AdamWCpu` does `p -= wd * p`), so the coefficient is
/// only comparable across records with the same `source`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OptimizerParams {
    /// First-moment decay.
    pub beta1: f64,
    /// Second-moment decay.
    pub beta2: f64,
    /// Denominator epsilon.
    pub eps: f64,
    /// Decoupled weight-decay coefficient. See the LIMIT above on how it is
    /// applied - the number alone does not fix the update rule.
    pub weight_decay: f64,
    /// The code path these four numbers were read from, e.g.
    /// "train_loop::AdamW". Empty only in a defaulted (absent) record.
    pub source: String,
}

/// The local evidence record written next to every checkpoint, always,
/// regardless of whether a database is reachable.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointRecord {
    /// Literal `CHECKPOINT_RECORD_SCHEMA`. Named by the constant and NOT
    /// repeated as a literal here: this comment claimed "/8" while the constant
    /// said "/9", which is how a comment that jokes about having outlived one
    /// bump goes on to outlive a second. The tag is one `grep` away; a stale
    /// copy of it is worse than no copy.
    pub schema: String,
    /// Ledger identity, UNSANITIZED (may differ from the on-disk directory).
    pub canon_name: String,
    pub seed: i64,
    pub step: i64,
    /// The file that was written, RELATIVE to the digest scope, or
    /// `outside-scope:<file name>` when it was written outside that tree. See
    /// `scope_relative_artifact_path`, the one function that decides this.
    ///
    /// Same treatment, and the same reason, as `source_digest_scope` at schema
    /// 7 and `trainer.path` before it: an absolute path here published the
    /// BUILDER'S HOME DIRECTORY in every locally produced sidecar, and those
    /// sidecars are committed as evidence and uploaded as CI artifacts. Two
    /// tracked records still carry `/Users/<name>/...` in this field, in the
    /// same artifact set whose provenance script proves that string no longer
    /// appears in the binary.
    ///
    /// CONSEQUENCE, stated because it is a real one: this string is no longer a
    /// location an outside reader can open. Nothing depended on that. The
    /// artifact is found by `{step}.bin` beside the sidecar - which is what
    /// `ckpt_replay::locate_artifact` already falls back to, because a recorded
    /// absolute path from the training machine did not resolve on a copied
    /// evidence directory either - and the file is then IDENTIFIED by `sha256`,
    /// which is the field that was ever evidence.
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
    pub format_version: u32,
    pub hidden: u32,
    pub d_model: u32,
    pub num_attn_layers: u32,
    pub optimizer: String,
    pub fake_quant_format: String,
    pub data_synthetic: bool,
    /// Total steps the run was configured for. Schema 2: without it the reader
    /// cannot tell where `gf16_floor_step` (70% of `steps_total`) fell.
    pub steps_total: u64,
    /// Cadence of the in-place `gf16_floor()` weight rewrite past the 70% mark.
    /// Schema 2: this MUTATES the artifact, and until now nothing recorded it.
    pub gf16_floor_every: u64,
    /// Eval cadence. Schema 2: recorded as the OBSERVATION parameter it is, so
    /// a reader can confirm it did not enter the recipe.
    pub eval_every: u64,
    /// Raw `val_bpb` measured at this checkpoint's step. Schema 2 name for the
    /// old ambiguous `bpb`.
    pub final_val_bpb: Option<f64>,
    /// Minimum raw `val_bpb` observed in the run up to this step.
    ///
    /// Schema 6 renamed this from `best_val_bpb` (old sidecars still load via
    /// the serde alias). It is a MINIMUM OVER NOISY DRAWS, not a better
    /// measurement of the same quantity: `evaluate` has a measured spread of
    /// ~0.034 bpb on byte-identical weights (see `docs/EVAL-UNCERTAINTY.md`),
    /// so the minimum of k readings is biased low by construction and biased
    /// further the more often the run evaluated. It is NOT a peer of
    /// `final_val_bpb`, which is one declared reading at one declared step, and
    /// it must not be quoted as the run's result.
    #[serde(alias = "best_val_bpb")]
    pub min_observed_val_bpb: Option<f64>,
    /// Last EMA value. An early-stopping signal, NOT a measurement: it is
    /// seeded from `init_bpb` (~7.0) and lags the raw reading.
    pub ema_bpb: Option<f64>,
    /// GIT_SHA env if set, else `git rev-parse HEAD` of the working tree.
    /// "" only when neither is available.
    pub git_sha: String,
    /// How `git_sha` was obtained: "verified-local" (git was actually queried),
    /// "asserted-by-environment" (GIT_SHA was exported and taken on trust, no
    /// tree inspected) or "unavailable". A value and the strength of the
    /// evidence for it are different facts and are recorded separately.
    pub git_provenance: String,
    /// True when the working tree had uncommitted changes at save time, `None`
    /// when no tree was inspected. A clean-looking `git_sha` over a dirty tree
    /// does not describe the code that produced the artifact, and an
    /// UNINSPECTED tree is not a clean one - schema 1 reported both as `false`.
    pub git_dirty: Option<bool>,
    /// The corpus the run actually consumed. Without this the artifact cannot
    /// be tied to the data it was measured against, which is half of what a
    /// provenance record is for.
    pub corpus: CorpusProvenance,
    /// RAILWAY_DEPLOYMENT_ID env, ABSENT when unset or blank.
    ///
    /// Schema 7 turned this from `String` into `Option<String>`. Schema 1-6
    /// wrote `run_id: ""` on every non-Railway run - which is the exact
    /// ''-vs-NULL defect `neon_writer` argues against at length for the
    /// database side, where 16 of 17 live rows carried an empty string and
    /// `WHERE sha IS NULL` therefore found nothing. `neon_writer` fixed it with
    /// `env_nonempty`; the fix was never ported to the PRIMARY evidence
    /// document, which is this one.
    ///
    /// Blank ON READ is absent too, so a schema 1-6 sidecar carrying `""`
    /// deserializes to `None` rather than to "the deployment whose id is the
    /// empty string". That is the same rule `ckpt_replay::present()` already
    /// enforces on every field it checks.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "blank_string_as_none"
    )]
    pub run_id: Option<String>,
    /// "pending" | "written" | "skipped-no-dsn" | "skipped-not-opted-in" |
    /// "failed" | "rejected".
    ///
    /// Every value but `"pending"` is `LedgerWrite::as_str`; `"pending"` is the
    /// absence of an outcome written by the first of the two sidecar writes
    /// (see `LEDGER_PENDING`).
    ///
    /// `"skipped-not-opted-in"` was added 2026-08-03. Before it, a run with a
    /// DSN in the environment and no `TRIOS_LEDGER_WRITE=1` recorded
    /// `"skipped-no-dsn"` here -- a cause the record could not stand behind,
    /// since a ledger WAS configured and this run declined it. The two are
    /// different facts about the world and only one of them is "nothing
    /// outside this checkout was asked for a number".
    ///
    /// NOT a schema bump. `ledger` is a free-text `String` and neither its
    /// name nor its type changed; a reader that dispatches on field presence,
    /// which is what `interop/triosckp_reader.py` does and what every schema
    /// block above documents, sees an identical record shape. Bumping the tag
    /// for a new value in an existing field would make the tag mean "some
    /// string somewhere is new", which is not a version an instrument can act
    /// on. What DOES have to change is any reader that enumerates the values,
    /// and the independent reader now rejects a `ledger` it does not know
    /// instead of echoing it.
    pub ledger: String,
    /// RFC3339 UTC.
    pub ts: String,

    // ---- schema 3 additions -------------------------------------------
    // Every field below is `#[serde(default)]`, so a schema 1 or 2 sidecar
    // still deserializes. A DEFAULTED value is not a measurement: readers
    // must decide by FIELD PRESENCE in the JSON (or by `schema`) before
    // citing any of them. `interop/triosckp_reader.py` reports exactly that.
    /// The training learning rate actually used, widened from the `f32` the
    /// optimizer stepped with - so `--lr 0.003` is recorded as
    /// 0.003000000026077032, the exact value, not the decimal that was typed.
    /// `None` only when genuinely unknown (a schema 1 or 2 sidecar being read
    /// back); never 0.0, which is a legal learning rate and would be a lie.
    /// Until schema 3 two runs an order of magnitude apart in `lr` produced
    /// provenance records differing only in the hash and the BPB.
    #[serde(default)]
    pub lr: Option<f64>,
    /// The EFFECTIVE attention scale: the factor the attention block is
    /// multiplied by on its way into the residual stream. Env-overridable via
    /// `TRIOS_ATTN_SCALE` (default 0.1), so until schema 3 an exported
    /// environment variable silently changed the architecture and left no
    /// trace in the record. Taken from the run configuration, not re-read from
    /// the environment at write time.
    #[serde(default)]
    pub attn_scale: f64,
    /// The EFFECTIVE attention window. Env-overridable via `TRIOS_ATTN_SEQ`
    /// (default 8); same argument as `attn_scale`. This is the configured
    /// window, which the forward pass clamps to the available prefix.
    #[serde(default)]
    pub attn_seq: u64,
    /// The machine that produced the artifact. See `PlatformProvenance`.
    #[serde(default)]
    pub platform: PlatformProvenance,
    /// Digest over the source tree that ran (`trios-source-tree/3`, see
    /// `resolve_source_digest`), or the literal `"not-computed"`.
    ///
    /// `git_sha` with `git_dirty: true` is honest but NOT reconstructive: it
    /// names a commit the tree did not match. This field describes the code
    /// that ran even when that code was never committed - but read the LIMIT
    /// on `resolve_source_digest` before citing it: it is a run-time read of
    /// the working directory, not the source the binary was compiled from.
    ///
    /// Cite it together with `platform.source_digest_scope`, which CLASSIFIES
    /// the directory this walk was relative to - the repository root, or an
    /// opaque `other:<hash>` that still distinguishes two throwaway clones from
    /// each other. It deliberately names no path (schema 7; see
    /// `resolve_source_digest_scope`), so it bounds the digest without
    /// publishing the builder's home directory. The digest alone says neither.
    #[serde(default)]
    pub source_sha256: String,

    // ---- schema 4 additions -------------------------------------------
    // Same rule as the schema 3 block above: both fields are
    // `#[serde(default)]`, so a schema 1/2/3 sidecar still deserializes, and a
    // DEFAULTED value is not a measurement. Readers must decide by FIELD
    // PRESENCE (an absent `trainer.sha256` is not "the empty binary"; an absent
    // `vocab` is not "vocab 0").
    /// The executable that produced this artifact. See `TrainerProvenance`.
    ///
    /// Until schema 4 nothing in the record named the program that ran, so a
    /// verifier had no hash to compare the binary it was about to execute
    /// against, and any executable at all could be presented as the trainer.
    #[serde(default)]
    pub trainer: TrainerProvenance,
    /// The alphabet size the trainer folded the corpus onto (`VOCAB`).
    ///
    /// This is a first-order property of every number the run reports: with
    /// `vocab = 128` the corpus bytes are taken `% 128`, which is injective for
    /// ASCII and NOT injective for anything else, so a "bits-per-byte" figure
    /// measured on a non-ASCII corpus would be measured on a coarser alphabet
    /// than the one it names. Recorded so the reader can see which alphabet the
    /// figure belongs to instead of assuming 256.
    #[serde(default)]
    pub vocab: u32,

    // ---- schema 5 addition --------------------------------------------
    /// Whether the in-place `gf16_floor()` weight rewrite ran during this run.
    ///
    /// Same rule as the schema 3 and 4 blocks: `#[serde(default)]`, so an older
    /// sidecar still deserializes, and a defaulted `false` is NOT a
    /// measurement - a reader must decide by field presence.
    ///
    /// This is byte 124 of the HASHED header, i.e. it already decided the
    /// artifact's identity, and until schema 5 it appeared in no sidecar at
    /// all: a reader holding the JSON could not tell whether the weights had
    /// been floored. The record carried `gf16_floor_every` (how often) without
    /// carrying whether it happened at all.
    #[serde(default)]
    pub gf16_enabled: bool,

    // ---- schema 6 additions: the sampling plan and its uncertainty ----
    // Same rule as every block above: `#[serde(default)]`, so a schema 1-5
    // sidecar still deserializes, and an ABSENT field is not a zero. These are
    // SIDECAR-ONLY: none of them enters the hashed `TRIOSCKP` header, none is
    // required for grading, and a record without them is a record that did not
    // state its sampling plan - which is the whole defect, and is visible.
    /// Number of eval windows `evaluate` actually averaged for
    /// `final_val_bpb`. Until schema 6 this was the hardcoded literal 40 in
    /// `train_loop`, appearing in no field of this record: the headline was a
    /// mean over 5.16% of a 100,000-byte val corpus and nothing said so.
    #[serde(default)]
    pub eval_chunks: Option<u32>,
    /// Tokens the eval actually looked at (`eval_chunks * eval_seq`). Compare
    /// against `corpus.val.bytes` for the coverage fraction.
    #[serde(default)]
    pub eval_tokens: Option<u64>,
    /// Window length in tokens, `SEQ + 1` = 129. The windows sit on an evenly
    /// spaced grid over the val stream, so the plan is fixed and aliased, not
    /// random - two runs at the same `eval_chunks` look at the same bytes.
    #[serde(default)]
    pub eval_seq: Option<u32>,
    /// Standard error of `final_val_bpb`: the sample stdev across the
    /// `eval_chunks` per-window readings divided by sqrt(`eval_chunks`).
    ///
    /// This is the WITHIN-GRID error only. It does not cover the much larger
    /// BETWEEN-GRID spread (which val prefix, which window offsets), measured
    /// at stdev ~0.034 bpb over seven grids on one fixed checkpoint in
    /// `docs/EVAL-UNCERTAINTY.md`. Quote the document, not this field alone,
    /// when comparing two runs.
    #[serde(default)]
    pub val_bpb_stderr: Option<f64>,
    /// The four hyperparameters `optimizer` was standing in for. See
    /// `OptimizerParams`, and read its LIMIT before citing them.
    #[serde(default)]
    pub optimizer_params: Option<OptimizerParams>,

    // ---- schema 9 addition: does the format LABEL name what ran? -------
    /// Whether `fake_quant_format` names arithmetic that actually executed.
    ///
    /// SIDECAR-ONLY, like the schema 6 block: it enters no hash, because it is
    /// a property of the label already in the header rather than a new input.
    /// It is derived from `fake_quant_format` by `format_label_faithful`, so it
    /// cannot disagree with the string it qualifies.
    ///
    /// `false` means the run was launched with `TRIOS_ALLOW_UNFAITHFUL_FORMAT=1`
    /// over a format the crate itself declares it cannot simulate from `f32`
    /// (`FormatKind::is_faithful()` returns false: identity passthrough,
    /// mantissa-mask stand-in, or a deferred encoder). Measured on this crate:
    /// three 20-step seed-47 runs under `TRIOS_FORMAT_TYPE=fp80` produced a
    /// payload bit-identical to the `f32` control after the 256-byte header,
    /// and the same `final_val_bpb` to all 16 digits - the `.bin` sha differed
    /// ONLY because the false label sat inside the hashed header, so the
    /// mislabel made an identity run look like a distinct artifact.
    ///
    /// `matrix_runner` has refused such a row since 2026-08-03 and stamps
    /// `format_faithful=false` when overridden; the binary that mints
    /// checkpoints did not, and the sidecar had no key for it at all.
    ///
    /// DEFAULT ON READ IS `true`, so every `/1`-`/8` document still
    /// deserializes. That default is NOT a measurement: a `/1`-`/8` record is
    /// SILENT about faithfulness, not asserting it - and those are exactly the
    /// records that could carry an `fp80` label with no refusal anywhere in the
    /// path. Decide by FIELD PRESENCE, as every schema block above says.
    #[serde(default = "format_faithful_default")]
    pub format_faithful: bool,
}

/// Forward-compatible default for [`CheckpointRecord::format_faithful`].
///
/// `#[serde(default)]` on a `bool` is `false`, which would read every schema
/// 1-8 sidecar as an admission of mislabelling it never made. `true` is the
/// non-accusing reading; the honest one is "this record does not say", and only
/// field presence can express that.
fn format_faithful_default() -> bool {
    true
}

/// Whether a `fake_quant_format` label names arithmetic that actually ran.
///
/// The single resolution of the question, so the sidecar cannot disagree with
/// the string in the hashed header: it takes the label itself, not the
/// environment, and answers with `FormatKind::is_faithful()`.
///
/// An UNRESOLVABLE label is `false`. A string this crate cannot map onto a
/// `FormatKind` cannot be claimed as a faithful measurement of anything - the
/// answer "I do not know what that is" is not the answer "yes".
pub fn format_label_faithful(label: &str) -> bool {
    crate::fake_quant::FormatKind::from_env(label).is_some_and(|k| k.is_faithful())
}

/// Deserialize a string field so that BLANK reads as ABSENT.
///
/// `""` and `"   "` are not values a run produced; they are what a run writes
/// when it has nothing to say. Reading them back as `Some("")` would carry the
/// defect forward into every consumer of an old sidecar.
fn blank_string_as_none<'de, D>(de: D) -> std::result::Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let raw = Option::<String>::deserialize(de)?;
    Ok(raw.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()))
}

/// The deployment that produced this artifact, or `None`.
///
/// The `env_nonempty` rule `neon_writer` applies to the database columns,
/// applied to the evidence document: an unset variable and a variable set to
/// whitespace are both the absence of a run id, never the empty run id.
pub fn resolve_run_id() -> Option<String> {
    match std::env::var("RAILWAY_DEPLOYMENT_ID") {
        Ok(v) if !v.trim().is_empty() => Some(v.trim().to_string()),
        _ => None,
    }
}

/// Schema tag stamped into every `CheckpointRecord`.
///
/// Version 2 was additive over version 1 plus two renames (`bpb` ->
/// `final_val_bpb`, `git_dirty: bool` -> `Option<bool>`).
///
/// Version 3 is purely additive over version 2: `lr`, `attn_scale`,
/// `attn_seq`, `platform` and `source_sha256`. Those were the first-order
/// inputs the record omitted - a missing learning rate, two undeclared
/// architecture env vars, and no platform at all, even though the cross-libc
/// experiment showed the same seed, corpus and source tree producing different
/// checkpoint hashes on macOS and on glibc.
///
/// Version 4 is purely additive over version 3: `trainer` and `vocab`. Both
/// are cases of the same defect - the record could not describe its own inputs.
/// Nothing named the EXECUTOR, so `ckpt_replay` hashed the binary it was about
/// to run and had nothing to compare it to (a `/bin/sh` stub that copied a
/// pre-baked file graded `VERIFIED`); and nothing named the ALPHABET, so a
/// figure produced by folding bytes `% 128` was labelled bits-per-byte whatever
/// the corpus was.
///
/// Version 5 is purely additive over version 4: `gf16_enabled`. Same defect
/// family once more - the record could not describe its own inputs. The flag is
/// byte 124 of the hashed header, so it had already decided the artifact's
/// identity, and no sidecar mentioned it; a reader could not tell whether the
/// weights had been through the in-place `gf16_floor()` rewrite, which is worth
/// ~0.116 bpb on this architecture.
///
/// Version 6 is additive over version 5 plus one rename. The additions -
/// `eval_chunks`, `eval_tokens`, `eval_seq`, `val_bpb_stderr` and
/// `optimizer_params` - are the same defect family a fifth time, in its two
/// remaining forms. First, the record stated a BPB to seventeen digits and said
/// nothing about how it was measured: `evaluate` averaged a hardcoded 40
/// windows of 129 tokens, 5,160 bytes, 5.16% of a 100,000-byte val corpus, on a
/// fixed aliased grid, and not one of those numbers appeared in any field. A
/// quantity sampled at 5% with no stated plan and no uncertainty is not a
/// measurement result, and the crate was comparing such quantities at the
/// fourth decimal. Second, `optimizer: "adamw"` was one string carrying four
/// hyperparameters, so two runs with different beta1 produced identical
/// provenance.
///
/// The rename is `best_val_bpb` -> `min_observed_val_bpb` (with a serde alias,
/// so schema 1-5 sidecars still load). The old name presented a minimum over
/// noisy draws as a peer of `final_val_bpb`; with a measured estimator spread
/// of ~0.034 bpb that minimum is optimistic by construction and gets more so
/// the more often a run evaluated.
///
/// Version 7 closes the bump `/6` disclosed as pending, and lands three
/// provenance fixes with it because they are one defect family in one record.
///
/// The pending item: `PlatformProvenance::source_digest_scope` was added after
/// the tag last moved, so a `/6` record carried a field `/6` did not name.
///
/// The additions: `platform.rustflags_source`, `platform.rustflags_sha256` and
/// `platform.remap_applied`. Same defect family as every block above - the
/// record could not describe its own inputs - and this time the undescribed
/// input DECIDES A VERDICT. `.cargo/config.toml` carries the
/// `--remap-path-prefix` flags that fix `trainer.sha256`; the file is
/// gitignored, so an auditor who clones at `git_sha` never receives it, builds
/// with the documented `cargo build --release --locked`, and is told
/// `TRAINER MISMATCH` by a check whose own suggested remedy ("or rebuild it")
/// is the thing that just failed. The file is now a hashed digest input as well
/// (`trios-source-tree/3`), so that auditor's `source_sha256` differs too -
/// which is the correct answer, not a bug: their tree is not the builder's.
///
/// The two changes of MEANING, both narrowing what the record publishes rather
/// than what it claims:
///   * `platform.source_digest_scope` is now a CLASSIFICATION
///     (`repository-root` | `other:<sha256>`), and `trainer.path` is relative
///     to it. Both used to be absolute paths, so every locally produced sidecar
///     published the builder's `$HOME` - in the same artifact set whose whole
///     point is that `strings` finds 0 such paths in the binary.
///   * `run_id` is `Option<String>`, absent instead of `""`. That is the
///     ''-vs-NULL defect `neon_writer` fixed for the database columns and never
///     ported to the evidence document, where `run_id: ""` appeared in every
///     non-Railway record ever written.
///
/// Old sidecars still deserialize: the new fields are `#[serde(default)]`, and
/// a `/1`-`/6` `run_id: ""` reads back as `None` by the same rule
/// `ckpt_replay::present()` already applies. A DEFAULTED value is still not a
/// measurement; decide by FIELD PRESENCE.
///
/// The BINARY header format version is deliberately NOT bumped, in 6 as in 5,
/// 4, 3 and 2: the sidecar is the evidence record, the `.bin` layout is
/// unchanged, and rewriting the format spec for a JSON field would invalidate
/// every existing artifact hash for nothing. (`vocab` was already byte 16 and
/// `gf16_enabled` byte 124 of the hashed header; schemas 4 and 5 only surface
/// them in the sidecar. The schema 6 fields are sidecar-only and enter no
/// hash: the eval plan describes an OBSERVATION of the weights, not the
/// weights, and putting it in the hashed header would make the artifact's
/// identity depend on how it was looked at - the exact defect
/// `TRIOS_GF16_FLOOR_EVERY` exists to undo.)
///
/// Version 8 is purely additive over version 7: `git_untracked` at the top
/// level, and `platform.libc_version`, `platform.libc_provenance` and
/// `platform.features`. All four are the same defect family as every block
/// above - the record could not describe its own inputs - and each of them
/// falsifies a sentence the record was previously making.
///
///   * `git_untracked` is a SIBLING of `git_dirty`, not a widening of it.
///     `git_dirty` comes from `git status --untracked-files=no` while
///     `source_sha256` walks `src/**/*.rs` off the filesystem, so an untracked
///     `.rs` is compiled in, moves the digest, and leaves `git_dirty: false` on
///     a tree that is not `git_sha`. Widening `git_dirty` would have changed
///     the meaning of every record already on disk without rewriting one of
///     them.
///   * `platform.libc_version` / `libc_provenance` put a VERSION under the
///     family. `libc` alone is `"gnu"` or `"undetermined"`, which cannot
///     separate glibc 2.28 from glibc 2.39 - in the exact dimension the
///     cross-architecture negative result turns on, where libm differences are
///     one of the two candidate causes and the two arms of the experiment
///     recorded `"gnu"` and `"undetermined"`.
///   * `platform.features` names the compiled feature set. It was already a
///     `trios-source-tree/2` digest input with no field naming it, so an
///     auditor running the published `:gf16` image met `TRAINER MISMATCH` with
///     three printed causes and the real one absent. The measured fact is that
///     the two feature builds hash differently (`8a357f46...` vs
///     `6a2874b9...`) and produced byte-identical artifacts; see
///     `compiled_feature_set`, whose earlier justification claimed the opposite
///     and is corrected there.
///
/// Old sidecars still deserialize: the `platform.*` additions are
/// `#[serde(default)]`, and `git_untracked` is absent from them, which is the
/// correct reading - a `/1`-`/7` record is SILENT about untracked files, not
/// asserting there were none. Decide by FIELD PRESENCE, as every block above
/// says.
///
/// NOTE for readers pinned to `/7`: `interop/triosckp_reader.py` is being
/// extended to know `/7`, and an `/8` record must degrade to a NOTE there
/// rather than a failure. Nothing in `/8` renames or removes a `/7` field, so
/// every `/7` code path keeps working on an `/8` document; the only correct
/// reaction to the higher version tag is "there are fields here I do not read".
///
/// Version 9 is purely additive over version 8: `format_faithful`. Same defect
/// family as every block above - the record could not describe its own inputs -
/// and this is the sharpest instance yet, because the undescribed input is a
/// LABEL THAT IS ALREADY HASHED. `fake_quant_format` occupies bytes 136..152 of
/// the `TRIOSCKP` header, so it decides the artifact's identity, and until now
/// nothing anywhere in the checkpoint path asked whether the format named there
/// had touched a single weight.
///
/// It had not. `TRIOS_FORMAT_TYPE=fp80` printed "QAT: FakeQuant enabled for
/// format Fp80", wrote `fake_quant_format: "fp80"` into the sidecar AND into
/// the hashed header, and executed nothing: `fake_quantize_model` returns
/// immediately for a format in `is_unsupported_in_f32()`. Measured over three
/// 20-step seed-47 runs, the `fp80` payload after the 256-byte header is
/// bit-identical to the `f32` control and `final_val_bpb` agrees to all 16
/// digits; the only difference in the `.bin` sha is the false label itself. The
/// mislabel therefore did not merely decorate an identity run - it made one
/// look like a distinct artifact.
///
/// The crate already knew this was fraud-shaped everywhere except here.
/// `fake_quant`'s own `unsupported_in_f32_implies_not_faithful` test says "The
/// arithmetic is defensible; the LABEL is not", and `matrix_runner` has refused
/// such a row since 2026-08-03 unless `TRIOS_ALLOW_UNFAITHFUL_FORMAT=1`, in
/// which case it stamps `format_faithful=false`. `trios-train` - the binary
/// that mints checkpoints - refused nothing and had no key to stamp. It now
/// refuses on the same predicate with the same wording, and when the operator
/// overrides it the artifact carries its own retraction.
///
/// The field is SIDECAR-ONLY and derived from `fake_quant_format` (see
/// `format_label_faithful`), so it enters no hash and cannot drift from the
/// string it qualifies. The BINARY header format version is deliberately NOT
/// bumped, for the same reason as in 8, 7, 6, 5, 4, 3 and 2.
///
/// Old sidecars still deserialize: the field defaults to `true` on read. Read
/// the note on `format_faithful` before citing that default - a `/1`-`/8`
/// record is SILENT, and those are precisely the records that could carry an
/// `fp80` label with nothing in the path to stop them.
///
/// # Version 9 also narrows `path`, and that is a REDEFINITION of /9
///
/// `path` no longer carries an absolute location; it carries the artifact
/// RELATIVE to the digest scope, or `outside-scope:<file name>`. See the field.
/// That is not additive - the same key changes meaning - so folding it into an
/// existing version number needs a reason, and the reason is that no `/9`
/// record has ever existed outside this working tree. Measured 2026-08-05:
///
/// ```text
/// $ git log --all -S'trios-checkpoint-record/9' --oneline   -> (no commits)
/// $ census of every *.json under checkpoints/ and evidence/ (154 files)
///       71  trios-checkpoint-record/8      22  trios-checkpoint-record/7
///       22  trios-checkpoint-record/6      18  trios-checkpoint-record/3
///        9  trios-checkpoint-record/1       5  trios-checkpoint-record/4
///        5  trios-checkpoint-record/2       1  trios-checkpoint-record/5
///        0  trios-checkpoint-record/9
/// ```
///
/// The highest tag any artifact anywhere carries is `/8`, and every `/8` record
/// has the absolute `path`. So no reader holds a `/9` document whose `path`
/// this widening could silently reinterpret, and no reader is stranded: `/9`
/// means BOTH `format_faithful` and the scope-relative `path`, always, in every
/// record that will ever bear the tag. Minting a `/10` instead would have left
/// `/9` permanently defined and permanently unemitted - a version number in the
/// reader's table that describes no artifact - which is the state this note
/// exists to record, not to institutionalise.
///
/// This is stated here rather than performed silently for the obvious reason: a
/// schema that can be redefined without saying so is not a schema.
pub const CHECKPOINT_RECORD_SCHEMA: &str = "trios-checkpoint-record/9";

/// The sidecar AS WRITTEN: every `CheckpointRecord` field, flattened, plus the
/// working-tree facts that only the writer is in a position to observe.
///
/// `git_untracked` lives here rather than on `CheckpointRecord` for one honest
/// reason, stated so nobody has to reverse-engineer it: the record is built by
/// an exhaustive struct literal in `train_loop::save_and_record`, and this
/// change is scoped to `checkpoint.rs` and `ckpt_replay.rs`. The value is a
/// property of the working tree at WRITE time - the same class of run-time read
/// as `source_sha256` and `git_dirty`, taken microseconds apart from them - so
/// resolving it at the write is not a compromise of meaning.
///
/// CONSEQUENCE, because it is a real one: `CheckpointRecord` has no field for
/// this key, so deserializing a sidecar into that struct DROPS it (serde
/// ignores unknown keys here). Readers that need it must read the JSON, which
/// is what `ckpt_replay` and `interop/triosckp_reader.py` both do. When the
/// field moves onto the struct, delete this wrapper rather than keeping two
/// sources for one key.
#[derive(serde::Serialize)]
struct SidecarDocument<'a> {
    #[serde(flatten)]
    record: &'a CheckpointRecord,
    /// `resolve_git_untracked()`, or `None` when no tree was inspected.
    ///
    /// A SIBLING of `git_dirty`, never a widening of it. Resolved only on the
    /// `GIT_PROVENANCE_VERIFIED` path, by the same rule that leaves `git_dirty`
    /// `None` when `GIT_SHA` was asserted by the environment: on that path the
    /// commit is taken on trust and no tree was examined, so reporting the
    /// untracked state of whatever directory happens to be underfoot would pair
    /// an inspected fact with an uninspected one.
    git_untracked: Option<bool>,
}

impl<'a> SidecarDocument<'a> {
    fn of(record: &'a CheckpointRecord) -> Self {
        let git_untracked = if record.git_provenance == GIT_PROVENANCE_VERIFIED {
            resolve_git_untracked()
        } else {
            None
        };
        Self {
            record,
            git_untracked,
        }
    }
}

/// Ledger value carried by the FIRST of the two sidecar writes. It is the
/// ABSENCE of an outcome rather than an outcome, which is the whole reason the
/// refinement guard below lets the second write move it.
const LEDGER_PENDING: &str = "pending";

/// The one escape hatch out of the refinement guard, `=1`. Deliberate,
/// per-invocation, and loud on stderr - never a record field, because "this
/// document was overwritten" is exactly the kind of claim a document that was
/// overwritten cannot be trusted to make.
const ALLOW_SIDECAR_OVERWRITE_ENV: &str = "TRIOS_ALLOW_SIDECAR_OVERWRITE";

/// One key on which a rewrite would CHANGE what the record already says
/// instead of adding to it. `key` is a dotted path (`corpus.val.sha256`).
struct SidecarDiff {
    key: String,
    existing: String,
    incoming: String,
}

/// True while `TRIOS_ALLOW_SIDECAR_OVERWRITE=1`.
fn allow_sidecar_overwrite() -> bool {
    std::env::var(ALLOW_SIDECAR_OVERWRITE_ENV).map(|v| v == "1") == Ok(true)
}

/// The two keys the documented two-phase write is allowed to move, and only
/// while the document ON DISK is still `pending`.
///
/// `ledger` because "pending" is the absence of an outcome, and `ts` because
/// the finalising write happens microseconds later and stamps itself. Once the
/// document on disk carries a terminal ledger the exemption is gone and both
/// keys are as immutable as every measurement beside them - so a SECOND RUN,
/// whose first write is `pending` over a terminal record, is refused on
/// `ledger` before it can touch anything else.
fn sidecar_finalization_exempt(existing: &serde_json::Value, key: &str) -> bool {
    existing.get("ledger").and_then(|v| v.as_str()) == Some(LEDGER_PENDING)
        && (key == "ledger" || key == "ts")
}

/// Every key on which `incoming` fails to be a REFINEMENT of `existing`, in
/// key order.
///
/// The rule: a key present in `existing` with a non-null value must appear in
/// `incoming` with an equal value. A null (or absent) value in `existing` is
/// the absence of information and may be filled; anything else may only be
/// repeated. Objects are compared key by key, so a nested block may gain a
/// field without the whole block reading as changed.
fn sidecar_refinement_diffs(
    existing: &serde_json::Value,
    incoming: &serde_json::Value,
) -> Vec<SidecarDiff> {
    let mut out = Vec::new();
    collect_sidecar_diffs(existing, incoming, "", existing, &mut out);
    out
}

fn collect_sidecar_diffs(
    existing: &serde_json::Value,
    incoming: &serde_json::Value,
    prefix: &str,
    root: &serde_json::Value,
    out: &mut Vec<SidecarDiff>,
) {
    match (existing, incoming) {
        (serde_json::Value::Object(old), serde_json::Value::Object(new)) => {
            for (key, old_value) in old {
                // Absent information is not a claim, so filling it in is the
                // one legal kind of rewrite.
                if old_value.is_null() {
                    continue;
                }
                if prefix.is_empty() && sidecar_finalization_exempt(root, key) {
                    continue;
                }
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                match new.get(key) {
                    // Dropping a key is a change: the new document says less
                    // than the one it would replace.
                    None => out.push(SidecarDiff {
                        key: path,
                        existing: old_value.to_string(),
                        incoming: "<absent>".to_string(),
                    }),
                    Some(new_value) => {
                        collect_sidecar_diffs(old_value, new_value, &path, root, out)
                    }
                }
            }
        }
        _ => {
            if existing != incoming {
                out.push(SidecarDiff {
                    key: if prefix.is_empty() {
                        "<document>".to_string()
                    } else {
                        prefix.to_string()
                    },
                    existing: existing.to_string(),
                    incoming: incoming.to_string(),
                });
            }
        }
    }
}

/// Refuse a sidecar write that would CHANGE what the file already records.
/// See `write_sidecar_scoped` for the rule and the escape hatch.
fn guard_sidecar_refinement(path: &Path, body: &[u8], rec: &CheckpointRecord) -> Result<()> {
    let Ok(existing_bytes) = std::fs::read(path) else {
        return Ok(());
    };
    let abs = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let existing: serde_json::Value = match serde_json::from_slice(&existing_bytes) {
        Ok(v) => v,
        Err(e) => {
            if allow_sidecar_overwrite() {
                eprintln!(
                    "WARN: {ALLOW_SIDECAR_OVERWRITE_ENV}=1 - replacing sidecar {abs:?}, \
                     which is not readable as JSON ({e})"
                );
                return Ok(());
            }
            anyhow::bail!(
                "refusing to overwrite sidecar {abs:?}: it already exists and does not \
                 parse as JSON ({e}), so this write cannot show that it would only ADD \
                 information. Move the file aside, or set \
                 {ALLOW_SIDECAR_OVERWRITE_ENV}=1 to replace it deliberately. Nothing was \
                 written and nothing was deleted."
            );
        }
    };
    let incoming: serde_json::Value = serde_json::from_slice(body)
        .context("failed to re-read the serialized sidecar document as JSON")?;
    let diffs = sidecar_refinement_diffs(&existing, &incoming);
    if diffs.is_empty() {
        return Ok(());
    }
    let keys: Vec<&str> = diffs.iter().map(|d| d.key.as_str()).collect();
    if allow_sidecar_overwrite() {
        eprintln!(
            "WARN: {ALLOW_SIDECAR_OVERWRITE_ENV}=1 - replacing sidecar {abs:?}, which \
             records a DIFFERENT value on {} key(s): {}. The record that was there is \
             gone.",
            diffs.len(),
            keys.join(", ")
        );
        return Ok(());
    }
    let first = &diffs[0];
    let existing_canon = existing
        .get("canon_name")
        .map(|v| v.to_string())
        .unwrap_or_else(|| "<absent>".to_string());
    let existing_step = existing
        .get("step")
        .map(|v| v.to_string())
        .unwrap_or_else(|| "<absent>".to_string());
    anyhow::bail!(
        "refusing to overwrite sidecar {abs:?}: it already records {}={} and this write \
         carries {}={} ({} key(s) differ in total: {}). The document on disk is \
         canon_name={} step={}; this write is canon_name={:?} step={}. A sidecar may only \
         GAIN information, never restate it - the record is the evidence for the artifact \
         beside it, and this function does not get to rewrite evidence in order to report \
         success. Give the run its own TRIOS_CANON_NAME or TRIOS_CHECKPOINT_DIR, or set \
         {ALLOW_SIDECAR_OVERWRITE_ENV}=1 to replace the record deliberately. Nothing was \
         written and nothing was deleted.",
        first.key,
        first.existing,
        first.key,
        first.incoming,
        diffs.len(),
        keys.join(", "),
        existing_canon,
        existing_step,
        rec.canon_name,
        rec.step
    );
}

/// Write the sidecar atomically (tmp + rename). Called twice per
/// checkpoint: once with `ledger = "pending"` right after the `.bin` lands,
/// once with the real ledger outcome. The two-phase write means a crash
/// between the artifact and the DB attempt still leaves an on-disk record
/// naming the file and its hash; "pending" is itself honest information.
pub fn write_sidecar(rec: &CheckpointRecord) -> Result<PathBuf> {
    write_sidecar_scoped(rec, None)
}

/// `write_sidecar` into the sweep's per-seed subdirectory, so the JSON sits
/// beside the `.bin` whose hash it reports rather than beside another seed's.
/// See `run_dir`.
///
/// # Refusal to overwrite
///
/// The write is a `fs::rename`, which is atomic and therefore also SILENT: it
/// replaces whatever was there. `save_scoped` has refused that for the `.bin`
/// since the sweep destroyed two of three artifacts - but it PERMITS re-saving
/// identical bytes, so a second run of the same recipe cleared the artifact
/// guard and then replaced the record beside it in place: same checkpoint
/// hash, `final_val_bpb` 4.5670576 -> 4.4738498, `eval_chunks` 40 -> 775, exit
/// 0, no warning. The BPB, its error bar, its coverage, its corpus hashes and
/// its timestamp - everything a counterparty is asked to check - moved while
/// the hash chain stayed green.
///
/// A write is now allowed only when it is a REFINEMENT of the document already
/// on disk: every key there with a non-null value must appear here with an
/// equal value, and only null or absent keys may be filled in. Any changed
/// value is an error naming the path, the first differing key and both values.
/// Same rule as `save_scoped`, same reason: a record that exists is evidence,
/// and this function does not get to rewrite evidence in order to report
/// success.
///
/// Two exemptions, both narrow. While the document on disk still reads
/// `ledger: "pending"`, this write may move `ledger` and `ts` - that is the
/// documented two-phase write above finishing itself, and it is over the
/// instant the ledger outcome lands. And `TRIOS_ALLOW_SIDECAR_OVERWRITE=1`
/// permits any write, printing a WARN naming the differing keys to stderr.
pub fn write_sidecar_scoped(rec: &CheckpointRecord, seed_scope: Option<u64>) -> Result<PathBuf> {
    let path = sidecar_path_scoped(
        &sanitize_run_name(&rec.canon_name),
        seed_scope,
        rec.step as usize,
    );
    let dir = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("sidecar path {path:?} has no parent"))?
        .to_path_buf();
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create sidecar dir {dir:?}"))?;
    let body = serde_json::to_vec_pretty(&SidecarDocument::of(rec))
        .context("failed to serialize CheckpointRecord")?;
    guard_sidecar_refinement(&path, &body, rec)?;
    let tmp = dir.join(format!("{}.json.tmp.{}", rec.step, std::process::id()));
    {
        let mut f =
            std::fs::File::create(&tmp).with_context(|| format!("failed to create {tmp:?}"))?;
        f.write_all(&body)
            .with_context(|| format!("failed to write {tmp:?}"))?;
        f.sync_all()
            .with_context(|| format!("failed to fsync {tmp:?}"))?;
    }
    std::fs::rename(&tmp, &path)
        .with_context(|| format!("failed to rename {tmp:?} -> {path:?}"))?;
    Ok(path)
}

/// True only for a name matching `{step}.bin` exactly. `*.tmp.*` leftovers from
/// an interrupted `save` are never checkpoints.
pub fn is_checkpoint_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|n| {
            n.strip_suffix(".bin")
                .is_some_and(|stem| !stem.is_empty() && stem.bytes().all(|b| b.is_ascii_digit()))
        })
        .unwrap_or(false)
}

/// EMA checkpoint averaging result.
#[derive(Debug, Clone)]
pub struct EmaCheckpoint {
    pub weights: Vec<f32>,
    pub averaged_count: usize,
    pub effective_decay: f64,
}

/// Post-hoc EMA over the last `n` checkpoint weight vectors.
///
/// Each checkpoint is weighted by its step number (later = heavier),
/// with exponential decay: weight_i = decay^(n - 1 - i).
///
/// Reference: Sanyal et al. 2024 - EMA of last N checkpoints improves
/// generalization by 0.03+ BPB at zero training cost.
///
/// # Arguments
/// * `checkpoints` - Weight vectors from N consecutive checkpoints
/// * `decay` - EMA decay factor (0.999 typical). Higher = more weight on earlier.
/// * `steps` - Step numbers corresponding to each checkpoint
///
/// # Returns
/// EMA-averaged weight vector, or error if checkpoints is empty.
pub fn ema_average(
    checkpoints: &[Vec<f32>],
    decay: f64,
    steps: &[usize],
) -> anyhow::Result<EmaCheckpoint> {
    anyhow::ensure!(
        !checkpoints.is_empty(),
        "need at least 1 checkpoint for EMA"
    );
    anyhow::ensure!(
        checkpoints.len() == steps.len(),
        "checkpoints ({}) and steps ({}) must have same length",
        checkpoints.len(),
        steps.len()
    );

    let n = checkpoints.len();
    let dim = checkpoints[0].len();

    for (i, ckpt) in checkpoints.iter().enumerate() {
        anyhow::ensure!(
            ckpt.len() == dim,
            "checkpoint {} has dim {}, expected {}",
            i,
            ckpt.len(),
            dim
        );
    }

    let mut ema = vec![0.0f64; dim];
    let mut total_weight = 0.0f64;

    for (i, (ckpt, &step)) in checkpoints.iter().zip(steps.iter()).enumerate() {
        let step_weight = step as f64;
        let positional_weight = decay.powi((n - 1 - i) as i32);
        let w = step_weight * positional_weight;
        total_weight += w;
        for j in 0..dim {
            ema[j] += w * ckpt[j] as f64;
        }
    }

    if total_weight > 0.0 {
        for v in ema.iter_mut() {
            *v /= total_weight;
        }
    }

    let weights: Vec<f32> = ema.iter().map(|&v| v as f32).collect();

    Ok(EmaCheckpoint {
        weights,
        averaged_count: n,
        effective_decay: decay,
    })
}

/// Sweep EMA over different N values (P4 requirement).
///
/// Tries N in {3, 5, 10, 20} and returns all results.
/// Caller picks the best one (lowest validation loss).
pub fn ema_sweep(
    all_checkpoints: &[Vec<f32>],
    all_steps: &[usize],
    decay: f64,
) -> Vec<(usize, EmaCheckpoint)> {
    let n_values = [3usize, 5, 10, 20];
    let mut results = Vec::new();

    for &n in &n_values {
        if n > all_checkpoints.len() {
            continue;
        }
        let start = all_checkpoints.len() - n;
        let ckpts = &all_checkpoints[start..];
        let steps = &all_steps[start..];
        if let Ok(ema) = ema_average(ckpts, decay, steps) {
            results.push((n, ema));
        }
    }

    results
}

// ===================================================================
// Window audit - the resume record (format `TRIOSRSM`, version 1)
//
// WHY A SECOND FILE AND NOT A BIGGER CHECKPOINT. `TRIOSCKP` carries weights
// only, and every digest this repository has published - the two
// cross-architecture 12000-step hashes, the seals in `evidence/SEALS.txt` -
// is a digest of those exact bytes. Adding optimizer moments to the container
// would invalidate all of them at once. The resume record therefore lands
// BESIDE the `.bin` as `{step}.resume`, with its own magic, its own version
// and its own digest, and `to_checkpoint_bytes` is not touched.
//
// WHAT IT BUYS. `ckpt_replay` is described as a spot-check verifier but
// re-executes from step 0, because a warm start was impossible: no moments,
// no step counter, no batch-sampler state. An auditor therefore paid the
// vendor's whole training budget to check one claim. With this record an
// auditor re-executes ONE challenged window of N steps out of T, at cost
// N/T. See `docs/WINDOW-AUDIT.md` for what that does and does not prove.
//
// LAYOUT. Little-endian throughout, floats as IEEE-754 bit patterns, exactly
// as `TRIOSCKP`. Three sections plus a trailing digest:
//
// ```text
//   header (344) | optimizer directory (32 * n) | payload | digest (32)
// ```
//
// Header, at absolute byte offsets:
//
// ```text
//    0   8  magic = b"TRIOSRSM"
//    8   4  format_version: u32 = 1
//   12   4  header_len: u32 = 344 (absolute offset of the directory)
//   16   8  seed: u64             24   8  step: u64 (steps COMPLETED)
//   32   8  rng_s: u64 (batch sampler, after the step above)
//   40   8  steps_total: u64      48   8  eval_every: u64
//   56   8  gf16_floor_every: u64
//   64   4  hidden: u32           68   4  d_model: u32
//   72   4  num_attn_layers: u32  76   4  vocab: u32
//   80   4  dim: u32              84   4  num_ctx: u32
//   88   4  base_lr: f32 bits     92   4  weight_decay: f32 bits
//   96   1  gf16_enabled: u8      97   1  data_synthetic: u8
//   98   1  ema_present: u8       99   1  best_present: u8
//  100   4  optimizer_count: u32
//  104   8  ema_bpb: f64 bits (meaningless unless ema_present)
//  112   8  min_observed_val_bpb: f64 bits (unless best_present)
//  120  64  weight_sha256: 64 lowercase hex ASCII
//  184  64  train_sha256: 64 lowercase hex ASCII
//  248  64  val_sha256: 64 lowercase hex ASCII
//  312  16  fake_quant_format: ASCII, NUL-padded
//  328   8  optimizer: ASCII, NUL-padded
//  336   8  reserved (must be zero, rejected if nonzero)
// ```
//
// Directory: `optimizer_count` entries of 32 bytes each, in the order the
// training loop steps them:
//
// ```text
//    0  16  name: ASCII, NUL-padded ("embed", "ctx0".."ctx5", "proj",
//                                    "attn_down", "attn_up", "head", "attn_w")
//   16   8  elements: u64 (length of m, and of v)
//   24   8  step: u64 (this instance's own bias-correction counter)
// ```
//
// Payload: for each directory entry in order, `elements` f32 of `m` followed
// by `elements` f32 of `v`.
//
// Trailer: SHA-256 over every preceding byte of the file. The digest is
// checked on load before a single moment is handed to an optimizer, so a
// half-written or edited record is a refusal and never a warm start from
// corrupted moments.
//
// `file_len = 344 + 32 * n + 8 * sum(elements) + 32`.
// ===================================================================

/// Magic prefix of every resume record.
pub const RESUME_MAGIC: &[u8; 8] = b"TRIOSRSM";

/// On-disk format version written by `save_resume`. Bump only together with a
/// reader branch that keeps v1 files loadable.
pub const RESUME_FORMAT_VERSION: u32 = 1;

/// Absolute byte length of the fixed header (= offset of the directory).
pub const RESUME_HEADER_LEN: usize = 344;

/// Byte length of one optimizer directory entry.
pub const RESUME_DIRECTORY_ENTRY_LEN: usize = 32;

/// Byte length of the trailing SHA-256.
pub const RESUME_DIGEST_LEN: usize = 32;

/// File extension of a resume record: `{step}.bin` pairs with `{step}.resume`.
pub const RESUME_EXTENSION: &str = "resume";

/// Every refusal this format can raise starts with these two words, so an
/// operator reading a failed audit can grep for one string and a test can
/// assert the run stopped for the reason it was supposed to stop for.
pub const RESUME_REFUSAL_PREFIX: &str = "RESUME REFUSED";

/// The named reasons. A refusal quotes exactly one of these in parentheses
/// immediately after `RESUME_REFUSAL_PREFIX`.
pub const RESUME_REASON_MAGIC: &str = "bad-magic";
/// See `RESUME_REASON_MAGIC`.
pub const RESUME_REASON_VERSION: &str = "unsupported-version";
/// See `RESUME_REASON_MAGIC`.
pub const RESUME_REASON_TRUNCATED: &str = "truncated";
/// See `RESUME_REASON_MAGIC`.
pub const RESUME_REASON_TRAILING: &str = "trailing-bytes";
/// See `RESUME_REASON_MAGIC`.
pub const RESUME_REASON_DIGEST: &str = "digest-mismatch";
/// See `RESUME_REASON_MAGIC`.
pub const RESUME_REASON_MALFORMED: &str = "malformed-field";
/// See `RESUME_REASON_MAGIC`.
pub const RESUME_REASON_WEIGHT_DIGEST: &str = "weight-digest-mismatch";
/// See `RESUME_REASON_MAGIC`.
pub const RESUME_REASON_SHAPE: &str = "shape-mismatch";
/// See `RESUME_REASON_MAGIC`.
pub const RESUME_REASON_CORPUS: &str = "corpus-mismatch";
/// See `RESUME_REASON_MAGIC`.
pub const RESUME_REASON_CADENCE: &str = "cadence-mismatch";
/// See `RESUME_REASON_MAGIC`.
pub const RESUME_REASON_RECIPE: &str = "recipe-mismatch";
/// See `RESUME_REASON_MAGIC`.
pub const RESUME_REASON_NOTHING_TO_DO: &str = "nothing-to-resume";
/// See `RESUME_REASON_MAGIC`. Raised by `train_loop`, not by this module.
pub const RESUME_REASON_MUON: &str = "muon-path";
/// See `RESUME_REASON_MAGIC`. Raised when the `.bin` or the `.resume` half of
/// the pair is missing.
pub const RESUME_REASON_MISSING: &str = "missing-file";

/// Build a refusal. Every rejection path in this module goes through here, so
/// the prefix and the reason token cannot drift between messages.
///
/// A refusal is deliberately an `Err` and never a fallback: resuming with
/// zeroed moments would run, would finish, and would produce different
/// weights than the monolithic run it claims to segment - the failure mode
/// this whole format exists to make impossible.
pub fn resume_refusal(reason: &str, detail: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{RESUME_REFUSAL_PREFIX} ({reason}): {detail}")
}

/// One optimizer instance's whole mutable state.
///
/// `name` is the instance, not the algorithm: the AdamW path runs
/// `7 + NUM_CTX` separate instances and each carries its own bias-correction
/// counter, so a single `step` for the run would be wrong for `attn_w`, which
/// is not stepped when the attention block is frozen.
#[derive(Debug, Clone, PartialEq)]
pub struct ResumeOptimizerState {
    pub name: String,
    pub step: u64,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

/// Everything the trainer needs to continue a run at the byte level, other
/// than the weights themselves (which stay in the paired `.bin`).
#[derive(Debug, Clone, PartialEq)]
pub struct ResumeRecord {
    pub seed: u64,
    /// Steps COMPLETED. The resumed run executes `step + 1 ..= steps_total`.
    pub step: u64,
    /// The batch sampler's LCG state after the step above. Without it the
    /// resumed run trains on different windows of the corpus and diverges
    /// immediately, while still looking like a healthy run.
    pub rng_s: u64,
    /// The total the cosine schedule was planned against. The resumed run
    /// must be given the SAME total: `cosine_lr` is a function of
    /// `(step, steps_total)`, so a segment run to a different total applies
    /// different learning rates to the same steps.
    pub steps_total: u64,
    pub eval_every: u64,
    pub gf16_floor_every: u64,
    pub hidden: u32,
    pub d_model: u32,
    pub num_attn_layers: u32,
    pub vocab: u32,
    pub dim: u32,
    pub num_ctx: u32,
    pub base_lr: f32,
    pub weight_decay: f32,
    pub gf16_enabled: bool,
    pub data_synthetic: bool,
    /// The run's running EMA and running minimum at the moment of capture.
    /// These are TRAJECTORY state, not a measurement of the paired weights -
    /// they are carried so a segmented run reports the same numbers a
    /// monolithic one would, and they are `None` when the run had not yet
    /// taken the reading in question.
    pub ema_bpb: Option<f64>,
    pub min_observed_val_bpb: Option<f64>,
    /// SHA-256 of the `.bin` this record pairs with, as `shasum -a 256`
    /// prints it. The binding is what stops moments from one run being
    /// pasted onto weights from another.
    pub weight_sha256: String,
    pub train_sha256: String,
    pub val_sha256: String,
    pub fake_quant_format: String,
    /// Always "adamw" in version 1. The Muon path is refused rather than
    /// half-serialised; see `RESUME_REASON_MUON`.
    pub optimizer: String,
    pub optimizers: Vec<ResumeOptimizerState>,
}

/// Where a resume record landed and what it hashes to.
#[derive(Debug, Clone)]
pub struct SavedResume {
    pub path: PathBuf,
    /// Lowercase hex SHA-256 of the file on disk. Equal to `shasum -a 256`.
    pub sha256: String,
    pub bytes: u64,
}

/// Copy `s` into a fixed-width NUL-padded ASCII field, refusing to truncate.
///
/// The twin of `train_loop::ascii_field`, which is private to that module. A
/// silently shortened optimizer-instance name would make the directory name a
/// tensor it does not name.
fn resume_ascii_field<const N: usize>(s: &str, what: &str) -> Result<[u8; N]> {
    if !s.is_ascii() || !s.bytes().all(|b| (0x20..=0x7e).contains(&b)) {
        return Err(resume_refusal(
            RESUME_REASON_MALFORMED,
            format!("{what} {s:?} is not printable ASCII"),
        ));
    }
    if s.len() > N {
        return Err(resume_refusal(
            RESUME_REASON_MALFORMED,
            format!("{what} {s:?} is {} bytes, field is {N}", s.len()),
        ));
    }
    let mut out = [0u8; N];
    out[..s.len()].copy_from_slice(s.as_bytes());
    Ok(out)
}

/// Inverse of `resume_ascii_field`.
fn read_resume_ascii(field: &[u8], what: &str) -> Result<String> {
    let end = field.iter().position(|&b| b == 0).unwrap_or(field.len());
    if field[end..].iter().any(|&b| b != 0) {
        return Err(resume_refusal(
            RESUME_REASON_MALFORMED,
            format!("{what} has bytes after its NUL terminator"),
        ));
    }
    let text = std::str::from_utf8(&field[..end]).map_err(|e| {
        resume_refusal(RESUME_REASON_MALFORMED, format!("{what} is not UTF-8: {e}"))
    })?;
    if !text.bytes().all(|b| (0x20..=0x7e).contains(&b)) {
        return Err(resume_refusal(
            RESUME_REASON_MALFORMED,
            format!("{what} {text:?} is not printable ASCII"),
        ));
    }
    Ok(text.to_string())
}

/// Write a 64-character lowercase-hex digest into a fixed field, refusing
/// anything that is not one. An empty or upper-case digest recorded here
/// would compare unequal to `shasum -a 256` output forever after.
fn write_hex64(s: &str, what: &str) -> Result<[u8; 64]> {
    if s.len() != 64
        || !s
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(resume_refusal(
            RESUME_REASON_MALFORMED,
            format!("{what} {s:?} is not 64 lowercase hex characters"),
        ));
    }
    let mut out = [0u8; 64];
    out.copy_from_slice(s.as_bytes());
    Ok(out)
}

/// Inverse of `write_hex64`.
fn read_hex64(field: &[u8], what: &str) -> Result<String> {
    if field.len() != 64
        || !field
            .iter()
            .all(|&b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(resume_refusal(
            RESUME_REASON_MALFORMED,
            format!("{what} is not 64 lowercase hex characters"),
        ));
    }
    Ok(String::from_utf8_lossy(field).into_owned())
}

fn resume_rd_u32(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn resume_rd_u64(bytes: &[u8], off: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[off..off + 8]);
    u64::from_le_bytes(b)
}

/// Serialize a resume record, digest included.
pub fn resume_to_bytes(rec: &ResumeRecord) -> Result<Vec<u8>> {
    if rec.optimizers.is_empty() {
        return Err(resume_refusal(
            RESUME_REASON_MALFORMED,
            "a resume record with no optimizer instances would restore nothing",
        ));
    }
    let mut elements: u64 = 0;
    for o in &rec.optimizers {
        if o.m.len() != o.v.len() {
            return Err(resume_refusal(
                RESUME_REASON_MALFORMED,
                format!(
                    "optimizer {:?} has {} first-moment and {} second-moment elements",
                    o.name,
                    o.m.len(),
                    o.v.len()
                ),
            ));
        }
        elements += o.m.len() as u64;
    }

    let n = rec.optimizers.len();
    let dir_len = n * RESUME_DIRECTORY_ENTRY_LEN;
    let payload_len = 8 * elements as usize;
    let mut out = vec![0u8; RESUME_HEADER_LEN + dir_len + payload_len + RESUME_DIGEST_LEN];

    out[0..8].copy_from_slice(RESUME_MAGIC);
    out[8..12].copy_from_slice(&RESUME_FORMAT_VERSION.to_le_bytes());
    out[12..16].copy_from_slice(&(RESUME_HEADER_LEN as u32).to_le_bytes());
    out[16..24].copy_from_slice(&rec.seed.to_le_bytes());
    out[24..32].copy_from_slice(&rec.step.to_le_bytes());
    out[32..40].copy_from_slice(&rec.rng_s.to_le_bytes());
    out[40..48].copy_from_slice(&rec.steps_total.to_le_bytes());
    out[48..56].copy_from_slice(&rec.eval_every.to_le_bytes());
    out[56..64].copy_from_slice(&rec.gf16_floor_every.to_le_bytes());
    out[64..68].copy_from_slice(&rec.hidden.to_le_bytes());
    out[68..72].copy_from_slice(&rec.d_model.to_le_bytes());
    out[72..76].copy_from_slice(&rec.num_attn_layers.to_le_bytes());
    out[76..80].copy_from_slice(&rec.vocab.to_le_bytes());
    out[80..84].copy_from_slice(&rec.dim.to_le_bytes());
    out[84..88].copy_from_slice(&rec.num_ctx.to_le_bytes());
    out[88..92].copy_from_slice(&rec.base_lr.to_le_bytes());
    out[92..96].copy_from_slice(&rec.weight_decay.to_le_bytes());
    out[96] = u8::from(rec.gf16_enabled);
    out[97] = u8::from(rec.data_synthetic);
    out[98] = u8::from(rec.ema_bpb.is_some());
    out[99] = u8::from(rec.min_observed_val_bpb.is_some());
    out[100..104].copy_from_slice(&(n as u32).to_le_bytes());
    out[104..112].copy_from_slice(&rec.ema_bpb.unwrap_or(0.0).to_le_bytes());
    out[112..120].copy_from_slice(&rec.min_observed_val_bpb.unwrap_or(0.0).to_le_bytes());
    out[120..184].copy_from_slice(&write_hex64(&rec.weight_sha256, "weight_sha256")?);
    out[184..248].copy_from_slice(&write_hex64(&rec.train_sha256, "train_sha256")?);
    out[248..312].copy_from_slice(&write_hex64(&rec.val_sha256, "val_sha256")?);
    out[312..328].copy_from_slice(&resume_ascii_field::<16>(
        &rec.fake_quant_format,
        "fake_quant_format",
    )?);
    out[328..336].copy_from_slice(&resume_ascii_field::<8>(&rec.optimizer, "optimizer")?);
    // 336..344 stays zero: reserved, rejected on load if nonzero.

    let mut off = RESUME_HEADER_LEN;
    for o in &rec.optimizers {
        out[off..off + 16].copy_from_slice(&resume_ascii_field::<16>(&o.name, "optimizer name")?);
        out[off + 16..off + 24].copy_from_slice(&(o.m.len() as u64).to_le_bytes());
        out[off + 24..off + 32].copy_from_slice(&o.step.to_le_bytes());
        off += RESUME_DIRECTORY_ENTRY_LEN;
    }
    for o in &rec.optimizers {
        for x in o.m.iter().chain(o.v.iter()) {
            out[off..off + 4].copy_from_slice(&x.to_le_bytes());
            off += 4;
        }
    }
    if off + RESUME_DIGEST_LEN != out.len() {
        return Err(resume_refusal(
            RESUME_REASON_MALFORMED,
            format!(
                "serialiser wrote {off} bytes into a {} byte buffer",
                out.len() - RESUME_DIGEST_LEN
            ),
        ));
    }
    let digest = Sha256::digest(&out[..off]);
    out[off..].copy_from_slice(&digest);
    Ok(out)
}

/// Inverse of `resume_to_bytes`. Every rejection in the load-validation list
/// happens here, before any caller can see a moment.
pub fn resume_from_bytes(bytes: &[u8]) -> Result<ResumeRecord> {
    let floor = RESUME_HEADER_LEN + RESUME_DIGEST_LEN;
    if bytes.len() < floor {
        return Err(resume_refusal(
            RESUME_REASON_TRUNCATED,
            format!(
                "{} bytes, need at least {floor} for a header and a digest",
                bytes.len()
            ),
        ));
    }
    if &bytes[0..8] != RESUME_MAGIC {
        return Err(resume_refusal(
            RESUME_REASON_MAGIC,
            "the first 8 bytes are not b\"TRIOSRSM\"; this is not a resume record",
        ));
    }
    let version = resume_rd_u32(bytes, 8);
    if version != RESUME_FORMAT_VERSION {
        return Err(resume_refusal(
            RESUME_REASON_VERSION,
            format!("format_version {version} (this build reads {RESUME_FORMAT_VERSION})"),
        ));
    }
    let header_len = resume_rd_u32(bytes, 12) as usize;
    if header_len != RESUME_HEADER_LEN {
        return Err(resume_refusal(
            RESUME_REASON_MALFORMED,
            format!("header_len {header_len} != {RESUME_HEADER_LEN}"),
        ));
    }
    if bytes[96] > 1 || bytes[97] > 1 || bytes[98] > 1 || bytes[99] > 1 {
        return Err(resume_refusal(
            RESUME_REASON_MALFORMED,
            "boolean header bytes 96..100 must be 0 or 1",
        ));
    }
    if bytes[336..344].iter().any(|&b| b != 0) {
        return Err(resume_refusal(
            RESUME_REASON_MALFORMED,
            "reserved bytes 336..344 are nonzero",
        ));
    }

    let n = resume_rd_u32(bytes, 100) as usize;
    if n == 0 {
        return Err(resume_refusal(
            RESUME_REASON_MALFORMED,
            "optimizer_count is 0; a record that restores nothing is not a warm start",
        ));
    }
    let dir_end = RESUME_HEADER_LEN + n * RESUME_DIRECTORY_ENTRY_LEN;
    if bytes.len() < dir_end + RESUME_DIGEST_LEN {
        return Err(resume_refusal(
            RESUME_REASON_TRUNCATED,
            format!(
                "{} bytes cannot hold a {n}-entry directory ({dir_end} bytes) and a digest",
                bytes.len()
            ),
        ));
    }

    let mut names = Vec::with_capacity(n);
    let mut counts = Vec::with_capacity(n);
    let mut steps = Vec::with_capacity(n);
    let mut elements: u64 = 0;
    for i in 0..n {
        let off = RESUME_HEADER_LEN + i * RESUME_DIRECTORY_ENTRY_LEN;
        names.push(read_resume_ascii(
            &bytes[off..off + 16],
            &format!("optimizer name {i}"),
        )?);
        let count = resume_rd_u64(bytes, off + 16);
        counts.push(count);
        steps.push(resume_rd_u64(bytes, off + 24));
        elements = elements.checked_add(count).ok_or_else(|| {
            resume_refusal(
                RESUME_REASON_MALFORMED,
                "directory element counts overflow u64",
            )
        })?;
    }

    let want_len = dir_end as u64 + 8 * elements + RESUME_DIGEST_LEN as u64;
    if (bytes.len() as u64) < want_len {
        return Err(resume_refusal(
            RESUME_REASON_TRUNCATED,
            format!(
                "file is {} bytes, the directory implies {want_len}",
                bytes.len()
            ),
        ));
    }
    if (bytes.len() as u64) > want_len {
        return Err(resume_refusal(
            RESUME_REASON_TRAILING,
            format!(
                "file is {} bytes, the directory implies {want_len}",
                bytes.len()
            ),
        ));
    }

    let split = bytes.len() - RESUME_DIGEST_LEN;
    let computed = sha256_hex(&bytes[..split]);
    let stored = bytes[split..].iter().fold(String::new(), |mut acc, b| {
        use std::fmt::Write as _;
        let _ = write!(acc, "{b:02x}");
        acc
    });
    if computed != stored {
        return Err(resume_refusal(
            RESUME_REASON_DIGEST,
            format!("stored {stored}, recomputed {computed} over {split} bytes"),
        ));
    }

    let mut off = dir_end;
    let mut optimizers = Vec::with_capacity(n);
    for i in 0..n {
        let count = counts[i] as usize;
        let mut m = Vec::with_capacity(count);
        let mut v = Vec::with_capacity(count);
        for _ in 0..count {
            m.push(f32::from_bits(resume_rd_u32(bytes, off)));
            off += 4;
        }
        for _ in 0..count {
            v.push(f32::from_bits(resume_rd_u32(bytes, off)));
            off += 4;
        }
        optimizers.push(ResumeOptimizerState {
            name: names[i].clone(),
            step: steps[i],
            m,
            v,
        });
    }
    if off != split {
        return Err(resume_refusal(
            RESUME_REASON_MALFORMED,
            format!("reader consumed {off} of {split} payload bytes"),
        ));
    }

    Ok(ResumeRecord {
        seed: resume_rd_u64(bytes, 16),
        step: resume_rd_u64(bytes, 24),
        rng_s: resume_rd_u64(bytes, 32),
        steps_total: resume_rd_u64(bytes, 40),
        eval_every: resume_rd_u64(bytes, 48),
        gf16_floor_every: resume_rd_u64(bytes, 56),
        hidden: resume_rd_u32(bytes, 64),
        d_model: resume_rd_u32(bytes, 68),
        num_attn_layers: resume_rd_u32(bytes, 72),
        vocab: resume_rd_u32(bytes, 76),
        dim: resume_rd_u32(bytes, 80),
        num_ctx: resume_rd_u32(bytes, 84),
        base_lr: f32::from_bits(resume_rd_u32(bytes, 88)),
        weight_decay: f32::from_bits(resume_rd_u32(bytes, 92)),
        gf16_enabled: bytes[96] != 0,
        data_synthetic: bytes[97] != 0,
        ema_bpb: (bytes[98] != 0).then(|| f64::from_bits(resume_rd_u64(bytes, 104))),
        min_observed_val_bpb: (bytes[99] != 0).then(|| f64::from_bits(resume_rd_u64(bytes, 112))),
        weight_sha256: read_hex64(&bytes[120..184], "weight_sha256")?,
        train_sha256: read_hex64(&bytes[184..248], "train_sha256")?,
        val_sha256: read_hex64(&bytes[248..312], "val_sha256")?,
        fake_quant_format: read_resume_ascii(&bytes[312..328], "fake_quant_format")?,
        optimizer: read_resume_ascii(&bytes[328..336], "optimizer")?,
        optimizers,
    })
}

/// `{step}.resume` beside `{step}.bin`.
pub fn resume_sidecar_path(checkpoint_path: &Path) -> PathBuf {
    checkpoint_path.with_extension(RESUME_EXTENSION)
}

/// Resolve `--resume-from` into the `(weights, resume record)` pair.
///
/// Either half may be named: `100.bin` and `100.resume` resolve to the same
/// pair. Both must exist, and the digest binding checked by `verify_resume`
/// is what proves they belong together - a matching filename is a convention,
/// not evidence.
pub fn resolve_resume_pair(arg: &Path) -> Result<(PathBuf, PathBuf)> {
    let ext = arg.extension().and_then(|e| e.to_str()).unwrap_or_default();
    let (weights, resume) = match ext {
        RESUME_EXTENSION => (arg.with_extension("bin"), arg.to_path_buf()),
        "bin" => (arg.to_path_buf(), resume_sidecar_path(arg)),
        other => {
            return Err(resume_refusal(
                RESUME_REASON_MISSING,
                format!(
                    "--resume-from {arg:?} has extension {other:?}; name the \
                     `{{step}}.bin` weights or the `{{step}}.{RESUME_EXTENSION}` record"
                ),
            ))
        }
    };
    if !weights.is_file() {
        return Err(resume_refusal(
            RESUME_REASON_MISSING,
            format!("weights {weights:?} do not exist"),
        ));
    }
    if !resume.is_file() {
        return Err(resume_refusal(
            RESUME_REASON_MISSING,
            format!(
                "resume record {resume:?} does not exist. Only checkpoints written by a \
                 build that carries this format have one; a weights-only artifact cannot \
                 be warm-started, and starting cold from it would silently zero the \
                 optimizer moments."
            ),
        ));
    }
    Ok((weights, resume))
}

/// Read and validate a resume record from disk.
pub fn load_resume_file(path: &Path) -> Result<ResumeRecord> {
    let raw = std::fs::read(path)
        .map_err(|e| resume_refusal(RESUME_REASON_MISSING, format!("{path:?}: {e}")))?;
    resume_from_bytes(&raw).with_context(|| format!("resume record {path:?}"))
}

/// Write a resume record beside the checkpoint it pairs with, atomically.
///
/// Same sequence as `save_scoped`: tmp in the same directory, `sync_all`
/// before close, rename, best-effort directory fsync, then hash the file as
/// it exists on disk. Unlike `save_scoped` this DOES overwrite: the record is
/// derived state that a re-run can legitimately regenerate, while the `.bin`
/// is the published evidence whose overwrite guard is unchanged. Nothing is
/// silent about it - the caller prints the path and the digest.
pub fn save_resume(checkpoint_path: &Path, rec: &ResumeRecord) -> Result<SavedResume> {
    let bytes = resume_to_bytes(rec)?;
    let final_path = resume_sidecar_path(checkpoint_path);
    let dir = final_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("resume path {final_path:?} has no parent"))?
        .to_path_buf();
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create resume dir {dir:?}"))?;

    let tmp_name = format!(
        "{}.tmp.{}",
        final_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("resume"),
        std::process::id()
    );
    let tmp_path = dir.join(tmp_name);
    {
        let mut f = std::fs::File::create(&tmp_path)
            .with_context(|| format!("failed to create {tmp_path:?}"))?;
        f.write_all(&bytes)
            .with_context(|| format!("failed to write {tmp_path:?}"))?;
        f.sync_all()
            .with_context(|| format!("failed to fsync {tmp_path:?}"))?;
    }
    std::fs::rename(&tmp_path, &final_path)
        .with_context(|| format!("failed to rename {tmp_path:?} -> {final_path:?}"))?;
    if let Ok(dir_handle) = std::fs::File::open(&dir) {
        let _ = dir_handle.sync_all();
    }

    let on_disk = std::fs::read(&final_path)
        .with_context(|| format!("failed to re-read resume record {final_path:?}"))?;
    Ok(SavedResume {
        sha256: sha256_hex(&on_disk),
        bytes: on_disk.len() as u64,
        path: final_path,
    })
}

/// What the run about to start declares about itself, for `verify_resume`.
///
/// Every field here is something that, if it differed, would make the
/// resumed run compute different weights than the monolithic run it claims to
/// be a segment of. Nothing in it is read from the environment by this
/// module: the caller passes what it is actually about to execute.
#[derive(Debug, Clone)]
pub struct ResumeExpectation {
    pub seed: u64,
    pub steps_total: u64,
    pub eval_every: u64,
    pub gf16_floor_every: u64,
    pub gf16_enabled: bool,
    pub hidden: u32,
    pub d_model: u32,
    pub num_attn_layers: u32,
    pub vocab: u32,
    pub dim: u32,
    pub num_ctx: u32,
    pub base_lr: f32,
    pub weight_decay: f32,
    pub fake_quant_format: String,
    /// SHA-256 of the weights file the caller just read.
    pub weight_sha256: String,
    pub train_sha256: String,
    pub val_sha256: String,
}

/// Refuse a resume that would not continue the run it claims to continue.
///
/// The checks are grouped by the reason they carry, so a failed audit says
/// which KIND of mismatch stopped it: wrong weights, wrong corpus, wrong
/// shape, wrong cadence, wrong recipe. There is no permissive mode.
pub fn verify_resume(rec: &ResumeRecord, want: &ResumeExpectation) -> Result<()> {
    if rec.weight_sha256 != want.weight_sha256 {
        return Err(resume_refusal(
            RESUME_REASON_WEIGHT_DIGEST,
            format!(
                "the record pairs with weights sha256={} but the file read is sha256={}. \
                 Optimizer moments from one run pasted onto weights from another produce a \
                 run that is a segment of neither.",
                rec.weight_sha256, want.weight_sha256
            ),
        ));
    }
    if rec.train_sha256 != want.train_sha256 || rec.val_sha256 != want.val_sha256 {
        return Err(resume_refusal(
            RESUME_REASON_CORPUS,
            format!(
                "the record was captured on train sha256={} val sha256={}; this run reads \
                 train sha256={} val sha256={}",
                rec.train_sha256, rec.val_sha256, want.train_sha256, want.val_sha256
            ),
        ));
    }
    if rec.hidden != want.hidden
        || rec.d_model != want.d_model
        || rec.num_attn_layers != want.num_attn_layers
        || rec.vocab != want.vocab
        || rec.dim != want.dim
        || rec.num_ctx != want.num_ctx
    {
        return Err(resume_refusal(
            RESUME_REASON_SHAPE,
            format!(
                "record hidden={} d_model={} layers={} vocab={} dim={} num_ctx={}; this run \
                 hidden={} d_model={} layers={} vocab={} dim={} num_ctx={}",
                rec.hidden,
                rec.d_model,
                rec.num_attn_layers,
                rec.vocab,
                rec.dim,
                rec.num_ctx,
                want.hidden,
                want.d_model,
                want.num_attn_layers,
                want.vocab,
                want.dim,
                want.num_ctx
            ),
        ));
    }
    if rec.eval_every != want.eval_every || rec.gf16_floor_every != want.gf16_floor_every {
        return Err(resume_refusal(
            RESUME_REASON_CADENCE,
            format!(
                "record eval_every={} gf16_floor_every={}; this run eval_every={} \
                 gf16_floor_every={}. Both cadences decide when the weights are rewritten \
                 or measured, so a segment run under a different one is not a segment of \
                 the same run.",
                rec.eval_every, rec.gf16_floor_every, want.eval_every, want.gf16_floor_every
            ),
        ));
    }
    if rec.seed != want.seed
        || rec.steps_total != want.steps_total
        || rec.base_lr.to_bits() != want.base_lr.to_bits()
        || rec.weight_decay.to_bits() != want.weight_decay.to_bits()
        || rec.gf16_enabled != want.gf16_enabled
        || rec.fake_quant_format != want.fake_quant_format
    {
        return Err(resume_refusal(
            RESUME_REASON_RECIPE,
            format!(
                "record seed={} steps_total={} lr={} wd={} gf16={} format={}; this run \
                 seed={} steps_total={} lr={} wd={} gf16={} format={}. The cosine schedule \
                 is a function of (step, steps_total, base_lr), so a segment run to a \
                 different total applies different learning rates to the same steps.",
                rec.seed,
                rec.steps_total,
                rec.base_lr,
                rec.weight_decay,
                rec.gf16_enabled,
                rec.fake_quant_format,
                want.seed,
                want.steps_total,
                want.base_lr,
                want.weight_decay,
                want.gf16_enabled,
                want.fake_quant_format
            ),
        ));
    }
    if rec.step >= rec.steps_total {
        return Err(resume_refusal(
            RESUME_REASON_NOTHING_TO_DO,
            format!(
                "the record is at step {} of {} - the run it describes is already finished",
                rec.step, rec.steps_total
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_average_single_checkpoint() {
        let ckpts = vec![vec![1.0f32, 2.0, 3.0]];
        let steps = vec![1000];
        let ema = ema_average(&ckpts, 0.999, &steps).unwrap();
        assert!((ema.weights[0] - 1.0).abs() < 1e-3);
        assert!((ema.weights[1] - 2.0).abs() < 1e-3);
        assert_eq!(ema.averaged_count, 1);
    }

    #[test]
    fn test_ema_average_two_checkpoints() {
        let ckpts = vec![vec![0.0f32], vec![2.0f32]];
        let steps = vec![1000, 2000];
        let ema = ema_average(&ckpts, 0.999, &steps).unwrap();
        assert!(ema.weights[0] > 0.0 && ema.weights[0] < 2.0);
        assert_eq!(ema.averaged_count, 2);
    }

    #[test]
    fn test_ema_average_later_heavier() {
        let ckpts = vec![vec![0.0f32], vec![10.0f32]];
        let steps = vec![100, 100];
        let ema = ema_average(&ckpts, 0.999, &steps).unwrap();
        assert!(ema.weights[0] > 5.0, "later checkpoint should weigh more");
    }

    #[test]
    fn test_ema_empty_fails() {
        let result = ema_average(&[], 0.999, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ema_dimension_mismatch_fails() {
        let ckpts = vec![vec![1.0f32], vec![1.0f32, 2.0f32]];
        let steps = vec![100, 200];
        let result = ema_average(&ckpts, 0.999, &steps);
        assert!(result.is_err());
    }

    #[test]
    fn test_ema_sweep() {
        let ckpts: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32]).collect();
        let steps: Vec<usize> = (0..20).map(|i| (i + 1) * 500).collect();
        let results = ema_sweep(&ckpts, &steps, 0.999);
        assert!(results.iter().any(|(n, _)| *n == 3));
        assert!(results.len() <= 4);
    }

    #[test]
    fn test_checkpoint_path() {
        let p = checkpoint_path("test-run", 5000);
        assert!(p.to_string_lossy().contains("test-run"));
        assert!(p.to_string_lossy().contains("5000"));
    }

    // ---- schema 4/5: the record must name its executor, its alphabet ----
    // ---- and whether the weights were floored -------------------------

    /// A record with every schema-4 and schema-5 field populated, so a
    /// round-trip test is testing the serialisation and not the defaults.
    fn schema4_record() -> CheckpointRecord {
        CheckpointRecord {
            schema: CHECKPOINT_RECORD_SCHEMA.to_string(),
            canon_name: "r4-bind".to_string(),
            seed: 47,
            step: 200,
            path: "checkpoints/r4-bind/200.bin".to_string(),
            sha256: "a".repeat(64),
            bytes: 1234,
            format_version: CHECKPOINT_FORMAT_VERSION,
            hidden: 64,
            d_model: 64,
            num_attn_layers: 2,
            optimizer: "adamw".to_string(),
            fake_quant_format: "f32".to_string(),
            data_synthetic: false,
            steps_total: 200,
            gf16_floor_every: 1,
            eval_every: 100,
            final_val_bpb: Some(2.61),
            min_observed_val_bpb: Some(2.60),
            ema_bpb: Some(3.10),
            git_sha: "deadbeef".to_string(),
            git_provenance: GIT_PROVENANCE_VERIFIED.to_string(),
            git_dirty: Some(true),
            corpus: CorpusProvenance::default(),
            // Absent, not `Some("")`: this fixture ran on no deployment, and
            // schema 7 says so by omitting the key.
            run_id: None,
            ledger: "skipped-no-dsn".to_string(),
            ts: "2026-08-03T00:00:00Z".to_string(),
            lr: Some(0.003_000_000_026_077_032),
            attn_scale: 0.1,
            attn_seq: 8,
            platform: PlatformProvenance::default(),
            source_sha256: "c".repeat(64),
            trainer: TrainerProvenance {
                path: "/tmp/target/release/trios-train".to_string(),
                sha256: "b".repeat(64),
                provenance: TRAINER_PROVENANCE_SELF_HASHED.to_string(),
            },
            vocab: 128,
            gf16_enabled: true,
            eval_chunks: Some(40),
            eval_tokens: Some(5_160),
            eval_seq: Some(129),
            val_bpb_stderr: Some(0.005_4),
            optimizer_params: Some(OptimizerParams {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.04,
                source: "train_loop::AdamW".to_string(),
            }),
            // Schema 9. `fp32` is faithful, so this fixture asserts the
            // ordinary case; the mislabelled one is covered below and in
            // `tests/format_label_truth.rs`.
            format_faithful: format_label_faithful("f32"),
        }
    }

    /// `/9` now means TWO things - `format_faithful` and the scope-relative
    /// `path` - because no `/9` record was ever persisted and the widening
    /// therefore reinterprets nothing. See the constant's own note for the
    /// census that licenses it. This test pins the tag; the one below pins the
    /// second half of what it now promises, so the pair cannot drift apart.
    #[test]
    fn schema_is_version_nine() {
        assert_eq!(CHECKPOINT_RECORD_SCHEMA, "trios-checkpoint-record/9");
        // The .bin layout did not change, so the artifact hashes did not.
        assert_eq!(CHECKPOINT_FORMAT_VERSION, 1);
    }

    /// The predicate the sidecar's `format_faithful` is derived from. It must
    /// answer for the STRING that was hashed into the header, and it must say
    /// "no" both for a format the crate cannot simulate from `f32` and for a
    /// label it cannot resolve at all.
    #[test]
    fn format_label_faithful_answers_for_the_hashed_label() {
        // No QAT: both spellings of the identity format.
        assert!(format_label_faithful("f32"));
        assert!(format_label_faithful("fp32"));
        // A kernel that really runs.
        assert!(format_label_faithful("fp16"));
        assert!(format_label_faithful("gf16"));
        // Wider than f32 - `fake_quantize_f32` is the identity, so a row
        // labelled `fp80` is an f32 row.
        assert!(!format_label_faithful("fp80"));
        assert!(!format_label_faithful("f64"));
        // Explicit identities inside the kernel, already refused by
        // `matrix_runner`.
        assert!(!format_label_faithful("int32"));
        assert!(!format_label_faithful("mxfp4"));
        // Unresolvable is not "yes".
        assert!(!format_label_faithful("int_8"));
        assert!(!format_label_faithful(""));
    }

    /// A schema 1-8 sidecar has no `format_faithful` key. It must still load,
    /// and it must load as `true` - the non-accusing reading - while the key's
    /// ABSENCE stays visible to any reader that looks at the JSON.
    #[test]
    fn a_pre_schema_nine_record_loads_without_format_faithful() {
        let rec = schema4_record();
        let mut v = serde_json::to_value(&rec).expect("as value");
        assert_eq!(v["format_faithful"], serde_json::json!(true));
        v.as_object_mut()
            .expect("object")
            .remove("format_faithful")
            .expect("the key must have been written");
        let back: CheckpointRecord = serde_json::from_value(v).expect("an /8 record must load");
        assert!(back.format_faithful);
    }

    /// The retraction must survive the write: an overridden unfaithful run has
    /// to be readable as such from the document alone.
    #[test]
    fn an_unfaithful_label_serializes_as_format_faithful_false() {
        let mut rec = schema4_record();
        rec.fake_quant_format = "fp80".to_string();
        rec.format_faithful = format_label_faithful(&rec.fake_quant_format);
        let json = serde_json::to_string(&SidecarDocument::of(&rec)).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&json).expect("as value");
        assert_eq!(v["fake_quant_format"], serde_json::json!("fp80"));
        assert_eq!(v["format_faithful"], serde_json::json!(false));
    }

    // ---- schema 8: the tree, the libc version and the feature set -------

    /// The written document must carry `git_untracked` as a SIBLING of
    /// `git_dirty`, and the two must stay separate keys: a reader that could
    /// only see one of them is back to the state this schema exists to fix.
    #[test]
    fn the_sidecar_document_carries_git_untracked_beside_git_dirty() {
        let rec = schema4_record();
        let json = serde_json::to_string(&SidecarDocument::of(&rec)).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&json).expect("as value");
        // Flattening must not drop or rename anything the record already said.
        assert_eq!(v["schema"], serde_json::json!(CHECKPOINT_RECORD_SCHEMA));
        assert_eq!(v["git_dirty"], serde_json::json!(true));
        assert_eq!(v["sha256"], serde_json::json!("a".repeat(64)));
        assert_eq!(v["optimizer_params"]["beta1"], serde_json::json!(0.9));
        // Present as a key even when the answer is unknown: absent means "this
        // record predates schema 8", which is a different statement.
        assert!(
            v.get("git_untracked").is_some(),
            "schema 8 must always write the key: {json}"
        );
        // Whatever git says on this machine, it is a boolean or an explicit
        // null - never a fabricated `false`.
        assert!(
            matches!(
                v["git_untracked"],
                serde_json::Value::Bool(_) | serde_json::Value::Null
            ),
            "git_untracked must be tri-state, got {}",
            v["git_untracked"]
        );
    }

    /// An asserted `git_sha` inspected no tree, so the untracked state of
    /// whatever directory is underfoot is not this record's to report - the
    /// same rule that leaves `git_dirty` `None` on that path.
    #[test]
    fn git_untracked_is_unknown_when_the_commit_was_only_asserted() {
        let mut rec = schema4_record();
        rec.git_provenance = GIT_PROVENANCE_ASSERTED.to_string();
        rec.git_dirty = None;
        let json = serde_json::to_string(&SidecarDocument::of(&rec)).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&json).expect("as value");
        assert_eq!(v["git_untracked"], serde_json::Value::Null);
        assert_eq!(v["git_dirty"], serde_json::Value::Null);
    }

    /// The banner is parsed, not guessed: an unrecognised one yields `None`
    /// and the caller records `LIBC_PROVENANCE_NONE`.
    #[test]
    fn the_ldd_banner_is_parsed_and_never_guessed() {
        assert_eq!(
            parse_libc_version_line("ldd (Ubuntu GLIBC 2.39-0ubuntu8.2) 2.39"),
            Some("2.39".to_string())
        );
        assert_eq!(
            parse_libc_version_line("ldd (GNU libc) 2.28\n"),
            Some("2.28".to_string())
        );
        assert_eq!(parse_libc_version_line(""), None);
        assert_eq!(parse_libc_version_line("musl libc (x86_64)"), None);
    }

    /// The libc dimension must never be blank AND silent at once: either a
    /// version is recorded, or a provenance value says why not.
    #[test]
    fn the_libc_dimension_is_answered_or_explicitly_unanswered() {
        let (version, provenance) = resolve_libc_version();
        assert!(
            [
                LIBC_PROVENANCE_LDD,
                LIBC_PROVENANCE_DARWIN_RELEASE,
                LIBC_PROVENANCE_UNSUPPORTED,
                LIBC_PROVENANCE_NONE,
            ]
            .contains(&provenance),
            "unknown libc provenance {provenance:?}"
        );
        match provenance {
            LIBC_PROVENANCE_LDD | LIBC_PROVENANCE_DARWIN_RELEASE => {
                let v = version.expect("a positive provenance must carry a version");
                assert!(!v.trim().is_empty(), "empty version with provenance");
            }
            // The two negative answers carry no version, and are distinct
            // answers: "no query for this target" is not "the query failed".
            _ => assert!(version.is_none(), "a failed query must not carry a version"),
        }
        let platform = resolve_platform_provenance();
        assert_eq!(platform.libc_provenance, provenance);
    }

    /// The feature set was a digest input that no field named. It is now a
    /// field, and it must be the SAME string that enters the digest.
    #[test]
    fn the_platform_record_names_the_compiled_feature_set() {
        let platform = resolve_platform_provenance();
        assert_eq!(platform.features, compiled_feature_set());
        assert!(
            platform.features.contains("gf16="),
            "the feature that decides the weights must be named: {}",
            platform.features
        );
    }

    /// The sampling plan and its uncertainty must survive a round trip, and
    /// their ABSENCE in an older sidecar must stay distinguishable from a
    /// measured zero - `eval_chunks: 0` would be "averaged over no windows",
    /// which is not what a schema 5 record is saying.
    #[test]
    fn the_eval_plan_round_trips_and_is_absent_not_zero_in_older_records() {
        let rec = schema4_record();
        let json = serde_json::to_string(&rec).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&json).expect("as value");
        assert_eq!(v["eval_chunks"], serde_json::json!(40));
        assert_eq!(v["eval_tokens"], serde_json::json!(5_160));
        assert_eq!(v["eval_seq"], serde_json::json!(129));
        assert_eq!(v["optimizer_params"]["beta1"], serde_json::json!(0.9));
        assert_eq!(
            v["optimizer_params"]["source"],
            serde_json::json!("train_loop::AdamW")
        );

        let back: CheckpointRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.eval_chunks, Some(40));
        assert_eq!(back.eval_tokens, Some(5_160));
        assert_eq!(back.eval_seq, Some(129));
        assert_eq!(back.val_bpb_stderr, Some(0.005_4));
        let params = back.optimizer_params.expect("schema 6 carries the params");
        assert_eq!(params.beta1, 0.9);
        assert_eq!(params.beta2, 0.999);
        assert_eq!(params.eps, 1e-8);
        assert_eq!(params.weight_decay, 0.04);

        // A pre-schema-6 sidecar still loads, and the reader can tell the
        // difference by field presence rather than by a defaulted number.
        let mut older = v.clone();
        for key in [
            "eval_chunks",
            "eval_tokens",
            "eval_seq",
            "val_bpb_stderr",
            "optimizer_params",
        ] {
            older.as_object_mut().unwrap().remove(key);
        }
        let older_json = serde_json::to_string(&older).expect("serialize older");
        let back: CheckpointRecord =
            serde_json::from_str(&older_json).expect("a pre-schema-6 sidecar must deserialize");
        assert_eq!(back.eval_chunks, None, "absent, not 0 windows");
        assert_eq!(back.val_bpb_stderr, None, "absent, not an error of 0.0");
        assert_eq!(back.optimizer_params, None, "absent, not beta1 = 0.0");
    }

    /// The rename must not orphan the records already on disk: every sidecar
    /// written before schema 6 spells this field `best_val_bpb`.
    #[test]
    fn min_observed_val_bpb_reads_an_older_best_val_bpb() {
        let rec = schema4_record();
        let json = serde_json::to_string(&rec).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&json).expect("as value");
        assert_eq!(
            v["min_observed_val_bpb"],
            serde_json::json!(2.60),
            "new records are written under the honest name"
        );
        assert!(
            v.get("best_val_bpb").is_none(),
            "and not under the old one as well"
        );

        let mut older = v.clone();
        let obj = older.as_object_mut().unwrap();
        let value = obj.remove("min_observed_val_bpb").expect("present");
        obj.insert("best_val_bpb".to_string(), value);
        let older_json = serde_json::to_string(&older).expect("serialize older");
        let back: CheckpointRecord =
            serde_json::from_str(&older_json).expect("a pre-schema-6 sidecar must deserialize");
        assert_eq!(
            back.min_observed_val_bpb,
            Some(2.60),
            "the alias must carry the old spelling forward"
        );
    }

    /// `gf16_enabled` decides the weights (it gates the in-place `gf16_floor()`
    /// rewrite) and was already byte 124 of the hashed header. Schema 5 puts it
    /// where a reader of the JSON can see it.
    #[test]
    fn gf16_enabled_round_trips_and_is_absent_not_false_in_older_records() {
        let rec = schema4_record();
        let json = serde_json::to_string(&rec).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&json).expect("as value");
        assert_eq!(v["gf16_enabled"], serde_json::json!(true));
        let back: CheckpointRecord = serde_json::from_str(&json).expect("deserialize");
        assert!(back.gf16_enabled);

        // A record written before schema 5 must still load, and its defaulted
        // `false` must be distinguishable from a measured `false` - which the
        // reader can only do by field presence.
        let mut older = v.clone();
        older.as_object_mut().unwrap().remove("gf16_enabled");
        let older_json = serde_json::to_string(&older).expect("serialize older");
        let back: CheckpointRecord =
            serde_json::from_str(&older_json).expect("a pre-schema-5 sidecar must deserialize");
        assert!(!back.gf16_enabled, "the default is false");
        assert!(
            older.get("gf16_enabled").is_none(),
            "and the JSON carries no such key, which is how a reader tells"
        );
    }

    #[test]
    fn trainer_and_vocab_round_trip() {
        let rec = schema4_record();
        let json = serde_json::to_string(&rec).expect("serialize");
        let back: CheckpointRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.trainer.sha256, "b".repeat(64));
        assert_eq!(back.trainer.path, "/tmp/target/release/trios-train");
        assert_eq!(back.trainer.provenance, TRAINER_PROVENANCE_SELF_HASHED);
        assert_eq!(back.vocab, 128);

        // And they are reachable at the dotted paths `ckpt_replay` digs for.
        let v: serde_json::Value = serde_json::from_str(&json).expect("as value");
        assert_eq!(v["trainer"]["sha256"], serde_json::json!("b".repeat(64)));
        assert_eq!(v["vocab"], serde_json::json!(128));
    }

    /// A `schema/1` sidecar must still LOAD - and must be distinguishable from
    /// a schema-4 record whose fields happen to be zero. `serde` cannot make
    /// that distinction for us, so the rule is that readers decide by FIELD
    /// PRESENCE in the JSON. This test asserts both halves.
    #[test]
    fn a_schema_one_record_loads_by_default_and_is_absent_not_zero() {
        let legacy = r#"{
            "schema": "trios-checkpoint-record/1",
            "canon_name": "legacy-run",
            "seed": 47,
            "step": 12000,
            "path": "checkpoints/legacy-run/12000.bin",
            "sha256": "ef6f0887",
            "bytes": 800000,
            "format_version": 1,
            "hidden": 384,
            "d_model": 64,
            "num_attn_layers": 2,
            "optimizer": "adamw",
            "fake_quant_format": "f32",
            "data_synthetic": false,
            "steps_total": 0,
            "gf16_floor_every": 0,
            "eval_every": 0,
            "final_val_bpb": null,
            "best_val_bpb": null,
            "ema_bpb": null,
            "git_sha": "",
            "git_provenance": "unavailable",
            "git_dirty": null,
            "corpus": {
                "train": {"path": "", "bytes": 0, "sha256": ""},
                "val": {"path": "", "bytes": 0, "sha256": ""}
            },
            "run_id": "",
            "ledger": "written",
            "ts": "2026-04-30T00:00:00Z"
        }"#;

        let rec: CheckpointRecord =
            serde_json::from_str(legacy).expect("a schema 1 sidecar must still deserialize");
        assert_eq!(rec.schema, "trios-checkpoint-record/1");
        assert_eq!(rec.trainer.sha256, "");
        assert_eq!(rec.trainer.provenance, "");
        assert_eq!(rec.vocab, 0);
        // The literal `"run_id": ""` above is what schema 1-6 actually wrote on
        // every non-Railway run. Schema 7 reads it as the absence of a claim,
        // not as the deployment whose id is the empty string.
        assert_eq!(rec.run_id, None);

        // The defaults above are NOT readings. What a reader must key on is
        // that the JSON does not carry the keys at all.
        let v: serde_json::Value = serde_json::from_str(legacy).expect("as value");
        assert!(v.get("trainer").is_none(), "schema 1 has no trainer field");
        assert!(v.get("vocab").is_none(), "schema 1 has no vocab field");
        // vocab 0 would be an alphabet of nothing, and an empty trainer hash is
        // not "the binary hashed to nothing" - both are the absence of a claim.
        assert_ne!(rec.vocab, 128, "a defaulted vocab must not look like 128");
    }

    #[test]
    fn resolve_trainer_provenance_hashes_the_running_executable() {
        let prov = resolve_trainer_provenance();
        // The test harness IS an executable on disk, so this path is exercised.
        assert_eq!(
            prov.provenance, TRAINER_PROVENANCE_SELF_HASHED,
            "current_exe was not hashable: {}",
            prov.provenance
        );
        assert_eq!(prov.sha256.len(), 64);
        assert!(prov.sha256.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(!prov.path.is_empty());

        // Schema 7: the path is RELATIVE to the digest scope, so it must not
        // start at the filesystem root and must not carry a home directory.
        assert!(
            !prov.path.starts_with('/'),
            "trainer.path must be scope-relative, got {}",
            prov.path
        );

        // The digest is over the bytes on disk, so an independent read of the
        // same path must reproduce it - which is what `shasum -a 256` does.
        // Cargo runs tests with the package root as the working directory,
        // which is the scope the relative path resolves against.
        let raw = std::fs::read(&prov.path).expect("re-read the executable");
        assert_eq!(prov.sha256, sha256_hex(&raw));
    }

    /// The branches of `scope_relative_path`, without touching the process
    /// working directory.
    #[test]
    fn the_trainer_path_is_scope_relative_and_never_absolute() {
        let root = Path::new("/lab/checkout");
        assert_eq!(
            scope_relative_path(
                Path::new("/lab/checkout/target/release/trios-train"),
                Some(root)
            ),
            "target/release/trios-train"
        );
        // An executable outside the walked tree is named, not located: a
        // sidecar must not publish `/Users/<someone>/...` just because the
        // binary was moved.
        let outside = scope_relative_path(Path::new("/elsewhere/bin/trios-train"), Some(root));
        assert_eq!(outside, format!("{PATH_OUTSIDE_SCOPE_PREFIX}trios-train"));
        assert!(!outside.contains("/elsewhere"));
        // No scope at all is the same case: still no path.
        assert_eq!(
            scope_relative_path(Path::new("/elsewhere/bin/trios-train"), None),
            format!("{PATH_OUTSIDE_SCOPE_PREFIX}trios-train")
        );
    }

    /// The same function now decides `CheckpointRecord::path`, so the two
    /// branches it adds are the ones a checkpoint actually takes: a path under
    /// the scope, and an ABSOLUTE `TRIOS_CHECKPOINT_DIR` outside it - which is
    /// how the two tracked evidence sidecars came to publish a home directory.
    #[test]
    fn the_checkpoint_path_is_scope_relative_and_never_absolute() {
        let root = Path::new("/lab/checkout");
        assert_eq!(
            scope_relative_path(
                Path::new("/lab/checkout/checkpoints/IGLA-run/12000.bin"),
                Some(root)
            ),
            "checkpoints/IGLA-run/12000.bin"
        );
        // An already-relative path is already scope-relative: it resolves
        // against the working directory, which is what `root` names.
        assert_eq!(
            scope_relative_path(Path::new("checkpoints/IGLA-run/12000.bin"), Some(root)),
            "checkpoints/IGLA-run/12000.bin"
        );
        // The defect, as it was measured: an absolute checkpoint dir under a
        // home directory. The name survives; the location does not.
        let leaked = scope_relative_path(
            Path::new("/Users/somebody/trios-trainer-igla/evidence/heldout/12000.bin"),
            Some(root),
        );
        assert_eq!(leaked, format!("{PATH_OUTSIDE_SCOPE_PREFIX}12000.bin"));
        assert!(!leaked.contains("/Users"));
        // A relative path that climbs OUT of the scope is not under it, and is
        // classified like any other outsider rather than passed through.
        let climbing = scope_relative_path(Path::new("../elsewhere/12000.bin"), Some(root));
        assert_eq!(climbing, format!("{PATH_OUTSIDE_SCOPE_PREFIX}12000.bin"));
    }

    /// The scope is a CLASSIFICATION. It must answer "which tree" and publish
    /// no path, and two different other-trees must stay distinguishable.
    #[test]
    fn the_source_digest_scope_names_no_path() {
        // Cargo runs tests with the package root as the working directory, and
        // this package is a git checkout, so this is the repository-root case.
        let here = std::env::current_dir().expect("cwd");
        assert_eq!(
            classify_source_digest_scope(&here),
            SOURCE_DIGEST_SCOPE_REPO_ROOT
        );

        let a = classify_source_digest_scope(Path::new("/lab/alpha-clone"));
        let b = classify_source_digest_scope(Path::new("/lab/beta-clone"));
        for (scope, leak) in [(&a, "alpha-clone"), (&b, "beta-clone")] {
            assert!(scope.starts_with(SOURCE_DIGEST_SCOPE_OTHER_PREFIX));
            assert!(!scope.contains(leak), "the scope published a path: {scope}");
            assert_eq!(
                scope.len(),
                SOURCE_DIGEST_SCOPE_OTHER_PREFIX.len() + 64,
                "the scope must be a sha256, not a path"
            );
        }
        assert_ne!(a, b, "two different trees must classify differently");
        assert_eq!(
            a,
            classify_source_digest_scope(Path::new("/lab/alpha-clone"))
        );
    }

    /// The flags file decides `trainer.sha256`, so the record must say whether
    /// it was there - and must never copy its contents, which are paths.
    #[test]
    fn the_build_flags_are_hashed_not_published() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (source, sha, remap) = resolve_build_flags_provenance(Some(dir.path()));
        assert_eq!(source, None, "an absent flags file is absent, not empty");
        assert_eq!(sha, None);
        assert!(!remap);

        let body =
            "[build]\nrustflags = [\n    \"--remap-path-prefix=/Users/somebody=/build\",\n]\n";
        std::fs::create_dir_all(dir.path().join(".cargo")).expect("mkdir .cargo");
        std::fs::write(dir.path().join(BUILD_FLAGS_PATH), body).expect("write flags");
        let (source, sha, remap) = resolve_build_flags_provenance(Some(dir.path()));
        assert_eq!(source.as_deref(), Some(BUILD_FLAGS_SOURCE_CARGO_CONFIG));
        assert_eq!(sha.as_deref(), Some(sha256_hex(body.as_bytes()).as_str()));
        assert!(
            remap,
            "the flags ask for remapping and the record must say so"
        );
        // The builder's home is IN the file and must not be in the record: the
        // flags are hashed, never copied.
        let recorded = format!("{source:?}{sha:?}");
        assert!(
            !recorded.contains("/Users/somebody"),
            "the flags were published verbatim: {recorded}"
        );

        // A flags file with no remap in it is present but claims nothing.
        std::fs::write(dir.path().join(BUILD_FLAGS_PATH), "[build]\n").expect("rewrite flags");
        let (source, sha, remap) = resolve_build_flags_provenance(Some(dir.path()));
        assert_eq!(source.as_deref(), Some(BUILD_FLAGS_SOURCE_CARGO_CONFIG));
        assert!(sha.is_some());
        assert!(!remap);
    }

    /// Presence and absence of the flags file must not digest alike: an auditor
    /// without it and a builder with it hold different trees, and that is the
    /// fact the digest exists to report.
    #[test]
    fn the_flags_file_changes_the_source_digest() {
        let with = digest_source_inputs(&[
            (BUILD_FLAGS_PATH.to_string(), Some(b"[build]\n".to_vec())),
            ("Cargo.toml".to_string(), Some(b"[package]\n".to_vec())),
        ]);
        let without = digest_source_inputs(&[
            (BUILD_FLAGS_PATH.to_string(), None),
            ("Cargo.toml".to_string(), Some(b"[package]\n".to_vec())),
        ]);
        let empty = digest_source_inputs(&[
            (BUILD_FLAGS_PATH.to_string(), Some(Vec::new())),
            ("Cargo.toml".to_string(), Some(b"[package]\n".to_vec())),
        ]);
        assert_ne!(with, without);
        assert_ne!(empty, without, "present-and-empty is not absent");
    }

    /// The ''-vs-NULL rule, on the primary evidence document this time.
    #[test]
    fn a_blank_run_id_is_absent_on_both_read_and_write() {
        let rec = schema4_record();
        let json = serde_json::to_string(&rec).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&json).expect("as value");
        assert!(
            v.get("run_id").is_none(),
            "an absent run_id must be omitted, never written as \"\""
        );

        // And a schema 1-6 sidecar that DID write "" reads back as absent.
        let mut blank = serde_json::to_value(&rec).expect("as value");
        blank["run_id"] = serde_json::json!("");
        let legacy: CheckpointRecord = serde_json::from_value(blank).expect("legacy loads");
        assert_eq!(legacy.run_id, None);

        // A real deployment id survives, trimmed.
        let mut real = serde_json::to_value(&rec).expect("as value");
        real["run_id"] = serde_json::json!(" dep-42 ");
        let back: CheckpointRecord = serde_json::from_value(real).expect("loads");
        assert_eq!(back.run_id.as_deref(), Some("dep-42"));
    }

    // ---- the source digest must hash what it claims to hash -----------

    #[test]
    fn source_digest_domain_is_version_three() {
        // The input set changed, so the digest of an unchanged tree changed
        // too. The tag is what stops a /2 and a /3 digest being compared.
        assert_eq!(SOURCE_DIGEST_DOMAIN, b"trios-source-tree/3\n");
    }

    /// Cargo runs tests with the package root as the working directory, so the
    /// walk below sees this repository.
    #[test]
    fn source_digest_covers_the_lock_the_toolchain_the_migrations_and_the_features() {
        let inputs = collect_source_digest_inputs().expect("the crate root must be walkable");
        let keys: Vec<&str> = inputs.iter().map(|(k, _)| k.as_str()).collect();

        for required in [
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            BUILD_FLAGS_PATH,
        ] {
            assert!(keys.contains(&required), "{required} is not a digest input");
        }
        assert!(
            keys.contains(&SOURCE_DIGEST_FEATURES_KEY),
            "the compiled feature set is not a digest input"
        );
        assert!(
            keys.iter().any(|k| k.starts_with("migration/src/")),
            "migration/src/**/*.rs is not a digest input: {keys:?}"
        );
        assert!(
            keys.contains(&"src/checkpoint.rs"),
            "src/**/*.rs is not a digest input"
        );

        // Every one of those three manifests exists in this tree, so each must
        // be hashed WITH bytes rather than as a declared-but-absent input.
        for (key, body) in &inputs {
            if ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"].contains(&key.as_str()) {
                assert!(body.is_some(), "{key} exists but was recorded as absent");
            }
        }
    }

    #[test]
    fn the_feature_set_names_every_declared_feature_with_its_state() {
        let set = compiled_feature_set();
        for name in [
            "ci-strict",
            "gf16",
            "gpu",
            "race",
            "smoke",
            "trios-integration",
        ] {
            assert!(
                set.contains(&format!("{name}=")),
                "feature {name} missing from {set:?}"
            );
        }
        // The state is recorded, not merely the name, so a build with the
        // feature off differs from one with it on.
        assert!(set.contains("gf16=0") || set.contains("gf16=1"));
        assert_eq!(set, compiled_feature_set(), "must be deterministic");
    }

    #[test]
    fn a_changed_input_changes_the_source_digest() {
        let base: Vec<SourceDigestInput> = vec![
            ("Cargo.lock".to_string(), Some(b"lock-a".to_vec())),
            ("src/lib.rs".to_string(), Some(b"fn main() {}".to_vec())),
            (
                SOURCE_DIGEST_FEATURES_KEY.to_string(),
                Some(b"gf16=0".to_vec()),
            ),
        ];
        let d0 = digest_source_inputs(&base);
        assert_eq!(d0.len(), 64);
        assert_eq!(d0, digest_source_inputs(&base), "must be deterministic");

        // The lock file is a real input: changing it changes the digest.
        let mut changed_lock = base.clone();
        changed_lock[0].1 = Some(b"lock-b".to_vec());
        assert_ne!(d0, digest_source_inputs(&changed_lock));

        // So is the feature set: the same sources compiled with `gf16` on are
        // a different program.
        let mut changed_features = base.clone();
        changed_features[2].1 = Some(b"gf16=1".to_vec());
        assert_ne!(d0, digest_source_inputs(&changed_features));

        // A declared-but-absent input is not an empty one.
        let mut absent = base.clone();
        absent[0].1 = None;
        let mut empty = base.clone();
        empty[0].1 = Some(Vec::new());
        assert_ne!(digest_source_inputs(&absent), digest_source_inputs(&empty));
        assert_ne!(d0, digest_source_inputs(&absent));
    }

    #[test]
    fn the_source_digest_is_a_hex_sha256_of_this_tree() {
        let d = resolve_source_digest();
        assert_ne!(
            d, SOURCE_DIGEST_NOT_COMPUTED,
            "the test harness runs in the crate root, so the tree is readable"
        );
        assert_eq!(d.len(), 64);
        assert!(d.chars().all(|c| c.is_ascii_hexdigit()));
        // Cached: every record from one run must name the same tree. (It is
        // deliberately NOT compared against a fresh walk here - an editor
        // saving a file mid-test would legitimately move that.)
        assert_eq!(d, resolve_source_digest());
    }

    /// A well-formed digest must say WHICH tree it ranged over. Without this,
    /// a run in a throwaway clone and a run in the repository are two 64-hex
    /// strings with nothing to tell them apart.
    ///
    /// Schema 7 answers that question with a CLASSIFICATION instead of the
    /// absolute path schema 6 published, so this asserts both halves: the tree
    /// is still identified, and the record no longer carries a path.
    #[test]
    fn a_computed_source_digest_classifies_the_directory_it_walked() {
        let digest = resolve_source_digest();
        let scope = resolve_source_digest_scope();
        // The harness runs in the crate root, so a digest WAS produced and the
        // walk therefore had a directory - not the sentinel.
        assert_ne!(digest, SOURCE_DIGEST_NOT_COMPUTED);
        assert_ne!(
            scope, SOURCE_DIGEST_SCOPE_NONE,
            "a digest was computed, so the walk had a directory"
        );

        // The walk reads `src` and `Cargo.toml` relative to the directory it
        // started in, so both must be there for the record to be the truth.
        // The directory is checked IN PROCESS; it is deliberately not the value
        // the record publishes.
        let root = source_digest_root().expect("a computed digest has a root");
        assert!(root.is_absolute(), "{root:?}");
        assert!(root.join("Cargo.toml").is_file(), "{root:?}");
        assert!(root.join("src").is_dir(), "{root:?}");

        // And the published value names no path at all. A path separator in
        // this field is the schema 6 defect returning.
        assert!(
            !scope.contains('/') && !scope.contains('\\'),
            "the scope published a path: {scope}"
        );
        assert!(!Path::new(&scope).is_absolute(), "{scope}");
        assert_eq!(
            scope, SOURCE_DIGEST_SCOPE_REPO_ROOT,
            "the crate root is this repository's git toplevel"
        );

        // Same cache as the digest, so the pair can never disagree.
        assert_eq!(scope, resolve_source_digest_scope());
        // And the resolver the record is actually built from carries it.
        assert_eq!(resolve_platform_provenance().source_digest_scope, scope);
    }

    /// The sentinel pair is the only shape allowed to say "no tree": a digest
    /// that was not computed must not name a directory it did not walk.
    #[test]
    fn the_scope_sentinel_and_the_digest_sentinel_agree() {
        let state = compute_source_digest_in(None);
        assert_eq!(state.digest, SOURCE_DIGEST_NOT_COMPUTED);
        assert_eq!(state.scope, SOURCE_DIGEST_SCOPE_NONE);
        assert_eq!(
            state.root, None,
            "no walk, so no directory to relativize to"
        );
    }

    #[test]
    fn a_failed_trainer_resolution_never_reports_an_empty_measurement() {
        // The failure shapes this type is allowed to produce, asserted on the
        // constructor rather than by breaking `current_exe`: an empty
        // `provenance` would read as a value that came out blank.
        let unavailable = TrainerProvenance {
            path: "/nonexistent".to_string(),
            sha256: String::new(),
            provenance: "unavailable: cannot read executable: no such file".to_string(),
        };
        assert!(unavailable.provenance.starts_with("unavailable: "));
        assert!(!unavailable.provenance.is_empty());
        assert!(unavailable.sha256.is_empty());
    }
}
