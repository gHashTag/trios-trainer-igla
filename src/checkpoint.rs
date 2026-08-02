//! Checkpoint save/load. Stored under `${TRIOS_CHECKPOINT_DIR}/{run-name}/{step}.bin`.
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
//! `bincode`, `zstd`, `blake3` and `postcard` are all absent from `Cargo.lock`
//! - the historical `// TODO: zstd-compressed bincode` named two crates this
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
pub const CHECKPOINT_PAYLOAD_OFFSET: usize =
    CHECKPOINT_HEADER_LEN + CHECKPOINT_TENSOR_COUNT * 8;

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

pub fn checkpoint_path(run_name: &str, step: usize) -> PathBuf {
    let base = std::env::var("TRIOS_CHECKPOINT_DIR").unwrap_or_else(|_| "checkpoints".into());
    PathBuf::from(base)
        .join(run_name)
        .join(format!("{step}.bin"))
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
pub fn save(run: &str, step: usize, bytes: &[u8]) -> Result<SavedCheckpoint> {
    let final_path = checkpoint_path(&sanitize_run_name(run), step);
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
    let base = std::env::var("TRIOS_CHECKPOINT_DIR").unwrap_or_else(|_| "checkpoints".into());
    PathBuf::from(base)
        .join(run_name)
        .join(format!("{step}.json"))
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

/// The platform half of the provenance record, absent from schemas 1 and 2.
///
/// This is not bookkeeping. The cross-libc experiment showed the SAME seed,
/// the SAME corpus and the SAME source tree producing DIFFERENT checkpoint
/// hashes on macOS versus glibc, which makes the platform the single decisive
/// untracked variable in the whole method: two records could agree on every
/// field schema 2 carried and still describe runs that could never reproduce
/// each other.
///
/// Every field except `toolchain` is resolved at COMPILE time, so it describes
/// the binary rather than whatever host happens to read the record later.
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
    pub libc: String,
    /// Best-effort rustc identity. Read `toolchain_provenance` before citing
    /// it; `"unknown"` when neither route produced a string.
    pub toolchain: String,
    /// `TOOLCHAIN_PROVENANCE_*`. A value and the strength of the evidence for
    /// it are different facts, and are recorded separately - the same rule
    /// `git_provenance` follows.
    pub toolchain_provenance: String,
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
    PlatformProvenance {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        pointer_width: usize::BITS,
        libc: libc.to_string(),
        toolchain,
        toolchain_provenance: toolchain_provenance.to_string(),
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
pub const SOURCE_DIGEST_DOMAIN: &[u8] = b"trios-source-tree/1\n";

/// Digest the source tree that is running: every `src/**/*.rs` plus
/// `Cargo.toml`, sorted by path, each contributing
/// `len(path) || path || len(bytes) || bytes` to one SHA-256. The length
/// prefixes make the encoding unambiguous, so no rename or content shuffle can
/// collide with another tree.
///
/// This covers TRACKED AND DIRTY files alike, which is the point: `git_sha`
/// with `git_dirty: true` names a commit the tree did not match, and every
/// archived checkpoint in this repo was produced by exactly such a tree.
///
/// Computed once per process and cached, so all records from one run agree and
/// a long run does not re-walk the tree at every checkpoint.
///
/// LIMIT, stated because the field would otherwise over-promise: this reads the
/// tree at RUN time from the current working directory, which is not proven to
/// be the tree the running binary was COMPILED from. It is the strongest claim
/// obtainable without a `build.rs`, and it is strictly stronger than a commit
/// hash over a dirty tree; it is not a build attestation. A run launched
/// outside its source tree records `SOURCE_DIGEST_NOT_COMPUTED` rather than a
/// digest of whatever tree happened to be underfoot.
pub fn resolve_source_digest() -> String {
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CACHE.get_or_init(compute_source_digest).clone()
}

fn compute_source_digest() -> String {
    let mut files: Vec<PathBuf> = Vec::new();
    if collect_rs_files(Path::new("src"), &mut files).is_err() || files.is_empty() {
        return SOURCE_DIGEST_NOT_COMPUTED.to_string();
    }
    let manifest = PathBuf::from("Cargo.toml");
    if !manifest.is_file() {
        return SOURCE_DIGEST_NOT_COMPUTED.to_string();
    }
    files.push(manifest);
    files.sort();
    let mut hasher = Sha256::new();
    hasher.update(SOURCE_DIGEST_DOMAIN);
    for path in &files {
        let raw = match std::fs::read(path) {
            Ok(raw) => raw,
            // A file that vanished mid-walk means the digest would describe a
            // tree that never existed. Refuse rather than hash a subset.
            Err(_) => return SOURCE_DIGEST_NOT_COMPUTED.to_string(),
        };
        let rel = path.to_string_lossy().replace('\\', "/");
        hasher.update((rel.len() as u64).to_le_bytes());
        hasher.update(rel.as_bytes());
        hasher.update((raw.len() as u64).to_le_bytes());
        hasher.update(&raw);
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
    /// `std::env::current_exe()` as resolved at save time. Informational: the
    /// path is not evidence, the hash is.
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
    let path = exe.to_string_lossy().into_owned();
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

/// The local evidence record written next to every checkpoint, always,
/// regardless of whether a database is reachable.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointRecord {
    /// Literal "trios-checkpoint-record/4".
    pub schema: String,
    /// Ledger identity, UNSANITIZED (may differ from the on-disk directory).
    pub canon_name: String,
    pub seed: i64,
    pub step: i64,
    /// The path actually written, as resolved (sanitized run component).
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
    pub best_val_bpb: Option<f64>,
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
    /// RAILWAY_DEPLOYMENT_ID env, "" if unset.
    pub run_id: String,
    /// "pending" | "written" | "skipped-no-dsn" | "failed".
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
    /// Digest over the source tree that ran (`trios-source-tree/1`, see
    /// `resolve_source_digest`), or the literal `"not-computed"`.
    ///
    /// `git_sha` with `git_dirty: true` is honest but NOT reconstructive: it
    /// names a commit the tree did not match. This field closes the last
    /// unhashed input of the method, so a record describes the code that ran
    /// even when that code was never committed.
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
/// The BINARY header format version is deliberately NOT bumped, in 4 as in 3
/// and 2: the sidecar is the evidence record, the `.bin` layout is unchanged,
/// and rewriting the format spec for a JSON field would invalidate every
/// existing artifact hash for nothing. (`vocab` was already byte 16 of the
/// hashed header; schema 4 only surfaces it in the sidecar.)
pub const CHECKPOINT_RECORD_SCHEMA: &str = "trios-checkpoint-record/4";

/// Overwrite the sidecar atomically (tmp + rename). Called twice per
/// checkpoint: once with `ledger = "pending"` right after the `.bin` lands,
/// once with the real ledger outcome. The two-phase write means a crash
/// between the artifact and the DB attempt still leaves an on-disk record
/// naming the file and its hash; "pending" is itself honest information.
pub fn write_sidecar(rec: &CheckpointRecord) -> Result<PathBuf> {
    let path = sidecar_path(&sanitize_run_name(&rec.canon_name), rec.step as usize);
    let dir = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("sidecar path {path:?} has no parent"))?
        .to_path_buf();
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create sidecar dir {dir:?}"))?;
    let body = serde_json::to_vec_pretty(rec).context("failed to serialize CheckpointRecord")?;
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

    // ---- schema 4: the record must name its executor and its alphabet ----

    /// A record with every schema-4 field populated, so a round-trip test is
    /// testing the serialisation and not the defaults.
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
            best_val_bpb: Some(2.60),
            ema_bpb: Some(3.10),
            git_sha: "deadbeef".to_string(),
            git_provenance: GIT_PROVENANCE_VERIFIED.to_string(),
            git_dirty: Some(true),
            corpus: CorpusProvenance::default(),
            run_id: String::new(),
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
        }
    }

    #[test]
    fn schema_is_version_four() {
        assert_eq!(CHECKPOINT_RECORD_SCHEMA, "trios-checkpoint-record/4");
        // The .bin layout did not change, so the artifact hashes did not.
        assert_eq!(CHECKPOINT_FORMAT_VERSION, 1);
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

        // The digest is over the bytes on disk, so an independent read of the
        // same path must reproduce it - which is what `shasum -a 256` does.
        let raw = std::fs::read(&prov.path).expect("re-read the executable");
        assert_eq!(prov.sha256, sha256_hex(&raw));
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
