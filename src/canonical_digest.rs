//! Canonical serialization digest for checkpoint artifacts.
//!
//! # The defect this closes
//!
//! The reproducibility criterion this crate offers as THE decidable test is
//! `sha256` over the raw checkpoint bytes. That digest is sensitive to a
//! distinction the arithmetic is not: IEEE 754 defines `-0.0 == +0.0`, every
//! forward pass in `train_loop.rs` treats the two identically, and no
//! measurable property of the model depends on which one a parameter holds --
//! yet their bit patterns differ (`0x80000000` vs `0x00000000`) and so their
//! `sha256` differs. A raw-bytes digest can therefore report MISMATCH between
//! two artifacts that are numerically the same object.
//!
//! This is not hypothetical. On the two published cross-architecture artifacts
//! (`evidence/xarch-aarch64-reference/12000.bin` and
//! `evidence/xarch-run-30767491098/12000.bin`, 212 992 parameters each) exactly
//! 2 073 parameters differ ONLY in the sign bit of zero. The repository's own
//! headline figure -- 43.70% divergence -- is 93 071/212 992, the NUMERICALLY
//! differing count, while the raw digest is reacting to 95 144. The published
//! statistic and the published criterion disagree about what a difference is.
//!
//! # The clause
//!
//! A canonical digest is defined over a *normalised* copy of the artifact:
//!
//! 1. header and tensor directory are hashed exactly as written (they carry no
//!    signed zero: every float in the header is a fixed configuration constant
//!    that the loader re-checks bit-for-bit anyway);
//! 2. every payload word whose bit pattern is `0x80000000` becomes
//!    `0x00000000`;
//! 3. byte order is little-endian and already fixed by the format's explicit
//!    `to_le_bytes`, so no swap is applied and none is needed;
//! 4. a payload containing NaN or an infinity has NO canonical digest. This is
//!    a refusal, not a normalisation: NaN has 2^24 - 2 distinct encodings and
//!    `NaN != NaN`, so any rule mapping them onto one representative would be
//!    an arbitrary choice dressed as a standard, and an artifact carrying a
//!    poisoned parameter is not a reproducibility claim to be graded -- it is a
//!    broken run.
//!
//! # What this does NOT buy
//!
//! Stated first because it is the part that gets dropped in retelling: on the
//! two published artifacts, canonicalisation removes 2 073 of 95 144 bitwise
//! differences and leaves 93 071 genuine numeric divergences. The
//! cross-architecture mismatch survives the clause completely intact. The
//! clause makes the criterion *well-founded* -- it stops the test from
//! reporting a difference where the arithmetic sees none -- it does not make
//! the artifacts equal, and nothing here weakens the published finding that a
//! checkpoint is not byte-portable across CPU architectures.
//!
//! # Scope of the container validation
//!
//! `validate_container` re-checks the FORMAT invariants that
//! `HybridModel::from_checkpoint_bytes` checks -- magic, format version,
//! header length, reserved and boolean header bytes, tensor count, and the
//! file length implied by the tensor directory (which is what rejects a
//! truncation). It deliberately does NOT re-check the build's shape constants
//! (`VOCAB`, `DIM`, `NUM_CTX`, `NGRAM`) or `HybridAttnConfig::validate`,
//! because a canonical digest must be computable by an independent
//! implementer holding only the format specification in `checkpoint.rs`; a
//! digest that no second implementation can reproduce is not a standard.

use crate::checkpoint::{
    sha256_hex, CHECKPOINT_FORMAT_VERSION, CHECKPOINT_HEADER_LEN, CHECKPOINT_MAGIC,
    CHECKPOINT_PAYLOAD_OFFSET, CHECKPOINT_TENSOR_COUNT,
};
use anyhow::Result;

/// Schema tag written into the JSON evidence record.
pub const CANONICAL_DIGEST_SCHEMA: &str = "canonical-digest/1";

/// Bit pattern of IEEE 754 negative zero in `f32`.
pub const NEGATIVE_ZERO_BITS: u32 = 0x8000_0000;

/// Substring every NaN/infinity refusal carries, so a caller (and a test) can
/// recognise the refusal without matching the whole sentence.
pub const REFUSAL_MARKER: &str = "REFUSING TO CANONICALIZE";

/// What one artifact's payload contains.
///
/// There is deliberately no `bitwise_differing` field: "how many parameters
/// differ" is not a property of one artifact. It is the caller's business,
/// answered by [`compare`] over a pair. Reporting it here would require this
/// struct to invent a second artifact it was never given.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct PayloadCensus {
    /// Number of `f32` words in the payload.
    pub params: usize,
    /// Parameters whose bit pattern is exactly `0x80000000`.
    pub negative_zero_count: usize,
    /// Parameters that are NaN under any encoding.
    pub nan_count: usize,
    /// Parameters that are `+inf` or `-inf`.
    pub infinite_count: usize,
}

/// Both digests of one artifact, plus the census that motivates the second.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CanonicalDigest {
    pub schema: String,
    /// `sha256` of the file exactly as it sits on disk. Equal to `shasum -a 256`.
    pub raw_sha256: String,
    /// `sha256` after clause 2 of the canonical serialization rule.
    pub canonical_sha256: String,
    pub census: PayloadCensus,
}

/// Read a little-endian `u32` at `off`. The caller has already length-checked.
fn rd_u32(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

/// Read a little-endian `u64` at `off`. The caller has already length-checked.
fn rd_u64(bytes: &[u8], off: usize) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&bytes[off..off + 8]);
    u64::from_le_bytes(w)
}

/// Check the container invariants of the on-disk format and return the number
/// of `f32` words in the payload.
///
/// Every error names the field and both values, because the point of this
/// function is to make a rejected artifact diagnosable without a hex editor.
pub fn validate_container(bytes: &[u8]) -> Result<usize> {
    anyhow::ensure!(
        bytes.len() >= CHECKPOINT_PAYLOAD_OFFSET,
        "checkpoint truncated: {} bytes, need at least {CHECKPOINT_PAYLOAD_OFFSET}",
        bytes.len()
    );
    anyhow::ensure!(&bytes[0..8] == CHECKPOINT_MAGIC, "bad magic (not TRIOSCKP)");
    let version = rd_u32(bytes, 8);
    anyhow::ensure!(
        version == CHECKPOINT_FORMAT_VERSION,
        "unsupported checkpoint format_version {version} \
         (this build reads {CHECKPOINT_FORMAT_VERSION})"
    );
    let header_len = rd_u32(bytes, 12) as usize;
    anyhow::ensure!(
        header_len == CHECKPOINT_HEADER_LEN,
        "header_len {header_len} != {CHECKPOINT_HEADER_LEN}"
    );
    anyhow::ensure!(
        bytes[126] == 0 && bytes[127] == 0,
        "reserved bytes 126..128 are nonzero"
    );
    anyhow::ensure!(
        bytes[124] <= 1 && bytes[125] <= 1,
        "boolean header bytes 124/125 must be 0 or 1"
    );
    let tensor_count = rd_u32(bytes, 52) as usize;
    anyhow::ensure!(
        tensor_count == CHECKPOINT_TENSOR_COUNT,
        "tensor_count {tensor_count} != {CHECKPOINT_TENSOR_COUNT}"
    );

    let mut total: u64 = 0;
    for i in 0..CHECKPOINT_TENSOR_COUNT {
        total = total.saturating_add(rd_u64(bytes, CHECKPOINT_HEADER_LEN + 8 * i));
    }
    let want_len = CHECKPOINT_PAYLOAD_OFFSET as u64 + 4 * total;
    anyhow::ensure!(
        bytes.len() as u64 == want_len,
        "checkpoint length {} != {want_len} implied by the tensor directory",
        bytes.len()
    );
    Ok(total as usize)
}

/// Census one artifact's payload WITHOUT refusing a poisoned one.
///
/// Separate from [`canonical_digest`] on purpose: an operator holding a
/// checkpoint full of NaN still needs to be told how many, and a function that
/// refuses can never report the count that justified the refusal.
pub fn payload_census(bytes: &[u8]) -> Result<PayloadCensus> {
    let params = validate_container(bytes)?;
    let mut census = PayloadCensus {
        params,
        ..Default::default()
    };
    for i in 0..params {
        let bits = rd_u32(bytes, CHECKPOINT_PAYLOAD_OFFSET + 4 * i);
        if bits == NEGATIVE_ZERO_BITS {
            census.negative_zero_count += 1;
        }
        let v = f32::from_bits(bits);
        if v.is_nan() {
            census.nan_count += 1;
        } else if v.is_infinite() {
            census.infinite_count += 1;
        }
    }
    Ok(census)
}

/// Payload bytes with every negative zero rewritten to positive zero, header
/// and tensor directory untouched.
///
/// Returns `Err` if the payload carries NaN or an infinity: clause 4.
pub fn canonicalize(bytes: &[u8]) -> Result<Vec<u8>> {
    let census = payload_census(bytes)?;
    anyhow::ensure!(
        census.nan_count == 0 && census.infinite_count == 0,
        "{REFUSAL_MARKER}: payload carries {} NaN and {} infinite parameter(s) \
         out of {}. A canonical digest is not defined for a poisoned artifact: \
         NaN has 2^24-2 encodings and compares unequal to itself, so mapping \
         them onto one representative would invent a rule rather than apply \
         one. Fix the run; do not normalise the wreckage.",
        census.nan_count,
        census.infinite_count,
        census.params
    );

    let mut out = bytes.to_vec();
    for i in 0..census.params {
        let off = CHECKPOINT_PAYLOAD_OFFSET + 4 * i;
        if rd_u32(&out, off) == NEGATIVE_ZERO_BITS {
            out[off..off + 4].copy_from_slice(&0f32.to_le_bytes());
        }
    }
    Ok(out)
}

/// Raw digest, canonical digest and the census, for one artifact.
pub fn canonical_digest(bytes: &[u8]) -> Result<CanonicalDigest> {
    let census = payload_census(bytes)?;
    let canonical = canonicalize(bytes)?;
    Ok(CanonicalDigest {
        schema: CANONICAL_DIGEST_SCHEMA.to_string(),
        raw_sha256: sha256_hex(bytes),
        canonical_sha256: sha256_hex(&canonical),
        census,
    })
}

/// The four-way census over a PAIR of artifacts.
///
/// `bitwise_differing` counts parameters whose 32 bits differ;
/// `numerically_differing` counts those the arithmetic can actually tell apart
/// (`a != b` under IEEE 754, which is what makes `-0.0` vs `+0.0` invisible
/// here); `signed_zero_only` is exactly the gap between the two, and is the
/// number the canonical clause removes.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PairCensus {
    pub schema: String,
    pub params: usize,
    pub bitwise_differing: usize,
    pub numerically_differing: usize,
    pub signed_zero_only: usize,
    pub zero_in_both: usize,
    pub raw_sha256_a: String,
    pub raw_sha256_b: String,
    pub canonical_sha256_a: String,
    pub canonical_sha256_b: String,
    pub raw_digests_equal: bool,
    pub canonical_digests_equal: bool,
    /// True only when the clause flips the verdict, i.e. the raw digests
    /// disagree and the canonical ones agree. On the published pair this is
    /// `false`, and saying so is the point.
    pub canonicalization_changes_verdict: bool,
    pub negative_zero_count_a: usize,
    pub negative_zero_count_b: usize,
}

/// Compare two artifacts under both digests.
///
/// Refuses unless both files pass [`validate_container`] and carry the same
/// parameter count: two artifacts of different shape are not two measurements
/// of one thing, and a percentage computed across them would be meaningless.
pub fn compare(a: &[u8], b: &[u8]) -> Result<PairCensus> {
    let da = canonical_digest(a)?;
    let db = canonical_digest(b)?;
    anyhow::ensure!(
        da.census.params == db.census.params,
        "artifacts hold {} and {} parameters; a per-parameter census across \
         two different shapes would not describe anything",
        da.census.params,
        db.census.params
    );

    let params = da.census.params;
    let mut bitwise_differing = 0usize;
    let mut numerically_differing = 0usize;
    let mut signed_zero_only = 0usize;
    let mut zero_in_both = 0usize;
    for i in 0..params {
        let off = CHECKPOINT_PAYLOAD_OFFSET + 4 * i;
        let (ba, bb) = (rd_u32(a, off), rd_u32(b, off));
        let (fa, fb) = (f32::from_bits(ba), f32::from_bits(bb));
        if fa == 0.0 && fb == 0.0 {
            zero_in_both += 1;
        }
        if ba != bb {
            bitwise_differing += 1;
            if fa == fb {
                // The only way two distinct bit patterns compare equal in
                // IEEE 754, NaN having been refused above.
                signed_zero_only += 1;
            }
        }
        if fa != fb {
            numerically_differing += 1;
        }
    }

    let raw_digests_equal = da.raw_sha256 == db.raw_sha256;
    let canonical_digests_equal = da.canonical_sha256 == db.canonical_sha256;
    Ok(PairCensus {
        schema: CANONICAL_DIGEST_SCHEMA.to_string(),
        params,
        bitwise_differing,
        numerically_differing,
        signed_zero_only,
        zero_in_both,
        raw_sha256_a: da.raw_sha256,
        raw_sha256_b: db.raw_sha256,
        canonical_sha256_a: da.canonical_sha256,
        canonical_sha256_b: db.canonical_sha256,
        raw_digests_equal,
        canonical_digests_equal,
        canonicalization_changes_verdict: !raw_digests_equal && canonical_digests_equal,
        negative_zero_count_a: da.census.negative_zero_count,
        negative_zero_count_b: db.census.negative_zero_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_zero_bits_are_what_the_platform_says() {
        assert_eq!((-0.0f32).to_bits(), NEGATIVE_ZERO_BITS);
        assert_eq!(0.0f32.to_bits(), 0);
        // The whole premise: the arithmetic cannot see the difference.
        assert!(-0.0f32 == 0.0f32);
    }

    #[test]
    fn a_truncated_buffer_is_refused_before_any_hashing() {
        let err = validate_container(&[0u8; 16]).expect_err("16 bytes is not a checkpoint");
        assert!(err.to_string().contains("truncated"), "{err}");
    }
}
