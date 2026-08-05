//! What the canonical serialization clause must do, and what it must not
//! pretend to do.
//!
//! Three properties, in order of what they defend:
//!
//! 1. two artifacts that differ ONLY in the sign of a zero have different raw
//!    digests and the SAME canonical digest -- this is the defect being
//!    closed;
//! 2. a payload carrying NaN has no canonical digest at all, and the refusal
//!    says so in words -- a canonical form for a poisoned artifact would be an
//!    invented rule, not an applied one;
//! 3. on the two PUBLISHED cross-architecture artifacts the clause removes
//!    exactly 2 073 encoding-only differences and the canonical digests STILL
//!    disagree. The third assertion is the honest limit written into the test
//!    suite, so that a future edit which "improves" the clause into declaring
//!    those two artifacts equal fails here.

use trios_trainer::canonical_digest::{
    canonical_digest, canonicalize, compare, payload_census, PayloadCensus,
    CANONICAL_DIGEST_SCHEMA, REFUSAL_MARKER,
};
use trios_trainer::checkpoint::{
    CHECKPOINT_FORMAT_VERSION, CHECKPOINT_HEADER_LEN, CHECKPOINT_MAGIC, CHECKPOINT_PAYLOAD_OFFSET,
    CHECKPOINT_TENSOR_COUNT,
};

const REFERENCE_AARCH64: &str = "evidence/xarch-aarch64-reference/12000.bin";
const CI_X86_64: &str = "evidence/xarch-run-30767491098/12000.bin";

/// Smallest buffer that passes the container check, carrying `payload`.
///
/// Only the fields `validate_container` reads are filled: magic, version,
/// header_len, tensor_count, the reserved/boolean bytes (left at zero) and a
/// tensor directory whose entries sum to `payload.len()`. Everything else is
/// zero, which is exactly the point -- the canonical digest is a property of
/// the container plus the payload, not of the model's shape.
fn synthetic_checkpoint(payload: &[f32]) -> Vec<u8> {
    let mut buf = vec![0u8; CHECKPOINT_PAYLOAD_OFFSET];
    buf[0..8].copy_from_slice(CHECKPOINT_MAGIC);
    buf[8..12].copy_from_slice(&CHECKPOINT_FORMAT_VERSION.to_le_bytes());
    buf[12..16].copy_from_slice(&(CHECKPOINT_HEADER_LEN as u32).to_le_bytes());
    buf[52..56].copy_from_slice(&(CHECKPOINT_TENSOR_COUNT as u32).to_le_bytes());
    // Whole payload declared as tensor 0; the remaining 18 entries stay 0.
    let dir0 = CHECKPOINT_HEADER_LEN;
    buf[dir0..dir0 + 8].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    for v in payload {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// The defect, in four parameters: one artifact holds `-0.0` where the other
/// holds `+0.0` and everything else is bit-identical. No forward pass can tell
/// them apart; `sha256` over the raw bytes can.
#[test]
fn signed_zero_twins_differ_raw_and_agree_canonically() {
    let a = synthetic_checkpoint(&[-0.0, 0.0, 1.5, -2.5]);
    let b = synthetic_checkpoint(&[0.0, 0.0, 1.5, -2.5]);
    assert_eq!(a.len(), b.len(), "the two buffers must be the same shape");
    assert_ne!(a, b, "the fixture must actually differ in its bytes");

    let da = canonical_digest(&a).expect("synthetic artifact A must canonicalize");
    let db = canonical_digest(&b).expect("synthetic artifact B must canonicalize");

    assert_ne!(
        da.raw_sha256, db.raw_sha256,
        "raw sha256 must react to the sign bit of zero -- that IS the defect"
    );
    assert_eq!(
        da.canonical_sha256, db.canonical_sha256,
        "canonical sha256 must not distinguish -0.0 from +0.0"
    );
    assert_eq!(da.schema, CANONICAL_DIGEST_SCHEMA);
    assert_eq!(
        da.census,
        PayloadCensus {
            params: 4,
            negative_zero_count: 1,
            nan_count: 0,
            infinite_count: 0,
        }
    );
    assert_eq!(db.census.negative_zero_count, 0);

    // The canonicalized bytes of A are literally B's bytes: normalisation is a
    // rewrite of the payload and touches nothing else.
    assert_eq!(canonicalize(&a).expect("A canonicalizes"), b);

    let pair = compare(&a, &b).expect("the pair censuses");
    assert_eq!(pair.params, 4);
    assert_eq!(pair.bitwise_differing, 1);
    assert_eq!(pair.numerically_differing, 0);
    assert_eq!(pair.signed_zero_only, 1);
    assert_eq!(pair.zero_in_both, 2);
    assert!(!pair.raw_digests_equal);
    assert!(pair.canonical_digests_equal);
    assert!(
        pair.canonicalization_changes_verdict,
        "here, and only here, the clause is allowed to flip a MISMATCH"
    );
}

/// A canonical digest is not defined for a poisoned artifact, and the refusal
/// must say which parameters poisoned it.
#[test]
fn a_nan_payload_has_no_canonical_digest() {
    let poisoned = synthetic_checkpoint(&[1.0, f32::NAN, -0.0, 3.0]);

    // The census still answers -- an operator needs the count that justifies
    // the refusal, so counting and refusing are deliberately separate.
    let census = payload_census(&poisoned).expect("a poisoned payload can still be censused");
    assert_eq!(census.nan_count, 1);
    assert_eq!(census.negative_zero_count, 1);

    let err = canonical_digest(&poisoned)
        .expect_err("a payload carrying NaN must not receive a canonical digest");
    let text = format!("{err:#}");
    assert!(
        text.contains(REFUSAL_MARKER),
        "the error must name the refusal, got: {text}"
    );
    assert!(
        text.contains("NaN"),
        "the error must name what it refused over, got: {text}"
    );
    assert!(
        canonicalize(&poisoned).is_err(),
        "canonicalize must refuse on the same terms as canonical_digest"
    );

    // An infinity is refused for the same reason and by the same clause.
    let infinite = synthetic_checkpoint(&[1.0, f32::INFINITY]);
    let err = canonicalize(&infinite).expect_err("an infinite parameter must be refused");
    assert!(format!("{err:#}").contains(REFUSAL_MARKER), "{err:#}");
}

/// The measured census over the two artifacts the repository publishes, and
/// the limit it does not cross.
///
/// Skips with a printed note if either artifact is absent, so a shallow
/// checkout does not turn a missing file into a false pass or a false failure.
#[test]
fn published_cross_architecture_pair_is_2073_signed_zeros_and_still_mismatches() {
    let (a, b) = match (std::fs::read(REFERENCE_AARCH64), std::fs::read(CI_X86_64)) {
        (Ok(a), Ok(b)) => (a, b),
        _ => {
            println!(
                "SKIP: {REFERENCE_AARCH64} and/or {CI_X86_64} not present in this \
                 checkout; the published-pair census cannot be measured here."
            );
            return;
        }
    };

    let c = compare(&a, &b).expect("both published artifacts must pass the container check");

    assert_eq!(c.params, 212_992, "parameter count of the published pair");
    assert_eq!(
        c.signed_zero_only, 2_073,
        "parameters differing ONLY in the sign bit of zero"
    );
    assert_eq!(c.bitwise_differing, 95_144, "bitwise-differing parameters");
    assert_eq!(
        c.numerically_differing, 93_071,
        "numerically-differing parameters -- the number behind the published \
         43.70% figure"
    );
    assert_eq!(c.zero_in_both, 33_503, "parameters that are zero in both");
    assert_eq!(
        c.bitwise_differing - c.numerically_differing,
        c.signed_zero_only,
        "the gap between the two counts is exactly the signed-zero population; \
         if it is not, one of the three is measuring something else"
    );

    // The honest limit. 2 073 of 95 144 differences are an artefact of the
    // encoding; the other 93 071 are real, and no serialization clause can
    // remove them.
    assert!(!c.raw_digests_equal, "the published pair is a MISMATCH");
    assert!(
        !c.canonical_digests_equal,
        "canonicalization must NOT make the two architectures agree -- if this \
         assertion ever fails, the clause has started hiding the finding it was \
         written to make well-founded"
    );
    assert!(
        !c.canonicalization_changes_verdict,
        "the verdict on the published pair is unchanged by the clause"
    );
}
