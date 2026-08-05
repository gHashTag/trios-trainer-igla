# Canonical serialization of a checkpoint artifact

Status: proposed clause, measured on the two published cross-architecture
artifacts. Implementation: `src/canonical_digest.rs`, `src/bin/canon_digest.rs`,
`tests/canonical_digest.rs`. Evidence:
`evidence/canonical-digest/signed-zero-census.json`.

## What this does NOT buy (read this first)

Canonicalization does not make the two published artifacts equal, and it does
not soften the cross-architecture finding. Of the 95 144 parameters whose bits
differ between the `aarch64-apple-darwin` reference and the `x86_64` Linux CI
run, **93 071 are genuine numeric divergence** and only 2 073 are an artefact of
the encoding. Under the clause below the two artifacts still produce different
digests:

```
RAW-VERDICT        MISMATCH
CANONICAL-VERDICT  MISMATCH
```

The clause makes the criterion **well-founded** -- it stops the test from
reporting a difference where the arithmetic sees none. It does not make the
artifacts equal. Nothing in this document changes the published conclusion that
a checkpoint is not byte-portable across CPU architectures.

## The defect

The reproducibility criterion this project offers as *the* decidable test is
`sha256` over the raw checkpoint bytes: two parties run the same pipeline, hash
the artifacts, and compare one string. The strength of that criterion is that it
needs no knowledge of the format -- `shasum -a 256` settles it.

Its weakness is that it is sensitive to a distinction the arithmetic ignores.
IEEE 754 defines `-0.0 == +0.0` as **true**; the two have different bit patterns
(`0x80000000` and `0x00000000`); no forward pass in `src/train_loop.rs`
distinguishes them, because every use of a parameter is an add or a multiply and
both operations give identical results for the two zeros. So a raw-bytes digest
can report MISMATCH between two artifacts that are the same object numerically.

Worse, the repository is already inconsistent about this. The headline figure
for the cross-architecture divergence, 43.70%, is `93071 / 212992` -- the
*numerically* differing count. The digest that decides PASS/FAIL is reacting to
95 144. **The published statistic and the published criterion disagree about
what a difference is.** That gap is what an opponent attacks, and today there is
no answer.

## The measured census

Two artifacts, both 852 272 bytes, payload starting at byte offset 304
(`CHECKPOINT_HEADER_LEN + CHECKPOINT_TENSOR_COUNT * 8`), 212 992 `f32`
parameters each:

| Quantity | Count | Share of params |
|---|---:|---:|
| parameters | 212 992 | 100.00% |
| bitwise differing (what `sha256` sees) | 95 144 | 44.67% |
| numerically differing (what the model sees) | 93 071 | 43.70% |
| differing ONLY in the sign bit of zero | 2 073 | 0.97% |
| zero in both artifacts | 33 503 | 15.73% |
| negative zeros, aarch64 reference | 8 119 | 3.81% |
| negative zeros, x86_64 CI run | 7 947 | 3.73% |

`bitwise_differing - numerically_differing = signed_zero_only` exactly, which is
the arithmetic identity that makes the table checkable: NaN having been refused,
the only way two distinct `f32` bit patterns can compare equal is a signed zero.

Reproduce:

```
cargo run --release --bin canon_digest -- \
  evidence/xarch-aarch64-reference/12000.bin \
  evidence/xarch-run-30767491098/12000.bin
```

Add `--json` for the machine-readable record; that is exactly how
`evidence/canonical-digest/signed-zero-census.json` was generated. The raw
digests it prints are the two already published in `README.md`
(`8a86fe69...` aarch64, `bb14ab18...` x86_64).

## The clause

> **Canonical form of a `TRIOSCKP` version-1 artifact.**
>
> 1. **Container.** The artifact MUST pass the container check: magic
>    `TRIOSCKP`, `format_version = 1`, `header_len = 152`, `tensor_count = 19`,
>    reserved bytes 126..128 zero, boolean header bytes 124 and 125 in `{0, 1}`,
>    and a total file length exactly equal to
>    `304 + 4 * sum(tensor directory)`. An artifact failing any of these has no
>    canonical form.
> 2. **Byte order.** All multi-byte quantities are little-endian. This is not a
>    normalisation step: the format already writes every integer and every float
>    through explicit `to_le_bytes`, so byte order is fixed at the point of
>    writing and no swap is applied on either architecture.
> 3. **Header and tensor directory.** Hashed exactly as written. They carry no
>    signed zero -- every float in the header is a fixed configuration constant
>    (`qk_gain`, `lr`, `attn_scale`, `ctx_weights[0..5]`) that the loader
>    re-checks bit-for-bit against the build's own constants.
> 4. **Negative zero.** Every payload word whose bit pattern is `0x80000000` is
>    rewritten to `0x00000000` before hashing.
> 5. **NaN and infinity.** An artifact whose payload contains any NaN or any
>    infinity has **no canonical digest**. This is a refusal, not a
>    normalisation.
> 6. **Digest.** The canonical digest is the SHA-256 of the byte sequence
>    produced by steps 3 and 4.

Two artifacts are *canonically identical* iff their canonical digests are equal.
Canonical identity implies numeric identity of every parameter, and numeric
identity of every parameter implies canonical identity -- which is precisely the
property the raw digest lacks.

### Why NaN is a refusal and not a normalisation

A NaN has 2^24 - 2 distinct `f32` encodings, and `NaN != NaN`. Any rule mapping
them onto one representative would be an arbitrary choice presented as a
standard, and it would make two artifacts "canonically identical" while their
models disagree on every prediction. More to the point, an artifact carrying a
poisoned parameter is not a reproducibility claim awaiting a grade -- it is a
broken run, and the correct output of a conformity check on a broken run is a
refusal that names the count. `canonical_digest` therefore returns an error
carrying the marker `REFUSING TO CANONICALIZE` and the NaN/infinity counts,
while `payload_census` still reports those counts: the function that refuses
must not be the only one that can see the evidence for the refusal.

### Why the header is hashed as written

A canonical form should normalise the fewest things that need normalising. The
header holds no parameter, no accumulated float and no free choice: every one of
its float fields is re-validated by
`HybridModel::from_checkpoint_bytes` against this build's constants, bit for
bit. Normalising it would add a rule with no defect behind it, and every rule
in a conformity clause is a rule a second implementer can get wrong.

## Relation to 243-FZ conformity assessment

The clause is a candidate for the "confirmation of conformity" criterion at the
one place the criterion has measurable technical content: it turns
"the artifacts are identical" from a statement about a file into a statement
about the model the file encodes, and it does so with a decision procedure that
a third party can execute with no knowledge of this crate beyond the format
specification in `src/checkpoint.rs`. A criterion that can return MISMATCH for
two identical models is not a criterion; a criterion that returns MISMATCH only
when the arithmetic actually differs is one.

It also carries its own scope statement. The clause is written to be checkable
by an independent implementation, so `validate_container` in
`src/canonical_digest.rs` deliberately checks only FORMAT invariants and not the
build's shape constants (`VOCAB`, `DIM`, `NUM_CTX`, `NGRAM`) or
`HybridAttnConfig::validate`. A digest that only this binary can compute would
not be a standard.

## Tests that pin this

`tests/canonical_digest.rs`:

1. `signed_zero_twins_differ_raw_and_agree_canonically` -- a four-parameter
   synthetic pair differing only in one zero's sign: raw digests differ,
   canonical digests are equal, and the canonicalized bytes of the first are
   literally the second's bytes.
2. `a_nan_payload_has_no_canonical_digest` -- a NaN payload is refused, the
   error names the refusal, and the census still reports the count. An infinity
   is refused on the same terms.
3. `published_cross_architecture_pair_is_2073_signed_zeros_and_still_mismatches`
   -- the whole table above, asserted on the two published artifacts, INCLUDING
   the assertion that the canonical digests still differ. That last assertion is
   the honest limit written into the suite: an edit that "improves" the clause
   into declaring the two architectures equal fails this test.
