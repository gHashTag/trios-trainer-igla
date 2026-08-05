#!/usr/bin/env python3
"""Independent TRIOSCKP reader.

Second implementation of the TRIOSCKP checkpoint container, written in Python 3
from the format specification in interop/SPEC-SNAPSHOT.txt alone. The Rust
encoder and decoder (`save`/`load` in src/checkpoint.rs, `to_checkpoint_bytes`
and `from_checkpoint_bytes` in src/train_loop.rs) were deliberately not read
while writing this file, so that agreement between the two implementations is
evidence about the specification and not about a shared reading of one source
tree.

Standard library only: struct, hashlib, json, argparse, sys, os.

Sidecar schema versions. The reader accepts trios-checkpoint-record/1 through
/9 and dispatches on FIELD PRESENCE, not on matching the `schema` string, so an
unseen tag is still read for everything it carries and a tag that claims fields
the record lacks is reported. For every record it names which of the post-schema-1
fields are absent - all of them are absent from the oldest archived checkpoints
here, which are schema 1.

Unrecognised keys. Every top-level key, every `platform` key and every key of the
four nested objects this reader knows (`corpus`, `corpus.train`, `corpus.val`,
`trainer`, `optimizer_params`) is checked against a table of names this reader
has meanings for. A key that is not in the table is reported as "present but NOT
INTERPRETED". It is a note, never a silent pass: a field the instrument cannot
see is a field the instrument cannot agree or disagree about, and reporting
agreement while ignoring it would overstate what the comparison covered.

Scope of the verdict. A PASS from this reader is not a statement about the whole
record. Every result names three disjoint sets: fields CROSS-CHECKED against the
decoded container bytes, fields ECHOED from the sidecar with no container
counterpart to check them against, and keys NOT INTERPRETED at all. Only the
first set is interlaboratory agreement. The provenance block (`platform`,
`source_sha256`, `trainer`) and the sampling plan (`eval_chunks`, `eval_tokens`,
`eval_seq`, `val_bpb_stderr`) are in the second set: this instrument cannot
confirm or refute a single one of them, and no citation of its agreement may
imply otherwise.

Derived architecture. One statement in the VERIFIED column is not a comparison
against a sidecar field but a measurement of the payload: the reader decodes the
tensor directory, walks the four layer-2 attention projections (wq2, wk2, wv2,
wo2) and counts how many of their f32 elements are exactly zero. A record may
declare `num_attn_layers: 2` while every byte of the second layer is zero, in
which case the second layer contributes nothing to any forward pass and the
EFFECTIVE count is one. That is reported as a measurement of the container and
never as an accusation - see `derive_architecture` for what it does and does not
license - and it is generic: a record whose layer-2 tensors hold any non-zero
element is reported as effective 2.

The `ledger` field. Unlike every other sidecar-only field, `ledger` is checked
against a closed vocabulary and an unknown value FAILS the record instead of
being echoed. It is the field a conformity report quotes to say that nothing
outside the checkout was told what was measured, so a string this reader cannot
interpret is a claim it must not pass on. See LEDGER_VALUES.

Forward compatibility. A tag ABOVE the highest version in this reader's table is
a note, not a failure: if the record carries every field of the highest known
version, everything this reader can check has been checked, and the only honest
report is that the record claims fields whose meaning is not yet known here. A
tag that claims a version this reader DOES know while omitting that version's
fields stays a failure - that is an over-promise about the record's own content,
and it is exactly what a conformity reader exists to catch.

SCOPE CAVEAT, so the independence claim above is not overstated: the container
decoding was written from the specification alone, but the sidecar field NAMES
for schemas 3 through 7 were derived by diffing the sidecar records on disk
against each other - not from the encoder, and for schemas 6 and 7 not from any
prose either. Agreement on `lr`, `attn_scale`, `attn_seq`, `vocab` and
`gf16_enabled` between sidecar and header is still a genuine cross-check of two
encodings of the same quantity; it is not evidence about the specification,
because the sidecar was never in the specification (see interop/README.md,
ambiguity A6).

SECOND SCOPE CAVEAT, narrower and newer, 2026-08-05. The schema 8 and 9 rows of
the table below were read from the version notes above `CHECKPOINT_RECORD_SCHEMA`
in src/checkpoint.rs - that is, from the ENCODER's own prose - and then confirmed
against the records on disk. The two halves of that sentence carry different
weight and are stated separately on purpose:

  * confirmed by artifact: every one of the 87 records on disk tagged /8 or /9
    carries `git_untracked`, `platform.libc_provenance` and `platform.features`,
    and all 16 /9 records carry `format_faithful`. That part is the same
    diff-derived evidence as schemas 6 and 7.
  * NOT independent: the reason `platform.libc_version` is listed as a known
    field and NOT as a marker came from the encoder, which serializes it only
    when a version query succeeded. No artifact on disk shows that case, so no
    amount of diffing would have found it.

Nothing here touches the container decode, which still does not read `save` /
`load` or `to_checkpoint_bytes` / `from_checkpoint_bytes`. But a schema table
partly copied from the writer is not a second opinion about the schema, and a
citation that leans on this reader's schema bookkeeping must say so.

Exit codes:
    0  every requested check passed
    1  at least one check failed (bad container, or sidecar disagreement)
    2  usage error (unreadable path, bad arguments)
"""

import argparse
import hashlib
import json
import os
import struct
import sys

MAGIC = b"TRIOSCKP"
SPEC_FORMAT_VERSION = 1
SPEC_HEADER_LEN = 152
SPEC_TENSOR_COUNT = 19
DIRECTORY_LEN = 8 * SPEC_TENSOR_COUNT
PAYLOAD_OFFSET = SPEC_HEADER_LEN + DIRECTORY_LEN  # 304, per the file_len formula
ELEMENT_SIZE = 4  # f32, implied by "file_len = 304 + 4 * sum(directory counts)"

# Canonical tensor order, spec lines 46-51. Order MUST NOT CHANGE in version 1.
TENSOR_NAMES = [
    "embed",
    "ctx0", "ctx1", "ctx2", "ctx3", "ctx4", "ctx5",
    "proj",
    "attn_down",
    "attn_up",
    "lm_head",
    "wq", "wk", "wv", "wo",
    "wq2", "wk2", "wv2", "wo2",
]

# The four projections of the SECOND attention layer, in directory order. Named
# here rather than sliced out of TENSOR_NAMES by position so that the derived
# check below reads as what it is - a statement about these four tensors - and
# breaks loudly rather than silently if the canonical order ever changes.
LAYER2_TENSORS = ["wq2", "wk2", "wv2", "wo2"]

# The f32 bit pattern of positive zero. The zero census below tests BYTES, not
# decoded floats, so "exactly zero" means these four bytes and nothing else: no
# tolerance, no rounding, and negative zero counted separately because it is a
# different bit pattern that is nonetheless numerically inert.
F32_POSITIVE_ZERO = b"\x00\x00\x00\x00"
F32_NEGATIVE_ZERO = b"\x00\x00\x00\x80"

# Header fields at absolute byte offsets, spec lines 19-39.
# (name, offset, struct format). All little-endian; floats are IEEE-754 bit
# patterns, which is exactly what struct's '<f' and '<d' decode.
U32_FIELDS = [
    ("vocab", 16),
    ("dim", 20),
    ("num_ctx", 24),
    ("hidden", 28),
    ("d_model", 32),
    ("num_heads", 36),
    ("attn_cfg_seq_len", 40),
    ("num_attn_layers", 44),
    ("ngram", 48),
    ("tensor_count", 52),
]

CTX_WEIGHT_COUNT = 6
CTX_WEIGHTS_OFFSET = 84
OPTIMIZER_OFFSET = 128
OPTIMIZER_LEN = 8
FAKE_QUANT_OFFSET = 136
FAKE_QUANT_LEN = 16
RESERVED_OFFSET = 126
RESERVED_LEN = 2


class CheckpointError(Exception):
    """A named container defect. `kind` is the stable machine-readable name."""

    def __init__(self, kind, detail):
        super().__init__("%s: %s" % (kind, detail))
        self.kind = kind
        self.detail = detail


def decode_ascii_field(raw, offset, length, field_name):
    """Decode an ASCII, NUL-padded fixed-width field.

    The spec says "ASCII, NUL-padded" and nothing more, so this reader enforces
    the strictest reading that the words support: bytes before the first NUL are
    the value, they must be ASCII, and every byte after the first NUL must also
    be NUL (no data hidden in the padding).
    """
    field = raw[offset:offset + length]
    nul = field.find(b"\x00")
    if nul < 0:
        value_bytes, padding = field, b""
    else:
        value_bytes, padding = field[:nul], field[nul:]
    if padding.strip(b"\x00"):
        raise CheckpointError(
            "NONZERO_STRING_PADDING",
            "%s padding after NUL is not all zero: %r" % (field_name, padding),
        )
    for byte in value_bytes:
        if byte < 0x20 or byte > 0x7E:
            raise CheckpointError(
                "NON_ASCII_STRING",
                "%s contains byte 0x%02x outside printable ASCII" % (field_name, byte),
            )
    return value_bytes.decode("ascii")


def parse_container(raw, path):
    """Decode a TRIOSCKP file image. Raises CheckpointError on any defect."""
    info = {"path": path, "bytes": len(raw)}

    if len(raw) < len(MAGIC):
        raise CheckpointError(
            "TRUNCATED_MAGIC",
            "file is %d bytes, need at least %d for the magic" % (len(raw), len(MAGIC)),
        )
    if raw[:len(MAGIC)] != MAGIC:
        raise CheckpointError(
            "BAD_MAGIC",
            "expected %r at offset 0, found %r" % (MAGIC, raw[:len(MAGIC)]),
        )

    if len(raw) < SPEC_HEADER_LEN:
        raise CheckpointError(
            "TRUNCATED_HEADER",
            "file is %d bytes, header alone is %d" % (len(raw), SPEC_HEADER_LEN),
        )

    info["format_version"] = struct.unpack_from("<I", raw, 8)[0]
    if info["format_version"] != SPEC_FORMAT_VERSION:
        raise CheckpointError(
            "UNSUPPORTED_FORMAT_VERSION",
            "this reader implements version %d, file declares %d"
            % (SPEC_FORMAT_VERSION, info["format_version"]),
        )

    info["header_len"] = struct.unpack_from("<I", raw, 12)[0]
    if info["header_len"] != SPEC_HEADER_LEN:
        raise CheckpointError(
            "BAD_HEADER_LEN",
            "version 1 fixes header_len at %d, file declares %d"
            % (SPEC_HEADER_LEN, info["header_len"]),
        )

    for name, offset in U32_FIELDS:
        info[name] = struct.unpack_from("<I", raw, offset)[0]

    if info["tensor_count"] != SPEC_TENSOR_COUNT:
        raise CheckpointError(
            "BAD_TENSOR_COUNT",
            "version 1 fixes tensor_count at %d, file declares %d"
            % (SPEC_TENSOR_COUNT, info["tensor_count"]),
        )

    info["qk_gain"] = struct.unpack_from("<d", raw, 56)[0]
    info["attn_cfg_lr"] = struct.unpack_from("<d", raw, 64)[0]
    info["train_lr"] = struct.unpack_from("<f", raw, 72)[0]
    info["attn_scale"] = struct.unpack_from("<f", raw, 76)[0]
    info["attn_seq"] = struct.unpack_from("<I", raw, 80)[0]
    info["ctx_weights"] = list(
        struct.unpack_from("<%df" % CTX_WEIGHT_COUNT, raw, CTX_WEIGHTS_OFFSET)
    )
    info["seed"] = struct.unpack_from("<Q", raw, 108)[0]
    info["step"] = struct.unpack_from("<Q", raw, 116)[0]
    info["gf16_enabled"] = raw[124]
    info["data_synthetic"] = raw[125]

    reserved = raw[RESERVED_OFFSET:RESERVED_OFFSET + RESERVED_LEN]
    if reserved != b"\x00" * RESERVED_LEN:
        raise CheckpointError(
            "RESERVED_NONZERO",
            "the %d reserved bytes at offset %d must be zero, found %r"
            % (RESERVED_LEN, RESERVED_OFFSET, reserved),
        )

    info["optimizer"] = decode_ascii_field(
        raw, OPTIMIZER_OFFSET, OPTIMIZER_LEN, "optimizer"
    )
    info["fake_quant_format"] = decode_ascii_field(
        raw, FAKE_QUANT_OFFSET, FAKE_QUANT_LEN, "fake_quant_format"
    )

    if len(raw) < PAYLOAD_OFFSET:
        raise CheckpointError(
            "TRUNCATED_DIRECTORY",
            "file is %d bytes, header plus %d-entry directory is %d"
            % (len(raw), SPEC_TENSOR_COUNT, PAYLOAD_OFFSET),
        )
    counts = list(
        struct.unpack_from("<%dQ" % SPEC_TENSOR_COUNT, raw, SPEC_HEADER_LEN)
    )
    info["directory"] = dict(zip(TENSOR_NAMES, counts))
    info["element_total"] = sum(counts)

    # Spec lines 53-55: attn_down (8) and attn_up (9) have identical element
    # counts. That makes them indistinguishable to the directory, but it also
    # makes inequality between them a detectable defect.
    if counts[8] != counts[9]:
        raise CheckpointError(
            "ATTN_COUNT_MISMATCH",
            "attn_down=%d and attn_up=%d must have identical element counts"
            % (counts[8], counts[9]),
        )

    expected_len = PAYLOAD_OFFSET + ELEMENT_SIZE * info["element_total"]
    info["expected_bytes"] = expected_len
    if len(raw) < expected_len:
        raise CheckpointError(
            "TRUNCATED_PAYLOAD",
            "directory declares %d elements, so file_len must be %d; file is %d "
            "bytes (%d missing)"
            % (info["element_total"], expected_len, len(raw), expected_len - len(raw)),
        )
    if len(raw) > expected_len:
        raise CheckpointError(
            "TRAILING_GARBAGE",
            "directory declares %d elements, so file_len must be %d; file is %d "
            "bytes (%d extra)"
            % (info["element_total"], expected_len, len(raw), len(raw) - expected_len),
        )

    info["architecture"] = derive_architecture(raw, counts, info["num_attn_layers"])

    return info


def derive_architecture(raw, counts, declared_layers):
    """Measure the layer-2 attention block in the PAYLOAD BYTES.

    Returns a dict describing what the container actually carries for the second
    attention layer, so that the effective layer count is a reading rather than a
    quotation. Every number in it is computed here; none of it comes from a
    sidecar.

    WHAT THIS ESTABLISHES. `num_attn_layers` in both the header and the sidecar
    is a DECLARATION of how many layers were allocated. It is not a statement
    that each of them holds anything. If wq2, wk2, wv2 and wo2 are entirely zero
    then the second layer's contribution to every forward pass is zero, whatever
    the arithmetic around it, so the parameters outside that block are the only
    ones the artifact can be said to carry. That is a property of these bytes and
    it is checkable by anyone holding the .bin alone.

    WHAT IT DOES NOT ESTABLISH, and the distinction has to survive being quoted.
    An all-zero block says the SERIALIZED model has an inert layer. It says
    nothing about intent: it cannot distinguish a design that deliberately
    allocates a second layer and freezes it from a training loop that meant to
    train one and failed to. It is not evidence of a defect and must never be
    presented as one - the trainer's own documentation states the layer is
    allocated, carried through every forward pass and provably inert, and this
    check simply moves that sentence from prose into a measurement.

    NOT A FLOAT COMPARISON. The census is over 4-byte groups, so a value is
    "exactly zero" only if its bit pattern is 00 00 00 00. Negative zero is
    counted apart: it is numerically inert in the same way and a different
    encoding, and folding the two together would hide which one the file holds.
    """
    offsets = {}
    cursor = PAYLOAD_OFFSET
    for name, count in zip(TENSOR_NAMES, counts):
        offsets[name] = (cursor, count)
        cursor += ELEMENT_SIZE * count

    census = []
    block_elements = 0
    positive_zero = 0
    negative_zero = 0
    for name in LAYER2_TENSORS:
        start, count = offsets[name]
        block = raw[start:start + ELEMENT_SIZE * count]
        zeros = 0
        negatives = 0
        for index in range(count):
            word = block[ELEMENT_SIZE * index:ELEMENT_SIZE * (index + 1)]
            if word == F32_POSITIVE_ZERO:
                zeros += 1
            elif word == F32_NEGATIVE_ZERO:
                negatives += 1
        census.append((name, zeros, negatives, count))
        block_elements += count
        positive_zero += zeros
        negative_zero += negatives

    element_total = sum(counts)
    inert = block_elements > 0 and positive_zero + negative_zero == block_elements
    # An unserialized block (every layer-2 count zero) is inert for the trivial
    # reason that it holds nothing, and is reported with its own wording rather
    # than folded into the all-zero case.
    unserialized = block_elements == 0

    if declared_layers >= 2 and (inert or unserialized):
        effective_layers = declared_layers - 1
    else:
        effective_layers = declared_layers

    return {
        "declared_layers": declared_layers,
        "effective_layers": effective_layers,
        "census": census,
        "block_elements": block_elements,
        "positive_zero": positive_zero,
        "negative_zero": negative_zero,
        "inert": inert,
        "unserialized": unserialized,
        "element_total": element_total,
        "outside_block": element_total - block_elements,
    }


def architecture_lines(arch):
    """The derived architecture report, as lines, worded as a measurement.

    Rendered under the VERIFIED heading because every number in it was computed
    from the container bytes. It is deliberately NOT added to the scope's
    `cross_checked` list: that list names sidecar KEYS which agree with the
    header, and this is a reading of the payload that no sidecar key states.
    """
    declared = arch["declared_layers"]
    effective = arch["effective_layers"]
    lines = [
        "num_attn_layers declared %d, effective %d (derived from the payload "
        "bytes, not read from the sidecar)" % (declared, effective),
    ]

    if declared < 2:
        lines.append(
            "the record declares fewer than 2 attention layers, so there is no "
            "layer-2 block to examine"
        )
        return lines

    if arch["unserialized"]:
        lines.append(
            "layer-2 projections %s carry 0 elements between them: the block is "
            "not serialized at all" % "/".join(LAYER2_TENSORS)
        )
    else:
        per_tensor = ", ".join(
            "%s %s/%s" % (name, format(zeros + negatives, ","), format(count, ","))
            for name, zeros, negatives, count in arch["census"]
        )
        lines.append(
            "layer-2 zero census %s/%s elements exactly zero (%s)"
            % (format(arch["positive_zero"] + arch["negative_zero"], ","),
               format(arch["block_elements"], ","), per_tensor)
        )
        if arch["negative_zero"]:
            lines.append(
                "of those, %s carry the NEGATIVE zero bit pattern (80 00 00 00 "
                "little-endian), which is numerically zero and a different "
                "encoding" % format(arch["negative_zero"], ",")
            )

    if arch["inert"] or arch["unserialized"]:
        lines.append(
            "serialized parameters %s total, %s in the layer-2 block, %s outside "
            "it; the block contributes nothing to a forward pass, so %s is the "
            "count the artifact can be said to carry"
            % (format(arch["element_total"], ","),
               format(arch["block_elements"], ","),
               format(arch["outside_block"], ","),
               format(arch["outside_block"], ","))
        )
        lines.append(
            "this is a statement about these bytes only: it shows the container "
            "carries an inert second layer, NOT that the training recipe "
            "intended one"
        )
    else:
        lines.append(
            "serialized parameters %s total, %s in the layer-2 block, of which "
            "%s are non-zero; the block is not inert in this artifact"
            % (format(arch["element_total"], ","),
               format(arch["block_elements"], ","),
               format(arch["block_elements"] - arch["positive_zero"]
                      - arch["negative_zero"], ","))
        )
    return lines


def read_checkpoint(path):
    """Read a file from disk, hash the bytes that were read back, and decode."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise CheckpointError("UNREADABLE_FILE", str(exc))
    info = parse_container(raw, path)
    info["sha256"] = hashlib.sha256(raw).hexdigest()
    return info


# Sidecar field name -> header field name. The specification snapshot describes
# the binary container only and says nothing about the sidecar, so this mapping
# was derived from the sidecar records themselves (see interop/README.md,
# ambiguity A6).
SIDECAR_CHECKS = [
    ("sha256", "sha256"),
    ("bytes", "bytes"),
    ("format_version", "format_version"),
    ("hidden", "hidden"),
    ("d_model", "d_model"),
    ("num_attn_layers", "num_attn_layers"),
    ("seed", "seed"),
    ("step", "step"),
    ("optimizer", "optimizer"),
    ("fake_quant_format", "fake_quant_format"),
    ("data_synthetic", "data_synthetic"),
]


# Fields introduced after schema 1, by the schema version that added them.
# Dispatch is by FIELD PRESENCE, never by matching the `schema` string: a tag
# is a claim about a record, the fields are the record. A tag this reader has
# never seen must still be read for everything it does carry, and a tag that
# over-promises must be caught rather than believed.
#
# A marker may name a nested field as `platform.<key>`: the schema 7 additions
# are all inside the `platform` object, so a table of top-level names alone
# cannot see them. Presence is tested by `record_has`.
#
# Schemas 6 and 7 were derived by diffing the records on disk, not from prose.
# Every name below is present in EVERY record on disk carrying that tag or a
# higher one; a field held by only some records of a version is not a marker,
# because inference must not depend on which run happened to write it.
SCHEMA_MARKERS = [
    (2, [
        "steps_total",
        "gf16_floor_every",
        "eval_every",
        "final_val_bpb",
        "git_provenance",
    ]),
    (3, [
        "lr",
        "attn_scale",
        "attn_seq",
        "platform",
        "source_sha256",
    ]),
    (4, [
        "trainer",
        "vocab",
    ]),
    (5, [
        "gf16_enabled",
    ]),
    (6, [
        "eval_chunks",
        "eval_tokens",
        "eval_seq",
        "val_bpb_stderr",
        "optimizer_params",
        "min_observed_val_bpb",
    ]),
    (7, [
        "platform.rustflags_sha256",
        "platform.rustflags_source",
        "platform.remap_applied",
        "platform.source_digest_scope",
    ]),
    (8, [
        "git_untracked",
        "platform.libc_provenance",
        "platform.features",
    ]),
    (9, [
        "format_faithful",
    ]),
]

# NOT a marker, and the omission is the point: `platform.libc_version` arrives
# with the three schema 8 names above and is present in every /8 and /9 record on
# disk, yet it is listed under PLATFORM_FIELDS as merely KNOWN. The writer emits
# it only when a version query for the target succeeded (see the SECOND SCOPE
# CAVEAT); on a target with no such query the field is absent from a perfectly
# well-formed /8 record. Making it a marker would infer schema 7 for that record
# and then FAIL it for over-promising a /8 tag - a false accusation produced
# entirely by this reader's bookkeeping, which is the exact defect the /4 round
# recorded in interop/README.md section 4. A marker must be a field the writer
# cannot omit, not a field this tree happens never to have omitted.
#
# The same reasoning applies to `platform.rustflags_sha256` and
# `platform.rustflags_source` at /7, which are marked and are also conditionally
# serialized (they are absent when no build-flags file was found). They are left
# as markers because every /7-and-higher record on disk carries them and changing
# an established version boundary is a separate decision from adding two new
# ones; the latent false-failure is recorded here rather than left to be
# rediscovered.

# Fields that a later schema STOPPED writing, by the version at which they
# disappear from every record carrying that tag. They stay known names - an
# older record carrying one is not an unrecognised field - but a record that
# reaches the retiring version and still carries the name gets a note, because
# either the record or this table is wrong about the version boundary.
#
# `bpb` -> `final_val_bpb` at /2 is documented in SPEC-SNAPSHOT.txt. The other
# two were read off the artifacts: no /6-or-higher record on disk carries
# `best_val_bpb` (it is superseded by `min_observed_val_bpb`), and no /7 record
# carries `run_id`.
#
# Nothing is retired at /8 or /9: both are additive over the version below them
# on every record on disk, and the /8 and /9 field sets are supersets of /7 key
# for key.
SCHEMA_RETIRED = [
    (2, ["bpb"]),
    (6, ["best_val_bpb"]),
    (7, ["run_id"]),
]

# Fields a later schema kept under the same NAME while changing what the value
# MEANS. Neither an addition nor a retirement, and the one kind of schema change
# a presence-based reader is structurally blind to: the key is there, the tag is
# consistent, and every check passes over a value that no longer says what the
# older records' value said.
#
# `path` at /9. Through /8 it was an absolute filesystem location, so every
# locally produced record published the writer's home directory. At /9 it is the
# artifact RELATIVE to `platform.source_digest_scope`, or the literal
# `outside-scope:<file name>` when it lies outside that scope. A consumer that
# joined `path` onto nothing and opened it worked for eight schema versions and
# stops working at the ninth, without a single check in this reader failing -
# `path` has no counterpart in the container and is ECHOED, as it always was.
# Saying so is the only thing this instrument can do about it.
SCHEMA_REDEFINED = [
    (9, ["path"]),
]

# The highest schema version whose field set this reader knows. A record whose
# tag is ABOVE this is a note, not a failure; see `compare_sidecar`.
MAX_KNOWN_SCHEMA = max(version for version, _ in SCHEMA_MARKERS)

# Post-schema-1 fields that the CONTAINER also carries, so they can be
# cross-checked against the decoded header rather than merely reported, listed
# under the schema version that introduced them:
#
#   schema 3  lr -> train_lr (offset 72), attn_scale (76), attn_seq (80)
#   schema 4  vocab -> the u32 at offset 16
#   schema 5  gf16_enabled -> the u8 at byte 124
#
# Absent from an older record, in which case the check is skipped, not failed:
# those records simply never made the claim.
SCHEMA_HEADER_CHECKS = [
    (3, [
        ("lr", "train_lr"),
        ("attn_scale", "attn_scale"),
        ("attn_seq", "attn_seq"),
    ]),
    (4, [
        ("vocab", "vocab"),
    ]),
    (5, [
        ("gf16_enabled", "gf16_enabled"),
    ]),
]

# The same pairs flattened in schema order. These are appended to the mandatory
# comparison whenever the record carries the field.
HEADER_CROSS_CHECKS = [pair for _, pairs in SCHEMA_HEADER_CHECKS for pair in pairs]

# ---------------------------------------------------------------------------
# The table of names this reader has a meaning for.
#
# Anything in a record that is NOT here is reported as present but not
# interpreted. That is the whole point of the table: a reader whose only
# outcomes are "agrees" and "disagrees" silently passes every field it cannot
# see, and a conformity instrument that cannot say "I do not know what this is"
# will report agreement over a shrinking fraction of the record as the record
# grows. Names were read off the artifacts under checkpoints/, never from the
# encoder.
# ---------------------------------------------------------------------------

SCHEMA1_FIELDS = [
    "schema",
    "canon_name", "seed", "step", "path",
    "sha256", "bytes", "format_version",
    "hidden", "d_model", "num_attn_layers",
    "optimizer", "fake_quant_format", "data_synthetic",
    "bpb", "ema_bpb",
    "git_sha", "git_dirty",
    "corpus", "run_id", "ledger", "ts",
]

# Known names that are NOT markers, by the version that introduced them. A
# non-marker is a field this reader understands but must not infer a version
# from, because some record carrying the version does not have it.
#
#   /2  best_val_bpb   documented as a /2 addition; retired at /6
#   /6  eval_seq etc.  are markers; nothing else was added at /6
SCHEMA_EXTRA_FIELDS = [
    (2, ["best_val_bpb"]),
]

# `platform` sub-keys, by the version that introduced them. NOTE, read off the
# artifacts and not to be smoothed over: `source_digest_scope` is listed at /7
# because every /7 record carries it, but three records on disk carry it under a
# /6 TAG (checkpoints/trios-train-rng47/2000.json and both of
# checkpoints/loc-v3/), written 2026-08-03 08:24-08:33 UTC, about an hour before
# the tag was cut to /7 at 09:25 UTC. The field shipped before its version did.
# That is why this reader reports later-schema fields found under an earlier tag
# instead of ignoring them.
PLATFORM_FIELDS = [
    (3, ["os", "arch", "pointer_width", "libc",
         "toolchain", "toolchain_provenance"]),
    (7, ["rustflags_sha256", "rustflags_source",
         "remap_applied", "source_digest_scope"]),
    (8, ["libc_version", "libc_provenance", "features"]),
]

# Nested objects this reader descends into when looking for unrecognised keys,
# with the sub-keys it knows. Every one of these sets was enumerated from the
# records on disk. `corpus` is two levels deep.
KNOWN_OBJECT_KEYS = {
    "platform": [key for _, keys in PLATFORM_FIELDS for key in keys],
    "trainer": ["path", "sha256", "provenance"],
    "optimizer_params": ["beta1", "beta2", "eps", "weight_decay", "source"],
    "corpus": ["train", "val"],
    "corpus.train": ["path", "bytes", "sha256"],
    "corpus.val": ["path", "bytes", "sha256"],
}

# Declaration order matters only for readable output; membership is what the
# unknown-key check uses.
KNOWN_TOP_LEVEL = list(SCHEMA1_FIELDS)
for _version, _keys in SCHEMA_MARKERS:
    for _key in _keys:
        if "." not in _key and _key not in KNOWN_TOP_LEVEL:
            KNOWN_TOP_LEVEL.append(_key)
for _version, _keys in SCHEMA_EXTRA_FIELDS:
    for _key in _keys:
        if _key not in KNOWN_TOP_LEVEL:
            KNOWN_TOP_LEVEL.append(_key)
KNOWN_TOP_LEVEL_SET = set(KNOWN_TOP_LEVEL)

# Every post-schema-2 addition, by version, in declaration order. These are the
# lists whose absence is reported for each record read.
PROVENANCE_FIELDS = [
    (version, markers) for version, markers in SCHEMA_MARKERS if version >= 3
]

SCHEMA_PREFIX = "trios-checkpoint-record/"

# Names under which a record might state a parameter count, so that the derived
# serialized total has something to be compared against when one appears. None of
# them occurs in any record on disk: the sidecar records geometry (`hidden`,
# `d_model`, `num_attn_layers`, `vocab`) and never a total, so the trainer's
# printed `params=196608` lives only in stdout and in prose. The derived total is
# therefore reported WITHOUT a counterpart, which is stated rather than left to be
# read as agreement.
SIDECAR_PARAM_COUNT_KEYS = [
    "params", "param_count", "parameters", "n_params", "num_params",
]

# ---------------------------------------------------------------------------
# The `ledger` vocabulary, and why an unknown value here is a FAILURE.
#
# `ledger` is the record's own statement about whether anything outside this
# checkout was told what was measured. It is the one field a conformity reader
# is asked to quote as evidence ("nothing external supplied a number"), and it
# is a closed vocabulary, not free text. A value this reader has no meaning for
# is therefore not the harmless "present but NOT INTERPRETED" case that applies
# to an unknown KEY: an unknown key adds a fact the instrument cannot see, an
# unknown `ledger` REPLACES a fact the instrument is about to report. Passing
# it through would let a reader downstream infer "not written" from a string
# that meant something else. So it fails, by name, and this table is the thing
# that has to be updated.
#
# PROVENANCE OF THIS TABLE, so the independence claim in the module docstring
# is not overstated. The container decoding was written from the specification
# alone and this file still does not read `save`/`load` or
# `to_checkpoint_bytes`/`from_checkpoint_bytes`. But `ledger` is a sidecar-only
# field (SCOPE CAVEAT above), and the values below were taken from the ledger
# writer's own outcome vocabulary in src/neon_writer.rs plus the records on
# disk. This check is therefore a VOCABULARY check, not interlaboratory
# evidence, and no agreement it reports may be cited as the latter.
#
# 2026-08-03: `skipped-not-opted-in` added. Before it existed, a run with a DSN
# in the environment and no TRIOS_LEDGER_WRITE=1 recorded `skipped-no-dsn`,
# which says nothing was configured. Something was; the run declined it. Every
# record written before that date carries the older, coarser value, and this
# reader cannot tell which of the two situations produced it - the artifacts
# under evidence/ are ambiguous on exactly this point and nothing here can
# disambiguate them after the fact.
LEDGER_VALUES = {
    "pending": "first of the two sidecar writes; no outcome recorded yet",
    "written": "a row was accepted by the ledger database",
    "skipped-no-dsn": "no DSN in the environment at all "
                      "(before 2026-08-03 this also covered the case below)",
    "skipped-not-opted-in": "a DSN WAS configured; the run did not set "
                            "TRIOS_LEDGER_WRITE=1 and wrote nothing",
    "failed": "a write was attempted and no row landed",
    "rejected": "the writer refused the value as unpublishable",
}


def record_has(record, name):
    """Presence test for a top-level or dotted (`platform.<key>`) field name.

    Only presence, never truthiness: `remap_applied: false` and
    `git_dirty: null` are fields the record carries and states a value for, and
    treating either as absent would misreport the record's schema.
    """
    if "." not in name:
        return name in record
    head, rest = name.split(".", 1)
    nested = record.get(head)
    if not isinstance(nested, dict):
        return False
    return record_has(nested, rest)


def infer_schema(record):
    """Highest schema version whose markers, and every earlier version's, are present.

    The schemas are purely additive, so a gap is not a version: a record
    carrying the schema 5 marker but missing a schema 4 one has not "reached"
    schema 5, it is a schema 3 record with one extra key. Scanning in ascending
    order and stopping at the first gap is what makes the over-promise check
    below meaningful.
    """
    inferred = 1
    for version, markers in SCHEMA_MARKERS:
        if not all(record_has(record, key) for key in markers):
            break
        inferred = version
    return inferred


def unrecognised_keys(record):
    """Every key in the record that this reader has no meaning for.

    Top level plus the nested objects in KNOWN_OBJECT_KEYS. Returned sorted, so
    the report is stable across JSON key orderings.
    """
    unknown = [k for k in record if k not in KNOWN_TOP_LEVEL_SET]
    for prefix, known in KNOWN_OBJECT_KEYS.items():
        nested = record
        for part in prefix.split("."):
            nested = nested.get(part) if isinstance(nested, dict) else None
        if not isinstance(nested, dict):
            continue
        known_set = set(known)
        unknown.extend(
            "%s.%s" % (prefix, k) for k in nested if k not in known_set
        )
    return sorted(unknown)


def echoed_keys(record, cross_checked):
    """Known keys the record carries that no container byte can confirm.

    Everything this reader recognises, minus what was actually compared against
    the decoded header. `platform` is enumerated key by key because the schema 7
    additions live inside it and naming the object alone would hide them; the
    other nested objects are named whole. `schema` is excluded because it is not
    echoed - the tag IS checked, against the record's own field presence.
    """
    echoed = [
        key for key in KNOWN_TOP_LEVEL
        if key in record and key not in cross_checked
        and key not in ("platform", "schema")
    ]
    platform = record.get("platform")
    if isinstance(platform, dict):
        for _version, keys in PLATFORM_FIELDS:
            echoed.extend("platform.%s" % k for k in keys if k in platform)
    elif "platform" in record:
        echoed.append("platform")
    return echoed


# ---------------------------------------------------------------------------
# The provenance seal: a digest over the ECHOED half of the record.
#
# Everything above tells you honestly that the provenance block is echoed and
# not verified. That honesty is worth nothing to an auditor holding one record:
# a sidecar rewritten to claim the wrong architecture, a rustc that never
# existed and a retracted BPB still prints RESULT PASS here, because the
# container it is checked against is untouched and the declaration is checked
# against nothing at all.
#
# The seal does not fix that by itself and is not a signature. It gives the
# declaration a digest that CAN be published out of band, so a forgery becomes
# detectable by anyone who read the published seal instead of undetectable by
# everyone. The normative definition lives in src/provenance_seal.rs; this is
# the second implementation of it, and the two agreeing on every record in
# evidence/ is the interop check. If they ever disagree at HEAD, the
# specification is wrong - not the record.
#
# Deliberately NOT derived from `echoed_keys` above: that set depends on which
# fields this reader happened to cross-check, so it would drift with the
# reader. A seal is a fixed field list or it is not a seal.
# ---------------------------------------------------------------------------

SEAL_PREFIX = "sha256:"
SEAL_ABSENT = "<absent>"

# `platform` sub-keys the seal always states, present or not. Every OTHER key
# found under `platform` is sealed too; these four are listed because their
# ABSENCE must also change the digest.
SEALED_PLATFORM_KEYS = ["arch", "libc", "os", "toolchain"]

# Sorted, and identical name-for-name to SEALED_TOP_LEVEL in
# src/provenance_seal.rs. `sha256` and `bytes` are in the set: the earlier
# defence that they were "sealed by the container digest already" was circular,
# because `sha256` IS the container digest and cannot seal itself. A seal that
# named no artifact bound its declaration to no particular .bin, so one
# authenticated declaration fitted every container of the same length - the
# x86_64 Linux record could be handed the aarch64 macOS weights (both 852272
# bytes) and still match the seal published for the Linux run. `seed` and `step`
# name which artifact the recipe was meant to produce; `gf16_floor_every` is a
# recipe parameter that mutates the weights and lives in no container header.
SEALED_TOP_LEVEL = [
    "bytes",
    "corpus",
    "eval_every",
    "final_val_bpb",
    "gf16_floor_every",
    "git_dirty",
    "git_sha",
    "schema",
    "seed",
    "sha256",
    "source_sha256",
    "step",
    "steps_total",
    "trainer",
]


def rust_f64_display(value):
    """Render a float exactly as Rust's `{}` on `f64` does.

    Rust is the normative side, so this reproduces IT rather than the other way
    round. `repr()` already chooses the shortest round-trip digits, and differs
    from Rust in two places only: it writes an integral value as `2.0` where
    Rust writes `2`, and it uses exponent notation where Rust never does.
    """
    if value != value:
        return "NaN"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    text = repr(float(value))
    if "e" in text or "E" in text:
        import decimal
        text = format(decimal.Decimal(text), "f")
    elif text.endswith(".0"):
        text = text[:-2]
    return text


def seal_escape(text):
    """Escape a string value so it cannot forge a line of the listing.

    Without this a toolchain string carrying a newline could contain
    `\\nplatform.arch=x86_64` and add a line the record never declared - a
    collision an attacker picks rather than one they have to find.
    """
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


def seal_render_scalar(value):
    """One scalar rendered per the seal specification. Containers never reach here."""
    if value is None:
        return "<null>"
    if isinstance(value, bool):
        # Checked before int: in Python `bool` IS an `int`, and `True` must
        # render as `true` and never as `1`.
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return rust_f64_display(value)
    if isinstance(value, str):
        return seal_escape(value)
    return "<unrenderable:%s>" % type(value).__name__


def seal_flatten(prefix, value, out):
    """Append `(key, value)` pairs for `value` under `prefix`, flattening containers."""
    if isinstance(value, dict):
        if not value:
            out.append((prefix, "<empty-object>"))
            return
        for key in sorted(value):
            seal_flatten("%s.%s" % (prefix, key), value[key], out)
        return
    if isinstance(value, list):
        if not value:
            out.append((prefix, "<empty-array>"))
            return
        for index, item in enumerate(value):
            seal_flatten("%s.%d" % (prefix, index), item, out)
        return
    out.append((prefix, seal_render_scalar(value)))


def provenance_listing(record):
    """The exact bytes the seal is taken over, as text.

    A field the record does NOT carry is rendered `key=<absent>` rather than
    skipped: deleting a field has to change the digest, or deletion is a free
    edit for a forger.
    """
    if not isinstance(record, dict):
        raise ValueError("a checkpoint record must be a JSON object, not %s"
                         % type(record).__name__)

    fields = []
    for key in SEALED_TOP_LEVEL:
        if key in record:
            seal_flatten(key, record[key], fields)
        else:
            fields.append((key, SEAL_ABSENT))

    platform = record.get("platform")
    if isinstance(platform, dict):
        for key in SEALED_PLATFORM_KEYS:
            if key in platform:
                seal_flatten("platform.%s" % key, platform[key], fields)
            else:
                fields.append(("platform.%s" % key, SEAL_ABSENT))
        for key in sorted(k for k in platform if k not in SEALED_PLATFORM_KEYS):
            seal_flatten("platform.%s" % key, platform[key], fields)
    elif "platform" in record:
        # Present but not an object. The four required keys are absent from it
        # whatever it is, and the thing itself is sealed under its own name so
        # the anomaly cannot be edited away.
        for key in SEALED_PLATFORM_KEYS:
            fields.append(("platform.%s" % key, SEAL_ABSENT))
        seal_flatten("platform", platform, fields)
    else:
        for key in SEALED_PLATFORM_KEYS:
            fields.append(("platform.%s" % key, SEAL_ABSENT))

    fields.sort(key=lambda pair: pair[0])
    deduped = []
    for key, value in fields:
        if deduped and deduped[-1][0] == key:
            continue
        deduped.append((key, value))
    return "".join("%s=%s\n" % (key, value) for key, value in deduped)


def provenance_seal(record):
    """`sha256:` plus the lowercase hex SHA-256 of the listing bytes."""
    listing = provenance_listing(record).encode("utf-8")
    return SEAL_PREFIX + hashlib.sha256(listing).hexdigest()


def seal_report(sidecar_path):
    """The one line every run prints about the record's declaration half.

    Printed whether the record passed, failed or was never opened: a reader
    that announced the seal only on success would be silent in exactly the case
    an auditor is looking at it.
    """
    if sidecar_path is None:
        return ("provenance-seal none - no sidecar was read, so this run says "
                "nothing about any declaration")
    try:
        with open(sidecar_path, "r") as handle:
            record = json.load(handle)
    except (OSError, ValueError) as exc:
        return "provenance-seal UNSEALABLE - %s: %s" % (sidecar_path, exc)
    try:
        seal = provenance_seal(record)
    except ValueError as exc:
        return "provenance-seal UNSEALABLE - %s" % exc
    return ("provenance-seal %s over %d declared field(s) [%s] - ECHOED, never "
            "verified here; compare it against the digest published for this "
            "record out of band (evidence/SEALS.txt, explained in "
            "docs/PROVENANCE-BINDING.md), or it authenticates "
            "nothing" % (
                seal,
                len(provenance_listing(record).splitlines()),
                ", ".join(line.split("=", 1)[0]
                          for line in provenance_listing(record).splitlines()),
            ))


def declared_schema(record):
    """The version the record's own `schema` tag claims, or None."""
    tag = record.get("schema")
    if not isinstance(tag, str) or not tag.startswith(SCHEMA_PREFIX):
        return None
    suffix = tag[len(SCHEMA_PREFIX):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def normalize_for_compare(key, sidecar_value, header_value):
    """Return (sidecar, header) coerced to comparable Python values."""
    if key in ("data_synthetic", "gf16_enabled"):
        # The header stores a u8; the sidecar stores a JSON boolean. Anything
        # other than 0 or 1 in the header is a value the sidecar cannot encode,
        # and comparing the raw byte is what surfaces that (ambiguity A4).
        if isinstance(sidecar_value, bool):
            sidecar_value = 1 if sidecar_value else 0
    if key == "sha256" and isinstance(sidecar_value, str):
        sidecar_value = sidecar_value.lower()
        header_value = header_value.lower()
    if key in ("lr", "attn_scale"):
        # The header stores f32 bit patterns; struct's '<f' widens them to the
        # Python float that is their exact f64 value, which is also exactly
        # what the record serializes. Equality is therefore the right test and
        # a tolerance would only hide a real disagreement.
        if isinstance(sidecar_value, int):
            sidecar_value = float(sidecar_value)
    return sidecar_value, header_value


def compare_sidecar(info, sidecar_path):
    """Cross-check a sidecar against a decoded container.

    Returns `(problems, notes, scope)`: mismatch strings (empty means
    agreement), informational lines about what the record does and does not
    carry, and a scope dict with three disjoint key lists - `cross_checked`,
    `echoed` and `uninterpreted` - so that no caller can report agreement
    without also reporting what the agreement did not cover.
    """
    empty_scope = {"cross_checked": [], "echoed": [], "uninterpreted": []}
    try:
        with open(sidecar_path, "r") as handle:
            record = json.load(handle)
    except OSError as exc:
        return ["sidecar unreadable: %s" % exc], [], dict(empty_scope)
    except ValueError as exc:
        return ["sidecar is not valid JSON: %s" % exc], [], dict(empty_scope)
    if not isinstance(record, dict):
        return (["sidecar is not a JSON object: %s" % type(record).__name__],
                [], dict(empty_scope))

    problems = []
    notes = []
    cross_checked = []

    declared = declared_schema(record)
    inferred = infer_schema(record)
    notes.append("schema tag %r, fields present up to schema %d"
                 % (record.get("schema"), inferred))
    if declared is not None and declared > inferred:
        if declared > MAX_KNOWN_SCHEMA and inferred == MAX_KNOWN_SCHEMA:
            # A version from the future. Every field this reader knows about is
            # present and has been checked; the tag claims fields whose names
            # this reader does not yet have, which it cannot verify and must not
            # pretend to. Reporting that is the whole of the honest answer -
            # failing the record would only mean "this instrument is older than
            # that one", which is not a defect in the artifact.
            notes.append(
                "NOTE: record declares schema/%d, this reader knows up to "
                "schema/%d; all schema/%d fields present and checked"
                % (declared, MAX_KNOWN_SCHEMA, MAX_KNOWN_SCHEMA)
            )
        else:
            # The record names a version whose fields it does not carry. That is
            # a defect in the record, not a version this reader cannot handle.
            missing = [
                key
                for version, markers in SCHEMA_MARKERS
                if version <= min(declared, MAX_KNOWN_SCHEMA)
                for key in markers
                if not record_has(record, key)
            ]
            problems.append(
                "schema tag claims version %d but the record carries only the "
                "schema %d field set (missing: %s)"
                % (declared, inferred, ", ".join(missing) or "none")
            )

    checks = list(SIDECAR_CHECKS)
    for sidecar_key, header_key in HEADER_CROSS_CHECKS:
        if sidecar_key in record and record[sidecar_key] is not None:
            checks.append((sidecar_key, header_key))

    for sidecar_key, header_key in checks:
        if sidecar_key not in record:
            problems.append("sidecar is missing field '%s'" % sidecar_key)
            continue
        left, right = normalize_for_compare(
            sidecar_key, record[sidecar_key], info[header_key]
        )
        cross_checked.append(sidecar_key)
        if left != right:
            problems.append(
                "%s: sidecar=%r decoded=%r" % (sidecar_key, left, right)
            )

    # The `ledger` vocabulary. Surfaced always, so a reader of the output never
    # has to open the record to learn what it claims about external recording,
    # and refused when unknown - see LEDGER_VALUES for why this one field is a
    # failure rather than a note.
    if "ledger" in record:
        value = record["ledger"]
        if not isinstance(value, str):
            problems.append(
                "ledger is %s, not a string: %r"
                % (type(value).__name__, value)
            )
        elif value in LEDGER_VALUES:
            notes.append("ledger %r: %s" % (value, LEDGER_VALUES[value]))
        else:
            problems.append(
                "ledger %r is not a value this reader knows (known: %s); "
                "refusing to echo it, because a reader downstream would read "
                "it as a statement about external recording"
                % (value, ", ".join(sorted(LEDGER_VALUES)))
            )
    else:
        notes.append("ledger: field ABSENT; this record makes no claim about "
                     "whether anything external was told what was measured")

    complete = True
    for version, fields in PROVENANCE_FIELDS:
        absent = [k for k in fields if not record_has(record, k)]
        if absent:
            complete = False
            notes.append("schema %d provenance fields ABSENT: %s"
                         % (version, ", ".join(absent)))
    if complete:
        notes.append("schema %d-%d provenance fields: all present"
                     % (PROVENANCE_FIELDS[0][0], MAX_KNOWN_SCHEMA))

    # A field of a LATER schema carried by a record that does not reach it. Not
    # a defect - it is how an additive schema actually rolls out, one field at a
    # time - but it is the difference between what the tag says and what the
    # record holds, and only a reader that names it can be trusted to have
    # looked. Observed on disk: platform.source_digest_scope under a /6 tag.
    for version, markers in SCHEMA_MARKERS:
        if version <= inferred:
            continue
        early = [k for k in markers if record_has(record, k)]
        if early:
            notes.append(
                "schema %d field(s) present under a record that reaches only "
                "schema %d: %s" % (version, inferred, ", ".join(early))
            )

    # A key whose MEANING a later schema changed while keeping the name. Nothing
    # here can fail: the value is echoed either way, and the whole hazard is that
    # every check passes. Naming the key and the version is the entire remedy
    # available to a reader that has no container counterpart for it.
    for version, fields in SCHEMA_REDEFINED:
        if inferred < version:
            continue
        changed = [k for k in fields if record_has(record, k)]
        if changed:
            notes.append(
                "field(s) REDEFINED at schema %d and present here, same name and "
                "a different meaning than in an earlier record: %s"
                % (version, ", ".join(changed))
            )

    # The derived parameter total, and whether anything in the record claims one.
    arch = info.get("architecture")
    if arch is not None:
        advertised = [k for k in SIDECAR_PARAM_COUNT_KEYS if k in record]
        if advertised:
            notes.append(
                "record states a parameter count (%s); the container serializes "
                "%s elements, %s of them outside the layer-2 block. This reader "
                "has no rule saying which of the two a %r field names, so the "
                "numbers are reported side by side and NOT compared"
                % (", ".join("%s=%r" % (k, record[k]) for k in advertised),
                   format(arch["element_total"], ","),
                   format(arch["outside_block"], ","),
                   advertised[0])
            )
        else:
            notes.append(
                "no field in this record states a parameter count, so the "
                "derived serialized total (%s elements) has no counterpart to "
                "agree or disagree with"
                % format(arch["element_total"], ",")
            )

    # A field a later schema stopped writing, still present in a record that
    # reaches that schema. Either the record or this reader's table is wrong
    # about where the boundary is; saying which fields are involved is what
    # makes the disagreement resolvable.
    for version, fields in SCHEMA_RETIRED:
        if inferred < version:
            continue
        lingering = [k for k in fields if record_has(record, k)]
        if lingering:
            notes.append(
                "field(s) retired at schema %d but still present: %s"
                % (version, ", ".join(lingering))
            )

    all_provenance = [k for _, fields in PROVENANCE_FIELDS for k in fields]
    null_valued = [k for k in all_provenance
                   if "." not in k and k in record and record[k] is None]
    if null_valued:
        notes.append("provenance fields present but null: %s"
                     % ", ".join(null_valued))

    # The two sets that make a PASS honest. `echoed` is read from the sidecar
    # and believed: no container byte can confirm or refute any of it, so the
    # provenance block and the eval grid are quoted, never verified.
    # `uninterpreted` is the set this reader has no meaning for at all - it must
    # be a NOTE and never a silent pass, because a field the instrument cannot
    # see is a field it cannot agree about.
    # Both sets are returned rather than folded into `notes`, so that every
    # caller has to render them: the scope is not an optional annotation on the
    # verdict, it is half of the verdict.
    echoed = echoed_keys(record, set(cross_checked))
    uninterpreted = unrecognised_keys(record)

    scope = {
        "cross_checked": cross_checked,
        "echoed": echoed,
        "uninterpreted": uninterpreted,
    }
    return problems, notes, scope


def print_report(info):
    print("file            %s" % info["path"])
    print("magic           TRIOSCKP")
    print("format_version  %d" % info["format_version"])
    print("header_len      %d" % info["header_len"])
    print("bytes           %d" % info["bytes"])
    print("sha256          %s" % info["sha256"])
    print("vocab           %d" % info["vocab"])
    print("dim             %d" % info["dim"])
    print("num_ctx         %d" % info["num_ctx"])
    print("hidden          %d" % info["hidden"])
    print("d_model         %d" % info["d_model"])
    print("num_heads       %d" % info["num_heads"])
    print("attn_cfg_seq_len %d" % info["attn_cfg_seq_len"])
    print("num_attn_layers %d" % info["num_attn_layers"])
    print("ngram           %d" % info["ngram"])
    print("tensor_count    %d" % info["tensor_count"])
    print("qk_gain         %r" % info["qk_gain"])
    print("attn_cfg_lr     %r" % info["attn_cfg_lr"])
    print("train_lr        %r" % info["train_lr"])
    print("attn_scale      %r" % info["attn_scale"])
    print("attn_seq        %d" % info["attn_seq"])
    print("ctx_weights     %s" % ", ".join("%r" % w for w in info["ctx_weights"]))
    print("seed            %d" % info["seed"])
    print("step            %d" % info["step"])
    print("gf16_enabled    %d" % info["gf16_enabled"])
    print("data_synthetic  %d" % info["data_synthetic"])
    print("optimizer       %s" % info["optimizer"])
    print("fake_quant_format %s" % info["fake_quant_format"])
    print("elements        %d (payload %d bytes)"
          % (info["element_total"], info["bytes"] - PAYLOAD_OFFSET))
    print("directory")
    for name in TENSOR_NAMES:
        print("  %-10s %d" % (name, info["directory"][name]))
    for line in architecture_lines(info["architecture"]):
        print("derived         %s" % line)


def architecture_summary(arch):
    """One line of the derived architecture reading, for a quoted verdict.

    The long form is `architecture_lines`. This is what goes on a RESULT line, so
    it carries the three numbers that cannot be dropped without changing the
    claim: declared layers, effective layers, and the zero census they rest on.
    """
    if arch["declared_layers"] < 2:
        return "%d attention layer(s) declared, no layer-2 block to examine" % (
            arch["declared_layers"],
        )
    if arch["unserialized"]:
        census = "layer-2 block not serialized (0 elements)"
    else:
        census = "layer-2 zeros %s/%s" % (
            format(arch["positive_zero"] + arch["negative_zero"], ","),
            format(arch["block_elements"], ","),
        )
    return "attention layers declared %d / effective %d, %s, %s parameters " \
           "serialized (%s outside the layer-2 block)" % (
               arch["declared_layers"], arch["effective_layers"], census,
               format(arch["element_total"], ","),
               format(arch["outside_block"], ","),
           )


def format_scope(scope, full):
    """Render the three-way scope of a sidecar comparison.

    `full` names every field in every set; that is the form a citation needs,
    because "agrees on 16 fields" without the other two lists is a claim about
    an unstated denominator. The compact form keeps the counts and still names
    the uninterpreted keys in full - those are the ones nobody can look up.
    """
    cross, echoed, unknown = (
        scope["cross_checked"], scope["echoed"], scope["uninterpreted"],
    )
    if full:
        return (
            "VERIFIED against the container (%d): %s | "
            "ECHOED from the sidecar, unverifiable here (%d): %s | "
            "NOT INTERPRETED (%d): %s"
            % (len(cross), ", ".join(cross) or "none",
               len(echoed), ", ".join(echoed) or "none",
               len(unknown), ", ".join(unknown) or "none")
        )
    return (
        "scope %d verified / %d echoed unverifiable / %d not interpreted%s"
        % (len(cross), len(echoed), len(unknown),
           (" [" + ", ".join(unknown) + "]") if unknown else "")
    )


def verify_one(bin_path, sidecar_path, verbose):
    """Verify a single checkpoint. Returns (ok, message, scope)."""
    try:
        info = read_checkpoint(bin_path)
    except CheckpointError as exc:
        return False, "%s: %s" % (exc.kind, exc.detail), None
    if verbose:
        print_report(info)
    if sidecar_path is None:
        return True, "container ok, sha256 %s, derived %s (NO sidecar read: " \
                     "nothing outside the container was checked)" % (
                         info["sha256"],
                         architecture_summary(info["architecture"]),
                     ), None
    problems, notes, scope = compare_sidecar(info, sidecar_path)
    if verbose:
        for note in notes:
            print("sidecar         %s" % note)
        print("scope           VERIFIED against the container (%d): %s"
              % (len(scope["cross_checked"]),
                 ", ".join(scope["cross_checked"]) or "none"))
        # Belongs on the VERIFIED side of the line and nowhere else: it is
        # measured in the payload, so a sidecar cannot make it agree. It is kept
        # out of the count above because that count is of sidecar KEYS.
        print("scope           VERIFIED, derived from the payload rather than "
              "compared to a key: %s" % architecture_summary(info["architecture"]))
        print("scope           ECHOED from the sidecar, NOT verified against "
              "the container (%d): %s"
              % (len(scope["echoed"]), ", ".join(scope["echoed"]) or "none"))
        print("scope           present but NOT INTERPRETED by this reader "
              "(%d): %s"
              % (len(scope["uninterpreted"]),
                 ", ".join(scope["uninterpreted"]) or "none"))
    if problems:
        return False, "SIDECAR_MISMATCH: " + "; ".join(problems), scope
    derived = architecture_summary(info["architecture"])
    if verbose:
        return True, "container ok, sha256 %s; derived %s; sidecar scope: %s" % (
            info["sha256"], derived, format_scope(scope, full=True),
        ), scope
    return True, "container ok, sha256 %s; derived %s; %s [%s]" % (
        info["sha256"], derived, format_scope(scope, full=False),
        "; ".join(notes),
    ), scope


def verify_all(root):
    """Pair every <step>.bin with <step>.json under root and verify each pair."""
    pairs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if not name.endswith(".bin"):
                continue
            bin_path = os.path.join(dirpath, name)
            sidecar_path = bin_path[: -len(".bin")] + ".json"
            pairs.append((bin_path, sidecar_path))

    if not pairs:
        print("no .bin files found under %s" % root)
        return 1

    failures = 0
    uninterpreted_total = {}
    for bin_path, sidecar_path in pairs:
        if not os.path.exists(sidecar_path):
            failures += 1
            print("FAIL %s  MISSING_SIDECAR: %s does not exist"
                  % (bin_path, sidecar_path))
            continue
        ok, message, scope = verify_one(bin_path, sidecar_path, verbose=False)
        print("SEAL %s  %s" % (bin_path, seal_report(sidecar_path)))
        if scope:
            for key in scope["uninterpreted"]:
                uninterpreted_total[key] = uninterpreted_total.get(key, 0) + 1
        if ok:
            print("PASS %s  %s" % (bin_path, message))
        else:
            failures += 1
            print("FAIL %s  %s" % (bin_path, message))

    print("summary: %d pair(s), %d passed, %d failed"
          % (len(pairs), len(pairs) - failures, failures))
    # The sweep aggregates what the per-pair lines only count. A key nobody in
    # this reader has a meaning for is the one thing an operator running the
    # whole tree needs pushed at them, because it is the field that will be
    # silently absent from every agreement claim made about these artifacts.
    if uninterpreted_total:
        print("keys present in the records but NOT INTERPRETED by this reader:")
        for key in sorted(uninterpreted_total):
            print("  %-40s in %d record(s)" % (key, uninterpreted_total[key]))
    else:
        print("no key in any record was left uninterpreted by this reader")
    print("NOTE: a PASS above covers the container bytes and the sidecar "
          "fields with a counterpart in them. Provenance (platform, "
          "source_sha256, trainer) and the eval grid (eval_chunks, "
          "eval_tokens, eval_seq, val_bpb_stderr) are ECHOED, never verified; "
          "run a single record without --verify-all for the field-by-field "
          "scope.")
    return 1 if failures else 0


def main(argv):
    parser = argparse.ArgumentParser(
        description="Independent reader for the TRIOSCKP checkpoint format "
                    "(version 1), implemented from the format specification.",
    )
    parser.add_argument(
        "checkpoint", nargs="?",
        help="path to a <step>.bin checkpoint file",
    )
    parser.add_argument(
        "--sidecar", metavar="PATH",
        help="cross-check the decoded header against this sidecar JSON record; "
             "defaults to the <step>.json beside the given <step>.bin",
    )
    parser.add_argument(
        "--no-sidecar", action="store_true",
        help="decode the container only, even if a sibling <step>.json exists",
    )
    parser.add_argument(
        "--verify-all", metavar="DIR", dest="verify_all_dir",
        help="walk DIR, pair every <step>.bin with <step>.json, verify each",
    )
    args = parser.parse_args(argv)

    if args.verify_all_dir is not None:
        if args.checkpoint is not None or args.sidecar is not None:
            parser.error("--verify-all cannot be combined with a checkpoint path")
        if not os.path.isdir(args.verify_all_dir):
            print("not a directory: %s" % args.verify_all_dir, file=sys.stderr)
            return 2
        return verify_all(args.verify_all_dir)

    if args.checkpoint is None:
        parser.error("a checkpoint path or --verify-all DIR is required")

    if args.sidecar is not None and args.no_sidecar:
        parser.error("--sidecar and --no-sidecar are contradictory")

    # A <step>.json handed in as the positional argument. It is not a container
    # and decoding it produced MAGIC_MISMATCH, which is a true statement about
    # the bytes and a useless one about the operator's intent: the naming
    # convention that pairs a sidecar to a container already runs in the other
    # direction three lines below, and running it both ways costs nothing. Says
    # out loud which file it swapped to, so the report still names what it read.
    if (args.checkpoint is not None
            and args.checkpoint.endswith(".json")
            and not args.no_sidecar
            and args.sidecar is None):
        sibling_bin = args.checkpoint[: -len(".json")] + ".bin"
        if os.path.exists(sibling_bin):
            args.sidecar = args.checkpoint
            args.checkpoint = sibling_bin
            print("container       %s (paired from the sidecar by the "
                  "<step>.bin convention)" % sibling_bin)

    if args.sidecar is not None and not os.path.exists(args.sidecar):
        print("sidecar not found: %s" % args.sidecar, file=sys.stderr)
        return 2

    # A bare <step>.bin used to report only the container and say nothing about
    # the record beside it, which reads as a clean pass over an artifact whose
    # provenance half was never opened. Pair it by the naming convention and say
    # out loud which file was paired; --no-sidecar keeps the old behaviour.
    if args.sidecar is None and not args.no_sidecar:
        sibling = args.checkpoint[:-len(".bin")] + ".json" \
            if args.checkpoint.endswith(".bin") else None
        if sibling and os.path.exists(sibling):
            args.sidecar = sibling
            print("sidecar         %s (paired by the <step>.json convention)"
                  % sibling)
        else:
            print("sidecar         none found beside %s; the container alone "
                  "carries no provenance" % args.checkpoint)

    ok, message, _scope = verify_one(args.checkpoint, args.sidecar, verbose=True)
    # Printed on EVERY run, pass or fail, and before the verdict: the verdict
    # is about bytes, and this is the digest of everything the verdict does not
    # cover. The Rust side prints the same string
    # (`ckpt_replay --record <json> --provenance-seal`); the two agreeing is
    # what makes the seal specification a specification and not a habit.
    print(seal_report(args.sidecar))
    if ok:
        print("RESULT PASS  %s" % message)
        return 0
    print("RESULT FAIL  %s" % message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
