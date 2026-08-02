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

Sidecar schema versions. The reader accepts trios-checkpoint-record/1, /2 and
/3 and dispatches on FIELD PRESENCE, not on matching the `schema` string, so an
unseen tag is still read for everything it carries and a tag that claims fields
the record lacks is reported. For every record it names which of the schema 3
provenance fields (lr, attn_scale, attn_seq, platform, source_sha256) are
absent - all five are absent from every archived checkpoint here, which are
schema 1.

SCOPE CAVEAT, so the independence claim above is not overstated: the container
decoding was written from the specification alone, but the schema 3 sidecar
field NAMES were taken from the record definition. Agreement on `lr`,
`attn_scale` and `attn_seq` between sidecar and header is still a genuine
cross-check of two encodings of the same quantity; it is not evidence about the
specification, because the sidecar was never in the specification (see
interop/README.md, ambiguity A6).

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

    return info


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
]

# Schema 3 fields that the CONTAINER also carries, so they can be cross-checked
# rather than merely reported. Header offsets 72, 76 and 80. Absent from a
# schema 1 or 2 record, in which case the check is skipped, not failed: those
# records simply never made the claim.
SCHEMA3_HEADER_CHECKS = [
    ("lr", "train_lr"),
    ("attn_scale", "attn_scale"),
    ("attn_seq", "attn_seq"),
]

# Schema 3 fields with no counterpart in the container. Their presence or
# absence is reported; nothing can confirm them from the bytes.
SCHEMA3_UNVERIFIABLE = ["platform", "source_sha256"]

# Every schema 3 addition, in declaration order. This is the list whose absence
# is reported for each record read.
SCHEMA3_FIELDS = dict(SCHEMA_MARKERS)[3]

SCHEMA_PREFIX = "trios-checkpoint-record/"


def infer_schema(record):
    """Highest schema version whose every marker field is present."""
    inferred = 1
    for version, markers in SCHEMA_MARKERS:
        if all(key in record for key in markers):
            inferred = version
    return inferred


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
    if key == "data_synthetic":
        # The header stores a u8; the sidecar stores a JSON boolean.
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

    Returns `(problems, notes, checked)`: mismatch strings (empty means
    agreement), informational lines about what the record does and does not
    carry, and the number of fields actually compared.
    """
    try:
        with open(sidecar_path, "r") as handle:
            record = json.load(handle)
    except OSError as exc:
        return ["sidecar unreadable: %s" % exc], [], 0
    except ValueError as exc:
        return ["sidecar is not valid JSON: %s" % exc], [], 0

    problems = []
    notes = []
    checked = 0

    declared = declared_schema(record)
    inferred = infer_schema(record)
    notes.append("schema tag %r, fields present up to schema %d"
                 % (record.get("schema"), inferred))
    if declared is not None and declared > inferred:
        # The record names a version whose fields it does not carry. That is a
        # defect in the record, not a version this reader cannot handle.
        problems.append(
            "schema tag claims version %d but the record carries only the "
            "schema %d field set" % (declared, inferred)
        )

    checks = list(SIDECAR_CHECKS)
    for sidecar_key, header_key in SCHEMA3_HEADER_CHECKS:
        if sidecar_key in record and record[sidecar_key] is not None:
            checks.append((sidecar_key, header_key))

    for sidecar_key, header_key in checks:
        if sidecar_key not in record:
            problems.append("sidecar is missing field '%s'" % sidecar_key)
            continue
        left, right = normalize_for_compare(
            sidecar_key, record[sidecar_key], info[header_key]
        )
        checked += 1
        if left != right:
            problems.append(
                "%s: sidecar=%r decoded=%r" % (sidecar_key, left, right)
            )

    absent = [k for k in SCHEMA3_FIELDS if k not in record]
    if absent:
        notes.append("schema 3 provenance fields ABSENT: %s" % ", ".join(absent))
    else:
        notes.append("schema 3 provenance fields: all present")
    null_valued = [k for k in SCHEMA3_FIELDS
                   if k in record and record[k] is None]
    if null_valued:
        notes.append("schema 3 provenance fields present but null: %s"
                     % ", ".join(null_valued))
    return problems, notes, checked


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


def verify_one(bin_path, sidecar_path, verbose):
    """Verify a single checkpoint. Returns (ok, message)."""
    try:
        info = read_checkpoint(bin_path)
    except CheckpointError as exc:
        return False, "%s: %s" % (exc.kind, exc.detail)
    if verbose:
        print_report(info)
    if sidecar_path is None:
        return True, "container ok, sha256 %s" % info["sha256"]
    problems, notes, checked = compare_sidecar(info, sidecar_path)
    if verbose:
        for note in notes:
            print("sidecar         %s" % note)
    if problems:
        return False, "SIDECAR_MISMATCH: " + "; ".join(problems)
    return True, "container ok, sidecar agrees on %d fields, sha256 %s [%s]" % (
        checked, info["sha256"], "; ".join(notes),
    )


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
    for bin_path, sidecar_path in pairs:
        if not os.path.exists(sidecar_path):
            failures += 1
            print("FAIL %s  MISSING_SIDECAR: %s does not exist"
                  % (bin_path, sidecar_path))
            continue
        ok, message = verify_one(bin_path, sidecar_path, verbose=False)
        if ok:
            print("PASS %s  %s" % (bin_path, message))
        else:
            failures += 1
            print("FAIL %s  %s" % (bin_path, message))

    print("summary: %d pair(s), %d passed, %d failed"
          % (len(pairs), len(pairs) - failures, failures))
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
        help="cross-check the decoded header against this sidecar JSON record",
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

    if args.sidecar is not None and not os.path.exists(args.sidecar):
        print("sidecar not found: %s" % args.sidecar, file=sys.stderr)
        return 2

    ok, message = verify_one(args.checkpoint, args.sidecar, verbose=True)
    if ok:
        print("RESULT PASS  %s" % message)
        return 0
    print("RESULT FAIL  %s" % message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
