#!/usr/bin/env python3
"""Measure the divergence between two TRIOSCKP checkpoint artifacts.

Written for one question: when the same seed, the same corpus and the same
source tree produce two checkpoints whose sha256 differ, HOW different are
they? "A different artifact" is not an answer a conformity assessor can use.
This tool turns the difference into numbers that can be re-derived by anyone
holding the two files.

Dependency-free on purpose: standard library only (struct, hashlib, json,
statistics, argparse, os, sys). numpy is deliberately NOT used, so that the
JSON this writes is a function of the two input files and of nothing else -
no BLAS, no vectorised summation order, no version-dependent reductions. The
arithmetic is plain Python floats in a fixed traversal order, which makes the
output bit-reproducible on any Python 3 that can run the file at all. That
matters more here than speed: 212,992 f32 elements take a few seconds.

The container is decoded per interop/SPEC-SNAPSHOT.txt: a 152-byte header, a
152-byte tensor directory of 19 u64 element counts, then the payload, with
file_len = 304 + 4 * sum(directory counts). Everything is little-endian.

Definitions, stated because every one of them is a choice a reader is
entitled to disagree with:

  differing parameter    fa[i] != fb[i] as IEEE-754 values. This is the
                         headline count. It treats +0.0 and -0.0 as EQUAL,
                         because they are the same number, even though they
                         are different bytes. `params_differing_bitwise`
                         reports the byte-level count as well, and
                         `params_signed_zero_only` is the gap between them.
  rel_l2                 ||a - b||_2 / ||b||_2. The denominator is the SECOND
                         file, so the caller decides which artifact is the
                         reference by argument order. `rel_l2_swapped` gives
                         the other normalisation so the choice cannot hide a
                         result.
  median_rel_diff        median over DIFFERING parameters of
                         |a - b| / max(|a|, |b|). Over all parameters the
                         median would be 0 whenever fewer than half of them
                         differ, which says nothing about the ones that do.
                         The max-magnitude denominator is bounded: a sign
                         flip scores 2.0, never infinity, and a value that is
                         zero in one file scores exactly 1.0.
  on_grid                every value of the tensor is an exact multiple of
                         1/16. This is how gf16-quantised tensors identify
                         themselves from the bytes alone, with no appeal to
                         the training source.

Exit codes:
    0  both files decoded and the comparison was written
    1  at least one file failed a format check, or the two are not comparable
    2  usage error
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import struct
import sys

MAGIC = b"TRIOSCKP"
SPEC_FORMAT_VERSION = 1
SPEC_HEADER_LEN = 152
SPEC_TENSOR_COUNT = 19
DIRECTORY_LEN = 8 * SPEC_TENSOR_COUNT
PAYLOAD_OFFSET = SPEC_HEADER_LEN + DIRECTORY_LEN  # 304
ELEMENT_SIZE = 4
RESERVED_OFFSET = 126
RESERVED_LEN = 2

# Canonical tensor order, interop/SPEC-SNAPSHOT.txt. MUST NOT CHANGE in v1.
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

# The gf16 quantisation step. 1/16 is exactly representable in binary, so
# every comparison against this grid is exact and no tolerance is involved.
GRID_STEP = 0.0625

# Tensors whose value spectrum the quantisation section reports in full.
QUANTIZATION_REPORT = ["embed", "proj", "lm_head"]

# Spacings are counted after rounding to this many decimals, so that two
# spacings which differ only in the last f64 bit are not counted as distinct
# grid steps. The reported value is the exact float, not the rounded key.
SPACING_ROUND = 12

DEFAULT_OUTPUT = os.path.join("docs", "cross-arch-divergence.json")


class CheckpointError(Exception):
    """A named container defect. `kind` is the stable machine-readable name."""

    def __init__(self, kind, detail):
        super().__init__("%s: %s" % (kind, detail))
        self.kind = kind
        self.detail = detail


def parse_container(raw, path):
    """Decode a TRIOSCKP image far enough to compare it. Raises on defects."""
    if len(raw) < len(MAGIC):
        raise CheckpointError(
            "TRUNCATED_MAGIC",
            "file is %d bytes, need at least %d for the magic"
            % (len(raw), len(MAGIC)),
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

    info = {"path": path, "bytes": len(raw)}
    info["format_version"] = struct.unpack_from("<I", raw, 8)[0]
    if info["format_version"] != SPEC_FORMAT_VERSION:
        raise CheckpointError(
            "UNSUPPORTED_FORMAT_VERSION",
            "this tool implements version %d, file declares %d"
            % (SPEC_FORMAT_VERSION, info["format_version"]),
        )
    info["header_len"] = struct.unpack_from("<I", raw, 12)[0]
    if info["header_len"] != SPEC_HEADER_LEN:
        raise CheckpointError(
            "BAD_HEADER_LEN",
            "version 1 fixes header_len at %d, file declares %d"
            % (SPEC_HEADER_LEN, info["header_len"]),
        )
    info["tensor_count"] = struct.unpack_from("<I", raw, 52)[0]
    if info["tensor_count"] != SPEC_TENSOR_COUNT:
        raise CheckpointError(
            "BAD_TENSOR_COUNT",
            "version 1 fixes tensor_count at %d, file declares %d"
            % (SPEC_TENSOR_COUNT, info["tensor_count"]),
        )

    reserved = raw[RESERVED_OFFSET:RESERVED_OFFSET + RESERVED_LEN]
    if reserved != b"\x00" * RESERVED_LEN:
        raise CheckpointError(
            "RESERVED_NONZERO",
            "the %d reserved bytes at offset %d must be zero, found %r"
            % (RESERVED_LEN, RESERVED_OFFSET, reserved),
        )

    if len(raw) < PAYLOAD_OFFSET:
        raise CheckpointError(
            "TRUNCATED_DIRECTORY",
            "file is %d bytes, header plus %d-entry directory is %d"
            % (len(raw), SPEC_TENSOR_COUNT, PAYLOAD_OFFSET),
        )
    counts = list(struct.unpack_from("<%dQ" % SPEC_TENSOR_COUNT, raw, SPEC_HEADER_LEN))
    info["counts"] = counts
    info["params"] = sum(counts)

    expected = PAYLOAD_OFFSET + ELEMENT_SIZE * info["params"]
    if len(raw) != expected:
        raise CheckpointError(
            "BAD_FILE_LENGTH",
            "directory declares %d elements, so file_len must be %d; file is %d"
            % (info["params"], expected, len(raw)),
        )

    info["seed"] = struct.unpack_from("<Q", raw, 108)[0]
    info["step"] = struct.unpack_from("<Q", raw, 116)[0]
    info["gf16_enabled"] = raw[124]
    info["data_synthetic"] = raw[125]
    info["header"] = raw[:SPEC_HEADER_LEN]
    info["directory_bytes"] = raw[SPEC_HEADER_LEN:PAYLOAD_OFFSET]
    info["values"] = struct.unpack_from("<%df" % info["params"], raw, PAYLOAD_OFFSET)
    info["words"] = struct.unpack_from("<%dI" % info["params"], raw, PAYLOAD_OFFSET)
    info["sha256"] = hashlib.sha256(raw).hexdigest()
    info["raw"] = raw
    return info


def read_checkpoint(path):
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise CheckpointError("UNREADABLE_FILE", str(exc))
    return parse_container(raw, path)


def on_grid(value):
    """True when `value` is an exact multiple of the 1/16 quantisation step.

    1/16 is a power of two, so the division is exact and the test needs no
    tolerance. A non-finite value is never on the grid, and saying so here
    keeps int() from raising on a poisoned checkpoint.
    """
    if not math.isfinite(value):
        return False
    quotient = value / GRID_STEP
    return quotient == int(quotient)


def spectrum(values):
    """Distinct values, their modal adjacent spacing, and the grid test."""
    distinct = sorted(set(values))
    report = {
        "count": len(values),
        "distinct": len(distinct),
        "min": distinct[0] if distinct else None,
        "max": distinct[-1] if distinct else None,
        "on_grid": all(on_grid(v) for v in distinct),
    }
    if len(distinct) < 2:
        report["modal_spacing"] = None
        report["modal_spacing_share"] = None
        report["spacings"] = 0
        return report

    tally = {}
    for i in range(len(distinct) - 1):
        gap = distinct[i + 1] - distinct[i]
        key = round(gap, SPACING_ROUND)
        entry = tally.get(key)
        if entry is None:
            tally[key] = [1, gap]
        else:
            entry[0] += 1
    total = len(distinct) - 1
    # Ties are broken by the smaller spacing, so the result never depends on
    # dict insertion order.
    best_key = min(tally, key=lambda k: (-tally[k][0], k))
    report["modal_spacing"] = tally[best_key][1]
    report["modal_spacing_share"] = tally[best_key][0] / total
    report["spacings"] = total
    return report


def compare(a, b):
    """Build the full comparison record for two decoded checkpoints."""
    if a["counts"] != b["counts"]:
        raise CheckpointError(
            "SHAPE_MISMATCH",
            "tensor directories differ: %r vs %r" % (a["counts"], b["counts"]),
        )

    n = a["params"]
    fa, fb = a["values"], b["values"]
    wa, wb = a["words"], b["words"]

    differing = []
    bitwise = 0
    zero_both = 0
    for i in range(n):
        if wa[i] != wb[i]:
            bitwise += 1
        if fa[i] != fb[i]:
            differing.append(i)
        elif fa[i] == 0.0:
            zero_both += 1

    sq = 0.0
    for i in differing:
        d = fa[i] - fb[i]
        sq += d * d
    norm_diff = sq ** 0.5
    norm_a = sum(x * x for x in fa) ** 0.5
    norm_b = sum(x * x for x in fb) ** 0.5

    rels = []
    max_abs = 0.0
    max_idx = None
    for i in differing:
        d = abs(fa[i] - fb[i])
        if d > max_abs:
            max_abs = d
            max_idx = i
        scale = max(abs(fa[i]), abs(fb[i]))
        if scale > 0.0:
            rels.append(d / scale)

    tensors = []
    offset = 0
    index_of = {}
    for name, count in zip(TENSOR_NAMES, a["counts"]):
        index_of[name] = (offset, count)
        local_diff = 0
        local_max = 0.0
        for i in range(offset, offset + count):
            if fa[i] != fb[i]:
                local_diff += 1
                d = abs(fa[i] - fb[i])
                if d > local_max:
                    local_max = d
        tensors.append({
            "name": name,
            "count": count,
            "differing": local_diff,
            "pct_differing": (100.0 * local_diff / count) if count else 0.0,
            "max_abs_diff": local_max,
            "distinct_a": len(set(fa[offset:offset + count])),
            "distinct_b": len(set(fb[offset:offset + count])),
            "on_grid_a": all(on_grid(v) for v in set(fa[offset:offset + count])),
            "on_grid_b": all(on_grid(v) for v in set(fb[offset:offset + count])),
        })
        offset += count

    def tensor_of(index):
        for name, (start, count) in index_of.items():
            if start <= index < start + count:
                return name
        return None

    # Of the differing parameters that live in gf16-quantised tensors, how
    # many differ by an exact multiple of the 1/16 grid step? This is the
    # mechanism claim, measured rather than asserted.
    grid_names = [t["name"] for t in tensors if t["on_grid_a"] and t["on_grid_b"]]
    grid_diff = 0
    grid_on_step = 0
    grid_steps_tally = {}
    for name in grid_names:
        start, count = index_of[name]
        for i in range(start, start + count):
            if fa[i] == fb[i]:
                continue
            grid_diff += 1
            delta = abs(fa[i] - fb[i])
            steps = delta / GRID_STEP
            if math.isfinite(steps) and steps == int(steps):
                grid_on_step += 1
                key = int(steps)
                grid_steps_tally[key] = grid_steps_tally.get(key, 0) + 1

    first_byte = None
    ra, rb = a["raw"], b["raw"]
    limit = min(len(ra), len(rb))
    for i in range(limit):
        if ra[i] != rb[i]:
            first_byte = i
            break

    record = {
        "tool": "scripts/compare_checkpoints.py",
        "container": "TRIOSCKP",
        "container_version": SPEC_FORMAT_VERSION,
        "spec": "interop/SPEC-SNAPSHOT.txt",
        "a": {
            "path": a["path"],
            "bytes": a["bytes"],
            "sha256": a["sha256"],
            "seed": a["seed"],
            "step": a["step"],
            "gf16_enabled": a["gf16_enabled"],
            "data_synthetic": a["data_synthetic"],
        },
        "b": {
            "path": b["path"],
            "bytes": b["bytes"],
            "sha256": b["sha256"],
            "seed": b["seed"],
            "step": b["step"],
            "gf16_enabled": b["gf16_enabled"],
            "data_synthetic": b["data_synthetic"],
        },
        "files_identical": a["sha256"] == b["sha256"],
        "header_bytes_equal": a["header"] == b["header"],
        "directory_bytes_equal": a["directory_bytes"] == b["directory_bytes"],
        "payload_offset": PAYLOAD_OFFSET,
        "params": n,
        "params_differing": len(differing),
        "frac_differing": len(differing) / n if n else 0.0,
        "pct_differing": (100.0 * len(differing) / n) if n else 0.0,
        "params_differing_bitwise": bitwise,
        "params_signed_zero_only": bitwise - len(differing),
        "params_zero_in_both": zero_both,
        "l2_a": norm_a,
        "l2_b": norm_b,
        "l2_diff": norm_diff,
        "rel_l2": (norm_diff / norm_b) if norm_b else None,
        "rel_l2_swapped": (norm_diff / norm_a) if norm_a else None,
        "rel_l2_denominator": "l2 norm of b (%s)" % b["path"],
        "median_rel_diff": statistics.median(rels) if rels else None,
        "median_rel_diff_definition":
            "median over differing parameters of |a-b| / max(|a|,|b|)",
        "median_rel_diff_n": len(rels),
        "max_abs_diff": max_abs,
        "max_abs_diff_index": max_idx,
        "max_abs_diff_tensor": tensor_of(max_idx) if max_idx is not None else None,
        "max_abs_diff_a": fa[max_idx] if max_idx is not None else None,
        "max_abs_diff_b": fb[max_idx] if max_idx is not None else None,
        "first_differing_byte": first_byte,
        "first_differing_byte_section": (
            None if first_byte is None
            else "header" if first_byte < SPEC_HEADER_LEN
            else "directory" if first_byte < PAYLOAD_OFFSET
            else "payload"
        ),
        "first_differing_param": (
            None if first_byte is None or first_byte < PAYLOAD_OFFSET
            else (first_byte - PAYLOAD_OFFSET) // ELEMENT_SIZE
        ),
        "tensors": tensors,
        "grid_step": GRID_STEP,
        "grid_tensors": grid_names,
        "grid_params_differing": grid_diff,
        "grid_params_differing_on_step": grid_on_step,
        "grid_step_histogram": {
            str(k): grid_steps_tally[k] for k in sorted(grid_steps_tally)
        },
        "quantization": {},
    }
    record["first_differing_param_tensor"] = (
        tensor_of(record["first_differing_param"])
        if record["first_differing_param"] is not None else None
    )
    for name in QUANTIZATION_REPORT:
        start, count = index_of[name]
        record["quantization"][name] = {
            "a": spectrum(fa[start:start + count]),
            "b": spectrum(fb[start:start + count]),
        }
    return record


def print_report(record):
    print("a                %s" % record["a"]["path"])
    print("  sha256         %s" % record["a"]["sha256"])
    print("b                %s" % record["b"]["path"])
    print("  sha256         %s" % record["b"]["sha256"])
    print("bytes            %d / %d" % (record["a"]["bytes"], record["b"]["bytes"]))
    print("header equal     %s" % record["header_bytes_equal"])
    print("directory equal  %s" % record["directory_bytes_equal"])
    print("params           %d" % record["params"])
    print("differing        %d (%.2f%%)"
          % (record["params_differing"], record["pct_differing"]))
    print("differing (bits) %d, of which signed-zero only %d"
          % (record["params_differing_bitwise"], record["params_signed_zero_only"]))
    print("rel L2           %.6f  (%s)"
          % (record["rel_l2"], record["rel_l2_denominator"]))
    print("median rel diff  %.6f  (%s)"
          % (record["median_rel_diff"], record["median_rel_diff_definition"]))
    print("max abs diff     %.6f  at index %s in %s (a=%r b=%r)"
          % (record["max_abs_diff"], record["max_abs_diff_index"],
             record["max_abs_diff_tensor"], record["max_abs_diff_a"],
             record["max_abs_diff_b"]))
    print("first diff byte  %s (%s)"
          % (record["first_differing_byte"], record["first_differing_byte_section"]))
    print("per tensor")
    print("  %-10s %8s %8s %8s %8s %6s" %
          ("name", "count", "differ", "pct", "maxabs", "grid"))
    for t in record["tensors"]:
        print("  %-10s %8d %8d %7.2f%% %8.6f %6s"
              % (t["name"], t["count"], t["differing"], t["pct_differing"],
                 t["max_abs_diff"], "yes" if t["on_grid_a"] and t["on_grid_b"] else "no"))
    print("gf16 grid step   %r" % record["grid_step"])
    print("  differing parameters in on-grid tensors: %d, of which an exact "
          "multiple of the step: %d"
          % (record["grid_params_differing"], record["grid_params_differing_on_step"]))
    for name in QUANTIZATION_REPORT:
        q = record["quantization"][name]
        print("  %-8s a: %d distinct over %d, modal spacing %r (%d of %d gaps)"
              % (name, q["a"]["distinct"], q["a"]["count"], q["a"]["modal_spacing"],
                 round(q["a"]["modal_spacing_share"] * q["a"]["spacings"]),
                 q["a"]["spacings"]))
        print("  %-8s b: %d distinct over %d, modal spacing %r (%d of %d gaps)"
              % ("", q["b"]["distinct"], q["b"]["count"], q["b"]["modal_spacing"],
                 round(q["b"]["modal_spacing_share"] * q["b"]["spacings"]),
                 q["b"]["spacings"]))


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv):
    parser = argparse.ArgumentParser(
        description="Measure how far apart two TRIOSCKP checkpoints are.",
    )
    parser.add_argument("a", help="first checkpoint (.bin)")
    parser.add_argument("b", help="second checkpoint (.bin); the L2 reference")
    parser.add_argument(
        "--json", action="store_true",
        help="write the comparison record to stdout and to no file",
    )
    parser.add_argument(
        "--out", metavar="PATH",
        help="write the comparison record to PATH (default: %s under the "
             "repository root, used when --json is not given)" % DEFAULT_OUTPUT,
    )
    args = parser.parse_args(argv)

    try:
        a = read_checkpoint(args.a)
        b = read_checkpoint(args.b)
        record = compare(a, b)
    except CheckpointError as exc:
        print("FAIL %s: %s" % (exc.kind, exc.detail), file=sys.stderr)
        return 1

    text = json.dumps(record, indent=2, sort_keys=True)

    if args.json:
        print(text)
        if args.out:
            with open(args.out, "w") as handle:
                handle.write(text + "\n")
        return 0

    print_report(record)
    out = args.out or os.path.join(repo_root(), DEFAULT_OUTPUT)
    directory = os.path.dirname(out)
    if directory and not os.path.isdir(directory):
        print("output directory does not exist: %s" % directory, file=sys.stderr)
        return 2
    with open(out, "w") as handle:
        handle.write(text + "\n")
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
