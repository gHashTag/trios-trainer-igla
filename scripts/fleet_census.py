#!/usr/bin/env python3
"""Re-derive the LOCAL sweep census from the log files themselves.

Why this script exists
----------------------
This repository has quoted three different "fleet" numbers -- 1,851 / 1,878 /
7,927 -- as if they were the same quantity. They are not. See
docs/FLEET-COUNT.md for the three-way decomposition. This script owns exactly
one of the three: the LOCAL sweep log population. It re-counts on every run so
the figure can never go stale the way 7,927 did.

WHICH POPULATION THIS COUNTS -- the decision, stated once
---------------------------------------------------------
DECISION: DEFAULT_GLOB stays `v6_*.log`. This script counts the v6 wave ONLY.
It is deliberately NOT widened to `*.log`, because the v6 wave is the only
population whose writer is committed in this tree and therefore the only one
whose truncation behaviour is known (see below). The glob is not widened, so
the report must instead SAY what it counted and what it left out: every run
prints the glob it counted, the `*.log` denominator, and each excluded family
with its count. A reader who runs `ls .trinity/results/*.log | wc -l` gets a
larger number, and the report tells them in advance exactly why and by how much.

The label everywhere -- here, in the report, in docs/FLEET-COUNT.md -- is
therefore "v6-wave sweep log files", never "sweep log files".

WHAT THE COUNTS MEAN, precisely
-------------------------------
The file count is a count of distinct CONFIGURATIONS that got a log, not a
count of processes launched, PROVIDED the writer truncates. Whether it does is
known for a small minority of the counted files and assumed for the rest, and
this script refuses to leave that distinction implicit.

THE COUNTED POPULATION HAS THREE FILENAME SHAPES, not one. Re-derived on this
tree 2026-08-07 by grouping the counted filenames on underscore-component
count; the three sum to the counted total exactly (165 + 7742 + 130 = 8037):

  7 components, opt-terminated    165  v6_{fmt}_h{hidden}_lr{lr}_seed{seed}_
                                       {step}_{opt}
  6 components                   7742  v6_{fmt}_h{hidden}_lr{lr}_seed{seed}_
                                       {step}
  4 components                    130  v6_{fmt}_seed{seed}_{step}

Only two of the three match a template committed anywhere in this repository:

  * 4-component -> cfbe605 (2026-05-25 10:37 +0700) scripts/auto_launch.sh:42
    name="${wave}_${fmt}_seed${seed}_${step}", redirect > "$log" at :54.
  * 7-component -> 85f33fd (2026-05-31 14:36 +0700) scripts/auto_launch.sh:72
    name="${wave}_${fmt}_h${hidden}_lr${lr}_seed${seed}_${step}_${opt}",
    redirect > "$log" at :84. This is also the template at HEAD.
  * 6-component -> NOTHING. `git log --all -S` on that template returns no
    commit, and the string is absent from the working tree as well.

So the truncation argument is MEASURED for 295 of the 8,037 counted files and
ASSUMED for the other 7,742, which is 96% of the population. That is the
identical condition docs/FLEET-COUNT.md gives as its reason for EXCLUDING the
v2_..v5_ families: no committed writer, therefore unknown semantics. It is
stated rather than buried, and --self-test case 7 keeps it honest by refusing
any counted file that matches none of the three shapes.

Subject to that, FOR THE v6 POPULATION:

  * file count  == EXACT count of configuration slots that produced a log
  * file count  == LOWER BOUND on the number of training processes launched
  * file count  != fleet size, and never was

and the exactness half rests on a separate MEASUREMENT, not on an argument
about the template. The two shapes that drop fields from the name (6-component
drops ${opt}; 4-component drops ${opt}, h${hidden} and lr${lr}) can only
collide if the dropped fields varied. Read out of the logs' own
`[trios-train] parsed` line on 2026-08-07: all 7,742 six-component logs record
optimizer=adamw, and all 130 four-component logs record optimizer=adamw
hidden=384 lr=0.003. The dropped fields were constant across every SURVIVING
log. Limit of that, stated: an overwrite destroys its own evidence, so it does
not by itself exclude a collision whose adamw arm wrote last -- which for the
six-component shape would have to hold 7,742 times out of 7,742. What is
observable is that no surviving log of either shorter shape records a value its
filename could not have carried. The same pass checked the other direction and
found 0 name/content mismatches in all 8,037 files.

That argument covers auto_launch.sh and NOTHING ELSE. scripts/format_sweep.sh:44
also truncates (`tee`), but it writes the `format_sweep_*` family, which this
glob EXCLUDES -- so its truncation is not evidence about anything counted here,
and citing it as such would be an argument for a population the count omits.
For the other excluded families (v2_/v3_/v4_/v5_/long_/ultra_/mega_/sweep_/
champion_) no writer is committed in this tree at all: `grep -rln 'trinity/
results' scripts/` names only auto_launch.sh, format_sweep.sh, two monitors
that read, and this file. Their truncation behaviour is UNKNOWN, not assumed --
which is the second reason not to widen the glob and silently pool them in.

The DONE: and val_bpb counts are per-FILE (a file is counted once no matter how
many matching lines it holds), for the same reason.

Usage
-----
  python3 scripts/fleet_census.py
  python3 scripts/fleet_census.py --results-dir <path> --glob 'v6_*.log'
  python3 scripts/fleet_census.py --self-test

Refusal contract
----------------
If the results directory does not exist, or the glob matches no file, this
script prints an explicit refusal naming the path and exits NONZERO WITHOUT
printing any count. A census that answers "0 logs" with exit 0 when the
directory has been deleted is a failure that reports success -- the exact
defect class this repository exists to remove.
"""

import argparse
import datetime
import glob as globmod
import io
import os
import re
import shutil
import sys
import tempfile

DEFAULT_GLOB = "v6_*.log"
DEFAULT_SUBDIR = os.path.join(".trinity", "results")

# The denominator a reader will reach for on their own:
#   ls .trinity/results/*.log | wc -l
DENOMINATOR_GLOB = "*.log"

# Families the census glob leaves out, named so the report can enumerate them
# instead of leaving the difference for the reader to discover. Order is the
# print order; anything unmatched by all of them lands in "other".
EXCLUDED_FAMILIES = (
    ("v2", "v2_*.log"),
    ("v3", "v3_*.log"),
    ("v4", "v4_*.log"),
    ("v5", "v5_*.log"),
    ("format_sweep", "format_sweep_*.log"),
)


# The three filename shapes the counted population actually has, re-derived
# 2026-08-07 by grouping every counted filename on underscore-component count.
# Each entry is (label, regex over the STEM, template, provenance).
#
# `provenance` is the load-bearing column and it is why this table exists at
# all. docs/FLEET-COUNT.md used to justify the whole population's truncation
# behaviour by citing ONE template, auto_launch.sh:72,84 -- which is the
# 7-component shape, 165 files, 2% of the count. The 7,742-file majority
# matches no template committed anywhere, so its truncation behaviour is
# ASSUMED. Naming that here, per shape, is the fix.
SHAPES = (
    (
        "7 components, opt-terminated",
        re.compile(r"^v6_[^_]+_h\d+_lr[0-9.]+_seed\d+_\d+_[^_]+$"),
        "v6_{fmt}_h{hidden}_lr{lr}_seed{seed}_{step}_{opt}",
        "COMMITTED: 85f33fd scripts/auto_launch.sh:72, redirect > at :84; "
        "also the template at HEAD. Truncation MEASURED.",
    ),
    (
        "6 components",
        re.compile(r"^v6_[^_]+_h\d+_lr[0-9.]+_seed\d+_\d+$"),
        "v6_{fmt}_h{hidden}_lr{lr}_seed{seed}_{step}",
        "NO COMMITTED TEMPLATE writes this shape (git log --all -S returns "
        "nothing, and the string is absent from the working tree). "
        "Truncation ASSUMED, not measured.",
    ),
    (
        "4 components",
        re.compile(r"^v6_[^_]+_seed\d+_\d+$"),
        "v6_{fmt}_seed{seed}_{step}",
        "COMMITTED: cfbe605 scripts/auto_launch.sh:42, redirect > at :54. "
        "Truncation MEASURED.",
    ),
)


class CensusRefusal(Exception):
    """Raised instead of returning a count that would be a lie."""


class UnknownShape(CensusRefusal):
    """A counted file whose name matches none of the three known shapes.

    This is a refusal and not a warning on purpose. The document's claim about
    what the count MEANS is made shape by shape; a file outside every shape is
    a file the claim does not cover, and printing the total anyway would be a
    count carrying an explanation that is not about all of it.
    """


def classify(basename):
    """Return the SHAPES label for a filename, or None if it matches none."""
    stem = basename[: -len(".log")] if basename.endswith(".log") else basename
    for label, pattern, _template, _provenance in SHAPES:
        if pattern.match(stem):
            return label
    return None


def shape_census(files):
    """Return [(label, count), ...] in SHAPES order.

    Raises UnknownShape, naming the offending files, if any counted file
    matches no shape. THE GUARD: it converts "the page's writer citation covers
    2% of the population" from something a reader has to notice into something
    this script will not pass over.
    """
    counts = {label: 0 for label, _, _, _ in SHAPES}
    unclassified = []
    for path in files:
        base = os.path.basename(path)
        label = classify(base)
        if label is None:
            unclassified.append(base)
        else:
            counts[label] += 1

    if unclassified:
        shown = ", ".join(sorted(unclassified)[:10])
        more = "" if len(unclassified) <= 10 else " (and %d more)" % (
            len(unclassified) - 10,
        )
        raise UnknownShape(
            "%d counted file(s) match NONE of the %d known filename shapes: "
            "%s%s. The known shapes are: %s. Every statement this census and "
            "docs/FLEET-COUNT.md make about what the count MEANS -- that the "
            "writer truncates, that the name is a pure function of the config, "
            "that no two configs collide -- is made shape by shape. A file "
            "outside every shape is a file none of those statements covers, so "
            "the total is not printed."
            % (
                len(unclassified),
                len(SHAPES),
                shown,
                more,
                "; ".join("%s (%s)" % (label, tmpl) for label, _, tmpl, _ in SHAPES),
            )
        )

    return [(label, counts[label]) for label, _, _, _ in SHAPES]


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def census(results_dir, pattern):
    """Return (files, done_files, val_bpb_files, pattern_used).

    Raises CensusRefusal rather than returning zero counts.
    """
    if not os.path.isdir(results_dir):
        raise CensusRefusal(
            "results directory does not exist: %s" % os.path.abspath(results_dir)
        )

    pattern_used = os.path.join(results_dir, pattern)
    files = sorted(globmod.glob(pattern_used))
    if not files:
        raise CensusRefusal(
            "glob matched no file: %s (directory exists, so either the sweep "
            "logs were deleted or the glob is wrong)" % os.path.abspath(pattern_used)
        )

    done = 0
    with_val_bpb = 0
    for path in files:
        saw_done = False
        saw_val = False
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if not saw_done and line.startswith("DONE:"):
                        saw_done = True
                    if not saw_val and "val_bpb" in line:
                        saw_val = True
                    if saw_done and saw_val:
                        break
        except OSError as exc:
            raise CensusRefusal("could not read %s: %s" % (path, exc))
        if saw_done:
            done += 1
        if saw_val:
            with_val_bpb += 1

    return files, done, with_val_bpb, pattern_used


def decompose(results_dir, pattern):
    """Return (total, matched, [(label, count), ...]) for the excluded families.

    `total` is over DENOMINATOR_GLOB unioned with the census glob, so the
    arithmetic always closes: matched + sum(excluded counts) == total, for any
    glob, including one that matches files outside `*.log`.

    A file the census glob matched is NEVER reported as excluded, even when an
    excluded family's glob would also match it.
    """
    denominator = set(globmod.glob(os.path.join(results_dir, DENOMINATOR_GLOB)))
    matched = set(globmod.glob(os.path.join(results_dir, pattern)))
    unmatched = denominator - matched

    families = []
    accounted = set()
    for label, family_glob in EXCLUDED_FAMILIES:
        hits = set(globmod.glob(os.path.join(results_dir, family_glob))) & unmatched
        hits -= accounted
        accounted |= hits
        families.append((label, len(hits)))
    families.append(("other", len(unmatched - accounted)))

    return len(denominator | matched), len(matched), families


def population_line(pattern, total, matched, families):
    """One line naming the glob counted AND every population excluded.

    Example:
      v6_*.log 8037 of 8195 *.log; excluded: v2 30, v3 36, v4 9, v5 11,
      format_sweep 19, other 53
    """
    shown = ["%s %d" % (label, count) for label, count in families if count]
    excluded = ", ".join(shown) if shown else "none (the glob matched every %s)" % (
        DENOMINATOR_GLOB,
    )
    return "%s %d of %d %s; excluded: %s" % (
        pattern,
        matched,
        total,
        DENOMINATOR_GLOB,
        excluded,
    )


def _wrap(text, width):
    """Greedy word wrap. No textwrap import for four lines of output."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def report(results_dir, pattern, stream=sys.stdout):
    files, done, with_val_bpb, pattern_used = census(results_dir, pattern)
    # Classify BEFORE anything is printed. The refusal contract of this script
    # is that nothing is emitted ahead of a refusal, so a reader never sees a
    # partial report and mistakes it for a whole one.
    shapes = shape_census(files)
    total, matched, families = decompose(results_dir, pattern)
    as_of = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")

    stream.write("LOCAL SWEEP CENSUS (re-derived, not transcribed)\n")
    stream.write("as-of:          %s\n" % as_of)
    stream.write("host:           %s\n" % os.uname().nodename)
    stream.write("glob:           %s\n" % os.path.abspath(pattern_used))
    stream.write("\n")
    # Each row carries its meaning INLINE. The previous version printed three
    # bare counts and left the reader to supply the labels, which is how 7,274
    # circulated for a day with no definition anywhere in the documents -- an
    # unlabelled number beside two labelled ones reads as a third fleet size.
    stream.write(
        "%-30s %6d   the counted population: %s only, %d of %d %s.\n"
        "%-30s %6s   NOT a fleet size.\n"
        % (
            "v6-wave log files",
            len(files),
            pattern,
            matched,
            total,
            DENOMINATOR_GLOB,
            "",
            "",
        )
    )
    stream.write(
        "%-30s %6d   RAN TO COMPLETION -- %d of %d. The completion count.\n"
        % ("files with a 'DONE:' line", done, done, len(files))
    )
    stream.write(
        "%-30s %6d   REACHED AN EVAL -- %d of %d emitted at least one\n"
        "%-30s %6s   reading. NOT a completion count (that is the %d above)\n"
        "%-30s %6s   and NOT a fleet size: a run killed after its first\n"
        "%-30s %6s   eval line is counted here.\n"
        % (
            "files containing 'val_bpb'",
            with_val_bpb,
            with_val_bpb,
            len(files),
            "",
            "",
            done,
            "",
            "",
            "",
            "",
        )
    )
    stream.write("\n")
    stream.write("population:     %s\n" % population_line(pattern, total, matched, families))
    stream.write("\n")

    # The shape decomposition. This is the block that keeps the truncation
    # argument honest: it says, per shape, whether a writer for it is committed
    # anywhere, and therefore whether the truncation behaviour behind the
    # "exact count of config slots" reading was MEASURED or ASSUMED.
    stream.write(
        "filename shapes (every counted file must match one, or this script "
        "REFUSES):\n"
    )
    measured = 0
    for (label, count), (_, _, template, provenance) in zip(shapes, SHAPES):
        stream.write("  %-30s %6d   %s\n" % (label, count, template))
        for chunk in _wrap(provenance, 58):
            stream.write("  %-30s %6s   %s\n" % ("", "", chunk))
        if provenance.startswith("COMMITTED"):
            measured += count
    shape_total = sum(count for _, count in shapes)
    stream.write("  %-30s %6d   == the counted population above\n" % ("classified", shape_total))
    stream.write(
        "  %-30s %6s   truncation MEASURED for %d of %d counted files "
        "(%.0f%%);\n"
        "  %-30s %6s   ASSUMED for the other %d. That is the SAME condition\n"
        "  %-30s %6s   docs/FLEET-COUNT.md gives for EXCLUDING v2_..v5_.\n"
        % (
            "",
            "",
            measured,
            shape_total,
            100.0 * measured / shape_total if shape_total else 0.0,
            "",
            "",
            shape_total - measured,
            "",
            "",
        )
    )
    stream.write("\n")

    stream.write(
        "These are counts of distinct CONFIGURATIONS that produced a log --\n"
        "an exact count of config slots, and a LOWER BOUND on launches, IF the\n"
        "writer truncates. For the shapes marked COMMITTED above that is\n"
        "measured: the name is a pure function of the config and the redirect is\n"
        "'> $log'. For the 6-component shape no writer is committed anywhere, so\n"
        "its truncation is ASSUMED. What IS measured for all three shapes is the\n"
        "observable half of the no-collision claim: the fields a shape drops from\n"
        "the name (optimizer, and for 4-component also hidden and lr) were\n"
        "constant across every SURVIVING log of that shape -- read out of the\n"
        "logs' own '[trios-train] parsed' line, not argued. An overwrite destroys\n"
        "its own evidence, so that does not by itself exclude a collision whose\n"
        "adamw arm wrote last. That argument is about\n"
        "auto_launch.sh and nothing else -- format_sweep.sh also truncates, but\n"
        "its files are EXCLUDED above, and for the remaining excluded families no\n"
        "writer is committed in this tree, so their behaviour is unknown rather\n"
        "than assumed.\n"
        "This is NOT a fleet size. See docs/FLEET-COUNT.md.\n"
    )
    stream.write(
        "The directory is gitignored (.gitignore:54 '.trinity/results/'), so\n"
        "this population exists on ONE workstation and cannot be re-derived\n"
        "from a clone.\n"
    )
    return len(files), done, with_val_bpb


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _write(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


SELF_TEST_CASES = 8


def self_test():
    """Prove both halves: the counts are right, AND the refusal actually fires.

    An assertion never seen to fail is not evidence, so the refusal cases run
    the missing-directory and empty-directory paths for real and require a
    nonzero exit with no count printed. Case 6 does the same for the
    excluded-population line: delete that line from report() and case 6 fails,
    naming the families that went missing. Cases 7 and 8 do it for the shape
    classification: 7 requires a counted file matching no shape to REFUSE with
    a nonzero exit and no count, and 8 requires the printed shape block to name
    every shape with its count and to mark, per shape, whether a writer is
    committed for it.

    Every fixture filename below is a REAL shape. That is not cosmetic: the
    fixtures used to be `v6_a.log`, which matches no shape the real population
    has, so the fixtures were exercising a classification the census would now
    refuse.
    """
    failures = []
    passed = []

    def ok(message):
        passed.append(message)
        print("[self-test] PASS %s" % message)

    tmp = tempfile.mkdtemp(prefix="fleet_census_selftest_")
    try:
        # Fixture: 5 logs; 2 with a 'DONE:' line; 3 containing 'val_bpb'.
        # All three real shapes are represented, so this case also exercises
        # the classifier on every branch.
        a = "v6_f32_h256_lr0.001_seed42_20000_adamw.log"   # 7 components
        b = "v6_f32_h256_lr0.001_seed43_20000.log"         # 6 components
        c = "v6_gf8_seed44_20000.log"                      # 4 components
        d = "v6_bf16_h384_lr0.003_seed45_50000.log"        # 6 components
        e = "v6_fp16_h512_lr0.0003_seed46_50000_muon.log"  # 7 components
        _write(os.path.join(tmp, a), "step=1\nDONE: ok\nval_bpb=2.61\n")
        _write(os.path.join(tmp, b), "DONE: ok\nnothing else\n")
        _write(os.path.join(tmp, c), "val_bpb=3.31\nstill running\n")
        _write(os.path.join(tmp, d), "Initial val_bpb=7.00\n")
        # Decoys that must NOT be counted: wrong prefix, and 'DONE:' not at
        # line start (a line that merely mentions DONE is not a completion).
        _write(os.path.join(tmp, "v2_z_seed1_100.log"), "DONE: ok\nval_bpb=1.0\n")
        _write(os.path.join(tmp, "v6_f32_seed47_20000.txt"), "DONE: ok\nval_bpb=1.0\n")
        _write(os.path.join(tmp, e), "not DONE: yet, no metric here\n")

        buf = io.StringIO()
        got = report(tmp, DEFAULT_GLOB, stream=buf)
        want = (5, 2, 3)
        if got != want:
            failures.append(
                "fixture counts wrong: got (files, done, val_bpb)=%r want %r" % (got, want)
            )
        else:
            ok("counts on fixture: 5 logs, 2 DONE:, 3 val_bpb")

        # 'DONE:' must be line-initial: fixture `e` says 'not DONE: yet'.
        if "DONE" in buf.getvalue() and got[1] != 2:
            failures.append("line-initial DONE: matching is broken")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- NUL-byte regression, observed in the real population -------------
    # Three real logs (v6_f32_seed44_20000.log, v6_fp16_seed49_20000.log,
    # v6_gf8_seed46_20000.log) carry NUL bytes from an interrupted write.
    # macOS `grep -l val_bpb` SKIPS them silently -- exit 1, no output, no
    # warning -- so the shell one-liner undercounts by 3 and looks correct.
    # This reader must not repeat that. A log with a metric in it counts,
    # whatever bytes surround the metric.
    tmp = tempfile.mkdtemp(prefix="fleet_census_nul_")
    try:
        with open(os.path.join(tmp, "v6_f32_seed44_20000.log"), "wb") as handle:
            handle.write(b"step=1\x00\x00\x00 val_bpb=2.61\nDONE: ok\n")
        _write(os.path.join(tmp, "v6_fp16_seed49_20000.log"), "val_bpb=3.0\n")

        buf = io.StringIO()
        got = report(tmp, DEFAULT_GLOB, stream=buf)
        want = (2, 1, 2)
        if got != want:
            failures.append(
                "NUL-byte regression: a log containing NUL bytes must still be "
                "read for 'val_bpb' and 'DONE:' -- macOS grep -l skips such "
                "files silently. got %r want %r" % (got, want)
            )
        else:
            ok(
                "NUL-containing log still counted "
                "(macOS 'grep -l' silently skips these)"
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- case 6: the report must name its glob AND every excluded family ---
    # The defect this guards: a row labelled "sweep log files" over a count
    # produced by a v6_-scoped glob. A reader who runs `ls *.log | wc -l` then
    # mints a fourth disagreeing figure. The count is not wrong; the label is,
    # and silence about the exclusions is what makes it wrong.
    tmp = tempfile.mkdtemp(prefix="fleet_census_population_")
    try:
        counted = [
            "v6_f32_h256_lr0.001_seed42_20000_adamw.log",
            "v6_f32_h256_lr0.001_seed43_20000.log",
            "v6_gf8_seed44_20000.log",
        ]
        excluded_fixture = {
            "v2": ["v2_a.log", "v2_b.log"],
            "v3": ["v3_a.log"],
            "v4": ["v4_a.log", "v4_b.log", "v4_c.log", "v4_d.log"],
            "v5": ["v5_a.log"],
            "format_sweep": ["format_sweep_gf8_seed42.log", "format_sweep_f32_seed42.log"],
            "other": ["long_gf8_seed42_20k.log", "ultra_f32_seed1_5k.log",
                      "mega_fp8_seed7_1k.log"],
        }
        for name in counted:
            _write(os.path.join(tmp, name), "val_bpb=2.0\nDONE: ok\n")
        for names in excluded_fixture.values():
            for name in names:
                _write(os.path.join(tmp, name), "val_bpb=2.0\nDONE: ok\n")
        # A non-.log file must not enter the denominator.
        _write(os.path.join(tmp, "cpu_train_f32_seed1.json"), "{}\n")

        total_expected = len(counted) + sum(len(v) for v in excluded_fixture.values())

        buf = io.StringIO()
        got = report(tmp, DEFAULT_GLOB, stream=buf)
        printed = buf.getvalue()

        if got[0] != len(counted):
            failures.append(
                "population case: the census counted %d files, expected %d "
                "(the glob must not pick up excluded families)" % (got[0], len(counted))
            )

        if DEFAULT_GLOB not in printed:
            failures.append(
                "POPULATION LINE BROKEN: the report never names the glob it "
                "counted (%r). A count whose scope is not printed is the label "
                "defect this case exists to catch. Output was: %r"
                % (DEFAULT_GLOB, printed)
            )

        denominator_phrase = "%d %s" % (total_expected, DENOMINATOR_GLOB)
        if denominator_phrase not in printed:
            failures.append(
                "POPULATION LINE BROKEN: the report never states the %r "
                "denominator (%r). A reader running `ls *.log | wc -l` would "
                "get %d with no warning. Output was: %r"
                % (DENOMINATOR_GLOB, denominator_phrase, total_expected, printed)
            )

        missing_families = []
        for label, names in excluded_fixture.items():
            term = "%s %d" % (label, len(names))
            if term not in printed:
                missing_families.append(term)
        if missing_families:
            failures.append(
                "POPULATION LINE BROKEN: the report does not enumerate every "
                "excluded family with its count. Missing: %s. Those files exist "
                "in the fixture and are NOT in the count, so a reader is owed "
                "them by name. Output was: %r"
                % (", ".join(sorted(missing_families)), printed)
            )

        # The arithmetic must close, or the enumeration is decorative.
        total, matched, families = decompose(tmp, DEFAULT_GLOB)
        if matched + sum(count for _, count in families) != total:
            failures.append(
                "population decomposition does not close: matched %d + excluded "
                "%r != total %d" % (matched, families, total)
            )
        elif not missing_families:
            ok(
                "report names its glob (%s), the %d %s denominator, and every "
                "excluded family: %s"
                % (
                    DEFAULT_GLOB,
                    total_expected,
                    DENOMINATOR_GLOB,
                    ", ".join("%s %d" % (l, c) for l, c in families if c),
                )
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- case 7: a counted file matching NO shape must REFUSE -------------
    # THE GUARD THIS PASS ADDS. docs/FLEET-COUNT.md justified the truncation
    # behaviour of the whole 8,037-file population by citing ONE template,
    # auto_launch.sh:72,84 -- which is the 7-component shape, 165 files, 2% of
    # the count. Nothing in the script noticed. A shape the table does not
    # carry is a shape whose writer provenance nobody has stated, so the count
    # is refused rather than printed with an explanation that is not about all
    # of it.
    tmp = tempfile.mkdtemp(prefix="fleet_census_shape_")
    try:
        _write(
            os.path.join(tmp, "v6_f32_h256_lr0.001_seed42_20000_adamw.log"),
            "val_bpb=2.0\nDONE: ok\n",
        )
        stray = "v6_f32_h256_lr0.001_seed42_20000_adamw_ctx512_extra.log"
        _write(os.path.join(tmp, stray), "val_bpb=2.0\nDONE: ok\n")

        buf = io.StringIO()
        shape_refused = None
        try:
            report(tmp, DEFAULT_GLOB, stream=buf)
        except UnknownShape as exc:
            shape_refused = str(exc)

        printed = buf.getvalue()
        if shape_refused is None:
            failures.append(
                "SHAPE GUARD BROKEN: %s matches none of the %d shapes in "
                "SHAPES, and the census counted it anyway. Output was: %r"
                % (stray, len(SHAPES), printed)
            )
        elif stray not in shape_refused:
            failures.append(
                "SHAPE GUARD BROKEN: the refusal fired but does not NAME the "
                "offending file %s. A refusal that does not say which file is "
                "unclassified sends the reader to grep 8,037 names. It said: "
                "%r" % (stray, shape_refused)
            )
        elif printed.strip():
            failures.append(
                "SHAPE GUARD BROKEN: nothing may be printed before a refusal, "
                "or a reader sees a partial report and takes it for a whole "
                "one. Output was: %r" % printed
            )
        else:
            rc = main(["--results-dir", tmp], stream=io.StringIO())
            if rc == 0:
                failures.append(
                    "SHAPE GUARD BROKEN: main() exited 0 for a population "
                    "holding an unclassified file (%s)" % stray
                )
            else:
                ok(
                    "unclassified filename refused, names the file, exit code "
                    "%d" % rc
                )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- case 8: the shape block must be PRINTED, with writer provenance ---
    # Case 7 proves the classifier refuses. This proves the report SAYS what it
    # classified. A guard that silently passes tells a reader nothing about the
    # 96% of the population whose writer is not committed anywhere; that fact
    # has to reach the page, not just the exit code.
    tmp = tempfile.mkdtemp(prefix="fleet_census_shapeprint_")
    try:
        shape_fixture = {
            "7 components, opt-terminated": [
                "v6_f32_h256_lr0.001_seed42_20000_adamw.log",
                "v6_gf8_h1024_lr0.0001_seed48_3000000_muon-cwd.log",
            ],
            "6 components": [
                "v6_f32_h256_lr0.0003_seed43_10000000000.log",
                "v6_bf16_h384_lr0.003_seed45_50000.log",
                "v6_gf4_h768_lr0.001_seed46_20000000.log",
            ],
            "4 components": ["v6_bf16_seed50_20000.log"],
        }
        for names in shape_fixture.values():
            for name in names:
                _write(os.path.join(tmp, name), "val_bpb=2.0\nDONE: ok\n")

        buf = io.StringIO()
        got = report(tmp, DEFAULT_GLOB, stream=buf)
        printed = buf.getvalue()

        expected_total = sum(len(v) for v in shape_fixture.values())
        missing_shapes = []
        for label, names in shape_fixture.items():
            term = "%-30s %6d" % (label, len(names))
            if term not in printed:
                missing_shapes.append("%s %d" % (label, len(names)))
        if got[0] != expected_total:
            failures.append(
                "shape print case: the census counted %d files, expected %d"
                % (got[0], expected_total)
            )
        elif missing_shapes:
            failures.append(
                "SHAPE BLOCK BROKEN: the report does not name every shape with "
                "its count. Missing: %s. Output was: %r"
                % (", ".join(missing_shapes), printed)
            )
        elif "MEASURED" not in printed or "ASSUMED" not in printed:
            failures.append(
                "SHAPE BLOCK BROKEN: the report prints shape counts without "
                "saying, per shape, whether a writer for it is COMMITTED "
                "(truncation MEASURED) or not (ASSUMED). That distinction is "
                "the whole point of the block: 7,742 of the 8,037 real files "
                "match no committed template. Output was: %r" % printed
            )
        else:
            ok(
                "shape block printed: %s; and marks MEASURED vs ASSUMED "
                "per shape"
                % ", ".join(
                    "%s %d" % (label, len(names))
                    for label, names in shape_fixture.items()
                )
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- refusal guard, exercised for real -------------------------------
    missing = os.path.join(tempfile.gettempdir(), "fleet_census_definitely_absent")
    if os.path.exists(missing):
        shutil.rmtree(missing, ignore_errors=True)

    buf = io.StringIO()
    refused = False
    try:
        report(missing, DEFAULT_GLOB, stream=buf)
    except CensusRefusal as exc:
        refused = True
        ok("refusal raised: %s" % exc)

    printed = buf.getvalue()
    if not refused:
        failures.append(
            "REFUSAL GUARD BROKEN: a missing directory must not produce a "
            "successful count. %s does not exist, yet the census returned "
            "instead of refusing. It printed: %r" % (missing, printed)
        )
    if printed.strip():
        failures.append(
            "REFUSAL GUARD BROKEN: a missing directory must not produce a "
            "successful count, and nothing may be printed before the refusal. "
            "Output was: %r" % printed
        )

    # And the same path through main(), to prove the EXIT CODE is nonzero.
    rc = main(["--results-dir", missing], stream=io.StringIO())
    if rc == 0:
        failures.append(
            "REFUSAL GUARD BROKEN: a missing directory must not produce a "
            "successful count; main() exited 0 for %s" % missing
        )
    else:
        ok("refusal exit code is nonzero: %d" % rc)

    # An empty-but-present directory must refuse too.
    empty = tempfile.mkdtemp(prefix="fleet_census_empty_")
    try:
        rc = main(["--results-dir", empty], stream=io.StringIO())
        if rc == 0:
            failures.append(
                "REFUSAL GUARD BROKEN: an empty results directory must not "
                "produce a successful '0 logs' count; main() exited 0 for %s" % empty
            )
        else:
            ok("empty directory refused, exit code %d" % rc)
    finally:
        shutil.rmtree(empty, ignore_errors=True)

    if failures:
        for line in failures:
            print("[self-test] FAIL %s" % line, file=sys.stderr)
        print(
            "[self-test] %d/%d checks passed, %d FAILED"
            % (len(passed), SELF_TEST_CASES, len(failures)),
            file=sys.stderr,
        )
        return 1
    if len(passed) != SELF_TEST_CASES:
        print(
            "[self-test] FAIL only %d of %d cases reported a result; a case that "
            "neither passes nor fails is not evidence" % (len(passed), SELF_TEST_CASES),
            file=sys.stderr,
        )
        return 1
    print("[self-test] %d/%d checks passed" % (len(passed), SELF_TEST_CASES))
    return 0


def main(argv=None, stream=sys.stdout):
    parser = argparse.ArgumentParser(
        description="Re-derive the local sweep census from .trinity/results/."
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="directory holding the sweep logs (default: <repo>/%s)" % DEFAULT_SUBDIR,
    )
    parser.add_argument(
        "--glob", default=DEFAULT_GLOB, help="glob for the logs (default: %s)" % DEFAULT_GLOB
    )
    parser.add_argument("--self-test", action="store_true", help="run the built-in checks")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    results_dir = args.results_dir or os.path.join(repo_root(), DEFAULT_SUBDIR)
    try:
        report(results_dir, args.glob, stream=stream)
    except UnknownShape as exc:
        print("REFUSED (unknown filename shape): %s" % exc, file=sys.stderr)
        print(
            "No count is printed. Either add the new shape to SHAPES in this "
            "file WITH its writer provenance -- which committed template, if "
            "any, produces it -- or the file does not belong in the counted "
            "population.",
            file=sys.stderr,
        )
        return 3
    except CensusRefusal as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        print(
            "No count is printed. A census that reports '0 logs' with a "
            "success exit when its input is gone is indistinguishable from a "
            "real measurement of an empty fleet.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
