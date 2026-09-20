#!/usr/bin/env python3
"""Does the 12000-step headline learning-rate schedule survive an ISA change?

WHY THIS IS A SEPARATE PROBE

`scripts/det_math_isa_probe.py --only lr` asks this question once, as the
scope-limit measurement (N4) attached to a ten-step experiment. It cannot ask
it the other way round: it always builds `lr_schedule_dump` with DEFAULT
features, because at the time it was written `det-math` replaced `exp` and
nothing else, so a det-math LR dump would have been the same dump.

That is no longer true. `train_loop::cosine_lr` now routes its cosine through
`det_math::cos_det` under `--features det-math`, and the interesting
measurement is the PAIR:

    default build     the baseline, and it is the control. 58 of 12000 lines
                      differ, first at step 4041, aarch64 3b25032e vs x86_64
                      3b25032d. If this arm ever stops reporting 58/4041 the
                      experiment has lost its control and nothing below is
                      readable.
    --features det-math   the result. 0 differing lines is the claim; anything
                      else is still a result and is printed in full.

Both arms must be run. A `cos_det` that closed the gap in BOTH builds would
mean something other than `cos_det` moved, and the measurement would not be
about `cos_det` at all.

WHAT IS HELD FIXED

ONE macOS host, ONE working tree, ONE pinned toolchain (`rust-toolchain.toml`),
one `cargo build --release --locked` per arm. The `--target` triple is the only
variable that moves. The x86_64 arm runs under `arch -x86_64`, which is
**Rosetta 2 binary translation, not native x86_64 silicon** - that caveat is
part of every number this script prints and may not be dropped when they are
quoted.

Unlike the training probes this one touches no checkpoint, no database and no
corpus: `src/bin/lr_schedule_dump.rs` reads nothing, takes no arguments and
writes 12000 lines to stdout. There is therefore no environment to scrub, and
saying so is cheaper than pretending to scrub one.

WHY BITS

The dump prints the u32 bit pattern of each f32, not a decimal rendering. The
question is byte-identity, and a decimal rendering rounds two different floats
onto one string.

Exit codes:
    0  a verdict was reached, whatever it was. A MISMATCH IS A RESULT.
    1  an artifact could not be produced: a build failed or a dump did not run
    2  usage error, or a precondition is not met - target not installed,
       Rosetta absent, or a binary changed underneath the probe
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HOST_TARGET = "aarch64-apple-darwin"
GUEST_TARGET = "x86_64-apple-darwin"
BINARY = "lr_schedule_dump"

# Both arms are built with an EXPLICIT --target, including the one that happens
# to be the host. `cargo build` without --target and `cargo build --target
# <host triple>` are not the same invocation and do not write to the same path;
# making both explicit is what keeps the triple the only moving variable.
TARGETS = (HOST_TARGET, GUEST_TARGET)
LAUNCHER = {HOST_TARGET: [], GUEST_TARGET: ["/usr/bin/arch", "-x86_64"]}
TRANSLATION = {
    HOST_TARGET: "native execution",
    GUEST_TARGET: "Rosetta 2 binary translation, not native x86_64 silicon",
}

# The published default-feature baseline, re-derived three times including once
# in standalone C outside this repository. `--expect-baseline` asserts it.
BASELINE_DIFFERING = 58
BASELINE_FIRST_STEP = 4041
BASELINE_FIRST_BITS = {HOST_TARGET: "3b25032e", GUEST_TARGET: "3b25032d"}

# `cargo build --bin X` and `cargo build --bin X --features det-math` write to
# the SAME path for a given target, so a probe that ran from that path would be
# comparing whichever build happened last. Every binary is copied out to its
# own name the moment it is built, hashed there, and RUN from there.
STASH = os.path.join("target", "lr-schedule-probe")

RULE = "=" * 74


class Precondition(Exception):
    """A condition the experiment needs, which is not met. Exit 2."""


class NoArtifact(Exception):
    """A build or run did not produce what it was supposed to. Exit 1."""


def say(line=""):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(cmd, label=None):
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - started
    if label:
        status = "ok" if proc.returncode == 0 else "rc=%d" % proc.returncode
        say("  %-52s %6.1fs  %s" % (label, elapsed, status))
    return proc.returncode, proc.stdout


def check_preconditions():
    say("PRECONDITIONS")
    if sys.platform != "darwin":
        raise Precondition(
            "this probe holds the OS fixed by running both arms on ONE macOS "
            "host under Rosetta 2; on %s there is no such pair to build"
            % sys.platform)

    rc, out = run(["rustup", "target", "list", "--installed"])
    if rc != 0:
        raise Precondition("rustup is not available: %s" % out.strip()[:200])
    installed = out.split()
    for target in TARGETS:
        if target not in installed:
            raise Precondition(
                "target %s is not installed. Install it with:\n"
                "    rustup target add %s" % (target, target))
    say("  targets installed              %s" % ", ".join(TARGETS))

    if not os.path.exists("/usr/bin/arch"):
        raise Precondition("/usr/bin/arch is missing; cannot select an ISA")
    rc, _ = run(["/usr/bin/arch", "-x86_64", "/usr/bin/true"])
    if rc != 0:
        raise Precondition(
            "Rosetta 2 cannot execute x86_64 binaries on this host "
            "(`arch -x86_64 /usr/bin/true` failed)")
    say("  rosetta 2                      executes x86_64")

    rc, out = run(["rustc", "--version"])
    rustc = out.strip() if rc == 0 else "unavailable"
    say("  rustc                          %s" % rustc)
    say("  host                           %s" % os.uname().machine)
    return rustc


def stash(rel, name):
    os.makedirs(os.path.join(REPO_ROOT, STASH), exist_ok=True)
    dest = os.path.join(STASH, name)
    shutil.copy2(os.path.join(REPO_ROOT, rel), os.path.join(REPO_ROOT, dest))
    return dest


def build_arm(target, det_math):
    """Build `lr_schedule_dump` for one target. Returns (path, sha256, file)."""
    cmd = ["cargo", "build", "--release", "--locked",
           "--target", target, "--bin", BINARY]
    if det_math:
        cmd += ["--features", "det-math"]
    rc, out = run(cmd, label=" ".join(cmd[1:]))
    if rc != 0:
        say(out[-4000:])
        raise NoArtifact("the %s build failed" % target)

    rel = "target/%s/release/%s" % (target, BINARY)
    if not os.path.isfile(os.path.join(REPO_ROOT, rel)):
        raise NoArtifact(
            "the %s build reported success but %s does not exist" % (target, rel))
    rel = stash(rel, "%s-%s-%s" % (BINARY, target,
                                   "detmath" if det_math else "default"))
    full = os.path.join(REPO_ROOT, rel)
    frc, fout = run(["file", "-b", full])
    return rel, sha256_file(full), (fout.strip() if frc == 0 else "unavailable")


def dump_arm(target, binary_rel):
    rc, out = run(LAUNCHER[target] + [os.path.join(REPO_ROOT, binary_rel)],
                  label="%s %s" % (BINARY, target))
    if rc != 0:
        say(out[-2000:])
        raise NoArtifact("%s failed on %s" % (BINARY, target))
    return out


def measure(det_math, out_dir):
    """Build, dump and diff both arms for one feature state."""
    arm = "det-math" if det_math else "default"
    say("ARM: %s" % ("--features det-math" if det_math
                     else "default features [the control]"))

    binaries, dumps = {}, {}
    for target in TARGETS:
        rel, sha, desc = build_arm(target, det_math)
        binaries[target] = {"path": rel, "sha256": sha, "file": desc}
    for target in TARGETS:
        dumps[target] = dump_arm(target, binaries[target]["path"])

    # The two BINARY hashes differ and that carries no information: different
    # machine code for a different instruction set cannot hash the same. Only
    # the DUMP was ever claimed to be portable.
    for target in TARGETS:
        now = sha256_file(os.path.join(REPO_ROOT, binaries[target]["path"]))
        if now != binaries[target]["sha256"]:
            raise Precondition(
                "the %s binary changed while the probe was running (%s -> %s). "
                "Re-run on a quiet tree."
                % (target, binaries[target]["sha256"], now))

    lines = {}
    for target in TARGETS:
        lines[target] = [l for l in dumps[target].splitlines()
                         if l.startswith("LR ")]
        path = os.path.join(out_dir, "lr-schedule-%s-%s.txt" % (arm, target))
        with open(path, "w") as fh:
            fh.write(dumps[target])
        binaries[target]["dump_path"] = os.path.relpath(path, REPO_ROOT)
        binaries[target]["dump_sha256"] = sha256_text(dumps[target])
        say("  %-28s binary %s" % (target, binaries[target]["sha256"][:16]))
        say("  %-28s dump   %s  -> %s"
            % ("", binaries[target]["dump_sha256"][:16],
               binaries[target]["dump_path"]))

    a, b = lines[HOST_TARGET], lines[GUEST_TARGET]
    if len(a) != len(b):
        raise Precondition("the two dumps have different lengths (%d vs %d)"
                           % (len(a), len(b)))
    if not a:
        raise NoArtifact("the dumps carry no LR lines at all")

    diffs = [(x, y) for x, y in zip(a, b) if x != y]
    diff_path = os.path.join(out_dir, "lr-schedule-%s-diff.txt" % arm)
    with open(diff_path, "w") as fh:
        fh.write("# %s vs %s (Rosetta 2)\n" % (HOST_TARGET, GUEST_TARGET))
        fh.write("# feature state: %s\n" % arm)
        fh.write("# lines compared: %d\n" % len(a))
        fh.write("# lines differing: %d\n" % len(diffs))
        for x, y in diffs:
            fh.write("- %s\n+ %s\n" % (x, y))

    first = diffs[0] if diffs else None
    say()
    say(RULE)
    say(" RESULT - %s" % arm)
    say(RULE)
    say(" steps compared         %d" % len(a))
    say(" lr values differing    %d of %d" % (len(diffs), len(a)))
    if first:
        say(" first differing step   %s" % first[0].split()[1])
        say("   %-8s %s   bits %s" % ("aarch64", first[0], first[0].split()[2]))
        say("   %-8s %s   bits %s" % ("x86_64", first[1], first[1].split()[2]))
    else:
        say(" first differing step   none: the schedule is byte-identical")
        say("                        across the ISA change at every step")
    say(RULE)
    say()

    return {
        "arm": arm,
        "det_math": det_math,
        "steps_compared": len(a),
        "differing": len(diffs),
        "first_differing_step": int(first[0].split()[1]) if first else None,
        "first_differing_bits": {
            HOST_TARGET: first[0].split()[2],
            GUEST_TARGET: first[1].split()[2],
        } if first else None,
        "binaries": binaries,
        "diff_path": os.path.relpath(diff_path, REPO_ROOT),
        "translation": TRANSLATION,
    }


def check_baseline(result):
    """The control. If this moved, nothing else in the run is readable."""
    problems = []
    if result["differing"] != BASELINE_DIFFERING:
        problems.append("expected %d differing lines, got %d"
                        % (BASELINE_DIFFERING, result["differing"]))
    if result["first_differing_step"] != BASELINE_FIRST_STEP:
        problems.append("expected first difference at step %d, got %s"
                        % (BASELINE_FIRST_STEP, result["first_differing_step"]))
    got_bits = result["first_differing_bits"] or {}
    for target, want in BASELINE_FIRST_BITS.items():
        if got_bits.get(target) != want:
            problems.append("expected %s bits %s at the first difference, got %s"
                            % (target, want, got_bits.get(target)))
    if problems:
        raise Precondition(
            "THE BASELINE MOVED. The default-feature arm is the control for "
            "this experiment and it no longer reproduces the published "
            "measurement:\n    %s\nReport the discrepancy; do not read the "
            "det-math arm until it is explained." % "\n    ".join(problems))
    say("BASELINE HOLDS: %d differing, first at step %d, %s vs %s."
        % (BASELINE_DIFFERING, BASELINE_FIRST_STEP,
           BASELINE_FIRST_BITS[HOST_TARGET], BASELINE_FIRST_BITS[GUEST_TARGET]))
    say()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Diff the 12000-step headline learning-rate schedule "
                    "across an ISA change, with and without --features "
                    "det-math.")
    parser.add_argument("--features", default="", metavar="LIST",
                        help="cargo features for the dump build. Only "
                             "'det-math' is meaningful here; empty means the "
                             "default build, which is the control.")
    parser.add_argument("--both", action="store_true",
                        help="run the default arm and the det-math arm in one "
                             "go and compare them. The default arm is checked "
                             "against the published baseline first.")
    parser.add_argument("--expect-baseline", action="store_true",
                        help="fail unless the default arm reproduces 58/12000 "
                             "first-diff 4041. Implied by --both.")
    parser.add_argument("--out-dir", metavar="DIR",
                        default=os.path.join(REPO_ROOT, "evidence", "cos-det-isa"),
                        help="where the dumps and diffs are written")
    parser.add_argument("--json", metavar="PATH", default=None,
                        help="also write the result record to PATH as JSON")
    args = parser.parse_args(argv)

    if args.features not in ("", "det-math"):
        raise Precondition(
            "unknown feature list %r; this probe understands '' and 'det-math'"
            % args.features)

    say(RULE)
    say(" LR SCHEDULE ISA PROBE - does the headline schedule cross the ISA?")
    say(RULE)
    rustc = check_preconditions()
    os.makedirs(args.out_dir, exist_ok=True)
    say("  out dir                        %s"
        % os.path.relpath(args.out_dir, REPO_ROOT))
    say()

    arms = [False, True] if args.both else [args.features == "det-math"]
    results = []
    for det_math in arms:
        result = measure(det_math, args.out_dir)
        if not det_math and (args.both or args.expect_baseline):
            check_baseline(result)
        results.append(result)

    if len(results) == 2:
        before, after = results[0]["differing"], results[1]["differing"]
        say(RULE)
        say(" BOTH ARMS")
        say(RULE)
        say(" default   %d of %d differing" % (before, results[0]["steps_compared"]))
        say(" det-math  %d of %d differing" % (after, results[1]["steps_compared"]))
        if after == 0 and before > 0:
            say(" cos_det CLOSES the schedule's cross-ISA gap, and the default")
            say(" arm proves it was still open in the same tree on the same")
            say(" host minutes earlier - so the feature is what closed it.")
        elif after == before:
            say(" NOTHING MOVED. Either the feature did not reach cosine_lr or")
            say(" the divergence is not in cos. Check that the det-math binary")
            say(" is not the default one under another name.")
        else:
            say(" PARTIAL: the gap narrowed but did not close. That is a")
            say(" result. The first surviving difference is named above; some")
            say(" operation reachable from cos_det is still platform-dependent.")
        say(RULE)
        say()

    say("SCOPE, which may not be dropped when these numbers are quoted:")
    say("  ONE macOS host, ONE pinned toolchain, ONE working tree. The x86_64")
    say("  arm ran under %s." % TRANSLATION[GUEST_TARGET])
    say("  This constrains a native x86_64 Linux arm; it does not replace it.")
    say()

    record = {
        "schema": "trios-lr-schedule-isa-probe/1",
        "arms": results,
        "baseline": {"differing": BASELINE_DIFFERING,
                     "first_differing_step": BASELINE_FIRST_STEP,
                     "first_differing_bits": BASELINE_FIRST_BITS},
        "rustc": rustc,
        "host": " ".join(os.uname()),
        "scope": TRANSLATION[GUEST_TARGET],
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
            fh.write("\n")
        say("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    if shutil.which("cargo") is None:
        sys.stderr.write("cargo is not on PATH\n")
        sys.exit(2)
    try:
        sys.exit(main())
    except Precondition as exc:
        sys.stderr.write("\nPRECONDITION NOT MET: %s\n" % exc)
        sys.exit(2)
    except NoArtifact as exc:
        sys.stderr.write("\nNO ARTIFACT: %s\n" % exc)
        sys.exit(1)
