#!/usr/bin/env python3
"""Reproduce the four-arm 12000-step cross-ISA result from this branch.

WHAT THIS REPRODUCES

`evidence/headline-isa-12000/` publishes four `trios-checkpoint-record/9`
sidecars, one per arm of a 2x2: {default features, `--features det-math`} x
{aarch64-apple-darwin, x86_64-apple-darwin}. Each ran 12000 gradient steps of
one configuration. The published result:

    det-math  aarch64  65f36543...   ==  det-math  x86_64  65f36543...  MATCH
    default   aarch64  8a86fe69...   !=  default   x86_64  cb7b24ca...  DIFFER

and the default aarch64 value is bit-equal to the published headline in
`evidence/r9-headline/12000.json`.

That is the strongest measurement this project owns. The previously published
cross-ISA MATCH reached TEN steps; this reaches twelve thousand. It was
produced by a shell script in /tmp that was never in the repository, so until
this file existed a third party could read the four hashes and had no way to
regenerate them. That is the gap this closes.

WHAT IT DOES NOT PROVE, AND THE LIMIT IS NOT SMALL

The x86_64 arm runs under Rosetta 2 binary translation on an aarch64 macOS
host. NO X86_64 PROCESSOR EXECUTES IT. Rosetta 2 may implement an x86_64
floating-point operation with any aarch64 sequence that yields the architected
result; nothing here measured whether it matches what Intel or AMD silicon
does. A MATCH from this script is a statement about a translated arm and must
be re-run on native x86_64 Linux before anyone calls it ISA portability.
`evidence/headline-isa-12000/PROVENANCE.txt` section 5 says the same thing at
greater length, and neither statement is decoration.

The published records also carry `git_dirty: true`. They were produced from a
working tree modified relative to e9eec901. This script prints the git state of
the tree it runs in so a reader can see whether their reproduction attempt is
being made under the same conditions or different ones -- rather than finding
out after twelve thousand steps that it was never going to match.

STRUCTURE

Reused verbatim in shape from `scripts/local_isa_probe.py` (do NOT edit that
file; another agent's finding lives in its comments) and
`scripts/det_math_isa_probe.py`: the same two target triples, the same
`/usr/bin/arch -x86_64` launcher, the same `env -i` allowlist proved empty of
every DSN alias by inspecting what a child actually receives, the same corpus
check against `data/MANIFEST.sha256`, and the same stash-then-run discipline
(`cargo build --bin X` and `cargo build --bin X --features det-math` write to
the SAME path, so a probe that ran from `target/release/` would compare
whichever build happened last).

COST

Four 12000-step runs. On the host that produced the published records the four
ran concurrently and took about fifteen minutes wall-clock. Do not run this
under a blanket timeout shared with other work: a kill mid-run leaves partial
checkpoints behind.

`--steps N` shortens the runs for a smoke test of this script itself. It is
NOT the headline experiment and the script says so in its own output and
refuses to compare against the published hashes.

EXIT CODES

    0  the det-math pair MATCHED
    3  the det-math pair DIFFERED. That is a result, not a crash -- but it
       contradicts the published record, so it must not be mistaken for
       success by a caller that only checks for zero.
    1  an artifact could not be produced: a build failed, a run failed, or a
       checkpoint a run was supposed to write is not on disk
    2  usage error, or a precondition is not met -- target not installed,
       Rosetta absent, corpus not matching data/MANIFEST.sha256
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

# From data/MANIFEST.sha256. The same two hashes the CI jobs check.
EXPECTED_TRAIN_SHA = "1a5aead1db78653f48ee799c4145ef71265f6aadd2c79ebffc9f0260cac1fb0d"
EXPECTED_VAL_SHA = "2088af36b1c7831083ef22c0f6e1999b1dece15b9fbe2d4695364e95d497d502"
TRAIN_DATA = "data/tiny_shakespeare.txt"
VAL_DATA = "data/tiny_shakespeare_val.txt"

DSN_ALIASES = (
    "DATABASE_URL",
    "NEON_DATABASE_URL",
    "TRIOS_NEON_DSN",
    "TRIOS_DATABASE_URL",
)

HEADLINE_STEPS = 12000

# The published four, from evidence/headline-isa-12000/. Compared against only
# when --steps is the headline value; at any other step count the comparison
# would be meaningless and is skipped rather than quietly passed.
PUBLISHED = {
    "dm-arm64": "65f36543520c5b88cd419ae458fe965d15b11d5d620448b8dc2188b9087dbd45",
    "dm-x86_64": "65f36543520c5b88cd419ae458fe965d15b11d5d620448b8dc2188b9087dbd45",
    "df-arm64": "8a86fe691aef64fcb637b90d4cf62650c217c3b000b6252846a8ab70c186012c",
    "df-x86_64": "cb7b24cad35b80e4f1b6f54fd724d0aa1d19d5a6610523ac14aae0e0d7904b34",
}

EVIDENCE_DIR = os.path.join("evidence", "headline-isa-12000")

# Byte-identical in content to the flag list the published runs used. The two
# data flags are appended by build_flags() so --steps can vary without the rest
# of the configuration drifting with it.
BASE_FLAGS = [
    "--seed", "47",
    "--hidden", "384",
    "--attn-layers", "2",
    "--eval-every", "1000",
    "--lr", "0.003",
    "--optimizer", "adamw",
]

# The four arms: (label, target, is_host, det_math, canon_name).
ARMS = (
    ("dm-arm64", HOST_TARGET, True, True, "h12k-dm-arm"),
    ("dm-x86_64", GUEST_TARGET, False, True, "h12k-dm-x86"),
    ("df-arm64", HOST_TARGET, True, False, "h12k-df-arm"),
    ("df-x86_64", GUEST_TARGET, False, False, "h12k-df-x86"),
)

STASH = os.path.join("target", "headline-isa-probe")

TRANSLATION_NOTE = (
    "the x86_64 arm ran under Rosetta 2 binary translation, "
    "NOT on native x86_64 silicon")

VERDICT_MATCH = "DET_MATH_PAIR_BYTE_IDENTICAL_AT_%d_STEPS"
VERDICT_DIFFER = "DET_MATH_PAIR_DIFFERS_AT_%d_STEPS"

RULE = "=" * 74


class Precondition(Exception):
    """A condition the experiment needs, which is not met. Exit 2."""


class NoArtifact(Exception):
    """A build or run did not produce the file it was supposed to. Exit 1."""


def say(line=""):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd, env=None, label=None):
    """Run a command, returning (rc, stdout+stderr, elapsed seconds)."""
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - started
    if label:
        status = "ok" if proc.returncode == 0 else "rc=%d" % proc.returncode
        say("  %-46s %7.1fs  %s" % (label, elapsed, status))
    return proc.returncode, proc.stdout, elapsed


def build_flags(steps):
    return BASE_FLAGS + [
        "--steps", str(steps),
        "--train-data", TRAIN_DATA,
        "--val-data", VAL_DATA,
    ]


def scrubbed_env(canon_name, run_dir):
    """`env -i` with an allowlist, expressed as a dict."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "TRINITY_AUTOMIGRATE": "0",
        "TRIOS_CANON_NAME": canon_name,
        "TRIOS_CHECKPOINT_INIT": "1",
        "TRIOS_CHECKPOINT_DIR": run_dir,
    }


def prove_env_carries_no_dsn(run_dir):
    """Prove the child environment is empty of every DSN alias.

    By inspecting what a child actually receives, not by asserting what this
    process intends to pass. The two are not the same claim.
    """
    env = scrubbed_env("headline-envcheck", run_dir)
    rc, out, _ = run([sys.executable, "-c",
                      "import os,sys\n"
                      "sys.stdout.write('\\n'.join('%s=%s' % kv "
                      "for kv in sorted(os.environ.items())))\n"], env=env)
    if rc != 0:
        raise Precondition("could not inspect the child environment")
    seen = out.splitlines()
    for alias in DSN_ALIASES:
        for line in seen:
            if line.startswith(alias + "="):
                raise Precondition(
                    "%s survives into the training environment" % alias)
    if "TRINITY_AUTOMIGRATE=0" not in seen:
        raise Precondition(
            "TRINITY_AUTOMIGRATE is not pinned to 0; automigrate defaults to ON")
    inherited = [a for a in DSN_ALIASES if a in os.environ]
    say("  no DSN alias reaches the trainer; automigrate is off")
    say("  (this shell has %s set; the scrub is what removes %s)"
        % (", ".join(inherited) if inherited else "no DSN alias",
           "them" if len(inherited) != 1 else "it"))


def git_state():
    """What a reader needs to judge whether their tree is the published one."""
    rc, out, _ = run(["git", "rev-parse", "HEAD"])
    head = out.strip() if rc == 0 else "unavailable"
    rc, out, _ = run(["git", "status", "--porcelain"])
    if rc != 0:
        return head, "unavailable", "unavailable"
    lines = [ln for ln in out.splitlines() if ln.strip()]
    dirty = sum(1 for ln in lines if not ln.startswith("??"))
    untracked = sum(1 for ln in lines if ln.startswith("??"))
    return head, dirty, untracked


def check_preconditions():
    say("PRECONDITIONS")

    if sys.platform != "darwin":
        raise Precondition(
            "this probe holds the OS fixed by running both arms on ONE macOS "
            "host under Rosetta 2; on %s there is no such pair to build. The "
            "native x86_64 re-run this result needs is a DIFFERENT experiment "
            "and this script is not it." % sys.platform)

    rc, out, _ = run(["rustup", "target", "list", "--installed"])
    if rc != 0:
        raise Precondition("rustup is not available: %s" % out.strip()[:200])
    installed = out.split()
    for target in (HOST_TARGET, GUEST_TARGET):
        if target not in installed:
            raise Precondition(
                "target %s is not installed. Install it with:\n"
                "    rustup target add %s" % (target, target))
    say("  targets installed              %s, %s" % (HOST_TARGET, GUEST_TARGET))

    if not os.path.exists("/usr/bin/arch"):
        raise Precondition("/usr/bin/arch is missing; cannot select an ISA")
    rc, _, _ = run(["/usr/bin/arch", "-x86_64", "/usr/bin/true"])
    if rc != 0:
        raise Precondition(
            "Rosetta 2 cannot execute x86_64 binaries on this host "
            "(`arch -x86_64 /usr/bin/true` failed)")
    say("  rosetta 2                      executes x86_64")

    for path, expected, name in (
        (TRAIN_DATA, EXPECTED_TRAIN_SHA, "train"),
        (VAL_DATA, EXPECTED_VAL_SHA, "val"),
    ):
        full = os.path.join(REPO_ROOT, path)
        if not os.path.isfile(full):
            raise Precondition(
                "%s is missing. The corpus is not in git; rebuild it as "
                ".github/workflows/cross-arch-repro.yml does." % path)
        got = sha256_file(full)
        if got != expected:
            raise Precondition(
                "%s corpus differs from data/MANIFEST.sha256\n"
                "    expected %s\n    got      %s" % (name, expected, got))
        say("  corpus %-23s %s" % (name, expected))

    rc, out, _ = run(["rustc", "--version"])
    rustc = out.strip() if rc == 0 else "unavailable"
    say("  rustc                          %s" % rustc)
    say("  host                           %s" % os.uname().machine)

    head, dirty, untracked = git_state()
    say("  git HEAD                       %s" % head)
    say("  git working tree               %s modified, %s untracked"
        % (dirty, untracked))
    say("  (the published records carry git_dirty: true at e9eec901; a tree "
        "in a\n   different state may not reproduce their bytes)")
    return rustc


def stash(rel, name):
    """Copy a freshly built binary out of a path a later cargo run can rewrite.

    `cargo build --bin X` and `cargo build --bin X --features det-math` write
    to the SAME path. Running from that path would compare whichever build
    happened last. Every binary is copied to its own name, hashed there, and
    run from there.
    """
    os.makedirs(os.path.join(REPO_ROOT, STASH), exist_ok=True)
    dest = os.path.join(STASH, name)
    shutil.copy2(os.path.join(REPO_ROOT, rel), os.path.join(REPO_ROOT, dest))
    return dest


def build_arm(label, target, is_host, det_math, binary="trios-train"):
    """Build one arm. Returns (relative path, sha256, `file` description)."""
    cmd = ["cargo", "build", "--release", "--locked", "--bin", binary]
    if det_math:
        cmd += ["--features", "det-math"]
    if not is_host:
        cmd[2:2] = ["--target", target]
    build_label = "cargo build %s %s%s" % (
        binary, "(host)" if is_host else "--target " + target,
        " --features det-math" if det_math else " [default features]")
    rc, out, _ = run(cmd, label=build_label)
    if rc != 0:
        say(out[-4000:])
        raise NoArtifact("the %s build failed" % label)

    rel = ("target/release/%s" % binary if is_host
           else "target/%s/release/%s" % (target, binary))
    full = os.path.join(REPO_ROOT, rel)
    if not os.path.isfile(full):
        raise NoArtifact("the %s build reported success but %s does not exist"
                         % (label, rel))
    rel = stash(rel, "%s-%s" % (binary, label))
    full = os.path.join(REPO_ROOT, rel)
    frc, fout, _ = run(["file", "-b", full])
    return rel, sha256_file(full), (fout.strip() if frc == 0 else "unavailable")


def train(binary_rel, canon_name, is_host, run_dir, steps):
    """Run one arm once. Returns the relative path of the final checkpoint."""
    cmd = [] if is_host else ["/usr/bin/arch", "-x86_64"]
    cmd += [os.path.join(REPO_ROOT, binary_rel)] + build_flags(steps)
    rc, out, elapsed = run(cmd, env=scrubbed_env(canon_name, run_dir),
                           label="train %s (%d steps)" % (canon_name, steps))
    if rc != 0:
        say(out[-4000:])
        raise NoArtifact("the run for %s exited %d" % (canon_name, rc))

    rel = os.path.join(run_dir, canon_name, "%d.bin" % steps)
    if not os.path.isfile(os.path.join(REPO_ROOT, rel)):
        raise NoArtifact("no step-%d artifact at %s" % (steps, rel))
    return rel, elapsed


def declared(bin_rel, *keys):
    """Read a dotted field out of the sidecar the trainer wrote."""
    side = os.path.join(REPO_ROOT, bin_rel[:-4] + ".json")
    if not os.path.isfile(side):
        return None
    try:
        with open(side, "r") as fh:
            node = json.load(fh)
    except (ValueError, OSError):
        return None
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def probe(run_dir, steps):
    say("ENVIRONMENT SCRUB")
    prove_env_carries_no_dsn(run_dir)
    say()

    say("BUILD - four arms from ONE tree with ONE compiler")
    binaries = {}
    for label, target, is_host, det_math, _canon in ARMS:
        # The two det-math arms and the two default arms share a build each;
        # only the target and the feature set move.
        key = (target, det_math)
        if key not in binaries:
            binaries[key] = build_arm(label, target, is_host, det_math)
        rel, digest, desc = binaries[key]
        say("  %-12s %s" % (label, digest))
        say("               %s" % desc)
    say()

    say("TRAIN - %d steps per arm, %s" % (steps, TRANSLATION_NOTE))
    results = {}
    for label, target, is_host, det_math, canon in ARMS:
        rel, _digest, _desc = binaries[(target, det_math)]
        ckpt, elapsed = train(rel, canon, is_host, run_dir, steps)
        results[label] = {
            "canon_name": canon,
            "target": target,
            "det_math": det_math,
            "checkpoint": ckpt,
            "sha256": sha256_file(os.path.join(REPO_ROOT, ckpt)),
            "bytes": os.path.getsize(os.path.join(REPO_ROOT, ckpt)),
            "declared_arch": declared(ckpt, "platform", "arch"),
            "declared_features": declared(ckpt, "platform", "features"),
            "final_val_bpb": declared(ckpt, "final_val_bpb"),
            "seconds": round(elapsed, 1),
        }
    say()

    say("THE TWO ARMS DID RUN ON DIFFERENT INSTRUCTION SETS")
    for label, _t, _h, _d, _c in ARMS:
        got = results[label]["declared_arch"]
        want = "aarch64" if results[label]["target"] == HOST_TARGET else "x86_64"
        if got != want:
            raise Precondition(
                "arm %s was built for %s but its checkpoint declares "
                "platform.arch=%r. The two arms did not run on different "
                "instruction sets and no comparison below means anything."
                % (label, results[label]["target"], got))
        say("  %-12s declares platform.arch %s" % (label, got))
    say()

    say(RULE)
    say(" RESULT - four arms, %d steps" % steps)
    say(RULE)
    for label, _t, _h, _d, _c in ARMS:
        r = results[label]
        say("  %-12s %s  (%d bytes, %.1fs)"
            % (label, r["sha256"], r["bytes"], r["seconds"]))
    say()

    dm_match = results["dm-arm64"]["sha256"] == results["dm-x86_64"]["sha256"]
    df_match = results["df-arm64"]["sha256"] == results["df-x86_64"]["sha256"]
    say("  det-math  aarch64 vs x86_64    %s" % ("MATCH" if dm_match else "DIFFER"))
    say("  default   aarch64 vs x86_64    %s" % ("MATCH" if df_match else "DIFFER"))
    say()

    if steps == HEADLINE_STEPS:
        say("AGAINST THE PUBLISHED RECORDS IN %s" % EVIDENCE_DIR)
        agreed = 0
        for label, _t, _h, _d, _c in ARMS:
            got = results[label]["sha256"]
            want = PUBLISHED[label]
            ok = got == want
            agreed += 1 if ok else 0
            say("  %-12s %s" % (label, "REPRODUCED" if ok else "DIFFERS"))
            if not ok:
                say("               published %s" % want)
                say("               this run  %s" % got)
        say("  %d of %d arms reproduced the published bytes" % (agreed, len(ARMS)))
    else:
        say("NOT THE HEADLINE EXPERIMENT")
        say("  --steps %d is not %d. These hashes are a smoke test of this"
            % (steps, HEADLINE_STEPS))
        say("  script and are NOT comparable to the published records; the")
        say("  comparison against %s is skipped rather than" % EVIDENCE_DIR)
        say("  quietly passed.")
    say()

    say("SCOPE")
    say("  %s%s." % (TRANSLATION_NOTE[0].upper(), TRANSLATION_NOTE[1:]))
    say("  A MATCH here is a statement about aarch64-apple-darwin against")
    say("  x86_64-apple-darwin-under-translation on ONE host. It must be")
    say("  re-run on native x86_64 Linux before it is called ISA")
    say("  portability in general.")
    say()
    verdict = (VERDICT_MATCH if dm_match else VERDICT_DIFFER) % steps
    say("VERDICT %s" % verdict)

    return {
        "steps": steps,
        "arms": results,
        "det_math_pair_match": dm_match,
        "default_pair_match": df_match,
        "translation": TRANSLATION_NOTE,
        "verdict": verdict,
    }, dm_match


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce the four-arm 12000-step cross-ISA result. "
                    "Builds both targets from one tree with the pinned "
                    "toolchain, runs the four arms, and prints the four "
                    "hashes with a MATCH/DIFFER verdict per pair.",
        epilog="Exit 0 if the det-math pair matched, 3 if it differed "
               "(a result, but not a success), 1 if an artifact could not be "
               "produced, 2 if a precondition is not met.")
    parser.add_argument("--steps", type=int, default=HEADLINE_STEPS,
                        help="steps per arm. Default %d, the headline. Any "
                             "other value is a smoke test of this script and "
                             "is not compared against the published hashes."
                             % HEADLINE_STEPS)
    parser.add_argument("--run-dir", metavar="DIR", default=None,
                        help="checkpoint base directory, relative to the repo "
                             "root. Defaults to a fresh timestamped directory, "
                             "which is what keeps a re-run from overwriting "
                             "the artifacts of the last one.")
    parser.add_argument("--json", metavar="PATH", default=None,
                        help="also write the result record to PATH as JSON")
    args = parser.parse_args(argv)

    if args.steps < 1:
        parser.error("--steps must be at least 1")

    run_dir = args.run_dir or os.path.join(
        "checkpoints", "headline-isa",
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))

    say(RULE)
    say(" HEADLINE ISA PROBE - four arms, %d steps, one host" % args.steps)
    say(RULE)

    rustc = check_preconditions()
    say("  run directory                  %s" % run_dir)
    say()

    record, dm_match = probe(run_dir, args.steps)
    record["rustc"] = rustc
    record["run_dir"] = run_dir

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
            fh.write("\n")
        say("wrote %s" % args.json)

    return 0 if dm_match else 3


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
