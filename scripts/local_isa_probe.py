#!/usr/bin/env python3
"""Separate the ISA from the OS as the cause of the cross-architecture mismatch.

GitHub Actions run 30767491098 measured that the documented seed-47 12000-step
checkpoint produced on x86_64 Linux is not byte-identical to the one recorded
on aarch64 macOS. That run changed TWO things at once - the instruction set
AND the operating system, and with the OS its libm and its libc - so it can
report THAT the artifacts disagree and cannot attribute the disagreement to
either variable. An experiment that varies two things attributes nothing.

This probe varies ONE. Both binaries are built from a single working tree by a
single pinned compiler (rust-toolchain.toml), differing only in `--target`, and
both run on one physical machine, under one kernel, against one filesystem and
one vendor's libm. The x86_64 arm executes under Rosetta 2. The OS, the libm
vendor and the compiler version are therefore held fixed and the ISA target is
the only thing that moves.

Read the scope limit with the result and do not drop it: Rosetta 2 is binary
translation, not native x86_64 silicon, and the x86_64 slice of Apple's libm is
not glibc's libm. This probe CONSTRAINS the native x86_64 Linux arm; it does
not replace it. See docs/DIVERGENCE-LOCALIZATION.md.

WHAT IT MEASURES

Two artifacts per arm, not one, because "the checkpoints differ" is not yet an
answer:

  0.bin   the weights as initialised, written before the first optimizer step.
          Requires TRIOS_CHECKPOINT_INIT=1; without it the earliest artifact
          any run can write is already one gradient step deep, and the
          init-versus-arithmetic question cannot be asked at all.
  10.bin  the weights after ten optimizer steps.

If 0.bin already differs across the ISA boundary the cause is initialisation
and the seeded RNG, which is outright fixable at no cost in speed. If 0.bin is
identical and 10.bin is not, the cause is the floating-point arithmetic of the
training loop - reduction order or libm - which is fixable only by constraining
the arithmetic, at a cost in speed. The two defects fit the same 12000-step
observation and have different fixes, which is why this distinction is worth a
measurement rather than an argument.

THE CONTROL

Each arm is run TWICE, with byte-identical flags, differing only in
TRIOS_CANON_NAME so the second run cannot overwrite the first. Without it a
cross-ISA mismatch is equally consistent with "the two instruction sets
disagree" and with "the trainer is not repeatable on x86_64 either and nobody
looked" - and the second reading is much the worse one, because it would make
determinism itself platform-contingent. The control is what licenses reading a
cross-ISA mismatch as a cross-ISA effect. If it fails, the verdict line below
is printed with its attribution explicitly voided.

Pass --skip-control to drop it. That makes the run faster and the result
uninterpretable; it exists for debugging this script, not for producing
evidence.

ENVIRONMENT

Both arms run under `env -i` with an ALLOWLIST - PATH, HOME,
TRINITY_AUTOMIGRATE=0, TRIOS_CANON_NAME, TRIOS_CHECKPOINT_INIT - not a
denylist. The DSN resolution chain is DATABASE_URL -> NEON_DATABASE_URL ->
TRIOS_NEON_DSN -> TRIOS_DATABASE_URL, and `TRINITY_AUTOMIGRATE` defaults to
"1" when unset, so an inherited DSN would run migrations against the live SSOT
before training started. `env -i` has nothing to enumerate and cannot fall
behind a fifth alias. The child environment is PROVED empty of all four rather
than asserted to be, before any run is taken.

Dependency-free on purpose: standard library only, in the manner of
scripts/compare_checkpoints.py. Nothing here writes to the repository; the
artifacts land under `checkpoints/`, which is gitignored.

RE-RUNNING

Every invocation writes into its OWN directory,
`checkpoints/isa-probe/<utc timestamp>/<canon name>/`, selected with
TRIOS_CHECKPOINT_DIR. This script therefore never deletes a checkpoint and
never overwrites one, and two invocations - or two agents - cannot collide.

That is not a convenience. `checkpoint::save` refuses to overwrite a sidecar
that would lose information, so a probe that reused one fixed directory would
fail on its second run, and the only ways to make it idempotent would be to
delete the previous evidence or to set TRIOS_ALLOW_SIDECAR_OVERWRITE=1. Both
would have this script destroy a measurement in order to take another one.
Allocating a fresh directory costs nothing and keeps every run's evidence.

VERDICT

Exactly one of these three lines is printed:

    the initial weights carry across the ISA change and the trained ones do
    not  -> the arithmetic is the cause
    both artifacts carry across  -> the ISA is not what run 30767491098 saw,
    and its OS/libm is the remaining candidate
    the initial weights already differ  -> initialisation / RNG is the cause

Exit codes:
    0  a verdict was reached, whatever it was. A MISMATCH IS A RESULT.
    1  an artifact could not be produced: a build failed, a run failed, or a
       checkpoint the run was supposed to write is not on disk
    2  usage error, or a precondition of the experiment is not met (target not
       installed, Rosetta absent, corpus not matching data/MANIFEST.sha256,
       the two arms did not actually run on different instruction sets, or a
       binary changed underneath the probe)
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

# The DSN aliases, in resolution order (src/neon_writer.rs, src/bin/trios-train.rs).
DSN_ALIASES = (
    "DATABASE_URL",
    "NEON_DATABASE_URL",
    "TRIOS_NEON_DSN",
    "TRIOS_DATABASE_URL",
)

# Byte-identical across all four runs. --attn-layers, --eval-every, --optimizer
# and the corpus paths are CLI defaults and are passed explicitly anyway, so
# that this invocation is textually the same experiment as the CI job's and a
# future change of default cannot silently move it.
#
# --eval-every is NOT a pure observation parameter on this trainer: gf16_floor
# mutates weights in place on an eval-gated schedule late in training. At 10
# steps that gate never fires, so it is not a confound here, but runs with a
# different eval cadence are not comparable to these.
TRAIN_FLAGS = [
    "--seed", "47",
    "--steps", "10",
    "--hidden", "384",
    "--attn-layers", "2",
    "--eval-every", "1000",
    "--lr", "0.003",
    "--optimizer", "adamw",
    "--train-data", TRAIN_DATA,
    "--val-data", VAL_DATA,
]

INIT_STEP = "0"
TRAINED_STEP = "10"

# Until 2026-08-05 this label read ONLY where it now reads SUFFICIENT, and it
# claimed an isolation this repository's own data denies. THREE step-10 hashes
# exist for the same declared inputs (seed 47, 10 steps, one corpus, one pinned
# rustc):
#
#   efef1cba  aarch64  macOS               (this probe, native)
#   5913542e  x86_64   macOS, Rosetta 2    (this probe, translated)
#   a32e9b2a  x86_64   Linux, glibc 2.39   (CI run 31004703001)
#
# The last two share a declared arch and differ anyway, so at FIXED architecture
# a change of OS and libc also moves the bytes. ONLY was therefore false as a
# machine-readable label even while the prose beside it said the right thing -
# and the label is what a machine reads. SUFFICIENT is what the experiment
# supports: varying the ISA alone is enough to diverge, which is not a claim
# that nothing else is.
#
# The retracted spelling is described here and nowhere written out, because it
# was never persisted: `git log --all -S` finds it in no commit (this script and
# its evidence are themselves uncommitted), so no artifact anywhere carries it
# and nobody can be holding one to grep for. Retractions are preserved in this
# repository; dead strings that would only match themselves are not.
VERDICT_ISA = "ISA_SUFFICIENT_TO_DIVERGE"
VERDICT_NOT_ISA = "ISA_IDENTICAL_SO_CAUSE_IS_OS_OR_LIBM"
VERDICT_RNG = "INIT_ALSO_DIVERGES_CAUSE_IS_RNG"

# The x86_64 step-10 hash measured on native x86_64 Linux by the
# `localize-divergence` job, CI run 31004703001, 2026-08-05. Quoted in the
# verdict reading so the third point is in the OUTPUT of the probe and not only
# in the documentation a reader may not have open.
LINUX_X86_STEP10 = "a32e9b2ab0043b91aaaae96f3e6945a419ef305e375fa2aaf2ba38a6deba19d1"

# Per-arm translation disclosure, recorded ON each arm rather than only once at
# the top level. A machine consumer slices `declared_platform["isa-probe-x86_64"]`
# out of the record, reads `arch: x86_64, os: macos`, and has no way to know it
# is not native silicon unless the arm says so itself.
TRANSLATION_NATIVE = "native execution"
TRANSLATION_ROSETTA = "Rosetta 2 binary translation, not native x86_64 silicon"

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
    """Run a command, streaming nothing, returning (rc, stdout+stderr)."""
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
        say("  %-34s %6.1fs  %s" % (label, elapsed, status))
    return proc.returncode, proc.stdout


def scrubbed_env(canon_name, run_dir):
    """`env -i` with an allowlist, expressed as a dict.

    subprocess with env=<dict> replaces the environment wholesale, which is
    exactly what `env -i` does. Nothing not named here reaches the trainer.
    """
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

    Proved by inspecting what a child actually receives, not by asserting what
    this process intends to pass. The two are not the same claim.
    """
    env = scrubbed_env("isa-probe-envcheck", run_dir)
    rc, out = run([sys.executable, "-c",
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
    for target in (HOST_TARGET, GUEST_TARGET):
        if target not in installed:
            raise Precondition(
                "target %s is not installed. Install it with:\n"
                "    rustup target add %s" % (target, target))
    say("  targets installed              %s, %s" % (HOST_TARGET, GUEST_TARGET))

    if not os.path.exists("/usr/bin/arch"):
        raise Precondition("/usr/bin/arch is missing; cannot select an ISA")
    rc, _ = run(["/usr/bin/arch", "-x86_64", "/usr/bin/true"])
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

    rc, out = run(["rustc", "--version"])
    rustc = out.strip() if rc == 0 else "unavailable"
    say("  rustc                          %s" % rustc)
    say("  host                           %s" % os.uname().machine)
    return rustc


def build_arm(target, is_host):
    """Build one arm. Returns (binary path, sha256, `file` description)."""
    cmd = ["cargo", "build", "--release", "--locked", "--bin", "trios-train"]
    if not is_host:
        cmd[2:2] = ["--target", target]
    label = "cargo build %s" % ("(host)" if is_host else "--target " + target)
    rc, out = run(cmd, label=label)
    if rc != 0:
        say(out[-4000:])
        raise NoArtifact("the %s build failed" % target)

    rel = ("target/release/trios-train" if is_host
           else "target/%s/release/trios-train" % target)
    full = os.path.join(REPO_ROOT, rel)
    if not os.path.isfile(full):
        raise NoArtifact("the %s build reported success but %s does not exist"
                         % (target, rel))
    digest = sha256_file(full)
    frc, fout = run(["file", "-b", full])
    return rel, digest, (fout.strip() if frc == 0 else "unavailable")


def train(binary_rel, canon_name, native, run_dir):
    """Run one arm once. Returns the two artifact paths.

    `native` selects the ISA: the host binary runs directly, the x86_64 one is
    launched through `arch -x86_64` so the choice is explicit in the process
    tree rather than inferred from the Mach-O header.
    """
    cmd = [] if native else ["/usr/bin/arch", "-x86_64"]
    cmd += [os.path.join(REPO_ROOT, binary_rel)] + TRAIN_FLAGS
    rc, out = run(cmd, env=scrubbed_env(canon_name, run_dir),
                  label="train %s" % canon_name)
    if rc != 0:
        say(out[-4000:])
        raise NoArtifact("the run for %s exited %d" % (canon_name, rc))

    produced = {}
    for step in (INIT_STEP, TRAINED_STEP):
        rel = os.path.join(run_dir, canon_name, "%s.bin" % step)
        full = os.path.join(REPO_ROOT, rel)
        if not os.path.isfile(full):
            if step == INIT_STEP:
                raise NoArtifact(
                    "no initial-weights artifact at %s. TRIOS_CHECKPOINT_INIT "
                    "did not take effect: this trainer cannot write 0.bin, so "
                    "the init-versus-arithmetic question cannot be asked."
                    % rel)
            raise NoArtifact("no step-%s artifact at %s" % (step, rel))
        produced[step] = rel
    return produced


def read_sidecar(bin_rel):
    """Read a checkpoint's sidecar. Returns {} when there is none."""
    side = os.path.join(REPO_ROOT, bin_rel[:-4] + ".json")
    if not os.path.isfile(side):
        return {}
    try:
        with open(side, "r") as fh:
            return json.load(fh)
    except (ValueError, OSError):
        return {}


def declared_platform(bin_rel):
    plat = read_sidecar(bin_rel).get("platform", {})
    return plat.get("os", "?"), plat.get("arch", "?")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Vary the ISA and nothing else, and report where the "
                    "cross-architecture divergence begins.")
    parser.add_argument(
        "--skip-control", action="store_true",
        help="do not repeat each arm. Faster, and the result cannot be "
             "attributed to the ISA. For debugging this script only.")
    parser.add_argument(
        "--json", metavar="PATH", default=None,
        help="also write the hashes and the verdict to PATH as JSON")
    parser.add_argument(
        "--run-dir", metavar="DIR", default=None,
        help="checkpoint base directory for this invocation, relative to the "
             "repository root. Defaults to a fresh timestamped directory, "
             "which is what keeps re-runs from overwriting evidence.")
    args = parser.parse_args(argv)

    run_dir = args.run_dir or os.path.join(
        "checkpoints", "isa-probe",
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))

    say(RULE)
    say(" LOCAL ISA PROBE - one host, one OS, one libm vendor, two ISAs")
    say(RULE)

    rustc = check_preconditions()
    say("  run directory                  %s" % run_dir)
    say()

    say("ENVIRONMENT SCRUB")
    prove_env_carries_no_dsn(run_dir)
    say()

    say("BUILD (one working tree, two targets)")
    arm_bin, arm_bin_sha, arm_file = build_arm(HOST_TARGET, is_host=True)
    x86_bin, x86_bin_sha, x86_file = build_arm(GUEST_TARGET, is_host=False)
    say("  %-28s %s" % (HOST_TARGET, arm_bin_sha))
    say("      %s" % arm_file)
    say("  %-28s %s" % (GUEST_TARGET, x86_bin_sha))
    say("      %s" % x86_file)
    say("  The two BINARY hashes differ, and that carries no information:")
    say("  different machine code for a different instruction set cannot hash")
    say("  the same. Only the CHECKPOINT was ever claimed to be portable.")
    say()

    say("RUNS")
    runs = [("isa-probe-arm64", arm_bin, True),
            ("isa-probe-x86_64", x86_bin, False)]
    if not args.skip_control:
        runs += [("isa-probe-arm64-b", arm_bin, True),
                 ("isa-probe-x86_64-b", x86_bin, False)]
    artifacts = {}
    for canon_name, binary, native in runs:
        artifacts[canon_name] = train(binary, canon_name, native, run_dir)
    say()

    # A binary that changed underneath the probe means the four artifacts were
    # not all produced by the two binaries whose hashes were printed, and the
    # comparison has no subject. Concurrent work in this tree makes that a live
    # possibility, so it is checked rather than assumed.
    for rel, was, name in ((arm_bin, arm_bin_sha, HOST_TARGET),
                           (x86_bin, x86_bin_sha, GUEST_TARGET)):
        now = sha256_file(os.path.join(REPO_ROOT, rel))
        if now != was:
            raise Precondition(
                "the %s binary changed while the probe was running "
                "(%s -> %s). The artifacts were not all produced by one pair "
                "of binaries; re-run on a quiet tree." % (name, was, now))
    say("BINARIES UNCHANGED THROUGHOUT - all four artifacts come from the two")
    say("binaries hashed above.")
    say()

    # The arms are only two ISAs if the trainer says so from inside each
    # process. `arch -x86_64` is a request; the sidecar is the observation.
    arm_os, arm_arch = declared_platform(artifacts["isa-probe-arm64"][INIT_STEP])
    x86_os, x86_arch = declared_platform(artifacts["isa-probe-x86_64"][INIT_STEP])
    say("DECLARED PLATFORM, read back out of each run's own sidecar")
    say("  isa-probe-arm64                os=%s arch=%s [%s]"
        % (arm_os, arm_arch, TRANSLATION_NATIVE))
    say("  isa-probe-x86_64               os=%s arch=%s [%s]"
        % (x86_os, x86_arch, TRANSLATION_ROSETTA))
    if arm_arch == x86_arch:
        raise Precondition(
            "both arms report arch=%s: the ISA was NOT varied and there is no "
            "experiment here" % arm_arch)
    if arm_os != x86_os:
        raise Precondition(
            "the arms report different operating systems (%s vs %s); this "
            "probe exists to hold the OS fixed" % (arm_os, x86_os))
    say("  ISA varied, OS held fixed: this is the single-variable experiment.")
    say()

    hashes = {}
    for canon_name, produced in artifacts.items():
        for step, rel in produced.items():
            hashes[(canon_name, step)] = sha256_file(
                os.path.join(REPO_ROOT, rel))

    # THE CONTROL, first, because it is what licenses reading the lines below.
    control_ok = None
    if not args.skip_control:
        say(RULE)
        say(" WITHIN-ARM REPEATABILITY CONTROL (n=2 per arm, same host, same flags)")
        say(RULE)
        control_ok = True
        for arm in ("isa-probe-arm64", "isa-probe-x86_64"):
            for step in (INIT_STEP, TRAINED_STEP):
                a = hashes[(arm, step)]
                b = hashes[(arm + "-b", step)]
                same = a == b
                control_ok = control_ok and same
                say(" %-20s step %-3s %s" % (
                    arm, step, "SELF-MATCH" if same else "SELF-MISMATCH"))
                if not same:
                    say("     run a : %s" % a)
                    say("     run b : %s" % b)
        if control_ok:
            say(" Both arms repeat themselves byte for byte. Determinism is a")
            say(" property of this trainer on BOTH instruction sets, so a")
            say(" cross-ISA mismatch below cannot be dismissed as the trainer")
            say(" simply being unrepeatable on one of them.")
        else:
            say(" AN ARM IS NOT REPEATABLE WITH ITSELF. This is a BIGGER")
            say(" finding than any cross-ISA mismatch: it would make")
            say(" determinism a property of one machine-and-run rather than")
            say(" of the trainer. The verdict below loses its control")
            say(" condition. Do not attribute anything to the ISA.")
        say()

    arm_init = hashes[("isa-probe-arm64", INIT_STEP)]
    x86_init = hashes[("isa-probe-x86_64", INIT_STEP)]
    arm_step = hashes[("isa-probe-arm64", TRAINED_STEP)]
    x86_step = hashes[("isa-probe-x86_64", TRAINED_STEP)]

    init_same = arm_init == x86_init
    step_same = arm_step == x86_step

    # The two independent readings, one line each. A single MATCH/MISMATCH
    # over the pair would be exactly as uninformative as the 12000-step job.
    say(RULE)
    say(" CROSS-ISA COMPARISON - two independent readings, one line each")
    say(RULE)
    say(" INIT   aarch64 %s" % arm_init)
    say(" INIT   x86_64  %s   %s" % (x86_init, "MATCH" if init_same else "MISMATCH"))
    say(" STEP10 aarch64 %s" % arm_step)
    say(" STEP10 x86_64  %s   %s" % (x86_step, "MATCH" if step_same else "MISMATCH"))
    say(RULE)
    say()

    if not init_same:
        verdict = VERDICT_RNG
        reading = [
            "The divergence is present BEFORE any gradient step. The seeded",
            "RNG and weight initialisation do not carry across the ISA change,",
            "so the cause is initialisation, not floating-point arithmetic,",
            "and it is fixable outright - integer RNG, explicit bit-level",
            "conversion - at no cost in speed.",
        ]
        if step_same:
            reading += [
                "",
                "BUT STEP10 MATCHED, WHICH IS CONTRADICTORY: different starting",
                "weights cannot yield identical weights ten steps later. Treat",
                "this run as broken and re-run before drawing any conclusion.",
            ]
    elif step_same:
        verdict = VERDICT_NOT_ISA
        reading = [
            "The artifact carries across the ISA change at BOTH points. On this",
            "host the instruction set alone does not produce a divergence, so",
            "the ISA is not what run 30767491098 saw, and the remaining",
            "candidate for that mismatch is the OS - its libm and its libc.",
            "The pitch's attribution of that mismatch to CPU architecture would",
            "then be wrong and must be corrected.",
            "",
            "This says nothing about 12000 steps: run 30767491098 measured a",
            "mismatch there. The next step would be to bisect the step count.",
        ]
    else:
        verdict = VERDICT_ISA
        reading = [
            "Initialisation is byte-identical across the ISA change and the",
            "divergence is introduced within ten optimizer steps. The cause is",
            "the floating-point arithmetic of the training loop - reduction",
            "order or libm - and it is fixable only by constraining that",
            "arithmetic, at a cost in speed.",
            "",
            "An ISA change ALONE is therefore sufficient to produce a",
            "divergence. That is not the same as showing it is the only cause",
            "operating in run 30767491098, which also varied the OS and libc.",
            "",
            "It is not the only cause, and that is measured rather than",
            "conceded. A THIRD step-10 hash exists for these same declared",
            "inputs, from native x86_64 Linux (glibc 2.39, CI run 31004703001):",
            "  x86_64 linux   %s" % LINUX_X86_STEP10,
            "  x86_64 rosetta %s" % x86_step,
            "Same declared architecture on both, different OS and libc,",
            "different bytes. So the isolation is one-way: the ISA suffices,",
            "and it is not alone. This verdict is named SUFFICIENT, not ONLY,",
            "for that reason.",
        ]

    say("VERDICT: %s" % verdict)
    say()
    for line in reading:
        say(("  " + line) if line else "")
    say()
    if control_ok is False:
        say("  ATTRIBUTION VOID: the within-arm control above FAILED. The")
        say("  verdict names a reading of the hashes; it is not licensed as a")
        say("  statement about the instruction set until an arm repeats.")
        say()
    elif control_ok is None:
        say("  ATTRIBUTION UNLICENSED: --skip-control was passed, so no arm was")
        say("  repeated and n=1 per arm cannot separate a between-ISA effect")
        say("  from within-arm non-repeatability. Re-run without it.")
        say()

    say("SCOPE, which may not be dropped when this result is quoted:")
    say("  Rosetta 2 is binary translation, not native x86_64 silicon, and the")
    say("  x86_64 slice of Apple's libm is not glibc's libm. This constrains")
    say("  the native x86_64 Linux arm; it does not replace it.")

    if args.json:
        record = {
            "schema": "trios-local-isa-probe/1",
            "verdict": verdict,
            "init": {"aarch64": arm_init, "x86_64": x86_init,
                     "match": init_same},
            "step10": {"aarch64": arm_step, "x86_64": x86_step,
                       "match": step_same},
            "control": {"ran": control_ok is not None, "passed": control_ok},
            "hashes": {"%s/%s" % k: v for k, v in sorted(hashes.items())},
            "binaries": {
                HOST_TARGET: {"path": arm_bin, "sha256": arm_bin_sha,
                              "file": arm_file},
                GUEST_TARGET: {"path": x86_bin, "sha256": x86_bin_sha,
                               "file": x86_file},
            },
            # `translation` is repeated on each arm, not only at the top level.
            # A consumer that reads one arm's entry sees `arch: x86_64,
            # os: macos` and would otherwise take it for native silicon.
            "declared_platform": {
                "isa-probe-arm64": {"os": arm_os, "arch": arm_arch,
                                    "translation": TRANSLATION_NATIVE},
                "isa-probe-x86_64": {"os": x86_os, "arch": x86_arch,
                                     "translation": TRANSLATION_ROSETTA},
            },
            "corpus": {"train": EXPECTED_TRAIN_SHA, "val": EXPECTED_VAL_SHA},
            "run_dir": run_dir,
            "flags": TRAIN_FLAGS,
            "rustc": rustc,
            "host": " ".join(os.uname()),
            "translation": "x86_64 arm executed under Rosetta 2",
        }
        with open(args.json, "w") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
            fh.write("\n")
        say()
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
