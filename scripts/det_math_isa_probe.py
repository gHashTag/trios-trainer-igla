#!/usr/bin/env python3
"""Does a bit-exact `exp` close the cross-ISA checkpoint divergence?

WHAT CAME BEFORE

`scripts/local_isa_probe.py` (do not edit it; other work cites it) established,
on ONE macOS host with ONE pinned rustc and ONE working tree, with the
`--target` triple as the only moving variable:

    INIT   (0.bin)  aarch64 == x86_64   4f854c82...   MATCH
    STEP10 (10.bin) aarch64 efef1cba... x86_64 5913542e...   MISMATCH

Initialisation and the seeded RNG carry across the ISA change; the arithmetic
of the training loop does not. Two candidate mechanisms were then removed by
measurement: FMA contraction is excluded by disassembly (neither binary
contains one), and reduction order is excluded by construction (`-O0` and
`-O3 + LTO` produce identical weights, which a re-associating compiler would
not). `src/bin/ulp_census.rs` measured what was left: `expf` disagrees between
the arms on 246 of 40010 census inputs, every disagreement exactly 1 ULP, while
`sqrtf` and `powi` agree everywhere.

WHAT THIS PROBE ADDS

Nobody had yet RUN the trainer with a deterministic `exp`. `--features
det-math` routes the two - exactly two, verified by grep - `.exp()` calls on
the training path through `det_math::exp_det`, which uses only correctly-rounded
IEEE-754 operations and calls into no vendor library. This probe rebuilds both
arms with that feature and re-asks the step-10 question.

SINCE THEN the same feature also routes `cosine_lr`'s cosine through
`det_math::cos_det`, so the step-10 hashes it prints MOVED - at ten steps
`warmup = 1` and the cosine branch runs on nine of them. That is expected and
is not a regression: the N1 control below still pins the DEFAULT build to its
published bytes, and the det-math hash is republished in docs/DET-MATH.md
alongside the run that produced it.

Both answers are results and neither is a failure of this probe:

    MATCH     libm was the whole of the remaining divergence, and the artifact
              is byte-portable across this ISA boundary once `exp` is pinned.
    MISMATCH  something else is also moving the bytes, and it is a source no
              probe in this repository has yet reached. The stage trace is then
              run to name the first divergent tensor.

FOUR MEASUREMENTS, and the first one is not optional

    N1 CONTROL.   Default features, aarch64, 10 steps. `0.bin` must still be
                  4f854c82... and `10.bin` must still be efef1cba... . These
                  edits touched `softmax` in two files and made `cosine_lr`
                  public; if the DEFAULT artifact moved, the edits changed the
                  published trainer and every number below is unreadable. The
                  probe stops.
    N2 THE RESULT. `--features det-math`, both targets, 10 steps.
    N3 THE PRICE.  Wall-clock of a 1000-step aarch64 run with and without the
                  feature, n=3 each. This is TRAINER wall-clock. A per-call
                  microbenchmark of `exp_det` against `expf` in isolation gives
                  a much larger ratio and is NOT the cost of the feature.
    N4 THE SCOPE LIMIT, AS IT STOOD WHEN THIS SCRIPT WAS WRITTEN. `cosine_lr`
                  called `cosf`, which the feature then did not replace, and
                  the 10-step probe has `warmup = 1` and reports the lr bits
                  identical at every step - so a 10-step MATCH did not extend
                  to the 12000-step headline. `lr_schedule_dump` is built for
                  both targets here and the two DEFAULT-feature dumps diffed,
                  which is still the baseline: 58 of 12000, first at step 4041.
                  N4 IS NO LONGER THE WHOLE STORY. `cosine_lr` now routes its
                  cosine through `det_math::cos_det` under the feature, and
                  `scripts/lr_schedule_isa_probe.py` runs BOTH arms and
                  measures 0 of 12000 differing with it. N4 as run here is the
                  control half of that pair, not a live limit.

Run them SEPARATELY (`--only isa`, `--only timing`, `--only lr`). N3 is six
1000-step runs; a kill mid-loop under one timeout would leave a partial
measurement behind, and this repository has been burned by that before.

ENVIRONMENT

Same discipline as `scripts/local_isa_probe.py`: every training run goes
through `env -i` expressed as an explicit ALLOWLIST dict, and the child
environment is PROVED empty of all four DSN aliases by inspecting what a child
actually receives, rather than asserted to be by describing what this process
intends to pass. `TRINITY_AUTOMIGRATE` defaults to "1" when unset, so an
inherited DSN would run migrations against the live SSOT before training
started.

CANON NAMES

FRESH names - `det-math-aarch64`, `det-math-x86_64`, `det-math-control` and
their `-b` controls. Reusing `isa-probe-*` would collide with the existing
probe's sidecars, and `checkpoint::save` refuses to overwrite a sidecar that
would lose information. That refusal is correct and fails closed; the fix is a
new name, never `TRIOS_ALLOW_SIDECAR_OVERWRITE=1`.

Exit codes:
    0  a verdict was reached, whatever it was. A MISMATCH IS A RESULT.
    1  an artifact could not be produced: a build failed, a run failed, or a
       checkpoint the run was supposed to write is not on disk
    2  usage error, or a precondition is not met - target not installed,
       Rosetta absent, corpus not matching data/MANIFEST.sha256, the two arms
       did not actually run on different instruction sets, a binary changed
       underneath the probe, or the N1 control moved the published artifact
"""

import argparse
import hashlib
import json
import os
import shutil
import statistics
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

# The published aarch64 default-feature artifacts, from scripts/local_isa_probe.py
# run on this host. N1 asserts these EXACTLY. They are the reason a reader can
# believe anything else this script prints: if they still hold, the det-math
# edits did not move the default build.
BASELINE_INIT = "4f854c82"
BASELINE_STEP10 = "efef1cba128a8c96"

# Byte-identical to scripts/local_isa_probe.py's TRAIN_FLAGS, so an arm of this
# probe is textually the same experiment as an arm of that one and the N1
# comparison is against a like-for-like run rather than a near-enough one.
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

# N3. Same shape, 1000 steps. --eval-every is kept at 1000 so the eval gate
# fires the same number of times in both timed arms; gf16_floor mutates weights
# on an eval-gated schedule, so an eval cadence that differed between the arms
# would time two different computations.
TIMING_STEPS = 1000
TIMING_REPEATS = 3

INIT_STEP = "0"
TRAINED_STEP = "10"

TRANSLATION_NATIVE = "native execution"
TRANSLATION_ROSETTA = "Rosetta 2 binary translation, not native x86_64 silicon"

VERDICT_MATCH = "DET_MATH_CLOSES_THE_ISA_GAP_AT_10_STEPS"
VERDICT_MISMATCH = "DET_MATH_INSUFFICIENT_ANOTHER_SOURCE_REMAINS"

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
        say("  %-42s %7.1fs  %s" % (label, elapsed, status))
    return proc.returncode, proc.stdout, elapsed


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
    env = scrubbed_env("det-math-envcheck", run_dir)
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


def check_preconditions(need_guest=True):
    say("PRECONDITIONS")

    if sys.platform != "darwin":
        raise Precondition(
            "this probe holds the OS fixed by running both arms on ONE macOS "
            "host under Rosetta 2; on %s there is no such pair to build"
            % sys.platform)

    rc, out, _ = run(["rustup", "target", "list", "--installed"])
    if rc != 0:
        raise Precondition("rustup is not available: %s" % out.strip()[:200])
    installed = out.split()
    wanted = (HOST_TARGET, GUEST_TARGET) if need_guest else (HOST_TARGET,)
    for target in wanted:
        if target not in installed:
            raise Precondition(
                "target %s is not installed. Install it with:\n"
                "    rustup target add %s" % (target, target))
    say("  targets installed              %s" % ", ".join(wanted))

    if need_guest:
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
    return rustc


def build_arm(target, is_host, det_math, binary="trios-train"):
    """Build one arm. Returns (relative path, sha256, `file` description)."""
    cmd = ["cargo", "build", "--release", "--locked", "--bin", binary]
    if det_math:
        cmd += ["--features", "det-math"]
    if not is_host:
        cmd[2:2] = ["--target", target]
    label = "cargo build %s %s%s" % (
        binary, "(host)" if is_host else "--target " + target,
        " --features det-math" if det_math else " [default features]")
    rc, out, _ = run(cmd, label=label)
    if rc != 0:
        say(out[-4000:])
        raise NoArtifact("the %s build failed" % target)

    rel = ("target/release/%s" % binary if is_host
           else "target/%s/release/%s" % (target, binary))
    full = os.path.join(REPO_ROOT, rel)
    if not os.path.isfile(full):
        raise NoArtifact("the %s build reported success but %s does not exist"
                         % (target, rel))
    rel = stash(rel, "%s-%s-%s" % (binary, target,
                                   "detmath" if det_math else "default"))
    full = os.path.join(REPO_ROOT, rel)
    frc, fout, _ = run(["file", "-b", full])
    return rel, sha256_file(full), (fout.strip() if frc == 0 else "unavailable")


# `cargo build --bin X` and `cargo build --bin X --features det-math` write to
# the SAME path, target/release/X. The default and the det-math host binaries
# therefore overwrite each other, and a probe that ran from that path would be
# comparing artifacts produced by whichever build happened last - or would trip
# its own "the binary changed underneath the probe" guard, which is what
# happened on the first attempt at this measurement (recorded here rather than
# quietly fixed: the guard did its job).
#
# Every binary is copied out to its own name under target/det-math-probe/ the
# moment it is built, hashed there, and RUN from there. Nothing in this probe
# executes a path that a later cargo invocation can rewrite.
STASH = os.path.join("target", "det-math-probe")


def stash(rel, name):
    os.makedirs(os.path.join(REPO_ROOT, STASH), exist_ok=True)
    dest = os.path.join(STASH, name)
    shutil.copy2(os.path.join(REPO_ROOT, rel), os.path.join(REPO_ROOT, dest))
    return dest


def train(binary_rel, canon_name, native, run_dir, flags=None, steps=None):
    """Run one arm once. Returns ({step: path}, elapsed seconds)."""
    cmd = [] if native else ["/usr/bin/arch", "-x86_64"]
    cmd += [os.path.join(REPO_ROOT, binary_rel)] + (flags or TRAIN_FLAGS)
    rc, out, elapsed = run(cmd, env=scrubbed_env(canon_name, run_dir),
                           label="train %s" % canon_name)
    if rc != 0:
        say(out[-4000:])
        raise NoArtifact("the run for %s exited %d" % (canon_name, rc))

    produced = {}
    for step in (INIT_STEP, steps or TRAINED_STEP):
        rel = os.path.join(run_dir, canon_name, "%s.bin" % step)
        full = os.path.join(REPO_ROOT, rel)
        if not os.path.isfile(full):
            if step == INIT_STEP:
                raise NoArtifact(
                    "no initial-weights artifact at %s. TRIOS_CHECKPOINT_INIT "
                    "did not take effect." % rel)
            raise NoArtifact("no step-%s artifact at %s" % (step, rel))
        produced[step] = rel
    return produced, elapsed


def read_sidecar(bin_rel):
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


# ---------------------------------------------------------------- N1 + N2 ---

def phase_isa(run_dir, rustc, skip_control):
    say("ENVIRONMENT SCRUB")
    prove_env_carries_no_dsn(run_dir)
    say()

    say("N1 CONTROL - default features, aarch64, 10 steps")
    say("  If either hash below moved, the det-math edits changed the PUBLISHED")
    say("  artifact and nothing after this point is readable.")
    ctl_bin, ctl_sha, ctl_file = build_arm(HOST_TARGET, True, det_math=False)
    ctl, _ = train(ctl_bin, "det-math-control", True, run_dir)
    ctl_init = sha256_file(os.path.join(REPO_ROOT, ctl[INIT_STEP]))
    ctl_step = sha256_file(os.path.join(REPO_ROOT, ctl[TRAINED_STEP]))
    say("  INIT   %s   expected prefix %s" % (ctl_init, BASELINE_INIT))
    say("  STEP10 %s   expected prefix %s" % (ctl_step, BASELINE_STEP10))
    if not ctl_init.startswith(BASELINE_INIT):
        raise Precondition(
            "N1 FAILED: the default-feature INIT artifact moved. Expected a "
            "hash starting %s, got %s." % (BASELINE_INIT, ctl_init))
    if not ctl_step.startswith(BASELINE_STEP10):
        raise Precondition(
            "N1 FAILED: the default-feature STEP10 artifact moved. Expected a "
            "hash starting %s, got %s. The det-math edits are not inert on the "
            "default build; fix that before reading N2."
            % (BASELINE_STEP10, ctl_step))
    say("  N1 PASSES: the default build still produces the published bytes.")
    say()

    say("N2 BUILD - one working tree, two targets, --features det-math")
    arm_bin, arm_sha, arm_file = build_arm(HOST_TARGET, True, det_math=True)
    x86_bin, x86_sha, x86_file = build_arm(GUEST_TARGET, False, det_math=True)
    say("  %-28s %s" % (HOST_TARGET, arm_sha))
    say("      %s" % arm_file)
    say("  %-28s %s" % (GUEST_TARGET, x86_sha))
    say("      %s" % x86_file)
    say("  The two BINARY hashes differ, and that carries no information:")
    say("  different machine code for a different instruction set cannot hash")
    say("  the same. Only the CHECKPOINT was ever claimed to be portable.")
    say("  The default-feature host binary hashed %s; it is a THIRD" % ctl_sha[:16])
    say("  binary and is not compared to these.")
    say()

    say("N2 RUNS")
    runs = [("det-math-aarch64", arm_bin, True),
            ("det-math-x86_64", x86_bin, False)]
    if not skip_control:
        runs += [("det-math-aarch64-b", arm_bin, True),
                 ("det-math-x86_64-b", x86_bin, False)]
    artifacts = {}
    for canon_name, binary, native in runs:
        artifacts[canon_name], _ = train(binary, canon_name, native, run_dir)
    say()

    for rel, was, name in ((arm_bin, arm_sha, HOST_TARGET),
                           (x86_bin, x86_sha, GUEST_TARGET),
                           (ctl_bin, ctl_sha, "control")):
        now = sha256_file(os.path.join(REPO_ROOT, rel))
        if now != was:
            raise Precondition(
                "the %s binary changed while the probe was running (%s -> %s). "
                "Re-run on a quiet tree." % (name, was, now))
    say("BINARIES UNCHANGED THROUGHOUT.")
    say()

    arm_os, arm_arch = declared_platform(artifacts["det-math-aarch64"][INIT_STEP])
    x86_os, x86_arch = declared_platform(artifacts["det-math-x86_64"][INIT_STEP])
    say("DECLARED PLATFORM, read back out of each run's own sidecar")
    say("  det-math-aarch64               os=%s arch=%s [%s]"
        % (arm_os, arm_arch, TRANSLATION_NATIVE))
    say("  det-math-x86_64                os=%s arch=%s [%s]"
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

    control_ok = None
    if not skip_control:
        say(RULE)
        say(" WITHIN-ARM REPEATABILITY CONTROL (n=2 per arm, det-math build)")
        say(RULE)
        control_ok = True
        for arm in ("det-math-aarch64", "det-math-x86_64"):
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
            say(" Both arms repeat themselves byte for byte.")
        else:
            say(" AN ARM IS NOT REPEATABLE WITH ITSELF. Attribution below is")
            say(" void. This is a bigger finding than any cross-ISA reading.")
        say()

    arm_init = hashes[("det-math-aarch64", INIT_STEP)]
    x86_init = hashes[("det-math-x86_64", INIT_STEP)]
    arm_step = hashes[("det-math-aarch64", TRAINED_STEP)]
    x86_step = hashes[("det-math-x86_64", TRAINED_STEP)]
    init_same = arm_init == x86_init
    step_same = arm_step == x86_step

    say(RULE)
    say(" N2 CROSS-ISA COMPARISON UNDER --features det-math")
    say(RULE)
    say(" INIT   aarch64 %s" % arm_init)
    say(" INIT   x86_64  %s   %s" % (x86_init, "MATCH" if init_same else "MISMATCH"))
    say(" STEP10 aarch64 %s" % arm_step)
    say(" STEP10 x86_64  %s   %s" % (x86_step, "MATCH" if step_same else "MISMATCH"))
    say(RULE)
    say()

    verdict = VERDICT_MATCH if step_same else VERDICT_MISMATCH
    say("VERDICT: %s" % verdict)
    say()
    if step_same:
        for line in [
            "With `exp` pinned to a bit-exact implementation, the step-10",
            "checkpoint is byte-identical across the ISA change on this host.",
            "libm's `expf` accounts for the whole of the divergence the earlier",
            "probe measured at this step count.",
            "",
            "SCOPE, which may not be dropped when this is quoted:",
            "  * TEN steps, not 12000. 11990 further steps of arithmetic were",
            "    not run, and no probe here has run them. What HAS been closed",
            "    since this text was first written is the schedule: cosine_lr",
            "    now goes through cos_det under this feature and",
            "    scripts/lr_schedule_isa_probe.py measures 0 of 12000 headline",
            "    learning rates differing, against 58 without it. So the site",
            "    that broke at step 4041 is gone; that removes the known",
            "    obstacle to a 12000-step claim, it does not make the claim.",
            "  * ONE macOS host. Rosetta 2 is binary translation, not native",
            "    x86_64 silicon, and Apple's x86_64 libm is not glibc's. This",
            "    constrains the native x86_64 Linux arm; it does not replace it.",
            "  * n=1 per arm for the hashes, with an n=2 within-arm control.",
        ]:
            say(("  " + line) if line else "")
    else:
        for line in [
            "Pinning `exp` was NOT sufficient. The step-10 checkpoints still",
            "differ, so a source of divergence remains that is not libm's",
            "`expf` - and it is a source no probe in this repository has yet",
            "reached. That is a finding, not a failure: it names the next",
            "measurement instead of leaving the question open.",
            "",
            "Run the stage trace on both arms to name the first divergent",
            "tensor under det-math before drawing any further conclusion.",
        ]:
            say(("  " + line) if line else "")
    say()
    if control_ok is False:
        say("  ATTRIBUTION VOID: the within-arm control FAILED.")
        say()
    elif control_ok is None:
        say("  ATTRIBUTION UNLICENSED: --skip-control was passed.")
        say()

    return {
        "schema": "trios-det-math-isa-probe/1",
        "verdict": verdict,
        "n1_control": {"init": ctl_init, "step10": ctl_step,
                       "baseline_init_prefix": BASELINE_INIT,
                       "baseline_step10_prefix": BASELINE_STEP10,
                       "passed": True},
        "n2_init": {"aarch64": arm_init, "x86_64": x86_init, "match": init_same},
        "n2_step10": {"aarch64": arm_step, "x86_64": x86_step,
                      "match": step_same},
        "control": {"ran": control_ok is not None, "passed": control_ok},
        "hashes": {"%s/%s" % k: v for k, v in sorted(hashes.items())},
        "binaries": {
            "control-" + HOST_TARGET: {"path": ctl_bin, "sha256": ctl_sha,
                                       "file": ctl_file},
            HOST_TARGET: {"path": arm_bin, "sha256": arm_sha, "file": arm_file},
            GUEST_TARGET: {"path": x86_bin, "sha256": x86_sha, "file": x86_file},
        },
        "declared_platform": {
            "det-math-aarch64": {"os": arm_os, "arch": arm_arch,
                                 "translation": TRANSLATION_NATIVE},
            "det-math-x86_64": {"os": x86_os, "arch": x86_arch,
                                "translation": TRANSLATION_ROSETTA},
        },
        "corpus": {"train": EXPECTED_TRAIN_SHA, "val": EXPECTED_VAL_SHA},
        "run_dir": run_dir,
        "flags": TRAIN_FLAGS,
        "rustc": rustc,
        "host": " ".join(os.uname()),
    }


# --------------------------------------------------------------------- N3 ---

def phase_timing(run_dir, rustc):
    """Wall-clock price of the feature, on aarch64, n=3 per arm."""
    say("N3 THE PRICE - trainer wall-clock, aarch64, %d steps, n=%d per arm"
        % (TIMING_STEPS, TIMING_REPEATS))
    say("  This is the cost of the FEATURE on the TRAINER. A per-call")
    say("  microbenchmark of exp_det against expf in isolation reports a much")
    say("  larger ratio; that number is not this number and must not be quoted")
    say("  as the trainer's cost.")
    say()
    prove_env_carries_no_dsn(run_dir)
    say()

    flags = list(TRAIN_FLAGS)
    flags[flags.index("--steps") + 1] = str(TIMING_STEPS)

    times = {}
    for det in (False, True):
        binary, _, _ = build_arm(HOST_TARGET, True, det_math=det)
        arm = "det-math" if det else "default"
        times[arm] = []
        for i in range(TIMING_REPEATS):
            canon = "det-math-timing-%s-%d" % (arm, i)
            _, elapsed = train(binary, canon, True, run_dir,
                               flags=flags, steps=str(TIMING_STEPS))
            times[arm].append(elapsed)
        say()

    say(RULE)
    say(" N3 RESULT")
    say(RULE)
    summary = {}
    for arm in ("default", "det-math"):
        vals = times[arm]
        summary[arm] = {"runs": vals, "min": min(vals),
                        "median": statistics.median(vals)}
        say(" %-10s min %7.2fs  median %7.2fs   (%s)"
            % (arm, min(vals), statistics.median(vals),
               ", ".join("%.2f" % v for v in vals)))
    ratio_min = summary["det-math"]["min"] / summary["default"]["min"]
    ratio_med = summary["det-math"]["median"] / summary["default"]["median"]
    say(" ratio det-math/default    min %.2f   median %.2f" % (ratio_min, ratio_med))
    say(RULE)
    say()
    return {"schema": "trios-det-math-timing/1", "steps": TIMING_STEPS,
            "repeats": TIMING_REPEATS, "target": HOST_TARGET,
            "seconds": summary,
            "ratio": {"min": ratio_min, "median": ratio_med},
            "rustc": rustc, "host": " ".join(os.uname()),
            "note": "trainer wall-clock, not a per-call microbenchmark"}


# --------------------------------------------------------------------- N4 ---

def phase_lr(out_dir, rustc):
    """Does the 12000-step LR schedule itself carry across the ISA change?

    DEFAULT FEATURES ONLY, deliberately. This arm is now the CONTROL for
    `scripts/lr_schedule_isa_probe.py`, which runs the same comparison with and
    without `det-math` and is the place to look for the result.
    """
    say("N4 THE SCOPE LIMIT - the learning-rate schedule across the ISA change")
    say("  Built with DEFAULT features, which is what makes this the control:")
    say("  the libm cosf that cosine_lr uses when the feature is off. If the")
    say("  schedule diverges here, the 12000-step headline diverges through a")
    say("  path the forward pass never touches, and no 10-step MATCH can speak")
    say("  for it. For the det-math arm of this same comparison run")
    say("  scripts/lr_schedule_isa_probe.py --both.")
    say()

    dumps = {}
    for target, is_host, launcher in ((HOST_TARGET, True, []),
                                      (GUEST_TARGET, False, ["/usr/bin/arch", "-x86_64"])):
        binary, sha, desc = build_arm(target, is_host, det_math=False,
                                      binary="lr_schedule_dump")
        rc, out, _ = run(launcher + [os.path.join(REPO_ROOT, binary)],
                         label="lr_schedule_dump %s" % target)
        if rc != 0:
            say(out[-2000:])
            raise NoArtifact("lr_schedule_dump failed on %s" % target)
        path = os.path.join(out_dir, "lr-schedule-%s.txt" % target)
        with open(path, "w") as fh:
            fh.write(out)
        dumps[target] = out
        say("  %-28s %s  wrote %s" % (target, sha[:16], os.path.relpath(path, REPO_ROOT)))
    say()

    a = [l for l in dumps[HOST_TARGET].splitlines() if l.startswith("LR ")]
    b = [l for l in dumps[GUEST_TARGET].splitlines() if l.startswith("LR ")]
    if len(a) != len(b):
        raise Precondition("the two dumps have different lengths (%d vs %d)"
                           % (len(a), len(b)))
    diffs = [(x, y) for x, y in zip(a, b) if x != y]
    first = diffs[0] if diffs else None

    diff_path = os.path.join(out_dir, "lr-schedule-diff.txt")
    with open(diff_path, "w") as fh:
        fh.write("# aarch64-apple-darwin vs x86_64-apple-darwin (Rosetta 2)\n")
        fh.write("# lines compared: %d\n" % len(a))
        fh.write("# lines differing: %d\n" % len(diffs))
        for x, y in diffs:
            fh.write("- %s\n+ %s\n" % (x, y))

    say(RULE)
    say(" N4 RESULT")
    say(RULE)
    say(" steps compared         %d" % len(a))
    say(" lr values differing    %d" % len(diffs))
    if first:
        say(" first differing step   %s" % first[0].split()[1])
        say("   aarch64  %s" % first[0])
        say("   x86_64   %s" % first[1])
        say(" With the DEFAULT cosf, a 10-step cross-ISA MATCH does NOT extend")
        say(" to the 12000-step headline: the schedule diverges on its own,")
        say(" through a path the forward pass never touches. This is the")
        say(" control; scripts/lr_schedule_isa_probe.py --both runs it against")
        say(" the det-math arm, where the same diff is 0.")
    else:
        say(" The schedule carries across the ISA change at every step. That")
        say(" removes cosf as an OBSTACLE to a 12000-step claim; it does not")
        say(" make one, because 11990 further steps of arithmetic were not run.")
    say(RULE)
    say()
    return {"schema": "trios-det-math-lr-scope/1", "steps_compared": len(a),
            "differing": len(diffs),
            "first_differing_step": int(first[0].split()[1]) if first else None,
            "first_differing": {"aarch64": first[0], "x86_64": first[1]} if first else None,
            "rustc": rustc, "host": " ".join(os.uname()),
            "features": "default",
            "note": "the DEFAULT-feature control: cosine_lr's libm cosf. The "
                    "det-math arm of this comparison is "
                    "scripts/lr_schedule_isa_probe.py"}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the trainer with a bit-exact exp and re-ask the "
                    "cross-ISA question.")
    parser.add_argument("--only", choices=("isa", "timing", "lr"), default="isa",
                        help="which measurement to take. Run them separately: "
                             "a kill mid-loop leaves a partial measurement.")
    parser.add_argument("--skip-control", action="store_true",
                        help="do not repeat each arm. Faster, and the result "
                             "cannot be attributed to the ISA.")
    parser.add_argument("--json", metavar="PATH", default=None,
                        help="also write the result record to PATH as JSON")
    parser.add_argument("--out-dir", metavar="DIR",
                        default=os.path.join(REPO_ROOT, "evidence", "det-math-isa"),
                        help="where N4 writes its dumps")
    parser.add_argument("--run-dir", metavar="DIR", default=None,
                        help="checkpoint base directory, relative to the repo "
                             "root. Defaults to a fresh timestamped directory, "
                             "which is what keeps re-runs from overwriting "
                             "evidence.")
    args = parser.parse_args(argv)

    run_dir = args.run_dir or os.path.join(
        "checkpoints", "det-math-isa",
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))

    say(RULE)
    say(" DET-MATH ISA PROBE - does a bit-exact exp close the cross-ISA gap?")
    say(RULE)

    rustc = check_preconditions(need_guest=args.only in ("isa", "lr"))
    say("  run directory                  %s" % run_dir)
    say("  measurement                    %s" % args.only)
    say()

    if args.only == "isa":
        record = phase_isa(run_dir, rustc, args.skip_control)
    elif args.only == "timing":
        record = phase_timing(run_dir, rustc)
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        record = phase_lr(args.out_dir, rustc)

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
