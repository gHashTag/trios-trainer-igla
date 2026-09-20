#!/usr/bin/env python3
"""Name the FIRST tensor that diverges across the ISA boundary.

WHY THIS EXISTS

`scripts/local_isa_probe.py` measured that 0.bin is byte-identical across the
ISA change and 10.bin is not. That bisects the divergence in TIME, and ten
steps is as narrow as time-bisection gets before it costs one run per step.
This probe bisects in SPACE instead: it hashes every intermediate tensor of ONE
forward/backward pair at step 1 and reports the first stage whose hash differs.
One run per arm, and the answer is a NAME.

The name is the whole point. The scouts already excluded the two explanations
that would have been reached for first:

  reduction order   `dot_sequential_4096` and `dot_split8_4096` hash
                    IDENTICALLY across the arms while disagreeing WITH EACH
                    OTHER on each arm - so the two ISAs associate a long sum
                    the same way, twice over. `-force-vector-width=1` moved not
                    one output byte.
  FMA contraction   excluded by disassembly: neither binary contains one.

What survives is libm (`expf`, ~13,300 calls per step, in two different
softmaxes; `cosf`, once per step) and the discontinuous NCA amplifier at
src/train_loop.rs, where `.round()` buckets a value and one ULP can flip a
bucket. Those three are distinguishable by WHERE the trace first differs, and
docs/DIVERGENCE-FIRST-TENSOR.md is the table that reads the answer. Read the
verdict THROUGH that table; the stage name alone is not yet an attribution.

WHAT IS HELD FIXED

Everything that `local_isa_probe.py` holds fixed, for the same reasons and by
the same means: both binaries are built from ONE working tree by ONE pinned
compiler (rust-toolchain.toml) differing only in `--target`, and both run on
ONE macOS host, so the OS, the libm vendor and the compiler do not move. The
x86_64 arm executes under Rosetta 2 - binary translation, NOT native x86_64
silicon - and that limit travels with every result this script prints.

THE CONTROL

Each arm runs TWICE under different `TRIOS_CANON_NAME`. If an arm's two trace
streams are not identical, then the trainer is not repeatable with itself on
that arm and a cross-arm difference cannot be attributed to the arm. The
verdict is then printed with its attribution explicitly VOIDED rather than
suppressed: the measurement still happened and hiding it would be the worse
failure.

CORROBORATION, NOT A GATE

Each traced run also writes 10.bin, and the script prints whether those hashes
equal the two constants `local_isa_probe.py` measured WITHOUT tracing:

    aarch64  efef1cba128a8c96e23124d1f139f73c11f8e00261b6148fcfb8cc427aaa0cac
    x86_64   5913542eb613abc3780ac959a0262af059f7b11ea6ed61b23bd0d62b9c8897ab

A match says the traced execution is the same execution that produced the
evidenced divergence - the trace observed it without moving it. A MISMATCH
would mean the instrument changed the experiment, and the stage verdict would
be worthless; the script says so in that case and exits 1. The standing proof
of inertness is `tests/stage_trace_inert.rs`; this is the same check taken in
situ, on the two arms this script actually ran.

ENVIRONMENT

`env -i` with an allowlist: PATH, HOME, TRINITY_AUTOMIGRATE=0,
TRIOS_CANON_NAME, TRIOS_TRACE_STAGE - plus TRIOS_CHECKPOINT_DIR, which the work
item's list did not name. It is here for a reason worth stating: checkpoints
are ON by default, `checkpoint::save` refuses to overwrite a sidecar that would
lose information, and without a fresh directory per invocation the SECOND run
of this script would either fail or need `TRIOS_ALLOW_SIDECAR_OVERWRITE=1`,
which destroys a previous measurement in order to take a new one. A timestamped
directory costs nothing and keeps every run's evidence.

EXIT CODES

    0  a verdict was reached, whatever it was. A DIVERGENCE IS A RESULT, and so
       is NO DIVERGENCE IN TRACED STAGES (it means the trace list is
       incomplete, which is a finding about this script and not a success).
    1  no verdict: a build failed, a run failed, the two streams do not have
       the same stage structure, or the trace moved the artifact.
    2  a precondition of the experiment is not met.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HOST_TARGET = "aarch64-apple-darwin"
GUEST_TARGET = "x86_64-apple-darwin"

# A target directory of this probe's own. The shared `target/` carries a build
# lock, and other agents are compiling in this tree concurrently; contending on
# that lock would make this script's runtime depend on unrelated work.
CARGO_TARGET_DIR = "/Users/playra/igla-target-stage-trace"

# From data/MANIFEST.sha256 - the same two hashes the CI jobs and
# scripts/local_isa_probe.py check.
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

# Byte-identical to evidence/xarch-local-isa/probe.json. Same experiment, one
# extra observation channel.
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

# Measured by scripts/local_isa_probe.py with the trace ABSENT. Recorded in
# evidence/xarch-local-isa/probe.json.
UNTRACED_STEP10 = {
    "aarch64": "efef1cba128a8c96e23124d1f139f73c11f8e00261b6148fcfb8cc427aaa0cac",
    "x86_64": "5913542eb613abc3780ac959a0262af059f7b11ea6ed61b23bd0d62b9c8897ab",
}

TRANSLATION_NATIVE = "native execution"
TRANSLATION_ROSETTA = "Rosetta 2 binary translation, not native x86_64 silicon"

EVIDENCE_DIR = os.path.join("evidence", "stage-trace-isa")

RULE = "=" * 74


class Precondition(Exception):
    """A condition the experiment needs, which is not met. Exit 2."""


class NoVerdict(Exception):
    """The comparison has no subject. Exit 1."""


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
    """Run a command. Returns (rc, stdout, stderr) kept SEPARATE.

    Separate, unlike local_isa_probe.py's merged capture, because the TRACE
    lines are on stderr and interleaving them with the trainer's stdout would
    make the stream order depend on buffering rather than on dataflow.
    """
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed = time.time() - started
    if label:
        status = "ok" if proc.returncode == 0 else "rc=%d" % proc.returncode
        say("  %-34s %6.1fs  %s" % (label, elapsed, status))
    return proc.returncode, proc.stdout, proc.stderr


def scrubbed_env(canon_name, run_dir):
    """`env -i` with an allowlist, expressed as a dict."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "TRINITY_AUTOMIGRATE": "0",
        "TRIOS_CANON_NAME": canon_name,
        "TRIOS_TRACE_STAGE": "1",
        "TRIOS_CHECKPOINT_DIR": run_dir,
    }


def prove_env_carries_no_dsn(run_dir):
    """Prove the child environment is empty of every DSN alias.

    Proved by inspecting what a child actually receives, not by asserting what
    this process intends to pass.
    """
    env = scrubbed_env("stage-trace-envcheck", run_dir)
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
    if "TRIOS_TRACE_STAGE=1" not in seen:
        raise Precondition(
            "TRIOS_TRACE_STAGE does not reach the trainer; there would be no "
            "trace to compare")
    say("  no DSN alias reaches the trainer; automigrate off; trace on")


def check_preconditions():
    say("PRECONDITIONS")

    if sys.platform != "darwin":
        raise Precondition(
            "this probe holds the OS fixed by running both arms on ONE macOS "
            "host under Rosetta 2; on %s there is no such pair to build"
            % sys.platform)

    rc, out, _ = run(["rustup", "target", "list", "--installed"])
    if rc != 0:
        raise Precondition("rustup is not available")
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
            "Rosetta 2 cannot execute x86_64 binaries on this host")
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
    say("  cargo target dir               %s" % CARGO_TARGET_DIR)
    return rustc


def build_env():
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = CARGO_TARGET_DIR
    return env


def build_arm(target, is_host):
    """Build one arm. Returns (binary path, sha256, `file` description).

    The host arm is built WITHOUT `--target`, exactly as
    scripts/local_isa_probe.py builds it, so this script's aarch64 binary is
    produced by the same command line that produced the evidenced efef1cba
    artifact. Making the two arms cosmetically symmetric by passing
    `--target aarch64-apple-darwin` here would change the host build's cargo
    invocation and put the corroboration check's subject in doubt.
    """
    cmd = ["cargo", "build", "--release", "--locked", "--bin", "trios-train"]
    if not is_host:
        cmd[2:2] = ["--target", target]
    label = "cargo build %s" % ("(host)" if is_host else "--target " + target)
    rc, out, err = run(cmd, env=build_env(), label=label)
    if rc != 0:
        say((out + err)[-4000:])
        raise NoVerdict("the %s build failed" % target)

    rel = ("release/trios-train" if is_host
           else "%s/release/trios-train" % target)
    full = os.path.join(CARGO_TARGET_DIR, rel)
    if not os.path.isfile(full):
        raise NoVerdict("the %s build reported success but %s does not exist"
                        % (target, full))
    frc, fout, _ = run(["file", "-b", full])
    return full, sha256_file(full), (fout.strip() if frc == 0 else "unavailable")


def parse_trace(stderr_text, label):
    """Extract the ordered stage list from one run's stderr.

    Returns [(name, sha256)] in emission order, which IS dataflow order: the
    trainer emits each stage where the value becomes final.

    A repeated stage name is refused rather than deduplicated. The trace is
    supposed to cover ONE forward/backward pair; a second `l1_q` would mean the
    narrowing failed and "the first stage that differs" would silently mean
    "the first of several same-named stages that differ", which is exactly the
    kind of quietly-wrong answer this exercise exists to remove.
    """
    stages = []
    seen = set()
    for line in stderr_text.splitlines():
        if not line.startswith("TRACE "):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise NoVerdict("malformed TRACE line in %s: %r" % (label, line))
        name, digest = parts[1], parts[2]
        if name in seen:
            raise NoVerdict(
                "stage %r appears twice in %s: the trace is not narrowed to "
                "one forward/backward pair" % (name, label))
        seen.add(name)
        stages.append((name, digest))
    if not stages:
        raise NoVerdict(
            "no TRACE lines in %s. TRIOS_TRACE_STAGE=1 did not take effect, "
            "so there is nothing to compare." % label)
    return stages


def train(binary, canon_name, native, run_dir):
    """Run one arm once. Returns (stages, raw stderr, step-10 checkpoint sha)."""
    cmd = [] if native else ["/usr/bin/arch", "-x86_64"]
    cmd += [binary] + TRAIN_FLAGS
    rc, _, err = run(cmd, env=scrubbed_env(canon_name, run_dir),
                     label="train %s" % canon_name)
    if rc != 0:
        say(err[-4000:])
        raise NoVerdict("the run for %s exited %d" % (canon_name, rc))
    stages = parse_trace(err, canon_name)

    ckpt = os.path.join(REPO_ROOT, run_dir, canon_name, "10.bin")
    if not os.path.isfile(ckpt):
        raise NoVerdict("no step-10 artifact at %s" % ckpt)
    return stages, err, sha256_file(ckpt)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Name the first tensor that diverges across the ISA "
                    "boundary, by stage trace at step 1.")
    parser.add_argument(
        "--skip-control", action="store_true",
        help="do not repeat each arm. Faster, and the result cannot be "
             "attributed to the ISA. For debugging this script only.")
    parser.add_argument(
        "--run-dir", metavar="DIR", default=None,
        help="checkpoint base directory for this invocation, relative to the "
             "repository root. Defaults to a fresh timestamped directory.")
    args = parser.parse_args(argv)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = args.run_dir or os.path.join("checkpoints", "stage-trace", stamp)

    say(RULE)
    say(" STAGE TRACE ACROSS THE ISA BOUNDARY - first divergent tensor, step 1")
    say(RULE)

    rustc = check_preconditions()
    say("  run directory                  %s" % run_dir)
    say()

    say("ENVIRONMENT SCRUB")
    prove_env_carries_no_dsn(run_dir)
    say()

    say("BUILD (one working tree, two targets, one pinned compiler)")
    arm_bin, arm_bin_sha, arm_file = build_arm(HOST_TARGET, is_host=True)
    x86_bin, x86_bin_sha, x86_file = build_arm(GUEST_TARGET, is_host=False)
    say("  %-28s %s" % (HOST_TARGET, arm_bin_sha))
    say("      %s" % arm_file)
    say("  %-28s %s" % (GUEST_TARGET, x86_bin_sha))
    say("      %s" % x86_file)
    say()

    say("RUNS (trace on stderr, one line per stage)")
    plan = [("aarch64", "stage-trace-arm64", arm_bin, True),
            ("x86_64", "stage-trace-x86_64", x86_bin, False)]
    if not args.skip_control:
        plan += [("aarch64-b", "stage-trace-arm64-b", arm_bin, True),
                 ("x86_64-b", "stage-trace-x86_64-b", x86_bin, False)]
    streams, raw, ckpts = {}, {}, {}
    for arm, canon, binary, native in plan:
        streams[arm], raw[arm], ckpts[arm] = train(
            binary, canon, native, run_dir)
    say("  stages per run                 %d" % len(streams["aarch64"]))
    say()

    for path, was, name in ((arm_bin, arm_bin_sha, HOST_TARGET),
                            (x86_bin, x86_bin_sha, GUEST_TARGET)):
        now = sha256_file(path)
        if now != was:
            raise Precondition(
                "the %s binary changed while the probe was running (%s -> %s). "
                "The streams were not all produced by one pair of binaries; "
                "re-run on a quiet tree." % (name, was, now))
    say("BINARIES UNCHANGED THROUGHOUT - every stream comes from the two")
    say("binaries hashed above.")
    say()

    # Corroboration: does the TRACED run still produce the evidenced artifact?
    say(RULE)
    say(" DID THE INSTRUMENT MOVE THE EXPERIMENT")
    say(RULE)
    inert = True
    for arm in ("aarch64", "x86_64"):
        want = UNTRACED_STEP10[arm]
        got = ckpts[arm]
        ok = want == got
        inert = inert and ok
        say(" %-8s 10.bin %s  %s" % (arm, got, "SAME AS UNTRACED" if ok
                                     else "CHANGED BY THE TRACE"))
        if not ok:
            say("          untraced %s" % want)
    if inert:
        say(" Both traced runs reproduce the checkpoint measured WITHOUT the")
        say(" trace. The instrument observed the divergence; it did not cause")
        say(" it.")
    say()
    if not inert:
        raise NoVerdict(
            "the trace changed the artifact it was supposed to observe. Every "
            "stage hash below describes an execution that is not the evidenced "
            "one, so there is no verdict to read. Fix the trace's inertness "
            "first; tests/stage_trace_inert.rs is the standing guard.")

    # Structure, before content. Two streams that do not list the same stages
    # in the same order are not comparable stage by stage.
    names_arm = [n for n, _ in streams["aarch64"]]
    names_x86 = [n for n, _ in streams["x86_64"]]
    if names_arm != names_x86:
        only_arm = [n for n in names_arm if n not in set(names_x86)]
        only_x86 = [n for n in names_x86 if n not in set(names_arm)]
        raise NoVerdict(
            "the two arms did not emit the same stage sequence. "
            "aarch64-only: %s; x86_64-only: %s. A stage-by-stage comparison "
            "has no subject." % (only_arm or "none", only_x86 or "none"))

    control_ok = None
    if not args.skip_control:
        say(RULE)
        say(" WITHIN-ARM REPEATABILITY CONTROL (n=2 per arm, same host, flags)")
        say(RULE)
        control_ok = True
        for arm in ("aarch64", "x86_64"):
            same = streams[arm] == streams[arm + "-b"]
            control_ok = control_ok and same
            say(" %-10s %s" % (arm, "SELF-MATCH over all %d stages" % len(
                streams[arm]) if same else "SELF-MISMATCH"))
            if not same:
                for (n1, h1), (n2, h2) in zip(streams[arm],
                                              streams[arm + "-b"]):
                    if (n1, h1) != (n2, h2):
                        say("     first self-difference at %s: %s vs %s"
                            % (n1, h1, h2))
                        break
        say(" CONTROL: %s" % ("PASSED" if control_ok else "FAILED"))
        if not control_ok:
            say(" An arm does not repeat itself. That is a BIGGER finding than")
            say(" any cross-arm difference: it would make determinism a")
            say(" property of one machine-and-run rather than of the trainer.")
        say()

    first = None
    for (name, h_arm), (_, h_x86) in zip(streams["aarch64"], streams["x86_64"]):
        if h_arm != h_x86:
            first = (name, h_arm, h_x86)
            break

    n_diff = sum(1 for (_, a), (_, b) in zip(streams["aarch64"],
                                             streams["x86_64"]) if a != b)

    say(RULE)
    say(" VERDICT")
    say(RULE)
    if first is None:
        say("NO DIVERGENCE IN TRACED STAGES")
        say("")
        say("This is NOT a success. 10.bin differs across the arms - the")
        say("hashes above are the two constants local_isa_probe.py measured -")
        say("so a divergence exists in a stage this trace does not cover. The")
        say("finding is that the stage list is INCOMPLETE, and the next move")
        say("is to name what is missing from it.")
        verdict_line = "NO DIVERGENCE IN TRACED STAGES"
    else:
        verdict_line = ("FIRST DIVERGENCE: %s aarch64=%s x86_64=%s"
                        % (first[0], first[1], first[2]))
        say(verdict_line)
        say("")
        say("%d of %d traced stages differ." % (n_diff, len(streams["aarch64"])))
        say("Read this THROUGH the table in docs/DIVERGENCE-FIRST-TENSOR.md.")
        say("The stage name is a location, not yet a cause.")
    if control_ok is False:
        say("")
        say("ATTRIBUTION VOIDED: the within-arm control FAILED. The line above")
        say("is a measurement that was taken; it is not evidence about the")
        say("ISA, because the trainer did not repeat itself on one of the arms.")
        verdict_line += "  [ATTRIBUTION VOIDED - within-arm control FAILED]"
    elif control_ok is None:
        say("")
        say("ATTRIBUTION VOIDED: run without --skip-control to attribute this")
        say("to the ISA.")
        verdict_line += "  [ATTRIBUTION VOIDED - control skipped]"
    say(RULE)
    say()

    ev = os.path.join(REPO_ROOT, EVIDENCE_DIR)
    os.makedirs(ev, exist_ok=True)
    for arm in streams:
        with open(os.path.join(ev, "trace-%s.txt" % arm), "w") as fh:
            fh.write("".join(l + "\n" for l in raw[arm].splitlines()
                             if l.startswith("TRACE ")))
    record = {
        "generated_utc": stamp,
        "rustc": rustc,
        "flags": TRAIN_FLAGS,
        "cargo_target_dir": CARGO_TARGET_DIR,
        "binaries": {
            HOST_TARGET: {"sha256": arm_bin_sha, "file": arm_file,
                          "translation": TRANSLATION_NATIVE},
            GUEST_TARGET: {"sha256": x86_bin_sha, "file": x86_file,
                           "translation": TRANSLATION_ROSETTA},
        },
        "corpus": {"train": EXPECTED_TRAIN_SHA, "val": EXPECTED_VAL_SHA},
        "checkpoint_step10": ckpts,
        "checkpoint_step10_untraced": UNTRACED_STEP10,
        "trace_is_inert_in_situ": inert,
        "control_ran": not args.skip_control,
        "control_passed": control_ok,
        "stages_compared": len(streams["aarch64"]),
        "stages_differing": n_diff,
        "first_divergence": None if first is None else {
            "stage": first[0], "aarch64": first[1], "x86_64": first[2]},
        "verdict": verdict_line,
        "scope": (
            "One macOS host, one pinned rustc, one working tree; the ISA "
            "target is the only variable. The x86_64 arm runs under Rosetta 2 "
            "binary translation, NOT native x86_64 silicon."
        ),
    }
    with open(os.path.join(ev, "verdict.json"), "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")
    say("evidence written to %s/ (trace-*.txt, verdict.json)" % EVIDENCE_DIR)
    say("NOTE: evidence/SEALS.txt has no line for this directory yet; another")
    say("work item owns that file. Sealing it is a follow-up.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Precondition as exc:
        say("")
        say("PRECONDITION NOT MET: %s" % exc)
        sys.exit(2)
    except NoVerdict as exc:
        say("")
        say("NO VERDICT: %s" % exc)
        sys.exit(1)
