#!/usr/bin/env python3
# IGLA-Coder execution-based pass@1 eval harness (Coder-Loop+6, wave B).
#
# WHAT THIS MEASURES
#   The Loop+5 harness asks "does the completion compile?" (compile@1). This
#   harness asks the strictly harder question: "does the generated function
#   produce the RIGHT answers?" (pass@1). For each micro-spec it
#     1. feeds the model a function signature + a one-line spec comment,
#     2. takes the completion,
#     3. wraps it with a fixed reference test main() that calls the function on
#        known inputs and asserts the known-correct outputs (HumanEval pattern),
#     4. compiles + runs; pass@1 = 1 only if it compiles AND all asserts hold.
#
# WHY SELF-CONTAINED MICRO-SPECS (and not parallel.jsonl)
#   The t27 parallel.jsonl carries only C<->Verilog file METADATA (paths, byte
#   counts) -- no executable I/O fixtures and no int main() harnesses (the
#   corpus uses in-body asserts as internal checks). So an honest, runnable
#   pass@1 needs reference test vectors, which we supply here. The specs are
#   drawn from the corpus's own arithmetic / bit-twiddling domain (mac, spi,
#   trit packing, modular ops) so they probe the same skill the model trained
#   on, while staying small enough that a CPU model has a fighting chance.
#
# PHI-AS-AXIS DISCIPLINE
#   Generator-agnostic. It measures the MODEL, never phi. To test "is phi
#   special as an axis" run this once per generator arm at EQUAL budget and diff
#   pass@1 (freeze -> hash -> null -> compare). The harness is the ruler.
#
# HONESTY
#   A tiny CPU model at any near-term budget will almost certainly score
#   pass@1 = 0. That is the expected result and the whole point: pass@1 is the
#   number we want to see move off zero, and it cannot be gamed by emitting
#   plausible-looking-but-wrong code (unlike compile@1).
#
# USAGE
#   # train+sample inline:
#   python3 eval_pass1.py --mode train --bin ../target/release/igla_coder \
#       --train ../data/code_train.bin --val ../data/code_val.bin \
#       --optimizer standard --seed 42 --steps 12000 --hidden 64 \
#       --fim-loss middle --max-new 220 --out eval_pass1.csv
#   # or score a saved checkpoint without retraining:
#   python3 eval_pass1.py --mode load --bin ../target/release/igla_coder \
#       --load ../ckpt_pilot.bin --max-new 220 --out eval_pass1.csv

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from shutil import which

COMPLETION_RE = re.compile(
    r"--- completion \(\d+ tokens\) ---\n(.*?)\n(?:HONESTY:|anchor:)", re.DOTALL
)

# C micro-specs. Each: a prompt the model continues, plus a reference test main
# that #includes the same header, declares nothing extra, and asserts known
# outputs. lang_id=1 is C. The prompt deliberately gives the signature + a spec
# comment so the task is "implement the body", not "invent an API".
C_SPECS = [
    {
        "name": "add_u32",
        "prompt": (
            "#include <stdint.h>\n"
            "/* return a + b (mod 2^32) */\n"
            "uint32_t add_u32(uint32_t a, uint32_t b) {\n"
        ),
        "test_main": (
            "#include <assert.h>\n"
            "int main(void){\n"
            "  assert(add_u32(2u,3u)==5u);\n"
            "  assert(add_u32(0u,0u)==0u);\n"
            "  assert(add_u32(0xFFFFFFFFu,1u)==0u);\n"
            "  return 0;\n}\n"
        ),
    },
    {
        "name": "clamp_u8",
        "prompt": (
            "#include <stdint.h>\n"
            "/* clamp x into [0,255] and return as uint8_t */\n"
            "uint8_t clamp_u8(int32_t x) {\n"
        ),
        "test_main": (
            "#include <assert.h>\n"
            "int main(void){\n"
            "  assert(clamp_u8(300)==255);\n"
            "  assert(clamp_u8(-5)==0);\n"
            "  assert(clamp_u8(42)==42);\n"
            "  return 0;\n}\n"
        ),
    },
    {
        "name": "popcount8",
        "prompt": (
            "#include <stdint.h>\n"
            "/* return the number of set bits in the low 8 bits of x */\n"
            "uint32_t popcount8(uint32_t x) {\n"
        ),
        "test_main": (
            "#include <assert.h>\n"
            "int main(void){\n"
            "  assert(popcount8(0u)==0u);\n"
            "  assert(popcount8(0xFFu)==8u);\n"
            "  assert(popcount8(0x0Fu)==4u);\n"
            "  return 0;\n}\n"
        ),
    },
    {
        "name": "mod_add",
        "prompt": (
            "#include <stdint.h>\n"
            "/* return (a + b) % m, assume m > 0 */\n"
            "uint32_t mod_add(uint32_t a, uint32_t b, uint32_t m) {\n"
        ),
        "test_main": (
            "#include <assert.h>\n"
            "int main(void){\n"
            "  assert(mod_add(5u,7u,10u)==2u);\n"
            "  assert(mod_add(0u,0u,3u)==0u);\n"
            "  assert(mod_add(9u,9u,5u)==3u);\n"
            "  return 0;\n}\n"
        ),
    },
]


def run_generate_train(args, prompt):
    cmd = [
        args.bin, "generate",
        "--train", args.train, "--val", args.val,
        "--optimizer", args.optimizer, "--seed", str(args.seed),
        "--steps", str(args.steps), "--hidden", str(args.hidden),
        "--lang-id", "1", "--max-new", str(args.max_new),
        "--temp", str(args.temp), "--prompt", prompt,
    ]
    if args.fim_loss:
        cmd += ["--fim-loss", args.fim_loss]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    m = COMPLETION_RE.search(out.stdout)
    return (m.group(1) if m else ""), out.returncode


def run_generate_load(args, prompt):
    cmd = [
        args.bin, "load-generate", "--load", args.load,
        "--seed", str(args.seed), "--lang-id", "1",
        "--max-new", str(args.max_new), "--temp", str(args.temp),
        "--prompt", prompt,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    m = COMPLETION_RE.search(out.stdout)
    return (m.group(1) if m else ""), out.returncode


def compile_and_run(full_src):
    """Return (compiles, passes). passes implies compiles."""
    d = tempfile.mkdtemp()
    src = os.path.join(d, "prog.c")
    exe = os.path.join(d, "prog")
    with open(src, "w") as f:
        f.write(full_src)
    try:
        c = subprocess.run(
            ["gcc", "-std=c11", "-O0", "-w", src, "-o", exe],
            capture_output=True, text=True, timeout=30,
        )
        if c.returncode != 0:
            return False, False
        r = subprocess.run([exe], capture_output=True, text=True, timeout=10)
        return True, (r.returncode == 0)
    except Exception:  # noqa: BLE001 -- never crash the sweep
        return False, False
    finally:
        for p in (src, exe):
            if os.path.exists(p):
                os.unlink(p)
        os.rmdir(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "load"], default="train")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--train", default="")
    ap.add_argument("--val", default="")
    ap.add_argument("--load", default="")
    ap.add_argument("--optimizer", default="standard")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=220, dest="max_new")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--fim-loss", default="", dest="fim_loss")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default="eval_pass1.csv")
    args = ap.parse_args()

    if not which("gcc"):
        print("gcc missing -- cannot run execution-based pass@1")
        sys.exit(0)
    if args.mode == "train" and not (args.train and args.val):
        ap.error("--mode train requires --train and --val")
    if args.mode == "load" and not args.load:
        ap.error("--mode load requires --load")

    print("anchor: phi^2 + phi^-2 = 3")
    print("HONEST: execution pass@1 of a tiny CPU model; 0 is the expected "
          "near-term result and the number we want to watch move.")

    rows = []
    n_compile = 0
    n_pass = 0
    for spec in C_SPECS:
        if args.mode == "train":
            completion, rc = run_generate_train(args, spec["prompt"])
        else:
            completion, rc = run_generate_load(args, spec["prompt"])
        full = spec["prompt"] + completion + "\n" + spec["test_main"]
        compiles, passes = compile_and_run(full)
        n_compile += int(compiles)
        n_pass += int(passes)
        print("  %-12s compiles=%s passes=%s (gen_rc=%s)"
              % (spec["name"], compiles, passes, rc))
        rows.append({
            "spec": spec["name"], "lang": "c",
            "optimizer": args.optimizer, "seed": args.seed,
            "steps": args.steps if args.mode == "train" else "",
            "mode": args.mode,
            "compiles": int(compiles), "passes": int(passes),
        })

    n = len(C_SPECS)
    with open(args.out, "w", newline="") as f:
        f.write("# IGLA-Coder execution pass@1 -- generator-agnostic; "
                "phi^2 + phi^-2 = 3\n")
        w = csv.DictWriter(
            f, fieldnames=["spec", "lang", "optimizer", "seed", "steps",
                           "mode", "compiles", "passes"])
        w.writeheader()
        w.writerows(rows)

    print("compile@1 = %d/%d = %.3f" % (n_compile, n, n_compile / n))
    print("pass@1    = %d/%d = %.3f" % (n_pass, n, n_pass / n))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
