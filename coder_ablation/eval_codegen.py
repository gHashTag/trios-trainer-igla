#!/usr/bin/env python3
# IGLA-Coder multi-language compile@1 eval harness (Coder-Loop+5, option B).
#
# WHAT THIS MEASURES
#   For each (lang, prompt) it drives `igla_coder generate`, captures the
#   completion, and asks the language's own toolchain a single binary question:
#   "does this syntactically check out?" (compile / syntax-only / parse). It
#   reports a per-language compile@1 rate. This is the P5 gate: a model-side
#   signal that is NOT BPB and NOT a self-graded heuristic -- the C compiler,
#   the Python parser, node, rustc, and gfortran are the judges.
#
# WHAT THIS IS NOT
#   - NOT pass@1 on unit tests (that needs spec test-suites; see Loop+6 menu).
#   - NOT a quality claim. A tiny CPU model trained for a smoke budget will
#     score ~0 here; that is the HONEST expected result and the harness exists
#     precisely so we can watch that number move off zero as training scales.
#
# PHI-AS-AXIS DISCIPLINE
#   The harness is generator-agnostic. It takes whatever the model emits and
#   measures it. It says nothing about phi. To compare g=phi vs g=standard you
#   run this harness once per arm at EQUAL budget and diff the compile@1 rates
#   (freeze -> hash -> null -> compare). The harness is the ruler, not the claim.
#
# USAGE
#   python3 eval_codegen.py --bin ../target/release/igla_coder \
#       --train ../data/code_train.bin --val ../data/code_val.bin \
#       --optimizer standard --seed 42 --steps 300 --hidden 64 \
#       --langs c,python,javascript --max-new 200 --out eval_codegen.csv
#
# Each checker is best-effort: if a toolchain is missing the lang is reported
# as "skipped(no-toolchain)" rather than counted as a failure, so the rate is
# never silently deflated by environment gaps.

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile

# t27 lang_id mapping mirrors prep_t27_corpus.py / igla_coder LANG sentinel use.
# Only langs we can actually CHECK on this box are listed; extend as toolchains
# become available. lang_id is the integer the model was conditioned on.
LANGS = {
    "c": {"lang_id": 1, "ext": ".c", "checker": "gcc"},
    "python": {"lang_id": 2, "ext": ".py", "checker": "py"},
    "javascript": {"lang_id": 3, "ext": ".js", "checker": "node"},
    "rust": {"lang_id": 4, "ext": ".rs", "checker": "rustc"},
    "fortran": {"lang_id": 5, "ext": ".f90", "checker": "gfortran"},
}

# Minimal, honest prompts: short headers the model should continue. Kept tiny so
# the smoke budget has a chance and so the test is about syntactic continuation.
PROMPTS = {
    "c": "#include <stdint.h>\nuint32_t add(uint32_t a, uint32_t b) {\n",
    "python": "def add(a, b):\n",
    "javascript": "function add(a, b) {\n",
    "rust": "fn add(a: u32, b: u32) -> u32 {\n",
    "fortran": "integer function add(a, b)\n  integer :: a, b\n",
}

# The model prints the completion between these markers (igla_coder generate).
COMPLETION_RE = re.compile(
    r"--- completion \(\d+ tokens\) ---\n(.*?)\nHONESTY:", re.DOTALL
)


def have(tool):
    from shutil import which

    return which(tool) is not None


def run_generate(args, lang_id, prompt):
    cmd = [
        args.bin,
        "generate",
        "--train",
        args.train,
        "--val",
        args.val,
        "--optimizer",
        args.optimizer,
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--hidden",
        str(args.hidden),
        "--lang-id",
        str(lang_id),
        "--max-new",
        str(args.max_new),
        "--temp",
        str(args.temp),
        "--prompt",
        prompt,
    ]
    if args.fim_loss:
        cmd += ["--fim-loss", args.fim_loss]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    m = COMPLETION_RE.search(out.stdout)
    completion = m.group(1) if m else ""
    return completion, out.returncode


def check_c(src):
    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run(
            ["gcc", "-fsyntax-only", "-std=c11", path],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return r.returncode == 0
    finally:
        os.unlink(path)


def check_py(src):
    r = subprocess.run(
        [sys.executable, "-c", "import sys,py_compile,tempfile; "
         "p=tempfile.mktemp(suffix='.py'); open(p,'w').write(sys.stdin.read()); "
         "py_compile.compile(p, doraise=True)"],
        input=src,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return r.returncode == 0


def check_node(src):
    r = subprocess.run(
        ["node", "--check", "-"],
        input=src,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if r.returncode == 0:
        return True
    # node --check on stdin may need a file on some versions; fall back.
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run(
            ["node", "--check", path], capture_output=True, text=True, timeout=20
        )
        return r.returncode == 0
    finally:
        os.unlink(path)


def check_rust(src):
    # wrap so a bare fn parses as a crate; emit metadata to a real temp file
    # (writing to /dev/null can fail on some setups and falsely report failure).
    d = tempfile.mkdtemp()
    path = os.path.join(d, "snippet.rs")
    out = os.path.join(d, "snippet.meta")
    with open(path, "w") as f:
        f.write(src + "\nfn main() {}\n")
    try:
        r = subprocess.run(
            ["rustc", "--edition", "2021", "--emit", "metadata",
             "--crate-type", "lib", "-A", "warnings", "-o", out, path],
            capture_output=True,
            text=True,
            timeout=40,
        )
        return r.returncode == 0
    finally:
        for p in (path, out):
            if os.path.exists(p):
                os.unlink(p)
        os.rmdir(d)


def check_fortran(src):
    with tempfile.NamedTemporaryFile("w", suffix=".f90", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run(
            ["gfortran", "-fsyntax-only", path],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return r.returncode == 0
    finally:
        os.unlink(path)


CHECKERS = {
    "gcc": check_c,
    "py": check_py,
    "node": check_node,
    "rustc": check_rust,
    "gfortran": check_fortran,
}
CHECKER_TOOL = {
    "gcc": "gcc",
    "py": sys.executable,
    "node": "node",
    "rustc": "rustc",
    "gfortran": "gfortran",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--optimizer", default="standard")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=200, dest="max_new")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--fim-loss", default="", dest="fim_loss",
                    help="all|middle (passed through to the trainer)")
    ap.add_argument("--langs", default="c,python,javascript")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="eval_codegen.csv")
    args = ap.parse_args()

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    rows = []
    print("anchor: phi^2 + phi^-2 = 3")
    print("HONEST: compile@1 of a tiny CPU model; ~0 at smoke budget is expected.")
    for lang in langs:
        if lang not in LANGS:
            print("  skip unknown lang:", lang)
            continue
        meta = LANGS[lang]
        checker_key = meta["checker"]
        tool = CHECKER_TOOL[checker_key]
        if not have(tool):
            print("  %-11s skipped(no-toolchain: %s)" % (lang, tool))
            rows.append({"lang": lang, "lang_id": meta["lang_id"],
                         "optimizer": args.optimizer, "seed": args.seed,
                         "steps": args.steps, "compiles": "",
                         "status": "skipped_no_toolchain"})
            continue
        prompt = PROMPTS.get(lang, "")
        completion, rc = run_generate(args, meta["lang_id"], prompt)
        full = prompt + completion
        ok = False
        status = "checked"
        try:
            ok = CHECKERS[checker_key](full)
        except Exception as e:  # noqa: BLE001 -- record, never crash the sweep
            status = "checker_error:" + type(e).__name__
        print("  %-11s compiles=%s (gen_rc=%s, %s)"
              % (lang, ok, rc, status))
        rows.append({"lang": lang, "lang_id": meta["lang_id"],
                     "optimizer": args.optimizer, "seed": args.seed,
                     "steps": args.steps, "compiles": int(ok),
                     "status": status})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["lang", "lang_id", "optimizer", "seed", "steps",
                        "compiles", "status"],
        )
        f.write("# IGLA-Coder compile@1 eval -- generator-agnostic; "
                "phi^2 + phi^-2 = 3\n")
        w.writeheader()
        w.writerows(rows)

    checked = [r for r in rows if r["compiles"] != ""]
    if checked:
        rate = sum(int(r["compiles"]) for r in checked) / len(checked)
        print("compile@1 over %d checked langs = %.3f" % (len(checked), rate))
    else:
        print("compile@1 = n/a (no toolchains available)")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
