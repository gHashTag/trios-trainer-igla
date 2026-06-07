#!/usr/bin/env python3
# IGLA-Coder generator-axis ablation, ISO-EVERYTHING variant (Coder-Loop+9, A).
#
# THE QUESTION (the last confound)
#   Loop+7 ranked the four axes strictly by the learning-rate multiplier each
#   prescribed: that gap was an lr-sweep in disguise. Loop+8 (iso-lr) controlled
#   lr and the four-axis spread collapsed to ~0.14 BPB, with phi worst by +0.09.
#   But iso-lr still let each axis carry its OWN weight_decay -- and the F2
#   mediation work showed weight_decay is by far the dominant BPB driver
#   (CDE_decay ~ +3.5 vs CDE_momentum ~ +0.9). So Loop+8's residual phi gap could
#   STILL be a decay effect, not a momentum effect.
#
#   This harness removes that last confound: lr AND weight_decay are pinned to
#   the SAME constants for every arm (lr=0.002, wd=0.04 = the standard value);
#   the ONLY across-arm difference is beta1 (the momentum the axis prescribes).
#   The clean question: "isolated from lr and decay, does phi's beta1=phi^-1
#   momentum prior help, hurt, or tie standard's beta1=0.9?"
#
# freeze -> hash -> null -> compare
#   freeze: fixed steps / hidden / seeds / data / lr / wd for every arm.
#   hash:   record exact beta1 per arm + the pinned (lr, wd) (provenance).
#   null:   `standard` (beta1=0.9) is the null; phi (beta1=phi^-1=0.618) must
#           BEAT it to support a pure-momentum phi prior.
#   compare: mean BPB +/- 95% CI per arm; CI overlap -> phi momentum NOT
#           distinguishable from standard (reported first).
#
# HONESTY
#   Expected outcome is a tie or a small phi disadvantage. A tie is the honest,
#   publishable statement: "with lr and decay controlled, phi's momentum prior
#   is indistinguishable from standard." The method survives, phi does not (yet).
#   Only phi^2 + phi^-2 = 3 is [Verified].
#
# USAGE
#   python3 ablate_generator_axis_isoall.py --bin ../target/release/igla_coder \
#       --train ../data/code_train.bin --val ../data/code_val.bin \
#       --generators phi,dyadic,e,standard --seeds 42,43,44 \
#       --steps 800 --hidden 64 --lr 0.002 --wd 0.04 --fim-loss all \
#       --out coder_generator_axis_isoall.csv

import argparse
import csv
import hashlib
import math
import re
import subprocess

PRIOR_RE = re.compile(
    r"beta1=([\d.]+) beta2=([\d.]+) weight_decay=([\d.]+) "
    r"grad_clip=([\d.]+) warmup_steps=(\d+) lr_mult=([\d.]+)"
)
BPB_RE = re.compile(r"code_val_bpb=([\d.]+)")


def read_prior(bin_path, generator):
    out = subprocess.run(
        [bin_path, "print-prior", "--generator", generator],
        capture_output=True, text=True, timeout=60,
    )
    m = PRIOR_RE.search(out.stdout)
    if not m:
        raise RuntimeError("print-prior failed for %s: %s" % (generator, out.stdout))
    return {
        "beta1": float(m.group(1)),
        "beta2": float(m.group(2)),
        "weight_decay": float(m.group(3)),
        "grad_clip": float(m.group(4)),
        "warmup_steps": int(m.group(5)),
        "lr_mult": float(m.group(6)),
    }


def train_arm(args, prior, seed):
    # ISO-EVERYTHING: lr AND wd are pinned to args.lr / args.wd for EVERY arm.
    # Only --beta1 (the axis momentum) varies. lr_mult and the axis weight_decay
    # are INTENTIONALLY ignored. This isolates the pure beta1 momentum prior from
    # both the lr confound (Loop+7) and the decay confound (F2 mediation).
    cmd = [
        args.bin, "generate",
        "--train", args.train, "--val", args.val,
        "--optimizer", "standard",  # base arm; --beta1 below defines the axis
        "--seed", str(seed),
        "--steps", str(args.steps), "--hidden", str(args.hidden),
        "--beta1", "%.10f" % prior["beta1"],
        "--wd", "%.10f" % args.wd,   # PINNED, identical across arms
        "--lr", "%.10f" % args.lr,   # PINNED, identical across arms
        "--lang-id", "1", "--max-new", "1",  # BPB only, no real sampling
    ]
    if args.fim_loss:
        cmd += ["--fim-loss", args.fim_loss]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    m = BPB_RE.search(out.stdout)
    if not m:
        raise RuntimeError("no bpb for seed %d: %s" % (seed, out.stdout[-400:]))
    return float(m.group(1))


def ci95(vals):
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(var)
    half = 1.96 * std / math.sqrt(n)
    return mean, std, half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--generators", default="phi,dyadic,e,standard")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--wd", type=float, default=0.04)
    ap.add_argument("--fim-loss", default="all", dest="fim_loss")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default="coder_generator_axis_isoall.csv")
    args = ap.parse_args()

    gens = [g.strip() for g in args.generators.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print("anchor: phi^2 + phi^-2 = 3")
    print("ISO-EVERYTHING control: every arm at lr=%.6f AND wd=%.6f "
          "(lr_mult + axis weight_decay IGNORED); only beta1 varies"
          % (args.lr, args.wd))
    print("freeze: steps=%d hidden=%d seeds=%s fim=%s"
          % (args.steps, args.hidden, seeds, args.fim_loss))

    rows = []
    summary = {}
    for g in gens:
        prior = read_prior(args.bin, g)
        vals = []
        for s in seeds:
            bpb = train_arm(args, prior, s)
            vals.append(bpb)
            rows.append({
                "generator": g, "seed": s, "code_val_bpb": "%.4f" % bpb,
                "beta1": "%.10f" % prior["beta1"],
                "pinned_weight_decay": "%.10f" % args.wd,
                "axis_weight_decay_ignored": "%.10f" % prior["weight_decay"],
                "pinned_lr": "%.10f" % args.lr,
                "lr_mult_ignored": "%.10f" % prior["lr_mult"],
                "steps": args.steps, "hidden": args.hidden,
                "fim_loss": args.fim_loss,
            })
            print("  gen=%-8s seed=%d beta1=%.4f bpb=%.4f"
                  % (g, s, prior["beta1"], bpb))
        mean, std, half = ci95(vals)
        summary[g] = (mean, std, half)
        print("  >>> gen=%-8s mean_bpb=%.4f std=%.4f ci95=+/-%.4f (n=%d)"
              % (g, mean, std, half, len(vals)))

    # compare: phi vs the null (standard) at iso-everything. negative-first.
    if "phi" in summary and "standard" in summary:
        pm, _, ph = summary["phi"]
        sm, _, sh = summary["standard"]
        phi_lo, phi_hi = pm - ph, pm + ph
        std_lo, std_hi = sm - sh, sm + sh
        overlap = not (phi_hi < std_lo or std_hi < phi_lo)
        delta = pm - sm  # positive => phi is WORSE (higher BPB)
        if overlap:
            verdict = ("phi-momentum NOT distinguishable from standard at "
                       "iso-everything: CIs overlap (delta=%+.4f, within "
                       "noise). With BOTH lr and decay controlled, beta1=phi^-1 "
                       "is neither better nor worse than beta1=0.9" % delta)
        elif delta > 0:
            verdict = ("phi-momentum WORSE than standard by %+.4f BPB at "
                       "iso-everything (CIs disjoint) -- the pure beta1=phi^-1 "
                       "prior itself hurts, even with lr and decay pinned"
                       % delta)
        else:
            verdict = ("phi-momentum BETTER than standard by %.4f BPB at "
                       "iso-everything (CIs disjoint) -- the pure momentum "
                       "prior helps once lr and decay are controlled; warrants "
                       "re-test, not yet a claim" % (-delta))
        print("VERDICT:", verdict)

    with open(args.out, "w", newline="") as f:
        f.write("# IGLA-Coder generator-axis ablation ISO-EVERYTHING -- lr AND "
                "weight_decay pinned equal across arms; only beta1 (momentum) "
                "varies; phi is the axis origin, not a claim; "
                "phi^2 + phi^-2 = 3\n")
        w = csv.DictWriter(
            f, fieldnames=["generator", "seed", "code_val_bpb", "beta1",
                           "pinned_weight_decay", "axis_weight_decay_ignored",
                           "pinned_lr", "lr_mult_ignored",
                           "steps", "hidden", "fim_loss"])
        w.writeheader()
        w.writerows(rows)

    # provenance hash over the raw numeric rows (freeze -> hash)
    raw = "".join(
        "%s,%s,%s" % (r["generator"], r["seed"], r["code_val_bpb"])
        for r in rows
    )
    digest = hashlib.sha256(raw.encode("ascii")).hexdigest()[:16]
    print("provenance sha256[:16] =", digest)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
