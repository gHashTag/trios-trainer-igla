#!/usr/bin/env python3
# IGLA-Coder generator-axis ablation (Coder-Loop+6, wave C).
#
# THE QUESTION
#   "Is phi special as an axis?" We treat phi as the coordinate-axis ORIGIN and
#   compare it head-to-head against control axes g=2 (dyadic), g=e, and the
#   tuned `standard` null baseline -- all at EQUAL budget, same seeds, same
#   data. Each arm's (beta1, weight_decay) is read from the SINGLE SOURCE OF
#   TRUTH (src/config_prior.rs) via `igla_coder print-prior`, then injected into
#   training with --beta1 / --wd. No math is duplicated in this script.
#
# freeze -> hash -> null -> compare
#   freeze: fixed steps / hidden / seeds / data for every arm.
#   hash:   we record the exact (beta1, wd) used per arm (provenance).
#   null:   `standard` is the null baseline; phi must BEAT it to be supported.
#   compare: report mean BPB +/- 95% CI per arm; if phi's CI overlaps standard
#            -> phi NOT supported (negative result, reported first).
#
# HONESTY
#   Carried result: phi is FALSIFIED on the coder track (phi^-3 decay CDE
#   +3.544 BPB). This ablation is expected to reproduce "phi loses". The value
#   is the clean, equal-budget, multi-axis control a referee would demand --
#   not a hope that phi wins.
#
# USAGE
#   python3 ablate_generator_axis.py --bin ../target/release/igla_coder \
#       --train ../data/code_train.bin --val ../data/code_val.bin \
#       --generators phi,dyadic,e,standard --seeds 42,43,44 \
#       --steps 800 --hidden 64 --out coder_generator_axis.csv

import argparse
import csv
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
    # Inject config-prior beta1 / wd via the existing override flags. lr is
    # scaled by the prior's lr_mult so the axis comparison is faithful to what
    # config_prior prescribes (not just the momentum/decay pair).
    lr = args.lr * prior["lr_mult"]
    cmd = [
        args.bin, "generate",
        "--train", args.train, "--val", args.val,
        "--optimizer", "standard",  # base arm; overrides below define the axis
        "--seed", str(seed),
        "--steps", str(args.steps), "--hidden", str(args.hidden),
        "--beta1", "%.10f" % prior["beta1"],
        "--wd", "%.10f" % prior["weight_decay"],
        "--lr", "%.10f" % lr,
        "--lang-id", "1", "--max-new", "1",  # no real sampling needed for BPB
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
    ap.add_argument("--fim-loss", default="middle", dest="fim_loss")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default="coder_generator_axis.csv")
    args = ap.parse_args()

    gens = [g.strip() for g in args.generators.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print("anchor: phi^2 + phi^-2 = 3")
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
                "weight_decay": "%.10f" % prior["weight_decay"],
                "lr_mult": "%.10f" % prior["lr_mult"],
                "steps": args.steps, "hidden": args.hidden,
                "fim_loss": args.fim_loss,
            })
            print("  gen=%-8s seed=%d bpb=%.4f" % (g, s, bpb))
        mean, std, half = ci95(vals)
        summary[g] = (mean, std, half)
        print("  >>> gen=%-8s mean_bpb=%.4f std=%.4f ci95=+/-%.4f (n=%d)"
              % (g, mean, std, half, len(vals)))

    # compare: phi vs the null (standard). negative-first verdict.
    if "phi" in summary and "standard" in summary:
        pm, _, ph = summary["phi"]
        sm, _, sh = summary["standard"]
        phi_lo, phi_hi = pm - ph, pm + ph
        std_lo, std_hi = sm - sh, sm + sh
        overlap = not (phi_hi < std_lo or std_hi < phi_lo)
        delta = pm - sm  # positive => phi is WORSE (higher BPB)
        if overlap:
            verdict = ("phi NOT supported: CI overlaps standard "
                       "(delta=%+.4f, within noise)" % delta)
        elif delta > 0:
            verdict = ("phi WORSE than standard by %+.4f BPB (CIs disjoint) "
                       "-- phi falsified on this axis" % delta)
        else:
            verdict = ("phi BETTER than standard by %.4f BPB (CIs disjoint) "
                       "-- would warrant re-test, not yet a claim" % (-delta))
        print("VERDICT:", verdict)

    with open(args.out, "w", newline="") as f:
        f.write("# IGLA-Coder generator-axis ablation -- phi is the axis "
                "origin, not a claim; phi^2 + phi^-2 = 3\n")
        w = csv.DictWriter(
            f, fieldnames=["generator", "seed", "code_val_bpb", "beta1",
                           "weight_decay", "lr_mult", "steps", "hidden",
                           "fim_loss"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
