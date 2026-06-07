#!/usr/bin/env python3
# IGLA-Coder generator-axis ablation, ISO-LR variant (Coder-Loop+8, wave A).
#
# THE QUESTION (refined from Loop+7)
#   Loop+7 found the four axes ranked STRICTLY by the learning-rate multiplier
#   each config_prior prescribes (standard lr_mult=1.0 best, phi 0.236, dyadic
#   0.125, e 0.050). That made the comparison a lr-sweep in disguise, not a test
#   of the (beta1, weight_decay) PRIOR. This harness controls that confound:
#   every arm trains at the SAME effective learning rate (lr_mult is dropped),
#   so the only thing that differs across arms is the (beta1, weight_decay) pair
#   the axis prescribes. Question becomes the clean one: "at iso-lr, is phi's
#   momentum/decay prior competitive with standard?"
#
# freeze -> hash -> null -> compare
#   freeze: fixed steps / hidden / seeds / data / EFFECTIVE lr for every arm.
#   hash:   record exact (beta1, wd, effective lr) per arm (provenance).
#   null:   `standard` is the null; phi must BEAT it (or at least tie) to be
#           supported as a momentum/decay prior independent of lr.
#   compare: mean BPB +/- 95% CI per arm; phi CI overlap with standard ->
#           phi NOT distinguishable from standard at iso-lr (reported first).
#
# HONESTY
#   This is NOT a hope that phi wins. The expected outcome is a near-tie: at
#   iso-lr the beta1/wd differences are second-order vs the lr that dominated
#   Loop+7. A tie is the honest, publishable "phi-as-axis is neither better nor
#   worse than standard once lr is controlled" statement. The method survives,
#   phi does not (yet).
#
# USAGE
#   python3 ablate_generator_axis_isolr.py --bin ../target/release/igla_coder \
#       --train ../data/code_train.bin --val ../data/code_val.bin \
#       --generators phi,dyadic,e,standard --seeds 42,43,44 \
#       --steps 800 --hidden 64 --lr 0.002 --fim-loss all \
#       --out coder_generator_axis_isolr.csv

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
    # ISO-LR: every arm uses args.lr directly. lr_mult is INTENTIONALLY ignored
    # so the only across-arm difference is the (beta1, weight_decay) prior. This
    # isolates the momentum/decay axis from the learning-rate confound that
    # dominated Loop+7.
    lr = args.lr
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
    ap.add_argument("--fim-loss", default="all", dest="fim_loss")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default="coder_generator_axis_isolr.csv")
    args = ap.parse_args()

    gens = [g.strip() for g in args.generators.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print("anchor: phi^2 + phi^-2 = 3")
    print("ISO-LR control: every arm at effective lr=%.6f (lr_mult IGNORED)"
          % args.lr)
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
                "effective_lr": "%.10f" % args.lr,
                "lr_mult_ignored": "%.10f" % prior["lr_mult"],
                "steps": args.steps, "hidden": args.hidden,
                "fim_loss": args.fim_loss,
            })
            print("  gen=%-8s seed=%d bpb=%.4f" % (g, s, bpb))
        mean, std, half = ci95(vals)
        summary[g] = (mean, std, half)
        print("  >>> gen=%-8s mean_bpb=%.4f std=%.4f ci95=+/-%.4f (n=%d)"
              % (g, mean, std, half, len(vals)))

    # compare: phi vs the null (standard) at iso-lr. negative-first verdict.
    if "phi" in summary and "standard" in summary:
        pm, _, ph = summary["phi"]
        sm, _, sh = summary["standard"]
        phi_lo, phi_hi = pm - ph, pm + ph
        std_lo, std_hi = sm - sh, sm + sh
        overlap = not (phi_hi < std_lo or std_hi < phi_lo)
        delta = pm - sm  # positive => phi is WORSE (higher BPB)
        if overlap:
            verdict = ("phi NOT distinguishable from standard at iso-lr: CIs "
                       "overlap (delta=%+.4f, within noise) -- the Loop+7 "
                       "gap was the lr confound, now controlled" % delta)
        elif delta > 0:
            verdict = ("phi WORSE than standard at iso-lr by %+.4f BPB (CIs "
                       "disjoint) -- the (beta1, wd) prior itself hurts, not "
                       "just lr" % delta)
        else:
            verdict = ("phi BETTER than standard at iso-lr by %.4f BPB (CIs "
                       "disjoint) -- the momentum/decay prior helps once lr is "
                       "controlled; warrants re-test, not yet a claim" % (-delta))
        print("VERDICT:", verdict)

    with open(args.out, "w", newline="") as f:
        f.write("# IGLA-Coder generator-axis ablation ISO-LR -- phi is the axis "
                "origin, not a claim; lr held equal across arms (lr_mult "
                "IGNORED); phi^2 + phi^-2 = 3\n")
        w = csv.DictWriter(
            f, fieldnames=["generator", "seed", "code_val_bpb", "beta1",
                           "weight_decay", "effective_lr", "lr_mult_ignored",
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
