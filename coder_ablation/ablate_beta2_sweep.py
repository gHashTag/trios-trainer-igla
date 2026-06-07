#!/usr/bin/env python3
# IGLA-Coder beta2 (second-moment) isolation sweep (Coder-Loop+10, C).
#
# THE QUESTION (the symmetric counterpart to the beta1 sweep)
#   Loop+9 controlled lr AND weight_decay and varied ONLY beta1 (first-moment
#   momentum); result was a TIE -- phi-momentum is NEUTRAL once lr+wd pinned.
#   That closed the beta1 axis. This harness asks the symmetric question for
#   the OTHER Adam moment: with lr, wd, AND beta1 all pinned to standard
#   (lr=0.002, wd=0.04, beta1=0.9), does varying beta2 (the second-moment /
#   RMS decay) -- specifically toward phi-native values -- help, hurt, or tie
#   the standard beta2=0.999?
#
#   This is the FIRST time beta2 is a free knob in this trainer (Loop+10 added
#   --beta2; before this it was hardcoded 0.999). It completes the per-knob
#   isolation grid: lr (Loop+7/8), wd (F2 mediation), beta1 (Loop+9), beta2
#   (here). grad_clip is NOT swept -- it is INERT in this trainer (printed in
#   the prior but never applied in opt_step), so sweeping it would be theatre.
#   We report grad_clip inert rather than fabricate a mediator (f2-mediation
#   rule: do NOT pin/vary a non-existent mediator).
#
# THE BETA2 GRID (all in (0,1), the valid Adam range)
#   0.999    standard AdamW (the null)
#   0.966    ~ 1 - phi^-7  (phi-native, still near 1)
#   0.764    ~ 1 - phi^-3  (phi-native, the phi^-3 decay anchor reflected)
#   0.618    = phi^-1      (phi-native, aggressive second-moment forgetting)
#   0.382    = 1/phi^2     (phi-native, very aggressive)
#
# freeze -> hash -> null -> compare
#   freeze: fixed steps / hidden / seeds / data / lr / wd / beta1 for every arm.
#   hash:   record exact beta2 per arm + the pinned (lr, wd, beta1).
#   null:   beta2=0.999 is the null; a phi-native beta2 must BEAT it to support
#           any phi second-moment prior.
#   compare: mean BPB +/- 95% CI per arm; CI overlap with the null -> that
#           beta2 is NOT distinguishable from standard (reported first).
#
# HONESTY
#   Expected outcome: standard beta2=0.999 wins or ties; aggressive phi-native
#   beta2 (0.382, 0.618) likely HURT (too much second-moment forgetting on a
#   tiny noisy CPU run). The honest publishable statement is most likely
#   "no phi-native beta2 beats standard 0.999" -- i.e. the second-moment knob
#   is also NOT a place where phi structure helps. Only phi^2 + phi^-2 = 3 is
#   [Verified]. The method survives, phi does not (yet).
#
# USAGE
#   python3 ablate_beta2_sweep.py --bin ../target/release/igla_coder \
#       --train ../data/code_train.bin --val ../data/code_val.bin \
#       --beta2s 0.999,0.966,0.764,0.618,0.382 --seeds 42,43,44 \
#       --steps 800 --hidden 64 --lr 0.002 --wd 0.04 --beta1 0.9 \
#       --fim-loss all --out coder_beta2_sweep.csv

import argparse
import csv
import hashlib
import math
import re
import subprocess

BPB_RE = re.compile(r"code_val_bpb=([\d.]+)")


def train_arm(args, beta2, seed):
    # ISO-EVERYTHING-ELSE: lr, wd, beta1 are ALL pinned to standard for every
    # arm. The ONLY across-arm difference is --beta2. This isolates the pure
    # second-moment prior, the symmetric counterpart to the Loop+9 beta1 sweep.
    cmd = [
        args.bin, "generate",
        "--train", args.train, "--val", args.val,
        "--optimizer", "standard",
        "--seed", str(seed),
        "--steps", str(args.steps), "--hidden", str(args.hidden),
        "--beta1", "%.10f" % args.beta1,  # PINNED, identical across arms
        "--beta2", "%.10f" % beta2,       # the ONLY varying knob
        "--wd", "%.10f" % args.wd,        # PINNED, identical across arms
        "--lr", "%.10f" % args.lr,        # PINNED, identical across arms
        "--lang-id", "1", "--max-new", "1",  # BPB only, no real sampling
    ]
    if args.fim_loss:
        cmd += ["--fim-loss", args.fim_loss]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    m = BPB_RE.search(out.stdout)
    if not m:
        raise RuntimeError("no bpb for beta2=%.4f seed %d: %s"
                           % (beta2, seed, out.stdout[-400:]))
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
    ap.add_argument("--beta2s", default="0.999,0.966,0.764,0.618,0.382")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--wd", type=float, default=0.04)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--fim-loss", default="all", dest="fim_loss")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default="coder_beta2_sweep.csv")
    args = ap.parse_args()

    beta2s = [float(b) for b in args.beta2s.split(",") if b.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    null_b2 = 0.999

    # phi-native annotation for the report
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    def annotate(b2):
        cands = {
            0.999: "standard (null)",
            round(1 - phi ** -7, 3): "~1-phi^-7",
            round(1 - phi ** -3, 3): "~1-phi^-3",
            round(phi ** -1, 3): "phi^-1",
            round(phi ** -2, 3): "1/phi^2",
        }
        return cands.get(round(b2, 3), "")

    print("anchor: phi^2 + phi^-2 = 3")
    print("ISO-EVERYTHING-ELSE control: every arm at lr=%.6f wd=%.6f beta1=%.6f "
          "(all standard); only beta2 varies" % (args.lr, args.wd, args.beta1))
    print("freeze: steps=%d hidden=%d seeds=%s fim=%s"
          % (args.steps, args.hidden, seeds, args.fim_loss))
    print("NOTE: grad_clip is INERT in this trainer (never applied in opt_step) "
          "-- NOT swept; reported inert in the loop report.")

    rows = []
    summary = {}
    for b2 in beta2s:
        vals = []
        for s in seeds:
            bpb = train_arm(args, b2, s)
            vals.append(bpb)
            rows.append({
                "beta2": "%.10f" % b2,
                "beta2_label": annotate(b2),
                "seed": s, "code_val_bpb": "%.4f" % bpb,
                "pinned_beta1": "%.10f" % args.beta1,
                "pinned_weight_decay": "%.10f" % args.wd,
                "pinned_lr": "%.10f" % args.lr,
                "steps": args.steps, "hidden": args.hidden,
                "fim_loss": args.fim_loss,
            })
            print("  beta2=%.4f (%-14s) seed=%d bpb=%.4f"
                  % (b2, annotate(b2), s, bpb))
        mean, std, half = ci95(vals)
        summary[b2] = (mean, std, half)
        print("  >>> beta2=%.4f mean_bpb=%.4f std=%.4f ci95=+/-%.4f (n=%d)"
              % (b2, mean, std, half, len(vals)))

    # compare: each phi-native beta2 vs the null (0.999). negative-first.
    if null_b2 in summary:
        nm, _, nh = summary[null_b2]
        n_lo, n_hi = nm - nh, nm + nh
        print("VERDICT (null = beta2=0.999, mean_bpb=%.4f):" % nm)
        any_better = False
        for b2 in beta2s:
            if b2 == null_b2:
                continue
            bm, _, bh = summary[b2]
            b_lo, b_hi = bm - bh, bm + bh
            overlap = not (b_hi < n_lo or n_hi < b_lo)
            delta = bm - nm  # positive => this beta2 is WORSE than null
            if overlap:
                tag = "NEUTRAL (CIs overlap)"
            elif delta > 0:
                tag = "WORSE (CIs disjoint)"
            else:
                tag = "BETTER (CIs disjoint) -- warrants re-test, not a claim"
                any_better = True
            print("  beta2=%.4f (%-14s) delta=%+.4f BPB vs null -> %s"
                  % (b2, annotate(b2), delta, tag))
        if not any_better:
            print("  SUMMARY: no phi-native beta2 beats standard 0.999 -- the "
                  "second-moment knob is NOT a place where phi structure helps. "
                  "Combined with Loop+9 (beta1 neutral) and F2 (wd is the real "
                  "driver, not phi), the per-knob isolation grid is closed for "
                  "phi-momentum/RMS. [Efit]")

    with open(args.out, "w", newline="") as f:
        f.write("# IGLA-Coder beta2 sweep -- lr, wd, beta1 ALL pinned standard; "
                "only beta2 (second-moment decay) varies; grad_clip INERT in "
                "this trainer, not swept; phi-native beta2 candidates are the "
                "axis origin, not a claim; phi^2 + phi^-2 = 3\n")
        w = csv.DictWriter(
            f, fieldnames=["beta2", "beta2_label", "seed", "code_val_bpb",
                           "pinned_beta1", "pinned_weight_decay", "pinned_lr",
                           "steps", "hidden", "fim_loss"])
        w.writeheader()
        w.writerows(rows)

    # provenance hash over the raw numeric rows (freeze -> hash)
    raw = "".join(
        "%s,%s,%s" % (r["beta2"], r["seed"], r["code_val_bpb"])
        for r in rows
    )
    digest = hashlib.sha256(raw.encode("ascii")).hexdigest()[:16]
    print("provenance sha256[:16] =", digest)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
