"""Emit the wd0-stratum coder ablation as F2 long-form CSV.

wd0 stratum pins weight_decay = 0 (the dominant mediator from the canonical
analysis). With decay removed, the only manipulated mediator is momentum
(beta1: 0.9 -> phi^-1). The PSE here is a Pearl Controlled Direct Effect of
momentum along the non-decay paths, NOT a marginal NDE -- hence the
`# INPUT STRATUM = wd0` banner and CDE-framing line.
"""
import csv

SHA = "9f8a8a1"
B1_OFF, B1_ON = 0.9, 0.6180339887

# wd pinned to 0 for both arms. key = m1_momentum (m2_decay always 0).
DATA = {
    0: {42: 3.7767, 43: 3.6408, 44: 3.6165},  # standard, b1=0.9
    1: {42: 3.4530, 43: 3.3494, 44: 3.4277},  # phi_b1,   b1=phi^-1
}

HEADER = [
    "mode", "arm", "seed", "x_phi_prior",
    "m1_momentum", "m2_decay", "beta1", "wd",
    "hidden", "n_steps", "lr", "code_val_bpb",
]


def main(path):
    with open(path, "w", newline="") as f:
        f.write(f"# W3C-PROV: source = trios-trainer-igla@{SHA}, "
                f"mode = optimizer_factorial_wd0\n")
        f.write("# INPUT STRATUM = wd0\n")
        f.write("# CDE-framing: wd0 stratum -- weight_decay pinned to 0; the "
                "momentum PSE is a Pearl CDE along non-decay paths\n")
        f.write("# generator = igla_coder ablation; outcome = code_val_bpb "
                "(bits-per-byte, lower better)\n")
        w = csv.writer(f)
        w.writerow(HEADER)
        for m1 in (0, 1):
            arm = "phi_b1" if m1 else "standard"
            b1 = B1_ON if m1 else B1_OFF
            for seed in sorted(DATA[m1]):
                w.writerow([
                    "optimizer_factorial_wd0", arm, seed, m1,
                    m1, 0, f"{b1:.10f}", "0.0000000000",
                    64, 300, 0.03, f"{DATA[m1][seed]:.4f}",
                ])


if __name__ == "__main__":
    main("coder_ablation_f2_wd0.csv")
    with open("coder_ablation_f2_wd0.csv") as f:
        print(f.read())
