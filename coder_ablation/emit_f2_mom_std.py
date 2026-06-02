"""Emit the mom_std-stratum coder ablation as F2 long-form CSV.

mom_std stratum pins momentum to standard (beta1 = 0.9). With momentum fixed,
the only manipulated mediator is weight decay (wd: 0.04 -> phi^-3 ~ 0.23607).
The PSE here is a Pearl Controlled Direct Effect of decay along the
non-momentum paths, NOT a marginal NDE -- hence the
`# INPUT STRATUM = mom_std` banner and CDE-framing line.

Design note (Loop+3): the originally proposed "warmup0" stratum was dropped
because the coder trainer has NO learning-rate warmup or schedule (lr is
constant); pinning a non-existent mediator would violate the f2-mediation-loop
rule that a stratum must pin a real mediator carrying >= 50% of the indirect
effect. The decay mediator carries the dominant IE (canonical CDE_decay
~ +3.54 BPB), so mom_std (pin momentum, vary decay) is the honest symmetric
completion of the 2x2 factorial begun by wd0 (pin decay, vary momentum).
"""
import csv

SHA = "7abf019"
B1_STD = 0.9
WD_OFF, WD_ON = 0.04, 0.2360679775  # baseline decay vs phi^-3

# momentum pinned to 0.9 for both cells. key = m2_decay (m1_momentum always 0).
DATA = {
    0: {42: 4.3111, 43: 4.5282, 44: 4.3456},  # wd = 0.04   (baseline decay)
    1: {42: 7.9469, 43: 7.9743, 44: 7.8957},  # wd = phi^-3 (phi decay)
}

HEADER = [
    "mode", "arm", "seed", "x_phi_prior",
    "m1_momentum", "m2_decay", "beta1", "wd",
    "hidden", "n_steps", "lr", "code_val_bpb",
]


def main(path):
    with open(path, "w", newline="") as f:
        f.write(f"# W3C-PROV: source = trios-trainer-igla@{SHA}, "
                f"mode = optimizer_factorial_mom_std\n")
        f.write("# INPUT STRATUM = mom_std\n")
        f.write("# CDE-framing: mom_std stratum -- beta1 pinned to 0.9; the "
                "decay PSE is a Pearl CDE along non-momentum paths\n")
        f.write("# generator = igla_coder ablation; outcome = code_val_bpb "
                "(bits-per-byte, lower better)\n")
        w = csv.writer(f)
        w.writerow(HEADER)
        for m2 in (0, 1):
            arm = "phi_wd" if m2 else "standard"
            wd = WD_ON if m2 else WD_OFF
            for seed in sorted(DATA[m2]):
                w.writerow([
                    "optimizer_factorial_mom_std", arm, seed, m2,
                    0, m2, f"{B1_STD:.10f}", f"{wd:.10f}",
                    64, 300, 0.03, f"{DATA[m2][seed]:.4f}",
                ])


if __name__ == "__main__":
    main("coder_ablation_f2_mom_std.csv")
    with open("coder_ablation_f2_mom_std.csv") as f:
        print(f.read())
