"""Emit the IGLA-Coder optimizer ablation as F2-contract long-form CSV.

Design: 2x2 factorial. Treatment X = phi-prior applied to the AdamW optimizer.
Two mediators of the code-BPB outcome:
  m1_momentum : beta1   0.9   (off) -> 0.6180339887 = phi^-1 (on)
  m2_decay    : wd      0.04  (off) -> 0.2360679775 = phi^-3 (on)
One row per (arm x seed). Self-contained: does NOT depend on the absent
f2_ablation_sweep / f2_dual_mediation binaries (only f2_harness.rs is on this
branch). Stratum = canonical (no nuisance pinned); PSEs are marginal NDE/NIE.
ASCII-only snake_case headers per ESS-DIVE / Frictionless convention.
"""
import csv
import sys

SHA = "9f8a8a1"  # trios-trainer-igla @ feat/igla-coder-v1 HEAD
B1_OFF, B1_ON = 0.9, 0.6180339887       # phi^-1
WD_OFF, WD_ON = 0.04, 0.2360679775      # phi^-3

# arm -> (m1_momentum_on, m2_decay_on); cells of the 2x2 factorial.
ARM_DESIGN = {
    "standard": (0, 0),   # neither mediator: baseline corner
    "phi_b1":   (1, 0),   # momentum only
    "phi_wd":   (0, 1),   # decay only
    "phi":      (1, 1),   # both = phi-canonical corner (full treatment)
}

# Measured code_val_bpb, hidden=64 steps=300 lr=0.03 (this loop, fresh runs).
DATA = {
    "standard": {42: 4.3111, 43: 4.5282, 44: 4.3456},
    "phi_b1":   {42: 5.0999, 43: 5.5880, 44: 5.1622},
    "phi_wd":   {42: 7.9469, 43: 7.9743, 44: 7.8957},
    "phi":      {42: 6.3656, 43: 7.9718, 44: 7.4221},
}

HEADER = [
    "mode", "arm", "seed", "x_phi_prior",
    "m1_momentum", "m2_decay", "beta1", "wd",
    "hidden", "n_steps", "lr", "code_val_bpb",
]


def main(path):
    with open(path, "w", newline="") as f:
        # W3C-PROV + INPUT STRATUM preamble (mandatory, before header).
        f.write(f"# W3C-PROV: source = trios-trainer-igla@{SHA}, "
                f"mode = optimizer_factorial_2x2\n")
        f.write("# INPUT STRATUM = canonical\n")
        f.write("# CDE-framing: canonical stratum -- PSEs are marginal "
                "NDE/NIE (no nuisance pinned)\n")
        f.write("# generator = igla_coder ablation; outcome = code_val_bpb "
                "(bits-per-byte, lower better)\n")
        w = csv.writer(f)
        w.writerow(HEADER)
        for arm, (m1, m2) in ARM_DESIGN.items():
            x = 1 if (m1 or m2) else 0
            b1 = B1_ON if m1 else B1_OFF
            wd = WD_ON if m2 else WD_OFF
            for seed in sorted(DATA[arm]):
                w.writerow([
                    "optimizer_factorial", arm, seed, x,
                    m1, m2, f"{b1:.10f}", f"{wd:.10f}",
                    64, 300, 0.03, f"{DATA[arm][seed]:.4f}",
                ])
    # Echo for inspection.
    with open(path) as f:
        sys.stdout.write(f.read())


if __name__ == "__main__":
    main("coder_ablation_f2.csv")
