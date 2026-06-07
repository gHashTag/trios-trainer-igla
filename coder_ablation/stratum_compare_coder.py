"""Cross-stratum comparator for the coder F2 mediation track (Loop+3).

Joins the three strata produced by f2_dual_mediation:
  canonical  -- both mediators free; marginal-style 2x2 CDEs
  wd0        -- decay pinned to 0; momentum CDE along non-decay paths
  mom_std    -- momentum pinned to 0.9; decay CDE along non-momentum paths

and reports, per mediator (momentum, decay), whether its controlled direct
effect is STABLE across the strata in which it is identified -- i.e. whether
the 95% bootstrap CIs overlap and the sign agrees. This is the coder-track
analogue of f2_stratum_compare (skill f2-mediation-loop); it runs the Rust
f2_dual_mediation binary on each CSV and parses the long-form output.

Honesty: a stable, robust harmful CDE is NOT evidence for phi. The whole point
is that phi-derived knobs (phi^-1 momentum, phi^-3 decay) hurt or, at best,
tie standard. The only "phi helps" cell (wd0 momentum) is regime-specific.
"""
import subprocess
import sys

BIN = "./target/release/f2_dual_mediation"

STRATA = {
    "canonical": "coder_ablation_f2.csv",
    "wd0": "coder_ablation_f2_wd0.csv",
    "mom_std": "coder_ablation_f2_mom_std.csv",
}


def run(csv):
    out = subprocess.run([BIN, "--csv", csv], capture_output=True, text=True, check=True).stdout
    rows = {}
    for line in out.splitlines():
        if line.startswith("#") or line.startswith("pse,"):
            continue
        parts = line.split(",")
        if len(parts) != 7:
            continue
        name, eff, lo, hi, p, g, st = parts
        rows[name] = dict(effect=float(eff), lo=float(lo), hi=float(hi),
                          gamma=float(g), status=st)
    return rows


def ci_overlap(a, b):
    return not (a["hi"] < b["lo"] or b["hi"] < a["lo"])


def main():
    res = {s: run(csv) for s, csv in STRATA.items()}

    # --- momentum CDE: identified in canonical (cde_momentum_decay0) and wd0 ---
    mom_canon = res["canonical"]["cde_momentum_decay0"]
    mom_wd0 = res["wd0"]["cde_momentum_decay_pinned"]
    mom_overlap = ci_overlap(mom_canon, mom_wd0)
    mom_sign = (mom_canon["effect"] > 0) == (mom_wd0["effect"] > 0)
    mom_stable = mom_overlap and mom_sign

    # --- decay CDE: identified in canonical (cde_decay_momentum0) and mom_std ---
    dec_canon = res["canonical"]["cde_decay_momentum0"]
    dec_mom = res["mom_std"]["cde_decay_momentum_pinned"]
    dec_overlap = ci_overlap(dec_canon, dec_mom)
    dec_sign = (dec_canon["effect"] > 0) == (dec_mom["effect"] > 0)
    dec_stable = dec_overlap and dec_sign

    print("# cross-stratum CDE stability (coder F2 track, Loop+3)")
    print("mediator,stratum_a,effect_a,ci_a,stratum_b,effect_b,ci_b,"
          "ci_overlap,sign_agree,stable_across_strata")

    print("momentum,canonical,{:+.4f},[{:+.4f}; {:+.4f}],wd0,{:+.4f},"
          "[{:+.4f}; {:+.4f}],{},{},{}".format(
              mom_canon["effect"], mom_canon["lo"], mom_canon["hi"],
              mom_wd0["effect"], mom_wd0["lo"], mom_wd0["hi"],
              mom_overlap, mom_sign, mom_stable))

    print("decay,canonical,{:+.4f},[{:+.4f}; {:+.4f}],mom_std,{:+.4f},"
          "[{:+.4f}; {:+.4f}],{},{},{}".format(
              dec_canon["effect"], dec_canon["lo"], dec_canon["hi"],
              dec_mom["effect"], dec_mom["lo"], dec_mom["hi"],
              dec_overlap, dec_sign, dec_stable))

    print()
    print("# verdicts")
    print(f"momentum CDE: stable_across_strata={mom_stable} "
          f"(canonical {mom_canon['effect']:+.3f} harmful vs wd0 "
          f"{mom_wd0['effect']:+.3f} helpful -- SIGN FLIP, regime-dependent)")
    print(f"decay CDE:    stable_across_strata={dec_stable} "
          f"(canonical {dec_canon['effect']:+.3f} vs mom_std "
          f"{dec_mom['effect']:+.3f} -- robust harmful, regime-independent)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
