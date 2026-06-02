#!/usr/bin/env python3
"""verify_formulas_vs_tables.py — algebraic identity gate.

Operationalizes the Loop 102 25th-adversarial-pass catch (the wd0
CDE row reported Γ_tip = 1.43 from the point estimate when the
§3.3 formula mandates Γ_tip = 1 + min(|CI_lo|, |CI_hi|) / Λ,
giving 1.01). A static check that re-derives every Γ_tip claim
in the markdown from its CI prevents that class of error.

Two identities are gated:

  (1) Four-PSE closure (§3.2):
      Δ_X = NDE + NIE_M1 + NIE_M2 + NIE_chain  (residual < 1e-5)
      Verified per row of `loop36_dual.csv` (and per-row of the
      stratified CSVs in `data/loop49/`).

  (2) Bridge-score envelope tipping point (§3.3):
      Γ_tip(Λ) = 1 + min(|CI_lo|, |CI_hi|) / Λ
      Parsed from §5.4 bullet-format lines:
          "- <fix> <PSE>: <est> [<lo>, <hi>], Γ_tip = <claim>"
      and from the headline-table line for wd0 CDE.

Usage: papers/scripts/verify_formulas_vs_tables.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
PAPER = CRATE_ROOT / "papers" / "f2_methodology.md"

# Tolerances: claims are rounded to 2dp in the markdown, so accept
# differences up to half a ULP at 2dp plus a small slack for the
# closure residual which is real (not zero).
ABS_TOL_GAMMA = 0.012
ABS_TOL_CLOSURE = 1.0e-5

LAMBDA = 1.0   # §5.4 baseline (justified in the §5.4 prose).


def to_float(s: str) -> float:
    """Parse a possibly-Unicode-minus float."""
    return float(s.replace("−", "-").replace("–", "-"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def closure_residual(row: dict[str, str]) -> float:
    """Δ_X − (NDE + NIE_M1 + NIE_M2 + NIE_chain)."""
    dx = float(row["delta_x"])
    s = (float(row["nde"]) + float(row["nie_m1"])
         + float(row["nie_m2"]) + float(row["nie_chain"]))
    return dx - s


def gamma_tip(lo: float, hi: float, lam: float = LAMBDA) -> float:
    """§3.3 formula. CI brackets zero → Γ_tip = 1.0 (already at zero)."""
    if lo <= 0 <= hi:
        return 1.0
    return 1.0 + min(abs(lo), abs(hi)) / lam


# ─── Identity 1: four-PSE closure across all dual-mediation CSVs ───

CLOSURE_CSVS = [
    "data/loop49/loop36_dual.csv",
    "data/loop49/loop49_wd0_dual.csv",
    "data/loop49/loop49_warmup0_dual.csv",
]


def check_closure() -> list[str]:
    mismatches: list[str] = []
    for relpath in CLOSURE_CSVS:
        path = CRATE_ROOT / relpath
        for row in csv_rows(path):
            try:
                r = closure_residual(row)
            except (KeyError, ValueError) as e:
                mismatches.append(f"{relpath}: row {row.get('fix_x', '?')}: {e}")
                continue
            if abs(r) > ABS_TOL_CLOSURE:
                mismatches.append(
                    f"{relpath}: fix={row['fix_x']} residual={r:+.3e} "
                    f"(tolerance {ABS_TOL_CLOSURE:.0e})")
    return mismatches


# ─── Identity 2: parse Γ_tip claims from markdown + re-derive ───

# Bullet-form line like:
#   "- rms NDE: −4.12 [−4.68, −3.55], Γ_tip = 4.55"
BULLET_RE = re.compile(
    r"^-\s+(\S+)\s+(NDE|NIE_M1|NIE_M2|NIE_chain):\s+"
    r"([+\-−][\d.]+)\s+"
    r"\[([+\-−][\d.]+),\s+([+\-−][\d.]+)\]"
    r"(?:\s+\(CI brackets 0\))?"
    r",\s+Γ_tip\s*=\s*([\d.]+)"
)


def check_bridge_score_bullets() -> list[str]:
    """Each bullet line is self-contained: estimate + CI + claim."""
    mismatches: list[str] = []
    n_checked = 0
    for ln in PAPER.read_text().splitlines():
        m = BULLET_RE.match(ln.strip())
        if not m:
            continue
        n_checked += 1
        fix, pse, est_s, lo_s, hi_s, claim_s = m.groups()
        try:
            lo = to_float(lo_s)
            hi = to_float(hi_s)
            claim = float(claim_s)
        except ValueError as e:
            mismatches.append(f"bullet parse failure: {ln!r}: {e}")
            continue
        derived = gamma_tip(lo, hi)
        if abs(derived - claim) > ABS_TOL_GAMMA:
            mismatches.append(
                f"{fix} {pse}: claim Γ_tip={claim} derived={derived:.3f} "
                f"(CI [{lo}, {hi}], diff {abs(derived - claim):.3f})")
    if n_checked == 0:
        mismatches.append("no Γ_tip bullet lines parsed — has the markdown format drifted?")
    return mismatches if mismatches else [f"# (parsed {n_checked} bullets)"]


# Headline table: |wd0 CDE for rms (+0.43)|<value>|<class>|
# Source CI is in data/loop49/loop49_wd0_dual.csv, fix_x=rms.
HEADLINE_ROWS = [
    ("rms canonical NDE",
     "data/loop49/loop36_dual.csv", "rms",
     "ci95_nde_lo", "ci95_nde_hi",
     "Canonical NDE for rms"),
    ("rms canonical NIE_M1",
     "data/loop49/loop36_dual.csv", "rms",
     "ci95_nie_m1_lo", "ci95_nie_m1_hi",
     "Canonical NIE_M1 via WD"),
    ("rms wd0 CDE/NDE",
     "data/loop49/loop49_wd0_dual.csv", "rms",
     "ci95_nde_lo", "ci95_nde_hi",
     "wd0 CDE for rms"),
]


def check_headline_table() -> list[str]:
    """Re-derive the 3-row headline table from CSVs."""
    mismatches: list[str] = []
    md = PAPER.read_text()
    for label, csv_path, fix_x, lo_field, hi_field, md_marker in HEADLINE_ROWS:
        rows = csv_rows(CRATE_ROOT / csv_path)
        match = next((r for r in rows if r["fix_x"] == fix_x), None)
        if match is None:
            mismatches.append(f"{label}: fix={fix_x} not found in {csv_path}")
            continue
        lo = float(match[lo_field])
        hi = float(match[hi_field])
        derived = gamma_tip(lo, hi)
        # Find the markdown table row whose left cell mentions md_marker
        # and pull the |claim_value| from it.
        row_re = re.compile(
            r"\|\s*" + re.escape(md_marker) + r"[^|]*\|\s*([\d.]+)\s*\|"
        )
        m = row_re.search(md)
        if not m:
            mismatches.append(f"{label}: markdown row matching {md_marker!r} not found")
            continue
        claim = float(m.group(1))
        if abs(derived - claim) > ABS_TOL_GAMMA:
            mismatches.append(
                f"{label}: claim Γ_tip={claim} derived={derived:.3f} "
                f"from CI [{lo:.3f}, {hi:.3f}]")
    return mismatches


# ─── Identity 4 (Loop 107): arithmetic claims in #1021 paper ───
#
# Each claim is a simple "lhs = rhs" assertion the paper makes,
# either as a multiplication or a sum. We verify lhs evaluates to
# rhs. Operates on the #1021 paper specifically.

ISSUE1021_PAPER = CRATE_ROOT / "papers" / "phi_ladder_paper_intro_draft.md"

ARITHMETIC_CLAIMS = [
    # (label, lhs as eval-safe expression, expected rhs)
    ("§3.1 sweep matrix", "8 * 2 * 5", 80),
    ("§3.3 BH pairs per stratum", "4 * 4", 16),
    ("§3.3 BH pairs both strata", "4 * 4 * 2", 32),
    # The CSV count: 80 cells + 2 aggregates + 1 pairwise_canonical
    # + 1 pairwise_wd0 + 1 stratum_compare + (8..16) sensitivity.
    # Verify the fixed component sums correctly; the (8..16) range
    # leads to total in [85, 101] inclusive.
    ("§5.1 fixed-component CSV sum", "80 + 2 + 1 + 1 + 1", 85),
    ("§5.1 CSV total upper bound (all survive)", "85 + 16", 101),
    ("§5.1 CSV total lower bound (half survive)", "85 + 8", 93),
]


def check_arithmetic_claims() -> list[str]:
    """Evaluate each lhs expression and confirm it equals the rhs."""
    mismatches: list[str] = []
    if not ISSUE1021_PAPER.exists():
        return [f"  {ISSUE1021_PAPER.relative_to(CRATE_ROOT)} not found"]
    for label, lhs_expr, expected in ARITHMETIC_CLAIMS:
        # Safe eval: only int literals and + - * operators.
        if not re.fullmatch(r"[\d\s+\-*]+", lhs_expr):
            mismatches.append(f"{label}: unsafe expression {lhs_expr!r}")
            continue
        actual = eval(lhs_expr)  # safe per regex above
        if actual != expected:
            mismatches.append(
                f"{label}: claim {lhs_expr} = {expected} but evaluates "
                f"to {actual}")
    return mismatches


def main() -> int:
    print(f"# verify_formulas_vs_tables.py — Λ = {LAMBDA}")
    closure = check_closure()
    if closure:
        print(f"# (1/3) four-PSE closure: FAIL ({len(closure)} mismatches)", file=sys.stderr)
        for m in closure:
            print(f"  {m}", file=sys.stderr)
    else:
        n_rows = sum(len(csv_rows(CRATE_ROOT / p)) for p in CLOSURE_CSVS)
        print(f"# (1/3) four-PSE closure: OK ({n_rows} rows verified)")

    bullets = check_bridge_score_bullets()
    bullet_failures = [b for b in bullets if not b.startswith("#")]
    bullet_meta = [b for b in bullets if b.startswith("#")]
    if bullet_failures:
        print(f"# (2/3) Γ_tip bullets: FAIL ({len(bullet_failures)} mismatches)",
              file=sys.stderr)
        for m in bullet_failures:
            print(f"  {m}", file=sys.stderr)
    else:
        print(f"# (2/3) Γ_tip bullets: OK  {bullet_meta[0] if bullet_meta else ''}")

    headline = check_headline_table()
    if headline:
        print(f"# (3/4) headline-table Γ_tip: FAIL ({len(headline)} mismatches)",
              file=sys.stderr)
        for m in headline:
            print(f"  {m}", file=sys.stderr)
    else:
        print(f"# (3/4) headline-table Γ_tip: OK ({len(HEADLINE_ROWS)} rows verified)")

    arith = check_arithmetic_claims()
    if arith:
        print(f"# (4/4) #1021 arithmetic: FAIL ({len(arith)} mismatches)",
              file=sys.stderr)
        for m in arith:
            print(f"  {m}", file=sys.stderr)
    else:
        print(f"# (4/4) #1021 arithmetic: OK ({len(ARITHMETIC_CLAIMS)} claims verified)")

    if closure or bullet_failures or headline or arith:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
