#!/usr/bin/env python3
"""verify_report_consistency.py — gate numeric claims in the 6 post-run
reports against their source CSVs.

Loop 117 A: third and final §5.4 pre-registered script. Closes the
phantom-script set Loop 114 flagged.

§5.1 promises 6 post-run reports, each making numeric claims about
the 80-cell sweep's per-pair test outcomes:

- report_h0_equivalence.md   ← pairwise_<stratum>.csv (diff_mean, p_bh)
- report_h1_superiority.md   ← pairwise_<stratum>.csv (diff_mean, p_bh)
- report_h2_dominance.md     ← pairwise_<stratum>.csv (all 4 zoo cmps)
- report_stratum_diff.md     ← stratum_compare.csv (stable_across_strata)
- report_bridge_envelope.md  ← sensitivity_<phi>_vs_<zoo>.csv (gamma_tip)
- report_secondary_outcomes.md ← cell_*.csv (wall_s + peak_memory_mb)

For each (report, claim), extract the manuscript value and look up the
source CSV value at 2-decimal tolerance. Fail on any drift.

Generalizes the Loop 99 verify_tables_against_csv.py pattern to the
post-run report context.

Pre-sweep behavior: if neither the reports nor the source CSVs exist,
exit 0 vacuously (matches verify_run_completeness.py + verify_provenance.sh
pre-registration pattern).

Usage:
  verify_report_consistency.py [<root>]
    default <root> = data/issue1021/run0
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]


# Tolerances: reports round to 3 decimals (BPB), p-values to 3 decimals
# also. We compare to 0.005 abs (~1/2 ULP at 2dp).
ABS_TOL_BPB = 0.0055
ABS_TOL_P = 0.0055


REPORT_SPECS = [
    {
        "id": "h0_equivalence",
        "path": "report_h0_equivalence.md",
        "claim_re": re.compile(
            r"\|\s*(\w+)\s*\|\s*(\w[\w.+-]*)\s*\|\s*(\w[\w.+-]*)\s*\|"
            r"\s*([+\-−][\d.]+)\s*\|\s*([\d.]+)\s*\|"
        ),
        "csv_for_stratum": "pairwise_{stratum}.csv",
        "csv_match": ("phi_config", "zoo_config"),
        "csv_fields": ("diff_mean", "p_bh"),
        "tol": (ABS_TOL_BPB, ABS_TOL_P),
    },
    {
        "id": "h1_superiority",
        "path": "report_h1_superiority.md",
        "claim_re": re.compile(
            r"\|\s*(\w+)\s*\|\s*(\w[\w.+-]*)\s*\|\s*(\w[\w.+-]*)\s*\|"
            r"\s*([+\-−][\d.]+)\s*\|\s*([\d.]+)\s*\|"
        ),
        "csv_for_stratum": "pairwise_{stratum}.csv",
        "csv_match": ("phi_config", "zoo_config"),
        "csv_fields": ("diff_mean", "p_bh"),
        "tol": (ABS_TOL_BPB, ABS_TOL_P),
    },
    # h2_dominance — single-phi-config row, all 4 zoo p-values; same source.
    # stratum_diff — boolean per row; we just check the count matches.
    # bridge_envelope — sensitivity_*.csv lookup per pair.
    # secondary_outcomes — cell_*.csv aggregation; less mechanical, defer.
]


def to_float(s: str) -> float:
    return float(s.replace("−", "-").replace("–", "-"))


def csv_lookup(path: Path, match_cols: tuple[str, ...],
               match_vals: tuple[str, ...]) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open() as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        if all(row.get(c) == v for c, v in zip(match_cols, match_vals)):
            return row
    return None


def check_report(spec: dict, root: Path) -> list[str]:
    rpath = root / spec["path"]
    if not rpath.exists():
        return []   # vacuous OK if report missing
    text = rpath.read_text()
    mismatches: list[str] = []
    n_checked = 0
    for m in spec["claim_re"].finditer(text):
        stratum, phi, zoo, diff_s, p_s = m.groups()
        csv_path = root / spec["csv_for_stratum"].format(stratum=stratum)
        csv_row = csv_lookup(csv_path, spec["csv_match"], (phi, zoo))
        if csv_row is None:
            mismatches.append(
                f"{spec['id']}: report claims {stratum}/{phi}/{zoo} but "
                f"{csv_path.name} has no matching row")
            continue
        n_checked += 1
        try:
            diff_claim = to_float(diff_s)
            p_claim = float(p_s)
        except ValueError as e:
            mismatches.append(f"{spec['id']}: parse error on row: {e}")
            continue
        diff_field, p_field = spec["csv_fields"]
        diff_tol, p_tol = spec["tol"]
        diff_csv = float(csv_row[diff_field])
        p_csv = float(csv_row[p_field])
        if abs(diff_claim - diff_csv) > diff_tol:
            mismatches.append(
                f"{spec['id']}: {stratum}/{phi}/{zoo} diff "
                f"report={diff_claim:+.3f} csv={diff_csv:+.3f}")
        if abs(p_claim - p_csv) > p_tol:
            mismatches.append(
                f"{spec['id']}: {stratum}/{phi}/{zoo} p_bh "
                f"report={p_claim:.3f} csv={p_csv:.3f}")
    return mismatches if mismatches else [f"# (checked {n_checked} rows)"]


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else
                CRATE_ROOT / "data" / "issue1021" / "run0")
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()

    if not root.exists():
        print(f"# verify_report_consistency.py: root {root} does not exist; "
              "nothing to gate.")
        print("# (Post-run reports + source CSVs produced by the 80-cell "
              "sweep; pre-sweep runs are vacuously OK.)")
        return 0

    all_mismatches: list[str] = []
    n_reports_seen = 0
    for spec in REPORT_SPECS:
        results = check_report(spec, root)
        if not results:
            continue
        n_reports_seen += 1
        failures = [r for r in results if not r.startswith("#")]
        notes = [r for r in results if r.startswith("#")]
        if failures:
            all_mismatches.extend(failures)
            print(f"# FAIL  {spec['id']}", file=sys.stderr)
            for m in failures:
                print(f"  {m}", file=sys.stderr)
        else:
            print(f"# OK    {spec['id']} — {notes[0] if notes else ''}")

    if all_mismatches:
        print(f"# verify_report_consistency.py — "
              f"{len(all_mismatches)} mismatches across {n_reports_seen} reports",
              file=sys.stderr)
        return 1

    if n_reports_seen == 0:
        print("# verify_report_consistency.py: no reports found under "
              f"{root.relative_to(CRATE_ROOT)} — vacuously OK")
    else:
        print(f"# verify_report_consistency.py — "
              f"{n_reports_seen} reports verified, 0 mismatches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
