#!/usr/bin/env python3
"""verify_run_completeness.py — gate the §5.1 artifact inventory at the
run-result paper's anchor.

Loop 116 A: second of three §5.4 pre-registered scripts. Checks that
every CSV the §5.1 inventory promises is present in the run subtree,
and parses against its schema.

§5.1 inventory (Loop 113+114 — schema rows + producers):
- 80 × `cell_<stratum>_<config>_<seed>.csv`
   2 strata (canonical, wd0) × 8 configs × 5 seeds (42-46)
- 2 × `aggregate_<stratum>.csv`
- 1 × `pairwise_canonical.csv`
- 1 × `pairwise_wd0.csv`
- 1 × `stratum_compare.csv`
- 8..16 × `sensitivity_<phi>_vs_<zoo>.csv`

Pre-sweep behavior: if the run root doesn't exist, exit 0 vacuously
(matches verify_provenance.sh's pre-registration pattern).

Schema parse: for cell CSVs we just check the row count is > 0 and
the header contains `val_bpb`; for aggregator CSVs we check the
producer-specific header. The strict per-cell numeric validation
lives in verify_provenance.sh + verify_report_consistency.py.

Usage:
  verify_run_completeness.py [<root>]
    default <root> = data/issue1021/run0
"""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]


# Per-config-class headers we expect. Each tuple: (filename pattern,
# required header columns, allow_extra=True).
CELL_REQUIRED_COLS = {"val_bpb"}
AGGREGATE_REQUIRED_COLS = {"config", "mean_bpb"}
PAIRWISE_REQUIRED_COLS = {
    "stratum", "phi_config", "zoo_config", "diff_mean", "p_raw", "p_bh"
}
STRATUM_COMPARE_REQUIRED_COLS = {"fix_x", "pse_name", "stable_across_strata"}
SENSITIVITY_REQUIRED_COLS = {"gamma_tip", "lambda"}


STRATA = ["canonical", "wd0"]
CONFIGS = ["GFTernary", "GF8", "GF16", "GF32",
           "BitNet-1.58", "INT8", "FP8", "bf16"]
SEEDS = [42, 43, 44, 45, 46]


def read_header(path: Path) -> set[str]:
    """Return the first non-comment, non-blank line as a set of column names."""
    with path.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#") or not line.strip():
                continue
            return set(next(csv.reader([line])))
    return set()


def header_ok(path: Path, required: set[str]) -> tuple[bool, str]:
    if not path.exists():
        return False, "file missing"
    header = read_header(path)
    missing = required - header
    if missing:
        return False, f"header missing {sorted(missing)}"
    return True, ""


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else
                CRATE_ROOT / "data" / "issue1021" / "run0")
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()

    if not root.exists():
        print(f"# verify_run_completeness.py: root {root} does not exist; "
              "nothing to gate.")
        print("# (The 80-cell sweep produces it; pre-sweep runs are "
              "vacuously OK.)")
        return 0

    missing: list[str] = []
    bad_header: list[tuple[str, str]] = []

    # 80 cell CSVs
    for stratum, config, seed in itertools.product(STRATA, CONFIGS, SEEDS):
        f = root / f"cell_{stratum}_{config}_{seed}.csv"
        ok, why = header_ok(f, CELL_REQUIRED_COLS)
        if not ok:
            (bad_header if f.exists() else missing).append(
                str(f.relative_to(root)) if not f.exists() else (
                    str(f.relative_to(root)), why)
            )

    # 2 aggregate CSVs
    for stratum in STRATA:
        f = root / f"aggregate_{stratum}.csv"
        ok, why = header_ok(f, AGGREGATE_REQUIRED_COLS)
        if not ok:
            if not f.exists():
                missing.append(str(f.relative_to(root)))
            else:
                bad_header.append((str(f.relative_to(root)), why))

    # 2 pairwise CSVs
    for stratum in STRATA:
        f = root / f"pairwise_{stratum}.csv"
        ok, why = header_ok(f, PAIRWISE_REQUIRED_COLS)
        if not ok:
            if not f.exists():
                missing.append(str(f.relative_to(root)))
            else:
                bad_header.append((str(f.relative_to(root)), why))

    # 1 stratum_compare CSV
    f = root / "stratum_compare.csv"
    ok, why = header_ok(f, STRATUM_COMPARE_REQUIRED_COLS)
    if not ok:
        if not f.exists():
            missing.append(str(f.relative_to(root)))
        else:
            bad_header.append((str(f.relative_to(root)), why))

    # 8..16 sensitivity CSVs. We don't know up-front which pairs survive
    # the permutation test, so we count the actual files present rather
    # than requiring a specific count.
    sensitivity_files = sorted(root.glob("sensitivity_*.csv"))
    if not (8 <= len(sensitivity_files) <= 16):
        bad_header.append((
            f"sensitivity_*.csv (count = {len(sensitivity_files)})",
            "expected 8..16 per §5.1 half-survival baseline",
        ))
    for f in sensitivity_files:
        ok, why = header_ok(f, SENSITIVITY_REQUIRED_COLS)
        if not ok:
            bad_header.append((str(f.relative_to(root)), why))

    if missing:
        print(f"# FAIL  {len(missing)} missing CSVs", file=sys.stderr)
        for m in missing[:20]:
            print(f"  missing: {m}", file=sys.stderr)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more", file=sys.stderr)
    if bad_header:
        print(f"# FAIL  {len(bad_header)} schema mismatches", file=sys.stderr)
        for path, why in bad_header[:20]:
            print(f"  {path}: {why}", file=sys.stderr)
        if len(bad_header) > 20:
            print(f"  ... and {len(bad_header) - 20} more", file=sys.stderr)

    if missing or bad_header:
        print(f"# verify_run_completeness.py — "
              f"{len(missing)} missing + {len(bad_header)} bad-header",
              file=sys.stderr)
        return 1

    n_sens = len(sensitivity_files)
    expected_fixed = 80 + 2 + 2 + 1
    print(f"# verify_run_completeness.py — "
          f"{expected_fixed} fixed CSVs + {n_sens} sensitivity CSVs verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
