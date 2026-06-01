#!/usr/bin/env python3
"""verify_tables_against_csv.py — ground every numeric table in the
paper against its source CSV at 2-decimal tolerance.

Operationalizes the Loop 98 22nd-adversarial-pass catch (Table 2 §5.2
had wd0 NDE magnitudes ~10x too large). A static check that compares
each row of each load-bearing table against the labeled CSV at
data/loop49*/ prevents the class of error from shipping again.

Each registered table has:
  - a markdown anchor (first heading line to grep for)
  - a CSV path (relative to crate root)
  - a row-label → CSV-row-key map (markdown left-column → CSV fix_x)
  - a column-header → (CSV field, optional CI low/high fields) map

Mismatches at the 2-decimal level abort with a precise diff. To
register a new table, add it to TABLE_REGISTRY below.

Usage: papers/scripts/verify_tables_against_csv.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
PAPER = CRATE_ROOT / "papers" / "f2_methodology.md"


# Tolerances are sloppy by design: paper rounds to 2 decimal places, so
# match to within 0.005 (half a ULP at 2dp).
ABS_TOL = 0.006


def csv_lookup(path: Path, row_key: str, row_col: str) -> dict[str, str]:
    """Read a CSV and return the row where row[row_col] == row_key."""
    with path.open() as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        if row[row_col] == row_key:
            return row
    raise KeyError(f"{row_key} not found in {row_col} column of {path}")


def parse_cell(cell: str) -> tuple[float, float | None, float | None]:
    """Extract (point, ci_lo, ci_hi) from a markdown table cell.

    Accepts forms:
      "-4.12"
      "-4.12 [-4.68, -3.55]"
      "**-4.12 [-4.68, -3.55]**"
    Returns (point, lo, hi) where lo/hi may be None if no CI was present.
    """
    s = cell.replace("**", "").replace("−", "-").replace("−", "-")
    s = s.replace("[", " ").replace("]", " ").replace(",", " ")
    nums = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", s)
    if not nums:
        raise ValueError(f"no numbers in cell: {cell!r}")
    pt = float(nums[0])
    lo = float(nums[1]) if len(nums) >= 2 else None
    hi = float(nums[2]) if len(nums) >= 3 else None
    return pt, lo, hi


def near(a: float, b: float) -> bool:
    return abs(a - b) <= ABS_TOL


# Each entry is a self-describing table validator. The "anchor" is a
# unique substring that locates the table's header line; the validator
# walks data rows under that header.
TABLE_REGISTRY = [
    {
        "id": "table_1_canonical_pse",
        "anchor": "|         | NDE   | NIE_M1 | NIE_M2 | NIE_chain |",
        "csv": "data/loop49/loop36_dual.csv",
        "row_col": "fix_x",
        "rows": ["rms", "dropout", "gradclip", "clamp", "smooth"],
        "cols": [
            # (markdown column index after the row label, csv point field, csv lo, csv hi)
            (0, "nde", None, None),
            (1, "nie_m1", None, None),
            (2, "nie_m2", None, None),
            (3, "nie_chain", None, None),
        ],
    },
    {
        "id": "table_2_nde_canonical",
        "anchor": "| fix     | NDE_canonical (95% CI)",
        "csv": "data/loop49/loop36_dual.csv",
        "row_col": "fix_x",
        "rows": ["rms", "dropout", "gradclip", "clamp", "smooth"],
        "cols": [
            (0, "nde", "ci95_nde_lo", "ci95_nde_hi"),
        ],
    },
    {
        "id": "table_2_nde_wd0",
        "anchor": "| fix     | NDE_canonical (95% CI)",
        "csv": "data/loop49/loop49_wd0_dual.csv",
        "row_col": "fix_x",
        "rows": ["rms", "dropout", "gradclip", "clamp", "smooth"],
        "cols": [
            (1, "nde", "ci95_nde_lo", "ci95_nde_hi"),
        ],
    },
    {
        "id": "table_2_nde_warmup0",
        "anchor": "| fix     | NDE_canonical (95% CI)",
        "csv": "data/loop49/loop49_warmup0_dual.csv",
        "row_col": "fix_x",
        "rows": ["rms", "dropout", "gradclip", "clamp", "smooth"],
        "cols": [
            (2, "nde", "ci95_nde_lo", "ci95_nde_hi"),
        ],
    },
]


def extract_table_rows(md_lines: list[str], anchor: str,
                       expected_rows: list[str]) -> dict[str, list[str]]:
    """Find the table starting at 'anchor', return {row_label: [cells]}."""
    for i, line in enumerate(md_lines):
        if anchor in line:
            start = i
            break
    else:
        raise LookupError(f"anchor not found: {anchor!r}")
    # Skip header + separator
    j = start + 2
    out: dict[str, list[str]] = {}
    while j < len(md_lines) and md_lines[j].startswith("|"):
        cells = [c.strip() for c in md_lines[j].strip().strip("|").split("|")]
        # Row label is cells[0]; strip ** markdown
        label = cells[0].replace("**", "").strip()
        if label in expected_rows:
            out[label] = cells[1:]
        j += 1
    return out


def validate_table(spec: dict) -> list[str]:
    """Return list of mismatch messages (empty if all rows match)."""
    md = PAPER.read_text().splitlines()
    csv_path = CRATE_ROOT / spec["csv"]
    table = extract_table_rows(md, spec["anchor"], spec["rows"])
    mismatches: list[str] = []
    for row_label in spec["rows"]:
        if row_label not in table:
            mismatches.append(f"{spec['id']}: row {row_label!r} missing from markdown table")
            continue
        cells = table[row_label]
        try:
            csv_row = csv_lookup(csv_path, row_label, spec["row_col"])
        except KeyError as e:
            mismatches.append(f"{spec['id']}: {e}")
            continue
        for col_idx, pt_field, lo_field, hi_field in spec["cols"]:
            if col_idx >= len(cells):
                mismatches.append(
                    f"{spec['id']}: row {row_label!r} missing column index {col_idx}"
                )
                continue
            try:
                pt, lo, hi = parse_cell(cells[col_idx])
            except ValueError as e:
                mismatches.append(f"{spec['id']}: {row_label!r} col {col_idx}: {e}")
                continue
            csv_pt = float(csv_row[pt_field])
            if not near(pt, csv_pt):
                mismatches.append(
                    f"{spec['id']}: {row_label!r} col {col_idx} ({pt_field}) "
                    f"paper={pt:+.3f} csv={csv_pt:+.3f} (diff {abs(pt - csv_pt):.3f})"
                )
            if lo_field and lo is not None:
                csv_lo = float(csv_row[lo_field])
                if not near(lo, csv_lo):
                    mismatches.append(
                        f"{spec['id']}: {row_label!r} col {col_idx} CI_lo "
                        f"paper={lo:+.3f} csv={csv_lo:+.3f}"
                    )
            if hi_field and hi is not None:
                csv_hi = float(csv_row[hi_field])
                if not near(hi, csv_hi):
                    mismatches.append(
                        f"{spec['id']}: {row_label!r} col {col_idx} CI_hi "
                        f"paper={hi:+.3f} csv={csv_hi:+.3f}"
                    )
    return mismatches


def main() -> int:
    all_mismatches: list[str] = []
    for spec in TABLE_REGISTRY:
        ms = validate_table(spec)
        if ms:
            all_mismatches.extend(ms)
            print(f"# {spec['id']}: FAIL ({len(ms)} mismatches)", file=sys.stderr)
            for m in ms:
                print(f"  {m}", file=sys.stderr)
        else:
            print(f"# {spec['id']}: OK ({len(spec['rows'])} rows × "
                  f"{len(spec['cols'])} cols verified)")
    if all_mismatches:
        print(f"# verify_tables_against_csv.py — {len(all_mismatches)} "
              f"mismatches across {len(TABLE_REGISTRY)} tables",
              file=sys.stderr)
        return 1
    print(f"# verify_tables_against_csv.py — {len(TABLE_REGISTRY)} "
          f"tables verified, 0 mismatches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
