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


def csv_lookup_multi(path: Path, key_cols: dict[str, str]) -> dict[str, str]:
    """Like csv_lookup but matches on multiple (column, value) pairs."""
    with path.open() as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        if all(row.get(k) == v for k, v in key_cols.items()):
            return row
    raise KeyError(f"{key_cols} not matched in {path}")


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


# Tables 3 + 4 (Loop 100) — single CSV row, multiple stratum columns
# (Table 3) and per-M_2 CSVs (Table 4). Different topology from the
# tables above, handled by their own validators below.
STRATUM_TABLE_REGISTRY = [
    {
        "id": "table_3_swap_nie_m1_3stratum",
        "anchor": "| stratum     | X = wd, NIE_M1 via rms (95% CI)",
        "csv": "data/loop49_swap/3stratum_swap.csv",
        "row_filter": {"fix_x": "wd", "pse_name": "NIE_M1"},
        # (markdown row label, csv estimate field, ci_lo field, ci_hi field)
        "stratum_rows": [
            ("canonical", "estimate_canonical", "ci95_lo_canonical", "ci95_hi_canonical"),
            ("wd0",       "estimate_wd0",       "ci95_lo_wd0",       "ci95_hi_wd0"),
            ("warmup0",   "estimate_warmup0",   "ci95_lo_warmup0",   "ci95_hi_warmup0"),
        ],
    },
]

# Table 4 (M_2 robustness grid): rows are M_2 alternative names; the
# warmup row sources from 3stratum_swap.csv (single CSV, stratum
# columns), the other 4 rows source from per-M_2 CSV files (one per
# stratum × M_2 = 12 CSVs total).
PER_M2_TABLE_REGISTRY = [
    {
        "id": "table_4_m2_robustness",
        "anchor": "| M_2 = …    | canonical",
        "rows": {
            # M_2 = warmup uses the 3stratum CSV with stratum-suffixed cols
            "warmup": {
                "csv_per_stratum": {
                    "canonical": ("data/loop49_swap/3stratum_swap.csv",
                                  {"fix_x": "wd", "pse_name": "NIE_M1"},
                                  "estimate_canonical", "ci95_lo_canonical", "ci95_hi_canonical"),
                    "wd0":      ("data/loop49_swap/3stratum_swap.csv",
                                  {"fix_x": "wd", "pse_name": "NIE_M1"},
                                  "estimate_wd0", "ci95_lo_wd0", "ci95_hi_wd0"),
                    "warmup0":  ("data/loop49_swap/3stratum_swap.csv",
                                  {"fix_x": "wd", "pse_name": "NIE_M1"},
                                  "estimate_warmup0", "ci95_lo_warmup0", "ci95_hi_warmup0"),
                },
            },
            # M_2 ∈ {gradclip, clamp, smooth, dropout} use per-stratum
            # per-M2 CSV files (canonical_swap_m2{m}.csv etc.) with the
            # standard dual-mediation schema (nie_m1 + CI fields).
            **{
                m: {
                    "csv_per_stratum": {
                        "canonical": (f"data/loop49_swap/canonical_swap_m2{m}.csv",
                                      {"fix_x": "wd"},
                                      "nie_m1", "ci95_nie_m1_lo", "ci95_nie_m1_hi"),
                        "wd0":      (f"data/loop49_swap/wd0_swap_m2{m}.csv",
                                      {"fix_x": "wd"},
                                      "nie_m1", "ci95_nie_m1_lo", "ci95_nie_m1_hi"),
                        "warmup0":  (f"data/loop49_swap/warmup0_swap_m2{m}.csv",
                                      {"fix_x": "wd"},
                                      "nie_m1", "ci95_nie_m1_lo", "ci95_nie_m1_hi"),
                    },
                }
                for m in ("gradclip", "clamp", "smooth", "dropout")
            },
        },
        "stratum_order": ["canonical", "wd0", "warmup0"],
    },
]


def validate_stratum_table(spec: dict) -> list[str]:
    md = PAPER.read_text().splitlines()
    csv_row = csv_lookup_multi(CRATE_ROOT / spec["csv"], spec["row_filter"])
    row_labels = [r[0] for r in spec["stratum_rows"]]
    table = extract_table_rows(md, spec["anchor"], row_labels)
    mismatches: list[str] = []
    for row_label, est_field, lo_field, hi_field in spec["stratum_rows"]:
        if row_label not in table:
            mismatches.append(f"{spec['id']}: row {row_label!r} missing")
            continue
        cells = table[row_label]
        if not cells:
            mismatches.append(f"{spec['id']}: row {row_label!r} empty")
            continue
        try:
            pt, lo, hi = parse_cell(cells[0])
        except ValueError as e:
            mismatches.append(f"{spec['id']}: {row_label!r}: {e}")
            continue
        csv_pt = float(csv_row[est_field])
        csv_lo = float(csv_row[lo_field])
        csv_hi = float(csv_row[hi_field])
        if not near(pt, csv_pt):
            mismatches.append(
                f"{spec['id']}: {row_label!r} estimate paper={pt:+.3f} csv={csv_pt:+.3f}")
        if lo is not None and not near(lo, csv_lo):
            mismatches.append(
                f"{spec['id']}: {row_label!r} CI_lo paper={lo:+.3f} csv={csv_lo:+.3f}")
        if hi is not None and not near(hi, csv_hi):
            mismatches.append(
                f"{spec['id']}: {row_label!r} CI_hi paper={hi:+.3f} csv={csv_hi:+.3f}")
    return mismatches


def validate_per_m2_table(spec: dict) -> list[str]:
    md = PAPER.read_text().splitlines()
    row_labels = list(spec["rows"].keys())
    table = extract_table_rows(md, spec["anchor"], row_labels)
    mismatches: list[str] = []
    for row_label in row_labels:
        if row_label not in table:
            mismatches.append(f"{spec['id']}: row {row_label!r} missing")
            continue
        cells = table[row_label]
        row_cfg = spec["rows"][row_label]
        for col_idx, stratum in enumerate(spec["stratum_order"]):
            if col_idx >= len(cells):
                mismatches.append(
                    f"{spec['id']}: {row_label!r} missing col {stratum}")
                continue
            try:
                pt, lo, hi = parse_cell(cells[col_idx])
            except ValueError as e:
                mismatches.append(f"{spec['id']}: {row_label!r}/{stratum}: {e}")
                continue
            csv_path, filt, est_f, lo_f, hi_f = row_cfg["csv_per_stratum"][stratum]
            try:
                csv_row = csv_lookup_multi(CRATE_ROOT / csv_path, filt)
            except KeyError as e:
                mismatches.append(f"{spec['id']}: {row_label!r}/{stratum}: {e}")
                continue
            csv_pt = float(csv_row[est_f])
            csv_lo = float(csv_row[lo_f])
            csv_hi = float(csv_row[hi_f])
            if not near(pt, csv_pt):
                mismatches.append(
                    f"{spec['id']}: {row_label!r}/{stratum} estimate "
                    f"paper={pt:+.3f} csv={csv_pt:+.3f}")
            if lo is not None and not near(lo, csv_lo):
                mismatches.append(
                    f"{spec['id']}: {row_label!r}/{stratum} CI_lo "
                    f"paper={lo:+.3f} csv={csv_lo:+.3f}")
            if hi is not None and not near(hi, csv_hi):
                mismatches.append(
                    f"{spec['id']}: {row_label!r}/{stratum} CI_hi "
                    f"paper={hi:+.3f} csv={csv_hi:+.3f}")
    return mismatches


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
    total_tables = 0
    for spec in TABLE_REGISTRY:
        total_tables += 1
        ms = validate_table(spec)
        if ms:
            all_mismatches.extend(ms)
            print(f"# {spec['id']}: FAIL ({len(ms)} mismatches)", file=sys.stderr)
            for m in ms:
                print(f"  {m}", file=sys.stderr)
        else:
            print(f"# {spec['id']}: OK ({len(spec['rows'])} rows × "
                  f"{len(spec['cols'])} cols verified)")
    for spec in STRATUM_TABLE_REGISTRY:
        total_tables += 1
        ms = validate_stratum_table(spec)
        if ms:
            all_mismatches.extend(ms)
            print(f"# {spec['id']}: FAIL ({len(ms)} mismatches)", file=sys.stderr)
            for m in ms:
                print(f"  {m}", file=sys.stderr)
        else:
            print(f"# {spec['id']}: OK ({len(spec['stratum_rows'])} "
                  f"stratum rows verified)")
    for spec in PER_M2_TABLE_REGISTRY:
        total_tables += 1
        ms = validate_per_m2_table(spec)
        if ms:
            all_mismatches.extend(ms)
            print(f"# {spec['id']}: FAIL ({len(ms)} mismatches)", file=sys.stderr)
            for m in ms:
                print(f"  {m}", file=sys.stderr)
        else:
            n_cells = len(spec["rows"]) * len(spec["stratum_order"])
            print(f"# {spec['id']}: OK ({n_cells} M_2 × stratum cells verified)")
    if all_mismatches:
        print(f"# verify_tables_against_csv.py — {len(all_mismatches)} "
              f"mismatches across {total_tables} tables", file=sys.stderr)
        return 1
    print(f"# verify_tables_against_csv.py — {total_tables} "
          f"tables verified, 0 mismatches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
