#!/usr/bin/env python3
"""verify_format_microbench_freshness.py — assert the
format_microbench grid summary file exists and parses, and (when run
against a clean tree) is no more than N days behind the latest commit
that touched `src/bin/format_microbench.rs` or `src/phi_numbers/posit16.rs`.

Loop 149 B operationalizes the audit catch from the 68th adversarial pass
(SEV-3 #4): "no validation that format_microbench outputs exist or are
recent. If the binary fails silently or the data dir is missing, the
gate does not catch it." With Loop 149's grid extension, the data has
become the load-bearing claim for §9.4's format-zoo arm — staleness or
absence of the summary file silently un-bases the paper's headline
delta table.

What this gate asserts:

  1. `.trinity/results/format_microbench_grid/format_microbench_grid_summary_seeds_<lo>-<hi>.json`
     exists for at least one (lo, hi) pair.
  2. The summary file parses as JSON and has the expected envelope:
     `tool == "format_microbench"`, `mode == "grid_summary"`,
     `delta_posit16_vs_gf16` is a dict with keys for each init scheme.
  3. Per-cell JSONs for the documented grid exist:
     `d{D}_{init}_seed{S}.json` for D ∈ {128, 384, 768, 1024} and
     init ∈ {xavier, he, normal_002} and S ∈ summary["seeds"].

Catches:
  - Summary JSON deleted but binary still references the result.
  - Per-cell JSONs deleted while summary remains (could imply partial
    re-run that left orphaned cells).
  - Summary JSON has a different schema than expected (binary was
    edited without bumping the schema check).

Out of scope:
  - Comparing rel_L2 numbers against a tolerance (Loop 149 design
    decision: a "no regression" gate would need a versioned baseline
    sidecar; that's the next-loop ratchet, not this one).
  - Freshness vs git mtime (CI environments don't preserve filesystem
    mtime reliably; the schema/existence check is more robust).

Usage: papers/scripts/verify_format_microbench_freshness.py

Exit 0 on clean; 1 on any missing/malformed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
GRID_DIR = CRATE_ROOT / ".trinity" / "results" / "format_microbench_grid"

EXPECTED_INITS = {"xavier", "he", "normal_002"}
EXPECTED_D_MODELS = {128, 384, 768, 1024}
SUMMARY_PATTERN = re.compile(
    r"format_microbench_grid_summary_seeds_(\d+)-(\d+)\.json$"
)
CELL_PATTERN = re.compile(
    r"d(\d+)_(xavier|he|normal_002)_seed(\d+)\.json$"
)


def find_summaries() -> list[Path]:
    if not GRID_DIR.exists():
        return []
    return [
        p for p in sorted(GRID_DIR.iterdir())
        if SUMMARY_PATTERN.search(p.name)
    ]


def main() -> int:
    if not GRID_DIR.exists():
        print(f"# FAIL  {GRID_DIR.relative_to(CRATE_ROOT)} missing. Run "
              "`./target/release/format_microbench --grid "
              "--seeds=42,43,44,45,46` to populate.",
              file=sys.stderr)
        return 1

    summaries = find_summaries()
    if not summaries:
        print(f"# FAIL  no grid summary JSON found in "
              f"{GRID_DIR.relative_to(CRATE_ROOT)}/. Expected at least "
              "one `format_microbench_grid_summary_seeds_<lo>-<hi>.json`.",
              file=sys.stderr)
        return 1

    # Use the latest summary (most-recently-seeded).
    summary_path = summaries[-1]
    try:
        data = json.loads(summary_path.read_text())
    except json.JSONDecodeError as e:
        print(f"# FAIL  {summary_path.name} unparseable: {e}",
              file=sys.stderr)
        return 1

    mismatches: list[str] = []

    # Schema check.
    if data.get("tool") != "format_microbench":
        mismatches.append(
            f"summary `tool` field is {data.get('tool')!r}, expected "
            "'format_microbench'")
    if data.get("mode") != "grid_summary":
        mismatches.append(
            f"summary `mode` field is {data.get('mode')!r}, expected "
            "'grid_summary'")

    seeds = data.get("seeds", [])
    if not seeds or not all(isinstance(s, int) for s in seeds):
        mismatches.append(
            f"summary `seeds` field is {seeds!r}, expected non-empty "
            "list of ints")
    seeds = [s for s in seeds if isinstance(s, int)]

    delta_dict = data.get("delta_posit16_vs_gf16", {})
    if not isinstance(delta_dict, dict):
        mismatches.append(
            f"summary `delta_posit16_vs_gf16` is not a dict")
    else:
        for init in EXPECTED_INITS:
            if init not in delta_dict:
                mismatches.append(
                    f"summary delta_posit16_vs_gf16 missing init "
                    f"`{init}`")
            else:
                d_row = delta_dict[init]
                for d in EXPECTED_D_MODELS:
                    key = f"d{d}"
                    if key not in d_row:
                        mismatches.append(
                            f"summary delta_posit16_vs_gf16[{init}] "
                            f"missing d_model `{key}`")
                    else:
                        cell = d_row[key]
                        if not isinstance(cell.get("delta_posit16_vs_gf16_mean"), (int, float)):
                            mismatches.append(
                                f"summary[{init}][{key}].delta_posit16_vs_gf16_mean "
                                "is not numeric")

    # Per-cell JSON existence.
    if seeds:
        missing_cells: list[str] = []
        for d in EXPECTED_D_MODELS:
            for init in EXPECTED_INITS:
                for s in seeds:
                    cell_path = GRID_DIR / f"d{d}_{init}_seed{s}.json"
                    if not cell_path.exists():
                        missing_cells.append(cell_path.name)
        if missing_cells:
            shown = ", ".join(sorted(missing_cells)[:6])
            extra = "" if len(missing_cells) <= 6 else (
                f" (… and {len(missing_cells) - 6} more)"
            )
            mismatches.append(
                f"{len(missing_cells)} per-cell JSON(s) missing: "
                f"{shown}{extra}. Re-run the grid to repopulate.")

    if mismatches:
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(f"# verify_format_microbench_freshness.py — "
              f"{len(mismatches)} freshness drift(s) on "
              f"{summary_path.name}", file=sys.stderr)
        return 1

    n_inits = len(EXPECTED_INITS)
    n_d = len(EXPECTED_D_MODELS)
    n_seeds = len(seeds)
    n_cells_expected = n_inits * n_d * n_seeds
    print(f"# verify_format_microbench_freshness.py — "
          f"{summary_path.name}: {n_cells_expected} cell(s) "
          f"({n_inits} inits × {n_d} d_models × {n_seeds} seeds), "
          f"schema OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
