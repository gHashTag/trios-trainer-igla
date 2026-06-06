#!/usr/bin/env python3
"""verify_quire_microbench_freshness.py — gate the F2 §9.4.2 regime
map against silent loss of its backing JSON.

Loop 153 follow-up to the format_microbench-freshness gate (Loop 149,
stage 41): the §9.4.2 dot-product-accuracy table is now a load-bearing
claim in the manuscript, so the JSON outputs of the
`quire_microbench` binary must stay on disk and parse to the expected
schema. If a future commit silently deletes them — or the binary
output diverges from what §9.4.2 reports — this stage fires.

What this gate asserts:
  1. `.trinity/results/quire_microbench_summary_seeds_<lo>-<hi>.json`
     exists for at least one (lo, hi) pair.
  2. The summary file parses as JSON with the expected envelope:
     `tool == "quire_microbench"`, `mode == "summary"`,
     `regimes` includes both "xavier" and "cancellation",
     `lengths` includes 64/256/1024/4096,
     `rel_err_by_regime_and_length` is a nested dict.
  3. Per-seed JSONs `quire_microbench_seed<S>.json` exist for every
     seed in `summary["seeds"]`.

Out of scope:
  - Comparing rel-error values against a tolerance (the table in
    §9.4.2 is in scientific-notation ranges; a tolerance gate would
    need a versioned baseline sidecar).
  - Schema validation of per-seed JSON contents (next-loop ratchet).

Usage: papers/scripts/verify_quire_microbench_freshness.py

Exit 0 on clean; 1 on missing/malformed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = CRATE_ROOT / ".trinity" / "results"

SUMMARY_PATTERN = re.compile(
    r"quire_microbench_summary_seeds_(\d+)-(\d+)\.json$"
)
EXPECTED_REGIMES = {"xavier", "cancellation"}
EXPECTED_LENGTHS = {64, 256, 1024, 4096}


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(CRATE_ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    if not RESULTS_DIR.exists():
        print(f"# FAIL  {_rel(RESULTS_DIR)} missing. Run "
              "`./target/release/quire_microbench` to populate.",
              file=sys.stderr)
        return 1

    summaries = sorted(
        p for p in RESULTS_DIR.iterdir()
        if SUMMARY_PATTERN.search(p.name)
    )
    if not summaries:
        print(f"# FAIL  no quire-microbench summary JSON found in "
              f"{_rel(RESULTS_DIR)}/. Expected at least one "
              "`quire_microbench_summary_seeds_<lo>-<hi>.json`.",
              file=sys.stderr)
        return 1

    summary_path = summaries[-1]
    try:
        data = json.loads(summary_path.read_text())
    except json.JSONDecodeError as e:
        print(f"# FAIL  {summary_path.name} unparseable: {e}",
              file=sys.stderr)
        return 1

    mismatches: list[str] = []
    if data.get("tool") != "quire_microbench":
        mismatches.append(
            f"summary `tool` is {data.get('tool')!r}, expected "
            "'quire_microbench'")
    if data.get("mode") != "summary":
        mismatches.append(
            f"summary `mode` is {data.get('mode')!r}, expected "
            "'summary'")

    seeds = data.get("seeds", [])
    if not seeds:
        mismatches.append("summary `seeds` is missing or empty")
    regimes = set(data.get("regimes", []))
    missing_r = EXPECTED_REGIMES - regimes
    if missing_r:
        mismatches.append(
            f"summary `regimes` missing: {sorted(missing_r)}")

    lengths = set(data.get("lengths", []))
    missing_l = EXPECTED_LENGTHS - lengths
    if missing_l:
        mismatches.append(
            f"summary `lengths` missing: {sorted(missing_l)}")

    by_rl = data.get("rel_err_by_regime_and_length", {})
    if not isinstance(by_rl, dict):
        mismatches.append(
            f"summary `rel_err_by_regime_and_length` not a dict")
    else:
        for regime in EXPECTED_REGIMES & regimes:
            regime_row = by_rl.get(regime, {})
            for L in EXPECTED_LENGTHS & lengths:
                key = f"L{L}"
                if key not in regime_row:
                    mismatches.append(
                        f"summary[{regime}][{key}] missing")

    # Per-seed JSON existence.
    for s in seeds:
        per_seed = RESULTS_DIR / f"quire_microbench_seed{s}.json"
        if not per_seed.exists():
            mismatches.append(
                f"per-seed JSON {per_seed.name} missing")

    if mismatches:
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(f"# verify_quire_microbench_freshness.py — "
              f"{len(mismatches)} freshness drift(s) on "
              f"{summary_path.name}", file=sys.stderr)
        return 1

    print(f"# verify_quire_microbench_freshness.py — "
          f"{summary_path.name}: {len(seeds)} seed(s) × "
          f"{len(EXPECTED_REGIMES)} regime(s) × "
          f"{len(EXPECTED_LENGTHS)} length(s) "
          f"= {len(seeds) * len(EXPECTED_REGIMES) * len(EXPECTED_LENGTHS)} "
          "cells, schema OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
