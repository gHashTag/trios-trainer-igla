#!/usr/bin/env python3
"""verify_bridge_bench_freshness.py — gate the F2 §9.4.3 sandbox-
scale training comparison against silent loss of its backing JSON.

Loop 155 follow-up to the quire-microbench-freshness gate
(Loop 153, stage 42): §9.4.3 reports the val BPB of f32 vs GF16 vs
Posit16 on a tiny bigram LM. The numbers in the paper depend on
the JSON outputs of `src/bin/bridge_bench` being present and
schema-correct.

Same catch-class as stage 41 and 42: this is a *drift catcher*,
not a direct submission-blocker. The camera-ready PDF would ship
unchanged even if every JSON in `.trinity/results/` were deleted.
The gate makes such deletion visible between loops, keeping the
reproducibility provenance honest. Discipline tier per the catch-
class taxonomy documented in `verify_quire_microbench_freshness.py`.

What this gate asserts:
  1. `.trinity/results/bridge_bench_summary_seeds_<lo>-<hi>.json`
     exists with the expected envelope (tool, mode, seeds,
     formats, val_bpb_by_format).
  2. Per-seed JSONs exist for every seed in summary["seeds"].
  3. The three formats `{"f32", "gf16", "posit16"}` are all present
     in summary["formats"].
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = CRATE_ROOT / ".trinity" / "results"
SUMMARY_PATTERN = re.compile(
    r"bridge_bench_summary_seeds_(\d+)-(\d+)\.json$"
)
EXPECTED_FORMATS = {"f32", "gf16", "posit16"}


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(CRATE_ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    if not RESULTS_DIR.exists():
        print(f"# FAIL  {_rel(RESULTS_DIR)} missing. Run "
              "`./target/release/bridge_bench` to populate.",
              file=sys.stderr)
        return 1

    summaries = sorted(
        p for p in RESULTS_DIR.iterdir()
        if SUMMARY_PATTERN.search(p.name)
    )
    if not summaries:
        print(f"# FAIL  no bridge-bench summary JSON found in "
              f"{_rel(RESULTS_DIR)}/. Expected at least one "
              "`bridge_bench_summary_seeds_<lo>-<hi>.json`.",
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
    if data.get("tool") != "bridge_bench":
        mismatches.append(
            f"summary `tool` is {data.get('tool')!r}, expected "
            "'bridge_bench'")
    if data.get("mode") != "summary":
        mismatches.append(
            f"summary `mode` is {data.get('mode')!r}, expected "
            "'summary'")
    seeds = data.get("seeds", [])
    if not seeds:
        mismatches.append("summary `seeds` is missing or empty")
    formats = set(data.get("formats", []))
    missing_f = EXPECTED_FORMATS - formats
    if missing_f:
        mismatches.append(
            f"summary `formats` missing: {sorted(missing_f)}")

    by_fmt = data.get("val_bpb_by_format", {})
    if not isinstance(by_fmt, dict):
        mismatches.append(
            "summary `val_bpb_by_format` not a dict")
    else:
        for fmt in EXPECTED_FORMATS & formats:
            cell = by_fmt.get(fmt, {})
            if not isinstance(cell.get("mean"), (int, float)):
                mismatches.append(
                    f"summary[{fmt}].mean is not numeric")
            if not isinstance(cell.get("std"), (int, float)):
                mismatches.append(
                    f"summary[{fmt}].std is not numeric")

    for s in seeds:
        per_seed = RESULTS_DIR / f"bridge_bench_seed{s}.json"
        if not per_seed.exists():
            mismatches.append(
                f"per-seed JSON {per_seed.name} missing")

    if mismatches:
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(f"# verify_bridge_bench_freshness.py — "
              f"{len(mismatches)} freshness drift(s) on "
              f"{summary_path.name}", file=sys.stderr)
        return 1

    print(f"# verify_bridge_bench_freshness.py — "
          f"{summary_path.name}: {len(seeds)} seed(s) × "
          f"{len(EXPECTED_FORMATS)} format(s) = "
          f"{len(seeds) * len(EXPECTED_FORMATS)} cells, schema OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
