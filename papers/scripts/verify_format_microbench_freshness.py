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
GRID_CONFIG = (
    CRATE_ROOT / "papers" / "scripts" / "format_microbench_grid_config.json"
)


def _load_config() -> tuple[set[str], set[int], list[int]] | str:
    """Load the shared grid config — the single source of truth for what
    cells the binary is *supposed* to produce. Falls back to documented
    defaults if the file is missing (binary has the same fallback)."""
    default_inits = {"xavier", "he", "normal_002"}
    default_d_models = {128, 384, 768, 1024}
    default_seeds = [42, 43, 44, 45, 46]
    if not GRID_CONFIG.exists():
        return default_inits, default_d_models, default_seeds
    try:
        cfg = json.loads(GRID_CONFIG.read_text())
    except json.JSONDecodeError as e:
        return f"grid config {GRID_CONFIG.name} unparseable: {e}"
    inits = set(cfg.get("inits", []))
    d_models = set(cfg.get("d_models", []))
    seeds = cfg.get("seeds", [])
    # 73rd-pass SEV-4 closure: an explicitly empty list in the config is
    # a misconfiguration, not a "no expectations" signal. We FAIL loudly
    # rather than silently report 0 cells as success — the gate is here
    # to catch grid-going-missing, and an empty grid IS the missing case.
    if cfg.get("inits") == [] or cfg.get("d_models") == [] or cfg.get("seeds") == []:
        return ("grid config has at least one empty list "
                "(inits/d_models/seeds): an empty grid is treated as "
                "misconfiguration. Either fix the config or remove the "
                "file to fall back to documented defaults.")
    inits = inits or default_inits
    d_models = d_models or default_d_models
    seeds = seeds or default_seeds
    return inits, d_models, list(seeds)


# Loop 151 A: hardcoded EXPECTED_* sets removed — derived from the
# shared grid config above instead.
SUMMARY_PATTERN = re.compile(
    r"format_microbench_grid_summary_seeds_(\d+)-(\d+)\.json$"
)
CELL_PATTERN = re.compile(
    r"d(\d+)_([A-Za-z_0-9]+)_seed(\d+)\.json$"
)


def find_summaries() -> list[Path]:
    if not GRID_DIR.exists():
        return []
    return [
        p for p in sorted(GRID_DIR.iterdir())
        if SUMMARY_PATTERN.search(p.name)
    ]


def _validate_cell(path: Path, expected_d: int, expected_init: str,
                   expected_seed: int) -> list[str]:
    """Per-cell JSON schema check. Loop 151 B closure of 71st-pass SEV-4.
    Loop 151 also folds the 73rd-pass SEV-2 closure: asserts the JSON's
    own (d_model, init, seed) fields match the values encoded in the
    filename. Without this check, a cell file named `d128_xavier_seed42.json`
    could contain data for `d=256, init=he, seed=43` and still pass.
    Returns a list of mismatch strings (empty == clean)."""
    out: list[str] = []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"{path.name}: unparseable JSON ({e})"]
    if data.get("tool") != "format_microbench":
        out.append(f"{path.name}: tool field is "
                   f"{data.get('tool')!r}, expected 'format_microbench'")
    if data.get("mode") != "grid_cell":
        out.append(f"{path.name}: mode field is "
                   f"{data.get('mode')!r}, expected 'grid_cell'")
    for k in ("init", "d_model", "seed", "vocab", "cell"):
        if k not in data:
            out.append(f"{path.name}: missing top-level key `{k}`")
    # Filename ↔ content consistency (73rd-pass SEV-2 closure).
    if data.get("d_model") != expected_d:
        out.append(f"{path.name}: content d_model = "
                   f"{data.get('d_model')} but filename encodes "
                   f"d_model = {expected_d}")
    if data.get("init") != expected_init:
        out.append(f"{path.name}: content init = "
                   f"{data.get('init')!r} but filename encodes "
                   f"init = {expected_init!r}")
    if data.get("seed") != expected_seed:
        out.append(f"{path.name}: content seed = "
                   f"{data.get('seed')} but filename encodes "
                   f"seed = {expected_seed}")
    cell = data.get("cell", {}) if isinstance(data, dict) else {}
    if not isinstance(cell, dict):
        return out + [f"{path.name}: `cell` is not an object"]
    headline = cell.get("headline", {})
    if not isinstance(headline, dict):
        out.append(f"{path.name}: missing or non-object `cell.headline`")
        return out
    required_headline = (
        "rel_l2_gf16", "rel_l2_posit16", "rel_l2_bf16",
        "delta_posit16_vs_gf16", "delta_posit16_vs_bf16",
    )
    for k in required_headline:
        if k not in headline:
            out.append(f"{path.name}: missing `cell.headline.{k}`")
        elif not isinstance(headline[k], (int, float)):
            out.append(f"{path.name}: `cell.headline.{k}` is "
                       f"{type(headline[k]).__name__}, expected number")
    return out


def main() -> int:
    cfg = _load_config()
    if isinstance(cfg, str):
        print(f"# FAIL  {cfg}", file=sys.stderr)
        return 1
    expected_inits, expected_d_models, expected_seeds = cfg

    if not GRID_DIR.exists():
        print(f"# FAIL  {GRID_DIR.relative_to(CRATE_ROOT)} missing. Run "
              "`./target/release/format_microbench --grid` to populate "
              "(the binary reads the grid config "
              "`papers/scripts/format_microbench_grid_config.json`).",
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
        for init in expected_inits:
            if init not in delta_dict:
                mismatches.append(
                    f"summary delta_posit16_vs_gf16 missing init "
                    f"`{init}`")
            else:
                d_row = delta_dict[init]
                for d in expected_d_models:
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

    # Per-cell JSON existence + schema validation. The schema check is the
    # Loop 151 B closure of 71st-pass SEV-4 — previously the gate only
    # asserted file presence, so a corrupted cell would slip through.
    if seeds:
        missing_cells: list[str] = []
        schema_errors: list[str] = []
        for d in expected_d_models:
            for init in expected_inits:
                for s in seeds:
                    cell_path = GRID_DIR / f"d{d}_{init}_seed{s}.json"
                    if not cell_path.exists():
                        missing_cells.append(cell_path.name)
                    else:
                        schema_errors.extend(
                            _validate_cell(cell_path, d, init, s)
                        )
        if missing_cells:
            shown = ", ".join(sorted(missing_cells)[:6])
            extra = "" if len(missing_cells) <= 6 else (
                f" (… and {len(missing_cells) - 6} more)"
            )
            mismatches.append(
                f"{len(missing_cells)} per-cell JSON(s) missing: "
                f"{shown}{extra}. Re-run the grid to repopulate.")
        if schema_errors:
            shown_errs = "; ".join(schema_errors[:4])
            more = "" if len(schema_errors) <= 4 else (
                f" (… and {len(schema_errors) - 4} more)"
            )
            mismatches.append(
                f"{len(schema_errors)} per-cell schema violation(s): "
                f"{shown_errs}{more}")

    if mismatches:
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(f"# verify_format_microbench_freshness.py — "
              f"{len(mismatches)} freshness drift(s) on "
              f"{summary_path.name}", file=sys.stderr)
        return 1

    n_inits = len(expected_inits)
    n_d = len(expected_d_models)
    n_seeds = len(seeds)
    n_cells_expected = n_inits * n_d * n_seeds
    print(f"# verify_format_microbench_freshness.py — "
          f"{summary_path.name}: {n_cells_expected} cell(s) "
          f"({n_inits} inits × {n_d} d_models × {n_seeds} seeds) "
          f"per grid config; summary + per-cell schemas OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
