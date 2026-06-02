#!/usr/bin/env python3
"""verify_preamble_per_producer.py — gate the §5.1 per-producer preamble
field list against actual binary source emissions.

Loop 114 C operationalizes the §5.1 claim that each producer binary
emits a specific set of `# prov:` fields. A static grep over each
binary's source compares the *actual* emission set against the
registered *expected* set; mismatches fail the gate.

Background:
- Loop 113 D rewrote §5.1 to scope the preamble field set per producer
  class (cell-level vs aggregator). The 37th adversarial pass (Loop
  114 A) was queued to verify the rewrite is accurate.
- This script discovers, statically, that some "aggregator" binaries
  emit NO preamble at all — §5.1's claim that aggregator CSVs carry
  the generatedAt/wasGeneratedBy/agent_git_sha/host/schema/version
  set is partially aspirational.

Registry entry shape: (binary_path, expected_fields).
- expected_fields = [] means "no preamble emitted; §5.1 must NOT
  claim a preamble for this binary".
- expected_fields = ["generatedAt", ...] means "binary must emit
  EXACTLY these fields (in some order) and §5.1 must list them".

Usage: papers/scripts/verify_preamble_per_producer.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]

# Pattern: `# prov:fieldname = ...` inside a writeln! call.
PROV_FIELD_RE = re.compile(r"#\s*prov:([a-zA-Z_]+)\s*=")


REGISTRY: list[tuple[str, list[str]]] = [
    # Cell-level CSV: full preamble per F2 protocol
    ("src/bin/f2_ablation_sweep.rs", [
        "generatedAt", "wasGeneratedBy", "agent_git_sha", "host",
        "trainer_internals_schema", "cargo_pkg_version",
    ]),
    # #1021 protocol contribution: same preamble set
    ("src/bin/f2_pairwise_perm.rs", [
        "generatedAt", "wasGeneratedBy", "agent_git_sha", "host",
        "phi_configs", "zoo_configs",
        "trainer_internals_schema", "cargo_pkg_version",
    ]),
    # Aggregator binaries — historically no preamble emitted; the §5.1
    # paragraph (post-Loop 114) must acknowledge this honestly.
    ("src/bin/f2_ablation_aggregate.rs", []),
    ("src/bin/f2_stratum_compare.rs", []),
    ("src/bin/f2_mediation_sensitivity.rs", []),
    ("src/bin/f2_dual_mediation.rs", []),
]


def emitted_fields(path: Path) -> list[str]:
    """Static grep: return the ordered list of prov:* field names the
    binary's source writes."""
    if not path.exists():
        return []
    text = path.read_text()
    return PROV_FIELD_RE.findall(text)


def check_entry(rel: str, expected: list[str]) -> list[str]:
    path = CRATE_ROOT / rel
    actual = emitted_fields(path)
    actual_set = set(actual)
    expected_set = set(expected)
    mismatches: list[str] = []
    missing = expected_set - actual_set
    extra = actual_set - expected_set
    for f in sorted(missing):
        mismatches.append(
            f"{rel}: missing # prov:{f} (expected per §5.1 / Loop 114 registry)")
    for f in sorted(extra):
        mismatches.append(
            f"{rel}: emits # prov:{f} but registry didn't expect it "
            "(silent extra field — update §5.1 or the registry)")
    return mismatches


def main() -> int:
    all_mismatches: list[str] = []
    rows_ok = 0
    for rel, expected in REGISTRY:
        ms = check_entry(rel, expected)
        if ms:
            all_mismatches.extend(ms)
            print(f"# FAIL  {rel}", file=sys.stderr)
            for m in ms:
                print(f"  {m}", file=sys.stderr)
        else:
            rows_ok += 1
            status = "no preamble (acknowledged)" if not expected \
                else f"{len(expected)} fields verified"
            print(f"# OK    {rel} — {status}")
    if all_mismatches:
        print(f"# verify_preamble_per_producer.py — "
              f"{len(all_mismatches)} mismatches across "
              f"{len(REGISTRY)} producers", file=sys.stderr)
        return 1
    print(f"# verify_preamble_per_producer.py — "
          f"{rows_ok}/{len(REGISTRY)} producers verified, 0 drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
