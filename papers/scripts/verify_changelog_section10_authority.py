#!/usr/bin/env python3
"""verify_changelog_section10_authority.py — assert CHANGELOG §10 is
the authoritative source of truth for `verify_*.py` stage additions.

Loop 143 C closes the "is §10 actually authoritative?" question. The
§10 discipline started around Loop 87 (first explicit per-loop gate
addition). Every `verify_*.py` stage added since should have a
corresponding §10 entry naming the gate filename. This gate:

  (a) Parses STAGES from `papers/scripts/run_all_checks.sh` for
      every `verify_*.py` entry.
  (b) Parses §10 for entries naming each gate filename.
  (c) Asserts each STAGES entry has a matching §10 entry (modulo
      legacy stages that predate the §10 discipline — allowlisted).

Legacy stages (added before Loop 87 §10 discipline) are exempted via
LEGACY_ALLOWLIST. New stages must land with §10 entries in the same
commit; the cascade discipline documented in GATE_AUTHORING_GUIDE.md
makes this part of the workflow.

Usage: papers/scripts/verify_changelog_section10_authority.py

Exit 0 if every non-legacy gate has a §10 entry; 1 on any missing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
RUN_ALL_CHECKS = CRATE_ROOT / "papers" / "scripts" / "run_all_checks.sh"
CHANGELOG = CRATE_ROOT / "papers" / "CHANGELOG.md"


# Gates that predate the §10 discipline (introduced Loop 87+).
# These are exempted from the §10-mention requirement.
LEGACY_ALLOWLIST = {
    "verify_paper_metadata.py",
    "verify_tables_against_csv.py",
    "verify_formulas_vs_tables.py",
    "verify_label_consistency.py",
    "verify_preamble_per_producer.py",
    "verify_run_completeness.py",
    "verify_report_consistency.py",
    "verify_provenance.sh",  # not .py but listed for symmetry
}


def parse_stages_gate_names() -> list[str] | str:
    """Return list of `verify_*.py` filenames from STAGES array."""
    if not RUN_ALL_CHECKS.exists():
        return f"missing {RUN_ALL_CHECKS.relative_to(CRATE_ROOT)}"
    text = RUN_ALL_CHECKS.read_text()
    m = re.search(r"STAGES=\(\s*\n(.*?)\n\)", text, re.DOTALL)
    if not m:
        return "STAGES=( ... ) block not found"
    body = m.group(1)
    return re.findall(r"papers/scripts/(verify_[A-Za-z0-9_]+\.py)", body)


def parse_section10_gates() -> set[str] | str:
    """Return set of gate filenames named in CHANGELOG §10 prose."""
    if not CHANGELOG.exists():
        return f"missing {CHANGELOG.relative_to(CRATE_ROOT)}"
    text = CHANGELOG.read_text()
    sec = re.search(
        r"^### 10\. CI gate evolution.*?(?=^###\s|^---|\Z)",
        text, re.MULTILINE | re.DOTALL,
    )
    if not sec:
        return "§10 'CI gate evolution' section not found"
    return set(re.findall(r"`(verify_[A-Za-z0-9_]+\.py)`", sec.group(0)))


def main() -> int:
    stages = parse_stages_gate_names()
    if isinstance(stages, str):
        print(f"# FAIL  parse stages: {stages}", file=sys.stderr)
        return 1
    section10 = parse_section10_gates()
    if isinstance(section10, str):
        print(f"# FAIL  parse §10: {section10}", file=sys.stderr)
        return 1

    print(f"# verify_changelog_section10_authority.py — "
          f"{len(stages)} verify_*.py stage(s); "
          f"{len(section10)} §10-mentioned gate(s)")

    mismatches: list[str] = []

    # (a) Every STAGES verify_*.py gate has a §10 entry (modulo legacy).
    missing_in_changelog: list[str] = []
    for gate in stages:
        if gate in LEGACY_ALLOWLIST:
            continue
        if gate not in section10:
            missing_in_changelog.append(gate)

    if missing_in_changelog:
        mismatches.append(
            f"{len(missing_in_changelog)} STAGES gate(s) not "
            f"mentioned in CHANGELOG §10: "
            f"{sorted(missing_in_changelog)}. Add a `- **Loop N "
            "<suffix>** — \\`<gate>\\` added (Nth stage)` entry "
            "to §10 in the commit that introduces the gate.")
    else:
        print(f"# OK    all {len(set(stages) - LEGACY_ALLOWLIST)} "
              "non-legacy STAGES gate(s) mentioned in §10")

    # (b) Every §10-mentioned gate has a STAGES entry. (A §10 mention
    # of a removed gate would be a stale entry — typically OK during
    # transition; we flag as WARN not FAIL.)
    extra_in_changelog = section10 - set(stages) - LEGACY_ALLOWLIST
    if extra_in_changelog:
        print(f"# WARN  §10 mentions {len(extra_in_changelog)} gate(s) "
              f"not in STAGES: {sorted(extra_in_changelog)} (likely "
              "stale entries from removed gates; remove or move to "
              "an explicit `(deprecated)` annotation)",
              file=sys.stderr)

    if mismatches:
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(f"# verify_changelog_section10_authority.py — "
              f"{len(mismatches)} §10-authority violation(s)",
              file=sys.stderr)
        return 1
    print(f"# verify_changelog_section10_authority.py — §10 is "
          "authoritative; all non-legacy stages have entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
