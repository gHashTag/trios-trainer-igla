#!/usr/bin/env python3
"""verify_stage_count_consistency.py — gate paper claims of "N stages"
against the actual count in run_all_checks.sh.

Loop 118 B operationalizes the SEV-2 stage-count-drift class that
the 39th + 40th adversarial passes both surfaced:
- 39th pass: F2 §E said "7 scripts + 1" while disk had 17 stages.
- 40th pass: F2 §E rewrite said "7 catalogued + 10 additional"
  but §E head says "**eight** auxiliary scripts".

Both were prose-drift bugs no static check caught. This script
parses the STAGES array in run_all_checks.sh, counts entries,
and asserts each registered "this paper says N stages" claim
matches.

Each entry in CLAIMS specifies:
- file: path relative to crate root
- anchor: a regex that locates the claim
- expected_n: the integer the claim should match
- description: short label for failure messages

Mismatches fail the gate with line-by-line diagnostics.

Usage: papers/scripts/verify_stage_count_consistency.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]

RUN_ALL_CHECKS = CRATE_ROOT / "papers" / "scripts" / "run_all_checks.sh"
STAGES_RE = re.compile(r"^\s*\"([^:\"]+):", re.MULTILINE)


def actual_stage_count() -> int:
    """Parse STAGES=(...) array in run_all_checks.sh."""
    text = RUN_ALL_CHECKS.read_text()
    # Find the STAGES=( block.
    m = re.search(r"STAGES=\(\s*\n(.*?)\n\)", text, re.DOTALL)
    if not m:
        raise RuntimeError("could not locate STAGES=( ... ) block")
    body = m.group(1)
    # Each line in the block that opens with `    "name:` is a stage.
    entries = re.findall(r'^\s+"([^:"]+):', body, re.MULTILINE)
    return len(entries)


# Registry of paper claims to verify.
# Each tuple: (relative_path, regex_pattern, expected_count_function, description)
CLAIMS: list[tuple[str, str, str]] = [
    # F2 paper §E: "Currently chains **18 stages** on disk"
    # Allow the wrapped form "**18 stages\*\*\n  on disk" via [\s\n]* gap.
    (
        "papers/f2_methodology.md",
        r"Currently chains\s+\*\*(\d+)\s*stages\*\*[\s\n]*on disk",
        "F2 §E run_all_checks.sh stage count",
    ),
    # #1021 paper §5.4: "18 stages on disk as of Loop 117"
    (
        "papers/phi_ladder_paper_intro_draft.md",
        r"\((\d+) stages on disk(?:\s+as of Loop \d+)?\)",
        "#1021 §5.4 on-disk gate count",
    ),
    # #1021 paper §5.4 decomposition: "12 F2-scope ... 6 #1021-scoped" = 18
    (
        "papers/phi_ladder_paper_intro_draft.md",
        r"\*\*(\d+)-stage on-disk gate\*\* breaks down",
        "#1021 §5.4 decomposition opener",
    ),
]


def find_claim(path: Path, pattern: str) -> tuple[int, int] | None:
    """Return (claimed_count, line_number) of the first match, or None.
    DOTALL so multi-line wrap doesn't break matching; line number is the
    line containing the digit group."""
    pat = re.compile(pattern, re.DOTALL)
    text = path.read_text()
    m = pat.search(text)
    if not m:
        return None
    # Line containing the captured digit group.
    digit_pos = m.start(1)
    line_no = text[:digit_pos].count("\n") + 1
    return int(m.group(1)), line_no


def main() -> int:
    actual = actual_stage_count()
    print(f"# verify_stage_count_consistency.py — STAGES array has {actual} entries")

    mismatches: list[str] = []
    for rel, pattern, desc in CLAIMS:
        path = CRATE_ROOT / rel
        if not path.exists():
            mismatches.append(f"{rel}: file missing for claim '{desc}'")
            continue
        found = find_claim(path, pattern)
        if found is None:
            mismatches.append(
                f"{rel}: no match for '{desc}' (pattern {pattern!r}) — "
                "paper claim might have moved or been reworded")
            continue
        claimed, line_no = found
        if claimed != actual:
            mismatches.append(
                f"{rel}:{line_no}: '{desc}' claims {claimed} but actual is {actual}")
        else:
            print(f"# OK  {rel}:{line_no}  — {desc} = {claimed}")

    if mismatches:
        print(f"# FAIL  {len(mismatches)} stage-count drift(s) "
              f"vs actual {actual}:", file=sys.stderr)
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        return 1
    print(f"# verify_stage_count_consistency.py — "
          f"{len(CLAIMS)} claims verified, 0 drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
