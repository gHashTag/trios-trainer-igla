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
    # CHANGELOG §10 "on-disk gate now runs **N stages**" (Loop 121 D).
    (
        "papers/CHANGELOG.md",
        r"on-disk gate now runs \*\*(\d+) stages\*\*",
        "CHANGELOG §10 on-disk gate count",
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


# Decomposition claims of the form "N1 F2-scope ... + N2 #1021-scoped = N total".
# Loop 119 B: gate the partition arithmetic, not just the total. Catches drift
# in claims like "12 F2 + 6 #1021 = 18" when stages are added to either bucket.
DECOMPOSITION_CLAIMS: list[tuple[str, str, str]] = [
    (
        "papers/phi_ladder_paper_intro_draft.md",
        # "**13 F2-scope stages**" ... line wrap ... "+ 6 #1021-scoped\n stages"
        r"\*\*(\d+) F2-scope stages\*\*[\s\S]*?\+\s+(\d+)\s+#1021-scoped[\s\S]*?stages",
        "#1021 §5.4 decomposition partition",
    ),
    # F2 paper §E partition (Loop 120 B): "seven appear as individual stages
    # ... the other twelve stages are gates introduced". The pattern matches
    # the numeric pair within the run_all_checks.sh paragraph; the 42nd pass
    # caught a "twelve" + "additional 10" residue in this paragraph that the
    # initial #1021-only gate could not catch. \s+ tolerates the line wrap
    # between "appear as" and "individual stages".
    (
        "papers/f2_methodology.md",
        r"(\w+)\s+appear\s+as\s+individual\s+stages[\s\S]*?other\s+(\w+)\s+stages\s+are\s+gates",
        "F2 §E catalogue partition (words)",
    ),
]


# DERIVED_CLAIMS: each claim's expected value is `actual - constant_offset`.
# Used for F2 §E's "additional N" sentence whose N should match the
# "non-catalogue" stage count = actual - 7 (the 7 catalogued-as-stage
# count out of 8 catalogue bullets; the 8th is run_all_checks.sh itself).
DERIVED_CLAIMS: list[tuple[str, str, int, str]] = [
    (
        "papers/f2_methodology.md",
        # "the additional N are documented" — N should equal actual - 7
        r"the additional (\d+) are documented",
        7,   # constant_offset: actual - 7 catalogued-as-stage = additional
        "F2 §E follow-up additional count (= actual - 7 catalogued-as-stage)",
    ),
]


def words_to_int(w: str) -> int | None:
    """Convert English number words (zero..twenty) + digits to int."""
    table = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20,
    }
    if w.isdigit():
        return int(w)
    return table.get(w.lower())


def find_decomposition(path: Path, pattern: str) -> tuple[int, int, int] | None:
    """Return (n1, n2, line_no_of_n1) or None. Accepts digits or English words."""
    pat = re.compile(pattern, re.DOTALL)
    text = path.read_text()
    m = pat.search(text)
    if not m:
        return None
    line_no = text[:m.start(1)].count("\n") + 1
    n1 = words_to_int(m.group(1))
    n2 = words_to_int(m.group(2))
    if n1 is None or n2 is None:
        return None
    return n1, n2, line_no


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

    # Loop 120 B: derived claims (N == actual - constant_offset).
    for rel, pattern, offset, desc in DERIVED_CLAIMS:
        path = CRATE_ROOT / rel
        if not path.exists():
            mismatches.append(f"{rel}: file missing for derived '{desc}'")
            continue
        found = find_claim(path, pattern)
        if found is None:
            mismatches.append(
                f"{rel}: no match for derived '{desc}' (pattern {pattern!r})")
            continue
        claimed, line_no = found
        expected = actual - offset
        if claimed != expected:
            mismatches.append(
                f"{rel}:{line_no}: '{desc}' claims {claimed} but expected "
                f"{expected} (= {actual} - {offset})")
        else:
            print(f"# OK  {rel}:{line_no}  — {desc} = {claimed}")

    # Loop 119 B: decomposition arithmetic.
    for rel, pattern, desc in DECOMPOSITION_CLAIMS:
        path = CRATE_ROOT / rel
        if not path.exists():
            mismatches.append(f"{rel}: file missing for decomposition '{desc}'")
            continue
        found = find_decomposition(path, pattern)
        if found is None:
            mismatches.append(
                f"{rel}: no match for decomposition '{desc}' "
                f"(pattern {pattern!r}) — paper claim might have moved or been reworded")
            continue
        n1, n2, line_no = found
        partition_sum = n1 + n2
        if partition_sum != actual:
            mismatches.append(
                f"{rel}:{line_no}: '{desc}' claims {n1}+{n2}={partition_sum} "
                f"but actual STAGES total is {actual}")
        else:
            print(f"# OK  {rel}:{line_no}  — {desc} = {n1}+{n2}={partition_sum}")

    if mismatches:
        print(f"# FAIL  {len(mismatches)} stage-count drift(s) "
              f"vs actual {actual}:", file=sys.stderr)
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        return 1
    print(f"# verify_stage_count_consistency.py — "
          f"{len(CLAIMS)} counts + {len(DECOMPOSITION_CLAIMS)} "
          f"decompositions + {len(DERIVED_CLAIMS)} derived verified, "
          "0 drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
