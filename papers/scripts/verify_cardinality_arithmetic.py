#!/usr/bin/env python3
"""verify_cardinality_arithmetic.py — gate "N items (a + b + c)" claims
across paper bodies for sum-of-parts == leading.

Loop 133 B generalizes the 55th-pass #14 catch (§1 claim-class enum
specifically) to a registry of cardinality-arithmetic claims. The
56th-pass #4 catch surfaced that the original 4-class hardcoded regex
in verify_submission_readiness.py was order-rigid and count-rigid;
this gate uses class-count-agnostic two-stage parsing.

For each registered (path, anchor_pattern, label):
  1. Find the anchor pattern, capturing (leading_count, parenthetical).
  2. Parse all integers from the parenthetical via `\\d+`.
  3. Assert sum-of-integers == leading_count.

If the anchor pattern doesn't match, the claim is dropped (the gate
does not enforce existence — that's the registry author's job; once
a claim is registered, drift in its arithmetic must fail).

Usage: papers/scripts/verify_cardinality_arithmetic.py

Exit 0 if all registered claims hold; 1 on any arithmetic drift or
parse failure on a previously-matching anchor.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]


# Each entry: (relative_path, anchor_regex_with_two_groups, label,
# [optional] frozen). Group 1 captures the leading count; group 2
# captures the parenthetical body. The body is parsed by extracting
# all `\d+` substrings and summing them. Order- and count-agnostic —
# works for 2-class through arbitrary-class partitions.
#
# Loop 135 B (57th-pass SEV-2 fix #1): optional 4th `frozen` field.
# Frozen entries are historical snapshots whose component distribution
# may differ from the current live state; the arithmetic (sum == lead)
# is still verified, but a `# INFO frozen` annotation flags the entry
# so future readers don't infer current-state from a snapshot.
#
# Scope note: this gate handles **parenthetical-shaped** enumerations.
# English-word partitions like F2 §E "seven appear ... other twenty
# stages" are covered by `verify_stage_count_consistency.py`'s
# DECOMPOSITION_CLAIMS (with `words_to_int` for English numerals).
# The two gates are complementary; this one stays narrow on shape.
CLAIMS: list[tuple] = [
    # SUBMISSION_CHECKLIST §1 (13/N) cross-paper claim-class
    # enumeration. Live mirror of verify_cross_paper_consistency.py's
    # `*_CLAIMS` lists; the class-registry-binding gate further
    # asserts the labels themselves resolve to actual classes.
    (
        "papers/SUBMISSION_CHECKLIST.md",
        r"(\d+) claims \(([^)]+)\)",
        "SUBMISSION_CHECKLIST §1 cross-paper claim-class enumeration",
    ),
    # CHANGELOG §10 Loop 131 B entry. The Loop 131 B narrative
    # documents the cross-paper class composition at THAT time
    # (1+5+1+4=11); Loop 133 A.iii reclassified 3 SCOPED → EXACT_PIN.
    # The arithmetic still holds (1+5+1+4 == 1+2+1+4+3 == 11) so the
    # gate passes either way, but the frozen=True flag makes the
    # historical-snapshot status explicit to future readers.
    (
        "papers/CHANGELOG.md",
        r"cross-paper claims now (\d+) across \d+ classes \(([^)]+)\)",
        "CHANGELOG §10 Loop 131 B cross-paper class enumeration (frozen)",
        True,  # frozen
    ),
]


def find_claim(path: Path, pattern: str) -> tuple[int, str, int] | None:
    """Return (leading_count, parenthetical_body, line_no) or None."""
    if not path.exists():
        return None
    text = path.read_text()
    m = re.search(pattern, text)
    if not m:
        return None
    leading = int(m.group(1))
    body = m.group(2)
    line_no = text[:m.start()].count("\n") + 1
    return leading, body, line_no


def sum_parts(body: str) -> tuple[int, list[int]]:
    """Extract all integer literals from the parenthetical body and
    return (sum, parts). Suppresses zero-padding by treating each
    `\\d+` substring as base-10."""
    parts = [int(n) for n in re.findall(r"\d+", body)]
    return sum(parts), parts


def main() -> int:
    mismatches: list[str] = []
    verified = 0
    for entry in CLAIMS:
        # Loop 135 B: tuple length disambiguates frozen flag.
        # 3-tuple = live (default frozen=False); 4-tuple = explicit
        # frozen value (typically True for historical snapshots).
        if len(entry) == 3:
            rel, pattern, label = entry
            frozen = False
        elif len(entry) == 4:
            rel, pattern, label, frozen = entry
        else:
            mismatches.append(
                f"CLAIMS registry entry has {len(entry)} fields; "
                f"expected 3 (live) or 4 (with frozen flag). Update.")
            continue
        path = CRATE_ROOT / rel
        found = find_claim(path, pattern)
        if found is None:
            mismatches.append(
                f"{rel}: anchor pattern not found for '{label}' "
                f"({pattern!r}) — claim may have been reworded or removed. "
                "If intentional, drop the CLAIMS entry."
            )
            print(f"# FAIL  miss  {label}", file=sys.stderr)
            continue
        leading, body, line_no = found
        total, parts = sum_parts(body)
        if not parts:
            mismatches.append(
                f"{rel}:{line_no}: '{label}' parenthetical "
                f"{body!r} contained no integers; arithmetic check "
                f"undefined.")
            print(f"# FAIL  empty {label}", file=sys.stderr)
            continue
        if total != leading:
            mismatches.append(
                f"{rel}:{line_no}: '{label}' leading {leading} != sum "
                f"{'+'.join(str(p) for p in parts)} = {total}")
            print(f"# FAIL  sum   {label}: {leading} != {total}",
                  file=sys.stderr)
            continue
        verified += 1
        tag = "OK   " if not frozen else "FROZ "
        print(f"# {tag} {rel}:{line_no}  {label}: {leading} = "
              f"{'+'.join(str(p) for p in parts)}"
              f"{' (historical snapshot — does not track live state)' if frozen else ''}")

    if mismatches:
        print(f"# verify_cardinality_arithmetic.py — "
              f"{len(mismatches)} arithmetic drift(s) across "
              f"{len(CLAIMS)} registered claim(s)",
              file=sys.stderr)
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        return 1
    print(f"# verify_cardinality_arithmetic.py — "
          f"{verified}/{len(CLAIMS)} cardinality-arithmetic claim(s) "
          "verified, 0 drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
