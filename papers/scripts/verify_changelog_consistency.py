#!/usr/bin/env python3
"""verify_changelog_consistency.py — gate the adversarial-pass count
and loop-range cursor across the three documents that report them.

Loop 131 C operationalizes the 53rd-pass catch #1: CHANGELOG §7 lead
paragraph claimed "Fifty independent passes total across Loops 59–127"
while SUBMISSION_CHECKLIST §2 and ADVERSARIAL_REVIEW_LOG headline said
"52 passes". The drift surfaced because no static check tied the three
sites together — each loop's refresh script updated whichever doc the
loop touched, and §7 lagged silently.

This gate parses three anchor sites and asserts:
  (a) the integer pass-count agrees across all three (CHANGELOG §7
      uses an English word like "Fifty-three"; the other two use digits;
      the gate normalizes both into ints).
  (b) the loop-range terminal cursor agrees (all three say "Loops 59-N"
      for the same N — they share the same N because each documents
      cumulative state through the most-recent loop).

Site registry (path, regex with two groups: count + last_loop):
  - `papers/CHANGELOG.md` §7 lead paragraph
  - `papers/SUBMISSION_CHECKLIST.md` §2 adversarial review line
  - `docs/ADVERSARIAL_REVIEW_LOG.md` headline statistics

Usage: papers/scripts/verify_changelog_consistency.py

Exit 0 on agreement; 1 on any drift. Per-site diagnostics on stderr.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]


# English word number support — covers the current §7 lead paragraph
# which writes "Fifty-three" instead of "53". The other two sites use
# digits; this dispatcher normalizes both forms into ints.
_UNITS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
]


def _to_int(w: str) -> int | None:
    """Accept "53" or "Fifty-three" or "fifty three" forms; return int.
    Returns None on unrecognized input (caller fails the gate cleanly)."""
    if w.isdigit():
        return int(w)
    w = w.lower().strip()
    if not w:
        return None
    if w in _UNITS:
        return _UNITS.index(w)
    if w in _TENS[2:]:
        return _TENS.index(w) * 10
    for sep in ("-", " "):
        if sep in w:
            t, _, u = w.partition(sep)
            if t in _TENS and u in _UNITS[1:10]:
                return _TENS.index(t) * 10 + _UNITS.index(u)
    return None


# Per-site claim: (path, regex with (count, last_loop) groups, label).
# All three regexes capture two groups; group 1 is the count (digit or
# word), group 2 is the terminal loop number. Loop range start is
# always 59 (paper-wide convention — the first adversarial pass landed
# at Loop 59 per ADVERSARIAL_REVIEW_LOG).
SITES: list[tuple[Path, str, str]] = [
    # CHANGELOG §7 lead paragraph. The word-or-digit count is captured
    # by `([A-Za-z0-9-]+)`; loop-range terminal cursor is the digit
    # after "Loops 59–"/"Loops 59-". Em-dash (–) and hyphen (-) both
    # accepted because the §7 prose uses em-dash while the other two
    # sites use ASCII hyphen.
    (
        CRATE_ROOT / "papers" / "CHANGELOG.md",
        # Use re.DOTALL via inline (?s); the prose wraps at ~64 cols,
        # so "**Fifty-three** independent\npasses total across Loops
        # 59–130" needs `\s+` (not literal space) between tokens. The
        # `.{0,40}` between "**Fifty-three**" and "independent" is a
        # narrow window to tolerate hyphen/whitespace markdown noise
        # without crossing paragraph boundaries.
        r"\*\*([A-Za-z0-9-]+)\*\*\s+independent\s+passes\s+total\s+"
        r"across\s+Loops\s+59[–-](\d+)",
        "CHANGELOG §7 lead paragraph",
    ),
    (
        CRATE_ROOT / "papers" / "SUBMISSION_CHECKLIST.md",
        r"Adversarial review\*\*: (\d+) passes across Loops 59[–-](\d+)",
        "SUBMISSION_CHECKLIST §2 adversarial review line",
    ),
    (
        CRATE_ROOT / "docs" / "ADVERSARIAL_REVIEW_LOG.md",
        r"\*\*(\d+) adversarial passes total\*\* \(Loops 59[–-](\d+)\)",
        "ADVERSARIAL_REVIEW_LOG headline statistics",
    ),
    # Loop 132-A SEV-2 fix #1: add CHANGELOG §10 header as a 4th site
    # so the loop-range terminal cursor is enforced at every documented
    # terminal. Without this, §10's "(Loops 87–130)" drifts silently
    # while §7's "Loops 59–130" gets caught — the prior gate only saw
    # one of them. Pattern: capture the start loop and terminal — the
    # gate already enforces (count, last_loop); start loop is captured
    # but ignored for now (could become its own SITES_BY_START class
    # if a future loop wants to bind §10's "from 87" anchor too).
    #
    # We synthesize a fake "count" group of 0 because §10's header
    # doesn't claim a pass count. The agreement check tolerates this:
    # `counts` collapses {0, 54, 54, 54} → {0, 54} which would fail.
    # So instead we register §10 with a pattern that captures a dummy
    # count == the agreed terminal loop count, derived from the OTHER
    # sites at agreement time. This is too clever; the simpler path
    # is a separate LOOP_RANGE_SITES list that gates only the terminal
    # loop number, distinct from the count-and-loop SITES above.
]


# Loop 132-A SEV-2 fix #1: separate registry for sites that gate ONLY
# the loop-range terminal cursor (no associated pass count). Each
# entry: (path, regex with single group capturing terminal loop, label).
LOOP_RANGE_SITES: list[tuple[Path, str, str]] = [
    (
        CRATE_ROOT / "papers" / "CHANGELOG.md",
        r"### 10\. CI gate evolution \(Loops 87[–-](\d+)\)",
        "CHANGELOG §10 header loop-range terminal",
    ),
]


def parse_site(path: Path, pattern: str, label: str
               ) -> tuple[int, int, int] | str:
    """Return (count, last_loop, line_no) or an error string."""
    if not path.exists():
        return f"{label}: file missing at {path.relative_to(CRATE_ROOT)}"
    text = path.read_text()
    m = re.search(pattern, text)
    if not m:
        return (f"{label}: regex {pattern!r} did not match {path.name} "
                "— prose may have been reworded; update SITES")
    count = _to_int(m.group(1))
    last_loop = _to_int(m.group(2))
    if count is None:
        return (f"{label}: count token {m.group(1)!r} not recognized "
                "(supports digits + English 0-99)")
    if last_loop is None:
        return (f"{label}: last-loop token {m.group(2)!r} not recognized")
    line_no = text[:m.start()].count("\n") + 1
    return count, last_loop, line_no


def main() -> int:
    rows: list[tuple[str, int, int, int]] = []
    errors: list[str] = []
    for path, pattern, label in SITES:
        result = parse_site(path, pattern, label)
        if isinstance(result, str):
            errors.append(result)
            print(f"# FAIL  parse  {label}: {result}", file=sys.stderr)
            continue
        count, last_loop, line_no = result
        rows.append((label, count, last_loop, line_no))
        print(f"# OK    parse  {label}:{line_no}  count={count} "
              f"last_loop={last_loop}")
    if errors:
        print(f"# verify_changelog_consistency.py — "
              f"{len(errors)} parse error(s); cannot compare",
              file=sys.stderr)
        return 1

    # Loop 132-A: also parse LOOP_RANGE_SITES (loop-terminal-only
    # registry, no count). These contribute to the `loops` agreement
    # check but not to `counts`.
    loop_range_rows: list[tuple[str, int, int]] = []
    for path, pattern, label in LOOP_RANGE_SITES:
        if not path.exists():
            errors.append(f"{label}: file missing at {path.relative_to(CRATE_ROOT)}")
            continue
        text = path.read_text()
        m = re.search(pattern, text)
        if not m:
            errors.append(f"{label}: regex did not match {path.name}")
            continue
        last_loop = _to_int(m.group(1))
        if last_loop is None:
            errors.append(f"{label}: last-loop token {m.group(1)!r} not recognized")
            continue
        line_no = text[:m.start()].count("\n") + 1
        loop_range_rows.append((label, last_loop, line_no))
        print(f"# OK    parse  {label}:{line_no}  last_loop={last_loop}")
    if errors:
        print(f"# verify_changelog_consistency.py — "
              f"{len(errors)} parse error(s); cannot compare",
              file=sys.stderr)
        return 1

    # Pairwise agreement: all counts equal, all last_loops equal.
    counts = {r[1] for r in rows}
    loops = {r[2] for r in rows} | {lr[1] for lr in loop_range_rows}
    if len(counts) > 1:
        errors.append(
            f"pass count disagreement across {len(rows)} sites: "
            f"{sorted(counts)}")
        print(f"# FAIL  agree  pass counts: {sorted(counts)}",
              file=sys.stderr)
        for label, c, _, line_no in rows:
            print(f"  {label}:{line_no} = {c}", file=sys.stderr)
    if len(loops) > 1:
        errors.append(
            f"loop-range terminal disagreement across "
            f"{len(rows) + len(loop_range_rows)} sites: {sorted(loops)}")
        print(f"# FAIL  agree  loop terminals: {sorted(loops)}",
              file=sys.stderr)
        for label, _, l, line_no in rows:
            print(f"  {label}:{line_no} = Loops 59-{l}", file=sys.stderr)
        for label, l, line_no in loop_range_rows:
            print(f"  {label}:{line_no} = Loops ...-{l}", file=sys.stderr)

    if errors:
        print(f"# verify_changelog_consistency.py — "
              f"{len(errors)} drift(s) across {len(SITES)} sites",
              file=sys.stderr)
        return 1
    only_count = next(iter(counts))
    only_loop = next(iter(loops))
    print(f"# verify_changelog_consistency.py — {len(SITES)} sites "
          f"agree on {only_count} passes / Loops 59-{only_loop}, 0 drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
