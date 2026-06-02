#!/usr/bin/env python3
"""verify_anchor_loop_coverage.py — assert every CHANGELOG §10
Loop-N entry has a corresponding commit on f2-methodology.

Loop 140 B operationalizes "documented in narrative but not in
code-archaeology". The CHANGELOG §10 lists per-loop CI gate
additions; this gate verifies that for each named Loop N, at least
one commit on the f2-methodology branch references "Loop N" in its
message. If §10 names a loop that has no corresponding commit, the
narrative is ahead of the history (or the commit grep is broken).

Tolerance: the most-recent loop entry is exempt from the commit
check (the current in-flight loop may have added §10 prose before
its commit lands; mirrors the verify_generator_consistency 1-loop
lag policy).

Usage: papers/scripts/verify_anchor_loop_coverage.py

Exit 0 on full coverage; 1 if any §10 loop entry (excluding the
most-recent) lacks a commit.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
CHANGELOG = CRATE_ROOT / "papers" / "CHANGELOG.md"


# Match `- **Loop N <suffix>**` at the start of a CHANGELOG §10 list
# item. The suffix may be empty, "A", "A.iii", "C", "B" etc. We capture
# just the integer for grep purposes; the suffix is informational.
_LOOP_ENTRY_RE = re.compile(
    r"^- \*\*Loop (\d+)(?:[A-Z. ()0-9iv]*)?\*\*",
    re.MULTILINE,
)


def parse_section10_loops() -> tuple[list[int], int] | str:
    """Return (sorted unique Loop N integers from §10, section line_no)
    or error string."""
    if not CHANGELOG.exists():
        return f"missing {CHANGELOG.relative_to(CRATE_ROOT)}"
    text = CHANGELOG.read_text()
    sec = re.search(
        r"^### 10\. CI gate evolution.*?(?=^###\s|^---|\Z)",
        text, re.MULTILINE | re.DOTALL,
    )
    if not sec:
        return "§10 'CI gate evolution' section not found"
    sec_text = sec.group(0)
    sec_offset = sec.start()
    loops = set()
    for m in _LOOP_ENTRY_RE.finditer(sec_text):
        loops.add(int(m.group(1)))
    if not loops:
        return ("§10 had no `- **Loop N**` list entries matched by "
                f"pattern {_LOOP_ENTRY_RE.pattern!r}")
    line_no = text[:sec_offset].count("\n") + 1
    return sorted(loops), line_no


def commits_mentioning_loop(n: int) -> int:
    """Return count of commits on the current branch whose message
    contains 'Loop {n}'. Falls back to 0 on git error."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", f"--grep=Loop {n}\\b", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=str(CRATE_ROOT),
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0
    return sum(1 for ln in result.stdout.splitlines() if ln.strip())


def main() -> int:
    parsed = parse_section10_loops()
    if isinstance(parsed, str):
        print(f"# FAIL  parse  {parsed}", file=sys.stderr)
        return 1
    loops, sec_line = parsed
    most_recent = max(loops)
    print(f"# verify_anchor_loop_coverage.py — §10:{sec_line} names "
          f"{len(loops)} unique Loop entries (most-recent = "
          f"{most_recent}, exempt from commit check)")

    mismatches: list[str] = []
    for n in loops:
        if n == most_recent:
            continue  # in-flight lag tolerance
        commit_count = commits_mentioning_loop(n)
        if commit_count == 0:
            mismatches.append(
                f"§10 names Loop {n} but `git log --grep='Loop {n}\\b'` "
                f"returns 0 commits on HEAD. Either the narrative is "
                "ahead of history, or the commit message lacks the "
                "expected 'Loop N' tag.")
        else:
            print(f"# OK    Loop {n}: {commit_count} matching commit(s)")

    if mismatches:
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(f"# verify_anchor_loop_coverage.py — "
              f"{len(mismatches)} §10 loop entr(ies) without commit "
              "coverage",
              file=sys.stderr)
        return 1
    print(f"# verify_anchor_loop_coverage.py — "
          f"{len(loops) - 1} historical §10 loop entries have ≥1 "
          "matching commit each (most-recent exempt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
