#!/usr/bin/env python3
"""verify_burn_down_trajectory.py — gate FALLBACK_BASELINES breadcrumb
monotonicity.

Loop 142 B operationalizes the "ratchet always tightens" invariant.
The burn-down history gate (Loop 137 B) asserts each entry's
arithmetic and most-recent agreement; this sister gate asserts the
trajectory shape:

  (a) Loop numbers strictly increase across breadcrumb entries.
  (b) Total (C) is monotonically non-increasing — debt only goes
      down, never up.

Per-file counts can swap (one file burns while another gains a new
attribution line, e.g., Loop 138 added 3 phi_ladder partition refs
even as the F2 baseline dropped 5). The total-monotonicity is the
load-bearing invariant.

Catches:
  - A future commit accidentally appends an entry with a higher
    total than the previous (e.g., misclick on the --update-baseline
    automation).
  - A loop number out-of-order, which would also break the burn-down
    history gate's most-recent selection.

Usage: papers/scripts/verify_burn_down_trajectory.py

Exit 0 if both invariants hold; 1 on any violation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
ANONYMIZER_GATE = (
    CRATE_ROOT / "papers" / "scripts" / "verify_anonymizer_completeness.py"
)


# Same shape as verify_burn_down_history._ENTRY_RE — keep them in
# sync (Loop 138/140 extended class with em-dash/en-dash/curly
# apostrophe; Loop 142 follow-up #6 added comma; Loop 144 66th-pass
# #7 added brackets/semicolons/pipes).
_ENTRY_RE = re.compile(
    r"Loop\s+(\d+)\s*[A-Za-z0-9.()\[\]\s§#+/\-—–',;|]*?:\s*(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)\.",
)

# Loop 144 — 66th-pass SEV-3 #8 (re-baseline escape hatch):
# breadcrumb lines tagged with `# RE-BASELINE` immediately before
# them are exempt from the monotonicity check between (prev, this)
# entries. Use for legitimate up-baselines (e.g., adding a new
# TMLR-bound paper that brings new anchors).
_REBASELINE_TAG_RE = re.compile(
    r"^\s*#\s+RE-BASELINE:[^\n]*$", re.MULTILINE,
)


def parse_breadcrumb() -> tuple[list[tuple[int, int, int, int, int]], set[int]] | str:
    """Return (entries, rebaseline_loop_set) or error.
    entries: list of (loop, A, B, C, line_no) in parse order.
    rebaseline_loop_set: loop numbers whose entry was preceded by
        `# RE-BASELINE` annotation; the monotonicity check between
        (prev, this) is skipped for those loops."""
    if not ANONYMIZER_GATE.exists():
        return f"missing {ANONYMIZER_GATE.relative_to(CRATE_ROOT)}"
    text = ANONYMIZER_GATE.read_text()
    sec = re.search(
        r"FALLBACK_BASELINES.*?(?=^FALLBACK_BASELINES\s*=)",
        text, re.DOTALL | re.MULTILINE,
    )
    if not sec:
        return "FALLBACK_BASELINES breadcrumb section not found"
    sec_text = sec.group(0)
    sec_offset = sec.start()
    # Collect rebaseline-tag line positions for matching.
    rebaseline_tag_ends = [
        cm.end() for cm in _REBASELINE_TAG_RE.finditer(sec_text)
    ]
    out: list[tuple[int, int, int, int, int]] = []
    rebaselined: set[int] = set()
    for m in _ENTRY_RE.finditer(sec_text):
        loop = int(m.group(1))
        a = int(m.group(2))
        b = int(m.group(3))
        c = int(m.group(4))
        line_no = text[:sec_offset + m.start()].count("\n") + 1
        # Check whether a RE-BASELINE tag appears immediately before
        # this entry (within ~80 chars; allows for one short line of
        # whitespace between).
        for tag_end in rebaseline_tag_ends:
            gap = m.start() - tag_end
            if 0 < gap < 80:
                rebaselined.add(loop)
                break
        out.append((loop, a, b, c, line_no))
    if not out:
        return "0 breadcrumb entries parsed"
    return out, rebaselined


def main() -> int:
    parsed = parse_breadcrumb()
    if isinstance(parsed, str):
        print(f"# FAIL  parse  {parsed}", file=sys.stderr)
        return 1
    entries, rebaselined = parsed

    mismatches: list[str] = []

    # (a) Loop numbers strictly increase.
    prev_loop = None
    for loop, _a, _b, _c, line_no in entries:
        if prev_loop is not None and loop <= prev_loop:
            mismatches.append(
                f"verify_anonymizer_completeness.py:{line_no}: "
                f"Loop {loop} appears after Loop {prev_loop} in "
                "the breadcrumb. Entries must be in strict "
                "loop-monotonic order; reorder or remove the "
                "out-of-order entry.")
        prev_loop = loop

    # (b) Total C monotonically non-increasing, except for entries
    # explicitly tagged with `# RE-BASELINE` immediately before.
    prev_total = None
    for loop, _a, _b, c, line_no in entries:
        if prev_total is not None and c > prev_total:
            if loop in rebaselined:
                print(f"# INFO  Loop {loop} total {c} > previous "
                      f"{prev_total} — accepted because preceded by "
                      "`# RE-BASELINE` tag")
            else:
                mismatches.append(
                    f"verify_anonymizer_completeness.py:{line_no}: "
                    f"Loop {loop} total {c} > previous total "
                    f"{prev_total}. The ratchet must only tighten; "
                    "a higher total means debt was added — verify the "
                    "intended action is a re-baseline (not a "
                    "regression). If intentional, add `# RE-BASELINE: "
                    "Loop N <reason>` immediately before the entry.")
        prev_total = c

    if mismatches:
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(f"# verify_burn_down_trajectory.py — "
              f"{len(mismatches)} monotonicity violation(s)",
              file=sys.stderr)
        return 1
    print(f"# verify_burn_down_trajectory.py — {len(entries)} "
          f"breadcrumb entries: Loop {entries[0][0]} → "
          f"Loop {entries[-1][0]}, total {entries[0][3]} → "
          f"{entries[-1][3]} (strict loop-monotonic + non-increasing "
          "total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
