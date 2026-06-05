#!/usr/bin/env python3
"""verify_burn_down_history.py — gate the FALLBACK_BASELINES
breadcrumb in verify_anonymizer_completeness.py against current
sidecar state.

Loop 137 B operationalizes the "documented history vs live state"
class. The FALLBACK_BASELINES comment carries a trajectory like:

    Loop 132 C baseline: 29 + 43 = 72.
    Loop 133 A.iv (F2 §E catalogue): 22 + 43 = 65.
    Loop 134 A.iii (#1021 §5.4 partition): 22 + 39 = 61.
    Loop 135 A.iii (#1021 §5.4 second pass): 22 + 32 = 54.
    Loop 136 A.iii (F2 §E remaining + §3.2/§4/§6.1): 16 + 32 = 48.

The most-recent entry must agree with the current sidecar JSON state
(per-file counts sum to the stated total). Older entries are
historical-only (sidecar didn't exist before Loop 134 B); the gate
parses them for shape validity (`A + B = C` arithmetic holds) but
doesn't bind them to anything.

Approach:
  1. Parse the breadcrumb from
     `papers/scripts/verify_anonymizer_completeness.py`.
  2. For each parsed tuple, assert `A + B == C` arithmetically.
  3. For the most-recent tuple, assert `A == sidecar["papers/f2_methodology.md"]`
     AND `B == sidecar["papers/phi_ladder_paper_intro_draft.md"]` AND
     `C == sum(sidecar.baselines)`.

Usage: papers/scripts/verify_burn_down_history.py

Exit 0 on agreement; 1 on any arithmetic or live-state drift.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
ANONYMIZER_GATE = (
    CRATE_ROOT / "papers" / "scripts" / "verify_anonymizer_completeness.py"
)
SIDECAR = CRATE_ROOT / "papers" / "scripts" / "anonymizer_baseline.json"


# Per-line breadcrumb entry shape:
#   "Loop <N> [<optional label>]: <A> + <B> = <C>."
# The label part is permissive — anything between the loop tag and the
# colon (e.g., "A.iv (F2 §E catalogue)") is captured-but-ignored.
# Label-token class extended in successive loops to keep up with
# breadcrumb label drift: Loop 138 (em-dash/en-dash/curly apostrophe),
# Loop 142 (comma), Loop 143 follow-up to 65th-pass #6 (documented in
# guide), Loop 144 (66th-pass #7 fix: brackets `[]`, semicolons `;`,
# pipes `|`). The fundamental separator is the colon — never add `:`
# to the class.
_ENTRY_RE = re.compile(
    r"Loop\s+(\d+)\s*[A-Za-z0-9.()\[\]\s§#+/\-—–',;|]*?:\s*(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)\.",
)

# Loop 144 — 66th-pass SEV-3 #7 (post-extension): catch the
# "looks-like-a-breadcrumb-but-didn't-parse" class. Any docstring
# line beginning with `#   Loop \d+` that doesn't match `_ENTRY_RE`
# emits a stderr WARN so a future contributor knows their intended
# entry silently dropped.
_BREADCRUMB_LINE_RE = re.compile(
    r"^\s*#\s+Loop\s+(\d+)[^\n]*$", re.MULTILINE,
)


def parse_breadcrumb() -> list[tuple[int, int, int, int, int]] | str:
    """Return list of (loop, A, B, C, line_no) tuples or error string."""
    if not ANONYMIZER_GATE.exists():
        return f"missing {ANONYMIZER_GATE.relative_to(CRATE_ROOT)}"
    text = ANONYMIZER_GATE.read_text()
    # Limit search to the FALLBACK_BASELINES doc block. The breadcrumb
    # lives in a `#` comment between FALLBACK_BASELINES introduction
    # and the `FALLBACK_BASELINES = {` line below.
    sec = re.search(
        r"FALLBACK_BASELINES.*?(?=^FALLBACK_BASELINES\s*=)",
        text, re.DOTALL | re.MULTILINE,
    )
    if not sec:
        return "FALLBACK_BASELINES breadcrumb section not found"
    sec_text = sec.group(0)
    sec_offset = sec.start()

    # Loop 144 fix #7: detector for breadcrumb-shaped lines that
    # didn't parse. A line beginning with `# Loop \d+` that doesn't
    # produce an _ENTRY_RE match likely contains a character outside
    # the label class — silent-drop class.
    parsed_starts = {m.start() for m in _ENTRY_RE.finditer(sec_text)}
    for cm in _BREADCRUMB_LINE_RE.finditer(sec_text):
        # Find the entry match (if any) within this line by checking
        # overlap with the line span; if no overlap, the line failed
        # to produce an entry.
        line_start, line_end = cm.span()
        has_entry = any(
            line_start <= s < line_end for s in parsed_starts
        )
        if not has_entry:
            file_line_no = text[:sec_offset + cm.start()].count("\n") + 1
            print(f"# WARN  verify_burn_down_history.py:{file_line_no}: "
                  f"Loop {cm.group(1)} line did NOT parse as a "
                  "breadcrumb entry — character outside label class? "
                  "Check `_ENTRY_RE` extension policy.",
                  file=sys.stderr)

    out: list[tuple[int, int, int, int, int]] = []
    for m in _ENTRY_RE.finditer(sec_text):
        loop = int(m.group(1))
        a = int(m.group(2))
        b = int(m.group(3))
        c = int(m.group(4))
        line_no = text[:sec_offset + m.start()].count("\n") + 1
        out.append((loop, a, b, c, line_no))
    if not out:
        return ("FALLBACK_BASELINES breadcrumb has 0 entries matching "
                f"'Loop N ... : A + B = C.' shape — registered pattern: "
                f"{_ENTRY_RE.pattern!r}")
    return out


def load_sidecar() -> dict[str, int] | str:
    if not SIDECAR.exists():
        return f"sidecar {SIDECAR.relative_to(CRATE_ROOT)} missing"
    try:
        data = json.loads(SIDECAR.read_text())
    except json.JSONDecodeError as e:
        return f"sidecar JSON parse error: {e}"
    bl = data.get("baselines")
    if not isinstance(bl, dict):
        return "sidecar lacks 'baselines' dict"
    out: dict[str, int] = {}
    for k, v in bl.items():
        if isinstance(v, int) and v >= 0:
            out[k] = v
    return out


def main() -> int:
    entries = parse_breadcrumb()
    if isinstance(entries, str):
        print(f"# FAIL  parse  {entries}", file=sys.stderr)
        return 1

    mismatches: list[str] = []

    # Shape check: A + B == C for every entry.
    for loop, a, b, c, line_no in entries:
        if a + b != c:
            mismatches.append(
                f"verify_anonymizer_completeness.py:{line_no}: "
                f"Loop {loop} breadcrumb says {a} + {b} = {c} but "
                f"{a} + {b} = {a + b}.")

    # Live binding for the most-recent entry.
    sidecar = load_sidecar()
    if isinstance(sidecar, str):
        mismatches.append(f"sidecar load: {sidecar}")
    else:
        entries_by_loop = sorted(entries, key=lambda e: e[0])
        latest_loop, latest_a, latest_b, latest_c, latest_line = (
            entries_by_loop[-1]
        )
        f2_actual = sidecar.get("papers/f2_methodology.md")
        phi_actual = sidecar.get("papers/phi_ladder_paper_intro_draft.md")
        total_actual = sum(sidecar.values())
        if f2_actual is None or phi_actual is None:
            mismatches.append(
                "sidecar missing one of the two expected keys "
                "(papers/f2_methodology.md, "
                "papers/phi_ladder_paper_intro_draft.md)"
            )
        else:
            if f2_actual != latest_a:
                mismatches.append(
                    f"breadcrumb Loop {latest_loop} A={latest_a} but "
                    f"sidecar f2_methodology.md={f2_actual}")
            if phi_actual != latest_b:
                mismatches.append(
                    f"breadcrumb Loop {latest_loop} B={latest_b} but "
                    f"sidecar phi_ladder...={phi_actual}")
            if total_actual != latest_c:
                mismatches.append(
                    f"breadcrumb Loop {latest_loop} C={latest_c} but "
                    f"sidecar total={total_actual}")
            if not mismatches:
                print(f"# OK    breadcrumb most-recent Loop "
                      f"{latest_loop}: {latest_a} + {latest_b} = "
                      f"{latest_c} (matches live sidecar)")

    if mismatches:
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(f"# verify_burn_down_history.py — "
              f"{len(mismatches)} drift(s) in burn-down history",
              file=sys.stderr)
        return 1
    print(f"# verify_burn_down_history.py — {len(entries)} historical "
          f"entries shape-valid; most-recent agrees with live sidecar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
