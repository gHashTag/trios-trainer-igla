#!/usr/bin/env python3
"""verify_deadline_freshness.py — assert no dated AOE deadline in
SUBMISSION_CHECKLIST §4 or `Anchor / version` is in the past.

Loop 145 C closes 67th-pass SEV-4 #6: the soft EOI deadline
2026-06-04 silently aged past its date because nothing in CI flagged
it. The checklist still listed it as "Next deadline" the day after.

The gate parses every `YYYY-MM-DD AOE` token in the checklist and
asserts the date is today or future (where today = UTC date — AOE is
UTC-12, so a strictly-greater-than today comparison is more lenient
than AOE; we use today's UTC date as a safe lower bound).

Exempted contexts:
  - Dates inside section labeled `### Checklist change log` (these
    are historical loop dates, not future deadlines).
  - Dates inside section labeled `## 6. Post-submission` (these
    document future workflow steps, not deadlines).
  - Dates explicitly annotated with `(passed)` or `(historical)`.

Catches:
  - Soft EOI deadline (e.g., 2026-06-04 AOE) that has aged out and
    is still framed as "Next deadline".
  - TMLR hard deadline (2026-09-30 AOE) when the submission window
    has closed.
  - Loop authors who copy a passed date into a new section without
    updating it.

Usage: papers/scripts/verify_deadline_freshness.py

Exit 0 if every parsed deadline is today-or-future; 1 on any past
deadline that lacks the `(passed)`/`(historical)` annotation.
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
CHECKLIST = CRATE_ROOT / "papers" / "SUBMISSION_CHECKLIST.md"

# `YYYY-MM-DD AOE` token; bold/un-bolded both accepted.
_DEADLINE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})\s+AOE")

# Headings whose contents are NOT future deadlines.
_EXEMPT_HEADINGS = (
    "### Checklist change log",
    "## 6. Post-submission",
)

# Annotations that mark an instance as deliberately documenting a
# past date (e.g., "EOI soft 2026-06-04 AOE has passed").
_PASSED_ANNOTATIONS = (
    "(passed)", "(historical)", "has passed",
)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return list of (heading_line, body_text) tuples, where the body
    runs until the next heading at any level."""
    lines = text.splitlines(keepends=True)
    out: list[tuple[str, list[str]]] = []
    current_heading = "(preamble)"
    current_body: list[str] = []
    for line in lines:
        if re.match(r"^#{1,6}\s+\S", line):
            if current_body or current_heading != "(preamble)":
                out.append((current_heading, current_body))
            current_heading = line.rstrip("\n")
            current_body = []
        else:
            current_body.append(line)
    if current_body or current_heading != "(preamble)":
        out.append((current_heading, current_body))
    return [(h, "".join(b)) for h, b in out]


def is_exempt(heading: str) -> bool:
    for ex in _EXEMPT_HEADINGS:
        if heading.strip().startswith(ex):
            return True
    return False


def main() -> int:
    if not CHECKLIST.exists():
        print(f"# FAIL  {CHECKLIST.relative_to(CRATE_ROOT)} missing",
              file=sys.stderr)
        return 1
    text = CHECKLIST.read_text()
    today = dt.date.today()  # local; UTC-12 AOE is more permissive.

    mismatches: list[str] = []
    parsed_count = 0
    exempt_count = 0
    annotated_count = 0

    for heading, body in split_sections(text):
        if is_exempt(heading):
            exempt_count += len(_DEADLINE_RE.findall(body))
            continue
        for m in _DEADLINE_RE.finditer(body):
            parsed_count += 1
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                deadline = dt.date(y, mo, d)
            except ValueError:
                mismatches.append(
                    f"under heading `{heading.strip()}`: malformed "
                    f"date {m.group(0)}")
                continue
            # Inspect context window ±120 chars for a passed-annotation.
            start = max(0, m.start() - 60)
            end = min(len(body), m.end() + 120)
            ctx = body[start:end]
            has_annotation = any(
                tag in ctx for tag in _PASSED_ANNOTATIONS
            )
            if deadline < today:
                if has_annotation:
                    annotated_count += 1
                    continue
                mismatches.append(
                    f"under heading `{heading.strip()}`: deadline "
                    f"{m.group(0)} ({deadline.isoformat()}) is in the "
                    f"past (today is {today.isoformat()}) and lacks "
                    "a `(passed)`, `(historical)`, or `has passed` "
                    "annotation. Either annotate the line or update "
                    "to the next active deadline.")

    print(f"# verify_deadline_freshness.py — {parsed_count} dated AOE "
          f"deadline(s) in submission sections; {annotated_count} "
          f"annotated as passed; {exempt_count} in exempt sections "
          f"(change log + post-submission)")

    if mismatches:
        for msg in mismatches:
            print(f"  {msg}", file=sys.stderr)
        print(f"# verify_deadline_freshness.py — {len(mismatches)} "
              "stale deadline(s)", file=sys.stderr)
        return 1
    print(f"# verify_deadline_freshness.py — all deadlines today-or-"
          f"future (or annotated as historical); today = "
          f"{today.isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
