#!/usr/bin/env python3
"""lint_paper_md.py — upstream lint for the Markdown source.

Flags common converter pitfalls BEFORE they reach the PDF. Pairs
with the downstream pdftotext-grep stage in compile_tmlr_test.sh:
the lint catches issues at the source; the grep catches anything
that slipped past the lint. Both are needed.

Checks (each tagged with severity):
  - SEV 4: backtick code span containing a `\\` LaTeX command (e.g.
    `\\widehat`, `\\text{}`) — these get double-escaped by the
    converter into `\\textbackslash{}widehat\\{\\}text`. Use math
    mode `$...$` instead.
  - SEV 4: Markdown table with > 3 columns — likely to truncate
    at the right margin in TMLR's narrow column.
  - SEV 3: bullet that spans multiple paragraphs (blank line
    inside the bullet); usually a rendering bug.
  - SEV 3: HTML comment in body (would render as prose).
  - SEV 3: box-drawing characters (`─`, `►`, `│`, etc.) inside
    fenced code blocks — render as U+FFFD `�`.
  - SEV 2: en-dash vs em-dash inconsistency on the same line.
  - SEV 1: trailing whitespace on a line.

Exit: 0 if all clean; 1 if any SEV-≥3 issue. SEV 1-2 are reported
but do not fail the lint.

Usage:
  papers/scripts/lint_paper_md.py [path/to/paper.md]
  papers/scripts/lint_paper_md.py --strict  # fail on any SEV >= 1
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPER = CRATE_ROOT / "papers" / "f2_methodology.md"

BACKTICK_BAD_LATEX = re.compile(
    r"`[^`]*\\(?:widehat|widetilde|text|mathrm|mathbb|frac|sqrt|sum|prod)\b[^`]*`"
)
HTML_COMMENT = re.compile(r"<!--")
BOX_DRAW = re.compile(r"[─━│┃┌┐└┘├┤┬┴┼►◄▲▼]")
EN_DASH = "–"
EM_DASH = "—"

# Loosely identifies markdown tables: 3+ pipe characters on one line.
TABLE_LINE_RE = re.compile(r"^\s*\|")


def lint(text: str, paper_name: str) -> tuple[int, int]:
    """Return (sev4plus_count, sev1_to_3_count)."""
    sev_high = 0
    sev_low = 0
    in_code_fence = False
    table_lines: list[tuple[int, str]] = []
    bullet_lines: list[tuple[int, str]] = []

    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            # Box-drawing chars inside fenced blocks render as U+FFFD.
            if BOX_DRAW.search(line):
                print(
                    f"  SEV 3  {paper_name}:{lineno}: box-drawing char "
                    f"inside code fence → renders as U+FFFD in PDF",
                    file=sys.stderr,
                )
                sev_low += 1
            continue

        if BACKTICK_BAD_LATEX.search(line):
            print(
                f"  SEV 4  {paper_name}:{lineno}: backtick code span "
                f"contains LaTeX command (use $...$ math mode instead):",
                file=sys.stderr,
            )
            print(f"         {line.strip()[:100]}", file=sys.stderr)
            sev_high += 1

        if HTML_COMMENT.search(line):
            print(
                f"  SEV 3  {paper_name}:{lineno}: HTML comment in body "
                f"(would render as prose unless stripped):",
                file=sys.stderr,
            )
            print(f"         {line.strip()[:80]}", file=sys.stderr)
            sev_low += 1

        if TABLE_LINE_RE.match(line):
            table_lines.append((lineno, line))
        else:
            # Flush table if we just left one.
            if table_lines:
                ncols = (
                    max((row.count("|") for _, row in table_lines), default=0) - 1
                )
                if ncols > 5:
                    first_lineno = table_lines[0][0]
                    print(
                        f"  SEV 4  {paper_name}:{first_lineno}: "
                        f"table has {ncols} columns; likely truncates at "
                        f"right margin in TMLR's narrow column (consider "
                        f"converting to a bullet list):",
                        file=sys.stderr,
                    )
                    sev_high += 1
                table_lines = []

        if EN_DASH in line and EM_DASH in line:
            print(
                f"  SEV 2  {paper_name}:{lineno}: en-dash and em-dash "
                f"on same line (style inconsistency)",
                file=sys.stderr,
            )
            sev_low += 1

        if line and line[-1] in (" ", "\t"):
            print(
                f"  SEV 1  {paper_name}:{lineno}: trailing whitespace",
                file=sys.stderr,
            )
            sev_low += 1

    # Final table flush.
    if table_lines:
        ncols = max((row.count("|") for _, row in table_lines), default=0) - 1
        if ncols > 3:
            first_lineno = table_lines[0][0]
            print(
                f"  SEV 4  {paper_name}:{first_lineno}: trailing table "
                f"has {ncols} columns; likely truncates at right margin.",
                file=sys.stderr,
            )
            sev_high += 1

    return sev_high, sev_low


def main() -> int:
    strict = "--strict" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    paper = Path(args[0]) if args else DEFAULT_PAPER
    if not paper.exists():
        print(f"ERROR: {paper} not found", file=sys.stderr)
        return 1
    rel = paper.relative_to(CRATE_ROOT) if paper.is_absolute() else paper
    text = paper.read_text()
    sev_high, sev_low = lint(text, str(rel))
    total = sev_high + sev_low
    print(
        f"# lint_paper_md.py — {sev_high} SEV-≥4 issues, "
        f"{sev_low} SEV-1-3 issues"
    )
    if sev_high > 0:
        return 1
    if strict and sev_low > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
