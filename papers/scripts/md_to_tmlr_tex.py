#!/usr/bin/env python3
r"""md_to_tmlr_tex.py — convert papers/f2_methodology.md to TMLR LaTeX.

Walks the Markdown source, emits LaTeX into a fresh body file that the
TMLR template skeleton can \input{}. The math is already in
dollar-delimited LaTeX so it passes through; the prose, tables,
figures, and citations need template-specific markup.

Output: papers/tmlr_submission_kit/f2_methodology_body.tex

Mappings:
  ## 1. Foo       → \section{Foo}\label{sec:1}
  ### 1.2 Foo     → \subsection{Foo}\label{sec:1.2}
  ### A. Foo      → \section{Foo}\label{app:a}   (in appendix block)
  §X.Y            → \cref{sec:X.Y}
  arXiv:NNNN      → \citet{arxiv:NNNN}            (placeholder, BibTeX pending)
  `file.rs`       → \texttt{file.rs}
  | a | b |       → tabular (best-effort)
  $...$, $$...$$  → pass through unchanged

This is NOT pandoc — it's a TMLR-specific shaper. Pandoc handles
most of the heavy lifting but breaks on the §X.Y refs, the
backtick file paths, and the citation style. This script does
the §X.Y and cite shaping; pandoc would still be useful for the
prose-to-LaTeX baseline, but for now we ship a single-script
pipeline.

Usage:
  papers/scripts/md_to_tmlr_tex.py
"""

from __future__ import annotations

import datetime as _dt
import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
PAPER = CRATE_ROOT / "papers" / "f2_methodology.md"
OUT = CRATE_ROOT / "papers" / "tmlr_submission_kit" / "f2_methodology_body.tex"


HEADER_2_RE = re.compile(r"^##\s+(\d+)\.\s+(.*)$")
HEADER_3_RE = re.compile(r"^###\s+(\d+\.\d+(?:\.\d+)?)\s+(.*)$")
APPENDIX_HEADER_RE = re.compile(r"^###\s+([A-Z])\.\s+(.*)$")
SECTION_REF_RE = re.compile(r"§(\d+(?:\.\d+){0,2})")
ARXIV_RE = re.compile(r"arXiv:(\d{4}\.\d{4,5})")
BACKTICK_RE = re.compile(r"`([^`]+)`")
BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
ITALIC_RE = re.compile(r"\*([^*]+)\*")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def latex_escape(text: str) -> str:
    """Escape LaTeX specials in raw prose. Skip pieces wrapped in code-spans."""
    # Order matters: backslash first.
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("^", "\\^{}")
        .replace("~", "\\~{}")
    )


def transform_inline(line: str) -> str:
    """Apply inline-level Markdown → LaTeX transforms to a single line.

    Handles backticks, bold, italics, §refs, arXiv cites, links.
    Math (dollar-delimited) is left untouched.
    """
    # Carve out math segments so they pass through verbatim.
    segments: list[tuple[str, str]] = []  # (kind, payload) where kind ∈ {math, text}
    i = 0
    while i < len(line):
        if line.startswith("$$", i):
            # display math — find closing $$
            end = line.find("$$", i + 2)
            if end < 0:
                end = len(line)
            else:
                end += 2
            segments.append(("math", line[i:end]))
            i = end
        elif line[i] == "$":
            end = line.find("$", i + 1)
            if end < 0:
                end = len(line)
            else:
                end += 1
            segments.append(("math", line[i:end]))
            i = end
        else:
            # Find next math start
            next_math = len(line)
            for sentinel in ("$$", "$"):
                idx = line.find(sentinel, i)
                if 0 <= idx < next_math:
                    next_math = idx
            segments.append(("text", line[i:next_math]))
            i = next_math

    out: list[str] = []
    for kind, payload in segments:
        if kind == "math":
            out.append(payload)
            continue
        # Backtick code spans → \texttt{}; escape LaTeX inside but
        # preserve the literal content's _ % # & ~ ^.
        def _bt(m: re.Match) -> str:
            inner = m.group(1)
            inner = (
                inner.replace("\\", "\\textbackslash{}")
                .replace("_", "\\_")
                .replace("&", "\\&")
                .replace("%", "\\%")
                .replace("#", "\\#")
                .replace("~", "\\~{}")
                .replace("^", "\\^{}")
                .replace("{", "\\{")
                .replace("}", "\\}")
            )
            return f"\\texttt{{{inner}}}"

        payload = BACKTICK_RE.sub(_bt, payload)
        payload = BOLD_RE.sub(r"\\textbf{\1}", payload)
        payload = ITALIC_RE.sub(r"\\emph{\1}", payload)
        payload = SECTION_REF_RE.sub(r"\\cref{sec:\1}", payload)
        payload = ARXIV_RE.sub(r"\\citep{arxiv:\1}", payload)
        payload = LINK_RE.sub(r"\\href{\2}{\1}", payload)
        out.append(payload)
    return "".join(out)


def main() -> int:
    if not PAPER.exists():
        print(f"ERROR: {PAPER} not found", file=sys.stderr)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    src = PAPER.read_text().splitlines()

    body: list[str] = []
    when = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body.append(
        f"% f2_methodology_body.tex — auto-generated by md_to_tmlr_tex.py at {when}."
    )
    body.append(
        "% Do not edit by hand; modify papers/f2_methodology.md and re-run the script."
    )
    body.append("")

    in_table = False
    table_buf: list[str] = []
    in_code = False
    in_math = False
    in_appendix = False
    seen_first_section = False

    i = 0
    while i < len(src):
        line = src[i]
        # Code fence
        if line.startswith("```"):
            if in_code:
                body.append("\\end{verbatim}")
                in_code = False
            else:
                body.append("\\begin{verbatim}")
                in_code = True
            i += 1
            continue
        if in_code:
            body.append(line)
            i += 1
            continue

        # Display math block ($$ ... $$ on separate lines, or on one line)
        if line.strip().startswith("$$"):
            # Collect until matching $$
            block = [line]
            j = i + 1
            while j < len(src):
                block.append(src[j])
                if src[j].strip().endswith("$$") and (
                    j != i or src[j].strip() != "$$"
                ):
                    break
                j += 1
            # Emit verbatim — math should pass through
            body.extend(block)
            i = j + 1
            continue

        # Headers
        m = HEADER_2_RE.match(line)
        if m:
            num, title = m.group(1), m.group(2)
            seen_first_section = True
            body.append("")
            body.append(f"\\section{{{transform_inline(title)}}}\\label{{sec:{num}}}")
            body.append("")
            i += 1
            continue
        m = HEADER_3_RE.match(line)
        if m:
            num, title = m.group(1), m.group(2)
            body.append("")
            body.append(
                f"\\subsection{{{transform_inline(title)}}}\\label{{sec:{num}}}"
            )
            body.append("")
            i += 1
            continue
        m = APPENDIX_HEADER_RE.match(line)
        if m:
            letter, title = m.group(1), m.group(2)
            if not in_appendix:
                body.append("")
                body.append("\\appendix")
                body.append("")
                in_appendix = True
            body.append(
                f"\\section{{{transform_inline(title)}}}\\label{{app:{letter.lower()}}}"
            )
            body.append("")
            i += 1
            continue

        # Skip H1 (title) and Abstract section header — TMLR template has its
        # own abstract environment; the body just inherits.
        if line.startswith("# ") or line.startswith("## Abstract"):
            i += 1
            continue

        # Horizontal rule → ignore (TMLR doesn't want them)
        if line.strip() == "---":
            i += 1
            continue

        # Table detection (very simple: | a | b | …)
        if line.lstrip().startswith("|") and "|" in line[2:]:
            if not in_table:
                in_table = True
                table_buf = []
            table_buf.append(line)
            i += 1
            continue
        else:
            if in_table:
                # Flush table
                rows = [
                    [c.strip() for c in row.strip().strip("|").split("|")]
                    for row in table_buf
                    if not re.match(r"^\s*\|[-:|\s]+\|\s*$", row)
                ]
                if rows:
                    ncols = max(len(r) for r in rows)
                    colspec = "l" * ncols
                    body.append("\\begin{table}[h]")
                    body.append("\\centering")
                    body.append(f"\\begin{{tabular}}{{{colspec}}}")
                    body.append("\\toprule")
                    for k, row in enumerate(rows):
                        cells = [transform_inline(c) for c in row]
                        body.append(" & ".join(cells) + " \\\\")
                        if k == 0 and len(rows) > 1:
                            body.append("\\midrule")
                    body.append("\\bottomrule")
                    body.append("\\end{tabular}")
                    body.append("\\end{table}")
                    body.append("")
                in_table = False
                table_buf = []
            # Fall through to prose handling

        # List items
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m:
            indent, marker, content = m.group(1), m.group(2), m.group(3)
            # Skip elaborate list nesting — flat itemize / enumerate only.
            kind = "enumerate" if re.match(r"\d+\.", marker) else "itemize"
            # Look-back: if previous non-empty body line is not \begin{kind},
            # open the environment. Look-ahead: if next non-blank line is
            # not a list item or continuation, close it.
            if not body or not body[-1].startswith(f"\\begin{{{kind}}}") and not any(
                b.startswith("\\item")
                and not (body[-1].startswith("\\end") if body else True)
                for b in body[-3:]
            ):
                # Heuristic open
                if not body or not (
                    body[-1].endswith("\\item")
                    or (body[-1].lstrip().startswith("\\item"))
                ):
                    body.append(f"\\begin{{{kind}}}")
            body.append(f"\\item {transform_inline(content)}")
            # Lookahead: if next line is not list, close.
            j = i + 1
            while j < len(src) and not src[j].strip():
                j += 1
            if j >= len(src) or not re.match(r"^\s*([-*]|\d+\.)\s+", src[j]):
                body.append(f"\\end{{{kind}}}")
            i += 1
            continue

        # Blank line → preserve as paragraph break
        if not line.strip():
            body.append("")
            i += 1
            continue

        # Default: prose
        body.append(transform_inline(line))
        i += 1

    if in_table and table_buf:
        body.append("% (warning: unflushed table at EOF)")
    if in_code:
        body.append("\\end{verbatim}")

    OUT.write_text("\n".join(body) + "\n")
    nlines = OUT.read_text().count("\n")
    print(f"# Wrote {OUT.relative_to(CRATE_ROOT)} ({nlines} lines)")
    if not seen_first_section:
        print("# warning: no section headers detected; output may be incomplete")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
