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
INLINE_SUBSUB_RE = re.compile(r"^\*\*(\d+\.\d+\.\d+)\s+(.+?)\.\*\*\s*(.*)$")
SECTION_REF_RE = re.compile(r"§(\d+(?:\.\d+){0,2})")
ARXIV_RE = re.compile(r"arXiv:(\d{4}\.\d{4,5})")
BACKTICK_RE = re.compile(r"`([^`]+)`")
BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
ITALIC_RE = re.compile(r"\*([^*]+)\*")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


# Unicode characters that pdflatex with utf8inputenc cannot render. We
# substitute LaTeX equivalents in the prose segments (math segments are
# unaffected — their contents pass through to math-mode rendering, which
# handles these natively).
UNICODE_PROSE_MAP = {
    "−": "$-$",       # U+2212 minus sign (vs. ASCII hyphen-minus)
    "—": "---",       # U+2014 em dash
    "–": "--",        # U+2013 en dash
    "…": "\\dots{}",  # U+2026 horizontal ellipsis
    "×": "$\\times$", # U+00D7
    "≥": "$\\geq$",   # U+2265
    "≤": "$\\leq$",   # U+2264
    "≈": "$\\approx$",# U+2248
    "→": "$\\to$",    # U+2192
    "±": "$\\pm$",    # U+00B1
    "·": "$\\cdot$",  # U+00B7
    "≪": "$\\ll$",    # U+226A
    "≫": "$\\gg$",    # U+226B
    "∞": "$\\infty$", # U+221E
    "ε": "$\\varepsilon$",  # U+03B5
    "θ": "$\\theta$",
    "λ": "$\\lambda$",
    "Λ": "$\\Lambda$",
    "γ": "$\\gamma$",
    "Γ": "$\\Gamma$",
    "Δ": "$\\Delta$",
    "Σ": "$\\Sigma$",
    "σ": "$\\sigma$",
    "μ": "$\\mu$",
    "π": "$\\pi$",
    "η": "$\\eta$",
    "φ": "$\\varphi$",
    "²": "$^2$",      # superscript 2
    "³": "$^3$",      # superscript 3
    "⁻": "$^{-}$",
    "⁶": "$^6$",
    "₁": "$_1$",
    "₂": "$_2$",
    "⊆": "$\\subseteq$",
    "∈": "$\\in$",
    "∀": "$\\forall$",
    "∃": "$\\exists$",
}


def unicode_to_latex(text: str) -> str:
    """Replace UTF-8 prose specials with LaTeX equivalents."""
    for ch, repl in UNICODE_PROSE_MAP.items():
        text = text.replace(ch, repl)
    return text


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
        # The typewriter font (Computer Modern Typewriter) lacks many
        # unicode symbols. Substitute ASCII equivalents inside texttt
        # so they actually render in the PDF (Loop 85 visual catch).
        BT_UNICODE_FALLBACK = {
            "≥": ">=",
            "≤": "<=",
            "≠": "!=",
            "≈": "~",
            "±": "+/-",
            "×": "x",
            "·": ".",
            "≪": "<<",
            "≫": ">>",
            "→": "->",
            "←": "<-",
            "⇒": "=>",
            "…": "...",
            "—": "--",
            "–": "-",
            # Combining macron / acute / etc. — strip; the base char remains.
            "̄": "",
            "́": "",
            "̀": "",
            "̃": "",
            "̂": "",
            "̈": "",
        }

        def _bt(m: re.Match) -> str:
            inner = m.group(1)
            for ch, repl in BT_UNICODE_FALLBACK.items():
                inner = inner.replace(ch, repl)
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

        # Carve out backtick content FIRST, so its native UTF-8 (Γ, Λ,
        # σ etc.) survives unmolested inside \texttt{} (xelatex handles
        # these via fontspec). Otherwise unicode_to_latex would replace
        # Γ → $\Gamma$ inside the backtick span, then _bt would escape
        # the backslash to \textbackslash{}, producing literal "\{}Gamma"
        # in the PDF.
        bt_placeholders: list[str] = []
        def _bt_capture(m: re.Match) -> str:
            bt_placeholders.append(_bt(m))
            return f"\x00BT{len(bt_placeholders) - 1}\x00"
        payload = BACKTICK_RE.sub(_bt_capture, payload)
        # NOW translate unicode prose specials to LaTeX math/macros.
        payload = unicode_to_latex(payload)
        # Now escape LaTeX specials in the surviving prose. `_` and `^`
        # are illegal in text mode (subscript/superscript triggers); we
        # escape them everywhere outside backtick blocks.
        payload = re.sub(r"(?<!\\)([&%#])", r"\\\1", payload)
        payload = re.sub(r"(?<!\\)_", r"\\_", payload)
        payload = re.sub(r"(?<!\\)\^", r"\\^{}", payload)
        # Markdown transforms (operate on escaped prose; the escapes survive).
        payload = BOLD_RE.sub(r"\\textbf{\1}", payload)
        payload = ITALIC_RE.sub(r"\\emph{\1}", payload)
        payload = SECTION_REF_RE.sub(r"\\cref{sec:\1}", payload)
        # arXiv id format like "2007.16031" contains a `.` (safe) but our
        # earlier `_` escape may have hit it; ARXIV_RE is anchored on the
        # literal "arXiv:" prefix which we have not modified, so this is fine.
        payload = ARXIV_RE.sub(r"\\citep{arxiv:\1}", payload)
        payload = LINK_RE.sub(r"\\href{\2}{\1}", payload)
        # Restore backtick blocks
        for k, repl in enumerate(bt_placeholders):
            payload = payload.replace(f"\x00BT{k}\x00", repl)
        out.append(payload)
    return "".join(out)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PAPER,
        help=f"Markdown source (default: {PAPER.relative_to(CRATE_ROOT)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT,
        help=f"LaTeX output (default: {OUT.relative_to(CRATE_ROOT)})",
    )
    args = parser.parse_args()
    paper = args.input
    out = args.output
    if not paper.exists():
        print(f"ERROR: {paper} not found", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    raw = paper.read_text()
    # Strip HTML comments (e.g. the anonymizer banner). They render as
    # prose in xelatex if left in.
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)
    src = raw.splitlines()

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
        m = INLINE_SUBSUB_RE.match(line)
        if m:
            num, title, tail = m.group(1), m.group(2), m.group(3)
            body.append("")
            body.append(
                f"\\subsubsection{{{transform_inline(title)}}}\\label{{sec:{num}}}"
            )
            body.append("")
            if tail.strip():
                body.append(transform_inline(tail))
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

        # List items — buffer the whole list (items + indented
        # continuation lines + blank separator lines) into one chunk.
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m:
            marker = m.group(2)
            kind = "enumerate" if re.match(r"\d+\.", marker) else "itemize"
            list_lines: list[tuple[str, str]] = []  # (role, content)
            j = i
            while j < len(src):
                lj = src[j]
                mi = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lj)
                if mi:
                    list_lines.append(("item", mi.group(3)))
                    j += 1
                    continue
                if not lj.strip():
                    # blank — peek ahead: if next non-blank is item or
                    # indented continuation, treat as separator; else stop
                    k = j + 1
                    while k < len(src) and not src[k].strip():
                        k += 1
                    if k < len(src) and (
                        re.match(r"^(\s*)([-*]|\d+\.)\s+", src[k])
                        or src[k].startswith("   ")
                    ):
                        list_lines.append(("blank", ""))
                        j += 1
                        continue
                    break
                if lj.startswith("   ") or lj.startswith("\t"):
                    list_lines.append(("cont", lj.lstrip()))
                    j += 1
                    continue
                break
            body.append(f"\\begin{{{kind}}}")
            current: list[str] = []
            for role, content in list_lines:
                if role == "item":
                    if current:
                        body.append("\\item " + " ".join(current))
                        current = []
                    current.append(transform_inline(content))
                elif role == "cont":
                    current.append(transform_inline(content))
                elif role == "blank":
                    pass
            if current:
                body.append("\\item " + " ".join(current))
            body.append(f"\\end{{{kind}}}")
            i = j
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

    out = out.resolve()
    out.write_text("\n".join(body) + "\n")
    nlines = out.read_text().count("\n")
    try:
        display = out.relative_to(CRATE_ROOT)
    except ValueError:
        display = out
    print(f"# Wrote {display} ({nlines} lines)")
    if not seen_first_section:
        print("# warning: no section headers detected; output may be incomplete")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
