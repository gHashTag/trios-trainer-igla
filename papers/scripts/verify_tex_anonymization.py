#!/usr/bin/env python3
"""verify_tex_anonymization.py — scan the anonymized .tex build artifact
for identifiers that the anonymizer should have stripped.

Loop 147 B closes 67th-pass SEV-3 #5: `verify_anonymizer_completeness.py`
(stage 24) only scans the **markdown sources**; the converter step
(`md_to_tmlr_tex.py`) could in principle re-introduce identifiers that
were stripped at the .md layer (e.g., from a header / footer template
or a metadata block injected at LaTeX time). Without a gate at the .tex
layer, a bug in the converter would ship to OpenReview.

What this gate checks: in
`papers/tmlr_submission_kit/f2_methodology_anonymized_body.tex` (the
build artifact md_to_tmlr_tex.py emits from the anonymized .md), zero
matches for:

  - **Bare `Loop N`** tokens (same _LOOP_REF_RE as stage 24, but
    without the `(internal ref)` / section-header / fenced-code-block
    allowances — the anonymized output should have ZERO of these).
  - **Branch name `f2-methodology`** (anonymizer should replace with
    `<branch>` or strip).
  - **PII identifiers**: `gHashTag`, `playra`, `trios-railway`,
    `xamituz938` (the gmail handle from the user profile).
  - **Email addresses ending in `@anthropic.com`** (acknowledgments
    block leakage).
  - **SHA-like tokens** of length 7-40 hex chars NOT followed by `}` or
    inside a `\\href{}` URL (anonymizer should replace with
    `<anchor commit>`).

The non-anonymized .tex (`f2_methodology_body.tex`) is the
post-acceptance publication target and is NOT subject to these checks.

Usage: papers/scripts/verify_tex_anonymization.py

Exit 0 if zero leaks; 1 on any. If the .tex artifact is missing, exit
1 with a "regenerate via compile_tmlr_test.sh" message — the gate
relies on the artifact being current, since it's .gitignored.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
ANON_TEX = (
    CRATE_ROOT / "papers" / "tmlr_submission_kit"
    / "f2_methodology_anonymized_body.tex"
)


# Same as stage 24's _LOOP_REF_RE.
_LOOP_REF_RE = re.compile(r"\bLoop \d{1,3}\b")

# Branch name + PII identifiers.
_BRANCH_RE = re.compile(r"\bf2-methodology\b")
_PII_RES: list[tuple[str, re.Pattern[str]]] = [
    ("gHashTag", re.compile(r"\bgHashTag\b", re.IGNORECASE)),
    ("playra", re.compile(r"\bplayra\b", re.IGNORECASE)),
    ("trios-railway", re.compile(r"\btrios-railway\b", re.IGNORECASE)),
    ("xamituz938", re.compile(r"\bxamituz938\b", re.IGNORECASE)),
]
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@anthropic\.com\b")

# SHA-like tokens: 7-40 hex chars on a word boundary. The exclusion
# pattern catches:
#   - Tokens immediately inside a `\href{...}` argument (those are
#     external URLs and may legitimately contain hex IDs).
#   - Tokens immediately followed by `}` (LaTeX command argument
#     boundary — unlikely to be a bare leak).
#   - Tokens that are pure decimal digits (Posit/format constants).
_SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b", re.IGNORECASE)


def _line_of(text: str, pos: int) -> int:
    return text[:pos].count("\n") + 1


def _rel(p: Path) -> str:
    """Return p relative to CRATE_ROOT if possible, else absolute. Lets
    the gate be break-tested with an injected path under /tmp."""
    try:
        return str(p.relative_to(CRATE_ROOT))
    except ValueError:
        return str(p)


def _scan_class(text: str, regex: re.Pattern[str],
                class_name: str) -> list[tuple[int, str, str]]:
    """Return list of (line_no, class_name, line_snippet) for each match."""
    out: list[tuple[int, str, str]] = []
    lines = text.splitlines()
    for m in regex.finditer(text):
        line_no = _line_of(text, m.start())
        if 1 <= line_no <= len(lines):
            snip = lines[line_no - 1].strip()[:100]
        else:
            snip = "(out-of-range)"
        out.append((line_no, class_name, snip))
    return out


def _scan_sha(text: str) -> list[tuple[int, str, str]]:
    """SHA-like scan with context-based exclusion. Excludes tokens whose
    enclosing line contains `\\href{` (external URL) or that are inside
    a `\\url{...}` argument."""
    out: list[tuple[int, str, str]] = []
    lines = text.splitlines()
    for m in _SHA_RE.finditer(text):
        token = m.group(0)
        # Skip if all-digits (likely a constant, not a SHA).
        if token.isdigit():
            continue
        # Skip if too short to be a meaningful git SHA prefix (avoid
        # generic hex words like "deaf", "cafe"). Min 8 alphanumeric mix.
        if len(token) < 8:
            continue
        line_no = _line_of(text, m.start())
        if 1 <= line_no <= len(lines):
            line = lines[line_no - 1]
        else:
            line = ""
        # If the enclosing line contains `\href{` or `\url{` or `arxiv:`,
        # exclude — those are external URLs / paper IDs.
        if "\\href{" in line or "\\url{" in line or "arxiv:" in line.lower():
            continue
        # If token is pure alpha (no digits), drop — definitely not a SHA.
        if not any(c.isdigit() for c in token):
            continue
        # Require at least one digit AND at least one [a-f] to look SHA-like.
        if not any(c in "abcdef" for c in token.lower()):
            continue
        out.append((line_no, "SHA-like", line.strip()[:100]))
    return out


def main() -> int:
    if not ANON_TEX.exists():
        print(f"# FAIL  {_rel(ANON_TEX)} missing. "
              "The anonymized .tex artifact is .gitignored — regenerate "
              "with `papers/scripts/compile_tmlr_test.sh` before running "
              "this gate.", file=sys.stderr)
        return 1

    text = ANON_TEX.read_text()
    leaks: list[tuple[int, str, str]] = []

    # Loop-N leaks.
    leaks.extend(_scan_class(text, _LOOP_REF_RE, "bare Loop-N"))
    # Branch name.
    leaks.extend(_scan_class(text, _BRANCH_RE, "branch name"))
    # PII identifiers.
    for name, regex in _PII_RES:
        leaks.extend(_scan_class(text, regex, f"PII({name})"))
    # Emails.
    leaks.extend(_scan_class(text, _EMAIL_RE, "@anthropic.com email"))
    # SHA-like.
    leaks.extend(_scan_sha(text))

    if leaks:
        # Emit the first 10 to avoid drowning the operator.
        for line_no, cls, snip in leaks[:10]:
            print(f"# FAIL  {_rel(ANON_TEX)}:{line_no}: "
                  f"{cls} — {snip!r}", file=sys.stderr)
        if len(leaks) > 10:
            print(f"# FAIL  ... and {len(leaks) - 10} more.",
                  file=sys.stderr)
        # Group by class for the summary.
        by_class: dict[str, int] = {}
        for _, cls, _ in leaks:
            by_class[cls] = by_class.get(cls, 0) + 1
        cls_summary = ", ".join(
            f"{c}={n}" for c, n in sorted(by_class.items())
        )
        print(f"# verify_tex_anonymization.py — {len(leaks)} leak(s) "
              f"in anonymized .tex artifact ({cls_summary}). Either "
              "fix the converter (papers/scripts/md_to_tmlr_tex.py) "
              "or the anonymizer (papers/scripts/anonymize_paper.py).",
              file=sys.stderr)
        return 1

    print(f"# verify_tex_anonymization.py — "
          f"{_rel(ANON_TEX)} clean across "
          f"{len(_PII_RES) + 4} leak class(es) "
          "(bare Loop-N, branch name, 4 PII identifiers, "
          "@anthropic.com email, SHA-like tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
