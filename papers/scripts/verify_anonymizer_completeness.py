#!/usr/bin/env python3
"""verify_anonymizer_completeness.py — gate paper bodies for bare
`Loop N` anchors that would leak through TMLR double-blind
anonymization.

Loop 132 C operationalizes the 54th-pass catch #9 *class*: the
phi_ladder line-646 `Loop 109 32nd-pass correction` was a bare
Loop-N anchor in paper prose (not in a section header, not in a
`(internal ref)` parenthetical). The anonymization pipeline
`papers/scripts/anonymize_paper.py` strips known internal terms
(`gHashTag`, `playra`, `trios-railway`, `@anthropic.com`) but does
not normalize `Loop N` references, so bare anchors survive into
the double-blind submission. A TMLR reviewer reading "Loop 109
32nd-pass correction" gets de-anonymizing signal (Loop discipline
is unique to this project's branch history).

This gate scans the two TMLR-bound papers
(`papers/f2_methodology.md` and `papers/phi_ladder_paper_intro_draft.md`)
for `\\bLoop \\d+\\b` occurrences and asserts each is in an allowed
context:
  (a) inside a Markdown section header (`#`/`##`/...).
  (b) inside an italic parenthetical that begins with "internal ref"
      (case-insensitive).
  (c) inside an HTML comment `<!-- ... -->`.
  (d) inside a fenced code block (` ``` ` or ` ~~~ `).

Any other occurrence is reported as an anonymizer leak.

**Ratchet mode**: the two papers carry legacy bare-anchor debt (70
occurrences at Loop 132 baseline — historical Loop references in
prose like "Loop 49 RmsNorm sign flip", "Loop 102 reported"). These
are tracked per-file as BASELINE counts; new additions FAIL the gate
but existing debt is allowed-with-warning. Reduction below baseline
is reported as "good news". A future cleanup loop can burn down the
debt by rewriting prose to use `(internal ref)` form and dropping
the BASELINE entries.

CHANGELOG.md, SUBMISSION_CHECKLIST.md, and ADVERSARIAL_REVIEW_LOG.md
are intentionally NOT scanned — they are project-internal admin docs
that don't ship in the submission bundle. The supplementary zip's
contents are also out of scope (gated separately by the supplementary
pack stage's pre-flight).

Usage: papers/scripts/verify_anonymizer_completeness.py

Exit 0 if no new leaks beyond baseline; 1 if any file's leak count
exceeds its registered baseline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]

# Loop 134 B: baselines now live in a JSON sidecar so a burn-down
# loop's baseline bump is a one-line diff to data, not a code edit.
# The sidecar path is fixed; if the file is missing or malformed,
# the gate falls back to FALLBACK_BASELINES and warns. This preserves
# the ratchet semantics — the gate never silently auto-lowers.
BASELINE_SIDECAR = CRATE_ROOT / "papers" / "scripts" / "anonymizer_baseline.json"

# Files subject to TMLR anonymization. Per-file baselines come from
# the sidecar (FALLBACK_BASELINES is the rescue value if the sidecar
# can't be read). Burn-down trajectory:
#   Loop 132 C baseline: 29 + 43 = 72.
#   Loop 133 A.iv (F2 §E catalogue): 22 + 43 = 65.
#   Loop 134 A.iii (#1021 §5.4 partition): 22 + 39 = 61.
#   Loop 135 A.iii (#1021 §5.4 second pass): 22 + 32 = 54.
#   Loop 136 A.iii (F2 §E remaining + §3.2/§4/§6.1): 16 + 32 = 48.
#   Loop 137 A.iii (#1021 §5.4 Gate-decomp prelude + 37th-pass refs): 16 + 27 = 43.
FALLBACK_BASELINES = {
    "papers/f2_methodology.md": 16,
    "papers/phi_ladder_paper_intro_draft.md": 27,
}


def _load_baselines() -> tuple[dict[str, int], list[str]]:
    """Read sidecar JSON; return (baselines, warnings)."""
    warnings: list[str] = []
    if not BASELINE_SIDECAR.exists():
        warnings.append(
            f"sidecar {BASELINE_SIDECAR.relative_to(CRATE_ROOT)} missing; "
            "using FALLBACK_BASELINES (in-code defaults)")
        return dict(FALLBACK_BASELINES), warnings
    try:
        data = json.loads(BASELINE_SIDECAR.read_text())
    except json.JSONDecodeError as e:
        warnings.append(
            f"sidecar JSON parse error ({e}); using FALLBACK_BASELINES")
        return dict(FALLBACK_BASELINES), warnings
    bl = data.get("baselines")
    if not isinstance(bl, dict):
        warnings.append(
            "sidecar lacks 'baselines' dict; using FALLBACK_BASELINES")
        return dict(FALLBACK_BASELINES), warnings
    out: dict[str, int] = {}
    for k, v in bl.items():
        # Loop 135 — 58th-pass SEV-3 fix #5: validate keys explicitly.
        # Without this, a typo'd path silently produces a `WARN file
        # missing; skipping` downstream that an operator can miss.
        if not isinstance(k, str) or not re.fullmatch(r"papers/.+\.md", k):
            warnings.append(
                f"sidecar baseline key {k!r} not 'papers/...md' shape; "
                "skipping")
            continue
        abs_path = CRATE_ROOT / k
        if not abs_path.exists():
            warnings.append(
                f"sidecar baseline key {k!r} points at non-existent "
                f"file {abs_path.relative_to(CRATE_ROOT)}; skipping")
            continue
        if isinstance(v, int) and v >= 0:
            out[k] = v
        else:
            warnings.append(
                f"sidecar baseline {k!r}={v!r} not non-negative int; "
                "skipping")
    return out, warnings


def _scan_targets() -> list[tuple[Path, int]]:
    """Resolve baselines into (absolute_path, baseline) tuples."""
    bl, warnings = _load_baselines()
    for w in warnings:
        print(f"# WARN  {w}", file=sys.stderr)
    return [(CRATE_ROOT / rel, count) for rel, count in bl.items()]


# Computed at module-init for backward compatibility with old import
# patterns that expected SCAN_TARGETS to be a module-level constant.
SCAN_TARGETS = _scan_targets()


# Strip ATX-style code fences and HTML comments before pattern
# matching, but preserve line-number fidelity by replacing fenced
# content with same-line-count whitespace. The fence regex matches
# both ``` and ~~~ openers with optional language tag, paired with
# the same fence (matching CommonMark §4.5 — we don't allow
# crossing fence types).
_FENCE_RE = re.compile(
    r"^(?P<fence>```+|~~~+)[^\n]*\n(?P<body>.*?)\n^(?P=fence)\s*$",
    re.MULTILINE | re.DOTALL,
)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def _blank_out(text: str, regex: re.Pattern) -> str:
    """Replace each regex match with whitespace of the same length to
    preserve line-number alignment for downstream scans."""
    def _replacer(m: re.Match) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))
    return regex.sub(_replacer, text)


# Match `Loop <N>` where N is 1-3 digit. Use \b boundary at both ends
# so "loops" / "Loophole" don't false-positive. Case-sensitive — we
# care about the canonical project usage which is always capitalized.
_LOOP_REF_RE = re.compile(r"\bLoop \d{1,3}\b")


def _line_of(text: str, pos: int) -> int:
    """1-indexed line number containing the offset `pos`."""
    return text[:pos].count("\n") + 1


def _line_text(text: str, line_no: int) -> str:
    """Return the full text of the 1-indexed line `line_no`."""
    lines = text.splitlines()
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1]
    return ""


def _is_allowed(text: str, scrubbed: str, match: re.Match) -> str | None:
    """Return None if the Loop-N occurrence is allowed, else a string
    describing the leak category (used in the diagnostic).

    `scrubbed` has fenced code blocks + HTML comments zeroed out, so
    if the original match position falls inside a whitespace region,
    the match was inside a code fence — categorized as allowed (d).
    """
    pos = match.start()
    # (d) Inside fenced code block or HTML comment — scrubbed text has
    # whitespace at this position.
    if scrubbed[pos] == " " or scrubbed[pos] == "\n":
        return None
    # Otherwise inspect the line.
    line_no = _line_of(text, pos)
    line = _line_text(text, line_no)
    stripped = line.lstrip()
    # (a) Section header: line starts with one or more `#` followed by space.
    if re.match(r"^#{1,6}\s", stripped):
        return None
    # (b) Italic / parenthetical "internal ref" — look for the smallest
    # enclosing parenthetical that contains the match, then check if its
    # opener contains "internal ref" (case-insensitive). The parenthetical
    # may span multiple lines, so search backward from `pos` for `(` and
    # forward for the matching `)`.
    paren_open = text.rfind("(", 0, pos)
    paren_close = text.find(")", pos)
    if paren_open != -1 and paren_close != -1:
        between = text[paren_open + 1:paren_close]
        # Only accept if the parenthetical clearly opens with the
        # "internal ref" idiom — this prevents false-allow from any
        # parenthetical happening to contain "Loop 109" deep inside.
        # We accept either "internal ref" or "adversarial-pass" framing.
        head = between[:80].lower()
        if "internal ref" in head or "adversarial-pass" in head:
            return None
    # No allowed context matched — this is a leak.
    return f"bare Loop-N anchor (line text: {stripped[:80]!r})"


def scan_file(path: Path) -> list[tuple[int, str]]:
    """Return list of (line_no, leak_description) for each disallowed
    Loop-N occurrence. Empty list means file is clean."""
    text = path.read_text()
    scrubbed = _blank_out(text, _FENCE_RE)
    scrubbed = _blank_out(scrubbed, _HTML_COMMENT_RE)
    leaks: list[tuple[int, str]] = []
    for m in _LOOP_REF_RE.finditer(text):
        reason = _is_allowed(text, scrubbed, m)
        if reason is not None:
            line_no = _line_of(text, m.start())
            leaks.append((line_no, reason))
    return leaks


def _update_baseline_json(new_counts: dict[str, int]) -> None:
    """Loop 136 B: atomically rewrite BASELINE_SIDECAR with new
    per-file counts. Preserves the "_comment" key and any other
    top-level metadata."""
    if BASELINE_SIDECAR.exists():
        data = json.loads(BASELINE_SIDECAR.read_text())
    else:
        data = {"baselines": {}}
    data["baselines"] = new_counts
    # Atomic write via temp file + rename.
    tmp = BASELINE_SIDECAR.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(BASELINE_SIDECAR)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update-baseline", action="store_true",
        help="On GOOD outcome (any file is below baseline), rewrite "
             "anonymizer_baseline.json with the new lower counts and "
             "exit 0 with 'ratchet re-armed' message. WARNING: this is "
             "a destructive action — the ratchet re-arms at the new "
             "lower bound and prevents a future regression to the "
             "previous baseline.",
    )
    args = parser.parse_args()

    over_baseline = 0
    total_leaks = 0
    total_scanned = 0
    new_counts: dict[str, int] = {}  # for --update-baseline
    # Loop 134 — 57th-pass SEV-4 fix #10: per-file disposition
    # tracking. Aggregate n_good/n_at/n_over so the final summary
    # surfaces all three counts; previously the summary lost burn-
    # down signal when one file went below baseline and another
    # stayed at-baseline.
    n_good = n_at = n_over = 0
    net_delta = 0  # sum of (current - baseline) across all files
    for path, baseline in SCAN_TARGETS:
        if not path.exists():
            print(f"# WARN  {path.relative_to(CRATE_ROOT)}: file missing; "
                  "skipping", file=sys.stderr)
            continue
        leaks = scan_file(path)
        total_scanned += 1
        total_leaks += len(leaks)
        net_delta += len(leaks) - baseline
        rel = path.relative_to(CRATE_ROOT)
        # Track new counts for --update-baseline. Always = current leak
        # count (over-baseline files keep their elevated count, GOOD
        # files lower, at-baseline stay same).
        new_counts[str(rel)] = len(leaks)
        if len(leaks) > baseline:
            over_baseline += len(leaks) - baseline
            # Emit only the LAST diff (most likely new addition) to
            # avoid drowning the operator in legacy-debt noise. A
            # full leak list is available by removing the slice.
            for line_no, reason in leaks[-min(10, len(leaks)):]:
                print(f"# FAIL  {rel}:{line_no}: {reason}",
                      file=sys.stderr)
            print(f"# FAIL  {rel}: {len(leaks)} bare anchors > "
                  f"baseline {baseline}. New additions must use "
                  "section-header or `(internal ref)` form.",
                  file=sys.stderr)
            n_over += 1
        elif len(leaks) < baseline:
            print(f"# GOOD  {rel}: {len(leaks)} bare anchors < "
                  f"baseline {baseline} — debt reduced by "
                  f"{baseline - len(leaks)}. Edit the baseline in "
                  f"papers/scripts/anonymizer_baseline.json to "
                  f"{len(leaks)} so the ratchet re-arms at the new "
                  f"lower bound.")
            n_good += 1
        else:
            print(f"# OK    {rel}: {len(leaks)} bare anchors == "
                  f"baseline {baseline} (legacy debt unchanged)")
            n_at += 1
    delta_str = (f"net Δ {net_delta:+d}" if net_delta != 0
                 else "net Δ 0")
    if args.update_baseline:
        if over_baseline > 0:
            print(f"# --update-baseline refused: {over_baseline} new "
                  f"bare anchor(s) beyond baseline. Fix the additions "
                  "first (cannot ratchet a regression).",
                  file=sys.stderr)
            return 1
        _update_baseline_json(new_counts)
        new_total = sum(new_counts.values())
        print(f"# verify_anonymizer_completeness.py — ratchet "
              f"re-armed at new baselines: total {new_total} "
              f"(was {sum(b for _, b in SCAN_TARGETS)}).")
        return 0
    if over_baseline > 0:
        print(f"# verify_anonymizer_completeness.py — {over_baseline} "
              f"NEW bare Loop-N anchor(s) beyond baseline across "
              f"{total_scanned} TMLR-bound papers ({n_over} over, "
              f"{n_at} at-baseline, {n_good} below; {delta_str}). "
              f"Total leak count {total_leaks}.",
              file=sys.stderr)
        return 1
    print(f"# verify_anonymizer_completeness.py — {total_scanned} "
          f"TMLR-bound papers scanned ({n_at} at-baseline, "
          f"{n_good} below; {delta_str}); {total_leaks} leak(s) "
          "at-or-below baseline (legacy debt ratchet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
