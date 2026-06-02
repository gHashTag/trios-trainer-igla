#!/usr/bin/env python3
"""regen_changelog_section7.py — generate a "passes 12-N at a glance"
table from the f2-methodology branch commit history.

Loop 132 B operationalizes the 54th-pass catch #7+#8 *class*: CHANGELOG
§7 accumulates per-loop "Loop N — <summary>" entries that are
hand-maintained and inevitably drift. Rather than fixing each drift
case, this generator extracts the cumulative state from
`git log --grep="adversarial pass" --grep="round-"` (the canonical
breadcrumb already referenced in the §7 lead paragraph) and emits a
compact Markdown table.

This is **informational** — the output `papers/CHANGELOG_section7_generated.md`
is NOT wired into the CI gate. The hand-maintained CHANGELOG.md §7
prose remains authoritative; this generator is a parallel artifact
that makes drift visible at-a-glance.

Future loops can:
  - Compare generated vs hand-maintained at submission time.
  - Eventually retire the hand-maintained §7 narrative and let the
    generator carry the cumulative state.
  - Add a CI gate that asserts (generated_count == hand_maintained_count).

Usage: papers/scripts/regen_changelog_section7.py
       [--output <path>] [--branch <name>]

Exit 0 if generation succeeds; 1 on git or parse errors.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = CRATE_ROOT / "papers" / "CHANGELOG_section7_generated.md"
DEFAULT_BRANCH = "f2-methodology"

# Capture either "(Loop N ...)" or "Loop N" or "(Loop N X)" forms from
# the first line of a commit message. The crate's commit convention
# is `<type>(<scope>): <short msg> (Loop N <part>)` with the loop tag
# in trailing parens, but earlier loops used inline forms — we accept
# both.
_LOOP_RE = re.compile(r"\bLoop[\s-]+(\d{1,4})\b")


def run_git_log(branch: str) -> list[str]:
    """Return one line per commit on `branch` whose message mentions
    "adversarial pass" or "round-N" — the canonical breadcrumb.

    Lines are formatted `<sha7> <subject>` from `git log --oneline`."""
    try:
        result = subprocess.run(
            [
                "git", "log", "--oneline",
                # Both --grep patterns: git ORs them by default.
                "--grep=adversarial pass",
                "--grep=round-",
                branch,
            ],
            capture_output=True, text=True, check=True,
            cwd=str(CRATE_ROOT),
        )
    except subprocess.CalledProcessError as e:
        print(f"# regen_changelog_section7.py: git log failed "
              f"(rc={e.returncode}): {e.stderr.strip()}", file=sys.stderr)
        return []
    except FileNotFoundError:
        print("# regen_changelog_section7.py: git not found in PATH",
              file=sys.stderr)
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def extract_loop(line: str) -> int | None:
    """Return the integer Loop N mentioned in `line`, or None."""
    m = _LOOP_RE.search(line)
    if not m:
        return None
    return int(m.group(1))


def render_table(commits: list[tuple[str, str, int]]) -> str:
    """Render an ordered Markdown table: SHA | Loop | Summary.

    `commits` is a list of (sha, subject_without_sha, loop_num) in
    git-log order (newest first). We emit oldest-first for readability,
    grouped by Loop for compactness when a single loop produces
    multiple commits.
    """
    by_loop: dict[int, list[tuple[str, str]]] = {}
    for sha, subject, loop in commits:
        by_loop.setdefault(loop, []).append((sha, subject))
    out: list[str] = []
    out.append("# CHANGELOG §7 — generated breadcrumb (regen_changelog_section7.py)")
    out.append("")
    out.append(f"Auto-generated from `git log --grep='adversarial pass' "
               f"--grep='round-'` on branch `{DEFAULT_BRANCH}`.")
    out.append("")
    out.append(f"Total commits matched: **{len(commits)}**; "
               f"distinct loops referenced: **{len(by_loop)}**.")
    out.append("")
    out.append("| Loop | Commits | First-line subjects |")
    out.append("|------|---------|----------------------|")
    for loop in sorted(by_loop):
        rows = by_loop[loop]
        shas = " ".join(f"`{sha}`" for sha, _ in rows)
        # Concatenate subjects; truncate any single subject to 80 chars
        # so the table doesn't blow out reviewer columns.
        subjects = " · ".join(
            (subj[:80] + "…") if len(subj) > 80 else subj
            for _, subj in rows
        )
        # Escape pipe chars in subjects so the table parses.
        subjects = subjects.replace("|", "\\|")
        out.append(f"| {loop} | {shas} | {subjects} |")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    args = parser.parse_args()

    lines = run_git_log(args.branch)
    commits: list[tuple[str, str, int]] = []
    skipped_no_loop = 0
    for line in lines:
        # Format: `<sha7> <subject>` from `--oneline`.
        sha, _, subject = line.partition(" ")
        loop = extract_loop(subject)
        if loop is None:
            skipped_no_loop += 1
            continue
        commits.append((sha, subject, loop))

    if not commits:
        print(f"# regen_changelog_section7.py: 0 matching commits "
              f"({skipped_no_loop} matched grep but had no Loop-N "
              "anchor in subject). Nothing to regenerate.",
              file=sys.stderr)
        return 1

    rendered = render_table(commits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    try:
        rel = args.output.relative_to(CRATE_ROOT) if args.output.is_absolute() else args.output
    except ValueError:
        rel = args.output
    print(f"# regen_changelog_section7.py — wrote {rel}: "
          f"{len(commits)} commit(s) across "
          f"{len(set(c[2] for c in commits))} loop(s) "
          f"(skipped {skipped_no_loop} loop-less subjects)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
