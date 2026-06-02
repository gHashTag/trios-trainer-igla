#!/usr/bin/env python3
"""verify_generator_consistency.py — bind regen_changelog_section7.py
generator output to the hand-maintained CHANGELOG §7 prose.

Loop 133 C operationalizes the "parallel generator vs hand-maintained
drift" class. The Loop 132 B generator emits a per-loop table at
`papers/CHANGELOG_section7_generated.md`; the CHANGELOG §7 lead
paragraph carries a breadcrumb floor "≥N commits across Loops 90–M".
This gate enforces:
  (a) generator's distinct-loop count is ≥ the §7 lead breadcrumb floor
  (the §7 says "≥35 commits"; generator must show ≥35 actual matches);
  (b) generator's terminal Loop M agrees with §7 lead's terminal loop
  (the gate already enforced by `verify_changelog_consistency.py`,
  re-checked here as a defense-in-depth).

Approach:
  1. Run `regen_changelog_section7.py --output <tmp>` to produce a
     fresh table from current git state.
  2. Parse the generated header for "Total commits matched: **N**;
     distinct loops referenced: **M**" and the per-row Loop column for
     the maximum Loop N seen.
  3. Parse `papers/CHANGELOG.md` §7 lead for "≥(\\d+) commits across
     Loops 90–(\\d+)" — the floor and terminal cursor.
  4. Assert generator_commits >= breadcrumb_floor AND generator_last_loop
     == breadcrumb_last_loop (or generator_last_loop >= breadcrumb_last
     if §7 lead intentionally lags by a loop).

Usage: papers/scripts/verify_generator_consistency.py

Exit 0 on agreement; 1 on drift or generator failure.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = CRATE_ROOT / "papers" / "scripts" / "regen_changelog_section7.py"
CHANGELOG = CRATE_ROOT / "papers" / "CHANGELOG.md"


def run_generator(out_path: Path) -> tuple[int, int, int] | str:
    """Return (total_commits, distinct_loops, max_loop) or error string."""
    if not GENERATOR.exists():
        return f"generator missing at {GENERATOR.relative_to(CRATE_ROOT)}"
    try:
        result = subprocess.run(
            [sys.executable, str(GENERATOR), "--output", str(out_path)],
            capture_output=True, text=True, check=True, timeout=30,
        )
    except subprocess.CalledProcessError as e:
        return (f"generator failed (rc={e.returncode}): "
                f"{(e.stderr or '').strip()[:200]}")
    except FileNotFoundError:
        return "python3 not found"
    if not out_path.exists():
        return f"generator exited 0 but no output at {out_path}"
    body = out_path.read_text()
    m_total = re.search(
        r"Total commits matched: \*\*(\d+)\*\*; distinct loops referenced: "
        r"\*\*(\d+)\*\*",
        body,
    )
    if not m_total:
        return ("generator output missing the 'Total commits matched ... "
                "distinct loops referenced' header line; expected the "
                "Loop 132 B format")
    total_commits = int(m_total.group(1))
    distinct_loops = int(m_total.group(2))
    loop_numbers = [int(n) for n in re.findall(r"^\| (\d+) \|", body,
                                                re.MULTILINE)]
    if not loop_numbers:
        return "generator output had no per-loop rows"
    return total_commits, distinct_loops, max(loop_numbers)


def parse_section7_breadcrumb() -> tuple[int, int, int] | str:
    """Parse §7 lead for '≥(\\d+) commits across Loops 90–(\\d+)'.

    Returns (commit_floor, range_start, range_end) or error string."""
    if not CHANGELOG.exists():
        return f"CHANGELOG missing at {CHANGELOG.relative_to(CRATE_ROOT)}"
    text = CHANGELOG.read_text()
    m = re.search(
        r"surfaces\s*≥(\d+)\s+commits\s+across\s+Loops\s+(\d+)[–-](\d+)",
        text,
    )
    if not m:
        return ("CHANGELOG §7 lead missing '≥N commits across Loops X-Y' "
                "breadcrumb; pattern may have drifted")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def main() -> int:
    with tempfile.NamedTemporaryFile(
        prefix="generated_section7_", suffix=".md",
        mode="w", delete=False,
    ) as f:
        tmp = Path(f.name)
    try:
        gen = run_generator(tmp)
        if isinstance(gen, str):
            print(f"# FAIL  generator: {gen}", file=sys.stderr)
            return 1
        total_commits, distinct_loops, max_loop = gen
        print(f"# OK    generator: {total_commits} commits, "
              f"{distinct_loops} distinct loops, max Loop {max_loop}")

        crumb = parse_section7_breadcrumb()
        if isinstance(crumb, str):
            print(f"# FAIL  breadcrumb: {crumb}", file=sys.stderr)
            return 1
        commit_floor, range_start, range_end = crumb
        print(f"# OK    breadcrumb: ≥{commit_floor} commits across "
              f"Loops {range_start}-{range_end}")

        errors: list[str] = []
        if total_commits < commit_floor:
            errors.append(
                f"generator surfaced {total_commits} commits but §7 "
                f"breadcrumb claims ≥{commit_floor}; either the floor "
                "is stale or commits matching the grep have been lost.")
        if max_loop < range_end - 1:
            # Loop 133 C refinement: tolerate 1-loop lag because the
            # in-flight current loop's commit hasn't landed yet at the
            # moment this gate runs during the commit-prep cycle. A
            # gap > 1 indicates real §7-ahead-of-history drift.
            errors.append(
                f"generator's max Loop ({max_loop}) is more than one "
                f"loop BELOW §7's breadcrumb range terminal "
                f"({range_end}); the §7 lead is ahead of the commit "
                "history by 2+ loops, which suggests the §7 was "
                "bumped without committing the corresponding loop.")
        elif max_loop == range_end - 1:
            print(f"# INFO  generator max Loop {max_loop} = breadcrumb "
                  f"terminal {range_end} − 1 — 1-loop in-flight lag "
                  "(expected during Loop N pre-commit cycle).")
        if max_loop > range_end:
            # This is the legitimate case where the lead is intentionally
            # pinned to the documented cumulative loop; generator picks
            # up later commits. Print as INFO not FAIL.
            print(f"# INFO  generator max Loop {max_loop} > breadcrumb "
                  f"terminal {range_end} — generator includes commits "
                  "after the §7 lead's last documented loop.")

        if errors:
            for e in errors:
                print(f"  {e}", file=sys.stderr)
            print(f"# verify_generator_consistency.py — "
                  f"{len(errors)} drift(s) between generator and §7",
                  file=sys.stderr)
            return 1
        print("# verify_generator_consistency.py — generator and §7 "
              "agree on commit floor + range terminal")
        return 0
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
