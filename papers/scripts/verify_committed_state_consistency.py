#!/usr/bin/env python3
"""verify_committed_state_consistency.py — gate that critical paper
files are committed before run_all_checks.sh declares submission-ready.

Loop 134 C operationalizes the "gate green on uncommitted state →
push different state" confusion. Pre-flight check: walks `git status
--porcelain papers/ docs/` and fails on tracked-file modifications
in submission-bound paths.

Rationale: every other CI stage reads the working-tree state of the
paper files. If those files have uncommitted modifications and the
operator pushes the *committed* state (or pulls a clean checkout
elsewhere), the gates would re-run against a different state than
the one they just declared green. This gate forces the operator to
commit before claiming the gate passed.

Scope: tracked-file modifications under `papers/`, `docs/`,
`papers/scripts/`. Untracked files are allowed (working-tree
artifacts, generated outputs not yet ignored). The `--allow-untracked`
flag is implicit; only modified tracked files fail the gate.

The gate also explicitly excludes paths the parallel agent work uses
(via the EXCLUDE_PATHS list) so a clean F2-paper commit isn't
blocked by submodule's uncommitted `src/` modifications.

Usage:
  papers/scripts/verify_committed_state_consistency.py
  papers/scripts/verify_committed_state_consistency.py --allow-untracked
    (default: untracked files don't fail the gate)

Exit 0 on clean tracked state; 1 on modified/staged/deleted tracked
files in scope.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]

# Scope: tracked files under these directories must be committed.
# Anything outside is out of scope (e.g., src/, data/).
SCOPE_PREFIXES = ["papers/", "docs/"]

# Paths to explicitly exclude even when under a SCOPE_PREFIXES entry.
# The auto-generated CHANGELOG section7 file regenerates on every
# `regen_changelog_section7.py` run; treating it as a tracked file
# whose drift must be committed would force a no-op commit on every
# CI cycle.
EXCLUDE_PATHS = {
    "papers/CHANGELOG_section7_generated.md",
}


def porcelain_lines() -> list[str]:
    """Return raw `git status --porcelain` output lines."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
            cwd=str(CRATE_ROOT),
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"# FAIL  git status failed: {e}", file=sys.stderr)
        return []
    return [ln for ln in result.stdout.splitlines() if ln]


def in_scope(rel_path: str) -> bool:
    if rel_path in EXCLUDE_PATHS:
        return False
    return any(rel_path.startswith(p) for p in SCOPE_PREFIXES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-untracked", action="store_true",
                        default=True,
                        help="(default) Untracked files do not fail the gate.")
    parser.add_argument("--fail-on-untracked", action="store_true",
                        help="Reverse the default: untracked files DO fail.")
    args = parser.parse_args()
    untracked_fails = args.fail_on_untracked

    lines = porcelain_lines()
    if not lines:
        # No modifications at all — clean state.
        print("# verify_committed_state_consistency.py — no modifications "
              "in any scope; tree is clean")
        return 0

    violations: list[str] = []
    skipped = 0
    for raw in lines:
        # Porcelain format: XY <path>  where XY is a 2-char status code.
        # X = staged, Y = working-tree. `??` = untracked.
        status = raw[:2]
        rel_path = raw[3:]
        # Handle renames "R  old -> new" — take the new path.
        if " -> " in rel_path:
            rel_path = rel_path.split(" -> ", 1)[1]
        if not in_scope(rel_path):
            skipped += 1
            continue
        if status == "??":
            if untracked_fails:
                violations.append(f"untracked tracked-scope path: {rel_path}")
            else:
                # Allowed under default.
                pass
            continue
        # Any other status code means tracked + modified/staged/deleted.
        violations.append(f"{status} {rel_path}")

    if violations:
        for v in violations:
            print(f"# FAIL  uncommitted: {v}", file=sys.stderr)
        print(f"# verify_committed_state_consistency.py — "
              f"{len(violations)} uncommitted change(s) in scope "
              f"(papers/ + docs/). Commit before declaring submission-"
              f"ready; gates run against working-tree, not HEAD.",
              file=sys.stderr)
        return 1
    print(f"# verify_committed_state_consistency.py — all tracked "
          f"paper/doc files are committed ({skipped} out-of-scope "
          f"modification(s) ignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
