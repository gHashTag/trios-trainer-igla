#!/usr/bin/env python3
"""verify_src_unchanged_during_paper_loop.py — detect src/ creep
during paper-only loops.

Loop 137 C operationalizes the 59th-pass #16 SEV-4 deferred catch.
Paper-only loops should not modify `src/` files; if they do, the
operator probably wants to split the commit (paper work and code
work in separate commits) for clearer history and review.

This is a MANUAL pre-flight tool, NOT a CI stage. Parallel agents
routinely modify `src/` during paper loops; wiring this as a CI
stage would fire on every dev cycle.

Approach:
  Walks `git status --porcelain src/` and reports any tracked-file
  modifications. Optionally reports line-counts via `git diff
  --stat src/` to surface scale (large modifications more likely
  warrant a split commit).

Usage:
  papers/scripts/verify_src_unchanged_during_paper_loop.py
  papers/scripts/verify_src_unchanged_during_paper_loop.py --stat
    (also show per-file line-count via `git diff --stat`)

Exit 0 if no src/ modifications; 1 if any (with breakdown on stderr).
Always informational — never wired as a CI stage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]


def src_modifications() -> list[str]:
    """Return list of 'XY path' porcelain lines under src/."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "src/"],
            capture_output=True, text=True, check=True,
            cwd=str(CRATE_ROOT),
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"# FAIL  git status failed: {e}", file=sys.stderr)
        return []
    return [ln for ln in result.stdout.splitlines() if ln.strip()]


def diff_stat() -> str:
    """Return `git diff --stat src/` output as a string."""
    try:
        result = subprocess.run(
            ["git", "diff", "--stat", "src/"],
            capture_output=True, text=True, check=True,
            cwd=str(CRATE_ROOT),
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stat", action="store_true",
        help="Also print `git diff --stat src/` for line-count detail.",
    )
    args = parser.parse_args()

    mods = src_modifications()
    if not mods:
        print("# verify_src_unchanged_during_paper_loop.py — src/ is "
              "clean; paper-only loop discipline intact")
        return 0

    print(f"# verify_src_unchanged_during_paper_loop.py — "
          f"{len(mods)} src/ modification(s) detected:", file=sys.stderr)
    tracked_mods = 0
    untracked = 0
    for line in mods:
        status = line[:2]
        path = line[3:]
        print(f"  {status} {path}", file=sys.stderr)
        if status == "??":
            untracked += 1
        else:
            tracked_mods += 1

    if args.stat:
        stat = diff_stat()
        if stat:
            print(file=sys.stderr)
            print("# git diff --stat src/:", file=sys.stderr)
            for line in stat.splitlines():
                print(f"  {line}", file=sys.stderr)

    print(
        f"# verify_src_unchanged_during_paper_loop.py — "
        f"{tracked_mods} tracked modification(s), {untracked} "
        f"untracked under src/. If this is a paper-only loop, "
        "consider `git stash push -- src/` before committing OR "
        "split the commit so src/ work lives in its own change.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
