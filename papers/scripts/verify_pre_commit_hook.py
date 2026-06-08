#!/usr/bin/env python3
"""verify_pre_commit_hook.py — detect whether the project's
pre-commit hook invokes verify_committed_state_consistency.py
--staged-only.

Loop 138 C operationalizes the "pre-commit pattern not installed"
class. Without a hook, the operator must remember to run --staged-only
manually before each commit. With a hook, git enforces it.

This is a MANUAL tool, NOT a CI stage. The hook is operator
convenience, not a paper-correctness invariant.

Probes:
  - `.git/hooks/pre-commit` (vanilla git)
  - `.husky/pre-commit` (npm husky)
  - `.git/hooks/pre-commit.sample` (unconfigured default — does NOT count)

Reports:
  - INSTALLED: hook exists, is executable, and references the gate.
  - PRESENT_BUT_NOT_WIRED: hook exists but doesn't reference the gate.
  - ABSENT: no hook installed.

Usage:
  papers/scripts/verify_pre_commit_hook.py
  papers/scripts/verify_pre_commit_hook.py --install
    (write a minimal hook to .git/hooks/pre-commit that invokes the
     gate; refuses to overwrite an existing hook)

Exit 0 if installed; 1 if absent or not-wired.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
GATE_CMD = "papers/scripts/verify_committed_state_consistency.py --staged-only"

HOOK_TEMPLATE = """#!/usr/bin/env bash
# pre-commit hook installed by verify_pre_commit_hook.py --install (Loop 138 C)
# Runs the committed-state staged-only check before every commit.
set -e
exec "$(git rev-parse --show-toplevel)/{cmd}"
""".lstrip()


def git_dir() -> Path:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True, text=True, check=True,
            cwd=str(CRATE_ROOT),
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"# FAIL  git rev-parse failed: {e}", file=sys.stderr)
        return CRATE_ROOT / ".git"
    return (CRATE_ROOT / result.stdout.strip()).resolve()


def hook_status(hook_path: Path) -> tuple[str, str]:
    """Return (status, detail) where status is one of:
    ABSENT, PRESENT_BUT_NOT_WIRED, INSTALLED."""
    if not hook_path.exists():
        return "ABSENT", f"no file at {hook_path}"
    try:
        text = hook_path.read_text()
    except OSError as e:
        return "ABSENT", f"cannot read {hook_path}: {e}"
    is_exec = os.access(hook_path, os.X_OK)
    refs_gate = "verify_committed_state_consistency.py" in text
    if not refs_gate:
        return "PRESENT_BUT_NOT_WIRED", (
            f"file at {hook_path} exists but does not reference "
            "verify_committed_state_consistency.py")
    if not is_exec:
        return "PRESENT_BUT_NOT_WIRED", (
            f"file at {hook_path} references the gate but is not "
            "executable (chmod +x required)")
    return "INSTALLED", f"executable + references gate"


def install_hook(target: Path) -> int:
    if target.exists():
        print(f"# FAIL  --install refused: {target} already exists. "
              "Move or remove it first.", file=sys.stderr)
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(HOOK_TEMPLATE.format(cmd=GATE_CMD))
    target.chmod(0o755)
    print(f"# verify_pre_commit_hook.py --install — wrote {target} "
          "(executable). Future `git commit` invocations will run "
          "the staged-only check automatically.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true",
                        help="Write a minimal pre-commit hook that "
                             "invokes the gate. Refuses to overwrite.")
    args = parser.parse_args()

    gd = git_dir()
    hook_paths = [
        gd / "hooks" / "pre-commit",
        CRATE_ROOT / ".husky" / "pre-commit",
    ]

    if args.install:
        # Default install target: vanilla git hooks dir.
        return install_hook(hook_paths[0])

    statuses: list[tuple[Path, str, str]] = []
    for p in hook_paths:
        status, detail = hook_status(p)
        statuses.append((p, status, detail))

    any_installed = any(s == "INSTALLED" for _, s, _ in statuses)
    for p, s, d in statuses:
        rel = p.relative_to(CRATE_ROOT) if p.is_relative_to(CRATE_ROOT) else p
        print(f"# {s:<22} {rel}: {d}")
    if any_installed:
        print("# verify_pre_commit_hook.py — pre-commit gate "
              "installed; staged-only check runs on every commit")
        return 0
    print("# verify_pre_commit_hook.py — pre-commit gate NOT "
          "installed. Run `papers/scripts/verify_pre_commit_hook.py "
          "--install` to wire it.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
