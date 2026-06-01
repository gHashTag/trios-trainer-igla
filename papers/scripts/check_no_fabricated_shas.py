#!/usr/bin/env python3
"""check_no_fabricated_shas.py — confirm every git SHA mentioned in
paper assets resolves to a real commit reachable from the current
repository.

Loop 77 caught a fabricated `b6f5c4` SHA in a README. This script
prevents the class by scanning every Markdown file under `papers/`,
`docs/`, and `data/` for 6–40 hex-character strings that look like
git SHAs, then runs `git cat-file -e <sha>` on each. Strings inside
verbatim code blocks (between ``` fences) are skipped — code
examples may include placeholder SHAs intentionally.

Exit 0 if every SHA-looking string resolves. Exit 1 with details on
stderr otherwise.

Usage: papers/scripts/check_no_fabricated_shas.py

Wired into papers/scripts/run_all_checks.sh as stage 7.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]

# Paths to scan. Anything outside this list is ignored.
SCAN_DIRS = [
    CRATE_ROOT / "papers",
    CRATE_ROOT / "docs",
    CRATE_ROOT / "data",
]
SCAN_EXTS = {".md", ".tex", ".sh", ".py"}
# Paths to exclude from scanning. These are non-F2 work (parallel
# branch experiments, third-party content) where SHAs may be from
# other repositories and not verifiable here.
EXCLUDE_PATHS = [
    CRATE_ROOT / "docs" / "preregistration",  # parallel-branch pre-regs
]

# Git SHAs are either short (7-12 hex chars) or full (40 hex chars).
# Excluded explicitly: 32-char hex (always an MD5 checksum in our
# manifests), 16-char (often a partial hash), 64-char (SHA-256).
# Bounded by word boundaries so we don't pick up parts of larger
# identifiers.
SHA_RE = re.compile(r"\b([0-9a-f]{7,12}|[0-9a-f]{40})\b")

# Allowed false positives: hex constants we know aren't SHAs. Add
# more here if the scanner false-flags something. Each must be the
# exact lowercase string the regex captures.
ALLOWLIST = {
    # Example: docs/F2_BINARIES.md examples include hash-of-config
    # values like "0xdead" — not matched by SHA_RE (starts with 0x),
    # but listed here for documentation.
}


def is_in_code_fence(text: str, position: int) -> bool:
    """True if `position` falls inside a ``` fenced code block."""
    fence_count = text.count("```", 0, position)
    return fence_count % 2 == 1


def collect_shas() -> dict[str, list[tuple[Path, int]]]:
    """Return {sha → [(file, line_number), ...]} for every SHA-like
    string in the scan tree, outside of code fences."""
    found: dict[str, list[tuple[Path, int]]] = {}
    for root in SCAN_DIRS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix not in SCAN_EXTS:
                continue
            if not path.is_file():
                continue
            if any(excl in path.parents or excl == path for excl in EXCLUDE_PATHS):
                continue
            try:
                text = path.read_text(errors="replace")
            except (OSError, UnicodeDecodeError):
                continue
            for m in SHA_RE.finditer(text):
                sha = m.group(1)
                if sha in ALLOWLIST:
                    continue
                # Skip if inside a fenced code block.
                if is_in_code_fence(text, m.start()):
                    continue
                # Skip strings that are too short to plausibly be a
                # SHA in *this* repo's history — but the regex is
                # already at 7+. Keep length filter for completeness.
                if len(sha) < 7:
                    continue
                # Filter: SHA tokens must contain at least one digit
                # and one letter to reduce false positives on
                # all-hex words. (e.g. "deadbeef" is allowed because
                # it has letters and digits, but a hypothetical word
                # like "abcdefa" — all letters — wouldn't pass.)
                if not (any(c.isdigit() for c in sha) and
                        any(c.isalpha() for c in sha)):
                    continue
                line_num = text[: m.start()].count("\n") + 1
                found.setdefault(sha, []).append((path, line_num))
    return found


def sha_exists(sha: str) -> bool:
    """Run `git cat-file -e <sha>` to check existence in any branch."""
    try:
        subprocess.run(
            ["git", "cat-file", "-e", sha],
            cwd=CRATE_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    candidates = collect_shas()
    fabricated: list[str] = []
    real = 0
    for sha, locations in sorted(candidates.items()):
        if sha_exists(sha):
            real += 1
        else:
            fabricated.append(sha)
            for path, line in locations:
                rel = path.relative_to(CRATE_ROOT)
                print(
                    f"  FAIL  fabricated SHA `{sha}` at "
                    f"{rel}:{line}",
                    file=sys.stderr,
                )

    print(
        f"# check_no_fabricated_shas.py — "
        f"{real} verified, {len(fabricated)} fabricated"
    )
    return 0 if not fabricated else 1


if __name__ == "__main__":
    sys.exit(main())
