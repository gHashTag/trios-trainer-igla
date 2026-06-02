#!/usr/bin/env python3
"""verify_label_consistency.py — deprecated-term gate.

Operationalizes the Loop 107 30th-adversarial-pass class: when a
substantive methodological change happens (e.g., Loop 105 drop of
four-PSE machinery for the #1021 paper), labels referring to the
dropped concept can survive in distant sections (§1.3 promise,
§4.4 framing, §1 abstract) without being caught by the section-
focused rounds 27-29. The 30th pass found two such Pearl-CDE label
residues; this gate prevents the next equivalent regression.

Each entry in DEPRECATIONS specifies:
- paper: relative path
- term: a regex pattern that must NOT match in the body
- reason: the prior decision that deprecated this term
- allowed_contexts: regex patterns whose lines are acceptable
  (e.g., disavowals like "this paper does **not** use X")

The gate fails if a deprecated term appears in a line that doesn't
match any allowed context.

Usage: papers/scripts/verify_label_consistency.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]


# Each tuple: (paper_relpath, deprecated_pattern, reason, allowed_context_patterns)
DEPRECATIONS: list[tuple[str, str, str, list[str]]] = [
    (
        "papers/phi_ladder_paper_intro_draft.md",
        r"\bPearl[- ]CDE\b",
        "Loop 105 §3.4 dropped four-PSE for the format comparison; "
        "this paper now uses stratified total-effect contrast, not Pearl CDE.",
        [
            # Disavowal phrasings.
            r"\bNOT Pearl[- ]CDE\b",
            r"\bnot Pearl[- ]CDE\b",
            r"\bnot a Pearl[- ]CDE\b",
            r"\bnot the Pearl[- ]CDE\b",
            # Companion-paper attribution (CDE is the F2 framework).
            r"\bcompanion(?:[^\n]*)Pearl[- ]CDE",
            r"\bcompanion paper(?:[^\n]*)Pearl[- ]CDE",
            r"\bF2(?:[^\n]*)deploys CDE",
            r"\bF2 companion(?:[^\n]*)Pearl[- ]CDE",
            r"\bF2 sandbox-scale finding(?:[^\n]*)Pearl[- ]CDE",
            # Contrast prose explicitly distinguishing the two estimands.
            r"\bdiffers:\s*F2 deploys",
            r"\bPearl[- ]CDE(?:[^\n]*)in the companion",
            r"\bPearl[- ]CDE in the companion's identification",
            # Methodological contrast: "a Pearl CDE would partition"
            # used to characterize per-PSE estimands in §3.5 Λ paragraph.
            r"\ba Pearl[- ]CDE would partition",
            r"\ba Pearl[- ]CDE bound",
            r"\bper-PSE NDE bound",
        ],
    ),
    (
        "papers/phi_ladder_paper_intro_draft.md",
        r"\bfour-PSE\s+decomposition\b",
        "Loop 105 §3.4 dropped four-PSE for the format comparison.",
        [
            r"\bnot deploy(?:[^\n]*)four-PSE",
            r"\bdo not\s+pre-register a four-PSE",
            r"\bdoes\s*\*\*not\*\*\s*deploy",
            r"\bdoes not deploy",
            r"\bdrop[\w ]*four-PSE",
            r"\bafter dropping the four-PSE",
            r"\b#### Why four-PSE",
            r"\bWhy four-PSE is dropped",
            r"\bfour-PSE machinery is\s+\*\*retained for future",
            r"\bthe four-PSE",
            r"\(3\)(?:[^\n]*)four-PSE",
            # §3.4 disavowal paragraph: "does not apply" + "remains
            # violated" are both rejection phrasings.
            r"four-PSE decomposition of\s*`f2_dual_mediation`\s*\*\*does not apply",
            r"\bremains violated\.\s*The four-PSE decomposition under these",
        ],
    ),
    (
        "papers/phi_ladder_paper_intro_draft.md",
        r"\bReporting will be\s*\*\*symmetric\*\*",
        "Loop 105 §4.4 introduced asymmetric primary/secondary; the §4 "
        "intro 'symmetric reporting' claim was inconsistent.",
        [],   # No allowed context — phrase should be gone.
    ),
]


def is_allowed(line: str, patterns: list[str]) -> bool:
    return any(re.search(p, line) for p in patterns)


def check_paper(rel: str, dep_pat: str, reason: str,
                allowed: list[str]) -> list[str]:
    path = CRATE_ROOT / rel
    if not path.exists():
        return [f"{rel}: file missing"]
    text = path.read_text()
    dep_re = re.compile(dep_pat)
    mismatches: list[str] = []
    for i, line in enumerate(text.splitlines(), 1):
        if not dep_re.search(line):
            continue
        if is_allowed(line, allowed):
            continue
        # Strip trailing whitespace, truncate for readability.
        excerpt = line.strip()
        if len(excerpt) > 100:
            excerpt = excerpt[:97] + "..."
        mismatches.append(f"{rel}:{i}: {dep_pat!r} in {excerpt!r}")
    return mismatches


def main() -> int:
    total_failures = 0
    total_rules = 0
    for paper, pat, reason, allowed in DEPRECATIONS:
        total_rules += 1
        failures = check_paper(paper, pat, reason, allowed)
        if failures:
            total_failures += len(failures)
            print(f"# FAIL  {paper} :: {pat}", file=sys.stderr)
            print(f"#       reason: {reason}", file=sys.stderr)
            for f in failures:
                print(f"  {f}", file=sys.stderr)
        else:
            print(f"# OK    {paper} :: {pat}")
    if total_failures:
        print(f"# verify_label_consistency.py — {total_failures} "
              f"deprecated-label leak(s) across {total_rules} rule(s)",
              file=sys.stderr)
        return 1
    print(f"# verify_label_consistency.py — "
          f"{total_rules} rules checked, 0 leaks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
