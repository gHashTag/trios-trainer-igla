#!/usr/bin/env python3
"""verify_tier_classification.py — gate the STAGE_TIERS array parity
+ tier-name validity in run_all_checks.sh.

Loop 142 A.iv: the 64th-pass #4 fix added a runtime WARN on
STAGES/STAGE_TIERS length mismatch but kept the gate running. This
stage promotes the parity check to a hard FAIL — any drift fires
the gate, so a new STAGES entry without a paired tier label is
caught at submission-readiness time rather than discovered at
deploy.

Asserts:
  (a) len(STAGE_TIERS) == len(STAGES).
  (b) Every tier ∈ {"submission", "discipline"}.
  (c) Tier contiguity: all "submission" entries precede any
      "discipline" entry. The summary's per-tier count is only
      meaningful when the partition is contiguous; a reordering
      that interleaves the two would still report a numeric
      total but obscure the actual gating boundary. (Loop 143 B
      addition — closes 65th-pass SEV-3 #8.)
  (d) Reports per-tier count for visibility (submission, discipline).

Usage: papers/scripts/verify_tier_classification.py

Exit 0 if all three invariants hold; 1 on any violation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
RUN_ALL_CHECKS = CRATE_ROOT / "papers" / "scripts" / "run_all_checks.sh"

ALLOWED_TIERS = {"submission", "discipline"}


def parse_stages_and_tiers() -> tuple[int, list[str]] | str:
    """Return (stages_count, tier_labels_list) or error string."""
    if not RUN_ALL_CHECKS.exists():
        return f"missing {RUN_ALL_CHECKS.relative_to(CRATE_ROOT)}"
    text = RUN_ALL_CHECKS.read_text()

    # Parse STAGES count.
    m_stages = re.search(r"STAGES=\(\s*\n(.*?)\n\)", text, re.DOTALL)
    if not m_stages:
        return "STAGES=( ... ) block not found"
    stages_body = m_stages.group(1)
    n_stages = len(re.findall(r'^\s+"', stages_body, re.MULTILINE))

    # Parse STAGE_TIERS list.
    m_tiers = re.search(r"STAGE_TIERS=\(\s*\n(.*?)\n\)", text, re.DOTALL)
    if not m_tiers:
        return "STAGE_TIERS=( ... ) block not found"
    tiers_body = m_tiers.group(1)
    tiers = re.findall(r'"([^"]+)"', tiers_body)

    return n_stages, tiers


def main() -> int:
    parsed = parse_stages_and_tiers()
    if isinstance(parsed, str):
        print(f"# FAIL  parse  {parsed}", file=sys.stderr)
        return 1
    n_stages, tiers = parsed
    print(f"# verify_tier_classification.py — STAGES={n_stages}, "
          f"STAGE_TIERS={len(tiers)}")

    mismatches: list[str] = []

    # (a) Parity.
    if len(tiers) != n_stages:
        mismatches.append(
            f"STAGE_TIERS length {len(tiers)} != STAGES length "
            f"{n_stages}. Add a tier label for every stage; the "
            "runtime WARN at run_all_checks.sh:151 is now a FAIL "
            "via this gate.")

    # (b) Allowed tier names.
    for i, tier in enumerate(tiers):
        if tier not in ALLOWED_TIERS:
            mismatches.append(
                f"STAGE_TIERS[{i}] = {tier!r} not in allowed set "
                f"{sorted(ALLOWED_TIERS)}.")

    # (c) Tier contiguity. Loop 143 B (65th-pass SEV-3 #8 closure):
    # all "submission" entries must precede any "discipline" entry.
    # The summary's per-tier counter is only meaningful when the
    # partition is contiguous.
    last_submission_idx = -1
    first_discipline_idx = len(tiers)
    for i, tier in enumerate(tiers):
        if tier == "submission":
            last_submission_idx = i
        elif tier == "discipline" and i < first_discipline_idx:
            first_discipline_idx = i
    if (last_submission_idx >= 0 and first_discipline_idx < len(tiers)
            and last_submission_idx > first_discipline_idx):
        mismatches.append(
            f"tier contiguity violated: last submission entry at "
            f"index {last_submission_idx} > first discipline entry "
            f"at index {first_discipline_idx}. The partition must "
            "be contiguous (all submission entries before any "
            "discipline entry).")
    else:
        print(f"# OK    contiguity: submission ends at "
              f"{last_submission_idx}, discipline starts at "
              f"{first_discipline_idx}")

    # (d) Per-tier count reporting.
    if not mismatches:
        from collections import Counter
        counts = Counter(tiers)
        for tier in sorted(counts):
            print(f"# OK    tier '{tier}': {counts[tier]} stage(s)")

    if mismatches:
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(f"# verify_tier_classification.py — "
              f"{len(mismatches)} tier-classification violation(s)",
              file=sys.stderr)
        return 1
    print(f"# verify_tier_classification.py — STAGES/STAGE_TIERS "
          f"parity holds; {len(tiers)} entries tier-labeled correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
