#!/usr/bin/env python3
"""verify_submission_readiness.py — gate SUBMISSION_CHECKLIST.md §1 against
the actual run_all_checks.sh STAGES array.

Loop 130 C operationalizes the submission-readiness audit finding (Loop
128 B) that the checklist's §1 sub-bullet enumeration drifts silently
when a new stage is added to STAGES but the checklist isn't updated
in lock-step.

Loop 129 C added the §1 "exits 0 with **N/N PASS**" pattern to
`verify_stage_count_consistency.py`, but that gate only checks the
top-line N. The sub-bullet enumeration `(1/21) ... (21/21)` is a
*separate* drift surface: stage added in code, count refreshed at the
top, but the sub-list still ends at 20.

This script:
1. Parses STAGES=( ... ) in `papers/scripts/run_all_checks.sh` to
   discover (actual_n, [stage_names]).
2. Parses §1 sub-bullets of the form `- [ ] (k/M) <name> — <desc>` in
   `papers/SUBMISSION_CHECKLIST.md`.
3. Asserts:
   - len(sub_bullets) == actual_n
   - every M in `(k/M)` equals actual_n
   - k values are exactly [1, 2, ..., actual_n] (monotonic, no gaps)
   - each sub-bullet's short-name roughly matches the STAGES entry
     name at the same index (Jaccard token overlap ≥ 0.3, case-folded)

Usage: papers/scripts/verify_submission_readiness.py

Exit 0 on clean; 1 on any drift.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
RUN_ALL_CHECKS = CRATE_ROOT / "papers" / "scripts" / "run_all_checks.sh"
SUBMISSION_CHECKLIST = CRATE_ROOT / "papers" / "SUBMISSION_CHECKLIST.md"


def parse_stages() -> list[str]:
    """Return ordered list of stage names from STAGES=( ... ) block."""
    text = RUN_ALL_CHECKS.read_text()
    m = re.search(r"STAGES=\(\s*\n(.*?)\n\)", text, re.DOTALL)
    if not m:
        raise RuntimeError("STAGES=( ... ) block not found in run_all_checks.sh")
    body = m.group(1)
    return re.findall(r'^\s+"([^:"]+):', body, re.MULTILINE)


# Match `- [ ] (k/M) <name> — <desc>` and capture (k, M, name, line_no).
# Em-dash is the canonical separator between name and description; we
# require it explicitly. Using `[—-]` here is wrong because it includes
# ASCII hyphen and would terminate names like "cross-reference audit"
# at the first hyphen.
_SUBBULLET_RE = re.compile(
    r"^\s*-\s*\[\s*\]\s*\((\d+)/(\d+)\)\s+(.+?)\s+—\s+",
    re.MULTILINE,
)


def parse_subbullets() -> list[tuple[int, int, str, int]]:
    """Return list of (k, M, name, line_no) for each §1 sub-bullet.

    Constrained to §1 — we extract the section bounded by the §1 heading
    and the next "## " heading so unrelated `(k/M)` patterns elsewhere
    in the file don't bleed in.
    """
    text = SUBMISSION_CHECKLIST.read_text()
    sec = re.search(
        r"^##\s*1\.\s+CI gates\b.*?(?=^##\s|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not sec:
        raise RuntimeError("§1 'CI gates' section not found in SUBMISSION_CHECKLIST.md")
    sec_text = sec.group(0)
    # Offset of §1 within the full file (for accurate line numbers).
    sec_offset = sec.start()
    out: list[tuple[int, int, str, int]] = []
    for m in _SUBBULLET_RE.finditer(sec_text):
        k = int(m.group(1))
        big_m = int(m.group(2))
        name = m.group(3).strip()
        # File-level line number: lines before (sec_offset + m.start()).
        abs_pos = sec_offset + m.start()
        line_no = text[:abs_pos].count("\n") + 1
        out.append((k, big_m, name, line_no))
    return out


def _tokenize(s: str) -> set[str]:
    """Lowercase, strip non-word punctuation, split on whitespace.

    Strip common decorations (#1021, parenthesized hints) so a checklist
    label like "cross-reference audit (companion paper)" tokenizes the
    same way as the STAGES entry "cross-ref audit".
    """
    s = s.lower()
    s = re.sub(r"\(.*?\)", " ", s)        # drop parenthetical hints
    s = re.sub(r"[^a-z0-9#]+", " ", s)    # keep #1021 etc.
    # Loop 130 C: strip very common short-words that add noise without
    # signal so token-overlap doesn't false-pass on stop-word matches.
    stop = {"the", "a", "and", "of", "vs", "to", "on", "for"}
    return {t for t in s.split() if t and t not in stop}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Loop 130 C: per-index name-similarity threshold. Checklist short-names
# are written for humans (e.g., "cross-reference audit") while STAGES
# names are written for grep (e.g., "cross-ref audit"); we want fuzzy
# enough to ignore "ref vs reference" but strict enough to catch a
# whole-stage swap (e.g., checklist says "metadata verify" but STAGES[1]
# is now "preamble per producer").
JACCARD_FLOOR = 0.30


def main() -> int:
    stages = parse_stages()
    subs = parse_subbullets()
    actual_n = len(stages)
    print(f"# verify_submission_readiness.py — STAGES={actual_n}, "
          f"§1 sub-bullets={len(subs)}")

    mismatches: list[str] = []

    # (A) Sub-bullet count must equal stage count.
    if len(subs) != actual_n:
        mismatches.append(
            f"SUBMISSION_CHECKLIST.md §1: sub-bullet count {len(subs)} "
            f"!= STAGES count {actual_n}. Either add the missing "
            f"`- [ ] (k/{actual_n}) ...` line(s) or refresh §1 to "
            f"reflect the current stage list.")

    # (B) Denominator M must match actual_n everywhere.
    for k, big_m, name, line_no in subs:
        if big_m != actual_n:
            mismatches.append(
                f"SUBMISSION_CHECKLIST.md:{line_no}: '({k}/{big_m}) {name}' "
                f"denominator {big_m} != actual {actual_n}. Update to "
                f"'(k/{actual_n})'.")

    # (C) Numerator k must be exactly [1..len(subs)] monotonic.
    for idx, (k, big_m, name, line_no) in enumerate(subs, start=1):
        if k != idx:
            mismatches.append(
                f"SUBMISSION_CHECKLIST.md:{line_no}: sub-bullet position "
                f"{idx} claims numerator {k}; expected {idx} (monotonic "
                f"renumbering required).")

    # (D) Fuzzy name match at each position (only when counts agree;
    # otherwise the per-index pairing is undefined).
    if len(subs) == actual_n:
        for i, (k, big_m, sub_name, line_no) in enumerate(subs):
            stage_name = stages[i]
            score = jaccard(_tokenize(sub_name), _tokenize(stage_name))
            if score < JACCARD_FLOOR:
                mismatches.append(
                    f"SUBMISSION_CHECKLIST.md:{line_no}: §1 stage {k} "
                    f"'{sub_name}' has Jaccard overlap {score:.2f} with "
                    f"STAGES[{i}] '{stage_name}' — below floor "
                    f"{JACCARD_FLOOR}. Either rename one to match or "
                    f"confirm the stages have actually been reordered.")
            else:
                print(f"# OK  ({k}/{actual_n}) '{sub_name}' ~ STAGES[{i}] "
                      f"'{stage_name}' (jaccard={score:.2f})")

    if mismatches:
        print(f"# verify_submission_readiness.py — "
              f"{len(mismatches)} drift(s) between §1 and STAGES",
              file=sys.stderr)
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        return 1
    print(f"# verify_submission_readiness.py — §1 fully aligned with "
          f"{actual_n} STAGES, 0 drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
