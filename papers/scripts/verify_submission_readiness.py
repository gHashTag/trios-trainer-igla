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
    try:
        text = RUN_ALL_CHECKS.read_text()
    except FileNotFoundError:
        raise RuntimeError(
            f"run_all_checks.sh missing at {RUN_ALL_CHECKS.relative_to(CRATE_ROOT)} "
            "— gate cannot verify §1 ↔ STAGES alignment without the source "
            "of truth for STAGES. Ensure papers/scripts/ is fully checked out."
        ) from None
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
# Loop 131-A SEV-4 fix #6: accept both `[ ]` (unchecked) and `[x]` /
# `[X]` (checked). Without the broadening, a single checked item would
# silently drop from `subs` and cause a confusing count-mismatch
# diagnostic instead of a dedicated "checked items detected" message.
_SUBBULLET_RE = re.compile(
    r"^\s*-\s*\[\s*([ xX])\s*\]\s*\((\d+)/(\d+)\)\s+(.+?)\s+—\s+",
    re.MULTILINE,
)


def parse_subbullets() -> tuple[list[tuple[int, int, str, int]], list[int]]:
    """Return (sub-bullet list, checked-line-numbers).

    Constrained to §1 — we extract the section bounded by the §1 heading
    and the next "## " heading so unrelated `(k/M)` patterns elsewhere
    in the file don't bleed in.

    Loop 131-A SEV-4 fix #6: also report which sub-bullets are marked
    `[x]` / `[X]` so main() can emit a dedicated diagnostic rather than
    let a single checked item cascade into a count-mismatch message.
    """
    try:
        text = SUBMISSION_CHECKLIST.read_text()
    except FileNotFoundError:
        raise RuntimeError(
            f"SUBMISSION_CHECKLIST.md missing at "
            f"{SUBMISSION_CHECKLIST.relative_to(CRATE_ROOT)} — gate cannot "
            "verify §1 ↔ STAGES alignment. Ensure papers/ is fully "
            "checked out."
        ) from None
    sec = re.search(
        r"^##\s*1\.\s+CI gates\b.*?(?=^##\s|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not sec:
        raise RuntimeError("§1 'CI gates' section not found in SUBMISSION_CHECKLIST.md")
    sec_text = sec.group(0)
    sec_offset = sec.start()
    out: list[tuple[int, int, str, int]] = []
    checked: list[int] = []
    for m in _SUBBULLET_RE.finditer(sec_text):
        marker = m.group(1)
        k = int(m.group(2))
        big_m = int(m.group(3))
        name = m.group(4).strip()
        abs_pos = sec_offset + m.start()
        line_no = text[:abs_pos].count("\n") + 1
        out.append((k, big_m, name, line_no))
        if marker in ("x", "X"):
            checked.append(line_no)
    return out, checked


def _tokenize(s: str) -> set[str]:
    """Lowercase, strip non-word punctuation, split on whitespace.

    Strip common decorations (#1021, parenthesized hints) so a checklist
    label like "cross-reference audit (companion paper)" tokenizes the
    same way as the STAGES entry "cross-ref audit".

    Loop 131-A SEV-4 fix #4: stop-word set extended symmetrically.
    Criterion: ≤3 chars AND grammatical-glue role (preposition,
    conjunction, copula). Was previously asymmetric (had "of" but
    missed "in", "by", "with", "at", "as", "or", "per") — a future
    rename like "preamble per producer" → "preamble in producer" would
    silently pass while semantically distinct.
    """
    s = s.lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9#]+", " ", s)
    stop = {
        "the", "a", "an", "and", "of", "or", "vs",
        "to", "on", "in", "at", "by", "as", "per", "with", "for", "is",
    }
    return {t for t in s.split() if t and t not in stop}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Loop 130 C: per-index absolute floor for token-overlap. Catches
# whole-stage swap (e.g., §1 says "metadata verify" but STAGES[1] is now
# "preamble per producer" → Jaccard 0.0).
JACCARD_FLOOR = 0.30


def _name_match_score(sub_tokens: set[str], stage_tokens: set[str]
                      ) -> float:
    """Wrap jaccard so it's the only place we declare the metric."""
    return jaccard(sub_tokens, stage_tokens)


def main() -> int:
    stages = parse_stages()
    subs, checked = parse_subbullets()
    actual_n = len(stages)
    print(f"# verify_submission_readiness.py — STAGES={actual_n}, "
          f"§1 sub-bullets={len(subs)} ({len(checked)} marked [x])")

    mismatches: list[str] = []

    # Loop 132-A SEV-3 fix #3: precheck STAGES for duplicate entries.
    # The nearest-neighbor check uses strict `<` against best_other,
    # which ties admit silently when STAGES contains a literal
    # duplicate (a copy-paste mistake). Catch the underlying issue
    # before the NN loop runs.
    seen: dict[str, int] = {}
    for i, name in enumerate(stages):
        if name in seen:
            mismatches.append(
                f"papers/scripts/run_all_checks.sh STAGES has duplicate "
                f"entry '{name}' at positions {seen[name]} and {i}. "
                f"Each stage name must be unique; otherwise the §1 "
                f"nearest-neighbor match is ambiguous.")
        else:
            seen[name] = i

    # Loop 131-A SEV-4 fix #6: dedicated diagnostic when sub-bullets
    # are marked `[x]`. The readiness audit assumes all-or-none
    # unchecked state (because a partially-checked checklist is a
    # mid-submission artifact that shouldn't pass the gate).
    if checked:
        mismatches.append(
            f"SUBMISSION_CHECKLIST.md §1: {len(checked)} sub-bullet(s) "
            f"marked [x] at line(s) {checked} — readiness audit "
            f"assumes all-or-none unchecked state. If submission is "
            f"in progress, this gate should be excluded from the CI run; "
            f"if accidental, replace [x] → [ ].")

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

    # (D) Name match at each position. Loop 131-A SEV-3 fix #1:
    # the Jaccard floor alone admits adjacent-swap silent-pass when two
    # stage names happen to share one token (e.g., "tables vs CSVs" ↔
    # "formulas vs tables" both have "tables" → Jaccard 1/3 = 0.333,
    # above the 0.30 floor). To catch this class, we additionally
    # require **nearest-neighbor**: the score for the §1 sub-bullet at
    # position i vs STAGES[i] must be ≥ max score vs STAGES[j] for any
    # j ≠ i. After a swap of stages i↔i+1, the §1 sub-bullet at i will
    # match STAGES[i+1] strictly better than STAGES[i], firing the
    # nearest-neighbor check.
    if len(subs) == actual_n:
        stage_tokens = [_tokenize(s) for s in stages]
        for i, (k, big_m, sub_name, line_no) in enumerate(subs):
            sub_tok = _tokenize(sub_name)
            stage_name = stages[i]
            score = _name_match_score(sub_tok, stage_tokens[i])
            # Find the maximum score against any OTHER stage.
            other_scores = [
                (j, _name_match_score(sub_tok, stage_tokens[j]))
                for j in range(actual_n) if j != i
            ]
            best_other_idx, best_other_score = max(
                other_scores, key=lambda p: p[1]
            ) if other_scores else (-1, 0.0)
            if score < JACCARD_FLOOR:
                mismatches.append(
                    f"SUBMISSION_CHECKLIST.md:{line_no}: §1 stage {k} "
                    f"'{sub_name}' has Jaccard overlap {score:.2f} with "
                    f"STAGES[{i}] '{stage_name}' — below floor "
                    f"{JACCARD_FLOOR}. Either rename one to match or "
                    f"confirm the stages have actually been reordered.")
            elif score < best_other_score:
                # Nearest-neighbor violation: §1 sub-bullet at position i
                # is MORE similar to a different STAGES entry than to
                # STAGES[i]. Most-likely cause: STAGES were reordered
                # without updating §1.
                mismatches.append(
                    f"SUBMISSION_CHECKLIST.md:{line_no}: §1 stage {k} "
                    f"'{sub_name}' matches STAGES[{i}] '{stage_name}' "
                    f"with Jaccard {score:.2f}, but matches STAGES["
                    f"{best_other_idx}] '{stages[best_other_idx]}' with "
                    f"{best_other_score:.2f} (higher). Reorder the §1 "
                    f"sub-bullets to track the current STAGES order.")
            else:
                print(f"# OK  ({k}/{actual_n}) '{sub_name}' ~ STAGES[{i}] "
                      f"'{stage_name}' (jaccard={score:.2f}, "
                      f"best_other={best_other_score:.2f})")

    # Loop 132-A SEV-3 fix #14: gate the §1 (13/M) "N claims (X EXACT +
    # Y SCOPED + Z ACKN + W RELATIONAL)" enumeration so the leading N
    # matches the sum X+Y+Z+W. Drift example: adding a 5th RELATIONAL
    # invariant (W: 4→5) without bumping the leading "11 claims" → 12.
    try:
        text = SUBMISSION_CHECKLIST.read_text()
    except FileNotFoundError:
        text = ""
    if text:
        enum_re = re.compile(
            r"(\d+) claims \((\d+) EXACT \+ (\d+) SCOPED "
            r"\+ (\d+) ACKN \+ (\d+) RELATIONAL\)"
        )
        m = enum_re.search(text)
        if m:
            leading = int(m.group(1))
            parts = [int(m.group(j)) for j in range(2, 6)]
            total = sum(parts)
            line_no = text[:m.start()].count("\n") + 1
            if leading != total:
                mismatches.append(
                    f"SUBMISSION_CHECKLIST.md:{line_no}: §1 enumeration "
                    f"'{leading} claims ({'+'.join(str(p) for p in parts)})' "
                    f"— leading {leading} != sum {total}. Update the "
                    f"leading count to match the partition.")
            else:
                print(f"# OK  §1 claim-class enumeration: {leading} = "
                      f"{'+'.join(str(p) for p in parts)}")

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
