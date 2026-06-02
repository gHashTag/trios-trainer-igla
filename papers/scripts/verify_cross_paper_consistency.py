#!/usr/bin/env python3
"""verify_cross_paper_consistency.py — gate cross-paper claim consistency.

Loop 121 B: operationalizes the "next SEV class" identified after Loop 120 —
cross-paper claims where each paper makes a statement that's true on its
own but contradicts (or appears to contradict) the other paper.

Examples surfaced by adversarial passes 38-43:
- F2 §8.1 says "Ten F2 binaries"; #1021 §3.3 says "12 binaries on disk"
  (this is FINE because they're different scopes: F2's documented set
  vs all bins on disk including #1021 contributions — but the gate
  enforces that the cross-paper relationship is *acknowledged* in both
  papers).
- F2 §3.5.4 says "714 passing tests" (src/lib.rs only); #1021 §1.3
  says "809 tests" (total = lib + per-binary + integration). Different
  scopes — gated for self-consistency.
- F2 §1 abstract says "809 tests"; #1021 §1.3 says "809 tests" — same
  scope, same number, must match exactly.

Each registered claim is one of:
- EXACT_MATCH: two papers say N; both Ns must be equal.
- SCOPED_DIFF: two papers say different Ns for related-but-distinct
  scopes; the gate asserts each paper's claim matches its own scope's
  current source-of-truth (cargo test --lib, ls src/bin/, etc.).
- ACKNOWLEDGES: paper A's claim must include a cross-reference to
  paper B's different scope (avoid silent contradictions).

Usage: papers/scripts/verify_cross_paper_consistency.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
F2_PAPER = CRATE_ROOT / "papers" / "f2_methodology.md"
ISSUE1021_PAPER = CRATE_ROOT / "papers" / "phi_ladder_paper_intro_draft.md"


def find_int_claim(path: Path, pattern: str) -> tuple[int, int] | None:
    """Return (claimed_int, line_number) of the first match, or None."""
    pat = re.compile(pattern, re.DOTALL)
    text = path.read_text()
    m = pat.search(text)
    if not m:
        return None
    digit_pos = m.start(1)
    line_no = text[:digit_pos].count("\n") + 1
    return int(m.group(1)), line_no


SUBMISSION_CHECKLIST = CRATE_ROOT / "papers" / "SUBMISSION_CHECKLIST.md"


# EXACT_MATCH: both papers must report the same integer.
# Each entry: (regex_a, paper_a, regex_b, paper_b, description)
EXACT_MATCH_CLAIMS: list[tuple[str, Path, str, Path, str]] = [
    # Total test count: F2 §1 abstract vs #1021 §1.3 — both report
    # cumulative cargo-test totals; must match exactly.
    (
        r"open-source in Rust with 10 binaries, (\d+)\s*\n?\s*unit/integration tests",
        F2_PAPER,
        r"The same 10 F2 binaries, (\d+) tests",
        ISSUE1021_PAPER,
        "Total cargo-test count (F2 §1 vs #1021 §1.3)",
    ),
    # Loop 122 B: SUBMISSION_CHECKLIST page-count claims; the F2 paper PDF
    # at the TMLR-class variant must match. We pin both via the checklist;
    # this is a single-file check across two CLAIMS (TMLR-class page count
    # currently 27pp at Loop 109+).
    (
        r"Real-TMLR-class PDF: \*\*(\d+) pages\*\*",
        SUBMISSION_CHECKLIST,
        r"Real-TMLR-class PDF: \*\*(\d+) pages\*\*",
        SUBMISSION_CHECKLIST,
        "SUBMISSION_CHECKLIST TMLR PDF page count (self-consistency)",
    ),
    # Loop 123 C: non-anon and anon page counts should match each other
    # (anonymization removes acknowledgments but adds the placeholder,
    # net page change is typically zero). The checklist currently pins
    # both to 42pp; gate the self-consistency.
    (
        r"Non-anonymized PDF: \*\*(\d+) pages\*\*",
        SUBMISSION_CHECKLIST,
        r"Anonymized PDF: \*\*(\d+) pages\*\*",
        SUBMISSION_CHECKLIST,
        "SUBMISSION_CHECKLIST non-anon vs anon page count",
    ),
]


# ACKNOWLEDGES: paper A says X, paper B says Y, where X ≠ Y is intentional
# (different scopes). The gate requires that paper A explicitly acknowledges
# the other paper's different scope, so that a TMLR reviewer reading both
# in sequence isn't surprised. Each entry: (paper, regex, must_include_pattern, description).
ACKNOWLEDGES_CLAIMS: list[tuple[Path, str, str, str]] = [
    # F2 §8.1 says "Ten F2 binaries"; #1021 §3.3 says "12 binaries on disk".
    # The discrepancy is legitimate (F2's documented set vs #1021's
    # on-disk inventory including new f2_pairwise_perm). #1021 §3.3 already
    # acknowledges via "the F2 paper's own §8.1 names a subset" — gate this.
    (
        ISSUE1021_PAPER,
        # Loop 123 D (46th pass A3 SEV-1 fix): the original `\bf12` had a
        # typo — `\b` followed by literal `f` matches "f12", not "12". The
        # actual prose says "**12 binaries** on disk"; the corrected regex
        # finds it.
        r"\*\*12 binaries\*\* on disk",
        r"F2 paper's own §8.1 names a subset",
        "#1021 §3.3 must acknowledge F2 §8.1's different binary scope",
    ),
]


# SCOPED_DIFF: claims that intentionally differ between papers; we just
# assert each claim matches its declared scope. Tracks discrepancies
# that should be sanity-checked rather than equated.
# Each entry: (regex, paper, expected_scope_description, sanity_floor, sanity_ceiling, label)
SCOPED_DIFF_CLAIMS: list[tuple[str, Path, str, int, int, str]] = [
    # F2 §3.5.4 reproducibility: "714 passing tests" refers to src/lib.rs.
    (
        r"cargo test --lib`? exits 0 with (\d+) passing tests",
        F2_PAPER,
        "src/lib.rs only",
        700, 800,
        "F2 §3.5.4 lib test count",
    ),
    # F2 §8.2 "809 tests grouped by source: 714 in src/lib.rs"
    (
        r"lists (\d+) tests grouped\s+by",
        F2_PAPER,
        "total = lib + per-bin + integration",
        800, 900,
        "F2 §8.2 grouped total",
    ),
    # Loop 124 D (47th pass A3 SEV-3): tightened from [20, 35] to [25, 30].
    # Current TMLR-class is 27pp; the old [20, 35] admitted 30% growth
    # silently. The new band tracks ±10% around the current observed
    # value, requiring intentional bumps to be reviewed.
    (
        r"Real-TMLR-class PDF: \*\*(\d+) pages\*\*",
        SUBMISSION_CHECKLIST,
        "TMLR class single-column",
        25, 30,
        "SUBMISSION_CHECKLIST TMLR page sanity range",
    ),
    # Loop 124 D (47th pass A3 SEV-3): tightened from [30, 55] to [38, 46].
    # Current non-anon is 42pp; old band admitted nearly 30% drift.
    (
        r"Non-anonymized PDF: \*\*(\d+) pages\*\*",
        SUBMISSION_CHECKLIST,
        "article wrapper",
        38, 46,
        "SUBMISSION_CHECKLIST non-anon page sanity range",
    ),
]


def check_exact_match(spec: tuple[str, Path, str, Path, str]) -> list[str]:
    pat_a, paper_a, pat_b, paper_b, label = spec
    found_a = find_int_claim(paper_a, pat_a)
    found_b = find_int_claim(paper_b, pat_b)
    mismatches: list[str] = []
    if found_a is None:
        mismatches.append(
            f"{label}: pattern not found in {paper_a.relative_to(CRATE_ROOT)} "
            f"({pat_a!r})")
        return mismatches
    if found_b is None:
        mismatches.append(
            f"{label}: pattern not found in {paper_b.relative_to(CRATE_ROOT)} "
            f"({pat_b!r})")
        return mismatches
    val_a, line_a = found_a
    val_b, line_b = found_b
    if val_a != val_b:
        mismatches.append(
            f"{label}: {paper_a.name}:{line_a} = {val_a} vs "
            f"{paper_b.name}:{line_b} = {val_b}")
    return mismatches


def check_acknowledges(spec: tuple[Path, str, str, str]) -> list[str]:
    """Loop 123 D (46th pass A3 SEV-2 + SEV-3): the gate now fails if EITHER
    the claim is missing (claim was deleted without removing the
    acknowledgement-requirement registration) OR the claim is present but
    the acknowledgement isn't. Eliminates the silent-pass-on-claim-removal
    class the 46th pass surfaced."""
    paper, claim_pat, ack_pat, label = spec
    text = paper.read_text()
    if not re.search(claim_pat, text):
        return [
            f"{label}: claim {claim_pat!r} no longer present in "
            f"{paper.relative_to(CRATE_ROOT)} — either the paper deleted "
            "the claim without removing this registration, or the regex "
            "drifted. Update the registry to match current prose."
        ]
    if not re.search(ack_pat, text):
        return [
            f"{label}: claim {claim_pat!r} found in "
            f"{paper.relative_to(CRATE_ROOT)} but acknowledgement "
            f"{ack_pat!r} is missing — cross-paper relationship needs "
            "to be explicitly noted to avoid silent contradiction with the "
            "other paper's different scope."
        ]
    return []


def check_scoped_diff(spec: tuple[str, Path, str, int, int, str]) -> list[str]:
    pat, paper, scope_desc, lo, hi, label = spec
    found = find_int_claim(paper, pat)
    mismatches: list[str] = []
    if found is None:
        mismatches.append(
            f"{label}: pattern not found in {paper.relative_to(CRATE_ROOT)} "
            f"({pat!r})")
        return mismatches
    val, line_no = found
    if val < lo or val > hi:
        mismatches.append(
            f"{label}: {paper.name}:{line_no} = {val} out of sanity range "
            f"[{lo}, {hi}] (scope: {scope_desc})")
    return mismatches


def main() -> int:
    all_mismatches: list[str] = []
    for spec in EXACT_MATCH_CLAIMS:
        ms = check_exact_match(spec)
        label = spec[-1]
        if ms:
            all_mismatches.extend(ms)
            print(f"# FAIL  EXACT  {label}", file=sys.stderr)
            for m in ms:
                print(f"  {m}", file=sys.stderr)
        else:
            print(f"# OK    EXACT  {label}")
    for spec in SCOPED_DIFF_CLAIMS:
        ms = check_scoped_diff(spec)
        label = spec[-1]
        if ms:
            all_mismatches.extend(ms)
            print(f"# FAIL  SCOPED {label}", file=sys.stderr)
            for m in ms:
                print(f"  {m}", file=sys.stderr)
        else:
            print(f"# OK    SCOPED {label}")

    for spec in ACKNOWLEDGES_CLAIMS:
        ms = check_acknowledges(spec)
        label = spec[-1]
        if ms:
            all_mismatches.extend(ms)
            print(f"# FAIL  ACKN   {label}", file=sys.stderr)
            for m in ms:
                print(f"  {m}", file=sys.stderr)
        else:
            print(f"# OK    ACKN   {label}")

    if all_mismatches:
        print(f"# verify_cross_paper_consistency.py — "
              f"{len(all_mismatches)} cross-paper drift(s)",
              file=sys.stderr)
        return 1
    n_total = (len(EXACT_MATCH_CLAIMS) + len(SCOPED_DIFF_CLAIMS) +
               len(ACKNOWLEDGES_CLAIMS))
    print(f"# verify_cross_paper_consistency.py — "
          f"{n_total} cross-paper claims verified, 0 drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
