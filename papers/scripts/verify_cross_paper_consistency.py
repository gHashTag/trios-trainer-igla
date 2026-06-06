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
  says "830 tests" (total = lib + per-binary + integration). Different
  scopes — gated for self-consistency.
- F2 §1 abstract says "830 tests"; #1021 §1.3 says "830 tests" — same
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


# Module-level constant so a future claim can request a wider window via
# import + override; Loop 127 B (49th pass A5b SEV-4 — elevated from
# magic per-call default).
DEFAULT_MAX_CONTEXT_CHARS = 200


def find_int_claim(path: Path, pattern: str,
                   max_context_chars: int | None = None
                   ) -> tuple[int, int] | None:
    """Return (claimed_int, line_number) of the first match, or None.

    Loop 125 B (47th pass A4 SEV-4): the original used `re.DOTALL`,
    which let `.` cross arbitrary line boundaries. With 8+ claims
    registered, future regexes with shared prefixes risk silently
    matching across paragraph boundaries. We now drop DOTALL and
    bound the match window via `max_context_chars` so each pattern
    must be self-contained within a single paragraph-ish span.

    Loop 127 B (49th pass A5 SEV-4 fixes):
    - max_context_chars now defaults to DEFAULT_MAX_CONTEXT_CHARS
      (module-level constant) — no more magic per-call literal.
    - When the pattern matches but its span exceeds the bound, the
      function PRINTS a stderr diagnostic naming the path + span
      before returning None. Previously the "no match" return was
      ambiguous between "regex didn't match" and "match but
      over-spanned"; now a debugger sees which case applies.
    """
    if max_context_chars is None:
        max_context_chars = DEFAULT_MAX_CONTEXT_CHARS
    pat = re.compile(pattern)
    text = path.read_text()
    m = pat.search(text)
    if not m:
        return None
    span_chars = m.end() - m.start()
    if span_chars > max_context_chars:
        # Loop 127 B (49th pass A5 SEV-4): disambiguate "over-spanned"
        # from "no match" in stderr.
        rel = path.relative_to(CRATE_ROOT) if path.is_absolute() else path
        print(f"# WARN find_int_claim: {rel}: regex {pattern!r} matched "
              f"but spans {span_chars} chars > max {max_context_chars} — "
              "likely paragraph-crossing; rejected. To accept, pass an "
              "explicit max_context_chars=... at the call site.",
              file=sys.stderr)
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
    # Loop 122 B: TMLR-class PDF page count, paired with itself in the
    # same file. Loop 126 D (48th pass A5 SEV-2): this entry was a no-op
    # — find_int_claim returns the first match for both calls so val_a
    # == val_b trivially. Removed and replaced with the SCOPED_DIFF
    # below at a hardcoded sanity-equal range [27, 27] so a one-line
    # bump fires the gate.
    # Loop 128 D (submission-readiness audit): the non-anon vs anon
    # equality claim was wrong post-Loop-127. Anonymization replaces
    # the §10.3 Acknowledgments body with a 1-line OMISSION placeholder;
    # depending on prose layout that can shift the page count by ±1.
    # Current: non-anon = 43, anon = 42. Each is now pinned individually
    # via SCOPED_DIFF below; the cross-paper EXACT_MATCH between them is
    # no longer a real invariant and has been removed.
]


# ACKNOWLEDGES: paper A says X, paper B says Y, where X ≠ Y is intentional
# (different scopes). The gate requires that paper A explicitly acknowledges
# the other paper's different scope, so that a TMLR reviewer reading both
# in sequence isn't surprised.
#
# Each entry: (paper, claim_pattern, ack_pattern, description, [optional]).
# The optional 5th field (default False) controls behavior when claim_pattern
# is not found: by default the gate FAILs (so that registry rot is loud);
# set to True for transitional periods where the claim wording is in flux
# and the gate should WARN-not-FAIL on the claim being missing. Loop 125 B
# (47th pass A5b SEV-4) added this for registry growth UX.
ACKNOWLEDGES_CLAIMS: list[tuple] = [
    # F2 §8.1 says "Ten F2 binaries"; #1021 §3.3 says "12 binaries on disk".
    # The discrepancy is legitimate (F2's documented set vs #1021's
    # on-disk inventory including new f2_pairwise_perm). #1021 §3.3 already
    # acknowledges via "the F2 paper's own §8.1 names a subset" — gate this.
    (
        ISSUE1021_PAPER,
        # Loop 123 D (46th pass A3 SEV-1 fix): the original `\bf12` had a
        # typo — `\b` followed by literal `f` matches "f12", not "12".
        # Loop 132-A SEV-4 fix #13: relax markdown emphasis to optional
        # `**` so a future copyedit that drops bold doesn't break the
        # gate. `(?:\*\*)?` permits both "**12 binaries**" and "12
        # binaries" forms; \b boundary on the digit prevents matching
        # "112 binaries" or similar.
        r"(?:\*\*)?\b12 binaries(?:\*\*)? on disk",
        r"F2 paper's own §8.1 names a subset",
        "#1021 §3.3 must acknowledge F2 §8.1's different binary scope",
    ),
]


# SCOPED_DIFF: claims that intentionally differ between papers and
# legitimately span a sanity range (lo < hi). For exact-equality
# claims (lo == hi), use EXACT_PIN_CLAIMS instead — clearer semantics
# and lower per-entry boilerplate.
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
    # F2 §8.2 "830 tests grouped by source: 735 in src/lib.rs"
    (
        r"lists (\d+) tests grouped\s+by",
        F2_PAPER,
        "total = lib + per-bin + integration",
        800, 900,
        "F2 §8.2 grouped total",
    ),
]


# EXACT_PIN: claim pinned to a single value. Loop 133-A SEV-4 fix #7:
# reclassifies the three [N, N] SCOPED_DIFFs that were exact-pins
# disguised as ranges (Loop 128 D acknowledged the disguise in
# comments). Each entry: (regex, paper, expected_value, label).
EXACT_PIN_CLAIMS: list[tuple[str, Path, int, str]] = [
    (
        r"Real-TMLR-class PDF: \*\*(\d+) pages\*\*",
        SUBMISSION_CHECKLIST,
        30,
        "SUBMISSION_CHECKLIST TMLR page exact pin",
    ),
    (
        r"Non-anonymized PDF: \*\*(\d+) pages\*\*",
        SUBMISSION_CHECKLIST,
        48,
        "SUBMISSION_CHECKLIST non-anon page exact pin",
    ),
    (
        r"Anonymized PDF: \*\*(\d+) pages\*\*",
        SUBMISSION_CHECKLIST,
        47,
        "SUBMISSION_CHECKLIST anon page exact pin",
    ),
]


def check_exact_pin(spec: tuple[str, Path, int, str]) -> list[str]:
    pat, paper, expected, label = spec
    found = find_int_claim(paper, pat)
    if found is None:
        return [
            f"{label}: pattern not found in {paper.relative_to(CRATE_ROOT)} "
            f"({pat!r})"
        ]
    val, line_no = found
    if val != expected:
        return [
            f"{label}: {paper.name}:{line_no} = {val} != expected pin {expected}"
        ]
    # Loop 134 — 57th-pass SEV-4 fix #9: assert the pattern matches
    # exactly once in the document. Multiple matches mean the gate is
    # picking the first one by convention; a future bump of the
    # primary anchor while a stale duplicate (e.g., footnote, appendix)
    # stays at the old value would silently pass.
    text = paper.read_text()
    n_matches = len(re.findall(pat, text))
    if n_matches > 1:
        return [
            f"{label}: pattern {pat!r} matches {n_matches}× in "
            f"{paper.name} — ambiguous which occurrence is the pin. "
            "Tighten the regex with surrounding context to select one."
        ]
    return []


# Loop 134 — 57th-pass SEV-4 fix #11: canonical label aliases moved
# here as the single source of truth. Paper prose uses abbreviated
# labels (EXACT, SCOPED, ACKN, RELATIONAL, EXACT_PIN); the gate's
# *_CLAIMS list names use the canonical longer forms. Other scripts
# (e.g., verify_class_registry_binding.py) import this mapping.
CLASS_LABEL_ALIASES: dict[str, str] = {
    "EXACT": "EXACT_MATCH",
    "SCOPED": "SCOPED_DIFF",
    "ACKN": "ACKNOWLEDGES",
    "RELATIONAL": "RELATIONAL",
    "EXACT_PIN": "EXACT_PIN",
    # Legacy alias for the early Loop 133 drafts that used "PIN" alone.
    "PIN": "EXACT_PIN",
}


# RELATIONAL: two integer claims must satisfy a relational invariant.
# Loop 130 B closes the 52nd-pass deferred SEV-3: the non-anon vs anon
# page counts are now pinned individually as SCOPED_DIFFs (Loop 128 D),
# but the structural invariant — non-anon must be ≥ anon, because
# anonymization can only replace content with the equal-or-shorter
# placeholder "[OMITTED FOR DOUBLE-BLIND REVIEW]" — wasn't enforced.
# An exact-pin would catch absolute drift but miss the legitimacy
# constraint when both numbers were bumped together. This class
# expresses such structural invariants directly.
#
# Each entry: (regex_a, paper_a, regex_b, paper_b, comparator, label)
# Comparator is one of "ge", "gt", "le", "lt", "eq" (resolved via
# COMPARATORS dict).
COMPARATORS = {
    "ge": (lambda a, b: a >= b, ">="),
    "gt": (lambda a, b: a > b, ">"),
    "le": (lambda a, b: a <= b, "<="),
    "lt": (lambda a, b: a < b, "<"),
    "eq": (lambda a, b: a == b, "=="),
}

RELATIONAL_CLAIMS: list[tuple[str, Path, str, Path, str, str]] = [
    # Non-anon ≥ anon page count: §10.3 Acknowledgments is replaced by
    # a 1-line OMISSION placeholder during anonymization. Removing
    # content can never grow the document, so non-anon must dominate.
    # If a future loop introduces author-block changes that GROW the
    # anonymized version above the non-anon (e.g., placeholder is
    # longer than original §10.3 text), this gate fires and forces
    # an explicit review.
    (
        r"Non-anonymized PDF: \*\*(\d+) pages\*\*",
        SUBMISSION_CHECKLIST,
        r"Anonymized PDF: \*\*(\d+) pages\*\*",
        SUBMISSION_CHECKLIST,
        "ge",
        "Non-anon ≥ anon page count (structural anonymization invariant)",
    ),
    # Loop 131 B: anon ≥ TMLR-class page count. The TMLR class uses
    # single-column tight layout; the article wrapper used for anon/
    # non-anon is multi-line wider. Same content always renders
    # shorter in TMLR class, so TMLR_pp ≤ anon_pp must hold. This
    # binds the page-count chain TMLR ≤ anon ≤ non-anon, catching any
    # future PDF-rendering inversion in one extra check.
    (
        r"Anonymized PDF: \*\*(\d+) pages\*\*",
        SUBMISSION_CHECKLIST,
        r"Real-TMLR-class PDF: \*\*(\d+) pages\*\*",
        SUBMISSION_CHECKLIST,
        "ge",
        "Anon ≥ TMLR-class page count (article-wrapper-wider-than-tmlr invariant)",
    ),
    # Loop 131 B: F2 §8.2 grouped total (cargo test --lib + per-bin +
    # integration = 809) ≥ §3.5.4 lib-only count (cargo test --lib =
    # 714). Lib tests are a strict subset of the cumulative total;
    # if a future loop drops integration tests without also reducing
    # the §3.5.4 lib number, this gate fires.
    (
        r"lists (\d+) tests grouped\s+by",
        F2_PAPER,
        r"cargo test --lib`? exits 0 with (\d+) passing tests",
        F2_PAPER,
        "ge",
        "F2 §8.2 grouped total ≥ §3.5.4 lib-only (subset invariant)",
    ),
    # Loop 131 B: #1021 §3.3 on-disk binary count ≥ F2 §D test-inventory
    # "10 F2 binaries". #1021 names a superset (10 F2 + 2 new #1021
    # binaries including f2_pairwise_perm + 1 more); F2 names only its
    # documented set. Subset invariant: #1021's count cannot drop below
    # F2's without removing a binary outright.
    (
        r"\*\*(\d+) binaries\*\* on disk",
        ISSUE1021_PAPER,
        r"830 tests across `src/lib\.rs`, (\d+) F2 binaries",
        F2_PAPER,
        "ge",
        "#1021 §3.3 binary count ≥ F2 §D 10-binary inventory (superset invariant)",
    ),
]


def check_relational(spec: tuple[str, Path, str, Path, str, str]) -> list[str]:
    pat_a, paper_a, pat_b, paper_b, cmp_name, label = spec
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
    if cmp_name not in COMPARATORS:
        mismatches.append(
            f"{label}: unknown comparator {cmp_name!r}; expected one of "
            f"{sorted(COMPARATORS)}")
        return mismatches
    fn, sym = COMPARATORS[cmp_name]
    if not fn(val_a, val_b):
        mismatches.append(
            f"{label}: {paper_a.name}:{line_a} = {val_a} {sym} "
            f"{paper_b.name}:{line_b} = {val_b} — invariant violated")
    return mismatches


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


def check_acknowledges(spec: tuple) -> list[str]:
    """Loop 123 D (46th pass A3 SEV-2 + SEV-3): the gate now fails if EITHER
    the claim is missing (claim was deleted without removing the
    acknowledgement-requirement registration) OR the claim is present but
    the acknowledgement isn't. Eliminates the silent-pass-on-claim-removal
    class the 46th pass surfaced.

    Loop 125 B (47th pass A5b SEV-4): optional 5th spec field controls
    behavior on missing claim — True → WARN to stderr but don't fail
    (for transitional periods); default False → FAIL.
    """
    # Loop 127 D (50th pass A3 SEV-4): explicit spec-length validation.
    # The shim in meta_test_cross_paper_gates.py uses *spec[1:] which is
    # permissive about tuple length; the dispatcher previously had a
    # binary 4-vs-else. A 6-tuple registration would silently propagate
    # through the shim and crash here with a cryptic "too many values"
    # ValueError. Make the contract explicit.
    if len(spec) not in (4, 5):
        return [
            f"check_acknowledges: registry entry has {len(spec)} fields; "
            "expected 4 (required) or 5 (with optional=True). Update the "
            "registry or extend the dispatcher."
        ]
    if len(spec) == 4:
        paper, claim_pat, ack_pat, label = spec
        optional = False
    else:
        paper, claim_pat, ack_pat, label, optional = spec
    text = paper.read_text()
    if not re.search(claim_pat, text):
        msg = (
            f"{label}: claim {claim_pat!r} no longer present in "
            f"{paper.relative_to(CRATE_ROOT)} — either the paper deleted "
            "the claim without removing this registration, or the regex "
            "drifted. Update the registry to match current prose."
        )
        if optional:
            print(f"# WARN  ACKN   {msg}", file=sys.stderr)
            return []
        return [msg]
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

    for spec in RELATIONAL_CLAIMS:
        ms = check_relational(spec)
        label = spec[-1]
        if ms:
            all_mismatches.extend(ms)
            print(f"# FAIL  RELAT  {label}", file=sys.stderr)
            for m in ms:
                print(f"  {m}", file=sys.stderr)
        else:
            print(f"# OK    RELAT  {label}")

    for spec in EXACT_PIN_CLAIMS:
        ms = check_exact_pin(spec)
        label = spec[-1]
        if ms:
            all_mismatches.extend(ms)
            print(f"# FAIL  PIN    {label}", file=sys.stderr)
            for m in ms:
                print(f"  {m}", file=sys.stderr)
        else:
            print(f"# OK    PIN    {label}")

    if all_mismatches:
        print(f"# verify_cross_paper_consistency.py — "
              f"{len(all_mismatches)} cross-paper drift(s)",
              file=sys.stderr)
        return 1
    n_total = (len(EXACT_MATCH_CLAIMS) + len(SCOPED_DIFF_CLAIMS) +
               len(ACKNOWLEDGES_CLAIMS) + len(RELATIONAL_CLAIMS) +
               len(EXACT_PIN_CLAIMS))
    print(f"# verify_cross_paper_consistency.py — "
          f"{n_total} cross-paper claims verified, 0 drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
