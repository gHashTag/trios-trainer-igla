#!/usr/bin/env python3
"""verify_stage_count_consistency.py — gate paper claims of "N stages"
against the actual count in run_all_checks.sh.

Loop 118 B operationalizes the SEV-2 stage-count-drift class that
the 39th + 40th adversarial passes both surfaced:
- 39th pass: F2 §E said "7 scripts + 1" while disk had 17 stages.
- 40th pass: F2 §E rewrite said "7 catalogued + 10 additional"
  but §E head says "**eight** auxiliary scripts".

Both were prose-drift bugs no static check caught. This script
parses the STAGES array in run_all_checks.sh, counts entries,
and asserts each registered "this paper says N stages" claim
matches.

Each entry in CLAIMS specifies:
- file: path relative to crate root
- anchor: a regex that locates the claim
- expected_n: the integer the claim should match
- description: short label for failure messages

Mismatches fail the gate with line-by-line diagnostics.

Usage: papers/scripts/verify_stage_count_consistency.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]

RUN_ALL_CHECKS = CRATE_ROOT / "papers" / "scripts" / "run_all_checks.sh"
STAGES_RE = re.compile(r"^\s*\"([^:\"]+):", re.MULTILINE)


def actual_stage_count() -> int:
    """Parse STAGES=(...) array in run_all_checks.sh."""
    text = RUN_ALL_CHECKS.read_text()
    # Find the STAGES=( block.
    m = re.search(r"STAGES=\(\s*\n(.*?)\n\)", text, re.DOTALL)
    if not m:
        raise RuntimeError("could not locate STAGES=( ... ) block")
    body = m.group(1)
    # Each line in the block that opens with `    "name:` is a stage.
    entries = re.findall(r'^\s+"([^:"]+):', body, re.MULTILINE)
    return len(entries)


# Registry of paper claims to verify.
# Each tuple: (relative_path, regex_pattern, expected_count_function, description)
CLAIMS: list[tuple[str, str, str]] = [
    # F2 paper §E: "Currently chains **18 stages** on disk"
    # Allow the wrapped form "**18 stages\*\*\n  on disk" via [\s\n]* gap.
    (
        "papers/f2_methodology.md",
        r"Currently chains\s+\*\*(\d+)\s*stages\*\*[\s\n]*on disk",
        "F2 §E run_all_checks.sh stage count",
    ),
    # #1021 paper §5.4: "18 stages on disk as of Loop 117"
    (
        "papers/phi_ladder_paper_intro_draft.md",
        r"\((\d+) stages on disk(?:\s+as of Loop \d+)?\)",
        "#1021 §5.4 on-disk gate count",
    ),
    # #1021 paper §5.4 decomposition: "12 F2-scope ... 6 #1021-scoped" = 18
    (
        "papers/phi_ladder_paper_intro_draft.md",
        r"\*\*(\d+)-stage on-disk gate\*\* breaks down",
        "#1021 §5.4 decomposition opener",
    ),
    # CHANGELOG §10 "on-disk gate now runs **N stages**" (Loop 121 D).
    (
        "papers/CHANGELOG.md",
        r"on-disk gate now runs \*\*(\d+) stages\*\*",
        "CHANGELOG §10 on-disk gate count",
    ),
    # SUBMISSION_CHECKLIST.md §1 "exits 0 with **N/N PASS**" (Loop 129 C —
    # operationalizes the submission-readiness audit finding that §1's
    # "13/13" wording drifted silently as new stages were added).
    (
        "papers/SUBMISSION_CHECKLIST.md",
        r"exits 0 with \*\*(\d+)/\d+ PASS\*\*",
        "SUBMISSION_CHECKLIST §1 exit-message stage count",
    ),
]


def find_claim(path: Path, pattern: str) -> tuple[int, int] | None:
    """Return (claimed_count, line_number) of the first match, or None.
    DOTALL so multi-line wrap doesn't break matching; line number is the
    line containing the digit group."""
    pat = re.compile(pattern, re.DOTALL)
    text = path.read_text()
    m = pat.search(text)
    if not m:
        return None
    # Line containing the captured digit group.
    digit_pos = m.start(1)
    line_no = text[:digit_pos].count("\n") + 1
    return int(m.group(1)), line_no


# Decomposition claims of the form "N1 F2-scope ... + N2 #1021-scoped = N total".
# Loop 119 B: gate the partition arithmetic, not just the total. Catches drift
# in claims like "12 F2 + 6 #1021 = 18" when stages are added to either bucket.
DECOMPOSITION_CLAIMS: list[tuple[str, str, str]] = [
    (
        "papers/phi_ladder_paper_intro_draft.md",
        # "**13 F2-scope stages**" ... line wrap ... "+ 6 #1021-scoped\n stages"
        r"\*\*(\d+) F2-scope stages\*\*[\s\S]*?\+\s+(\d+)\s+#1021-scoped[\s\S]*?stages",
        "#1021 §5.4 decomposition partition",
    ),
    # F2 paper §E partition (Loop 120 B): "seven appear as individual stages
    # ... the other twelve stages are gates introduced". The pattern matches
    # the numeric pair within the run_all_checks.sh paragraph; the 42nd pass
    # caught a "twelve" + "additional 10" residue in this paragraph that the
    # initial #1021-only gate could not catch. \s+ tolerates the line wrap
    # between "appear as" and "individual stages".
    (
        "papers/f2_methodology.md",
        r"(\w+)\s+appear\s+as\s+individual\s+stages[\s\S]*?other\s+(\w+)\s+stages\s+are\s+gates",
        "F2 §E catalogue partition (words)",
    ),
]


# DERIVED_CLAIMS: each claim's expected value is `actual - constant_offset`.
# Used for F2 §E's "additional N" sentence whose N should match the
# "non-catalogue" stage count = actual - 7 (the 7 catalogued-as-stage
# count out of 8 catalogue bullets; the 8th is run_all_checks.sh itself).
DERIVED_CLAIMS: list[tuple[str, str, str, str]] = [
    (
        "papers/f2_methodology.md",
        # "the additional N are documented" — N should equal actual - 7
        r"the additional (\d+) are documented",
        "catalogued_as_stage",   # offset key resolved at runtime
        "F2 §E follow-up additional count "
        "(= actual - catalogued_as_stage_offset())",
    ),
]


def resolve_offset(key: str) -> int:
    """Resolve a named offset to an integer. Loop 122 C: replaces hard-coded 7."""
    if key == "catalogued_as_stage":
        return catalogued_as_stage_offset()
    if isinstance(key, int):
        return key
    raise ValueError(f"unknown offset key {key!r}")


_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
          "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
          "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def words_to_int(w: str) -> int | None:
    """Convert English number words (0..99) + digits to int.

    Loop 122 C (45th pass A5): extended from the original 0..20 cap so the
    gate doesn't break silently when the on-disk stage count grows past 20.
    Handles "twenty-one", "twenty one", "twenty 1" forms (the last via the
    digit shortcut).
    """
    if w.isdigit():
        return int(w)
    w = w.lower().strip()
    if not w:
        # Loop 123 D (46th pass A1 SEV-4): empty string used to fall
        # through to `_TENS.index("") * 10 = 0`. Reject explicitly.
        return None
    if w in _UNITS:
        return _UNITS.index(w)
    # Loop 123 D: _TENS[0] and _TENS[1] are both "" — only treat
    # explicit ten-word matches as valid, never the empty placeholder.
    if w in _TENS[2:]:
        return _TENS.index(w) * 10
    # Compound forms: "twenty-one", "thirty-five", etc.
    for sep in ("-", " "):
        if sep in w:
            tens_part, _, units_part = w.partition(sep)
            if tens_part in _TENS and units_part in _UNITS[1:10]:
                return _TENS.index(tens_part) * 10 + _UNITS.index(units_part)
    return None


def catalogued_as_stage_offset() -> int:
    """Derive the F2 §E catalogue-as-stage count rather than hard-coding 7.

    The §E catalogue lists 8 scripts; one of them (`run_all_checks.sh`)
    is the orchestrator and not a stage of itself. The remaining 7
    contribute to STAGES. Loop 122 C (45th pass A5): future-proof by
    computing this from the catalogue rather than the literal 7.
    """
    paper = (CRATE_ROOT / "papers" / "f2_methodology.md").read_text()
    # Count bullets of the form "- **`papers/scripts/<name>`** ..." within
    # the §E catalogue body (between "### E. Reviewer-grade tooling
    # catalogue" and the closing "supplementary-zip packer" or section
    # boundary).
    cat_re = re.compile(r"###\s+E\.\s+Reviewer-grade tooling catalogue(.*?)(?=###|\Z)",
                        re.DOTALL)
    m = cat_re.search(paper)
    if not m:
        # Loop 123 D (46th pass A2 SEV-2): warn loudly instead of silently
        # masking the rename. The fallback 7 keeps the gate working but the
        # stderr trail makes the drift visible.
        print("# WARN catalogued_as_stage_offset: §E section heading "
              "not found; falling back to hard-coded 7. The §E catalogue "
              "may have been renamed.", file=sys.stderr)
        return 7
    body = m.group(1)
    n_bullets = len(re.findall(r"^- \*\*`papers/scripts/", body, re.MULTILINE))
    has_orchestrator = bool(re.search(r"run_all_checks\.sh", body))
    if n_bullets == 0:
        # Loop 123 D (46th pass A2 SEV-2): the bullet-format change
        # used to clamp to 0 silently. Now we warn AND fall back.
        print("# WARN catalogued_as_stage_offset: §E catalogue has 0 "
              "bullets in expected `- **`papers/scripts/...`` format. "
              "Bullet style may have changed (e.g., `* **` instead of "
              "`- **`); falling back to hard-coded 7.", file=sys.stderr)
        return 7
    # Loop 123 D (46th pass A2 SEV-3): content-based detection of the
    # orchestrator rather than positional `- 1`. If `run_all_checks.sh`
    # is no longer in the catalogue, the offset = bullet count directly.
    return n_bullets - 1 if has_orchestrator else n_bullets


def find_decomposition(path: Path, pattern: str) -> tuple[int, int, int] | None:
    """Return (n1, n2, line_no_of_n1) or None. Accepts digits or English words."""
    pat = re.compile(pattern, re.DOTALL)
    text = path.read_text()
    m = pat.search(text)
    if not m:
        return None
    line_no = text[:m.start(1)].count("\n") + 1
    n1 = words_to_int(m.group(1))
    n2 = words_to_int(m.group(2))
    if n1 is None or n2 is None:
        return None
    return n1, n2, line_no


def main() -> int:
    actual = actual_stage_count()
    print(f"# verify_stage_count_consistency.py — STAGES array has {actual} entries")

    mismatches: list[str] = []
    for rel, pattern, desc in CLAIMS:
        path = CRATE_ROOT / rel
        if not path.exists():
            mismatches.append(f"{rel}: file missing for claim '{desc}'")
            continue
        found = find_claim(path, pattern)
        if found is None:
            mismatches.append(
                f"{rel}: no match for '{desc}' (pattern {pattern!r}) — "
                "paper claim might have moved or been reworded")
            continue
        claimed, line_no = found
        if claimed != actual:
            mismatches.append(
                f"{rel}:{line_no}: '{desc}' claims {claimed} but actual is {actual}")
        else:
            print(f"# OK  {rel}:{line_no}  — {desc} = {claimed}")

    # Loop 120 B: derived claims (N == actual - resolved_offset).
    # Loop 122 C: offset is now a key resolved to a computed value
    # via `resolve_offset`, future-proofing against §E catalogue growth.
    for rel, pattern, offset_key, desc in DERIVED_CLAIMS:
        path = CRATE_ROOT / rel
        if not path.exists():
            mismatches.append(f"{rel}: file missing for derived '{desc}'")
            continue
        found = find_claim(path, pattern)
        if found is None:
            mismatches.append(
                f"{rel}: no match for derived '{desc}' (pattern {pattern!r})")
            continue
        claimed, line_no = found
        try:
            offset = resolve_offset(offset_key)
        except ValueError as e:
            # Loop 123 D (46th pass A5 SEV-3): catch typo'd offset keys
            # cleanly instead of crashing main() with a traceback.
            mismatches.append(
                f"{rel}: derived '{desc}' has invalid offset key "
                f"{offset_key!r}: {e}")
            continue
        expected = actual - offset
        if claimed != expected:
            mismatches.append(
                f"{rel}:{line_no}: '{desc}' claims {claimed} but expected "
                f"{expected} (= {actual} - {offset})")
        else:
            print(f"# OK  {rel}:{line_no}  — {desc} = {claimed}")

    # Loop 119 B: decomposition arithmetic.
    for rel, pattern, desc in DECOMPOSITION_CLAIMS:
        path = CRATE_ROOT / rel
        if not path.exists():
            mismatches.append(f"{rel}: file missing for decomposition '{desc}'")
            continue
        found = find_decomposition(path, pattern)
        if found is None:
            mismatches.append(
                f"{rel}: no match for decomposition '{desc}' "
                f"(pattern {pattern!r}) — paper claim might have moved or been reworded")
            continue
        n1, n2, line_no = found
        partition_sum = n1 + n2
        if partition_sum != actual:
            mismatches.append(
                f"{rel}:{line_no}: '{desc}' claims {n1}+{n2}={partition_sum} "
                f"but actual STAGES total is {actual}")
        else:
            print(f"# OK  {rel}:{line_no}  — {desc} = {n1}+{n2}={partition_sum}")

    if mismatches:
        print(f"# FAIL  {len(mismatches)} stage-count drift(s) "
              f"vs actual {actual}:", file=sys.stderr)
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        return 1
    print(f"# verify_stage_count_consistency.py — "
          f"{len(CLAIMS)} counts + {len(DECOMPOSITION_CLAIMS)} "
          f"decompositions + {len(DERIVED_CLAIMS)} derived verified, "
          "0 drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
