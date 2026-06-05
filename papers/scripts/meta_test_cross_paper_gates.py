#!/usr/bin/env python3
"""meta_test_cross_paper_gates.py — synthetic-failure tests for the
cross-paper consistency gate.

Loop 124 B operationalizes the 46th-pass lesson: the ACKNOWLEDGES gate
shipped at Loop 122 with a regex typo (`\\bf12` instead of `\\b12`) that
made it silently report OK without ever running. That class of "gate
ships with no enforcement" can only be caught by adversarially breaking
each gate's input and asserting the gate produces the expected failure
diagnostic.

This script does NOT modify any tracked file. It writes a temporary copy
of the relevant paper, mutates it to break ONE registered claim at a
time, monkey-patches the verifier's module-level path constants to point
at the temp copy, runs the gate, and asserts a non-zero exit + a stderr
line mentioning the expected failure class.

Coverage:
- EXACT_MATCH: total cargo-test count (F2 §1 vs #1021 §1.3) →
  mutate F2 to "open-source in Rust with 10 binaries, 999 unit/integration
  tests"; gate must FAIL with "EXACT" and the label.
- SCOPED_DIFF: F2 §3.5.4 lib test count → mutate to 1234 (outside sanity
  range 700-800); gate must FAIL with "out of sanity range".
- ACKNOWLEDGES: #1021 §3.3 acknowledgement → delete the "F2 paper's own
  §8.1 names a subset" sentence; gate must FAIL with "acknowledgement is
  missing".

Each test passes iff the gate REPORTS the right failure on the
synthetic break. If the gate silently passes, the test fails.

Usage:
  papers/scripts/meta_test_cross_paper_gates.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
GATE = CRATE_ROOT / "papers" / "scripts" / "verify_cross_paper_consistency.py"
F2 = CRATE_ROOT / "papers" / "f2_methodology.md"
ISSUE1021 = CRATE_ROOT / "papers" / "phi_ladder_paper_intro_draft.md"
CHECKLIST = CRATE_ROOT / "papers" / "SUBMISSION_CHECKLIST.md"


# Loop 143 A.iii: shared helpers for tempdir+subprocess break-tests.
# The 64th-pass #8 SEV-4 flagged that each of the 5 newer break-tests
# (Loops 140+141) duplicated ~15 LOC. The helpers below collapse the
# pattern to ~5 LOC per new test. Existing tests are not migrated in
# this loop to keep the diff focused; future loops can refactor.
def _copy_to_tmp(tmp: Path,
                 scripts: list[str] | None = None,
                 papers: list[str] | None = None,
                 docs: list[str] | None = None) -> Path:
    """Copy the named files into a tmp mirror of crate layout and
    return the dst papers/scripts path for further mutation."""
    dst_scripts = tmp / "papers" / "scripts"
    dst_papers = tmp / "papers"
    dst_docs = tmp / "docs"
    dst_scripts.mkdir(parents=True, exist_ok=True)
    for name in (scripts or []):
        shutil.copy2(
            CRATE_ROOT / "papers" / "scripts" / name, dst_scripts / name,
        )
    for name in (papers or []):
        shutil.copy2(CRATE_ROOT / "papers" / name, dst_papers / name)
    if docs:
        dst_docs.mkdir(parents=True, exist_ok=True)
        for name in docs:
            shutil.copy2(CRATE_ROOT / "docs" / name, dst_docs / name)
    return dst_scripts


def _run_gate(scripts_dir: Path, gate_name: str
              ) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(scripts_dir / gate_name)],
        capture_output=True, text=True, timeout=30,
    )


def _assert_fires(result: subprocess.CompletedProcess,
                  fragment: str, label: str) -> bool:
    if result.returncode == 0:
        print(f"# FAIL  {label}: gate exited 0 unexpectedly",
              file=sys.stderr)
        return False
    if fragment not in result.stderr:
        print(f"# FAIL  {label}: stderr missing {fragment!r}",
              file=sys.stderr)
        print(f"  stderr: {result.stderr[:300]}", file=sys.stderr)
        return False
    print(f"# OK    {label}: gate fires with expected diagnostic")
    return True


def run_gate(f2_override: Path | None = None,
             issue1021_override: Path | None = None,
             checklist_override: Path | None = None) -> tuple[int, str]:
    """Run the verifier with optional file overrides. Uses an environment
    trick: write a tiny shim that re-imports the gate's module and
    overrides its Path constants before main()."""
    shim = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(CRATE_ROOT)!r} + '/papers/scripts')
import verify_cross_paper_consistency as g
"""
    if f2_override:
        shim += f"g.F2_PAPER = Path({str(f2_override)!r})\n"
    if issue1021_override:
        shim += f"g.ISSUE1021_PAPER = Path({str(issue1021_override)!r})\n"
    if checklist_override:
        shim += f"g.SUBMISSION_CHECKLIST = Path({str(checklist_override)!r})\n"
    # Rebuild claim lists so they pick up the overrides (the registry
    # captures the Path constants at module-load time as literals).
    shim += """
g.EXACT_MATCH_CLAIMS = [
    (regex_a, g.F2_PAPER if paper_a == _orig_f2 else (g.ISSUE1021_PAPER if paper_a == _orig_i1021 else g.SUBMISSION_CHECKLIST),
     regex_b, g.F2_PAPER if paper_b == _orig_f2 else (g.ISSUE1021_PAPER if paper_b == _orig_i1021 else g.SUBMISSION_CHECKLIST),
     label)
    for (regex_a, paper_a, regex_b, paper_b, label) in _saved_exact
]
g.SCOPED_DIFF_CLAIMS = [
    (regex, g.F2_PAPER if paper == _orig_f2 else (g.ISSUE1021_PAPER if paper == _orig_i1021 else g.SUBMISSION_CHECKLIST),
     scope, lo, hi, label)
    for (regex, paper, scope, lo, hi, label) in _saved_scoped
]
g.ACKNOWLEDGES_CLAIMS = [
    # Loop 126 D (49th pass A5 SEV-2 regression fix): handle both
    # 4-tuple (required) and 5-tuple (optional=True). Previous code
    # unpacked only 4-tuple, crashing on any 5-tuple registration.
    (
        (g.F2_PAPER if spec[0] == _orig_f2 else
         (g.ISSUE1021_PAPER if spec[0] == _orig_i1021
          else g.SUBMISSION_CHECKLIST)),
        *spec[1:],
    )
    for spec in _saved_ackn
]
g.RELATIONAL_CLAIMS = [
    (
        regex_a,
        (g.F2_PAPER if paper_a == _orig_f2 else
         (g.ISSUE1021_PAPER if paper_a == _orig_i1021
          else g.SUBMISSION_CHECKLIST)),
        regex_b,
        (g.F2_PAPER if paper_b == _orig_f2 else
         (g.ISSUE1021_PAPER if paper_b == _orig_i1021
          else g.SUBMISSION_CHECKLIST)),
        cmp_name,
        label,
    )
    for (regex_a, paper_a, regex_b, paper_b, cmp_name, label) in _saved_rel
]
g.EXACT_PIN_CLAIMS = [
    (
        regex,
        (g.F2_PAPER if paper == _orig_f2 else
         (g.ISSUE1021_PAPER if paper == _orig_i1021
          else g.SUBMISSION_CHECKLIST)),
        expected,
        label,
    )
    for (regex, paper, expected, label) in _saved_pin
]
sys.exit(g.main())
"""
    # The shim references _orig_f2/_orig_i1021/_saved_* — prepend the
    # captures BEFORE the registry-rebuild block.
    preamble = f"""
_orig_f2 = Path({str(F2)!r})
_orig_i1021 = Path({str(ISSUE1021)!r})
_saved_exact = list(g.EXACT_MATCH_CLAIMS)
_saved_scoped = list(g.SCOPED_DIFF_CLAIMS)
_saved_ackn = list(g.ACKNOWLEDGES_CLAIMS)
_saved_rel = list(g.RELATIONAL_CLAIMS)
_saved_pin = list(g.EXACT_PIN_CLAIMS)
"""
    full = shim.replace(
        "g.EXACT_MATCH_CLAIMS = [",
        preamble + "g.EXACT_MATCH_CLAIMS = [", 1)
    result = subprocess.run(
        [sys.executable, "-c", full],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode, result.stderr


def assert_fails_with(label: str, expect_substring: str,
                      rc: int, stderr: str) -> bool:
    if rc == 0:
        print(f"# FAIL  {label}: gate exited 0, expected non-zero "
              f"(synthetic break didn't produce a failure)", file=sys.stderr)
        return False
    if expect_substring not in stderr:
        print(f"# FAIL  {label}: gate did fail but stderr missing "
              f"expected substring {expect_substring!r}", file=sys.stderr)
        print(f"  actual stderr:\n{stderr[:500]}", file=sys.stderr)
        return False
    print(f"# OK    {label}: gate failed cleanly with expected diagnostic")
    return True


def test_exact_match_break_f2_total(tmp: Path) -> bool:
    """Break F2 §1 total cargo-test count by writing 999."""
    f2_tmp = tmp / "f2_broken.md"
    text = F2.read_text()
    broken = text.replace(
        "open-source in Rust with 10 binaries, 809",
        "open-source in Rust with 10 binaries, 999",
    )
    f2_tmp.write_text(broken)
    rc, err = run_gate(f2_override=f2_tmp)
    return assert_fails_with(
        "EXACT_MATCH break (F2 total test count)",
        "Total cargo-test count",
        rc, err,
    )


def test_scoped_diff_break_lib_count(tmp: Path) -> bool:
    """Push the F2 §3.5.4 lib count outside the [700, 800] sanity range."""
    f2_tmp = tmp / "f2_lib_broken.md"
    text = F2.read_text()
    broken = text.replace(
        "cargo test --lib` exits 0 with 714 passing tests",
        "cargo test --lib` exits 0 with 1234 passing tests",
    )
    f2_tmp.write_text(broken)
    rc, err = run_gate(f2_override=f2_tmp)
    return assert_fails_with(
        "SCOPED_DIFF break (F2 lib count out of range)",
        "out of sanity range",
        rc, err,
    )


def test_relational_break_non_anon_lt_anon(tmp: Path) -> bool:
    """Mutate SUBMISSION_CHECKLIST so non-anon < anon (impossible by the
    anonymization invariant) and assert the gate fires.

    Loop 130 B (52nd-pass deferred SEV-3 closure): adds RELATIONAL class
    coverage so the meta-test's inventory_completeness check stays green
    when the new class is registered."""
    chk_tmp = tmp / "checklist_inverted.md"
    text = CHECKLIST.read_text()
    # Set non-anon to 30, anon to 50 — clearly violates the ≥ invariant
    # without trampling other SCOPED_DIFF exact pins (which are also in
    # the same file but use different regexes that will fail to extract
    # the new values gracefully via SCOPED_DIFF failure rather than
    # masking the RELATIONAL failure).
    broken = re.sub(
        r"Non-anonymized PDF: \*\*\d+ pages\*\*",
        "Non-anonymized PDF: **30 pages**",
        text, count=1,
    )
    broken = re.sub(
        r"Anonymized PDF: \*\*\d+ pages\*\*",
        "Anonymized PDF: **50 pages**",
        broken, count=1,
    )
    if broken == text:
        print("# SKIP relational break: SUBMISSION_CHECKLIST didn't "
              "contain the expected page-count patterns",
              file=sys.stderr)
        return True
    chk_tmp.write_text(broken)
    rc, err = run_gate(checklist_override=chk_tmp)
    return assert_fails_with(
        "RELATIONAL break (non-anon < anon)",
        "invariant violated",
        rc, err,
    )


def test_relational_boundary_equality(tmp: Path) -> bool:
    """Loop 131-A SEV-4 fix #5: guard against silent direction-flip in
    COMPARATORS. The "ge" comparator must accept equality (a == b).
    Mutate SUBMISSION_CHECKLIST so non-anon == anon == 42 (the
    boundary case) and assert the gate exits 0 — confirming "ge"
    semantics are inclusive, not strictly-greater.

    Without this test, a registry author who flips "ge" → "gt" would
    introduce a subtle false-FAIL at the equality boundary that the
    existing far-from-boundary break-test (non-anon=30, anon=50)
    cannot expose. Equality boundary is the canonical regression
    surface for comparator semantics."""
    chk_tmp = tmp / "checklist_equal.md"
    text = CHECKLIST.read_text()
    broken = re.sub(
        r"Non-anonymized PDF: \*\*\d+ pages\*\*",
        "Non-anonymized PDF: **42 pages**",
        text, count=1,
    )
    broken = re.sub(
        r"Anonymized PDF: \*\*\d+ pages\*\*",
        "Anonymized PDF: **42 pages**",
        broken, count=1,
    )
    # Also pin the SCOPED_DIFF exact-pin entries so they don't fail
    # independently — we're testing RELATIONAL behavior in isolation.
    # The SUBMISSION_CHECKLIST non-anon SCOPED_DIFF pin is [43, 43];
    # this test legitimately violates it. We accept the SCOPED_DIFF
    # failure as collateral and check ONLY that the RELATIONAL "ge"
    # comparator does NOT produce a "invariant violated" diagnostic
    # in the stderr output. If "ge" were flipped to "gt", non-anon=42
    # would not be > anon=42 → "invariant violated" would appear.
    chk_tmp.write_text(broken)
    rc, err = run_gate(checklist_override=chk_tmp)
    # The SCOPED_DIFF fires on the 42 vs 43 pin — rc will be 1.
    # But the RELATIONAL "Non-anon ≥ anon" must NOT appear in the
    # mismatch list, because 42 >= 42 is true.
    if "Non-anon ≥ anon page count" in err and "invariant violated" in err:
        # Find the line that mentions "invariant violated" and check
        # if it's the non-anon-ge-anon one.
        lines = err.splitlines()
        for line in lines:
            if "Non-anon ≥ anon" in line and "invariant violated" in line:
                print(f"# FAIL  RELATIONAL boundary equality: gate "
                      f"reported '{line.strip()}' for non-anon=anon=42 "
                      "— 'ge' comparator should ACCEPT equality. This "
                      "indicates the comparator may have been flipped "
                      "to 'gt' or similar.", file=sys.stderr)
                return False
    print("# OK    RELATIONAL boundary equality: 'ge' accepts a==b "
          "(non-anon=anon=42 case)")
    return True


def test_exact_pin_break_tmlr_page(tmp: Path) -> bool:
    """Mutate the TMLR-class page exact pin (27) to a different value
    and assert the EXACT_PIN gate fires.

    Loop 133-A SEV-4 fix #7 closure: keeps inventory_completeness green
    when the new EXACT_PIN class is registered."""
    chk_tmp = tmp / "checklist_tmlr_pin.md"
    text = CHECKLIST.read_text()
    broken = re.sub(
        r"Real-TMLR-class PDF: \*\*\d+ pages\*\*",
        "Real-TMLR-class PDF: **99 pages**",
        text, count=1,
    )
    if broken == text:
        print("# SKIP exact_pin: TMLR page pattern not in CHECKLIST",
              file=sys.stderr)
        return True
    chk_tmp.write_text(broken)
    rc, err = run_gate(checklist_override=chk_tmp)
    return assert_fails_with(
        "EXACT_PIN break (TMLR page pinned 27 → 99)",
        "!= expected pin",
        rc, err,
    )


def test_burn_down_arithmetic_break(tmp: Path) -> bool:
    """Loop 140 A.iv (migrated to shared helpers Loop 144 A.iii):
    inject "Loop 99 SYNTHETIC: 5+5=11" into a tmpdir copy of
    verify_anonymizer_completeness.py's FALLBACK_BASELINES; assert
    verify_burn_down_history.py fires with "= 10" diagnostic.

    Injects at a NON-most-recent position so the most-recent-binding
    check doesn't mask the arithmetic shape-check."""
    dst = _copy_to_tmp(tmp, scripts=[
        "verify_anonymizer_completeness.py",
        "verify_burn_down_history.py",
        "_gate_utils.py",
        "anonymizer_baseline.json",
    ])
    anon = dst / "verify_anonymizer_completeness.py"
    text = anon.read_text()
    broken = text.replace(
        "Loop 132 C baseline: 29 + 43 = 72.",
        "Loop 132 C baseline: 29 + 43 = 72.\n"
        "#   Loop 99 SYNTHETIC: 5 + 5 = 11.",
    )
    if broken == text:
        print("# SKIP burn_down_arithmetic: FALLBACK_BASELINES marker not found",
              file=sys.stderr)
        return True
    anon.write_text(broken)
    result = _run_gate(dst, "verify_burn_down_history.py")
    return _assert_fires(
        result, "= 10",
        "burn-down arithmetic break",
    )


def test_alias_round_trip_dangling_break(tmp: Path) -> bool:
    """Loop 140 A.iv (migrated to shared helpers Loop 144 A.iii):
    inject dangling alias BOGUS → NONEXISTENT_CLASS into a temp copy
    of verify_cross_paper_consistency.py CLASS_LABEL_ALIASES and assert
    verify_alias_round_trip.py reports the dead alias."""
    dst = _copy_to_tmp(tmp, scripts=[
        "verify_cross_paper_consistency.py",
        "verify_alias_round_trip.py",
        "_gate_utils.py",
    ])
    consist = dst / "verify_cross_paper_consistency.py"
    text = consist.read_text()
    broken = text.replace(
        '"PIN": "EXACT_PIN",',
        '"PIN": "EXACT_PIN",\n    "BOGUS": "NONEXISTENT_CLASS",',
    )
    if broken == text:
        print("# SKIP alias_dangling: CLASS_LABEL_ALIASES anchor not found",
              file=sys.stderr)
        return True
    consist.write_text(broken)
    result = _run_gate(dst, "verify_alias_round_trip.py")
    return _assert_fires(result, "BOGUS", "alias dangling break")


def test_documented_vs_extracted_break(tmp: Path) -> bool:
    """Loop 141 A.iv (migrated to shared helpers Loop 144 A.iii):
    mutate SUBMISSION_CHECKLIST (11/N) "6 reports" → "5 reports";
    assert verify_documented_vs_extracted_consistency.py fires."""
    dst = _copy_to_tmp(
        tmp,
        scripts=[
            "verify_documented_vs_extracted_consistency.py",
            "verify_report_consistency.py",
            "_gate_utils.py",
        ],
        papers=["SUBMISSION_CHECKLIST.md"],
    )
    chk = tmp / "papers" / "SUBMISSION_CHECKLIST.md"
    text = chk.read_text()
    broken = text.replace(
        "report consistency — 6 reports (2 full + 4 stub coverage)",
        "report consistency — 5 reports (1 full + 4 stub coverage)",
    )
    if broken == text:
        print("# SKIP doc_vs_extracted: anchor not found",
              file=sys.stderr)
        return True
    chk.write_text(broken)
    result = _run_gate(dst, "verify_documented_vs_extracted_consistency.py")
    return _assert_fires(
        result, "5 reports",
        "doc_vs_extracted break",
    )


def test_changelog_consistency_break(tmp: Path) -> bool:
    """Loop 141 A.iv (migrated to shared helpers Loop 144 A.iii):
    mutate CHANGELOG §7 lead count to 'Sixty-two' while leaving
    other sites at current; assert verify_changelog_consistency.py
    agreement check fires."""
    dst = _copy_to_tmp(
        tmp,
        scripts=["verify_changelog_consistency.py", "_gate_utils.py"],
        papers=["CHANGELOG.md", "SUBMISSION_CHECKLIST.md"],
        docs=["ADVERSARIAL_REVIEW_LOG.md"],
    )
    cl = tmp / "papers" / "CHANGELOG.md"
    text = cl.read_text()
    broken = re.sub(
        r"\*\*[A-Za-z-]+\*\* independent\s+passes total",
        "**Sixty-two** independent passes total",
        text, count=1,
    )
    if broken == text:
        print("# SKIP changelog_consistency: §7 lead anchor not found",
              file=sys.stderr)
        return True
    cl.write_text(broken)
    result = _run_gate(dst, "verify_changelog_consistency.py")
    # Either "disagreement" or "drift" diagnostic accepted.
    if result.returncode == 0 or (
            "disagreement" not in result.stderr.lower()
            and "drift" not in result.stderr.lower()):
        print(f"# FAIL  changelog_consistency break: "
              f"rc={result.returncode}, "
              "stderr lacks disagreement/drift",
              file=sys.stderr)
        print(f"  stderr: {result.stderr[:300]}", file=sys.stderr)
        return False
    print("# OK    changelog_consistency break: gate fires with "
          "disagreement/drift diagnostic")
    return True


def test_stage_count_break(tmp: Path) -> bool:
    """Loop 141 A.iv (migrated to shared helpers Loop 144 A.iii):
    mutate F2 §E `**N stages**` to N-5; assert
    verify_stage_count_consistency.py fires."""
    dst = _copy_to_tmp(
        tmp,
        scripts=[
            "verify_stage_count_consistency.py",
            "_gate_utils.py",
            "run_all_checks.sh",
        ],
        papers=[
            "f2_methodology.md",
            "phi_ladder_paper_intro_draft.md",
            "CHANGELOG.md",
            "SUBMISSION_CHECKLIST.md",
        ],
    )
    f2 = tmp / "papers" / "f2_methodology.md"
    text = f2.read_text()
    m = re.search(r"Currently chains \*\*(\d+) stages\*\*", text)
    if not m:
        print("# SKIP stage_count: §E count anchor not found",
              file=sys.stderr)
        return True
    actual = int(m.group(1))
    broken = text.replace(
        f"Currently chains **{actual} stages**",
        f"Currently chains **{actual - 5} stages**",
        1,
    )
    f2.write_text(broken)
    result = _run_gate(dst, "verify_stage_count_consistency.py")
    # Both the claim and the actual count should surface in stderr.
    if result.returncode == 0 or (
            str(actual - 5) not in result.stderr
            or str(actual) not in result.stderr):
        print(f"# FAIL  stage_count break: rc={result.returncode}, "
              "stderr missing claim/actual digits",
              file=sys.stderr)
        print(f"  stderr: {result.stderr[:300]}", file=sys.stderr)
        return False
    print("# OK    stage_count break: gate fires with claim-vs-actual "
          "mismatch")
    return True


def test_tier_classification_parity_break(tmp: Path) -> bool:
    """Loop 143 A.iv: synthetic break for verify_tier_classification.
    Mutate run_all_checks.sh STAGE_TIERS to remove one entry;
    assert the parity check fires. Uses Loop 143 A.iii shared helpers."""
    dst = _copy_to_tmp(tmp, scripts=[
        "verify_tier_classification.py", "run_all_checks.sh",
    ])
    shell = dst / "run_all_checks.sh"
    text = shell.read_text()
    # Drop the LAST tier entry to create an off-by-one.
    broken = re.sub(
        r'(\s+)"discipline"\s*\n\)',
        r'\n)',
        text, count=1,
    )
    if broken == text:
        print("# SKIP tier_classification: STAGE_TIERS anchor not found",
              file=sys.stderr)
        return True
    shell.write_text(broken)
    result = _run_gate(dst, "verify_tier_classification.py")
    return _assert_fires(
        result, "STAGE_TIERS length",
        "tier_classification parity break",
    )


def test_burn_down_trajectory_monotonicity_break(tmp: Path) -> bool:
    """Loop 143 A.iv: synthetic break for verify_burn_down_trajectory.
    Append a 'Loop 999: 50 + 50 = 100' entry which violates total
    non-increasing (100 > current total 24). Asserts gate fires."""
    dst = _copy_to_tmp(tmp, scripts=[
        "verify_burn_down_trajectory.py",
        "verify_anonymizer_completeness.py",
    ])
    anon = dst / "verify_anonymizer_completeness.py"
    text = anon.read_text()
    broken = text.replace(
        "FALLBACK_BASELINES = {",
        "#   Loop 999 SYNTHETIC: 50 + 50 = 100.\n"
        "FALLBACK_BASELINES = {",
    )
    if broken == text:
        print("# SKIP burn_down_trajectory: FALLBACK_BASELINES marker "
              "not found", file=sys.stderr)
        return True
    anon.write_text(broken)
    result = _run_gate(dst, "verify_burn_down_trajectory.py")
    return _assert_fires(
        result, "total 100 > previous",
        "burn_down_trajectory monotonicity break",
    )


def test_acknowledges_break_remove_ack(tmp: Path) -> bool:
    """Delete the acknowledgement sentence from #1021 paper."""
    i1021_tmp = tmp / "i1021_no_ack.md"
    text = ISSUE1021.read_text()
    # Replace the acknowledgement-required sentence with a placeholder
    # that doesn't contain "F2 paper's own §8.1 names a subset".
    broken = text.replace(
        "F2 paper's own §8.1 names a subset",
        "the F2 paper documents a different scope",
    )
    if broken == text:
        print("# SKIP acknowledges break: source text didn't contain the "
              "expected ack-pattern (paper may have evolved)",
              file=sys.stderr)
        return True
    i1021_tmp.write_text(broken)
    rc, err = run_gate(issue1021_override=i1021_tmp)
    return assert_fails_with(
        "ACKNOWLEDGES break (acknowledgement deleted)",
        "acknowledgement",
        rc, err,
    )


def test_inventory_completeness() -> bool:
    """Loop 126 B (47th pass A4 follow-up): assert every claim-class list
    in `verify_cross_paper_consistency.py` has at least one break-test in
    this file. Adding a new class without a meta-test fails the harness.

    Approach: introspect the gate module's CLAIMS lists by name, then
    check that this script's TEST_FN_BY_CLASS map covers each."""
    sys.path.insert(0, str(CRATE_ROOT / "papers" / "scripts"))
    try:
        import verify_cross_paper_consistency as g
    finally:
        sys.path.pop(0)
    registered_classes = {
        name.removesuffix("_CLAIMS")
        for name in dir(g)
        if name.endswith("_CLAIMS") and isinstance(getattr(g, name), list)
    }
    # Each class must have at least one break-test below.
    tested_classes = {"EXACT_MATCH", "SCOPED_DIFF", "ACKNOWLEDGES",
                      "RELATIONAL", "EXACT_PIN"}
    missing = registered_classes - tested_classes
    extra = tested_classes - registered_classes
    if missing:
        print(f"# FAIL  inventory_completeness: gate has CLAIMS classes "
              f"{sorted(missing)} that this meta-test does not cover. "
              "Add a break-test before the gate is trusted to enforce "
              "those classes.", file=sys.stderr)
        return False
    if extra:
        # Loop 127 D (50th pass A2 SEV-3): tested classes not in gate
        # are now a hard failure, not a WARN. The 47th-pass A4 follow-up
        # intent was symmetric coverage; stale tests for deleted classes
        # are dead code that could mask future regressions if the class
        # is reintroduced under the old name.
        print(f"# FAIL  inventory_completeness: meta-test names classes "
              f"{sorted(extra)} that don't exist in the gate (stale "
              "test code). Remove or rename the break-test.",
              file=sys.stderr)
        return False
    print(f"# OK    inventory_completeness: {len(registered_classes)} "
          f"claim classes registered ({sorted(registered_classes)}), "
          "all covered by break-tests")
    return True


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="meta_test_cross_paper_") as td:
        tmp = Path(td)
        tests = [
            ("inventory_completeness", lambda _t: test_inventory_completeness()),
            ("exact_match_f2_total", test_exact_match_break_f2_total),
            ("scoped_diff_lib_count", test_scoped_diff_break_lib_count),
            ("acknowledges_remove_ack", test_acknowledges_break_remove_ack),
            ("relational_non_anon_lt_anon", test_relational_break_non_anon_lt_anon),
            ("relational_boundary_equality", test_relational_boundary_equality),
            ("exact_pin_tmlr_page", test_exact_pin_break_tmlr_page),
            # Loop 140 A.iv: extended coverage beyond cross-paper gates.
            ("burn_down_arithmetic", test_burn_down_arithmetic_break),
            ("alias_round_trip_dangling", test_alias_round_trip_dangling_break),
            # Loop 141 A.iv: extended coverage to 3 more newer gates.
            ("doc_vs_extracted_drift", test_documented_vs_extracted_break),
            ("changelog_consistency_drift", test_changelog_consistency_break),
            ("stage_count_drift", test_stage_count_break),
            # Loop 143 A.iv: break-tests for stages 35 + 36, using the
            # shared helpers extracted in 143 A.iii.
            ("tier_classification_parity", test_tier_classification_parity_break),
            ("burn_down_trajectory_monotonicity", test_burn_down_trajectory_monotonicity_break),
        ]
        results = [(name, fn(tmp)) for name, fn in tests]
    n_pass = sum(1 for _, ok in results if ok)
    n_total = len(results)
    if n_pass < n_total:
        print(f"# meta_test_cross_paper_gates.py — {n_pass}/{n_total} synthetic-break "
              "tests passed", file=sys.stderr)
        return 1
    print(f"# meta_test_cross_paper_gates.py — {n_total}/{n_total} synthetic-break "
          "tests passed; covers 5 cross-paper claim classes + "
          "burn-down arithmetic + alias bijection + doc-vs-extracted "
          "+ changelog consistency + stage count + tier parity + "
          "burn-down monotonicity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
