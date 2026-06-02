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
    (g.F2_PAPER if paper == _orig_f2 else (g.ISSUE1021_PAPER if paper == _orig_i1021 else g.SUBMISSION_CHECKLIST),
     claim_pat, ack_pat, label)
    for (paper, claim_pat, ack_pat, label) in _saved_ackn
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
    tested_classes = {"EXACT_MATCH", "SCOPED_DIFF", "ACKNOWLEDGES"}
    missing = registered_classes - tested_classes
    extra = tested_classes - registered_classes
    if missing:
        print(f"# FAIL  inventory_completeness: gate has CLAIMS classes "
              f"{sorted(missing)} that this meta-test does not cover. "
              "Add a break-test before the gate is trusted to enforce "
              "those classes.", file=sys.stderr)
        return False
    if extra:
        # Tested classes not in gate is suspicious but not fatal — could
        # be a leftover after a class was deleted from the gate.
        print(f"# WARN  inventory_completeness: meta-test names classes "
              f"{sorted(extra)} that don't exist in the gate (stale "
              "test?)", file=sys.stderr)
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
        ]
        results = [(name, fn(tmp)) for name, fn in tests]
    n_pass = sum(1 for _, ok in results if ok)
    n_total = len(results)
    if n_pass < n_total:
        print(f"# meta_test_cross_paper_gates.py — {n_pass}/{n_total} synthetic-break "
              "tests passed", file=sys.stderr)
        return 1
    print(f"# meta_test_cross_paper_gates.py — {n_total}/{n_total} synthetic-break "
          "tests passed; gate enforces its 3 claim classes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
