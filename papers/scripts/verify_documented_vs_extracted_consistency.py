#!/usr/bin/env python3
"""verify_documented_vs_extracted_consistency.py — bind §1 sub-bullet
description metadata to live verifier-source-extracted counts.

Loop 135 C generalizes the class-registry-binding gate (Loop 134 A.iv,
stage 27) which binds the (13/27) sub-bullet to verify_cross_paper_
consistency.py. This gate adds bindings for other §1 sub-bullets
whose descriptions carry numeric metadata that drifts when the
underlying verifier changes but the §1 prose isn't updated in
lock-step.

Each binding is (sub_bullet_kid, description_regex, extractor_fn,
label). The extractor returns the live count by reading the
corresponding verifier source / running the verifier and parsing
output.

Active bindings (Loop 136 grown 2 → 4):
  (11/N) report consistency — "6 reports (2 full + 4 stub coverage)"
          → REPORT_SPECS + EXISTS_STUBS lengths in
          verify_report_consistency.py.
  (12/N) stage count consistency — "5 counts + 2 decompositions +
          1 derived" → CLAIMS + DECOMPOSITION_CLAIMS + DERIVED_CLAIMS
          list lengths in verify_stage_count_consistency.py.
  (14/N) cross-paper gate meta-test — "7 synthetic-break tests PASS
          (5 classes covered)" → count of `def test_*` functions in
          meta_test_cross_paper_gates.py.
  (24/N) anonymizer completeness — "48 at Loop 136" → sum of
          baselines in papers/scripts/anonymizer_baseline.json.

Usage: papers/scripts/verify_documented_vs_extracted_consistency.py

Exit 0 on agreement; 1 on any documented-vs-extracted drift.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_CHECKLIST = CRATE_ROOT / "papers" / "SUBMISSION_CHECKLIST.md"
SCRIPTS_DIR = CRATE_ROOT / "papers" / "scripts"


def _parse_subbullet(k: int, description_regex: str
                     ) -> tuple[tuple[int, ...], int] | str:
    """Find SUBMISSION_CHECKLIST §1 sub-bullet `(k/N) <name> — <desc>`
    and extract integer groups from <desc> via description_regex.

    Returns ((integer_tuple), line_no) or error string."""
    if not SUBMISSION_CHECKLIST.exists():
        return f"missing {SUBMISSION_CHECKLIST.relative_to(CRATE_ROOT)}"
    text = SUBMISSION_CHECKLIST.read_text()
    sec = re.search(
        r"^##\s*1\.\s+CI gates\b.*?(?=^##\s|\Z)",
        text, re.MULTILINE | re.DOTALL,
    )
    if not sec:
        return "§1 'CI gates' section not found"
    sec_text = sec.group(0)
    sec_offset = sec.start()
    # Multi-line tolerant: capture from em-dash up to next `- [ ]`
    # checkbox (start of next sub-bullet) or blank line. The §1 prose
    # uses 2-space indent continuations for wrapped lines.
    bullet_re = re.compile(
        rf"^\s*-\s*\[\s*[ xX]\s*\]\s*\({k}/\d+\)\s+(.+?)\s+—\s+(.+?)(?=^\s*-\s*\[|\n\n|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    m = bullet_re.search(sec_text)
    if not m:
        return f"sub-bullet ({k}/N) not found in §1"
    # Collapse multi-line description into single line for regex matching.
    description = re.sub(r"\s+", " ", m.group(2)).strip()
    line_no = (
        text[:sec_offset + m.start()].count("\n") + 1
    )
    dm = re.search(description_regex, description)
    if not dm:
        return (f"description {description!r} did not match "
                f"{description_regex!r}")
    nums = tuple(int(g) for g in dm.groups())
    return nums, line_no


def _import_gate(name: str):
    path = SCRIPTS_DIR / name
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        path.stem, str(path),
    )
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # Loop 136 — 59th-pass SEV-3 fix #1: emit traceback to stderr
        # before returning None. The 58th-pass closure #7 added this
        # to verify_class_registry_binding.py but the duplicated
        # helper here was missed, silently swallowing import errors.
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None
    return mod


def extract_report_consistency_counts() -> tuple[int, int] | str:
    """Return (n_full, n_stub) from verify_report_consistency.py."""
    mod = _import_gate("verify_report_consistency.py")
    if mod is None:
        return "could not import verify_report_consistency.py"
    full = getattr(mod, "REPORT_SPECS", None)
    stubs = getattr(mod, "EXISTS_STUBS", None)
    if not isinstance(full, list) or not isinstance(stubs, list):
        return "REPORT_SPECS or EXISTS_STUBS missing/non-list"
    return len(full), len(stubs)


def extract_meta_test_count() -> int | str:
    """Count `def test_*` functions in meta_test_cross_paper_gates.py."""
    path = SCRIPTS_DIR / "meta_test_cross_paper_gates.py"
    if not path.exists():
        return "meta_test_cross_paper_gates.py missing"
    text = path.read_text()
    # Match top-level `def test_<name>(` at column 0.
    matches = re.findall(r"^def (test_[A-Za-z0-9_]+)\(", text, re.MULTILINE)
    return len(matches)


def extract_stage_count_counts() -> tuple[int, int, int] | str:
    """Return (n_counts, n_decomp, n_derived) from
    verify_stage_count_consistency.py CLAIMS / DECOMPOSITION_CLAIMS /
    DERIVED_CLAIMS list lengths."""
    mod = _import_gate("verify_stage_count_consistency.py")
    if mod is None:
        return "could not import verify_stage_count_consistency.py"
    claims = getattr(mod, "CLAIMS", None)
    decomp = getattr(mod, "DECOMPOSITION_CLAIMS", None)
    derived = getattr(mod, "DERIVED_CLAIMS", None)
    for name, v in (("CLAIMS", claims),
                    ("DECOMPOSITION_CLAIMS", decomp),
                    ("DERIVED_CLAIMS", derived)):
        if not isinstance(v, list):
            return f"{name} missing/non-list in verify_stage_count_consistency.py"
    return len(claims), len(decomp), len(derived)


def extract_anonymizer_total() -> int | str:
    """Return total bare-anchor count across all SCAN_TARGETS by
    reading the JSON sidecar. Operates from the live data, not from
    re-running the scanner — the gate compares to §1 anyway."""
    sidecar = SCRIPTS_DIR / "anonymizer_baseline.json"
    if not sidecar.exists():
        return f"sidecar {sidecar.name} missing"
    import json
    try:
        data = json.loads(sidecar.read_text())
    except json.JSONDecodeError as e:
        return f"sidecar JSON parse error: {e}"
    bl = data.get("baselines")
    if not isinstance(bl, dict):
        return "sidecar lacks 'baselines' dict"
    total = 0
    for k, v in bl.items():
        if isinstance(v, int) and v >= 0:
            total += v
    return total


def main() -> int:
    mismatches: list[str] = []

    # Binding 1: (11/27) report consistency — "6 reports (2 full + 4 stub coverage)"
    s1 = _parse_subbullet(
        11,
        r"(\d+) reports \((\d+) full \+ (\d+) stub",
    )
    if isinstance(s1, str):
        mismatches.append(f"(11/N): {s1}")
    else:
        (claimed_total, claimed_full, claimed_stub), line_no = s1
        actual = extract_report_consistency_counts()
        if isinstance(actual, str):
            mismatches.append(f"(11/N) report-consistency import: {actual}")
        else:
            actual_full, actual_stub = actual
            actual_total = actual_full + actual_stub
            if (claimed_full, claimed_stub, claimed_total) != \
               (actual_full, actual_stub, actual_total):
                mismatches.append(
                    f"SUBMISSION_CHECKLIST:{line_no}: (11/N) report "
                    f"consistency claims {claimed_total} reports "
                    f"({claimed_full} full + {claimed_stub} stub) "
                    f"but verify_report_consistency.py has "
                    f"{actual_full} REPORT_SPECS + {actual_stub} "
                    f"EXISTS_STUBS = {actual_total} total.")
            else:
                print(f"# OK    §1:{line_no} (11/N) report consistency: "
                      f"{claimed_full} full + {claimed_stub} stub = "
                      f"{claimed_total} (matches verify_report_consistency.py)")

    # Binding 2: (14/27) meta-test — "7 synthetic-break tests PASS (5 classes covered)"
    s2 = _parse_subbullet(
        14,
        r"(\d+) synthetic-break tests PASS \((\d+) classes",
    )
    if isinstance(s2, str):
        mismatches.append(f"(14/N): {s2}")
    else:
        (claimed_tests, claimed_classes), line_no = s2
        actual_tests = extract_meta_test_count()
        if isinstance(actual_tests, str):
            mismatches.append(f"(14/N) meta-test extract: {actual_tests}")
        else:
            # The meta-test count includes inventory_completeness which
            # is a class-coverage check, not a per-class break-test. The
            # §1 sub-bullet's "7 synthetic-break tests" counts all
            # def test_* including inventory. Direct compare.
            if claimed_tests != actual_tests:
                mismatches.append(
                    f"SUBMISSION_CHECKLIST:{line_no}: (14/N) meta-test "
                    f"claims {claimed_tests} synthetic-break tests but "
                    f"meta_test_cross_paper_gates.py has {actual_tests} "
                    f"def test_* functions.")
            else:
                print(f"# OK    §1:{line_no} (14/N) meta-test: "
                      f"{claimed_tests} tests = "
                      f"{actual_tests} def test_* (matches)")

    # Binding 3 (Loop 136 A.iv): (12/N) stage count — "5 counts + 2
    # decompositions + 1 derived" → CLAIMS + DECOMPOSITION_CLAIMS +
    # DERIVED_CLAIMS list lengths in verify_stage_count_consistency.py.
    s3 = _parse_subbullet(
        12,
        r"(\d+) counts \+ (\d+) decompositions \+ (\d+) derived",
    )
    if isinstance(s3, str):
        mismatches.append(f"(12/N): {s3}")
    else:
        (cl_counts, cl_decomp, cl_derived), line_no = s3
        actual = extract_stage_count_counts()
        if isinstance(actual, str):
            mismatches.append(f"(12/N) stage-count extract: {actual}")
        else:
            ac_counts, ac_decomp, ac_derived = actual
            if (cl_counts, cl_decomp, cl_derived) != \
               (ac_counts, ac_decomp, ac_derived):
                mismatches.append(
                    f"SUBMISSION_CHECKLIST:{line_no}: (12/N) stage "
                    f"count claims {cl_counts} counts + {cl_decomp} "
                    f"decompositions + {cl_derived} derived but "
                    f"verify_stage_count_consistency.py has "
                    f"{ac_counts} CLAIMS + {ac_decomp} "
                    f"DECOMPOSITION_CLAIMS + {ac_derived} DERIVED_CLAIMS.")
            else:
                print(f"# OK    §1:{line_no} (12/N) stage count: "
                      f"{cl_counts}+{cl_decomp}+{cl_derived} = "
                      f"{ac_counts}+{ac_decomp}+{ac_derived} (matches)")

    # Binding 4 (Loop 136 A.iv): (24/N) anonymizer — "X at Loop Y"
    # leading number == sum of baselines in anonymizer_baseline.json.
    s4 = _parse_subbullet(
        24,
        r"(\d+) at Loop \d+",
    )
    if isinstance(s4, str):
        mismatches.append(f"(24/N): {s4}")
    else:
        (claimed_total,), line_no = s4
        actual_total = extract_anonymizer_total()
        if isinstance(actual_total, str):
            mismatches.append(f"(24/N) anonymizer extract: {actual_total}")
        elif claimed_total != actual_total:
            mismatches.append(
                f"SUBMISSION_CHECKLIST:{line_no}: (24/N) anonymizer "
                f"ratchet claims {claimed_total} but sidecar sums to "
                f"{actual_total}.")
        else:
            print(f"# OK    §1:{line_no} (24/N) anonymizer ratchet: "
                  f"{claimed_total} = sum(sidecar baselines)")

    if mismatches:
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(f"# verify_documented_vs_extracted_consistency.py — "
              f"{len(mismatches)} drift(s) between §1 description "
              "metadata and verifier source state",
              file=sys.stderr)
        return 1
    # Loop 136 — 59th-pass SEV-5 fix #19: name the gated file in the
    # OK-path summary so a reader doesn't have to inspect the source
    # to find which file was checked.
    print(f"# verify_documented_vs_extracted_consistency.py — 4/4 "
          f"documented-vs-extracted binding(s) verified against "
          f"{SUBMISSION_CHECKLIST.relative_to(CRATE_ROOT)} §1, 0 drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
