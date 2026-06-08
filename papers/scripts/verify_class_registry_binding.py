#!/usr/bin/env python3
"""verify_class_registry_binding.py — gate the §1 sub-bullet class-
class enumeration against the actual `_CLAIMS` lists in
verify_cross_paper_consistency.py.

Loop 134 A operationalizes the SEV the 55th-pass #14 catch surfaced:
the §1 sub-bullet (13/M) "N claims (1 EXACT + 2 SCOPED + 1 ACKN +
4 RELATIONAL + 3 EXACT_PIN)" names labels and counts. The
verify_cardinality_arithmetic.py gate (Loop 133 B) checks the sum
holds, but doesn't bind the *labels* to the actual class registry.
A drift like "11 claims (1 EXACT + 5 SCOPED + 1 ACKN + 4 RELATIONAL)"
(stale composition) sums to 11 and arithmetic-passes; this gate
catches the label-vs-registry drift.

Approach:
  1. Parse `papers/SUBMISSION_CHECKLIST.md` §1 (13/M) sub-bullet:
     extract pairs of (count, label) like `(1 EXACT)`, `(2 SCOPED)`,
     `(1 ACKN)`, `(4 RELATIONAL)`, `(3 EXACT_PIN)`.
  2. Import `verify_cross_paper_consistency` and discover `*_CLAIMS`
     attributes that are lists.
  3. Normalize §1 labels to gate-attribute names:
       EXACT      → EXACT_MATCH
       SCOPED     → SCOPED_DIFF
       ACKN       → ACKNOWLEDGES
       RELATIONAL → RELATIONAL
       EXACT_PIN  → EXACT_PIN
       PIN        → EXACT_PIN (legacy alias)
  4. Assert each (count, normalized_label): count == len(*_CLAIMS).
  5. Also assert no _CLAIMS list is unrepresented in §1 (registry
     completeness).

Usage: papers/scripts/verify_class_registry_binding.py

Exit 0 on perfect binding; 1 on any count or coverage drift.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_CHECKLIST = CRATE_ROOT / "papers" / "SUBMISSION_CHECKLIST.md"
GATE_MODULE = CRATE_ROOT / "papers" / "scripts" / "verify_cross_paper_consistency.py"


# Loop 134 — 57th-pass SEV-4 fix #11: LABEL_NORMALIZATION moved to
# `verify_cross_paper_consistency.py` as `CLASS_LABEL_ALIASES` (single
# source of truth). Imported lazily in main() because the gate module
# load is the same one parsed for *_CLAIMS attributes.
LABEL_NORMALIZATION: dict[str, str] | None = None  # lazy-loaded


def parse_section1_claim_classes() -> tuple[list[tuple[int, str]], int] | str:
    """Return ((count, label) pairs, line_no) or error string."""
    if not SUBMISSION_CHECKLIST.exists():
        return f"missing {SUBMISSION_CHECKLIST.relative_to(CRATE_ROOT)}"
    text = SUBMISSION_CHECKLIST.read_text()
    m = re.search(r"(\d+) claims \(([^)]+)\)", text)
    if not m:
        return ("§1 'N claims (...)' anchor not found in "
                f"{SUBMISSION_CHECKLIST.name}")
    body = m.group(2)
    line_no = text[:m.start()].count("\n") + 1
    # Each part is "<digit> <LABEL>" possibly separated by "+", ",", or whitespace.
    pairs = re.findall(r"(\d+)\s+([A-Z][A-Z_]*)", body)
    if not pairs:
        return (f"§1 enumeration body {body!r} has no (count, LABEL) "
                "pairs in the canonical 'N LABEL' form")
    return [(int(c), lab) for c, lab in pairs], line_no


def load_gate_classes() -> tuple[dict[str, int], dict[str, str]] | str:
    """Import verify_cross_paper_consistency and return ({prefix:
    list_length}, CLASS_LABEL_ALIASES) for each discovered *_CLAIMS
    attribute. The aliases dict is the canonical source of label
    normalization (Loop 134 SEV-4 fix #11).

    Loop 137 A.iv: import via shared _gate_utils.import_gate helper
    so the traceback-emission fix (Loop 135 #7) stays in one place."""
    sys.path.insert(0, str(GATE_MODULE.parent))
    try:
        from _gate_utils import import_gate
    finally:
        sys.path.pop(0)
    mod = import_gate(GATE_MODULE.name)
    if mod is None:
        return f"failed to import {GATE_MODULE.name} (see traceback above)"
    classes: dict[str, int] = {}
    for name in dir(mod):
        if name.endswith("_CLAIMS"):
            v = getattr(mod, name)
            if isinstance(v, list):
                classes[name.removesuffix("_CLAIMS")] = len(v)
    aliases = getattr(mod, "CLASS_LABEL_ALIASES", None)
    if not isinstance(aliases, dict):
        return ("gate module lacks CLASS_LABEL_ALIASES dict; cannot "
                "normalize §1 labels. Loop 134 moved this dict into "
                "verify_cross_paper_consistency.py as the single source.")
    return classes, aliases


def main() -> int:
    s1 = parse_section1_claim_classes()
    if isinstance(s1, str):
        print(f"# FAIL  parse  {s1}", file=sys.stderr)
        return 1
    pairs, line_no = s1
    loaded = load_gate_classes()
    if isinstance(loaded, str):
        print(f"# FAIL  import {loaded}", file=sys.stderr)
        return 1
    classes, aliases = loaded
    print(f"# verify_class_registry_binding.py — §1:{line_no} "
          f"has {len(pairs)} (count, LABEL) pair(s); gate registry has "
          f"{len(classes)} class(es): {sorted(classes)}")

    mismatches: list[str] = []
    seen_classes: set[str] = set()
    for count, label in pairs:
        normalized = aliases.get(label)
        if normalized is None:
            mismatches.append(
                f"§1:{line_no}: label '{label}' has no normalization "
                "rule; either add to CLASS_LABEL_ALIASES (in "
                "verify_cross_paper_consistency.py) or rename the §1 "
                "label to a known form ({})".format(sorted(aliases)))
            continue
        if normalized not in classes:
            mismatches.append(
                f"§1:{line_no}: label '{label}' → '{normalized}' has "
                f"no matching *_CLAIMS list in the gate "
                f"(known: {sorted(classes)})")
            continue
        actual = classes[normalized]
        if count != actual:
            mismatches.append(
                f"§1:{line_no}: label '{label}' claims {count} but "
                f"{normalized}_CLAIMS list has {actual} entries")
        else:
            print(f"# OK    {label} → {normalized}: §1 claims {count} "
                  f"= len({normalized}_CLAIMS)")
        seen_classes.add(normalized)

    # Registry completeness: every *_CLAIMS list should be named in §1.
    unrepresented = set(classes) - seen_classes
    if unrepresented:
        mismatches.append(
            f"§1:{line_no}: gate has *_CLAIMS lists "
            f"{sorted(unrepresented)} that are NOT named in §1 "
            "enumeration. Add to §1 or remove from gate.")

    if mismatches:
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(f"# verify_class_registry_binding.py — {len(mismatches)} "
              "binding drift(s) between §1 and gate registry",
              file=sys.stderr)
        return 1
    print(f"# verify_class_registry_binding.py — all "
          f"{len(seen_classes)} gate class(es) bound to §1 enumeration, "
          "0 drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
