#!/usr/bin/env python3
"""verify_alias_round_trip.py — assert CLASS_LABEL_ALIASES is
bijective on the gate's actual class names.

Loop 138 B closes the 60th-pass deferred SEV-4 #6 helper-extraction
concern: an aliases dict that's only checked forward (alias →
canonical) lets dead aliases (alias pointing to a removed class)
and unaliased canonical names (a new class with no aliases) drift
silently.

This gate asserts:
  Forward: every alias key resolves to a canonical name that has a
           matching *_CLAIMS list in verify_cross_paper_consistency.
  Inverse: every *_CLAIMS list has at least one alias pointing to
           its canonical name (so paper authors can reference it
           from §1 prose).

Usage: papers/scripts/verify_alias_round_trip.py

Exit 0 on bijective coverage; 1 on any dead alias or unaliased
canonical.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
try:
    from _gate_utils import import_gate
finally:
    sys.path.pop(0)


def main() -> int:
    mod = import_gate("verify_cross_paper_consistency.py")
    if mod is None:
        print("# FAIL  load verify_cross_paper_consistency.py "
              "(see traceback above)", file=sys.stderr)
        return 1
    aliases = getattr(mod, "CLASS_LABEL_ALIASES", None)
    if not isinstance(aliases, dict):
        print("# FAIL  CLASS_LABEL_ALIASES not a dict in gate module",
              file=sys.stderr)
        return 1

    # Discover _CLAIMS list names.
    classes: set[str] = set()
    for name in dir(mod):
        if name.endswith("_CLAIMS"):
            v = getattr(mod, name)
            if isinstance(v, list):
                classes.add(name.removesuffix("_CLAIMS"))

    print(f"# verify_alias_round_trip.py — {len(aliases)} aliases, "
          f"{len(classes)} *_CLAIMS classes: {sorted(classes)}")

    mismatches: list[str] = []

    # Forward: every alias resolves to a known class.
    for short, canonical in aliases.items():
        if canonical not in classes:
            mismatches.append(
                f"alias '{short}' → '{canonical}' but no "
                f"{canonical}_CLAIMS list exists in the gate "
                "(dead alias).")
        else:
            print(f"# OK    forward: '{short}' → '{canonical}' "
                  "(has matching _CLAIMS list)")

    # Inverse: every class has at least one alias mapping to it.
    reverse: dict[str, list[str]] = {}
    for short, canonical in aliases.items():
        reverse.setdefault(canonical, []).append(short)
    for cls in sorted(classes):
        if cls not in reverse:
            mismatches.append(
                f"*_CLAIMS class '{cls}' has no alias in "
                "CLASS_LABEL_ALIASES — paper authors cannot reference "
                "it from §1 prose without adding an alias entry.")
        else:
            short_list = reverse[cls]
            print(f"# OK    inverse: '{cls}' ← {{{', '.join(repr(s) for s in short_list)}}}")

    if mismatches:
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(f"# verify_alias_round_trip.py — {len(mismatches)} "
              "bijection violation(s) in CLASS_LABEL_ALIASES",
              file=sys.stderr)
        return 1
    print(f"# verify_alias_round_trip.py — bijection holds: "
          f"{len(aliases)} aliases cover {len(classes)} classes "
          "in both directions, 0 drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
