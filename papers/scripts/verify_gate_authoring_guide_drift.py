#!/usr/bin/env python3
"""verify_gate_authoring_guide_drift.py — assert the discipline
documented in GATE_AUTHORING_GUIDE.md still matches live code.

Loop 144 B operationalizes "the guide tells contributors X but the
code does Y" drift. The guide cites several `verify_*.py` filenames
and a canonical breadcrumb label-class regex. This gate:

  (a) Every `verify_*.py` filename mentioned in the guide must
      exist on disk in papers/scripts/.
  (b) The breadcrumb label-class snippet in the guide (currently in
      §6 "Stage 6: breadcrumb discipline") must match the live regex
      in `verify_burn_down_history.py:_ENTRY_RE`.

Both checks defend against the class the 66th-pass #6 caught: the
guide said "avoid commas" but the regex had been extended to accept
them; the contradiction was invisible until reviewed.

Usage: papers/scripts/verify_gate_authoring_guide_drift.py

Exit 0 if guide and code agree; 1 on any drift.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
GUIDE = CRATE_ROOT / "papers" / "scripts" / "GATE_AUTHORING_GUIDE.md"
BURN_DOWN_HISTORY = (
    CRATE_ROOT / "papers" / "scripts" / "verify_burn_down_history.py"
)
SCRIPTS_DIR = CRATE_ROOT / "papers" / "scripts"


# Match `verify_*.py` filenames in the guide (likely backticked).
_GUIDE_FILE_RE = re.compile(r"`(verify_[A-Za-z0-9_]+\.py)`")

# Match the label class string in the guide, e.g.,
# "The label class accepts `[A-Za-z0-9.()\s§#+/\-—–',]`".
_GUIDE_LABEL_CLASS_RE = re.compile(
    r"label class accepts\s*`\[([^`]+)\]`"
)

# Match the live regex in verify_burn_down_history.py. The class
# itself contains escaped brackets `\[\]`, so we extract by
# anchoring on the surrounding `\s*[` opener and `]*?:` closer
# (non-greedy capture of everything between).
_LIVE_LABEL_CLASS_RE = re.compile(
    r'r"Loop\\s\+\(\\d\+\)\\s\*\[(.*?)\]\*\?:', re.DOTALL,
)


def parse_guide_files() -> set[str] | str:
    if not GUIDE.exists():
        return f"missing {GUIDE.relative_to(CRATE_ROOT)}"
    return set(_GUIDE_FILE_RE.findall(GUIDE.read_text()))


def parse_guide_label_class() -> str | None:
    if not GUIDE.exists():
        return None
    m = _GUIDE_LABEL_CLASS_RE.search(GUIDE.read_text())
    return m.group(1) if m else None


def parse_live_label_class() -> str | None:
    if not BURN_DOWN_HISTORY.exists():
        return None
    m = _LIVE_LABEL_CLASS_RE.search(BURN_DOWN_HISTORY.read_text())
    return m.group(1) if m else None


def main() -> int:
    mismatches: list[str] = []

    # (a) Existence of all guide-cited filenames.
    cited = parse_guide_files()
    if isinstance(cited, str):
        print(f"# FAIL  parse guide: {cited}", file=sys.stderr)
        return 1
    missing: list[str] = []
    for name in cited:
        if not (SCRIPTS_DIR / name).exists():
            missing.append(name)
    if missing:
        mismatches.append(
            f"guide cites {len(missing)} verify_*.py file(s) that "
            f"don't exist: {sorted(missing)}. Either the file was "
            "renamed/removed and the guide wasn't updated, or the "
            "filename is wrong.")
    else:
        print(f"# OK    all {len(cited)} guide-cited verify_*.py "
              "filename(s) exist on disk")

    # (b) Label-class agreement.
    guide_class = parse_guide_label_class()
    live_class = parse_live_label_class()
    if guide_class is None:
        mismatches.append(
            "guide §6 'label class accepts `[...]`' anchor not "
            "found; the guide may have been reworded.")
    elif live_class is None:
        mismatches.append(
            "verify_burn_down_history.py _ENTRY_RE label-class "
            "anchor not found; the code may have been reformatted.")
    elif guide_class != live_class:
        mismatches.append(
            f"label-class drift: guide says `[{guide_class}]` but "
            f"code has `[{live_class}]`. Update one to match the "
            "other — discipline-by-comment is fragile when the "
            "regex extends silently.")
    else:
        print(f"# OK    label class agrees: `[{guide_class}]`")

    if mismatches:
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(f"# verify_gate_authoring_guide_drift.py — "
              f"{len(mismatches)} drift(s) between guide and code",
              file=sys.stderr)
        return 1
    print(f"# verify_gate_authoring_guide_drift.py — guide aligned "
          "with live code on cited filenames + label-class regex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
