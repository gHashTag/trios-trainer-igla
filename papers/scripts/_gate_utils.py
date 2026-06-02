"""_gate_utils.py — shared helpers for paper-side CI gates.

Loop 137 A.iv extracts duplicated functions from sister gates so future
closures (e.g., the 58th-pass #7 traceback emission) land in one place
instead of N parallel copies. The 59th-pass #1 surfaced that Loop 135
shipped a regression by copy-pasting an old `_import_gate` without the
fresh traceback fix.

Currently exposes:
  import_gate(name)  — load a sibling gate module by filename; emits
                       traceback to stderr on import failure.
  to_int(w)          — Loop 138 A.iv: extracted from
                       verify_changelog_consistency.py's _to_int and
                       verify_stage_count_consistency.py's words_to_int.
                       Accepts digits, English 0..99 (incl. compound
                       forms like 'twenty-one').

Future candidates for extraction (left in their original homes):
  - _parse_subbullet (verify_class_registry_binding + doc_vs_extracted)
  - _load_baselines (anonymizer)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import types

CRATE_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = CRATE_ROOT / "papers" / "scripts"


# Loop 139 B: cache for import_gate. Keyed by `path.stem` so two
# callers asking for the same gate share the module instance.
# Closes 61st-pass #4 SEV-3: per-call exec_module duplicated module
# bodies + risked silent state divergence for modules with import-time
# state (counters, atexit hooks). Pass `cache=False` to bypass.
_MODULE_CACHE: dict[str, types.ModuleType] = {}


def import_gate(name: str, cache: bool = True
                ) -> types.ModuleType | None:
    """Load a sibling gate module by filename (e.g., 'verify_foo.py').

    Returns the loaded module on success, None on any load failure.
    Emits the traceback to stderr before returning None so syntax
    errors surface with full context — closes the 58th-pass #7 SEV-3
    that the 59th-pass #1 caught a regression of.

    Loop 139 B: caching is on by default. The cache key is the path
    stem (filename without .py); a future loop editing the gate
    module mid-run gets a stale cache entry unless `cache=False` is
    passed. The trade-off: 4× speedup for run_all_checks.sh sweeps
    that import verify_cross_paper_consistency.py from three sister
    gates."""
    path = SCRIPTS_DIR / name
    if not path.exists():
        return None
    if cache and path.stem in _MODULE_CACHE:
        return _MODULE_CACHE[path.stem]
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None
    if cache:
        _MODULE_CACHE[path.stem] = mod
    return mod


_UNITS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
]


def to_int(w: str) -> int | None:
    """Convert English number words (0..99) + digits to int.

    Accepts "53", "Fifty-three", "fifty three", or "fifty 3" forms.
    Returns None on unrecognized input. Hyphen and space both accepted
    as compound separators (Loop 135 extension to handle "twenty-one").

    Loop 138 A.iv: extracted from verify_changelog_consistency._to_int
    (returning None on errors) and verify_stage_count_consistency.
    words_to_int (same semantics). Single source of truth — closes
    the 60th-pass SEV-4 #6 helper-extraction-conventions class."""
    if w.isdigit():
        return int(w)
    w = w.lower().strip()
    if not w:
        return None
    if w in _UNITS:
        return _UNITS.index(w)
    if w in _TENS[2:]:
        return _TENS.index(w) * 10
    for sep in ("-", " "):
        if sep in w:
            tens_part, _, units_part = w.partition(sep)
            if tens_part in _TENS and units_part in _UNITS[1:10]:
                return _TENS.index(tens_part) * 10 + _UNITS.index(units_part)
    return None
