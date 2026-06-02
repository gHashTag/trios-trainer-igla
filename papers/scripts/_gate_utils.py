"""_gate_utils.py — shared helpers for paper-side CI gates.

Loop 137 A.iv extracts duplicated functions from sister gates so future
closures (e.g., the 58th-pass #7 traceback emission) land in one place
instead of N parallel copies. The 59th-pass #1 surfaced that Loop 135
shipped a regression by copy-pasting an old `_import_gate` without the
fresh traceback fix.

Currently exposes:
  import_gate(name)  — load a sibling gate module by filename; emits
                       traceback to stderr on import failure.

Future candidates for extraction (left in their original homes for
now to avoid touching too much in one loop):
  - _parse_subbullet (verify_class_registry_binding + doc_vs_extracted)
  - _to_int English-numeral helper (changelog + stage_count)
  - _load_baselines (anonymizer)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import types

CRATE_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = CRATE_ROOT / "papers" / "scripts"


def import_gate(name: str) -> types.ModuleType | None:
    """Load a sibling gate module by filename (e.g., 'verify_foo.py').

    Returns the loaded module on success, None on any load failure.
    Emits the traceback to stderr before returning None so syntax
    errors surface with full context — closes the 58th-pass #7 SEV-3
    that the 59th-pass #1 caught a regression of."""
    path = SCRIPTS_DIR / name
    if not path.exists():
        return None
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
    return mod
