#!/usr/bin/env python3
"""verify_module_cache_consistency.py — assert _gate_utils.import_gate's
caching behavior holds.

Loop 139 B operationalizes the 61st-pass #4 SEV-3 concern: per-call
`exec_module` was re-executing gate bodies and risked silent state
divergence. Loop 139 B added the `cache=True` default; this gate
asserts the cache contract:

  (a) Two calls to `import_gate('foo')` with cache=True return the
      SAME module instance (id equality).
  (b) Two calls with cache=False return DIFFERENT instances.
  (c) Mixing cache=True after cache=False does NOT poison the cache
      (the cache=False call doesn't write).

A regression that disabled caching (e.g., removed the `_MODULE_CACHE`
lookup) would fail (a). A regression that always cached would fail (b).

Usage: papers/scripts/verify_module_cache_consistency.py

Exit 0 if cache contract holds; 1 on any regression.
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
    # Use verify_cross_paper_consistency as the canary — it's the
    # heaviest gate module so caching has the biggest payoff. Any
    # module that exists in papers/scripts/ would work.
    target = "verify_cross_paper_consistency.py"

    mismatches: list[str] = []

    # (a) Two cached calls → same instance.
    a1 = import_gate(target, cache=True)
    a2 = import_gate(target, cache=True)
    if a1 is None or a2 is None:
        mismatches.append(
            f"baseline import failed for {target} — cannot test "
            "cache contract")
    elif id(a1) != id(a2):
        mismatches.append(
            f"cache=True regression: two import_gate('{target}', "
            f"cache=True) calls returned DIFFERENT instances "
            f"(id={id(a1)} vs id={id(a2)}). Cache lookup likely "
            "broken.")
    else:
        print(f"# OK    cache=True: 2 calls to import_gate "
              f"('{target}') return same instance "
              f"(id={id(a1)})")

    # (b) Two uncached calls → different instances.
    b1 = import_gate(target, cache=False)
    b2 = import_gate(target, cache=False)
    if b1 is None or b2 is None:
        mismatches.append(
            f"cache=False import failed for {target}")
    elif id(b1) == id(b2):
        mismatches.append(
            f"cache=False regression: two import_gate('{target}', "
            f"cache=False) calls returned SAME instance "
            f"(id={id(b1)}). cache=False is failing to bypass "
            "the cache.")
    else:
        print(f"# OK    cache=False: 2 calls return distinct "
              f"instances (id={id(b1)}, id={id(b2)})")

    # (c) cache=False doesn't poison cache=True.
    c1 = import_gate(target, cache=True)
    if c1 is None:
        mismatches.append("post-bypass cache=True call returned None")
    elif a1 is not None and id(c1) != id(a1):
        mismatches.append(
            f"cache poisoning regression: cache=True after cache=False "
            f"returned a NEW instance (id={id(c1)}) instead of the "
            f"original cached one (id={id(a1)})")
    else:
        print(f"# OK    cache=True after cache=False: still returns "
              f"original cached instance (id={id(c1)})")

    if mismatches:
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(f"# verify_module_cache_consistency.py — "
              f"{len(mismatches)} cache-contract violation(s)",
              file=sys.stderr)
        return 1
    print(f"# verify_module_cache_consistency.py — all 3 cache "
          "contract clauses hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
