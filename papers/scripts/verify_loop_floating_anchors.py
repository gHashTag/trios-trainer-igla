#!/usr/bin/env python3
"""verify_loop_floating_anchors.py — gate "as of Loop N" anchors so
the loop number tracks current state instead of drifting silently.

Loop 139 C operationalizes the 61st-pass #7 SEV-3 catch: phi_ladder
§5.4 line 831 reads "30 stages on disk as of Loop 138". The
stage-count gate enforces "30 stages" but the regex captures Loop
138 as a non-binding group — a future loop bumping STAGES to 31
without touching "Loop 138" leaves the prose stuck through Loop
145+.

This gate scans the registered TMLR-bound paper bodies for
`(... as of Loop N)` patterns and asserts:
  N matches the §7 lead loop (current cumulative state) within a
  1-loop in-flight tolerance (the current Loop's edits may
  legitimately bump §7 first, then update the prose in a follow-up).

Per-anchor labels can also exempt entries (e.g., the CHANGELOG
"as of Loop 131 B" entry is a frozen historical reference and not
gated here).

Usage: papers/scripts/verify_loop_floating_anchors.py

Exit 0 on agreement; 1 on any drift beyond the 1-loop tolerance.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPTS_DIR))
try:
    from _gate_utils import import_gate
finally:
    sys.path.pop(0)


# Files to scan for live "as of Loop N" anchors. The CHANGELOG entries
# carrying "as of Loop X" are historical-snapshot annotations
# (Loop 135 B introduced the convention with "(as of Loop 131 B;
# superseded by Loop 133 A.iii)") — those are intentionally FROZEN
# and excluded from this gate.
# Loop 139 follow-up (62nd-pass SEV-4 fix #13): derive scan targets
# from the anonymizer baseline sidecar so adding a new TMLR-bound
# paper there auto-includes it here. Falls back to the explicit list
# if the sidecar can't be loaded.
def _scan_targets() -> list[Path]:
    try:
        import json
        sidecar = (
            CRATE_ROOT / "papers" / "scripts" / "anonymizer_baseline.json"
        )
        data = json.loads(sidecar.read_text())
        bl = data.get("baselines")
        # Loop 140 — 63rd-pass SEV-4 fix #3: require non-empty dict.
        # An empty `{"baselines": {}}` previously yielded `[]` and
        # silently scanned 0 files (gate would pass with "0 live
        # anchors all within tolerance"). Now falls through to the
        # explicit 2-paper fallback in that case.
        if isinstance(bl, dict) and bl:
            return [CRATE_ROOT / k for k in bl if isinstance(k, str)]
    except Exception:
        pass
    return [
        CRATE_ROOT / "papers" / "f2_methodology.md",
        CRATE_ROOT / "papers" / "phi_ladder_paper_intro_draft.md",
    ]


SCAN_TARGETS = _scan_targets()


# Match `as of Loop N` (capturing N) with optional context. The
# preceding "frozen-historical" idiom `*as of Loop X B; superseded`
# is excluded by requiring a non-italic preface (no `*` before "as").
_ANCHOR_RE = re.compile(
    r"(?<!\*)\bas of Loop (\d+)\b",
)


def get_changelog_lead_loop() -> int | str:
    """Resolve the §7 lead loop via verify_changelog_consistency's
    SITES registry: CHANGELOG §7 lead paragraph's terminal loop."""
    mod = import_gate("verify_changelog_consistency.py")
    if mod is None:
        return "could not import verify_changelog_consistency.py"
    sites = getattr(mod, "SITES", None)
    if not isinstance(sites, list) or not sites:
        return "SITES list missing/empty"
    # First entry is CHANGELOG §7 lead.
    path, pattern, _label = sites[0]
    if not path.exists():
        return f"{path} missing"
    text = path.read_text()
    m = re.search(pattern, text)
    if not m:
        return f"§7 lead pattern did not match {path.name}"
    # Group 2 is the last_loop; group 1 is the pass count.
    return int(m.group(2))


def main() -> int:
    lead = get_changelog_lead_loop()
    if isinstance(lead, str):
        print(f"# FAIL  resolve lead loop: {lead}", file=sys.stderr)
        return 1
    print(f"# verify_loop_floating_anchors.py — §7 lead loop = {lead}; "
          "scanning for 'as of Loop N' anchors")

    mismatches: list[str] = []
    scanned = 0
    for path in SCAN_TARGETS:
        if not path.exists():
            print(f"# WARN  {path.relative_to(CRATE_ROOT)}: missing; "
                  "skipping", file=sys.stderr)
            continue
        text = path.read_text()
        for m in _ANCHOR_RE.finditer(text):
            scanned += 1
            n = int(m.group(1))
            line_no = text[:m.start()].count("\n") + 1
            rel = path.relative_to(CRATE_ROOT)
            # Tolerance: gate accepts N == lead or N == lead - 1 (the
            # 1-loop in-flight lag where §7 was bumped first and the
            # prose update lands in the next sweep). Drift beyond
            # that fires.
            if abs(n - lead) > 1:
                mismatches.append(
                    f"{rel}:{line_no}: 'as of Loop {n}' but §7 lead "
                    f"is Loop {lead} (drift of {abs(n - lead)}). "
                    "Bump the prose to current state.")
            else:
                print(f"# OK    {rel}:{line_no}  'as of Loop {n}' "
                      f"(lead {lead}, in 1-loop tolerance)")

    if mismatches:
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(f"# verify_loop_floating_anchors.py — "
              f"{len(mismatches)} drift(s) across {scanned} live "
              "'as of Loop N' anchor(s)",
              file=sys.stderr)
        return 1
    print(f"# verify_loop_floating_anchors.py — {scanned} live "
          f"'as of Loop N' anchor(s) all within 1-loop tolerance of "
          f"lead {lead}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
