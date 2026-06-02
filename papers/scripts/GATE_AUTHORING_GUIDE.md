# Paper-side CI gate authoring guide

Distillation of ~14 loops of gate-evolution learnings (Loops 128–142).
This guide is for adding a new `verify_*.py` stage to
`papers/scripts/run_all_checks.sh` without retreading the patterns
the prior 30+ gates already exercise.

This is **internal methodology**, not a CI artifact. The companion
papers (`papers/f2_methodology.md`, `papers/phi_ladder_paper_intro_draft.md`)
do not depend on it shipping or being green.

## When to add a new gate

Add a gate when **two paper-side claims can drift relative to each
other** and the drift is invisible to existing gates. Examples that
seeded prior loops:

- §1 sub-bullet says "11 claims (1 EXACT + 5 SCOPED ...)" but the
  live `_CLAIMS` lists in `verify_cross_paper_consistency.py` contain
  a different distribution → `verify_class_registry_binding.py`.
- CHANGELOG §7 lead says "Fifty-two passes" but SUBMISSION_CHECKLIST
  §2 says "53 passes" → `verify_changelog_consistency.py`.
- FALLBACK_BASELINES breadcrumb claims `Loop 138: 11 + 27 = 38` but
  the sidecar lives at `16 + 32 = 48` → `verify_burn_down_history.py`.

**Do not** add a gate that only catches a one-off bug. The discipline
is: each gate must encode a *class* of drift that will recur.

## Stage 0: shared helpers in `_gate_utils.py`

Before authoring a new gate, scan `_gate_utils.py` for reusable
primitives:

- `import_gate(name, cache=True)` — load a sister gate module.
  Returns the module on success or `None` on failure (traceback to
  stderr). Cache is on by default; pass `cache=False` to bypass.
- `to_int(w)` — convert digits or English numerals 0..99 (including
  compound forms like "twenty-one") to int. Use anywhere paper prose
  alternates between "61" and "Sixty-one".

When you find yourself duplicating a helper, hoist it here. The
59th-pass #1 caught a regression where `_import_gate` was
copy-pasted with the old un-fixed traceback path; the centralization
in Loop 137 A.iv prevents that class permanently.

## Stage 1: gate skeleton

A canonical gate has this shape:

```python
#!/usr/bin/env python3
"""verify_<class>_<thing>.py — gate <claim A> against <claim B>."""

from __future__ import annotations
import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]

def main() -> int:
    mismatches: list[str] = []
    # ... parse + assert ...
    if mismatches:
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(f"# verify_<thing>.py — {len(mismatches)} drift(s)",
              file=sys.stderr)
        return 1
    print(f"# verify_<thing>.py — N invariants verified, 0 drift")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

Conventions:

- `CRATE_ROOT = Path(__file__).resolve().parents[2]` (gates live two
  dirs deep under crate root).
- Print OK lines to stdout, FAIL lines to stderr. The
  `run_all_checks.sh` orchestrator captures both.
- Tail summary line names the gate filename and the verified count.
  This is the line `run_all_checks.sh` echoes via `tail -1`.

## Stage 2: synthetic break-test in `meta_test_cross_paper_gates.py`

Every gate must ship with a break-test that proves the gate fires
when fed mutated input. Without this, a regex regression silently
disables the gate (the 46th-pass `\bf12` typo bug — Loop 122 — is
the canonical case).

Pattern: tempdir + subprocess + assert non-zero exit + diagnostic
fragment. See `test_burn_down_arithmetic_break` (Loop 140 A.iv) and
`test_documented_vs_extracted_break` (Loop 141 A.iv) for templates.

```python
def test_<gate>_break(tmp: Path) -> bool:
    import shutil
    src_dir = CRATE_ROOT / "papers" / "scripts"
    dst_dir = tmp / "papers" / "scripts"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ["verify_<gate>.py", "_gate_utils.py", ...]:
        shutil.copy2(src_dir / name, dst_dir / name)
    # ... copy paper files ...
    # ... mutate one file to break the invariant ...
    result = subprocess.run(
        [sys.executable, str(dst_dir / "verify_<gate>.py")],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        print("# FAIL  break: gate exited 0 despite mutation",
              file=sys.stderr)
        return False
    if "<unique fragment>" not in result.stderr:
        print("# FAIL  break: stderr missing diagnostic fragment",
              file=sys.stderr)
        return False
    print("# OK    break: gate fires with diagnostic")
    return True
```

**Pin to a unique fragment** (Loop 141 #9): `"5"` and `"6"` are not
diagnostic; `"claims 5 reports"` is.

**Inject at a non-most-recent position** (Loop 140 #12): mutating
the latest breadcrumb entry would mask the arithmetic check via the
live-binding check.

## Stage 3: tier classification

Every new stage gets a tier label in `STAGE_TIERS` (parallel array
in `run_all_checks.sh`). Allowed tiers:

- `submission` — must pass for submission readiness (PDF rendering,
  metadata parity, label leaks, anonymization-breaking drift).
- `discipline` — internal drift catchers + registry binders.

`verify_tier_classification.py` (stage 35) asserts STAGE_TIERS
length matches STAGES; `verify_anchor_loop_coverage.py` (33) +
`verify_dependency_graph.py` (34) round out the meta-gate trio.

If unsure, default to `discipline`; promote to `submission` only if
the gate's fail mode is visible in the submission bundle (e.g.,
anonymization leak, page count drift).

## Stage 4: cascade discipline

After adding a stage, the following paper sites must be refreshed
in the same commit:

- F2 §E catalogue: `**N stages**` + "other twenty-N stages" prose
  + enumeration list
- #1021 §5.4: `(N stages on disk as of Loop M)` + `**N-stage on-disk
  gate**` partition + enumeration
- CHANGELOG §10: `now runs **N stages**`
- SUBMISSION_CHECKLIST §1: `**N/N PASS**` header + every `(k/N)`
  sub-bullet + 1 new `(N/N) <gate-name> — <description>` line

The `verify_stage_count_consistency.py` gate gates 5 sites (counts)
+ 2 (decompositions) + 1 (derived). Failing it lists the exact
file:line. The gate-catches-its-own-author's-drift pattern means
you fix as you go.

## Stage 5: dependency-graph hygiene

If your gate calls `_gate_utils.import_gate("verify_X.py")`,
`verify_dependency_graph.py` (stage 34) automatically picks up the
edge and asserts:

- No cycles in the import DAG.
- STAGES execution order respects dependency direction (a later
  stage may import an earlier one).

If you add an inter-gate import, ensure your gate's STAGES index is
greater than the imported gate's. The dependency-graph gate will
fire on order violations.

## Stage 6: breadcrumb discipline (if your gate has a ratchet)

If your gate has a ratchet baseline (like
`verify_anonymizer_completeness.py`), maintain the breadcrumb in
the FALLBACK_BASELINES docstring. Each loop's burn-down or
re-baseline gets a one-line entry:

```
#   Loop N <suffix> (short scope): A + B = C.
```

`verify_burn_down_history.py` (29) gates that the most-recent entry
matches live sidecar; `verify_burn_down_trajectory.py` (36) gates
that Loop numbers strictly increase and total C is monotonically
non-increasing.

The label class accepts `[A-Za-z0-9.()\s§#+/\-—–']`. Avoid commas
and quotation marks in labels (Loop 142 lesson: a comma in the
label silently dropped the entry from the parse).

## Bottom-line discipline rules

1. **Verify before publish**: every gate fires on at least one
   synthetic break.
2. **Catch a class, not a case**: the gate's regex must match the
   *pattern* of drift, not just yesterday's specific instance.
3. **Two sites of truth → one binder**: any time a paper claim is
   mirrored in code, bind them with a gate.
4. **Tier-label every stage**: an untiered stage silently
   misclassifies in the summary.
5. **Document the burn-down trajectory**: the FALLBACK_BASELINES
   breadcrumb is the project's only audit of what was burned when.

## See also

- `papers/CHANGELOG.md` §10 — per-loop gate introduction history.
- `papers/CHANGELOG.md` §7 — adversarial review pass log.
- `papers/SUBMISSION_CHECKLIST.md` §1 — gate roster mirrored to
  human-readable sub-bullets.
- `papers/scripts/run_all_checks.sh` — STAGES array (source of truth).
- `papers/scripts/_gate_utils.py` — shared helpers; extend here
  before duplicating.
