# F2 Loop Numbering — Convention Guide

## What is a "loop"?

A **loop** is a single self-contained research/implementation cycle that follows
this pattern:

1. **Audit** — survey weak spots in the current state of the F2 framework.
2. **Research** — find 2024-2026 papers/sources that inform the upcoming changes.
3. **Decomposed plan** — write a list of concrete fixes (file:line + gap).
4. **Implement** — make the changes, add unit/integration tests.
5. **Report** — summarize what landed + empirical findings if any.
6. **Three collaboration options for next loop** — propose follow-up directions.

The audit-research-plan-implement-report-options cadence is the recurring
contract; the user invokes it by repeating one fixed directive.

## Loop number ranges

| Range | Theme | Anchor binaries |
|---|---|---|
| **24-26** | Multi-seed ablation foundation (cumulative + LOCO + pairwise) | `f2_ablation_sweep`, `f2_ablation_aggregate`, `f2_harness` |
| **27-29** | Interaction scoring (iLOCO, Möbius, BH-FDR) + DOT/Mermaid viz | `f2_iloco_score`, `f2_iloco_dot` |
| **30-33** | Mediation analysis (Baron-Kenny) + W3C-PROV provenance | `f2_mediation`, `f2_provenance_check` |
| **34-39** | Dual-mediation + delta-method SE + tcorrected CIs + `race::stats` centralization | `f2_dual_mediation`, `race::stats` |
| **40-44** | Bridge-score sensitivity + tipping-point + Warmup0 stratum + JSONL adapter | `f2_mediation_sensitivity`, `f2_to_jsonl`, `Stratum::Warmup0` |
| **45+** | Hardening (streaming, schema validation, naming conventions) | — |

## Cross-file "Loop N fix M" comments

Every change carries a comment of the form `Loop 41 fix 3: ...` or
`Loop 42 fix 6: ...`. Two purposes:

1. **Traceability**: any line of code can be traced back to its loop +
   audit punch list item.
2. **Reviewer guidance**: a "Loop 31 fix" comment means "this change
   addressed a specific audit-surfaced gap, not a drive-by edit."

To find every change from a specific loop:
```bash
grep -rn "Loop 30 fix" src/ tests/ docs/
```

## When to bump the loop number

You **don't**. Loops are bumped externally by the user's invocation of the
recurring directive. The implementation just continues from the last loop's
"Three collaboration options" and applies the next audit cycle.

## When to add a new doc

If a loop introduces a new framework concept (e.g., Stratum policy in Loop 45,
or a schema-validation contract in some future loop), drop a short doc into
`docs/` (e.g., `docs/F2_BINARIES.md`, this file, `docs/F2_WEIGHT_DECAY.md`).
Keep documents under ~100 lines; cross-link liberally.

## Cumulative finding (as of Loop 46)

The F2 framework's central empirical finding (Loops 30-40):

> At the default `weight_decay = 0.1` regime, every non-mediator "fix" has a
> +5 BPB indirect effect through WD that is canceled by a −4.5 BPB direct
> effect — a textbook **suppression pattern** (Cohen & Cohen 1983). RmsNorm is
> the only fix with intrinsic effect, with NDE & NIE_M1 robust to Γ ≥ 4.5 of
> unmeasured confounding (Loop 41 tipping-point analysis on Loop 36 data).

Subsequent loops (40-46) focused on tooling robustness, schema validation,
notebook integration, and reproducibility.
