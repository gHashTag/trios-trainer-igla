# P1 Null Result — Muon vs AdamW

**Date:** 2026-04-27
**Phase:** P1 - Optimizer Lab
**Hypothesis:** Muon (η2D=0.0235, η1D=0.007) reduces final BPB by >=0.05 vs AdamW at 12K steps
**Outcome:** **FALSIFIED**

## Evidence

From `assertions/seed_results.jsonl` (3000-step comparison):

| Optimizer | Seed 42 BPB | Seed 43 BPB | Seed 44 BPB | Avg BPB | vs AdamW |
|-----------|--------------|--------------|--------------|----------|----------|
| **AdamW** | 2.6840 | 2.6970 | 2.6998 | 2.6936 | baseline |
| **Muon** | 2.7473 | 2.7784 | 2.7551 | 2.7603 | **+0.07** |

**Key Finding:** Muon performed **WORSE** than AdamW by approximately **0.07 BPB** at 3000 steps.

## Decision

According to TRAINING_FLOW_V2.md P1 specification:
> "Falsification: Muon does not beat AdamW by >= 0.05 on this corpus → proceed with AdamW for P2 and document the null result in `docs/audit/P1_null.md` (do not pretend gain)."

**DECISION:** Proceed with AdamW for P2-P5. Do NOT use Muon in gate2-final.toml.

## Implications

- `gate2-final.toml` optimizer should be changed from `muon` to `adamw`
- P2-P4 experiments should use AdamW as baseline
- Muon+CWD (Cautious Weight Decay) was NOT evaluated (not needed since base Muon already worse)

## Next Steps

1. Update `configs/gate2-final.toml` to use AdamW
2. P2 muP transfer — use AdamW
3. P3 Schedule-Free/WSD — benchmark against AdamW cosine baseline
4. P4 Multi-Obj + EMA — use AdamW
5. P5 Gate-2 Push — use AdamW configuration
