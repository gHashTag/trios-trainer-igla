# Epic #181 — φ Falsifiability Ablations: Experimental Design

**Date:** 2026-05-31
**Status:** Design draft
**Honest label:** `[Open conjecture]` — phi as design-prior has never been ablated

---

## 1. Hypothesis

**Null hypothesis H₀:** Replacing all φ-anchored hyperparameters with conventional values does not change final BPB by more than 0.05 (the P1 null threshold).

**Alternative H₁:** φ-anchored values improve BPB by ≥ 0.05 compared to conventional baselines.

---

## 2. Treatment vs Control

### Fixed across both arms
| Parameter | Value |
|-----------|-------|
| Seed | 1597 (same for both) |
| Steps | 81,000 |
| Format | bf16 |
| Hidden | 128 |
| Optimizer | AdamW |
| Data | tiny_shakespeare (~1.1M chars) |
| Eval frequency | 5,000 steps |

### Control arm (conventional)
| Parameter | Value | Source |
|-----------|-------|--------|
| lr | 0.001 | Standard default |
| weight_decay | 0.01 | HuggingFace default |
| beta1 | 0.9 | AdamW default |
| beta2 | 0.999 | AdamW default |
| warmup | 500 | Linear ramp |

### Treatment arm (φ-anchored)
| Parameter | Value | φ Derivation |
|-----------|-------|--------------|
| lr | φ⁻³ ≈ 0.23607 | `1/φ³` (honest anchor) |
| weight_decay | φ⁻³ ≈ 0.23607 | Same anchor |
| beta1 | 1/φ ≈ 0.618 | Golden-ratio decay |
| beta2 | 0.95 | Heuristic (not φ-derived, fixed) |
| warmup | 500 | Same as control |

---

## 3. Primary outcome

Final validation BPB at step 81,000.

Secondary outcomes:
- Convergence speed (steps to BPB < 3.0, < 2.5)
- EMA BPB stability (variance of last 10 evals)
- Wall-clock time per step

---

## 4. Falsification criteria

Per P1 Optimizer Lab rules (TRAINING_FLOW_V2.md):
- If Treatment BPB > Control BPB + 0.05 → **FALSIFIED** (phi does not help)
- If |Treatment − Control| < 0.05 → **NULL** (phi is aesthetic)
- If Treatment BPB < Control BPB − 0.05 → **EMPIRICAL_FIT** (phi may help; more seeds needed)

---

## 5. Statistical power

Single-seed comparison is **preliminary only**.
To claim significance, need 3 seeds × 2 arms = 6 runs minimum.
For Epic #181 Phase A, we run 1 seed pair as a pilot.

---

## 6. Known confounders

| Confounder | Mitigation |
|------------|------------|
| Sub-Chinchilla data | Both arms use identical corpus; BPB not comparable to Pile/OWT |
| lr=0.236 very high | Clip to 0.01 if INV-1 violation detected; document if clipped |
| wd=0.236 very high | Same clipping rule; AdamW decouples wd from gradients |

---

## 7. Implementation checklist

- [ ] Deploy Control service to Railway (`short-wave-bf16-sgdm-control`)
- [ ] Deploy Treatment service to Railway (`short-wave-bf16-sgdm-phi`)
- [ ] Wait for both to reach 81K steps
- [ ] Harvest final BPB values to `gardener_harvest.log`
- [ ] Compute delta, classify per §4
- [ ] Update `HONEST_FINDINGS.md` with result

---

## 8. Honest principle

> Only one φ-fact is `[Verified]`: φ² + φ⁻² = 3.
> This experiment tests whether φ is useful as a **design-prior**, not whether it
> "explains physics". Any outcome is valuable: success, null, or falsification.

---

*Append-only document. Phase results will be added as sections, not edits.*
