# Honest Findings — IGLA RACE

**Date:** 2026-05-31
**Wave:** 8 (Wave-8 honest checkpoint)
**Policy:** No fake proofs. No cosmetic edits to hide gaps.

---

## 1. Bug φ⁻³ — weight_decay honest value

**Status:** FIXED in code
**Severity:** Comment bug (code was correct, comments lied)

### What was wrong
- `src/optimizer.rs` comments said `weight_decay = α_φ ≈ 0.11803`
- `src/invariants.rs` named the constant `ALPHA_PHI = 0.11803398874989485`
- The **actual** code computed `1.0 / (phi * phi * phi) = φ⁻³ ≈ 0.2360679774997897`

### What 0.11803 really is
`0.11803398874989485 = φ⁻³ / 2` — half of the true weight-decay anchor.

### Fix applied
- `invariants.rs`: renamed `ALPHA_PHI` → `PHI_INV3_HALF`, added `PHI_INV3 = 0.2360679774997897`
- `optimizer.rs`: all comments updated to `φ⁻³ ≈ 0.23607`
- `ema.rs`: trace comments updated to reference `PHI_INV3_HALF`
- Tests updated to verify both values explicitly

### Honest label
`[CODE_CORRECTION]` — no numerical result changed (code already used 0.23607), only comments/constant names were misleading.

---

## 2. Honest Champion BPB

**Status:** RETRACTED prior values; current honest record below
**Context:** sub-Chinchilla run on unique data (~1.1M characters), so **all BPB are preliminary** (not comparable to standard benchmarks).

### Retracted values
| Value | Claimed in | Status |
|-------|------------|--------|
| 2.5193 | `src/invariants.rs` `BPB_CHAMPION` | **RETRACTED** — stale placeholder |
| 2.2393 | `tests/champion_reproduction.rs`, docs | **RETRACTED** — old replication target |
| 2.2111 | `DEPLOYMENT_BLOCKER.md`, `README.md` | **RETRACTED** — superseded by Wave-8 |

### Honest champion (Wave-8)
| Field | Value |
|-------|-------|
| BPB | **2.1919** |
| Seed | 43 |
| Steps | ~81 000 |
| Hidden | 828 |
| Data | ~1.1M characters (unique corpus) |
| Benchmark | sub-Chinchilla (preliminary) |

### Honest label
`[PRELIMINARY_BPB]` — not comparable to standard Pile/OWT BPB because data distribution and tokenization differ. Use only for internal IGLA RACE tracking.

---

## 3. Design-Prior Principle for φ

**Status:** Preserved
**Policy:** φ is a **falsifiable design-prior**, not a physical constant.

### Verified fact (Coq-proven)
`φ² + φ⁻² = 3` — this is the **only** `[Verified]` statement involving φ in this codebase.

### Retracted claims
| Claim | Status | Reason |
|-------|--------|--------|
| `δ_CP = 3/φ²` | **RETRACTED** | Not used in any production code or derivation |
| Any ToE / prize claim | **FORBIDDEN** | Per `CLAUDE.md` rule 5 |

### What φ is allowed to do
- Anchor hyperparameters (LR, weight_decay, EMA decay) as a **convenience constant**
- Serve as a **falsifiable prior** for grid-search initialization
- Provide **traceability** (every φ-based constant derives from `PHI`, `PHI_SQ`, etc.)

### What φ is NOT allowed to do
- Explain physical constants without derivation
- Be promoted to `verified` without proof
- Be used to claim a Theory of Everything

### Honest label
`[DESIGN_PRIOR]` — φ is a heuristic, not physics.

---

## 4. How to Verify

```bash
# Check φ⁻³ values
grep -n "PHI_INV3" src/invariants.rs src/optimizer.rs src/race/ema.rs

# Check champion constants
grep -n "BPB_CHAMPION\|2.1919\|2.2393\|2.2111" src/invariants.rs docs/audit/*.md

# Check no δ_CP usage
grep -rn "delta_CP\|δ_CP\|3/phi²" src/ proofs/ derivations/ 2>/dev/null || echo "No δ_CP found — honest"
```

---

*This file is append-only. If a finding is superseded, add a new section — do not delete the old one.*
