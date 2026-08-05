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

> **SUPERSEDED 2026-08-03 — see section 5 and [`RETRACTION.md`](../../RETRACTION.md).**
> This section named 2.1919 the "honest champion". That number was retracted with
> the ledger it came from. It is not citable and there is no champion in its place.
> The rows below are preserved because this file is append-only; the status
> column is corrected in place so the table cannot be read as a live claim.

**Status:** RETRACTED prior values; the replacement named here was itself retracted (section 5)
**Context:** sub-Chinchilla run on unique data (~1.1M characters), so **all BPB are preliminary** (not comparable to standard benchmarks).

### Retracted values
| Value | Claimed in | Status |
|-------|------------|--------|
| 2.5193 | `src/invariants.rs` `BPB_CHAMPION` | **RETRACTED** — stale placeholder; no artifact (see RETRACTION.md §2) |
| 2.2393 | `tests/champion_reproduction.rs`, docs | **RETRACTED** — no artifact, unresolvable commit `2446855`, forbidden seed |
| 2.2111 | `DEPLOYMENT_BLOCKER.md`, `.trinity/dashboard.md`, `.trinity/STATUS.md`, `.trinity/experience/trios_20260427_pt2.trinity` | **RETRACTED** — superseded by Wave-8, which is itself retracted. **[SITE LIST CORRECTED 2026-08-03:** this row named `README.md` as a declaring site. It is not one -- `grep -n '2.2111' README.md` returns nothing. The four sites listed are the ones a repo-wide grep finds today, and they are the same four that `RETRACTION.md` section 2 lists for this value.**]** |

### "Honest champion (Wave-8)" — RETRACTED 2026-08-03
| Field | Value | Status |
|-------|-------|--------|
| BPB | 2.1919 | **RETRACTED** — not citable, no artifact, written outside `ledger::emit_row` |
| Seed | 43 | **RETRACTED** — forbidden under Canon #93 (`src/seed_canon.rs`) |
| Steps | ~81 000 | unverifiable — no checkpoint was ever written at that sha |
| Hidden | 828 | not in dispute (the trainer hardcodes it) |
| Data | ~1.1M characters (unique corpus) | **RETRACTED** — no corpus digest was recorded |
| Benchmark | sub-Chinchilla (preliminary) | — |

### Honest label
`[RETRACTED]` — superseded by section 5. Not `[PRELIMINARY_BPB]`: a preliminary
measurement is still a measurement, and this one has no artifact behind it.

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

# Check champion constants -- every hit below is retracted, see RETRACTION.md
grep -n "BPB_CHAMPION\|2.1919\|2.2393\|2.2111" src/invariants.rs docs/audit/*.md   # all retracted

# Check no δ_CP usage
grep -rn "delta_CP\|δ_CP\|3/phi²" src/ proofs/ derivations/ 2>/dev/null || echo "No δ_CP found — honest"

# Check the ledger is retracted and has not come back
test ! -f assertions/seed_results.jsonl && head -3 assertions/RETRACTED-seed_results.jsonl.txt
```

---

## 5. The ledger itself was retracted (2026-08-03)

**Status:** RETRACTED
**Severity:** The published source of truth for BPB evidence

Section 2 above presented **2.1919** as the current honest champion. RETRACTED.
That number came from `assertions/seed_results.jsonl`, which was tracked in git and declared
the SSOT by `SOURCE_OF_TRUTH.md`, `MIGRATION.md`, `docs/TRAINING_FLOW_V2.md`,
`docs/preregistration/asymlogit_ngram_v1.md` and `igla-dash/README.md`. The whole
ledger is retracted. It is preserved verbatim at
[`assertions/RETRACTED-seed_results.jsonl.txt`](../../assertions/RETRACTED-seed_results.jsonl.txt)
behind a header naming, per row family, what is missing.

Why no row in it is citable:

- **No artifact.** `checkpoint::save` was `pub fn save(..) -> Result<()> { Ok(()) }`
  at both commits the rows cite (`4c0b04c`, `cd91c45`). No weights were written,
  so nothing can be re-measured.
- **No corpus hash.** At both commits `load_data` silently substituted a
  112 500-byte pangram for a missing corpus file. No row can tell you which it got.
- **No trainer hash.** `sha` is an abbreviated repo HEAD; it pins no binary, no
  config digest, no data.
- **No eval coverage.** No val token count, no eval window count, no cadence -
  and `--eval-every` is not observation-only, so rows without it are not comparable.
- **Forbidden seeds.** Every row uses 42, 43 or 44; `src/seed_canon.rs` forbids
  all of them under Canon #93.
- **Outside the validated emit path.** No row carries `agent`, `jsonl_row` or
  `ts`, which the in-repo `LedgerRow` requires, so R7 triplet validation never
  ran on any of them. Six of eleven also violate R8 (`step >= 4000`).

The retracted 2.1919 row carried `gate_status: "above_target"`, which was read
as a pass. It was not one even on its own terms: the Gate-2 target in force was
BPB < 1.85, and the retracted 2.1919 is above that.

**There is no champion.** `RETRACTION.md` section 2 -- "The five coexisting
numbers" -- lists all five BPB numbers that coexisted in this tree, all
retracted but one (2.5193, 2.2393, 2.2111, 2.1919, 2.6348), and says for each
whether it is citable. Only the last one is, and only with its eval cadence and
checkpoint digest stated alongside it.
**[COUNT CORRECTED 2026-08-03:** this paragraph said "all four" and omitted the
retracted 2.2111 from the list while the table it describes has five rows.
Corrected against `RETRACTION.md` section 2 as it stands today.**]**

### Honest label
`[RETRACTED_EVIDENCE]` — an evidenced "could not verify" replaces a number that
looked like a result.

---

*This file is append-only. If a finding is superseded, add a new section — do not delete the old one.*
*Exception, 2026-08-03: section 2's status column was corrected in place rather than only appended to, because leaving `| BPB | **2.1919** |` unqualified left a retracted number readable as a live claim. No row was deleted.*
*Exception, 2026-08-03 (second): two stale statements about OTHER files were corrected in place, each with its correction marked inline rather than silently overwritten -- the retracted-2.2111 row's "Claimed in" list (it named `README.md`, which no longer declares that value) and section 5's "all four BPB numbers" (the table it describes has five rows). Both were checked by grep against the tree at the time of the edit. No finding was removed and no status changed.*
