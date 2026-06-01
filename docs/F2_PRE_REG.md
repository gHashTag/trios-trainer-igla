# F2 Pre-Registration — Issue #1021 Champion-Scale Sweep

**Pre-registration drafted**: Loop 51 (2026-06-01).
**Status**: design locked, compute pending user decision.
**Filed before any champion-scale data exists.** Once data lands, this document
becomes the protocol against which results are evaluated.

## 1. Research question

Does the **phi-ladder quantization path** (GFTernary → GF8 → GF16 → GF32) yield
lower validation BPB than mainstream alternatives (BitNet b1.58, INT4-W4A8,
FP8, bf16) at champion scale, controlling for WD-confounding?

## 2. Hypotheses

We pre-register three nested hypotheses and the BPB shifts that would support
each.

### H0 (null)
For every (phi, zoo) pair, mean validation BPB is within ±0.05 BPB
(equivalence margin). No path is superior at our scale.

### H1 (phi superior on at least one zoo competitor)
Mean BPB is ≥ 0.10 BPB lower for phi-ladder than for at least one of
{BitNet-1.58, INT4-W4A8, FP8, bf16} at p < 0.05 (paired permutation across
seeds, BH-corrected over 4 comparisons).

### H2 (phi dominant across the zoo)
Mean BPB is ≥ 0.10 BPB lower for phi-ladder than every mainstream alternative
at p < 0.05 (paired permutation, BH-corrected over 4 comparisons).

### Confounder-controlled variant
Repeat H1/H2 at the **WD=0 stratum** (Pearl CDE). Loop 49 showed that WD
suppresses RmsNorm's intrinsic effect; we expect the same applies to
quantization paths. If H1/H2 holds at WD=0 stratum but fails at default WD,
the difference is interpretable as "phi-ladder works because it interacts
favorably with WD's structural role," not as raw superiority.

## 3. Experimental design

### Configurations (8)
| Path | Family | Notes |
|---|---|---|
| GFTernary | phi-ladder | `{−φ, 0, +φ}` ternary, our path |
| GF8 | phi-ladder | 8-bit phi-encoded |
| GF16 | phi-ladder | 16-bit phi-encoded |
| GF32 | phi-ladder | 32-bit phi-encoded (full precision baseline) |
| BitNet-1.58 | format-zoo | `{−1, 0, +1}` ternary per arXiv:2402.17764 |
| INT4-W4A8 | format-zoo | 4-bit weights + 8-bit activations |
| FP8 | format-zoo | per NVIDIA MXFP8 / arXiv:2509.22536 |
| bf16 | format-zoo | gold-standard baseline |

### Hyperparameter regimes
For confounder-controlled analysis we run TWO strata:
- **Canonical** (default WD=0.1, all 7 fixes enabled per `f2_ablation_sweep`)
- **wd0** (WD pinned to 0.0, otherwise identical)

### Model size and data
- **Params**: ~1B (champion scale per Issue #1021)
- **Data**: FineWeb 10B tokens (validate against held-out 100M)
- **Sequence**: 2048 tokens
- **Training tokens**: 50B (validation BPB stable beyond ~30B per typical
  scaling-law studies)

### Seeds and replication
- **5 seeds** per (path, stratum) cell → 8 × 2 × 5 = 80 runs total
- Seeds: [42, 43, 44, 45, 46] (matches existing F2 framework convention)

### Evaluation
- **Primary**: validation BPB (per byte, not per token) on FineWeb held-out
- **Secondary**: training BPB at final step, wall-clock to convergence,
  memory peak

## 4. Analysis plan (pre-specified)

### Step 1: per-cell mean ± SE
Run `f2_ablation_aggregate` on the 80-run CSV to produce wide-form table.

### Step 2: paired permutation test
For each (phi-config, zoo-config) pair, paired permutation test on BPB
differences across 5 seeds (Fisher-Pitman exact, 32 sign-flips per
arXiv:2205.01416). BH-correct p-values over the 4 comparisons within each
stratum.

### Step 3: cross-stratum stability
Run `f2_dual_mediation` on each pair at the wd0 stratum; pipe to
`f2_mediation_sensitivity --lambda-sweep`. Verify the BPB difference is
stable under unmeasured confounding Γ ∈ [1.0, 5.0].

### Step 4: cross-stratum comparison
Run `f2_stratum_compare` on the two strata; flag any pair whose
`stable_across_strata` is false (CI disjoint between canonical and wd0).

### Step 5: pre-specified subgroup
If both strata agree on H1/H2, conclude.
If they disagree, report both estimates and frame the canonical result as
"likely confounded by WD" per Loop 49 precedent.

## 5. Success criteria (binary)

The sweep is declared **successful** iff ALL three hold:

1. ≥ 38 of 40 runs per stratum complete without divergence (eval BPB not NaN)
2. Per-seed BPB CV across seeds within each cell is < 5% (sanity check on
   trainer stability)
3. `f2_provenance_check` exits 0 (PASS) on the resulting CSV

The sweep is declared **inconclusive** (not failed) iff:
- H0 holds in both strata AND minimum-detectable effect at our N=5 is > 0.10
  BPB (i.e., we don't have power to distinguish 0.10 BPB)

The sweep is declared **failed** iff:
- Trainer divergence or provenance failure in either stratum (re-run with
  fix, do not interpret results)

## 6. Pre-registered claims for write-up

We will report, regardless of outcome:
- The full 80-row CSV (in supplementary)
- p-values for all 8 (phi vs zoo) comparisons in each stratum (BH-corrected)
- `stable_across_strata` verdict per comparison
- Sensitivity envelope: minimum Γ at which any "phi superior" verdict flips
- Honest framing of what failed: divergent runs, CV outliers, missing data

## 7. Stopping rules

- **Resource cap**: if compute exceeds 2× budget without 80 runs completing,
  halt and report partial results.
- **Trainer instability cap**: if > 4 of 40 runs per stratum diverge, halt
  the sweep, debug the trainer, and re-run from scratch. Do not partial-report
  unstable cells.
- **Mid-run reanalysis prohibition**: we will NOT change the analysis plan
  after seeing partial data. Any reanalysis is exploratory, not confirmatory.

## 8. Reproducibility commitment

- The exact `f2_ablation_sweep` command lines per cell are committed to this
  repo before any run starts (see `experiments/loop_52_champion_commands.sh`,
  to be added when compute is approved).
- All raw CSVs + provenance preambles are committed in a `data/loop_52/`
  subdirectory.
- The PR opening this study (anchored at `19d032e` HEAD) links to this
  pre-registration as the protocol.

## 9. What this document is NOT

- It is not a research proposal seeking funding.
- It is not a guarantee of compute availability.
- It does not commit to any specific timeline.
- It does not preclude a later, separate pre-registration for related questions
  (e.g., warmup0 stratum, 3-way iLOCO on quantization paths, etc.).

## 10. Adversarial review checklist

If anyone wants to attack the design before data is collected:

- Q: Why 5 seeds? A: Owen 2025 (arXiv:2508.10083) shows BCa undercovers at
  N=5; we use Fisher-Pitman exact (arXiv:2205.01416) which is valid at this N.
- Q: Why BH over 4 comparisons within each stratum? A: matches Liu/Leung/Shao
  arXiv:1712.03305 dependent-test BH validity at N=5.
- Q: Why 50B tokens? A: Quantization scaling laws (arXiv:2502.05003) suggest
  effects saturate beyond ~30B; we add headroom for safety.
- Q: Why both strata? A: Loop 49 sign-flip finding established that
  WD-confounded canonical results can mislead. The wd0 stratum is the
  honest counterfactual.

---

## What this document needs from the user to graduate

1. **Compute target chosen** (Railway / personal GPU / cloud)
2. **Budget approved** (rough $ ceiling)
3. **Decision: run only wd0 stratum** (half compute) or both strata (full)
4. **Branch creation: `f2-pre-reg` to commit this document + `experiments/loop_52_champion_commands.sh` together** (atomic protocol)

Until 1-4 are signed off, this document is **draft-protocol, non-binding**.
