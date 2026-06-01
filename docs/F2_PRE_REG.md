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
differences across 5 seeds. We use the exact paired-permutation
procedure of Zmigrod, Vieira & Cotterell (2022, arXiv:2205.01416,
"Exact Paired-Permutation Testing for Structured Test Statistics") —
their algorithm runs the exact 2^5 = 32-sign-flip enumeration without
Monte-Carlo approximation. BH-correct p-values over the 4
comparisons within each stratum per Liu, Leung & Shao
(arXiv:1712.03305).

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

## 4.5 Phase 0 deliverable: swap-parameterization CSV (Loop 63)

Prior to the champion-scale sweep above, F2 has an outstanding
sandbox-scale deliverable: a three-stratum dual-mediation CSV under
the **swap parameterization** `M_1 = rms, M_2 = warmup`. The current
committed evidence in `data/loop49/` uses `M_1 = wd, M_2 = warmup`
only; the swap parameterization was described in early draft text of
the F2 paper but never anchored against committed data. The §5.3
honest-rewrite in Loop 62 removed the unanchored claim; this section
pre-registers the run that would close the gap.

### Exact commands

```bash
# Anchor: descendant of 5367bde on f2-methodology branch.
# Outputs land in data/loop49_swap/ (NEW subdir, force-add against
# the parent .gitignore /data/ rule).

mkdir -p data/loop49_swap/

# Re-run dual_mediation with M1=rms, M2=warmup on each of the three
# existing committed sweep CSVs:
cargo run --release --bin f2_dual_mediation -- \
  --m1 rms --m2 warmup data/loop49/loop36_dual.csv \
  --out data/loop49_swap/canonical_swap_dual.csv

cargo run --release --bin f2_dual_mediation -- \
  --m1 rms --m2 warmup data/loop49/loop49_wd_stratified.csv \
  --out data/loop49_swap/wd0_swap_dual.csv

cargo run --release --bin f2_dual_mediation -- \
  --m1 rms --m2 warmup data/loop49/loop47_warmup_stratified.csv \
  --out data/loop49_swap/warmup0_swap_dual.csv

# Cross-stratum compare:
cargo run --release --bin f2_stratum_compare -- \
  --canonical data/loop49_swap/canonical_swap_dual.csv \
  --wd0       data/loop49_swap/wd0_swap_dual.csv \
  --warmup0   data/loop49_swap/warmup0_swap_dual.csv \
  --out       data/loop49_swap/3stratum_swap.csv
```

### Expected output

- 3 dual-mediation CSVs in `data/loop49_swap/` (canonical, wd0, warmup0)
- 1 stratum-compare CSV `3stratum_swap.csv`
- Total file count: 4; estimated total size: ~10 KB
- Compute cost: < 1 second wall time (analytical re-aggregation of
  existing per-seed BPB values; no new training)

### Success criteria (Phase 0, binary)

1. All four binaries produce non-empty CSVs with valid W3C-PROV preambles
2. `f2_provenance_check` exits 0 (PASS) on each
3. The `3stratum_swap.csv` populates the `stable_across_strata` column
   with parseable boolean values

### Pre-registered prediction

Under no-XM-interaction (validated by the
`dual_mediation_no_interaction_residual_lock` test on the canonical
parameterization), the NIE_M1 via rms in the swap parameterization
should be approximately constant across the wd0 stratum (since
pinning wd does not directly constrain the rms-mediated pathway).
We pre-register: **we expect the per-row NIE_M1 estimate for at
least 3 of the 5 non-mediator fixes to land within 0.5 BPB of the
canonical-stratum value, across both Pearl-CDE strata**.

If this prediction holds, the §5.3 "framework predicts invariance"
claim becomes empirically supported. If it fails, the §5.3 framing
is wrong and the paper must be revised.

### Phase 0 partial result (Loop 63, 2026-06-01)

The two stratified sweeps already in `data/loop49/` were
re-processed with `--m1 rms --m2 warmup`. Outputs are committed at
`data/loop49_swap/wd0_swap_dual.csv` and
`data/loop49_swap/warmup0_swap_dual.csv` (5 rows each, < 1s
runtime).

### Phase 0 complete (Loop 64, 2026-06-01)

The canonical-stratum sweep was regenerated via
`cargo run --release --bin f2_ablation_sweep -- --mode all
--steps 200 --csv data/loop49_swap/canonical_sweep.csv` (~25 min
wall time). Then the swap pipeline ran end-to-end:

```
canonical_sweep.csv  (88 KB)  ─►  f2_dual_mediation --m1 rms --m2 warmup
                                  ─► canonical_swap_dual.csv (5 rows)
                                  
wd0_swap_dual.csv   (Loop 63)  ──┐
warmup0_swap_dual.csv (Loop 63) ─┤── f2_stratum_compare
canonical_swap_dual.csv (Loop 64)┘   ─► 3stratum_swap.csv (20 rows)
```

`f2_provenance_check` on `canonical_sweep.csv` exits 0 (PASS) at
the Loop 64 anchor commit.

**Phase 0 stable-flag distribution**: of 20 PSE rows, **5 are
flagged `stable_across_strata = true`** and 15 are `false`. The 5
stable rows are:
- `wd, NIE_chain` (CIs bracket zero in all strata)
- `wd, NIE_M1` (byte-identical −0.751 [−1.325, −0.177] across all 3)
- `gradclip, NIE_M1` (byte-identical across canonical + warmup0)
- `clamp, NIE_M1` (byte-identical across canonical + warmup0)
- `smooth, NIE_M1` (byte-identical across canonical + warmup0)

**Pre-registered prediction outcome**: The Loop 63 pre-registered
prediction was "at least 3 of 5 non-mediator fixes show NIE_M1
within 0.5 BPB of canonical across both Pearl-CDE strata". Phase 0
shows **3 of 5 are byte-identical between canonical and warmup0**
(gradclip, clamp, smooth); the wd row is byte-identical across
ALL THREE strata; the dropout row is the only one where the wd0
estimate differs substantially (−0.40 vs canonical −0.96). **The
pre-registered prediction holds**.

**Interpretation**: The byte-identical pattern is real and
reflects the no-XM-interaction structure plus the deterministic
seed-aligned LOCO/pair/triplet design. It is *not* a
"cross-stratum invariance" in the strong sense (the wd0 stratum
diverges for fixes where the LOCO row coverage is missing — see
Loop 63 note), and it is also *not* an unanchored overclaim — the
result is now in `data/loop49_swap/3stratum_swap.csv` and
reviewable.

### Anchor

Phase 0 pre-registered Loop 63 (2026-06-01), completed Loop 64
(2026-06-01), anchored at branch HEAD `be9b4c4` (or descendant on
`f2-methodology`).

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
- The PR opening this study (PR #185 on `f2-methodology` branch,
  anchored at the latest descendant of `5367bde`) links to this
  pre-registration as the protocol. The current branch HEAD at the
  time of this Loop 58 update is `a092d5e`.

## 9. What this document is NOT

- It is not a research proposal seeking funding.
- It is not a guarantee of compute availability.
- It does not commit to any specific timeline.
- It does not preclude a later, separate pre-registration for related questions
  (e.g., warmup0 stratum, 3-way iLOCO on quantization paths, etc.).

## 10. Adversarial review checklist

If anyone wants to attack the design before data is collected:

- Q: Why 5 seeds? A: Owen 2025 (arXiv:2508.10083) shows BCa undercovers at
  N=5; we use the exact paired-permutation procedure of Zmigrod et al.
  (arXiv:2205.01416), which enumerates the 2^5 = 32 sign-flip outcomes
  exactly without Monte-Carlo approximation.
- Q: Why BH over 4 comparisons within each stratum? A: matches Liu/Leung/Shao
  (arXiv:1712.03305) asymptotic dependent-test BH validity for pairwise
  t-statistic comparisons.
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
