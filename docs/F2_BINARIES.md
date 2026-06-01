# F2 Binaries — Index

The F2 framework (Loops 28-46) provides a complete pipeline for path-specific
ablation analysis. All binaries read/write long-form CSV (one row per
observation) and share the `race::stats` numerics layer + `race::ablation`
taxonomy.

**Total: 11 F2 binaries** (as of Loop 46). Verify with
`grep -c '^\[\[bin\]\]' Cargo.toml | grep f2_`.

| Binary | Purpose | Loop |
|---|---|---|
| `f2_ablation_sweep` | Run cumulative + LOCO + pairwise + triplet sweeps, including stratified modes (`wd_stratified`, `warmup_stratified`) | 24-42 |
| `f2_ablation_aggregate` | Wide-form CSV + paired-t p-values with race::stats t-CDF | 26 |
| `f2_iloco_score` | Pairwise and 3-way iLOCO (Möbius), BH-FDR, optional `--permutation`, `--control-variate` | 28 |
| `f2_iloco_dot` | Graphviz DOT + Mermaid output for interaction networks | 29 |
| `f2_mediation` | Baron-Kenny indirect/direct effect with percentile/exact bootstrap CI | 30 |
| `f2_dual_mediation` | Zhao-Luo 4-PSE decomposition (NDE/NIE_M1/NIE_M2/NIE_chain) with delta-method SE + t-corrected 95% CI | 34-36 |
| `f2_mediation_sensitivity` | Additive bridge-score envelope (arXiv:2605.18724 Thm 2); `--tipping-point`, `--lambda-sweep`, `--wide-form`, `--lambda-grid` | 40-46 |
| `f2_provenance_check` | W3C-PROV preamble verifier with PASS/WARN/FAIL exit codes; stratum-detection INFO banner | 33-43 |
| `f2_to_jsonl` | Long-form CSV → JSON Lines for Jupyter/Marimo notebooks (streaming, numeric inference, NaN→null) | 44-46 |
| `f2_harness` | Legacy single-config harness (pre-stratification) | 26 |
| `f2_pareto_sweep` | Multi-precision Pareto frontier sweep (1000-step baseline) | 26 |

## Typical pipeline

```bash
# 1. Run an ablation sweep (200 steps, ~6min sandbox or 1000 steps for prod)
f2_ablation_sweep --mode wd_stratified --steps 200 --csv out_sweep.csv

# 2. Verify provenance (validates schema/git SHA/host/timestamp)
f2_provenance_check out_sweep.csv

# 3. Score path-specific effects via dual-mediation
f2_dual_mediation --m1 wd --m2 warmup out_sweep.csv --out out_dual.csv

# 4. Compute additive sensitivity envelopes (multiple Λ values)
f2_mediation_sensitivity --lambda-sweep out_dual.csv --out out_sens.csv

# 5. Convert to JSON Lines for notebook analysis
f2_to_jsonl out_sens.csv --out out_sens.jsonl
```

## Cross-binary contract

- **CSV row** = `mode, fix_name, fix_index, cumulative_n, seed, bpb, config_hash, wall_s`
  (preserved by every binary that emits long-form data)
- **Provenance preamble** = `#`-prefixed W3C-PROV (arXiv:2312.07852) lines.
  Every long-form reader skips `#` and `mode,`-prefixed lines.
- **Stratum** = `Stratum::Canonical`, `Stratum::Wd0`, `Stratum::Warmup0`
  (`race::ablation`). Mode column carries the prefix (`wd0_loco`, `warmup0_pairwise`).
- **Canonical fix names** = `["rms", "warmup", "gradclip", "clamp", "smooth", "wd", "dropout"]`
  (`race::ablation::CANONICAL_FIX_NAMES`).

## Interpreting stratified results (Loop 47 audit fix 1)

When `f2_ablation_sweep` is run with `--mode wd_stratified` or
`--mode warmup_stratified`, the resulting CSV contains rows tagged with the
stratum prefix (`wd0_*` or `warmup0_*`). When this CSV is consumed by
`f2_dual_mediation` (or `f2_mediation_sensitivity`), the reported
`NDE` / `NIE_M1` / `NIE_M2` / `NIE_chain` values are **Pearl Controlled Direct
and Indirect Effects** at the disabled-value of the stratum variable —
*not* marginal natural effects.

| Sweep mode | Stratum tag | What NDE means |
|---|---|---|
| canonical (`--mode loco/pairwise/triplet`) | `pairwise`, `loco`, `triplet` | Marginal NDE per Zhao-Luo (arXiv:2007.16031) — direct effect of X with M1, M2 free to flow |
| `--mode wd_stratified` | `wd0_*` | CDE at WD pinned to 0.0 — direct effect of X with WD's pathway structurally blocked |
| `--mode warmup_stratified` | `warmup0_*` | CDE at warmup_steps=0 — direct effect of X with warmup's pathway structurally blocked |

The `f2_dual_mediation` binary surfaces the input stratum in the output
preamble (`# INPUT STRATUM = warmup0`) so downstream analysis cannot mistake
a CDE for a marginal effect. Loop 47 audit fix 2 added this check; the
`detect_input_stratum_classifies_each_prefix` unit test locks it.

**Practical guidance:**
- If you want **marginal** NDE/NIE → use canonical-mode CSV.
- If you want CDE at a fixed mediator → use the corresponding stratified mode.
- **Don't mix strata in one CSV** — the stratum-detector reports `mixed` and
  the result is causally undefined.

## References

- arXiv:2007.16031 (Zhao-Luo 2-mediator decomposition)
- arXiv:2502.06661 (iLOCO interaction scoring)
- arXiv:1710.02011 (Miles-Shpitser efficient influence function)
- arXiv:2508.10083 (Owen Aug 2025: BCa undercoverage at small N)
- arXiv:2605.18724 (Ohnishi-Li bridge-score sensitivity, Theorem 2)
- arXiv:2312.07852 (RO-Crate provenance preamble)
- arXiv:2011.04216 (DoWhy flat record-per-estimate convention)
