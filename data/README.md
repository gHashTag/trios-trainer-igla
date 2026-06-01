# `data/` — empirical evidence for the F2 methodology paper

This directory holds every committed CSV that backs a numerical
claim in `papers/f2_methodology.md`. Two subdirectories:

| Directory | Bytes | Backs paper section | One-sentence purpose |
|---|---|---|---|
| [`loop49/`](loop49/) | ~80 KB (6 files + README) | §5.1, §5.2, §5.4, Figures 1, 3, 4 | The headline sandbox-ablation matrix and the canonical / wd0 / warmup0 three-stratum cross-comparison (canonical parameterization, M_1 = wd, M_2 = warmup). |
| [`loop49_swap/`](loop49_swap/) | ~152 KB (15 files) | §5.3, §6.4, Figures 5, 6 | Phase 0 swap-parameterization data: re-runs of `f2_dual_mediation` with M_1 = rms and every candidate M_2 ∈ {warmup, gradclip, clamp, smooth, dropout}, across all three strata. |

Both directories together = ~232 KB of CSV (no binaries, no PDFs,
no PNGs). Each file opens with a W3C-PROV / RO-Crate preamble per
§3.5.1 of the paper.

## How to verify integrity

```bash
# Provenance check on every committed sweep CSV — exits 0 on PASS,
# 1 on WARN (older git SHA, schema OK), 2 or 3 on FAIL.
cargo run --release --bin f2_provenance_check -- \
    data/loop49/loop49_wd_stratified.csv \
    data/loop49/loop47_warmup_stratified.csv \
    data/loop49_swap/canonical_sweep.csv
```

## How to regenerate every paper figure from these CSVs

```bash
papers/scripts/figure_regen.sh
# → papers/figures/fig{1,2,3,4,5,6}_*.png (~10 s wall time)
```

## What is NOT in this directory

- **Per-seed training logs**: F2 binaries operate on aggregated
  sweep CSVs (mode × fix_name × seed × BPB); raw training logs are
  not committed.
- **Champion-scale data**: pre-registered in `docs/F2_PRE_REG.md`,
  not yet executed.
- **Trainer source data** (FineWeb tokens, etc.): the sandbox uses
  a synthetic counter task per §4.1; no external dataset shipped.

## File-level provenance

For per-file MD5 checksums and reproduction commands, see
[`loop49/README.md`](loop49/README.md). The `loop49_swap/`
subdirectory's contents are described in `docs/F2_PRE_REG.md` §4.5
(Phase 0 deliverable, completed Loop 64, robustness-extended Loop 68).

Anchor commit: `583b417` is the earliest commit on the
`f2-methodology` branch at which every file in this tree is
committed. Any descendant of `583b417` is also a valid anchor.
