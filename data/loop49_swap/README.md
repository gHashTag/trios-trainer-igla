# `data/loop49_swap/` — Phase 0 swap-parameterization evidence

This directory holds the empirical evidence for §5.3 (cross-stratum
stability flag — wd × NIE_M1 byte-identical result) and §6.4 (full
M_2 robustness landscape) of `papers/f2_methodology.md`.

All files produced by `f2_dual_mediation --m1 rms --m2 <m>` applied
to the three sweep CSVs (canonical, wd0, warmup0) for each candidate
M_2 ∈ {warmup, gradclip, clamp, smooth, dropout}, plus the
cross-stratum compare CSV.

## File manifest (17 files, ~152 KB total)

| File | Bytes | MD5 |
|---|---:|---|
| `3stratum_swap.csv` | 2456 | `1b56c4d917ff9bdf119a8d935df87ac9` |
| `canonical_sweep.csv` | 88411 | `446f0999e69c39ed65d538f4d2bff523` |
| `canonical_swap_dual.csv` | 2070 | `6df55cccaa08ba9a9c9275901ab2bdcd` |
| `canonical_swap_m2clamp.csv` | 2065 | `7b2d383867e14a7a64a24b816c94f90e` |
| `canonical_swap_m2dropout.csv` | 2066 | `6b7765c8b91c7339f03bfe79b154bf0f` |
| `canonical_swap_m2gradclip.csv` | 2064 | `e0551c5b69e6d6b04e7c8c3a643d4a1a` |
| `canonical_swap_m2smooth.csv` | 2065 | `a37b2cba9c14cab81babbd09cdafb891` |
| `wd0_swap_dual.csv` | 2039 | `bb846bc2dc76888accd2faab1c95d82b` |
| `wd0_swap_m2clamp.csv` | 2031 | `d02f8eee03d5009eb8b1c55047b84db9` |
| `wd0_swap_m2dropout.csv` | 2039 | `035864271d70a7ce9beb60c3eee99c87` |
| `wd0_swap_m2gradclip.csv` | 2022 | `2f3799fb8de5172d298d40402b55fa5c` |
| `wd0_swap_m2smooth.csv` | 2031 | `b95f88fff841a65fce1e05e71e4e7b8b` |
| `warmup0_swap_dual.csv` | 2037 | `1996f07f3131e8823a85026ac95c9f1c` |
| `warmup0_swap_m2clamp.csv` | 2037 | `a55ca560d734a0ad6f6795d8c450d84e` |
| `warmup0_swap_m2dropout.csv` | 2038 | `3d79e61aad6068f471c0f8a3b8a645fe` |
| `warmup0_swap_m2gradclip.csv` | 2038 | `8382d44e0f995d1be6df73e21bbb1d74` |
| `warmup0_swap_m2smooth.csv` | 2037 | `d023094601f4473d4e4bdb4ae7bb5782` |

(Checksums computed at Loop 76 commit, `3783ab7` and descendants.)

## File naming convention

```
<stratum>_swap_dual.csv            — M_2 = warmup (default), stratum ∈ {canonical, wd0, warmup0}
<stratum>_swap_m2<m>.csv           — M_2 = m,           stratum ∈ {canonical, wd0, warmup0}
canonical_sweep.csv                — input sweep CSV regenerated Loop 64 (~25 min wall)
3stratum_swap.csv                  — cross-stratum compare for M_2 = warmup
```

## How each file was produced

```bash
# canonical_sweep.csv — fresh sweep at the f2-methodology branch HEAD.
cargo run --release --bin f2_ablation_sweep -- \
    --mode all --steps 200 \
    --csv data/loop49_swap/canonical_sweep.csv

# canonical_swap_dual.csv — swap parameterization on the canonical sweep.
cargo run --release --bin f2_dual_mediation -- \
    --m1 rms --m2 warmup data/loop49_swap/canonical_sweep.csv \
    --out data/loop49_swap/canonical_swap_dual.csv

# wd0_swap_dual.csv — same swap, on the committed wd_stratified sweep.
cargo run --release --bin f2_dual_mediation -- \
    --m1 rms --m2 warmup data/loop49/loop49_wd_stratified.csv \
    --out data/loop49_swap/wd0_swap_dual.csv

# warmup0_swap_dual.csv — same swap, on the committed warmup_stratified sweep.
cargo run --release --bin f2_dual_mediation -- \
    --m1 rms --m2 warmup data/loop49/loop47_warmup_stratified.csv \
    --out data/loop49_swap/warmup0_swap_dual.csv

# Alternative M_2 robustness CSVs (12 files) — Loop 68 sweep.
for m2 in gradclip clamp smooth dropout; do
    for stratum in canonical:data/loop49_swap/canonical_sweep.csv \
                   wd0:data/loop49/loop49_wd_stratified.csv \
                   warmup0:data/loop49/loop47_warmup_stratified.csv; do
        s=${stratum%%:*}; csv=${stratum#*:}
        cargo run --release --bin f2_dual_mediation -- \
            --m1 rms --m2 $m2 $csv \
            --out data/loop49_swap/${s}_swap_m2${m2}.csv
    done
done

# 3stratum_swap.csv — cross-stratum compare for M_2 = warmup.
cargo run --release --bin f2_stratum_compare -- \
    --canonical data/loop49_swap/canonical_swap_dual.csv \
    --wd0       data/loop49_swap/wd0_swap_dual.csv \
    --warmup0   data/loop49_swap/warmup0_swap_dual.csv \
    --out       data/loop49_swap/3stratum_swap.csv
```

## Verification

```bash
cargo run --release --bin f2_provenance_check -- \
    data/loop49_swap/canonical_sweep.csv
# Exits 0 (PASS) or 1 (WARN, older git SHA but schema OK).
```

Anchor: `583b417` is the earliest commit on `f2-methodology` at
which every file in this directory is committed. Any descendant
is also a valid anchor.
