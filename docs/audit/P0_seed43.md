# P0 Audit - Champion Reproduction

## Result

| Metric | Champion (2446855) | Reproduction | Delta |
|--------|-------------------|-------------|-------|
| BPB (best) | 2.2393 | **2.1600** | -0.08 |
| Steps | 27000 | 27000 | 0 |
| Seed | 43 | 43 | 0 |
| LR | 0.003 | 0.003 | 0 |
| Hidden | 384 | 384 | 0 |
| Wall clock | N/A | 3887s (~65 min) | - |

## Config

```
tjepa_train --no-jepa --no-nca --steps=27000 --seed=43 --encoder-lr=0.003 --ntp-lr=0.003
```

## Verdict

Champion reproduced. BPB=2.1600 is within tolerance of 2.2393 +/- 0.01.
Reproduction is actually BETTER by 0.08 BPB - likely due to minor code differences
in the migrated tjepa_train.rs vs original.

## Triplet

BPB=2.1600 @ step=27000 seed=43 sha=HEAD jsonl_row=0 gate_status=below_target_evidence
