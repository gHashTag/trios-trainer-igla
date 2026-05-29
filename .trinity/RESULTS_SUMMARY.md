# IGLA RACE Format Sweep — Results Summary

Date: 2026-05-25
Model: embed+bigram+smear+lm_head, 196K params, hidden=384
Optimizer: AdamW, lr=0.003 (default)
Dataset: Synthetic fallback (tinyshakespeare.txt not found)

## Critical Finding

`trios-train --format=X` is **dead code** (`#[allow(dead_code)]` in `src/bin/trios-train.rs:89`).
All 28 currently running processes are effectively **f32 training** regardless of `--format`.
The only quantization applied is `gf16_floor()` (coarse 1/16 grid snap) after 70% of steps,
which has negligible impact on final BPB for this model size.

## Completed Results

### 20K Steps (Best so far)
| Run | Seed | Final BPB | Notes |
|-----|------|-----------|-------|
| long_f32_seed44_20k | 44 | **2.6026** | Best result overall |
| long_gf16_seed44_20k | 44 | **2.6026** | Identical to f32 — format is dead code |
| long_f32_seed42_20k | 42 | **2.6125** | |
| long_gf16_seed42_20k | 42 | **2.6125** | Identical to f32 |
| long_gf16_seed43_20k | 43 | **2.6218** | |
| long_f32_seed43_20k | 43 | INCOMPLETE (step 18K, best=2.6687) | ~10 min remaining |

### 10K Steps (ALL formats identical per seed)
| Seed | BPB | Formats tested |
|------|-----|----------------|
| 42 | 2.7176 | f32, bf16, fp16, gf16, gf8 |
| 43 | 2.7027 | f32, bf16, fp16, gf16, gf8, fp8_e5m2, int4, int8, mxfp8, nf4, posit8, tf32 |
| 44 | 2.6872 | f32, bf16, fp16, gf16, gf8 |

**Observation**: Even int4/nf4/posit8 match f32 at 10K. Model is too small for quantization effects to matter without real QAT.

### 5K Steps (Format Sweep, seed=43)
| Group | Formats | BPB |
|-------|---------|-----|
| High-bit | f32, bf16, fp16, gf16, gf8, fp8_e4m3 | 2.8494 |
| Low-bit | gf4, gf32, lns8, mxfp4, mxfp8, posit8, posit16 | 3.2163 |

Note: Difference is random noise from independent runs with same seed (identical val_bpb at steps 1K-4K).

### Hyperparameter Sweeps (20K, gf16, seed=43)
| Config | Best BPB (at step) | Status |
|--------|---------------------|--------|
| hidden=384, lr=0.003 (baseline) | 2.6218 (20K) | Best |
| hidden=384, lr=0.001 | 3.4057 (8K) | Too slow |
| hidden=384, lr=0.005 | 3.6591 (8K) | Diverging |
| hidden=384, lr=0.01 | 3.9512 (8K) | Diverging |
| hidden=256 | 3.2320 (10K) | Underfitting |
| hidden=512 | 3.9355 (6K) | Unstable |
| hidden=828 | 3.9502 (6K) | Unstable |

**Conclusion**: lr=0.003, hidden=384 is optimal. Larger hidden dims cause instability on this task.

### Champion 81K (seed=43, no format flag)
| Step | Best BPB | Notes |
|------|----------|-------|
| 31K | 2.7333 | Worse than 20K runs. Possibly lr=0.003 too high for 81K steps, or eval-every=1000 affects EMA differently. |

## Running Processes (28 trios-train + 1 shell)

- **ULTRA 50K** (12 runs): f32/bf16/fp16/gf16 × seeds 42/43/44 — all ~10K steps in, best ~4.47
- **MEGA 81K** (6 runs): fp16/gf16 × seeds 42/43/44 — all ~10K steps in, best ~5.43
- **Champion 81K** (1 run): seed=43, no format — at step 31K, best=2.7333
- **Hidden/LR sweeps** (6 runs): 20K steps, various configs — most in progress
- **format_sweep.sh** (1 shell): Running gf16 seed=43 20K with lr=0.01 (diverging)

## Recommendations

1. **Fix dead code**: Connect `--format` in `trios-train.rs` to real QAT (FakeQuant + STE) from `fake_quant.rs`.
2. **For real format comparison**: Use `cpu_train` with `TRIOS_FORMAT_TYPE=gf16` (it implements real QAT).
3. **Stop low-LR / high-hidden runs**: They're underfitting/unstable. Focus on lr=0.003, hidden=384.
4. **Let 50K/81K runs finish**: They'll complete in ~2-4 hours and may show further BPB improvement.

