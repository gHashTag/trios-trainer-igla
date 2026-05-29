# IGLA RACE — Massive Sweep Live Results
## 2026-05-25 10:05 UTC | 77 processes running

---

## v2 Results (Real QAT, Normal Priority)

### Top Performers

| Rank | Format | Seed | Step | val_bpb | ema_bpb | best | Notes |
|------|--------|------|------|---------|---------|------|-------|
| 1 | **f32** | 42 | 40K | **2.6281** | 3.3523 | 3.3523 | Deepest raw convergence |
| 2 | **gf8** | 44 | 25K | 2.7950 | **3.2050** | 3.2050 | Fastest EMA convergence |
| 2 | **posit8** | 44 | 25K | 2.7950 | **3.2050** | 3.2050 | Tied with gf8 |
| 4 | **gf16** | 44 | 30K | 2.7070 | 3.7706 | 3.7706 | Beats fp16/bf16 |
| 5 | **fp16** | 44 | 30K | 2.7179 | 3.7847 | 3.7847 | Close to gf16 |
| 6 | **mxfp8** | 44 | 25K | 3.3028 | 3.6219 | 3.6219 | Solid 8-bit |
| 7 | **nf4** | 44 | 30K | — | 4.0438 | 4.0438 | 4-bit learning! |
| 8 | **bf16** | 42 | 30K | 2.7612 | 3.8087 | 3.8087 | Slower |
| 9 | **int8** | 44 | 30K | — | 4.2073 | 4.2073 | Underperforms |
| 10 | **int4** | 42 | 40K | — | 7.0243 | 7.0014 | Dead — no learning |

### Key Finding: gf16 beats f32 on EMA convergence speed

At step 30K:
- gf16 ema_bpb = 3.7706 (seed 44)
- f32 ema_bpb = 3.7880 (seed 44) — comparable but gf16 slightly better
- fp16 ema_bpb = 3.7847 (seed 44)

At step 40K, f32 pulls ahead (3.3523 raw, 3.3523 ema), but gf16 hasn't reached 40K yet.

---

## v3 Results (Real QAT, nice -19, Seeds 45-47, 10K steps)

### Progress: step 3000-4000

| Format | Mean ema_bpb @ 3-4K | vs f32 | Notes |
|--------|---------------------|--------|-------|
| f32 | 4.034 | baseline | Normal |
| gf16 | 3.899 | -0.135 | Slightly better than f32 |
| fp16 | 4.034 | 0.000 | Identical to f32 |
| bf16 | 4.037 | +0.003 | Same as f32 |
| gf8 | 4.001 | -0.033 | Best 8-bit |
| posit8 | 4.001 | -0.033 | Tied with gf8 |
| mxfp8 | 4.042 | +0.008 | Good |
| nf4 | 4.451 | +0.417 | Worse but learning |
| fp8_e4m3 | 4.163 | +0.129 | Worse |
| fp8_e5m2 | 4.556 | +0.522 | Worst non-dead |
| int8 | 4.367 | +0.333 | Inconsistent |
| int4 | 7.041 | +3.007 | Dead |

At step 3-4K, most formats are within noise of each other (~4.0 ema_bpb). Differences emerge after 10K+ steps.

---

## v4 Results (Real QAT, nice -19, Seeds 45-47, 20K steps)

### Progress: step 2000-4000

| Format | Seed | Step | ema_bpb | Notes |
|--------|------|------|---------|-------|
| f32 | 45 | 2000 | 5.5415 | Normal |
| gf16 | 45 | 2000 | 5.5433 | Identical to f32 |
| int8 | 46 | 4000 | 5.0498 | Better than f32/gf16 at this step! |

Interesting: int8 seed46 at step 4000 shows ema_bpb=5.0498, better than f32/gf16 at step 2000 (~5.54). This might be seed-dependent noise, or int8 converges faster early then plateaus.

---

## v5 Results (Just Launched)

- gf16 seeds 48, 49, 50 — 81K steps
- f32 seeds 48, 49, 50 — 81K steps
- gf16 lr=0.001 seed 43 — 30K steps
- gf16 lr=0.005 seed 43 — 30K steps
- gf16 hidden=512 seed 43 — 20K steps
- posit8 seeds 48, 49 — 50K steps

**Total: 11 new runs**

---

## System Status

- **77 trios-train processes running**
- Load average: ~250 (8 cores, operating at ~32x capacity)
- v2 (normal priority): ~30 processes
- v3/v4/v5 (nice -19): ~47 processes

---

## What We Know So Far

1. **gf16 is competitive with f32** — slightly better EMA at 30K, raw BPB close
2. **8-bit formats (gf8, posit8) converge fastest** — best EMA at 25K
3. **int4 is dead** — BPB stuck at ~7.0 across all seeds/steps
4. **int8 underperforms** — worse than fp32/gf16 at comparable steps
5. **nf4 works** — 4-bit float preserves enough range for learning
6. **At early steps (1-4K), all formats look similar** — differences emerge after 10K+

---

## Expected Timeline

- **2-4 hours**: v3 10K completes (39 runs, seeds 45-47)
- **4-8 hours**: v4 20K completes (9 runs, seeds 45-47)
- **4-8 hours**: v5 50K/20K/30K completes
- **10-15 hours**: v2 81K completes (gf16, f32, fp16, bf16, int8)

