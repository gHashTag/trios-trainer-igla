# IGLA RACE — Massive Format Sweep Status
## 2026-05-25 02:37 UTC

### Running Processes: 74 total
- **v2** (30 runs): Real QAT, normal priority — 81K/50K steps
- **v3** (39 runs): Real QAT, nice -19 — 10K steps, seeds 45-47
- **v4** (9 runs): Real QAT, nice -19 — 20K steps, seeds 45-47

---

## v2 Results (Real QAT — the money data)

### Step 30000 / 25000 (best so far)

| Format | Seed | Step | Best BPB | Notes |
|--------|------|------|----------|-------|
| **gf8** | 44 | 25000 | **3.2050** | 🏆 LEADER |
| **posit8** | 44 | 25000 | **3.2050** | 🏆 Tied leader |
| **gf16** | 44 | 30000 | **3.7706** | Beats f32! |
| **fp16** | 44 | 30000 | **3.7847** | Close to gf16 |
| **f32** | 42 | 30000 | **3.7998** | Baseline |
| **mxfp8** | 44 | 25000 | **3.6219** | Good 8-bit |
| **nf4** | 44 | 25000 | **4.1158** | 4-bit learning! |
| **int8** | 42 | 30000 | **4.2357** | Worse than nf4 |
| **bf16** | 44 | 20000 | **4.4227** | Slower convergence |
| **fp8_e5m2** | 44 | 20000 | **4.2899** | Decent |
| **int4** | 44 | 30000 | **6.9935** | 💀 DEAD — no learning |
| **fp8_e4m3** | — | — | N/A | Not started yet |

### Key Findings

1. **gf16 beats f32** (3.7706 vs 3.7998 @ 30K). Difference: -0.029 BPB. Small but real — QAT works.
2. **8-bit formats dominate**: gf8 and posit8 both at 3.2050 — ~0.57 BPB better than f32. 8 bits enough for this model.
3. **int4 is dead**: BPB stuck at ~7.0 (initialization value). 4 bits insufficient for backprop.
4. **nf4 learns**: 4.1158 @ 25K — much better than int4. NF4 (normal float) preserves dynamic range.
5. **bf16 lags**: 4.4227 @ 20K — slower than fp16/gf16. Possibly due to different rounding in QAT.

### Expected Completion
- 50K runs (gf8, posit8, mxfp8, nf4, fp8_e5m2, fp8_e4m3): ~4-6 hours
- 81K runs (f32, fp16, bf16, gf16, int8, int4): ~10-15 hours

---

## v3 / v4 Runs (Seeds 45-47, nice -19)

Just launched. Will provide cross-seed statistical validation. Expected first eval in 20-40 minutes.

---

## Total Data Points When Complete

| Category | Runs | Steps | Data quality |
|----------|------|-------|-------------|
| v2 81K | 18 | 81K | Best (real QAT, 3 seeds) |
| v2 50K | 12 | 50K | Good (real QAT, 2 seeds) |
| v3 10K | 39 | 10K | Medium (3 new seeds) |
| v4 20K | 9 | 20K | Good (3 new seeds) |
| **Total** | **78** | **Mixed** | **First honest format comparison** |

---

## Next Actions

1. Wait for v2 50K completion (~4-6h) — kill nice -19 jobs if needed to free CPU
2. Collect v3/v4 first evals (~30-60 min)
3. When load drops, launch 81K runs for gf8/posit8 (best performers)
4. Generate final BPB vs format plot

