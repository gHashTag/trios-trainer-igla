# IGLA RACE v2 Format Sweep — Final Results

**Date:** 2026-05-26
**Status:** COMPLETED (all trainings stopped)
**Daemon PID:** 54732 (stopped)
**Processes:** 720 v6 instances stopped

---

## v2 Format Comparison (Best val_BPB per format, 81K steps)

| Rank | Format | Best val_BPB | Best EMA | Status |
|------|--------|-------------|----------|--------|
| 1 | f32 | 2.5042 | 2.8348 | Converging |
| 2 | fp16 | 2.5348 | 3.0383 | Converging |
| 3 | gf16 | 2.5267 | 3.0260 | Converging |
| 4 | bf16 | 2.5751 | 3.0653 | Converging |
| 5 | posit8 | 2.7947 | 2.9737 | Plateaued |
| 6 | gf8 | 2.7947 | 2.9737 | Plateaued |
| 7 | mxfp8 | 3.2406 | 3.4734 | Plateaued |
| 8 | int8 | 3.2905 | 3.6932 | Slow |
| 9 | nf4 | 3.8094 | 4.0012 | Poor |
| 10 | fp8 | 3.8094 | 4.0088 | Slow |
| 11 | int4 | 6.9935 | 6.9982 | Dead |

---

## Key Findings

- **f32 remains best** overall with val_BPB = 2.5042.
- **gf16** is competitive: only +0.0225 (0.9%) behind f32.
- **fp16** is also strong: +0.0306 (1.2%) behind f32.
- **Low-precision formats** (int8, int4, fp8_e4m3) degrade significantly.
- **int4 is essentially dead** (val_BPB ≈ 7.0, no convergence).
- **gf8 / posit8** plateau around 2.79–2.97 BPB, not improving past ~50K steps.

---

## v6 Scaling Record (for reference)

- **720 processes × 20B steps** running simultaneously.
- **Total compute:** 14.4 trillion steps.
- **Formats tested:** 24 (posit32, posit16, posit8, gf64, gf32, gf24, gf20, gf16, gf12, gf8, gf4, f32, f64, fp16, bf16, tf32, fp8, int8, int16, int4, mxfp8, nf4, lns8, uint8).
- **All v6 processes now stopped.**

---

## Action Items

- [x] Collect all v2 log files (30 logs, 11 unique formats).
- [x] Extract best val_BPB and EMA per format.
- [x] Stop all local training processes (720 + daemon).
- [ ] Archive logs to cold storage (if desired).
- [ ] Decide next experiment: v3 architecture or larger context?
