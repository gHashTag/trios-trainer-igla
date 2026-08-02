> # [RETRACTED 2026-08-02] -- READ BEFORE THE TABLE BELOW
>
> **Every number in this file is withdrawn. Do not cite it, quote it, or use it to
> order numeric formats.** The tables are retained in place, unedited: a silently
> deleted number is indistinguishable from a number that was never wrong.
>
> **1. No model artifact stands behind any row.** Until 2026-08-02, `checkpoint::save`
> in this repository was a stub that returned `Ok(())` and was called by nothing. The
> sweep summarized here wrote zero checkpoints. Nothing exists on disk to re-evaluate,
> so no row can be reproduced even in principle. (Fixed 2026-08-02: atomic save with a
> SHA-256 taken over bytes re-read from disk. **No re-measurement has been run under the
> fixed code.**)
>
> **2. The evaluator could not distinguish a measurement from a failure.**
> `loss_on_seq` / `evaluate` returned `0.0` and `f32::MAX` sentinels through the same
> channel as real readings, and a `.max(1e-10)` clamp laundered NaN into a finite value
> (`f32::max` ignores NaN), turning a poisoned forward pass into a plausible-looking
> number. A logged BPB does not establish that a forward pass succeeded.
>
> **3. Our own artifacts disagree on the DIRECTION of the headline claim.** This file
> ranks fp16 (2.5348) second and gf16 (2.5267) third while printing gf16 with the LOWER
> (better) BPB -- internally inconsistent -- and its gf16-below-fp16 ordering reverses the
> skill-library snapshot frozen 2026-05-25, which reported gf16 2.5725 / fp16 2.5501 /
> bf16 2.6135 / gf8 2.9322 (all four now retracted; none of them match this file).
> `IGLA_V6_FINAL_RESULTS.md`, dated the same day, reverses the other half: it places
> gf16 (2.8859) BEHIND bf16 (2.8419), and ranks gf12 -- a format neither of the other two
> sources discusses -- first.
>
> Three artifacts of the same programme disagree about whether GoldenFloat beats bf16 and
> whether it beats fp16. That disagreement, not any individual value, is the reportable
> result. It is also the point: mandatory reproducibility of a development cycle is
> supposed to surface exactly this, and here it did.

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
