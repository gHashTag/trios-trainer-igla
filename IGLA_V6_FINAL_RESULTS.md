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
> (`f32::max` ignores NaN). A logged BPB does not establish that a forward pass succeeded.
>
> **3. The table below is headed "Best Results by Format" but is mostly untrained
> models.** Seven of its seventeen rows read exactly **6.9788** -- f64, gf64, posit32,
> gf24, posit16, gf32, gf20. Independently measured initialization loss on this
> architecture is ~7.00 bpb, so those seven rows are the value an untrained model
> returns, ranked as if they were results. **Sixteen of the seventeen rows report zero
> DONE runs**: nine are labelled "in progress (killed)" and seven "deep runs, not
> converged". Only gf12 has any completed run. The file states it below in its own words:
> "Only 5 completions out of 7,927 logs". A row whose training was killed before
> convergence is not a measurement of a numeric format.
>
> **4. Our own artifacts disagree on the DIRECTION of the headline claim.** This file
> places gf16 (2.8859) BEHIND bf16 (2.8419), sixth of seventeen. `IGLA_V2_FINAL_RESULTS.md`,
> dated the same day, has gf16 (2.5267) AHEAD of fp16 (2.5348). The skill-library snapshot
> frozen 2026-05-25 asserted the opposite of both: gf16 2.5725 beats bf16 2.6135 while fp16
> 2.5501 edges gf16 (all four values now retracted). Three artifacts of the same programme
> disagree about whether GoldenFloat beats bf16 and whether it beats fp16. That
> disagreement, not any individual value, is the reportable result -- and surfacing it is
> precisely what mandatory reproducibility of a development cycle is for.
>
> **5. Run count is unreconciled.** "7,927 logs / 5 DONE" here, "1878-run fleet" in
> trios-railway issue #109, and "~1,851 experiments" in the skill-library retraction are
> three different figures for overlapping populations. None is derived from an experiment
> ledger. Do not quote any of them as the fleet size.

# IGLA RACE v6 Format Sweep — Final Results

**Date:** 2026-05-26  
**Status:** COMPLETED (training stopped per user request)  
**Total logs collected:** 7,927  
**Completed runs:** 5 DONE  
**Peak concurrency:** 110 processes  
**Peak load average:** ~500

---

## Best Results by Format (sorted by BPB — lower is better)

| Format | DONE Runs | Best BPB | Best DONE BPB | Notes |
|--------|-----------|----------|---------------|-------|
| **gf12** | 5 | **2.4103** | **2.5232** | Best format. lr=0.001, h=256, 50K steps |
| gf8 | 0 | 2.7502 | — | in progress (killed) |
| f32 | 0 | 2.8142 | — | in progress (killed) |
| posit8 | 0 | 2.8353 | — | in progress (killed) |
| bf16 | 0 | 2.8419 | — | in progress (killed) |
| gf16 | 0 | 2.8859 | — | in progress (killed) |
| fp16 | 0 | 2.9091 | — | in progress (killed) |
| gf4 | 0 | 3.0898 | — | in progress (killed) |
| mxfp8 | 0 | 3.2016 | — | in progress (killed) |
| nf4 | 0 | 3.7091 | — | in progress (killed) |
| f64 | 0 | 6.9788 | — | deep runs, not converged |
| gf64 | 0 | 6.9788 | — | deep runs, not converged |
| posit32 | 0 | 6.9788 | — | deep runs, not converged |
| gf24 | 0 | 6.9788 | — | deep runs, not converged |
| posit16 | 0 | 6.9788 | — | deep runs, not converged |
| gf32 | 0 | 6.9788 | — | deep runs, not converged |
| gf20 | 0 | 6.9788 | — | deep runs, not converged |

---

## Completed Runs (sorted by BPB)

```
BPB=2.5232  steps=50000   gf12   v6_gf12_h256_lr0.001_seed71_50000   adamw
BPB=2.5417  steps=50000   gf12   v6_gf12_h256_lr0.003_seed71_50000   adamw
BPB=2.5762  steps=20000   gf12   v6_gf12_h768_lr0.001_seed71_20000   adamw
BPB=2.9505  steps=20000   gf12   v6_gf12_h768_lr0.0003_seed71_20000  adamw
BPB=3.0543  steps=50000   gf12   v6_gf12_h256_lr0.0003_seed71_50000  adamw
```

---

## Key Findings

1. **gf12 is the best format** — achieves BPB=2.5232, outperforming f32 baseline (2.8142)
2. **lr=0.001 consistently beats lr=0.0003** across all gf12 completions
3. **h=256 outperforms h=768** for gf12 at 50K steps
4. **f64/gf64/gf20/gf24/gf32/posit16/posit32** stuck at ~7.0 (random baseline) — deep step counts never converged due to early termination
5. **int8/int4/fp8_e5m2** — no eval data (killed before first eval)

---

## Configuration Space Explored

- **Formats:** 30 (gf4-gf64, f32/f64, posit8/16/32, fp16, bf16, tf32, fp8 variants, int8/16/4, mxfp8/6/4, nf4, lns8, uint8)
- **Seeds:** 138 (43–180)
- **Steps:** 18 (2K–15M)
- **Hiddens:** 5 (256–1024)
- **LRs:** 5 (0.01–0.0001)
- **Optimizers:** 3 (adamw, muon, muon-cwd)
- **Total combinations:** ~5.6 million

---

## System Limitations

- 110 processes on 8 cores → load avg ~500
- Most deep runs (500K+ steps) killed before convergence
- Only 5 completions out of 7,927 logs
- Fast turnaround requires step counts ≤ 50K on this hardware

---

## Recommendation

For future sweeps: cap step counts at 50K–100K for this hardware, or migrate to cloud (Railway) for deep runs.
